"""Runnable FusionRL control plane for groups, policy versions and chunks.

The scheduler does not create GPU processes, move weights/KV blocks or implement
dynamic Megatron membership. Backends execute its decisions and report actual
provenance. These pure-Python controllers can also be driven by a simulator.

FusionRL uses fixed dispatch-order group quotas to avoid completion-time-biased
batch membership. This can wait for a straggler; it is a deliberate modification
of QwenGyre's oldest-ready streaming policy. Seer-inspired chunk placement is a
bounded online heuristic, not a reproduction of its production cache system.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import math
from typing import Mapping, Sequence

from .trajectory import (
    AdmittedGroup, PolicyVersionMismatch, QuarantinedGroup, RolloutGroup,
    admit_group, policy_versions, validate_complete_group,
)


def _integer(value: int, name: str, minimum: int = 0) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


@dataclass(frozen=True)
class SchedulerConfig:
    mode: str = "strict_sync"
    groups_per_update: int = 1
    executions_per_group: int = 4
    burst_steps: int = 1
    extra_dispatch_groups: int = 0
    max_pending_groups: int = 32
    max_policy_lag: int = 0
    jmax: int = 5

    def __post_init__(self) -> None:
        if self.mode not in {"strict_sync", "bounded_async"}:
            raise ValueError("mode must be strict_sync or bounded_async")
        for name in ("groups_per_update", "executions_per_group", "burst_steps", "max_pending_groups", "jmax"):
            _integer(getattr(self, name), name, 1)
        for name in ("extra_dispatch_groups", "max_policy_lag"):
            _integer(getattr(self, name), name)
        if self.target_depth > self.max_pending_groups:
            raise ValueError("boundary dispatch depth exceeds the bounded pending-group capacity")
        if self.mode == "strict_sync" and (self.burst_steps != 1 or self.extra_dispatch_groups or self.max_policy_lag):
            raise ValueError("strict_sync requires one update per publication, no extra dispatch and zero token lag")

    @property
    def target_depth(self) -> int:
        return self.groups_per_update * self.burst_steps + self.extra_dispatch_groups

    @property
    def ideal_stationary_mean_dispatch_lag(self) -> float:
        """No-drop fixed-population accounting value, NOT a measured/max bound.

        Quarantines, replacements, finite windows or changing counts invalidate
        an unconditional claim that observed lag equals this expression.
        """
        return self.extra_dispatch_groups / self.groups_per_update + (self.burst_steps - 1) / 2


@dataclass(frozen=True)
class GroupSpec:
    group_id: str
    conditioning_key: str
    domain: str = "default"

    def __post_init__(self) -> None:
        if not self.group_id or not self.conditioning_key or not self.domain:
            raise ValueError("group_id, conditioning_key and domain must be explicit")


@dataclass(frozen=True)
class GroupTicket:
    group_id: str
    conditioning_key: str
    domain: str
    dispatch_version: int
    expected_executions: int
    sequence: int


class FusionScheduler:
    """Boundary-dispatched, complete-group scheduler with exactly-once use.

    Call ``dispatch`` at startup or after ``publish``; submit terminal groups
    with ``complete``; call ``begin_update`` to receive the immutable admitted
    groups. After the optimizer succeeds, ``commit_update`` consumes those
    groups and advances the trainer version. ``publish`` is called only at a
    completed burst boundary. Optimizer execution remains the caller's job.

    In async mode the lag gate is applied at update admission to actual token
    versions, not dispatch metadata. Old tokens are masked before ratios and
    denominators; their original execution/reward slots remain in the group.
    """

    def __init__(self, config: SchedulerConfig | None = None,
                 quotas: Mapping[str, int] | None = None, policy_version: int = 0):
        self.config = config or SchedulerConfig()
        _integer(policy_version, "policy version")
        self.policy_version = policy_version
        self.published_version = policy_version
        self.quotas = dict(quotas or {"default": self.config.groups_per_update})
        for domain, count in self.quotas.items():
            if not domain:
                raise ValueError("domain quota keys must be nonempty")
            _integer(count, "domain quota", 1)
        if sum(self.quotas.values()) != self.config.groups_per_update:
            raise ValueError("domain quotas must sum to groups_per_update")
        self._tickets: dict[str, GroupTicket] = {}
        self._ready: dict[str, RolloutGroup] = {}
        self._quarantined: dict[str, str] = {}
        self._seen: set[str] = set()
        self._consumed: list[str] = []
        self._next_sequence = 0
        self._dispatch_open = True
        self._steps_in_burst = 0
        self._needs_publication = False
        self._inflight: tuple[str, ...] | None = None
        self._admitted: tuple[AdmittedGroup, ...] | None = None
        self.dispatch_lags: list[int] = []
        self.admitted_token_lags: list[int] = []
        self.quarantine_history: list[tuple[str, str]] = []
        self.cancellation_history: list[tuple[str, str]] = []

    @property
    def pending_count(self) -> int:
        return len(self._tickets)

    @property
    def ready_count(self) -> int:
        return len(self._ready)

    @property
    def consumed_groups(self) -> tuple[str, ...]:
        return tuple(self._consumed)

    @property
    def quarantined_groups(self) -> dict[str, str]:
        return dict(self._quarantined)

    @property
    def needs_publication(self) -> bool:
        return self._needs_publication

    def dispatch(self, specs: Sequence[GroupSpec]) -> tuple[GroupTicket, ...]:
        if not self._dispatch_open or self._inflight is not None:
            raise RuntimeError("new groups can be dispatched only at publication boundaries")
        specs = tuple(specs)
        if len({spec.group_id for spec in specs}) != len(specs):
            raise ValueError("duplicate group ID in dispatch transaction")
        if any(spec.group_id in self._seen for spec in specs):
            raise ValueError("group identity reuse/replay is disabled")
        if any(spec.domain not in self.quotas for spec in specs):
            raise ValueError("dispatch domain is absent from fixed update quotas")
        if self.pending_count + len(specs) > self.config.target_depth:
            raise ValueError("dispatch would exceed the boundary group quota")
        if self.pending_count + len(specs) == self.config.target_depth:
            counts = {domain: sum(ticket.domain == domain for ticket in self._tickets.values())
                      + sum(spec.domain == domain for spec in specs) for domain in self.quotas}
            if any(counts[domain] < quota * self.config.burst_steps for domain, quota in self.quotas.items()):
                raise ValueError("a full boundary buffer must satisfy each domain quota for every update in the burst")
        # Validate the transaction before mutating any state.
        tickets = tuple(GroupTicket(spec.group_id, spec.conditioning_key, spec.domain,
                        self.published_version, self.config.executions_per_group,
                        self._next_sequence + offset) for offset, spec in enumerate(specs))
        for ticket in tickets:
            self._tickets[ticket.group_id] = ticket
            self._seen.add(ticket.group_id)
        self._next_sequence += len(tickets)
        return tickets

    def complete(self, group: RolloutGroup) -> None:
        if group.group_id not in self._tickets:
            raise ValueError("completed group has no live dispatch ticket")
        if group.group_id in self._ready or group.group_id in self._quarantined:
            raise ValueError("a dispatch group can report a terminal result only once")
        ticket = self._tickets[group.group_id]
        try:
            if group.expected_executions != ticket.expected_executions:
                raise QuarantinedGroup(group.group_id, "replicate count differs from the original dispatch quota")
            key = validate_complete_group(group)
            if key != ticket.conditioning_key:
                raise QuarantinedGroup(group.group_id, "conditioning key differs from the dispatch contract")
            if group.dispatch_version != ticket.dispatch_version:
                raise QuarantinedGroup(group.group_id, "dispatch provenance differs from the original ticket")
            if self.config.mode == "strict_sync" and any(v != self.policy_version for v in policy_versions(group)):
                raise PolicyVersionMismatch(group.group_id, "strict synchronous generation changed its frozen policy")
            if any(v > self.published_version for v in policy_versions(group)):
                raise PolicyVersionMismatch(group.group_id, "rollout claims weights that have not been published")
        except (QuarantinedGroup, PolicyVersionMismatch) as error:
            # A version violation is a terminal, invalid outcome too. Leaving the
            # slot neither ready nor quarantined would block every later update
            # and would let a different record be submitted for the same slot.
            self._quarantined[group.group_id] = error.reason
            self.quarantine_history.append((group.group_id, error.reason))
            raise
        self._ready[group.group_id] = group

    def quarantine(self, group_id: str, reason: str) -> None:
        """Invalidate a live group after completion, e.g. when integrity
        monitoring cannot clear it. It can then be retried in its slot."""
        if not reason:
            raise ValueError("quarantine requires an explicit recorded reason")
        if group_id not in self._tickets:
            raise ValueError("only a live dispatched group can be quarantined")
        if self._inflight is not None and group_id in self._inflight:
            raise RuntimeError("cannot quarantine a group owned by the current optimizer update")
        if group_id in self._quarantined:
            return
        self._ready.pop(group_id, None)
        self._quarantined[group_id] = reason
        self.quarantine_history.append((group_id, reason))

    def replace_quarantined(self, old_group_id: str, new_group_id: str) -> GroupTicket:
        """Retry the entire invalid group in its reserved logical slot.

        The replacement keeps task/domain, replicate count, position and old
        behavior snapshot. The backend must still retain that snapshot. This
        operation does not transform an invalid score into a policy failure.
        """
        if old_group_id not in self._quarantined:
            raise ValueError("only a quarantined group can be replaced")
        if not new_group_id or new_group_id in self._seen:
            raise ValueError("replacement requires a fresh dispatch group ID")
        if self._inflight is not None:
            raise RuntimeError("cannot replace group slots during an optimizer update")
        old = self._tickets[old_group_id]
        replacement = replace(old, group_id=new_group_id)
        del self._tickets[old_group_id]
        del self._quarantined[old_group_id]
        self._tickets[new_group_id] = replacement
        self._seen.add(new_group_id)
        return replacement

    def begin_update(self) -> tuple[AdmittedGroup, ...] | None:
        if self._inflight is not None:
            raise RuntimeError("an update already owns a batch; duplicate sample claims are disabled")
        if self._needs_publication:
            raise RuntimeError("the completed burst must publish before another update")
        if self._dispatch_open:
            if self.pending_count != self.config.target_depth:
                return None
            self._dispatch_open = False
        selected: list[GroupTicket] = []
        for domain, count in self.quotas.items():
            candidates = sorted((ticket for ticket in self._tickets.values() if ticket.domain == domain),
                                key=lambda ticket: ticket.sequence)
            if len(candidates) < count:
                return None
            selected.extend(candidates[:count])
        # Membership depends on dispatch order and quota, never completion order.
        for ticket in selected:
            if ticket.group_id in self._quarantined:
                raise QuarantinedGroup(ticket.group_id, self._quarantined[ticket.group_id])
            if ticket.group_id not in self._ready:
                return None
        groups = tuple(self._ready[ticket.group_id] for ticket in selected)
        if self.config.mode == "strict_sync":
            for group in groups:
                if any(version != self.policy_version for version in policy_versions(group)):
                    raise PolicyVersionMismatch(group.group_id, "strict synchronous updates cannot mask away stale generation")
        admitted = tuple(admit_group(group, self.policy_version, self.config.max_policy_lag, self.config.jmax)
                         for group in groups)
        self._inflight = tuple(ticket.group_id for ticket in selected)
        self._admitted = admitted
        return admitted

    def commit_update(self) -> int:
        if self._inflight is None or self._admitted is None:
            raise RuntimeError("no admitted batch is waiting for optimizer commit")
        for group_id, group in zip(self._inflight, self._admitted):
            ticket = self._tickets.pop(group_id)
            del self._ready[group_id]
            self._consumed.append(group_id)
            self.dispatch_lags.append(self.policy_version - ticket.dispatch_version)
            self.admitted_token_lags.extend(self.policy_version - target.policy_version
                                            for execution in group.executions for target in execution.targets)
        self.policy_version += 1
        self._steps_in_burst += 1
        self._inflight = self._admitted = None
        self._needs_publication = self._steps_in_burst == self.config.burst_steps
        return self.policy_version

    def publish(self) -> int:
        if not self._needs_publication or self._inflight is not None:
            raise RuntimeError("weights can be published only after the completed burst's final update")
        self.published_version = self.policy_version
        self._steps_in_burst = 0
        self._needs_publication = False
        self._dispatch_open = True
        return self.published_version

    def cancel_pending(self, reason: str) -> dict:
        """Close old-phase data at a published stage barrier, without training it.

        The caller must cancel corresponding backend jobs before mutating a
        model in another phase. Returned IDs identify those jobs. This reference
        scheduler owns data admission only; it cannot cancel external tools.
        Cancellation invalidates the stationary no-drop dispatch-lag identity.
        """
        if not reason:
            raise ValueError("stage cancellation requires an explicit recorded reason")
        if not self._dispatch_open or self._inflight is not None or self._needs_publication or self._steps_in_burst:
            raise RuntimeError("pending work can be cancelled only at a published boundary with no active update")
        ids = tuple(ticket.group_id for ticket in sorted(self._tickets.values(), key=lambda item: item.sequence))
        result = {
            "groups": len(ids), "ready_groups": len(self._ready),
            "quarantined_groups": len(self._quarantined),
            "execution_slots": sum(ticket.expected_executions for ticket in self._tickets.values()),
            "group_ids": ids, "reason": reason,
        }
        self.cancellation_history.extend((group_id, reason) for group_id in ids)
        self._tickets.clear()
        self._ready.clear()
        self._quarantined.clear()
        return result


BoundaryScheduler = FusionScheduler


@dataclass(frozen=True)
class CacheIdentity:
    policy_version: int
    tokenizer_id: str
    layout_id: str
    prefix_digest: str
    model_id: str = "policy"

    @classmethod
    def from_tokens(cls, policy_version: int, tokenizer_id: str, layout_id: str,
                    prefix_token_ids: Sequence[int], model_id: str = "policy") -> "CacheIdentity":
        _integer(policy_version, "cache policy version")
        if not tokenizer_id or not layout_id or not model_id:
            raise ValueError("cache identity requires tokenizer, layout and model identifiers")
        tokens = tuple(prefix_token_ids)
        for token in tokens:
            _integer(token, "prefix token")
        digest = hashlib.sha256(",".join(map(str, tokens)).encode("ascii")).hexdigest()
        return cls(policy_version, tokenizer_id, layout_id, digest, model_id)


def can_reuse_cache(cache: CacheIdentity | None, *, policy_version: int, tokenizer_id: str,
                    layout_id: str, prefix_token_ids: Sequence[int], model_id: str = "policy") -> bool:
    """Require exact version/model/tokenizer/layout/prefix identity.

    ``layout_id`` must cover cache dtype, attention format and parallel shards.
    Snapshot versions must be unique within ``model_id``; a production backend
    should bind ``model_id`` to a checkpoint/run digest, not a display label.
    """
    expected = CacheIdentity.from_tokens(policy_version, tokenizer_id, layout_id, prefix_token_ids, model_id)
    return cache is not None and cache == expected


@dataclass(frozen=True)
class RolloutRequest:
    request_id: str
    group_id: str
    policy_version: int
    prompt_token_ids: tuple[int, ...]
    max_new_tokens: int
    generated_token_ids: tuple[int, ...] = ()
    is_probe: bool = False
    tokenizer_id: str = "default-tokenizer"
    model_id: str = "policy"

    def __post_init__(self) -> None:
        object.__setattr__(self, "prompt_token_ids", tuple(self.prompt_token_ids))
        object.__setattr__(self, "generated_token_ids", tuple(self.generated_token_ids))
        _integer(self.policy_version, "request policy version")
        _integer(self.max_new_tokens, "maximum new tokens", 1)
        if not self.request_id or not self.group_id or not self.tokenizer_id or not self.model_id:
            raise ValueError("request and conditioning identities must be explicit")
        if len(self.generated_token_ids) > self.max_new_tokens:
            raise ValueError("request already exceeds its generation limit")
        for token in self.prompt_token_ids + self.generated_token_ids:
            _integer(token, "request token")


@dataclass(frozen=True)
class InferenceEngine:
    engine_id: str
    policy_version: int
    free_kv_tokens: int
    tokenizer_id: str = "default-tokenizer"
    layout_id: str = "default-layout"
    model_id: str = "policy"
    prefill_cost_per_token: float = 1.0
    decode_cost_per_token: float = 1.0
    caches: Mapping[str, CacheIdentity] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _integer(self.policy_version, "engine policy version")
        _integer(self.free_kv_tokens, "available KV token capacity")
        if not self.engine_id or not self.tokenizer_id or not self.layout_id or not self.model_id:
            raise ValueError("engine identity and formats must be explicit")
        if not math.isfinite(self.prefill_cost_per_token) or self.prefill_cost_per_token < 0:
            raise ValueError("prefill cost must be finite and nonnegative")
        if not math.isfinite(self.decode_cost_per_token) or self.decode_cost_per_token <= 0:
            raise ValueError("decode cost must be finite and positive")


@dataclass(frozen=True)
class ChunkDecision:
    request_id: str
    engine_id: str
    policy_version: int
    max_new_tokens: int
    estimated_remaining: int
    estimated_cost: float
    cache_action: str
    cache_identity: CacheIdentity


class LengthPredictor:
    """Conservative maximum completed-sibling length; no trained predictor."""

    def __init__(self) -> None:
        self.completed_max: dict[str, int] = {}
        self.observation_count: dict[str, int] = {}

    def observe(self, group_id: str, length: int) -> None:
        _integer(length, "completed generation length")
        self.completed_max[group_id] = max(length, self.completed_max.get(group_id, 0))
        self.observation_count[group_id] = self.observation_count.get(group_id, 0) + 1

    def predict(self, group_id: str, max_new_tokens: int) -> int:
        return min(max_new_tokens, self.completed_max.get(group_id, max_new_tokens))


class ChunkScheduler:
    """Version-frozen, memory-aware generation chunks with a fairness bound.

    First request per group acts as a length probe. Probes prefer less generated
    work, ordinary requests prefer estimated remaining length, and an age rule
    prevents starvation. Cost-per-token estimates choose among compatible
    engines; ``target_chunk_cost`` can also bound chunk size. These values are
    inputs from profiling, not simulated GPU performance claims.

    ``free_kv_tokens`` is the capacity available to the candidate INCLUDING its
    existing prefix. The caller supplies fresh engine snapshots after each
    decision and reserves the returned footprint before another concurrent
    call. The reference controller issues only one in-flight chunk per engine.
    """

    def __init__(self, policy_version: int = 0, mode: str = "strict_sync", chunk_tokens: int = 128,
                 max_wait_rounds: int = 16, target_chunk_cost: float = math.inf):
        _integer(policy_version, "policy version")
        _integer(chunk_tokens, "chunk token limit", 1)
        _integer(max_wait_rounds, "maximum waiting rounds", 1)
        if mode not in {"strict_sync", "bounded_async"}:
            raise ValueError("unknown scheduling mode")
        if math.isnan(target_chunk_cost) or target_chunk_cost <= 0:
            raise ValueError("target chunk cost must be positive")
        self.policy_version = policy_version
        self.mode = mode
        self.chunk_tokens = chunk_tokens
        self.max_wait_rounds = max_wait_rounds
        self.target_chunk_cost = target_chunk_cost
        self.predictor = LengthPredictor()
        self.requests: dict[str, RolloutRequest] = {}
        self.completed: set[str] = set()
        self._group_versions: dict[str, int] = {}
        self._group_signatures: dict[str, tuple] = {}
        self._probe_groups: set[str] = set()
        self._last_scheduled: dict[str, int] = {}
        self._round = 0
        self._inflight: dict[str, ChunkDecision] = {}

    def submit(self, request: RolloutRequest) -> None:
        if request.request_id in self.requests:
            raise ValueError("request identity reuse is disabled")
        if request.policy_version > self.policy_version:
            raise PolicyVersionMismatch(request.group_id, "request claims a future policy")
        if self.mode == "strict_sync" and request.policy_version != self.policy_version:
            raise PolicyVersionMismatch(request.group_id, "strict chunk request does not use the current frozen policy")
        if request.group_id in self._group_versions and self._group_versions[request.group_id] != request.policy_version:
            raise PolicyVersionMismatch(request.group_id, "chunks/replicates in a request group must retain one policy snapshot")
        signature = (request.prompt_token_ids, request.max_new_tokens, request.tokenizer_id, request.model_id)
        if request.group_id in self._group_signatures and self._group_signatures[request.group_id] != signature:
            raise ValueError("length-prediction groups must share prompt, generation limit, tokenizer and model")
        if request.is_probe and request.group_id in self._probe_groups:
            raise ValueError("a group already has its designated probe")
        if request.group_id not in self._probe_groups:
            request = replace(request, is_probe=True)
            self._probe_groups.add(request.group_id)
        self._group_versions[request.group_id] = request.policy_version
        self._group_signatures[request.group_id] = signature
        self.requests[request.request_id] = request
        self._last_scheduled[request.request_id] = self._round
        if len(request.generated_token_ids) == request.max_new_tokens:
            self.completed.add(request.request_id)
            self.predictor.observe(request.group_id, len(request.generated_token_ids))

    def update_policy_version(self, version: int) -> None:
        _integer(version, "new policy version")
        if version < self.policy_version:
            raise ValueError("policy versions cannot move backwards")
        if version == self.policy_version:
            return
        if self.mode == "strict_sync" and any(request_id not in self.completed for request_id in self.requests):
            raise RuntimeError("strict mode cannot publish weights while rollout chunks remain unfinished")
        self.policy_version = version

    def _rank(self, request: RolloutRequest) -> tuple:
        age = self._round - self._last_scheduled[request.request_id]
        if age >= self.max_wait_rounds:
            return (-1, -age, request.request_id)
        generated = len(request.generated_token_ids)
        if request.is_probe:
            return (0, generated, request.request_id)
        remaining = max(1, self.predictor.predict(request.group_id, request.max_new_tokens) - generated)
        return (1, -remaining, request.request_id)

    def choose(self, engines: Sequence[InferenceEngine]) -> ChunkDecision | None:
        self._round += 1
        occupied_engines = {decision.engine_id for decision in self._inflight.values()}
        requests = sorted((request for request in self.requests.values()
                           if request.request_id not in self.completed and request.request_id not in self._inflight),
                          key=self._rank)
        for request in requests:
            prefix = request.prompt_token_ids + request.generated_token_ids
            hard_remaining = request.max_new_tokens - len(request.generated_token_ids)
            candidates: list[ChunkDecision] = []
            for engine in engines:
                if engine.engine_id in occupied_engines:
                    continue
                if (engine.policy_version, engine.tokenizer_id, engine.model_id) != (
                        request.policy_version, request.tokenizer_id, request.model_id):
                    continue
                chunk = min(self.chunk_tokens, hard_remaining, engine.free_kv_tokens - len(prefix))
                if chunk < 1:
                    continue
                reuse = can_reuse_cache(engine.caches.get(request.request_id), policy_version=request.policy_version,
                                        tokenizer_id=engine.tokenizer_id, layout_id=engine.layout_id,
                                        prefix_token_ids=prefix, model_id=engine.model_id)
                prefill_cost = 0.0 if reuse else len(prefix) * engine.prefill_cost_per_token
                if math.isfinite(self.target_chunk_cost):
                    # Permit at least one token when an unavoidable prefill alone
                    # exceeds the target budget, so the soft cost bound cannot deadlock.
                    budget_tokens = max(1, int((self.target_chunk_cost - prefill_cost) / engine.decode_cost_per_token))
                    chunk = min(chunk, budget_tokens)
                estimated = max(1, self.predictor.predict(request.group_id, request.max_new_tokens)
                                - len(request.generated_token_ids))
                candidates.append(ChunkDecision(
                    request.request_id, engine.engine_id, request.policy_version, chunk, estimated,
                    prefill_cost + chunk * engine.decode_cost_per_token,
                    "reuse" if reuse else "re_prefill",
                    CacheIdentity.from_tokens(request.policy_version, engine.tokenizer_id,
                                              engine.layout_id, prefix, engine.model_id),
                ))
            if candidates:
                decision = min(candidates, key=lambda item: (item.estimated_cost / item.max_new_tokens,
                                -item.max_new_tokens, item.engine_id))
                self._inflight[request.request_id] = decision
                self._last_scheduled[request.request_id] = self._round
                return decision
        return None

    def finish_chunk(self, request_id: str, emitted_token_ids: Sequence[int],
                     behavior_policy_version: int, finished: bool = False) -> RolloutRequest:
        if request_id not in self._inflight:
            raise ValueError("request has no in-flight chunk to finish")
        decision = self._inflight[request_id]
        request = self.requests[request_id]
        if behavior_policy_version != decision.policy_version:
            raise PolicyVersionMismatch(request.group_id, "backend changed policy during a generation chunk")
        emitted = tuple(emitted_token_ids)
        for token in emitted:
            _integer(token, "emitted token")
        if len(emitted) > decision.max_new_tokens:
            raise ValueError("backend exceeded the reserved chunk token limit")
        if not emitted and not finished:
            raise ValueError("a nonterminal chunk must make progress")
        updated = replace(request, generated_token_ids=request.generated_token_ids + emitted)
        self.requests[request_id] = updated
        del self._inflight[request_id]
        if finished or len(updated.generated_token_ids) >= updated.max_new_tokens:
            self.completed.add(request_id)
            self.predictor.observe(updated.group_id, len(updated.generated_token_ids))
        return updated


@dataclass(frozen=True)
class ElasticState:
    """Execution-count state for a QwenGyre-inspired role transition guard."""

    cells: int
    training_cells: int
    per_cell_capacity: int
    standalone_capacity: int
    dispatched: int
    launched: int
    terminated: int

    def __post_init__(self) -> None:
        for name in ("cells", "per_cell_capacity"):
            _integer(getattr(self, name), name, 1)
        for name in ("training_cells", "standalone_capacity", "dispatched", "launched", "terminated"):
            _integer(getattr(self, name), name)
        if self.training_cells > self.cells or not self.terminated <= self.launched <= self.dispatched:
            raise ValueError("invalid elastic role/execution counters")
        if self.running > self.rollout_capacity:
            raise ValueError("current rollout capacity cannot serve all live executions")

    @property
    def rollout_capacity(self) -> int:
        return self.standalone_capacity + (self.cells - self.training_cells) * self.per_cell_capacity

    @property
    def running(self) -> int:
        return self.launched - self.terminated

    @property
    def outstanding(self) -> int:
        return self.dispatched - self.terminated

    @property
    def launch_count(self) -> int:
        return min(self.dispatched - self.launched, self.rollout_capacity - self.running)

    def can_add_training_cell(self, ready_groups: int, groups_per_update: int,
                              minimum_ready_fraction: float = 1.0, stable_for: float = 0.0,
                              required_stability: float = 0.0, groups_still_needed: int | None = None) -> bool:
        _integer(ready_groups, "ready groups")
        _integer(groups_per_update, "groups per update", 1)
        if not 0 < minimum_ready_fraction <= 1 or stable_for < 0 or required_stability < 0:
            raise ValueError("invalid supply/stability threshold")
        remaining = groups_per_update if groups_still_needed is None else groups_still_needed
        _integer(remaining, "groups still needed")
        supply = min(ready_groups, remaining) >= minimum_ready_fraction * groups_per_update
        return (self.training_cells < self.cells and supply and stable_for >= required_stability
                and self.outstanding <= self.rollout_capacity - self.per_cell_capacity)

    def add_training_cell(self, **supply_parameters) -> "ElasticState":
        if not self.can_add_training_cell(**supply_parameters):
            raise RuntimeError("role transition would strand work or lacks sustained training supply")
        return replace(self, training_cells=self.training_cells + 1)
