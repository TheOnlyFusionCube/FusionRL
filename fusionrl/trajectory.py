"""Immutable rollout records and provenance-preserving training admission.

This module is deliberately independent of a tokenizer or inference engine. The
caller records the *actual* input/output IDs and behavior probabilities. It must
not reconstruct them from rendered chat messages or the latest model weights.

Inspired by QwenGyre Sections 5 and Appendix B, with explicit FusionRL changes:
deterministic path selection, stale-token masking before all loss statistics,
and whole-group quarantine for invalid infrastructure/evaluation outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable


ROLE_PRIORITY = {"main": 0, "main_summary": 1, "subagent": 2, "subagent_summary": 3}
PROVENANCE = frozenset({"policy", "system", "user", "tool", "evaluator", "harness"})
TERMINAL_STATUSES = frozenset({"completed", "timed_out", "cancelled", "infra_failed", "evaluator_failed"})


class AdmissionError(ValueError):
    """An execution group cannot currently be used for an update."""

    def __init__(self, group_id: str, reason: str):
        self.group_id = group_id
        self.reason = reason
        super().__init__(f"Group {group_id!r}: {reason}")


class IncompleteGroup(AdmissionError):
    """Original replicate slots have not all reached terminal outcomes."""


class QuarantinedGroup(AdmissionError):
    """An invalid outcome invalidates the entire group's reward comparison."""


class PolicyVersionMismatch(AdmissionError):
    """Policy provenance violates the execution mode's version contract."""


def _nonnegative_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def _tokens(values: Iterable[int], name: str) -> tuple[int, ...]:
    result = tuple(values)
    for token in result:
        _nonnegative_int(token, name)
    return result


@dataclass(frozen=True)
class GenerationEvent:
    """One immutable model-request record, not a reconstructed chat turn.

    ``event_id`` identifies one sampling event within its execution. Shared
    prefixes reuse that ID. Independently sampled equal text uses different IDs.
    ``provenance`` is required for every output position; a role alone is never
    evidence that a token came from the policy. Non-policy positions may have
    ``None`` behavior metadata and are always masked from the objective.
    """

    event_id: str
    input_token_ids: tuple[int, ...]
    output_token_ids: tuple[int, ...]
    behavior_log_probs: tuple[float | None, ...]
    policy_versions: tuple[int | None, ...]
    provenance: tuple[str, ...]
    role: str = "main"
    branch_id: str = "main"
    actor_id: str = "policy"

    def __post_init__(self) -> None:
        if not self.event_id:
            raise ValueError("event_id must identify the original generation event")
        object.__setattr__(self, "input_token_ids", _tokens(self.input_token_ids, "input token"))
        object.__setattr__(self, "output_token_ids", _tokens(self.output_token_ids, "output token"))
        for name in ("behavior_log_probs", "policy_versions", "provenance"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        n = len(self.output_token_ids)
        if any(len(getattr(self, name)) != n for name in ("behavior_log_probs", "policy_versions", "provenance")):
            raise ValueError("output IDs, behavior log probabilities, policy versions and provenance must align")
        if self.role not in ROLE_PRIORITY:
            raise ValueError(f"Unknown trajectory role: {self.role!r}")
        if not self.branch_id or not self.actor_id:
            raise ValueError("branch_id and actor_id must be explicit")
        for origin, logp, version in zip(self.provenance, self.behavior_log_probs, self.policy_versions):
            if origin not in PROVENANCE:
                raise ValueError(f"Unknown output provenance: {origin!r}")
            if origin == "policy":
                if logp is None or not math.isfinite(logp) or logp > 0:
                    raise ValueError("policy targets require finite, nonpositive actual behavior log probabilities")
                if version is None:
                    raise ValueError("policy targets require actual behavior policy versions")
                _nonnegative_int(version, "behavior policy version")


@dataclass(frozen=True)
class TrajectoryPath:
    path_id: str
    event_ids: tuple[str, ...]
    role: str = "main"
    branch_id: str = "main"

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_ids", tuple(self.event_ids))
        if not self.path_id or not self.branch_id:
            raise ValueError("path_id and branch_id must be explicit")
        if self.role not in ROLE_PRIORITY:
            raise ValueError(f"Unknown trajectory role: {self.role!r}")
        if len(set(self.event_ids)) != len(self.event_ids):
            raise ValueError("a path cannot repeat the same generation event")


@dataclass(frozen=True)
class Assessment:
    score: float | None
    valid: bool
    artifact_id: str
    reason: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.valid, bool):
            raise ValueError("assessment validity must be an explicit boolean")
        if self.valid and (self.score is None or not math.isfinite(self.score)):
            raise ValueError("a valid assessment needs a finite scalar score")
        if self.valid and not self.artifact_id:
            raise ValueError("a valid assessment must identify the assessed artifact/state")


@dataclass(frozen=True)
class ExecutionRecord:
    execution_id: str
    group_id: str
    events: tuple[GenerationEvent, ...]
    paths: tuple[TrajectoryPath, ...]
    assessment: Assessment | None
    status: str = "completed"
    conditioning_key: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "events", tuple(self.events))
        object.__setattr__(self, "paths", tuple(self.paths))
        if not self.execution_id or not self.group_id:
            raise ValueError("execution_id and group_id must be explicit")
        if self.status not in TERMINAL_STATUSES | {"running", "queued"}:
            raise ValueError(f"Unknown execution status: {self.status!r}")
        event_ids = {event.event_id for event in self.events}
        if len(event_ids) != len(self.events):
            raise ValueError("generation event IDs must be unique within an execution")
        if len({path.path_id for path in self.paths}) != len(self.paths):
            raise ValueError("path IDs must be unique within an execution")
        for path in self.paths:
            if not set(path.event_ids) <= event_ids:
                raise ValueError(f"path {path.path_id!r} references an unknown generation event")


@dataclass(frozen=True)
class RolloutGroup:
    group_id: str
    executions: tuple[ExecutionRecord, ...]
    expected_executions: int
    dispatch_version: int = 0
    conditioning_key: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "executions", tuple(self.executions))
        _nonnegative_int(self.dispatch_version, "dispatch version")
        _nonnegative_int(self.expected_executions, "expected executions")
        if self.expected_executions < 1:
            raise ValueError("expected_executions must be positive")
        if not self.group_id:
            raise ValueError("group_id must include a unique dispatch group identity")
        if len({execution.execution_id for execution in self.executions}) != len(self.executions):
            raise ValueError("execution IDs must be unique within a group")


@dataclass(frozen=True)
class TrainingTarget:
    target_id: tuple[str, int]
    token_id: int
    context: tuple[int, ...]
    behavior_logp: float
    policy_version: int
    execution_id: str
    group_id: str
    path_id: str
    role: str
    branch_id: str
    actor_id: str

    @property
    def behavior_log_prob(self) -> float:
        """Readable alias for callers using the singular long field name."""
        return self.behavior_logp


@dataclass(frozen=True)
class AdmittedExecution:
    execution_id: str
    group_id: str
    targets: tuple[TrainingTarget, ...]
    assessment: Assessment
    original_index: int
    selected_paths: tuple[str, ...]
    masked_provenance: int = 0
    masked_stale: int = 0
    masked_shared_occurrences: int = 0
    masked_by_path_cap: int = 0

    @property
    def reward(self) -> float:
        return float(self.assessment.score)


@dataclass(frozen=True)
class AdmittedGroup:
    group_id: str
    executions: tuple[AdmittedExecution, ...]
    expected_executions: int
    dispatch_version: int
    conditioning_key: str

    @property
    def rewards(self) -> tuple[float, ...]:
        return tuple(execution.reward for execution in self.executions)

    @property
    def target_count(self) -> int:
        return sum(len(execution.targets) for execution in self.executions)


def policy_versions(group: RolloutGroup) -> tuple[int, ...]:
    """Actual policy versions, including tokens that a later mask might exclude."""
    return tuple(
        version
        for execution in group.executions
        for event in execution.events
        for version, origin in zip(event.policy_versions, event.provenance)
        if origin == "policy"
    )


def validate_complete_group(group: RolloutGroup) -> str:
    """Validate all original replicate slots and return their conditioning key.

    Empty keys support small examples. A production caller should populate a
    fingerprint of task revision, prompt, effort, harness, rubric and config.
    Distinct explicit keys are always rejected, even if group IDs match.
    """
    if len(group.executions) < group.expected_executions:
        raise IncompleteGroup(group.group_id, "not all original replicate slots are present")
    if len(group.executions) > group.expected_executions:
        raise QuarantinedGroup(group.group_id, "extra executions cannot silently replace original replicate slots")
    keys = {group.conditioning_key} if group.conditioning_key else set()
    for execution in group.executions:
        if execution.group_id != group.group_id:
            raise QuarantinedGroup(group.group_id, "execution belongs to a different dispatch group")
        if execution.conditioning_key:
            keys.add(execution.conditioning_key)
        if execution.status not in TERMINAL_STATUSES:
            raise IncompleteGroup(group.group_id, "an original execution has not terminated")
        if execution.status in {"infra_failed", "evaluator_failed"}:
            raise QuarantinedGroup(group.group_id, "infrastructure/evaluator failure is not a policy reward")
        if execution.assessment is None or not execution.assessment.valid:
            raise QuarantinedGroup(group.group_id, "at least one original outcome lacks a valid assessment")
    if len(keys) > 1:
        raise QuarantinedGroup(group.group_id, "replicates have incompatible conditioning keys")
    return next(iter(keys), "")


def _admit_execution(execution: ExecutionRecord, index: int, current_version: int,
                     max_policy_lag: int, jmax: int) -> AdmittedExecution:
    events = {event.event_id: event for event in execution.events}
    eligible: set[tuple[str, int]] = set()
    provenance_count = stale_count = 0
    for event in execution.events:
        for offset, (origin, version) in enumerate(zip(event.provenance, event.policy_versions)):
            if origin != "policy":
                provenance_count += 1
                continue
            if version > current_version:
                raise PolicyVersionMismatch(execution.group_id, "behavior tokens come from a future trainer version")
            if current_version - version > max_policy_lag:
                stale_count += 1
                continue
            eligible.add((event.event_id, offset))

    ids_by_path = {
        path.path_id: tuple((event_id, offset)
                            for event_id in path.event_ids
                            for offset in range(len(events[event_id].output_token_ids))
                            if (event_id, offset) in eligible)
        for path in execution.paths
    }
    available = set(eligible)
    candidates = list(execution.paths)
    selected_paths: list[str] = []
    targets: list[TrainingTarget] = []
    while candidates and len(selected_paths) < jmax:
        remaining = {path.path_id: tuple(tid for tid in ids_by_path[path.path_id] if tid in available)
                     for path in candidates}
        candidates = [path for path in candidates if remaining[path.path_id]]
        if not candidates:
            break
        # Deterministic modification of QwenGyre's probability-proportional draw.
        path = min(candidates, key=lambda candidate: (ROLE_PRIORITY[candidate.role],
                   -len(remaining[candidate.path_id]), candidate.path_id))
        selected_paths.append(path.path_id)
        for event_id, offset in remaining[path.path_id]:
            event = events[event_id]
            targets.append(TrainingTarget(
                target_id=(event_id, offset),
                token_id=event.output_token_ids[offset],
                context=event.input_token_ids + event.output_token_ids[:offset],
                behavior_logp=float(event.behavior_log_probs[offset]),
                policy_version=event.policy_versions[offset],
                execution_id=execution.execution_id,
                group_id=execution.group_id,
                path_id=path.path_id,
                role=event.role,
                branch_id=event.branch_id,
                actor_id=event.actor_id,
            ))
        available.difference_update(remaining[path.path_id])
        candidates.remove(path)
    unique_covered = {tid for ids in ids_by_path.values() for tid in ids}
    shared_occurrences = sum(len(ids) for ids in ids_by_path.values()) - len(unique_covered)
    return AdmittedExecution(
        execution.execution_id, execution.group_id, tuple(targets), execution.assessment,
        index, tuple(selected_paths), provenance_count, stale_count,
        shared_occurrences, len(available),
    )


def admit_group(group: RolloutGroup, current_version: int, max_policy_lag: int = 0,
                jmax: int = 5) -> AdmittedGroup:
    """Admit complete original groups with fixed target sets and empty slots.

    Stale/provenance masks precede path selection, sequence ratios and all loss
    denominators. They remove tokens, not execution slots or valid rewards.
    ``FusionScheduler`` imposes the stronger all-token version check in strict
    synchronous mode. No advantage normalization or policy loss occurs here.
    """
    _nonnegative_int(current_version, "current version")
    _nonnegative_int(max_policy_lag, "maximum policy lag")
    if isinstance(jmax, bool) or not isinstance(jmax, int) or jmax < 1:
        raise ValueError("jmax must be a positive integer")
    key = validate_complete_group(group)
    executions = tuple(_admit_execution(execution, index, current_version, max_policy_lag, jmax)
                       for index, execution in enumerate(group.executions))
    return AdmittedGroup(group.group_id, executions, group.expected_executions, group.dispatch_version, key)
