"""Train a real Hugging Face causal LM with the FusionRL learner.

Scope: single-turn tasks with a programmatic verifier (math, exact answers,
or any ``module:function`` grader), on one GPU or on data-parallel replicas
launched with ``torchrun`` (the model must fit on each GPU). Sharded
multi-node training of large models uses the pinned verl integration
instead. Agentic tasks need an environment runner plus the integrity layer
(attested images, monitor) and are rejected here rather than run unsafely.

What the runner guarantees:

* Rollouts record exactly the probabilities tokens were sampled from
  (temperature 1, full support), and each execution's random stream depends
  only on (seed, group, replicate), so resume and world size do not change it.
* Updates use complete groups in dispatch order, the execution-balanced loss
  with the ORIGINAL B*N denominator on every rank and microbatch, and a SUM
  all-reduce of gradients (equivalent to one process holding the batch).
* Before each strict-sync update the rollout/training log-prob gap is
  measured on the batch itself; a gap beyond ``parity`` aborts the run.
* Evaluator failures quarantine and regenerate the whole group; verifier
  hack flags become confirmed hacks with a penalty; a sustained hack rate
  halts the run.
* Checkpoints at publication boundaries hold weights, optimizer, RNG and the
  data/queue ledger; ``resume`` continues exactly.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import time
from typing import Any

import numpy as np

from . import __version__
from .backends.hf import HFPolicy, disable_dropout
from .integrity.monitor import GroupIntegrityDecision, IntegrityLedger
from .learner import LearnerBatch, learner_loss, shard_batch
from .pipeline import GroupSupervision, prepare_step
from .run_config import RunConfig
from .scheduler import FusionScheduler, GroupSpec, GroupTicket, SchedulerConfig
from .trajectory import Assessment, ExecutionRecord, GenerationEvent, QuarantinedGroup, RolloutGroup, TrajectoryPath
from .verifiers import load_verifier

MAX_GROUP_RETRIES = 3


class ParityError(RuntimeError):
    """Rollout and training engines disagree about the sampled tokens' probabilities."""


class IntegrityHalt(RuntimeError):
    """Sustained reward hacking; training stops instead of optimizing it."""


def _seed(*parts: Any) -> int:
    return int.from_bytes(hashlib.sha256(json.dumps(parts).encode()).digest()[:8], "little") & (2**63 - 1)


class _Dist:
    def __init__(self) -> None:
        import torch
        import torch.distributed as dist
        self.torch, self.dist = torch, dist
        self.world = int(os.environ.get("WORLD_SIZE", "1"))
        self.rank = int(os.environ.get("RANK", "0"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if self.world > 1 and not dist.is_initialized():
            dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")

    @property
    def enabled(self) -> bool:
        return self.world > 1

    def gather(self, obj: Any) -> list[Any]:
        if not self.enabled:
            return [obj]
        result: list[Any] = [None] * self.world
        self.dist.all_gather_object(result, obj)
        return result

    def sum_(self, tensor):
        if self.enabled:
            self.dist.all_reduce(tensor, op=self.dist.ReduceOp.SUM)
        return tensor

    def max_(self, tensor):
        if self.enabled:
            self.dist.all_reduce(tensor, op=self.dist.ReduceOp.MAX)
        return tensor

    def barrier(self) -> None:
        if self.enabled:
            self.dist.barrier()


def load_tasks(path: str | Path) -> list[dict]:
    tasks, seen = [], set()
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            task = json.loads(line)
            if not isinstance(task, dict) or not isinstance(task.get("task_id"), str) or "prompt" not in task:
                raise ValueError(f"line {number}: a task needs task_id and prompt")
            if task["task_id"] in seen:
                raise ValueError(f"line {number}: duplicate task_id {task['task_id']!r}")
            if "environment" in task or "docker_image" in task:
                raise ValueError(f"line {number}: agentic tasks need an attested environment runner, "
                                 "not the single-turn runner")
            seen.add(task["task_id"])
            tasks.append(task)
    if not tasks:
        raise ValueError("the training file has no tasks")
    return tasks


def _messages(task: dict, system_prompt: str) -> list[dict]:
    prompt = task["prompt"]
    messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else [dict(m) for m in prompt]
    if system_prompt and not any(m.get("role") == "system" for m in messages):
        messages.insert(0, {"role": "system", "content": system_prompt})
    return messages


def _resolve_device_and_precision(cfg: RunConfig):
    import torch
    device = cfg.model.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("model.device is cuda but CUDA is unavailable")
    precision = cfg.model.precision
    if precision == "auto":
        precision = "bf16-autocast" if device == "cuda" and torch.cuda.is_bf16_supported() else "fp32"
    if precision.startswith("bf16") and device == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("bf16 requested on a GPU without bf16 support")
    return device, precision


def load_model(cfg: RunConfig, device: str, precision: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import transformers
    # Always pass the dtype: a checkpoint's own config often says bf16, which
    # would silently replace fp32 master weights. ``dtype`` replaced
    # ``torch_dtype`` in transformers 4.56.
    major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
    dtype_key = "dtype" if (major, minor) >= (4, 56) else "torch_dtype"
    kwargs: dict[str, Any] = {dtype_key: torch.bfloat16 if precision == "bf16" else torch.float32,
                              "trust_remote_code": cfg.model.trust_remote_code}
    if cfg.model.attn_implementation:
        kwargs["attn_implementation"] = cfg.model.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(cfg.model.path, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.path, trust_remote_code=cfg.model.trust_remote_code)
    return model.to(device), tokenizer


class Runner:
    def __init__(self, cfg: RunConfig, output: str | Path, *, model: Any = None, tokenizer: Any = None,
                 progress: bool = True):
        import torch
        cfg.validate()
        self.cfg, self.output, self.progress = cfg, Path(output), progress
        self.torch = torch
        self.dist = _Dist()
        self.device, self.precision = _resolve_device_and_precision(cfg)
        if self.device == "cuda":
            torch.cuda.set_device(self.dist.local_rank)
            self.device = f"cuda:{self.dist.local_rank}"
        if model is None:
            model, tokenizer = load_model(cfg, self.device, self.precision)
        self.model = model.to(self.device)
        if cfg.model.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        self.model.train()
        disable_dropout(self.model)
        autocast = torch.bfloat16 if self.precision == "bf16-autocast" else None
        self.policy = HFPolicy(self.model, tokenizer, max_tokens_per_microbatch=cfg.learner.max_tokens_per_microbatch,
                               autocast_dtype=autocast)
        self.tokenizer = tokenizer
        self.verify = load_verifier(cfg.data.verifier)
        self.tasks = load_tasks(cfg.data.train)
        self.prompt_ids = {}
        for task in self.tasks:
            ids = self.policy.encode_chat(_messages(task, cfg.data.system_prompt))
            if len(ids) > cfg.data.max_prompt_tokens:
                raise ValueError(f"task {task['task_id']!r} prompt has {len(ids)} tokens "
                                 f"> data.max_prompt_tokens={cfg.data.max_prompt_tokens}")
            self.prompt_ids[task["task_id"]] = ids
        self.task_by_id = {task["task_id"]: task for task in self.tasks}
        o = cfg.optim
        fused = self.device.startswith("cuda")
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=o.learning_rate, betas=(o.beta1, o.beta2),
                                           eps=o.eps, weight_decay=o.weight_decay, fused=fused or None)
        self.ledger = IntegrityLedger(cfg.integrity.task_revoke_rate, cfg.integrity.task_min_executions,
                                      cfg.integrity.halt_rate, cfg.integrity.halt_window)
        # Mutable run state (all of it is checkpointed).
        self.step = 0
        self.policy_version = 0
        self.group_counter = 0
        self.quarantined = 0
        self.epoch, self.cursor = 0, 0
        self.order = self._epoch_order(0)
        self.records: dict[str, RolloutGroup] = {}
        self.meta: dict[str, dict] = {}
        self.controller = self._controller(0)

    # -------------------------------------------------------------- helpers

    def _controller(self, version: int) -> FusionScheduler:
        s = self.cfg.schedule
        return FusionScheduler(SchedulerConfig(s.mode, s.groups_per_update, s.executions_per_group, s.burst_steps,
                                               s.extra_dispatch_groups, s.max_pending_groups, s.max_policy_lag,
                                               s.jmax), policy_version=version)

    def _epoch_order(self, epoch: int) -> list[str]:
        ids = [task["task_id"] for task in self.tasks]
        random.Random(_seed(self.cfg.run.seed, "epoch", epoch)).shuffle(ids)
        return ids

    def _next_task(self) -> str:
        if self.cursor >= len(self.order):
            self.epoch, self.cursor = self.epoch + 1, 0
            self.order = self._epoch_order(self.epoch)
        task_id = self.order[self.cursor]
        self.cursor += 1
        return task_id

    def _conditioning_key(self, task_id: str) -> str:
        payload = {"task": task_id, "prompt": list(self.prompt_ids[task_id]), "verifier": self.cfg.data.verifier,
                   "route": self.cfg.reward.route, "tokenizer": self.policy.fingerprint,
                   "max_new_tokens": self.cfg.rollout.max_new_tokens}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def _log(self, message: str) -> None:
        if self.progress and self.dist.rank == 0:
            print(message, flush=True)

    # -------------------------------------------------------------- rollout

    def _generate(self, tickets: list[GroupTicket]) -> dict[str, list[dict]]:
        """Each rank samples its share of groups; results are exchanged."""
        torch = self.torch
        mine = [t for index, t in enumerate(tickets) if index % self.dist.world == self.dist.rank]
        rows = [(t, r) for t in mine for r in range(t.expected_executions)]
        local: dict[str, list[dict]] = {t.group_id: [None] * t.expected_executions for t in mine}
        batch = self.cfg.rollout.generation_batch_size
        for start in range(0, len(rows), batch):
            chunk = rows[start:start + batch]
            prompts = [self.prompt_ids[self.meta[t.group_id]["task_id"]] for t, _ in chunk]
            generators = [torch.Generator(device=self.device).manual_seed(
                _seed(self.cfg.run.seed, "rollout", t.group_id, r)) for t, r in chunk]
            samples = self.policy.generate(prompts, max_new_tokens=self.cfg.rollout.max_new_tokens,
                                           generator=generators)
            for (ticket, replicate), (tokens, logps) in zip(chunk, samples):
                task = self.task_by_id[self.meta[ticket.group_id]["task_id"]]
                text = self.tokenizer.decode(list(tokens), skip_special_tokens=True)
                verdict = self.verify(task, text)
                local[ticket.group_id][replicate] = {
                    "tokens": list(tokens), "logps": list(logps), "score": verdict.score, "valid": verdict.valid,
                    "hack": verdict.hack, "detail": verdict.detail,
                    "truncated": not tokens or tokens[-1] not in self.policy.eos_ids}
        merged: dict[str, list[dict]] = {}
        for part in self.dist.gather(local):
            merged.update(part)
        return merged

    def _build_group(self, ticket: GroupTicket, results: list[dict]) -> RolloutGroup:
        prompt = self.prompt_ids[self.meta[ticket.group_id]["task_id"]]
        executions = []
        for replicate, result in enumerate(results):
            execution_id = f"{ticket.group_id}:e{replicate}"
            tokens = tuple(result["tokens"])
            event = GenerationEvent(f"{execution_id}:r0", prompt, tokens, tuple(result["logps"]),
                                    tuple(ticket.dispatch_version for _ in tokens), tuple("policy" for _ in tokens))
            artifact = hashlib.sha256(json.dumps([list(prompt), list(tokens)]).encode()).hexdigest()
            valid = bool(result["valid"])
            executions.append(ExecutionRecord(
                execution_id, ticket.group_id, (event,), (TrajectoryPath(f"{execution_id}:main", (event.event_id,)),),
                Assessment(float(result["score"]) if valid else None, valid, artifact, result["detail"]),
                status="completed" if valid else "evaluator_failed", conditioning_key=ticket.conditioning_key))
        return RolloutGroup(ticket.group_id, tuple(executions), ticket.expected_executions,
                            ticket.dispatch_version, ticket.conditioning_key)

    def _complete(self, tickets: list[GroupTicket]) -> None:
        pending = list(tickets)
        for _ in range(MAX_GROUP_RETRIES + 1):
            if not pending:
                return
            results = self._generate(pending)
            retry = []
            for ticket in pending:
                group = self._build_group(ticket, results[ticket.group_id])
                self.meta[ticket.group_id]["results"] = results[ticket.group_id]
                try:
                    self.controller.complete(group)
                    self.records[ticket.group_id] = group
                except QuarantinedGroup as error:
                    self.quarantined += 1
                    meta = self.meta.pop(ticket.group_id)
                    meta["retries"] = meta.get("retries", 0) + 1
                    new_id = f"{ticket.group_id.split('-retry')[0]}-retry{meta['retries']}"
                    self._log(f"quarantined {ticket.group_id}: {error.reason}; retrying as {new_id}")
                    self.meta[new_id] = meta
                    retry.append(self.controller.replace_quarantined(ticket.group_id, new_id))
            pending = retry
        raise RuntimeError(f"{len(pending)} groups failed evaluation {MAX_GROUP_RETRIES + 1} times")

    def _refill(self) -> None:
        missing = self.controller.config.target_depth - self.controller.pending_count
        specs = []
        for _ in range(missing):
            task_id = self._next_task()
            group_id = f"g{self.group_counter:08d}"
            self.group_counter += 1
            self.meta[group_id] = {"task_id": task_id}
            specs.append(GroupSpec(group_id, self._conditioning_key(task_id)))
        self._complete(list(self.controller.dispatch(specs)))

    # --------------------------------------------------------------- update

    def _assign_rows(self, batch: LearnerBatch) -> list[int]:
        """Whole groups per rank, balanced by token count, deterministic."""
        groups: dict[str, list[int]] = {}
        for row, group_id in enumerate(batch.group_ids):
            groups.setdefault(group_id, []).append(row)
        load = [0] * self.dist.world
        owner: dict[str, int] = {}
        for group_id, rows in sorted(groups.items(), key=lambda kv: (-sum(len(batch.targets[r]) for r in kv[1]), kv[0])):
            rank = min(range(self.dist.world), key=lambda k: (load[k], k))
            owner[group_id] = rank
            load[rank] += sum(len(batch.targets[r]) + 1 for r in rows)
        return [row for row, group_id in enumerate(batch.group_ids) if owner[group_id] == self.dist.rank]

    def _microbatches(self, batch: LearnerBatch, rows: list[int]) -> list[list[int]]:
        budget = self.cfg.learner.max_tokens_per_microbatch
        result, current, used = [], [], 0
        for row in rows:
            size = max((t.scoring_view()[1] + 1 for t in batch.targets[row]), default=0)
            if current and used + size > budget:
                result.append(current)
                current, used = [], 0
            current.append(row)
            used += size
        if current:
            result.append(current)
        return result

    def _update(self) -> dict[str, Any]:
        torch = self.torch
        admitted = self.controller.begin_update()
        if admitted is None:
            raise RuntimeError("the controller could not form a complete quota batch")
        supervision = {}
        hacks_total = 0
        for group in admitted:
            results = self.meta[group.group_id]["results"]
            hacks = tuple(bool(r["hack"]) for r in results)
            hacks_total += sum(hacks)
            decision = GroupIntegrityDecision(hacks, False, "verifier", ())
            self.ledger.record(self.meta[group.group_id]["task_id"], decision)
            supervision[group.group_id] = GroupSupervision(
                execution_ids=tuple(e.execution_id for e in group.executions),
                artifact_ids=tuple(e.assessment.artifact_id for e in group.executions),
                conditioning_key=group.conditioning_key, route=self.cfg.reward.route,
                scores=tuple(group.rewards) if self.cfg.reward.route == "scalar" else None,
                confirmed_hacks=hacks, hack_penalty=self.cfg.reward.hack_penalty)
        if self.ledger.should_halt:
            raise IntegrityHalt(f"hack rate {self.ledger.window_hack_rate:.3f} over the last "
                                f"{self.cfg.integrity.halt_window} executions")
        prepared = prepare_step(admitted, supervision)
        batch = prepared.batch
        global_count = len(batch.execution_ids)
        self.optimizer.zero_grad(set_to_none=True)
        totals = torch.zeros(6, dtype=torch.float64, device=self.device)  # loss, tokens, |gap|, clipped, nonempty, rows
        gap_max = torch.zeros(1, dtype=torch.float64, device=self.device)
        rows = self._assign_rows(batch)
        for micro in self._microbatches(batch, rows):
            sub = shard_batch(batch, micro)
            if sub.target_count == 0:
                continue
            loss, stat = learner_loss(sub, sequence_log_prob_fn=self.policy.sequence_log_probs,
                                      config=self.cfg.loss, global_execution_count=global_count)
            loss.backward()
            nonempty = stat["execution_count"] - stat["empty_executions"]
            totals += torch.tensor([stat["loss"], stat["target_tokens"], stat["abs_log_ratio_sum"],
                                    stat["clipped_executions"], nonempty, len(micro)], dtype=torch.float64,
                                   device=self.device)
            gap_max = torch.maximum(gap_max, torch.tensor([stat["max_abs_log_ratio"]], dtype=torch.float64,
                                                          device=self.device))
        self.dist.sum_(totals)
        self.dist.max_(gap_max)
        loss_value, tokens, gap_sum, clipped, nonempty, _ = totals.tolist()
        mean_gap = gap_sum / tokens if tokens else 0.0
        strict = self.cfg.schedule.mode == "strict_sync"
        if strict and tokens and (mean_gap > self.cfg.parity.max_mean_abs_gap or
                                  gap_max.item() > self.cfg.parity.max_abs_gap):
            raise ParityError(f"rollout/training log-prob gap mean={mean_gap:.4g} max={gap_max.item():.4g} exceeds "
                              f"parity limits ({self.cfg.parity.max_mean_abs_gap}, {self.cfg.parity.max_abs_gap})")
        if strict and nonempty >= 8 and clipped / nonempty > self.cfg.parity.max_strict_clip_fraction:
            raise ParityError(f"{clipped / nonempty:.0%} of executions were clipped before any update; rollout and "
                              "training numerics disagree by more than the execution clip range")
        params = [p for p in self.model.parameters() if p.requires_grad]
        if self.dist.enabled:
            for p in params:
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
                self.dist.sum_(p.grad)
        signal = any(abs(a) > 1e-12 and t for a, t in zip(batch.advantages, batch.targets))
        grad_norm = 0.0
        applied = False
        if signal:
            grad_norm = float(torch.nn.utils.clip_grad_norm_(params, self.cfg.optim.gradient_clip).item())
            if not np.isfinite(grad_norm):
                raise FloatingPointError("non-finite gradient; update not committed")
            if grad_norm > 0:
                warmup = self.cfg.optim.warmup_steps
                scale = min(1.0, (self.step + 1) / warmup) if warmup else 1.0
                for group in self.optimizer.param_groups:
                    group["lr"] = self.cfg.optim.learning_rate * scale
                self.optimizer.step()
                applied = True
        self.optimizer.zero_grad(set_to_none=True)
        self.controller.commit_update()
        results = [r for g in admitted for r in self.meta[g.group_id]["results"]]
        lags = [self.controller.policy_version - 1 - t.policy_version for g in admitted for e in g.executions
                for t in e.targets]
        for group in admitted:
            self.records.pop(group.group_id, None)
            self.meta.pop(group.group_id, None)
        return {
            "step": self.step + 1, "policy_version": self.controller.policy_version,
            "reward_mean": float(np.mean([r["score"] for r in results])),
            "zero_signal_groups": sum(not g.diagnostics["has_learning_signal"] for g in prepared.grading.values()),
            "confirmed_hacks": hacks_total, "loss": loss_value, "target_tokens": tokens,
            "clip_fraction": clipped / nonempty if nonempty else 0.0,
            "rollout_train_gap_mean": mean_gap, "rollout_train_gap_max": float(gap_max.item()),
            "response_tokens_mean": float(np.mean([len(r["tokens"]) for r in results])),
            "truncated_fraction": float(np.mean([r["truncated"] for r in results])),
            "behavior_lag_max": max(lags, default=0), "optimizer_applied": applied, "gradient_norm": grad_norm,
            "learning_rate": self.optimizer.param_groups[0]["lr"],
        }

    # ----------------------------------------------------------- checkpoints

    def _state(self) -> dict:
        return {"step": self.step, "policy_version": self.controller.policy_version,
                "group_counter": self.group_counter, "epoch": self.epoch, "cursor": self.cursor,
                "consumed_groups": len(self.controller.consumed_groups), "ledger": self.ledger.state_dict(),
                "fusionrl_version": __version__, "tokenizer_fingerprint": self.policy.fingerprint,
                "config": self.cfg.to_dict()}

    def save_checkpoint(self) -> Path | None:
        torch = self.torch
        root = self.output / "checkpoints"
        final = root / f"step-{self.step:08d}"
        if self.dist.rank == 0:
            temporary = root / f".tmp-step-{self.step:08d}"
            shutil.rmtree(temporary, ignore_errors=True)
            temporary.mkdir(parents=True)
            self.model.save_pretrained(temporary / "model", safe_serialization=True)
            self.tokenizer.save_pretrained(temporary / "model")
            torch.save(self.optimizer.state_dict(), temporary / "optimizer.pt")
            (temporary / "state.json").write_text(json.dumps(self._state(), indent=2) + "\n")
            shutil.rmtree(final, ignore_errors=True)
            temporary.rename(final)
            (root / "latest").write_text(final.name + "\n")
            kept = sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("step-"))
            for old in kept[:-self.cfg.checkpoint.keep_last]:
                shutil.rmtree(old)
        self.dist.barrier()
        return final

    def load_checkpoint(self) -> bool:
        torch = self.torch
        root = self.output / "checkpoints"
        if not (root / "latest").exists():
            return False
        path = root / (root / "latest").read_text().strip()
        state = json.loads((path / "state.json").read_text())
        if state["tokenizer_fingerprint"] != self.policy.fingerprint:
            raise ValueError("checkpoint was trained with a different tokenizer")
        _load_weights(self.model, path / "model")
        self.optimizer.load_state_dict(torch.load(path / "optimizer.pt", map_location=self.device, weights_only=True))
        self.step, self.group_counter = state["step"], state["group_counter"]
        self.epoch, self.cursor = state["epoch"], state["cursor"]
        self.order = self._epoch_order(self.epoch)
        self.ledger.load_state_dict(state["ledger"])
        self.controller = self._controller(state["policy_version"])
        self.records.clear()
        self.meta.clear()
        return True

    # ------------------------------------------------------------------ run

    def run(self, *, resume: bool = False) -> dict[str, Any]:
        self.output.mkdir(parents=True, exist_ok=True)
        metrics_path = self.output / "metrics.jsonl"
        resumed = self.load_checkpoint() if resume else False
        if not resumed and self.dist.rank == 0:
            if metrics_path.exists() or (self.output / "checkpoints").exists():
                raise FileExistsError("output has a previous run; pass resume=True or choose a fresh directory")
            (self.output / "run_config.json").write_text(json.dumps(self.cfg.to_dict(), indent=2) + "\n")
        if resumed and self.dist.rank == 0 and metrics_path.exists():
            lines = [line for line in metrics_path.read_text().splitlines() if json.loads(line)["step"] <= self.step]
            metrics_path.write_text("".join(line + "\n" for line in lines))
        self.dist.barrier()
        started = time.perf_counter()
        last: dict[str, Any] = {}
        if self.step < self.cfg.run.steps:
            self._refill()
        while self.step < self.cfg.run.steps:
            tick = time.perf_counter()
            last = self._update()
            self.step += 1
            last["step_seconds"] = time.perf_counter() - tick
            if self.controller.needs_publication:
                # In-process rollouts read the live weights: publication is the version bump.
                self.controller.publish()
                if self.step % self.cfg.checkpoint.every == 0 or self.step == self.cfg.run.steps:
                    self.save_checkpoint()
                if self.step < self.cfg.run.steps:
                    self._refill()
            if self.dist.rank == 0:
                with metrics_path.open("a") as handle:
                    handle.write(json.dumps(last) + "\n")
            self._log(f"step {self.step}/{self.cfg.run.steps} reward={last['reward_mean']:.3f} "
                      f"loss={last['loss']:.4f} gap={last['rollout_train_gap_mean']:.2e}")
        summary = {"implementation": f"FusionRL {__version__}", "steps": self.step, "resumed": resumed,
                   "world_size": self.dist.world, "device": self.device, "precision": self.precision,
                   "elapsed_seconds": time.perf_counter() - started, "last": last,
                   "quarantined_groups_retried": self.quarantined, "revoked_tasks": dict(self.ledger.revoked)}
        if self.dist.rank == 0:
            (self.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        return summary


def _load_weights(model: Any, directory: Path) -> None:
    from safetensors.torch import load_file
    index = directory / "model.safetensors.index.json"
    if index.exists():
        files = sorted(set(json.loads(index.read_text())["weight_map"].values()))
    else:
        files = ["model.safetensors"]
    state: dict = {}
    for name in files:
        state.update(load_file(str(directory / name)))
    missing, unexpected = model.load_state_dict(state, strict=False)
    tied = set(getattr(model, "_tied_weights_keys", None) or [])
    if unexpected or (set(missing) - tied):
        raise ValueError(f"checkpoint mismatch: missing={sorted(set(missing) - tied)[:5]} unexpected={unexpected[:5]}")
    if hasattr(model, "tie_weights"):
        model.tie_weights()


def run_training(cfg: RunConfig, output: str | Path, *, resume: bool = False, progress: bool = True,
                 model: Any = None, tokenizer: Any = None) -> dict[str, Any]:
    return Runner(cfg, output, model=model, tokenizer=tokenizer, progress=progress).run(resume=resume)
