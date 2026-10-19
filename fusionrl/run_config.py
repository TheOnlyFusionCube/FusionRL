"""Strict configuration for training a real Hugging Face causal LM."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
import json
import math
from pathlib import Path
from typing import Any

from .config import Schedule, _strict_dataclass
from .objective import ExecutionLossConfig


@dataclass(frozen=True)
class ModelSpec:
    path: str = ""
    precision: str = "auto"            # auto | fp32 | bf16-autocast | bf16
    device: str = "auto"               # auto | cpu | cuda
    gradient_checkpointing: bool = False
    attn_implementation: str = ""      # e.g. sdpa, flash_attention_2; empty = model default
    trust_remote_code: bool = False


@dataclass(frozen=True)
class DataSpec:
    train: str = ""
    verifier: str = "math_answer"
    system_prompt: str = ""
    max_prompt_tokens: int = 1024


@dataclass(frozen=True)
class RewardSpec:
    route: str = "binary"              # binary | scalar
    hack_penalty: float = 1.0


@dataclass(frozen=True)
class OptimSpec:
    learning_rate: float = 1e-6
    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    warmup_steps: int = 0
    gradient_clip: float = 1.0


@dataclass(frozen=True)
class RolloutSpec:
    max_new_tokens: int = 1024
    generation_batch_size: int = 64


@dataclass(frozen=True)
class LearnerSpec:
    max_tokens_per_microbatch: int = 16384


@dataclass(frozen=True)
class ParitySpec:
    """Abort before an update whose rollout/training log-prob gap is too large.

    In strict sync the update's first forward uses the same weights that
    sampled the batch, so the gap measures engine numerics only. A large gap
    means a sampler or precision mismatch that would silently bias learning.
    """
    max_mean_abs_gap: float = 0.05
    max_abs_gap: float = 5.0
    # Executions clipped before any update mean the engines disagree by more
    # than the +/-0.5% execution clip; their signal would be silently dropped.
    max_strict_clip_fraction: float = 0.2


@dataclass(frozen=True)
class CheckpointSpec:
    every: int = 50
    keep_last: int = 2


@dataclass(frozen=True)
class IntegritySpec:
    task_revoke_rate: float = 0.25
    task_min_executions: int = 16
    halt_rate: float = 0.05
    halt_window: int = 1024


@dataclass(frozen=True)
class RunSpec:
    steps: int = 100
    seed: int = 0


@dataclass(frozen=True)
class RunConfig:
    schema_version: int = 1
    model: ModelSpec = field(default_factory=ModelSpec)
    data: DataSpec = field(default_factory=DataSpec)
    schedule: Schedule = field(default_factory=Schedule)
    reward: RewardSpec = field(default_factory=RewardSpec)
    loss: ExecutionLossConfig = field(default_factory=ExecutionLossConfig)
    optim: OptimSpec = field(default_factory=OptimSpec)
    rollout: RolloutSpec = field(default_factory=RolloutSpec)
    learner: LearnerSpec = field(default_factory=LearnerSpec)
    parity: ParitySpec = field(default_factory=ParitySpec)
    checkpoint: CheckpointSpec = field(default_factory=CheckpointSpec)
    integrity: IntegritySpec = field(default_factory=IntegritySpec)
    run: RunSpec = field(default_factory=RunSpec)

    def validate(self) -> "RunConfig":
        from .config import FusionConfig, Train
        if self.schema_version != 1 or type(self.schema_version) is not int:
            raise ValueError("Unsupported run schema_version")
        # Reuse the shared schedule/loss guards (strict sync, quotas, lag, sampler).
        FusionConfig(schedule=self.schedule, loss=self.loss,
                     train=Train(steps=self.run.steps, seed=self.run.seed)).validate()
        m, d, r, o = self.model, self.data, self.reward, self.optim
        if not m.path:
            raise ValueError("model.path must name a checkpoint directory or hub id")
        if m.precision not in {"auto", "fp32", "bf16-autocast", "bf16"} or m.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("Unknown model precision or device")
        if not d.train or not d.verifier:
            raise ValueError("data.train and data.verifier are required")
        if d.max_prompt_tokens < 1:
            raise ValueError("data.max_prompt_tokens must be positive")
        if r.route not in {"binary", "scalar"}:
            raise ValueError("The real-model runner supports binary and calibrated scalar verifiers; "
                             "GRS/GAR need grader channels a runner must supply explicitly")
        if not math.isfinite(r.hack_penalty) or r.hack_penalty < 0:
            raise ValueError("reward.hack_penalty must be finite and nonnegative")
        for name in ("learning_rate", "eps", "gradient_clip"):
            if not math.isfinite(getattr(o, name)) or getattr(o, name) <= 0:
                raise ValueError(f"optim.{name} must be finite and positive")
        if not (0 <= o.beta1 < 1 and 0 <= o.beta2 < 1) or o.weight_decay < 0 or o.warmup_steps < 0:
            raise ValueError("Invalid optimizer hyperparameters")
        if self.rollout.max_new_tokens < 1 or self.rollout.generation_batch_size < 1:
            raise ValueError("Rollout sizes must be positive")
        if self.learner.max_tokens_per_microbatch < 1:
            raise ValueError("learner.max_tokens_per_microbatch must be positive")
        if not (self.parity.max_mean_abs_gap > 0 and self.parity.max_abs_gap > 0
                and 0 < self.parity.max_strict_clip_fraction <= 1):
            raise ValueError("Parity thresholds must be positive")
        if self.checkpoint.every < 1 or self.checkpoint.keep_last < 1:
            raise ValueError("Checkpoint cadence and retention must be positive")
        if self.checkpoint.every % self.schedule.burst_steps:
            raise ValueError("Checkpoints are taken at publication boundaries: every must be a multiple of burst_steps")
        i = self.integrity
        if not (0 < i.task_revoke_rate <= 1 and 0 < i.halt_rate <= 1) or i.task_min_executions < 1 or i.halt_window < 1:
            raise ValueError("Invalid integrity thresholds")
        if self.run.steps % self.schedule.burst_steps:
            raise ValueError("run.steps must be a multiple of schedule.burst_steps")
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_SECTIONS = {"model": ModelSpec, "data": DataSpec, "schedule": Schedule, "reward": RewardSpec,
             "loss": ExecutionLossConfig, "optim": OptimSpec, "rollout": RolloutSpec, "learner": LearnerSpec,
             "parity": ParitySpec, "checkpoint": CheckpointSpec, "integrity": IntegritySpec, "run": RunSpec}


def run_config_from_dict(data: dict[str, Any]) -> RunConfig:
    unknown = set(data) - {f.name for f in fields(RunConfig)}
    if unknown:
        raise ValueError(f"Unknown run config keys: {sorted(unknown)}")
    values = {k: _strict_dataclass(_SECTIONS[k], v) if k in _SECTIONS else v for k, v in data.items()}
    return RunConfig(**values).validate()


def load_run_config(path: str | Path) -> RunConfig:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Run config must contain a JSON object")
    return run_config_from_dict(data)
