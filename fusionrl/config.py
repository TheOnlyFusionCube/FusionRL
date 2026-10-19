"""Strict configuration parsing and cross-source compatibility guards."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
import json
import math
from pathlib import Path
from typing import Any, get_args, get_origin, get_type_hints

from .objective import ExecutionLossConfig


@dataclass(frozen=True)
class Schedule:
    mode: str = "strict_sync"
    groups_per_update: int = 4
    executions_per_group: int = 8
    burst_steps: int = 1
    extra_dispatch_groups: int = 0
    max_pending_groups: int = 64
    max_policy_lag: int = 0
    jmax: int = 5
    dispatch_policy: str = "boundary"
    sample_reuse: bool = False


@dataclass(frozen=True)
class Reward:
    route: str = "binary"
    length_mode: str = "none"
    gar_cap: float = 2.0
    effort_levels: tuple[int, ...] = (25, 50, 75, 100)
    effort_k0: float = 0.2
    effort_tau: float = 25.0
    effort_length_norm: float = 4.0
    effort_cap: float = 0.2
    length_min_pass_rate: float = 0.5
    length_percentile: float = 50.0
    length_max_deduction: float = 0.2
    length_tolerance: float = 0.0
    length_saturation: float = 2.0
    length_exponent: float = 1.0


@dataclass(frozen=True)
class Train:
    steps: int = 100
    learning_rate: float = 0.01
    seed: int = 7
    max_generated_tokens: int = 4
    sft_steps: int = 0
    opd_steps: int = 0
    gradient_clip: float = 1.0


@dataclass(frozen=True)
class PolicyContract:
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    cache_policy: str = "invalidate_on_version_change"
    model_kind: str = "dense"
    freeze_moe_router: bool = False
    recorded_routing_replay: bool = False


@dataclass(frozen=True)
class FusionConfig:
    schema_version: int = 1
    schedule: Schedule = field(default_factory=Schedule)
    reward: Reward = field(default_factory=Reward)
    loss: ExecutionLossConfig = field(default_factory=ExecutionLossConfig)
    train: Train = field(default_factory=Train)
    policy: PolicyContract = field(default_factory=PolicyContract)

    def validate(self) -> "FusionConfig":
        s, r, t, p = self.schedule, self.reward, self.train, self.policy
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("Unsupported FusionRL schema_version")
        for name, value in (("sample_reuse",s.sample_reuse),("freeze_moe_router",p.freeze_moe_router),("recorded_routing_replay",p.recorded_routing_replay)):
            if type(value) is not bool:
                raise ValueError(f"{name} must be a JSON boolean")
        if s.mode not in {"strict_sync", "bounded_async"}:
            raise ValueError("Choose strict_sync or bounded_async")
        for name in ("groups_per_update", "executions_per_group", "burst_steps", "max_pending_groups", "jmax"):
            x = getattr(s, name)
            if isinstance(x, bool) or not isinstance(x, int) or x <= 0:
                raise ValueError(f"schedule.{name} must be a positive integer")
        if s.executions_per_group < 2:
            raise ValueError("Group-relative advantages need at least two executions")
        for name in ("extra_dispatch_groups", "max_policy_lag"):
            x = getattr(s, name)
            if isinstance(x, bool) or not isinstance(x, int) or x < 0:
                raise ValueError(f"schedule.{name} must be a nonnegative integer")
        if s.dispatch_policy != "boundary" or s.sample_reuse:
            raise ValueError("FusionRL implements boundary dispatch without sample reuse")
        if s.mode == "strict_sync" and (s.max_policy_lag or s.extra_dispatch_groups or s.burst_steps != 1):
            raise ValueError("Strict sync requires zero lag, zero carry-over, and one update per publication")
        if s.max_pending_groups < s.groups_per_update * s.burst_steps + s.extra_dispatch_groups:
            raise ValueError("Pending capacity cannot fit the configured dispatch window")
        if r.route not in {"binary", "grs", "gar", "scalar"} or r.length_mode not in {"none", "effort", "group_relative"}:
            raise ValueError("Unknown reward route or length mode")
        if r.route == "gar" and r.length_mode != "none":
            raise ValueError("GAR requires binary outcomes: use quality factors for concision, or choose binary/GRS for length costs")
        if not r.effort_levels or len(set(r.effort_levels)) != len(r.effort_levels):
            raise ValueError("Effort levels must be nonempty and unique")
        if any(isinstance(x, bool) or not isinstance(x, int) or not 1 <= x <= 100 for x in r.effort_levels):
            raise ValueError("Each reasoning effort must be an integer in [1,100]")
        for key in ("gar_cap", "effort_tau", "effort_length_norm", "length_saturation"):
            if not math.isfinite(getattr(r,key)) or getattr(r,key) <= 0:
                raise ValueError(f"reward.{key} must be finite and positive")
        if r.gar_cap < 1 or r.length_saturation <= r.length_tolerance:
            raise ValueError("GAR cap must be >=1 and length saturation must exceed tolerance")
        for key in ("effort_k0", "effort_cap", "length_max_deduction", "length_tolerance"):
            if not math.isfinite(getattr(r,key)) or getattr(r,key) < 0:
                raise ValueError(f"reward.{key} must be finite and nonnegative")
        if not 0 <= r.length_min_pass_rate <= 1 or not 0 < r.length_percentile < 100 or not math.isfinite(r.length_exponent) or r.length_exponent < 1:
            raise ValueError("Invalid group-relative length parameters")
        for key in ("steps", "max_generated_tokens", "seed", "sft_steps", "opd_steps"):
            x = getattr(t,key)
            if isinstance(x,bool) or not isinstance(x,int) or x < 0:
                raise ValueError(f"train.{key} must be a nonnegative integer")
        if t.steps == 0 or t.max_generated_tokens == 0:
            raise ValueError("Training steps and generation budget must be positive")
        if not math.isfinite(t.learning_rate) or not math.isfinite(t.gradient_clip) or min(t.learning_rate,t.gradient_clip) <= 0:
            raise ValueError("Learning rate and gradient clip must be finite and positive")
        if p.temperature != 1 or p.top_p != 1 or p.top_k != -1:
            raise ValueError("Portable profiles require full-support temperature-1 sampling; engine-specific truncation needs audited probability support")
        if p.cache_policy != "invalidate_on_version_change":
            raise ValueError("Generic backends must re-prefill after a policy change")
        if p.model_kind not in {"dense", "moe"}:
            raise ValueError("Unknown model kind")
        if p.model_kind == "moe" and not (p.freeze_moe_router and p.recorded_routing_replay):
            raise ValueError("MoE use requires both router freezing and a recorded routing replay backend")
        if p.model_kind == "dense" and (p.freeze_moe_router or p.recorded_routing_replay):
            raise ValueError("Dense models have no MoE routing to freeze or replay")
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _check_json_type(owner: str, name: str, expected: Any, value: Any) -> Any:
    """Reject JSON values whose type differs from the field annotation.

    JSON booleans are not numbers here: ``"learning_rate": true`` would
    otherwise train at 1.0, and ``"temperature": true`` would pass as 1.
    """
    where = f"{owner}.{name}"
    if expected is bool:
        if type(value) is not bool:
            raise ValueError(f"{where} must be a JSON boolean")
    elif expected is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{where} must be an integer")
    elif expected is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{where} must be a number")
        return float(value)
    elif expected is str:
        if not isinstance(value, str):
            raise ValueError(f"{where} must be a string")
    elif get_origin(expected) is tuple:
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"{where} must be a JSON array")
        item = get_args(expected)[0]
        return tuple(_check_json_type(owner, f"{name}[{i}]", item, x) for i, x in enumerate(value))
    return value


def _strict_dataclass(cls: Any, data: Any) -> Any:
    if not isinstance(data, dict):
        raise ValueError(f"{cls.__name__} must be a JSON object")
    unknown = set(data) - {x.name for x in fields(cls)}
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} keys: {sorted(unknown)}")
    hints = get_type_hints(cls)
    checked = {key: _check_json_type(cls.__name__, key, hints[key], value) for key, value in data.items()}
    return cls(**checked)


def config_from_dict(data: dict[str, Any]) -> FusionConfig:
    unknown = set(data) - {x.name for x in fields(FusionConfig)}
    if unknown:
        raise ValueError(f"Unknown FusionRL keys: {sorted(unknown)}")
    nested = {"schedule":Schedule,"reward":Reward,"loss":ExecutionLossConfig,"train":Train,"policy":PolicyContract}
    values = {k: _strict_dataclass(nested[k],v) if k in nested else v for k,v in data.items()}
    return FusionConfig(**values).validate()


def load_config(path: str | Path) -> FusionConfig:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Config must contain a JSON object")
    return config_from_dict(data)

