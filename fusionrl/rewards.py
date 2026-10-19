"""Auditable group rewards for FusionRL; NumPy is the only dependency.

Equations are independently implemented from the source reports. GRS and GAR
are alternative routes. GAR consumes binary corrected outcomes only and cannot
be combined with additive cost shaping in this conservative implementation.
FusionRL's explicit scalar route accepts externally calibrated [0,1] rewards.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np


def _vector(values: Sequence[float], name: str, size: int | None = None) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or result.size == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional sequence")
    if size is not None and result.size != size:
        raise ValueError(f"{name} must contain {size} values")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result.copy()


def _binary(values: Sequence[float], name: str = "outcomes") -> np.ndarray:
    result = _vector(values, name)
    if not np.all((result == 0.0) | (result == 1.0)):
        raise ValueError(f"{name} must be binary; GAR must never receive shaped rewards")
    return result


def _unit_scores(values: Sequence[float] | None, name: str, size: int) -> np.ndarray:
    if values is None:
        raise ValueError(f"{name} is required for the GRS route")
    result = _vector(values, name, size)
    if np.any((result < 0.0) | (result > 1.0)):
        raise ValueError(f"{name} must lie in [0, 1]")
    return result


def _finite_scalar(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True)
class AdvantageResult:
    """All intermediate channels are retained for audits and metric logging."""

    advantages: np.ndarray
    corrected_outcomes: np.ndarray
    effective_rewards: np.ndarray
    base_advantages: np.ndarray
    route_advantages: np.ndarray
    cost_advantages: np.ndarray
    diagnostics: dict[str, Any] = field(default_factory=dict)


def compute_advantages(
    outcomes: Sequence[float],
    *,
    route: str = "binary",
    solution_scores: Sequence[float] | None = None,
    behavior_scores: Sequence[float] | None = None,
    quality_factors: Sequence[float | None] | None = None,
    confirmed_hacks: Sequence[bool] | None = None,
    cost_penalties: Sequence[float] | None = None,
    gar_cap: float = 2.0,
    hack_penalty: float = 0.0,
) -> AdvantageResult:
    """Compute centered scalar advantages for one complete prompt group.

    ``outcomes`` is the binary verifier result, except for the explicit
    ``scalar`` route, where it is an already calibrated reward in [0,1].
    Confirmed hacks become zero rewards before statistics or rubric multiplication. GRS multiplies this
    result by two bounded rubric scores. GAR redistributes positive binary
    advantages, caps the common multiplier, then recenters the whole group.
    Missing or malformed GAR rankings fall back to binary advantages. Only
    factors for corrected passing trajectories are required to be in (0, 1].

    ``cost_penalties`` must be signed, nonpositive penalties. They are centered
    separately and added after binary/GRS/scalar advantages; this equals centering
    the sum of reward and cost. Nonzero costs with GAR are rejected, preserving
    GAR's binary derivation and avoiding two simultaneous concision signals.
    ``hack_penalty`` (>= 0) subtracts that amount from every confirmed hack
    as its own centered channel, after zeroing its reward. Zeroing alone gives
    no signal against a hack when its groupmates also failed; the penalty
    makes the hack strictly worse than an honest failure. It is an integrity
    channel, not a length controller, so it is allowed with every route.
    No standard-deviation normalization or token-level shaping is applied.
    """
    if route not in {"binary", "grs", "gar", "scalar"}:
        raise ValueError("route must be one of: binary, grs, gar, scalar")
    if route == "scalar":
        corrected = _vector(outcomes, "scalar rewards")
        if np.any((corrected < 0.0) | (corrected > 1.0)):
            raise ValueError("scalar rewards must be calibrated to [0,1]")
    else:
        corrected = _binary(outcomes)
    n = corrected.size
    cap = _finite_scalar(gar_cap, "gar_cap")
    if cap < 1.0:
        raise ValueError("gar_cap must be at least 1")
    penalty = _finite_scalar(hack_penalty, "hack_penalty")
    if penalty < 0.0:
        raise ValueError("hack_penalty must be nonnegative")

    hacks = np.zeros(n, dtype=bool)
    if confirmed_hacks is not None:
        raw_hacks = np.asarray(confirmed_hacks)
        if raw_hacks.ndim != 1 or raw_hacks.size != n:
            raise ValueError(f"confirmed_hacks must contain {n} boolean values")
        if raw_hacks.dtype.kind != "b":
            raise ValueError("confirmed_hacks must contain booleans, not truthy strings")
        hacks = raw_hacks.astype(bool, copy=True)
        corrected[hacks] = 0.0

    cost = np.zeros(n, dtype=np.float64)
    if cost_penalties is not None:
        cost = _vector(cost_penalties, "cost_penalties", n)
        if np.any(cost > 0.0):
            raise ValueError("cost_penalties must be nonpositive")
    if route == "gar" and np.any(cost != 0.0):
        raise ValueError("GAR cannot be combined with nonzero additive cost penalties")
    if route != "grs" and (solution_scores is not None or behavior_scores is not None):
        raise ValueError("solution_scores and behavior_scores belong only to the GRS route")
    if route != "gar" and quality_factors is not None:
        raise ValueError("quality_factors belong only to the GAR route")

    effective_rewards = corrected.copy()
    if route == "grs":
        sol = _unit_scores(solution_scores, "solution_scores", n)
        beh = _unit_scores(behavior_scores, "behavior_scores", n)
        effective_rewards *= sol * beh

    base = effective_rewards - effective_rewards.mean()
    routed = base.copy()
    diag: dict[str, Any] = {
        "route": route,
        "group_size": n,
        "corrected_pass_rate": float(corrected.mean()) if route != "scalar" else None,
        "corrected_reward_mean": float(corrected.mean()),
        "confirmed_hack_count": int(hacks.sum()),
        "gar_fallback": None,
        "gar_multiplier_uncapped": None,
        "gar_multiplier_used": None,
        "gar_cap_active": False,
    }
    if route == "gar":
        passing = corrected == 1.0
        if not passing.any() or passing.all():
            # Neither case has positive binary advantage to redistribute.
            diag["gar_fallback"] = "unanimous_group"
        elif quality_factors is None:
            diag["gar_fallback"] = "missing_grader_output"
        else:
            try:
                factors = np.asarray(quality_factors, dtype=np.float64)
                valid_shape = factors.ndim == 1 and factors.size == n
                valid_values = valid_shape and bool(np.all(
                    np.isfinite(factors[passing])
                    & (factors[passing] > 0.0)
                    & (factors[passing] <= 1.0)
                ))
            except (TypeError, ValueError, OverflowError):
                valid_values = False
            if not valid_values:
                diag["gar_fallback"] = "invalid_grader_output"
            else:
                positive_mass = float(base[passing].sum())
                denominator = float((factors[passing] * base[passing]).sum())
                if denominator <= 0.0 or not np.isfinite(denominator):
                    diag["gar_fallback"] = "degenerate_grader_output"
                else:
                    multiplier = positive_mass / denominator
                    if not np.isfinite(multiplier):
                        # Extremely tiny positive factors can overflow the ratio;
                        # treat them as unusable grading rather than emit inf logs.
                        diag["gar_fallback"] = "numerically_unsafe_grader_output"
                    else:
                        applied = min(cap, multiplier)
                        routed[passing] *= applied * factors[passing]
                        routed -= routed.mean()
                        diag.update({
                            "gar_multiplier_uncapped": multiplier,
                            "gar_multiplier_used": applied,
                            "gar_cap_active": multiplier > cap,
                        })

    centered_cost = cost - cost.mean()
    hack_channel = -penalty * hacks.astype(np.float64)
    centered_hack = hack_channel - hack_channel.mean()
    diag["hack_penalty"] = penalty
    combined = routed + centered_cost + centered_hack
    # The arithmetic mean is removed once more only to eliminate rounding drift.
    combined -= combined.mean()
    diag["advantage_sum"] = float(combined.sum())
    diag["has_learning_signal"] = bool(np.any(np.abs(combined) > 1e-12))
    return AdvantageResult(
        advantages=combined,
        corrected_outcomes=corrected,
        effective_rewards=effective_rewards,
        base_advantages=base,
        route_advantages=routed,
        cost_advantages=centered_cost,
        diagnostics=diag,
    )


def deepseek_effort_penalties(
    reasoning_tokens: Sequence[float],
    *,
    budget: float,
    min_budget: float,
    k0: float,
    tau: float,
    length_norm: float,
    cap: float,
) -> np.ndarray:
    """Return -min(cap, k0*exp(-(b-b_min)/tau)*L_reason/length_norm).

    Token counts refer to reasoning tokens, never prompt/tool/context tokens.
    The group shares one effort budget; modifying this convention creates a
    different conditioning problem and requires separate validation.
    """
    length = _vector(reasoning_tokens, "reasoning_tokens")
    b = _finite_scalar(budget, "budget")
    bmin = _finite_scalar(min_budget, "min_budget")
    scale = _finite_scalar(k0, "k0")
    decay = _finite_scalar(tau, "tau")
    norm = _finite_scalar(length_norm, "length_norm")
    maximum = _finite_scalar(cap, "cap")
    if np.any(length < 0.0):
        raise ValueError("reasoning_tokens must be nonnegative")
    if bmin < 0.0 or b < bmin:
        raise ValueError("budget must be at least min_budget >= 0")
    if scale < 0.0 or maximum < 0.0 or decay <= 0.0 or norm <= 0.0:
        raise ValueError("k0/cap must be nonnegative; tau/length_norm must be positive")
    coefficient = scale * np.exp(-(b - bmin) / decay)
    if coefficient == 0.0 or maximum == 0.0:
        return np.zeros_like(length)
    with np.errstate(over="ignore", invalid="raise"):
        return -np.minimum(maximum, coefficient * (length / norm))


def mimo_length_penalties(
    lengths: Sequence[float],
    outcomes: Sequence[float],
    *,
    min_pass_rate: float,
    percentile: float,
    max_deduction: float,
    tolerance: float,
    saturation: float,
    exponent: float,
) -> np.ndarray:
    """Success-gated group-relative length penalty (MiMo Eq. 4).

    Supply hack-corrected binary outcomes. ``percentile`` uses [0, 100] units;
    this implementation chooses NumPy's explicit linear quantile convention.
    Only groups with pass_rate STRICTLY GREATER than min_pass_rate qualify.
    Length counts generated model tokens across the whole trajectory.
    """
    length = _vector(lengths, "lengths")
    success = _binary(outcomes)
    if length.size != success.size or np.any(length < 0.0):
        raise ValueError("lengths must be nonnegative and match outcomes")
    gate = _finite_scalar(min_pass_rate, "min_pass_rate")
    quantile = _finite_scalar(percentile, "percentile")
    maximum = _finite_scalar(max_deduction, "max_deduction")
    delta = _finite_scalar(tolerance, "tolerance")
    sat = _finite_scalar(saturation, "saturation")
    power = _finite_scalar(exponent, "exponent")
    if not 0.0 <= gate <= 1.0 or not 0.0 < quantile < 100.0:
        raise ValueError("min_pass_rate must be in [0,1]; percentile in (0,100)")
    if maximum < 0.0 or delta < 0.0 or sat <= delta or power < 1.0:
        raise ValueError("require deduction>=0, tolerance>=0, saturation>tolerance, exponent>=1")
    penalty = np.zeros_like(length)
    passing = success == 1.0
    if not passing.any() or float(success.mean()) <= gate:
        return penalty
    reference = float(np.quantile(length[passing], quantile / 100.0, method="linear"))
    if reference <= 0.0:
        raise ValueError("successful reference length must be positive")
    ramp = np.clip((length[passing] / reference - 1.0 - delta) / (sat - delta), 0.0, 1.0)
    penalty[passing] = -maximum * np.power(ramp, power)
    return penalty


def choose_cost_penalties(
    *,
    deepseek: Sequence[float] | None = None,
    mimo: Sequence[float] | None = None,
) -> np.ndarray | None:
    """Reject simultaneous effort controllers, even if one currently returns zeros."""
    if deepseek is not None and mimo is not None:
        raise ValueError("Choose one cost controller: DeepSeek effort OR MiMo relative length")
    selected = deepseek if deepseek is not None else mimo
    if selected is None:
        return None
    penalty = _vector(selected, "cost_penalties")
    if np.any(penalty > 0.0):
        raise ValueError("cost_penalties must be nonpositive")
    return penalty
