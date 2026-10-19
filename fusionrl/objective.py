"""Execution-balanced policy loss; QwenGyre Appendix B, equations 26--30.

The behavior law is recorded at sampling time. It is NOT a recomputed old policy.
The sequence statistic and token importance weights are detached. Only the final
log probability term has a policy gradient. No second PPO/GSPO/GRPO loss is used.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ExecutionLossConfig:
    clip_low: float = 0.995
    clip_high: float = 1.005
    token_is_cap: float = 5.0
    token_loss_cap: float = 100.0

    def __post_init__(self) -> None:
        values = (self.clip_low, self.clip_high, self.token_is_cap, self.token_loss_cap)
        if not all(math.isfinite(v) for v in values):
            raise ValueError("Loss parameters must be finite")
        if not (0 < self.clip_low <= 1 <= self.clip_high):
            raise ValueError("Clipping bounds must bracket 1 and be positive")
        if self.token_is_cap <= 0 or self.token_loss_cap <= 0:
            raise ValueError("Importance and token-loss caps must be positive")


def _denominator(rows: int, global_execution_count: int | None) -> int:
    value = rows if global_execution_count is None else global_execution_count
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError("global_execution_count must be an integer")
    if value < rows or value <= 0:
        raise ValueError("Denominator must include every original execution slot")
    return int(value)


def execution_loss_numpy(
    log_probs: Any,
    behavior_log_probs: Any,
    advantages: Any,
    action_mask: Any,
    *,
    config: ExecutionLossConfig | None = None,
    global_execution_count: int | None = None,
) -> tuple[float, dict[str, float]]:
    """Independent float64 numerical oracle (no automatic differentiation).

    Rows are ORIGINAL executions, including rows with zero eligible targets.
    Columns are unique admitted target tokens, NOT every token in every path.
    ``action_mask`` is the admission/provenance mask, before policy clipping.
    """
    cfg = config or ExecutionLossConfig()
    lp = np.asarray(log_probs, dtype=np.float64)
    bp = np.asarray(behavior_log_probs, dtype=np.float64)
    adv = np.asarray(advantages, dtype=np.float64)
    mask = np.asarray(action_mask)
    if lp.ndim != 2 or bp.shape != lp.shape or mask.shape != lp.shape:
        raise ValueError("Log probabilities and mask must have shape [executions, tokens]")
    if adv.shape != (lp.shape[0],) or mask.dtype != np.bool_:
        raise ValueError("Need one advantage per execution and a boolean mask")
    den = _denominator(lp.shape[0], global_execution_count)
    if not np.isfinite(adv).all():
        raise ValueError("Non-finite advantage")
    if not (np.isfinite(lp[mask]).all() and np.isfinite(bp[mask]).all()):
        raise ValueError("Admitted log probabilities must be finite")
    if (lp[mask] > 1e-6).any() or (bp[mask] > 1e-6).any():
        raise ValueError("A log probability cannot be positive")
    lp = np.where(mask, lp, 0.0)
    bp = np.where(mask, bp, 0.0)
    delta = lp - bp
    lengths = mask.sum(axis=1)
    mean_log_ratio = (delta * mask).sum(axis=1) / np.maximum(lengths, 1)
    clipped = ((adv > 0) & (mean_log_ratio > math.log(cfg.clip_high))) | (
        (adv < 0) & (mean_log_ratio < math.log(cfg.clip_low))
    )
    weights = np.exp(np.minimum(delta, math.log(cfg.token_is_cap)))
    token_losses = np.minimum(
        cfg.token_loss_cap, -(~clipped).astype(np.float64)[:, None] * weights * adv[:, None] * lp
    )
    per_execution = (token_losses * mask).sum(axis=1) / np.maximum(lengths, 1)
    loss = float(per_execution.sum() / den)
    nonempty = lengths > 0
    metrics = {
        "execution_count": float(lp.shape[0]),
        "normalization_count": float(den),
        "target_tokens": float(lengths.sum()),
        "empty_executions": float((~nonempty).sum()),
        "clip_fraction": float(clipped[nonempty].mean()) if nonempty.any() else 0.0,
        "mean_log_ratio": float(mean_log_ratio[nonempty].mean()) if nonempty.any() else 0.0,
        "abs_log_ratio_sum": float((np.abs(delta) * mask).sum()),
        "max_abs_log_ratio": float((np.abs(delta) * mask).max()) if mask.any() else 0.0,
        "clipped_executions": float(clipped[nonempty].sum()),
        "loss": loss,
    }
    return loss, metrics


def execution_loss(
    log_probs: Any,
    behavior_log_probs: Any,
    advantages: Any,
    action_mask: Any,
    *,
    config: ExecutionLossConfig | None = None,
    global_execution_count: int | None = None,
) -> tuple[Any, dict[str, float]]:
    """PyTorch policy loss with exact stop-gradient semantics.

    For data parallel execution, pass the ORIGINAL global execution count.
    If the backend averages gradients over D ranks, it must multiply this local
    loss by D once. Targets of an execution may not be split across microbatches
    unless its global target count AND clipping statistic are first supplied by
    a separate reduce; this function deliberately requires complete rows.
    """
    import torch

    cfg = config or ExecutionLossConfig()
    if not all(torch.is_tensor(x) for x in (log_probs, behavior_log_probs, advantages, action_mask)):
        raise TypeError("execution_loss expects torch tensors")
    if log_probs.ndim != 2 or behavior_log_probs.shape != log_probs.shape or action_mask.shape != log_probs.shape:
        raise ValueError("Log probabilities and mask must have shape [executions, tokens]")
    if advantages.shape != (log_probs.shape[0],) or action_mask.dtype != torch.bool:
        raise ValueError("Need one advantage per execution and a boolean mask")
    if not log_probs.is_floating_point() or not behavior_log_probs.is_floating_point() or not advantages.is_floating_point():
        raise TypeError("Log probabilities and advantages must be floating point")
    if len({x.device for x in (log_probs, behavior_log_probs, advantages, action_mask)}) != 1:
        raise ValueError("All loss tensors must be on the same device")
    den = _denominator(log_probs.shape[0], global_execution_count)
    if not torch.isfinite(advantages).all().item():
        raise ValueError("Non-finite advantage")
    if not (torch.isfinite(log_probs[action_mask]).all() & torch.isfinite(behavior_log_probs[action_mask]).all()).item():
        raise ValueError("Admitted log probabilities must be finite")
    if (log_probs[action_mask] > 1e-6).any().item() or (behavior_log_probs[action_mask] > 1e-6).any().item():
        raise ValueError("A log probability cannot be positive")
    # Keep reductions in at least float32; preserve float64 for numerical tests.
    dtype = torch.float64 if log_probs.dtype == torch.float64 else torch.float32
    lp = torch.where(action_mask, log_probs.to(dtype), 0.0)
    bp = torch.where(action_mask, behavior_log_probs.detach().to(dtype), 0.0)
    adv = advantages.detach().to(dtype)
    lengths = action_mask.sum(dim=1)
    safe_lengths = lengths.clamp_min(1).to(dtype)
    delta = lp.detach() - bp
    mean_log_ratio = (delta * action_mask).sum(dim=1) / safe_lengths
    clipped = ((adv > 0) & (mean_log_ratio > math.log(cfg.clip_high))) | (
        (adv < 0) & (mean_log_ratio < math.log(cfg.clip_low))
    )
    weights = torch.exp(torch.clamp(delta, max=math.log(cfg.token_is_cap))).detach()
    token_losses = torch.clamp(
        -(~clipped).to(dtype)[:, None] * weights * adv[:, None] * lp,
        max=cfg.token_loss_cap,
    )
    per_execution = (token_losses * action_mask).sum(dim=1) / safe_lengths
    loss = per_execution.sum() / den
    nonempty = lengths > 0
    metrics = {
        "execution_count": float(log_probs.shape[0]),
        "normalization_count": float(den),
        "target_tokens": float(lengths.sum().item()),
        "empty_executions": float((~nonempty).sum().item()),
        "clip_fraction": float(clipped[nonempty].float().mean().item()) if nonempty.any().item() else 0.0,
        "mean_log_ratio": float(mean_log_ratio[nonempty].mean().item()) if nonempty.any().item() else 0.0,
        # Token-level |log pi_theta - log p_roll| over admitted targets. Before
        # the first update of a strict-sync batch this is exactly the
        # rollout-engine versus training-engine mismatch.
        "abs_log_ratio_sum": float((delta.abs() * action_mask).sum().item()),
        "max_abs_log_ratio": float((delta.abs() * action_mask).max().item()) if action_mask.any().item() else 0.0,
        "clipped_executions": float(clipped[nonempty].sum().item()),
        "loss": float(loss.detach().item()),
    }
    return loss, metrics

