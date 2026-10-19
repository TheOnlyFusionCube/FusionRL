"""Explicit verl hooks; math is delegated to the shared FusionRL objective."""

from __future__ import annotations

from collections import defaultdict
import os

from . import ADVANTAGE_NAME, LOSS_NAME
from .contracts import VerlContractError, validate_config, validate_execution_batch


def binary_advantage(token_level_rewards, response_mask, index, config=None, **kwargs):
    """Outcome-only centered binary reward; one row must equal one execution."""
    import torch
    from fusionrl.rewards import compute_advantages

    if token_level_rewards.ndim != 2 or token_level_rewards.shape != response_mask.shape:
        raise VerlContractError("reward and action-mask tensors must be aligned [executions,tokens]")
    if len(index) != len(response_mask):
        raise VerlContractError("one prompt UID is required for each execution")
    rewards = token_level_rewards.sum(dim=-1).detach()
    if not bool(torch.isfinite(rewards).all()) or not bool(((rewards == 0) | (rewards == 1)).all()):
        raise VerlContractError("the verl adapter accepts binary verified outcomes only")
    if bool((torch.count_nonzero(token_level_rewards, dim=-1) > 1).any()):
        raise VerlContractError("dense/process reward channels require a separate transport adapter")
    groups = defaultdict(list)
    for row, uid in enumerate(index):
        groups[str(uid)].append(row)
    row_advantages = torch.zeros_like(rewards)
    for rows in groups.values():
        if len(rows) < 2:
            raise VerlContractError("each binary reward group needs at least two executions")
        computed = compute_advantages(rewards[rows].cpu().tolist(), route="binary").advantages
        row_advantages[rows] = torch.as_tensor(computed, device=rewards.device, dtype=rewards.dtype)
    advantages = row_advantages[:, None] * response_mask
    return advantages, advantages.clone()


def policy_loss(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    loss_agg_mode="seq-mean-token-mean",
    config=None,
    rollout_is_weights=None,
):
    """old_log_prob is recorded behavior probability by the explicit patch."""
    import torch
    from fusionrl.objective import ExecutionLossConfig, execution_loss

    if config is None or loss_agg_mode != "seq-mean-token-mean":
        raise VerlContractError("FusionRL requires explicit execution-balanced actor configuration")
    if rollout_is_weights is not None:
        raise VerlContractError("a second importance-sampling correction would change the objective")
    if advantages.shape != response_mask.shape or log_prob.shape != response_mask.shape:
        raise VerlContractError("policy tensors must have matching [execution,token] shape")
    mask = response_mask.bool()
    count = mask.sum(dim=1)
    row_adv = torch.where(mask, advantages, torch.zeros_like(advantages)).sum(dim=1) / count.clamp_min(1)
    broadcast = row_adv[:, None].expand_as(advantages)
    if not torch.allclose(advantages[mask], broadcast[mask], rtol=1e-5, atol=1e-7):
        raise VerlContractError("token-shaped auxiliary advantages are unsupported by the execution objective")
    info = config.global_batch_info
    global_count = info.get("global_batch_size")
    if global_count is None or int(global_count) <= 0:
        raise VerlContractError("the logical execution count must be supplied for every microbatch")
    expected = int(config.ppo_mini_batch_size) * int(config.rollout_n)
    if int(global_count) != expected:
        raise VerlContractError("global_batch_size must count the complete sampled executions")
    dp_size = int(info.get("dp_size", 1))
    if dp_size < 1:
        raise VerlContractError("dp_size must be positive")
    loss, metrics = execution_loss(
        log_prob,
        old_log_prob.detach(),
        row_adv.detach(),
        mask,
        config=ExecutionLossConfig(),
        global_execution_count=int(global_count),
    )
    # FSDP/DDP averages gradients across workers. Each worker contributes its
    # microbatch execution sum divided by the SAME original global count.
    loss = loss * dp_size
    result = {f"actor/fusionrl_{key}": value for key, value in metrics.items()}
    result["actor/fusionrl_global_execution_count"] = float(global_count)
    return loss, result


def ensure_registered() -> None:
    from verl.trainer.ppo.core_algos import ADV_ESTIMATOR_REGISTRY, POLICY_LOSS_REGISTRY
    from verl.trainer.ppo.core_algos import register_adv_est, register_policy_loss

    if ADVANTAGE_NAME in ADV_ESTIMATOR_REGISTRY and ADV_ESTIMATOR_REGISTRY[ADVANTAGE_NAME] is not binary_advantage:
        raise VerlContractError("FusionRL advantage registry name is already occupied")
    if LOSS_NAME in POLICY_LOSS_REGISTRY and POLICY_LOSS_REGISTRY[LOSS_NAME] is not policy_loss:
        raise VerlContractError("FusionRL loss registry name is already occupied")
    if ADVANTAGE_NAME not in ADV_ESTIMATOR_REGISTRY:
        register_adv_est(ADVANTAGE_NAME)(binary_advantage)
    if LOSS_NAME not in POLICY_LOSS_REGISTRY:
        register_policy_loss(LOSS_NAME)(policy_loss)


def validate_and_register(config) -> None:
    validate_config(config)
    if os.getenv("WEBDEV_GRADE_MODE") or os.getenv("DROP_INFRA_FROM_GROUP", "0") == "1":
        raise VerlContractError("domain-specific trainer environment overrides are incompatible with this profile")
    ensure_registered()


def guard_training_batch(batch, config, training_step: int):
    """Run after balancing and before likelihood/advantage computation."""
    import transfer_queue as tq

    # TaskRunnerV1 runs in another Ray process; main()'s registries are local
    # to the launcher process and cannot be assumed to exist here.
    ensure_registered()
    fields = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id, select_fields=["extra_fields"])
    validate_execution_batch(
        batch.keys,
        batch.tags,
        list(fields["extra_fields"]),
        rollout_n=int(config.actor_rollout_ref.rollout.n),
        prompt_count=int(config.data.train_batch_size),
        training_step=training_step,
    )
    return batch


def copy_behavior_log_probs(batch):
    """Preserve true rollout likelihoods instead of recomputing a third policy."""
    import torch
    import transfer_queue as tq

    data = tq.kv_batch_get(
        keys=batch.keys,
        partition_id=batch.partition_id,
        select_fields=["rollout_log_probs", "response_mask"],
    )
    if "rollout_log_probs" not in data:
        raise VerlContractError("recorded behavior log probabilities are mandatory")
    padded = data.to_padded_tensor()
    probs, mask = padded["rollout_log_probs"], padded["response_mask"].bool()
    if probs.shape != mask.shape or not bool(torch.isfinite(probs[mask]).all()):
        raise VerlContractError("behavior log probabilities must cover every generated action")
    if bool((probs[mask] > 1e-5).any()):
        raise VerlContractError("behavior log probabilities must not be positive")
    data["old_log_probs"] = data["rollout_log_probs"]
    tq.kv_batch_put(keys=batch.keys, partition_id=batch.partition_id, fields=data.select("old_log_probs"))
    return batch

