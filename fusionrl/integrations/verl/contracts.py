"""Dependency-light config, batch, and pinned-source validation."""

from __future__ import annotations

import ast
from collections import Counter
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

from . import ADVANTAGE_NAME, LOSS_NAME, UPSTREAM_COMMIT


class VerlContractError(ValueError):
    """An input lies outside the verified FusionRL-to-verl contract."""


def get_path(config: Any, path: str, default: Any = None) -> Any:
    value = config
    for key in path.split("."):
        if not hasattr(value, "get"):
            return default
        value = value.get(key, default)
    return value


def validate_config(config: Any) -> None:
    """Accept only the implemented strict-sync binary execution profile.

    The portable core supports more routes. Those routes are not silently
    approximated by this upstream adapter's narrower tensor transport.
    """
    requirements = {
        "trainer.use_v1": True,
        "trainer.v1.trainer_mode": "sync",
        "algorithm.adv_estimator": ADVANTAGE_NAME,
        "algorithm.use_kl_in_reward": False,
        "algorithm.norm_adv_by_std_in_grpo": False,
        "algorithm.filter_groups.enable": False,
        "actor_rollout_ref.actor.policy_loss.loss_mode": LOSS_NAME,
        "actor_rollout_ref.actor.loss_agg_mode": "seq-mean-token-mean",
        "actor_rollout_ref.actor.ppo_epochs": 1,
        "actor_rollout_ref.actor.entropy_coeff": 0,
        "actor_rollout_ref.actor.use_kl_loss": False,
        "actor_rollout_ref.rollout.temperature": 1.0,
        "actor_rollout_ref.rollout.top_p": 1.0,
        "actor_rollout_ref.rollout.top_k": -1,
        "actor_rollout_ref.rollout.do_sample": True,
        "actor_rollout_ref.rollout.calculate_log_probs": True,
        "actor_rollout_ref.rollout.multi_turn.enable": False,
        "actor_rollout_ref.rollout.agent.default_agent_loop": "single_turn_agent",
        "actor_rollout_ref.rollout.over_sample_rate": 0,
        "actor_rollout_ref.rollout.nnodes": 0,
    }
    failures = []
    for path, expected in requirements.items():
        actual = get_path(config, path)
        if actual != expected:
            failures.append(f"{path} must be {expected!r}, got {actual!r}")

    disabled = [
        "algorithm.rollout_correction.bypass_mode",
        "algorithm.rollout_correction.rollout_is",
        "algorithm.rollout_correction.rollout_rs",
        "algorithm.use_pf_ppo",
        "algorithm.group_advantage_by_harness",
        "algorithm.length_penalty.enable",
        "algorithm.tool_call_error_penalty.enable",
        "algorithm.repetition_penalty.enable",
        "algorithm.deep_failure_mask.enable",
        "algorithm.arvo_penalties.enable",
        "distillation.enabled",
        "critic.enable",
        "reward.reward_model.enable",
        "actor_rollout_ref.model.mtp.enable",
        "actor_rollout_ref.actor.use_prefix_grouper",
        "actor_rollout_ref.rollout.agent.agent_loop_manager_class",
        "trainer.v1.sampler.custom_sampler.path",
        "trainer.v1.sampler.sync_refill_failed_groups",
    ]
    for path in disabled:
        if get_path(config, path):
            failures.append(f"{path} is unsupported by this adapter")

    strategy = get_path(config, "actor_rollout_ref.actor.strategy")
    if strategy not in {"fsdp", "fsdp2"}:
        failures.append("only the FSDP/FSDP2 engine contract is implemented")
    if get_path(config, "actor_rollout_ref.rollout.name") not in {"vllm", "sglang"}:
        failures.append("rollout engine must be vllm or sglang with recorded log probabilities")
    if get_path(config, "actor_rollout_ref.actor.loss_scale_factor") is not None:
        failures.append("loss_scale_factor must be null; the core owns execution normalization")
    if get_path(config, "algorithm.invalid_reward_value") is not None:
        failures.append("sentinel rewards are unsupported; infrastructure failure aborts the batch")
    if get_path(config, "actor_rollout_ref.rollout.min_p", 0.0) != 0.0:
        failures.append("min_p must be zero to retain full sampler support")
    if get_path(config, "actor_rollout_ref.rollout.presence_penalty", 0.0) != 0.0:
        failures.append("presence_penalty must be zero")
    if get_path(config, "actor_rollout_ref.rollout.repetition_penalty", 1.0) != 1.0:
        failures.append("repetition_penalty must be one")
    if get_path(config, "actor_rollout_ref.rollout.custom", {}):
        failures.append("custom rollout overrides require a separate audited sampler adapter")

    n = get_path(config, "actor_rollout_ref.rollout.n")
    batch = get_path(config, "data.train_batch_size")
    minibatch = get_path(config, "actor_rollout_ref.actor.ppo_mini_batch_size")
    if not isinstance(n, int) or isinstance(n, bool) or n < 2:
        failures.append("rollout.n must be an integer >= 2")
    if not isinstance(batch, int) or isinstance(batch, bool) or batch < 1:
        failures.append("data.train_batch_size must be a positive integer")
    if batch != minibatch:
        failures.append("one optimizer batch must contain the entire sampled prompt batch")
    if failures:
        raise VerlContractError("FusionRL verl contract rejected:\n- " + "\n- ".join(failures))


def validate_rollout_request(agent_name: str, sampling_params: Mapping[str, Any]) -> None:
    """Validate actual runtime inputs after per-dataset/per-prompt overrides."""
    if agent_name != "single_turn_agent":
        raise VerlContractError("dataset agent_name overrides require a separate audited agent adapter")
    required = {"temperature": 1.0, "top_p": 1.0, "top_k": -1, "repetition_penalty": 1.0, "logprobs": True}
    for key, expected in required.items():
        if sampling_params.get(key) != expected:
            raise VerlContractError(f"actual rollout {key} must be {expected!r}")
    extra = set(sampling_params) - set(required)
    if extra:
        raise VerlContractError(f"unreviewed runtime sampling overrides: {sorted(extra)!r}")


def validate_execution_batch(
    keys: Sequence[str],
    tags: Sequence[Mapping[str, Any]],
    extra_fields: Sequence[Mapping[str, Any]],
    *,
    rollout_n: int,
    prompt_count: int,
    training_step: int,
) -> int:
    """Check one physical row per logical execution and complete prompt groups.

    Upstream increments training_step before the update. The current rollout
    weight version is therefore training_step - 1 (see upstream metrics).
    Empty *represented* executions remain in the denominator. Missing rows,
    infrastructure failures, and artificial DP padding abort the whole update.
    """
    expected = rollout_n * prompt_count
    if len(keys) != expected or len(tags) != expected or len(extra_fields) != expected:
        raise VerlContractError(f"expected {expected} represented executions and metadata, got {len(keys)} rows")
    if len(set(keys)) != len(keys):
        raise VerlContractError("duplicate trajectory keys")
    groups: Counter[str] = Counter()
    sessions: set[tuple[str, str]] = set()
    policy_version = training_step - 1
    for key, tag, raw_extra in zip(keys, tags, extra_fields, strict=True):
        parts = key.rsplit("_", 2)
        if len(parts) != 3 or not parts[0] or not parts[1].isdigit() or parts[2] != "0":
            raise VerlContractError(f"only one-row execution keys uid_session_0 are supported: {key!r}")
        uid, session, _ = parts
        if "_" in uid:
            raise VerlContractError("upstream ReplayBuffer requires underscore-free prompt UIDs")
        if not 0 <= int(session) < rollout_n:
            raise VerlContractError(f"session index is outside the configured prompt group: {key!r}")
        if (uid, session) in sessions:
            raise VerlContractError("multiple output rows from one execution require the full path adapter")
        sessions.add((uid, session))
        groups[uid] += 1
        if tag.get("is_padding", False):
            raise VerlContractError("synthetic DP padding would corrupt the execution denominator")
        if tag.get("status") != "success":
            raise VerlContractError("every represented execution must have a completed transport record")
        lo, hi = tag.get("min_global_steps"), tag.get("max_global_steps")
        if lo != policy_version or hi != policy_version:
            raise VerlContractError(
                f"missing/stale/mixed behavior provenance for {key}: versions {(lo, hi)}, expected {policy_version}"
            )
        extra = getattr(raw_extra, "data", raw_extra)
        if not isinstance(extra, Mapping):
            raise VerlContractError("extra_fields must be a mapping for each execution")
        if extra.get("is_infra", 0) or extra.get("webdev_group_pending", False):
            raise VerlContractError("infrastructure failures or ungraded group rewards cannot enter this adapter")
    if len(groups) != prompt_count or any(count != rollout_n for count in groups.values()):
        raise VerlContractError("each prompt must contain exactly rollout.n independent executions")
    return expected


def check_source_tree(root: Path | str, *, patched: bool = False, check_git: bool = True) -> dict[str, str]:
    """Validate fixed source fingerprints and actual AST interfaces, offline."""
    root = Path(root).resolve()
    manifest = json.loads(Path(__file__).with_name("source_contract.json").read_text())
    if check_git:
        proc = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], text=True, capture_output=True, check=True)
        if proc.stdout.strip() != UPSTREAM_COMMIT:
            raise VerlContractError(f"verl must be checked out at {UPSTREAM_COMMIT}")
    checked = {}
    for relative, contract in manifest["files"].items():
        path = root / relative
        if not path.is_file():
            raise VerlContractError(f"missing upstream contract file: {relative}")
        source = path.read_bytes()
        expected = contract["patched_sha256" if patched else "original_sha256"]
        digest = hashlib.sha256(source).hexdigest()
        if digest != expected:
            raise VerlContractError(f"unexpected source for {relative}; exact pin or exact patch differs")
        tree = ast.parse(source, filename=str(path))
        functions = {node.name: node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
        for name, parameters in contract.get("functions", {}).items():
            if name not in functions:
                raise VerlContractError(f"missing upstream function {name} in {relative}")
            fn = functions[name]
            actual = [a.arg for a in (*fn.args.posonlyargs, *fn.args.args, *fn.args.kwonlyargs)]
            if actual != parameters:
                raise VerlContractError(f"signature mismatch for {name}: {actual!r}")
        checked[relative] = digest
    return checked

