"""Check a real-model run before committing GPU time.

Reports hardware, precision support, the memory a full fine-tune needs, the
tokenizer fingerprint, selective-head consistency, and the rollout/training
log-prob gap measured on prompts from the actual training file. Exit status
is nonzero when any check fails.
"""
from __future__ import annotations

from typing import Any

from .run_config import RunConfig


def _bytes_per_parameter(precision: str) -> int:
    # weights + gradients + AdamW (two fp32 moments)
    return {"fp32": 4 + 4 + 8, "bf16-autocast": 4 + 4 + 8, "bf16": 2 + 2 + 8}[precision]


def run_preflight(cfg: RunConfig, *, prompts: int = 8, model: Any = None, tokenizer: Any = None) -> dict[str, Any]:
    import torch

    from .backends.hf import HFPolicy
    from .runner import _messages, _resolve_device_and_precision, load_model, load_tasks

    report: dict[str, Any] = {"checks": {}, "torch": torch.__version__}
    device, precision = _resolve_device_and_precision(cfg)
    report.update({"device": device, "precision": precision})
    if device == "cuda":
        properties = torch.cuda.get_device_properties(0)
        report["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        report["gpu_memory_gib"] = round(properties.total_memory / 2**30, 2)
        report["checks"]["nccl_available"] = bool(torch.distributed.is_nccl_available())
        report["checks"]["bf16_supported"] = bool(torch.cuda.is_bf16_supported())
    if model is None:
        model, tokenizer = load_model(cfg, device, precision)
    from .backends.hf import disable_dropout
    model = model.to(device)
    model.train()
    report["dropout_settings_disabled"] = disable_dropout(model)
    parameters = sum(p.numel() for p in model.parameters())
    need = parameters * _bytes_per_parameter(precision) / 2**30
    report["parameters"] = parameters
    report["estimated_train_state_gib"] = round(need, 2)
    if device == "cuda":
        # Activations, KV cache and fragmentation need headroom beyond the state.
        report["checks"]["fits_single_gpu"] = need < 0.8 * report["gpu_memory_gib"]
    autocast = torch.bfloat16 if precision == "bf16-autocast" else None
    policy = HFPolicy(model, tokenizer, max_tokens_per_microbatch=cfg.learner.max_tokens_per_microbatch,
                      autocast_dtype=autocast)
    report["tokenizer_fingerprint"] = policy.fingerprint
    report["scoring"] = policy.scoring
    report["head_consistency_max_abs"] = policy.check_head_consistency()
    report["checks"]["head_consistency"] = report["head_consistency_max_abs"] < 1e-2 or policy.scoring == "full"
    report["checks"]["eos_token_defined"] = bool(policy.eos_ids)
    tasks = load_tasks(cfg.data.train)[:prompts]
    encoded = [policy.encode_chat(_messages(task, cfg.data.system_prompt)) for task in tasks]
    report["checks"]["prompts_within_limit"] = all(len(p) <= cfg.data.max_prompt_tokens for p in encoded)
    generator = torch.Generator(device=device).manual_seed(cfg.run.seed)
    parity = policy.rollout_parity(encoded, max_new_tokens=min(cfg.rollout.max_new_tokens, 256), generator=generator)
    report["rollout_parity"] = parity.__dict__
    report["checks"]["rollout_parity"] = parity.within(cfg.parity.max_mean_abs_gap, cfg.parity.max_abs_gap)
    report["passed"] = all(report["checks"].values())
    return report
