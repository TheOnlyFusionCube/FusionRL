"""Optional, separate full-vocabulary on-policy distillation stage.

Reverse KL is a FusionRL choice, not an undisclosed equation attributed to the
DeepSeek report. Student prefixes must be freshly sampled for this stage.
"""
from __future__ import annotations

from typing import Any


def full_vocabulary_reverse_kl(
    student_logits: Any,
    teacher_logits: Any,
    target_mask: Any,
    *,
    student_tokenizer_fingerprint: str,
    teacher_tokenizer_fingerprint: str,
) -> Any:
    """Mean KL(student || teacher) on student-visited, identically tokenized contexts.

    A fingerprint must include token-to-id mapping, special tokens, tokenizer
    normalization and the chat-template contract. Equal vocabulary SIZE alone
    is insufficient. Teacher logits are always detached.
    """
    import torch

    if not all(torch.is_tensor(x) for x in (student_logits,teacher_logits,target_mask)):
        raise TypeError("OPD inputs must be torch tensors")
    if not student_logits.is_floating_point() or not teacher_logits.is_floating_point():
        raise TypeError("OPD logits must be floating point")
    if not student_tokenizer_fingerprint or student_tokenizer_fingerprint != teacher_tokenizer_fingerprint:
        raise ValueError("Full-vocabulary OPD requires identical tokenizer fingerprints")
    if student_logits.shape != teacher_logits.shape or student_logits.ndim < 2:
        raise ValueError("Teacher and student must provide aligned full-vocabulary logits")
    if target_mask.shape != student_logits.shape[:-1] or target_mask.dtype != torch.bool:
        raise ValueError("Invalid OPD target mask")
    if len({student_logits.device, teacher_logits.device, target_mask.device}) != 1:
        raise ValueError("OPD inputs must share a device")
    if not torch.isfinite(student_logits[target_mask]).all().item() or not torch.isfinite(teacher_logits[target_mask]).all().item():
        raise ValueError("Non-finite OPD logits")
    # Select before softmax so NaN in masked padding cannot contaminate gradients.
    dtype=torch.float64 if torch.float64 in (student_logits.dtype,teacher_logits.dtype) else torch.float32
    student = student_logits[target_mask].to(dtype)
    teacher = teacher_logits.detach()[target_mask].to(dtype)
    if not target_mask.any().item():
        return student.sum() * 0.0
    student_logp = torch.log_softmax(student, dim=-1)
    teacher_logp = torch.log_softmax(teacher, dim=-1)
    return (student_logp.exp() * (student_logp - teacher_logp)).sum(dim=-1).mean()

