"""Static integrity checks on task records, before any image is touched.

These checks only read the dataset row. They cannot see git objects or file
timestamps inside an image; ``environment.audit_image`` does that. A task is
trainable only after BOTH pass and an attestation is stored.
"""
from __future__ import annotations

from collections import Counter
from typing import Iterable

from ..data import TaskRecord
from .environment import EnvFinding, test_patch_needles


def audit_task_record(record: TaskRecord) -> list[EnvFinding]:
    findings: list[EnvFinding] = []
    instance = record.private_instance
    visible = "\n".join(message.get("content", "") for message in record.model_input())
    visible += "\n" + str(instance.get("problem_statement", ""))
    test_patch = instance.get("test_patch") or ""
    if record.subset == "music":
        return findings
    image = str(instance.get("docker_image", ""))
    if "@sha256:" not in image:
        findings.append(EnvFinding("mutable_image_reference", "warning",
                                   f"{image or 'missing image'} is a mutable tag; the attestation must bind the "
                                   "hardened image ID that is actually run"))
    if record.subset == "code" and not test_patch:
        findings.append(EnvFinding("missing_hidden_tests", "warning", "code task has no test_patch to grade with"))
    needles = test_patch_needles(test_patch, limit=10**6)
    leaked = [needle for needle in needles if needle in visible]
    # Issues legitimately quote a few lines that tests later reuse; a leak is
    # a substantial share of the hidden test's distinctive added lines.
    if leaked and len(leaked) >= 3 and len(leaked) >= 0.3 * len(needles):
        findings.append(EnvFinding("hidden_tests_in_prompt", "leak",
                                   f"{len(leaked)}/{len(needles)} distinctive hidden-test lines appear in "
                                   "model-visible text", tuple(leaked[:5])))
    elif leaked:
        findings.append(EnvFinding("test_lines_quoted_in_prompt", "warning",
                                   f"{len(leaked)}/{len(needles)} hidden-test lines also appear in the issue text",
                                   tuple(leaked[:5])))
    command = str(instance.get("test_command", ""))
    cwd = str(instance.get("cwd", "")).rstrip("/")
    if cwd and cwd + "/" in command:
        findings.append(EnvFinding("grading_files_in_workspace", "warning",
                                   f"test command runs a script inside the agent workspace ({command[:120]})"))
    return findings


def summarize_dataset(records: Iterable[TaskRecord]) -> dict:
    counts: Counter = Counter()
    leaking: list[str] = []
    total = 0
    for record in records:
        total += 1
        findings = audit_task_record(record)
        for finding in findings:
            counts[(finding.check, finding.severity)] += 1
        if any(f.severity == "leak" for f in findings):
            leaking.append(record.task_id)
    return {"tasks": total, "static_leak_tasks": len(leaking), "leaking_task_ids": leaking[:100],
            "findings": {f"{check}:{severity}": n for (check, severity), n in sorted(counts.items())},
            "note": "Static checks only. Every trainable task still needs an image audit and attestation."}
