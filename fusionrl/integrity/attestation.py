"""Fail-closed admission: train only on environments that passed an audit.

An attestation binds a task to the exact hardened image ID that was audited,
the network flags under which egress was probed and found blocked, and the
audit/rules versions. ``IntegrityGate`` refuses a task when any of these is
missing or differs at dispatch time, or when the task has been revoked by
the integrity ledger. Re-run the audit after changing the image, runner
network or audit version.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Mapping, Sequence

from .environment import AUDIT_VERSION, AuditReport
from .monitor import IntegrityLedger
from .rules import RULES_VERSION

ATTESTATION_SCHEMA = "fusionrl-environment-attestation-v1"
_IMAGE_ID = re.compile(r"^sha256:[0-9a-f]{64}$")


class IntegrityError(RuntimeError):
    """A task may not be used for training under the current evidence."""


@dataclass(frozen=True)
class EnvironmentAttestation:
    task_id: str
    image_id: str                   # Content-addressed ID of the hardened image actually run.
    source_image: str
    network_args: tuple[str, ...]   # Exact runner flags under which egress was probed.
    network_reachable: tuple[str, ...]
    audit_passed: bool
    audit_digest: str
    audit_version: str = AUDIT_VERSION
    rules_version: str = RULES_VERSION
    created_unix: float = 0.0
    schema: str = ATTESTATION_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "network_args", tuple(self.network_args))
        object.__setattr__(self, "network_reachable", tuple(self.network_reachable))
        if not self.task_id:
            raise ValueError("attestation needs a task_id")
        if not _IMAGE_ID.match(self.image_id):
            raise ValueError("attestation image_id must be a content-addressed sha256 image ID, not a tag")

    @property
    def passed(self) -> bool:
        return self.audit_passed and not self.network_reachable

    @classmethod
    def from_report(cls, task_id: str, report: AuditReport, *, source_image: str, network_args: Sequence[str],
                    network_probe: Mapping[str, bool]) -> "EnvironmentAttestation":
        if not network_probe:
            raise ValueError("an attestation requires a network probe result")
        return cls(task_id=task_id, image_id=report.subject, source_image=source_image,
                   network_args=tuple(network_args),
                   network_reachable=tuple(sorted(url for url, ok in network_probe.items() if ok)),
                   audit_passed=report.passed, audit_digest=report.digest, created_unix=time.time())

    def to_dict(self) -> dict:
        return asdict(self)


def _filename(task_id: str) -> str:
    return hashlib.sha256(task_id.encode()).hexdigest()[:32] + ".json"


class AttestationStore:
    """One JSON file per task under a directory (works on shared storage)."""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)

    def save(self, attestation: EnvironmentAttestation, report: AuditReport | None = None) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        payload = {"attestation": attestation.to_dict(), "report": report.to_dict() if report else None}
        path = self.directory / _filename(attestation.task_id)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        temporary.replace(path)
        return path

    def load(self, task_id: str) -> EnvironmentAttestation | None:
        path = self.directory / _filename(task_id)
        if not path.exists():
            return None
        data = json.loads(path.read_text())["attestation"]
        if data.get("schema") != ATTESTATION_SCHEMA or data.get("task_id") != task_id:
            raise IntegrityError(f"attestation file for {task_id!r} is malformed or belongs to another task")
        return EnvironmentAttestation(**data)


class IntegrityGate:
    """Check a task immediately before dispatching any of its executions."""

    def __init__(self, store: AttestationStore, *, ledger: IntegrityLedger | None = None,
                 audit_version: str = AUDIT_VERSION, rules_version: str = RULES_VERSION):
        self.store = store
        self.ledger = ledger
        self.audit_version = audit_version
        self.rules_version = rules_version

    def check(self, task_id: str, *, image_id: str, network_args: Sequence[str]) -> EnvironmentAttestation:
        if self.ledger is not None:
            if self.ledger.should_halt:
                raise IntegrityError(f"training halted: hack rate {self.ledger.window_hack_rate:.3f} "
                                     "over the recent window")
            if task_id in self.ledger.revoked:
                raise IntegrityError(f"task {task_id!r} revoked: {self.ledger.revoked[task_id]}; re-audit it")
        attestation = self.store.load(task_id)
        if attestation is None:
            raise IntegrityError(f"task {task_id!r} has no environment attestation")
        if not attestation.passed:
            raise IntegrityError(f"task {task_id!r} failed its environment audit or network probe")
        if attestation.image_id != image_id:
            raise IntegrityError(f"task {task_id!r} would run image {image_id}, but {attestation.image_id} "
                                 "was audited")
        if tuple(network_args) != attestation.network_args:
            raise IntegrityError(f"task {task_id!r} network flags differ from the probed flags")
        if attestation.audit_version != self.audit_version or attestation.rules_version != self.rules_version:
            raise IntegrityError(f"task {task_id!r} was attested under an older audit or rules version")
        return attestation
