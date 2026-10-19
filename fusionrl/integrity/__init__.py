"""Reward-hacking defenses: environment audits, attestations, monitoring, rules.

Layers (each assumes the others can fail):

1. ``environment``: audit and harden task images so solution leaks are gone.
2. ``attestation``: refuse to train on a task without a passing audit of the
   exact image and network flags that will run.
3. ``rules``: tell the model precisely which channels are prohibited.
4. ``monitor``: scan transcripts; confirmed or attempted use of a prohibited
   channel is a hack; revoke leaky tasks and halt on sustained hacking.
5. ``grading``: drop candidate edits to hidden tests before grading.
"""
from .attestation import AttestationStore, EnvironmentAttestation, IntegrityError, IntegrityGate
from .grading import SanitizedPatch, sanitize_candidate_patch
from .monitor import (GroupIntegrityDecision, HackFinding, HackPolicy, IntegrityLedger, TranscriptStep,
                      judge_group, scan_transcript)
from .rules import RULES_VERSION, integrity_rules, with_integrity_rules

__all__ = [
    "AttestationStore", "EnvironmentAttestation", "GroupIntegrityDecision", "HackFinding", "HackPolicy",
    "IntegrityError", "IntegrityGate", "IntegrityLedger", "RULES_VERSION", "SanitizedPatch", "TranscriptStep",
    "integrity_rules", "judge_group", "sanitize_candidate_patch", "scan_transcript", "with_integrity_rules",
]
