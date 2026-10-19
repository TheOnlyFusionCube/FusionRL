"""Grader-side hardening for repository tasks.

The grader owns the hidden tests. Before applying ``test_patch`` it removes
every part of the candidate's diff that touches files the test patch defines
or grading-owned paths, so a policy cannot pass by editing, deleting or
replacing tests. Removed sections are reported for the integrity monitor.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

from .environment import patch_files

_SECTION = re.compile(r"^diff --git a/(\S+) b/(\S+)$", re.MULTILINE)
# Test-only configuration a candidate never needs to change to fix the code.
_TEST_INFRA = re.compile(r"(^|/)(conftest\.py|pytest\.ini|jest\.config\.[cm]?[jt]s|\.mocharc[^/]*|"
                         r"karma\.conf\.js|phpunit\.xml(\.dist)?|vitest\.config\.[cm]?[jt]s)$")


@dataclass(frozen=True)
class SanitizedPatch:
    patch: str
    removed_files: tuple[str, ...]

    @property
    def tampered(self) -> bool:
        return bool(self.removed_files)


def sanitize_candidate_patch(candidate: str, *, test_patch: str, protected_paths: Iterable[str] = (),
                             protect_test_infrastructure: bool = True) -> SanitizedPatch:
    """Drop candidate diff sections that touch hidden-test or grading files."""
    protected = set(patch_files(test_patch)) | {p.strip().lstrip("./") for p in protected_paths if p.strip()}
    starts = [m.start() for m in _SECTION.finditer(candidate)]
    if not starts:
        if candidate.strip():
            raise ValueError("candidate patch is not a git-style unified diff")
        return SanitizedPatch(candidate, ())
    kept, removed = [candidate[:starts[0]]], []
    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else len(candidate)
        section = candidate[start:end]
        match = _SECTION.match(section)
        paths = {match.group(1), match.group(2)} - {"/dev/null"}
        if paths & protected or (protect_test_infrastructure and any(_TEST_INFRA.search(p) for p in paths)):
            removed.extend(sorted(paths))
        else:
            kept.append(section)
    return SanitizedPatch("".join(kept), tuple(dict.fromkeys(removed)))
