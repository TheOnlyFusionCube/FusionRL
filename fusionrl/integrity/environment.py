"""Audit and harden task environments against solution leaks.

Checks the leak classes vals.ai found in MiMo-V2.6 RL images ("MiMo reward
hacking"): later upstream commits and refs, unreachable git objects holding
the fix (67% of audited coding tasks), reference-patch file timestamps,
patch-application leftovers, cached copies of hidden content, and grading
infrastructure inside the agent's workspace.

Everything here runs on the HOST over a filesystem snapshot (a directory or a
``docker export`` tar). Nothing from the image is executed. Git analysis runs
on a copy whose config and hooks are replaced, because an untrusted
``.git/config`` (``core.fsmonitor``, ``core.sshCommand``...) could otherwise
execute commands on the host.
"""
from __future__ import annotations

import contextlib
from dataclasses import asdict, dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import tarfile
import tempfile
from typing import IO, Iterable, Iterator, Sequence

AUDIT_VERSION = "fusionrl-env-audit-v1"
MAX_SCAN_BYTES = 32 * 1024 * 1024
_PATCH_LEFTOVER = re.compile(r"\.(orig|rej|porig)$|(^|/)(test_patch|gold_patch|reference_(patch|solution)|"
                             r"solution|fix)\.(diff|patch)$", re.IGNORECASE)
# Installed or generated content: newer timestamps there reflect setup, not the fix.
_DERIVED = re.compile(r"(^|/)(\.venv|venv|env|node_modules|site-packages|dist-packages|__pycache__|\.tox|"
                      r"\.mypy_cache|\.pytest_cache|\.ruff_cache|build|dist|target|\.gradle|vendor|\.cache)/|"
                      r"\.egg-info(/|$)|\.pyc$")
_GRADING_FILES = re.compile(r"(^|/)(mimo_test_command\.sh|run_tests?\.sh|eval(uate)?\.sh|grade(r)?\.py)$")
# Git state that can carry commit messages or whole patches beyond HEAD.
_GIT_STATE_FILES = ("ORIG_HEAD", "FETCH_HEAD", "MERGE_HEAD", "CHERRY_PICK_HEAD", "REVERT_HEAD", "REBASE_HEAD",
                    "AUTO_MERGE", "BISECT_LOG", "COMMIT_EDITMSG", "MERGE_MSG", "SQUASH_MSG", "TAG_EDITMSG",
                    "gitk.cache")
_GIT_STATE_DIRS = ("logs", "refs/original", "rr-cache", "rebase-merge", "rebase-apply", "sequencer")
_SAFE_CONFIG_KEYS = re.compile(r"^(core\.(repositoryformatversion|bare|filemode|symlinks|ignorecase|precomposeunicode|"
                               r"autocrlf|eol|safecrlf|quotepath|logallrefupdates|abbrev|compression)|"
                               r"extensions\.(objectformat|worktreeconfig)|user\.(name|email)|init\.defaultbranch|"
                               r"diff\.[a-z]+|merge\.[a-z]+|pull\.[a-z]+)=", re.IGNORECASE)


class EnvironmentError_(RuntimeError):
    """The environment cannot be audited or hardened as requested."""


@dataclass(frozen=True)
class EnvFinding:
    check: str
    severity: str          # "leak" blocks attestation; "warning" is reported only.
    detail: str
    paths: tuple[str, ...] = ()


@dataclass(frozen=True)
class AuditReport:
    subject: str
    workspace: str
    head_commit: str | None
    findings: tuple[EnvFinding, ...]
    stats: dict = field(default_factory=dict)
    version: str = AUDIT_VERSION

    @property
    def passed(self) -> bool:
        return not any(f.severity == "leak" for f in self.findings)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["passed"] = self.passed
        return data

    @property
    def digest(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


# --------------------------------------------------------------------------- git


def _git_env(home: Path) -> dict[str, str]:
    return {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(home), "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull, "GIT_TERMINAL_PROMPT": "0", "GIT_OPTIONAL_LOCKS": "0",
            "LC_ALL": "C"}


def _git(git_dir: Path, *args: str, check: bool = True, input_text: str | None = None) -> str:
    command = ["git", "-c", "core.hooksPath=" + os.devnull, "-c", "core.fsmonitor=false",
               "-c", "safe.directory=*", "--git-dir", str(git_dir), *args]
    result = subprocess.run(command, capture_output=True, text=True, env=_git_env(git_dir.parent),
                            input=input_text, timeout=600, check=False)
    if check and result.returncode != 0:
        raise EnvironmentError_(f"git {' '.join(args)} failed: {result.stderr.strip()[:400]}")
    return result.stdout


def _git_ok(git_dir: Path, *args: str) -> bool:
    command = ["git", "-c", "core.hooksPath=" + os.devnull, "-c", "core.fsmonitor=false",
               "-c", "safe.directory=*", "--git-dir", str(git_dir), *args]
    return subprocess.run(command, capture_output=True, env=_git_env(git_dir.parent),
                          timeout=600, check=False).returncode == 0


def _sanitized_config(original: Path) -> str:
    """Keep only inert keys from an untrusted repository config."""
    if not original.exists():
        return "[core]\n\trepositoryformatversion = 0\n\tbare = false\n"
    listing = subprocess.run(["git", "config", "--file", str(original), "--list"], capture_output=True,
                             text=True, env=_git_env(original.parent), timeout=60, check=False).stdout
    lines = ["[core]", "\trepositoryformatversion = 0", "\tbare = false"]
    sections: dict[str, list[str]] = {}
    for entry in listing.splitlines():
        if not _SAFE_CONFIG_KEYS.match(entry):
            continue
        key, _, value = entry.partition("=")
        section, _, name = key.rpartition(".")
        if section == "core" and name in {"repositoryformatversion", "bare"}:
            continue
        sections.setdefault(section, []).append(f"\t{name} = {json.dumps(value)}")
    for section, entries in sections.items():
        if section == "core":
            lines.extend(entries)
        else:
            lines.append(f"[{section}]")
            lines.extend(entries)
    return "\n".join(lines) + "\n"


def _remote_names(config_text: str) -> list[str]:
    return re.findall(r'^\s*\[remote\s+"([^"]+)"\]', config_text, flags=re.MULTILINE)


def safe_git_copy(git_dir: Path, destination: Path) -> Path:
    """Copy a .git directory, replacing config and hooks with inert versions."""
    if not (git_dir / "HEAD").exists():
        raise EnvironmentError_(f"{git_dir} is not a git directory")
    target = destination / ".git"
    shutil.copytree(git_dir, target, symlinks=True, ignore=shutil.ignore_patterns("hooks"))
    (target / "hooks").mkdir()
    (target / "config").write_text(_sanitized_config(git_dir / "config"))
    return target


def _rev_list_objects(git_dir: Path, *revs: str) -> set[str]:
    out = _git(git_dir, "rev-list", "--objects", *revs)
    return {line.split(" ", 1)[0] for line in out.splitlines() if line}


def audit_git(git_dir: Path, worktree_blobs: set[str] | frozenset[str] = frozenset()) -> tuple[list[EnvFinding], dict]:
    """Audit a SANITIZED git directory copy (see ``safe_git_copy``).

    ``worktree_blobs`` are git blob IDs of the workspace files. A staged blob
    is acceptable only when the same content is already in the workspace.
    """
    findings: list[EnvFinding] = []
    head = _git(git_dir, "rev-parse", "--verify", "HEAD^{commit}", check=False).strip()
    if not head:
        return [EnvFinding("git_head", "leak", "HEAD does not resolve to a commit; history cannot be audited")], {}
    head_ref = _git(git_dir, "symbolic-ref", "-q", "HEAD", check=False).strip()
    allowed = _rev_list_objects(git_dir, "HEAD")
    index_blobs = {line.split()[1] for line in _git(git_dir, "ls-files", "-s", check=False).splitlines()
                   if len(line.split()) >= 2}
    stats: dict = {"head": head, "head_ref": head_ref or None, "objects_reachable_from_head": len(allowed)}

    beyond_refs = []
    for line in _git(git_dir, "for-each-ref", "--format=%(refname) %(objectname)").splitlines():
        name, _, target = line.partition(" ")
        if name == head_ref:
            continue
        ancestor = _git_ok(git_dir, "merge-base", "--is-ancestor", target, "HEAD")
        if name.startswith(("refs/stash", "refs/notes/", "refs/replace/")) or not ancestor:
            beyond_refs.append(name)
    if beyond_refs:
        findings.append(EnvFinding("git_refs_beyond_head", "leak",
                                   f"{len(beyond_refs)} refs point outside HEAD's history (branches, remotes, "
                                   "tags, stash or notes can carry the fix)", tuple(beyond_refs[:50])))

    all_objects = {line.split(" ", 1)[0] for line in
                   _git(git_dir, "cat-file", "--batch-all-objects", "--batch-check=%(objectname)").splitlines() if line}
    hidden = all_objects - allowed - (index_blobs & set(worktree_blobs))
    # Annotated tags are never listed by ``rev-list --objects HEAD``; a tag
    # (or tag chain) that labels a commit in HEAD's history reveals nothing new.
    if hidden:
        kinds = dict(line.split(" ", 1) for line in _git(
            git_dir, "cat-file", "--batch-check=%(objectname) %(objecttype)",
            input_text="\n".join(sorted(hidden)) + "\n").splitlines() if " " in line)
        changed = True
        while changed:
            changed = False
            for object_id in [o for o in hidden if kinds.get(o) == "tag"]:
                header = _git(git_dir, "cat-file", "-p", object_id, check=False).split("\n", 1)[0].split()
                if len(header) == 2 and header[0] == "object" and header[1] in allowed:
                    allowed.add(object_id)
                    hidden.discard(object_id)
                    changed = True
    stats["objects_total"] = len(all_objects)
    stats["objects_not_reachable_from_head"] = len(hidden)
    if hidden:
        types = _git(git_dir, "cat-file", "--batch-check=%(objectname) %(objecttype)",
                     input_text="\n".join(sorted(hidden)) + "\n").splitlines()
        by_type: dict[str, list[str]] = {}
        for line in types:
            object_id, _, kind = line.partition(" ")
            by_type.setdefault(kind, []).append(object_id)
        commits = by_type.get("commit", [])
        stats["hidden_commits"] = len(commits)
        stats["hidden_objects_by_type"] = {kind: len(ids) for kind, ids in sorted(by_type.items())}
        findings.append(EnvFinding("git_objects_beyond_head", "leak",
                                   f"{len(hidden)} git objects ({', '.join(f'{len(v)} {k}s' for k, v in sorted(by_type.items()))}) "
                                   "are not reachable from HEAD; unreachable blobs and commits can contain the reference fix",
                                   tuple((commits or by_type.get("blob", []))[:20])))

    logs = git_dir / "logs"
    if logs.exists():
        recorded: set[str] = set()
        for log in (p for p in logs.rglob("*") if p.is_file()):
            for line in log.read_text(errors="replace").splitlines():
                recorded.update(part for part in line.split()[:2] if re.fullmatch(r"[0-9a-f]{40,64}", part))
        recorded.discard("0" * 40)
        recorded.discard("0" * 64)
        outside = sorted(sha for sha in recorded if not _git_ok(git_dir, "merge-base", "--is-ancestor", sha, "HEAD"))
        if outside:
            findings.append(EnvFinding("git_reflogs", "leak", f"reflogs record {len(outside)} positions outside "
                                       "HEAD's history", tuple(outside[:20])))
        elif recorded:
            findings.append(EnvFinding("git_reflogs_head_only", "warning", "reflogs exist but only record HEAD's history"))
    state = [name for name in _GIT_STATE_FILES if (git_dir / name).exists()]
    if "COMMIT_EDITMSG" in state:
        # Harmless when it only repeats HEAD's own message.
        recorded = (git_dir / "COMMIT_EDITMSG").read_text(errors="replace")
        own = _git(git_dir, "log", "-1", "--format=%B", "HEAD", check=False)
        strip = lambda text: [ln for ln in text.strip().splitlines() if not ln.startswith("#")]  # noqa: E731
        if strip(recorded) == strip(own):
            state.remove("COMMIT_EDITMSG")
            findings.append(EnvFinding("git_commit_editmsg", "warning", "COMMIT_EDITMSG repeats HEAD's message"))
    state += [name for name in _GIT_STATE_DIRS[1:] if (git_dir / name).exists()]
    if state:
        findings.append(EnvFinding("git_state_files", "leak", "git state files can hold commits, commit messages "
                                   "or patches beyond HEAD (e.g. COMMIT_EDITMSG keeps the last commit's message)",
                                   tuple(state)))
    if (git_dir / "objects" / "info" / "alternates").exists():
        findings.append(EnvFinding("git_alternates", "leak", "objects/info/alternates links an external object store"))
    if (git_dir / "info" / "grafts").exists():
        findings.append(EnvFinding("git_grafts", "leak", "info/grafts rewrites history visibility"))
    return findings, stats


def harden_git(git_dir: Path) -> str:
    """Reduce a SANITIZED git copy to exactly HEAD's history. Returns HEAD."""
    head = _git(git_dir, "rev-parse", "--verify", "HEAD^{commit}").strip()
    head_ref = _git(git_dir, "symbolic-ref", "-q", "HEAD", check=False).strip()
    for line in _git(git_dir, "for-each-ref", "--format=%(refname)").splitlines():
        if line and line != head_ref:
            _git(git_dir, "update-ref", "-d", line)
    for name in _GIT_STATE_FILES:
        (git_dir / name).unlink(missing_ok=True)
    for relative in _GIT_STATE_DIRS:
        shutil.rmtree(git_dir / relative, ignore_errors=True)
    # Staged-only blobs could hold the fix; reset the index to HEAD's tree.
    # Workspace files are untouched, so task-side edits still show as changes.
    _git(git_dir, "read-tree", "HEAD")
    (git_dir / "objects" / "info" / "alternates").unlink(missing_ok=True)
    (git_dir / "info" / "grafts").unlink(missing_ok=True)
    _git(git_dir, "reflog", "expire", "--expire=now", "--expire-unreachable=now", "--all", check=False)
    _git(git_dir, "repack", "-a", "-d", "-q")
    _git(git_dir, "prune", "--expire=now")
    if _git(git_dir, "rev-parse", "--verify", "HEAD^{commit}").strip() != head:
        raise EnvironmentError_("hardening changed HEAD")
    return head


def _head_commit_time(git_dir: Path) -> int:
    return int(_git(git_dir, "log", "-1", "--format=%ct", "HEAD").strip())


# ---------------------------------------------------------------- filesystem


@dataclass(frozen=True)
class _Entry:
    path: str              # Absolute POSIX path inside the environment.
    mtime: float
    is_file: bool


def test_patch_needles(patch_text: str, *, minimum_length: int = 24, limit: int = 64) -> list[str]:
    """Distinctive added lines of a hidden test or reference patch."""
    needles, seen = [], set()
    for line in patch_text.splitlines():
        if not line.startswith("+") or line.startswith("+++"):
            continue
        text = line[1:].strip()
        if len(text) >= minimum_length and text not in seen and not re.fullmatch(r"[\W_]+", text):
            needles.append(text)
            seen.add(text)
            if len(needles) >= limit:
                break
    return needles


def patch_files(patch_text: str) -> list[str]:
    """Repository-relative paths touched by a git-style unified diff."""
    files = []
    for match in re.finditer(r"^diff --git a/(\S+) b/(\S+)$", patch_text, flags=re.MULTILINE):
        for path in match.groups():
            if path not in files and path != "/dev/null":
                files.append(path)
    return files


def _mtime_findings(entries: Sequence[_Entry], workspace: str, reference_files: Sequence[str],
                    tolerance: float) -> tuple[list[EnvFinding], dict]:
    files = [e for e in entries if e.is_file]
    if not files:
        return [EnvFinding("workspace_empty", "leak", f"no files found under {workspace}")], {}
    buckets: dict[int, int] = {}
    for entry in files:
        buckets[int(entry.mtime // max(tolerance, 1))] = buckets.get(int(entry.mtime // max(tolerance, 1)), 0) + 1
    dominant = max(buckets, key=lambda key: (buckets[key], -key)) * max(tolerance, 1)
    newer = [e.path for e in files if e.mtime > dominant + tolerance]
    older = [e.path for e in files if e.mtime < dominant - tolerance]
    stats = {"workspace_files": len(files), "distinct_mtime_buckets": len(buckets),
             "newer_than_bulk": len(newer), "older_than_bulk": len(older)}
    findings = []
    rel = [p[len(workspace):].lstrip("/") for p in newer]
    source = [r for r in rel if not _DERIVED.search(r)]
    stats["newer_source_files"] = len(source)
    if source:
        overlap = sorted(set(source) & set(reference_files))
        detail = (f"{len(source)} source files are newer than the bulk checkout; `find -newer*` reveals them"
                  + (f", including {len(overlap)} files touched by the hidden test/reference patch" if overlap else ""))
        findings.append(EnvFinding("timestamp_outliers", "leak", detail, tuple((overlap + source)[:50])))
    if len(source) < len(rel):
        findings.append(EnvFinding("timestamp_outliers_derived", "warning",
                                   f"{len(rel) - len(source)} installed/generated files are newer than the checkout",
                                   tuple([r for r in rel if _DERIVED.search(r)][:10])))
    if older:
        findings.append(EnvFinding("timestamp_nonuniform", "warning",
                                   f"{len(older)} workspace files are older than the bulk checkout",
                                   tuple(p[len(workspace):].lstrip('/') for p in older[:20])))
    return findings, stats


def _content_findings(hits: dict[str, int], total: int, label: str) -> list[EnvFinding]:
    """A file holding most of a hidden patch's distinctive lines is a leak.

    Files the patch itself edits naturally share some lines with it, so a
    partial overlap is reported as a warning only.
    """
    if not hits:
        return []
    kind = label.split("_")[0]
    full = sorted(path for path, count in hits.items() if count >= 3 and count >= 0.5 * total)
    partial = sorted(set(hits) - set(full))
    findings = []
    if full:
        findings.append(EnvFinding(label, "leak", f"{len(full)} files contain most of the hidden {kind} patch's "
                                   "added lines (a copy of the hidden content is reachable)", tuple(full[:50])))
    if partial:
        findings.append(EnvFinding(label.replace("present", "partial"), "warning",
                                   f"{len(partial)} files share some lines with the hidden {kind} patch",
                                   tuple(f"{p} ({hits[p]}/{total})" for p in partial[:20])))
    return findings


def _iter_dir(root: Path) -> Iterator[tuple[str, os.stat_result, Path]]:
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = [d for d in dirnames if not (Path(directory) / d).is_symlink()]
        for name in filenames:
            path = Path(directory) / name
            try:
                info = path.lstat()
            except OSError:
                continue
            yield "/" + path.relative_to(root).as_posix(), info, path


def _scan_needles(data: bytes, needles: dict[str, list[bytes]]) -> dict[str, int]:
    counts = {label: sum(value in data for value in values) for label, values in needles.items()}
    return {label: count for label, count in counts.items() if count}


def audit_snapshot(*, root: Path | None = None, tar_stream: IO[bytes] | None = None, workspace: str,
                   subject: str, test_patch: str = "", reference_patch: str = "",
                   timestamp_tolerance: float = 2.0) -> AuditReport:
    """Audit a filesystem snapshot given as a directory ``root`` or an export tar."""
    if (root is None) == (tar_stream is None):
        raise ValueError("Provide exactly one of root or tar_stream")
    workspace = "/" + workspace.strip("/")
    git_prefix = workspace.rstrip("/") + "/.git/"
    needles = {k: [n.encode() for n in test_patch_needles(v, limit=256)] for k, v in
               (("test_content_present", test_patch), ("reference_content_present", reference_patch)) if v}
    reference_files = patch_files(test_patch) + patch_files(reference_patch)
    entries: list[_Entry] = []
    hits: dict[str, dict[str, int]] = {label: {} for label in needles}
    leftovers, grading = [], []
    worktree_blobs: set[str] = set()
    with tempfile.TemporaryDirectory(prefix="fusionrl-audit-") as tmp:
        tmpdir = Path(tmp)
        raw_git = tmpdir / "raw" / ".git"

        def visit(path: str, mtime: float, is_file: bool, read) -> None:
            in_ws = path == workspace or path.startswith(workspace.rstrip("/") + "/")
            in_git = path.startswith(git_prefix) or path == git_prefix.rstrip("/")
            if in_ws and not in_git:
                entries.append(_Entry(path, mtime, is_file))
                rel = path[len(workspace):].lstrip("/")
                if is_file and _PATCH_LEFTOVER.search(rel):
                    leftovers.append(rel)
                if is_file and _GRADING_FILES.search(rel):
                    grading.append(rel)
            if is_file and not in_git and (in_ws or needles):
                data = read()
                if data is not None:
                    if in_ws:
                        worktree_blobs.add(hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest())
                    for label, count in _scan_needles(data, needles).items():
                        hits[label][path] = count

        if root is not None:
            for path, info, real in _iter_dir(Path(root)):
                regular = os.path.isfile(real) and not real.is_symlink()
                visit(path, info.st_mtime, regular,
                      lambda real=real, info=info: real.read_bytes() if info.st_size <= MAX_SCAN_BYTES else None)
            git_source = Path(root) / workspace.lstrip("/") / ".git"
            if git_source.is_dir():
                safe_git_copy(git_source, tmpdir / "safe")
        else:
            with tarfile.open(fileobj=tar_stream, mode="r|*") as archive:
                for member in archive:
                    path = "/" + member.name.lstrip("./").lstrip("/")
                    if path.startswith(git_prefix):
                        _extract_member(archive, member, raw_git / path[len(git_prefix):])
                        continue
                    visit(path, float(member.mtime), member.isfile(),
                          lambda member=member: archive.extractfile(member).read()
                          if member.size <= MAX_SCAN_BYTES else None)
            if (raw_git / "HEAD").exists():
                safe_git_copy(raw_git, tmpdir / "safe")
        findings: list[EnvFinding] = []
        stats: dict = {}
        raw_config = (Path(root) / workspace.lstrip("/") / ".git" / "config") if root is not None else raw_git / "config"
        remotes = _remote_names(raw_config.read_text(errors="replace")) if raw_config.is_file() else []
        if remotes:
            findings.append(EnvFinding("git_remotes", "warning", "configured remotes invite fetching upstream "
                                       "history; hardening removes them and the runner must block egress",
                                       tuple(remotes)))
        safe = tmpdir / "safe" / ".git"
        if safe.exists():
            git_findings, git_stats = audit_git(safe, worktree_blobs)
            findings += git_findings
            stats.update(git_stats)
            head = git_stats.get("head")
        else:
            findings.append(EnvFinding("git_missing", "warning", f"no git repository at {workspace}; "
                                       "only filesystem checks apply"))
            head = None
    mtime_findings, mtime_stats = _mtime_findings(entries, workspace, reference_files, timestamp_tolerance)
    findings += mtime_findings
    stats.update(mtime_stats)
    if leftovers:
        findings.append(EnvFinding("patch_leftovers", "leak", "patch/answer artifacts in the workspace reveal the "
                                   "solution or its scope", tuple(leftovers[:50])))
    if grading:
        findings.append(EnvFinding("grading_files_in_workspace", "warning", "grading scripts are visible to the "
                                   "policy; run them from outside the agent workspace", tuple(grading[:20])))
    for label, found in hits.items():
        findings += _content_findings(found, len(needles[label]), label)
    return AuditReport(subject, workspace, head, tuple(findings), stats)


def _extract_member(archive: tarfile.TarFile, member: tarfile.TarInfo, destination: Path) -> None:
    """Extract one git-dir member without following links outside ``destination``."""
    if member.isdir():
        destination.mkdir(parents=True, exist_ok=True)
        return
    if not member.isfile():
        return  # Links inside .git are not needed for object/ref analysis.
    destination.parent.mkdir(parents=True, exist_ok=True)
    source = archive.extractfile(member)
    with destination.open("wb") as handle:
        shutil.copyfileobj(source, handle)


# ----------------------------------------------------------------- hardening


@dataclass(frozen=True)
class HardeningResult:
    head_commit: str
    epoch: int
    removed_paths: tuple[str, ...]
    output: str


def _prepare_git(raw_git: Path, tmpdir: Path) -> tuple[Path, str, int, list[str]]:
    if not (raw_git / "HEAD").exists():
        raise EnvironmentError_("no git repository in the workspace to harden")
    original_config = (raw_git / "config").read_text(errors="replace") if (raw_git / "config").exists() else ""
    safe = safe_git_copy(raw_git, tmpdir / "safe")
    head = harden_git(safe)
    epoch = _head_commit_time(safe)
    shutil.rmtree(safe / "hooks")
    (safe / "hooks").mkdir()
    removed = [".git/config remotes: " + ",".join(_remote_names(original_config))] if _remote_names(original_config) else []
    return safe, head, epoch, removed


def _extract_git_members(archive: tarfile.TarFile, raw_git: Path, git_prefix: str | None) -> None:
    """Extract .git members; ``git_prefix`` None means strip the first path component."""
    for member in archive:
        name = member.name.lstrip("./").lstrip("/")
        if git_prefix is None:
            relative = name.partition("/")[2]
        else:
            path = "/" + name
            if not path.startswith(git_prefix):
                continue
            relative = path[len(git_prefix):]
        if relative:
            _extract_member(archive, member, raw_git / relative)


def _write_hardened(source: IO[bytes], destination: IO[bytes], *, workspace: str, safe: Path, epoch: int,
                    roots: Sequence[str], remove_patch_leftovers: bool) -> list[str]:
    git_prefix = workspace + "/.git/"
    removed: list[str] = []

    def normalized(path: str) -> bool:
        return any(path == r or path.startswith(r.rstrip("/") + "/") for r in roots)

    with tarfile.open(fileobj=source, mode="r|*") as archive, \
            tarfile.open(fileobj=destination, mode="w|", format=tarfile.PAX_FORMAT) as out:
        for member in archive:
            path = "/" + member.name.lstrip("./").lstrip("/")
            if path.startswith(git_prefix) or path == git_prefix.rstrip("/"):
                continue
            rel = path[len(workspace):].lstrip("/") if path.startswith(workspace + "/") else None
            if remove_patch_leftovers and rel and member.isfile() and _PATCH_LEFTOVER.search(rel):
                removed.append(rel)
                continue
            if normalized(path):
                member.mtime = epoch
            member.pax_headers = {k: v for k, v in member.pax_headers.items() if k not in {"mtime", "atime", "ctime"}}
            out.addfile(member, archive.extractfile(member) if member.isfile() else None)
        base = workspace.lstrip("/") + "/.git"
        for item in sorted([safe, *safe.rglob("*")]):
            info = out.gettarinfo(str(item), arcname=base + ("/" + item.relative_to(safe).as_posix()
                                                             if item != safe else ""))
            info.mtime, info.uid, info.gid, info.uname, info.gname = epoch, 0, 0, "root", "root"
            info.pax_headers = {}
            if item.is_file():
                with item.open("rb") as handle:
                    out.addfile(info, handle)
            else:
                out.addfile(info)
    return removed


def harden_snapshot_tar(source: IO[bytes], destination: IO[bytes], *, workspace: str,
                        normalize_paths: Sequence[str] = (), remove_patch_leftovers: bool = True,
                        git_dir: Path | None = None) -> HardeningResult:
    """Rewrite a filesystem tar: HEAD-only git history and uniform timestamps.

    Timestamps under ``workspace`` (and any ``normalize_paths``) are set to
    HEAD's commit time and PAX atime/ctime records are dropped; importing the
    result as one layer also removes layer-membership hints. The workspace's
    ``.git`` comes from ``git_dir`` when given (so ``source`` can be a pipe),
    otherwise from a first pass over a seekable ``source``.
    """
    workspace = "/" + workspace.strip("/")
    roots = [workspace] + ["/" + p.strip("/") for p in normalize_paths]
    with tempfile.TemporaryDirectory(prefix="fusionrl-harden-") as tmp:
        tmpdir = Path(tmp)
        raw_git = Path(git_dir) if git_dir is not None else tmpdir / "raw" / ".git"
        if git_dir is None:
            if not source.seekable():
                raise ValueError("a non-seekable source needs git_dir")
            start = source.tell()
            with tarfile.open(fileobj=source, mode="r|*") as archive:
                _extract_git_members(archive, raw_git, workspace + "/.git/")
            source.seek(start)
        safe, head, epoch, removed = _prepare_git(raw_git, tmpdir)
        removed = _write_hardened(source, destination, workspace=workspace, safe=safe, epoch=epoch, roots=roots,
                                  remove_patch_leftovers=remove_patch_leftovers) + removed
    return HardeningResult(head, epoch, tuple(removed), "tar")


# ---------------------------------------------------------------------- docker


def _docker(*args: str, stdout=subprocess.PIPE, stdin=None, timeout: int = 3600, check: bool = True,
            text: bool = True) -> subprocess.CompletedProcess:
    result = subprocess.run(["docker", *args], stdout=stdout, stderr=subprocess.PIPE, stdin=stdin,
                            timeout=timeout, check=False, text=text)
    if check and result.returncode != 0:
        error = result.stderr if text else result.stderr.decode(errors="replace")
        raise EnvironmentError_(f"docker {args[0]} failed: {error.strip()[:400]}")
    return result


def image_id(image: str) -> str:
    return _docker("image", "inspect", "--format", "{{.Id}}", image).stdout.strip()


@contextlib.contextmanager
def _streamed(*command: str) -> Iterator[IO[bytes]]:
    """Yield a docker command's stdout as a stream, then drain and check it."""
    process = subprocess.Popen(["docker", *command], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        yield process.stdout
        while process.stdout.read(1 << 20):
            pass  # A tar reader stops at the end marker; drain padding so docker can exit.
    finally:
        process.stdout.close()
        error = process.stderr.read().decode(errors="replace")
        process.stderr.close()
        if process.wait() != 0:
            raise EnvironmentError_(f"docker {command[0]} failed: {error.strip()[:400]}")


@contextlib.contextmanager
def _container(image: str) -> Iterator[str]:
    container = _docker("create", image, "/fusionrl-never-run").stdout.strip()
    try:
        yield container
    finally:
        _docker("rm", "-f", container, check=False)


def audit_image(image: str, *, workspace: str, test_patch: str = "", reference_patch: str = "") -> AuditReport:
    """Audit an image's filesystem without running it (streamed; no full copy on disk)."""
    with _container(image) as container, _streamed("export", container) as stream:
        report = audit_snapshot(tar_stream=stream, workspace=workspace, subject=image_id(image),
                                test_patch=test_patch, reference_patch=reference_patch)
    config = json.loads(_docker("image", "inspect", "--format", "{{json .Config}}", image).stdout) or {}
    proxies = sorted({item.partition("=")[0] for item in config.get("Env") or []
                      if item.partition("=")[0].lower() in {"http_proxy", "https_proxy", "all_proxy", "ftp_proxy"}})
    if proxies:
        report = AuditReport(report.subject, report.workspace, report.head_commit, report.findings + (
            EnvFinding("proxy_configured", "leak", "the image configures an outbound proxy for the policy",
                       tuple(proxies)),), report.stats)
    return report


def _import_changes(image: str) -> list[str]:
    config = json.loads(_docker("image", "inspect", "--format", "{{json .Config}}", image).stdout) or {}
    changes = []
    for item in config.get("Env") or []:
        key, _, value = item.partition("=")
        changes.append(f"ENV {key}={json.dumps(value)}")
    if config.get("WorkingDir"):
        changes.append(f"WORKDIR {config['WorkingDir']}")
    if config.get("User"):
        changes.append(f"USER {config['User']}")
    if config.get("Entrypoint"):
        changes.append("ENTRYPOINT " + json.dumps(config["Entrypoint"]))
    if config.get("Cmd"):
        changes.append("CMD " + json.dumps(config["Cmd"]))
    return changes


def harden_image(image: str, tag: str, *, workspace: str, normalize_paths: Sequence[str] = ()) -> tuple[str, HardeningResult]:
    """Build a single-layer hardened copy of ``image``; returns its image ID.

    Streams ``docker export`` through the rewriter into ``docker import``; only
    the workspace's ``.git`` is staged on disk.
    """
    workspace = "/" + workspace.strip("/")
    args = ["import"]
    for change in _import_changes(image) + [f"LABEL org.fusionrl.hardened={AUDIT_VERSION}",
                                             f"LABEL org.fusionrl.source-image={image_id(image)}"]:
        args += ["--change", change]
    with tempfile.TemporaryDirectory(prefix="fusionrl-harden-") as tmp, _container(image) as container:
        raw_git = Path(tmp) / "raw" / ".git"
        with _streamed("cp", f"{container}:{workspace}/.git", "-") as stream:
            with tarfile.open(fileobj=stream, mode="r|*") as archive:
                _extract_git_members(archive, raw_git, None)
        importer = subprocess.Popen(["docker", *args, "-", tag], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE)
        try:
            with _streamed("export", container) as source:
                result = harden_snapshot_tar(source, importer.stdin, workspace=workspace,
                                             normalize_paths=normalize_paths, git_dir=raw_git)
        finally:
            importer.stdin.close()
            _, error = importer.communicate()
        if importer.returncode != 0:
            raise EnvironmentError_(f"docker import failed: {error.decode(errors='replace').strip()[:400]}")
    return image_id(tag), result


# host:port pairs. Raw IPs catch networks that block DNS but not egress.
DEFAULT_PROBE_TARGETS = ("github.com:443", "api.github.com:443", "codeload.github.com:443", "gitlab.com:443",
                         "pypi.org:443", "files.pythonhosted.org:443", "registry.npmjs.org:443",
                         "proxy.golang.org:443", "example.com:80", "1.1.1.1:443", "8.8.8.8:53")


def probe_network(probe_image: str, network_args: Sequence[str],
                  targets: Iterable[str] = DEFAULT_PROBE_TARGETS, timeout: int = 4) -> dict[str, bool]:
    """Return {host:port: reachable} from a probe container using the runner's network flags.

    Uses a raw TCP connect (``nc``), not an HTTP client: busybox ``wget`` fails
    on HTTPS even when the host is reachable, which would report a false pass.
    ``probe_image`` needs ``sh`` and ``nc`` (busybox is enough). Every target
    must be unreachable for an attestation to pass.
    """
    result = {}
    for target in targets:
        host, _, port = target.rpartition(":")
        if not host or not port.isdigit():
            raise ValueError(f"probe target must be host:port, not {target!r}")
        script = (f"nc -w {int(timeout)} {shlex.quote(host)} {int(port)} </dev/null >/dev/null 2>&1 "
                  "&& echo REACHABLE || echo BLOCKED")
        run = _docker("run", "--rm", *network_args, probe_image, "sh", "-c", script, check=False,
                      timeout=timeout * 4 + 30)
        if "REACHABLE" not in run.stdout and "BLOCKED" not in run.stdout:
            raise EnvironmentError_(f"network probe could not run: {run.stderr.strip()[:300]}")
        result[target] = "REACHABLE" in run.stdout
    return result
