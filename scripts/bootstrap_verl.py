#!/usr/bin/env python3
"""Fetch the exact public upstream revision and apply the reviewed local patch.

This does not install large GPU packages, pull task images, or run training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from fusionrl.integrations.verl import UPSTREAM_COMMIT, UPSTREAM_URL
from fusionrl.integrations.verl.contracts import VerlContractError, check_source_tree


def run(*args, cwd=None, capture=False):
    return subprocess.run(args, cwd=cwd, text=True, capture_output=capture, check=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--submodules", action="store_true", help="also fetch the two pinned MiMo harness submodules (not needed for the strict single-turn profile)")
    args = parser.parse_args(argv)
    destination = args.destination.resolve()
    patch = PROJECT_ROOT / "patches" / "verl-fusionrl-strict-sync.patch"
    if not destination.exists():
        if args.check_only:
            parser.error("--check-only needs an existing checkout")
        destination.mkdir(parents=True)
        run("git", "init", str(destination))
        run("git", "remote", "add", "origin", UPSTREAM_URL, cwd=destination)
        run("git", "fetch", "--depth", "1", "origin", UPSTREAM_COMMIT, cwd=destination)
        run("git", "checkout", "--detach", "FETCH_HEAD", cwd=destination)
    if not (destination / ".git").exists():
        parser.error("destination already exists but is not a git checkout; choose a fresh path")
    try:
        checked = check_source_tree(destination, patched=True)
        patch_status = "already applied"
    except VerlContractError:
        checked = check_source_tree(destination, patched=False)
        if args.check_only:
            print(json.dumps({"source_contract": "passed", "patch": "not applied", "files": checked}, indent=2))
            return 0
        run("git", "apply", "--check", str(patch), cwd=destination)
        run("git", "apply", str(patch), cwd=destination)
        checked = check_source_tree(destination, patched=True)
        patch_status = "applied"
    if args.submodules and not args.check_only:
        run("git", "submodule", "update", "--init", "third_party/mimoagent-osr", "third_party/uni_agent", cwd=destination)
    print(json.dumps({"commit": UPSTREAM_COMMIT, "patch": patch_status, "files": checked}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

