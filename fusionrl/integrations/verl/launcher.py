"""Launch the exact patched upstream source, or inspect it without GPUs."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

from .contracts import check_source_tree


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verl-root", required=True, type=Path)
    parser.add_argument("--check-only", action="store_true", help="verify the patched source contract; do not import GPU dependencies")
    parser.add_argument("overrides", nargs=argparse.REMAINDER, help="Hydra key=value overrides, after --")
    args = parser.parse_args(argv)
    root = args.verl_root.resolve()
    checked = check_source_tree(root, patched=True)
    if args.check_only:
        print(json.dumps({"source_contract": "passed", "files": checked}, indent=2))
        return 0
    required = ["torch", "ray", "hydra", "omegaconf", "transfer_queue", "tensordict", "torchdata"]
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    if missing:
        parser.error("install the pinned upstream GPU environment first; missing " + ", ".join(missing))
    overrides = args.overrides
    if overrides[:1] == ["--"]:
        overrides = overrides[1:]
    package_root = Path(__file__).resolve().parents[3]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(root), str(package_root), env.get("PYTHONPATH", "")])
    config_dir = Path(__file__).with_name("config")
    command = [
        sys.executable, "-m", "verl.trainer.main_ppo",
        "--config-path", str(config_dir), "--config-name", "fusionrl_binary",
        *overrides,
    ]
    # No shell evaluation; Hydra receives each literal argument separately.
    return subprocess.run(command, cwd=root, env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())

