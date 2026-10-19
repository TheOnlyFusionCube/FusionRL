#!/usr/bin/env python3
"""Fetch pinned MiMo inputs explicitly; never run them or pull Docker images."""

from __future__ import annotations

import argparse
from pathlib import Path

from fusionrl.data import MIMO_DATASET, MIMO_FILES, MIMO_REVISION, load_parquet, save_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--subsets", nargs="+", choices=sorted(MIMO_FILES), default=["code"])
    parser.add_argument("--include-general-environments", action="store_true",
                        help="Also fetch general/envs/** (the full source dataset is about 12.1 GB)")
    parser.add_argument("--export-jsonl", action="store_true",
                        help="Convert each selected parquet to a private canonical task envelope")
    args = parser.parse_args()
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError as exc:
        raise SystemExit("Install huggingface_hub, and pyarrow when exporting JSONL") from exc
    args.output.mkdir(parents=True, exist_ok=True)
    files = ["README.md", "image-mapping.jsonl"] + [MIMO_FILES[s] for s in dict.fromkeys(args.subsets)]
    for filename in files:
        local = hf_hub_download(repo_id=MIMO_DATASET, repo_type="dataset", revision=MIMO_REVISION,
                                filename=filename, local_dir=args.output)
        print(f"Fetched pinned input: {local}")
    if args.include_general_environments:
        snapshot_download(repo_id=MIMO_DATASET, repo_type="dataset", revision=MIMO_REVISION,
                          allow_patterns=["general/envs/**"], local_dir=args.output)
    if args.export_jsonl:
        for subset in dict.fromkeys(args.subsets):
            target = args.output / "canonical" / f"{subset}.jsonl"
            count = save_jsonl(load_parquet(args.output / MIMO_FILES[subset], subset), target)
            print(f"Exported {count} trusted {subset} task envelopes to {target}")


if __name__ == "__main__":
    main()
