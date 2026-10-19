#!/usr/bin/env python3
"""Audit (and optionally harden and attest) MiMo task images in bulk.

Pulls each task's image, audits it without running it, and writes one JSON
line per task. With --harden it builds a hardened single-layer copy, and with
--attest it re-audits that copy, probes egress under the runner's network
flags and stores an attestation that ``IntegrityGate`` will accept.

Example:
  python scripts/audit_mimo_images.py --tasks code.jsonl --image-map image-mapping.jsonl \\
      --registry-mirror mirror.gcr.io --sample 20 --output audit.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import shlex
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fusionrl.data import load_jsonl, read_image_mapping, resolve_image_alias  # noqa: E402
from fusionrl.integrity.attestation import AttestationStore, EnvironmentAttestation  # noqa: E402
from fusionrl.integrity.environment import audit_image, harden_image, probe_network  # noqa: E402


def _pull(candidates: list[str]) -> str:
    """Pull the first candidate that succeeds, retrying transient registry errors."""
    errors = []
    for image in candidates:
        for attempt in range(3):
            result = subprocess.run(["docker", "pull", "-q", image], capture_output=True, text=True, check=False)
            if result.returncode == 0:
                return image
            errors.append(f"{image}: {result.stderr.strip()[:200]}")
            time.sleep(20 * (attempt + 1))
    raise RuntimeError("; ".join(errors[-2:]))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks", type=Path, required=True, help="canonical JSONL from `fusionrl prepare-mimo`")
    parser.add_argument("--image-map", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--registry-mirror", default="", help="e.g. mirror.gcr.io (replaces docker.io)")
    parser.add_argument("--sample", type=int, default=0, help="audit a seeded random sample of this size")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--task-ids", nargs="+", help="audit only these canonical task IDs")
    parser.add_argument("--harden", action="store_true")
    parser.add_argument("--attest", action="store_true", help="implies --harden")
    parser.add_argument("--store", type=Path, help="attestation directory (required with --attest)")
    parser.add_argument("--probe-image", help="image with sh and nc (required with --attest)")
    parser.add_argument("--network-args", default="--network none")
    parser.add_argument("--hardened-prefix", default="fusionrl-hardened")
    parser.add_argument("--keep-source", action="store_true", help="do not remove pulled source images")
    args = parser.parse_args(argv)
    if args.attest and not (args.store and args.probe_image):
        parser.error("--attest needs --store and --probe-image")
    if args.output.exists():
        parser.error("refusing to overwrite an existing audit log")
    mapping = read_image_mapping(args.image_map)
    records = [r for r in load_jsonl(args.tasks) if r.subset != "music"]
    if args.task_ids:
        wanted = set(args.task_ids)
        records = [r for r in records if r.task_id in wanted]
        if len(records) != len(wanted):
            parser.error("some --task-ids are not in --tasks")
    if args.sample:
        records = random.Random(args.seed).sample(records, min(args.sample, len(records)))
    network_args = shlex.split(args.network_args)
    store = AttestationStore(args.store) if args.store else None
    summary = {"tasks": 0, "raw_failed": 0, "errors": 0, "attested": 0, "by_check": {}}
    with args.output.open("w") as log:
        for record in records:
            original = resolve_image_alias(record, mapping)
            candidates = [original]
            if args.registry_mirror and original.startswith("docker.io/"):
                candidates.insert(0, args.registry_mirror.rstrip("/") + "/" + original[len("docker.io/"):])
            source = candidates[0]
            workspace = record.private_instance.get("cwd") or "/testbed"
            test_patch = record.private_instance.get("test_patch") or ""
            row = {"task_id": record.task_id, "image": source, "workspace": workspace}
            try:
                source = row["image"] = _pull(candidates)
                raw = audit_image(source, workspace=workspace, test_patch=test_patch)
                row["raw"] = raw.to_dict()
                summary["raw_failed"] += int(not raw.passed)
                for finding in raw.findings:
                    if finding.severity == "leak":
                        summary["by_check"][finding.check] = summary["by_check"].get(finding.check, 0) + 1
                if args.harden or args.attest:
                    tag = f"{args.hardened_prefix}/{record.task_id.replace(':', '-')}:hardened"
                    hardened_id, _ = harden_image(source, tag, workspace=workspace)
                    row["hardened_image_id"] = hardened_id
                    if args.attest:
                        report = audit_image(tag, workspace=workspace, test_patch=test_patch)
                        probe = probe_network(args.probe_image, network_args)
                        attestation = EnvironmentAttestation.from_report(
                            record.task_id, report, source_image=source, network_args=network_args,
                            network_probe=probe)
                        store.save(attestation, report)
                        row["attested"] = attestation.passed
                        summary["attested"] += int(attestation.passed)
            except Exception as error:  # noqa: BLE001 - record and continue the batch.
                row["error"] = str(error)[:500]
                summary["errors"] += 1
            finally:
                if not args.keep_source:
                    subprocess.run(["docker", "rmi", "-f", source], capture_output=True, check=False)
            summary["tasks"] += 1
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(json.dumps({"task": record.task_id, "raw_passed": row.get("raw", {}).get("passed"),
                              "leaks": [f["check"] for f in row.get("raw", {}).get("findings", [])
                                        if f["severity"] == "leak"], "error": row.get("error")}), flush=True)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
