"""``fusionrl integrity ...`` commands."""
from __future__ import annotations

import json
import shlex
from pathlib import Path


def register(commands) -> None:
    parser = commands.add_parser("integrity", help="Audit, harden and attest task environments against solution leaks")
    sub = parser.add_subparsers(dest="integrity_command", required=True)

    audit = sub.add_parser("audit-image", help="Audit an image's filesystem without running it")
    audit.add_argument("--image", required=True)
    audit.add_argument("--workspace", required=True)
    _task_args(audit)

    harden = sub.add_parser("harden-image", help="Write a single-layer, HEAD-only, timestamp-normalized copy")
    harden.add_argument("--image", required=True)
    harden.add_argument("--tag", required=True)
    harden.add_argument("--workspace", required=True)

    attest = sub.add_parser("attest", help="Audit a hardened image, probe egress and store an attestation")
    attest.add_argument("--image", required=True, help="hardened image that the runner will execute")
    attest.add_argument("--source-image", required=True)
    attest.add_argument("--workspace", required=True)
    attest.add_argument("--store", type=Path, required=True)
    attest.add_argument("--network-args", default="--network none",
                        help="exact docker network flags the runner uses (default: --network none)")
    attest.add_argument("--probe-image", required=True, help="image with sh and wget, e.g. busybox")
    _task_args(attest, required=True)

    dataset = sub.add_parser("audit-dataset", help="Static checks over canonical task JSONL")
    dataset.add_argument("--input", type=Path, required=True)


def _task_args(parser, required: bool = False) -> None:
    parser.add_argument("--tasks", type=Path, required=required, help="canonical task JSONL (for hidden tests)")
    parser.add_argument("--task-id", required=required)


def _task(args):
    if not args.tasks:
        return None
    from ..data import load_jsonl
    for record in load_jsonl(args.tasks):
        if record.task_id == args.task_id:
            return record
    raise ValueError(f"task {args.task_id!r} not found in {args.tasks}")


def run(args) -> None:
    from .environment import audit_image, harden_image, probe_network
    command = args.integrity_command
    if command == "audit-image":
        record = _task(args)
        report = audit_image(args.image, workspace=args.workspace,
                             test_patch=(record.private_instance.get("test_patch") or "") if record else "")
        print(json.dumps(report.to_dict(), indent=2))
    elif command == "harden-image":
        image, result = harden_image(args.image, args.tag, workspace=args.workspace)
        print(json.dumps({"image_id": image, "head_commit": result.head_commit, "epoch": result.epoch,
                          "removed": list(result.removed_paths)}, indent=2))
    elif command == "attest":
        from .attestation import AttestationStore, EnvironmentAttestation
        record = _task(args)
        report = audit_image(args.image, workspace=args.workspace,
                             test_patch=record.private_instance.get("test_patch") or "")
        network_args = shlex.split(args.network_args)
        probe = probe_network(args.probe_image, network_args)
        attestation = EnvironmentAttestation.from_report(record.task_id, report, source_image=args.source_image,
                                                         network_args=network_args, network_probe=probe)
        path = AttestationStore(args.store).save(attestation, report)
        print(json.dumps({"passed": attestation.passed, "attestation": str(path),
                          "leaks": [f.check for f in report.findings if f.severity == "leak"],
                          "reachable": list(attestation.network_reachable)}, indent=2))
        if not attestation.passed:
            raise SystemExit(1)
    elif command == "audit-dataset":
        from ..data import load_jsonl
        from .dataset import summarize_dataset
        print(json.dumps(summarize_dataset(load_jsonl(args.input)), indent=2))
    else:  # pragma: no cover - argparse enforces the choices.
        raise ValueError(command)

