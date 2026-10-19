#!/usr/bin/env python3
"""Build a deterministic source ZIP with an internal SHA-256 manifest."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]
IGNORED={"__pycache__",".git",".venv",".pytest_cache","build","dist","runs","external"}
RELEASE_DIRS={"fusionrl","configs","docs","examples","patches","scripts","tests"}
RELEASE_FILES={".gitignore","LICENSE","NOTICE","README.md","SOURCES.md","sources.json",
               "pyproject.toml","requirements-cpu.txt"}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    output=args.output.resolve()
    if output==ROOT or ROOT in output.parents:
        parser.error("Place the release ZIP outside the source folder")
    files=[]
    for path in sorted(ROOT.rglob("*")):
        relative=path.relative_to(ROOT)
        if relative.parts[0] not in RELEASE_DIRS and relative.as_posix() not in RELEASE_FILES:
            continue
        if not path.is_file() or path.is_symlink():
            continue
        if any(part in IGNORED or part.endswith(".egg-info") for part in relative.parts):
            continue
        if path.suffix in {".pyc",".pyo",".zip"} or path.name=="MANIFEST.sha256":
            continue
        files.append(path)
    manifest=ROOT/"MANIFEST.sha256"
    manifest.write_text("".join(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(ROOT).as_posix()}\n" for p in files))
    files.append(manifest)
    output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(output,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        for path in sorted(files):
            info=zipfile.ZipInfo("FusionRL/"+path.relative_to(ROOT).as_posix(),date_time=(2026,10,19,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED
            info.external_attr=(0o100755 if path.suffix==".sh" else 0o100644)<<16
            archive.writestr(info,path.read_bytes(),compress_type=zipfile.ZIP_DEFLATED,compresslevel=9)
    with zipfile.ZipFile(output) as archive:
        bad=archive.testzip()
        if bad:
            raise RuntimeError(f"Archive CRC verification failed: {bad}")
    print(f"files={len(files)} bytes={output.stat().st_size} sha256={hashlib.sha256(output.read_bytes()).hexdigest()}")


if __name__=="__main__":
    main()

