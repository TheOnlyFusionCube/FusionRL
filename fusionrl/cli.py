"""Small explicit command line; expensive downloads and runs are opt-in."""
from __future__ import annotations

import argparse
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys

from . import __version__
from .config import FusionConfig, load_config


def main(argv: list[str] | None = None) -> None:
    parser=argparse.ArgumentParser(prog="fusionrl",description="FusionRL reference training and compatibility checks")
    parser.add_argument("--version",action="version",version=__version__)
    commands=parser.add_subparsers(dest="command",required=True)
    validate=commands.add_parser("validate-config",help="Reject incompatible option combinations")
    validate.add_argument("config",type=Path)
    train=commands.add_parser("train-demo",help="Train an actual tiny causal CPU policy in a synthetic environment")
    train.add_argument("--config",type=Path)
    train.add_argument("--output",type=Path,required=True)
    train.add_argument("--steps",type=int)
    train.add_argument("--quiet",action="store_true")
    commands.add_parser("doctor",help="Inspect dependency availability; never downloads or starts services")
    data=commands.add_parser("prepare-mimo",help="Normalize an already downloaded MiMo parquet file to canonical JSONL")
    data.add_argument("--subset",required=True,choices=["code","general","webdev","music","cyber"])
    data.add_argument("--input",type=Path,required=True)
    data.add_argument("--output",type=Path,required=True)
    args=parser.parse_args(argv)
    try:
        if args.command=="validate-config":
            config=load_config(args.config)
            print(json.dumps({"valid":True,"config":config.to_dict()},indent=2))
        elif args.command=="doctor":
            report={"fusionrl_version":__version__,"python":sys.version.split()[0],
                "dependencies":{x:importlib.util.find_spec(x) is not None for x in ["numpy","torch","pyarrow","huggingface_hub","verl"]},
                "notes":["Torch is needed for policy training; NumPy-only components can run without it.",
                         "A detected verl package is not an integration pass; check the exact pinned source contract."]}
            print(json.dumps(report,indent=2))
        elif args.command=="train-demo":
            from .trainer import run_demo
            config=load_config(args.config) if args.config else FusionConfig().validate()
            if args.steps is not None:
                config=replace(config,train=replace(config.train,steps=args.steps)).validate()
            print(json.dumps(run_demo(config,args.output,progress=not args.quiet),indent=2))
        elif args.command=="prepare-mimo":
            from .data import load_parquet,save_jsonl
            if args.output.exists():
                raise FileExistsError("Refusing to overwrite an existing canonical dataset")
            records=load_parquet(args.input,args.subset)
            count=save_jsonl(records,args.output)
            print(json.dumps({"output":str(args.output),"subset":args.subset,"tasks":count}))
    except (ValueError,TypeError,FileNotFoundError,FileExistsError,ImportError,RuntimeError) as error:
        parser.exit(2,f"FusionRL: {error}\n")


if __name__=="__main__":
    main()

