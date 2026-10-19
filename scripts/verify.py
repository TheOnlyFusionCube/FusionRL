#!/usr/bin/env python3
"""Run meaningful offline checks and write an auditable result bundle."""
from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
import platform
import sys
import time
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


class Results(unittest.TextTestResult):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.records=[]

    def addSuccess(self,test):
        self.records.append({"test":test.id(),"status":"passed"})
        super().addSuccess(test)

    def addSkip(self,test,reason):
        self.records.append({"test":test.id(),"status":"skipped","reason":reason})
        super().addSkip(test,reason)

    def addFailure(self,test,error):
        self.records.append({"test":test.id(),"status":"failed"})
        super().addFailure(test,error)

    def addError(self,test,error):
        self.records.append({"test":test.id(),"status":"error"})
        super().addError(test,error)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--require-source",action="store_true")
    args=parser.parse_args()
    if args.require_source and not os.environ.get("FUSIONRL_VERL_SOURCE"):
        parser.error("Set FUSIONRL_VERL_SOURCE to the pristine pinned checkout")
    log=io.StringIO()
    started=time.perf_counter()
    suite=unittest.defaultTestLoader.discover(str(ROOT/"tests"))
    result=unittest.TextTestRunner(stream=log,verbosity=2,resultclass=Results).run(suite)
    source_available=bool(os.environ.get("FUSIONRL_VERL_SOURCE"))
    report={"tests_run":result.testsRun,"failures":len(result.failures),"errors":len(result.errors),
        "skipped":len(result.skipped),"passed":result.testsRun-len(result.failures)-len(result.errors)-len(result.skipped),
        "successful":result.wasSuccessful(),"source_checkout_supplied":source_available,
        "python":platform.python_version(),"elapsed_seconds":time.perf_counter()-started,"tests":result.records}
    for name in ("numpy","torch","hydra","pyarrow"):
        try:
            module=__import__(name)
            report[name+"_version"]=getattr(module,"__version__","unknown")
        except ImportError:
            report[name+"_version"]=None
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/"tests.json").write_text(json.dumps(report,indent=2)+"\n")
    (args.output/"tests.txt").write_text(log.getvalue())
    print(json.dumps({k:v for k,v in report.items() if k!="tests"},indent=2))
    if not result.wasSuccessful():
        print(log.getvalue(),file=sys.stderr)
    raise SystemExit(0 if result.wasSuccessful() and (not args.require_source or not result.skipped) else 1)


if __name__=="__main__":
    main()

