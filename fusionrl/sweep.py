"""Repeat one demo profile over several seeds and report the spread.

A single seed of the synthetic demo can move by several points when only the
random stream changes. Comparisons between profiles should use this report,
not one run. It is still the same synthetic task, not a held-out benchmark.
"""
from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path
from typing import Any, Sequence

from .config import FusionConfig

STAGES = ("initial", "after_sft", "after_rl", "final")


def _describe(values: Sequence[float]) -> dict[str, float]:
    count = len(values)
    mean = sum(values) / count
    # Sample standard deviation; a sweep always has at least two seeds.
    std = math.sqrt(sum((x - mean) ** 2 for x in values) / (count - 1))
    return {"mean": mean, "std": std, "min": min(values), "max": max(values)}


def run_sweep(config: FusionConfig, seeds: Sequence[int], output: str | Path,
              *, progress: bool = True) -> dict[str, Any]:
    from .trainer import run_demo

    seeds = list(seeds)
    if len(seeds) < 2:
        raise ValueError("A sweep needs at least two seeds")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Sweep seeds must be distinct")
    if any(isinstance(s, bool) or not isinstance(s, int) or s < 0 for s in seeds):
        raise ValueError("Sweep seeds must be nonnegative integers")
    output = Path(output)
    if (output / "sweep.json").exists():
        raise FileExistsError("Choose a fresh output directory; an existing sweep is not overwritten")
    runs = []
    for seed in seeds:
        seeded = replace(config, train=replace(config.train, seed=seed)).validate()
        if progress:
            print(f"seed {seed}", flush=True)
        summary = run_demo(seeded, output / f"seed-{seed}", progress=False)
        runs.append({"seed": seed, **{stage: summary[stage] for stage in STAGES},
                     "observed_max_admitted_token_lag": summary["observed_max_admitted_token_lag"]})
    aggregate = {
        stage: {metric: _describe([run[stage][metric] for run in runs]) for metric in runs[0][stage]}
        for stage in STAGES
    }
    report = {
        "evidence_scope": "Repeated seeds of the synthetic CPU demo; not a held-out or real-task benchmark",
        "mode": config.schedule.mode, "route": config.reward.route, "length_mode": config.reward.length_mode,
        "seeds": seeds, "aggregate": aggregate, "runs": runs,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "sweep.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report
