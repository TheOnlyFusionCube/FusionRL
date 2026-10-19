# FusionRL

**One execution-level RL learner, with compatible data, grading and scheduling contracts.**

FusionRL combines selected ideas from the six requested sources. It is a runnable research reference with a source-pinned integration for XiaomiMiMo/verl. It now also trains real Hugging Face causal LMs (`fusionrl train`, single GPU or `torchrun` data parallel; see [docs/real_models.md](docs/real_models.md)). It also defends against the environment leaks vals.ai found in MiMo RL tasks, with audited and hardened images, an attestation gate, a trajectory monitor and explicit rules; see [docs/integrity.md](docs/integrity.md). Its CPU demo trainer performs actual autoregressive policy sampling, reward computation, gradient updates and checkpoint export; it does not simulate or claim GPU throughput.

Start with [the design and compatibility decisions](docs/design.md), then [the algorithm](docs/algorithm.md). The exact source references and revisions are in [SOURCES.md](SOURCES.md) and [sources.json](sources.json). The source reports and datasets are not interchangeable implementations of one algorithm; FusionRL explicitly changes their boundaries before combining them.

## Quick start

Use Python 3.10 or newer in a fresh environment. The CPU validation used Python 3.13, NumPy 2.5.4 and PyTorch 2.6.0+cpu. Recorded run values reproduce exactly only under that PyTorch build.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.6.0
python -m pip install -e .
python -m fusionrl doctor
python -m fusionrl validate-config configs/strict_sync.json
python -m fusionrl train-demo --config configs/strict_sync.json --output runs/first
python -m fusionrl sweep --config configs/strict_sync.json --seeds 1 2 3 4 5 --output runs/first_sweep
python -m unittest discover -s tests -v
```

Choose a new output directory for each run. The trainer preserves existing evidence and refuses to overwrite a completed run. It writes `summary.json`, per-step `metrics.jsonl`, the resolved configuration, a complete example rollout, and `checkpoint.pt`. Load the checkpoint with `torch.load(path, weights_only=True)` into `TinyCausalPolicy`. It is a final model export, not a resumable distributed queue checkpoint.

A single seed is a noisy summary: final scores of one profile can differ by more than 20 points between seeds. Use `sweep` to compare profiles; it writes each seed's run plus a `sweep.json` with mean, standard deviation, minimum and maximum per metric.

For a very short installation check, use `configs/smoke.json`. For asynchronous admission, use `configs/bounded_async.json`. A generated synthetic token count is a demonstration variable; it is not an LLM reasoning-token measurement.

## Real models and reward-hacking defenses

```bash
python -m pip install -e '.[real]'
fusionrl preflight --config configs/real/qwen2.5-1.5b-math-1gpu.json   # hardware, memory, parity
fusionrl train --config configs/real/qwen2.5-1.5b-math-1gpu.json --output runs/math [--resume]
fusionrl integrity audit-dataset --input code.jsonl                     # static task checks
fusionrl integrity audit-image --image IMG --workspace /testbed         # git, timestamps, leftovers
python scripts/audit_mimo_images.py --tasks code.jsonl --image-map image-mapping.jsonl \
    --attest --store attestations/ --probe-image busybox --output audit.jsonl
```

The real-model path was checked on CPU. With Qwen2.5-0.5B-Instruct, rollout and training log-probabilities agree to about 2e-6 and checkpoints reload bit-exact. With a tiny Llama in the test suite, resume reproduces an uninterrupted run exactly and two data-parallel ranks match one process. It has not been run on a GPU here. Run `preflight` on your hardware first.

In at least 6 of 9 audited MiMo code-task images, file timestamps reveal the reference fix's scope, as vals.ai reported. None of the 9 held unreachable fix commits. All 2,602 code tasks use mutable image tags and grade from a script path inside the agent workspace. Hardening removes the leak without changing task behavior. Do not train on MiMo environments that have not been attested.

## Included profiles

| Configuration | Purpose | Important contract |
|---|---|---|
| `strict_sync.json` | Synchronous binary-reward reference | Every generated token comes from the current frozen policy |
| `bounded_async.json` | Delayed-group reference | Actual behavior versions and probabilities retained; token-lag cap applied at admission |
| `gar.json` | Passing-solution quality redistribution | Binary base outcomes; no additive length controller |
| `grs_effort.json` | Rubric scores with effort-dependent cost | Fixed effort per group; quality rubric and token cost have distinct roles |
| `scalar.json` | Already-calibrated continuous rewards | Fixed calibration into [0,1]; no hidden thresholding |
| `staged.json` | Optional SFT, RL and separate OPD | Fresh student prefixes after the RL queue barrier; identical tokenizer mapping |

These are small demonstration settings, not copied production batch sizes or tuned recommendations. A production task can select a different grading route per group through `pipeline.prepare_step`; all groups in one update share an original replicate count and the same learner objective.

## What is implemented

- Immutable generation records, original causal contexts, branch identity, role/provenance masks, a deterministic path cap, and shared-target deduplication.
- Binary, GRS, GAR and explicitly calibrated scalar reward routes; hack correction before group statistics; guarded alternative length costs.
- The QwenGyre execution loss: detached token importance weights, a shared sign-aware execution clipping decision, and original-execution normalization. No nested PPO/GSPO correction is added.
- Strict synchronous and bounded asynchronous group controllers with publication boundaries, complete-group quotas, actual-token lag gates, exactly-once consumption and stage cancellation.
- A Seer-inspired chunk scheduler and elastic resource guards. These return decisions for a backend to execute.
- A real tiny causal CPU trainer with batched incremental rollout sampling, optional synthetic SFT and full-vocabulary reverse-KL distillation, numerical oracle, multi-seed sweeps, tests, and recorded validation runs.
- A MiMo parquet/JSONL adapter covering the actual five subset schemas, plus a revision-pinned download script.
- A four-file patch and checked FSDP/FSDP2 integration for the pinned MiMo verl fork's strict-sync, single-row, binary-reward path, with a fail-fast rollout/training parity guard.
- A real-model trainer for Hugging Face causal LMs, with exact behavior probabilities, per-execution reproducible sampling, memory-bounded sequence scoring, data parallelism, resumable checkpoints and a preflight check.
- Reward-hacking defenses: streamed image audits and hardening, attestation-gated admission with egress probes, explicit channel rules, a trajectory monitor with task revocation and halting, hack penalties, and grader-side patch sanitization.

## Scope of the GPU integration

The **portable core** handles multiple causal paths, rich grading routes and bounded asynchronous records. The **supplied verl adapter** deliberately supports a narrower transport: strict sync, binary verification, full-support sampling and one physical row per execution. It rejects unsupported combinations before they can silently alter the objective. See [docs/verl.md](docs/verl.md) for the exact support matrix and launch procedure.

Distributed Ray/CUDA execution, the real MiMo environments, private model graders, MoE routing replay, multimodal processors, shared RDMA KV storage, and live elastic worker membership have not been executed here. The archive includes the tested mathematical core, control-plane decisions and an audited adapter contract; deploying the richer core requires the backend integrations listed in [docs/production.md](docs/production.md).

## Data preparation

```bash
python -m pip install -e '.[data]'
python scripts/fetch_mimo.py --help
python -m fusionrl prepare-mimo --subset music --input /path/to/music.parquet --output /path/to/music.jsonl
```

The importer preserves the original public prompt separately from verifier assets and environment metadata. Music is a symbolic composition task; code, cyber, general and web development have different environment needs. Downloading or normalizing a row does not execute its commands or instantiate its harness. See [docs/data.md](docs/data.md).

## Verification

The packaged [validation report](docs/validation.md) gives the executed test count, environment, source-contract checks and synthetic run results. The raw run evidence is in `examples/validation/`. Source-dependent integration tests require an explicit pristine upstream checkout:

```bash
python -m pip install -e '.[data]' hydra-core==1.3.2
FUSIONRL_VERL_SOURCE=/absolute/path/to/pristine-pinned-verl python -m unittest discover -s tests -v
```

Without that checkout the source-dependent checks are skipped with explicit messages; parquet tests also require the data extra. The included bootstrap script applies a patch; a source-test checkout must remain pristine. Installed numerical, scheduling, grading and CPU training tests do not need a GPU or network.

## Package map

| Path | Contents |
|---|---|
| `fusionrl/pipeline.py`, `learner.py`, `objective.py` | Group grading to execution-normalized gradients |
| `fusionrl/rewards.py`, `data.py` | Reward math and source-specific task import |
| `fusionrl/trajectory.py`, `scheduler.py` | Admission, provenance, chunks, quotas and resource guards |
| `fusionrl/trainer.py`, `tiny_policy.py`, `distillation.py`, `sweep.py` | Executable CPU stages and seed sweeps |
| `fusionrl/runner.py`, `run_config.py`, `backends/hf.py`, `verifiers.py`, `preflight.py` | Real-model training |
| `fusionrl/integrity/` | Environment audit/hardening, attestation, rules, monitor, grading |
| `fusionrl/integrations/verl/`, `patches/` | Pin-aware integration and local upstream patch |
| `configs/`, `scripts/`, `.github/` | Ready demonstration profiles, data fetch, upstream bootstrap and CI |
| `docs/`, `tests/`, `examples/validation/` | Design rationale, reproducible checks and measured evidence |

License: Apache-2.0 for FusionRL code; see `LICENSE` and `NOTICE`. No model checkpoints from the source organizations, task containers, or full source reports are embedded. The tiny checkpoints in validation evidence were generated by this package.

