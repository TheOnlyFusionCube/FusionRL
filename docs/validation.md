# Executed validation

FusionRL 0.1.0 was validated on October 19, 2026. The final suite ran **121 tests: 121 passed, zero failures, zero errors and zero skips** (with the pinned verl source, a Docker daemon and a probe image supplied). Six fresh demonstration runs then exercised the final implementation, and each profile was repeated over eight seeds. The raw records, resolved run configurations, per-step metrics, example rollouts and tiny model exports are included in [examples/validation](../examples/validation/).

These results establish the behavior of the mathematical core, data import, control-plane contracts and CPU training path. They do not establish real-model quality, production throughput or distributed GPU correctness.

## Environment

| Component | Executed version |
|---|---|
| Python | 3.13.16 |
| NumPy | 2.5.4 |
| PyTorch | 2.6.0+cpu |
| Hydra | 1.3.2 |
| Transformers | 5.20.2 |
| Docker Engine | 29.9.1 |
| PyArrow | 26.0.0 |
| Trainable demonstration policy | 7,246-parameter autoregressive GRU |
| Demonstration task | `synthetic-symbol-copy-v1` |
| Demonstration seed | 7 |

The project was also installed as an editable Python package and its installed `fusionrl doctor` entry point executed successfully from outside the project directory. A GPU build of the pinned upstream backend has a separate dependency contract; see [verl.md](verl.md).

## Test coverage

The machine-readable report is [tests.json](../examples/validation/tests.json), and the individual test names and outcomes are in [tests.txt](../examples/validation/tests.txt).

| Test module | Passed | Principal evidence |
|---|---:|---|
| Configuration | 4 | Schema validation, JSON field types, and rejection of incompatible routes or scheduling settings |
| Data | 10 | All five MiMo subset schemas, separation of prompt/private metadata, actual parquet streaming and CLI record counts |
| Integrity | 15 | Monitor catches each exploit vals.ai reported and ignores ordinary commands; fail-closed group decisions; task revocation and halting; rules; grader patch sanitization; hack penalty; leaky-repository audit, hardening (fix commit unrecoverable, uniform timestamps), hostile git config never executed; Docker image round trip; `--network none` probe |
| Distillation | 2 | Full-vocabulary reverse-KL values, gradients and tokenizer-contract enforcement |
| Objective | 10 | Independent numerical oracle, sign-aware execution clipping, detached importance weights, one-sided loss cap and original-execution denominator |
| Pipeline | 3 | Supervision identity binding and integration of alternative grading routes with admission and learning |
| Rewards | 14 | Binary, scalar, GRS and GAR algebra; hack correction; guarded alternative length costs |
| Scheduler | 17 | Complete groups, fixed quotas, publication boundaries, actual-token lag, quarantine and in-place retry of version-mismatched groups, exactly-once consumption, cancellation and chunk/resource guards |
| Real model | 10 | Tiny Llama/GPT-2: recorded vs independent log-probs, selective vs full head, Gumbel sampler vs softmax, dropout disabled for scoring, tokenizer fingerprint, sequence vs context learner paths, resume equals uninterrupted run, 2 DDP ranks equal 1 process, parity and integrity aborts, evaluator-failure retry |
| Training | 5 | Real causal-policy updates and checkpoint evidence, batched sampling probabilities, delayed-group behavior, stage integration and seed sweeps |
| Trajectory | 16 | Original contexts, shared-target deduplication, provenance and stale masks, invalid-group quarantine, retained original slots |
| verl contract | 15 | Core/adapter value and gradient equivalence, microbatch and DP reduction, rollout/training parity guard, runtime restrictions, real source fingerprints/AST, patch application and Hydra composition |
| **Total** | **121** | **Zero skipped** |

The full source checks used a pristine checkout of `XiaomiMiMo/verl` at `79090e58195e80a9cfe251c95ae7a78a2132b6e2`. Patch application occurred in a disposable copy. Hydra composition resolved the actual pinned upstream YAML tree. These checks substantiate a specific interface and reduction contract; they do not execute Ray workers, FSDP collectives or GPU kernels.

The parquet checks create synthetic fixtures with the five observed schemas. They exercise PyArrow, row-group streaming, large-string fields, canonical JSONL output and the CLI's consumed-iterator count. They do not execute downloaded task environments. The public source schemas and revisions were separately inspected, as recorded in [SOURCES.md](../SOURCES.md).

## Final demonstration results

Every profile starts from the same seed-7 initialization. Before training, the mean probability assigned to the correct first output token is **7.0664%**, and greedy completion accuracy is **9.375%** on the demo's fixed 32-prompt evaluation set. These prompts use the same synthetic symbol-copy task family as training; this is a functional check, not a held-out benchmark.

The first five profiles each perform 100 consumed RL updates. The staged profile performs 5 SFT steps, 40 consumed RL updates and 5 OPD steps. An update with no eligible signal or a completely zero gradient is consumed and recorded without an optimizer step, so an RL update count is not a claim that every update changed the weights.

| Profile | Final correct-first-token probability | Final greedy completion accuracy | Maximum admitted token lag |
|---|---:|---:|---:|
| Strict synchronous binary | 89.3037% | 96.8750% | 0 |
| Bounded asynchronous binary | 80.7276% | 100.0000% | 1 |
| GAR | 74.6152% | 75.0000% | 0 |
| GRS with effort cost | 98.4712% | 100.0000% | 0 |
| Calibrated scalar | 85.5694% | 87.5000% | 0 |
| SFT → RL → OPD | 77.9908% | 100.0000% | 0 |

### Seed spread

A single seed is not a reliable summary of a profile. Seed 7 is the worst of eight seeds for GAR, for example, while an earlier implementation's seed-7 GAR run reached 98%. `fusionrl sweep` repeats a profile and reports the spread; the table below uses seeds 1–8 (mean ± sample standard deviation, with the minimum). The raw reports are in `examples/validation/sweeps/`.

| Profile | Correct-first-token probability | Greedy completion accuracy |
|---|---:|---:|
| Strict synchronous binary | 94.4 ± 4.7% (min 86.3%) | 98.0 ± 4.4% (min 87.5%) |
| Bounded asynchronous binary | 83.0 ± 2.9% (min 78.6%) | 100.0 ± 0.0% (min 100.0%) |
| GAR | 93.9 ± 8.2% (min 74.6%) | 96.9 ± 8.8% (min 75.0%) |
| GRS with effort cost | 96.8 ± 4.5% (min 85.7%) | 98.4 ± 4.4% (min 87.5%) |
| Calibrated scalar | 91.7 ± 5.9% (min 85.6%) | 94.5 ± 6.0% (min 87.5%) |
| SFT → RL → OPD | 81.6 ± 6.6% (min 67.8%) | 98.0 ± 4.4% (min 87.5%) |

Batched rollout changed the order of random draws, not the sampling law. Under the previous one-request-at-a-time sampler the same eight-seed sweep gave 95.7 ± 4.0% (strict synchronous) and 93.2 ± 10.2% (GAR) correct-first-token probability, consistent with the table above. Exact run values also depend on the PyTorch build: the recorded runs reproduce bit-for-bit under 2.6.0+cpu but not under later releases.

The bounded-asynchronous run is a deterministic delayed-queue simulation. It records the old policy's actual sampling probabilities and versions, exercises admission of delayed groups, and cancels four pending groups at the final stage boundary. Its configured lag ceiling is two versions; the observed admitted maximum is one. It does not run rollout and optimization concurrently on separate hardware.

The staged run's correct-first-token probability progresses from 7.0664% initially to 14.8912% after SFT, 73.4671% after RL and 77.9908% after OPD. Its teacher is a transparent synthetic oracle with an identical tokenizer mapping. The included OPD loss trace uses newly sampled student prefixes, so it need not decrease monotonically from step to step.

These profiles test different reward definitions and stage budgets. Their numbers are not controlled comparisons of the methods, and they do not justify ranking GAR, GRS, scalar grading or synchronous scheduling on real tasks. Recorded wall times are local CPU observations with concurrent verification work, not a throughput benchmark.

### Evidence files

Each profile directory under `examples/validation/` contains (the `sweeps/` directory holds one aggregate seed report per profile):

- `config.json`: the resolved configuration used for that run.
- `summary.json`: environment, seed, stage counts, evaluation measurements and observed policy lag.
- `metrics.jsonl`: per-update loss, reward, admission and optimization diagnostics.
- `rollout_example.json`: actual generated token contexts, probabilities, versions and execution identities.
- `checkpoint.pt`: this package's tiny final model export, loadable with `torch.load(path, weights_only=True)`.

The model export does not contain a resumable distributed queue or original organizations' model weights. The source reports, dataset rows and environment images are referenced and pinned separately rather than embedded in these evidence files.

## Reproduction

Install the package and CPU PyTorch as described in the [README](../README.md). The numerical and CPU training tests need no GPU or network after installation. To execute all checks, install the data and Hydra dependencies and provide a pristine source checkout:

```bash
python -m pip install -e '.[data]' hydra-core==1.3.2

FUSIONRL_VERL_SOURCE=/absolute/path/to/pristine-pinned-verl \
  python scripts/verify.py --require-source --output runs/verification

python -m fusionrl train-demo --config configs/strict_sync.json --output runs/strict_sync
python -m fusionrl train-demo --config configs/bounded_async.json --output runs/bounded_async
python -m fusionrl train-demo --config configs/gar.json --output runs/gar
python -m fusionrl train-demo --config configs/grs_effort.json --output runs/grs_effort
python -m fusionrl train-demo --config configs/scalar.json --output runs/scalar
python -m fusionrl train-demo --config configs/staged.json --output runs/staged
python -m fusionrl sweep --config configs/gar.json --seeds 1 2 3 4 5 6 7 8 --output runs/gar_sweep
```

Use fresh output directories; completed training evidence is protected against accidental overwrite. The source pin and pristine-check procedure are documented in [verl.md](verl.md). Missing optional dependencies or missing upstream source cause explicit test skips; `--require-source` treats any skip as an incomplete verification run.

The release builder creates a ZIP with fixed archive timestamps, excludes local environments and caches, and includes `MANIFEST.sha256` for every payload file. Build it with `python scripts/make_release.py --output ../FusionRL.zip`. The manifest covers the code and validation evidence; it excludes itself to avoid a recursive checksum.

## Real-model checks

Run on CPU with **Qwen2.5-0.5B-Instruct** (494M parameters, fp32), using the model's own tokenizer and chat template:

| Check | Result |
|---|---|
| `fusionrl preflight`: selective head vs full forward | max difference 0.0 |
| `fusionrl preflight`: rollout vs training log-prob gap, 154 sampled tokens | mean 2.1e-6, max 1.7e-5 |
| `fusionrl train`, 3 steps (2 groups × 4 executions, ≤48 new tokens) | gap ≤ 3.9e-5 every step; ~20–60 s/step on 4 CPU cores |
| Checkpoint reload into a fresh model | all 290 tensors bit-exact; tied embeddings preserved |

These establish the plumbing and numerics on a real architecture and tokenizer. They are not a learning result. A 300-step run on the tiny Llama arithmetic fixture raised mean reward from 0.09 to about 0.21. No GPU run was possible here.

## Integrity checks on real MiMo data

- Static audit of all 2,602 code tasks: no hidden tests leaked into prompts; 2,602 mutable image tags; 2,602 grading commands at a workspace path.
- Images (9 code tasks: `001457` plus a seeded sample of 8): in at least 6 of 9, source-file timestamps reveal the fix's scope, and in 5 of those the newer files include the file the hidden test patch edits. 0 of 9 held unreachable fix commits. 1 of 9 was clean. Details and caveats are in [integrity.md](integrity.md#what-was-found-in-the-real-dataset). Raw reports are in `examples/validation/integrity/mimo_image_audits.jsonl`: the `audit_sample` rows used the first audit version, before its false positives were fixed, and the `audit_recheck` rows used the corrected one.
- Hardened copies of `001457`, `000146` and `002520` pass the audit. `001457` behaves identically after hardening (same 24 Jest tests pass with `--network none`).

## Remaining deployment evidence

A deployment must still validate actual rollout-to-training probability equivalence, distributed value/gradient reductions, model and tokenizer compatibility, recovery from persisted queues, and the domain's real environment/verifier behavior. Shared KV transport, live elastic membership, MoE routing replay and multimodal execution require concrete backend implementations. [production.md](production.md) specifies those interfaces and the measurements needed before making quality, throughput or cost claims.
