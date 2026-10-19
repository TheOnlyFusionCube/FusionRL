# Executed validation

FusionRL 0.1.0 was validated on October 19, 2026. The final suite ran **92 tests: 92 passed, zero failures, zero errors and zero skips**. Six fresh demonstration runs then exercised the final implementation. The raw records, resolved run configurations, per-step metrics, example rollouts and tiny model exports are included in [examples/validation](../examples/validation/).

These results establish the behavior of the mathematical core, data import, control-plane contracts and CPU training path. They do not establish real-model quality, production throughput or distributed GPU correctness.

## Environment

| Component | Executed version |
|---|---|
| Python | 3.12.14 |
| NumPy | 2.3.5 |
| PyTorch | 2.6.0+cpu |
| Hydra | 1.3.2 |
| PyArrow | 19.0.1 |
| Trainable demonstration policy | 7,246-parameter autoregressive GRU |
| Demonstration task | `synthetic-symbol-copy-v1` |
| Demonstration seed | 7 |

The project was also installed as an editable Python package and its installed `fusionrl doctor` entry point executed successfully from outside the project directory. A GPU build of the pinned upstream backend has a separate dependency contract; see [verl.md](verl.md).

## Test coverage

The machine-readable report is [tests.json](../examples/validation/tests.json), and the individual test names and outcomes are in [tests.txt](../examples/validation/tests.txt).

| Test module | Passed | Principal evidence |
|---|---:|---|
| Configuration | 3 | Schema validation and rejection of incompatible routes or scheduling settings |
| Data | 10 | All five MiMo subset schemas, separation of prompt/private metadata, actual parquet streaming and CLI record counts |
| Distillation | 2 | Full-vocabulary reverse-KL values, gradients and tokenizer-contract enforcement |
| Objective | 10 | Independent numerical oracle, sign-aware execution clipping, detached importance weights, one-sided loss cap and original-execution denominator |
| Pipeline | 3 | Supervision identity binding and integration of alternative grading routes with admission and learning |
| Rewards | 14 | Binary, scalar, GRS and GAR algebra; hack correction; guarded alternative length costs |
| Scheduler | 17 | Complete groups, fixed quotas, publication boundaries, actual-token lag, exactly-once consumption, cancellation and chunk/resource guards |
| Training | 3 | Real causal-policy updates and checkpoint evidence, delayed-group behavior and stage integration |
| Trajectory | 16 | Original contexts, shared-target deduplication, provenance and stale masks, invalid-group quarantine, retained original slots |
| verl contract | 14 | Core/adapter value and gradient equivalence, microbatch and DP reduction, runtime restrictions, real source fingerprints/AST, patch application and Hydra composition |
| **Total** | **92** | **Zero skipped** |

The full source checks used a pristine checkout of `XiaomiMiMo/verl` at `79090e58195e80a9cfe251c95ae7a78a2132b6e2`. Patch application occurred in a disposable copy. Hydra composition resolved the actual pinned upstream YAML tree. These checks substantiate a specific interface and reduction contract; they do not execute Ray workers, FSDP collectives or GPU kernels.

The parquet checks create synthetic fixtures with the five observed schemas. They exercise PyArrow, row-group streaming, large-string fields, canonical JSONL output and the CLI's consumed-iterator count. They do not execute downloaded task environments. The public source schemas and revisions were separately inspected, as recorded in [SOURCES.md](../SOURCES.md).

## Final demonstration results

Every profile starts from the same seed-7 initialization. Before training, the mean probability assigned to the correct first output token is **7.0664%**, and greedy completion accuracy is **9.375%** on the demo's fixed 32-prompt evaluation set. These prompts use the same synthetic symbol-copy task family as training; this is a functional check, not a held-out benchmark.

The first five profiles each perform 100 consumed RL updates. The staged profile performs 5 SFT steps, 40 consumed RL updates and 5 OPD steps. An update with no eligible signal or a completely zero gradient is consumed and recorded without an optimizer step, so an RL update count is not a claim that every update changed the weights.

| Profile | Final correct-first-token probability | Final greedy completion accuracy | Maximum admitted token lag |
|---|---:|---:|---:|
| Strict synchronous binary | 96.0873% | 100.0000% | 0 |
| Bounded asynchronous binary | 84.6315% | 100.0000% | 1 |
| GAR | 98.2716% | 100.0000% | 0 |
| GRS with effort cost | 97.6329% | 100.0000% | 0 |
| Calibrated scalar | 95.2015% | 100.0000% | 0 |
| SFT → RL → OPD | 72.4691% | 93.7500% | 0 |

The bounded-asynchronous run is a deterministic delayed-queue simulation. It records the old policy's actual sampling probabilities and versions, exercises admission of delayed groups, and cancels four pending groups at the final stage boundary. Its configured lag ceiling is two versions; the observed admitted maximum is one. It does not run rollout and optimization concurrently on separate hardware.

The staged run's correct-first-token probability progresses from 7.0664% initially to 14.8912% after SFT, 66.9298% after RL and 72.4691% after OPD. Its teacher is a transparent synthetic oracle with an identical tokenizer mapping. The included OPD loss trace uses newly sampled student prefixes, so it need not decrease monotonically from step to step.

These profiles test different reward definitions and stage budgets. Their numbers are not controlled comparisons of the methods, and they do not justify ranking GAR, GRS, scalar grading or synchronous scheduling on real tasks. Recorded wall times are local CPU observations with concurrent verification work, not a throughput benchmark.

### Evidence files

Each profile directory under `examples/validation/` contains:

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
```

Use fresh output directories; completed training evidence is protected against accidental overwrite. The source pin and pristine-check procedure are documented in [verl.md](verl.md). Missing optional dependencies or missing upstream source cause explicit test skips; `--require-source` treats any skip as an incomplete verification run.

The release builder creates a ZIP with fixed archive timestamps, excludes local environments and caches, and includes `MANIFEST.sha256` for every payload file. Build it with `python scripts/make_release.py --output ../FusionRL.zip`. The manifest covers the code and validation evidence; it excludes itself to avoid a recursive checksum.

## Remaining deployment evidence

A deployment must still validate actual rollout-to-training probability equivalence, distributed value/gradient reductions, model and tokenizer compatibility, recovery from persisted queues, and the domain's real environment/verifier behavior. Shared KV transport, live elastic membership, MoE routing replay and multimodal execution require concrete backend implementations. [production.md](production.md) specifies those interfaces and the measurements needed before making quality, throughput or cost claims.
