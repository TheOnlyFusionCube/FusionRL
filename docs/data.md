# Data contracts and the MiMo adapter

FusionRL stores a **task definition**, then collects fresh on-policy or
staleness-bounded trajectories from its environment. The MiMo release supplies
tasks, environment assets and verifier metadata. It does not supply the sampled
token probabilities, routing records, candidate sets or current-policy
trajectories needed by the FusionRL learner. Importing the dataset therefore
does not by itself launch a complete RL experiment.

## Verified source snapshot

The source is [XiaomiMiMo/MiMo-V2.6-RL-oss](https://huggingface.co/datasets/XiaomiMiMo/MiMo-V2.6-RL-oss),
pinned to `971cab2ab421fd88db29770db0d0e52014b63615` (source update October 18,
2026; inspected October 19, 2026). Its card declares Apache-2.0. Counts below
come from the live Hugging Face dataset rows API and the configuration paths
from the [pinned card](https://huggingface.co/datasets/XiaomiMiMo/MiMo-V2.6-RL-oss/blob/971cab2ab421fd88db29770db0d0e52014b63615/README.md).
The large environment assets and Docker images remain upstream inputs; they
are not copied into this source package.

| Config | Parquet path | Rows | Public task family | Schema difference |
|---|---|---:|---|---|
| `code` | `code.parquet` | 2,602 | Repository work | JSON instance contains executable-test metadata |
| `cyber` | `cyber.parquet` | 1,000 | Contained vulnerability reproduction | No required `ability` column |
| `general` | `general/train.parquet` | 989 | General agent tasks | Multiple environment `dataset_type` values |
| `webdev` | `webdev.parquet` | 2,093 | Web artifacts | Visual-verifier environment |
| `music` | `music.parquet` | 1,000 | Symbolic music composition | No required `agent_name`, `instance_id`, or `instance_json` |

Total: 7,684 rows. Row counts describe this inspected snapshot, not a guaranteed
property of a later `main` revision. Music is text in ABC notation and should
not be silently routed to an audio-generation environment.

## Canonical task envelope

`fusionrl.data.normalize_mimo_row` validates the source message list and
preserves the original private `extra_info` and `reward_model`. For the four
agent subsets, it also parses `extra_info.instance_json`, checks matching
instance identifiers and environment types, and requires a public problem
statement and a Docker alias. For music, identity comes from `extra_info.src_id`;
the composition parameters remain available to its rule-based verifier.

`TaskRecord.model_input()` returns only the original `prompt`. It never appends
`private_extra_info`, `private_instance`, or `private_reward_model`. The trusted
runner may use these fields to initialize an environment and verifier. In
particular, code `test_patch` and general `tests_files` must remain outside the
actor's prompt and mutable workspace. They are not demonstrations or target
answers. A harness must deliberately construct its public instructions: for
example, a cyber row's short prompt and its full `instance_json.problem_statement`
are different fields with different roles. This adapter preserves both and
does not guess how an external harness combines them.

`save_jsonl` writes complete trusted envelopes; `load_jsonl` checks schema
version 1. These files are coordinator inputs, not chat-template training
datasets. `load_parquet` streams batches using optional `pyarrow`. It preserves
environment-specific metadata rather than mapping every general task to one
executor. Unsupported environment types must fail at runner registration.

## Continuous reward calibration

FusionRL's explicitly named `scalar` grading route accepts an already calibrated
verifier or judge score in `[0,1]` and computes `score - group_mean(score)`.
It does not invent a pass threshold. For example, `[0.2, 0.5, 0.8]` produces
`[-0.3, 0, 0.3]`. Confirmed hacking zeroes the relevant score before the group
mean is calculated. The route rejects GRS solution/behavior inputs and GAR
quality factors, keeping each execution's reward semantics unambiguous.

A deployment must pin the task-specific rubric, aggregation formula, score
range, calibration procedure and grader version. A score of `0.8` from one
verifier is not automatically equivalent to `0.8` from another. The source
parquet import supplies task metadata; it does **not** supply or validate a
production calibration adapter. Reward calibration should be checked against
held-out evidence before mixing task families under a shared update budget.

One cost channel can be added to scalar rewards, with its own group mean
removed. The MiMo relative-length controller additionally needs explicit,
hack-corrected **binary success flags** supplied by the task verifier. Fractional
scores are not success flags, and FusionRL never silently thresholds them.
An effort controller without a success gate remains an alternative. GAR remains
restricted to binary outcomes with no nonzero additive cost channel.

## Fetch and convert

From an installed FusionRL checkout, install the optional data dependencies
specified in `pyproject.toml`, then run:

```bash
python scripts/fetch_mimo.py --output ./external_data/mimo --subsets code general webdev music cyber --export-jsonl
```

The script downloads the five selected parquet files, the source card and
`image-mapping.jsonl` at the immutable revision. Add
`--include-general-environments` to fetch `general/envs/**` separately. Those
assets account for most of the roughly 12.1 GB upstream dataset. The script
does not run a task, build a container or start a paid training job.

The source [image map](https://huggingface.co/datasets/XiaomiMiMo/MiMo-V2.6-RL-oss/blob/971cab2ab421fd88db29770db0d0e52014b63615/image-mapping.jsonl)
uses `dataset_image` and `dockerhub_image` keys. `read_image_mapping` loads
this map; `resolve_image_alias` requires an exact entry. A production deployment
should record the resolved immutable container digest, because pinning a
dataset commit does not freeze a registry tag's contents.

## Environment integrity

Treat MiMo task images as untrusted until audited. vals.ai found reference fixes reachable through git history, unreachable objects and file timestamps in this task family. A FusionRL audit of a real image confirmed the timestamp leak. All code rows also use mutable image tags and a grading script path inside the workspace. Use `fusionrl integrity audit-dataset`, `scripts/audit_mimo_images.py --attest` and `IntegrityGate` before any training run; see [integrity.md](integrity.md).

## Required environment boundaries

A registered runner owns reset, action execution, timeout handling, evaluation
and cleanup. Each rollout receives fresh writable state and an immutable
initial task. Trusted evaluation owns hidden tests and grading credentials.
Infrastructure failures reject the affected sample group and do not become
negative model rewards. Candidate outputs and documents are evidence for a
grader, not instructions for the grader. Keep raw verifier results, confirmed
hack findings and final route decisions separate for reproducibility.

The publication's environment designs and grading workflow are described in
[MiMo-V2.6 §§4.2–4.3 and §7](https://arxiv.org/html/2610.11959v1).
FusionRL's schema separation and explicit registration are implementation
choices that enforce these boundaries; they are not a claim to reproduce the
private MiMo production stack.
