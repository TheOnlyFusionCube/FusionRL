# FusionRL integration with XiaomiMiMo/verl

## Status and supported scope

This package includes a concrete adapter and a small, commit-specific patch for the public XiaomiMiMo fork. The adapter calls FusionRL's shared advantage and execution-loss implementations. Its tests cover scalar values, gradients, distribution across microbatches, configuration restrictions, source interfaces, and application of the patch to the actual pinned files. **A GPU, Ray, or distributed model-training run of this integration has not been executed.** The standalone CPU reference pipeline has its own validation results in the main documentation.

The backend deliberately supports a smaller scope than FusionRL's portable algorithm core:

| Capability | This verl adapter | Portable FusionRL core |
|---|---|---|
| Binary verified outcomes, centered by prompt group | Implemented | Implemented |
| QwenGyre execution-balanced loss | Implemented through the shared core | Implemented |
| Strict synchronous policy versions | Required, checked on actual rollout tags | Supported |
| Bounded asynchronous admission | Rejected | Available with required provenance |
| One physical row per execution | Required and checked | General execution/path representation |
| Multiple causal paths or overlapping session outputs | Rejected | Explicit path identity and unique target selection |
| GAR or GRS reward routing | Rejected by this adapter | Implemented as alternative routes |
| Effort costs or depth-dependent penalties | Rejected by this adapter | Constrained profiles in the core |
| MiMo Code / General / Cyber / Webdev harnesses | Data adapters and upstream references only | Environment contracts; external runners still needed |
| FSDP / FSDP2 | Integration contract inspected and CPU-tested | Backend-independent tensor loss |
| Megatron context/pipeline/expert parallel training | Rejected | Requires an additional validated reduction adapter |

Selecting an unsupported combination raises a descriptive error. Registering a new loss name alone is insufficient to claim full integration.

## Exact upstream revision

- Repository: https://github.com/XiaomiMiMo/verl
- Branch inspected: `mimo-oss`.
- Commit: `79090e58195e80a9cfe251c95ae7a78a2132b6e2`.
- Inspection date: October 19, 2026.
- MiMo agent submodule: `third_party/mimoagent-osr`, commit `c3df8469f250b15e7906487a6d52be4c1827a509`.
- Uni-Agent submodule: `third_party/uni_agent`, commit `c63e0b01c375ebede95e01fe92bc367df24e5bf3`.

The submodule pins are taken from the commit's gitlinks; `.gitmodules` records their repository URLs. The README's newer repository spelling may differ from `.gitmodules`. The single-turn integration does not use either submodule. [Pinned repository](https://github.com/XiaomiMiMo/verl/tree/79090e58195e80a9cfe251c95ae7a78a2132b6e2), [submodule routing](https://github.com/XiaomiMiMo/verl/blob/79090e58195e80a9cfe251c95ae7a78a2132b6e2/.gitmodules).

`fusionrl/integrations/verl/source_contract.json` contains SHA-256 fingerprints and parsed function signatures for ten relevant source files, both before and after the patch. The bootstrap script rejects a different revision or unexpected edits in these files. A future upstream revision needs a new audit and patch, rather than weakening the fingerprint checks.

## What the patch changes

`patches/verl-fusionrl-strict-sync.patch` changes four files:

1. `verl/trainer/main_ppo.py`: validate the FusionRL configuration and register its hooks before Ray startup.
2. `verl/trainer/ppo/v1/trainer_base.py`: check complete logical executions after batch balancing; register the advantage estimator in the Ray trainer process; copy the actual rollout likelihood into the field consumed by the policy loss.
3. `verl/workers/utils/losses.py`: explicitly register the FusionRL policy loss inside each actor worker before lookup.
4. `verl/experimental/agent_loop/agent_loop.py`: check the actual runtime agent name and sampling parameters, including overrides coming from dataset rows.

The unchanged upstream code continues to own model execution, TransferQueue transport, checkpointing, optimizer updates, and ordinary scheduling. The patch branches only on the exact FusionRL names. It does not replace existing algorithms or globally monkeypatch their implementations.

The registry names are `fusionrl_binary` and `fusionrl_execution`. The actual upstream loss call ends with the keyword `rollout_is_weights`, despite an older type annotation describing the last tensor differently. This interface was read from the real caller. [Loss dispatch](https://github.com/XiaomiMiMo/verl/blob/79090e58195e80a9cfe251c95ae7a78a2132b6e2/verl/workers/utils/losses.py).

## Probability, advantage, and reduction semantics

The loss needs the probability that sampled each token. Recomputing an old model under the training engine would introduce a third distribution. The patch explicitly preserves recorded `rollout_log_probs` as `old_log_probs` for this one loss. The misleading legacy field name does not change its meaning.

The profile requires temperature 1, top-p 1, top-k disabled, repetition penalty 1, and log-probability capture. It checks both the configured values and the values passed into the actual rollout request. This avoids incorrectly using unadjusted model likelihoods as probabilities of a truncated sampler. The adapter rejects additional rollout importance weights, bypass-mode losses, reference KL penalties, entropy bonuses, distillation losses, and dense advantage edits.

Advantages are one scalar per execution, obtained by calling `fusionrl.rewards.compute_advantages(..., route="binary")` on each complete prompt group. A successful verifier outcome is 1; failure is 0. Infrastructure failure is not silently converted into a negative training example. An infrastructure marker, incomplete group, pending visual grade, or missing policy version aborts the update. This conservative backend behavior differs from the portable core, which can preserve invalid/empty original execution slots in a richer admission record.

Because the profile is strict sync with one minibatch and one epoch, every microbatch is scored under the weights that sampled it, so `log_prob - old_log_prob` is pure rollout/training engine mismatch. The loss reports `actor/fusionrl_rollout_train_gap_mean` and `_max`. It raises if the mean per-token gap exceeds `FUSIONRL_MAX_ROLLOUT_TRAIN_GAP` (default 0.1), or if more than `FUSIONRL_MAX_STRICT_CLIP_FRACTION` (default 0.2) of at least 8 executions are clipped before any update. The execution clip is ±0.5% on the mean log ratio, so a vLLM-versus-FSDP precision mismatch of that size would otherwise silently drop data. Set both variables in the Ray runtime environment if your deployment needs different limits.

Each microbatch passes the **original global number of executions** to `execution_loss`. It does not pass a token count or divide by the number of surviving targets. The adapter multiplies each local contribution by the FSDP data-parallel size once, compensating for distributed gradient averaging. The pinned FSDP loop accumulates `loss.backward()` directly across microbatches and does not divide each loss again by the microbatch count. Tests compare both values and gradients against the shared core, including an empty execution. [FSDP accumulation](https://github.com/XiaomiMiMo/verl/blob/79090e58195e80a9cfe251c95ae7a78a2132b6e2/verl/workers/engine/fsdp/transformer_impl.py).

The profile requires one complete sampled prompt batch per optimizer minibatch and one optimizer epoch. Sequence rows remain intact during microbatching. A real, represented execution with zero eligible tokens still belongs in the denominator. A missing execution cannot be reconstructed by adding arbitrary padding: the batch guard rejects missing records and synthetic data-parallel padding. Choose a batch size divisible by the number of data-parallel workers.

Strict synchronous version checking follows the pinned trainer's convention. It loads rollout weights at version 0, increments the training step before the first update, and measures rollout staleness relative to `global_steps - 1`. Therefore an update at step 1 requires `min_global_steps == max_global_steps == 0`. The LLM client derives those tags from the rollout server's actual weight-version metadata. Missing or mixed versions are rejected. [Trainer convention](https://github.com/XiaomiMiMo/verl/blob/79090e58195e80a9cfe251c95ae7a78a2132b6e2/verl/trainer/ppo/v1/trainer_base.py), [LLM version propagation](https://github.com/XiaomiMiMo/verl/blob/79090e58195e80a9cfe251c95ae7a78a2132b6e2/verl/workers/rollout/llm_server.py).

## Bootstrap and inspect

Run these commands from the extracted FusionRL project. The bootstrap downloads public source and applies a local patch; it does not install CUDA libraries, download weights, start task containers, or run training.

```bash
python scripts/bootstrap_verl.py --destination /absolute/path/to/fusionrl-verl

export FUSIONRL_VERL_ROOT=/absolute/path/to/fusionrl-verl
bash scripts/train_verl.sh --check-only
```

To check an existing original checkout without modifying it:

```bash
python scripts/bootstrap_verl.py \
  --destination /absolute/path/to/original-verl \
  --check-only
```

To fetch the separately pinned MiMo harness submodules for future environment integration, add `--submodules` to a normal bootstrap invocation. That does not make those harnesses supported by this launcher.

## GPU environment and launch contract

Use a separate GPU environment built from the pinned upstream's installation recipe and dependency constraints. The source records `TransferQueue==0.1.8`; its SGLang extra records `sglang[srt,openai]==0.5.8` and `torch==2.9.1`. It also restricts `tensordict` and `transformers`. The included profile defaults to vLLM; choose the appropriate upstream container/environment for the selected engine. These are upstream constraints, **not a tested FusionRL hardware bill of materials**. [Upstream setup](https://github.com/XiaomiMiMo/verl/blob/79090e58195e80a9cfe251c95ae7a78a2132b6e2/setup.py), [requirements](https://github.com/XiaomiMiMo/verl/blob/79090e58195e80a9cfe251c95ae7a78a2132b6e2/requirements.txt).

The launcher requires an actual compatible model checkpoint, separate training/validation Parquet files, and a trusted binary verifier. It checks the patched source first. Missing dependencies or unsupported configurations stop before training. A concrete invocation has this form:

```bash
export FUSIONRL_VERL_ROOT=/absolute/path/to/fusionrl-verl
bash scripts/train_verl.sh -- \
  actor_rollout_ref.model.path=/absolute/path/to/model \
  data.train_files=/absolute/path/to/train.parquet \
  data.val_files=/absolute/path/to/validation.parquet \
  reward.custom_reward_function.path=/absolute/path/to/trusted_binary_verifier.py \
  reward.custom_reward_function.name=compute_score \
  trainer.n_gpus_per_node=1
```

The verifier uses the upstream hook signature:

```python
def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    # Call a trusted verifier appropriate to the dataset.
    # Return {"score": 0.0} or {"score": 1.0}.
    ...
```

That schematic signature is not a production verifier implementation. Generated code, shell commands, and task artifacts must run in the domain's isolated execution service. A dataset containing MiMo `agent_name` values cannot be passed into the single-turn launcher unchanged: the runtime guard rejects incompatible dispatch.

Ordinary upstream RL Parquet records carry `data_source`, a `prompt` message list, `reward_model` with its scoring metadata, and `extra_info`. FusionRL's canonical `TaskRecord` preserves additional private verifier/environment metadata and is intentionally not a universal replacement for every upstream Parquet schema.

## Why the other MiMo harnesses need explicit adapters

The release has five different training environments. Code uses Uni-Agent's framework adapter and TransferQueue session capture; General, Cyber, and Webdev use native verl agent loops around MiMo harnesses; Music uses a separate single-turn symbolic scorer. Their Docker/Kubernetes, judge, and transport requirements differ. The current Code README also requires `anti_hack_cleanup: true`; preserve that setting when implementing its environment adapter. [Pinned README](https://github.com/XiaomiMiMo/verl/blob/79090e58195e80a9cfe251c95ae7a78a2132b6e2/README.md).

Three upstream details require particular care in a larger integration:

- The native `prompt-mean` aggregation assigns equal total weight to prompts, not executions, and its configuration currently accepts only the vanilla PPO loss. FusionRL's chosen execution objective therefore does not enable it. [Actor constraints](https://github.com/XiaomiMiMo/verl/blob/79090e58195e80a9cfe251c95ae7a78a2132b6e2/verl/workers/config/actor.py).
- V1's multi-output session helper computes terminal-outcome advantages once per session only for the built-in GRPO identifier. A custom estimator name takes a generic route. Counting all session outputs as independent executions would be incorrect; this adapter rejects them. [Session advantage helper](https://github.com/XiaomiMiMo/verl/blob/79090e58195e80a9cfe251c95ae7a78a2132b6e2/verl/trainer/ppo/v1/utils.py).
- Webdev rewrites placeholder rewards using group judgments after sampling. Filtering uniform placeholder rewards before grading would discard useful groups. Its visual artifacts also require shared storage between rollout workers and the driver. A correct GAR/GRS visual adapter must complete grading before advantage computation and distinguish judge failures from low-quality outputs. [Group reward path](https://github.com/XiaomiMiMo/verl/blob/79090e58195e80a9cfe251c95ae7a78a2132b6e2/recipes/design/webdev/group_reward.py).

A future full agent adapter should transport original execution IDs, path IDs, exact sampled token IDs and log probabilities, per-token policy versions, group identity, verifier versions, validity, and auxiliary grader channels. It must preserve original execution denominators across path filtering and distributed partitioning. Those records already have a portable representation in FusionRL; wiring them through every upstream harness remains explicit integration work.

## Verification

The adapter tests run with the rest of the package. To additionally check the real pristine upstream checkout and apply the patch inside a disposable copy:

```bash
FUSIONRL_VERL_SOURCE=/absolute/path/to/original-verl \
  python -m unittest discover -s tests -p test_verl_contract.py -v
```

The supplied integration was checked against the source pin and all adapter tests passed, including the Torch loss/gradient tests. GPU kernels, model compatibility, distributed checkpoint recovery, external verifiers, and task environment services remain to be validated in the intended deployment environment.
