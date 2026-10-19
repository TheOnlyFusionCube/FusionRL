# Training real models

FusionRL has two GPU paths:

| Path | Use it for | Parallelism |
|---|---|---|
| `fusionrl train` (this page) | Single-turn tasks with a programmatic verifier on models that fit on one GPU | One GPU, or data-parallel replicas via `torchrun` |
| [verl integration](verl.md) | Large models, sharded training, vLLM/SGLang rollout servers | FSDP/FSDP2 through the pinned upstream fork |

Both use the same execution-balanced loss and fail fast on rollout/training probability mismatch.

## Quick start

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cu124 torch
python -m pip install -e '.[real]'
fusionrl preflight --config configs/real/qwen2.5-1.5b-math-1gpu.json
fusionrl train --config configs/real/qwen2.5-1.5b-math-1gpu.json --output runs/qwen-math
# 8 GPUs on one node (each holds a full replica):
torchrun --nproc-per-node 8 -m fusionrl train --config ... --output runs/qwen-math-8x
# continue after an interruption:
fusionrl train --config ... --output runs/qwen-math --resume
```

Task file: one JSON object per line.

```json
{"task_id": "gsm8k-0001", "prompt": "Natalia sold clips to 48 friends...", "answer": "72"}
{"task_id": "chat-7", "prompt": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}], "answer": "x"}
```

Verifiers: `math_answer` (last `\boxed{}` or last number, numeric tolerance), `exact_match`, or your own `module:function(task, response)` returning a score in [0,1], or `{"score", "valid", "hack", "detail"}`. Use `valid: false` for a grader or sandbox failure: the whole group is quarantined and regenerated, never trained as a zero reward. A raised exception is treated the same way.

## What `preflight` checks

Run it on the target machine before committing GPU time. It exits nonzero on failure.

- CUDA devices and memory, bf16 support, NCCL availability.
- Estimated training state: 16 bytes/parameter for fp32 or bf16-autocast (weights, gradients, AdamW moments), 12 for pure bf16. A full fine-tune of a 7B model in bf16-autocast needs about 105 GiB before activations, so it does not fit on one 80 GB GPU. Use the verl/FSDP path for it.
- Tokenizer fingerprint: covers the token-ID mapping, special tokens, normalizer/pre-tokenizer and chat template. Checkpoints refuse to resume under a different tokenizer.
- Selective-head consistency: the scorer applies the LM head only at target positions. This is checked against the model's own full forward, which catches logit scaling or soft-capping. If they differ, it falls back to full logits automatically.
- Rollout/training parity: generates from prompts in your training file, rescores the tokens through the training path, and compares with the recorded probabilities.

## Correctness properties

- **Exact behavior probabilities.** The sampler uses the model's raw logits at temperature 1 with full support, and records the log probability each token was drawn from. It deliberately ignores the checkpoint's `generation_config`. Qwen2.5-Instruct, for example, ships temperature 0.7, top-p 0.8, top-k 20 and a repetition penalty. Inheriting those would make recorded probabilities disagree with the scored ones.
- **Reproducible randomness.** Each execution's noise stream depends only on (seed, group, replicate) and uses Gumbel-max sampling, which is exact. Resuming reproduces an uninterrupted run bit for bit, and changing the GPU count does not change the data (tested).
- **Original denominator everywhere.** Every rank and microbatch divides by the global B·N. Whole executions stay together, and gradients are SUM-all-reduced. Two data-parallel ranks match one process (tested).
- **Memory.** Targets share one token tuple per generation event, so context storage is linear in sequence length (previously quadratic). Scoring runs one forward per sequence, packed by `learner.max_tokens_per_microbatch`, and never materializes a full `[tokens, vocab]` logit tensor.
- **Parity guard.** In strict sync the first forward of each update uses the weights that sampled the batch, so its log-ratio is pure engine numerics. The run aborts if the mean per-token gap exceeds `parity.max_mean_abs_gap`, or if more than `parity.max_strict_clip_fraction` of executions are clipped before any update (the execution clip is ±0.5%, so a mismatch that size silently drops data).
- **Integrity.** Verifier hack flags zero the reward and add `reward.hack_penalty`. A sustained hack rate halts the run and leaky tasks are revoked. Agentic task records are refused. See [integrity.md](integrity.md).
- **Checkpoints.** Taken at publication boundaries: `save_pretrained` weights, tokenizer, optimizer state, and the data/queue/integrity ledger. Writes are atomic, keep the last K, and `--resume` truncates metrics to the checkpoint.

## Validated here

On CPU, with a tiny random Llama (unit tests) and with **Qwen2.5-0.5B-Instruct** (manual run):

| Check | Result |
|---|---|
| Qwen2.5-0.5B preflight: selective vs full head | max difference 0.0 |
| Qwen2.5-0.5B rollout vs training log-prob gap (fp32, 154 tokens) | mean 2.1e-6, max 1.7e-5 |
| Qwen2.5-0.5B: 3 training steps with checkpoint | gap ≤ 3.9e-5 on every step; 290 saved tensors reload bit-exact (tied embeddings) |
| Tiny Llama: recorded log-probs vs an independent forward | ≤ 2e-6, including left-padded rows |
| Gumbel sampler vs softmax (4,000 draws) | max frequency error < 0.03 |
| Resume vs uninterrupted run | identical weights and metrics |
| 2 data-parallel ranks (gloo) vs 1 process | weights equal within 2e-6 |
| Injected sampler skew (−0.3 nats/token) | run aborts with `ParityError` |

**Not validated here:** CUDA, NCCL, bf16 numerics on GPU, throughput, or a learning curve on a real benchmark. No GPU was available. Expect a bf16-autocast gap around 1e-3 to 1e-2 per token. Run `preflight` on your hardware and treat its parity report as the source of truth.

## Limits

- Generation uses the training model in-process (HF forward with KV cache). That is simple and exact, but slower than a vLLM/SGLang server. For high-throughput rollouts on large models, use the verl path.
- Models must fit on one GPU, since data parallelism replicates them. No FSDP, tensor or pipeline parallelism here.
- Only binary and calibrated scalar verifiers. GRS/GAR need grader channels that a runner must supply explicitly.
- No multi-turn or agentic environments. Those need an environment runner built on the integrity layer.
