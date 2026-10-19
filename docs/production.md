# Connecting a real training backend

The portable implementation is complete for its defined CPU reference and admission/loss APIs. A real agent-training deployment supplies the following concrete responsibilities. These are integration requirements, not performance claims or silently implemented services.

## Minimal supported GPU path

Use the exact pinned XiaomiMiMo/verl revision and patch as described in [verl.md](verl.md). Start with a compatible causal model, a single-turn dataset and a binary verifier. Keep each physical row equal to one independent execution. Use a batch divisible by the data-parallel world size. Preserve the full-support sampler settings and actual behavior probabilities. This is the adapter whose source signatures, numerical reductions and Hydra composition were tested.

Install the upstream CUDA/inference/TransferQueue dependencies in a separate environment using the pinned repository's requirements. The CPU validation's PyTorch wheel is not a GPU environment lock. The archive neither launches a cluster nor silently downloads weights or task containers.

## Interface for richer agent executions

A richer harness should perform these operations for each task group:

1. Resolve the pinned task, environment image digest, initial repository state, verifier/rubric revision and model-visible prompt. Reserve N independent original execution IDs. Set one effort and one harness for the entire group.
2. For every model request, write `GenerationEvent` directly from the engine's actual input/output IDs, behavior log probabilities, policy versions and provenance. Preserve media/processor references and hashes in the backend's durable context store; do not reduce a multimodal request to text and assume equivalence.
3. Track branches and summaries through `TrajectoryPath`. Assign generation IDs at sampling time. A repeated prefix references the same event; an independently sampled identical string gets a fresh event ID.
4. Assess the actual artifact/state. Keep public observations separate from hidden test assets and grader instructions. Report a valid policy failure separately from an infrastructure or evaluator failure.
5. Submit complete `RolloutGroup` records to the controller. Bind `GroupSupervision` to the exact execution and artifact order and the same conditioning key. Supply required GRS scores, GAR factors or calibrated scalar grades explicitly.
6. Call `prepare_step`, then score every retained target through `learner_loss` using its original context. Preserve per-execution reduction across tensor packing, microbatches and data-parallel workers.
7. After a successful update, commit the batch, publish the actual checkpoint to inference engines, wait for their acknowledgment, and refill the reserved group quotas. Bind each version to the installed weight digest.

MoE models require a concrete routing implementation: router freezing alone does not make experts invariant after other weights change. Store and replay the route belonging to each generation segment when that backend contract is selected. Multimodal processors, speculative decoding and restricted samplers similarly need explicit probability/context equivalence checks. Their mere presence in a model configuration is not backend support.

## Failure, recovery and elastic resources

For a quarantined group, retain its logical task/domain slot and regenerate the whole group with a fresh identity. A completed group whose token versions violate the mode contract (a changed frozen policy in strict sync, or unpublished weights) is quarantined the same way; the controller then refuses another record for that slot and `replace_quarantined` retries it. Keep the required old snapshot until retry completion or explicitly cancel it at a published boundary. Do not replace an invalid execution with an unrelated ready sample.

Persist group tickets, immutable payload references, assessment versions, consumed IDs, publication digests and optimizer state together in a real recovery checkpoint. The tiny model's export intentionally contains no resumable distributed queue. The exact-once guarantees of a live controller do not by themselves survive a process crash without this ledger.

`ChunkScheduler` returns placement, token budget and cache-reuse decisions. An engine must implement interruption and report progress. `CacheIdentity` prevents generic reuse across a different model, policy, tokenizer, layout or prefix. Without a compatible shared-cache transport, re-prefill the exact prefix under the same frozen request policy. `ElasticState` guards waterlevel and stable training supply; backend resource ownership, process-group reconfiguration and weight migration remain explicit operations.

## What to measure before expanding scope

Use held-out tasks separated by repository/task family and provenance, and keep evaluation assets outside training prompts. Establish the actual rollout-to-training log-probability discrepancy on identical contexts. Validate objective values and gradients through the real engine, including empty executions and shared paths. Measure completion distributions, invalid-group counts, stale-token masks, execution clipping, per-domain membership and resource utilization.

Compare success, solution quality and tokens at matched inference settings and training compute. Ablate one change at a time: execution weighting, quality grading, effort cost, then asynchronous scheduling. Repeat seeds and retain confidence intervals appropriate to the benchmark. No synthetic demo result in this archive supports a frontier-model quality, throughput or cost prediction.

