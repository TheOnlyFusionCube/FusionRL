# Systems and trajectory contracts

FusionRL provides executable controllers for data admission, policy versions, generation chunks and resource-transition guards. They run in ordinary Python and are exercised by the test suite. A GPU backend must carry out their decisions and report the policy and probabilities it actually used. These controllers do not implement a distributed inference engine, shared RDMA cache, live Megatron membership, or remote tool cancellation.

## Source boundary

[Seer v3](https://arxiv.org/pdf/2511.14617v3), Sections 3.2–3.4 and Appendix A.1, supplies the scheduling motivation: split generation into movable chunks, estimate sibling lengths, and optimize a synchronous rollout boundary. Its production implementation combines shared KV storage with adaptive grouped speculative decoding. FusionRL implements a smaller scheduling heuristic; it does not claim Seer's measured acceleration.

[QwenGyre v1](https://arxiv.org/html/2609.33848v1), Sections 4–5 and Appendix B, supplies the execution/data separation, publication-boundary dispatch, exact token provenance, deduplicated trajectory targets and execution-level normalization. Its scheduling-lag analysis in Appendix A concerns stationary group means under specific population assumptions. FusionRL changes the ready-group selection, invalid-outcome policy and trajectory sampling. No source throughput or convergence result transfers to these changes.

## Two execution modes, one data contract

| Contract | `strict_sync` | `bounded_async` |
|---|---|---|
| Dispatch | One update's groups at a published boundary | Bounded groups at a published boundary, with optional extra groups |
| Publication | Every optimizer update | After `burst_steps` updates |
| Group completion | Every original execution must terminate and receive a valid assessment | Same |
| Actual policy provenance | Every generated policy token must match the frozen current version | Recorded per token; admission retains only targets within `max_policy_lag` |
| Empty admitted execution | Original slot and reward remain | Same, including slots made empty by stale masking |
| Group reuse | Disabled | Disabled |
| Learning interpretation | Synchronous data collection | Off-policy data collection with an explicit admission rule |

Strict mode rejects a stale token even if a later mask could hide it. Set `burst_steps=1`, `extra_dispatch_groups=0` and `max_policy_lag=0`; incompatible configurations fail at construction. A pause between chunks does not authorize a policy update.

In asynchronous mode, a dispatch version records when a group entered the queue. It does not prove which weights generated any token. A long execution can contain calls under different published policies only if the backend records each call's real policy version and behavior probability. Its original causal context must remain intact. A backend that pins executions to old snapshots must retain those snapshots until the work finishes or is explicitly cancelled.

### Operation sequence

`FusionScheduler` has the following state transitions:

1. `dispatch(specs)` reserves bounded group tickets at startup or a published boundary. The final buffer fill must contain enough groups from each domain to satisfy every update in the burst.
2. `complete(group)` registers a terminal group. The group must retain its ticket's identity, conditioning key, dispatch version and original replicate count.
3. `begin_update()` returns admitted groups when the predetermined members are complete. It returns `None` while those members are unfinished. Repeated claims while an update owns a batch fail.
4. The caller performs the optimizer update. `commit_update()` then consumes the groups exactly once and advances the trainer version. Committing without an admitted batch fails.
5. At the end of a burst, `publish()` opens the next dispatch boundary. The backend must actually distribute the matching weight snapshot before using that version for generation.

The controller's version counter is an orchestration record. Production integration must bind it to a checkpoint digest and acknowledge backend weight installation. Calling `publish()` alone does not move model weights.

### Fixed domain quotas and completion-time bias

The caller supplies integer groups-per-update quotas, such as `{"math": 1, "code": 1}`. Membership is determined by dispatch order within each domain. If an older member is unfinished, a later completed member does not replace it. This deliberately trades some throughput for transparent sampling. Rollout length and reward evaluation speed therefore cannot decide which original groups enter a particular update.

The pending limit counts all unconsumed groups: queued, running, complete, quarantined and claimed. Completing a group does not release a slot. A publication boundary refills only the consumed quota. Replaying a previously dispatched group identity fails.

### Staleness is measured at admission

`admit_group` compares the current trainer version with each actual policy-token version. Tokens beyond `max_policy_lag` are masked before selecting paths, computing any sequence ratio or forming the token denominator. Future behavior versions fail validation. The context of a retained token still includes the original preceding tokens, including preceding tokens excluded as training targets.

The scheduler records dispatch lag separately from admitted-token lag. The optional `ideal_stationary_mean_dispatch_lag` property is a no-drop accounting reference, not a guarantee or a maximum. Quarantine, replacement, cancellation, finite windows and changing quotas require measured accounting. A zero-target execution still occupies its original normalization slot; a group containing only masked targets can contribute zero gradient and should be visible in metrics rather than silently replaced by faster data.

## Exact records and target identity

`GenerationEvent` stores exact input IDs, exact output IDs, aligned behavior log probabilities, aligned policy versions and explicit provenance. `actor_id`, `role` and `branch_id` supply additional context. A role string is never sufficient evidence that output was generated by the policy.

`TrajectoryPath` references immutable generation events. `TrainingTarget.context` is the event's original input followed by the earlier output tokens from that same event. It is not reconstructed from a later chat transcript. A compaction summary creates a new context branch; it cannot change the conditioning of an earlier output.

Target identity is `(generation_event_id, output_offset)` within an execution. Repeated paths can reference the same generation event, so that event's targets are assigned once. Independently sampled equal text has distinct event IDs and remains distinct data. Input tokens, tool results, evaluator feedback and harness-injected assistant content remain context and receive no policy loss.

Each execution also has a conditioning key. Production callers should fingerprint the task revision, serialized prompt, effort setting, harness revision, rubric version and configuration generation. The rollout group adds a unique dispatch identity. Matching `group_id` strings cannot override conflicting explicit conditioning keys. Empty keys are allowed only by the low-level record layer for simple examples; scheduler tickets require a nonempty key.

### Deterministic path cap

FusionRL admits at most `jmax` paths per execution, defaulting to five. It ranks the role classes `main`, `main_summary`, `subagent` and `subagent_summary` in that order. Within a class, it chooses the path with the most remaining eligible targets, breaking ties by path ID. Selecting a path removes its target identities from the remaining candidates. Its final target mask is fixed before any loss calculation.

This deterministic rule is an explicit modification of the source's stochastic selection. It supports repeatable debugging; it does not make the capped objective an unbiased estimate of an uncapped full-tree objective. Role priority and path caps need task-specific validation. Losses must average over admitted tokens within each original execution, then preserve the original execution weighting across groups and microbatches.

## Valid scores, invalid outcomes and stage barriers

`Assessment` distinguishes a finite valid score tied to an assessed artifact/state from an unresolved evaluation. A valid zero remains useful. A timeout can be scored if its preserved work is assessable under the task's actual scoring rule. Time elapsed, termination status or trajectory length does not manufacture a task reward.

Infrastructure and evaluator failures quarantine the whole group. They do not enter group advantage statistics as zeros. `replace_quarantined` reserves the same logical slot for a fresh complete group, retaining its task, domain, replicate count and behavior snapshot. The backend must regenerate all original executions under that contract; it may not fill the slot using a convenient subset of old completions. The replacement receives a new group identity.

`cancel_pending(reason)` provides a clean boundary before a different training phase such as on-policy distillation. It requires a published boundary with no active update, returns the group identities and counts, records the reason, and removes the cancelled data from admission. It preserves seen identities so they cannot reappear through replay. The caller must cancel corresponding real backend jobs and keep their tool environments separate before another phase mutates the model. A new phase starts with a coherent new configuration and policy snapshot.

## Seer-inspired generation scheduler

`ChunkScheduler` makes actual request/engine placement decisions; `finish_chunk` advances recorded progress. One request per homogeneous prompt group serves as a probe. Before any completed sibling, the predictor uses the generation limit; afterward it uses the maximum completed sibling length. Probes prefer smaller generated lengths, other requests prefer estimated remaining work, and an age rule prevents starvation.

Homogeneous prediction groups must share prompt tokens, generation limit, tokenizer, model and frozen policy. An agent's divergent tool-conditioned calls should use separate scheduling groups even when they belong to the same learning group. Generation length is not a predictor of arbitrary external tool latency. This controller does not claim otherwise.

Each engine report gives the token capacity available to a candidate, including that candidate's existing prefix. The scheduler bounds each chunk by that capacity and the request's remaining generation limit. Profiled prefill/decode costs select among compatible engines; a soft target cost can further shorten chunks. Expensive unavoidable prefill may exceed that soft target to preserve progress. Only one chunk is in flight per engine in this reference implementation; backend concurrency requires explicit footprint reservations.

### Cache reuse is conditional

`CacheIdentity` binds the policy version, model identity, tokenizer, cache layout and exact token-prefix digest. `layout_id` must cover dtype, attention/cache format and parallel-shard arrangement. Every component must match before reuse. A different weight snapshot invalidates old KV even if the visible prefix is unchanged.

A decision with `cache_action="re_prefill"` requires a full prefix evaluation under the same requested policy snapshot. It is a correct fallback when compatible cache transfer is unavailable, with a different performance cost. The reference decision contains no RDMA operation. Grouped speculative decoding is intentionally not enabled by this controller: a backend must first demonstrate exact target-distribution verification and accurate behavior log probabilities.

## Elastic capacity guard

`ElasticState` applies execution-unit accounting. Outstanding work includes queued and running executions. It permits another cell to enter training only when remaining rollout capacity covers that entire outstanding population and a sustained ready-data threshold is met. The ready-data count is capped by the groups the current batch still needs. The state is immutable; `add_training_cell` returns the permitted next state or rejects the transition.

The backend must preserve live harness/workspace state, reserve destination service, hand off or recompute compatible caches, and acknowledge takeover before releasing source resources. Retrying a model request must not rerun an already committed tool action. Dynamic gradient participants must read the same frozen training weights, claim disjoint microbatches, preserve global execution normalization and close membership before the single optimizer owner updates weights.

## Verification included

The trajectory tests cover shared generation identity, independently repeated text, compaction contexts, provenance masks, stale tokens, empty original slots, complete-group admission, conditioning mismatches and deterministic path caps. The scheduler tests cover boundary state transitions, exactly-once use, fixed domain membership, asynchronous behavior lag, atomic quota errors, quarantine replacement, phase cancellation, chunk memory limits, cache identity, fairness and elastic waterlevel guards. These tests verify the reference contracts. They do not benchmark distributed hardware or validate learning improvements on the source workloads.
