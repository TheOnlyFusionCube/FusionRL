import unittest
from dataclasses import replace

from fusionrl.scheduler import (
    CacheIdentity, ChunkScheduler, ElasticState, FusionScheduler, GroupSpec,
    InferenceEngine, RolloutRequest, SchedulerConfig, can_reuse_cache,
)
from fusionrl.trajectory import (
    Assessment, ExecutionRecord, GenerationEvent, PolicyVersionMismatch,
    QuarantinedGroup, RolloutGroup, TrajectoryPath,
)


def finished_group(ticket, token_version=None, invalid=False):
    version = ticket.dispatch_version if token_version is None else token_version
    records = []
    for index in range(ticket.expected_executions):
        event = GenerationEvent(f"output-{index}", (1,), (2,), (-0.7,), (version,), ("policy",))
        records.append(ExecutionRecord(
            f"{ticket.group_id}-e{index}", ticket.group_id, (event,),
            (TrajectoryPath("main", (event.event_id,)),),
            Assessment(None, False, "state", "infra") if invalid else Assessment(float(index % 2), True, "state"),
            conditioning_key=ticket.conditioning_key,
        ))
    return RolloutGroup(ticket.group_id, tuple(records), ticket.expected_executions,
                        ticket.dispatch_version, ticket.conditioning_key)


class GroupSchedulerTests(unittest.TestCase):
    def test_strict_roundtrip_and_exactly_once_consumption(self):
        scheduler = FusionScheduler(SchedulerConfig(executions_per_group=2))
        ticket, = scheduler.dispatch([GroupSpec("g0", "key")])
        self.assertIsNone(scheduler.begin_update())
        scheduler.complete(finished_group(ticket))
        admitted = scheduler.begin_update()
        self.assertEqual(admitted[0].group_id, "g0")
        with self.assertRaises(RuntimeError):
            scheduler.begin_update()
        self.assertEqual(scheduler.commit_update(), 1)
        self.assertEqual(scheduler.published_version, 0)
        self.assertEqual(scheduler.publish(), 1)
        self.assertEqual(scheduler.consumed_groups, ("g0",))
        with self.assertRaises(ValueError):
            scheduler.dispatch([GroupSpec("g0", "key")])

    def test_strict_rejects_stale_instead_of_masking(self):
        scheduler = FusionScheduler(SchedulerConfig(executions_per_group=2), policy_version=2)
        ticket, = scheduler.dispatch([GroupSpec("g0", "key")])
        with self.assertRaises(PolicyVersionMismatch):
            scheduler.complete(finished_group(ticket, token_version=1))
        self.assertEqual(scheduler.ready_count, 0)

    def test_async_preserves_boundary_quota_and_masks_actual_token_lag(self):
        cfg = SchedulerConfig(mode="bounded_async", executions_per_group=2, burst_steps=2,
                              extra_dispatch_groups=1, max_policy_lag=1)
        scheduler = FusionScheduler(cfg)
        tickets = scheduler.dispatch([GroupSpec(f"g{i}", f"task{i}") for i in range(3)])
        scheduler.complete(finished_group(tickets[1]))
        scheduler.complete(finished_group(tickets[2]))
        self.assertIsNone(scheduler.begin_update())  # g0 is the predetermined member.
        scheduler.complete(finished_group(tickets[0]))
        self.assertEqual(scheduler.begin_update()[0].group_id, "g0")
        scheduler.commit_update()
        with self.assertRaises(RuntimeError):
            scheduler.dispatch([GroupSpec("midburst", "key")])
        with self.assertRaises(RuntimeError):
            scheduler.publish()
        self.assertEqual(scheduler.begin_update()[0].group_id, "g1")
        scheduler.commit_update()
        scheduler.publish()
        scheduler.dispatch([GroupSpec("g3", "task3"), GroupSpec("g4", "task4")])
        stale = scheduler.begin_update()[0]
        self.assertEqual(stale.group_id, "g2")
        self.assertEqual(len(stale.executions), 2)
        self.assertEqual(stale.target_count, 0)
        self.assertEqual(sum(execution.masked_stale for execution in stale.executions), 2)
        scheduler.commit_update()
        self.assertTrue(all(lag <= 1 for lag in scheduler.admitted_token_lags))

    def test_dispatch_lag_is_not_behavior_lag(self):
        cfg = SchedulerConfig(mode="bounded_async", executions_per_group=2,
                              extra_dispatch_groups=1, max_policy_lag=0)
        scheduler = FusionScheduler(cfg)
        old, survivor = scheduler.dispatch([GroupSpec("old", "a"), GroupSpec("survivor", "b")])
        scheduler.complete(finished_group(old))
        scheduler.begin_update()
        scheduler.commit_update()
        scheduler.publish()
        scheduler.dispatch([GroupSpec("new", "c")])
        scheduler.complete(finished_group(survivor, token_version=1))
        admitted = scheduler.begin_update()[0]
        self.assertEqual(admitted.dispatch_version, 0)
        self.assertEqual(admitted.target_count, 2)
        scheduler.commit_update()
        self.assertEqual(scheduler.dispatch_lags[-1], 1)
        self.assertEqual(scheduler.admitted_token_lags[-1], 0)

    def test_fixed_domain_mix_does_not_take_first_ready_groups(self):
        cfg = SchedulerConfig(mode="bounded_async", groups_per_update=2, executions_per_group=2,
                              extra_dispatch_groups=2, max_policy_lag=2)
        scheduler = FusionScheduler(cfg, quotas={"math": 1, "code": 1})
        tickets = scheduler.dispatch([GroupSpec("m0", "m0", "math"), GroupSpec("c0", "c0", "code"),
                                      GroupSpec("m1", "m1", "math"), GroupSpec("c1", "c1", "code")])
        for ticket in tickets[2:]:
            scheduler.complete(finished_group(ticket))
        self.assertIsNone(scheduler.begin_update())
        for ticket in tickets[:2]:
            scheduler.complete(finished_group(ticket))
        self.assertEqual([group.group_id for group in scheduler.begin_update()], ["m0", "c0"])

    def test_final_boundary_fill_rejects_impossible_domain_mix_atomically(self):
        scheduler = FusionScheduler(SchedulerConfig(groups_per_update=2), quotas={"math": 1, "code": 1})
        with self.assertRaises(ValueError):
            scheduler.dispatch([GroupSpec("m0", "key", "math"), GroupSpec("m1", "key", "math")])
        self.assertEqual(scheduler.pending_count, 0)
        scheduler.dispatch([GroupSpec("m0", "key", "math"), GroupSpec("c0", "key", "code")])
        self.assertEqual(scheduler.pending_count, 2)

    def test_invalid_group_can_only_be_replaced_as_a_whole(self):
        scheduler = FusionScheduler(SchedulerConfig(executions_per_group=2))
        ticket, = scheduler.dispatch([GroupSpec("bad", "key")])
        with self.assertRaises(QuarantinedGroup):
            scheduler.complete(finished_group(ticket, invalid=True))
        with self.assertRaises(QuarantinedGroup):
            scheduler.begin_update()
        replacement = scheduler.replace_quarantined("bad", "retry")
        self.assertEqual(replacement.sequence, ticket.sequence)
        self.assertEqual(replacement.dispatch_version, ticket.dispatch_version)
        self.assertEqual(scheduler.pending_count, 1)
        scheduler.complete(finished_group(replacement))
        self.assertEqual(scheduler.begin_update()[0].group_id, "retry")

    def test_stage_barrier_cancels_old_data_without_permitting_replay(self):
        cfg = SchedulerConfig(mode="bounded_async", executions_per_group=2,
                              extra_dispatch_groups=1, max_policy_lag=1)
        scheduler = FusionScheduler(cfg)
        first, leftover = scheduler.dispatch([GroupSpec("first", "a"), GroupSpec("leftover", "b")])
        scheduler.complete(finished_group(first))
        scheduler.complete(finished_group(leftover))
        scheduler.begin_update()
        with self.assertRaises(RuntimeError):
            scheduler.cancel_pending("OPD barrier")
        scheduler.commit_update()
        scheduler.publish()
        report = scheduler.cancel_pending("OPD barrier")
        self.assertEqual(report["groups"], 1)
        self.assertEqual(report["ready_groups"], 1)
        self.assertEqual(report["execution_slots"], 2)
        self.assertEqual(scheduler.pending_count, 0)
        with self.assertRaises(ValueError):
            scheduler.dispatch([GroupSpec("leftover", "b")])


class CacheAndChunkTests(unittest.TestCase):
    def test_cache_requires_every_identity_component(self):
        cache = CacheIdentity.from_tokens(2, "tok", "bf16-tp2", (1, 2), "model-digest")
        arguments = dict(policy_version=2, tokenizer_id="tok", layout_id="bf16-tp2",
                         prefix_token_ids=(1, 2), model_id="model-digest")
        self.assertTrue(can_reuse_cache(cache, **arguments))
        for name, mismatch in [("policy_version", 3), ("tokenizer_id", "new-tok"),
                               ("layout_id", "fp8-tp4"), ("prefix_token_ids", (1, 3)),
                               ("model_id", "other-model")]:
            with self.subTest(field=name):
                self.assertFalse(can_reuse_cache(cache, **{**arguments, name: mismatch}))

    def test_chunk_respects_memory_and_frozen_policy(self):
        scheduler = ChunkScheduler(chunk_tokens=10)
        scheduler.submit(RolloutRequest("r", "g", 0, (1, 2), 20))
        self.assertIsNone(scheduler.choose([InferenceEngine("wrong", 1, 100)]))
        decision = scheduler.choose([InferenceEngine("engine", 0, 5)])
        self.assertEqual(decision.max_new_tokens, 3)
        self.assertEqual(decision.cache_action, "re_prefill")
        with self.assertRaises(RuntimeError):
            scheduler.update_policy_version(1)
        with self.assertRaises(PolicyVersionMismatch):
            scheduler.finish_chunk("r", (3,), behavior_policy_version=1)
        scheduler.finish_chunk("r", (3, 4, 5), behavior_policy_version=0, finished=True)
        scheduler.update_policy_version(1)
        self.assertEqual(scheduler.policy_version, 1)

    def test_one_chunk_per_engine_prevents_unreserved_concurrent_placement(self):
        scheduler = ChunkScheduler(chunk_tokens=1)
        scheduler.submit(RolloutRequest("r0", "g", 0, (1,), 3))
        scheduler.submit(RolloutRequest("r1", "g", 0, (1,), 3))
        engine = InferenceEngine("engine", 0, 100)
        decision = scheduler.choose([engine])
        self.assertIsNone(scheduler.choose([engine]))
        scheduler.finish_chunk(decision.request_id, (2,), 0)
        self.assertIsNotNone(scheduler.choose([engine]))

    def test_group_prediction_drives_remaining_long_work_after_probes(self):
        scheduler = ChunkScheduler(chunk_tokens=5, max_wait_rounds=100)
        for rid, gid in [("a0", "a"), ("a1", "a"), ("b0", "b"), ("b1", "b")]:
            scheduler.submit(RolloutRequest(rid, gid, 0, (1,), 10))
        engine = InferenceEngine("engine", 0, 100)
        self.assertEqual(scheduler.choose([engine]).request_id, "a0")
        scheduler.finish_chunk("a0", (2, 2), 0, finished=True)
        self.assertEqual(scheduler.choose([engine]).request_id, "b0")
        scheduler.finish_chunk("b0", (2, 2, 2, 2), 0, finished=True)
        decision = scheduler.choose([engine])
        self.assertEqual(decision.request_id, "b1")
        self.assertEqual(decision.estimated_remaining, 4)

    def test_age_guard_prevents_probe_starvation_of_other_requests(self):
        scheduler = ChunkScheduler(chunk_tokens=1, max_wait_rounds=2)
        scheduler.submit(RolloutRequest("probe", "g", 0, (1,), 100))
        scheduler.submit(RolloutRequest("ordinary", "g", 0, (1,), 100))
        engine = InferenceEngine("engine", 0, 200)
        self.assertEqual(scheduler.choose([engine]).request_id, "probe")
        scheduler.finish_chunk("probe", (2,), 0)
        self.assertEqual(scheduler.choose([engine]).request_id, "ordinary")

    def test_cost_choice_uses_matching_cache_and_returns_reprefill_for_mismatch(self):
        scheduler = ChunkScheduler(chunk_tokens=4)
        scheduler.submit(RolloutRequest("r", "g", 0, (1, 2, 3, 4), 4))
        cache = CacheIdentity.from_tokens(0, "default-tokenizer", "default-layout", (1, 2, 3, 4))
        cached = InferenceEngine("cached", 0, 100, caches={"r": cache})
        uncached = InferenceEngine("uncached", 0, 100)
        decision = scheduler.choose([uncached, cached])
        self.assertEqual(decision.engine_id, "cached")
        self.assertEqual(decision.cache_action, "reuse")
        self.assertEqual(decision.estimated_cost, 4)

    def test_homogeneous_prediction_groups_reject_different_prompts(self):
        scheduler = ChunkScheduler()
        scheduler.submit(RolloutRequest("r0", "g", 0, (1,), 5))
        with self.assertRaises(ValueError):
            scheduler.submit(RolloutRequest("r1", "g", 0, (2,), 5))


class ElasticGuardTests(unittest.TestCase):
    def test_waterlevel_covers_queued_and_running_executions(self):
        state = ElasticState(cells=4, training_cells=0, per_cell_capacity=10,
                             standalone_capacity=0, dispatched=35, launched=20, terminated=5)
        self.assertEqual(state.running, 15)
        self.assertEqual(state.outstanding, 30)
        self.assertEqual(state.launch_count, 15)
        next_state = state.add_training_cell(ready_groups=2, groups_per_update=2)
        self.assertEqual(next_state.rollout_capacity, 30)
        self.assertFalse(next_state.can_add_training_cell(ready_groups=2, groups_per_update=2))

    def test_supply_hysteresis_and_remaining_batch_cap(self):
        state = ElasticState(4, 0, 10, 0, 5, 5, 5)
        self.assertFalse(state.can_add_training_cell(4, 4, stable_for=1, required_stability=2))
        self.assertFalse(state.can_add_training_cell(4, 4, minimum_ready_fraction=0.5, groups_still_needed=1))
        self.assertTrue(state.can_add_training_cell(4, 4, minimum_ready_fraction=0.5,
                                                   stable_for=2, required_stability=2))


if __name__ == "__main__":
    unittest.main()
