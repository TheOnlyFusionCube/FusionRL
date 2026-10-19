import unittest
from dataclasses import replace

from fusionrl.trajectory import (
    Assessment, ExecutionRecord, GenerationEvent, IncompleteGroup,
    PolicyVersionMismatch, QuarantinedGroup, RolloutGroup, TrajectoryPath,
    admit_group,
)


def event(name, context=(1,), output=(2,), versions=None, provenance=None, role="main"):
    provenance = provenance or ("policy",) * len(output)
    versions = versions if versions is not None else (2,) * len(output)
    logp = tuple(-0.5 if origin == "policy" else None for origin in provenance)
    return GenerationEvent(name, context, output, logp, versions, provenance, role=role)


def execution(events=(), paths=(), name="e0", score=1.0, key="task:effort:harness:rubric:v1"):
    return ExecutionRecord(name, "g0", tuple(events), tuple(paths),
                           Assessment(score, True, f"artifact:{name}"), conditioning_key=key)


def group(*executions, expected=None):
    return RolloutGroup("g0", tuple(executions), expected or len(executions),
                        dispatch_version=2, conditioning_key="task:effort:harness:rubric:v1")


class TrajectoryAdmissionTests(unittest.TestCase):
    def test_shared_prefix_counted_once_and_context_never_reconstructed(self):
        prefix = event("prefix", (10,), (20,))
        main = event("main", (10, 20, 91), (30,))
        branch = event("branch", (10, 20, 92), (40,), role="subagent")
        record = execution([prefix, main, branch], [
            TrajectoryPath("main", ("prefix", "main")),
            TrajectoryPath("branch", ("prefix", "branch"), role="subagent"),
        ])
        admitted = admit_group(group(record), 2).executions[0]
        self.assertEqual([target.target_id for target in admitted.targets],
                         [("prefix", 0), ("main", 0), ("branch", 0)])
        self.assertEqual([target.context for target in admitted.targets],
                         [(10,), (10, 20, 91), (10, 20, 92)])
        self.assertEqual(admitted.masked_shared_occurrences, 1)

    def test_equal_independent_outputs_are_distinct_generation_events(self):
        first = event("independent_1", (7,), (8,))
        second = event("independent_2", (7,), (8,))
        record = execution([first, second], [TrajectoryPath("a", ("independent_1",)),
                                            TrajectoryPath("b", ("independent_2",))])
        targets = admit_group(group(record), 2).executions[0].targets
        self.assertEqual(len(targets), 2)
        self.assertNotEqual(targets[0].target_id, targets[1].target_id)

    def test_compaction_does_not_recondition_old_tokens(self):
        before = event("before", (10, 11, 12), (13, 14))
        after = event("after", (999, 25), (30,))
        record = execution([before, after], [TrajectoryPath("pre", ("before",)),
                                            TrajectoryPath("post", ("after",))])
        targets = admit_group(group(record), 2).executions[0].targets
        by_id = {target.target_id: target.context for target in targets}
        self.assertEqual(by_id[("before", 1)], (10, 11, 12, 13))
        self.assertEqual(by_id[("after", 0)], (999, 25))

    def test_provenance_overrides_assistant_like_role_and_context_is_preserved(self):
        mixed = event("mixed", (1,), (70, 80), versions=(None, 2),
                      provenance=("harness", "policy"))
        record = execution([mixed], [TrajectoryPath("main", ("mixed",))])
        admitted = admit_group(group(record), 2).executions[0]
        self.assertEqual(len(admitted.targets), 1)
        self.assertEqual(admitted.targets[0].context, (1, 70))
        self.assertEqual(admitted.targets[0].token_id, 80)
        self.assertEqual(admitted.masked_provenance, 1)

    def test_stale_tokens_masked_before_selection_but_remain_in_context(self):
        mixed = event("mixed", (1,), (20, 21), versions=(0, 2))
        record = execution([mixed], [TrajectoryPath("main", ("mixed",))])
        admitted = admit_group(group(record), 2, max_policy_lag=1).executions[0]
        self.assertEqual([target.token_id for target in admitted.targets], [21])
        self.assertEqual(admitted.targets[0].context, (1, 20))
        self.assertEqual(admitted.masked_stale, 1)

    def test_empty_execution_slots_and_valid_zero_reward_are_preserved(self):
        active = execution([event("active")], [TrajectoryPath("main", ("active",))])
        empty = execution(name="e1", score=0)
        admitted = admit_group(group(active, empty), 2)
        self.assertEqual(len(admitted.executions), 2)
        self.assertEqual(admitted.rewards, (1.0, 0.0))
        self.assertEqual(admitted.executions[1].targets, ())
        self.assertEqual(admitted.executions[1].original_index, 1)

    def test_infrastructure_zero_does_not_enter_group_statistics(self):
        failed = replace(execution(score=0), status="infra_failed")
        with self.assertRaises(QuarantinedGroup):
            admit_group(group(failed), 2)

    def test_invalid_evaluation_is_not_a_valid_zero(self):
        invalid = replace(execution(score=0), assessment=Assessment(None, False, "state", "evaluator unavailable"))
        with self.assertRaises(QuarantinedGroup):
            admit_group(group(invalid), 2)
        self.assertEqual(admit_group(group(execution(score=0)), 2).rewards, (0.0,))

    def test_assessable_partial_work_is_permitted(self):
        partial = replace(execution(score=0.25), status="timed_out")
        self.assertEqual(admit_group(group(partial), 2).rewards, (0.25,))

    def test_incomplete_original_group_never_updates(self):
        with self.assertRaises(IncompleteGroup):
            admit_group(group(execution(), expected=2), 2)
        with self.assertRaises(IncompleteGroup):
            admit_group(group(replace(execution(), status="running")), 2)

    def test_conditioning_key_cannot_be_hidden_by_matching_group_id(self):
        first = execution(name="e0")
        second = execution(name="e1", key="same-task:different-effort")
        with self.assertRaises(QuarantinedGroup):
            admit_group(group(first, second), 2)

    def test_path_cap_is_deterministic_and_obeys_role_priority(self):
        main = event("main")
        sub = event("sub", output=(3, 4, 5), role="subagent")
        paths = [TrajectoryPath("zmain", ("main",)), TrajectoryPath("asub", ("sub",), role="subagent")]
        one = admit_group(group(execution([main, sub], paths)), 2, jmax=1).executions[0]
        two = admit_group(group(execution([sub, main], paths[::-1])), 2, jmax=1).executions[0]
        self.assertEqual(one.targets, two.targets)
        self.assertEqual(one.selected_paths, ("zmain",))
        self.assertEqual(one.masked_by_path_cap, 3)

    def test_future_behavior_policy_is_rejected(self):
        record = execution([event("future", versions=(3,))], [TrajectoryPath("main", ("future",))])
        with self.assertRaises(PolicyVersionMismatch):
            admit_group(group(record), 2, max_policy_lag=100)

    def test_missing_behavior_metadata_cannot_be_silently_recomputed(self):
        with self.assertRaises(ValueError):
            GenerationEvent("bad", (1,), (2,), (None,), (2,), ("policy",))
        with self.assertRaises(ValueError):
            GenerationEvent("bad", (1,), (2,), (-0.1,), (None,), ("policy",))

    def test_duplicate_generation_identity_is_rejected(self):
        duplicate = event("duplicate")
        with self.assertRaises(ValueError):
            execution([duplicate, duplicate], [TrajectoryPath("main", ("duplicate",))])

    def test_original_replicate_count_requires_an_integer(self):
        for invalid in (2.0, True, 0):
            with self.subTest(count=invalid):
                with self.assertRaises(ValueError):
                    RolloutGroup("g0", (execution(),), invalid)


if __name__ == "__main__":
    unittest.main()
