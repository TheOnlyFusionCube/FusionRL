"""Independent invariants and hand-calculated examples for reward routing."""

import unittest

import numpy as np

from fusionrl.rewards import (
    choose_cost_penalties,
    compute_advantages,
    deepseek_effort_penalties,
    mimo_length_penalties,
)


class RewardTests(unittest.TestCase):
    def test_explicit_scalar_route_centers_without_thresholding(self):
        result = compute_advantages([0.2, 0.5, 0.8], route="scalar")
        np.testing.assert_allclose(result.advantages, [-0.3, 0, 0.3], atol=1e-15)
        np.testing.assert_allclose(result.effective_rewards, [0.2, 0.5, 0.8])
        self.assertIsNone(result.diagnostics["corrected_pass_rate"])

    def test_scalar_hacks_and_costs_are_applied_before_final_centering(self):
        result = compute_advantages([0.2, 0.5, 0.8], route="scalar",
                                    confirmed_hacks=[False, False, True],
                                    cost_penalties=[0, -0.1, -0.2])
        expected = np.array([0.2, 0.4, -0.2])
        np.testing.assert_allclose(result.corrected_outcomes, [0.2, 0.5, 0])
        np.testing.assert_allclose(result.advantages, expected - expected.mean())

    def test_scalar_range_and_grading_conflicts_are_rejected(self):
        for rewards in ([-0.1, 0.5], [0.2, 1.1], [0.2, float("nan")]):
            with self.subTest(rewards=rewards), self.assertRaises(ValueError):
                compute_advantages(rewards, route="scalar")
        for kwargs in ({"solution_scores": [1, 1]}, {"behavior_scores": [1, 1]},
                       {"quality_factors": [1, 1]}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                compute_advantages([0.2, 0.8], route="scalar", **kwargs)
        with self.assertRaises(ValueError):
            compute_advantages([0.2, 0.8], route="binary")

    def test_hacks_change_all_group_statistics_before_grading(self):
        result = compute_advantages([1, 1, 0, 0], confirmed_hacks=[True, False, False, False])
        np.testing.assert_allclose(result.corrected_outcomes, [0, 1, 0, 0])
        np.testing.assert_allclose(result.advantages, [-0.25, 0.75, -0.25, -0.25])
        self.assertEqual(result.diagnostics["confirmed_hack_count"], 1)

    def test_grs_preserves_failure_gate_and_all_pass_learning(self):
        result = compute_advantages([1, 1, 1], route="grs", solution_scores=[1, 0.8, 0.2],
                                    behavior_scores=[1, 0.5, 1])
        np.testing.assert_allclose(result.effective_rewards, [1, 0.4, 0.2])
        self.assertTrue(result.diagnostics["has_learning_signal"])
        result = compute_advantages([0, 1], route="grs", solution_scores=[1, 1],
                                    behavior_scores=[1, 1], confirmed_hacks=[False, True])
        np.testing.assert_array_equal(result.advantages, [0, 0])

    def test_uncapped_gar_redistributes_positive_mass_and_leaves_failures(self):
        result = compute_advantages([1, 1, 0, 0], route="gar", quality_factors=[1, 0.5, None, None])
        np.testing.assert_allclose(result.advantages, [2 / 3, 1 / 3, -0.5, -0.5])
        self.assertAlmostEqual(result.advantages[:2].sum(), 1.0)
        self.assertAlmostEqual(result.advantages.sum(), 0.0)
        self.assertFalse(result.diagnostics["gar_cap_active"])

    def test_capped_gar_recenters_both_successes_and_failures(self):
        result = compute_advantages([1, 1, 0, 0], route="gar",
                                    quality_factors=[0.1, 0.2, None, None], gar_cap=2)
        # Raw capped values are [0.1,0.2,-0.5,-0.5], mean -0.175.
        np.testing.assert_allclose(result.advantages, [0.275, 0.375, -0.325, -0.325])
        self.assertTrue(result.diagnostics["gar_cap_active"])

    def test_unusable_gar_output_falls_back_to_corrected_binary(self):
        for factors in (None, [0, 1, 1], [float("nan"), 1, 1], [1], ["bad", 1, 1]):
            with self.subTest(factors=factors):
                result = compute_advantages([1, 1, 0], route="gar", quality_factors=factors)
                np.testing.assert_allclose(result.advantages, [1 / 3, 1 / 3, -2 / 3])
                self.assertIsNotNone(result.diagnostics["gar_fallback"])
        for outcomes in ([1, 1], [0, 0]):
            result = compute_advantages(outcomes, route="gar", quality_factors=[1, 0.5])
            np.testing.assert_array_equal(result.advantages, [0, 0])

    def test_route_conflicts_are_executable_errors(self):
        for kwargs in (
            {"route": "gar", "cost_penalties": [0, -0.1]},
            {"route": "gar", "solution_scores": [1, 1]},
            {"route": "binary", "quality_factors": [1, 1]},
            {"route": "grs", "solution_scores": [1, 1], "behavior_scores": [1, 1], "quality_factors": [1, 1]},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                compute_advantages([1, 0], **kwargs)
        with self.assertRaises(ValueError):
            compute_advantages([1, 0.25], route="gar")
        with self.assertRaises(ValueError):
            choose_cost_penalties(deepseek=[0, 0], mimo=[0, 0])

    def test_cost_centering_equals_centering_combined_rewards(self):
        result = compute_advantages([1, 0, 1], cost_penalties=[-0.2, -0.1, 0])
        shaped = np.array([0.8, -0.1, 1])
        np.testing.assert_allclose(result.advantages, shaped - shaped.mean())
        with self.assertRaises(ValueError):
            compute_advantages([1, 0], cost_penalties=[0, 0.1])

    def test_effort_decay_and_cap(self):
        kwargs = dict(min_budget=1, k0=0.2, tau=2, length_norm=100, cap=0.3)
        low = deepseek_effort_penalties([0, 100, 200], budget=1, **kwargs)
        high = deepseek_effort_penalties([0, 100, 200], budget=3, **kwargs)
        np.testing.assert_allclose(low, [0, -0.2, -0.3])
        np.testing.assert_allclose(high, [0, -0.2 / np.e, -0.4 / np.e])
        self.assertTrue(np.all(high >= low))

    def test_length_penalty_only_successes_and_strict_gate(self):
        kwargs = dict(min_pass_rate=0.5, percentile=50, max_deduction=0.2,
                      tolerance=0, saturation=1, exponent=1)
        # All successes have median 150; 300 tokens reaches saturation.
        penalties = mimo_length_penalties([100, 150, 300, 999], [1, 1, 1, 0], **kwargs)
        np.testing.assert_allclose(penalties, [0, 0, -0.2, 0])
        gated = mimo_length_penalties([100, 1000, 5000, 9000], [1, 1, 0, 0], **kwargs)
        np.testing.assert_array_equal(gated, [0, 0, 0, 0])

    def test_finite_nonempty_and_boolean_contracts(self):
        for bad in ([], [0, float("nan")], [[0, 1]], [0, 2]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                compute_advantages(bad)
        with self.assertRaises(ValueError):
            compute_advantages([1, 0], confirmed_hacks=["false", "true"])

    def test_tiny_gar_factors_cannot_create_nonfinite_diagnostics(self):
        result = compute_advantages([1, 1, 0], route="gar", quality_factors=[1e-320, 1e-320, None])
        self.assertEqual(result.diagnostics["gar_fallback"], "numerically_unsafe_grader_output")
        self.assertTrue(np.isfinite(result.advantages).all())
        self.assertIsNone(result.diagnostics["gar_multiplier_uncapped"])


if __name__ == "__main__":
    unittest.main()
