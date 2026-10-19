import unittest

from fusionrl.config import config_from_dict


class ConfigurationTests(unittest.TestCase):
    def test_defaults_valid(self):
        config_from_dict({})

    def test_incompatible_combinations_rejected(self):
        cases = [
            {"reward":{"route":"gar","length_mode":"effort"}},
            {"reward":{"route":"gar","length_mode":"group_relative"}},
            {"schedule":{"mode":"strict_sync","max_policy_lag":1}},
            {"schedule":{"mode":"strict_sync","burst_steps":2}},
            {"schedule":{"sample_reuse":True}},
            {"schedule":{"dispatch_policy":"continuous"}},
            {"policy":{"top_p":.9}},
            {"policy":{"model_kind":"moe"}},
            {"policy":{"cache_policy":"reuse_across_checkpoints"}},
            {"nonexistent_option":True},
            {"schedule":{"staleness":2}},
            {"schema_version":True},
            {"policy":{"model_kind":"moe","freeze_moe_router":"false","recorded_routing_replay":"false"}},
            {"reward":{"length_exponent":float("nan")}},
        ]
        for case in cases:
            with self.subTest(case=case),self.assertRaises(ValueError):
                config_from_dict(case)

    def test_json_types_are_enforced(self):
        cases = [
            {"train":{"learning_rate":True}},
            {"train":{"learning_rate":"0.01"}},
            {"train":{"steps":2.0}},
            {"reward":{"gar_cap":True}},
            {"reward":{"effort_levels":"25"}},
            {"reward":{"effort_levels":[25.0,50]}},
            {"loss":{"clip_high":"1.01"}},
            {"policy":{"temperature":True}},
            {"schedule":{"mode":["strict_sync"]}},
        ]
        for case in cases:
            with self.subTest(case=case),self.assertRaises(ValueError):
                config_from_dict(case)
        # Integer literals remain valid for float fields and are stored as floats.
        config = config_from_dict({"train":{"learning_rate":1},"reward":{"effort_levels":[25,50]}})
        self.assertIsInstance(config.train.learning_rate, float)
        self.assertEqual(config.reward.effort_levels, (25, 50))

    def test_guarded_async_profile(self):
        config_from_dict({"schedule":{"mode":"bounded_async","max_policy_lag":2,"extra_dispatch_groups":4}})


if __name__ == "__main__":
    unittest.main()

