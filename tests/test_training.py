import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from fusionrl.config import config_from_dict

try:
    import torch
except ImportError:
    torch=None


@unittest.skipIf(torch is None,"PyTorch extra is not installed")
class EndToEndTests(unittest.TestCase):
    def test_recorded_behavior_is_initial_sampler_not_updated_model(self):
        from fusionrl.trainer import run_demo
        from fusionrl.tiny_policy import TinyCausalPolicy
        cfg=config_from_dict({"schedule":{"groups_per_update":2,"executions_per_group":4},"train":{"steps":4,"seed":11}})
        with tempfile.TemporaryDirectory() as tmp:
            summary=run_demo(cfg,tmp,progress=False)
            trace=json.loads((Path(tmp)/"rollout_example.json").read_text())
            torch.manual_seed(11)
            initial=TinyCausalPolicy().eval()
            for execution in trace["group"]["executions"]:
                event=execution["events"][0]
                context=tuple(event["input_token_ids"])
                outputs=tuple(event["output_token_ids"])
                with torch.no_grad():
                    expected=initial.log_probs([context+outputs[:i] for i in range(len(outputs))],list(outputs))
                np.testing.assert_allclose(expected,event["behavior_log_probs"],atol=5e-7,rtol=0)
            checkpoint=torch.load(Path(tmp)/"checkpoint.pt",weights_only=True)
            self.assertTrue(any(not torch.equal(initial.state_dict()[k],v) for k,v in checkpoint["state_dict"].items()))
            self.assertEqual(summary["observed_max_admitted_token_lag"],0)
            with self.assertRaises(FileExistsError):
                run_demo(cfg,tmp,progress=False)

    def test_async_uses_delayed_versions_and_finishes_at_clean_stage(self):
        from fusionrl.trainer import run_demo
        cfg=config_from_dict({"schedule":{"mode":"bounded_async","groups_per_update":2,"executions_per_group":4,
            "extra_dispatch_groups":2,"max_policy_lag":1},"train":{"steps":4,"opd_steps":2}})
        with tempfile.TemporaryDirectory() as tmp:
            summary=run_demo(cfg,tmp,progress=False)
            self.assertEqual(summary["observed_max_admitted_token_lag"],1)
            self.assertEqual(summary["discarded_pending_groups_at_stage_boundary"],2)
            self.assertEqual(len(summary["opd_losses"]),2)
            self.assertTrue(np.isfinite(summary["opd_losses"]).all())

    def test_small_configured_grading_stages_execute(self):
        from fusionrl.trainer import run_demo
        for route,length_mode in [("gar","none"),("grs","effort"),("scalar","none"),("binary","group_relative")]:
            cfg=config_from_dict({"schedule":{"groups_per_update":1,"executions_per_group":4},
                "reward":{"route":route,"length_mode":length_mode},"train":{"steps":2,"sft_steps":1}})
            with self.subTest(route=route),tempfile.TemporaryDirectory() as tmp:
                summary=run_demo(cfg,tmp,progress=False)
                self.assertEqual(summary["sft_steps"],1)
                self.assertEqual(summary["rl_steps"],2)


if __name__=="__main__":
    unittest.main()

