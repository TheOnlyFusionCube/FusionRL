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

    def test_batched_sampling_records_the_drawn_probabilities(self):
        from fusionrl.tiny_policy import TinyCausalPolicy
        torch.manual_seed(3)
        model=TinyCausalPolicy().eval()
        prompts=[model.prompt(s,b) for s in range(8) for b in (25,100)]*4
        samples=model.sample_batch(prompts,max_tokens=4,generator=torch.Generator().manual_seed(5))
        self.assertEqual(len(samples),len(prompts))
        self.assertTrue(any(len(tokens)>1 for tokens,_ in samples))
        for prompt,(tokens,logps) in zip(prompts,samples):
            self.assertTrue(1<=len(tokens)<=4)
            self.assertTrue(all(t>=8 for t in tokens[:-1]))  # Stops at the first answer symbol.
            with torch.no_grad():
                expected=model.log_probs([prompt+tokens[:i] for i in range(len(tokens))],list(tokens))
            np.testing.assert_allclose(expected,logps,atol=5e-7,rtol=0)
        # Greedy evaluation via batched decoding matches a step-by-step argmax loop.
        successes=0
        for prompt,answer in [(model.prompt(s,b),s) for s in range(8) for b in (25,100)]:
            context=prompt
            for _ in range(4):
                token=int(model.logits([context]).argmax(dim=-1).item())
                context=context+(token,)
                if token<8:
                    successes+=int(token==answer)
                    break
        self.assertAlmostEqual(model.evaluate((25,100),4)["greedy_completion_accuracy"],successes/16)

    def test_seed_sweep_reports_spread(self):
        from fusionrl.sweep import run_sweep
        cfg=config_from_dict({"schedule":{"groups_per_update":1,"executions_per_group":4},"train":{"steps":2}})
        with tempfile.TemporaryDirectory() as tmp:
            report=run_sweep(cfg,[1,2],tmp,progress=False)
            self.assertEqual(report["seeds"],[1,2])
            stats=report["aggregate"]["final"]["first_token_correct_probability"]
            values=[run["final"]["first_token_correct_probability"] for run in report["runs"]]
            self.assertAlmostEqual(stats["mean"],sum(values)/2)
            self.assertLessEqual(stats["min"],stats["max"])
            self.assertTrue((Path(tmp)/"seed-2"/"summary.json").exists())
            with self.assertRaises(FileExistsError):
                run_sweep(cfg,[3,4],tmp,progress=False)
        with self.assertRaises(ValueError):
            run_sweep(cfg,[1],"unused",progress=False)

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

