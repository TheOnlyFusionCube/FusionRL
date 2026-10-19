"""Checks the actual integration boundary without claiming a GPU training run."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
import unittest

from fusionrl.integrations.verl import ADVANTAGE_NAME, LOSS_NAME
from fusionrl.integrations.verl.contracts import (
    VerlContractError,
    check_source_tree,
    validate_config,
    validate_execution_batch,
    validate_rollout_request,
)

PROJECT = Path(__file__).resolve().parents[1]
HAS_TORCH = importlib.util.find_spec("torch") is not None


def config_fixture():
    return {
        "trainer": {"use_v1": True, "v1": {"trainer_mode": "sync"}},
        "data": {"train_batch_size": 2},
        "algorithm": {
            "adv_estimator": ADVANTAGE_NAME,
            "use_kl_in_reward": False,
            "norm_adv_by_std_in_grpo": False,
            "filter_groups": {"enable": False},
        },
        "actor_rollout_ref": {
            "actor": {
                "strategy": "fsdp2",
                "policy_loss": {"loss_mode": LOSS_NAME},
                "loss_agg_mode": "seq-mean-token-mean",
                "ppo_epochs": 1,
                "ppo_mini_batch_size": 2,
                "entropy_coeff": 0,
                "use_kl_loss": False,
            },
            "rollout": {
                "name": "vllm",
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": -1,
                "do_sample": True,
                "calculate_log_probs": True,
                "multi_turn": {"enable": False},
                "agent": {"default_agent_loop": "single_turn_agent"},
                "over_sample_rate": 0,
                "nnodes": 0,
                "n": 2,
            },
        },
    }


def batch_fixture():
    keys = ["aa_0_0", "aa_1_0", "bb_0_0", "bb_1_0"]
    tags = [dict(status="success", min_global_steps=3, max_global_steps=3) for _ in keys]
    return keys, tags, [{} for _ in keys]


class ConfigContractTests(unittest.TestCase):
    def test_supported_profile(self):
        validate_config(config_fixture())

    def test_async_and_truncated_sampler_rejected(self):
        for section, key, value in [
            ("trainer", "v1", {"trainer_mode": "colocate_async"}),
            ("algorithm", "adv_estimator", "gar"),
        ]:
            config = config_fixture()
            config[section][key] = value
            with self.assertRaises(VerlContractError):
                validate_config(config)
        for key, value in [("top_p", .95), ("temperature", .6), ("top_k", 20), ("calculate_log_probs", False)]:
            config = config_fixture()
            config["actor_rollout_ref"]["rollout"][key] = value
            with self.assertRaises(VerlContractError):
                validate_config(config)

    def test_second_correction_or_prompt_mean_rejected(self):
        config = config_fixture()
        config["algorithm"]["rollout_correction"] = {"rollout_is": "token"}
        with self.assertRaises(VerlContractError):
            validate_config(config)
        config = config_fixture()
        config["actor_rollout_ref"]["actor"]["loss_agg_mode"] = "prompt-mean"
        with self.assertRaises(VerlContractError):
            validate_config(config)

    def test_batch_microbatch_configuration_rejected(self):
        config = config_fixture()
        config["actor_rollout_ref"]["actor"]["ppo_mini_batch_size"] = 1
        with self.assertRaises(VerlContractError):
            validate_config(config)

    def test_actual_dataset_sampler_overrides_are_checked(self):
        params = {"temperature": 1.0, "top_p": 1.0, "top_k": -1, "repetition_penalty": 1.0, "logprobs": True}
        validate_rollout_request("single_turn_agent", params)
        with self.assertRaises(VerlContractError):
            validate_rollout_request("webdev", params)
        with self.assertRaises(VerlContractError):
            validate_rollout_request("single_turn_agent", {**params, "temperature": 0.0})
        with self.assertRaises(VerlContractError):
            validate_rollout_request("single_turn_agent", {**params, "min_p": .1})


class BatchContractTests(unittest.TestCase):
    def check(self, keys, tags, extra):
        return validate_execution_batch(keys, tags, extra, rollout_n=2, prompt_count=2, training_step=4)

    def test_complete_batch(self):
        self.assertEqual(self.check(*batch_fixture()), 4)

    def test_missing_empty_execution_cannot_silently_shrink_denominator(self):
        keys, tags, extra = batch_fixture()
        with self.assertRaises(VerlContractError):
            self.check(keys[:-1], tags[:-1], extra[:-1])

    def test_padding_staleness_multipath_and_infra_rejected(self):
        for kind in ["padding", "stale", "missing_version", "multiple_outputs", "infra", "pending"]:
            keys, tags, extra = batch_fixture()
            if kind == "padding":
                tags[0]["is_padding"] = True
            elif kind == "stale":
                tags[0]["min_global_steps"] = 2
            elif kind == "missing_version":
                tags[0].pop("max_global_steps")
            elif kind == "multiple_outputs":
                keys[1] = "aa_0_1"
            elif kind == "infra":
                extra[0]["is_infra"] = 1
            elif kind == "pending":
                extra[0]["webdev_group_pending"] = True
            with self.subTest(kind=kind), self.assertRaises(VerlContractError):
                self.check(keys, tags, extra)


@unittest.skipUnless(HAS_TORCH, "requires the optional Torch execution backend")
class LossAdapterTests(unittest.TestCase):
    def test_binary_advantage_and_nonbinary_failure(self):
        import numpy as np
        import torch
        from fusionrl.integrations.verl.plugin import binary_advantage

        rewards = torch.tensor([[0., 1.], [0., 0.], [0., 0.], [0., 1.]])
        mask = torch.tensor([[1, 1], [1, 0], [0, 0], [1, 1]])
        adv, _ = binary_advantage(rewards, mask, np.array(["a", "a", "b", "b"]))
        expected = torch.tensor([[.5, .5], [-.5, 0.], [0., 0.], [.5, .5]])
        torch.testing.assert_close(adv, expected)
        with self.assertRaises(VerlContractError):
            binary_advantage(rewards * .5, mask, np.array(["a", "a", "b", "b"]))

    def test_policy_loss_matches_core_and_global_microbatch_sum(self):
        import torch
        from fusionrl.integrations.verl.plugin import policy_loss
        from fusionrl.objective import execution_loss

        old = torch.full((4, 3), -2., dtype=torch.float64)
        delta = torch.tensor([[.001, -.002, .003], [.004, -.001, 0.], [0., 0., 0.], [-.003, .002, .001]], dtype=torch.float64)
        new = (old + delta).requires_grad_()
        mask = torch.tensor([[1, 1, 1], [1, 0, 0], [0, 0, 0], [1, 1, 0]], dtype=torch.bool)
        scalar_adv = torch.tensor([.5, -.5, -.5, .5], dtype=torch.float64)
        adv = scalar_adv[:, None] * mask
        config = SimpleNamespace(ppo_mini_batch_size=2, rollout_n=2, global_batch_info={"dp_size": 1, "global_batch_size": 4})
        actual, _ = policy_loss(old, new, adv, mask, config=config)
        expected, _ = execution_loss(new, old, scalar_adv, mask, global_execution_count=4)
        torch.testing.assert_close(actual, expected)
        whole_grad = torch.autograd.grad(actual, new, retain_graph=True)[0]
        pieces = [policy_loss(old[i:i+1], new[i:i+1], adv[i:i+1], mask[i:i+1], config=config)[0] for i in range(4)]
        split = sum(pieces)
        torch.testing.assert_close(split, actual)
        torch.testing.assert_close(torch.autograd.grad(split, new)[0], whole_grad)
        config.global_batch_info["dp_size"] = 2
        ddp, _ = policy_loss(old, new, adv, mask, config=config)
        torch.testing.assert_close(ddp / 2, expected)

    def test_extra_weights_rejected(self):
        import torch
        from fusionrl.integrations.verl.plugin import policy_loss

        data = torch.ones((2, 2))
        config = SimpleNamespace(ppo_mini_batch_size=1, rollout_n=2, global_batch_info={"dp_size": 1, "global_batch_size": 2})
        with self.assertRaises(VerlContractError):
            policy_loss(data, data, data, data.bool(), config=config, rollout_is_weights=data)


class PinnedSourceTests(unittest.TestCase):
    def test_manifest_has_expected_api_and_exact_hashes(self):
        manifest = json.loads((PROJECT / "fusionrl/integrations/verl/source_contract.json").read_text())
        core = manifest["files"]["verl/trainer/ppo/core_algos.py"]
        self.assertEqual(core["functions"]["compute_policy_loss_vanilla"][-1], "rollout_is_weights")
        self.assertEqual(core["functions"]["register_adv_est"], ["name_or_enum"])
        for contract in manifest["files"].values():
            self.assertEqual(len(contract["original_sha256"]), 64)
            self.assertEqual(len(contract["patched_sha256"]), 64)

    @unittest.skipUnless(os.getenv("FUSIONRL_VERL_SOURCE"), "set FUSIONRL_VERL_SOURCE to verify against the real pinned source checkout")
    def test_actual_source_signatures_and_patch_application(self):
        source = Path(os.environ["FUSIONRL_VERL_SOURCE"])
        checked = check_source_tree(source, patched=False)
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            for relative in checked:
                output = target / relative
                output.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source / relative, output)
            patch = PROJECT / "patches/verl-fusionrl-strict-sync.patch"
            subprocess.run(["git", "apply", "--check", str(patch)], cwd=target, check=True, capture_output=True)
            subprocess.run(["git", "apply", str(patch)], cwd=target, check=True, capture_output=True)
            self.assertEqual(set(check_source_tree(target, patched=True, check_git=False)), set(checked))

    @unittest.skipUnless(os.getenv("FUSIONRL_VERL_SOURCE") and importlib.util.find_spec("hydra"), "requires the original pinned source and hydra-core for configuration composition")
    def test_real_hydra_defaults_compose_and_validate(self):
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf

        source = Path(os.environ["FUSIONRL_VERL_SOURCE"]) / "verl/trainer/config"
        profile = PROJECT / "fusionrl/integrations/verl/config/fusionrl_binary.yaml"
        with tempfile.TemporaryDirectory() as directory:
            config_root = Path(directory) / "config"
            shutil.copytree(source, config_root)
            # Only substitute the search-path URI: loading the identical YAML
            # tree by file avoids importing the unrelated Ray/GPU package.
            text = profile.read_text().replace("pkg://verl.trainer.config", "file://" + str(config_root))
            (config_root / profile.name).write_text(text)
            with initialize_config_dir(version_base=None, config_dir=str(config_root)):
                config = compose(config_name="fusionrl_binary", overrides=[
                    "actor_rollout_ref.model.path=/models/validated-model",
                    "data.train_files=/data/train.parquet",
                    "data.val_files=/data/validation.parquet",
                    "reward.custom_reward_function.path=/verifiers/binary.py",
                ])
                OmegaConf.resolve(config)
                validate_config(config)
                self.assertEqual(config.actor_rollout_ref.actor._target_, "verl.workers.config.FSDPActorConfig")
                self.assertTrue(config.transfer_queue.enable)


if __name__ == "__main__":
    unittest.main()
