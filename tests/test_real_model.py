"""Real Hugging Face causal-LM path, on a tiny randomly initialized Llama (CPU, offline)."""
import dataclasses
import json
import os
from pathlib import Path
import socket
import tempfile
import unittest

import numpy as np

try:
    import torch
    import transformers  # noqa: F401
    from tokenizers import Tokenizer, models, pre_tokenizers
except ImportError:
    torch = None

WORDS = ["<pad>", "<s>", "</s>", "<unk>", "user", ":", "assistant", "system", "What", "is", "plus", "?", "\n"] + \
    [str(i) for i in range(20)]


def make_fixture(root: Path, *, steps=4, every=2, extra=None):
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
    tok = Tokenizer(models.WordLevel({w: i for i, w in enumerate(WORDS)}, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="<pad>", bos_token="<s>", eos_token="</s>",
                                        unk_token="<unk>")
    tokenizer.chat_template = ("{% for m in messages %}{{ m['role'] }} : {{ m['content'] }} \n {% endfor %}"
                               "{% if add_generation_prompt %}assistant : {% endif %}")
    torch.manual_seed(0)
    model = LlamaForCausalLM(LlamaConfig(vocab_size=len(WORDS), hidden_size=32, intermediate_size=64,
                                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                                         max_position_embeddings=128, pad_token_id=0, bos_token_id=1, eos_token_id=2))
    model.save_pretrained(root / "model")
    tokenizer.save_pretrained(root / "model")
    with (root / "train.jsonl").open("w") as handle:
        for a in range(4):
            for b in range(4):
                handle.write(json.dumps({"task_id": f"add-{a}-{b}", "prompt": f"What is {a} plus {b} ?",
                                         "answer": str(a + b)}) + "\n")
    config = {"model": {"path": str(root / "model"), "device": "cpu"},
              "data": {"train": str(root / "train.jsonl"), "verifier": "math_answer"},
              "schedule": {"groups_per_update": 2, "executions_per_group": 4},
              "optim": {"learning_rate": 0.02},
              "rollout": {"max_new_tokens": 3, "generation_batch_size": 5},
              "learner": {"max_tokens_per_microbatch": 40},
              "checkpoint": {"every": every}, "run": {"steps": steps, "seed": 3}}
    for key, value in (extra or {}).items():
        config.setdefault(key, {}).update(value)
    (root / "run.json").write_text(json.dumps(config))
    return model, tokenizer


def final_weights(output: Path):
    from safetensors.torch import load_file
    latest = (output / "checkpoints" / "latest").read_text().strip()
    return load_file(str(output / "checkpoints" / latest / "model" / "model.safetensors"))


def _ddp_worker(rank, world, port, root, output):
    os.environ.update({"RANK": str(rank), "WORLD_SIZE": str(world), "LOCAL_RANK": str(rank),
                       "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": str(port)})
    torch.set_num_threads(1)
    from fusionrl.run_config import load_run_config
    from fusionrl.runner import run_training
    run_training(load_run_config(Path(root) / "run.json"), output, progress=False)
    torch.distributed.destroy_process_group()


@unittest.skipIf(torch is None, "requires torch, transformers and tokenizers")
class HFBackendTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.model, self.tokenizer = make_fixture(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def policy(self, **kw):
        from fusionrl.backends.hf import HFPolicy
        return HFPolicy(self.model, self.tokenizer, **kw)

    def test_recorded_logprobs_match_an_independent_forward(self):
        policy = self.policy(max_tokens_per_microbatch=16)
        prompt = policy.encode_chat([{"role": "user", "content": "What is 1 plus 2 ?"}])
        prompts = [prompt, prompt[:4], prompt[2:]]
        samples = policy.generate(prompts, max_new_tokens=6, generator=torch.Generator().manual_seed(0))
        for p, (tokens, logps) in zip(prompts, samples):
            with torch.no_grad():
                reference = torch.log_softmax(self.model(torch.tensor([list(p) + list(tokens)])).logits[0], -1)
            expected = [reference[len(p) + k - 1, t].item() for k, t in enumerate(tokens)]
            np.testing.assert_allclose(logps, expected, atol=2e-6)
        report = policy.rollout_parity(prompts, max_new_tokens=6, generator=torch.Generator().manual_seed(1))
        self.assertTrue(report.within(1e-5, 1e-4), report)

    def test_selective_scoring_matches_full_logits_and_is_differentiable(self):
        selective, full = self.policy(scoring="selective"), self.policy(scoring="full")
        self.assertLess(selective.check_head_consistency(), 1e-5)
        sequences = [tuple(range(4, 15)), tuple(range(5, 9)), (4, 5)]
        positions = [[3, 7, 10], [1, 3], [1]]
        a = selective.sequence_log_probs(sequences, positions)
        b = full.sequence_log_probs(sequences, positions)
        torch.testing.assert_close(a, b, atol=1e-5, rtol=0)
        a.sum().backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in self.model.parameters()))

    def test_gumbel_sampling_matches_the_softmax(self):
        policy = self.policy()
        prompt = policy.encode_chat([{"role": "user", "content": "What is 3 plus 3 ?"}])
        with torch.no_grad():
            probs = torch.softmax(self.model(torch.tensor([list(prompt)])).logits[0, -1].double(), -1)
        draws = policy.generate([prompt] * 4000, max_new_tokens=1, generator=torch.Generator().manual_seed(2))
        counts = np.bincount([t[0] for t, _ in draws], minlength=len(WORDS)) / 4000
        self.assertLess(np.abs(counts - probs.numpy()).max(), 0.03)
        # A row's sample does not depend on the other rows in its batch.
        gens = lambda: [torch.Generator().manual_seed(s) for s in (10, 11, 12)]  # noqa: E731
        together = policy.generate([prompt, prompt[:3], prompt], max_new_tokens=4, generator=gens())
        alone = policy.generate([prompt], max_new_tokens=4, generator=gens()[2:])
        self.assertEqual(together[2][0], alone[0][0])

    def test_dropout_is_disabled_for_scoring(self):
        from transformers import GPT2Config, GPT2LMHeadModel
        from fusionrl.backends.hf import HFPolicy, disable_dropout
        torch.manual_seed(0)
        model = GPT2LMHeadModel(GPT2Config(vocab_size=len(WORDS), n_embd=32, n_layer=2, n_head=4, n_positions=64,
                                           bos_token_id=1, eos_token_id=2, pad_token_id=0)).train()
        policy = HFPolicy(model, self.tokenizer)
        prompt = policy.encode_chat([{"role": "user", "content": "What is 1 plus 2 ?"}])
        noisy = policy.rollout_parity([prompt] * 4, max_new_tokens=6, generator=torch.Generator().manual_seed(0))
        self.assertGreater(noisy.mean_abs_gap, 1e-3)  # Dropout makes scoring a different network.
        self.assertGreater(disable_dropout(model), 0)
        clean = policy.rollout_parity([prompt] * 4, max_new_tokens=6, generator=torch.Generator().manual_seed(0))
        self.assertLess(clean.max_abs_gap, 1e-5)
        self.assertTrue(model.training)

    def test_tokenizer_fingerprint_covers_mapping_and_template(self):
        from fusionrl.backends.hf import tokenizer_fingerprint
        base = tokenizer_fingerprint(self.tokenizer)
        self.tokenizer.chat_template = "{{ messages[0]['content'] }}"
        self.assertNotEqual(base, tokenizer_fingerprint(self.tokenizer))

    def test_sequence_and_context_learner_paths_agree(self):
        from fusionrl.learner import learner_loss
        from fusionrl.pipeline import GroupSupervision, prepare_step
        from fusionrl.trajectory import (Assessment, ExecutionRecord, GenerationEvent, RolloutGroup,
                                         TrajectoryPath, admit_group)
        policy = self.policy(max_tokens_per_microbatch=12)
        prompt = policy.encode_chat([{"role": "user", "content": "What is 2 plus 2 ?"}])
        samples = policy.generate([prompt] * 4, max_new_tokens=4, generator=torch.Generator().manual_seed(4))
        executions = []
        for i, (tokens, logps) in enumerate(samples):
            event = GenerationEvent(f"e{i}:r", prompt, tokens, logps, (0,) * len(tokens), ("policy",) * len(tokens))
            executions.append(ExecutionRecord(f"e{i}", "g", (event,), (TrajectoryPath("m", (event.event_id,)),),
                                              Assessment(float(i % 2), True, f"a{i}"), conditioning_key="k"))
        group = admit_group(RolloutGroup("g", tuple(executions), 4, conditioning_key="k"), 0)
        step = prepare_step([group], {"g": GroupSupervision(tuple(e.execution_id for e in group.executions),
                                                            tuple(e.assessment.artifact_id for e in group.executions),
                                                            "k")})

        def context_fn(contexts, token_ids):
            return torch.cat([policy.sequence_log_probs([tuple(c) + (t,)], [[len(c)]])
                              for c, t in zip(contexts, token_ids)])

        loss_a, _ = learner_loss(step.batch, sequence_log_prob_fn=policy.sequence_log_probs)
        grads_a = torch.autograd.grad(loss_a, list(self.model.parameters()), allow_unused=True)
        loss_b, _ = learner_loss(step.batch, context_fn)
        grads_b = torch.autograd.grad(loss_b, list(self.model.parameters()), allow_unused=True)
        self.assertAlmostEqual(loss_a.item(), loss_b.item(), places=6)
        for a, b in zip(grads_a, grads_b):
            if a is not None:
                torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)


@unittest.skipIf(torch is None, "requires torch, transformers and tokenizers")
class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def config(self, **overrides):
        from fusionrl.run_config import load_run_config
        cfg = load_run_config(self.root / "run.json")
        return dataclasses.replace(cfg, **overrides)

    def test_resume_reproduces_an_uninterrupted_run(self):
        from fusionrl.runner import run_training
        make_fixture(self.root, steps=4, every=2)
        cfg = self.config()
        run_training(cfg, self.root / "full", progress=False)
        half = dataclasses.replace(cfg, run=dataclasses.replace(cfg.run, steps=2))
        run_training(half, self.root / "split", progress=False)
        summary = run_training(cfg, self.root / "split", resume=True, progress=False)
        self.assertTrue(summary["resumed"])
        a, b = final_weights(self.root / "full"), final_weights(self.root / "split")
        for key in a:
            torch.testing.assert_close(a[key], b[key], atol=0, rtol=0)
        full = (self.root / "full" / "metrics.jsonl").read_text().splitlines()
        split = (self.root / "split" / "metrics.jsonl").read_text().splitlines()
        strip = lambda rows: [{k: v for k, v in json.loads(r).items() if k != "step_seconds"} for r in rows]  # noqa: E731
        self.assertEqual(strip(full), strip(split))
        with self.assertRaises(FileExistsError):
            run_training(cfg, self.root / "full", progress=False)

    def test_two_data_parallel_ranks_match_one_process(self):
        import torch.multiprocessing as mp
        from fusionrl.runner import run_training
        make_fixture(self.root, steps=3, every=3)
        run_training(self.config(), self.root / "single", progress=False)
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        mp.spawn(_ddp_worker, args=(2, port, str(self.root), str(self.root / "ddp")), nprocs=2, join=True)
        self.assertEqual(json.loads((self.root / "ddp" / "summary.json").read_text())["world_size"], 2)
        a, b = final_weights(self.root / "single"), final_weights(self.root / "ddp")
        for key in a:
            torch.testing.assert_close(a[key], b[key], atol=2e-6, rtol=1e-5)

    def test_parity_and_integrity_guards(self):
        from fusionrl.runner import IntegrityHalt, ParityError, Runner
        make_fixture(self.root, steps=2, every=2, extra={"parity": {"max_mean_abs_gap": 1e-3}})
        runner = Runner(self.config(), self.root / "parity", progress=False)
        real = runner.policy.generate

        def skewed(*args, **kwargs):  # Simulates a rollout engine with a different sampler/precision.
            return [(t, tuple(x - 0.3 for x in lp)) for t, lp in real(*args, **kwargs)]
        runner.policy.generate = skewed
        with self.assertRaises(ParityError):
            runner.run()

        (self.root / "hack.py").write_text("def verify(task, response):\n    return {'score': 1.0, 'hack': True}\n")
        import sys
        sys.path.insert(0, str(self.root))
        try:
            make_fixture(self.root, steps=2, every=2, extra={
                "data": {"verifier": "hack:verify"},
                "integrity": {"halt_rate": 0.5, "halt_window": 8}})
            with self.assertRaises(IntegrityHalt):
                Runner(self.config(), self.root / "halt", progress=False).run()
        finally:
            sys.path.remove(str(self.root))

    def test_evaluator_failures_are_retried_not_trained(self):
        import sys
        from fusionrl.runner import run_training
        (self.root / "flaky.py").write_text(
            "calls = {'n': 0}\n"
            "def verify(task, response):\n"
            "    calls['n'] += 1\n"
            "    if calls['n'] == 3:\n"
            "        raise RuntimeError('sandbox crashed')\n"
            "    return 1.0 if task['answer'] in response else 0.0\n")
        sys.path.insert(0, str(self.root))
        try:
            make_fixture(self.root, steps=2, every=2, extra={"data": {"verifier": "flaky:verify"}})
            summary = run_training(self.config(), self.root / "flaky", progress=False)
        finally:
            sys.path.remove(str(self.root))
        self.assertEqual(summary["steps"], 2)
        self.assertEqual(summary["quarantined_groups_retried"], 1)


if __name__ == "__main__":
    unittest.main()
