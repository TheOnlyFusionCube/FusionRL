"""Hugging Face causal-LM backend for the portable FusionRL learner.

Three properties matter for correct RL on a real model:

1. **Exact behavior probabilities.** ``generate`` samples from the full
   temperature-1 softmax of the model's raw logits with its own loop, so the
   recorded log probability is the one the token was drawn from. It ignores a
   checkpoint's ``generation_config`` (which often sets top-p/top-k or a
   temperature) instead of silently inheriting a truncated sampler.
2. **Memory-bounded scoring.** ``sequence_log_probs`` runs each shared event
   sequence once, packs sequences into microbatches by a token budget, and
   applies the LM head only at target positions. A full ``[tokens, vocab]``
   logit tensor (32k x 150k floats is ~19 GB) is never materialized.
3. **Checked numerics.** ``check_head_consistency`` compares the selective
   head with the model's own full forward (catching logit scaling or
   soft-capping the selective path would miss) and ``rollout_parity`` measures
   generation-versus-training log-probability gaps before a costly run.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Sequence


def disable_dropout(model: Any) -> int:
    """Zero every dropout so training-mode scoring equals the sampling network.

    Train mode is kept (gradient checkpointing needs it), but RL must score
    tokens with the same deterministic network that sampled them. Returns the
    number of settings changed.
    """
    import torch

    changed = 0
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout) and module.p:
            module.p = 0.0
            changed += 1
        for name in ("attention_dropout", "dropout", "attn_dropout", "resid_dropout", "hidden_dropout"):
            value = getattr(module, name, None)
            if isinstance(value, float) and value:
                setattr(module, name, 0.0)
                changed += 1
    for name in dir(model.config):
        if name.endswith(("dropout", "pdrop", "dropout_prob", "dropout_rate")):
            value = getattr(model.config, name, None)
            if isinstance(value, float) and value:
                setattr(model.config, name, 0.0)
                changed += 1
    return changed


def tokenizer_fingerprint(tokenizer: Any) -> str:
    """Hash the token-ID mapping, special tokens, normalization and chat template.

    Equal vocabulary size is not equivalence. Two tokenizers with this same
    fingerprint map text and chat turns to identical IDs.
    """
    vocab = tokenizer.get_vocab()
    payload: dict[str, Any] = {
        "vocab": sorted(vocab.items(), key=lambda item: item[1]),
        "special": {k: str(v) for k, v in sorted(getattr(tokenizer, "special_tokens_map", {}).items())},
        "added": sorted((str(k), int(v)) for k, v in getattr(tokenizer, "get_added_vocab", dict)().items()),
        "chat_template": getattr(tokenizer, "chat_template", None),
        "class": type(tokenizer).__name__,
    }
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is not None:
        state = json.loads(backend.to_str())
        payload["pipeline"] = {k: state.get(k) for k in ("normalizer", "pre_tokenizer", "post_processor", "decoder")}
        model = state.get("model", {})
        payload["model_type"] = model.get("type")
        if "merges" in model:
            payload["merges"] = hashlib.sha256(json.dumps(model["merges"]).encode()).hexdigest()
    encoded = json.dumps(payload, sort_keys=True, default=str).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ParityReport:
    tokens: int
    mean_abs_gap: float
    max_abs_gap: float
    mean_signed_gap: float

    def within(self, mean_abs: float, max_abs: float) -> bool:
        return self.mean_abs_gap <= mean_abs and self.max_abs_gap <= max_abs


class HFPolicy:
    """Wrap ``AutoModelForCausalLM`` + tokenizer for FusionRL rollouts and scoring."""

    def __init__(self, model: Any, tokenizer: Any, *, max_tokens_per_microbatch: int = 16384,
                 scoring: str = "auto", autocast_dtype: Any = None):
        import torch

        if scoring not in {"auto", "selective", "full"}:
            raise ValueError("scoring must be auto, selective or full")
        if max_tokens_per_microbatch < 1:
            raise ValueError("max_tokens_per_microbatch must be positive")
        self.model = model
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens_per_microbatch
        self.autocast_dtype = autocast_dtype
        self.fingerprint = tokenizer_fingerprint(tokenizer)
        pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        if pad is None:
            raise ValueError("tokenizer needs a pad or eos token for batching")
        self.pad_id = int(pad)
        eos = model.generation_config.eos_token_id if getattr(model, "generation_config", None) else None
        if eos is None:
            eos = tokenizer.eos_token_id
        self.eos_ids = frozenset([eos] if isinstance(eos, int) else (eos or []))
        self.scoring = scoring
        if scoring == "auto":
            self.scoring = "selective" if self.check_head_consistency() < 1e-3 else "full"
        self._torch = torch

    @property
    def device(self):
        return next(self.model.parameters()).device

    # ------------------------------------------------------------- prompts

    def encode_chat(self, messages: Sequence[dict], *, add_generation_prompt: bool = True) -> tuple[int, ...]:
        if getattr(self.tokenizer, "chat_template", None):
            ids = self.tokenizer.apply_chat_template(list(messages), add_generation_prompt=add_generation_prompt,
                                                     tokenize=True)
            if isinstance(ids, dict) or hasattr(ids, "input_ids"):
                ids = ids["input_ids"]
        else:
            text = "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"
            ids = self.tokenizer(text, add_special_tokens=True)["input_ids"]
        return tuple(int(x) for x in ids)

    # ------------------------------------------------------------- scoring

    def _autocast(self):
        import contextlib
        torch = self._torch if hasattr(self, "_torch") else __import__("torch")
        if self.autocast_dtype is None:
            return contextlib.nullcontext()
        return torch.autocast(device_type=self.device.type, dtype=self.autocast_dtype)

    def _hidden(self, input_ids, attention_mask):
        base = getattr(self.model, "base_model", None)
        if base is None or base is self.model:
            raise RuntimeError("selective scoring needs a separable base model")
        out = base(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        return out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]

    def _head(self, hidden):
        logits = self.model.get_output_embeddings()(hidden).float()
        cap = getattr(self.model.config, "final_logit_softcapping", None)
        if cap:
            logits = cap * self._torch.tanh(logits / cap)
        return logits

    def _microbatches(self, lengths: Sequence[int]) -> list[list[int]]:
        order = sorted(range(len(lengths)), key=lambda i: -lengths[i])
        batches, current, longest = [], [], 0
        for i in order:
            candidate = max(longest, lengths[i])
            if current and candidate * (len(current) + 1) > self.max_tokens:
                batches.append(current)
                current, longest = [], 0
                candidate = lengths[i]
            current.append(i)
            longest = candidate
        if current:
            batches.append(current)
        return batches

    def sequence_log_probs(self, sequences: Sequence[Sequence[int]], positions: Sequence[Sequence[int]]):
        """Log p(seq[p] | seq[:p]) for every listed p, flattened in input order.

        Differentiable; one forward per sequence, packed by token budget.
        """
        torch = self._torch
        if len(sequences) != len(positions):
            raise ValueError("one position list per sequence is required")
        for seq, pos in zip(sequences, positions):
            if pos and (min(pos) < 1 or max(pos) >= len(seq)):
                raise ValueError("positions must satisfy 1 <= p < len(sequence)")
        device = self.device
        pieces: list[tuple[int, Any]] = []
        lengths = [max(pos) + 1 if pos else 0 for pos in positions]  # Later tokens cannot affect earlier logits.
        for batch in self._microbatches(lengths):
            batch = [i for i in batch if lengths[i] > 0]
            if not batch:
                continue
            width = max(lengths[i] for i in batch)
            ids = torch.full((len(batch), width), self.pad_id, dtype=torch.long, device=device)
            mask = torch.zeros((len(batch), width), dtype=torch.long, device=device)
            for row, i in enumerate(batch):
                ids[row, :lengths[i]] = torch.tensor(sequences[i][:lengths[i]], dtype=torch.long, device=device)
                mask[row, :lengths[i]] = 1
            rows = torch.tensor([row for row, i in enumerate(batch) for _ in positions[i]], device=device)
            cols = torch.tensor([p - 1 for i in batch for p in positions[i]], device=device)
            targets = torch.tensor([sequences[i][p] for i in batch for p in positions[i]], device=device)
            with self._autocast():
                if self.scoring == "selective":
                    logits = self._head(self._hidden(ids, mask)[rows, cols])
                else:
                    full = self.model(input_ids=ids, attention_mask=mask, use_cache=False).logits
                    logits = full[rows, cols].float()
            logp = torch.log_softmax(logits.float(), dim=-1).gather(1, targets[:, None]).squeeze(1)
            start = 0
            for i in batch:
                pieces.append((i, logp[start:start + len(positions[i])]))
                start += len(positions[i])
        by_index = dict(pieces)
        empty = self.model.get_output_embeddings().weight.sum() * 0
        ordered = [by_index[i] if i in by_index else empty.float().reshape(1)[:0] for i in range(len(sequences))]
        return torch.cat(ordered) if ordered else empty.float().reshape(1)[:0]

    def check_head_consistency(self, probe: Sequence[int] | None = None) -> float:
        """Max |selective - full| log-prob difference on a probe sequence."""
        import torch
        self._torch = torch
        vocab = self.model.get_output_embeddings().weight.shape[0]
        ids = list(probe) if probe is not None else [int(x) % vocab for x in range(3, 3 + 16)]
        positions = list(range(1, len(ids)))
        previous, self.scoring = getattr(self, "scoring", "auto"), "selective"
        try:
            with torch.no_grad():
                selective = self.sequence_log_probs([ids], [positions])
                self.scoring = "full"
                full = self.sequence_log_probs([ids], [positions])
        except RuntimeError:
            return math.inf
        finally:
            self.scoring = previous
        return float((selective - full).abs().max().item())

    # ---------------------------------------------------------- generation

    def generate(self, prompts: Sequence[Sequence[int]], *, max_new_tokens: int, generator: Any,
                 stop_ids: Sequence[int] | None = None) -> list[tuple[tuple[int, ...], tuple[float, ...]]]:
        """Sample with temperature 1 and full support; return tokens and their log probs.

        ``generator`` is one ``torch.Generator`` or one per prompt. Sampling
        uses the Gumbel-max trick with each row's own noise stream, which draws
        exactly from softmax(logits) and makes a row's sample independent of
        which other rows share its batch (up to floating-point ties). Left-pads
        the batch, decodes with the KV cache and stops each row at an
        end-of-sequence token (kept as the final policy token).
        """
        torch = self._torch
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if not prompts:
            return []
        if any(len(p) == 0 for p in prompts):
            raise ValueError("prompts must be nonempty")
        stops = frozenset(stop_ids) if stop_ids is not None else self.eos_ids
        device = self.device
        generators = list(generator) if isinstance(generator, (list, tuple)) else None
        if generators is not None and len(generators) != len(prompts):
            raise ValueError("one generator per prompt is required")
        width = max(len(p) for p in prompts)
        ids = torch.full((len(prompts), width), self.pad_id, dtype=torch.long, device=device)
        mask = torch.zeros((len(prompts), width), dtype=torch.long, device=device)
        for row, prompt in enumerate(prompts):
            ids[row, width - len(prompt):] = torch.tensor(list(prompt), dtype=torch.long, device=device)
            mask[row, width - len(prompt):] = 1
        tokens: list[list[int]] = [[] for _ in prompts]
        logps: list[list[float]] = [[] for _ in prompts]
        done = torch.zeros(len(prompts), dtype=torch.bool, device=device)
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad(), self._autocast():
                position_ids = (mask.cumsum(dim=1) - 1).clamp_min(0)
                out = self.model(input_ids=ids, attention_mask=mask, position_ids=position_ids, use_cache=True)
                past = out.past_key_values
                logits = out.logits[:, -1, :].float()
                for _ in range(max_new_tokens):
                    lp = torch.log_softmax(logits, dim=-1)
                    if generators is None:
                        uniform = torch.rand(lp.shape, generator=generator, device=device, dtype=torch.float64)
                    else:
                        uniform = torch.stack([torch.rand(lp.shape[1], generator=g, device=device,
                                                          dtype=torch.float64) for g in generators])
                    tiny = torch.finfo(torch.float64).tiny
                    gumbel = -torch.log(-torch.log(uniform.clamp(tiny, 1.0 - 1e-16)))
                    drawn = (lp.double() + gumbel).argmax(dim=-1)
                    chosen = lp.gather(1, drawn[:, None]).squeeze(1)
                    for row in (~done).nonzero().flatten().tolist():
                        tokens[row].append(int(drawn[row]))
                        logps[row].append(float(chosen[row]))
                    done |= torch.tensor([int(t) in stops for t in drawn.tolist()], device=device)
                    if bool(done.all()):
                        break
                    mask = torch.cat([mask, torch.ones((len(prompts), 1), dtype=mask.dtype, device=device)], dim=1)
                    step_positions = (mask.sum(dim=1, keepdim=True) - 1)
                    out = self.model(input_ids=drawn[:, None], attention_mask=mask, position_ids=step_positions,
                                     past_key_values=past, use_cache=True)
                    past = out.past_key_values
                    logits = out.logits[:, -1, :].float()
        finally:
            self.model.train(was_training)
        return [(tuple(t), tuple(lp)) for t, lp in zip(tokens, logps)]

    def rollout_parity(self, prompts: Sequence[Sequence[int]], *, max_new_tokens: int, generator: Any) -> ParityReport:
        """Generate, then rescore the same tokens through the training path."""
        torch = self._torch
        samples = self.generate(prompts, max_new_tokens=max_new_tokens, generator=generator)
        sequences = [tuple(p) + t for p, (t, _) in zip(prompts, samples)]
        positions = [list(range(len(p), len(p) + len(t))) for p, (t, _) in zip(prompts, samples)]
        with torch.no_grad():
            scored = self.sequence_log_probs(sequences, positions).double()
        recorded = torch.tensor([x for _, lps in samples for x in lps], dtype=torch.float64, device=scored.device)
        gap = scored - recorded
        if gap.numel() == 0:
            return ParityReport(0, 0.0, 0.0, 0.0)
        return ParityReport(int(gap.numel()), float(gap.abs().mean()), float(gap.abs().max()), float(gap.mean()))
