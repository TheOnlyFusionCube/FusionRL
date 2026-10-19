"""Small autoregressive CPU model for executable algorithm verification.

This is a synthetic environment test, not a pretrained language model or an
estimate of coding, reasoning, or agent benchmark performance.
"""
from __future__ import annotations

import torch
from torch import nn


class TinyCausalPolicy(nn.Module):
    vocab_size = 14
    tokenizer_fingerprint = "fusionrl-synthetic-symbol-v1:14:exact-id-map"

    def __init__(self, hidden_size: int = 32):
        super().__init__()
        self.embedding = nn.Embedding(self.vocab_size,hidden_size)
        self.rnn = nn.GRU(hidden_size,hidden_size,batch_first=True)
        self.head = nn.Linear(hidden_size,self.vocab_size)

    @staticmethod
    def prompt(symbol: int, effort: int) -> tuple[int, ...]:
        if not 0 <= symbol < 8 or not 1 <= effort <= 100:
            raise ValueError("Invalid synthetic task")
        # Distinct effort tokens condition generation; never compare subgroups.
        effort_bin = min(3,(effort-1)//25)
        return (8,symbol,9,10+effort_bin)

    def logits(self, contexts: list[tuple[int,...]]) -> torch.Tensor:
        device = next(self.parameters()).device
        if not contexts:
            return self.head.weight[:0,:].sum(dim=1)[:,None].expand(0,self.vocab_size)
        self._validate_contexts(contexts)
        lengths = torch.tensor([len(c) for c in contexts],device=device)
        padded = nn.utils.rnn.pad_sequence([torch.tensor(c,dtype=torch.long,device=device) for c in contexts],batch_first=True)
        output,_ = self.rnn(self.embedding(padded))
        last = output[torch.arange(len(contexts),device=device),lengths-1]
        return self.head(last)

    def log_probs(self, contexts: list[tuple[int,...]], token_ids: list[int]) -> torch.Tensor:
        if len(contexts) != len(token_ids):
            raise ValueError("Context and target counts must agree")
        logits = self.logits(contexts)
        ids=torch.tensor(token_ids,dtype=torch.long,device=logits.device)
        return torch.log_softmax(logits,dim=-1).gather(1,ids[:,None]).squeeze(1)

    def _validate_contexts(self, contexts: list[tuple[int,...]]) -> None:
        if any(not c or any(not isinstance(x,int) or x < 0 or x >= self.vocab_size for x in c) for c in contexts):
            raise ValueError("Contexts must be nonempty valid token sequences")

    @torch.no_grad()
    def _decode(self, prompts: list[tuple[int,...]], max_tokens: int,
                generator: torch.Generator | None) -> list[tuple[tuple[int,...],tuple[float,...]]]:
        """Decode many prompts together, carrying the GRU state between steps.

        Each step runs one recurrent update for the still-active rows instead of
        re-encoding every full context. With ``generator`` the next token is
        sampled from the full temperature-1 distribution; without it, greedy.
        The recorded log probability is the one the token was drawn from.
        """
        if isinstance(max_tokens,bool) or not isinstance(max_tokens,int) or max_tokens < 1:
            raise ValueError("max_tokens must be a positive integer")
        if not prompts:
            return []
        self._validate_contexts(prompts)
        if self.rnn.num_layers != 1:
            raise ValueError("Incremental decoding assumes a single-layer GRU")
        device = next(self.parameters()).device
        lengths = torch.tensor([len(c) for c in prompts],device=device)
        padded = nn.utils.rnn.pad_sequence([torch.tensor(c,dtype=torch.long,device=device) for c in prompts],batch_first=True)
        # Trailing padding cannot affect earlier outputs of a causal GRU.
        output,_ = self.rnn(self.embedding(padded))
        hidden = output[torch.arange(len(prompts),device=device),lengths-1]
        rows = torch.arange(len(prompts),device=device)
        tokens: list[list[int]] = [[] for _ in prompts]
        logps: list[list[float]] = [[] for _ in prompts]
        for _ in range(max_tokens):
            lp = torch.log_softmax(self.head(hidden),dim=-1)
            if generator is None:
                drawn = lp.argmax(dim=-1)
            else:
                drawn = torch.multinomial(lp.exp(),1,generator=generator).squeeze(1)
            chosen = lp.gather(1,drawn[:,None]).squeeze(1)
            for row,token,logp in zip(rows.tolist(),drawn.tolist(),chosen.tolist()):
                tokens[row].append(int(token))
                logps[row].append(float(logp))
            keep = drawn >= 8  # First ordinary symbol is the terminal answer.
            if not bool(keep.any()):
                break
            rows, hidden, drawn = rows[keep], hidden[keep], drawn[keep]
            step,_ = self.rnn(self.embedding(drawn)[:,None,:],hidden[None].contiguous())
            hidden = step[:,0]
        return [(tuple(t),tuple(l)) for t,l in zip(tokens,logps)]

    def sample_batch(self, prompts: list[tuple[int,...]], *, max_tokens: int,
                     generator: torch.Generator) -> list[tuple[tuple[int,...],tuple[float,...]]]:
        """Sample one completion per prompt with its actual behavior log probabilities."""
        return self._decode(list(prompts),max_tokens,generator)

    def sample(self, prompt: tuple[int,...], *, max_tokens: int, generator: torch.Generator) -> tuple[tuple[int,...],tuple[float,...]]:
        return self.sample_batch([prompt],max_tokens=max_tokens,generator=generator)[0]

    @torch.no_grad()
    def evaluate(self, efforts: tuple[int,...], max_tokens: int = 4) -> dict[str,float]:
        prompts=[self.prompt(symbol,b) for symbol in range(8) for b in efforts]
        answers=[symbol for symbol in range(8) for _ in efforts]
        probs=self.logits(prompts).softmax(dim=-1)
        first=float(probs[torch.arange(len(prompts)),torch.tensor(answers,device=probs.device)].mean().item())
        completions=self._decode(prompts,max_tokens,None)
        successes=sum(tokens[-1]==answer for (tokens,_),answer in zip(completions,answers))
        return {"first_token_correct_probability":first,"greedy_completion_accuracy":successes/len(prompts)}
