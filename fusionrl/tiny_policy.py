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
        if any(not c or any(not isinstance(x,int) or x < 0 or x >= self.vocab_size for x in c) for c in contexts):
            raise ValueError("Contexts must be nonempty valid token sequences")
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

    @torch.no_grad()
    def sample(self, prompt: tuple[int,...], *, max_tokens: int, generator: torch.Generator) -> tuple[tuple[int,...],tuple[float,...]]:
        tokens,logps=[],[]
        for _ in range(max_tokens):
            lp=torch.log_softmax(self.logits([prompt+tuple(tokens)])[0],dim=-1)
            token=int(torch.multinomial(lp.exp(),1,generator=generator).item())
            tokens.append(token)
            logps.append(float(lp[token].item()))
            if token < 8:  # First ordinary symbol is the terminal answer.
                break
        return tuple(tokens),tuple(logps)

    @torch.no_grad()
    def evaluate(self, efforts: tuple[int,...], max_tokens: int = 4) -> dict[str,float]:
        prompts=[self.prompt(symbol,b) for symbol in range(8) for b in efforts]
        answers=torch.tensor([symbol for symbol in range(8) for _ in efforts],device=self.head.weight.device)
        probs=self.logits(prompts).softmax(dim=-1)
        first=float(probs[torch.arange(len(prompts)),answers].mean().item())
        successes=0
        for context,answer in zip(prompts,answers.tolist()):
            for _ in range(max_tokens):
                token=int(self.logits([context]).argmax(dim=-1).item())
                context=context+(token,)
                if token < 8:
                    successes+=int(token==answer)
                    break
        return {"first_token_correct_probability":first,"greedy_completion_accuracy":successes/len(prompts)}

