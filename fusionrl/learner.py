"""Connect admitted provenance records to a differentiable causal policy.

No chat re-rendering, branch concatenation, or reward recomputation occurs here.
The policy callback must score each target under its supplied original context.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

from .objective import ExecutionLossConfig, execution_loss
from .trajectory import AdmittedGroup, TrainingTarget


@dataclass(frozen=True)
class LearnerBatch:
    execution_ids: tuple[str, ...]
    targets: tuple[tuple[TrainingTarget, ...], ...]
    advantages: tuple[float, ...]
    group_ids: tuple[str, ...]

    @property
    def target_count(self) -> int:
        return sum(len(row) for row in self.targets)


def assemble_batch(groups: Sequence[AdmittedGroup], group_advantages: Sequence[Sequence[float]]) -> LearnerBatch:
    if not groups or len(groups) != len(group_advantages):
        raise ValueError("Need a complete advantage vector for each admitted group")
    if len({g.group_id for g in groups}) != len(groups):
        raise ValueError("A group may be consumed only once in an update")
    if len({g.expected_executions for g in groups}) != 1:
        raise ValueError("One optimizer batch must use a common original group size")
    ids, targets, values, group_ids = [], [], [], []
    for group, advantages in zip(groups,group_advantages):
        if len(group.executions)!=group.expected_executions:
            raise ValueError("An admitted group cannot omit original execution slots")
        vector = np.asarray(advantages,dtype=np.float64)
        if vector.shape != (len(group.executions),) or not np.isfinite(vector).all():
            raise ValueError("Advantages must retain every original execution slot")
        for execution, advantage in zip(group.executions,vector):
            ids.append(execution.execution_id)
            group_ids.append(group.group_id)
            targets.append(execution.targets)
            values.append(float(advantage))
    if len(set(ids)) != len(ids):
        raise ValueError("Execution identities must be unique across an optimizer batch")
    return LearnerBatch(tuple(ids),tuple(targets),tuple(values),tuple(group_ids))


def learner_loss(
    batch: LearnerBatch,
    log_prob_fn: Callable[[list[tuple[int, ...]], list[int]], Any],
    *,
    config: ExecutionLossConfig | None = None,
) -> tuple[Any, dict[str,float]]:
    """Score exact contexts and preserve equal aggregate weight per execution."""
    import torch

    contexts, token_ids, row_ids, col_ids, behavior = [], [], [], [], []
    for i,row in enumerate(batch.targets):
        for j,target in enumerate(row):
            contexts.append(target.context)
            token_ids.append(target.token_id)
            row_ids.append(i)
            col_ids.append(j)
            behavior.append(target.behavior_logp)
    values = log_prob_fn(contexts,token_ids)
    if not torch.is_tensor(values) or values.shape != (len(token_ids),):
        raise ValueError("Policy callback must return one differentiable log probability per target")
    if not values.is_floating_point():
        raise TypeError("Policy log probabilities must be floating point")
    # Never round stored behavior probabilities to BF16/FP16: a .005 sequence
    # clip boundary is smaller than their quantization step near many logps.
    values=values.to(torch.float64 if values.dtype==torch.float64 else torch.float32)
    rows, width = len(batch.targets), max(1,max(map(len,batch.targets),default=0))
    indices = torch.tensor([i*width+j for i,j in zip(row_ids,col_ids)],dtype=torch.long,device=values.device)
    lp = torch.zeros(rows*width,dtype=values.dtype,device=values.device).index_copy(0,indices,values).view(rows,width)
    # Preserve a connected zero if all original slots have no trainable targets.
    lp = lp + values.sum()*0
    bp = torch.zeros(rows*width,dtype=values.dtype,device=values.device)
    bp.index_copy_(0,indices,torch.tensor(behavior,dtype=values.dtype,device=values.device))
    mask = torch.zeros(rows*width,dtype=torch.bool,device=values.device)
    mask[indices] = True
    advantages = torch.tensor(batch.advantages,dtype=values.dtype,device=values.device)
    return execution_loss(lp,bp.view(rows,width),advantages,mask.view(rows,width),config=config)

