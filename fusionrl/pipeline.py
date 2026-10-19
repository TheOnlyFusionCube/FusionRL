"""Mix compatible task-specific grading routes in ONE optimizer batch."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from .learner import LearnerBatch, assemble_batch
from .rewards import AdvantageResult, compute_advantages
from .trajectory import AdmittedGroup


@dataclass(frozen=True)
class GroupSupervision:
    """Externally evaluated grades, bound to exact executions and artifacts.

    ``scores`` overrides the recorded base scores only for a declared, calibrated
    scalar grader. Other routes use immutable recorded binary assessments.
    GRS rubric and GAR ranking arrays use the same ordered execution IDs.
    """
    execution_ids: tuple[str,...]
    artifact_ids: tuple[str,...]
    conditioning_key: str
    route: str = "binary"
    scores: tuple[float,...] | None = None
    solution_scores: tuple[float,...] | None = None
    behavior_scores: tuple[float,...] | None = None
    quality_factors: tuple[float | None,...] | None = None
    confirmed_hacks: tuple[bool,...] | None = None
    cost_penalties: tuple[float,...] | None = None
    gar_cap: float = 2.0
    hack_penalty: float = 0.0


@dataclass(frozen=True)
class PreparedStep:
    batch: LearnerBatch
    grading: dict[str,AdvantageResult]


def prepare_step(groups: Sequence[AdmittedGroup], supervision: Mapping[str,GroupSupervision]) -> PreparedStep:
    """Center each homogeneous group, then concatenate for one execution loss.

    A batch may contain GAR code groups, GRS quality groups and calibrated scalar
    groups. Grades from different groups never share a baseline or a ranking.
    The group key must include its chosen grading route/configuration version.
    """
    if set(supervision)!={g.group_id for g in groups}:
        raise ValueError("Exactly one supervision record is required per admitted group")
    results={}
    for group in groups:
        grade=supervision[group.group_id]
        if tuple(grade.execution_ids)!=tuple(e.execution_id for e in group.executions):
            raise ValueError("Grader order/identity differs from the admitted execution order")
        if tuple(grade.artifact_ids)!=tuple(e.assessment.artifact_id for e in group.executions):
            raise ValueError("Grader results refer to different assessed artifacts")
        if not grade.conditioning_key or grade.conditioning_key!=group.conditioning_key:
            raise ValueError("Grader configuration differs from the immutable group contract")
        if grade.scores is not None and grade.route!="scalar":
            raise ValueError("Binary/GRS/GAR routes cannot overwrite verified base outcomes")
        outcomes=grade.scores if grade.scores is not None else group.rewards
        results[group.group_id]=compute_advantages(outcomes,route=grade.route,
            solution_scores=grade.solution_scores,behavior_scores=grade.behavior_scores,
            quality_factors=grade.quality_factors,confirmed_hacks=grade.confirmed_hacks,
            cost_penalties=grade.cost_penalties,gar_cap=grade.gar_cap,hack_penalty=grade.hack_penalty)
    batch=assemble_batch(groups,[results[g.group_id].advantages for g in groups])
    return PreparedStep(batch,results)

