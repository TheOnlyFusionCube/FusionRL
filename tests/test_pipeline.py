from dataclasses import replace
import unittest

import numpy as np

from fusionrl.learner import assemble_batch
from fusionrl.pipeline import GroupSupervision,prepare_step
from fusionrl.trajectory import AdmittedExecution,AdmittedGroup,Assessment,TrainingTarget


def make_group(name,rewards):
    executions=[]
    for i,reward in enumerate(rewards):
        target=TrainingTarget(("sample",0),2,(1,),-.5,0,f"{name}:e{i}",name,"main","main","main","policy")
        executions.append(AdmittedExecution(f"{name}:e{i}",name,(target,),Assessment(reward,True,f"artifact:{name}:{i}"),i,("main",)))
    return AdmittedGroup(name,tuple(executions),len(rewards),0,f"conditioning:{name}")


def grade(group,**kwargs):
    return GroupSupervision(tuple(e.execution_id for e in group.executions),
        tuple(e.assessment.artifact_id for e in group.executions),group.conditioning_key,**kwargs)


class PipelineTests(unittest.TestCase):
    def test_different_routes_share_loss_batch_without_sharing_baselines(self):
        gar=make_group("gar",[0,1,1])
        grs=make_group("grs",[1,1,1])
        scalar=make_group("scalar",[.2,.5,.8])
        prepared=prepare_step([gar,grs,scalar],{
            "gar":grade(gar,route="gar",quality_factors=(None,1.,.5)),
            "grs":grade(grs,route="grs",solution_scores=(1.,1.,1.),behavior_scores=(.25,.5,1.)),
            "scalar":grade(scalar,route="scalar")})
        self.assertEqual(len(prepared.batch.targets),9)
        for result in prepared.grading.values():
            self.assertAlmostEqual(float(result.advantages.sum()),0)
        np.testing.assert_allclose(prepared.grading["scalar"].advantages,[-.3,0,.3],atol=1e-15)
        self.assertTrue(prepared.grading["grs"].diagnostics["has_learning_signal"])

    def test_artifact_and_execution_binding_are_enforced(self):
        group=make_group("a",[0,1])
        good=grade(group)
        for bad in [replace(good,execution_ids=tuple(reversed(good.execution_ids))),
                    replace(good,artifact_ids=("wrong","wrong")),replace(good,conditioning_key="other")]:
            with self.assertRaises(ValueError):
                prepare_step([group],{"a":bad})

    def test_group_size_and_missing_slots_cannot_change_weighting(self):
        a,b=make_group("a",[0,1]),make_group("b",[0,0,1])
        with self.assertRaises(ValueError):
            assemble_batch([a,b],[[0,0],[0,0,0]])
        with self.assertRaises(ValueError):
            assemble_batch([replace(a,executions=a.executions[:1])],[[0]])


if __name__=="__main__":
    unittest.main()

