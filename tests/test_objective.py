import math
import unittest

import numpy as np

from fusionrl.objective import ExecutionLossConfig, execution_loss_numpy

try:
    import torch
except ImportError:
    torch = None


class NumpyObjectiveTests(unittest.TestCase):
    def test_empty_original_slots_keep_weight(self):
        full, _ = execution_loss_numpy([[-2.0], [0.0]], [[-2.0], [0.0]], [1.0, -1.0], [[True], [False]])
        one, _ = execution_loss_numpy([[-2.0]], [[-2.0]], [1.0], [[True]])
        self.assertAlmostEqual(full * 2, one)

    def test_sign_aware_clipping(self):
        cfg = ExecutionLossConfig(clip_low=.9, clip_high=1.1)
        lp = np.array([[-1.0], [-2.0], [-1.0], [-2.0]])
        bp = np.array([[-2.0], [-1.0], [-2.0], [-1.0]])
        loss, m = execution_loss_numpy(lp, bp, [1., -1., -1., 1.], np.ones((4,1), bool), config=cfg)
        expected = (-math.e + 2/math.e)/4
        self.assertAlmostEqual(loss, expected)
        self.assertEqual(m["clip_fraction"], .5)

    def test_masked_nan_ignored_but_active_nan_rejected(self):
        x, _ = execution_loss_numpy([[float("nan"), -2]], [[float("nan"), -2]], [1.], [[False,True]])
        self.assertEqual(x, 2)
        with self.assertRaises(ValueError):
            execution_loss_numpy([[float("nan")]], [[-2]], [1.], [[True]])

    def test_denominator_cannot_drop_empty_slots(self):
        with self.assertRaises(ValueError):
            execution_loss_numpy([[0.],[0.]], [[0.],[0.]], [0.,0.], [[False],[False]], global_execution_count=1)


@unittest.skipIf(torch is None, "PyTorch extra is not installed")
class TorchObjectiveTests(unittest.TestCase):
    def test_matches_independent_numpy_oracle(self):
        from fusionrl.objective import execution_loss
        rng = np.random.default_rng(17)
        lp = -rng.uniform(.1,6,(11,9))
        bp = lp + rng.normal(0,.2,(11,9))
        bp = np.minimum(bp,0)
        adv = rng.normal(0,1,11)
        mask = rng.uniform(0,1,(11,9)) > .3
        mask[3,:] = False
        for lo, hi in [(.995,1.005),(.8,1.2)]:
            cfg = ExecutionLossConfig(lo,hi)
            expected, _ = execution_loss_numpy(lp,bp,adv,mask,config=cfg)
            actual, _ = execution_loss(torch.tensor(lp),torch.tensor(bp),torch.tensor(adv),torch.tensor(mask),config=cfg)
            self.assertAlmostEqual(actual.item(),expected,places=11)

    def test_only_logp_receives_exact_semigradient(self):
        from fusionrl.objective import execution_loss
        lp = torch.tensor([[-2.,-3.],[-1.,0.]],dtype=torch.float64,requires_grad=True)
        bp = lp.detach().clone().requires_grad_()
        adv = torch.tensor([1.,-1.],dtype=torch.float64,requires_grad=True)
        mask = torch.tensor([[True,True],[True,False]])
        loss,_ = execution_loss(lp,bp,adv,mask)
        loss.backward()
        np.testing.assert_allclose(lp.grad, [[-.25,-.25],[.5,0]],atol=1e-12)
        self.assertIsNone(bp.grad)
        self.assertIsNone(adv.grad)

    def test_importance_cap_and_sequence_clipping_have_distinct_roles(self):
        from fusionrl.objective import execution_loss
        lp = torch.tensor([[-1.],[-1.]],requires_grad=True)
        bp = torch.tensor([[-10.],[-10.]])
        loss,_ = execution_loss(lp,bp,torch.tensor([-1.,1.]),torch.ones((2,1),dtype=torch.bool))
        loss.backward()
        np.testing.assert_allclose(lp.grad, [[2.5],[0]],atol=1e-6)

    def test_upper_loss_cap_not_absolute_clip(self):
        from fusionrl.objective import execution_loss
        lp = torch.tensor([[-200.],[-200.]],requires_grad=True)
        loss,_ = execution_loss(lp,lp.detach().clone(),torch.tensor([1.,-1.]),torch.ones((2,1),dtype=torch.bool))
        self.assertEqual(loss.item(), -50)
        loss.backward()
        np.testing.assert_allclose(lp.grad, [[0],[.5]])

    def test_empty_execution_zero_gradient(self):
        from fusionrl.objective import execution_loss
        lp = torch.tensor([[-1.,-2.]],requires_grad=True)
        loss,m = execution_loss(lp,lp.detach(),torch.tensor([1.]),torch.zeros((1,2),dtype=torch.bool))
        loss.backward()
        self.assertEqual(loss.item(),0)
        self.assertEqual(m["empty_executions"],1)
        self.assertTrue(torch.equal(lp.grad,torch.zeros_like(lp)))

    def test_distributed_sum_count_equivalence(self):
        from fusionrl.objective import execution_loss
        lp = torch.tensor([[-1.],[-2.],[-3.],[-4.]],requires_grad=True)
        adv = torch.tensor([1.,-.5,.25,-.75])
        mask = torch.tensor([[True],[True],[False],[True]])
        all_loss,_=execution_loss(lp,lp.detach(),adv,mask)
        shards=[]
        for sl in [slice(0,2),slice(2,4)]:
            part,_=execution_loss(lp[sl],lp.detach()[sl],adv[sl],mask[sl],global_execution_count=4)
            shards.append(part*2)
        self.assertAlmostEqual(torch.stack(shards).mean().item(),all_loss.item())


if __name__ == "__main__":
    unittest.main()

