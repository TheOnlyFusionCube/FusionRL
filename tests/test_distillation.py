import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch extra is not installed")
class DistillationTests(unittest.TestCase):
    def test_full_vocab_shared_mapping_and_teacher_detach(self):
        from fusionrl.distillation import full_vocabulary_reverse_kl
        student = torch.tensor([[1.,0.,-1.]],requires_grad=True)
        teacher = torch.tensor([[0.,1.,-1.]],requires_grad=True)
        mask = torch.tensor([True])
        loss = full_vocabulary_reverse_kl(student,teacher,mask,student_tokenizer_fingerprint="v1",teacher_tokenizer_fingerprint="v1")
        self.assertGreater(loss.item(),0)
        loss.backward()
        self.assertIsNotNone(student.grad)
        self.assertIsNone(teacher.grad)
        with self.assertRaises(ValueError):
            full_vocabulary_reverse_kl(student,teacher,mask,student_tokenizer_fingerprint="v1",teacher_tokenizer_fingerprint="permuted_v1")

    def test_mask_and_identity(self):
        from fusionrl.distillation import full_vocabulary_reverse_kl
        x=torch.tensor([[1.,2.],[float('nan'),float('nan')]],requires_grad=True)
        loss=full_vocabulary_reverse_kl(x,x.detach(),torch.tensor([True,False]),student_tokenizer_fingerprint="v",teacher_tokenizer_fingerprint="v")
        self.assertAlmostEqual(loss.item(),0,places=6)
        loss.backward()
        self.assertTrue(torch.isfinite(x.grad).all())


if __name__ == "__main__":
    unittest.main()

