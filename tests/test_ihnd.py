import os
import unittest

import numpy as np
import torch

from ihnd.data import collate_subjects, read_graph
from ihnd.model import IHND, HypergraphDiffusion
from ihnd.mtd import torch_strength
from ihnd.objective import IHNDLoss, intervention_indices


class IHNDTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = os.environ.get("IHND_TEST_GRAPH")
        if not path:
            raise unittest.SkipTest("Set IHND_TEST_GRAPH to an MTD graph from a real participant")
        cls.x, cls.edges, cls.weights = read_graph(path)
        if len(cls.edges) < 8:
            raise unittest.SkipTest("At least eight retained hyperedges are required")

    def batch(self, count=16, device="cpu"):
        metadata = {"subject_id": os.environ.get("IHND_TEST_SUBJECT", "participant"), "site": os.environ.get("IHND_TEST_SITE", "site"), "label": int(os.environ.get("IHND_TEST_LABEL", "1"))}
        values = (metadata, torch.from_numpy(self.x[:, :32].copy()), torch.from_numpy(self.edges[:count].copy()), torch.from_numpy(self.weights[:count].copy()))
        return collate_subjects([values]).to(device)

    def test_diffusion_matches_explicit_operator(self):
        batch = self.batch(8)
        z = batch.x[:, :4].double()
        weights = batch.weights.double()
        layer = HypergraphDiffusion(4).double()
        actual = layer(z, batch.node_index, batch.edge_index, weights)
        degree = z.new_zeros(len(z)).index_add_(0, batch.node_index, weights[batch.edge_index])
        inverse = torch.where(degree > 0, degree.clamp_min(1e-12).rsqrt(), 0)
        gradient = z.new_zeros((len(batch.node_index), len(z)))
        for row, (node, edge) in enumerate(zip(batch.node_index, batch.edge_index)):
            members = batch.node_index[batch.edge_index == edge]
            gradient[row, node] += inverse[node]
            gradient[row, members] -= inverse[members]/len(members)
        averages = torch.stack([z[batch.node_index[batch.edge_index == edge]].mean(dim=0) for edge in range(len(weights))])
        scores = layer.attention(torch.cat((z[batch.node_index], averages[batch.edge_index]), dim=-1)).squeeze(-1)
        attention = torch.zeros_like(scores)
        for node in torch.unique(batch.node_index):
            selected = batch.node_index == node
            attention[selected] = scores[selected].softmax(dim=0)
        expected = z-gradient.T@(attention[:, None]*(gradient@z))
        torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)

    def test_stop_gradient_and_partition(self):
        torch.manual_seed(42)
        batch = self.batch()
        model = IHND(batch.x.shape[1], hidden_dim=8, gradient_checkpointing=True)
        criterion = IHNDLoss()
        model.train()
        output = model(batch)
        torch.testing.assert_close(output["invariant_weights"]+batch.weights*(1-output["mask"]), batch.weights)
        objective = criterion(output, batch)
        (objective["invariant_risk"]+objective["risk_variance"]).backward()
        self.assertTrue(all(p.grad is None for p in model.shortcut_head.parameters()))
        self.assertTrue(any(p.grad is not None for p in model.encoder.parameters()))
        self.assertTrue(any(p.grad is not None for p in model.rationale.parameters()))
        model.zero_grad(set_to_none=True)
        criterion(model(batch), batch)["shortcut_loss"].backward()
        self.assertTrue(all(p.grad is None for p in model.encoder.parameters()))
        self.assertTrue(all(p.grad is None for p in model.rationale.parameters()))
        self.assertTrue(any(p.grad is not None for p in model.shortcut_head.parameters()))

    def test_inference_uses_only_invariant_branch(self):
        batch = self.batch()
        model = IHND(batch.x.shape[1], hidden_dim=8).eval()
        with torch.no_grad():
            before = model(batch)
            for parameter in model.shortcut_head.parameters():
                parameter.add_(100)
            after = model(batch)
        self.assertNotIn("shortcut_logits", before)
        torch.testing.assert_close(before["logits"], after["logits"], rtol=0, atol=0)
        torch.testing.assert_close(before["mask"], before["rationale_probability"])

    def test_mtd_strength_matches_numpy(self):
        q = self.x.astype(np.float64)
        edges = self.edges[:16]
        expected = []
        for edge in edges:
            values = q[edge[edge >= 0]]
            product = values.prod(axis=0)
            deviation = product.std(ddof=0)
            same = np.all(values >= 0, axis=0) | np.all(values <= 0, axis=0)
            expected.append(np.mean(np.abs((product-product.mean())/deviation)*same) if deviation > 0 else 0.0)
        actual = torch_strength(torch.from_numpy(q[None]), torch.from_numpy(edges))[0]
        np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-12, atol=1e-12)

    def test_single_site_donor_fallback(self):
        batch = self.batch()
        donor = intervention_indices(batch.sites, 4)
        self.assertEqual(tuple(donor.shape), (4, 1))
        self.assertEqual(int(donor.sum()), 0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_native_strength_matches_torch(self):
        from ihnd.native import NativeStrength
        kernel = NativeStrength("cuda")
        indices = np.unique(np.linspace(0, len(self.edges)-1, 96, dtype=int))
        edges = torch.as_tensor(self.edges[indices], dtype=torch.int32, device="cuda")
        q = torch.as_tensor(self.x[None], device="cuda")
        observed = torch.empty(len(edges), device="cuda")
        counts = torch.zeros(len(edges), dtype=torch.int32, device="cuda")
        kernel(q, edges, observed, observed, counts)
        expected = torch_strength(q, edges)[0]
        torch.testing.assert_close(observed, expected, rtol=3e-6, atol=3e-7)
        rng = np.random.default_rng(42)
        shuffled = rng.permuted(np.broadcast_to(self.x, (7, *self.x.shape)), axis=2).copy()
        null_q = torch.as_tensor(shuffled, device="cuda")
        output = torch.empty((7, len(edges)), device="cuda")
        kernel(null_q, edges, observed, output, counts)
        null_reference = torch_strength(null_q, edges)
        torch.testing.assert_close(output, null_reference, rtol=3e-6, atol=3e-7)
        kernel(null_q, edges, observed, output, counts, mode=1)
        torch.testing.assert_close(counts, (null_reference >= observed).sum(dim=0).to(torch.int32), rtol=0, atol=0)
        kernel.close()


if __name__ == "__main__":
    unittest.main()
