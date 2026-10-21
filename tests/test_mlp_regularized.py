"""Fast checks for the fixed regularized MLP configuration."""
from io import BytesIO
from pathlib import Path
import unittest

import numpy as np
import torch

from coat_eda.mlp import load_baseline_assignments
from coat_eda.mlp_regularized import INPUT_DIM, RegularizedMLP, calculate_metrics, load_regularized_mlp


class RegularizedMLPTests(unittest.TestCase):
    def test_layer_shapes_and_finite_values(self) -> None:
        model = RegularizedMLP(); logits = model(torch.zeros((2, INPUT_DIM))); probabilities = torch.sigmoid(logits)
        self.assertEqual(tuple(logits.shape), (2,)); self.assertEqual(model.network[0].out_features, 128)
        self.assertEqual(model.network[3].out_features, 32); self.assertTrue(torch.isfinite(logits).all())
        self.assertTrue(torch.all((probabilities >= 0) & (probabilities <= 1)))

    def test_persisted_ids_match_original_mlp_source(self) -> None:
        assignments = load_baseline_assignments(Path("reports/baseline/split_assignments.csv"))
        self.assertEqual(len(assignments), 30_000); self.assertEqual(assignments.id.nunique(), 30_000)
        self.assertEqual(set(assignments.subset), {"train", "internal_validation"})

    def test_metrics_and_checkpoint_round_trip(self) -> None:
        metrics = calculate_metrics(np.array([0, 1, 1, 0]), np.array([.1, .9, .4, .2]))
        self.assertAlmostEqual(metrics["f1_label_1"], 2 / 3); self.assertTrue(np.isfinite(metrics["roc_auc"]))
        model = RegularizedMLP(); buffer = BytesIO()
        torch.save({"model_state_dict": model.state_dict(), "config": {"input_dim": INPUT_DIM}}, buffer); buffer.seek(0)
        restored = load_regularized_mlp(buffer, torch.device("cpu"))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(model.parameters(), restored.parameters())))


if __name__ == "__main__": unittest.main()
