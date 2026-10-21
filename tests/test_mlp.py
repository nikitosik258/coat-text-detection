"""Fast MLP component checks; no dataset download or full training required."""
import unittest
from pathlib import Path
from io import BytesIO

import numpy as np
import torch

from coat_eda.mlp import INPUT_DIM, MLP, SparseIndexDataset, load_baseline_assignments, load_mlp_for_prediction, set_seed


class MLPTests(unittest.TestCase):
    def test_uses_the_persisted_baseline_ids(self) -> None:
        path = Path("reports/baseline/split_assignments.csv")
        self.assertTrue(path.exists(), "MLP requires the baseline assignment artifact")
        assignments = load_baseline_assignments(path)
        self.assertEqual(len(assignments), 30_000)
        self.assertEqual(assignments.groupby(["subset", "label"]).size().to_dict(),
                         {("internal_validation", 0): 3000, ("internal_validation", 1): 3000,
                          ("train", 0): 12000, ("train", 1): 12000})
    def test_input_output_shape_and_finite_probabilities(self) -> None:
        model = MLP(); logits = model(torch.zeros((3, INPUT_DIM)))
        probabilities = torch.sigmoid(logits)
        loss = torch.nn.BCEWithLogitsLoss()(logits, torch.tensor([0.0, 1.0, 0.0]))
        self.assertEqual(tuple(logits.shape), (3,))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertTrue(torch.isfinite(probabilities).all())
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.all((probabilities >= 0) & (probabilities <= 1)))

    def test_sparse_dataset_densifies_only_batch(self) -> None:
        from scipy.sparse import csr_matrix
        dataset = SparseIndexDataset(csr_matrix(np.eye(4, INPUT_DIM, dtype=np.float32)), np.array([0, 1, 0, 1]))
        features, labels = dataset.collate([0, 2])
        self.assertEqual(tuple(features.shape), (2, INPUT_DIM)); self.assertEqual(tuple(labels.shape), (2,))

    def test_checkpoint_round_trip(self) -> None:
        set_seed(); model = MLP()
        # Keep the short-lived artifact inside the writable project workspace.
        checkpoint = BytesIO()
        torch.save({"model_state_dict": model.state_dict(), "config": {"input_dim": INPUT_DIM}, "best_epoch": 1}, checkpoint)
        checkpoint.seek(0)
        restored = load_mlp_for_prediction(checkpoint, torch.device("cpu"))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(model.parameters(), restored.parameters())))


if __name__ == "__main__": unittest.main()
