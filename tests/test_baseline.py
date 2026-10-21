"""Fast tests for deterministic, leakage-free baseline components."""
import unittest

import numpy as np
import pandas as pd
from datasets import Dataset

from coat_eda.baseline import calculate_metrics, make_pipeline
from coat_eda.data_prep import make_assignments, validate_assignments


class BaselineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = Dataset.from_dict({"id": list(range(40)), "text": [f"text {i}" for i in range(40)],
                                          "label": [0] * 20 + [1] * 20})

    def test_split_is_balanced_and_has_no_shared_ids(self) -> None:
        assignments = make_assignments(self.dataset, sample_per_class=10)
        validate_assignments(assignments, sample_per_class=10)
        self.assertEqual(len(assignments), 20)
        self.assertFalse(set(assignments.query("subset == 'train'").id) &
                         set(assignments.query("subset == 'internal_validation'").id))
        self.assertEqual(assignments.groupby(["subset", "label"]).size().to_dict(),
                         {("internal_validation", 0): 2, ("internal_validation", 1): 2, ("train", 0): 8, ("train", 1): 8})

    def test_tfidf_is_fitted_only_on_training_texts(self) -> None:
        pipeline = make_pipeline()
        pipeline.fit(["common trainonlytoken", "common trainonlytoken other"], [0, 1])
        vocabulary = pipeline.named_steps["tfidf"].vocabulary_
        self.assertNotIn("validationonlytoken", vocabulary)
        self.assertIn("trainonlytoken", vocabulary)

    def test_metrics_for_known_predictions(self) -> None:
        metrics = calculate_metrics(np.array([0, 1, 1, 0]), np.array([0, 1, 0, 0]), np.array([0.1, 0.9, 0.4, 0.2]))
        self.assertEqual(metrics["confusion_matrix"], [[2, 0], [1, 1]])
        self.assertAlmostEqual(metrics["accuracy"], 0.75)
        self.assertAlmostEqual(metrics["precision_label_1"], 1.0)
        self.assertAlmostEqual(metrics["recall_label_1"], 0.5)
        self.assertAlmostEqual(metrics["f1_label_1"], 2 / 3)
        self.assertAlmostEqual(metrics["roc_auc"], 1.0)


if __name__ == "__main__":
    unittest.main()
