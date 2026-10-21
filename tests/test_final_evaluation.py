"""Fast, inference-only unit tests for final evaluation safeguards."""
import unittest
import json
from pathlib import Path
import numpy as np
import pandas as pd
from coat_eda.final_evaluation import prediction_frame, score, validate_prediction_frames

class FinalEvaluationTests(unittest.TestCase):
    def test_shared_metrics_use_label_one_and_probabilities(self):
        result = score(np.array([0,1,1,0]), np.array([.1,.9,.4,.2]))
        self.assertAlmostEqual(result["f1_label_1"], 2/3); self.assertAlmostEqual(result["roc_auc"], 1.0)
    def test_prediction_alignment_and_threshold(self):
        first = prediction_frame(np.array([3,4]), np.array([0,1]), np.array([.2,.8])); second = first.copy(); validate_prediction_frames({"a":first,"b":second})
        self.assertEqual(first.predicted_label.tolist(), [0,1])
    def test_misaligned_or_invalid_probabilities_fail(self):
        first = prediction_frame(np.array([3]), np.array([0]), np.array([.2])); bad = prediction_frame(np.array([4]), np.array([0]), np.array([.2]))
        with self.assertRaises(ValueError): validate_prediction_frames({"a":first,"b":bad})
        with self.assertRaises(ValueError): score(np.array([0,1]), np.array([.1, np.nan]))
    def test_saved_prediction_files_and_metrics_are_consistent(self):
        root = Path("reports/final_evaluation")
        payload = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
        expected_ids = None
        for row in payload:
            frame = pd.read_csv(root / f"{row['key']}_predictions.csv")
            self.assertEqual(len(frame), 24628)
            self.assertTrue(frame.probability_ai.between(0, 1).all())
            self.assertAlmostEqual(score(frame.true_label.to_numpy(), frame.probability_ai.to_numpy())["f1_label_1"], row["f1_label_1"])
            if expected_ids is None: expected_ids = frame.id.to_numpy()
            else: self.assertTrue(np.array_equal(expected_ids, frame.id.to_numpy()))
if __name__ == "__main__": unittest.main()
