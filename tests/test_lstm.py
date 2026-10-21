"""Fast unit tests for tokenizer, vocabulary, padding, LSTM and persistence logic."""
from io import BytesIO
from pathlib import Path
import unittest
import numpy as np
import torch
from coat_eda.mlp import load_baseline_assignments
from coat_eda.lstm import LSTMClassifier, Vocabulary, collate_sequences, load_lstm, metrics, tokenize

class LSTMTests(unittest.TestCase):
    def test_fixed_ids_are_disjoint(self):
        assignments = load_baseline_assignments(Path("reports/baseline/split_assignments.csv")); self.assertEqual(len(assignments), 30000)
        train = set(assignments.loc[assignments.subset == "train", "id"]); valid = set(assignments.loc[assignments.subset == "internal_validation", "id"]); self.assertFalse(train & valid)
    def test_vocab_train_only_unknown_and_padding(self):
        vocab = Vocabulary.build(["Привет, мир!"])
        self.assertEqual(Vocabulary.from_dict(vocab.to_dict()).token_to_id, vocab.token_to_id)
        self.assertEqual(tokenize("Привет, мир!"), ["Привет", ",", "мир", "!"])
        self.assertEqual(vocab.encode("unknown")[0], 1); self.assertNotIn("unknown", vocab.token_to_id)
        padded, lengths, _ = collate_sequences([(vocab.encode("Привет"), 0.), (vocab.encode("Привет мир !"), 1.)])
        self.assertEqual(tuple(padded.shape), (2, 3)); self.assertEqual(lengths.tolist(), [1, 3]); self.assertEqual(padded[0, 1].item(), 0)
    def test_model_lengths_metrics_and_checkpoint(self):
        model = LSTMClassifier(10); tokens, lengths, labels = collate_sequences([([2], 0.), ([2, 3, 4], 1.)]); logits = model(tokens, lengths)
        self.assertEqual(tuple(logits.shape), (2,)); self.assertTrue(torch.isfinite(logits).all())
        self.assertTrue(torch.isfinite(torch.nn.BCEWithLogitsLoss()(logits, labels)))
        result = metrics(np.array([0, 1, 1, 0]), np.array([.1, .9, .4, .2])); self.assertAlmostEqual(result["f1_label_1"], 2/3)
        blob = BytesIO(); torch.save({"model_state_dict": model.state_dict(), "config": {"embedding_dim": 128, "hidden_size": 128, "dropout": .3}, "vocab_size": 10}, blob); blob.seek(0)
        restored = load_lstm(blob, torch.device("cpu")); self.assertTrue(all(torch.equal(a, b) for a,b in zip(model.parameters(), restored.parameters())))
if __name__ == "__main__": unittest.main()
