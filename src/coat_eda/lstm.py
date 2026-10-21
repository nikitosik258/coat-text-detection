"""Word-level LSTM experiment on the fixed CoAT train/internal-validation IDs."""
from __future__ import annotations

import argparse
import copy
import json
import random
import re
from collections import Counter
from pathlib import Path
from time import perf_counter
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score
from torch import Tensor, nn
from torch.nn.utils import clip_grad_norm_, rnn
from torch.utils.data import DataLoader, Dataset

from coat_eda.data_prep import load_texts_for_assignments
from coat_eda.mlp import load_baseline_assignments, set_seed

TOKEN_RE = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)
PAD, UNK = "<PAD>", "<UNK>"
CONFIG = {"embedding_dim": 128, "hidden_size": 128, "dropout": 0.3, "max_vocab_size": 20_000,
          "max_sequence_length": 192, "learning_rate": 0.001, "weight_decay": 1e-4, "batch_size": 64,
          "max_epochs": 15, "patience": 3, "gradient_clip_norm": 1.0, "random_state": 42, "threshold": 0.5}


def tokenize(text: str) -> list[str]:
    """Unicode word/punctuation tokenizer preserving case and Russian characters."""
    return TOKEN_RE.findall(text)


class Vocabulary:
    def __init__(self, token_to_id: dict[str, int]) -> None:
        self.token_to_id = token_to_id

    @classmethod
    def build(cls, texts: list[str], max_size: int = CONFIG["max_vocab_size"]) -> "Vocabulary":
        counts = Counter(token for text in texts for token in tokenize(text))
        # Stable tie-breaking makes vocabulary reproducible across runs/platforms.
        ranked = sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))[: max_size - 2]
        return cls({PAD: 0, UNK: 1, **{token: index + 2 for index, (token, _) in enumerate(ranked)}})

    def encode(self, text: str, max_length: int = CONFIG["max_sequence_length"]) -> list[int]:
        ids = [self.token_to_id.get(token, 1) for token in tokenize(text)[:max_length]]
        return ids or [1]  # pack_padded_sequence requires a positive sequence length

    def to_dict(self) -> dict[str, Any]:
        return {"pad_token": PAD, "unk_token": UNK, "token_to_id": self.token_to_id}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Vocabulary":
        mapping = {token: int(index) for token, index in payload["token_to_id"].items()}
        if mapping.get(PAD) != 0 or mapping.get(UNK) != 1: raise ValueError("Invalid vocabulary special tokens")
        return cls(mapping)

    def __len__(self) -> int: return len(self.token_to_id)


class SequenceDataset(Dataset[tuple[list[int], float]]):
    def __init__(self, texts: list[str], labels: np.ndarray, vocab: Vocabulary) -> None:
        self.sequences = [vocab.encode(text) for text in texts]
        self.labels = np.asarray(labels, dtype=np.float32)

    def __len__(self) -> int: return len(self.labels)
    def __getitem__(self, index: int) -> tuple[list[int], float]: return self.sequences[index], self.labels[index]


def collate_sequences(batch: list[tuple[list[int], float]]) -> tuple[Tensor, Tensor, Tensor]:
    sequences, labels = zip(*batch); lengths = torch.tensor([len(sequence) for sequence in sequences], dtype=torch.long)
    padded = rnn.pad_sequence([torch.tensor(sequence, dtype=torch.long) for sequence in sequences], batch_first=True, padding_value=0)
    return padded, lengths, torch.tensor(labels, dtype=torch.float32)


def build_loader(texts: list[str], labels: np.ndarray, vocab: Vocabulary, shuffle: bool) -> DataLoader:
    generator = torch.Generator().manual_seed(CONFIG["random_state"])
    return DataLoader(SequenceDataset(texts, labels, vocab), batch_size=CONFIG["batch_size"], shuffle=shuffle,
                      num_workers=0, collate_fn=collate_sequences, generator=generator)


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 128, hidden_size: int = 128, dropout: float = .3) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(dropout); self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, token_ids: Tensor, lengths: Tensor) -> Tensor:
        embedded = self.embedding(token_ids)
        packed = rnn.pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed)
        return self.classifier(self.dropout(hidden[-1])).squeeze(1)


def metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    predicted = (probabilities >= CONFIG["threshold"]).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, predicted, pos_label=1, average="binary", zero_division=0)
    return {"accuracy": float(accuracy_score(y_true, predicted)), "precision_label_1": float(precision),
            "recall_label_1": float(recall), "f1_label_1": float(f1), "roc_auc": float(roc_auc_score(y_true, probabilities)),
            "confusion_matrix": confusion_matrix(y_true, predicted, labels=[0, 1]).tolist(), "labels": [0, 1]}


@torch.no_grad()
def predict(model: LSTMClassifier, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, float]:
    model.eval(); outputs: list[np.ndarray] = []; started = perf_counter()
    for tokens, lengths, _ in loader: outputs.append(torch.sigmoid(model(tokens.to(device), lengths)).cpu().numpy())
    probabilities = np.concatenate(outputs); elapsed = perf_counter() - started
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any(): raise RuntimeError("Invalid probabilities")
    return probabilities, elapsed


def train(train_loader: DataLoader, validation_loader: DataLoader, validation_labels: np.ndarray, vocab_size: int, checkpoint: Path) -> tuple[LSTMClassifier, list[dict[str, float]], int, float]:
    set_seed(CONFIG["random_state"]); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMClassifier(vocab_size).to(device); optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
    loss_fn = nn.BCEWithLogitsLoss(); best_loss, best_epoch, stale, best_state = float("inf"), 0, 0, None; history = []; started = perf_counter()
    for epoch in range(1, CONFIG["max_epochs"] + 1):
        model.train(); loss_sum = 0.; examples = 0
        for tokens, lengths, labels in train_loader:
            optimizer.zero_grad(set_to_none=True); logits = model(tokens.to(device), lengths); loss = loss_fn(logits, labels.to(device))
            if not torch.isfinite(loss): raise RuntimeError("NaN/Inf training loss")
            loss.backward(); clip_grad_norm_(model.parameters(), CONFIG["gradient_clip_norm"]); optimizer.step()
            loss_sum += loss.item() * len(labels); examples += len(labels)
        model.eval(); val_loss_sum = 0.; val_examples = 0; outputs = []
        with torch.no_grad():
            for tokens, lengths, labels in validation_loader:
                logits = model(tokens.to(device), lengths); loss = loss_fn(logits, labels.to(device))
                if not torch.isfinite(loss): raise RuntimeError("NaN/Inf validation loss")
                val_loss_sum += loss.item() * len(labels); val_examples += len(labels); outputs.append(torch.sigmoid(logits).cpu().numpy())
        probabilities = np.concatenate(outputs); value = val_loss_sum / val_examples; epoch_metrics = metrics(validation_labels, probabilities)
        history.append({"epoch": epoch, "train_loss": loss_sum / examples, "validation_loss": value, "validation_f1": epoch_metrics["f1_label_1"], "validation_roc_auc": epoch_metrics["roc_auc"]})
        if value < best_loss: best_loss, best_epoch, stale, best_state = value, epoch, 0, copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= CONFIG["patience"]: break
    seconds = perf_counter() - started
    if best_state is None: raise RuntimeError("No LSTM checkpoint")
    model.load_state_dict(best_state); torch.save({"model_state_dict": model.state_dict(), "config": CONFIG, "vocab_size": vocab_size, "best_epoch": best_epoch}, checkpoint)
    return model, history, best_epoch, seconds


def tokenization_statistics(train_texts: list[str], validation_texts: list[str], vocab: Vocabulary) -> dict[str, Any]:
    raw_lengths = np.array([len(tokenize(text)) for text in train_texts + validation_texts])
    validation_tokens = [token for text in validation_texts for token in tokenize(text)]
    truncated = float(np.mean(raw_lengths > CONFIG["max_sequence_length"]))
    return {"tokenizer_pattern": TOKEN_RE.pattern, "max_vocab_size": CONFIG["max_vocab_size"], "vocabulary_size": len(vocab),
            "max_sequence_length": CONFIG["max_sequence_length"], "lengths_over_train_and_internal_validation": {
                "mean": float(raw_lengths.mean()), "median": float(np.median(raw_lengths)), "q90": float(np.quantile(raw_lengths, .90)), "q95": float(np.quantile(raw_lengths, .95)), "q99": float(np.quantile(raw_lengths, .99))},
            "fraction_truncated_at_max_length": truncated,
            "validation_unknown_token_fraction": float(sum(token not in vocab.token_to_id for token in validation_tokens) / len(validation_tokens)) if validation_tokens else 0.}


def save_plots(history: list[dict[str, float]], matrix: list[list[int]], output: Path) -> None:
    epochs = [row["epoch"] for row in history]; plt.rcParams.update({"font.size": 11})
    def line(values: list[float], ylabel: str, title: str, file: str, second: list[float] | None = None) -> None:
        fig, ax = plt.subplots(figsize=(7, 4.5)); ax.plot(epochs, values, marker="o", label="Train loss" if second else None)
        if second: ax.plot(epochs, second, marker="o", label="Validation loss"); ax.legend()
        ax.set(xlabel="Epoch", ylabel=ylabel, title=title); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(output / file, dpi=300); plt.close(fig)
    line([r["train_loss"] for r in history], "BCEWithLogitsLoss", "LSTM learning curves", "training_curves.png", [r["validation_loss"] for r in history])
    line([r["validation_f1"] for r in history], "F1 (label=1)", "LSTM validation F1", "f1_by_epoch.png")
    line([r["validation_roc_auc"] for r in history], "ROC-AUC", "LSTM validation ROC-AUC", "roc_auc_by_epoch.png")
    fig, ax = plt.subplots(figsize=(5, 4)); image = ax.imshow(matrix, cmap="Blues"); fig.colorbar(image, ax=ax)
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["0 (human)", "1 (AI)"], yticklabels=["0 (human)", "1 (AI)"], xlabel="Predicted label", ylabel="True label", title="LSTM: internal validation")
    for row in range(2):
        for col in range(2): ax.text(col, row, str(matrix[row][col]), ha="center", va="center")
    fig.tight_layout(); fig.savefig(output / "confusion_matrix.png", dpi=300); plt.close(fig)


def load_vocab(path: Path | str) -> Vocabulary:
    """Load the persisted train-only tokenizer vocabulary."""
    return Vocabulary.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
def load_lstm(path: Path, device: torch.device | None = None) -> LSTMClassifier:
    target = device or torch.device("cuda" if torch.cuda.is_available() else "cpu"); checkpoint = torch.load(path, map_location=target, weights_only=True)
    model = LSTMClassifier(int(checkpoint["vocab_size"]), checkpoint["config"]["embedding_dim"], checkpoint["config"]["hidden_size"], checkpoint["config"]["dropout"]).to(target)
    model.load_state_dict(checkpoint["model_state_dict"]); model.eval(); return model
def predict_texts(model: LSTMClassifier, vocab: Vocabulary, texts: list[str], device: torch.device) -> np.ndarray:
    loader = build_loader(texts, np.zeros(len(texts), dtype=np.float32), vocab, False); probabilities, _ = predict(model, loader, device); return probabilities


def run(output: Path = Path("reports/lstm"), models: Path = Path("models")) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True); models.mkdir(parents=True, exist_ok=True)
    assignments = load_baseline_assignments(Path("reports/baseline/split_assignments.csv")); train_frame, validation_frame = load_texts_for_assignments(assignments)
    if set(train_frame.id) & set(validation_frame.id): raise RuntimeError("Overlapping train/validation IDs")
    vocab = Vocabulary.build(train_frame.text.tolist()); (models / "lstm_vocab.json").write_text(json.dumps(vocab.to_dict(), ensure_ascii=False), encoding="utf-8")
    stats = tokenization_statistics(train_frame.text.tolist(), validation_frame.text.tolist(), vocab); (output / "tokenization_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    train_loader = build_loader(train_frame.text.tolist(), train_frame.label.to_numpy(), vocab, True); validation_loader = build_loader(validation_frame.text.tolist(), validation_frame.label.to_numpy(), vocab, False)
    model, history, best_epoch, training_seconds = train(train_loader, validation_loader, validation_frame.label.to_numpy(), len(vocab), models / "lstm_best.pt")
    probabilities, prediction_seconds = predict(model, validation_loader, next(model.parameters()).device); result = metrics(validation_frame.label.to_numpy(), probabilities)
    result.update({"best_epoch": best_epoch, "training_seconds": training_seconds, "prediction_seconds": prediction_seconds, "train_examples": len(train_frame), "internal_validation_examples": len(validation_frame), "device": str(next(model.parameters()).device), "official_validation_used": False, "official_test_used": False, "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)})
    pd.DataFrame(history).to_csv(output / "training_history.csv", index=False, encoding="utf-8"); (output / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame([{k: v for k, v in result.items() if k not in {"confusion_matrix", "labels"}}]).to_csv(output / "metrics.csv", index=False, encoding="utf-8")
    pd.DataFrame(result["confusion_matrix"], index=["true_0", "true_1"], columns=["pred_0", "pred_1"]).to_csv(output / "confusion_matrix.csv", encoding="utf-8")
    predictions = (probabilities >= CONFIG["threshold"]).astype(int); pd.DataFrame({"id": validation_frame.id, "true_label": validation_frame.label, "predicted_label": predictions, "probability_ai": probabilities}).to_csv(output / "validation_predictions.csv", index=False, encoding="utf-8")
    (output / "training_config.json").write_text(json.dumps({**CONFIG, "vocab_size": len(vocab), "device": str(next(model.parameters()).device)}, indent=2), encoding="utf-8")
    save_plots(history, result["confusion_matrix"], output); return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LSTM for CoAT fixed split"); parser.add_argument("--output-dir", type=Path, default=Path("reports/lstm")); parser.add_argument("--model-dir", type=Path, default=Path("models"))
    args = parser.parse_args(); print(json.dumps(run(args.output_dir, args.model_dir), ensure_ascii=False, indent=2))
if __name__ == "__main__": main()
