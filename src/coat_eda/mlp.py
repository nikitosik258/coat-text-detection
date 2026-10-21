"""PyTorch MLP trained on the fixed CoAT split and baseline TF-IDF features."""
from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from coat_eda.data_prep import RANDOM_STATE, load_texts_for_assignments, validate_assignments

INPUT_DIM = 30_000
CONFIG = {"input_dim": INPUT_DIM, "hidden_dim_1": 256, "hidden_dim_2": 64, "dropout_1": 0.3,
          "dropout_2": 0.2, "learning_rate": 0.001, "batch_size": 64, "max_epochs": 20,
          "patience": 4, "random_state": RANDOM_STATE, "threshold": 0.5}


def set_seed(seed: int = RANDOM_STATE) -> None:
    """Set all relevant PRNGs before model construction and training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False


class MLP(nn.Module):
    """30000 -> 256 -> 64 -> 1 logit binary classifier."""
    def __init__(self, input_dim: int = INPUT_DIM) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1),
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.network(features).squeeze(1)


class SparseIndexDataset(Dataset[int]):
    """Stores sparse matrices; densification happens only for a single mini-batch."""
    def __init__(self, features: csr_matrix, labels: np.ndarray) -> None:
        self.features = features.astype(np.float32)
        self.labels = np.asarray(labels, dtype=np.float32)

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, index: int) -> int:
        return index

    def collate(self, indices: list[int]) -> tuple[Tensor, Tensor]:
        batch = self.features[indices].toarray()  # bounded: batch_size x 30,000 only
        return torch.from_numpy(batch), torch.from_numpy(self.labels[indices])


def build_loader(features: csr_matrix, labels: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = SparseIndexDataset(features, labels)
    generator = torch.Generator().manual_seed(RANDOM_STATE)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0,
                      collate_fn=dataset.collate, generator=generator)


def load_baseline_assignments(path: Path) -> pd.DataFrame:
    assignments = pd.read_csv(path)
    validate_assignments(assignments)
    return assignments


def load_baseline_vectorizer(model_path: Path):
    """Retrieve frozen TF-IDF fitted by baseline; never refit on MLP data."""
    pipeline = joblib.load(model_path)
    if "tfidf" not in pipeline.named_steps:
        raise ValueError("Saved baseline pipeline lacks the fitted 'tfidf' step")
    vectorizer = pipeline.named_steps["tfidf"]
    if not hasattr(vectorizer, "vocabulary_") or len(vectorizer.vocabulary_) != INPUT_DIM:
        raise ValueError(f"Expected a fitted {INPUT_DIM}-feature TF-IDF vectorizer")
    return vectorizer


def calculate_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    predictions = (probabilities >= CONFIG["threshold"]).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, predictions, pos_label=1, average="binary", zero_division=0
    )
    return {"accuracy": float(accuracy_score(y_true, predictions)), "precision_label_1": float(precision),
            "recall_label_1": float(recall), "f1_label_1": float(f1),
            "roc_auc": float(roc_auc_score(y_true, probabilities)),
            "confusion_matrix": confusion_matrix(y_true, predictions, labels=[0, 1]).tolist(), "labels": [0, 1]}


@torch.no_grad()
def predict_probabilities(model: MLP, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, float]:
    model.eval()
    probabilities: list[np.ndarray] = []
    started = perf_counter()
    for features, _ in loader:
        logits = model(features.to(device)).cpu()
        probabilities.append(torch.sigmoid(logits).numpy())
    elapsed = perf_counter() - started
    output = np.concatenate(probabilities)
    if not np.isfinite(output).all() or ((output < 0) | (output > 1)).any():
        raise RuntimeError("Invalid probabilities produced by model")
    return output, elapsed


def save_curves(history: list[dict[str, float]], output_dir: Path) -> None:
    epochs = [entry["epoch"] for entry in history]
    plt.rcParams.update({"font.size": 11})
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(epochs, [entry["train_loss"] for entry in history], marker="o", label="Train loss")
    ax.plot(epochs, [entry["validation_loss"] for entry in history], marker="o", label="Validation loss")
    ax.set(xlabel="Epoch", ylabel="BCEWithLogitsLoss", title="MLP learning curves")
    ax.legend(); ax.grid(alpha=0.25); fig.tight_layout(); fig.savefig(output_dir / "training_curves.png", dpi=300); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(epochs, [entry["validation_f1"] for entry in history], marker="o", color="#1f77b4")
    ax.set(xlabel="Epoch", ylabel="F1 (label=1)", title="MLP validation F1 by epoch")
    ax.grid(alpha=0.25); fig.tight_layout(); fig.savefig(output_dir / "f1_by_epoch.png", dpi=300); plt.close(fig)


def save_confusion_matrix(matrix: list[list[int]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(matrix, cmap="Blues"); fig.colorbar(image, ax=ax)
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["0 (human)", "1 (AI)"],
           yticklabels=["0 (human)", "1 (AI)"], xlabel="Predicted label", ylabel="True label",
           title="MLP: internal validation")
    for row in range(2):
        for col in range(2): ax.text(col, row, str(matrix[row][col]), ha="center", va="center")
    fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)


def train_mlp(train_features: csr_matrix, train_labels: np.ndarray, validation_features: csr_matrix,
              validation_labels: np.ndarray, model_path: Path) -> tuple[MLP, list[dict[str, float]], int, float]:
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(train_features.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    criterion = nn.BCEWithLogitsLoss()
    train_loader = build_loader(train_features, train_labels, CONFIG["batch_size"], shuffle=True)
    validation_loader = build_loader(validation_features, validation_labels, CONFIG["batch_size"], shuffle=False)
    best_loss, best_epoch, stale_epochs, best_state = float("inf"), 0, 0, None
    history: list[dict[str, float]] = []
    started = perf_counter()
    for epoch in range(1, CONFIG["max_epochs"] + 1):
        model.train(); train_loss_sum = 0.0; train_count = 0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features); loss = criterion(logits, labels)
            if not torch.isfinite(loss): raise RuntimeError("NaN/Inf training loss")
            loss.backward(); optimizer.step()
            train_loss_sum += loss.item() * len(labels); train_count += len(labels)
        model.eval(); validation_loss_sum = 0.0; validation_count = 0; probs: list[np.ndarray] = []
        with torch.no_grad():
            for features, labels in validation_loader:
                logits = model(features.to(device)); loss = criterion(logits, labels.to(device))
                if not torch.isfinite(loss): raise RuntimeError("NaN/Inf validation loss")
                validation_loss_sum += loss.item() * len(labels); validation_count += len(labels)
                probs.append(torch.sigmoid(logits).cpu().numpy())
        val_probabilities = np.concatenate(probs)
        val_loss = validation_loss_sum / validation_count
        val_f1 = calculate_metrics(validation_labels, val_probabilities)["f1_label_1"]
        history.append({"epoch": epoch, "train_loss": train_loss_sum / train_count,
                        "validation_loss": val_loss, "validation_f1": val_f1})
        if val_loss < best_loss:
            best_loss, best_epoch, stale_epochs, best_state = val_loss, epoch, 0, copy.deepcopy(model.state_dict())
        else:
            stale_epochs += 1
            if stale_epochs >= CONFIG["patience"]: break
    elapsed = perf_counter() - started
    if best_state is None: raise RuntimeError("No model checkpoint was created")
    model.load_state_dict(best_state)
    torch.save({"model_state_dict": model.state_dict(), "config": CONFIG, "best_epoch": best_epoch}, model_path)
    return model, history, best_epoch, elapsed


def load_mlp_for_prediction(model_path: Path, device: torch.device | None = None) -> MLP:
    """Reload a trained MLP checkpoint for inference without retraining."""
    target = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=target, weights_only=True)
    model = MLP(int(checkpoint["config"]["input_dim"])).to(target)
    model.load_state_dict(checkpoint["model_state_dict"]); model.eval()
    return model


def load_mlp_predictor(model_path: Path = Path("models/mlp_best.pt"),
                       baseline_model_path: Path = Path("models/tfidf_logreg.joblib"),
                       device: torch.device | None = None) -> tuple[MLP, Any, torch.device]:
    """Load frozen TF-IDF and MLP objects for inference without fitting."""
    target = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return load_mlp_for_prediction(model_path, target), load_baseline_vectorizer(baseline_model_path), target


def predict_texts(model: MLP, vectorizer: Any, texts: list[str], device: torch.device,
                  batch_size: int = CONFIG["batch_size"]) -> np.ndarray:
    """Return label-1 probabilities from already fitted inference objects."""
    features = vectorizer.transform(texts)
    loader = build_loader(features, np.zeros(len(texts), dtype=np.float32), batch_size, shuffle=False)
    probabilities, _ = predict_probabilities(model, loader, device)
    return probabilities


def run(output_dir: Path = Path("reports/mlp"), model_dir: Path = Path("models"),
        assignments_path: Path = Path("reports/baseline/split_assignments.csv"),
        baseline_model_path: Path = Path("models/tfidf_logreg.joblib")) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True); model_dir.mkdir(parents=True, exist_ok=True)
    assignments = load_baseline_assignments(assignments_path)
    train_frame, validation_frame = load_texts_for_assignments(assignments)
    if set(train_frame.id) & set(validation_frame.id): raise RuntimeError("Overlapping IDs in persisted split")
    vectorizer = load_baseline_vectorizer(baseline_model_path)
    train_features = vectorizer.transform(train_frame.text)
    validation_features = vectorizer.transform(validation_frame.text)
    model_path = model_dir / "mlp_best.pt"
    model, history, best_epoch, training_seconds = train_mlp(
        train_features, train_frame.label.to_numpy(), validation_features, validation_frame.label.to_numpy(), model_path
    )
    device = next(model.parameters()).device
    validation_loader = build_loader(validation_features, validation_frame.label.to_numpy(), CONFIG["batch_size"], shuffle=False)
    probabilities, prediction_seconds = predict_probabilities(model, validation_loader, device)
    metrics = calculate_metrics(validation_frame.label.to_numpy(), probabilities)
    metrics.update({"best_epoch": best_epoch, "training_seconds": training_seconds, "prediction_seconds": prediction_seconds,
                    "train_examples": len(train_frame), "internal_validation_examples": len(validation_frame),
                    "device": str(device), "official_validation_used": False, "official_test_used": False})
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False, encoding="utf-8")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame([{key: value for key, value in metrics.items() if key not in {"confusion_matrix", "labels"}}]).to_csv(output_dir / "metrics.csv", index=False, encoding="utf-8")
    pd.DataFrame(metrics["confusion_matrix"], index=["true_0", "true_1"], columns=["pred_0", "pred_1"]).to_csv(output_dir / "confusion_matrix.csv", encoding="utf-8")
    (output_dir / "training_config.json").write_text(json.dumps({**CONFIG, "device": str(device), "tfidf_source": str(baseline_model_path), "assignments_source": str(assignments_path)}, indent=2), encoding="utf-8")
    save_curves(history, output_dir); save_confusion_matrix(metrics["confusion_matrix"], output_dir / "confusion_matrix.png")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CoAT MLP on frozen baseline TF-IDF features")
    parser.add_argument("--output-dir", type=Path, default=Path("reports/mlp")); parser.add_argument("--model-dir", type=Path, default=Path("models"))
    args = parser.parse_args(); print(json.dumps(run(args.output_dir, args.model_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
