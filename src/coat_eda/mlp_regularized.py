"""Regularized MLP experiment on the fixed CoAT split and frozen TF-IDF."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score
from torch import Tensor, nn
from torch.utils.data import DataLoader

from coat_eda.data_prep import load_texts_for_assignments
from coat_eda.mlp import (
    INPUT_DIM, build_loader, load_baseline_assignments, load_baseline_vectorizer, set_seed,
)

CONFIG = {"input_dim": INPUT_DIM, "hidden_dim_1": 128, "hidden_dim_2": 32, "dropout_1": 0.4,
          "dropout_2": 0.4, "learning_rate": 0.0003, "weight_decay": 1e-4, "batch_size": 64,
          "max_epochs": 20, "patience": 4, "random_state": 42, "threshold": 0.5}


class RegularizedMLP(nn.Module):
    """30000 -> 128 -> 32 -> 1 logit architecture fixed for this experiment."""
    def __init__(self, input_dim: int = INPUT_DIM) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, 32), nn.ReLU(), nn.Dropout(0.4), nn.Linear(32, 1),
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.network(features).squeeze(1)


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
def predict_probabilities(model: RegularizedMLP, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, float]:
    model.eval(); outputs: list[np.ndarray] = []; started = perf_counter()
    for features, _ in loader:
        outputs.append(torch.sigmoid(model(features.to(device))).cpu().numpy())
    elapsed = perf_counter() - started
    probabilities = np.concatenate(outputs)
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise RuntimeError("Invalid probabilities")
    return probabilities, elapsed


def train_regularized_mlp(train_features: csr_matrix, train_labels: np.ndarray, validation_features: csr_matrix,
                          validation_labels: np.ndarray, model_path: Path) -> tuple[RegularizedMLP, list[dict[str, float]], int, float]:
    set_seed(CONFIG["random_state"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RegularizedMLP(train_features.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
    criterion = nn.BCEWithLogitsLoss()
    train_loader = build_loader(train_features, train_labels, CONFIG["batch_size"], shuffle=True)
    validation_loader = build_loader(validation_features, validation_labels, CONFIG["batch_size"], shuffle=False)
    best_loss, best_epoch, stale_epochs, best_state = float("inf"), 0, 0, None
    history: list[dict[str, float]] = []; started = perf_counter()
    for epoch in range(1, CONFIG["max_epochs"] + 1):
        model.train(); total_loss = 0.0; total_examples = 0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True); logits = model(features); loss = criterion(logits, labels)
            if not torch.isfinite(loss): raise RuntimeError("NaN/Inf training loss")
            loss.backward(); optimizer.step(); total_loss += loss.item() * len(labels); total_examples += len(labels)
        model.eval(); validation_loss_sum = 0.0; validation_count = 0; outputs: list[np.ndarray] = []
        with torch.no_grad():
            for features, labels in validation_loader:
                logits = model(features.to(device)); loss = criterion(logits, labels.to(device))
                if not torch.isfinite(loss): raise RuntimeError("NaN/Inf validation loss")
                validation_loss_sum += loss.item() * len(labels); validation_count += len(labels)
                outputs.append(torch.sigmoid(logits).cpu().numpy())
        probabilities = np.concatenate(outputs); validation_loss = validation_loss_sum / validation_count
        epoch_metrics = calculate_metrics(validation_labels, probabilities)
        history.append({"epoch": epoch, "train_loss": total_loss / total_examples, "validation_loss": validation_loss,
                        "validation_f1": epoch_metrics["f1_label_1"], "validation_roc_auc": epoch_metrics["roc_auc"]})
        if validation_loss < best_loss:
            best_loss, best_epoch, stale_epochs, best_state = validation_loss, epoch, 0, copy.deepcopy(model.state_dict())
        else:
            stale_epochs += 1
            if stale_epochs >= CONFIG["patience"]: break
    training_seconds = perf_counter() - started
    if best_state is None: raise RuntimeError("No checkpoint created")
    model.load_state_dict(best_state)
    torch.save({"model_state_dict": model.state_dict(), "config": CONFIG, "best_epoch": best_epoch}, model_path)
    return model, history, best_epoch, training_seconds


def load_regularized_mlp(model_path: Path, device: torch.device | None = None) -> RegularizedMLP:
    target = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=target, weights_only=True)
    model = RegularizedMLP(int(checkpoint["config"]["input_dim"])).to(target)
    model.load_state_dict(checkpoint["model_state_dict"]); model.eval()
    return model


def load_regularized_predictor(model_path: Path = Path("models/mlp_regularized_best.pt"),
                               baseline_model_path: Path = Path("models/tfidf_logreg.joblib"),
                               device: torch.device | None = None) -> tuple[RegularizedMLP, Any, torch.device]:
    target = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return load_regularized_mlp(model_path, target), load_baseline_vectorizer(baseline_model_path), target


def predict_texts(model: RegularizedMLP, vectorizer: Any, texts: list[str], device: torch.device) -> np.ndarray:
    features = vectorizer.transform(texts)
    loader = build_loader(features, np.zeros(len(texts), dtype=np.float32), CONFIG["batch_size"], shuffle=False)
    probabilities, _ = predict_probabilities(model, loader, device)
    return probabilities


def save_figures(history: list[dict[str, float]], matrix: list[list[int]], output_dir: Path) -> None:
    epochs = [row["epoch"] for row in history]; plt.rcParams.update({"font.size": 11})
    fig, ax = plt.subplots(figsize=(7, 4.5)); ax.plot(epochs, [r["train_loss"] for r in history], marker="o", label="Train loss")
    ax.plot(epochs, [r["validation_loss"] for r in history], marker="o", label="Validation loss")
    ax.set(xlabel="Epoch", ylabel="BCEWithLogitsLoss", title="Regularized MLP learning curves"); ax.legend(); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(output_dir / "training_curves.png", dpi=300); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4.5)); ax.plot(epochs, [r["validation_f1"] for r in history], marker="o")
    ax.set(xlabel="Epoch", ylabel="F1 (label=1)", title="Regularized MLP validation F1"); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(output_dir / "f1_by_epoch.png", dpi=300); plt.close(fig)
    fig, ax = plt.subplots(figsize=(5, 4)); image = ax.imshow(matrix, cmap="Blues"); fig.colorbar(image, ax=ax)
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["0 (human)", "1 (AI)"], yticklabels=["0 (human)", "1 (AI)"],
           xlabel="Predicted label", ylabel="True label", title="Regularized MLP: internal validation")
    for row in range(2):
        for col in range(2): ax.text(col, row, str(matrix[row][col]), ha="center", va="center")
    fig.tight_layout(); fig.savefig(output_dir / "confusion_matrix.png", dpi=300); plt.close(fig)


def create_comparison(output_dir: Path, regularized: dict[str, Any]) -> pd.DataFrame:
    baseline = json.loads(Path("reports/baseline/metrics.json").read_text(encoding="utf-8"))
    original = json.loads(Path("reports/mlp/metrics.json").read_text(encoding="utf-8"))
    sources = [("Logistic Regression", baseline, "fit_seconds"), ("Original MLP", original, "training_seconds"),
               ("Regularized MLP", regularized, "training_seconds")]
    rows = []
    for name, values, time_key in sources:
        row = {"model": name, "accuracy": values["accuracy"], "precision": values["precision_label_1"],
               "recall": values["recall_label_1"], "f1": values["f1_label_1"], "roc_auc": values["roc_auc"],
               "training_seconds": values[time_key]}
        rows.append(row)
    frame = pd.DataFrame(rows)
    original_row = frame.loc[frame.model == "Original MLP"].iloc[0]
    for metric in ["accuracy", "precision", "recall", "f1", "roc_auc", "training_seconds"]:
        frame[f"delta_vs_original_mlp_{metric}"] = frame[metric] - original_row[metric]
    frame.to_csv(output_dir / "model_comparison.csv", index=False, encoding="utf-8")
    positions = np.arange(len(frame)); width = .34
    fig, ax = plt.subplots(figsize=(8, 4.8)); ax.bar(positions - width / 2, frame.f1, width, label="F1")
    ax.bar(positions + width / 2, frame.roc_auc, width, label="ROC-AUC")
    ax.set(xticks=positions, xticklabels=frame.model, ylabel="Score", title="Internal validation: model comparison", ylim=(0, 1))
    ax.legend(); ax.grid(axis="y", alpha=.25); fig.tight_layout(); fig.savefig(output_dir / "model_comparison.png", dpi=300); plt.close(fig)
    return frame


def run(output_dir: Path = Path("reports/mlp_regularized"), model_dir: Path = Path("models")) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True); model_dir.mkdir(parents=True, exist_ok=True)
    assignments_path = Path("reports/baseline/split_assignments.csv"); baseline_model_path = Path("models/tfidf_logreg.joblib")
    assignments = load_baseline_assignments(assignments_path)
    train_frame, validation_frame = load_texts_for_assignments(assignments)
    if set(train_frame.id) & set(validation_frame.id): raise RuntimeError("Overlapping persisted IDs")
    vectorizer = load_baseline_vectorizer(baseline_model_path)
    train_features, validation_features = vectorizer.transform(train_frame.text), vectorizer.transform(validation_frame.text)
    model, history, best_epoch, training_seconds = train_regularized_mlp(
        train_features, train_frame.label.to_numpy(), validation_features, validation_frame.label.to_numpy(), model_dir / "mlp_regularized_best.pt"
    )
    probabilities, prediction_seconds = predict_probabilities(model, build_loader(validation_features, validation_frame.label.to_numpy(), CONFIG["batch_size"], False), next(model.parameters()).device)
    metrics = calculate_metrics(validation_frame.label.to_numpy(), probabilities)
    metrics.update({"best_epoch": best_epoch, "training_seconds": training_seconds, "prediction_seconds": prediction_seconds,
                    "train_examples": len(train_frame), "internal_validation_examples": len(validation_frame),
                    "device": str(next(model.parameters()).device), "official_validation_used": False, "official_test_used": False})
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False, encoding="utf-8")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame([{k: v for k, v in metrics.items() if k not in {"confusion_matrix", "labels"}}]).to_csv(output_dir / "metrics.csv", index=False, encoding="utf-8")
    pd.DataFrame(metrics["confusion_matrix"], index=["true_0", "true_1"], columns=["pred_0", "pred_1"]).to_csv(output_dir / "confusion_matrix.csv", encoding="utf-8")
    (output_dir / "training_config.json").write_text(json.dumps({**CONFIG, "tfidf_source": str(baseline_model_path), "assignments_source": str(assignments_path)}, indent=2), encoding="utf-8")
    save_figures(history, metrics["confusion_matrix"], output_dir); create_comparison(output_dir, metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fixed regularized CoAT MLP experiment")
    parser.add_argument("--output-dir", type=Path, default=Path("reports/mlp_regularized")); parser.add_argument("--model-dir", type=Path, default=Path("models"))
    args = parser.parse_args(); print(json.dumps(run(args.output_dir, args.model_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
