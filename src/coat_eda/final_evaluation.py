"""One-time final inference of fixed CoAT models on official validation only."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import torch
from datasets import load_dataset
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             precision_recall_fscore_support, precision_recall_curve, roc_auc_score, roc_curve)

from coat_eda.data_prep import load_texts_for_assignments
from coat_eda.lstm import load_lstm, load_vocab, predict_texts as lstm_predict
from coat_eda.mlp import load_mlp_predictor, predict_texts as mlp_predict
from coat_eda.mlp_regularized import load_regularized_predictor, predict_texts as regularized_predict

DATASET_ID, CONFIG, THRESHOLD = "RussianNLP/coat", "binary", 0.5
MODEL_SPECS = [("Logistic Regression", "logreg", "models/tfidf_logreg.joblib"),
               ("Original MLP", "mlp_original", "models/mlp_best.pt"),
               ("Regularized MLP", "mlp_regularized", "models/mlp_regularized_best.pt"),
               ("LSTM", "lstm", "models/lstm_best.pt")]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""): digest.update(chunk)
    return digest.hexdigest()


def score(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    """Shared, fixed-threshold metric implementation for every final model."""
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Probabilities must be finite values in [0, 1]")
    predicted = (probabilities >= THRESHOLD).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, predicted, pos_label=1, average="binary", zero_division=0)
    return {"accuracy": float(accuracy_score(y_true, predicted)), "precision_label_1": float(precision),
            "recall_label_1": float(recall), "f1_label_1": float(f1), "roc_auc": float(roc_auc_score(y_true, probabilities)),
            "average_precision": float(average_precision_score(y_true, probabilities)),
            "confusion_matrix": confusion_matrix(y_true, predicted, labels=[0, 1]).tolist()}


def prediction_frame(ids: np.ndarray, labels: np.ndarray, probabilities: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame({"id": ids, "true_label": labels, "predicted_label": (probabilities >= THRESHOLD).astype(int), "probability_ai": probabilities})
    if len(frame) != len(ids) or frame.probability_ai.isna().any(): raise ValueError("Invalid prediction frame")
    return frame


def validate_prediction_frames(frames: dict[str, pd.DataFrame]) -> None:
    reference: pd.DataFrame | None = None
    for name, frame in frames.items():
        if list(frame.columns) != ["id", "true_label", "predicted_label", "probability_ai"]: raise ValueError(f"Unexpected columns for {name}")
        if not np.isfinite(frame.probability_ai).all() or not frame.probability_ai.between(0, 1).all(): raise ValueError(f"Invalid probabilities for {name}")
        if not (frame.predicted_label.to_numpy() == (frame.probability_ai.to_numpy() >= THRESHOLD).astype(int)).all(): raise ValueError(f"Threshold mismatch for {name}")
        if reference is None: reference = frame
        elif not np.array_equal(reference.id.to_numpy(), frame.id.to_numpy()) or not np.array_equal(reference.true_label.to_numpy(), frame.true_label.to_numpy()): raise ValueError("Model outputs are misaligned")


def profile_validation(validation: pd.DataFrame) -> dict[str, Any]:
    required = {"id", "text", "label"}
    if not required.issubset(validation.columns): raise ValueError(f"Missing validation columns: {required - set(validation.columns)}")
    labels = validation.label.value_counts().sort_index().to_dict()
    if set(labels) != {0, 1}: raise ValueError(f"Expected labels 0 and 1, got {labels}")
    assignments = pd.read_csv("reports/baseline/split_assignments.csv")
    train_frame, internal_frame = load_texts_for_assignments(assignments)
    used_ids = set(train_frame.id) | set(internal_frame.id)
    used_text_hashes = {hashlib.sha256(text.encode("utf-8")).hexdigest() for text in pd.concat([train_frame.text, internal_frame.text])}
    hashes = validation.text.map(lambda text: hashlib.sha256(text.encode("utf-8")).hexdigest())
    result = {"dataset": DATASET_ID, "configuration": CONFIG, "split": "validation", "examples": int(len(validation)),
              "columns": validation.columns.tolist(), "class_counts": {str(key): int(value) for key, value in labels.items()},
              "empty_or_whitespace_texts": int((validation.text.str.strip() == "").sum()),
              "duplicate_exact_text_rows": int(len(validation) - validation.text.nunique()),
              "overlapping_ids_with_train_and_internal_validation": int(validation.id.isin(used_ids).sum()),
              "overlapping_exact_texts_with_train_and_internal_validation": int(hashes.isin(used_text_hashes).sum())}
    if any(result[key] for key in ("empty_or_whitespace_texts", "duplicate_exact_text_rows", "overlapping_ids_with_train_and_internal_validation", "overlapping_exact_texts_with_train_and_internal_validation")):
        raise ValueError(f"Official validation integrity check failed: {result}")
    return result


def evaluate_models(validation: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], list[dict[str, Any]]]:
    texts, labels, ids = validation.text.tolist(), validation.label.to_numpy(), validation.id.to_numpy()
    outputs: dict[str, pd.DataFrame] = {}; rows: list[dict[str, Any]] = []
    # All neural models are intentionally loaded on CPU for comparable reported inference conditions.
    device = torch.device("cpu")
    pipeline = joblib.load("models/tfidf_logreg.joblib")
    started = perf_counter(); classes = list(pipeline.classes_); probabilities = pipeline.predict_proba(texts)[:, classes.index(1)]; elapsed = perf_counter() - started
    outputs["logreg"] = prediction_frame(ids, labels, probabilities); rows.append({"model": "Logistic Regression", "key": "logreg", **score(labels, probabilities), "inference_seconds": elapsed, "seconds_per_text": elapsed / len(validation), "device": "cpu"})
    model, vectorizer, _ = load_mlp_predictor(Path("models/mlp_best.pt"), Path("models/tfidf_logreg.joblib"), device)
    started = perf_counter(); probabilities = mlp_predict(model, vectorizer, texts, device); elapsed = perf_counter() - started
    outputs["mlp_original"] = prediction_frame(ids, labels, probabilities); rows.append({"model": "Original MLP", "key": "mlp_original", **score(labels, probabilities), "inference_seconds": elapsed, "seconds_per_text": elapsed / len(validation), "device": "cpu"})
    model, vectorizer, _ = load_regularized_predictor(Path("models/mlp_regularized_best.pt"), Path("models/tfidf_logreg.joblib"), device)
    started = perf_counter(); probabilities = regularized_predict(model, vectorizer, texts, device); elapsed = perf_counter() - started
    outputs["mlp_regularized"] = prediction_frame(ids, labels, probabilities); rows.append({"model": "Regularized MLP", "key": "mlp_regularized", **score(labels, probabilities), "inference_seconds": elapsed, "seconds_per_text": elapsed / len(validation), "device": "cpu"})
    model, vocab = load_lstm(Path("models/lstm_best.pt"), device), load_vocab(Path("models/lstm_vocab.json"))
    started = perf_counter(); probabilities = lstm_predict(model, vocab, texts, device); elapsed = perf_counter() - started
    outputs["lstm"] = prediction_frame(ids, labels, probabilities); rows.append({"model": "LSTM", "key": "lstm", **score(labels, probabilities), "inference_seconds": elapsed, "seconds_per_text": elapsed / len(validation), "device": "cpu"})
    validate_prediction_frames(outputs); return outputs, rows


def save_figure(fig: plt.Figure, output: Path, name: str, tight: bool = True) -> None:
    if tight: fig.tight_layout()
    fig.savefig(output / f"{name}.png", dpi=300); fig.savefig(output / f"{name}.pdf"); plt.close(fig)


def create_figures(frames: dict[str, pd.DataFrame], metrics: pd.DataFrame, output: Path) -> None:
    plt.rcParams.update({"font.size": 10}); positions = np.arange(len(metrics)); width = .34
    fig, ax = plt.subplots(figsize=(7.2, 4)); ax.bar(positions-width/2, metrics.f1_label_1, width, label="F1")
    ax.bar(positions+width/2, metrics.roc_auc, width, label="ROC-AUC"); ax.set(xticks=positions, xticklabels=metrics.model, ylabel="Score", ylim=(0, 1), title="Official validation: model comparison")
    ax.legend(); ax.grid(axis="y", alpha=.25); save_figure(fig, output, "model_comparison")
    fig, ax = plt.subplots(figsize=(6.4, 4.5))
    for _, row in metrics.iterrows():
        frame = frames[row.key]; fpr, tpr, _ = roc_curve(frame.true_label, frame.probability_ai); ax.plot(fpr, tpr, label=f"{row.model} (AUC={row.roc_auc:.3f})")
    ax.plot([0,1],[0,1],"--",color="gray",label="Random"); ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curves")
    ax.legend(fontsize=8); ax.grid(alpha=.25); save_figure(fig, output, "roc_curves")
    fig, ax = plt.subplots(figsize=(6.4, 4.5))
    for _, row in metrics.iterrows():
        frame = frames[row.key]; recall, precision, _ = precision_recall_curve(frame.true_label, frame.probability_ai); ax.plot(recall, precision, label=f"{row.model} (AP={row.average_precision:.3f})")
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision-recall curves"); ax.legend(fontsize=8); ax.grid(alpha=.25); save_figure(fig, output, "pr_curves")
    fig, ax = plt.subplots(figsize=(7.2, 4)); ax.bar(metrics.model, metrics.seconds_per_text * 1000); ax.set(ylabel="Milliseconds per text", title="Inference time (CPU; preprocessing included)")
    ax.tick_params(axis="x", rotation=15); ax.grid(axis="y", alpha=.25); save_figure(fig, output, "inference_time")
    fig, axes = plt.subplots(2, 2, figsize=(7, 6));
    for ax, (_, row) in zip(axes.flat, metrics.iterrows()):
        matrix = np.array(row.confusion_matrix); image = ax.imshow(matrix, cmap="Blues", vmin=0); ax.set(title=row.model, xticks=[0,1], yticks=[0,1], xlabel="Predicted", ylabel="True")
        for r in range(2):
            for c in range(2): ax.text(c, r, str(matrix[r,c]), ha="center", va="center")
        pd.DataFrame(matrix, index=["true_0", "true_1"], columns=["pred_0", "pred_1"]).to_csv(output / f"confusion_matrix_{row.key}.csv", encoding="utf-8")
        single, single_ax = plt.subplots(figsize=(4,3.5)); single_ax.imshow(matrix, cmap="Blues"); single_ax.set(title=f"{row.model}: confusion matrix", xticks=[0,1], yticks=[0,1], xlabel="Predicted", ylabel="True")
        for r in range(2):
            for c in range(2): single_ax.text(c,r,str(matrix[r,c]),ha="center",va="center")
        save_figure(single, output, f"confusion_matrix_{row.key}")
    fig.subplots_adjust(wspace=.45, hspace=.55, right=.88)
    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=.75); save_figure(fig, output, "confusion_matrices", tight=False)


def summary_markdown(profile: dict[str, Any], metrics: pd.DataFrame) -> str:
    columns = [("model", "Model"), ("accuracy", "Accuracy"), ("precision_label_1", "Precision"),
               ("recall_label_1", "Recall"), ("f1_label_1", "F1"), ("roc_auc", "ROC-AUC"),
               ("inference_seconds", "Inference seconds")]
    header = "| " + " | ".join(label for _, label in columns) + " |"
    separator = "|" + "|".join("---" for _ in columns) + "|"
    metric_rows = ["| " + " | ".join(str(row[key]) if key == "model" else f"{row[key]:.4f}" for key, _ in columns) + " |"
                   for _, row in metrics.iterrows()]
    table = "\n".join([header, separator, *metric_rows])
    return f"""# Final official-validation evaluation\n\nThis report evaluates fixed checkpoints only; no model fitting, threshold selection, vocabulary rebuilding, or TF-IDF fitting was performed.\n\nOfficial CoAT `binary` validation contains {profile['examples']} examples, with class counts {profile['class_counts']}. Integrity checks found no empty texts, exact duplicates, or overlap with the previously used train/internal-validation data.\n\n## Metrics\n\n{table}\n\nScores use label `1` as the positive AI class and a fixed 0.5 threshold; ROC-AUC and average precision use probabilities. Inference was measured on CPU over the full validation set and includes text preprocessing/vectorization/tokenization.\n\n## Limitations\n\nThis is a single held-out official dataset split, not evidence about arbitrary contemporary LLMs. No statistical significance test was performed, so metric differences are descriptive only.\n"""


def run(output: Path = Path("reports/final_evaluation")) -> pd.DataFrame:
    output.mkdir(parents=True, exist_ok=True); validation = load_dataset(DATASET_ID, CONFIG, split="validation").to_pandas(); profile = profile_validation(validation)
    frames, rows = evaluate_models(validation); metrics = pd.DataFrame(rows)
    for key, frame in frames.items(): frame.to_csv(output / f"{key}_predictions.csv", index=False, encoding="utf-8")
    logreg = metrics.loc[metrics.key == "logreg"].iloc[0]
    for metric in ("accuracy", "f1_label_1", "roc_auc"):
        metrics[f"delta_vs_logreg_{metric}"] = metrics[metric] - logreg[metric]
    metrics.to_json(output / "metrics.json", orient="records", indent=2); metrics.to_csv(output / "metrics.csv", index=False, encoding="utf-8"); metrics.to_csv(output / "model_comparison.csv", index=False, encoding="utf-8")
    (output / "dataset_profile.json").write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
    config = {"dataset": DATASET_ID, "configuration": CONFIG, "split": "validation", "threshold": THRESHOLD, "device": "cpu", "preprocessing_included_in_inference_timing": True, "python": sys.version, "platform": platform.platform(), "torch": torch.__version__, "scikit_learn": sklearn.__version__, "models": {key: {"path": path, "sha256": sha256(Path(path))} for _, key, path in MODEL_SPECS}, "lstm_vocab": {"path": "models/lstm_vocab.json", "sha256": sha256(Path("models/lstm_vocab.json"))}}
    (output / "evaluation_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    create_figures(frames, metrics, output); (output / "evaluation_summary.md").write_text(summary_markdown(profile, metrics), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Final fixed-model evaluation on CoAT official validation"); parser.add_argument("--output-dir", type=Path, default=Path("reports/final_evaluation"))
    args = parser.parse_args(); print(run(args.output_dir)[["model", "f1_label_1", "roc_auc"]].to_string(index=False))
if __name__ == "__main__": main()
