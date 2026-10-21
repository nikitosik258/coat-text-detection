"""Leakage-free TF-IDF (word 1--2 grams) + Logistic Regression baseline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score
from sklearn.pipeline import Pipeline

from coat_eda.data_prep import (
    RANDOM_STATE, load_texts_for_assignments, make_assignments, save_assignments, validate_assignments,
)


def make_pipeline() -> Pipeline:
    """The vectorizer is fitted only when this pipeline receives training texts."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=30_000, min_df=2, sublinear_tf=True)),
        ("classifier", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, class_weight=None)),
    ])


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, positive_scores: np.ndarray) -> dict[str, Any]:
    """Return binary metrics where label 1 is the positive (AI) class."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, pos_label=1, average="binary", zero_division=0
    )
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)), "precision_label_1": float(precision),
        "recall_label_1": float(recall), "f1_label_1": float(f1),
        "roc_auc": float(roc_auc_score(y_true, positive_scores)),
        "confusion_matrix": matrix.tolist(), "labels": [0, 1],
    }


def save_confusion_matrix(matrix: list[list[int]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["0 (human)", "1 (AI)"],
           yticklabels=["0 (human)", "1 (AI)"], xlabel="Predicted label", ylabel="True label",
           title="TF-IDF + Logistic Regression: internal validation")
    for row in range(2):
        for col in range(2):
            ax.text(col, row, str(matrix[row][col]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_split_profile(train_frame: pd.DataFrame, validation_frame: pd.DataFrame, path: Path) -> None:
    rows = []
    for subset, frame in (("train", train_frame), ("internal_validation", validation_frame)):
        for label, label_frame in frame.groupby("label"):
            chars = label_frame.text.str.len()
            words = label_frame.text.str.findall(r"\S+").str.len()
            rows.append({"subset": subset, "label": int(label), "examples": int(len(label_frame)),
                         "char_length_mean": float(chars.mean()), "char_length_median": float(chars.median()),
                         "word_length_mean": float(words.mean()), "word_length_median": float(words.median())})
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8")


def run(output_dir: Path = Path("reports/baseline"), model_dir: Path = Path("models")) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    # No official validation/test data are loaded anywhere in this workflow.
    from datasets import load_dataset
    official_train = load_dataset("RussianNLP/coat", "binary", split="train")
    assignments = make_assignments(official_train)
    validate_assignments(assignments)
    save_assignments(assignments, output_dir)
    train_frame, validation_frame = load_texts_for_assignments(assignments)
    if set(train_frame.id) & set(validation_frame.id):
        raise RuntimeError("ID overlap detected immediately before fitting")
    save_split_profile(train_frame, validation_frame, output_dir / "prepared_split_profile.csv")

    pipeline = make_pipeline()
    started = perf_counter()
    pipeline.fit(train_frame.text, train_frame.label)
    fit_seconds = perf_counter() - started
    started = perf_counter()
    predictions = pipeline.predict(validation_frame.text)
    probabilities = pipeline.predict_proba(validation_frame.text)[:, 1]
    predict_seconds = perf_counter() - started
    metrics = calculate_metrics(validation_frame.label.to_numpy(), predictions, probabilities)
    metrics.update({"fit_seconds": fit_seconds, "predict_seconds": predict_seconds,
                    "train_examples": len(train_frame), "internal_validation_examples": len(validation_frame),
                    "official_validation_used": False, "official_test_used": False,
                    "tfidf_vocabulary_size": len(pipeline.named_steps["tfidf"].vocabulary_)})
    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame([{key: value for key, value in metrics.items() if key not in {"confusion_matrix", "labels"}}]).to_csv(
        output_dir / "metrics.csv", index=False, encoding="utf-8"
    )
    pd.DataFrame(metrics["confusion_matrix"], index=["true_0", "true_1"], columns=["pred_0", "pred_1"]).to_csv(
        output_dir / "confusion_matrix.csv", encoding="utf-8"
    )
    save_confusion_matrix(metrics["confusion_matrix"], output_dir / "confusion_matrix.png")
    joblib.dump(pipeline, model_dir / "tfidf_logreg.joblib")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the CoAT TF-IDF + Logistic Regression baseline")
    parser.add_argument("--output-dir", type=Path, default=Path("reports/baseline"))
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir, args.model_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
