"""Reproducible EDA for RussianNLP/coat, configuration ``binary``.

The script only reads Hugging Face data and writes derived reports.  It never
modifies examples, removes duplicates, or combines official dataset splits.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import ClassLabel, Dataset, DatasetDict, load_dataset

DATASET_ID = "RussianNLP/coat"
CONFIG = "binary"
EXPECTED_SPLITS = ("train", "validation", "test")
WORD_RE = re.compile(r"\S+", flags=re.UNICODE)


def json_default(value: Any) -> Any:
    """Convert NumPy values to values accepted by the standard JSON encoder."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Not JSON serializable: {type(value)!r}")


def write_json(value: Any, path: Path) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")


def find_columns(dataset: Dataset) -> tuple[str, str]:
    """Identify the textual and class-label columns from dataset features."""
    string_columns = [name for name, feature in dataset.features.items() if getattr(feature, "dtype", None) == "string"]
    text_column = "text" if "text" in string_columns else (string_columns[0] if len(string_columns) == 1 else None)
    label_column = "label" if "label" in dataset.column_names else None
    if label_column is None:
        class_labels = [name for name, feature in dataset.features.items() if isinstance(feature, ClassLabel)]
        label_column = class_labels[0] if len(class_labels) == 1 else None
    if not text_column or not label_column:
        raise ValueError(f"Could not infer text/label columns. Features: {dataset.features}")
    return text_column, label_column


def label_name(feature: Any, value: Any) -> str:
    if isinstance(feature, ClassLabel):
        return feature.int2str(int(value))
    return str(value)


def text_key(text: str) -> str:
    """Hash exact text to keep report files compact and avoid copying data."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def describe(values: list[int]) -> dict[str, float | int]:
    series = pd.Series(values, dtype="float64")
    return {
        "min": int(series.min()), "max": int(series.max()),
        "mean": float(series.mean()), "median": float(series.median()),
    }


def analyze_split(dataset: Dataset, split: str, text_column: str, label_column: str) -> tuple[dict[str, Any], list[dict[str, Any]], set[str], dict[str, set[str]]]:
    """Calculate split-local statistics while preserving the original Dataset."""
    label_feature = dataset.features[label_column]
    classes: Counter[str] = Counter()
    char_lengths: list[int] = []
    word_lengths: list[int] = []
    hashes: list[str] = []
    labels_by_hash: dict[str, set[str]] = defaultdict(set)
    empty = 0

    for row in dataset:
        raw = row[text_column]
        text = raw if isinstance(raw, str) else ""
        label = label_name(label_feature, row[label_column])
        key = text_key(text)
        classes[label] += 1
        char_lengths.append(len(text))
        word_lengths.append(len(WORD_RE.findall(text)))
        hashes.append(key)
        labels_by_hash[key].add(label)
        empty += int(not text.strip())

    duplicate_rows = len(hashes) - len(set(hashes))
    rows = []
    for unit, values in (("characters", char_lengths), ("words", word_lengths)):
        rows.append({"split": split, "unit": unit, **describe(values)})
    result = {
        "split": split,
        "examples": len(dataset),
        "empty_or_whitespace_texts": empty,
        "unique_exact_texts": len(set(hashes)),
        "duplicate_rows_within_split": duplicate_rows,
        "duplicate_text_groups_within_split": sum(count > 1 for count in Counter(hashes).values()),
        "class_counts": dict(sorted(classes.items())),
    }
    return result, rows, set(hashes), labels_by_hash


def plot_class_distribution(class_df: pd.DataFrame, output: Path) -> None:
    pivot = class_df.pivot(index="class", columns="split", values="count").fillna(0)
    ax = pivot.plot(kind="bar", figsize=(10, 6))
    ax.set_title("CoAT: class distribution by official split")
    ax.set_xlabel("Class")
    ax.set_ylabel("Number of texts")
    ax.legend(title="Split")
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()


def plot_length_histograms(lengths: dict[str, dict[str, list[int]]], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for unit, axis in zip(("characters", "words"), axes):
        for split, split_lengths in lengths.items():
            values = split_lengths[unit]
            axis.hist(values, bins=60, alpha=0.45, label=split, density=True)
        axis.set_title(f"Text length distribution ({unit})")
        axis.set_xlabel(unit.capitalize())
        axis.set_ylabel("Density")
        axis.legend()
    fig.suptitle("CoAT text lengths by official split")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def run(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    data: DatasetDict = load_dataset(DATASET_ID, CONFIG)
    missing = [split for split in EXPECTED_SPLITS if split not in data]
    if missing:
        raise ValueError(f"Expected official splits are absent: {missing}; available: {list(data)}")

    text_column, label_column = find_columns(data["train"])
    schema = {
        "dataset": DATASET_ID, "configuration": CONFIG,
        "splits": {split: len(data[split]) for split in data},
        "columns": data["train"].column_names,
        "features": {name: str(feature) for name, feature in data["train"].features.items()},
        "text_column": text_column, "label_column": label_column,
        "label_names": getattr(data["train"].features[label_column], "names", None),
    }
    write_json(schema, output_dir / "schema.json")

    split_results, length_rows, sets, all_labels = [], [], {}, defaultdict(set)
    class_rows, lengths_for_plot = [], {}
    for split in EXPECTED_SPLITS:
        result, rows, hashes, labels = analyze_split(data[split], split, text_column, label_column)
        split_results.append(result)
        length_rows.extend(rows)
        sets[split] = hashes
        for key, label_set in labels.items():
            all_labels[key].update(label_set)
        for label, count in result["class_counts"].items():
            class_rows.append({"split": split, "class": label, "count": count})
        # Recompute only lengths required for the graphic; data remain unmodified.
        lengths_for_plot[split] = {"characters": [len(x if isinstance(x, str) else "") for x in data[split][text_column]],
                                   "words": [len(WORD_RE.findall(x if isinstance(x, str) else "")) for x in data[split][text_column]]}

    class_df = pd.DataFrame(class_rows)
    length_df = pd.DataFrame(length_rows)
    class_df.to_csv(output_dir / "class_distribution.csv", index=False, encoding="utf-8")
    length_df.to_csv(output_dir / "length_statistics.csv", index=False, encoding="utf-8")
    pd.DataFrame(split_results).drop(columns=["class_counts"]).to_csv(output_dir / "quality_checks.csv", index=False, encoding="utf-8")

    overlap_rows = []
    for left, right in combinations(EXPECTED_SPLITS, 2):
        overlap_rows.append({"split_a": left, "split_b": right, "shared_unique_exact_texts": len(sets[left] & sets[right]),
                             "share_of_a_unique_pct": 100 * len(sets[left] & sets[right]) / len(sets[left]) if sets[left] else 0,
                             "share_of_b_unique_pct": 100 * len(sets[left] & sets[right]) / len(sets[right]) if sets[right] else 0})
    overlap_df = pd.DataFrame(overlap_rows)
    overlap_df.to_csv(output_dir / "split_overlaps.csv", index=False, encoding="utf-8")

    conflict_rows = [{"text_sha256": key, "labels": " | ".join(sorted(labels)), "label_count": len(labels)}
                     for key, labels in all_labels.items() if len(labels) > 1]
    pd.DataFrame(conflict_rows, columns=["text_sha256", "labels", "label_count"]).to_csv(output_dir / "conflicting_labels.csv", index=False, encoding="utf-8")

    train_counts = dict(class_df[class_df["split"] == "train"].set_index("class")["count"])
    min_class = min(train_counts.values())
    maximum_balanced = 2 * min_class
    feasible_targets = [n for n in range(20_000, 30_001, 2) if n // 2 <= min_class]
    recommendation = max(feasible_targets) if feasible_targets else None
    feasibility = pd.DataFrame([{"train_class": label, "available_examples": count, "required_per_class_for_20000": 10_000,
                                  "required_per_class_for_30000": 15_000} for label, count in sorted(train_counts.items())])
    feasibility.to_csv(output_dir / "balanced_subsample_feasibility.csv", index=False, encoding="utf-8")

    plot_class_distribution(class_df, output_dir / "class_distribution.png")
    plot_length_histograms(lengths_for_plot, output_dir / "length_distributions.png")
    summary = {"schema": schema, "split_results": split_results, "cross_split_overlaps": overlap_rows,
               "conflicting_label_texts": len(conflict_rows),
               "balanced_train_subset": {"only_train_used": True, "maximum_balanced_examples": maximum_balanced,
                  "feasible_in_20000_to_30000": bool(feasible_targets), "largest_recommended_target": recommendation,
                  "examples_per_class_at_recommendation": recommendation // 2 if recommendation else None}}
    write_json(summary, output_dir / "summary.json")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EDA for RussianNLP/coat (binary)")
    parser.add_argument("--output-dir", type=Path, default=Path("reports/eda"))
    args = parser.parse_args()
    summary = run(args.output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=json_default))


if __name__ == "__main__":
    main()
