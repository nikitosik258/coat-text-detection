"""Deterministic train-only preparation shared by all future model baselines."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset, load_dataset
from sklearn.model_selection import train_test_split

DATASET_ID = "RussianNLP/coat"
CONFIG = "binary"
RANDOM_STATE = 42
SAMPLE_PER_CLASS = 15_000
ASSIGNMENT_COLUMNS = ["id", "label", "subset"]


def make_assignments(train: Dataset, sample_per_class: int = SAMPLE_PER_CLASS, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """Select equal numbers per class then create a stratified 80/20 split.

    Only the official train split is accepted. IDs, rather than copied text, are
    saved so subsequent models can reconstruct precisely the same examples.
    """
    frame = train.select_columns(["id", "label"]).to_pandas()
    available_labels = sorted(frame["label"].unique().tolist())
    if available_labels != [0, 1]:
        raise ValueError(f"Expected binary labels [0, 1] in official train, got {available_labels}")
    selected = []
    rng = np.random.default_rng(random_state)
    for label in available_labels:
        candidates = frame.loc[frame["label"] == label]
        if len(candidates) < sample_per_class:
            raise ValueError(f"Class {label} has {len(candidates)}, fewer than {sample_per_class}")
        chosen_positions = rng.choice(len(candidates), size=sample_per_class, replace=False)
        selected.append(candidates.iloc[chosen_positions])
    sampled = pd.concat(selected, ignore_index=True)
    train_part, validation_part = train_test_split(
        sampled, test_size=0.20, random_state=random_state, stratify=sampled["label"], shuffle=True
    )
    assignment = pd.concat([
        train_part.assign(subset="train"),
        validation_part.assign(subset="internal_validation"),
    ], ignore_index=True)[ASSIGNMENT_COLUMNS]
    validate_assignments(assignment, sample_per_class=sample_per_class)
    return assignment.sort_values(["subset", "id"]).reset_index(drop=True)


def validate_assignments(assignments: pd.DataFrame, sample_per_class: int = SAMPLE_PER_CLASS) -> None:
    """Fail loudly if an assignment cannot support leakage-free comparison."""
    if list(assignments.columns) != ASSIGNMENT_COLUMNS:
        raise ValueError(f"Expected columns {ASSIGNMENT_COLUMNS}")
    if len(assignments) != sample_per_class * 2 or assignments["id"].nunique() != len(assignments):
        raise ValueError("Assignments must contain exactly one row per selected ID")
    expected_subsets = {"train", "internal_validation"}
    if set(assignments["subset"]) != expected_subsets:
        raise ValueError("Assignments must contain train and internal_validation only")
    train_ids = set(assignments.loc[assignments.subset == "train", "id"])
    valid_ids = set(assignments.loc[assignments.subset == "internal_validation", "id"])
    if train_ids & valid_ids:
        raise ValueError("Training and internal validation IDs overlap")
    counts = assignments.groupby(["subset", "label"]).size()
    expected = {("train", 0): sample_per_class * 0.8, ("train", 1): sample_per_class * 0.8,
                ("internal_validation", 0): sample_per_class * 0.2, ("internal_validation", 1): sample_per_class * 0.2}
    if any(counts.get(key, 0) != value for key, value in expected.items()):
        raise ValueError(f"Unexpected stratified class counts: {counts.to_dict()}")


def save_assignments(assignments: pd.DataFrame, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "split_assignments.csv"
    assignments.to_csv(path, index=False, encoding="utf-8")
    metadata = {
        "dataset": DATASET_ID, "configuration": CONFIG, "source_official_split": "train",
        "random_state": RANDOM_STATE, "sample_per_class": SAMPLE_PER_CLASS,
        "subsets": assignments.groupby("subset").size().to_dict(),
        "class_counts": {subset: {str(label): int(count) for label, count in group.groupby("label").size().items()}
                         for subset, group in assignments.groupby("subset")},
    }
    (output_dir / "split_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_texts_for_assignments(assignments: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reload official train and join only the persisted assignment IDs to text."""
    train = load_dataset(DATASET_ID, CONFIG, split="train")
    source = train.select_columns(["id", "text", "label"]).to_pandas()
    merged = assignments.merge(source, on=["id", "label"], how="left", validate="one_to_one")
    if merged["text"].isna().any() or len(merged) != len(assignments):
        raise ValueError("Some persisted IDs could not be reconstructed from official train")
    return (merged.loc[merged.subset == "train", ["id", "text", "label"]].copy(),
            merged.loc[merged.subset == "internal_validation", ["id", "text", "label"]].copy())
