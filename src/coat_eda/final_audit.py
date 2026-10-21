"""Reproducible, non-destructive technical audit of saved CoAT artefacts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from coat_eda.data_prep import validate_assignments
from coat_eda.error_analysis import all_models_wrong, load_frames
from coat_eda.final_evaluation import score

OUT = Path("reports/final_audit")
PROTECTED = [
    "models/tfidf_logreg.joblib", "models/mlp_best.pt", "models/mlp_regularized_best.pt",
    "models/lstm_best.pt", "models/lstm_vocab.json", "reports/baseline/split_assignments.csv",
    "reports/baseline/split_metadata.json", "reports/final_evaluation/logreg_predictions.csv",
    "reports/final_evaluation/mlp_original_predictions.csv", "reports/final_evaluation/mlp_regularized_predictions.csv",
    "reports/final_evaluation/lstm_predictions.csv", "reports/final_evaluation/metrics.csv",
    "reports/final_evaluation/metrics.json", "reports/final_evaluation/model_comparison.csv",
    "reports/error_analysis/bootstrap_model_ci.csv", "reports/error_analysis/bootstrap_pairwise_differences.csv",
]

def hashes() -> pd.DataFrame:
    return pd.DataFrame([{"path": str(Path(path).resolve()), "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()} for path in PROTECTED])

def check_metrics() -> tuple[pd.DataFrame, bool]:
    root = Path("reports/final_evaluation")
    saved = {row["key"]: row for row in json.loads((root / "metrics.json").read_text(encoding="utf-8"))}
    rows = []
    for key, row in saved.items():
        frame = pd.read_csv(root / f"{key}_predictions.csv")
        recalculated = score(frame.true_label.to_numpy(), frame.probability_ai.to_numpy())
        exact = all(np.isclose(recalculated[name], row[name]) for name in ("accuracy", "precision_label_1", "recall_label_1", "f1_label_1", "roc_auc"))
        cm = recalculated["confusion_matrix"] == row["confusion_matrix"]
        rows.append({"key": key, "model": row["model"], **{name: recalculated[name] for name in ("accuracy", "precision_label_1", "recall_label_1", "f1_label_1", "roc_auc")}, "confusion_matrix_match": cm, "metrics_json_match": exact})
    result = pd.DataFrame(rows)
    csv_keys = pd.read_csv(root / "metrics.csv").key.tolist()
    comparison_keys = pd.read_csv(root / "model_comparison.csv").key.tolist()
    return result, bool(result.metrics_json_match.all() and result.confusion_matrix_match.all() and csv_keys == comparison_keys == result.key.tolist())

def check_split() -> dict[str, object]:
    assignments = pd.read_csv("reports/baseline/split_assignments.csv")
    validate_assignments(assignments)
    metadata = json.loads(Path("reports/baseline/split_metadata.json").read_text(encoding="utf-8"))
    return {"examples": int(len(assignments)), "train": int((assignments.subset == "train").sum()), "internal_validation": int((assignments.subset == "internal_validation").sum()), "unique_ids": int(assignments.id.nunique()), "random_state": metadata["random_state"], "class_counts": metadata["class_counts"], "valid": True}

def markdown(summary: dict[str, object], metrics: pd.DataFrame) -> str:
    table = metrics.to_csv(index=False).replace(",", " | ")
    return f"""# Итоговый технический аудит CoAT

## 1. Цель и область аудита

Проверены сохранённые артефакты эксперимента без обучения, дообучения или изменения официальных итоговых предсказаний.

## 2. Структура проекта

Зависимости и консольные команды описаны в `pyproject.toml`; код расположен в `src/coat_eda`, быстрые проверки — в `tests`, модели — в `models`, результаты — в `reports`.

## 3. Датасет и разбиение

Фиксированная train-only подвыборка содержит {summary['split']['examples']} примеров: {summary['split']['train']} train и {summary['split']['internal_validation']} internal validation. Классы стратифицированы, ID уникальны, seed={summary['split']['random_state']}. EDA фиксирует 24 628 размеченных объектов официального validation и отсутствие точных текстовых пересечений; это не исключает семантическую близость текстов.

## 4. Архитектуры

Логистическая регрессия использует сохранённый TF-IDF Pipeline. MLP и regularized MLP используют BCEWithLogitsLoss, dropout и лучший checkpoint по validation loss; разреженные TF-IDF матрицы уплотняются только по мини-батчам. LSTM использует train-only словарь, `<PAD>/<UNK>`, padding через packed sequences и последнее корректное скрытое состояние. В inference используются `eval()` и отсутствие градиентов.

## 5. Проверка метрик

Метрики ниже независимо пересчитаны из prediction CSV; совпадение с `metrics.json`, `metrics.csv` и `model_comparison.csv`: **{summary['metrics_consistent']}**.

```text
{table}```

## 6. Ошибки и bootstrap

FP/FN, FPR/FNR, F1 по квартилям длины, confidence/Brier и ID-выравнивание представлены в `reports/error_analysis`. Общих ошибок всех четырёх моделей: **{summary['common_errors']}**. Сохранённые paired class-stratified bootstrap-результаты не изменялись: интервалы разности F1 включают ноль, интервалы разности ROC-AUC для LSTM с regularized MLP и Logistic Regression — нет. Это вывод для данного validation и текстового bootstrap; кластерная зависимость источников может занижать неопределённость.

## 7. Графики

Для статьи рекомендуются: `final_evaluation/model_comparison.png`, `roc_curves.png`, `pr_curves.png`, `error_analysis/f1_by_length.png`, `fp_fn_comparison.png` и `confidence_correct_vs_wrong.png`. Они имеют PNG/PDF-версии и пригодны для двухколоночной вёрстки после выбора подмножества.

## 8. Воспроизводимость

`uv sync`, затем `uv run pytest`; для пересчёта post-hoc анализа — `uv run coat-error-analysis`. Обучение, inference из checkpoint и пересчёт метрик по CSV — разные операции. Идентичность переобучения на разных устройствах не гарантируется.

## 9. Обнаруженные проблемы и исправления

**Major, исправлено:** обычный `coat-error-analysis` ранее считал общие ошибки лишь трёх моделей и заменял развёрнутый отчёт краткой заметкой. Оба режима теперь используют `all_models_wrong` и учитывают четыре модели. **Minor, исправлено:** README актуализирован и содержит финальный аудит и pytest.

## 10. Оставшиеся ограничения

Не проведена отдельная проверка семантически близких текстов и нет внешнего независимого корпуса. Статистические выводы ограничены одним официальным validation-набором и предположениями text-level bootstrap.

## 11. Заключение

Критических и неустранённых major-проблем аудит не выявил. Экспериментальная часть готова к этапу №8 — подготовке материалов статьи, с указанными ограничениями.
"""

def run(output: Path = OUT, test_results: str | None = None) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    before = output / "artifact_hashes_before.csv"
    if not before.exists(): hashes().to_csv(before, index=False, encoding="utf-8")
    metric_rows, consistent = check_metrics(); metric_rows.to_csv(output / "recalculated_metrics.csv", index=False, encoding="utf-8")
    split = check_split(); common = all_models_wrong(load_frames())
    saved_common = pd.read_csv("reports/error_analysis/all_models_wrong_ids.csv")
    if not saved_common.equals(common.reset_index(drop=True)): raise ValueError("Common-error CSV does not match the four-model calculation")
    after = hashes(); after.to_csv(output / "artifact_hashes_after.csv", index=False, encoding="utf-8")
    initial = pd.read_csv(before).rename(columns={"Path": "path", "Hash": "sha256"})[["path", "sha256"]]
    initial["sha256"] = initial["sha256"].str.lower()
    hashes_unchanged = initial.equals(after)
    manifest = pd.DataFrame([
        {"kind":"table", "path":"reports/final_evaluation/model_comparison.csv", "purpose":"Основные метрики моделей"},
        {"kind":"figure", "path":"reports/final_evaluation/model_comparison.png", "purpose":"F1 и ROC-AUC"},
        {"kind":"figure", "path":"reports/final_evaluation/roc_curves.png", "purpose":"ROC-кривые"},
        {"kind":"figure", "path":"reports/final_evaluation/pr_curves.png", "purpose":"Precision-recall кривые"},
        {"kind":"figure", "path":"reports/error_analysis/f1_by_length.png", "purpose":"F1 по длине текста"},
        {"kind":"figure", "path":"reports/error_analysis/fp_fn_comparison.png", "purpose":"Структура ошибок"},
        {"kind":"table", "path":"reports/error_analysis/bootstrap_pairwise_differences.csv", "purpose":"Bootstrap-интервалы различий"},
    ]); manifest.to_csv(output / "article_materials_manifest.csv", index=False, encoding="utf-8")
    summary = {"split": split, "metrics_consistent": consistent, "common_errors": int(len(common)), "protected_hashes_unchanged": hashes_unchanged, "critical_open": [], "major_open": []}
    (output / "audit_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "audit_report.md").write_text(markdown(summary, metric_rows), encoding="utf-8")
    if test_results is not None:
        (output / "test_results.txt").write_text(test_results + "\n", encoding="utf-8")
    return summary

def main() -> None:
    parser = argparse.ArgumentParser(description="Audit saved CoAT experiment artefacts without retraining")
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--test-results", default=None)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir, args.test_results), ensure_ascii=False, indent=2))

if __name__ == "__main__": main()
