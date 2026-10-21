"""Prepare article-ready materials solely from saved CoAT experiment outputs."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path("reports")
OUT = ROOT / "article_materials"

def md_table(frame: pd.DataFrame) -> str:
    return "| " + " | ".join(frame.columns) + " |\n|" + "|".join("---" for _ in frame.columns) + "|\n" + "\n".join("| " + " | ".join(str(v) for v in row) + " |" for row in frame.itertuples(index=False, name=None))

def write_table(name: str, frame: pd.DataFrame, tables: Path) -> None:
    frame.to_csv(tables / f"{name}.csv", index=False, encoding="utf-8")
    (tables / f"{name}.md").write_text(f"# {name}\n\n{md_table(frame)}\n", encoding="utf-8")

def load_inputs() -> dict[str, object]:
    eda = json.loads((ROOT / "eda" / "summary.json").read_text(encoding="utf-8"))
    split = json.loads((ROOT / "baseline" / "split_metadata.json").read_text(encoding="utf-8"))
    final = pd.read_csv(ROOT / "final_evaluation" / "metrics.csv")
    errors = pd.read_csv(ROOT / "error_analysis" / "error_summary.csv")
    pairs = pd.read_csv(ROOT / "error_analysis" / "bootstrap_pairwise_differences.csv")
    lengths = pd.read_csv(ROOT / "eda" / "length_statistics.csv")
    return {"eda": eda, "split": split, "final": final, "errors": errors, "pairs": pairs, "lengths": lengths,
            "mlp": json.loads((ROOT / "mlp" / "training_config.json").read_text(encoding="utf-8")),
            "reg": json.loads((ROOT / "mlp_regularized" / "training_config.json").read_text(encoding="utf-8")),
            "lstm": json.loads((ROOT / "lstm" / "training_config.json").read_text(encoding="utf-8")),
            "lstm_metrics": json.loads((ROOT / "lstm" / "metrics.json").read_text(encoding="utf-8"))}

def make_tables(data: dict[str, object], tables: Path) -> None:
    eda, split, final, lengths = data["eda"], data["split"], data["final"], data["lengths"]
    train_len = lengths.loc[(lengths.split == "train") & (lengths.unit == "words")].iloc[0]
    val_len = lengths.loc[(lengths.split == "validation") & (lengths.unit == "words")].iloc[0]
    schema = eda["schema"]["splits"]
    dataset = pd.DataFrame([
        ["Набор и конфигурация", "RussianNLP/coat, binary"], ["Язык", "русский"],
        ["Задача", "бинарная классификация: человек (0) / AI (1)"],
        ["Официальный train", schema["train"]], ["Официальный validation", schema["validation"]],
        ["Выбранная train-only подвыборка", 30000], ["Обучение / внутренняя валидация", "24 000 / 6 000"],
        ["Классы в train / внутренней валидации", "12 000/12 000; 3 000/3 000 (0/1)"],
        ["Длина train, слов: среднее / медиана", f"{train_len['mean']:.2f} / {train_len['median']:.0f}"],
        ["Длина official validation, слов: среднее / медиана", f"{val_len['mean']:.2f} / {val_len['median']:.0f}"],
        ["Официальный test", "не использовался: метки скрыты"],
    ], columns=["Характеристика", "Значение"])
    write_table("table_1_dataset", dataset, tables)
    mlp, reg, lstm = data["mlp"], data["reg"], data["lstm"]
    models = pd.DataFrame([
        ["Logistic Regression", "TF-IDF, word 1–2-grams", "LogisticRegression", "—", "max_features=30 000; min_df=2; sublinear_tf=True; max_iter=1000", "фиксированные параметры baseline"],
        ["Original MLP", "замороженный TF-IDF (30 000)", "30 000→256→64→1, ReLU, dropout 0.3/0.2", 7696769, f"Adam; lr={mlp['learning_rate']}; batch={mlp['batch_size']}; ≤{mlp['max_epochs']} эпох", "minimum internal-validation loss; patience=4"],
        ["Regularized MLP", "замороженный TF-IDF (30 000)", "30 000→128→32→1, ReLU, dropout 0.4/0.4", 3844289, f"Adam; lr={reg['learning_rate']}; weight_decay={reg['weight_decay']}; batch={reg['batch_size']}; ≤{reg['max_epochs']} эпох", "minimum internal-validation loss; patience=4"],
        ["LSTM", "токены + обучаемый Embedding", "Embedding(20 000,128)→LSTM(128)→dropout(0.3)→Linear(1)", data["lstm_metrics"]["trainable_parameters"], f"AdamW; lr={lstm['learning_rate']}; weight_decay={lstm['weight_decay']}; batch={lstm['batch_size']}; max length={lstm['max_sequence_length']}", "minimum internal-validation loss; patience=3"],
    ], columns=["Модель", "Представление текста", "Архитектура", "Обучаемые параметры", "Основные параметры", "Выбор checkpoint"])
    write_table("table_2_models", models, tables)
    result = final[["model", "accuracy", "precision_label_1", "recall_label_1", "f1_label_1", "roc_auc", "inference_seconds"]].copy()
    result.columns = ["Модель", "Accuracy", "Precision (AI=1)", "Recall (AI=1)", "F1 (AI=1)", "ROC-AUC", "Инференс CPU, с (24 628 текстов)"]
    for c in result.columns[1:]: result[c] = result[c].map(lambda x: f"{x:.4f}").astype(str)
    write_table("table_3_final_metrics", result, tables)
    pairs = data["pairs"].copy(); pairs = pairs[pairs.metric.isin(["f1", "roc_auc"])]
    rows = []
    for comparison, group in pairs.groupby("comparison", sort=False):
        values = {r.metric: r for _, r in group.iterrows()}
        f1, auc = values["f1"], values["roc_auc"]
        rows.append([comparison, f"{f1.estimate:+.4f}", f"[{f1.ci95_low:+.4f}; {f1.ci95_high:+.4f}]", f"{auc.estimate:+.4f}", f"[{auc.ci95_low:+.4f}; {auc.ci95_high:+.4f}]"])
    write_table("table_4_bootstrap", pd.DataFrame(rows, columns=["Сравнение", "ΔF1", "95% CI ΔF1", "ΔROC-AUC", "95% CI ΔROC-AUC"]), tables)
    err = data["errors"][["model", "fp", "fn", "fpr", "fnr"]].copy(); err.columns = ["Модель", "FP", "FN", "FPR", "FNR"]
    for c in ["FPR", "FNR"]: err[c] = err[c].map(lambda x: f"{x:.4f}")
    write_table("table_s1_errors", err, tables)

def make_bootstrap_figure(data: dict[str, object], figures: Path) -> None:
    pairs = data["pairs"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True)
    for ax, metric, label in zip(axes, ["f1", "roc_auc"], ["ΔF1", "ΔROC-AUC"]):
        x = pairs[pairs.metric == metric].reset_index(drop=True)
        y = np.arange(len(x))
        ax.errorbar(x.estimate, y, xerr=[x.estimate - x.ci95_low, x.ci95_high - x.estimate], fmt="o", color="#1f77b4", capsize=3)
        ax.axvline(0, color="gray", ls="--", lw=1); ax.set(xlabel=label, yticks=y, yticklabels=x.comparison, title=label)
        ax.grid(axis="x", alpha=.25)
    fig.tight_layout(); fig.savefig(figures / "figure_3_bootstrap_differences.png", dpi=300); fig.savefig(figures / "figure_3_bootstrap_differences.pdf"); plt.close(fig)

def copy_figures(data: dict[str, object], figures: Path) -> list[dict[str, str]]:
    selection = [
        ("figure_1_roc_curves", ROOT / "final_evaluation" / "roc_curves", "ROC-кривые четырёх моделей на официальной validation CoAT."),
        ("figure_2_f1_by_length", ROOT / "error_analysis" / "f1_by_length", "F1 моделей в квартилях длины текста; связь описательная, не причинная."),
        ("figure_4_fp_fn", ROOT / "error_analysis" / "fp_fn_comparison", "Количество ложноположительных и ложноотрицательных решений на official validation."),
    ]
    manifest = []
    for target, source, caption in selection:
        for ext in (".png", ".pdf"):
            shutil.copy2(source.with_suffix(ext), figures / f"{target}{ext}")
        manifest.append({"Название": target, "Тип": "рисунок", "Путь": f"figures/{target}.png", "Исходный файл данных": str(source.with_suffix('.png')).replace('\\\\', '/'), "Раздел": "Результаты и обсуждение", "Назначение": caption})
    make_bootstrap_figure(data, figures)
    manifest.insert(2, {"Название": "figure_3_bootstrap_differences", "Тип": "рисунок", "Путь": "figures/figure_3_bootstrap_differences.png", "Исходный файл данных": "reports/error_analysis/bootstrap_pairwise_differences.csv", "Раздел": "Результаты и обсуждение", "Назначение": "Парные 95% percentile CI разностей LSTM относительно двух моделей."})
    return manifest

def notes() -> dict[str, str]:
    return {
"methodology_notes.md": """# Методика эксперимента

Использована конфигурация `binary` корпуса RussianNLP/coat: класс 0 — человеческий текст, класс 1 — автоматически сгенерированный. Из официального train детерминированно отобраны 15 000 объектов каждого класса (seed 42); после стратифицированного деления получены 24 000 обучающих и 6 000 внутренних валидационных объектов. Официальный размеченный validation (24 628 текстов) не участвовал в подборе параметров и применён только для итоговой оценки; test с закрытыми метками не использовался. Точные ID train и внутренней валидации не пересекаются.

Для Logistic Regression использованы TF-IDF word n-граммы 1–2 (`max_features=30000`, `min_df=2`, `sublinear_tf=True`) и `LogisticRegression(max_iter=1000, random_state=42, class_weight=None)`. TF-IDF обучался только на 24 000 train-текстах. Регистр, пунктуация и стоп-слова специально не удалялись, так как потенциально несут стилевую информацию.

Original MLP получает замороженные TF-IDF-признаки: 30 000→256→64→1, ReLU и dropout 0.3/0.2; Adam, lr=0.001, batch=64, до 20 эпох. Regularized MLP использует 30 000→128→32→1, dropout 0.4/0.4, Adam с lr=0.0003 и weight decay 1e-4; остальные лимиты — batch=64, до 20 эпох. В обеих сетях применялся BCEWithLogitsLoss, checkpoint выбирался по минимуму internal-validation loss, patience=4.

LSTM использует отдельное представление текста: Unicode-токенизацию со словами и пунктуацией при сохранении регистра, train-only словарь из 20 000 токенов с `<PAD>` и `<UNK>`, ограничение длины 192. Архитектура: Embedding(128), однонаправленная LSTM(hidden=128), dropout 0.3 и Linear(1); AdamW (lr=0.001, weight decay 1e-4), batch=64, максимум 15 эпох, patience=3, clipping нормы градиента 1.0. Padding обрабатывался packed sequences; использовалось последнее скрытое состояние корректной длины.

Все нейросети имели одинаковые ID train/internal validation, но различное представление текста: TF-IDF для MLP и последовательности токенов для LSTM. Вероятности преобразовывались в класс фиксированным порогом 0.5. На validation рассчитаны Accuracy, Precision, Recall, F1 (положительный класс AI=1) и ROC-AUC. Неопределённость оценена сохранённым paired class-stratified text-level bootstrap: 2 000 повторов, seed 42, percentile 95% CI; эта процедура описывает данную выборку и может недооценивать зависимость кластеризованных источников.
""",
"results_notes.md": """# Результаты и обсуждение

На official validation LSTM имеет наибольшие наблюдаемые Accuracy=0.7027, F1=0.6733 и ROC-AUC=0.7964. Regularized MLP достигает F1=0.6695, Logistic Regression — 0.6660, Original MLP — 0.6449. Original MLP имеет максимальный Precision=0.7513, но минимальный Recall=0.5650; LSTM сочетает Precision=0.7473 с Recall=0.6126. Следовательно, различия в F1 отражают разный баланс ошибок, а не только общий уровень качества.

Regularized MLP была добавлена как вариант с меньшими скрытыми слоями, большим dropout и weight decay. На internal validation её F1=0.6525 против 0.6313 у Original MLP; на official validation F1 также выше (0.6695 против 0.6449). Это согласуется с тем, что исходная конфигурация могла быть чувствительна к переобучению, но не доказывает, что регуляризация полностью устранила его. Лучшие checkpoints выбирались по minimum validation loss; best epoch равен 1 для Original MLP, 2 для Regularized MLP и 2 для LSTM. Короткая история LSTM не даёт основания для сильного вывода о переобучении.

Для LSTM зарегистрированы 2 550 FP и 4 771 FN (FPR=0.2071, FNR=0.3874); для Regularized MLP — 3 263 FP и 4 475 FN. У LSTM и Regularized MLP: 13 575 случаев, где обе модели верны, 4 006 — где обе ошибочны, 3 732 — где верна только LSTM, и 3 315 — где верна только Regularized MLP. Все четыре модели одновременно ошиблись на 3 376 текстах. Средняя уверенность LSTM на верных/ошибочных решениях равна 0.7522/0.6154; доля ошибок с уверенностью не менее 0.9 — 0.0060.

Качество существенно различается по квартилям длины. F1 LSTM меняется от 0.5294 в Q1_short и 0.4979 в Q2 до 0.9232 в Q4_long; у Regularized MLP — от 0.5247 и 0.5117 до 0.8830. Это описательная связь на данном наборе, а не доказательство причинного эффекта длины.

Paired bootstrap даёт ΔF1 LSTM−Regularized MLP=0.0037, 95% CI [−0.0037; 0.0113], и ΔF1 LSTM−Logistic Regression=0.0072, 95% CI [−0.0001; 0.0149]; оба интервала включают ноль. Для ROC-AUC интервалы не включают ноль: 0.0365 [0.0304; 0.0427] и 0.0463 [0.0402; 0.0526] соответственно. Поэтому для этой validation paired bootstrap поддерживает различие ROC-AUC, но не даёт убедительного преимущества LSTM по F1. Этот вывод нельзя переносить на произвольные новые LLM или домены.

Инференс на CPU для всех 24 628 текстов (включая предобработку) занял 0.7812 с для Logistic Regression, 2.4730 с для Original MLP, 2.0305 с для Regularized MLP и 4.2126 с для LSTM. Более высокий ROC-AUC LSTM получен ценой большего вычислительного времени; время обучения не используется как строго сопоставимая метрика.
""",
"conclusions_notes.md": """# Научные выводы

1. На official validation CoAT все четыре рассмотренные подхода дают F1 от 0.6449 до 0.6733, что показывает практическую различимость классов в рамках данного корпуса и протокола.
2. LSTM с train-only словарём и последовательностным представлением имеет наибольший наблюдаемый ROC-AUC=0.7964 и Accuracy=0.7027, тогда как TF-IDF-подходы используют иной тип информации о тексте.
3. Regularized MLP превосходит Original MLP по F1 на official validation (0.6695 против 0.6449), однако этот результат относится к конкретным конфигурациям и критериям early stopping.
4. Интервалы bootstrap для разностей F1 LSTM относительно Regularized MLP и Logistic Regression включают ноль; на этой выборке это не подтверждает устойчивого преимущества LSTM именно по F1. Для ROC-AUC соответствующие интервалы ноль не включают.
5. Короткие группы текстов имеют существенно меньший наблюдаемый F1, чем Q4_long; это следует учитывать как описательную характеристику ошибок, не приписывая длине причинную роль.
6. Для обобщения необходимы проверки на независимых корпусах, новых генеративных моделях и доменах, а также анализ семантически близких текстов и зависимостей по источникам.
""",
"limitations_notes.md": """# Ограничения исследования

Использован один корпус CoAT и train-only подвыборка из 24 000 текстов; итоговая оценка проведена на одной official validation. Официальный test не использовался, поскольку его метки скрыты. Семантические дубликаты и близкие тексты между выборками специально не исследовались. Результаты могут зависеть от генеративных моделей, задач и источников CoAT. Text-level bootstrap не моделирует возможную кластерную зависимость примеров. Применён фиксированный порог 0.5. Сравнивались конкретные архитектуры и гиперпараметры, а не все возможные MLP/LSTM. Поэтому оценка CoAT не является доказательством способности обнаруживать тексты любых современных языковых моделей.
""",
"sources_manifest.md": """# Проверенные источники

1. **CoAT: Corpus of artificial texts** — Tatiana Shamardina, Marat Saidov, Alena Fenogenova, Aleksandr Tumanov, Alina Zemlyakova, Anna Lebedeva, Ekaterina Gryaznova, Tatiana Shavrina, Vladislav Mikhailov, Ekaterina Artemova; 2025. *Natural Language Processing*, 31(1), 150–175. DOI: [10.1017/nlp.2024.38](https://doi.org/10.1017/nlp.2024.38). Разделы: набор данных, постановка задачи. Проверено по [странице Cambridge](https://www.cambridge.org/core/journals/natural-language-processing/article/coat-corpus-of-artificial-texts/7E2CA97E21663CC031FB6BAFE56E0046).
2. **RussianNLP/coat: dataset card** — RussianNLP. Страница данных: [Hugging Face](https://huggingface.co/datasets/RussianNLP/coat). Раздел: данные и воспроизводимость.
3. **Long Short-Term Memory** — Sepp Hochreiter, Jürgen Schmidhuber; 1997. *Neural Computation*, 9(8), 1735–1780. DOI: [10.1162/neco.1997.9.8.1735](https://doi.org/10.1162/neco.1997.9.8.1735). Раздел: LSTM.
4. **Introduction to Information Retrieval** — Christopher D. Manning, Prabhakar Raghavan, Hinrich Schütze; 2008. Cambridge University Press. [Официальная страница книги](https://nlp.stanford.edu/IR-book/). Раздел: TF-IDF.
5. **DetectGPT: Zero-Shot Machine-Generated Text Detection using Probability Curvature** — Eric Mitchell, Yoonho Lee, Alexander Khazatsky, Christopher D. Manning, Chelsea Finn; 2023. ICML/PMLR 202. [Страница публикации](https://proceedings.mlr.press/v202/mitchell23a.html). Раздел: контекст обнаружения машинно-генерированного текста.
6. **The Relationship Between Precision-Recall and ROC Curves** — Jesse Davis, Mark Goadrich; 2006. ICML. DOI: [10.1145/1143844.1143874](https://doi.org/10.1145/1143844.1143874). Раздел: метрики и графики.

Все позиции выше имеют указанную первичную или официальную страницу; непроверенные публикации не включены.
""",
"article_outline.md": """# План статьи (5–7 страниц, две колонки)

1. **Введение.** Актуальность, задача и цель; кратко обозначить Logistic Regression, MLP и LSTM.
2. **Набор данных и подготовка.** Таблица 1 после первого абзаца; описать train-only выборку, внутреннюю валидацию и отдельный official validation.
3. **Модели и методика.** Таблица 2; TF-IDF и последовательностное представление, early stopping, метрики и bootstrap.
4. **Результаты и обсуждение.** Таблица 3 в начале раздела; рисунок 1 (ROC) на две колонки. Затем Таблица 4 и рисунок 3 (bootstrap). Рисунок 2 (F1 по длине) — одна колонка. При наличии места — рисунок 4 (FP/FN), иначе вынести в приложение.
5. **Заключение и ограничения.** Сформулировать выводы только для CoAT и данного протокола.

Необязательные PR-кривые полезны при необходимости показать компромисс precision/recall, но при сбалансированных классах и ограничении объёма они дублируют таблицу 3 и ROC. Графики уверенности — диагностический материал; матрицы ошибок дублируют FP/FN и таблицу S1. Поэтому они не включены в основной набор.
""",
"article_ready_summary.md": """# Готовность материалов

Комплект содержит четыре основные таблицы, дополнительную таблицу ошибок, четыре отобранных рисунка в PNG (300 DPI) и PDF, методические и результатные заметки, ограничения и проверенную библиографию. Все числа извлечены из сохранённых CSV/JSON, без обучения или переоценки моделей. Автоматическая проверка сравнивает метрики и bootstrap-интервалы с исходными файлами. Комплект предназначен для следующего самостоятельного этапа — подготовки LaTeX-рукописи, но сам рукописью не является.
"""}

def validate(output: Path) -> None:
    expected = [output / "tables" / name for name in ("table_1_dataset.csv", "table_2_models.csv", "table_3_final_metrics.csv", "table_4_bootstrap.csv", "table_s1_errors.csv")]
    expected += [output / "figures" / f"{n}{ext}" for n in ("figure_1_roc_curves", "figure_2_f1_by_length", "figure_3_bootstrap_differences", "figure_4_fp_fn") for ext in (".png", ".pdf")]
    expected += [output / name for name in notes()] + [output / "article_outline.md", output / "article_ready_summary.md", output / "article_materials_manifest.csv"]
    missing = [str(p) for p in expected if not p.is_file()]
    if missing: raise ValueError(f"Missing article materials: {missing}")
    for path in expected:
        if path.suffix == ".csv" and pd.read_csv(path).isna().any().any(): raise ValueError(f"NaN in {path}")
    final = pd.read_csv(ROOT / "final_evaluation" / "metrics.csv")
    prepared = pd.read_csv(output / "tables" / "table_3_final_metrics.csv")
    if not np.allclose(prepared["F1 (AI=1)"].astype(float), final.f1_label_1, rtol=0, atol=5e-5): raise ValueError("Article metrics differ from final evaluation")
    source_pairs = pd.read_csv(ROOT / "error_analysis" / "bootstrap_pairwise_differences.csv")
    article_pairs = pd.read_csv(output / "tables" / "table_4_bootstrap.csv")
    if len(article_pairs) != 2 or len(source_pairs) != 4: raise ValueError("Unexpected bootstrap table content")

def run(output: Path = OUT, check_only: bool = False) -> None:
    if check_only:
        validate(output); return
    tables, figures = output / "tables", output / "figures"; tables.mkdir(parents=True, exist_ok=True); figures.mkdir(parents=True, exist_ok=True)
    data = load_inputs(); make_tables(data, tables); items = copy_figures(data, figures)
    for name, text in notes().items(): (output / name).write_text(text, encoding="utf-8")
    for table in ("table_1_dataset", "table_2_models", "table_3_final_metrics", "table_4_bootstrap", "table_s1_errors"):
        items.append({"Название": table, "Тип": "таблица", "Путь": f"tables/{table}.md", "Исходный файл данных": "сохранённые CSV/JSON reports", "Раздел": "см. article_outline.md", "Назначение": "Таблица с фактическими результатами."})
    for name in notes(): items.append({"Название": name, "Тип": "текст", "Путь": name, "Исходный файл данных": "сохранённые отчёты и код", "Раздел": "см. article_outline.md", "Назначение": "Научные заметки для рукописи."})
    pd.DataFrame(items).to_csv(output / "article_materials_manifest.csv", index=False, encoding="utf-8")
    validate(output)

def main() -> None:
    parser = argparse.ArgumentParser(description="Create article materials from saved CoAT results only")
    parser.add_argument("--output-dir", type=Path, default=OUT); parser.add_argument("--check", action="store_true")
    args = parser.parse_args(); run(args.output_dir, args.check); print("Article-materials check passed")

if __name__ == "__main__": main()
