# CoAT: обнаружение автоматически сгенерированного текста

This project provides a reproducible experimental pipeline for the `binary`
configuration of [`RussianNLP/coat`](https://huggingface.co/datasets/RussianNLP/coat):
EDA, a fixed train/internal-validation split, TF-IDF +
Logistic Regression, two MLP variants, LSTM, final inference on official
validation, and post-hoc error analysis. It never changes source data, removes
duplicates automatically, combines official splits, or creates an application interface.

## Requirements

Install [uv](https://docs.astral.sh/uv/) and run from the project root:

```powershell
uv sync
uv run coat-eda
```

Alternatively:

```powershell
uv run python -m coat_eda.eda --output-dir reports/eda
```

The first run downloads the dataset through Hugging Face `datasets`; its cache is
managed outside this repository.  To place results elsewhere, pass `--output-dir`.

## Output

All generated artefacts go to `reports/eda/`:

- `schema.json` and `summary.json` — dataset structure and machine-readable results;
- `class_distribution.csv`, `length_statistics.csv`, `quality_checks.csv` — counts,
  text-length statistics and duplicate/empty-text statistics by official split;
- `split_overlaps.csv`, `conflicting_labels.csv` — exact-text leakage and label conflicts;
- `balanced_subsample_feasibility.csv` — train-only balanced-subsample capacity;
- `class_distribution.png`, `length_distributions.png` — required visualizations.

Text identity checks use SHA-256 hashes in reports so source texts are not copied
into analysis outputs.  Empty text means an empty or whitespace-only string; word
counts use Unicode non-whitespace tokens; duplicate and overlap checks use exact
text equality.

## TF-IDF + Logistic Regression baseline

Run the deterministic baseline (seed `42`) with:

```powershell
uv sync
uv run coat-baseline
```

It samples 15,000 examples per class **only from official `train`**, then creates
stratified `train` (24,000) and `internal_validation` (6,000) sets. The reusable
ID assignments are saved as `reports/baseline/split_assignments.csv`; later MLP
and LSTM experiments must consume this file rather than construct a new split.
The vectorizer is fitted inside a scikit-learn `Pipeline` on the training subset
only. Official `validation` is not used by this command, and `test` is excluded
because its labels are hidden.

Generated baseline reports include metrics, a confusion matrix (CSV and PNG),
split profiles and split metadata in `reports/baseline/`. The trained pipeline is
written to `models/tfidf_logreg.joblib`.

Run the automated checks with:

```powershell
uv run pytest
```

## MLP on frozen TF-IDF features

The MLP reads the existing `reports/baseline/split_assignments.csv` and the
already fitted TF-IDF step from `models/tfidf_logreg.joblib`; it does not create
a new split or fit TF-IDF again. It keeps features sparse until each mini-batch
is converted to a bounded dense tensor.

```powershell
uv sync
uv run coat-mlp
```

Its architecture is `30000 → 256 → 64 → 1` with ReLU and dropout (0.3, 0.2).
Training uses Adam, learning rate 0.001, batches of 64, at most 20 epochs, and
early stopping (patience 4) on internal-validation loss. Results are written to
`reports/mlp/`; the best checkpoint is `models/mlp_best.pt`. The checkpoint can
be loaded through `coat_eda.mlp.load_mlp_predictor` and used with
`coat_eda.mlp.predict_texts` without retraining.

## Regularized MLP experiment

This separate experiment preserves all original MLP outputs. It uses the same
persisted IDs and frozen baseline TF-IDF, with architecture `30000 → 128 → 32 → 1`,
dropout 0.4/0.4, Adam learning rate 0.0003, and weight decay 1e-4.

```powershell
uv run python -m coat_eda.mlp_regularized
```

Results, including per-epoch ROC-AUC and a three-model comparison, are written
to `reports/mlp_regularized/`; the checkpoint is `models/mlp_regularized_best.pt`.

## LSTM with trainable embeddings

The LSTM uses the same persisted 24,000/6,000 IDs but tokenizes text directly:
Unicode words and punctuation are retained with case preserved. Its vocabulary is
built from train only (20,000 tokens including `<PAD>`/`<UNK>`), and sequences
are dynamically padded and capped at 192 tokens.

```powershell
uv run coat-lstm
```

The model is `Embedding(128) → unidirectional LSTM(128) → Dropout(0.3) → Linear(1)`.
Results are in `reports/lstm/`; checkpoint and vocabulary are `models/lstm_best.pt`
and `models/lstm_vocab.json`. Load them with `load_lstm` and `load_vocab`.

## Final official-validation evaluation

This command performs inference only on the official labelled CoAT `validation`
split. It reloads all four fixed checkpoints and their frozen preprocessing;
it does not fit TF-IDF, rebuild the LSTM vocabulary, train, tune, or use `test`.

```powershell
uv run coat-final-eval
```

Outputs in `reports/final_evaluation/` include aligned prediction CSVs, metrics,
data-integrity checks, reproducibility metadata, plots (PNG/PDF), confusion
matrices, and a Markdown summary. Inference timing is CPU timing for the full
official validation set and includes input preprocessing.

## Error analysis and uncertainty

This post-hoc stage uses the saved final-evaluation prediction CSV files only;
it does not train models, alter checkpoints, tune a threshold, or change data.

```powershell
uv run coat-error-analysis
```

Outputs in `reports/error_analysis/` include FP/FN summaries, length and
confidence analyses, model disagreements, deterministic short error snippets,
and 2,000 paired class-stratified bootstrap replicates with 95% percentile CIs.

To apply the targeted Stage-6 reporting repair without rerunning bootstrap:

```powershell
uv run coat-error-analysis --repair
```

## Final technical audit

The audit is inference-free and does not rerun training, final evaluation, or
bootstrap. It recalculates metrics from the saved prediction CSV files, checks
the persisted split and protected artefact hashes, and writes a Russian report.

```powershell
uv run coat-final-audit
uv run pytest
```

Outputs are stored in `reports/final_audit/`. Re-training commands (`coat-baseline`,
`coat-mlp`, `coat-lstm`) create new experiment artefacts; loading checkpoints for
inference is instead performed by `coat-final-eval`. Metrics may differ slightly
when training is repeated on a different device; saved prediction CSVs are the
source for post-hoc metric recalculation.

## Materials for the scientific article

The following command does not train or evaluate a model. It prepares Russian
tables, figure copies and evidence-based notes for a future article from the
saved reports only:

```powershell
uv run coat-article-materials
uv run coat-article-materials --check
```

The complete package is created in `reports/article_materials/`. It is an
article-preparation kit, not a LaTeX manuscript.
