# Final official-validation evaluation

This report evaluates fixed checkpoints only; no model fitting, threshold selection, vocabulary rebuilding, or TF-IDF fitting was performed.

Official CoAT `binary` validation contains 24628 examples, with class counts {'0': 12314, '1': 12314}. Integrity checks found no empty texts, exact duplicates, or overlap with the previously used train/internal-validation data.

## Metrics

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Inference seconds |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.6790 | 0.6940 | 0.6402 | 0.6660 | 0.7501 | 0.7812 |
| Original MLP | 0.6890 | 0.7513 | 0.5650 | 0.6449 | 0.7683 | 2.4730 |
| Regularized MLP | 0.6858 | 0.7061 | 0.6366 | 0.6695 | 0.7599 | 2.0305 |
| LSTM | 0.7027 | 0.7473 | 0.6126 | 0.6733 | 0.7964 | 4.2126 |

Scores use label `1` as the positive AI class and a fixed 0.5 threshold; ROC-AUC and average precision use probabilities. Inference was measured on CPU over the full validation set and includes text preprocessing/vectorization/tokenization.

## Limitations

This is a single held-out official dataset split, not evidence about arbitrary contemporary LLMs. No statistical significance test was performed, so metric differences are descriptive only.
