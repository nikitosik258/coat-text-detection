# table_2_models

| Модель | Представление текста | Архитектура | Обучаемые параметры | Основные параметры | Выбор checkpoint |
|---|---|---|---|---|---|
| Logistic Regression | TF-IDF, word 1–2-grams | LogisticRegression | — | max_features=30 000; min_df=2; sublinear_tf=True; max_iter=1000 | фиксированные параметры baseline |
| Original MLP | замороженный TF-IDF (30 000) | 30 000→256→64→1, ReLU, dropout 0.3/0.2 | 7696769 | Adam; lr=0.001; batch=64; ≤20 эпох | minimum internal-validation loss; patience=4 |
| Regularized MLP | замороженный TF-IDF (30 000) | 30 000→128→32→1, ReLU, dropout 0.4/0.4 | 3844289 | Adam; lr=0.0003; weight_decay=0.0001; batch=64; ≤20 эпох | minimum internal-validation loss; patience=4 |
| LSTM | токены + обучаемый Embedding | Embedding(20 000,128)→LSTM(128)→dropout(0.3)→Linear(1) | 2692225 | AdamW; lr=0.001; weight_decay=0.0001; batch=64; max length=192 | minimum internal-validation loss; patience=3 |
