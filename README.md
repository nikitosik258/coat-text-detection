# CoAT: обнаружение автоматически сгенерированных текстов

Воспроизводимый Python-проект для исследования «Применение нейронных сетей для обнаружения автоматически сгенерированного текста». Используется конфигурация `binary` датасета [`RussianNLP/coat`](https://huggingface.co/datasets/RussianNLP/coat).

Проект включает EDA, фиксированное разбиение данных, TF-IDF + Logistic Regression, две MLP, LSTM, итоговую оценку на official `validation`, анализ ошибок, bootstrap, технический аудит и материалы для будущей статьи. Исходные данные не изменяются; official split не объединяются, а `test` с закрытыми метками не участвует в оценке.

## Установка

Требуется Python 3.10+ и [uv](https://docs.astral.sh/uv/). Выполняйте команды из корня проекта.

```powershell
uv sync
```

При первом запуске библиотека `datasets` загрузит CoAT из Hugging Face; кэш находится вне репозитория.

## Основные команды

```powershell
# Откройте ноутбуки в VS Code или JupyterLab
uv run jupyter lab notebooks/

# Фиксированная выборка и базовая модель
uv run coat-baseline

# Нейросетевые модели
uv run coat-mlp
uv run python -m coat_eda.mlp_regularized
uv run coat-lstm

# Инференс сохранённых моделей на official validation
uv run coat-final-eval

# Анализ ошибок, аудит и материалы статьи выполняются в ноутбуках
uv run pytest

```

Команды обучения создают экспериментальные артефакты. Для просмотра и анализа уже полученных результатов используйте ноутбуки из `notebooks/`, а не повторное обучение.

## Результаты проекта

| Директория | Содержимое |
|---|---|
| `reports/eda/` | схема CoAT, распределения, длина текстов, дубликаты и пересечения |
| `reports/baseline/` | фиксированные ID, профиль разбиения, метрики базовой модели |
| `reports/mlp/` | результаты Original MLP |
| `reports/mlp_regularized/` | результаты Regularized MLP |
| `reports/lstm/` | результаты LSTM и статистики токенизации |
| `reports/final_evaluation/` | итоговые prediction CSV, метрики, графики и матрицы ошибок |
| `reports/error_analysis/` | FP/FN, длина, уверенность, разногласия моделей и bootstrap |
| `reports/final_audit/` | аудит структуры, метрик и защищённых артефактов |
| `reports/article_materials/` | таблицы, рисунки и заметки для подготовки статьи |
| `models/` | TF-IDF Pipeline, checkpoints MLP/LSTM и словарь LSTM |

## EDA

Ноутбук: `notebooks/01_eda.ipynb`.

Результаты сохраняются в `reports/eda/`: описание схемы, статистики по official split, точные дубликаты и пересечения, проверка возможности сбалансированной выборки и графики. Проверка идентичности текстов основана на SHA-256-хэшах; тексты не копируются в отчёты. Пустой текст — пустая или состоящая только из пробелов строка.

В нём данные загружаются через `datasets`, а результаты сохраняются в `reports/eda/`.

## Подготовка выборки и Logistic Regression

```powershell
uv run coat-baseline
```

Из official `train` выбираются 15 000 текстов каждого класса при `random_state=42`; затем выполняется стратифицированное разделение на 24 000 обучающих и 6 000 внутренних валидационных объектов. Повторно используемые ID находятся в `reports/baseline/split_assignments.csv`.

TF-IDF обучается только на обучающей части внутри `Pipeline`: word n-граммы `(1, 2)`, `max_features=30000`, `min_df=2`, `sublinear_tf=True`. Классификатор: `LogisticRegression(max_iter=1000, random_state=42, class_weight=None)`. Official `validation` не участвует в этой команде, `test` исключён. Pipeline сохраняется в `models/tfidf_logreg.joblib`.

## Original MLP и Regularized MLP

```powershell
uv run coat-mlp
uv run python -m coat_eda.mlp_regularized
```

Обе модели используют сохранённые ID и уже обученный TF-IDF; новый split и векторизатор не создаются. Преобразование разрежённых признаков в плотные выполняется только для мини-батча.

- Original MLP: `30000 → 256 → 64 → 1`, ReLU, dropout 0.3/0.2, Adam (`lr=0.001`), batch 64, до 20 эпох, early stopping с `patience=4`. Результаты: `reports/mlp/`, checkpoint: `models/mlp_best.pt`.
- Regularized MLP: `30000 → 128 → 32 → 1`, dropout 0.4/0.4, Adam (`lr=0.0003`, `weight_decay=1e-4`). Результаты: `reports/mlp_regularized/`, checkpoint: `models/mlp_regularized_best.pt`.

## LSTM

```powershell
uv run coat-lstm
```

LSTM использует те же ID, но прямую токенизацию текста: сохраняются Unicode-слова, пунктуация и регистр. Словарь строится только по train, содержит до 20 000 токенов с `<PAD>` и `<UNK>`; длина последовательности ограничена 192 токенами.

Архитектура: `Embedding(128) → LSTM(128) → Dropout(0.3) → Linear(1)`. Результаты: `reports/lstm/`; checkpoint и словарь: `models/lstm_best.pt`, `models/lstm_vocab.json`.

## Итоговая оценка

```powershell
uv run coat-final-eval
```

Команда выполняет только инференс четырёх сохранённых моделей на размеченном official CoAT `validation`. Она не обучает TF-IDF, не пересобирает словарь LSTM, не настраивает порог и не использует `test`.

В `reports/final_evaluation/` находятся выровненные prediction CSV, метрики, проверки целостности, PNG/PDF-графики и матрицы ошибок. Время инференса измерено на CPU для 24 628 текстов с учётом предобработки.

## Анализ ошибок и неопределённости

Ноутбук: `notebooks/02_error_analysis.ipynb`.

Этап использует только сохранённые prediction CSV. Он не обучает модели, не меняет checkpoints и не подбирает порог. `reports/error_analysis/` содержит FP/FN, анализ длины и уверенности, расхождения моделей, примеры ошибок, а также 2 000 парных стратифицированных bootstrap-повторов с 95% percentile CI.

Он использует сохранённые prediction CSV и bootstrap-результаты; переобучение и повторная оценка моделей не выполняются.

## Технический аудит

Ноутбук: `notebooks/03_final_audit.ipynb`.

Аудит не запускает обучение, итоговый инференс или bootstrap заново. Он пересчитывает метрики из сохранённых prediction CSV, проверяет fixed split и SHA-256 защищённых артефактов. Результаты записываются в `reports/final_audit/`.

## Материалы для научной статьи

Ноутбук: `notebooks/04_article_materials.ipynb`.

Команда использует только сохранённые CSV/JSON и создаёт в `reports/article_materials/` русскоязычные таблицы, отобранные PNG/PDF-рисунки, методические заметки, выводы, ограничения, структуру статьи и проверенный перечень источников. Это комплект для следующего самостоятельного этапа подготовки рукописи, а не LaTeX-статья.

## Тесты

```powershell
uv run pytest
```

Тесты не требуют полного обучения. Они проверяют разбиение, отсутствие пересечения ID, расчёт метрик и загрузку моделей для инференса. Ноутбуки предназначены для интерактивного анализа и формирования отчётов.
