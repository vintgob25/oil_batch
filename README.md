# oil_batch MVP

MVP-скрипт для извлечения товарных строк из PDF-заказов и обновления Excel-базы batch/used.

## Что делает

`oil_batch_mvp.py`:
- конвертирует страницы PDF в PNG;
- отправляет каждую страницу в vision-модель;
- получает по каждой строке заказа:
  - `product`
  - `qty_raw`
  - `batch`
  - `confidence`
  - `reason`
- нормализует batch-коды и confidence;
- обновляет колонку `Used` в базе по найденным batch;
- сохраняет итоговый Excel с листами:
  - `BATCH_DB_UPDATED`
  - `PARSED_LINES`
  - `NEEDS_REVIEW`

## Требования

- Python 3.10+
- OpenAI API key

Установка зависимостей:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Переменные окружения

- `OPENAI_API_KEY` — API ключ (обязательно)
- `OPENAI_VISION_MODEL` — имя модели (опционально, по умолчанию `gpt-4.1-mini`)
- `OIL_BATCH_DEBUG_DIR` — опционально, папка для debug JSON по страницам (raw/normalized ответы модели)

Пример:

```bash
cp .env.example .env
# отредактируйте .env
set -a
source .env
set +a
```

## Запуск

```bash
python oil_batch_mvp.py \
  --pdf ./sample.pdf \
  --batch-db ./Example.xlsx \
  --out ./result.xlsx
```

Опционально:

```bash
python oil_batch_mvp.py \
  --pdf ./sample.pdf \
  --batch-db ./Example.xlsx \
  --out ./result.xlsx \
  --work-dir ./work \
  --confidence-threshold 0.85
```
