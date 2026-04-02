# oil_batch MVP (Windows quick start)

Этот проект нужен для первого локального прогона:
- берёт `ntcnpdf.pdf`,
- парсит строки заказа через vision-модель,
- обновляет `Example.xlsx`,
- создаёт `result.xlsx`.

## 1) Установить Python

1. Скачайте Python 3.10+ с официального сайта: https://www.python.org/downloads/windows/  
2. При установке обязательно включите галочку **Add Python to PATH**.

Проверка в PowerShell:

```powershell
python --version
```

## 2) Открыть папку проекта

В PowerShell:

```powershell
cd C:\path\to\oil_batch
```

## 3) Создать `.env` из `.env.example`

```powershell
copy .env.example .env
notepad .env
```

Вставьте ваш API-ключ в `OPENAI_API_KEY`.

## 4) Установить зависимости

Рекомендуется через виртуальное окружение:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 5) Положить входные файлы рядом со скриптом

В папке проекта должны лежать:
- `oil_batch_mvp.py`
- `ntcnpdf.pdf`
- `Example.xlsx`

## 6) Запуск одной командой

```powershell
python oil_batch_mvp.py --pdf .\ntcnpdf.pdf --batch-db .\Example.xlsx --out .\result.xlsx
```

После успешного запуска появится файл:
- `result.xlsx`

---

## Пример готового `.env`

```env
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx
OPENAI_VISION_MODEL=gpt-4.1-mini
OIL_BATCH_DEBUG_DIR=.\work\debug_ai
```

`OIL_BATCH_DEBUG_DIR` можно оставить пустым, если debug-json не нужен.
