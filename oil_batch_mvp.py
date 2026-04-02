from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
import pandas as pd
from openai import APIConnectionError, APITimeoutError, OpenAI
from rapidfuzz import fuzz, process
from dotenv import load_dotenv

# =========================
# CONFIG
# =========================
SUPPORTED_IMAGE_EXT = {".png", ".jpg", ".jpeg"}
BATCH_COL_NAME = "Batch"
USED_COL_NAME = "Used"
REVIEW_SHEET_NAME = "NEEDS_REVIEW"
RESULTS_SHEET_NAME = "PARSED_LINES"
SUGGESTED_SHEET_NAME = "SUGGESTED_MATCHES"
OPENAI_MODEL_ENV = "OPENAI_VISION_MODEL"
OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
DEFAULT_OPENAI_MODEL = "gpt-4.1-mini"
DEBUG_OUTPUT_DIR_ENV = "OIL_BATCH_DEBUG_DIR"
logger = logging.getLogger(__name__)
load_dotenv()

# =========================
# DATA MODELS
# =========================
@dataclass
class ParsedLine:
    pdf_file: str
    page: int
    product: str
    qty_raw: str
    qty_value: float | None
    qty_unit: str | None
    batch: str
    confidence: float
    reason: str
    status: str  # ok / review / not_found / skipped


# =========================
# HELPERS
# =========================
def normalize_batch(value) -> str:
    if value is None:
        return ""

    # pandas часто читает пустые ячейки как NaN (float)
    if pd.isna(value):
        return ""

    value = str(value).strip()
    if not value:
        return ""

    value = value.upper()
    value = re.sub(r"\s+", "", value)
    value = re.sub(r"[^A-Z0-9-]", "", value)
    return value



def parse_qty(qty_raw: str | None) -> tuple[float | None, str | None]:
    if not qty_raw:
        return None, None

    text = qty_raw.strip().lower()
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*(oz|lb|lbs|ml|l)?", text)
    if not m:
        return None, None

    value = float(m.group(1))
    unit = (m.group(2) or "").lower() or None
    return value, unit



def ensure_required_columns(df: pd.DataFrame) -> None:
    missing = [c for c in (BATCH_COL_NAME, USED_COL_NAME) if c not in df.columns]
    if missing:
        raise ValueError(
            f"В базе нет обязательных колонок: {missing}. Ожидаю минимум '{BATCH_COL_NAME}' и '{USED_COL_NAME}'."
        )



def pdf_to_images(pdf_path: Path, output_dir: Path, dpi: int = 140) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(pdf_path)
    image_paths: list[Path] = []

    for page_index in range(len(doc)):
        page = doc.load_page(page_index)
        pix = page.get_pixmap(dpi=dpi)
        img_path = output_dir / f"{pdf_path.stem}_page_{page_index + 1}.png"
        pix.save(img_path)
        image_paths.append(img_path)

    return image_paths



def image_to_base64(image_path: Path) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# =========================
# AI PARSER
# =========================
def parse_page_with_ai(image_path: Path) -> list[dict[str, Any]]:
    """
    MVP-заглушка.

    Здесь нужно подключить vision-модель.
    Она должна вернуть JSON-список по всем строкам заказа на странице.

    Ожидаемый формат каждого элемента:
    {
      "product": "Abyssinian Oil",
      "qty_raw": "16 oz",
      "batch": "ABSNCACAC001A",
      "confidence": 0.97,
      "reason": "clear handwritten batch near product row"
    }

    ВАЖНО:
    - confidence от 0 до 1
    - batch уже лучше просить модель возвращать без пробелов
    - если batch не читается, пусть возвращает ""
    """
    api_key = os.getenv(OPENAI_API_KEY_ENV, "").strip()
    model_name = os.getenv(OPENAI_MODEL_ENV, DEFAULT_OPENAI_MODEL).strip() or DEFAULT_OPENAI_MODEL
    debug_dir_raw = os.getenv(DEBUG_OUTPUT_DIR_ENV, "").strip()
    debug_dir = Path(debug_dir_raw) if debug_dir_raw else None

    logger.info("AI parse started for page image: %s", image_path)

    def write_debug_payload(payload: Any, suffix: str) -> None:
        if not debug_dir:
            return
        try:
            debug_dir.mkdir(parents=True, exist_ok=True)
            debug_path = debug_dir / f"{image_path.stem}_{suffix}.json"
            with debug_path.open("w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=2)
            logger.info("Debug JSON saved: %s", debug_path)
        except Exception as exc:
            logger.warning("Failed to write debug JSON for %s: %s", image_path, exc)

    if not api_key:
        logger.error("Переменная окружения %s не задана. AI-разбор пропущен.", OPENAI_API_KEY_ENV)
        return []

    image_base64 = image_to_base64(image_path)
    client = OpenAI(api_key=api_key, timeout=120.0, max_retries=0)
    prompt = (
        "Ты анализируешь скан страницы заказа и должен извлечь batch-коды максимально точно.\n"
        "Важные правила:\n"
        "1) Для каждой товарной строки найди batch ТОЛЬКО как рукописный код на той же горизонтальной линии, "
        "справа или непосредственно рядом с этой строкой.\n"
        "2) Batch должен выглядеть как код (смесь букв и цифр, без случайных слов).\n"
        "3) Игнорируй посторонние пометки и шум: слова вроде 'neo', стрелки, служебные заметки, каракули.\n"
        "4) Не угадывай batch. Если код не виден или есть сомнение — верни пустую строку \"\" и низкий confidence.\n"
        "5) Не придумывай значения и не копируй код с другой строки.\n"
        "6) В reason кратко укажи, почему уверен/не уверен (например: 'код справа на одной линии' или 'нечитаемо').\n"
        "Верни СТРОГО JSON-объект формата: "
        '{"rows":[{"product":"...","qty_raw":"...","batch":"...","confidence":0.0,"reason":"..."}]}. '
        "Без markdown, без комментариев, без любого текста вне JSON."
    )

    max_attempts = 4
    response = None
    for attempt in range(1, max_attempts + 1):
        if attempt > 1:
            sleep_seconds = 2 ** (attempt - 1)  # 2s, 4s, 8s before retries
            logger.warning(
                "Retrying OpenAI call for %s: attempt %s/%s in %ss",
                image_path,
                attempt,
                max_attempts,
                sleep_seconds,
            )
            time.sleep(sleep_seconds)

        try:
            response = client.responses.create(
                model=model_name,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt},
                            {"type": "input_image", "image_url": f"data:image/png;base64,{image_base64}"},
                        ],
                    }
                ],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "batch_rows",
                        "schema": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "rows": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "additionalProperties": False,
                                        "properties": {
                                            "product": {"type": "string"},
                                            "qty_raw": {"type": "string"},
                                            "batch": {"type": "string"},
                                            "confidence": {"type": "number"},
                                            "reason": {"type": "string"},
                                       },
                                       "required": ["product", "qty_raw", "batch", "confidence", "reason"],
                                    },
                                },
                            },
                            "required": ["rows"],
                        },
                        "strict": True,
                    }
                },
            )
            break
        except (APITimeoutError, APIConnectionError) as exc:
            if attempt == max_attempts:
                logger.exception(
                    "OpenAI timeout/connect error после %s попыток для %s: %s",
                    max_attempts,
                    image_path,
                    exc,
                )
                return []
            logger.warning(
                "Timeout/connect ошибка OpenAI на попытке %s/%s для %s: %s",
                attempt,
                max_attempts,
                image_path,
                exc,
            )
        except Exception as exc:
            err_text = str(exc).lower()
            is_retryable_network = any(
                marker in err_text
                for marker in ("timeout", "connecttimeout", "handshake", "ssl", "connection reset")
            )
            if is_retryable_network and attempt < max_attempts:
                logger.warning(
                    "Временная сетевая ошибка OpenAI на попытке %s/%s для %s: %s",
                    attempt,
                    max_attempts,
                    image_path,
                    exc,
                )
                continue
            if is_retryable_network:
                logger.exception(
                    "OpenAI сетевая ошибка после %s попыток для %s: %s",
                    max_attempts,
                    image_path,
                    exc,
                )
                return []
            logger.exception("Ошибка вызова vision-модели для %s: %s", image_path, exc)
            return []

    if response is None:
        logger.error("OpenAI не вернул ответ после %s попыток для %s", max_attempts, image_path)
        return []

    raw_text = getattr(response, "output_text", "") or ""
    logger.info("Raw model response for %s: %s", image_path, raw_text[:2000])
    write_debug_payload({"image": str(image_path), "raw_response": raw_text}, "raw_response")
    try:
        payload = json.loads(raw_text)
    except Exception as exc:
        logger.exception("Невалидный JSON от модели для %s: %s; raw=%r", image_path, exc, raw_text[:500])
        return []

    if not isinstance(payload, dict):
        logger.error("Ожидался объект JSON верхнего уровня, но получено: %s", type(payload).__name__)
        return []

    rows = payload.get("rows", [])
    if not isinstance(rows, list):
        logger.error("Поле 'rows' должно быть списком, но получено: %s", type(rows).__name__)
        return []

    normalized_rows: list[dict[str, Any]] = []
    for idx, item in enumerate(rows):
        if not isinstance(item, dict):
            logger.error("Элемент #%s не объект: %r", idx, item)
            return []
        try:
            normalized_rows.append(
                {
                    "product": str(item.get("product", "")).strip(),
                    "qty_raw": str(item.get("qty_raw", "")).strip(),
                    "batch": str(item.get("batch", "")).strip(),
                    "confidence": float(item.get("confidence", 0.0) or 0.0),
                    "reason": str(item.get("reason", "")).strip(),
                }
            )
        except Exception as exc:
            logger.exception("Не удалось нормализовать элемент #%s: %r, err=%s", idx, item, exc)
            return []

    filtered_rows: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str, str]] = set()
    filtered_count = 0
    garbage_markers = ("ngbc", "wholesale", "waterless")

    for row in normalized_rows:
        product = str(row.get("product", "")).strip()
        qty_raw = str(row.get("qty_raw", "")).strip()
        batch = str(row.get("batch", "")).strip()
        confidence = float(row.get("confidence", 0.0) or 0.0)
        qty_value, _ = parse_qty(qty_raw)

        is_too_short = len(product) <= 3
        has_garbage_marker = any(marker in product.lower() for marker in garbage_markers)
        has_empty_or_zero_qty = (not qty_raw) or (qty_value in (None, 0.0))
        has_empty_batch_low_conf = (not batch) and confidence < 0.5

        if is_too_short or has_garbage_marker or has_empty_or_zero_qty or has_empty_batch_low_conf:
            filtered_count += 1
            continue

        dedupe_key = (product.lower(), qty_raw.lower(), batch.lower())
        if dedupe_key in seen_keys:
            filtered_count += 1
            continue
        seen_keys.add(dedupe_key)
        filtered_rows.append(row)

    write_debug_payload(filtered_rows, "normalized_rows")
    logger.info(
        "Parsed %s rows from %s (filtered out %s garbage/duplicate rows)",
        len(filtered_rows),
        image_path,
        filtered_count,
    )
    return filtered_rows


# =========================
# PIPELINE
# =========================
def parse_pdf(pdf_path: Path, work_dir: Path, confidence_threshold: float) -> list[ParsedLine]:
    images_dir = work_dir / "pages"
    image_paths = pdf_to_images(pdf_path, images_dir)

    results: list[ParsedLine] = []

    checkpoint_path = work_dir / "parsed_lines_checkpoint.json"

    def save_checkpoint() -> None:
        try:
            work_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_payload = [asdict(line) for line in results]
            checkpoint_path.write_text(
                json.dumps(checkpoint_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            logger.info("Checkpoint saved after page processing: %s", checkpoint_path)
        except Exception as exc:
            logger.warning("Failed to save checkpoint %s: %s", checkpoint_path, exc)

    for page_no, image_path in enumerate(image_paths, start=1):
        logger.info("Processing page %s (%s)", page_no, image_path.name)
        raw_items = parse_page_with_ai(image_path)
        logger.info("Page %s: recognized %s rows", page_no, len(raw_items))

        if not raw_items:
            results.append(
                ParsedLine(
                    pdf_file=pdf_path.name,
                    page=page_no,
                    product="",
                    qty_raw="",
                    qty_value=None,
                    qty_unit=None,
                    batch="",
                    confidence=0.0,
                    reason="AI parser returned no rows",
                    status="review",
                )
            )
            save_checkpoint()
            if page_no < len(image_paths):
                time.sleep(1)
            continue

        for item in raw_items:
            product = str(item.get("product", "")).strip()
            qty_raw = str(item.get("qty_raw", "")).strip()
            batch = normalize_batch(str(item.get("batch", "")))
            confidence = float(item.get("confidence", 0.0) or 0.0)
            reason = str(item.get("reason", "")).strip()
            qty_value, qty_unit = parse_qty(qty_raw)

            if not batch or confidence < confidence_threshold:
                status = "review"
            else:
                status = "ok"

            results.append(
                ParsedLine(
                    pdf_file=pdf_path.name,
                    page=page_no,
                    product=product,
                    qty_raw=qty_raw,
                    qty_value=qty_value,
                    qty_unit=qty_unit,
                    batch=batch,
                    confidence=confidence,
                    reason=reason,
                    status=status,
                )
            )

        save_checkpoint()
        if page_no < len(image_paths):
            time.sleep(1)

    return results



def build_updates(parsed_lines: list[ParsedLine], batch_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ensure_required_columns(batch_df)

    df = pd.DataFrame([asdict(x) for x in parsed_lines])
    if df.empty:
        df = pd.DataFrame(
            columns=[
                "pdf_file",
                "page",
                "product",
                "qty_raw",
                "qty_value",
                "qty_unit",
                "batch",
                "confidence",
                "reason",
                "status",
            ]
        )

    batch_df = batch_df.copy()
    batch_df["_batch_norm"] = batch_df[BATCH_COL_NAME].astype(str).map(normalize_batch)
    batch_norm_to_original: dict[str, list[str]] = {}
    for _, batch_row in batch_df.iterrows():
        batch_norm = normalize_batch(batch_row.get(BATCH_COL_NAME))
        if not batch_norm:
            continue
        batch_norm_to_original.setdefault(batch_norm, []).append(str(batch_row.get(BATCH_COL_NAME, "")).strip())
    fuzzy_candidates = list(batch_norm_to_original.keys())

    ok_rows = []
    review_rows = []

    for _, row in df.iterrows():
        batch = normalize_batch(row.get("batch"))
        qty_value = row.get("qty_value")
        qty_unit_raw = row.get("qty_unit")
        if pd.isna(qty_unit_raw):
            qty_unit = ""
        else:
            qty_unit = str(qty_unit_raw).lower()
        base_status = row.get("status", "review")

        out_row = row.to_dict()
        out_row["suggested_batch"] = ""
        out_row["similarity_score"] = None

        if base_status != "ok":
            out_row["review_note"] = "Низкая уверенность или batch пустой"
            review_rows.append(out_row)
            continue

        if qty_value is None:
            out_row["status"] = "review"
            out_row["review_note"] = "Не удалось распарсить количество"
            review_rows.append(out_row)
            continue

        if qty_unit not in {"oz", ""}:
            out_row["status"] = "review"
            out_row["review_note"] = f"Пока автоматически обновляем только oz. Найдено: {qty_unit}"
            review_rows.append(out_row)
            continue

        mask = batch_df["_batch_norm"] == batch
        matched = batch_df[mask]

        if matched.empty:
            if batch and fuzzy_candidates:
                best_match = process.extractOne(batch, fuzzy_candidates, scorer=fuzz.ratio)
                if best_match:
                    best_batch_norm, similarity_score, _ = best_match
                    out_row["similarity_score"] = float(similarity_score)
                    if similarity_score >= 90:
                        suggested_values = batch_norm_to_original.get(best_batch_norm, [])
                        out_row["status"] = "suggested_match"
                        out_row["suggested_batch"] = suggested_values[0] if suggested_values else best_batch_norm
                        out_row["review_note"] = (
                            "Точного совпадения нет. Предложен ближайший batch по fuzzy matching "
                            f"(similarity={similarity_score:.1f})."
                        )
                        review_rows.append(out_row)
                        continue
            out_row["status"] = "not_found"
            out_row["review_note"] = "Batch не найден в базе"
            review_rows.append(out_row)
            continue

        if len(matched) > 1:
            out_row["status"] = "review"
            out_row["review_note"] = "В базе найдено несколько одинаковых batch"
            review_rows.append(out_row)
            continue

        idx = matched.index[0]
        current_used = batch_df.at[idx, USED_COL_NAME]
        current_used = 0 if pd.isna(current_used) else float(current_used)
        batch_df.at[idx, USED_COL_NAME] = current_used + float(qty_value)

        out_row["status"] = "updated"
        out_row["review_note"] = "Used увеличен автоматически"
        ok_rows.append(out_row)

    batch_df = batch_df.drop(columns=["_batch_norm"])
    ok_df = pd.DataFrame(ok_rows)
    review_df = pd.DataFrame(review_rows)
    return batch_df, pd.concat([ok_df, review_df], ignore_index=True)



def save_results(output_xlsx: Path, updated_batch_df: pd.DataFrame, parsed_df: pd.DataFrame) -> None:
    with pd.ExcelWriter(output_xlsx, engine="openpyxl") as writer:
        updated_batch_df.to_excel(writer, sheet_name="BATCH_DB_UPDATED", index=False)
        parsed_df.to_excel(writer, sheet_name=RESULTS_SHEET_NAME, index=False)

        review_df = parsed_df[parsed_df["status"].isin(["review", "not_found"])] if not parsed_df.empty else parsed_df
        review_df.to_excel(writer, sheet_name=REVIEW_SHEET_NAME, index=False)
        suggested_df = (
            parsed_df[parsed_df["status"] == "suggested_match"] if not parsed_df.empty else parsed_df
        )
        suggested_df.to_excel(writer, sheet_name=SUGGESTED_SHEET_NAME, index=False)


# =========================
# CLI
# =========================
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="MVP: разбор PDF заказов и обновление базы batch/used")
    parser.add_argument("--pdf", required=True, help="Путь к PDF файлу")
    parser.add_argument("--batch-db", required=True, help="Путь к Excel базе batch")
    parser.add_argument("--out", default="result.xlsx", help="Куда сохранить результат")
    parser.add_argument("--work-dir", default="./work", help="Временная рабочая папка")
    parser.add_argument("--confidence-threshold", type=float, default=0.85)
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    batch_db_path = Path(args.batch_db)
    output_path = Path(args.out)
    work_dir = Path(args.work_dir)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF не найден: {pdf_path}")
    if not batch_db_path.exists():
        raise FileNotFoundError(f"Excel база не найдена: {batch_db_path}")

    batch_df = pd.read_excel(batch_db_path)
    print("Колонки Excel:", list(batch_df.columns))
    print(batch_df.head(10))
    parsed_lines = parse_pdf(pdf_path, work_dir, args.confidence_threshold)
    updated_batch_df, parsed_df = build_updates(parsed_lines, batch_df)
    save_results(output_path, updated_batch_df, parsed_df)

    print(f"Готово. Результат сохранён в: {output_path}")


if __name__ == "__main__":
    main()
