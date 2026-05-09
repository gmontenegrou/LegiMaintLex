from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import re
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_PROVIDER = "mistral"
DEFAULT_INPUT_CSV = Path("src/data/maintreg_database_clean.csv")
ARTICLE_NUMBER_RE = re.compile(
    r"^\s*(?:Art\.|Article)\s+(.{1,32}?)(?:\s*\.\s*-|\s*[.):-]|\s*$)",
    re.IGNORECASE,
)
CURRENT_LEGAL_ACT_TYPES = {"artifact", "source"}
CURRENT_LEGAL_ACT_RE = re.compile(
    r"^(?:(?:l|le|la|les|du|de|des|au|aux|a|ce|cet|cette)\s+)*"
    r"present(?:e|s|es)?\s+"
    r"(?:arrete|arret|decret|loi|decision|ordonnance|reglement|acte|texte)$"
)

CTX_COLS = [
    "document_key",
    "article_id",
    "article_key",
    "article_number",
    "corpus_id",
    "date",
]

TRIPLET_COLS = CTX_COLS + [
    "triplet_index",
    "head",
    "head_type",
    "head_resolution_mode",
    "head_resolved_entity_id",
    "relation",
    "tail",
    "tail_type",
    "tail_resolution_mode",
    "tail_resolved_entity_id",
    "topic",
    "head_start",
    "head_end",
    "head_spans",
    "head_span_text",
    "head_span_mode",
    "tail_start",
    "tail_end",
    "tail_spans",
    "tail_span_text",
    "tail_span_mode",
]

ENTITY_COLS = CTX_COLS + [
    "triplet_index",
    "entity_role",
    "entity",
    "entity_type",
    "entity_resolution_mode",
    "resolved_entity_id",
    "has_mention",
    "entity_start",
    "entity_end",
    "entity_spans",
    "entity_span_text",
    "entity_span_mode",
]

MENTION_COLS = CTX_COLS + [
    "triplet_index",
    "mention_role",
    "mention",
    "mention_type",
    "mention_resolution_mode",
    "resolved_entity_id",
    "mention_start",
    "mention_end",
    "mention_spans",
    "mention_text",
    "mention_span_mode",
]


def set_max_csv_field_size() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    set_max_csv_field_size()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    extras = sorted({key for row in rows for key in row} - set(columns))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns + extras)
        writer.writeheader()
        writer.writerows(rows)


def blank(value: Any) -> bool:
    return value is None or str(value).strip().lower() in {
        "",
        "nan",
        "none",
        "null",
        "na",
    }


def first(*values: Any) -> str:
    for value in values:
        if not blank(value):
            return str(value)
    return ""


def normalized_text(value: Any) -> str:
    return re.sub(r"\s+", " ", first(value)).strip()


def folded_text(value: Any) -> str:
    text = re.sub(r"[’‘ʼ`´]", " ", normalized_text(value))
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = text.encode("ascii", "ignore").decode("ascii").lower()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def slug(value: Any, max_len: int = 80) -> str:
    text = re.sub(r"[’‘ʼ`´]", " ", normalized_text(value))
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-").lower()
    return (text or "document")[:max_len].strip("-") or "document"


def short_hash(*values: Any) -> str:
    text = "||".join(normalized_text(value) for value in values)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def as_int(value: Any) -> int | None:
    if blank(value):
        return None
    try:
        return int(float(str(value).strip()))
    except ValueError:
        return None


def parse_literal(value: Any) -> Any:
    if blank(value):
        return None
    if isinstance(value, (dict, list)):
        return value
    text = str(value).strip()
    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(re.sub(r"\bnan\b", "None", text, flags=re.IGNORECASE))
        except Exception:
            pass
    return None


def as_mapping(value: Any) -> dict[str, Any]:
    parsed = parse_literal(value)
    return parsed if isinstance(parsed, dict) else {}


def as_triplets(value: Any) -> list[dict[str, Any]]:
    parsed = parse_literal(value)
    if isinstance(parsed, list):
        return [item for item in parsed if isinstance(item, dict)]
    if isinstance(parsed, dict) and isinstance(parsed.get("triplets"), list):
        return [item for item in parsed["triplets"] if isinstance(item, dict)]
    return []


def date_to_xsd(value: Any) -> str:
    if blank(value):
        return ""

    text = str(value).strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        return text
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}T.+", text):
        return text[:10]

    match = re.fullmatch(r"(\d{2})[./-](\d{2})[./-](\d{4})", text)
    if match:
        day, month, year = match.groups()
        return f"{year}-{month}-{day}"

    try:
        timestamp = float(text)
    except ValueError:
        return ""

    if timestamp > 10_000_000_000:
        timestamp /= 1000
    try:
        return datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime("%Y-%m-%d")
    except (OSError, OverflowError, ValueError):
        return ""


def extract_article_number(content: Any) -> str:
    text = first(content)
    match = ARTICLE_NUMBER_RE.search(text)
    if not match:
        return ""
    return re.sub(r"\s+", " ", match.group(1)).strip()


def article_number(
    input_row: dict[str, str],
    provenance: dict[str, Any],
    content: str,
) -> str:
    return first(
        input_row.get("num"),
        provenance.get("number"),
        extract_article_number(content),
    )


def article_key(article_id: str, number: str, source_row_id: str) -> str:
    return first(number, f"{article_id}-{source_row_id}", article_id, source_row_id)


def document_key(input_row: dict[str, str], provenance: dict[str, Any]) -> str:
    title = normalized_text(
        first(
            input_row.get("title"),
            provenance.get("title"),
            input_row.get("corpus_id"),
            input_row.get("id"),
            "document",
        )
    )
    return f"{slug(title)}-{short_hash(title)}"


def is_current_legal_act_reference(label: str, type_value: str) -> bool:
    type_folded = folded_text(type_value)
    return (
        type_folded in CURRENT_LEGAL_ACT_TYPES
        and CURRENT_LEGAL_ACT_RE.search(folded_text(label)) is not None
    )


def entity_resolution(
    label: str,
    type_value: str,
    document_key_value: str,
) -> tuple[str, str]:
    if is_current_legal_act_reference(label, type_value):
        return "current_legal_act", document_key_value
    return "global_entity", ""


def get_segments(triplet: dict[str, Any], side: str) -> list[tuple[int, int]]:
    spans = triplet.get(f"{side}_spans")
    if isinstance(spans, list):
        segments = [
            (item.get("start"), item.get("end"))
            for item in spans
            if isinstance(item, dict)
        ]
        segments = [
            (start, end)
            for start, end in segments
            if isinstance(start, int) and isinstance(end, int) and end > start
        ]
        if segments:
            return segments

    start, end = triplet.get(f"{side}_start"), triplet.get(f"{side}_end")
    return (
        [(start, end)]
        if isinstance(start, int) and isinstance(end, int) and end > start
        else []
    )


def clip(segments: list[tuple[int, int]], text: str) -> list[tuple[int, int]]:
    size = len(text)
    return [
        (max(0, min(start, size)), max(0, min(end, size)))
        for start, end in segments
        if max(0, min(end, size)) > max(0, min(start, size))
    ]


def infer_segment(text: str, mention: str) -> list[tuple[int, int]]:
    if not mention:
        return []
    start = text.find(mention)
    if start >= 0:
        return [(start, start + len(mention))]
    match = re.search(re.escape(mention), text, flags=re.IGNORECASE)
    return [(match.start(), match.end())] if match else []


def side_span(
    triplet: dict[str, Any],
    side: str,
    content: str,
    infer_missing_spans: bool,
) -> tuple[list[tuple[int, int]], str]:
    segments = clip(get_segments(triplet, side), content)
    if segments:
        return segments, "embedded"
    if infer_missing_spans:
        segments = infer_segment(content, first(triplet.get(side)))
        if segments:
            return segments, "inferred"
    return [], ""


def span_start_end(segments: list[tuple[int, int]]) -> tuple[str, str]:
    return (str(segments[0][0]), str(segments[-1][1])) if segments else ("", "")


def span_string(segments: list[tuple[int, int]]) -> str:
    return "; ".join(f"{start}-{end}" for start, end in segments)


def span_text(text: str, segments: list[tuple[int, int]]) -> str:
    return " [...] ".join(text[start:end] for start, end in segments).strip()


def indexed_row(
    rows: list[dict[str, str]],
    source_index: int | None,
    fallback_index: int,
) -> dict[str, str]:
    if source_index is not None and 0 <= source_index < len(rows):
        return rows[source_index]
    return rows[fallback_index] if 0 <= fallback_index < len(rows) else {}


def article_context(
    input_row: dict[str, str],
    provenance: dict[str, Any],
    document_key_value: str,
    article_id: str,
    article_key_value: str,
    article_number_value: str,
) -> dict[str, str]:
    return {
        "document_key": document_key_value,
        "article_id": article_id,
        "article_key": article_key_value,
        "article_number": article_number_value,
        "corpus_id": first(input_row.get("corpus_id"), provenance.get("corpus_id")),
        "date": date_to_xsd(input_row.get("date")),
    }


def triplets_column(rows: list[dict[str, str]], preferred: str) -> str:
    columns = set(rows[0]) if rows else set()
    for candidate in (preferred, "legal_triplets_with_spans", "legal_triplets"):
        if candidate in columns:
            return candidate
    raise ValueError("No triplets column found.")


def flatten(
    input_csv: Path,
    results_csv: Path,
    preferred_triplets_column: str,
    infer_missing_spans: bool,
    max_rows: int | None,
) -> tuple[
    list[dict[str, str]], list[dict[str, str]], list[dict[str, str]], dict[str, int]
]:
    input_rows = read_csv(input_csv)
    result_rows = read_csv(results_csv)[:max_rows]
    column = triplets_column(result_rows, preferred_triplets_column)
    triplet_rows: list[dict[str, str]] = []
    entity_rows: list[dict[str, str]] = []
    mention_rows: list[dict[str, str]] = []
    missing = {"head": 0, "tail": 0}

    for result_pos, result_row in enumerate(result_rows):
        source_index = as_int(result_row.get("index"))
        input_row = indexed_row(input_rows, source_index, result_pos)
        provenance = as_mapping(result_row.get("provenance"))
        article_id = first(
            input_row.get("id"),
            provenance.get("id_local"),
            result_row.get("article_id"),
        )
        content = first(result_row.get("content"), input_row.get("content"))
        source_row_id = first(result_row.get("index"), result_pos)
        document_key_value = document_key(input_row, provenance)
        article_number_value = article_number(input_row, provenance, content)
        article_key_value = article_key(article_id, article_number_value, source_row_id)
        context = article_context(
            input_row,
            provenance,
            document_key_value,
            article_id,
            article_key_value,
            article_number_value,
        )

        for fallback_idx, triplet in enumerate(as_triplets(result_row.get(column))):
            triplet_idx = first(triplet.get("triplet_index"), fallback_idx)
            head = first(triplet.get("head"))
            head_type = first(triplet.get("head_type"))
            tail = first(triplet.get("tail"))
            tail_type = first(triplet.get("tail_type"))
            head_resolution_mode, head_resolved_id = entity_resolution(
                head,
                head_type,
                document_key_value,
            )
            tail_resolution_mode, tail_resolved_id = entity_resolution(
                tail,
                tail_type,
                document_key_value,
            )
            spans = {
                "head": side_span(triplet, "head", content, infer_missing_spans),
                "tail": side_span(triplet, "tail", content, infer_missing_spans),
            }
            for role, (segments, _) in spans.items():
                if not segments:
                    missing[role] += 1

            triplet_row = {
                **context,
                "triplet_index": triplet_idx,
                "head": head,
                "head_type": head_type,
                "head_resolution_mode": head_resolution_mode,
                "head_resolved_entity_id": head_resolved_id,
                "relation": first(triplet.get("relation")),
                "tail": tail,
                "tail_type": tail_type,
                "tail_resolution_mode": tail_resolution_mode,
                "tail_resolved_entity_id": tail_resolved_id,
                "topic": first(
                    triplet.get("topic"), result_row.get("context_main_topic")
                ),
            }
            for role in ("head", "tail"):
                segments, mode = spans[role]
                start, end = span_start_end(segments)
                triplet_row[f"{role}_start"] = start
                triplet_row[f"{role}_end"] = end
                triplet_row[f"{role}_spans"] = span_string(segments)
                triplet_row[f"{role}_span_text"] = span_text(content, segments)
                triplet_row[f"{role}_span_mode"] = mode
            triplet_rows.append(triplet_row)

            for role in ("head", "tail"):
                segments, mode = spans[role]
                start, end = span_start_end(segments)
                label = triplet_row[role]
                type_ = triplet_row[f"{role}_type"]
                resolution_mode = triplet_row[f"{role}_resolution_mode"]
                resolved_id = triplet_row[f"{role}_resolved_entity_id"]
                entity_rows.append(
                    {
                        **context,
                        "triplet_index": triplet_idx,
                        "entity_role": role,
                        "entity": label,
                        "entity_type": type_,
                        "entity_resolution_mode": resolution_mode,
                        "resolved_entity_id": resolved_id,
                        "has_mention": "true" if segments else "false",
                        "entity_start": start,
                        "entity_end": end,
                        "entity_spans": span_string(segments),
                        "entity_span_text": span_text(content, segments),
                        "entity_span_mode": mode,
                    }
                )
                if segments:
                    mention_rows.append(
                        {
                            **context,
                            "triplet_index": triplet_idx,
                            "mention_role": role,
                            "mention": label,
                            "mention_type": type_,
                            "mention_resolution_mode": resolution_mode,
                            "resolved_entity_id": resolved_id,
                            "mention_start": start,
                            "mention_end": end,
                            "mention_spans": span_string(segments),
                            "mention_text": span_text(content, segments),
                            "mention_span_mode": mode,
                        }
                    )

    stats = {
        "triplets": len(triplet_rows),
        "entities": len(entity_rows),
        "mentions": len(mention_rows),
        "missing_head_spans": missing["head"],
        "missing_tail_spans": missing["tail"],
    }
    return triplet_rows, entity_rows, mention_rows, stats


def default_results_csv(provider: str) -> Path:
    root = Path("exp/kg") / provider
    candidates = {
        "mistral": [
            "full_constrained_extraction_by_mistral_mistral-large-latest_nrows_6370.csv",
        ],
        "openai": [
            "full_constrained_extraction_by_openai_gpt-4.1_nrows_6370.csv",
        ],
    }
    for name in candidates.get(provider, []):
        path = root / name
        if path.exists():
            return path
    for path in sorted(root.glob("full_constrained_extraction_by_*.csv")):
        if not path.name.endswith("_input.csv"):
            return path
    raise FileNotFoundError(f"No results CSV found for provider: {provider}")


def default_flat_output(
    results_csv: Path,
    kind: str,
    output_dir: Path | None = None,
) -> Path:
    return (output_dir or results_csv.parent) / f"{results_csv.stem}_{kind}_flat.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Flatten KG extraction results to CSV."
    )
    parser.add_argument("--provider", default=DEFAULT_PROVIDER)
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--results-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--triplets-output-csv", type=Path, default=None)
    parser.add_argument("--entities-output-csv", type=Path, default=None)
    parser.add_argument("--mentions-output-csv", type=Path, default=None)
    parser.add_argument("--triplets-column", default="legal_triplets")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--no-infer-spans", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_csv = args.results_csv or default_results_csv(args.provider)
    triplets, entities, mentions, stats = flatten(
        input_csv=args.input_csv,
        results_csv=results_csv,
        preferred_triplets_column=args.triplets_column,
        infer_missing_spans=not args.no_infer_spans,
        max_rows=args.max_rows,
    )
    outputs = {
        "Triplets": (
            args.triplets_output_csv
            or default_flat_output(results_csv, "legal_triplets", args.output_dir),
            triplets,
            TRIPLET_COLS,
        ),
        "Entities": (
            args.entities_output_csv
            or default_flat_output(results_csv, "legal_entities", args.output_dir),
            entities,
            ENTITY_COLS,
        ),
        "Mentions": (
            args.mentions_output_csv
            or default_flat_output(results_csv, "legal_mentions", args.output_dir),
            mentions,
            MENTION_COLS,
        ),
    }
    for label, (path, rows, columns) in outputs.items():
        write_csv(path, rows, columns)
        print(f"{label}: {len(rows)} -> {path}")

    print(f"Input CSV: {args.input_csv}")
    print(f"Results CSV: {results_csv}")
    print(f"Missing head spans: {stats['missing_head_spans']}")
    print(f"Missing tail spans: {stats['missing_tail_spans']}")


if __name__ == "__main__":
    main()
