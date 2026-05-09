from __future__ import annotations

import argparse
import ast
import csv
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_CORPUS_CSV = Path("src/data/maintreg_database_clean.csv")
DEFAULT_MISSING_METADATA_CSV = Path("src/data/missing_metadata.csv")

DATE_PATTERN = re.compile(r"\d{2}\.\d{2}\.\d{4}")


def clean_fieldname(value: str) -> str:
    return value.lstrip("\ufeff").strip('"')


def read_dict_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        raw_fieldnames = reader.fieldnames
        if not raw_fieldnames:
            raise ValueError(f"No header found in {path}")

        fieldnames = [clean_fieldname(fieldname) for fieldname in raw_fieldnames]
        rows = []
        for row in reader:
            clean_row = {
                fieldname: row.get(raw_fieldname, "")
                for raw_fieldname, fieldname in zip(raw_fieldnames, fieldnames)
            }
            rows.append(clean_row)

    return fieldnames, rows


def is_blank(value: Any) -> bool:
    if value is None:
        return True
    text = str(value).strip()
    return text == "" or text.lower() in {"nan", "none", "null", "na"}


def parse_metadata(value: Any) -> dict[str, Any]:
    if is_blank(value):
        return {}
    if isinstance(value, dict):
        return value

    try:
        parsed = ast.literal_eval(str(value))
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def timestamp_ms_to_date(value: Any) -> str:
    if is_blank(value):
        return ""
    try:
        timestamp_ms = int(float(str(value)))
    except ValueError:
        return ""

    return datetime.fromtimestamp(
        timestamp_ms / 1000,
        tz=timezone.utc,
    ).strftime("%d.%m.%Y")


def date_from_metadata(metadata: dict[str, Any]) -> str:
    date_debut = str(metadata.get("date_debut") or "").strip()
    if DATE_PATTERN.fullmatch(date_debut):
        return date_debut

    return timestamp_ms_to_date(metadata.get("version_date_debut"))


def build_metadata_by_article_id(
    missing_metadata_csv: Path,
) -> tuple[dict[str, str], dict[str, str], list[str]]:
    date_by_id: dict[str, str] = {}
    metadata_by_id: dict[str, str] = {}
    warnings: list[str] = []

    _, rows = read_dict_rows(missing_metadata_csv)
    for row_number, row in enumerate(rows, start=2):
        metadata_text = str(row.get("metadata") or "").strip()
        metadata = parse_metadata(row.get("metadata"))
        article_id = str(metadata.get("id") or row.get("id") or "").strip()
        date = date_from_metadata(metadata)

        if not article_id:
            continue

        if metadata_text:
            previous_metadata = metadata_by_id.get(article_id)
            if previous_metadata and previous_metadata != metadata_text:
                warnings.append(
                    f"Conflicting metadata for {article_id} at row {row_number}: "
                    "kept first value"
                )
            else:
                metadata_by_id[article_id] = metadata_text

        if date:
            previous = date_by_id.get(article_id)
            if previous and previous != date:
                warnings.append(
                    f"Conflicting dates for {article_id} at row {row_number}: "
                    f"kept {previous}, ignored {date}"
                )
                continue

            date_by_id[article_id] = date

    return date_by_id, metadata_by_id, warnings


def enrich_corpus_metadata(
    corpus_csv: Path,
    missing_metadata_csv: Path,
    output_csv: Path,
) -> tuple[int, int, int, list[str]]:
    date_by_id, metadata_by_id, warnings = build_metadata_by_article_id(
        missing_metadata_csv
    )

    fieldnames, rows = read_dict_rows(corpus_csv)
    if "id" not in fieldnames or "date" not in fieldnames:
        raise ValueError("Corpus CSV must contain 'id' and 'date' columns.")
    if "metadata" not in fieldnames:
        fieldnames.append("metadata")
        for row in rows:
            row["metadata"] = ""

    filled_dates = 0
    filled_metadata = 0
    still_missing_ids: list[str] = []

    for row in rows:
        article_id = str(row.get("id") or "").strip()

        recovered_metadata = metadata_by_id.get(article_id)
        if recovered_metadata and row.get("metadata") != recovered_metadata:
            row["metadata"] = recovered_metadata
            filled_metadata += 1

        if not is_blank(row.get("date")):
            continue

        recovered_date = date_by_id.get(article_id)
        if recovered_date:
            row["date"] = recovered_date
            filled_dates += 1
        elif article_id not in still_missing_ids:
            still_missing_ids.append(article_id)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            quoting=csv.QUOTE_ALL,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    return filled_dates, filled_metadata, len(still_missing_ids), warnings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fill empty corpus date values and add a metadata column from "
            "src/data/missing_metadata.csv."
        )
    )
    parser.add_argument("--corpus-csv", type=Path, default=DEFAULT_CORPUS_CSV)
    parser.add_argument(
        "--missing-metadata-csv",
        type=Path,
        default=DEFAULT_MISSING_METADATA_CSV,
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Defaults to overwriting --corpus-csv.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_csv = args.output_csv or args.corpus_csv

    filled_dates, filled_metadata, still_missing, warnings = enrich_corpus_metadata(
        corpus_csv=args.corpus_csv,
        missing_metadata_csv=args.missing_metadata_csv,
        output_csv=output_csv,
    )

    print(f"Filled dates: {filled_dates}")
    print(f"Filled metadata values: {filled_metadata}")
    print(f"Still missing dates: {still_missing}")
    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"- {warning}")
    print(f"Output CSV: {output_csv}")


if __name__ == "__main__":
    main()
