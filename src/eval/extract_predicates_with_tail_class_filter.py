from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


DEFAULT_INPUT_CSV = Path(
    "exp/new_ontology/openai/ontology_extended_openai_from_kg.csv"
)
DEFAULT_OUTPUT_DIR = Path("exp/evaluation_results/kg_predicate_tail_class")

CLASSES = [
    "Action",
    "Actor",
    "Artifact",
    "Condition",
    "Definition",
    "Location",
    "Modality",
    "Reason",
    "Reference",
    "Situation",
    "Source",
    "Time",
]


def is_blank(value: Any) -> bool:
    if value is None:
        return True
    return str(value).strip().lower() in {"", "nan", "none", "null", "na"}


def local_name(value: Any) -> str:
    if is_blank(value):
        return ""

    text = str(value).strip()
    if "#" in text:
        return text.rsplit("#", 1)[-1]
    if "/" in text:
        return text.rstrip("/").rsplit("/", 1)[-1]
    if ":" in text:
        return text.split(":", 1)[-1]
    return text


def as_int(value: Any, default: int = 0) -> int:
    if is_blank(value):
        return default
    try:
        return int(float(str(value).strip()))
    except ValueError:
        return default


def infer_expected_tail_classes(
    predicate_local: str,
    class_names: Iterable[str],
) -> list[str]:
    predicate_text = str(predicate_local).strip().lower()
    if not predicate_text:
        return []
    return [class_name for class_name in class_names if class_name.lower() in predicate_text]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def annotated_signature_rows(
    rows: list[dict[str, str]],
    class_names: list[str],
) -> list[dict[str, Any]]:
    annotated = []

    for row in rows:
        predicate = row.get("predicate", "")
        predicate_local = row.get("predicate_local") or local_name(predicate)
        range_iri = row.get("range", "")
        range_local = row.get("range_local") or local_name(range_iri)
        expected_tail_classes = infer_expected_tail_classes(
            predicate_local=predicate_local,
            class_names=class_names,
        )
        support = as_int(row.get("signature_support_count"), default=1)

        has_tail_class_hint = bool(expected_tail_classes)
        tail_class_matches_hint = (
            not has_tail_class_hint or range_local in expected_tail_classes
        )
        evaluation_status = (
            "no_tail_class_hint"
            if not has_tail_class_hint
            else (
                "kept_tail_class_match"
                if tail_class_matches_hint
                else "excluded_tail_class_mismatch"
            )
        )

        out = dict(row)
        out.update(
            {
                "predicate_local": predicate_local,
                "range_local": range_local,
                "expected_tail_classes_from_predicate": "|".join(
                    expected_tail_classes
                ),
                "has_tail_class_hint": str(has_tail_class_hint).lower(),
                "tail_class_matches_hint": str(tail_class_matches_hint).lower(),
                "evaluation_status": evaluation_status,
                "signature_support_count": support,
            }
        )
        annotated.append(out)

    return annotated


def predicate_review_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("predicate", ""))].append(row)

    review_rows = []
    for predicate, group in grouped.items():
        predicate_local = str(group[0].get("predicate_local", local_name(predicate)))
        expected_tail_classes = sorted(
            {
                class_name
                for row in group
                for class_name in str(
                    row.get("expected_tail_classes_from_predicate", "")
                ).split("|")
                if class_name
            }
        )
        matching_ranges = sorted(
            {
                str(row.get("range_local", ""))
                for row in group
                if row.get("evaluation_status") == "kept_tail_class_match"
            }
        )
        mismatching_ranges = sorted(
            {
                str(row.get("range_local", ""))
                for row in group
                if row.get("evaluation_status") == "excluded_tail_class_mismatch"
            }
        )
        no_hint_ranges = sorted(
            {
                str(row.get("range_local", ""))
                for row in group
                if row.get("evaluation_status") == "no_tail_class_hint"
            }
        )

        total_support = sum(as_int(row.get("signature_support_count")) for row in group)
        excluded_support = sum(
            as_int(row.get("signature_support_count"))
            for row in group
            if row.get("evaluation_status") == "excluded_tail_class_mismatch"
        )
        kept_support = total_support - excluded_support

        if not expected_tail_classes:
            predicate_status = "no_tail_class_hint"
        elif mismatching_ranges:
            predicate_status = "review_has_mismatching_tail_class"
        else:
            predicate_status = "all_tail_classes_match"

        review_rows.append(
            {
                "predicate": predicate,
                "predicate_local": predicate_local,
                "predicate_status": predicate_status,
                "expected_tail_classes_from_predicate": "|".join(
                    expected_tail_classes
                ),
                "matching_tail_classes": "|".join(matching_ranges),
                "mismatching_tail_classes": "|".join(mismatching_ranges),
                "no_hint_tail_classes": "|".join(no_hint_ranges),
                "signature_count": len(group),
                "kept_signature_count": sum(
                    1
                    for row in group
                    if row.get("evaluation_status") != "excluded_tail_class_mismatch"
                ),
                "excluded_signature_count": sum(
                    1
                    for row in group
                    if row.get("evaluation_status") == "excluded_tail_class_mismatch"
                ),
                "support_count": total_support,
                "kept_support_count": kept_support,
                "excluded_support_count": excluded_support,
            }
        )

    status_order = {
        "review_has_mismatching_tail_class": 0,
        "all_tail_classes_match": 1,
        "no_tail_class_hint": 2,
    }
    return sorted(
        review_rows,
        key=lambda row: (
            status_order.get(str(row["predicate_status"]), 99),
            row["predicate_local"],
        ),
    )


def count_rows(
    annotated_rows: list[dict[str, Any]],
    review_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    signature_status_counts = Counter(
        str(row.get("evaluation_status", "")) for row in annotated_rows
    )
    predicate_status_counts = Counter(
        str(row.get("predicate_status", "")) for row in review_rows
    )

    support_by_status: Counter[str] = Counter()
    for row in annotated_rows:
        support_by_status[str(row.get("evaluation_status", ""))] += as_int(
            row.get("signature_support_count")
        )

    counts = [
        {"metric": "signature_rows_total", "value": len(annotated_rows)},
        {"metric": "predicates_total", "value": len(review_rows)},
    ]
    counts.extend(
        {
            "metric": f"signature_rows_{status}",
            "value": count,
        }
        for status, count in sorted(signature_status_counts.items())
    )
    counts.extend(
        {
            "metric": f"signature_support_{status}",
            "value": count,
        }
        for status, count in sorted(support_by_status.items())
    )
    counts.extend(
        {
            "metric": f"predicates_{status}",
            "value": count,
        }
        for status, count in sorted(predicate_status_counts.items())
    )
    return counts


def ordered_fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    preferred = [
        "predicate",
        "predicate_local",
        "domain",
        "domain_local",
        "range",
        "range_local",
        "expected_tail_classes_from_predicate",
        "has_tail_class_hint",
        "tail_class_matches_hint",
        "evaluation_status",
        "signature_support_count",
        "predicate_support_count",
    ]
    existing = {key for row in rows for key in row}
    extras = sorted(existing - set(preferred))
    return [key for key in preferred if key in existing] + extras


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract KG-derived predicates and flag signatures where the predicate "
            "mentions a class name but the tail/range class does not match it."
        )
    )
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Defaults to '<input stem>_tail_class_filter'.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input_csv}")

    rows = read_csv_rows(args.input_csv)
    annotated = annotated_signature_rows(rows, class_names=CLASSES)
    kept = [
        row
        for row in annotated
        if row.get("evaluation_status") != "excluded_tail_class_mismatch"
    ]
    excluded = [
        row
        for row in annotated
        if row.get("evaluation_status") == "excluded_tail_class_mismatch"
    ]
    review = predicate_review_rows(annotated)
    counts = count_rows(annotated_rows=annotated, review_rows=review)

    prefix = args.output_prefix or f"{args.input_csv.stem}_tail_class_filter"
    signatures_path = args.output_dir / f"{prefix}_signatures.csv"
    kept_path = args.output_dir / f"{prefix}_kept_signatures.csv"
    excluded_path = args.output_dir / f"{prefix}_excluded_signatures.csv"
    review_path = args.output_dir / f"{prefix}_predicates_for_review.csv"
    counts_path = args.output_dir / f"{prefix}_counts.csv"

    signature_fieldnames = ordered_fieldnames(annotated)
    write_csv_rows(signatures_path, annotated, signature_fieldnames)
    write_csv_rows(kept_path, kept, signature_fieldnames)
    write_csv_rows(excluded_path, excluded, signature_fieldnames)
    write_csv_rows(
        review_path,
        review,
        [
            "predicate",
            "predicate_local",
            "predicate_status",
            "expected_tail_classes_from_predicate",
            "matching_tail_classes",
            "mismatching_tail_classes",
            "no_hint_tail_classes",
            "signature_count",
            "kept_signature_count",
            "excluded_signature_count",
            "support_count",
            "kept_support_count",
            "excluded_support_count",
        ],
    )
    write_csv_rows(counts_path, counts, ["metric", "value"])

    for row in counts:
        print(f"{row['metric']}: {row['value']}")
    print(f"All annotated signatures: {signatures_path}")
    print(f"Kept signatures: {kept_path}")
    print(f"Excluded signatures: {excluded_path}")
    print(f"Predicates for review: {review_path}")
    print(f"Counts: {counts_path}")


if __name__ == "__main__":
    main()
