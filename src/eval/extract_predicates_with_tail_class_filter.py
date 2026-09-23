from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from rdflib import Graph, URIRef

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.kg_creation.update_ontology_from_populated_kg import (
    PREDICATE_NAMESPACES,
    bind_namespaces,
    build_signature_counts,
    collect_direct_triples,
    collect_reified_triples,
)

DEFAULT_INPUT_CSV = Path("exp/new_ontology/openai/ontology_extended_openai_from_kg.csv")
DEFAULT_FULL_KG_DIRS = [
    Path("exp/kg/full/mistral"),
    Path("exp/kg/full/openai"),
]
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
    return [
        class_name for class_name in class_names if class_name.lower() in predicate_text
    ]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(
    path: Path, rows: list[dict[str, Any]], fieldnames: list[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def signature_rows_from_kg(
    kg_path: Path,
    predicate_namespaces: Iterable[str],
    min_support: int,
    include_direct_triples: bool,
    include_reified_statements: bool,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    graph = Graph()
    bind_namespaces(graph)
    graph.parse(str(kg_path), format="turtle")

    observed_triples: set[tuple[URIRef, URIRef, URIRef]] = set()
    if include_reified_statements:
        observed_triples.update(collect_reified_triples(graph, predicate_namespaces))
    if include_direct_triples:
        observed_triples.update(collect_direct_triples(graph, predicate_namespaces))

    signature_counts, property_counts, skipped = build_signature_counts(
        graph=graph,
        observed_triples=observed_triples,
    )

    rows = []
    for (predicate, domain, range_), support in sorted(
        signature_counts.items(),
        key=lambda item: (
            local_name(item[0][0]),
            local_name(item[0][1]),
            local_name(item[0][2]),
        ),
    ):
        if property_counts[predicate] < min_support or support < min_support:
            continue
        rows.append(
            {
                "source_kg": str(kg_path),
                "predicate": str(predicate),
                "predicate_local": local_name(predicate),
                "domain": str(domain),
                "domain_local": local_name(domain),
                "range": str(range_),
                "range_local": local_name(range_),
                "signature_support_count": support,
                "predicate_support_count": property_counts[predicate],
            }
        )

    stats = {
        "kg_triples_parsed": len(graph),
        "observed_relation_triples": len(observed_triples),
        "skipped_relation_triples_without_class": skipped,
        "object_properties_seen": len(property_counts),
        "signature_rows_before_support_filter": len(signature_counts),
        "signature_rows_after_support_filter": len(rows),
    }
    return rows, stats


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
                "expected_tail_classes_from_predicate": "|".join(expected_tail_classes),
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
                "expected_tail_classes_from_predicate": "|".join(expected_tail_classes),
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
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=None,
        help=(
            "Precomputed predicate/domain/range signature CSV. If no KG inputs are "
            "provided, defaults to exp/new_ontology/openai/ontology_extended_openai_from_kg.csv."
        ),
    )
    parser.add_argument(
        "--kg",
        type=Path,
        action="append",
        default=[],
        help="KG Turtle file to process. Can be repeated.",
    )
    parser.add_argument(
        "--kg-dir",
        type=Path,
        action="append",
        default=[],
        help="Directory containing KG Turtle files to process. Can be repeated.",
    )
    parser.add_argument(
        "--kg-glob",
        action="append",
        default=[],
        help="Glob pattern for KG Turtle files, for example 'exp/kg/full/openai/*.ttl'.",
    )
    parser.add_argument(
        "--full-kg",
        action="store_true",
        help="Process all Turtle KGs in exp/kg/full/mistral and exp/kg/full/openai.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--output-prefix",
        default=None,
        help=(
            "For CSV mode, defaults to '<input stem>_tail_class_filter'. "
            "For single-KG mode, defaults to '<kg stem>_tail_class_filter'."
        ),
    )
    parser.add_argument(
        "--predicate-namespace",
        action="append",
        choices=sorted(PREDICATE_NAMESPACES),
        default=None,
        help="Predicate namespace to extract from KG inputs. Can be repeated. Default: semlegm.",
    )
    parser.add_argument(
        "--min-support",
        type=int,
        default=1,
        help="Minimum observed support needed to keep a KG-derived signature.",
    )
    parser.add_argument(
        "--no-direct-triples",
        action="store_true",
        help="Do not read direct subject-predicate-object triples from KG inputs.",
    )
    parser.add_argument(
        "--no-reified-statements",
        action="store_true",
        help="Do not read rdf:Statement/rdf:subject/rdf:predicate/rdf:object triples from KG inputs.",
    )
    return parser.parse_args()


def write_tail_class_outputs(
    rows: list[dict[str, Any]],
    output_dir: Path,
    prefix: str,
) -> dict[str, Path]:
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

    signatures_path = output_dir / f"{prefix}_signatures.csv"
    kept_path = output_dir / f"{prefix}_kept_signatures.csv"
    excluded_path = output_dir / f"{prefix}_excluded_signatures.csv"
    review_path = output_dir / f"{prefix}_predicates_for_review.csv"
    counts_path = output_dir / f"{prefix}_counts.csv"

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

    return {
        "signatures": signatures_path,
        "kept": kept_path,
        "excluded": excluded_path,
        "review": review_path,
        "counts": counts_path,
    }


def resolve_kg_paths(args: argparse.Namespace) -> list[Path]:
    paths = set(args.kg)

    kg_dirs = list(args.kg_dir)
    if args.full_kg:
        kg_dirs.extend(DEFAULT_FULL_KG_DIRS)

    for kg_dir in kg_dirs:
        if not kg_dir.exists():
            raise FileNotFoundError(f"KG directory not found: {kg_dir}")
        paths.update(path for path in kg_dir.glob("*.ttl") if path.is_file())

    for pattern in args.kg_glob:
        paths.update(path for path in Path().glob(pattern) if path.is_file())

    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "KG file(s) not found: " + ", ".join(str(path) for path in sorted(missing))
        )

    return sorted(paths, key=lambda path: str(path))


def main() -> None:
    args = parse_args()
    kg_paths = resolve_kg_paths(args)

    if kg_paths:
        selected_names = args.predicate_namespace or ["semlegm"]
        predicate_namespaces = [PREDICATE_NAMESPACES[name] for name in selected_names]
        all_counts = []

        for kg_path in kg_paths:
            rows, kg_stats = signature_rows_from_kg(
                kg_path=kg_path,
                predicate_namespaces=predicate_namespaces,
                min_support=args.min_support,
                include_direct_triples=not args.no_direct_triples,
                include_reified_statements=not args.no_reified_statements,
            )
            prefix = args.output_prefix or f"{kg_path.stem}_tail_class_filter"
            paths = write_tail_class_outputs(rows, args.output_dir, prefix)

            for metric, value in kg_stats.items():
                all_counts.append(
                    {
                        "source_kg": str(kg_path),
                        "output_prefix": prefix,
                        "metric": metric,
                        "value": value,
                    }
                )

            print(f"KG: {kg_path}")
            for metric, value in kg_stats.items():
                print(f"{metric}: {value}")
            print(f"All annotated signatures: {paths['signatures']}")
            print(f"Kept signatures: {paths['kept']}")
            print(f"Excluded signatures: {paths['excluded']}")
            print(f"Predicates for review: {paths['review']}")
            print(f"Counts: {paths['counts']}")

        if len(kg_paths) > 1:
            batch_counts_path = (
                args.output_dir / "kg_tail_class_filter_batch_counts.csv"
            )
            write_csv_rows(
                batch_counts_path,
                all_counts,
                ["source_kg", "output_prefix", "metric", "value"],
            )
            print(f"Batch KG counts: {batch_counts_path}")
        return

    input_csv = args.input_csv or DEFAULT_INPUT_CSV
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    rows = read_csv_rows(input_csv)
    prefix = args.output_prefix or f"{input_csv.stem}_tail_class_filter"
    paths = write_tail_class_outputs(rows, args.output_dir, prefix)

    counts = read_csv_rows(paths["counts"])
    for row in counts:
        print(f"{row['metric']}: {row['value']}")
    print(f"All annotated signatures: {paths['signatures']}")
    print(f"Kept signatures: {paths['kept']}")
    print(f"Excluded signatures: {paths['excluded']}")
    print(f"Predicates for review: {paths['review']}")
    print(f"Counts: {paths['counts']}")


if __name__ == "__main__":
    main()
