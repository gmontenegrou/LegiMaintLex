from __future__ import annotations

import argparse
import csv
import re
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import RDF, RDFS, XSD


SEMLEG = Namespace("https://w3id.org/semleg#")
SEMLEGM = Namespace("https://w3id.org/semleg/maintenance#")
PROV = Namespace("http://www.w3.org/ns/prov#")

DEFAULT_KG = Path("exp/kg/rdf/openai/csv_to_rml_mapping_openai_rules.ttl")
DEFAULT_MANUAL_CSV = Path(
    "exp/evaluation_results/kg_predicate_tail_class/"
    "ontology_extended_openai_from_kg_tail_class_filter_excluded_signatures.csv"
)
DEFAULT_CURATION_TTL = Path(
    "exp/evaluation_results/kg_predicate_tail_class/openai_statement_curation.ttl"
)
DEFAULT_CURATED_KG = Path(
    "exp/evaluation_results/kg_predicate_tail_class/openai_kg_curated.ttl"
)
DEFAULT_AUDIT_CSV = Path(
    "exp/evaluation_results/kg_predicate_tail_class/openai_statement_curation_audit.csv"
)

ENTITY_TYPE_RE = re.compile(r"/entity/([^/]+)/", re.IGNORECASE)
ENTITY_TYPE_TO_CLASS = {
    "action": "Action",
    "actor": "Actor",
    "artifact": "Artifact",
    "condition": "Condition",
    "definition": "Definition",
    "location": "Location",
    "modality": "Modality",
    "reason": "Reason",
    "reference": "Reference",
    "situation": "Situation",
    "source": "Source",
    "time": "Time",
}
KNOWN_CLASS_NAMES = set(ENTITY_TYPE_TO_CLASS.values())

ANNOTATION_COLS = (
    "manual_annotation",
    "mannual_annotation",
    "annotation",
    "decision",
)
CURATED_PREDICATE_COLS = (
    "curated_predicate",
    "curated_predicate_iri",
    "new_predicate",
    "new_predicate_iri",
)
CURATED_PREDICATE_LOCAL_COLS = (
    "curated_predicate_local",
    "new_predicate_local",
)

STATUS_BY_ANNOTATION = {
    "1": SEMLEGM.AcceptedDespiteHeuristicMismatch,
    "0": SEMLEGM.PredicateCorrected,
    "-1": SEMLEGM.SemanticValidityNotVerified,
}


def bind_namespaces(graph: Graph) -> None:
    graph.bind("rdf", RDF)
    graph.bind("rdfs", RDFS)
    graph.bind("xsd", XSD)
    graph.bind("semleg", SEMLEG)
    graph.bind("semlegm", SEMLEGM)
    graph.bind("prov", PROV)


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


def semleg_class_iri(value: Any) -> URIRef | None:
    name = local_name(value)
    if name not in KNOWN_CLASS_NAMES:
        return None
    return URIRef(SEMLEG[name])


def class_iri_from_entity_path(entity: URIRef) -> URIRef | None:
    match = ENTITY_TYPE_RE.search(str(entity))
    if not match:
        return None
    class_name = ENTITY_TYPE_TO_CLASS.get(match.group(1).lower())
    return URIRef(SEMLEG[class_name]) if class_name else None


def normalize_semleg_class(term: URIRef) -> URIRef | None:
    name = local_name(term)
    if name not in KNOWN_CLASS_NAMES:
        return None
    if str(term).startswith(str(SEMLEG)) or str(term).startswith(str(SEMLEGM)):
        return URIRef(SEMLEG[name])
    return None


def entity_classes(graph: Graph, entity: URIRef) -> set[URIRef]:
    classes = {
        normalized
        for rdf_type in graph.objects(entity, RDF.type)
        if isinstance(rdf_type, URIRef)
        for normalized in [normalize_semleg_class(rdf_type)]
        if normalized is not None
    }
    if classes:
        return classes

    inferred = class_iri_from_entity_path(entity)
    return {inferred} if inferred is not None else set()


def detect_delimiter(path: Path) -> str:
    first_line = path.read_text(encoding="utf-8-sig").splitlines()[0]
    return ";" if first_line.count(";") > first_line.count(",") else ","


def read_manual_rows(path: Path) -> list[dict[str, str]]:
    delimiter = detect_delimiter(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def first_present(row: dict[str, str], names: tuple[str, ...]) -> str:
    for name in names:
        value = row.get(name)
        if not is_blank(value):
            return str(value).strip()
    return ""


def expected_tail_classes(row: dict[str, str]) -> list[str]:
    raw_value = row.get("expected_tail_classes_from_predicate", "")
    return [value for value in str(raw_value).split("|") if value]


def class_name_pattern(class_name: str) -> re.Pattern[str]:
    return re.compile(rf"{re.escape(class_name)}s?", flags=re.IGNORECASE)


def choose_replaced_class(predicate_local: str, class_names: list[str]) -> str:
    for class_name in sorted(class_names, key=len, reverse=True):
        if class_name_pattern(class_name).search(predicate_local):
            return class_name
    return class_names[0] if class_names else ""


def replace_class_name(predicate_local: str, old_class: str, new_class: str) -> str:
    pattern = class_name_pattern(old_class)

    def replacement(match: re.Match[str]) -> str:
        matched = match.group(0)
        return f"{new_class}s" if matched.lower().endswith("s") else new_class

    replaced = pattern.sub(replacement, predicate_local, count=1)
    return replaced if replaced != predicate_local else f"{predicate_local}{new_class}"


def curated_predicate_for_row(row: dict[str, str]) -> URIRef:
    explicit_iri = first_present(row, CURATED_PREDICATE_COLS)
    if explicit_iri:
        return URIRef(explicit_iri)

    explicit_local = first_present(row, CURATED_PREDICATE_LOCAL_COLS)
    if explicit_local:
        return URIRef(SEMLEGM[explicit_local])

    predicate_local = row.get("predicate_local") or local_name(row.get("predicate"))
    range_local = row.get("range_local") or local_name(row.get("range"))
    expected_classes = expected_tail_classes(row)
    replaced_class = choose_replaced_class(predicate_local, expected_classes)
    curated_local = replace_class_name(
        predicate_local=predicate_local,
        old_class=replaced_class,
        new_class=range_local,
    )
    return URIRef(SEMLEGM[curated_local])


def manual_annotation(row: dict[str, str]) -> str:
    value = first_present(row, ANNOTATION_COLS)
    if value not in {"1", "0", "-1"}:
        raise ValueError(f"Invalid manual annotation value: {value!r}")
    return value


def build_signature_lookup(
    rows: list[dict[str, str]],
) -> dict[tuple[URIRef, URIRef, URIRef], dict[str, Any]]:
    lookup = {}
    for row in rows:
        annotation = manual_annotation(row)
        predicate = URIRef(row["predicate"])
        domain = semleg_class_iri(row["domain"] or row.get("domain_local"))
        range_ = semleg_class_iri(row["range"] or row.get("range_local"))
        if domain is None or range_ is None:
            continue

        lookup[(predicate, domain, range_)] = {
            "row": row,
            "annotation": annotation,
            "status": STATUS_BY_ANNOTATION[annotation],
            "curated_predicate": (
                curated_predicate_for_row(row) if annotation == "0" else None
            ),
            "expected_tail_classes": [
                class_iri
                for class_name in expected_tail_classes(row)
                for class_iri in [semleg_class_iri(class_name)]
                if class_iri is not None
            ],
        }
    return lookup


def add_curation_vocabulary(graph: Graph, source_csv: Path) -> None:
    graph.add((SEMLEGM.CurationStatus, RDF.type, RDFS.Class))
    graph.add((SEMLEGM.CurationRule, RDF.type, RDFS.Class))
    graph.add((SEMLEGM.TailClassMentionRule, RDF.type, SEMLEGM.CurationRule))
    graph.add(
        (
            SEMLEGM.TailClassMentionRule,
            RDFS.label,
            Literal("Tail class mention rule", lang="en"),
        )
    )

    for status in STATUS_BY_ANNOTATION.values():
        graph.add((status, RDF.type, SEMLEGM.CurationStatus))

    run = SEMLEGM.openaiTailClassManualCurationRun
    graph.add((run, RDF.type, PROV.Activity))
    graph.add((run, PROV.used, Literal(source_csv.as_posix(), datatype=XSD.string)))
    graph.add(
        (
            run,
            PROV.generatedAtTime,
            Literal(datetime.now(timezone.utc).isoformat(), datatype=XSD.dateTime),
        )
    )


def add_statement_annotations(
    graph: Graph,
    statement: URIRef,
    annotation: str,
    status: URIRef,
    llm_predicate: URIRef,
    curated_predicate: URIRef | None,
    domain_class: URIRef,
    range_class: URIRef,
    expected_classes: list[URIRef],
    source_csv: Path,
) -> None:
    graph.add((statement, SEMLEGM.manualAnnotation, Literal(int(annotation))))
    graph.add((statement, SEMLEGM.curationStatus, status))
    graph.add((statement, SEMLEGM.curationRule, SEMLEGM.TailClassMentionRule))
    graph.add((statement, SEMLEGM.llmExtractedPredicate, llm_predicate))
    graph.add((statement, SEMLEGM.observedDomainClass, domain_class))
    graph.add((statement, SEMLEGM.observedTailClass, range_class))
    graph.add(
        (
            statement,
            SEMLEGM.curationSourceFile,
            Literal(source_csv.as_posix(), datatype=XSD.string),
        )
    )

    for expected_class in expected_classes:
        graph.add((statement, SEMLEGM.expectedTailClass, expected_class))

    if annotation == "1":
        graph.add((statement, SEMLEGM.semanticValidityVerified, Literal(True)))
        graph.add((statement, SEMLEGM.effectivePredicate, llm_predicate))
    elif annotation == "0":
        graph.add((statement, SEMLEGM.semanticValidityVerified, Literal(True)))
        graph.add((statement, SEMLEGM.curatedPredicate, curated_predicate))
        graph.add((statement, SEMLEGM.effectivePredicate, curated_predicate))
        graph.add((statement, SEMLEGM.curationAction, SEMLEGM.PredicateReplacement))
    elif annotation == "-1":
        graph.add((statement, SEMLEGM.semanticValidityVerified, Literal(False)))
        graph.add((statement, SEMLEGM.excludedFromFinalOntology, Literal(True)))


def write_audit_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "statement",
        "annotation",
        "curation_status",
        "subject",
        "object",
        "domain_class",
        "range_class",
        "llm_predicate",
        "curated_predicate",
        "direct_old_triple_removed",
        "direct_new_triple_added",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def curate_kg(
    kg: Graph,
    signature_lookup: dict[tuple[URIRef, URIRef, URIRef], dict[str, Any]],
    source_csv: Path,
) -> tuple[Graph, list[dict[str, Any]], Counter[str]]:
    curation = Graph()
    bind_namespaces(curation)
    add_curation_vocabulary(curation, source_csv=source_csv)

    audit_rows = []
    counts: Counter[str] = Counter()

    for statement in kg.subjects(RDF.type, RDF.Statement):
        predicates = [
            value
            for value in kg.objects(statement, RDF.predicate)
            if isinstance(value, URIRef)
        ]
        subjects = [
            value
            for value in kg.objects(statement, RDF.subject)
            if isinstance(value, URIRef)
        ]
        objects = [
            value
            for value in kg.objects(statement, RDF.object)
            if isinstance(value, URIRef)
        ]

        matched = None
        for predicate in predicates:
            for subject in subjects:
                domain_classes = entity_classes(kg, subject)
                for object_ in objects:
                    range_classes = entity_classes(kg, object_)
                    for domain_class in domain_classes:
                        for range_class in range_classes:
                            key = (predicate, domain_class, range_class)
                            if key in signature_lookup:
                                matched = (
                                    predicate,
                                    subject,
                                    object_,
                                    domain_class,
                                    range_class,
                                    signature_lookup[key],
                                )
                                break
                        if matched:
                            break
                    if matched:
                        break
                if matched:
                    break
            if matched:
                break

        if not matched:
            continue

        (
            llm_predicate,
            subject,
            object_,
            domain_class,
            range_class,
            curation_info,
        ) = matched
        annotation = curation_info["annotation"]
        status = curation_info["status"]
        curated_predicate = curation_info["curated_predicate"]

        add_statement_annotations(
            graph=curation,
            statement=statement,
            annotation=annotation,
            status=status,
            llm_predicate=llm_predicate,
            curated_predicate=curated_predicate,
            domain_class=domain_class,
            range_class=range_class,
            expected_classes=curation_info["expected_tail_classes"],
            source_csv=source_csv,
        )

        direct_old_removed = False
        direct_new_added = False

        counts[f"statements_annotation_{annotation}"] += 1
        counts[f"statements_status_{local_name(status)}"] += 1
        audit_rows.append(
            {
                "statement": str(statement),
                "annotation": annotation,
                "curation_status": str(status),
                "subject": str(subject),
                "object": str(object_),
                "domain_class": str(domain_class),
                "range_class": str(range_class),
                "llm_predicate": str(llm_predicate),
                "curated_predicate": str(curated_predicate or ""),
                "direct_old_triple_removed": str(direct_old_removed).lower(),
                "direct_new_triple_added": str(direct_new_added).lower(),
            }
        )

    counts["statements_curated_total"] = len(audit_rows)
    return curation, audit_rows, counts


def copy_kg_and_append_curation(
    source_kg: Path,
    target_kg: Path,
    curation: Graph,
) -> None:
    target_kg.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_kg, target_kg)

    curation_nt = curation.serialize(format="nt")
    with target_kg.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write("\n\n# ---------------------------------------------------------------------------\n")
        handle.write("# Statement-level manual curation layer appended to the raw KG.\n")
        handle.write("# For corrected predicates, use semlegm:effectivePredicate or semlegm:curatedPredicate.\n")
        handle.write("# The original rdf:predicate triples are intentionally preserved in append mode.\n\n")
        handle.write(curation_nt)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a statement-level curation layer and a complete curated KG "
            "from manual tail-class annotations."
        )
    )
    parser.add_argument("--kg", type=Path, default=DEFAULT_KG)
    parser.add_argument("--manual-csv", type=Path, default=DEFAULT_MANUAL_CSV)
    parser.add_argument("--curation-ttl", type=Path, default=DEFAULT_CURATION_TTL)
    parser.add_argument("--curated-kg", type=Path, default=DEFAULT_CURATED_KG)
    parser.add_argument("--audit-csv", type=Path, default=DEFAULT_AUDIT_CSV)
    parser.add_argument(
        "--kg-mode",
        choices=["append"],
        default="append",
        help=(
            "Build the complete KG by copying the raw KG and appending the "
            "curation layer. This is fast and preserves the original triples; "
            "queries should use semlegm:effectivePredicate for corrected statements."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.kg.exists():
        raise FileNotFoundError(f"KG file not found: {args.kg}")
    if not args.manual_csv.exists():
        raise FileNotFoundError(f"Manual annotation CSV not found: {args.manual_csv}")

    manual_rows = read_manual_rows(args.manual_csv)
    signature_lookup = build_signature_lookup(manual_rows)

    kg = Graph()
    bind_namespaces(kg)
    kg.parse(str(args.kg), format="turtle")

    curation, audit_rows, counts = curate_kg(
        kg=kg,
        signature_lookup=signature_lookup,
        source_csv=args.manual_csv,
    )

    args.curation_ttl.parent.mkdir(parents=True, exist_ok=True)
    curation.serialize(destination=str(args.curation_ttl), format="turtle")
    copy_kg_and_append_curation(
        source_kg=args.kg,
        target_kg=args.curated_kg,
        curation=curation,
    )
    write_audit_csv(args.audit_csv, audit_rows)

    print(f"Manual signatures loaded: {len(signature_lookup)}")
    print(f"Input KG triples: {len(kg)}")
    print("Curated KG mode: append")
    print(
        "Curated KG note: original rdf:predicate triples are preserved; "
        "use semlegm:effectivePredicate for curated queries."
    )
    for metric, value in sorted(counts.items()):
        print(f"{metric}: {value}")
    print(f"Curation layer: {args.curation_ttl}")
    print(f"Curated KG: {args.curated_kg}")
    print(f"Audit CSV: {args.audit_csv}")


if __name__ == "__main__":
    main()
