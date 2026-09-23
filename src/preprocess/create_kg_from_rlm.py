from __future__ import annotations

import argparse
import csv
import importlib.util
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from rdflib import BNode, Graph, Literal, URIRef
from rdflib.namespace import RDF

RR = URIRef("http://www.w3.org/ns/r2rml#")
RML = URIRef("http://semweb.mmlab.be/ns/rml#")

RR_TRIPLES_MAP = URIRef(f"{RR}TriplesMap")
RR_SUBJECT_MAP = URIRef(f"{RR}subjectMap")
RR_PREDICATE_OBJECT_MAP = URIRef(f"{RR}predicateObjectMap")
RR_OBJECT_MAP = URIRef(f"{RR}objectMap")
RR_PREDICATE_MAP = URIRef(f"{RR}predicateMap")
RR_CLASS = URIRef(f"{RR}class")
RR_TEMPLATE = URIRef(f"{RR}template")
RR_CONSTANT = URIRef(f"{RR}constant")
RR_DATATYPE = URIRef(f"{RR}datatype")
RR_TERM_TYPE = URIRef(f"{RR}termType")
RR_IRI = URIRef(f"{RR}IRI")
RR_BLANK_NODE = URIRef(f"{RR}BlankNode")
RR_LITERAL = URIRef(f"{RR}Literal")
RR_PREDICATE = URIRef(f"{RR}predicate")
RR_OBJECT = URIRef(f"{RR}object")

RML_LOGICAL_SOURCE = URIRef(f"{RML}logicalSource")
RML_SOURCE = URIRef(f"{RML}source")
RML_REFERENCE = URIRef(f"{RML}reference")

DEFAULT_OUTPUT_DIR = Path("exp/kg/rdf")


def set_max_csv_field_size() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def default_mapping(provider: str) -> Path:
    return (
        Path("exp/kg")
        / provider
        / f"csv_to_rml_mapping_{provider}_rules.ttl"
    )


def default_output(provider: str, mapping: Path, output_format: str) -> Path:
    extension = {"turtle": "ttl", "nt": "nt", "xml": "rdf", "json-ld": "jsonld"}.get(
        output_format,
        "ttl",
    )
    return DEFAULT_OUTPUT_DIR / provider / f"{mapping.stem}.{extension}"


def resolve_source_path(source: str, base_dir: Path) -> Path:
    path = Path(source)
    return path if path.is_absolute() else base_dir / path


def read_csv_rows(path: Path, max_rows: int | None = None) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"RML source CSV not found: {path}")

    set_max_csv_field_size()
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader):
            if max_rows is not None and index >= max_rows:
                break
            rows.append(row)
    return rows


def first_object(graph: Graph, subject: Any, predicate: URIRef) -> Any | None:
    return next(graph.objects(subject, predicate), None)


def text_value(value: Any) -> str:
    return "" if value is None else str(value)


def expand_template(template: str, row: dict[str, str]) -> str:
    def replace(match: re.Match[str]) -> str:
        return text_value(row.get(match.group(1), ""))

    return re.sub(r"\{([^{}]+)\}", replace, template)


def bnode_id(value: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_")
    return clean or "blank"


def raw_map_value(graph: Graph, term_map: Any, row: dict[str, str]) -> str:
    constant = first_object(graph, term_map, RR_CONSTANT)
    if constant is not None:
        return text_value(constant)

    reference = first_object(graph, term_map, RML_REFERENCE)
    if reference is not None:
        return text_value(row.get(text_value(reference), ""))

    template = first_object(graph, term_map, RR_TEMPLATE)
    if template is not None:
        return expand_template(text_value(template), row)

    return ""


def term_from_map(
    graph: Graph,
    term_map: Any,
    row: dict[str, str],
    default_term_type: URIRef,
) -> URIRef | BNode | Literal | None:
    value = raw_map_value(graph, term_map, row)
    if value == "":
        return None

    term_type = first_object(graph, term_map, RR_TERM_TYPE) or default_term_type
    datatype = first_object(graph, term_map, RR_DATATYPE)

    if term_type == RR_BLANK_NODE:
        return BNode(bnode_id(value))
    if term_type == RR_IRI:
        return URIRef(value)
    if datatype is not None:
        return Literal(value, datatype=datatype)
    return Literal(value)


def logical_source_rows(
    mapping_graph: Graph,
    triples_map: Any,
    source_cache: dict[Any, list[dict[str, str]]],
    base_dir: Path,
    max_rows_per_source: int | None,
) -> list[dict[str, str]]:
    source_node = first_object(mapping_graph, triples_map, RML_LOGICAL_SOURCE)
    if source_node is None:
        return []
    if source_node not in source_cache:
        source_path_literal = first_object(mapping_graph, source_node, RML_SOURCE)
        if source_path_literal is None:
            raise ValueError(f"Missing rml:source for logical source: {source_node}")
        source_path = resolve_source_path(text_value(source_path_literal), base_dir)
        source_cache[source_node] = read_csv_rows(source_path, max_rows_per_source)
    return source_cache[source_node]


def add_subject_classes(
    mapping_graph: Graph,
    output_graph: Graph,
    subject_map: Any,
    subject: URIRef | BNode,
) -> None:
    for cls in mapping_graph.objects(subject_map, RR_CLASS):
        output_graph.add((subject, RDF.type, cls))


def predicates_from_pom(
    mapping_graph: Graph,
    predicate_object_map: Any,
    row: dict[str, str],
) -> list[URIRef]:
    predicates = [p for p in mapping_graph.objects(predicate_object_map, RR_PREDICATE)]
    for predicate_map in mapping_graph.objects(predicate_object_map, RR_PREDICATE_MAP):
        predicate = term_from_map(mapping_graph, predicate_map, row, RR_IRI)
        if isinstance(predicate, URIRef):
            predicates.append(predicate)
    return predicates


def objects_from_pom(
    mapping_graph: Graph,
    predicate_object_map: Any,
    row: dict[str, str],
) -> list[URIRef | BNode | Literal]:
    objects: list[URIRef | BNode | Literal] = [
        obj for obj in mapping_graph.objects(predicate_object_map, RR_OBJECT)
    ]
    for object_map in mapping_graph.objects(predicate_object_map, RR_OBJECT_MAP):
        obj = term_from_map(mapping_graph, object_map, row, RR_LITERAL)
        if obj is not None:
            objects.append(obj)
    return objects


def run_lite_rml(
    mapping_path: Path,
    output_path: Path,
    output_format: str,
    base_dir: Path,
    max_rows_per_source: int | None,
) -> int:
    mapping_graph = Graph()
    mapping_graph.parse(str(mapping_path), format="turtle")

    output_graph = Graph()
    for prefix, namespace in mapping_graph.namespaces():
        output_graph.bind(prefix, namespace)

    source_cache: dict[Any, list[dict[str, str]]] = {}
    for triples_map in mapping_graph.subjects(RDF.type, RR_TRIPLES_MAP):
        subject_map = first_object(mapping_graph, triples_map, RR_SUBJECT_MAP)
        if subject_map is None:
            continue

        rows = logical_source_rows(
            mapping_graph,
            triples_map,
            source_cache,
            base_dir,
            max_rows_per_source,
        )
        for row in rows:
            subject = term_from_map(mapping_graph, subject_map, row, RR_IRI)
            if not isinstance(subject, (URIRef, BNode)):
                continue

            add_subject_classes(mapping_graph, output_graph, subject_map, subject)
            for pom in mapping_graph.objects(triples_map, RR_PREDICATE_OBJECT_MAP):
                for predicate in predicates_from_pom(mapping_graph, pom, row):
                    for obj in objects_from_pom(mapping_graph, pom, row):
                        output_graph.add((subject, predicate, obj))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_graph.serialize(destination=str(output_path), format=output_format)
    return len(output_graph)


def run_pyrml(mapping_path: Path, output_path: Path, output_format: str) -> None:
    if importlib.util.find_spec("pyrml") is None:
        raise RuntimeError(
            "PyRML is not installed in this environment. "
            "Install it or run with --engine lite."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "pyrml",
        "-m",
        str(mapping_path),
        "-o",
        str(output_path),
        "-f",
        output_format,
    ]
    subprocess.run(command, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an RDF KG from an RML mapping for Mistral or OpenAI."
    )
    parser.add_argument(
        "--provider",
        choices=["mistral", "openai", "combined", "all"],
        default="mistral",
    )
    parser.add_argument("--mapping", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--format",
        default="turtle",
        choices=["turtle", "nt", "xml", "json-ld"],
        help="RDF serialization format.",
    )
    parser.add_argument(
        "--engine",
        choices=["lite", "pyrml"],
        default="lite",
        help="Use the built-in RML subset engine or an installed PyRML module.",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("."),
        help="Base directory for relative rml:source paths.",
    )
    parser.add_argument(
        "--max-rows-per-source",
        type=int,
        default=None,
        help="Debug option for the lite engine.",
    )
    return parser.parse_args()


def create_kg_for_provider(provider: str, args: argparse.Namespace) -> None:
    mapping = args.mapping or default_mapping(provider)
    output = args.output or default_output(provider, mapping, args.format)
    if not mapping.exists():
        raise FileNotFoundError(f"RML mapping not found: {mapping}")

    if args.engine == "pyrml":
        run_pyrml(mapping, output, args.format)
        print(f"RDF output: {output}")
        return

    triple_count = run_lite_rml(
        mapping_path=mapping,
        output_path=output,
        output_format=args.format,
        base_dir=args.base_dir,
        max_rows_per_source=args.max_rows_per_source,
    )
    print(f"Provider: {provider}")
    print(f"RML mapping: {mapping}")
    print(f"RDF triples: {triple_count}")
    print(f"RDF output: {output}")


def main() -> None:
    args = parse_args()
    if args.provider == "all":
        if args.mapping is not None or args.output is not None:
            raise ValueError(
                "--provider all cannot be combined with --mapping or --output"
            )
        for provider in ("mistral", "openai"):
            create_kg_for_provider(provider, args)
        return

    create_kg_for_provider(args.provider, args)


if __name__ == "__main__":
    main()
