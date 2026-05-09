"""
Generate an inline factorized Turtle Light file from an OWL/Turtle ontology.

Behavior:
- Loads ontology from src/data/semleg-ontology-filtered.ttl
- Finds classes in the SEMLEG and SEMLEGM namespaces
- Serializes each class as a compact Turtle Light block
- Removes namespace prefixes and language/datatype suffixes from literals
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Set

from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import OWL, RDF, RDFS

DEFAULT_INPUT_OWL = Path("src/new_output/dataset/ontology_extended_from_canonical.ttl")
DEFAULT_OUTPUT_TTL = Path("src/data/semleg-triple-ont-auto.ttl")


SEMLEG = Namespace("https://w3id.org/semleg#")
SEMLEGM = Namespace("https://w3id.org/semleg/maintenance#")
EXCLUDED_PREDICATE_LOCAL_NAMES = {
    "confidenceValue",
    "textualEvidenceCount",
}


def is_iri(term) -> bool:
    return isinstance(term, URIRef)


def local_name(iri: URIRef) -> str:
    value = str(iri)
    if "#" in value:
        return value.split("#")[-1]
    return value.rstrip("/").split("/")[-1]


def all_classes(g: Graph) -> Set[URIRef]:
    classes: Set[URIRef] = set()
    for cls in g.subjects(RDF.type, OWL.Class):
        if is_iri(cls):
            classes.add(cls)
    for cls in g.subjects(RDF.type, RDFS.Class):
        if is_iri(cls):
            classes.add(cls)
    return classes


def all_object_properties(g: Graph) -> Set[URIRef]:
    props: Set[URIRef] = set()
    for prop in g.subjects(RDF.type, OWL.ObjectProperty):
        if is_iri(prop):
            props.add(prop)
    return props


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serialize ontology resources as inline factorized Turtle Light."
    )
    parser.add_argument("--input-owl", type=Path, default=DEFAULT_INPUT_OWL)
    parser.add_argument("--output-ttl", type=Path, default=DEFAULT_OUTPUT_TTL)
    parser.add_argument(
        "--mode",
        choices=("classes", "all"),
        default="all",
        help="classes = only classes, all = classes + object properties",
    )
    return parser.parse_args()


def render_iri(iri: URIRef) -> str:
    return f":{local_name(iri)}"


def escape_literal(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def render_term(term) -> str:
    if isinstance(term, URIRef):
        return render_iri(term)
    if isinstance(term, Literal):
        return f'"{escape_literal(str(term))}"'
    return f'"{escape_literal(str(term))}"'


def predicate_sort_key(predicate: URIRef) -> tuple[int, str]:
    special_order = {
        str(RDF.type): 0,
        str(RDFS.label): 1,
        str(RDFS.comment): 2,
        str(RDFS.domain): 3,
        str(RDFS.range): 4,
    }
    return (special_order.get(str(predicate), 99), local_name(predicate).lower())


def subject_sort_key(g: Graph, subject: URIRef) -> tuple[int, str]:
    types = set(g.objects(subject, RDF.type))
    if OWL.Class in types or RDFS.Class in types:
        return (0, local_name(subject).lower())
    if OWL.ObjectProperty in types:
        return (1, local_name(subject).lower())
    return (2, local_name(subject).lower())


def subjects_for_mode(g: Graph, mode: str) -> list[URIRef]:
    if mode == "classes":
        return sorted(all_classes(g), key=lambda x: local_name(x).lower())

    subjects = all_classes(g) | all_object_properties(g)
    return sorted(subjects, key=lambda subject: subject_sort_key(g, subject))


def render_type_values(values: list[str]) -> list[str]:
    preferred = []
    for value in values:
        if value == ":Class":
            preferred.append(value)
        elif value == ":ObjectProperty":
            preferred.append(value)

    if preferred:
        return list(dict.fromkeys(preferred))

    return list(dict.fromkeys(values))


def build_turtle_light(g: Graph, mode: str) -> str:
    subjects = subjects_for_mode(g, mode)
    blocks: list[str] = []

    for subject in subjects:
        predicate_map: dict[URIRef, list[str]] = {}

        for predicate, obj in sorted(
            g.predicate_objects(subject),
            key=lambda item: (
                predicate_sort_key(item[0]),
                render_term(item[1]).lower(),
            ),
        ):
            if local_name(predicate) in EXCLUDED_PREDICATE_LOCAL_NAMES:
                continue
            predicate_map.setdefault(predicate, []).append(render_term(obj))

        statements: list[str] = []
        for predicate in sorted(predicate_map, key=predicate_sort_key):
            if predicate == RDF.type:
                values = render_type_values(predicate_map[predicate])
                if not values:
                    continue
                statements.append("a " + ", ".join(values))
                continue

            values = ", ".join(dict.fromkeys(predicate_map[predicate]))
            statements.append(f"{render_iri(predicate)} {values}")

        if statements:
            blocks.append(f"{render_iri(subject)} " + " ; ".join(statements) + " .")

    return "\n".join(blocks) + ("\n" if blocks else "")


def main() -> None:
    args = parse_args()
    if not args.input_owl.exists():
        raise FileNotFoundError(f"Input OWL not found: {args.input_owl}")

    graph = Graph()
    graph.parse(str(args.input_owl))

    ttl_text = build_turtle_light(graph, mode=args.mode)
    args.output_ttl.parent.mkdir(parents=True, exist_ok=True)
    args.output_ttl.write_text(ttl_text, encoding="utf-8")
    print(f"Turtle Light written to: {args.output_ttl}")
    print(f"Mode: {args.mode}")


if __name__ == "__main__":
    main()
