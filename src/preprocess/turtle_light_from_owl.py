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
from typing import Any, Set

from rdflib import BNode, Graph, Literal, Namespace, URIRef
from rdflib.collection import Collection
from rdflib.namespace import OWL, RDF, RDFS

DEFAULT_INPUT_OWL = Path("exp/new_ontology/ontology_extended_from_canonical.ttl")
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


def unique_preserve_order(values: list[Any]) -> list[Any]:
    seen = set()
    unique = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique


def class_expression_members(g: Graph, expression: Any) -> list[URIRef]:
    if isinstance(expression, URIRef):
        return [expression]
    if not isinstance(expression, BNode):
        return []

    union_list = next(g.objects(expression, OWL.unionOf), None)
    if union_list is None:
        return []

    members: list[URIRef] = []
    for member in Collection(g, union_list):
        members.extend(class_expression_members(g, member))
    return unique_preserve_order(members)


def literal_enumeration_members(g: Graph, expression: Any) -> list[Literal]:
    if not isinstance(expression, BNode):
        return []

    one_of_list = next(g.objects(expression, OWL.oneOf), None)
    if one_of_list is None:
        return []

    return [
        member for member in Collection(g, one_of_list) if isinstance(member, Literal)
    ]


def is_restriction(g: Graph, term: Any) -> bool:
    if not isinstance(term, BNode):
        return False
    return (
        (term, RDF.type, OWL.Restriction) in g
        or (term, OWL.onProperty, None) in g
        or (term, OWL.allValuesFrom, None) in g
    )


def render_class_expression(g: Graph, expression: Any) -> str:
    members = class_expression_members(g, expression)
    if not members:
        return render_term(g, expression)
    if len(members) == 1:
        return render_iri(members[0])
    return "(" + " OR ".join(render_iri(member) for member in members) + ")"


def render_literal_enumeration(g: Graph, expression: Any) -> str:
    members = literal_enumeration_members(g, expression)
    return "(" + " OR ".join(render_term(g, member) for member in members) + ")"


def render_restriction(g: Graph, restriction: BNode) -> str:
    on_properties = [
        prop
        for prop in g.objects(restriction, OWL.onProperty)
        if isinstance(prop, URIRef)
    ]
    all_values_from = list(g.objects(restriction, OWL.allValuesFrom))

    parts = ["a :Restriction"]
    if on_properties:
        rendered_props = ", ".join(render_iri(prop) for prop in on_properties)
        parts.append(f":onProperty {rendered_props}")
    if all_values_from:
        rendered_ranges = ", ".join(
            render_class_expression(g, range_expression)
            for range_expression in all_values_from
        )
        parts.append(f":allValuesFrom {rendered_ranges}")

    return "[" + " ; ".join(parts) + "]"


def render_bnode(g: Graph, term: BNode) -> str:
    class_members = class_expression_members(g, term)
    if class_members:
        return render_class_expression(g, term)

    literal_members = literal_enumeration_members(g, term)
    if literal_members:
        return render_literal_enumeration(g, term)

    if is_restriction(g, term):
        return render_restriction(g, term)

    return "[]"


def render_term(g: Graph, term) -> str:
    if isinstance(term, URIRef):
        return render_iri(term)
    if isinstance(term, Literal):
        return f'"{escape_literal(str(term))}"'
    if isinstance(term, BNode):
        return render_bnode(g, term)
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
                render_term(g, item[1]).lower(),
            ),
        ):
            if local_name(predicate) in EXCLUDED_PREDICATE_LOCAL_NAMES:
                continue
            predicate_map.setdefault(predicate, []).append(render_term(g, obj))

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
