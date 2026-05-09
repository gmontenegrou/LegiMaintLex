from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

from rdflib import Graph, Literal, URIRef
from rdflib.namespace import OWL, RDF, RDFS

provider = "openai"
DEFAULT_BASE_RML = Path("src/data/csv_to_rml_mapping.ttl")
DEFAULT_ONTOLOGY = Path(
    f"exp/new_ontology/{provider}/ontology_extended_{provider}.ttl"
)
DEFAULT_OUTPUT = Path(
    f"exp/kg/{provider}/csv_to_rml_mapping_{provider}_rules.ttl"
)
DEFAULT_PROVIDER_LABELS = {
    "mistral": "Mistral AI",
    "openai": "OpenAI",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate ontology-driven extraction rules and append them to the "
            "base RML mapping."
        )
    )
    parser.add_argument("--base-rml", type=Path, default=DEFAULT_BASE_RML)
    parser.add_argument("--ontology", type=Path, default=DEFAULT_ONTOLOGY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--extraction-provider",
        default=provider,
        help="Provider identifier used in generated PROV activities.",
    )
    parser.add_argument(
        "--extraction-provider-label",
        default=None,
        help="Human-readable provider label used in generated PROV activities.",
    )
    parser.add_argument(
        "--extraction-model",
        default="gpt-4.1",
        help="Model name used in generated PROV activities.",
    )
    parser.add_argument(
        "--extraction-strategy",
        default="signature-driven",
        help="Extraction strategy used in generated PROV activities.",
    )
    parser.add_argument(
        "--relation-run-id",
        default=None,
        help=(
            "Optional run identifier suffix for the relation extraction activity. "
            "Defaults to '<provider>_<model>' after IRI-safe normalization."
        ),
    )
    return parser.parse_args()


def unique_sorted(values: Iterable) -> list:
    return sorted(set(values), key=lambda value: str(value))


def local_name(term: URIRef) -> str:
    value = str(term)
    if "#" in value:
        return value.rsplit("#", 1)[1]
    return value.rstrip("/").rsplit("/", 1)[-1]


def qname(graph: Graph, term: URIRef) -> str:
    return graph.namespace_manager.normalizeUri(term)


def safe_resource_id(graph: Graph, *terms_or_values: object) -> str:
    parts = []
    for value in terms_or_values:
        if isinstance(value, URIRef):
            parts.append(qname(graph, value))
        else:
            parts.append(str(value))
    return re.sub(r"[^A-Za-z0-9]+", "_", "_".join(parts)).strip("_")


def safe_instance_id(*values: object) -> str:
    value = "_".join(str(item) for item in values if str(item).strip())
    value = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()
    return value or "unspecified"


def provider_label(provider_id: str, explicit_label: str | None) -> str:
    if explicit_label:
        return explicit_label
    return DEFAULT_PROVIDER_LABELS.get(provider_id.lower(), provider_id)


def concretize_base_mapping(base_text: str, args: argparse.Namespace) -> str:
    provider_id = safe_instance_id(args.extraction_provider)
    model_id = safe_instance_id(args.extraction_model)
    run_id = (
        safe_instance_id(args.relation_run_id)
        if args.relation_run_id
        else f"{provider_id}_{model_id}"
    )
    label = provider_label(args.extraction_provider, args.extraction_provider_label)
    relation_activity = f"inst:relationExtractionRun_{run_id}"
    provider_agent = f"inst:{provider_id}"
    extended_ontology = f"inst:extendedOntology_{provider_id}"

    replacements = {
        "inst:relationExtractionRun": relation_activity,
        "inst:extractionProvider": provider_agent,
        "inst:extendedOntology": extended_ontology,
        '"Signature-driven relation extraction"': f'"Signature-driven relation extraction by {label}"',
        '"Extraction provider"': f'"{label}"',
        '"unspecified-model"': f'"{args.extraction_model}"',
        '"signature-driven"': f'"{args.extraction_strategy}"',
        '"unspecified-ontology"': f'"{args.ontology.as_posix()}"',
    }
    for old, new in replacements.items():
        base_text = base_text.replace(old, new)
    return base_text


def render_ref(graph: Graph, term: URIRef) -> str:
    return qname(graph, term)


def render_literal(graph: Graph, literal: Literal) -> str:
    return literal.n3(graph.namespace_manager)


def render_literal_values(
    graph: Graph,
    predicate: URIRef,
    values: list[Literal],
    indent: str = "    ",
) -> list[str]:
    if not values:
        return []

    rendered = [render_literal(graph, value) for value in values]
    return [f"{indent}{render_ref(graph, predicate)} {', '.join(rendered)}"]


def render_object_values(
    graph: Graph,
    predicate: str,
    objects: list[URIRef],
    indent: str = "    ",
) -> list[str]:
    if not objects:
        return []

    rendered = [render_ref(graph, value) for value in objects]
    return [f"{indent}{predicate} {', '.join(rendered)}"]


def statement(subject: str, predicate_lines: list[str]) -> str:
    if not predicate_lines:
        return f"{subject} ."
    lines = [subject]
    for index, line in enumerate(predicate_lines):
        suffix = " ." if index == len(predicate_lines) - 1 else " ;"
        lines.append(f"{line}{suffix}")
    return "\n".join(lines)


def ontology_classes(graph: Graph) -> list[URIRef]:
    classes = set(graph.subjects(RDF.type, OWL.Class))
    classes.update(graph.subjects(RDF.type, RDFS.Class))
    return unique_sorted(term for term in classes if isinstance(term, URIRef))


def ontology_object_properties(graph: Graph) -> list[URIRef]:
    return unique_sorted(
        term
        for term in graph.subjects(RDF.type, OWL.ObjectProperty)
        if isinstance(term, URIRef)
    )


def render_class_rule(graph: Graph, cls: URIRef) -> str:
    labels = unique_sorted(graph.objects(cls, RDFS.label))
    comments = unique_sorted(graph.objects(cls, RDFS.comment))
    resource = f"map:ClassRule_{safe_resource_id(graph, cls)}"

    lines = [
        "    a map:ExtractionClassRule",
        f"    map:classIri {render_ref(graph, cls)}",
        f'    map:typeValue "{local_name(cls)}"',
        "    map:sourceOntology map:OntologyExtendedMistral",
    ]
    lines.extend(render_literal_values(graph, RDFS.label, labels))
    lines.extend(render_literal_values(graph, RDFS.comment, comments))
    return statement(resource, lines)


def render_relation_rule(graph: Graph, prop: URIRef) -> str:
    labels = unique_sorted(graph.objects(prop, RDFS.label))
    comments = unique_sorted(graph.objects(prop, RDFS.comment))
    domains = unique_sorted(graph.objects(prop, RDFS.domain))
    ranges = unique_sorted(graph.objects(prop, RDFS.range))
    resource = f"map:RelationRule_{safe_resource_id(graph, prop)}"

    lines = [
        "    a map:ExtractionRelationRule",
        f"    map:predicateIri {render_ref(graph, prop)}",
        f'    map:relationValue "{local_name(prop)}"',
        f'    map:relationIri "{str(prop)}"^^xsd:anyURI',
        "    map:sourceOntology map:OntologyExtendedMistral",
    ]
    lines.extend(render_object_values(graph, "map:domainClass", domains))
    lines.extend(render_object_values(graph, "map:rangeClass", ranges))
    lines.extend(render_literal_values(graph, RDFS.label, labels))
    lines.extend(render_literal_values(graph, RDFS.comment, comments))
    return statement(resource, lines)


def render_domain_range_rule(
    graph: Graph,
    prop: URIRef,
    domain: URIRef,
    range_: URIRef,
) -> str:
    relation_resource = f"map:RelationRule_{safe_resource_id(graph, prop)}"
    rule_id = safe_resource_id(graph, prop, domain, range_)
    resource = f"map:DomainRangeRule_{rule_id}"

    lines = [
        "    a map:ExtractionDomainRangeRule",
        f"    map:relationRule {relation_resource}",
        f"    map:predicateIri {render_ref(graph, prop)}",
        f"    map:headClass {render_ref(graph, domain)}",
        f"    map:tailClass {render_ref(graph, range_)}",
        f'    map:relationValue "{local_name(prop)}"',
        f'    map:headTypeValue "{local_name(domain)}"',
        f'    map:tailTypeValue "{local_name(range_)}"',
        f'    map:normalizedPredicateIri "{str(prop)}"^^xsd:anyURI',
        "    map:sourceOntology map:OntologyExtendedMistral",
    ]
    return statement(resource, lines)


def render_rules(graph: Graph, ontology_path: Path) -> str:
    classes = ontology_classes(graph)
    properties = ontology_object_properties(graph)
    domain_range_rules: list[tuple[URIRef, URIRef, URIRef]] = []

    for prop in properties:
        domains = unique_sorted(graph.objects(prop, RDFS.domain))
        ranges = unique_sorted(graph.objects(prop, RDFS.range))
        for domain in domains:
            for range_ in ranges:
                if isinstance(domain, URIRef) and isinstance(range_, URIRef):
                    domain_range_rules.append((prop, domain, range_))

    header = "\n".join(
        [
            "",
            "# ---------------------------------------------------------------------------",
            "# Ontology-driven extraction rules generated from ontology_extended.",
            "@prefix owl: <http://www.w3.org/2002/07/owl#> .",
            "@prefix semlegm: <https://w3id.org/semleg/maintenance#> .",
        ]
    )
    ontology_summary = "\n".join(
        [
            "map:OntologyExtendedMistral",
            "    a map:ExtractionOntology ;",
            f'    map:sourcePath "{ontology_path.as_posix()}" ;',
            f"    map:classCount {len(classes)} ;",
            f"    map:objectPropertyCount {len(properties)} ;",
            f"    map:domainRangeRuleCount {len(domain_range_rules)} .",
        ]
    )
    rule_classes = "\n".join(
        [
            "map:ExtractionClassRule a rdfs:Class .",
            "map:ExtractionRelationRule a rdfs:Class .",
            "map:ExtractionDomainRangeRule a rdfs:Class .",
        ]
    )

    blocks = [
        header,
        ontology_summary,
        rule_classes,
        "# Class rules: allowed values for legal_triplets[].head_type and tail_type.",
    ]

    blocks.extend(render_class_rule(graph, cls) for cls in classes)
    blocks.append("")
    blocks.append("# Relation rules: allowed predicates and their ontology metadata.")
    blocks.extend(render_relation_rule(graph, prop) for prop in properties)
    blocks.append("")
    blocks.append("# Domain/range rules: allowed extraction signatures.")
    blocks.extend(
        render_domain_range_rule(graph, prop, domain, range_)
        for prop, domain, range_ in domain_range_rules
    )
    blocks.append("")
    return "\n\n".join(blocks)


def main() -> None:
    args = parse_args()
    if not args.base_rml.exists():
        raise FileNotFoundError(f"Base RML mapping not found: {args.base_rml}")
    if not args.ontology.exists():
        raise FileNotFoundError(f"Ontology not found: {args.ontology}")

    graph = Graph()
    graph.parse(str(args.ontology), format="turtle")

    base_text = concretize_base_mapping(
        args.base_rml.read_text(encoding="utf-8").rstrip(),
        args,
    )
    rules_text = render_rules(graph, args.ontology)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(base_text + rules_text, encoding="utf-8")

    class_count = len(ontology_classes(graph))
    property_count = len(ontology_object_properties(graph))
    domain_range_count = sum(
        len(unique_sorted(graph.objects(prop, RDFS.domain)))
        * len(unique_sorted(graph.objects(prop, RDFS.range)))
        for prop in ontology_object_properties(graph)
    )
    print(f"Output RML mapping: {args.output}")
    print(f"Classes: {class_count}")
    print(f"Object properties: {property_count}")
    print(f"Domain/range rules: {domain_range_count}")


if __name__ == "__main__":
    main()
