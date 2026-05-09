from __future__ import annotations

import argparse
import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from rdflib import BNode, Graph, Literal, Namespace, URIRef
from rdflib.collection import Collection
from rdflib.namespace import OWL, RDF, RDFS


DEFAULT_OUTPUT_DIR = Path("src/exp/evaluation_results")
DEFAULT_TRIPLETS_COLUMN = "legal_triplets"
SEMLEG_NS = "https://w3id.org/semleg#"
SEMLEGM_NS = "https://w3id.org/semleg/maintenance#"


@dataclass(frozen=True)
class EvaluationTarget:
    provider: str
    csv_path: Path
    ontology_path: Path
    kg_path: Path | None = None


@dataclass(frozen=True)
class TripletParseResult:
    json_syntax_valid: bool
    legal_triplets_json_valid: bool
    json_error: str
    parseable: bool
    parser: str
    triplets: list[dict[str, Any]]


@dataclass(frozen=True)
class OntologyRules:
    ontology_path: Path
    graph: Graph
    signatures_full: set[tuple[str, str, str]]
    signatures_local: set[tuple[str, str, str]]
    class_uris_by_local: dict[str, set[str]]
    relation_uris_by_local: dict[str, set[str]]


DEFAULT_TARGETS = {
    "mistral": EvaluationTarget(
        provider="mistral",
        csv_path=Path(
            "src/exp/dataset_full/mistral/"
            "full_constrained_extraction_by_mistral_mistral-large-latest_nrows_6370.csv"
        ),
        ontology_path=Path("src/exp/ontology_extension/owl/ontology_extended_mistral.ttl"),
        kg_path=Path("src/exp/dataset_full/kg/mistral/csv_to_rml_mapping_mistral_rules.ttl"),
    ),
    "openai": EvaluationTarget(
        provider="openai",
        csv_path=Path(
            "src/exp/dataset_full/openai/"
            "full_constrained_extraction_by_openai_gpt-4.1_nrows_6370.csv"
        ),
        ontology_path=Path("src/exp/ontology_extension/owl/ontology_extended_openai.ttl"),
        kg_path=Path("src/exp/dataset_full/kg/openai/csv_to_rml_mapping_openai_rules.ttl"),
    ),
}

# Canonical aliases used by legacy outputs.
CANONICAL_CURIE_PREFIX_MAP = {
    "sleimi": "semleg",
}

CANONICAL_URI_NS_MAP = {
    "https://w3id.org/sleimi#": "https://w3id.org/semleg#",
}

SH = Namespace("http://www.w3.org/ns/shacl#")
SHAPE_NS = Namespace("https://w3id.org/semleg/shapes#")


def is_blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    return str(value).strip().lower() in {"", "nan", "none", "null", "na"}


def ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def local_name(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    if "#" in text:
        return text.rsplit("#", 1)[-1]
    if "/" in text:
        return text.rstrip("/").rsplit("/", 1)[-1]
    if ":" in text:
        return text.split(":", 1)[1]
    return text


def extract_triplet_dicts(payload: Any) -> list[dict[str, Any]] | None:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("legal_triplets", "triples", "triplets"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    return None


def parse_triplets_cell(value: Any) -> TripletParseResult:
    if isinstance(value, list):
        triplets = extract_triplet_dicts(value) or []
        return TripletParseResult(
            json_syntax_valid=True,
            legal_triplets_json_valid=True,
            json_error="none",
            parseable=True,
            parser="native_list",
            triplets=triplets,
        )

    if is_blank(value):
        return TripletParseResult(
            json_syntax_valid=False,
            legal_triplets_json_valid=False,
            json_error="blank",
            parseable=False,
            parser="none",
            triplets=[],
        )

    text = str(value).strip()
    json_error = "none"
    try:
        payload = json.loads(text)
        triplets = extract_triplet_dicts(payload)
        return TripletParseResult(
            json_syntax_valid=True,
            legal_triplets_json_valid=triplets is not None,
            json_error="none" if triplets is not None else "json_payload_not_triplets",
            parseable=triplets is not None,
            parser="json" if triplets is not None else "none",
            triplets=triplets or [],
        )
    except json.JSONDecodeError as exc:
        json_error = f"json_decode_error:{exc.msg}"

    try:
        payload = ast.literal_eval(text)
        triplets = extract_triplet_dicts(payload)
        if triplets is not None:
            return TripletParseResult(
                json_syntax_valid=False,
                legal_triplets_json_valid=True,
                json_error=json_error,
                parseable=True,
                parser="python_literal",
                triplets=triplets,
            )
    except (SyntaxError, ValueError):
        pass

    return TripletParseResult(
        json_syntax_valid=False,
        legal_triplets_json_valid=False,
        json_error=json_error,
        parseable=False,
        parser="none",
        triplets=[],
    )


def unique_uri_refs(values: Iterable[Any]) -> set[URIRef]:
    return {value for value in values if isinstance(value, URIRef)}


def build_local_uri_map(uris: Iterable[URIRef]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for uri in uris:
        out.setdefault(local_name(uri), set()).add(str(uri))
    return out


def load_ontology_rules(ontology_path: Path) -> OntologyRules:
    graph = Graph()
    graph.parse(str(ontology_path), format="turtle")

    classes = unique_uri_refs(graph.subjects(RDF.type, OWL.Class))
    classes.update(unique_uri_refs(graph.subjects(RDF.type, RDFS.Class)))

    properties = unique_uri_refs(graph.subjects(RDF.type, OWL.ObjectProperty))
    properties.update(unique_uri_refs(graph.subjects(RDFS.domain, None)))
    properties.update(unique_uri_refs(graph.subjects(RDFS.range, None)))

    signatures_full: set[tuple[str, str, str]] = set()
    for prop in properties:
        domains = unique_uri_refs(graph.objects(prop, RDFS.domain))
        ranges = unique_uri_refs(graph.objects(prop, RDFS.range))
        classes.update(domains)
        classes.update(ranges)
        for domain in domains:
            for range_cls in ranges:
                signatures_full.add((str(domain), str(prop), str(range_cls)))

    signatures_local = {
        (local_name(head), local_name(relation), local_name(tail))
        for head, relation, tail in signatures_full
    }

    return OntologyRules(
        ontology_path=ontology_path,
        graph=graph,
        signatures_full=signatures_full,
        signatures_local=signatures_local,
        class_uris_by_local=build_local_uri_map(classes),
        relation_uris_by_local=build_local_uri_map(properties),
    )


def bind_common_namespaces(target_graph: Graph, source_graph: Graph) -> None:
    for prefix, namespace in source_graph.namespaces():
        if prefix:
            target_graph.bind(prefix, namespace, override=False)
    target_graph.bind("sh", SH, override=True)
    target_graph.bind("semlegsh", SHAPE_NS, override=True)


def shape_uri_for_class(class_uri: str) -> URIRef:
    return URIRef(f"{SHAPE_NS}{local_name(class_uri)}Shape")


def build_shacl_graph_from_rules(rules: OntologyRules) -> Graph:
    shacl_graph = Graph()
    bind_common_namespaces(shacl_graph, rules.graph)

    ranges_by_domain_relation: dict[str, dict[str, set[str]]] = {}
    for domain_uri, relation_uri, range_uri in rules.signatures_full:
        ranges_by_domain_relation.setdefault(domain_uri, {}).setdefault(
            relation_uri,
            set(),
        ).add(range_uri)

    for domain_uri, ranges_by_relation in sorted(ranges_by_domain_relation.items()):
        node_shape = shape_uri_for_class(domain_uri)
        shacl_graph.add((node_shape, RDF.type, SH.NodeShape))
        shacl_graph.add((node_shape, SH.targetClass, URIRef(domain_uri)))
        shacl_graph.add(
            (
                node_shape,
                RDFS.label,
                Literal(f"{local_name(domain_uri)} domain/range shape"),
            )
        )

        for relation_uri, range_uris in sorted(ranges_by_relation.items()):
            property_shape = BNode()
            shacl_graph.add((node_shape, SH.property, property_shape))
            shacl_graph.add((property_shape, SH.path, URIRef(relation_uri)))

            sorted_ranges = sorted(range_uris)
            if len(sorted_ranges) == 1:
                shacl_graph.add(
                    (property_shape, SH["class"], URIRef(sorted_ranges[0]))
                )
                continue

            sh_or_list = BNode()
            shacl_graph.add((property_shape, SH["or"], sh_or_list))
            range_options = []
            for range_uri in sorted_ranges:
                range_option = BNode()
                shacl_graph.add((range_option, SH["class"], URIRef(range_uri)))
                range_options.append(range_option)
            Collection(shacl_graph, sh_or_list, range_options)

    return shacl_graph


def export_shacl_shapes(
    targets: Iterable[EvaluationTarget],
    output_dir: Path,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []

    for target in targets:
        rules = load_ontology_rules(target.ontology_path)
        shacl_graph = build_shacl_graph_from_rules(rules)
        output_path = output_dir / f"{target.provider}_ontology_extended_shapes.ttl"
        shacl_graph.serialize(destination=str(output_path), format="turtle")
        output_paths.append(output_path)
        print(
            f"[SHACL] {target.provider}: saved {len(rules.signatures_local)} "
            f"domain/range signatures as {output_path}"
        )

    return output_paths


def expand_curie_or_uri(graph: Graph, token: str) -> str:
    raw = token.strip()
    if raw.startswith(("http://", "https://")):
        for old_ns, new_ns in CANONICAL_URI_NS_MAP.items():
            if raw.startswith(old_ns):
                return new_ns + raw[len(old_ns) :]
        return raw

    if ":" in raw:
        prefix, rest = raw.split(":", 1)
        mapped_prefix = CANONICAL_CURIE_PREFIX_MAP.get(prefix, prefix)
        raw = f"{mapped_prefix}:{rest}"
        try:
            return str(graph.namespace_manager.expand_curie(raw))
        except Exception:
            return raw

    return raw


def normalize_term(
    graph: Graph,
    token: Any,
    uri_by_local: dict[str, set[str]],
) -> tuple[str, str]:
    if is_blank(token):
        return "", ""

    expanded = expand_curie_or_uri(graph, str(token))
    name = local_name(expanded)
    candidates = uri_by_local.get(name, set())
    if expanded.startswith(("http://", "https://")):
        return expanded, name
    if len(candidates) == 1:
        return next(iter(candidates)), name
    return expanded, name


def component_error_label(
    head_exists: bool,
    relation_exists: bool,
    tail_exists: bool,
) -> str:
    errors = []
    if not head_exists:
        errors.append("head_class_not_in_ontology")
    if not relation_exists:
        errors.append("relation_not_in_ontology")
    if not tail_exists:
        errors.append("tail_class_not_in_ontology")
    return "none" if not errors else "|".join(errors)


def shacl_violation_reason(
    full_signature_match: bool,
    head_exists: bool,
    relation_exists: bool,
    tail_exists: bool,
) -> str:
    if full_signature_match:
        return "none"
    if not (head_exists and relation_exists and tail_exists):
        return "component_not_in_ontology"
    return "domain_range_signature_not_allowed"


def evaluate_file(
    target: EvaluationTarget,
    rules: OntologyRules,
    triplets_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    df = pd.read_csv(target.csv_path)
    if triplets_column not in df.columns:
        raise ValueError(f"'{triplets_column}' column not found in {target.csv_path}")

    row_metrics: list[dict[str, Any]] = []
    triplet_details: list[dict[str, Any]] = []

    total_input_tokens_sum = 0
    total_output_tokens_sum = 0
    if "total_input_tokens" in df.columns:
        total_input_tokens_sum = int(
            pd.to_numeric(df["total_input_tokens"], errors="coerce").fillna(0).sum()
        )
    if "total_output_tokens" in df.columns:
        total_output_tokens_sum = int(
            pd.to_numeric(df["total_output_tokens"], errors="coerce").fillna(0).sum()
        )

    rows_json_valid = 0
    rows_json_syntax_valid = 0
    rows_parseable = 0
    rows_empty_triplets = 0
    total_triplets = 0
    valid_triples = 0
    invalid_triples = 0
    missing_head_class = 0
    missing_relation = 0
    missing_tail_class = 0
    domain_range_violations = 0
    total_class_mentions = 0
    class_mentions_defined_in_ontology = 0
    class_mentions_not_in_ontology = 0
    extracted_class_locals: set[str] = set()
    extracted_class_locals_defined_in_ontology: set[str] = set()
    extracted_class_locals_not_in_ontology: set[str] = set()
    total_object_property_mentions = 0
    object_property_mentions_defined_in_ontology = 0
    object_property_mentions_not_in_ontology = 0
    extracted_object_property_locals: set[str] = set()
    extracted_object_property_locals_defined_in_ontology: set[str] = set()
    extracted_object_property_locals_not_in_ontology: set[str] = set()
    total_relation_type_mentions = 0
    relation_type_mentions_defined_in_ontology = 0
    relation_type_mentions_not_in_ontology = 0
    extracted_relation_types: set[tuple[str, str, str]] = set()
    extracted_relation_types_defined_in_ontology: set[tuple[str, str, str]] = set()
    extracted_relation_types_not_in_ontology: set[tuple[str, str, str]] = set()

    class_locals = set(rules.class_uris_by_local)
    relation_locals = set(rules.relation_uris_by_local)

    for row_idx, row in df.iterrows():
        parsed = parse_triplets_cell(row.get(triplets_column))
        if parsed.json_syntax_valid:
            rows_json_syntax_valid += 1
        if parsed.legal_triplets_json_valid:
            rows_json_valid += 1
        if parsed.parseable:
            rows_parseable += 1

        triplets = parsed.triplets
        n_triplets = len(triplets)
        if n_triplets == 0:
            rows_empty_triplets += 1

        row_valid = 0
        row_invalid = 0
        row_missing_head = 0
        row_missing_relation = 0
        row_missing_tail = 0
        row_domain_range_violations = 0
        source_index = row.get("index", row_idx)

        for triplet_idx, triplet in enumerate(triplets):
            head_uri, head_local = normalize_term(
                rules.graph,
                triplet.get("head_type"),
                rules.class_uris_by_local,
            )
            relation_uri, relation_local = normalize_term(
                rules.graph,
                triplet.get("relation"),
                rules.relation_uris_by_local,
            )
            tail_uri, tail_local = normalize_term(
                rules.graph,
                triplet.get("tail_type"),
                rules.class_uris_by_local,
            )

            signature_full = (head_uri, relation_uri, tail_uri)
            signature_local = (head_local, relation_local, tail_local)
            full_uri_match = signature_full in rules.signatures_full
            full_local_match = signature_local in rules.signatures_local
            signature_match = full_uri_match or full_local_match

            head_exists = head_local in class_locals
            relation_exists = relation_local in relation_locals
            tail_exists = tail_local in class_locals
            violation = not signature_match
            reason = shacl_violation_reason(
                signature_match,
                head_exists,
                relation_exists,
                tail_exists,
            )

            for class_local, class_exists in (
                (head_local, head_exists),
                (tail_local, tail_exists),
            ):
                total_class_mentions += 1
                if class_local:
                    extracted_class_locals.add(class_local)
                if class_exists:
                    class_mentions_defined_in_ontology += 1
                    extracted_class_locals_defined_in_ontology.add(class_local)
                else:
                    class_mentions_not_in_ontology += 1
                    if class_local:
                        extracted_class_locals_not_in_ontology.add(class_local)

            total_object_property_mentions += 1
            if relation_local:
                extracted_object_property_locals.add(relation_local)
            if relation_exists:
                object_property_mentions_defined_in_ontology += 1
                extracted_object_property_locals_defined_in_ontology.add(relation_local)
            else:
                object_property_mentions_not_in_ontology += 1
                if relation_local:
                    extracted_object_property_locals_not_in_ontology.add(relation_local)

            relation_type = (head_local, relation_local, tail_local)
            total_relation_type_mentions += 1
            extracted_relation_types.add(relation_type)
            if signature_match:
                relation_type_mentions_defined_in_ontology += 1
                extracted_relation_types_defined_in_ontology.add(relation_type)
            else:
                relation_type_mentions_not_in_ontology += 1
                extracted_relation_types_not_in_ontology.add(relation_type)

            if signature_match:
                row_valid += 1
                valid_triples += 1
            else:
                row_invalid += 1
                invalid_triples += 1

            if not head_exists:
                row_missing_head += 1
                missing_head_class += 1
            if not relation_exists:
                row_missing_relation += 1
                missing_relation += 1
            if not tail_exists:
                row_missing_tail += 1
                missing_tail_class += 1
            if reason == "domain_range_signature_not_allowed":
                row_domain_range_violations += 1
                domain_range_violations += 1

            triplet_details.append(
                {
                    "provider": target.provider,
                    "file": str(target.csv_path),
                    "ontology_extended": str(rules.ontology_path),
                    "row_index": int(row_idx),
                    "source_index": source_index,
                    "triplet_index": int(triplet_idx),
                    "head": triplet.get("head"),
                    "head_type_raw": triplet.get("head_type"),
                    "head_type_local": head_local,
                    "head_type_uri": head_uri,
                    "relation_raw": triplet.get("relation"),
                    "relation_local": relation_local,
                    "relation_uri": relation_uri,
                    "tail": triplet.get("tail"),
                    "tail_type_raw": triplet.get("tail_type"),
                    "tail_type_local": tail_local,
                    "tail_type_uri": tail_uri,
                    "topic": triplet.get("topic"),
                    "signature_full_match": bool(full_uri_match),
                    "signature_local_match": bool(full_local_match),
                    "shacl_violation": bool(violation),
                    "shacl_violation_reason": reason,
                    "head_class_exists_in_ontology": bool(head_exists),
                    "relation_exists_in_ontology": bool(relation_exists),
                    "tail_class_exists_in_ontology": bool(tail_exists),
                    "component_error_label": component_error_label(
                        head_exists,
                        relation_exists,
                        tail_exists,
                    ),
                }
            )

        total_triplets += n_triplets
        row_metrics.append(
            {
                "provider": target.provider,
                "file": str(target.csv_path),
                "ontology_extended": str(rules.ontology_path),
                "row_index": int(row_idx),
                "source_index": source_index,
                "legal_triplets_json_syntax_valid": bool(parsed.json_syntax_valid),
                "legal_triplets_json_valid": bool(parsed.legal_triplets_json_valid),
                "legal_triplets_json_error": parsed.json_error,
                "legal_triplets_parseable": bool(parsed.parseable),
                "legal_triplets_parser": parsed.parser,
                "nb_triplets": int(n_triplets),
                "nb_valid_triples_shacl": int(row_valid),
                "nb_invalid_triples_shacl_violations": int(row_invalid),
                "invalid_triples_shacl_violation_ratio": ratio(row_invalid, n_triplets),
                "nb_head_class_not_in_ontology": int(row_missing_head),
                "nb_class_mentions": int(n_triplets * 2),
                "nb_class_mentions_defined_in_ontology": int(
                    (n_triplets * 2) - row_missing_head - row_missing_tail
                ),
                "class_mentions_defined_in_ontology_ratio": ratio(
                    (n_triplets * 2) - row_missing_head - row_missing_tail,
                    n_triplets * 2,
                ),
                "nb_relation_not_in_ontology": int(row_missing_relation),
                "relation_not_in_ontology_ratio": ratio(
                    row_missing_relation,
                    n_triplets,
                ),
                "nb_object_property_mentions": int(n_triplets),
                "nb_object_property_mentions_defined_in_ontology": int(
                    n_triplets - row_missing_relation
                ),
                "object_property_mentions_defined_in_ontology_ratio": ratio(
                    n_triplets - row_missing_relation,
                    n_triplets,
                ),
                "nb_tail_class_not_in_ontology": int(row_missing_tail),
                "nb_domain_range_signature_not_allowed": int(
                    row_domain_range_violations
                ),
                "domain_range_signature_not_allowed_ratio": ratio(
                    row_domain_range_violations,
                    n_triplets,
                ),
                "nb_relation_type_mentions": int(n_triplets),
                "nb_relation_type_mentions_defined_in_ontology": int(row_valid),
                "relation_type_mentions_defined_in_ontology_ratio": ratio(
                    row_valid,
                    n_triplets,
                ),
            }
        )

    overall = {
        "provider": target.provider,
        "file": str(target.csv_path),
        "ontology_extended": str(rules.ontology_path),
        "nb_rows": int(len(df)),
        "sum_total_input_tokens": int(total_input_tokens_sum),
        "sum_total_output_tokens": int(total_output_tokens_sum),
        "nb_rows_legal_triplets_json_syntax_valid": int(rows_json_syntax_valid),
        "nb_rows_legal_triplets_json_valid": int(rows_json_valid),
        "nb_rows_legal_triplets_json_invalid": int(len(df) - rows_json_valid),
        "json_validity_legal_triplets_ratio": ratio(rows_json_valid, len(df)),
        "json_syntax_validity_legal_triplets_ratio": ratio(
            rows_json_syntax_valid,
            len(df),
        ),
        "nb_rows_legal_triplets_parseable": int(rows_parseable),
        "nb_rows_legal_triplets_unparseable": int(len(df) - rows_parseable),
        "parseable_legal_triplets_ratio": ratio(rows_parseable, len(df)),
        "json_total_rows": int(len(df)),
        "json_valid_rows": int(rows_json_valid),
        "json_invalid_rows": int(len(df) - rows_json_valid),
        "json_valid_ratio": ratio(rows_json_valid, len(df)),
        "json_strict_syntax_valid_rows": int(rows_json_syntax_valid),
        "json_strict_syntax_valid_ratio": ratio(rows_json_syntax_valid, len(df)),
        "json_parseable_rows": int(rows_parseable),
        "json_parseable_ratio": ratio(rows_parseable, len(df)),
        "nb_rows_empty_triplets": int(rows_empty_triplets),
        "nb_total_triplets": int(total_triplets),
        "nb_valid_triples_shacl": int(valid_triples),
        "nb_invalid_triples_shacl_violations": int(invalid_triples),
        "invalid_triples_shacl_violation_ratio": ratio(
            invalid_triples,
            total_triplets,
        ),
        "shacl_conformance_ratio": ratio(valid_triples, total_triplets),
        "nb_triples_head_class_not_in_ontology": int(missing_head_class),
        "nb_triples_relation_not_in_ontology": int(missing_relation),
        "relation_not_in_ontology_ratio": ratio(missing_relation, total_triplets),
        "nb_triples_tail_class_not_in_ontology": int(missing_tail_class),
        "nb_total_class_mentions": int(total_class_mentions),
        "nb_class_mentions_defined_in_ontology": int(
            class_mentions_defined_in_ontology
        ),
        "nb_class_mentions_not_in_ontology": int(class_mentions_not_in_ontology),
        "class_mentions_defined_in_ontology_ratio": ratio(
            class_mentions_defined_in_ontology,
            total_class_mentions,
        ),
        "nb_unique_extracted_classes": int(len(extracted_class_locals)),
        "nb_unique_extracted_classes_defined_in_ontology": int(
            len(extracted_class_locals_defined_in_ontology)
        ),
        "nb_unique_extracted_classes_not_in_ontology": int(
            len(extracted_class_locals_not_in_ontology)
        ),
        "unique_extracted_classes_defined_in_ontology_ratio": ratio(
            len(extracted_class_locals_defined_in_ontology),
            len(extracted_class_locals),
        ),
        "unique_extracted_classes_not_in_ontology": "|".join(
            sorted(extracted_class_locals_not_in_ontology)
        ),
        "classes_total_mentions": int(total_class_mentions),
        "classes_mentions_defined_in_ontology": int(
            class_mentions_defined_in_ontology
        ),
        "classes_mentions_not_in_ontology": int(class_mentions_not_in_ontology),
        "classes_mentions_defined_in_ontology_ratio": ratio(
            class_mentions_defined_in_ontology,
            total_class_mentions,
        ),
        "classes_unique_extracted": int(len(extracted_class_locals)),
        "classes_unique_defined_in_ontology": int(
            len(extracted_class_locals_defined_in_ontology)
        ),
        "classes_unique_not_in_ontology": int(
            len(extracted_class_locals_not_in_ontology)
        ),
        "classes_unique_defined_in_ontology_ratio": ratio(
            len(extracted_class_locals_defined_in_ontology),
            len(extracted_class_locals),
        ),
        "nb_triples_domain_range_signature_not_allowed": int(
            domain_range_violations
        ),
        "domain_range_signature_not_allowed_ratio": ratio(
            domain_range_violations,
            total_triplets,
        ),
        "object_property_total_mentions": int(total_object_property_mentions),
        "object_property_mentions_defined_in_ontology": int(
            object_property_mentions_defined_in_ontology
        ),
        "object_property_mentions_not_in_ontology": int(
            object_property_mentions_not_in_ontology
        ),
        "object_property_mentions_defined_in_ontology_ratio": ratio(
            object_property_mentions_defined_in_ontology,
            total_object_property_mentions,
        ),
        "object_property_unique_extracted": int(
            len(extracted_object_property_locals)
        ),
        "object_property_unique_defined_in_ontology": int(
            len(extracted_object_property_locals_defined_in_ontology)
        ),
        "object_property_unique_not_in_ontology": int(
            len(extracted_object_property_locals_not_in_ontology)
        ),
        "object_property_unique_defined_in_ontology_ratio": ratio(
            len(extracted_object_property_locals_defined_in_ontology),
            len(extracted_object_property_locals),
        ),
        "object_property_unique_not_in_ontology_values": "|".join(
            sorted(extracted_object_property_locals_not_in_ontology)
        ),
        "relation_type_total_mentions": int(total_relation_type_mentions),
        "relation_type_mentions_defined_in_ontology": int(
            relation_type_mentions_defined_in_ontology
        ),
        "relation_type_mentions_not_in_ontology": int(
            relation_type_mentions_not_in_ontology
        ),
        "relation_type_mentions_defined_in_ontology_ratio": ratio(
            relation_type_mentions_defined_in_ontology,
            total_relation_type_mentions,
        ),
        "relation_type_unique_extracted": int(len(extracted_relation_types)),
        "relation_type_unique_defined_in_ontology": int(
            len(extracted_relation_types_defined_in_ontology)
        ),
        "relation_type_unique_not_in_ontology": int(
            len(extracted_relation_types_not_in_ontology)
        ),
        "relation_type_unique_defined_in_ontology_ratio": ratio(
            len(extracted_relation_types_defined_in_ontology),
            len(extracted_relation_types),
        ),
        "nb_ontology_signatures": int(len(rules.signatures_local)),
        "nb_ontology_classes": int(len(rules.class_uris_by_local)),
        "nb_ontology_relations": int(len(rules.relation_uris_by_local)),
    }

    return pd.DataFrame(row_metrics), pd.DataFrame(triplet_details), overall


def ontology_type_uris(rules: OntologyRules) -> set[str]:
    return {uri for uris in rules.class_uris_by_local.values() for uri in uris}


def ontology_relation_uris(rules: OntologyRules) -> set[str]:
    return {uri for uris in rules.relation_uris_by_local.values() for uri in uris}


def is_semleg_relation_candidate(predicate: URIRef) -> bool:
    predicate_text = str(predicate)
    return predicate_text.startswith(SEMLEG_NS) or predicate_text.startswith(SEMLEGM_NS)


def first_label(graph: Graph, resource: URIRef) -> str:
    label = next(graph.objects(resource, RDFS.label), None)
    return "" if label is None else str(label)


def first_graph_object(graph: Graph, subject: Any, predicate: URIRef) -> Any | None:
    return next(graph.objects(subject, predicate), None)


def iter_direct_relation_assertions(
    kg_graph: Graph,
) -> Iterable[tuple[str, Any | None, URIRef, URIRef, URIRef]]:
    for subject, predicate, obj in kg_graph:
        if not isinstance(subject, URIRef) or not isinstance(predicate, URIRef):
            continue
        if not isinstance(obj, URIRef):
            continue
        yield "direct", None, subject, predicate, obj


def iter_reified_relation_assertions(
    kg_graph: Graph,
) -> Iterable[tuple[str, Any | None, URIRef, URIRef, URIRef]]:
    seen = set()
    for statement in kg_graph.subjects(RDF.predicate, None):
        subject = first_graph_object(kg_graph, statement, RDF.subject)
        predicate = first_graph_object(kg_graph, statement, RDF.predicate)
        obj = first_graph_object(kg_graph, statement, RDF.object)
        if not isinstance(subject, URIRef) or not isinstance(predicate, URIRef):
            continue
        if not isinstance(obj, URIRef):
            continue

        key = (statement, subject, predicate, obj)
        if key in seen:
            continue
        seen.add(key)
        yield "reified", statement, subject, predicate, obj


def ontology_types_for_resources(
    kg_graph: Graph,
    rules: OntologyRules,
) -> dict[URIRef, set[str]]:
    allowed_types = ontology_type_uris(rules)
    resource_types: dict[URIRef, set[str]] = {}
    for resource, _, class_uri in kg_graph.triples((None, RDF.type, None)):
        if not isinstance(resource, URIRef):
            continue
        class_uri_text = str(class_uri)
        if class_uri_text not in allowed_types:
            continue
        resource_types.setdefault(resource, set()).add(class_uri_text)
    return resource_types


def kg_signature_match(
    head_types: set[str],
    relation_uri: str,
    tail_types: set[str],
    rules: OntologyRules,
) -> tuple[bool, bool, str, str]:
    for head_type in sorted(head_types):
        for tail_type in sorted(tail_types):
            signature_full = (head_type, relation_uri, tail_type)
            signature_local = (
                local_name(head_type),
                local_name(relation_uri),
                local_name(tail_type),
            )
            if signature_full in rules.signatures_full:
                return True, True, head_type, tail_type
            if signature_local in rules.signatures_local:
                return True, False, head_type, tail_type
    return False, False, "", ""


def evaluate_kg(
    target: EvaluationTarget,
    rules: OntologyRules,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if target.kg_path is None:
        raise ValueError(f"No KG path configured for provider: {target.provider}")

    kg_graph = Graph()
    kg_graph.parse(str(target.kg_path), format="turtle")

    relation_uris = ontology_relation_uris(rules)
    resource_types = ontology_types_for_resources(kg_graph, rules)
    detail_rows: list[dict[str, Any]] = []

    total_relation_assertions = 0
    valid_relation_assertions = 0
    invalid_relation_assertions = 0
    missing_head_type = 0
    missing_tail_type = 0
    domain_range_violations = 0
    predicate_not_in_ontology = 0

    reified_assertions = list(iter_reified_relation_assertions(kg_graph))
    assertions = (
        reified_assertions
        if reified_assertions
        else list(iter_direct_relation_assertions(kg_graph))
    )
    assertion_model = "reified" if reified_assertions else "direct"

    for assertion_source, statement, subject, predicate, obj in assertions:
        relation_uri = str(predicate)
        head_types = resource_types.get(subject, set())
        tail_types = resource_types.get(obj, set())

        if relation_uri not in relation_uris:
            if (
                is_semleg_relation_candidate(predicate)
                and head_types
                and tail_types
            ):
                predicate_not_in_ontology += 1
                detail_rows.append(
                    {
                        "provider": target.provider,
                        "kg_file": str(target.kg_path),
                        "ontology_extended": str(rules.ontology_path),
                        "assertion_source": assertion_source,
                        "statement_uri": "" if statement is None else str(statement),
                        "subject_uri": str(subject),
                        "subject_label": first_label(kg_graph, subject),
                        "subject_types": "|".join(
                            sorted(local_name(t) for t in head_types)
                        ),
                        "predicate_uri": relation_uri,
                        "predicate_local": local_name(relation_uri),
                        "object_uri": str(obj),
                        "object_label": first_label(kg_graph, obj),
                        "object_types": "|".join(
                            sorted(local_name(t) for t in tail_types)
                        ),
                        "matched_subject_type": "",
                        "matched_object_type": "",
                        "signature_full_match": False,
                        "signature_local_match": False,
                        "shacl_violation": True,
                        "shacl_violation_reason": "predicate_not_in_ontology",
                        "subject_has_ontology_type": True,
                        "object_has_ontology_type": True,
                    }
                )
            continue

        total_relation_assertions += 1
        head_exists = bool(head_types)
        tail_exists = bool(tail_types)

        signature_match, full_uri_match, matched_head_type, matched_tail_type = (
            kg_signature_match(head_types, relation_uri, tail_types, rules)
        )
        if signature_match:
            valid_relation_assertions += 1
            reason = "none"
        else:
            invalid_relation_assertions += 1
            if not head_exists or not tail_exists:
                reason = "resource_type_missing_in_kg"
            else:
                reason = "domain_range_signature_not_allowed"
                domain_range_violations += 1

        if not head_exists:
            missing_head_type += 1
        if not tail_exists:
            missing_tail_type += 1

        detail_rows.append(
            {
                "provider": target.provider,
                "kg_file": str(target.kg_path),
                "ontology_extended": str(rules.ontology_path),
                "assertion_source": assertion_source,
                "statement_uri": "" if statement is None else str(statement),
                "subject_uri": str(subject),
                "subject_label": first_label(kg_graph, subject),
                "subject_types": "|".join(sorted(local_name(t) for t in head_types)),
                "predicate_uri": relation_uri,
                "predicate_local": local_name(relation_uri),
                "object_uri": str(obj),
                "object_label": first_label(kg_graph, obj),
                "object_types": "|".join(sorted(local_name(t) for t in tail_types)),
                "matched_subject_type": local_name(matched_head_type),
                "matched_object_type": local_name(matched_tail_type),
                "signature_full_match": bool(full_uri_match),
                "signature_local_match": bool(signature_match and not full_uri_match),
                "shacl_violation": bool(not signature_match),
                "shacl_violation_reason": reason,
                "subject_has_ontology_type": bool(head_exists),
                "object_has_ontology_type": bool(tail_exists),
            }
        )

    overall = {
        "provider": target.provider,
        "kg_file": str(target.kg_path),
        "ontology_extended": str(rules.ontology_path),
        "nb_kg_triples": int(len(kg_graph)),
        "kg_relation_assertion_model": assertion_model,
        "nb_kg_reified_relation_assertions": int(len(reified_assertions)),
        "nb_kg_relation_assertions_scanned": int(len(assertions)),
        "nb_kg_relation_assertions": int(total_relation_assertions),
        "nb_valid_kg_relation_assertions_shacl": int(valid_relation_assertions),
        "nb_invalid_kg_relation_assertions_shacl": int(invalid_relation_assertions),
        "invalid_kg_relation_assertions_shacl_ratio": ratio(
            invalid_relation_assertions,
            total_relation_assertions,
        ),
        "kg_shacl_conformance_ratio": ratio(
            valid_relation_assertions,
            total_relation_assertions,
        ),
        "nb_kg_relation_assertions_subject_type_missing": int(missing_head_type),
        "nb_kg_relation_assertions_object_type_missing": int(missing_tail_type),
        "nb_kg_relation_assertions_domain_range_signature_not_allowed": int(
            domain_range_violations
        ),
        "nb_kg_relation_assertions_predicate_not_in_ontology": int(
            predicate_not_in_ontology
        ),
        "predicate_not_in_ontology_ratio_over_known_relation_assertions": ratio(
            predicate_not_in_ontology,
            total_relation_assertions,
        ),
        "nb_kg_candidate_relation_assertions": int(
            total_relation_assertions + predicate_not_in_ontology
        ),
        "nb_invalid_kg_candidate_relation_assertions": int(
            invalid_relation_assertions + predicate_not_in_ontology
        ),
        "invalid_kg_candidate_relation_assertions_ratio": ratio(
            invalid_relation_assertions + predicate_not_in_ontology,
            total_relation_assertions + predicate_not_in_ontology,
        ),
        "kg_candidate_relation_conformance_ratio": ratio(
            valid_relation_assertions,
            total_relation_assertions + predicate_not_in_ontology,
        ),
        "nb_kg_typed_resources": int(len(resource_types)),
    }
    return pd.DataFrame(detail_rows), overall


def targets_from_args(args: argparse.Namespace) -> list[EvaluationTarget]:
    if args.input:
        if args.provider == "all":
            raise ValueError("--input requires --provider mistral or --provider openai")
        ontology_path = args.ontology or DEFAULT_TARGETS[args.provider].ontology_path
        kg_path = args.kg or DEFAULT_TARGETS[args.provider].kg_path
        return [
            EvaluationTarget(
                provider=args.provider,
                csv_path=args.input,
                ontology_path=ontology_path,
                kg_path=kg_path,
            )
        ]

    if args.kg and args.provider == "all":
        raise ValueError("--kg requires --provider mistral or --provider openai")

    if args.kg:
        base = DEFAULT_TARGETS[args.provider]
        return [
            EvaluationTarget(
                provider=base.provider,
                csv_path=base.csv_path,
                ontology_path=args.ontology or base.ontology_path,
                kg_path=args.kg,
            )
        ]

    providers = DEFAULT_TARGETS if args.provider == "all" else {args.provider: DEFAULT_TARGETS[args.provider]}
    return list(providers.values())


def run_comparison(
    targets: Iterable[EvaluationTarget],
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    triplets_column: str = DEFAULT_TRIPLETS_COLUMN,
    include_kg: bool = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    overall_rows: list[dict[str, Any]] = []
    kg_overall_rows: list[dict[str, Any]] = []

    for target in targets:
        rules = load_ontology_rules(target.ontology_path)
        print(f"[INFO] Provider: {target.provider}")
        print(f"[INFO] Loaded ontology_extended: {target.ontology_path}")
        print(f"[INFO] Ontology signatures: {len(rules.signatures_local)}")
        print(f"[INFO] CSV: {target.csv_path}")
        if target.kg_path is not None:
            print(f"[INFO] KG: {target.kg_path}")

        row_df, detail_df, overall = evaluate_file(
            target=target,
            rules=rules,
            triplets_column=triplets_column,
        )
        overall_rows.append(overall)

        stem = f"{target.provider}_{target.csv_path.stem}"
        row_out = output_dir / f"{stem}_row_json_shacl_metrics.csv"
        detail_out = output_dir / f"{stem}_triplet_shacl_violations.csv"
        row_df.to_csv(row_out, index=False)
        detail_df.to_csv(detail_out, index=False)

        print(
            f"[FILE] {target.csv_path}\n"
            f"       json_validity={overall['json_validity_legal_triplets_ratio']:.4f} | "
            f"json_valid_rows={overall['nb_rows_legal_triplets_json_valid']} | "
            f"json_invalid_rows={overall['nb_rows_legal_triplets_json_invalid']} | "
            f"total_triplets={overall['nb_total_triplets']} | "
            f"invalid_triples={overall['nb_invalid_triples_shacl_violations']} | "
            f"invalid_triples_ratio={overall['invalid_triples_shacl_violation_ratio']:.4f}"
        )

        if include_kg:
            if target.kg_path is None:
                print(f"[WARN] No KG configured for provider: {target.provider}")
            elif not target.kg_path.exists():
                print(f"[WARN] KG not found, skipping: {target.kg_path}")
            else:
                kg_detail_df, kg_overall = evaluate_kg(target=target, rules=rules)
                kg_overall_rows.append(kg_overall)
                kg_stem = f"{target.provider}_{target.kg_path.stem}"
                kg_detail_out = output_dir / f"{kg_stem}_kg_shacl_violations.csv"
                kg_detail_df.to_csv(kg_detail_out, index=False)
                print(
                    f"[KG] {target.kg_path}\n"
                    f"     kg_relation_assertions={kg_overall['nb_kg_relation_assertions']} | "
                    f"kg_invalid_assertions={kg_overall['nb_invalid_kg_relation_assertions_shacl']} | "
                    f"kg_invalid_ratio={kg_overall['invalid_kg_relation_assertions_shacl_ratio']:.4f} | "
                    f"predicate_not_in_ontology={kg_overall['nb_kg_relation_assertions_predicate_not_in_ontology']} | "
                    f"candidate_invalid_ratio={kg_overall['invalid_kg_candidate_relation_assertions_ratio']:.4f}"
                )

    overall_df = pd.DataFrame(overall_rows)
    overall_out = output_dir / "legal_triplets_json_validity_and_shacl_violations.csv"
    overall_df.to_csv(overall_out, index=False)
    if kg_overall_rows:
        kg_overall_df = pd.DataFrame(kg_overall_rows)
        kg_overall_out = output_dir / "kg_shacl_violations.csv"
        kg_overall_df.to_csv(kg_overall_out, index=False)
    print(f"\n[INFO] Saved overall metrics: {overall_out}")
    if kg_overall_rows:
        print(f"[INFO] Saved KG overall metrics: {kg_overall_out}")
    print(f"[INFO] Saved row and triplet details in: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate legal_triplets JSON validity and invalid triples / SHACL-style "
            "domain-range violations against provider-specific ontology_extended TTL files."
        )
    )
    parser.add_argument(
        "--provider",
        choices=["mistral", "openai", "all"],
        default="all",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Optional extraction CSV. Requires --provider mistral or --provider openai.",
    )
    parser.add_argument(
        "--ontology",
        type=Path,
        default=None,
        help="Optional ontology_extended TTL override for --input.",
    )
    parser.add_argument(
        "--kg",
        type=Path,
        default=None,
        help="Optional generated KG TTL override. Requires a single provider.",
    )
    parser.add_argument(
        "--include-kg",
        action="store_true",
        help="Also evaluate generated KG relation assertions against ontology_extended.",
    )
    parser.add_argument(
        "--export-shacl",
        action="store_true",
        help="Export SHACL NodeShapes generated from ontology domain/range rules.",
    )
    parser.add_argument(
        "--only-export-shacl",
        action="store_true",
        help="Export SHACL shapes and skip CSV/KG evaluation.",
    )
    parser.add_argument(
        "--shacl-output-dir",
        type=Path,
        default=None,
        help="Directory for generated SHACL files. Defaults to --output-dir.",
    )
    parser.add_argument(
        "--triplets-column",
        default=DEFAULT_TRIPLETS_COLUMN,
        help="Column containing the extracted triplets.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    targets = targets_from_args(args)
    if args.export_shacl or args.only_export_shacl:
        export_shacl_shapes(
            targets=targets,
            output_dir=args.shacl_output_dir or args.output_dir,
        )
    if args.only_export_shacl:
        return
    run_comparison(
        targets=targets,
        output_dir=args.output_dir,
        triplets_column=args.triplets_column,
        include_kg=args.include_kg,
    )


if __name__ == "__main__":
    main()
