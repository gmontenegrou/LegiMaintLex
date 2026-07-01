import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_PREFIXES = {
    "owl": "http://www.w3.org/2002/07/owl#",
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "xsd": "http://www.w3.org/2001/XMLSchema#",
    "semleg": "https://w3id.org/semleg#",
    "iof": "https://spec.industrialontologies.org/ontology/core/Core/",
    "semlegm": "https://w3id.org/semleg/maintenance#",
}

DEFAULT_INPUT_CSV = Path("exp/new_ontology/relations_simplify_canonical.csv")
DEFAULT_BASE_ONTOLOGY = Path("src/data/semleg-ontology-filtered.ttl")
DEFAULT_OUTPUT_TTL = Path("exp/new_ontology/ontology_extended_from_canonical.ttl")
DEFAULT_OPENAI_JSON = Path(
    "exp/new_ontology/openai/ontology_discovery_accumulated_candidate_model_final_two.json"
)
DEFAULT_MISTRAL_JSON = Path(
    "exp/new_ontology/mistral/ontology_discovery_accumulated_candidate_model_final_two.json"
)


def _as_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value).strip()


def _to_qname(value: str, default_prefix: str = "semlegm") -> str:
    v = _as_str(value)
    if not v:
        return f"{default_prefix}:Unknown"
    if ":" in v:
        return v
    return f"{default_prefix}:{v}"


def _qname_to_iri(qname: str, base_iri: str, prefix_map: dict[str, str]) -> str:
    q = _as_str(qname)
    if not q:
        return f"{base_iri}Unknown"
    if q.startswith("http://") or q.startswith("https://"):
        return q
    if ":" in q:
        prefix, local = q.split(":", 1)
        ns = prefix_map.get(prefix)
        if ns:
            return f"{ns}{local}"
    return f"{base_iri}{q}"


def _build_prefix_map_from_graph(graph) -> dict[str, str]:
    """
    Use the ontology's own namespace bindings first, then fill known defaults.
    """
    prefix_map: dict[str, str] = {}
    for prefix, ns in graph.namespace_manager.namespaces():
        if prefix:
            prefix_map[prefix] = str(ns)
    for prefix, ns in DEFAULT_PREFIXES.items():
        prefix_map.setdefault(prefix, ns)
    return prefix_map


def _ordered_unique(values: list[Any]) -> list[Any]:
    seen = set()
    unique = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique


def _local_name(iri: str) -> str:
    value = _as_str(iri)
    if not value:
        return ""
    if "#" in value:
        return value.rsplit("#", 1)[-1]
    if "/" in value:
        return value.rsplit("/", 1)[-1]
    if ":" in value:
        return value.rsplit(":", 1)[-1]
    return value


def _validate_input_columns(df: pd.DataFrame):
    required = {"domain", "range", "object_prop_candidate"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in canonical CSV: {missing}")


def _coerce_confidence(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_candidate_json(
    input_json_path: Path,
    confidence_threshold: float,
) -> pd.DataFrame:
    with open(input_json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    rows = []
    for entry in payload.get("candidates_object_props", []) or []:
        if not isinstance(entry, dict):
            continue

        confidence = _coerce_confidence(
            entry.get("confidence_score", entry.get("confidence"))
        )
        if confidence is None or confidence <= confidence_threshold:
            continue

        textual_evidence = entry.get("textual_evidence") or []
        if not isinstance(textual_evidence, list):
            textual_evidence = [textual_evidence]
        definition = _as_str(entry.get("definition"))

        rows.append(
            {
                "domain": _as_str(entry.get("domain")),
                "range": _as_str(entry.get("range")),
                "object_prop_candidate": _as_str(entry.get("object_prop_candidate")),
                "confidence_value_prom": confidence,
                "textual_evidence_count_total": len(
                    [_as_str(ev) for ev in textual_evidence if _as_str(ev)]
                ),
                "aligned_CQ": _as_str(entry.get("aligned_CQ")),
                "definition_text": definition,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "domain",
                "range",
                "object_prop_candidate",
                "confidence_value_prom",
                "textual_evidence_count_total",
                "aligned_CQ",
                "definition_text",
            ]
        )

    df = pd.DataFrame(rows)
    return (
        df.groupby(["domain", "range", "object_prop_candidate"], as_index=False)
        .agg(
            {
                "confidence_value_prom": "mean",
                "textual_evidence_count_total": "sum",
                "aligned_CQ": "first",
                "definition_text": "first",
            }
        )
        .reset_index(drop=True)
    )


def build_agreement_candidates(
    openai_df: pd.DataFrame,
    mistral_df: pd.DataFrame,
) -> pd.DataFrame:
    join_cols = ["domain", "range", "object_prop_candidate"]
    merged = openai_df.merge(
        mistral_df,
        on=join_cols,
        how="inner",
        suffixes=("_openai", "_mistral"),
    )

    if merged.empty:
        return pd.DataFrame(
            columns=join_cols
            + [
                "confidence_value_prom",
                "textual_evidence_count_total",
                "aligned_CQ",
                "definition_text",
            ]
        )

    return pd.DataFrame(
        {
            "domain": merged["domain"],
            "range": merged["range"],
            "object_prop_candidate": merged["object_prop_candidate"],
            "confidence_value_prom": (
                merged["confidence_value_prom_openai"]
                + merged["confidence_value_prom_mistral"]
            )
            / 2.0,
            "textual_evidence_count_total": (
                merged["textual_evidence_count_total_openai"]
                + merged["textual_evidence_count_total_mistral"]
            ),
            "aligned_CQ": merged["aligned_CQ_openai"].where(
                merged["aligned_CQ_openai"].astype(str).str.strip() != "",
                merged["aligned_CQ_mistral"],
            ),
            "definition_text": merged["definition_text_openai"].where(
                merged["definition_text_openai"].astype(str).str.strip() != "",
                merged["definition_text_mistral"],
            ),
        }
    ).reset_index(drop=True)


def _select_canonical_relation(row: pd.Series, preference: str) -> str:
    openai_rel = _as_str(row.get("canonical_relation_openai"))
    mistral_rel = _as_str(row.get("canonical_relation_mistral"))
    agreement = row.get("canonical_relation_agreement")

    if preference == "openai":
        return openai_rel or mistral_rel or _as_str(row.get("object_prop_candidate"))

    if preference == "mistral":
        return mistral_rel or openai_rel or _as_str(row.get("object_prop_candidate"))

    # agreement_then_openai (default)
    if pd.notna(agreement) and bool(agreement) and openai_rel:
        return openai_rel
    return openai_rel or mistral_rel or _as_str(row.get("object_prop_candidate"))


def _prepare_rows(df: pd.DataFrame, preference: str) -> pd.DataFrame:
    out = df.copy()
    out["canonical_relation"] = out.apply(
        lambda r: _select_canonical_relation(r, preference=preference),
        axis=1,
    )

    agg_map = {}
    if "confidence_value_prom" in out.columns:
        agg_map["confidence_value_prom"] = "mean"
    if "textual_evidence_count_total" in out.columns:
        agg_map["textual_evidence_count_total"] = "sum"
    if "definition_text" in out.columns:
        agg_map["definition_text"] = "first"

    if agg_map:
        out = (
            out.groupby(["domain", "range", "canonical_relation"], as_index=False)
            .agg(agg_map)
            .reset_index(drop=True)
        )
    else:
        out = (
            out[["domain", "range", "canonical_relation"]]
            .drop_duplicates()
            .reset_index(drop=True)
        )

    return out


def _extend_ontology_from_prepared_df(
    prepared: pd.DataFrame,
    base_ontology_path: Path,
    output_ttl_path: Path,
    base_iri: str,
    include_metadata: bool,
    metadata_mode: str,
):
    from rdflib import BNode, Graph, Literal, URIRef
    from rdflib.collection import Collection
    from rdflib.namespace import OWL, RDF, RDFS, XSD

    def make_class_expression(graph: Graph, classes: list[URIRef]):
        unique_classes = _ordered_unique(classes)
        if not unique_classes:
            return None

        if len(unique_classes) == 1:
            return unique_classes[0]

        union_node = BNode()
        class_list_node = BNode()
        Collection(graph, class_list_node, unique_classes)
        graph.add((union_node, RDF.type, OWL.Class))
        graph.add((union_node, OWL.unionOf, class_list_node))
        return union_node

    def add_class_expression_property(
        graph: Graph,
        prop: URIRef,
        predicate: URIRef,
        classes: list[URIRef],
    ):
        class_expression = make_class_expression(graph, classes)
        if class_expression is not None:
            graph.add((prop, predicate, class_expression))

    def add_scoped_range_restriction(
        graph: Graph,
        domain_class: URIRef,
        prop: URIRef,
        range_classes: list[URIRef],
    ):
        range_expression = make_class_expression(graph, range_classes)
        if range_expression is None:
            return

        restriction = BNode()
        graph.add((restriction, RDF.type, OWL.Restriction))
        graph.add((restriction, OWL.onProperty, prop))
        graph.add((restriction, OWL.allValuesFrom, range_expression))
        graph.add((domain_class, RDFS.subClassOf, restriction))

    def add_literal_set_property(
        graph: Graph,
        prop: URIRef,
        predicate: URIRef,
        values: list[Any],
        datatype: URIRef,
        mode: str,
    ):
        unique_values = _ordered_unique(values)
        try:
            unique_values = sorted(unique_values)
        except TypeError:
            pass

        literals = [Literal(value, datatype=datatype) for value in unique_values]
        if not literals:
            return

        if mode == "property" or len(literals) == 1:
            for literal in literals:
                graph.add((prop, predicate, literal))
            return

        data_range = BNode()
        literal_list = BNode()
        Collection(graph, literal_list, literals)
        graph.add((data_range, RDF.type, RDFS.Datatype))
        graph.add((data_range, OWL.oneOf, literal_list))
        graph.add((prop, predicate, data_range))

    def normalize_repeated_class_expression_properties(graph: Graph):
        for prop in list(graph.subjects(RDF.type, OWL.ObjectProperty)):
            for predicate in (RDFS.domain, RDFS.range):
                class_expressions = _ordered_unique(
                    list(graph.objects(prop, predicate))
                )
                if len(class_expressions) <= 1:
                    continue

                graph.remove((prop, predicate, None))
                add_class_expression_property(
                    graph=graph,
                    prop=prop,
                    predicate=predicate,
                    classes=class_expressions,
                )

    g = Graph()
    g.parse(str(base_ontology_path), format="turtle")
    prefix_map = _build_prefix_map_from_graph(g)

    existing_ns_values = {str(ns) for _, ns in g.namespace_manager.namespaces()}
    if base_iri not in existing_ns_values:
        g.bind("semlegm", base_iri, override=False)

    if include_metadata:
        g.add((URIRef(f"{base_iri}confidenceValue"), RDF.type, OWL.AnnotationProperty))
        g.add(
            (
                URIRef(f"{base_iri}textualEvidenceCount"),
                RDF.type,
                OWL.AnnotationProperty,
            )
        )

    normalize_repeated_class_expression_properties(g)

    existing_object_properties: dict[str, URIRef] = {}
    existing_classes: dict[str, URIRef] = {}

    for subject in g.subjects(RDF.type, OWL.ObjectProperty):
        local = _local_name(str(subject))
        if local and local not in existing_object_properties:
            existing_object_properties[local] = subject

    for subject in g.subjects(RDF.type, OWL.Class):
        local = _local_name(str(subject))
        if local and local not in existing_classes:
            existing_classes[local] = subject

    property_constraints: dict[
        URIRef,
        dict[str, Any],
    ] = {}

    for _, row in prepared.iterrows():
        prop_q = _to_qname(row["canonical_relation"], default_prefix="semlegm")
        dom_q = _as_str(row["domain"])
        rng_q = _as_str(row["range"])

        prop_candidate = URIRef(_qname_to_iri(prop_q, base_iri, prefix_map))
        dom_candidate = URIRef(_qname_to_iri(dom_q, base_iri, prefix_map))
        rng_candidate = URIRef(_qname_to_iri(rng_q, base_iri, prefix_map))

        prop_local = _local_name(str(prop_candidate))
        dom_local = _local_name(str(dom_candidate))
        rng_local = _local_name(str(rng_candidate))

        prop = existing_object_properties.get(prop_local, prop_candidate)
        dom = existing_classes.get(dom_local, dom_candidate)
        rng = existing_classes.get(rng_local, rng_candidate)

        g.add((dom, RDF.type, OWL.Class))
        g.add((rng, RDF.type, OWL.Class))
        g.add((prop, RDF.type, OWL.ObjectProperty))
        constraints = property_constraints.setdefault(
            prop,
            {
                "domains": [],
                "ranges": [],
                "ranges_by_domain": {},
                "confidence_values": [],
                "evidence_counts": [],
            },
        )
        constraints["domains"].append(dom)
        constraints["ranges"].append(rng)
        constraints["ranges_by_domain"].setdefault(dom, []).append(rng)

        if prop_local and prop_local not in existing_object_properties:
            existing_object_properties[prop_local] = prop
        if dom_local and dom_local not in existing_classes:
            existing_classes[dom_local] = dom
        if rng_local and rng_local not in existing_classes:
            existing_classes[rng_local] = rng

        if include_metadata:
            confidence = row.get("confidence_value_prom")
            evidence_count = row.get("textual_evidence_count_total")
            definition_text = _as_str(row.get("definition_text"))

            if confidence is not None and not (
                isinstance(confidence, float) and pd.isna(confidence)
            ):
                constraints["confidence_values"].append(float(confidence))

            if evidence_count is not None and not (
                isinstance(evidence_count, float) and pd.isna(evidence_count)
            ):
                constraints["evidence_counts"].append(int(evidence_count))

            if definition_text:
                g.add(
                    (
                        prop,
                        RDFS.comment,
                        Literal(definition_text, datatype=XSD.string),
                    )
                )

    for prop, constraints in property_constraints.items():
        g.remove((prop, RDFS.domain, None))
        g.remove((prop, RDFS.range, None))
        add_class_expression_property(g, prop, RDFS.domain, constraints["domains"])
        add_class_expression_property(g, prop, RDFS.range, constraints["ranges"])
        unique_domains = _ordered_unique(constraints["domains"])
        unique_ranges = _ordered_unique(constraints["ranges"])
        if len(unique_domains) > 1 and len(unique_ranges) > 1:
            for dom, ranges in constraints["ranges_by_domain"].items():
                add_scoped_range_restriction(g, dom, prop, ranges)

        if include_metadata:
            add_literal_set_property(
                graph=g,
                prop=prop,
                predicate=URIRef(f"{base_iri}confidenceValue"),
                values=constraints["confidence_values"],
                datatype=XSD.decimal,
                mode=metadata_mode,
            )
            add_literal_set_property(
                graph=g,
                prop=prop,
                predicate=URIRef(f"{base_iri}textualEvidenceCount"),
                values=constraints["evidence_counts"],
                datatype=XSD.integer,
                mode=metadata_mode,
            )

    output_ttl_path.parent.mkdir(parents=True, exist_ok=True)
    g.serialize(destination=str(output_ttl_path), format="turtle")


def extend_ontology_from_canonical_csv(
    input_csv_path: Path,
    base_ontology_path: Path,
    output_ttl_path: Path,
    base_iri: str = "https://w3id.org/semleg/maintenance#",
    relation_preference: str = "agreement_then_openai",
    include_metadata: bool = True,
    metadata_mode: str = "one-of",
):
    df = pd.read_csv(input_csv_path)
    _validate_input_columns(df)
    prepared = _prepare_rows(df, preference=relation_preference)
    _extend_ontology_from_prepared_df(
        prepared=prepared,
        base_ontology_path=base_ontology_path,
        output_ttl_path=output_ttl_path,
        base_iri=base_iri,
        include_metadata=include_metadata,
        metadata_mode=metadata_mode,
    )


def extend_ontology_from_dataframe(
    df: pd.DataFrame,
    base_ontology_path: Path,
    output_ttl_path: Path,
    base_iri: str = "https://w3id.org/semleg/maintenance#",
    include_metadata: bool = True,
    metadata_mode: str = "one-of",
):
    _validate_input_columns(df)
    prepared = df.rename(columns={"object_prop_candidate": "canonical_relation"}).copy()
    _extend_ontology_from_prepared_df(
        prepared=prepared,
        base_ontology_path=base_ontology_path,
        output_ttl_path=output_ttl_path,
        base_iri=base_iri,
        include_metadata=include_metadata,
        metadata_mode=metadata_mode,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Extend an existing ontology TTL from canonical relation mappings "
            "or directly from ontology discovery JSON outputs."
        )
    )
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--openai-json", type=Path, default=DEFAULT_OPENAI_JSON)
    parser.add_argument("--mistral-json", type=Path, default=DEFAULT_MISTRAL_JSON)
    parser.add_argument("--base-ontology", type=Path, default=DEFAULT_BASE_ONTOLOGY)
    parser.add_argument("--output-ttl", type=Path, default=DEFAULT_OUTPUT_TTL)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("exp/new_ontology"),
        help="Output directory for provider-specific TTL files when using discovery JSON inputs.",
    )
    parser.add_argument("--base-iri", default="https://w3id.org/semleg/maintenance#")
    parser.add_argument(
        "--relation-preference",
        choices=["agreement_then_openai", "openai", "mistral"],
        default="agreement_then_openai",
        help="How to choose canonical relation when providers disagree.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.7,
        help="Minimum confidence required to keep a discovered relation candidate.",
    )
    parser.add_argument(
        "--from-discovery-json",
        action="store_true",
        help=(
            "Build OWL directly from ontology discovery JSON outputs and generate "
            "three variants: openai-only, mistral-only, and agreement-only."
        ),
    )
    parser.add_argument(
        "--discovery-source",
        choices=["openai", "mistral", "agreement", "all"],
        default="all",
        help=(
            "When using --from-discovery-json, choose which ontology variant to build."
        ),
    )
    parser.add_argument(
        "--no-metadata",
        action="store_true",
        help="Do not add confidence/evidence metadata triples.",
    )
    parser.add_argument(
        "--metadata-mode",
        choices=["property", "one-of"],
        default="one-of",
        help=(
            "How to write repeated metadata values. 'property' writes repeated "
            "literal triples directly on the property. 'one-of' writes repeated "
            "literal values as an OWL data enumeration with owl:oneOf."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.from_discovery_json:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        selected_source = args.discovery_source
        outputs = []

        openai_df = None
        mistral_df = None

        if selected_source in {"openai", "all", "agreement"}:
            openai_df = load_candidate_json(
                input_json_path=args.openai_json,
                confidence_threshold=args.confidence_threshold,
            )

        if selected_source in {"mistral", "all", "agreement"}:
            mistral_df = load_candidate_json(
                input_json_path=args.mistral_json,
                confidence_threshold=args.confidence_threshold,
            )

        if selected_source in {"openai", "all"}:
            outputs.append(
                ("openai", openai_df, args.output_dir / "ontology_extended_openai.ttl")
            )

        if selected_source in {"mistral", "all"}:
            outputs.append(
                (
                    "mistral",
                    mistral_df,
                    args.output_dir / "ontology_extended_mistral.ttl",
                )
            )

        if selected_source in {"agreement", "all"}:
            agreement_df = build_agreement_candidates(openai_df, mistral_df)
            outputs.append(
                (
                    "agreement",
                    agreement_df,
                    args.output_dir / "ontology_extended_agreement.ttl",
                )
            )

        for label, df_variant, output_path in outputs:
            extend_ontology_from_dataframe(
                df=df_variant,
                base_ontology_path=args.base_ontology,
                output_ttl_path=output_path,
                base_iri=args.base_iri,
                include_metadata=not args.no_metadata,
                metadata_mode=args.metadata_mode,
            )
            print(
                f"[{label}] Extended ontology saved at: {output_path} "
                f"({len(df_variant)} relations retained)"
            )
        return

    extend_ontology_from_canonical_csv(
        input_csv_path=args.input_csv,
        base_ontology_path=args.base_ontology,
        output_ttl_path=args.output_ttl,
        base_iri=args.base_iri,
        relation_preference=args.relation_preference,
        include_metadata=not args.no_metadata,
        metadata_mode=args.metadata_mode,
    )

    print(f"Extended ontology saved at: {args.output_ttl}")


if __name__ == "__main__":
    main()
