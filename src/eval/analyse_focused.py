from __future__ import annotations

import argparse
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

import pandas as pd
from rdflib import Graph, Namespace
from rdflib.term import Node, URIRef

DEFAULT_KG = "exp/kg/full/mistral/LegiMaintLex_mistral_fusion_0_9.ttl"
DEFAULT_PATTERN = r"\bportes?\b|\baccessibilité?\b|\bportails?\b"
DEFAULT_OUTPUT_DIR = "src/data/corpus_analysis_new"
DEFAULT_PREFIX = "batiment"

DCTERMS = Namespace("http://purl.org/dc/terms/")
PROV = Namespace("http://www.w3.org/ns/prov#")
RDF = Namespace("http://www.w3.org/1999/02/22-rdf-syntax-ns#")
SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")
RDFS = Namespace("http://www.w3.org/2000/01/rdf-schema#")
SEMLEG = Namespace("https://w3id.org/semleg#")
CNT = Namespace("http://www.w3.org/2011/content#")
ELI = Namespace("http://data.europa.eu/eli/ontology#")
OA = Namespace("http://www.w3.org/ns/oa#")

LABEL_PREDICATES = (SKOS.altLabel, SKOS.prefLabel, RDFS.label)


def local_name(uri: object | None) -> str | None:
    """Return a compact final URI fragment/path component."""
    if uri is None or pd.isna(uri):
        return None
    value = str(uri)
    return re.split(r"[/#]", value.rstrip("/#"))[-1] or value


def entity_class(uri: object | None) -> str | None:
    """Infer the entity class from LegiMaintLex resource URIs."""
    if uri is None or pd.isna(uri):
        return None
    match = re.search(r"/entity/([^/]+)/", str(uri))
    if not match:
        return "Unknown"
    raw = match.group(1).replace("_", " ").replace("-", " ")
    return raw.title().replace(" ", "")


def first_graph_value(
    graph: Graph,
    subject: Node | None,
    predicates: Iterable[URIRef],
) -> str | None:
    """Return the first RDF value found for subject over an ordered predicate list."""
    if subject is None:
        return None
    for predicate in predicates:
        value = graph.value(subject, predicate)
        if value is not None:
            return str(value)
    return None


def load_graph(kg_path: str | Path, rdf_format: str = "turtle") -> Graph:
    graph = Graph()
    graph.parse(str(kg_path), format=rdf_format)
    logging.info("Loaded RDF graph: %s triples", f"{len(graph):,}")
    return graph


def extract_articles_with_subject_pattern(graph: Graph, pattern: str) -> pd.DataFrame:
    """Find articles whose dcterms:subject matches a regex pattern."""
    compiled = re.compile(pattern, flags=re.IGNORECASE)
    rows: list[dict[str, object]] = []

    for article_uri, subject_literal in graph.subject_objects(DCTERMS.subject):
        subject_value = str(subject_literal)
        matches = compiled.findall(subject_value)
        if matches:
            rows.append(
                {
                    "article_uri": str(article_uri),
                    "dcterms_subject": subject_value,
                    "matches": matches,
                    "match_count": len(matches),
                }
            )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("article_uri").reset_index(drop=True)
    logging.info("Articles matching pattern: %s", f"{len(df):,}")
    return df


def extract_article_triples(graph: Graph, articles_df: pd.DataFrame) -> pd.DataFrame:
    """Extract RDF statements linked to focused articles through prov:hadPrimarySource."""
    rows: list[dict[str, object]] = []

    for article_uri in (
        articles_df.get("article_uri", pd.Series(dtype=str)).dropna().drop_duplicates()
    ):
        article = URIRef(article_uri)
        for triple_uri in graph.subjects(PROV.hadPrimarySource, article):
            rdf_subject = graph.value(triple_uri, RDF.subject)
            rdf_predicate = graph.value(triple_uri, RDF.predicate)
            rdf_object = graph.value(triple_uri, RDF.object)
            extraction_run = graph.value(triple_uri, PROV.wasGeneratedBy)

            subject_uri = str(rdf_subject) if rdf_subject else None
            predicate_uri = str(rdf_predicate) if rdf_predicate else None
            object_uri = str(rdf_object) if rdf_object else None

            rows.append(
                {
                    "article_uri": article_uri,
                    "triple_uri": str(triple_uri),
                    "is_rdf_statement": (triple_uri, RDF.type, RDF.Statement) in graph,
                    "subject": subject_uri,
                    "subject_prefLabel": first_graph_value(
                        graph, rdf_subject, LABEL_PREDICATES
                    ),
                    "subject_class": entity_class(subject_uri),
                    "predicate": predicate_uri,
                    "predicate_local": local_name(predicate_uri),
                    "object": object_uri,
                    "object_prefLabel": first_graph_value(
                        graph, rdf_object, LABEL_PREDICATES
                    ),
                    "object_class": entity_class(object_uri),
                    "extraction_run": str(extraction_run) if extraction_run else None,
                }
            )

    df = pd.DataFrame(rows)
    logging.info("Extracted focused triples: %s", f"{len(df):,}")
    return df


def extract_article_content(graph: Graph, triples_df: pd.DataFrame) -> pd.DataFrame:
    """Extract article metadata and full text through semleg:hasContent/cnt:chars."""
    rows: list[dict[str, object]] = []

    for article_uri in (
        triples_df.get("article_uri", pd.Series(dtype=str)).dropna().drop_duplicates()
    ):
        article = URIRef(article_uri)
        content_node = graph.value(article, SEMLEG.hasContent)
        content = graph.value(content_node, CNT.chars) if content_node else None
        content_text = str(content) if content else None

        rows.append(
            {
                "article_uri": article_uri,
                "article_local": local_name(article_uri),
                "document_uri": first_graph_value(graph, article, (ELI.is_part_of,)),
                "id_local": first_graph_value(graph, article, (ELI.id_local,)),
                "article_number": first_graph_value(graph, article, (ELI.number,)),
                "dcterms_subject": first_graph_value(
                    graph, article, (DCTERMS.subject,)
                ),
                "source_date_start": first_graph_value(
                    graph, article, (OA.sourceDateStart,)
                ),
                "content_uri": str(content_node) if content_node else None,
                "content": content_text,
                "content_length": len(content_text) if content_text else 0,
            }
        )

    df = pd.DataFrame(rows)
    found = df["content"].notna().sum() if "content" in df else 0
    logging.info("Articles with content: %s / %s", f"{found:,}", f"{len(df):,}")
    return df


def iter_entity_mentions(triples_df: pd.DataFrame):
    for _, triple in triples_df.iterrows():
        for role in ("subject", "object"):
            uri = triple.get(role)
            if uri is None or pd.isna(uri):
                continue
            label = triple.get(f"{role}_prefLabel")
            class_name = triple.get(f"{role}_class")
            yield {
                "article_uri": triple.get("article_uri"),
                "entity_uri": str(uri),
                "entity_label": label if pd.notna(label) else local_name(uri),
                "entity_class": (
                    class_name if pd.notna(class_name) else entity_class(uri)
                ),
                "role": role,
                "predicate": triple.get("predicate_local")
                or local_name(triple.get("predicate")),
            }


def build_entities_by_article(triples_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate entity designations by article for article-level dashboards."""
    entities: dict[tuple[str, str], dict[str, object]] = {}

    for mention in iter_entity_mentions(triples_df):
        key = (str(mention["article_uri"]), mention["entity_uri"])
        if key not in entities:
            entities[key] = {
                "article_uri": mention["article_uri"],
                "entity_uri": mention["entity_uri"],
                "entity_label": mention["entity_label"],
                "entity_class": mention["entity_class"],
                "as_subject_count": 0,
                "as_object_count": 0,
                "predicates": Counter(),
            }
        entities[key][f"as_{mention['role']}_count"] += 1
        if mention["predicate"]:
            entities[key]["predicates"][str(mention["predicate"])] += 1

    rows = []
    for entity in entities.values():
        predicate_counter: Counter = entity.pop("predicates")
        total = int(entity["as_subject_count"]) + int(entity["as_object_count"])
        rows.append(
            {
                **entity,
                "total_count": total,
                "predicates": "; ".join(sorted(predicate_counter)),
                "predicate_counts": "; ".join(
                    f"{predicate}:{count}"
                    for predicate, count in sorted(predicate_counter.items())
                ),
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(
            ["article_uri", "total_count", "entity_label"],
            ascending=[True, False, True],
        )
    return df


def build_entities_by_class(triples_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate focused entities by inferred class across focused articles."""
    entities: dict[str, dict[str, object]] = {}

    for mention in iter_entity_mentions(triples_df):
        key = mention["entity_uri"]
        if key not in entities:
            entities[key] = {
                "entity_uri": mention["entity_uri"],
                "entity_label": mention["entity_label"],
                "entity_class": mention["entity_class"],
                "as_subject_count": 0,
                "as_object_count": 0,
                "articles": set(),
                "predicates": Counter(),
            }
        entities[key][f"as_{mention['role']}_count"] += 1
        entities[key]["articles"].add(mention["article_uri"])
        if mention["predicate"]:
            entities[key]["predicates"][str(mention["predicate"])] += 1

    rows = []
    for entity in entities.values():
        articles = entity.pop("articles")
        predicate_counter: Counter = entity.pop("predicates")
        total = int(entity["as_subject_count"]) + int(entity["as_object_count"])
        rows.append(
            {
                **entity,
                "total_count": total,
                "article_count": len(articles),
                "article_uris": "; ".join(sorted(str(article) for article in articles)),
                "predicates": "; ".join(sorted(predicate_counter)),
                "predicate_counts": "; ".join(
                    f"{predicate}:{count}"
                    for predicate, count in sorted(predicate_counter.items())
                ),
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(
            ["entity_class", "total_count", "entity_label"],
            ascending=[True, False, True],
        )
    return df


def build_article_edges(triples_df: pd.DataFrame) -> pd.DataFrame:
    """Build a normalized edge list for per-article graph rendering."""
    columns = [
        "article_uri",
        "triple_uri",
        "subject",
        "subject_prefLabel",
        "subject_class",
        "predicate",
        "predicate_local",
        "object",
        "object_prefLabel",
        "object_class",
    ]
    existing = [column for column in columns if column in triples_df.columns]
    return triples_df.loc[:, existing].copy()


def write_outputs(
    articles_df: pd.DataFrame,
    triples_df: pd.DataFrame,
    content_df: pd.DataFrame,
    output_dir: Path,
    prefix: str,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "articles": output_dir / "portes_uri.csv",
        "triples": output_dir / f"{prefix}_triples.csv",
        "content": output_dir / f"{prefix}_articles_content.csv",
        "entities_by_article": output_dir / f"{prefix}_entities_by_article.csv",
        "entities_by_class": output_dir / f"{prefix}_entities_by_class.csv",
        "article_edges": output_dir / f"{prefix}_article_edges.csv",
    }

    outputs = {
        "articles": articles_df,
        "triples": triples_df,
        "content": content_df,
        "entities_by_article": build_entities_by_article(triples_df),
        "entities_by_class": build_entities_by_class(triples_df),
        "article_edges": build_article_edges(triples_df),
    }

    for name, df in outputs.items():
        df.to_csv(paths[name], index=False)
        logging.info("Wrote %s rows to %s", f"{len(df):,}", paths[name])

    return paths


def run_analysis(
    kg: str | Path = DEFAULT_KG,
    pattern: str = DEFAULT_PATTERN,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    prefix: str = DEFAULT_PREFIX,
    rdf_format: str = "turtle",
) -> dict[str, pd.DataFrame]:
    graph = load_graph(kg, rdf_format)
    articles_df = extract_articles_with_subject_pattern(graph, pattern)
    triples_df = extract_article_triples(graph, articles_df)
    content_df = extract_article_content(graph, triples_df)

    write_outputs(
        articles_df=articles_df,
        triples_df=triples_df,
        content_df=content_df,
        output_dir=Path(output_dir),
        prefix=prefix,
    )

    return {
        "articles": articles_df,
        "triples": triples_df,
        "content": content_df,
        "entities_by_article": build_entities_by_article(triples_df),
        "entities_by_class": build_entities_by_class(triples_df),
        "article_edges": build_article_edges(triples_df),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate focused KG dashboard CSVs from an RDF graph."
    )
    parser.add_argument("--kg", default=DEFAULT_KG, help="Path to the RDF/Turtle KG.")
    parser.add_argument(
        "--pattern", default=DEFAULT_PATTERN, help="Regex over dcterms:subject."
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for generated CSV files.",
    )
    parser.add_argument(
        "--prefix", default=DEFAULT_PREFIX, help="Output filename prefix."
    )
    parser.add_argument(
        "--format", default="turtle", help="RDF format passed to rdflib."
    )
    parser.add_argument("--log-level", default="INFO", help="Python logging level.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s:%(message)s",
    )
    run_analysis(
        kg=args.kg,
        pattern=args.pattern,
        output_dir=args.output_dir,
        prefix=args.prefix,
        rdf_format=args.format,
    )


if __name__ == "__main__":
    main()
