import argparse
import ast
import json
import os
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from mistralai import Mistral
from openai import OpenAI
from rdflib import Graph, URIRef
from rdflib.namespace import OWL, RDF, RDFS

MISTRAL_EMBED_MODEL = "mistral-embed"
OPENAI_EMBED_MODEL = "text-embedding-3-large"
DEFAULT_BATCH_SIZE = 128
DEFAULT_ENTITY_SIMILARITY_THRESHOLD = 0.7
DEFAULT_RELATION_SIMILARITY_THRESHOLD = 0.7
DEFAULT_EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openai").strip().lower()
DEFAULT_TOPIC_FILTER = "maintenanceActivity"
DEFAULT_RELATION_CANONICAL_STRATEGY = "frequency"

_client_cache = {}
_embedding_cache = {}


def get_default_embedding_model(provider: str) -> str:
    provider = provider.strip().lower()
    if provider == "mistral":
        return MISTRAL_EMBED_MODEL
    if provider == "openai":
        return OPENAI_EMBED_MODEL
    raise ValueError("Unsupported embedding provider. Use 'mistral' or 'openai'.")


def get_embedding_client(provider: str):
    provider = provider.strip().lower()
    if provider in _client_cache:
        return _client_cache[provider]

    if provider == "mistral":
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("Missing MISTRAL_API_KEY environment variable.")
        _client_cache[provider] = Mistral(api_key=api_key)
        return _client_cache[provider]

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Missing OPENAI_API_KEY environment variable.")
        _client_cache[provider] = OpenAI(api_key=api_key)
        return _client_cache[provider]

    raise ValueError("Unsupported embedding provider. Use 'mistral' or 'openai'.")


def extract_embedding_vector(item):
    if isinstance(item, dict):
        return item.get("embedding", [])
    return getattr(item, "embedding", [])


def get_embeddings(
    items,
    provider=DEFAULT_EMBEDDING_PROVIDER,
    model=None,
    batch_size=DEFAULT_BATCH_SIZE,
):
    if not items:
        return np.empty((0, 0), dtype=np.float32)

    provider = provider.strip().lower()
    model = model or get_default_embedding_model(provider)
    client = get_embedding_client(provider)

    missing = [
        text for text in items if (provider, model, text) not in _embedding_cache
    ]

    for i in range(0, len(missing), batch_size):
        batch = missing[i : i + batch_size]
        if provider == "mistral":
            response = client.embeddings.create(model=model, inputs=batch)
        else:
            response = client.embeddings.create(model=model, input=batch)

        data = getattr(response, "data", None)
        if data is None and isinstance(response, dict):
            data = response.get("data", [])

        for text, emb_item in zip(batch, data):
            _embedding_cache[(provider, model, text)] = np.array(
                extract_embedding_vector(emb_item), dtype=np.float32
            )

    return np.vstack([_embedding_cache[(provider, model, text)] for text in items])


def cluster_by_similarity(
    items,
    similarity_threshold,
    provider=DEFAULT_EMBEDDING_PROVIDER,
    model=None,
    batch_size=DEFAULT_BATCH_SIZE,
):
    if len(items) <= 1:
        return [items] if items else []

    embeddings = get_embeddings(
        items,
        provider=provider,
        model=model,
        batch_size=batch_size,
    )
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    emb_norm = embeddings / norms
    cos_sim = emb_norm @ emb_norm.T

    n = len(items)
    visited = np.zeros(n, dtype=bool)
    clusters = []

    for i in range(n):
        if visited[i]:
            continue

        cluster = [items[i]]
        visited[i] = True

        for j in range(i + 1, n):
            if visited[j]:
                continue
            if cos_sim[i, j] >= similarity_threshold:
                cluster.append(items[j])
                visited[j] = True

        clusters.append(cluster)

    return clusters


def local_name(value):
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


def normalized_label_key(value):
    if value is None:
        return ""
    text = re.sub(r"\s+", " ", str(value)).strip()
    text = re.sub(r"[’‘ʼ`´]", " ", text)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = text.encode("ascii", "ignore").decode("ascii").lower()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def unique_uri_refs(values):
    return {value for value in values if isinstance(value, URIRef)}


def default_ontology_path(provider):
    return Path(f"src/exp/ontology_extension/owl/ontology_extended_{provider}.ttl")


def load_relation_ontology_index(ontology_path):
    graph = Graph()
    graph.parse(str(ontology_path), format="turtle")

    properties = unique_uri_refs(graph.subjects(RDF.type, OWL.ObjectProperty))
    properties.update(unique_uri_refs(graph.subjects(RDFS.domain, None)))
    properties.update(unique_uri_refs(graph.subjects(RDFS.range, None)))

    relation_locals = {local_name(prop) for prop in properties}
    signatures_local = set()
    for prop in properties:
        domains = unique_uri_refs(graph.objects(prop, RDFS.domain))
        ranges = unique_uri_refs(graph.objects(prop, RDFS.range))
        for domain in domains:
            for range_ in ranges:
                signatures_local.add(
                    (local_name(domain), local_name(prop), local_name(range_))
                )

    return {
        "ontology_path": str(ontology_path),
        "relation_locals": relation_locals,
        "signatures_local": signatures_local,
    }


def choose_canonical_label(cluster, counts):
    return max(cluster, key=lambda label: (counts.get(label, 0), len(label), label))


def exact_normalized_entity_map(values, counts, group_value):
    grouped = {}
    for value in values:
        key = normalized_label_key(value)
        if key:
            grouped.setdefault(key, []).append(value)

    canonical_map = {}
    records = []
    for variants in grouped.values():
        unique_variants = sorted(set(variants))
        if len(unique_variants) <= 1:
            continue
        canonical_label = choose_canonical_label(unique_variants, counts)
        for variant in unique_variants:
            canonical_map[variant] = canonical_label
        records.extend(
            build_cluster_records(
                cluster_kind="entity_exact",
                group_value=group_value,
                cluster=unique_variants,
                canonical_label=canonical_label,
                counts=counts,
            )
        )
    return canonical_map, records


def choose_relation_canonical_label(
    cluster,
    counts,
    head_type,
    tail_type,
    strategy=DEFAULT_RELATION_CANONICAL_STRATEGY,
    ontology_index=None,
):
    if strategy == "frequency":
        return choose_canonical_label(cluster, counts)

    if strategy != "ontology-aware":
        raise ValueError(f"Unsupported relation canonical strategy: {strategy}")

    if ontology_index is None:
        raise ValueError(
            "relation_canonical_strategy='ontology-aware' requires an ontology index."
        )

    head_local = local_name(head_type)
    tail_local = local_name(tail_type)
    signatures_local = ontology_index["signatures_local"]
    relation_locals = ontology_index["relation_locals"]

    exact_signature_candidates = [
        label
        for label in cluster
        if (head_local, local_name(label), tail_local) in signatures_local
    ]
    if exact_signature_candidates:
        return choose_canonical_label(exact_signature_candidates, counts)

    ontology_relation_candidates = [
        label for label in cluster if local_name(label) in relation_locals
    ]
    if ontology_relation_candidates:
        return choose_canonical_label(ontology_relation_candidates, counts)

    return choose_canonical_label(cluster, counts)


def build_cluster_records(cluster_kind, group_value, cluster, canonical_label, counts):
    canonical_count = int(counts.get(canonical_label, 0))
    cluster_sorted = sorted(
        cluster,
        key=lambda label: (counts.get(label, 0), len(label), label),
        reverse=True,
    )

    records = []
    for variant in cluster_sorted:
        records.append(
            {
                "cluster_kind": cluster_kind,
                "group": group_value,
                "canonical_label": canonical_label,
                "variant_label": variant,
                "variant_count": int(counts.get(variant, 0)),
                "canonical_count": canonical_count,
                "cluster_size": len(cluster),
                "is_canonical": variant == canonical_label,
                "variants_in_cluster": json.dumps(cluster_sorted, ensure_ascii=False),
            }
        )
    return records


def build_mapping_dataframe(records):
    if not records:
        return pd.DataFrame(
            columns=[
                "cluster_kind",
                "group",
                "canonical_label",
                "variant_label",
                "variant_count",
                "canonical_count",
                "cluster_size",
                "is_canonical",
                "variants_in_cluster",
            ]
        )

    mapping_df = pd.DataFrame(records)
    return mapping_df.sort_values(
        by=[
            "cluster_kind",
            "group",
            "canonical_label",
            "variant_count",
            "variant_label",
        ],
        ascending=[True, True, True, False, True],
    ).reset_index(drop=True)


def apply_canonical_map(df, columns, canonical_map):
    df_out = df.copy()
    for column in columns:
        df_out[column] = df_out[column].map(
            lambda value: canonical_map.get(value, value)
        )
    return df_out


def merge_entities_global(
    df,
    head_text_column="head",
    head_type_column="head_type",
    tail_text_column="tail",
    tail_type_column="tail_type",
    similarity_threshold=DEFAULT_ENTITY_SIMILARITY_THRESHOLD,
    embedding_provider=DEFAULT_EMBEDDING_PROVIDER,
    embedding_model=None,
    batch_size=DEFAULT_BATCH_SIZE,
    excluded_types=("Reference",),
    skip_values_with_digits=True,
):
    df = df.copy()
    counts = pd.concat(
        [
            df[head_text_column].dropna().astype(str),
            df[tail_text_column].dropna().astype(str),
        ],
        ignore_index=True,
    ).value_counts()

    entity_frames = []
    for text_column, type_column in (
        (head_text_column, head_type_column),
        (tail_text_column, tail_type_column),
    ):
        subset = df[[text_column, type_column]].copy()
        subset.columns = ["entity_text", "entity_type"]
        entity_frames.append(subset)

    entities_df = pd.concat(entity_frames, ignore_index=True).drop_duplicates()
    unique_types = [
        entity_type
        for entity_type in entities_df["entity_type"].dropna().unique()
        if str(entity_type) not in excluded_types
    ]

    canonical_map = {}
    records = []

    for entity_type in unique_types:
        print(f"Merging entities globally for type '{entity_type}'...")
        subset = entities_df[entities_df["entity_type"] == entity_type]
        entities = [
            value
            for value in subset["entity_text"].dropna().astype(str).unique().tolist()
            if value
        ]
        if skip_values_with_digits:
            entities = [value for value in entities if not re.search(r"\d", value)]
        if not entities:
            continue

        exact_map, exact_records = exact_normalized_entity_map(
            entities,
            counts,
            group_value=str(entity_type),
        )
        records.extend(exact_records)
        semantic_entities = sorted({exact_map.get(value, value) for value in entities})
        semantic_counts = {
            value: int(
                sum(
                    counts.get(original, 0)
                    for original in entities
                    if exact_map.get(original, original) == value
                )
            )
            for value in semantic_entities
        }

        clusters = cluster_by_similarity(
            semantic_entities,
            similarity_threshold=similarity_threshold,
            provider=embedding_provider,
            model=embedding_model,
            batch_size=batch_size,
        )

        semantic_map = {}
        for cluster in clusters:
            canonical_label = choose_canonical_label(cluster, semantic_counts)
            for variant in cluster:
                semantic_map[variant] = canonical_label
            records.extend(
                build_cluster_records(
                    cluster_kind="entity",
                    group_value=str(entity_type),
                    cluster=cluster,
                    canonical_label=canonical_label,
                    counts=semantic_counts,
                )
            )
        for value in entities:
            exact_canonical = exact_map.get(value, value)
            canonical_map[value] = semantic_map.get(exact_canonical, exact_canonical)

    merged_df = apply_canonical_map(
        df,
        columns=[head_text_column, tail_text_column],
        canonical_map=canonical_map,
    )
    return merged_df, canonical_map, build_mapping_dataframe(records)


def merge_entities_by_type(
    df,
    text_column="head",
    type_column="head_type",
    similarity_threshold=DEFAULT_ENTITY_SIMILARITY_THRESHOLD,
    embedding_provider=DEFAULT_EMBEDDING_PROVIDER,
    embedding_model=None,
    batch_size=DEFAULT_BATCH_SIZE,
    excluded_types=("Reference",),
    skip_values_with_digits=True,
):
    df = df.copy()
    unique_types = [
        entity_type
        for entity_type in df[type_column].dropna().unique()
        if str(entity_type) not in excluded_types
    ]
    counts = df[text_column].dropna().astype(str).value_counts()
    canonical_map = {}
    records = []

    for entity_type in unique_types:
        print(f"Merging entities for type '{entity_type}'...")
        subset = df[df[type_column] == entity_type]
        entities = [
            value
            for value in subset[text_column].dropna().astype(str).unique().tolist()
            if value
        ]
        if skip_values_with_digits:
            entities = [value for value in entities if not re.search(r"\d", value)]
        if not entities:
            continue

        exact_map, exact_records = exact_normalized_entity_map(
            entities,
            counts,
            group_value=str(entity_type),
        )
        records.extend(exact_records)
        semantic_entities = sorted({exact_map.get(value, value) for value in entities})
        semantic_counts = {
            value: int(
                sum(
                    counts.get(original, 0)
                    for original in entities
                    if exact_map.get(original, original) == value
                )
            )
            for value in semantic_entities
        }

        clusters = cluster_by_similarity(
            semantic_entities,
            similarity_threshold=similarity_threshold,
            provider=embedding_provider,
            model=embedding_model,
            batch_size=batch_size,
        )

        semantic_map = {}
        for cluster in clusters:
            canonical_label = choose_canonical_label(cluster, semantic_counts)
            for variant in cluster:
                semantic_map[variant] = canonical_label
            records.extend(
                build_cluster_records(
                    cluster_kind="entity",
                    group_value=str(entity_type),
                    cluster=cluster,
                    canonical_label=canonical_label,
                    counts=semantic_counts,
                )
            )
        for value in entities:
            exact_canonical = exact_map.get(value, value)
            canonical_map[value] = semantic_map.get(exact_canonical, exact_canonical)

    merged_df = apply_canonical_map(
        df,
        columns=[text_column],
        canonical_map=canonical_map,
    )
    return merged_df, canonical_map, build_mapping_dataframe(records)


def merge_column_by_similarity(
    df,
    text_column,
    similarity_threshold,
    embedding_provider=DEFAULT_EMBEDDING_PROVIDER,
    embedding_model=None,
    batch_size=DEFAULT_BATCH_SIZE,
    cluster_kind="relation",
    group_value="all",
):
    df = df.copy()
    values = [
        value
        for value in df[text_column].dropna().astype(str).unique().tolist()
        if value
    ]
    if not values:
        return df, {}, build_mapping_dataframe([])

    clusters = cluster_by_similarity(
        values,
        similarity_threshold=similarity_threshold,
        provider=embedding_provider,
        model=embedding_model,
        batch_size=batch_size,
    )

    counts = df[text_column].dropna().astype(str).value_counts()
    canonical_map = {}
    records = []

    for cluster in clusters:
        canonical_label = choose_canonical_label(cluster, counts)
        for variant in cluster:
            canonical_map[variant] = canonical_label
        records.extend(
            build_cluster_records(
                cluster_kind=cluster_kind,
                group_value=group_value,
                cluster=cluster,
                canonical_label=canonical_label,
                counts=counts,
            )
        )

    merged_df = apply_canonical_map(
        df,
        columns=[text_column],
        canonical_map=canonical_map,
    )
    return merged_df, canonical_map, build_mapping_dataframe(records)


def merge_relations(
    df,
    relation_column="relation",
    head_type_column="head_type",
    tail_type_column="tail_type",
    similarity_threshold=DEFAULT_RELATION_SIMILARITY_THRESHOLD,
    embedding_provider=DEFAULT_EMBEDDING_PROVIDER,
    embedding_model=None,
    batch_size=DEFAULT_BATCH_SIZE,
    relation_canonical_strategy=DEFAULT_RELATION_CANONICAL_STRATEGY,
    ontology_index=None,
):
    df = df.copy()
    counts = (
        df.groupby([head_type_column, tail_type_column])[relation_column]
        .value_counts()
        .to_dict()
    )
    canonical_map = {}
    records = []

    signature_df = (
        df[[head_type_column, tail_type_column]]
        .dropna()
        .drop_duplicates()
        .reset_index(drop=True)
    )

    for _, signature_row in signature_df.iterrows():
        head_type = signature_row[head_type_column]
        tail_type = signature_row[tail_type_column]
        signature_mask = (df[head_type_column] == head_type) & (
            df[tail_type_column] == tail_type
        )
        subset = df.loc[signature_mask]
        relations = [
            value
            for value in subset[relation_column].dropna().astype(str).unique().tolist()
            if value
        ]
        if not relations:
            continue

        print(f"Merging relations for signature ('{head_type}', '{tail_type}')...")
        clusters = cluster_by_similarity(
            relations,
            similarity_threshold=similarity_threshold,
            provider=embedding_provider,
            model=embedding_model,
            batch_size=batch_size,
        )

        signature_counts = {
            relation: counts.get((head_type, tail_type, relation), 0)
            for relation in relations
        }
        signature_group = f"{head_type}|||{tail_type}"

        for cluster in clusters:
            canonical_label = choose_relation_canonical_label(
                cluster,
                signature_counts,
                head_type=head_type,
                tail_type=tail_type,
                strategy=relation_canonical_strategy,
                ontology_index=ontology_index,
            )
            for variant in cluster:
                canonical_map[(head_type, tail_type, variant)] = canonical_label
            records.extend(
                build_cluster_records(
                    cluster_kind="relation",
                    group_value=signature_group,
                    cluster=cluster,
                    canonical_label=canonical_label,
                    counts=signature_counts,
                )
            )

    merged_df = df.copy()
    merged_df[relation_column] = merged_df.apply(
        lambda row: canonical_map.get(
            (row[head_type_column], row[tail_type_column], row[relation_column]),
            row[relation_column],
        ),
        axis=1,
    )
    return merged_df, canonical_map, build_mapping_dataframe(records)


def normalize_entities(
    df,
    text_column="head",
    type_column="head_type",
    similarity_threshold=DEFAULT_ENTITY_SIMILARITY_THRESHOLD,
    embedding_provider=DEFAULT_EMBEDDING_PROVIDER,
    embedding_model=None,
    batch_size=DEFAULT_BATCH_SIZE,
):
    merged_df, canonical_map, mapping_df = merge_entities_by_type(
        df=df,
        text_column=text_column,
        type_column=type_column,
        similarity_threshold=similarity_threshold,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        batch_size=batch_size,
    )
    return merged_df, canonical_map, mapping_df


def normalize_all_entities(
    df,
    head_text_column="head",
    head_type_column="head_type",
    tail_text_column="tail",
    tail_type_column="tail_type",
    similarity_threshold=DEFAULT_ENTITY_SIMILARITY_THRESHOLD,
    embedding_provider=DEFAULT_EMBEDDING_PROVIDER,
    embedding_model=None,
    batch_size=DEFAULT_BATCH_SIZE,
):
    return merge_entities_global(
        df=df,
        head_text_column=head_text_column,
        head_type_column=head_type_column,
        tail_text_column=tail_text_column,
        tail_type_column=tail_type_column,
        similarity_threshold=similarity_threshold,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        batch_size=batch_size,
    )


def normalize_relations(
    df,
    relation_column="relation",
    head_type_column="head_type",
    tail_type_column="tail_type",
    similarity_threshold=DEFAULT_RELATION_SIMILARITY_THRESHOLD,
    embedding_provider=DEFAULT_EMBEDDING_PROVIDER,
    embedding_model=None,
    batch_size=DEFAULT_BATCH_SIZE,
    relation_canonical_strategy=DEFAULT_RELATION_CANONICAL_STRATEGY,
    ontology_index=None,
):
    return merge_relations(
        df=df,
        relation_column=relation_column,
        head_type_column=head_type_column,
        tail_type_column=tail_type_column,
        similarity_threshold=similarity_threshold,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        batch_size=batch_size,
        relation_canonical_strategy=relation_canonical_strategy,
        ontology_index=ontology_index,
    )


def parse_legal_triplets_cell(value):
    if isinstance(value, list):
        return value
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            parsed = json.loads(stripped)
            return parsed if isinstance(parsed, list) else []
        except Exception:
            pass
        try:
            parsed = ast.literal_eval(stripped)
            return parsed if isinstance(parsed, list) else []
        except Exception:
            return []
    return []


def explode_triplets_dataframe(
    df,
    triplets_column="triples",
    topic_filter=DEFAULT_TOPIC_FILTER,
    source_text_column="content",
):
    keep_all = topic_filter is None or topic_filter == "all"
    allowed_topics = None
    if not keep_all and isinstance(topic_filter, (list, set, tuple)):
        allowed_topics = set(topic_filter)

    rows = []
    for idx, row in df.iterrows():
        source_text = row.get(source_text_column, "")
        triplets = parse_legal_triplets_cell(row.get(triplets_column))
        for triplet in triplets:
            if not isinstance(triplet, dict):
                continue

            topic = triplet.get("topic")
            if not keep_all:
                if allowed_topics is not None and topic not in allowed_topics:
                    continue
                if allowed_topics is None and topic != topic_filter:
                    continue

            flat_triplet = triplet.copy()
            flat_triplet["topic"] = topic
            flat_triplet["text"] = source_text
            flat_triplet["_source_row_index"] = idx
            rows.append(flat_triplet)
    if not rows:
        return pd.DataFrame(
            columns=[
                "head",
                "head_type",
                "relation",
                "tail",
                "tail_type",
                "topic",
                "text",
                "_source_row_index",
            ]
        )
    return pd.DataFrame(rows)


def rebuild_triplets_in_dataframe(
    df,
    normalized_triplets_df,
    triplets_column="triples",
    topic_filter=None,
):
    df_out = df.copy()

    keep_all = topic_filter is None or topic_filter == "all"
    allowed_topics = None
    if not keep_all and isinstance(topic_filter, (list, set, tuple)):
        allowed_topics = set(topic_filter)

    def keep_topic(triplet):
        if keep_all:
            return True
        topic = triplet.get("topic")
        if allowed_topics is not None:
            return topic in allowed_topics
        return topic == topic_filter

    grouped = {}
    for _, row in normalized_triplets_df.iterrows():
        src_idx = row["_source_row_index"]
        row_dict = row.drop(labels=["_source_row_index"]).to_dict()
        if keep_topic(row_dict):
            grouped.setdefault(src_idx, []).append(row_dict)

    new_col = []
    for idx in df_out.index:
        if idx in grouped:
            triplets = grouped[idx]
        else:
            original_triplets = parse_legal_triplets_cell(
                df_out.at[idx, triplets_column]
            )
            triplets = [
                t for t in original_triplets if isinstance(t, dict) and keep_topic(t)
            ]
        new_col.append(triplets)

    df_out[triplets_column] = new_col
    return df_out


def ensure_parent_dir(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def write_mapping_outputs(mapping_df, csv_path, json_path):
    ensure_parent_dir(csv_path)
    mapping_df.to_csv(csv_path, index=False)

    grouped_payload = []
    if not mapping_df.empty:
        group_keys = ["cluster_kind", "group", "canonical_label"]
        for group_values, group_df in mapping_df.groupby(
            group_keys, dropna=False, sort=True
        ):
            cluster_kind, group_value, canonical_label = group_values
            grouped_payload.append(
                {
                    "cluster_kind": cluster_kind,
                    "group": group_value,
                    "canonical_label": canonical_label,
                    "canonical_count": int(group_df["canonical_count"].iloc[0]),
                    "cluster_size": int(group_df["cluster_size"].iloc[0]),
                    "variants": [
                        {
                            "label": row["variant_label"],
                            "count": int(row["variant_count"]),
                            "is_canonical": bool(row["is_canonical"]),
                        }
                        for _, row in group_df.sort_values(
                            by=["variant_count", "variant_label"],
                            ascending=[False, True],
                        ).iterrows()
                    ],
                }
            )

    ensure_parent_dir(json_path)
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(grouped_payload, fh, ensure_ascii=False, indent=2)


def build_default_paths(provider, input_file=None):
    dir_path = Path(f"src/experiments/{provider}")
    input_path = (
        Path(input_file)
        if input_file
        else dir_path / f"classes_guided_extraction_by_{provider}.csv"
    )
    output_flat = dir_path / f"{input_path.stem}_normalized.csv"
    output_nested = dir_path / f"triplets_in_row_{input_path.stem}_normalized.csv"
    entity_mapping_csv = dir_path / f"{input_path.stem}_entity_mapping.csv"
    entity_mapping_json = dir_path / f"{input_path.stem}_entity_mapping.json"
    relation_mapping_csv = dir_path / f"{input_path.stem}_relation_mapping.csv"
    relation_mapping_json = dir_path / f"{input_path.stem}_relation_mapping.json"
    return {
        "input": input_path,
        "flat_output": output_flat,
        "nested_output": output_nested,
        "entity_mapping_csv": entity_mapping_csv,
        "entity_mapping_json": entity_mapping_json,
        "relation_mapping_csv": relation_mapping_csv,
        "relation_mapping_json": relation_mapping_json,
    }


def run_merge_pipeline(
    input_file,
    output_file,
    output_file_one_by_row,
    entity_mapping_csv,
    entity_mapping_json,
    relation_mapping_csv,
    relation_mapping_json,
    embedding_provider=DEFAULT_EMBEDDING_PROVIDER,
    embedding_model=None,
    topic_filter=DEFAULT_TOPIC_FILTER,
    entity_similarity_threshold=DEFAULT_ENTITY_SIMILARITY_THRESHOLD,
    relation_similarity_threshold=DEFAULT_RELATION_SIMILARITY_THRESHOLD,
    relation_canonical_strategy=DEFAULT_RELATION_CANONICAL_STRATEGY,
    ontology_file=None,
    batch_size=DEFAULT_BATCH_SIZE,
):
    embedding_provider = embedding_provider.strip().lower()
    embedding_model = embedding_model or get_default_embedding_model(embedding_provider)

    print(f"Using embedding provider: {embedding_provider}, model: {embedding_model}")
    print(f"Relation canonical strategy: {relation_canonical_strategy}")

    ontology_index = None
    if relation_canonical_strategy == "ontology-aware":
        if ontology_file is None:
            ontology_file = default_ontology_path(embedding_provider)
        ontology_file = Path(ontology_file)
        if not ontology_file.exists():
            raise FileNotFoundError(
                f"Ontology file required for ontology-aware relation merge: {ontology_file}"
            )
        ontology_index = load_relation_ontology_index(ontology_file)
        print(
            "Ontology-aware relation merge loaded: "
            f"{ontology_file} | "
            f"relations={len(ontology_index['relation_locals'])} | "
            f"signatures={len(ontology_index['signatures_local'])}"
        )

    df = pd.read_csv(input_file)
    triplets_column = "legal_triplets" if "legal_triplets" in df.columns else "triples"
    print(f"Using triplets column: {triplets_column}")
    flat_triplets_df = explode_triplets_dataframe(
        df,
        triplets_column=triplets_column,
        topic_filter=topic_filter,
    )
    if flat_triplets_df.empty:
        print(
            "No triplets found in the selected triplets column after parsing/topic filtering. "
            f"triplets_column={triplets_column!r}, topic_filter={topic_filter!r}"
        )
        return None

    print("Global entity merge started...")
    merged_entities_df, entity_map, entity_mapping_df = merge_entities_global(
        flat_triplets_df,
        similarity_threshold=entity_similarity_threshold,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        batch_size=batch_size,
    )

    print("Relation merge started...")
    merged_relations_df, relation_map, relation_mapping_df = merge_relations(
        merged_entities_df,
        similarity_threshold=relation_similarity_threshold,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        batch_size=batch_size,
        relation_canonical_strategy=relation_canonical_strategy,
        ontology_index=ontology_index,
    )

    ensure_parent_dir(output_file)
    merged_relations_df.to_csv(output_file, index=False)

    rebuilt_df = rebuild_triplets_in_dataframe(
        df,
        merged_relations_df,
        triplets_column=triplets_column,
        topic_filter=topic_filter,
    )
    rebuilt_df[triplets_column] = rebuilt_df[triplets_column].map(
        lambda triplets: json.dumps(triplets, ensure_ascii=False)
    )

    ensure_parent_dir(output_file_one_by_row)
    rebuilt_df.to_csv(output_file_one_by_row, index=False)

    write_mapping_outputs(entity_mapping_df, entity_mapping_csv, entity_mapping_json)
    write_mapping_outputs(
        relation_mapping_df, relation_mapping_csv, relation_mapping_json
    )

    print("Normalization finished:")
    print(f"  Flat output: {output_file}")
    print(f"  Nested output: {output_file_one_by_row}")
    print(f"  Entity mapping CSV: {entity_mapping_csv}")
    print(f"  Entity mapping JSON: {entity_mapping_json}")
    print(f"  Relation mapping CSV: {relation_mapping_csv}")
    print(f"  Relation mapping JSON: {relation_mapping_json}")

    return {
        "normalized_triplets": merged_relations_df,
        "entity_map": entity_map,
        "entity_mapping_df": entity_mapping_df,
        "relation_map": relation_map,
        "relation_mapping_df": relation_mapping_df,
        "nested_df": rebuilt_df,
    }


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Merge semantically similar entities and predicates in extracted triplets "
            "using OpenAI or Mistral embeddings."
        )
    )
    parser.add_argument(
        "--provider",
        default=DEFAULT_EMBEDDING_PROVIDER,
        choices=["openai", "mistral"],
        help="Embedding provider to use.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Embedding model override. If omitted, the provider default is used.",
    )
    parser.add_argument(
        "--input-file",
        default=None,
        help="Input CSV with a legal_triplets column.",
    )
    parser.add_argument(
        "--output-file",
        default=None,
        help="Flat normalized triplets CSV output path.",
    )
    parser.add_argument(
        "--output-file-one-by-row",
        default=None,
        help="Row-level nested CSV output path.",
    )
    parser.add_argument(
        "--entity-mapping-csv",
        default=None,
        help="CSV output for entity merge mappings.",
    )
    parser.add_argument(
        "--entity-mapping-json",
        default=None,
        help="JSON output for entity merge mappings.",
    )
    parser.add_argument(
        "--relation-mapping-csv",
        default=None,
        help="CSV output for relation merge mappings.",
    )
    parser.add_argument(
        "--relation-mapping-json",
        default=None,
        help="JSON output for relation merge mappings.",
    )
    parser.add_argument(
        "--topic-filter",
        default=DEFAULT_TOPIC_FILTER,
        help='Topic filter. Use "all" to keep all triplets.',
    )
    parser.add_argument(
        "--entity-similarity-threshold",
        type=float,
        default=DEFAULT_ENTITY_SIMILARITY_THRESHOLD,
        help="Cosine similarity threshold for entity merges.",
    )
    parser.add_argument(
        "--relation-similarity-threshold",
        type=float,
        default=DEFAULT_RELATION_SIMILARITY_THRESHOLD,
        help="Cosine similarity threshold for relation merges.",
    )
    parser.add_argument(
        "--relation-canonical-strategy",
        choices=["frequency", "ontology-aware"],
        default=DEFAULT_RELATION_CANONICAL_STRATEGY,
        help=(
            "How to pick the canonical relation label inside each similarity cluster. "
            "'frequency' keeps the previous behavior. 'ontology-aware' first prefers "
            "a relation valid for the head_type/tail_type signature, then any relation "
            "defined in the ontology, then falls back to frequency."
        ),
    )
    parser.add_argument(
        "--ontology-file",
        type=Path,
        default=None,
        help=(
            "Ontology TTL used by --relation-canonical-strategy ontology-aware. "
            "Defaults to src/exp/ontology_extension/owl/ontology_extended_{provider}.ttl."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Embedding batch size.",
    )
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    default_paths = build_default_paths(args.provider, input_file=args.input_file)
    run_merge_pipeline(
        input_file=args.input_file or default_paths["input"],
        output_file=args.output_file or default_paths["flat_output"],
        output_file_one_by_row=(
            args.output_file_one_by_row or default_paths["nested_output"]
        ),
        entity_mapping_csv=(
            args.entity_mapping_csv or default_paths["entity_mapping_csv"]
        ),
        entity_mapping_json=(
            args.entity_mapping_json or default_paths["entity_mapping_json"]
        ),
        relation_mapping_csv=(
            args.relation_mapping_csv or default_paths["relation_mapping_csv"]
        ),
        relation_mapping_json=(
            args.relation_mapping_json or default_paths["relation_mapping_json"]
        ),
        embedding_provider=args.provider,
        embedding_model=args.model,
        topic_filter=args.topic_filter,
        entity_similarity_threshold=args.entity_similarity_threshold,
        relation_similarity_threshold=args.relation_similarity_threshold,
        relation_canonical_strategy=args.relation_canonical_strategy,
        ontology_file=args.ontology_file,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
