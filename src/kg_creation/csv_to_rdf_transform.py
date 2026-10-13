from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import re
import subprocess
import sys
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

from ontologies_and_kg_combination import (
    COMBINED_EMBEDDING_PROVIDER,
    COMBINED_PROVIDER,
    patch_combined_mapping_text,
    prepare_combined_inputs,
    provider_assets,
    provider_choices,
    rml_provenance_fields,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

BASE_RML = Path("src/data/csv_to_rml_mapping.ttl")
BUILD_RML_RULES = Path("src/preprocess/build_rml_extraction_rules_from_ontology.py")
MERGE_ELEMENTS = Path("src/kg_creation/elements_merge_and_filtered.py")
FLATTEN_KG = Path("src/preprocess/flatten_kg_results.py")
CREATE_KG = Path("src/preprocess/create_kg_from_rlm.py")
INSTANCE_NS = "https://w3id.org/semleg/legimaintlex/resource"
SEMLEG_NS = "https://w3id.org/semleg#"
SEMLEGM_NS = "https://w3id.org/semleg/maintenance#"
DEFAULT_MERGE_BATCH_SIZE = 128
DEFAULT_ENTITY_SIMILARITY_THRESHOLD = 0.7
DEFAULT_RELATION_SIMILARITY_THRESHOLD = 0.7
DEFAULT_RELATION_CANONICAL_STRATEGY = "ontology-aware"
DEFAULT_FILTER_STRATEGY = "article"
DEFAULT_TOPIC_FILTER = "all"
PIPELINE_STAGES = (
    "build-rules",
    "merge",
    "flatten",
    "prepare-rml-sources",
    "create-kg",
)
ARTICLE_NUMBER_RE = re.compile(
    r"^\s*(?:Art\.|Article)\s+(.{1,32}?)(?:\s*\.\s*-|\s*[.):-]|\s*$)",
    re.IGNORECASE,
)
CURRENT_LEGAL_ACT_TYPES = {"artifact", "source"}
CURRENT_LEGAL_ACT_RE = re.compile(
    r"^(?:(?:l|le|la|les|du|de|des|au|aux|a|ce|cet|cette)\s+)*"
    r"present(?:e|s|es)?\s+"
    r"(?:arrete|arret|decret|loi|decision|ordonnance|reglement|acte|texte)$"
)

TRIPLET_RML_COLUMNS = [
    "source_row_id",
    "extraction_provider",
    "extraction_provider_label",
    "extraction_model",
    "extraction_source_file",
    "extraction_provider_uri",
    "relation_extraction_run_uri",
    "relation_extraction_label",
    "mention_extraction_activity_uri",
    "document_key",
    "article_id",
    "article_key",
    "article_number",
    "article_uri",
    "triple_uri",
    "triplet_index",
    "head",
    "head_pref_label",
    "head_resolution_mode",
    "head_resolved_entity_id",
    "head_uri",
    "head_type",
    "head_type_iri",
    "head_start",
    "head_end",
    "relation",
    "relation_iri",
    "tail",
    "tail_pref_label",
    "tail_resolution_mode",
    "tail_resolved_entity_id",
    "tail_uri",
    "tail_type",
    "tail_type_iri",
    "tail_start",
    "tail_end",
    "topic",
]

MENTION_RML_COLUMNS = [
    "source_row_id",
    "extraction_provider",
    "extraction_provider_label",
    "extraction_model",
    "extraction_source_file",
    "extraction_provider_uri",
    "relation_extraction_run_uri",
    "relation_extraction_label",
    "mention_extraction_activity_uri",
    "triplet_index",
    "mention_role",
    "mention_uri",
    "entity_uri",
    "document_key",
    "article_uri",
    "article_key",
    "article_number",
    "mention_start",
    "mention_end",
    "mention_resolution_mode",
    "resolved_entity_id",
]


@dataclass(frozen=True)
class ProviderConfig:
    provider: str
    provider_label: str
    extraction_model: str
    results_csv: Path
    ontology_ttl: Path
    rml_mapping: Path


@dataclass(frozen=True)
class RmlSourceFiles:
    documents: Path
    triplets: Path
    mentions: Path


@dataclass(frozen=True)
class MergeOutputFiles:
    flat_output: Path
    nested_output: Path
    entity_mapping_csv: Path
    entity_mapping_json: Path
    relation_mapping_csv: Path
    relation_mapping_json: Path


def provider_config(provider: str) -> ProviderConfig:
    assets = provider_assets(provider)

    return ProviderConfig(
        provider=assets.provider,
        provider_label=assets.provider_label,
        extraction_model=assets.extraction_model,
        results_csv=assets.results_csv,
        ontology_ttl=assets.ontology_ttl,
        rml_mapping=assets.rml_mapping,
    )


def command_to_text(command: list[str]) -> str:
    return " ".join(f'"{part}"' if " " in part else part for part in command)


def run_step(name: str, command: list[str], dry_run: bool) -> None:
    print(f"\n== {name} ==")
    print(command_to_text(command))
    if dry_run:
        return
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def check_path(path: Path, label: str) -> None:
    if not (REPO_ROOT / path).exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def set_max_csv_field_size() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def read_csv(path: Path) -> list[dict[str, str]]:
    check_path(path, "CSV")
    set_max_csv_field_size()
    with (REPO_ROOT / path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]], columns: list[str]) -> None:
    target = REPO_ROOT / path
    target.parent.mkdir(parents=True, exist_ok=True)
    extras = sorted({key for row in rows for key in row} - set(columns))
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns + extras)
        writer.writeheader()
        writer.writerows(rows)


def blank(value: Any) -> bool:
    return value is None or str(value).strip().lower() in {
        "",
        "nan",
        "none",
        "null",
        "na",
    }


def first(*values: Any) -> str:
    for value in values:
        if not blank(value):
            return str(value)
    return ""


def normalized_text(value: Any) -> str:
    return re.sub(r"\s+", " ", first(value)).strip()


def folded_text(value: Any) -> str:
    text = re.sub(r"[’‘ʼ`´]", " ", normalized_text(value))
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = text.encode("ascii", "ignore").decode("ascii").lower()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def short_label(value: Any, max_chars: int = 50) -> str:
    text = normalized_text(value)
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars].rsplit(" ", 1)[0].strip()
    return cut or text[:max_chars].strip()


def label_sort_key(label: str, count: int) -> tuple[int, int, int, str]:
    stripped = label.strip()
    sentence_initial_upper = (
        len(stripped) > 1
        and stripped[0].isupper()
        and stripped[1:].lower() == stripped[1:]
    )
    return (
        count,
        0 if sentence_initial_upper else 1,
        len(stripped),
        stripped,
    )


def choose_label_variant(labels: list[str]) -> str:
    counts: dict[str, int] = {}
    for label in labels:
        if label:
            counts[label] = counts.get(label, 0) + 1
    if not counts:
        return ""
    return max(counts, key=lambda label: label_sort_key(label, counts[label]))


def collapse_combined_entity_labels(triplet_rows: list[dict[str, str]]) -> None:
    labels_by_entity: dict[str, dict[str, list[str]]] = {}
    for row in triplet_rows:
        for role in ("head", "tail"):
            entity = row.get(f"{role}_uri", "")
            if not entity:
                continue
            labels = labels_by_entity.setdefault(entity, {"pref": [], "alt": []})
            pref_label = first(
                row.get(f"{role}_pref_label"), short_label(row.get(role))
            )
            alt_label = first(row.get(role))
            if pref_label:
                labels["pref"].append(pref_label)
            if alt_label:
                labels["alt"].append(alt_label)

    selected_labels = {
        entity: {
            "pref": choose_label_variant(labels["pref"]),
            "alt": choose_label_variant(labels["alt"]),
        }
        for entity, labels in labels_by_entity.items()
    }

    for row in triplet_rows:
        for role in ("head", "tail"):
            entity = row.get(f"{role}_uri", "")
            selected = selected_labels.get(entity)
            if not selected:
                continue
            row[f"{role}_pref_label"] = selected["pref"]
            row[role] = selected["alt"]


def as_int(value: Any) -> int | None:
    if blank(value):
        return None
    try:
        return int(float(str(value).strip()))
    except ValueError:
        return None


def parse_literal(value: Any) -> Any:
    if blank(value):
        return None
    if isinstance(value, (dict, list)):
        return value
    text = str(value).strip()
    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(re.sub(r"\bnan\b", "None", text, flags=re.IGNORECASE))
        except Exception:
            pass
    return None


def as_mapping(value: Any) -> dict[str, Any]:
    parsed = parse_literal(value)
    return parsed if isinstance(parsed, dict) else {}


def as_triplets(value: Any) -> list[dict[str, Any]]:
    parsed = parse_literal(value)
    if isinstance(parsed, list):
        return [item for item in parsed if isinstance(item, dict)]
    if isinstance(parsed, dict) and isinstance(parsed.get("triplets"), list):
        return [item for item in parsed["triplets"] if isinstance(item, dict)]
    return []


def date_to_xsd(value: Any) -> str:
    if blank(value):
        return ""
    text = str(value).strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        return text
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}T.+", text):
        return text[:10]

    match = re.fullmatch(r"(\d{2})[./-](\d{2})[./-](\d{4})", text)
    if match:
        day, month, year = match.groups()
        return f"{year}-{month}-{day}"

    try:
        timestamp = float(text)
    except ValueError:
        return ""
    if timestamp > 10_000_000_000:
        timestamp /= 1000
    try:
        return datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime("%Y-%m-%d")
    except (OSError, OverflowError, ValueError):
        return ""


def extract_article_number(content: Any) -> str:
    text = first(content)
    match = ARTICLE_NUMBER_RE.search(text)
    if not match:
        return ""
    return re.sub(r"\s+", " ", match.group(1)).strip()


def article_number(
    input_row: dict[str, str],
    provenance: dict[str, Any],
    content: str,
) -> str:
    return first(
        input_row.get("num"),
        provenance.get("number"),
        extract_article_number(content),
    )


def article_key(article_id: str, number: str, source_row_id: str) -> str:
    parts = [
        first(number, "article"),
        first(article_id),
        first(source_row_id),
    ]
    return "-".join(part for part in parts if not blank(part))


def document_key(input_row: dict[str, str], provenance: dict[str, Any]) -> str:
    title = normalized_text(
        first(
            input_row.get("title"),
            provenance.get("title"),
            input_row.get("corpus_id"),
            input_row.get("id"),
            "document",
        )
    )
    return f"{slug(title, 80)}-{short_hash(title)}"


def slug(value: Any, max_len: int = 72) -> str:
    text = first(value, "unnamed")
    text = re.sub(r"[’‘ʼ`´]", " ", text)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-").lower()
    return (text or "unnamed")[:max_len].strip("-") or "unnamed"


def short_hash(*values: Any) -> str:
    text = "||".join(first(value) for value in values)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def iri_part(value: Any) -> str:
    return quote(first(value, "unknown"), safe="._-~")


def article_uri(document_key_value: str, article_key_value: str) -> str:
    return (
        f"{INSTANCE_NS}/document/{iri_part(document_key_value)}"
        f"/article/{iri_part(article_key_value)}"
    )


def document_uri(document_key_value: str) -> str:
    return f"{INSTANCE_NS}/document/{iri_part(document_key_value)}"


def entity_uri(label: str, type_value: str) -> str:
    label_key = folded_text(label)
    type_key = folded_text(type_value)
    return (
        f"{INSTANCE_NS}/entity/{slug(type_value or 'Entity', 32)}"
        f"/{slug(label)}-{short_hash(type_key, label_key)}"
    )


def is_current_legal_act_reference(label: str, type_value: str) -> bool:
    type_folded = folded_text(type_value)
    return (
        type_folded in CURRENT_LEGAL_ACT_TYPES
        and CURRENT_LEGAL_ACT_RE.search(folded_text(label)) is not None
    )


def resolved_entity_uri(
    label: str,
    type_value: str,
    document_key_value: str,
) -> tuple[str, str, str]:
    if is_current_legal_act_reference(label, type_value):
        return document_uri(document_key_value), "current_legal_act", document_key_value
    return entity_uri(label, type_value), "global_entity", ""


def triple_uri(
    article_id: str,
    article_key_value: str,
    source_row_id: str,
    triplet_index: str,
    extraction_provider: str = "",
) -> str:
    extraction_anchor = first(article_id, source_row_id)
    article_anchor = first(article_key_value, source_row_id)
    provider_scope = (
        f"/{iri_part(extraction_provider)}" if not blank(extraction_provider) else ""
    )
    return (
        f"{INSTANCE_NS}/extraction{provider_scope}/{iri_part(extraction_anchor)}"
        f"/article/{iri_part(article_anchor)}/triple/{iri_part(triplet_index)}"
    )


def mention_uri(
    article_id: str,
    article_key_value: str,
    source_row_id: str,
    triplet_index: str,
    role: str,
    extraction_provider: str = "",
) -> str:
    extraction_anchor = first(article_id, source_row_id)
    article_anchor = first(article_key_value, source_row_id)
    provider_scope = (
        f"/{iri_part(extraction_provider)}" if not blank(extraction_provider) else ""
    )
    return (
        f"{INSTANCE_NS}/extraction{provider_scope}/{iri_part(extraction_anchor)}"
        f"/article/{iri_part(article_anchor)}"
        f"/triple/{iri_part(triplet_index)}/mention/{iri_part(role)}"
    )


def get_segments(triplet: dict[str, Any], side: str) -> list[tuple[int, int]]:
    spans = triplet.get(f"{side}_spans")
    if isinstance(spans, list):
        out = []
        for item in spans:
            if not isinstance(item, dict):
                continue
            start, end = item.get("start"), item.get("end")
            if isinstance(start, int) and isinstance(end, int) and end > start:
                out.append((start, end))
        if out:
            return out

    start, end = triplet.get(f"{side}_start"), triplet.get(f"{side}_end")
    if isinstance(start, int) and isinstance(end, int) and end > start:
        return [(start, end)]
    return []


def clip_segments(
    segments: list[tuple[int, int]],
    content: str,
) -> list[tuple[int, int]]:
    size = len(content)
    return [
        (max(0, min(start, size)), max(0, min(end, size)))
        for start, end in segments
        if max(0, min(end, size)) > max(0, min(start, size))
    ]


def infer_segment(content: str, mention: str) -> list[tuple[int, int]]:
    if not mention:
        return []
    start = content.find(mention)
    if start >= 0:
        return [(start, start + len(mention))]
    match = re.search(re.escape(mention), content, flags=re.IGNORECASE)
    return [(match.start(), match.end())] if match else []


def side_segments(
    triplet: dict[str, Any],
    side: str,
    content: str,
    infer_missing_spans: bool,
) -> list[tuple[int, int]]:
    segments = clip_segments(get_segments(triplet, side), content)
    if segments or not infer_missing_spans:
        return segments
    return infer_segment(content, first(triplet.get(side)))


def segment_bounds(segments: list[tuple[int, int]]) -> tuple[str, str]:
    if not segments:
        return "", ""
    return str(segments[0][0]), str(segments[-1][1])


def parse_prefixes(ttl_text: str) -> dict[str, str]:
    return {
        prefix: namespace
        for prefix, namespace in re.findall(
            r"@prefix\s+([A-Za-z][\w-]*):\s+<([^>]+)>\s*\.",
            ttl_text,
        )
    }


def resolve_ttl_term(term: str, prefixes: dict[str, str]) -> str:
    value = term.strip().rstrip(" .;")
    if value.startswith("<") and value.endswith(">"):
        return value[1:-1]
    if ":" in value:
        prefix, suffix = value.split(":", 1)
        if prefix in prefixes:
            return prefixes[prefix] + suffix
    return value if value.startswith(("http://", "https://")) else ""


def ontology_lookup(mapping: Path) -> tuple[dict[str, str], dict[str, str]]:
    text = (REPO_ROOT / mapping).read_text(encoding="utf-8")
    prefixes = parse_prefixes(text)
    type_iris = {
        "Action": SEMLEG_NS + "Action",
        "Actor": SEMLEG_NS + "Actor",
        "Artifact": SEMLEG_NS + "Artifact",
        "Condition": SEMLEG_NS + "Condition",
        "Definition": SEMLEG_NS + "Definition",
        "Location": SEMLEG_NS + "Location",
        "Reason": SEMLEG_NS + "Reason",
        "Situation": SEMLEG_NS + "Situation",
        "Source": SEMLEG_NS + "Source",
        "Time": SEMLEG_NS + "Time",
        "Modality": SEMLEGM_NS + "Modality",
    }
    relation_iris = {
        "hasCondition": SEMLEG_NS + "hasCondition",
        "hasDefinition": SEMLEG_NS + "hasDefinition",
        "hasSubSource": SEMLEG_NS + "hasSubSource",
        "hasTime": SEMLEG_NS + "hasTime",
    }
    for iri_term, type_value in re.findall(
        r"map:ClassRule_[\s\S]*?map:classIri\s+([^;\n]+)\s*;"
        r"[\s\S]*?map:typeValue\s+\"([^\"]+)\"",
        text,
    ):
        iri = resolve_ttl_term(iri_term, prefixes)
        if iri:
            type_iris[type_value] = iri

    for relation_value, relation_iri in re.findall(
        r"map:RelationRule_[\s\S]*?map:relationValue\s+\"([^\"]+)\"\s*;"
        r"[\s\S]*?map:relationIri\s+\"([^\"]+)\"",
        text,
    ):
        relation_iris[relation_value] = relation_iri
    return type_iris, relation_iris


def lookup_iri(value: str, mapping: dict[str, str], fallback_ns: str) -> str:
    if value.startswith(("http://", "https://")):
        return value
    if value in mapping:
        return mapping[value]
    for key, iri in mapping.items():
        if key.lower() == value.lower():
            return iri
    return fallback_ns + re.sub(r"[^A-Za-z0-9_]+", "", value)


def indexed_row(
    rows: list[dict[str, str]],
    source_row_id: int | None,
    fallback_position: int,
) -> dict[str, str]:
    if source_row_id is not None and 0 <= source_row_id < len(rows):
        return rows[source_row_id]
    if 0 <= fallback_position < len(rows):
        return rows[fallback_position]
    return {}


def rml_source_files(config: ProviderConfig, results_csv: Path) -> RmlSourceFiles:
    source_dir = Path("exp/kg") / config.provider / "rml_sources"
    stem = results_csv.stem
    return RmlSourceFiles(
        documents=source_dir / f"{stem}_documents_rml.csv",
        triplets=source_dir / f"{stem}_legal_triplets_rml.csv",
        mentions=source_dir / f"{stem}_legal_mentions_rml.csv",
    )


def merge_output_files(results_csv: Path) -> MergeOutputFiles:
    output_dir = results_csv.parent
    stem = results_csv.stem
    merge_stem = f"{stem}_elements_merged"
    return MergeOutputFiles(
        flat_output=output_dir / f"{merge_stem}.csv",
        nested_output=output_dir / f"triplets_in_row_{merge_stem}.csv",
        entity_mapping_csv=output_dir / f"{stem}_entity_mapping.csv",
        entity_mapping_json=output_dir / f"{stem}_entity_mapping.json",
        relation_mapping_csv=output_dir / f"{stem}_relation_mapping.csv",
        relation_mapping_json=output_dir / f"{stem}_relation_mapping.json",
    )


def results_csv_after_merge(results_csv: Path, args: argparse.Namespace) -> Path:
    if args.skip_merge:
        return results_csv
    return merge_output_files(results_csv).nested_output


def apply_start_at(args: argparse.Namespace) -> None:
    start_index = PIPELINE_STAGES.index(args.start_at)
    if start_index > PIPELINE_STAGES.index("build-rules"):
        args.skip_build_rules = True
    if start_index > PIPELINE_STAGES.index("merge"):
        args.skip_merge = True
    if start_index > PIPELINE_STAGES.index("flatten"):
        args.skip_flatten = True
    if start_index > PIPELINE_STAGES.index("prepare-rml-sources"):
        args.skip_prepare_rml_sources = True


def create_document_rml_source(
    input_corpus: Path,
    output_csv: Path,
    included_source_row_ids: set[int] | None = None,
) -> int:
    rows = read_csv(input_corpus)
    filtered_rows = []
    for source_row_id, row in enumerate(rows):
        if (
            included_source_row_ids is not None
            and source_row_id not in included_source_row_ids
        ):
            continue
        number = article_number(row, {}, first(row.get("content")))
        row["document_key"] = document_key(row, {})
        row["article_number"] = number
        row["article_key"] = article_key(
            first(row.get("id")), number, str(source_row_id)
        )
        row["date"] = date_to_xsd(row.get("date"))
        filtered_rows.append(row)
    columns = list(filtered_rows[0].keys()) if filtered_rows else []
    write_csv(output_csv, filtered_rows, columns)
    return len(filtered_rows)


def filtered_source_row_ids(
    results_csv: Path,
    max_rows: int | None = None,
) -> set[int]:
    """Return corpus row ids that have at least one extracted triplet."""
    result_rows = read_csv(results_csv)
    if max_rows is not None:
        result_rows = result_rows[:max_rows]

    source_row_ids = set()
    for result_position, result_row in enumerate(result_rows):
        if not as_triplets(result_row.get("legal_triplets")):
            continue
        source_index = as_int(result_row.get("index"))
        source_row_ids.add(
            source_index if source_index is not None else result_position
        )
    return source_row_ids


def create_triplet_and_mention_rml_sources(
    config: ProviderConfig,
    input_corpus: Path,
    results_csv: Path,
    triplets_csv: Path,
    mentions_csv: Path,
    infer_missing_spans: bool,
    max_rows: int | None,
) -> tuple[int, int]:
    input_rows = read_csv(input_corpus)
    result_rows = read_csv(results_csv)
    if max_rows is not None:
        result_rows = result_rows[:max_rows]
    type_iris, relation_iris = ontology_lookup(config.rml_mapping)
    triplet_rows: list[dict[str, str]] = []
    mention_rows: list[dict[str, str]] = []

    for result_position, result_row in enumerate(result_rows):
        source_index = as_int(result_row.get("index"))
        input_row = indexed_row(input_rows, source_index, result_position)
        provenance = as_mapping(result_row.get("provenance"))
        source_row_id = first(result_row.get("index"), result_position)
        article_id = first(
            input_row.get("id"),
            provenance.get("id_local"),
            result_row.get("article_id"),
            source_row_id,
        )
        content = first(result_row.get("content"), input_row.get("content"))
        doc_key = document_key(input_row, provenance)
        number = article_number(input_row, provenance, content)
        key = article_key(article_id, number, source_row_id)
        article = article_uri(doc_key, key)
        row_topic = first(result_row.get("context_main_topic"))
        extraction_provider = first(
            result_row.get("extraction_provider"),
            "" if config.provider != COMBINED_PROVIDER else config.provider,
        )
        extraction_model = first(
            result_row.get("extraction_model"),
            config.extraction_model,
        )
        extraction_provenance = rml_provenance_fields(
            extraction_provider or config.provider,
            model=extraction_model,
            source_file=result_row.get("extraction_source_file") or results_csv,
        )
        uri_provider_scope = first(result_row.get("extraction_provider"))

        for fallback_index, triplet in enumerate(
            as_triplets(result_row.get("legal_triplets"))
        ):
            triplet_index = first(triplet.get("triplet_index"), fallback_index)
            head = first(triplet.get("head"))
            tail = first(triplet.get("tail"))
            head_type = first(triplet.get("head_type"))
            tail_type = first(triplet.get("tail_type"))
            relation = first(triplet.get("relation"))
            head_segments = side_segments(triplet, "head", content, infer_missing_spans)
            tail_segments = side_segments(triplet, "tail", content, infer_missing_spans)
            head_start, head_end = segment_bounds(head_segments)
            tail_start, tail_end = segment_bounds(tail_segments)
            head_uri, head_resolution_mode, head_resolved_id = resolved_entity_uri(
                head,
                head_type,
                doc_key,
            )
            tail_uri, tail_resolution_mode, tail_resolved_id = resolved_entity_uri(
                tail,
                tail_type,
                doc_key,
            )

            triplet_rows.append(
                {
                    "source_row_id": source_row_id,
                    **extraction_provenance,
                    "document_key": doc_key,
                    "article_id": article_id,
                    "article_key": key,
                    "article_number": number,
                    "article_uri": article,
                    "triple_uri": triple_uri(
                        article_id,
                        key,
                        source_row_id,
                        triplet_index,
                        uri_provider_scope,
                    ),
                    "triplet_index": triplet_index,
                    "head": head,
                    "head_pref_label": short_label(head),
                    "head_resolution_mode": head_resolution_mode,
                    "head_resolved_entity_id": head_resolved_id,
                    "head_uri": head_uri,
                    "head_type": head_type,
                    "head_type_iri": lookup_iri(head_type, type_iris, SEMLEG_NS),
                    "head_start": head_start,
                    "head_end": head_end,
                    "relation": relation,
                    "relation_iri": lookup_iri(relation, relation_iris, SEMLEGM_NS),
                    "tail": tail,
                    "tail_pref_label": short_label(tail),
                    "tail_resolution_mode": tail_resolution_mode,
                    "tail_resolved_entity_id": tail_resolved_id,
                    "tail_uri": tail_uri,
                    "tail_type": tail_type,
                    "tail_type_iri": lookup_iri(tail_type, type_iris, SEMLEG_NS),
                    "tail_start": tail_start,
                    "tail_end": tail_end,
                    "topic": first(triplet.get("topic"), row_topic),
                }
            )

            for role, uri, segments, mode, resolved_id in (
                (
                    "head",
                    head_uri,
                    head_segments,
                    head_resolution_mode,
                    head_resolved_id,
                ),
                (
                    "tail",
                    tail_uri,
                    tail_segments,
                    tail_resolution_mode,
                    tail_resolved_id,
                ),
            ):
                start, end = segment_bounds(segments)
                if not start or not end:
                    continue
                mention_rows.append(
                    {
                        "source_row_id": source_row_id,
                        **extraction_provenance,
                        "triplet_index": triplet_index,
                        "mention_role": role,
                        "mention_uri": mention_uri(
                            article_id,
                            key,
                            source_row_id,
                            triplet_index,
                            role,
                            uri_provider_scope,
                        ),
                        "entity_uri": uri,
                        "document_key": doc_key,
                        "article_uri": article,
                        "article_key": key,
                        "article_number": number,
                        "mention_start": start,
                        "mention_end": end,
                        "mention_resolution_mode": mode,
                        "resolved_entity_id": resolved_id,
                    }
                )

    if config.provider == COMBINED_PROVIDER:
        collapse_combined_entity_labels(triplet_rows)
    write_csv(triplets_csv, triplet_rows, TRIPLET_RML_COLUMNS)
    write_csv(mentions_csv, mention_rows, MENTION_RML_COLUMNS)
    return len(triplet_rows), len(mention_rows)


def patch_mapping_sources(config: ProviderConfig, sources: RmlSourceFiles) -> None:
    mapping_path = REPO_ROOT / config.rml_mapping
    text = mapping_path.read_text(encoding="utf-8")

    replacements = {
        "RawLegalDocumentSource": sources.documents,
        "NormalizedTripletSource": sources.triplets,
        "NormalizedMentionSource": sources.mentions,
    }
    for source_name, source_path in replacements.items():
        text = re.sub(
            rf'(map:{source_name}\s+[\s\S]*?rml:source\s+)"[^"]+"',
            rf'\1"{source_path.as_posix()}"',
            text,
            count=1,
        )

    text = text.replace("/article/{num}", "/article/{article_key}")
    text = text.replace("/document/{id}", "/document/{document_key}")
    text = re.sub(
        r'(rr:predicate\s+eli:number\s*;\s*rr:objectMap\s*\[\s*rml:reference\s+)"num"',
        r'\1"article_number"',
        text,
        count=1,
    )

    text = re.sub(
        r'(rml:reference\s+"date"\s*;\s*rr:datatype\s+)xsd:(?:string|dateTime|date)',
        r"\1xsd:date",
        text,
        count=1,
    )
    text = text.replace('rml:reference "head_alt_label"', 'rml:reference "head"')
    text = text.replace('rml:reference "tail_alt_label"', 'rml:reference "tail"')
    if config.provider == COMBINED_PROVIDER:
        text = patch_combined_mapping_text(text)
    mapping_path.write_text(text, encoding="utf-8")


def prepare_rml_sources(
    config: ProviderConfig,
    input_corpus: Path,
    results_csv: Path,
    infer_missing_spans: bool,
    max_rows: int | None,
    dry_run: bool,
) -> None:
    sources = rml_source_files(config, results_csv)
    print("\n== Prepare generated RML CSV sources ==")
    print(f"Documents: {sources.documents}")
    print(f"Triplets:  {sources.triplets}")
    print(f"Mentions:  {sources.mentions}")
    print(f"Patch mapping: {config.rml_mapping}")
    if dry_run:
        return

    included_source_row_ids = filtered_source_row_ids(results_csv, max_rows)
    document_count = create_document_rml_source(
        input_corpus,
        sources.documents,
        included_source_row_ids=included_source_row_ids,
    )
    triplet_count, mention_count = create_triplet_and_mention_rml_sources(
        config=config,
        input_corpus=input_corpus,
        results_csv=results_csv,
        triplets_csv=sources.triplets,
        mentions_csv=sources.mentions,
        infer_missing_spans=infer_missing_spans,
        max_rows=max_rows,
    )
    patch_mapping_sources(config, sources)
    print(f"Documents: {document_count}")
    print(f"Triplets: {triplet_count}")
    print(f"Mentions: {mention_count}")


def build_commands(
    config: ProviderConfig,
    input_csv: Path | None,
    args: argparse.Namespace,
) -> list[tuple[str, list[str]]]:
    results_csv = input_csv or config.results_csv
    check_path(results_csv, "Extraction results CSV")
    if not args.skip_build_rules:
        check_path(config.ontology_ttl, "Ontology TTL")
        check_path(BASE_RML, "Base RML mapping")
    elif not args.skip_prepare_rml_sources or not args.skip_create_kg:
        check_path(config.rml_mapping, "Generated RML mapping")

    commands: list[tuple[str, list[str]]] = []

    if not args.skip_build_rules:
        commands.append(
            (
                "Build ontology-driven RML rules",
                [
                    sys.executable,
                    str(BUILD_RML_RULES),
                    "--base-rml",
                    str(BASE_RML),
                    "--ontology",
                    str(config.ontology_ttl),
                    "--output",
                    str(config.rml_mapping),
                    "--extraction-provider",
                    config.provider,
                    "--extraction-provider-label",
                    config.provider_label,
                    "--extraction-model",
                    config.extraction_model,
                    "--extraction-strategy",
                    "signature-driven",
                ],
            )
        )

    if not args.skip_merge:
        check_path(MERGE_ELEMENTS, "Elements merge script")
        merge_outputs = merge_output_files(results_csv)
        merge_provider = args.merge_provider or (
            COMBINED_EMBEDDING_PROVIDER
            if config.provider == COMBINED_PROVIDER
            else config.provider
        )
        merge_command = [
            sys.executable,
            str(MERGE_ELEMENTS),
            "--provider",
            merge_provider,
            "--input-file",
            str(results_csv),
            "--output-file",
            str(merge_outputs.flat_output),
            "--output-file-one-by-row",
            str(merge_outputs.nested_output),
            "--entity-mapping-csv",
            str(merge_outputs.entity_mapping_csv),
            "--entity-mapping-json",
            str(merge_outputs.entity_mapping_json),
            "--relation-mapping-csv",
            str(merge_outputs.relation_mapping_csv),
            "--relation-mapping-json",
            str(merge_outputs.relation_mapping_json),
            "--topic-filter",
            args.topic_filter,
            "--filter-strategy",
            args.filter_strategy,
            "--entity-similarity-threshold",
            str(args.entity_similarity_threshold),
            "--relation-similarity-threshold",
            str(args.relation_similarity_threshold),
            "--relation-canonical-strategy",
            args.relation_canonical_strategy,
            "--ontology-file",
            str(config.ontology_ttl),
            "--batch-size",
            str(args.merge_batch_size),
        ]
        if args.merge_model:
            merge_command.extend(["--model", args.merge_model])
        commands.append(("Merge extracted elements across all data", merge_command))

    if not args.skip_flatten:
        flatten_results_csv = results_csv_after_merge(results_csv, args)
        flatten_command = [
            sys.executable,
            str(FLATTEN_KG),
            "--provider",
            config.provider,
            "--results-csv",
            str(flatten_results_csv),
        ]
        if args.input_corpus:
            flatten_command.extend(["--input-csv", str(args.input_corpus)])
        if args.max_rows is not None:
            flatten_command.extend(["--max-rows", str(args.max_rows)])
        if args.no_infer_spans:
            flatten_command.append("--no-infer-spans")
        commands.append(("Create flat CSV intermediates", flatten_command))

    return commands


def create_kg_command(config: ProviderConfig, args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        str(CREATE_KG),
        "--provider",
        config.provider,
        "--mapping",
        str(config.rml_mapping),
        "--format",
        args.rdf_format,
        "--engine",
        args.engine,
    ]
    if args.kg_max_rows_per_source is not None:
        command.extend(["--max-rows-per-source", str(args.kg_max_rows_per_source)])
    if args.kg_output is not None:
        command.extend(["--output", str(args.kg_output)])
    return command


def providers_from_args(args: argparse.Namespace) -> list[str]:
    return ["mistral", "openai"] if args.provider == "all" else [args.provider]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the CSV-to-RDF pipeline: build RML rules, create flat CSV "
            "intermediates, prepare RML CSV sources, and apply RML rules."
        )
    )
    parser.add_argument(
        "--provider",
        choices=provider_choices(include_all=True),
        default="mistral",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Extraction results CSV. Only valid with a single provider.",
    )
    parser.add_argument(
        "--input-corpus",
        type=Path,
        default=Path("src/data/maintreg_database_clean.csv"),
        help="Corpus CSV used by flatten_kg_results.py.",
    )
    parser.add_argument(
        "--start-at",
        choices=PIPELINE_STAGES,
        default="build-rules",
        help=(
            "Start the pipeline at this stage, automatically skipping previous stages. "
            "Use 'flatten' to skip ontology rule generation and element merge."
        ),
    )
    parser.add_argument(
        "--skip-merge",
        action="store_true",
        help="Skip semantic entity/relation merging before flattening.",
    )
    parser.add_argument("--skip-build-rules", action="store_true")
    parser.add_argument("--skip-flatten", action="store_true")
    parser.add_argument("--skip-prepare-rml-sources", action="store_true")
    parser.add_argument("--skip-create-kg", action="store_true")
    parser.add_argument(
        "--merge-provider",
        choices=["mistral", "openai"],
        default=None,
        help=(
            "Embedding provider for the pre-flatten merge step. Defaults to the "
            "extraction provider, or OpenAI for combined mode."
        ),
    )
    parser.add_argument(
        "--merge-model",
        default=None,
        help="Embedding model override for the pre-flatten merge step.",
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
        "--merge-batch-size",
        type=int,
        default=DEFAULT_MERGE_BATCH_SIZE,
        help="Embedding batch size for the pre-flatten merge step.",
    )
    parser.add_argument(
        "--filter-strategy",
        choices=["article", "triplet"],
        default=DEFAULT_FILTER_STRATEGY,
        help=(
            "Topic filtering strategy for the pre-flatten merge: use the article "
            "domain (article) or each triplet topic (triplet)."
        ),
    )
    parser.add_argument(
        "--topic-filter",
        default=DEFAULT_TOPIC_FILTER,
        help=(
            'Topic regex or "all" for the pre-flatten merge. With the article '
            "strategy, it is matched against the article domain."
        ),
    )
    parser.add_argument(
        "--relation-canonical-strategy",
        choices=["frequency", "ontology-aware"],
        default=DEFAULT_RELATION_CANONICAL_STRATEGY,
        help=(
            "Canonical relation strategy for the pre-KG merge. "
            "Use 'ontology-aware' for final KG creation; use 'frequency' to keep "
            "the older label-frequency behavior."
        ),
    )
    parser.add_argument(
        "--rdf-format", default="turtle", choices=["turtle", "nt", "xml", "json-ld"]
    )
    parser.add_argument("--engine", default="lite", choices=["lite", "pyrml"])
    parser.add_argument(
        "--kg-output",
        type=Path,
        default=None,
        help="Output RDF file path passed to create_kg_from_rlm.py.",
    )
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--kg-max-rows-per-source", type=int, default=None)
    parser.add_argument("--no-infer-spans", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    apply_start_at(args)
    providers = providers_from_args(args)
    if args.input is not None and len(providers) > 1:
        raise ValueError(
            "--input can only be used with --provider mistral or --provider openai"
        )
    if args.input is not None and COMBINED_PROVIDER in providers:
        raise ValueError(
            "--input cannot be used with --provider combined. "
            "Combined mode builds its input from the OpenAI and Mistral CSVs."
        )
    if args.kg_output is not None and len(providers) > 1:
        raise ValueError("--kg-output can only be used with a single provider")

    for provider in providers:
        if provider == COMBINED_PROVIDER:
            prepare_combined_inputs(dry_run=args.dry_run)
        config = provider_config(provider)
        results_csv = args.input or config.results_csv
        merged_results_csv = results_csv_after_merge(results_csv, args)
        print(f"\n######## Provider: {provider} ########")
        for name, command in build_commands(config, args.input, args):
            run_step(name, command, dry_run=args.dry_run)
        if not args.skip_prepare_rml_sources:
            prepare_rml_sources(
                config=config,
                input_corpus=args.input_corpus,
                results_csv=merged_results_csv,
                infer_missing_spans=not args.no_infer_spans,
                max_rows=args.max_rows,
                dry_run=args.dry_run,
            )
        if not args.skip_create_kg:
            run_step(
                "Apply generated RML mapping to generated CSV sources",
                create_kg_command(config, args),
                dry_run=args.dry_run,
            )


if __name__ == "__main__":
    main()
