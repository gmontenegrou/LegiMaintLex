from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rdflib import Graph

REPO_ROOT = Path(__file__).resolve().parents[2]
INSTANCE_NS = "https://w3id.org/semleg/legimaintlex/resource"

COMBINED_PROVIDER = "combined"
COMBINED_EMBEDDING_PROVIDER = "openai"
PROVIDER_ORDER = ("mistral", "openai")


@dataclass(frozen=True)
class ProviderAssets:
    provider: str
    provider_label: str
    extraction_model: str
    results_csv: Path
    ontology_ttl: Path
    rml_mapping: Path


@dataclass(frozen=True)
class CombinedOutputs:
    results_csv: Path
    ontology_ttl: Path
    manifest_json: Path


PROVIDER_ASSETS = {
    "mistral": ProviderAssets(
        provider="mistral",
        provider_label="Mistral AI",
        extraction_model="mistral-large-latest",
        results_csv=Path(
            "exp/kg/mistral/"
            "full_constrained_extraction_by_mistral_mistral-large-latest_nrows_6370.csv"
        ),
        ontology_ttl=Path("exp/new_ontology/mistral/ontology_extended_mistral.ttl"),
        rml_mapping=Path("exp/kg/mistral/csv_to_rml_mapping_mistral_rules.ttl"),
    ),
    "openai": ProviderAssets(
        provider="openai",
        provider_label="OpenAI",
        extraction_model="gpt-4.1",
        results_csv=Path(
            "exp/kg/openai/full_constrained_extraction_by_openai_gpt-4.1_nrows_6370.csv"
        ),
        ontology_ttl=Path("exp/new_ontology/openai/ontology_extended_openai.ttl"),
        rml_mapping=Path("exp/kg/openai/csv_to_rml_mapping_openai_rules.ttl"),
    ),
}

COMBINED_ASSETS = ProviderAssets(
    provider=COMBINED_PROVIDER,
    provider_label="OpenAI + Mistral AI",
    extraction_model="gpt-4.1 + mistral-large-latest",
    results_csv=Path(
        "exp/kg/combined/"
        "full_constrained_extraction_by_combined_openai_mistral_nrows_6370.csv"
    ),
    ontology_ttl=Path("exp/new_ontology/combined/ontology_extended_combined.ttl"),
    rml_mapping=Path("exp/kg/combined/csv_to_rml_mapping_combined_rules.ttl"),
)

COMBINED_OUTPUTS = CombinedOutputs(
    results_csv=COMBINED_ASSETS.results_csv,
    ontology_ttl=COMBINED_ASSETS.ontology_ttl,
    manifest_json=Path("exp/kg/combined/combined_inputs_manifest.json"),
)

PROVENANCE_COLUMNS = [
    "extraction_provider",
    "extraction_provider_label",
    "extraction_model",
    "extraction_source_file",
    "extraction_provider_uri",
    "relation_extraction_run_uri",
    "relation_extraction_label",
    "mention_extraction_activity_uri",
]


def repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def safe_instance_id(*values: object) -> str:
    value = "_".join(str(item) for item in values if str(item).strip())
    value = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()
    return value or "unspecified"


def provider_assets(provider: str) -> ProviderAssets:
    provider = provider.strip().lower()
    if provider == COMBINED_PROVIDER:
        return COMBINED_ASSETS
    if provider in PROVIDER_ASSETS:
        return PROVIDER_ASSETS[provider]
    raise ValueError(f"Unsupported provider: {provider}")


def provider_choices(include_all: bool = False) -> list[str]:
    choices = [*PROVIDER_ORDER, COMBINED_PROVIDER]
    if include_all:
        choices.append("all")
    return choices


def extraction_provider_uri(provider: str) -> str:
    return f"{INSTANCE_NS}/{safe_instance_id(provider)}"


def relation_extraction_run_uri(provider: str, model: str) -> str:
    return f"{INSTANCE_NS}/relationExtractionRun_{safe_instance_id(provider, model)}"


def mention_extraction_activity_uri(provider: str, model: str) -> str:
    return f"{INSTANCE_NS}/mentionExtractionActivity_{safe_instance_id(provider, model)}"


def rml_provenance_fields(
    provider: str,
    model: str | None = None,
    source_file: Path | str | None = None,
) -> dict[str, str]:
    try:
        assets = provider_assets(provider)
        provider_label = assets.provider_label
        model = model or assets.extraction_model
    except ValueError:
        provider_label = provider
        model = model or "unspecified-model"

    return {
        "extraction_provider": provider,
        "extraction_provider_label": provider_label,
        "extraction_model": model,
        "extraction_source_file": str(source_file or ""),
        "extraction_provider_uri": extraction_provider_uri(provider),
        "relation_extraction_run_uri": relation_extraction_run_uri(provider, model),
        "relation_extraction_label": (
            f"Signature-driven relation extraction by {provider_label}"
        ),
        "mention_extraction_activity_uri": mention_extraction_activity_uri(
            provider, model
        ),
    }


def set_max_csv_field_size() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    absolute = repo_path(path)
    if not absolute.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    set_max_csv_field_size()
    with absolute.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def write_csv(path: Path, rows: list[dict[str, str]], columns: list[str]) -> None:
    absolute = repo_path(path)
    absolute.parent.mkdir(parents=True, exist_ok=True)
    extras = sorted({key for row in rows for key in row} - set(columns))
    with absolute.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns + extras)
        writer.writeheader()
        writer.writerows(rows)


def unique_columns(column_groups: list[list[str]]) -> list[str]:
    columns: list[str] = []
    for group in column_groups:
        for column in group:
            if column not in columns:
                columns.append(column)
    return columns


def interleave_rows(provider_rows: dict[str, list[dict[str, str]]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    max_len = max((len(items) for items in provider_rows.values()), default=0)
    for index in range(max_len):
        for provider in PROVIDER_ORDER:
            items = provider_rows.get(provider, [])
            if index < len(items):
                rows.append(items[index])
    return rows


def combine_extraction_results(
    output_csv: Path = COMBINED_OUTPUTS.results_csv,
) -> tuple[Path, int]:
    provider_rows: dict[str, list[dict[str, str]]] = {}
    column_groups: list[list[str]] = []

    for provider in PROVIDER_ORDER:
        assets = provider_assets(provider)
        rows, columns = read_csv(assets.results_csv)
        column_groups.append(columns)
        enriched_rows: list[dict[str, str]] = []
        provenance = rml_provenance_fields(
            provider=provider,
            model=assets.extraction_model,
            source_file=assets.results_csv,
        )
        for row in rows:
            enriched_rows.append({**row, **provenance})
        provider_rows[provider] = enriched_rows

    combined_rows = interleave_rows(provider_rows)
    columns = unique_columns(column_groups + [PROVENANCE_COLUMNS])
    write_csv(output_csv, combined_rows, columns)
    return output_csv, len(combined_rows)


def combine_ontologies(
    output_ttl: Path = COMBINED_OUTPUTS.ontology_ttl,
) -> tuple[Path, int]:
    graph = Graph()
    for provider in PROVIDER_ORDER:
        ontology_path = repo_path(provider_assets(provider).ontology_ttl)
        if not ontology_path.exists():
            raise FileNotFoundError(f"Ontology TTL not found: {ontology_path}")
        graph.parse(str(ontology_path), format="turtle")

    output_path = repo_path(output_ttl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    graph.serialize(destination=str(output_path), format="turtle")
    return output_ttl, len(graph)


def write_manifest(
    outputs: CombinedOutputs = COMBINED_OUTPUTS,
    row_count: int | None = None,
    ontology_triples: int | None = None,
) -> None:
    payload: dict[str, Any] = {
        "provider": COMBINED_PROVIDER,
        "providers": list(PROVIDER_ORDER),
        "outputs": {key: str(value) for key, value in asdict(outputs).items()},
        "inputs": {
            provider: {
                "results_csv": str(provider_assets(provider).results_csv),
                "ontology_ttl": str(provider_assets(provider).ontology_ttl),
                "extraction_model": provider_assets(provider).extraction_model,
            }
            for provider in PROVIDER_ORDER
        },
        "row_count": row_count,
        "ontology_triples": ontology_triples,
    }
    manifest_path = repo_path(outputs.manifest_json)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def prepare_combined_inputs(force: bool = True, dry_run: bool = False) -> CombinedOutputs:
    outputs = COMBINED_OUTPUTS
    row_count: int | None = None
    ontology_triples: int | None = None

    if dry_run:
        print("Combined inputs would be prepared:")
        print(f"  Results CSV: {outputs.results_csv}")
        print(f"  Ontology TTL: {outputs.ontology_ttl}")
        print(f"  Manifest: {outputs.manifest_json}")
        return outputs

    if force or not repo_path(outputs.results_csv).exists():
        _, row_count = combine_extraction_results(outputs.results_csv)
        print(f"Combined extraction rows: {row_count} -> {outputs.results_csv}")

    if force or not repo_path(outputs.ontology_ttl).exists():
        _, ontology_triples = combine_ontologies(outputs.ontology_ttl)
        print(f"Combined ontology triples: {ontology_triples} -> {outputs.ontology_ttl}")

    write_manifest(outputs, row_count=row_count, ontology_triples=ontology_triples)
    print(f"Combined manifest: {outputs.manifest_json}")
    return outputs


def replace_constant_iri_prefix(text: str, iri_prefix: str, reference: str) -> str:
    pattern = (
        rf"rr:constant\s+{re.escape(iri_prefix)}[A-Za-z0-9_]*\s*;\s*"
        r"rr:termType\s+rr:IRI"
    )
    replacement = f'rml:reference "{reference}" ;\n            rr:termType rr:IRI'
    return re.sub(pattern, replacement, text)


def replace_constant_iri(text: str, iri: str, reference: str) -> str:
    pattern = (
        rf"rr:constant\s+{re.escape(iri)}\s*;\s*"
        r"rr:termType\s+rr:IRI"
    )
    replacement = f'rml:reference "{reference}" ;\n            rr:termType rr:IRI'
    return re.sub(pattern, replacement, text)


def replace_literal_for_predicate(block: str, predicate: str, reference: str) -> str:
    pattern = (
        rf"(rr:predicate\s+{re.escape(predicate)}\s*;\s*"
        r"rr:objectMap\s*\[\s*)"
        r'rr:constant\s+"[^"]*"\s*;\s*rr:datatype\s+xsd:string'
    )
    replacement = rf'\1rml:reference "{reference}" ;' "\n            rr:datatype xsd:string"
    return re.sub(pattern, replacement, block)


def patch_block(text: str, block_name: str, patcher) -> str:
    pattern = rf"({re.escape(block_name)}[\s\S]*?)(?=\n\nmap:|\Z)"

    def replace(match: re.Match[str]) -> str:
        return patcher(match.group(1))

    return re.sub(pattern, replace, text, count=1)


def patch_combined_mapping_text(text: str) -> str:
    text = text.replace(
        'rr:template "target-{source_row_id}-{triplet_index}-{mention_role}"',
        'rr:template "target-{extraction_provider}-{source_row_id}-{triplet_index}-{mention_role}"',
    )
    text = text.replace(
        'rr:template "selector-{source_row_id}-{triplet_index}-{mention_role}"',
        'rr:template "selector-{extraction_provider}-{source_row_id}-{triplet_index}-{mention_role}"',
    )
    text = replace_constant_iri_prefix(
        text,
        "inst:relationExtractionRun",
        "relation_extraction_run_uri",
    )
    text = replace_constant_iri(
        text,
        "inst:mentionExtractionActivity",
        "mention_extraction_activity_uri",
    )
    text = replace_constant_iri(text, "inst:combined", "extraction_provider_uri")

    text = patch_block(
        text,
        "map:RelationExtractionActivityTriplesMap",
        lambda block: replace_literal_for_predicate(
            replace_literal_for_predicate(
                block,
                "rdfs:label",
                "relation_extraction_label",
            ),
            "semleg:usesModel",
            "extraction_model",
        ),
    )
    text = patch_block(
        text,
        "map:ExtractionProviderAgentTriplesMap",
        lambda block: replace_literal_for_predicate(
            block,
            "rdfs:label",
            "extraction_provider_label",
        ),
    )
    return text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build combined OpenAI+Mistral inputs for KG creation."
    )
    parser.add_argument("--no-force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare_combined_inputs(force=not args.no_force, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
