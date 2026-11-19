"""Create document-level curation decisions from the current corpus only.

The output is a JSON hand-off for the later RDF/RML phase. It intentionally
does not read the unified Legifrance dataset and does not modify SKOS labels.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from datetime import date
from pathlib import Path
from urllib.parse import quote

INSTANCE_NS = "https://w3id.org/semleg/legimaintlex/resource"
VALIDATED = "inst:ValidatedForMaintenance"
NOT_SELECTED = "inst:NotSelectedForMaintenance"
CURATION_ACTIVITY_START = "2026-10-05T00:00:00Z"
CURATION_ACTIVITY_END = "2026-11-09T23:59:59Z"


def clean(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def key(value: object) -> str:
    return clean(value).casefold()


def first_non_empty(*values: object) -> str:
    for value in values:
        text = clean(value)
        if text:
            return text
    return ""


def source_key(value: object) -> str:
    text = unicodedata.normalize("NFKD", clean(value))
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-").lower()
    return text or "unnamed"


def document_key(title: str) -> str:
    normalized = clean(title)
    slug = source_key(normalized)[:80].strip("-") or "unnamed"
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:12]
    return f"{slug}-{digest}"


def document_uri(title: str) -> str:
    return f"{INSTANCE_NS}/document/{quote(document_key(title), safe='._-~')}"


def read_csv(
    path: Path, delimiter: str, encodings: tuple[str, ...]
) -> list[dict[str, str]]:
    for encoding in encodings:
        try:
            with path.open(encoding=encoding, newline="") as handle:
                return list(csv.DictReader(handle, delimiter=delimiter))
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("unknown", b"", 0, 1, f"Could not decode {path}")


def parse_literal(value: object) -> object:
    text = clean(value)
    if not text:
        return None
    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(text)
        except (ValueError, SyntaxError, json.JSONDecodeError):
            continue
    return None


def parse_candidates(row: dict[str, str]) -> list[dict[str, object]]:
    parsed = parse_literal(row.get("candidates"))
    if not isinstance(parsed, list):
        return []
    result = []
    for index, candidate in enumerate(parsed, start=1):
        if not isinstance(candidate, dict):
            continue
        title = first_non_empty(
            candidate.get("document_title"),
            candidate.get("title"),
        )
        if title:
            result.append(
                {
                    "candidate_rank": candidate.get("candidate", index),
                    "title": title,
                    "topics": candidate.get("topics") or [],
                }
            )
    return result


def load_seed_titles(path: Path) -> dict[str, list[str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        clean(reference): sorted({clean(title) for title in titles if clean(title)})
        for reference, titles in data.items()
    }


def corpus_index(path: Path) -> dict[str, list[dict[str, object]]]:
    rows = read_csv(path, ",", ("utf-8-sig", "cp1252"))
    index: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row_number, row in enumerate(rows, start=2):
        if key(row.get("type")) not in {"arrete", "arrêté"}:
            continue
        title = clean(row.get("title"))
        if not title:
            continue
        index[key(title)].append(
            {
                "row_number": row_number,
                "source_id": clean(row.get("id")),
                "corpus_id": clean(row.get("corpus_id")),
                "title": title,
                "article_number": clean(row.get("num")),
                "date": clean(row.get("date")),
                "type": clean(row.get("type")),
                "topics": clean(row.get("domain")),
            }
        )
    return dict(index)


def manual_evidence(
    path: Path,
) -> tuple[dict[str, list[dict[str, object]]], dict[str, list[dict[str, object]]]]:
    rows = read_csv(path, ";", ("utf-8-sig", "cp1252"))
    selected: dict[str, list[dict[str, object]]] = defaultdict(list)
    candidates: dict[str, list[dict[str, object]]] = defaultdict(list)

    for row_number, row in enumerate(rows, start=2):
        selected_title = first_non_empty(row.get("title"), row.get("new"))
        evidence = {
            "source_row": row_number,
            "date": clean(row.get("date")),
            "topic": clean(row.get("topic")),
            "validated_input": clean(row.get("validated")),
            "confidence": clean(row.get("confidence")),
            "decision": clean(row.get("decision")),
            "comment": clean(row.get("comment")),
        }
        if selected_title:
            selected[key(selected_title)].append(
                {**evidence, "selected_title": selected_title}
            )

        for candidate in parse_candidates(row):
            candidates[key(candidate["title"])].append(
                {
                    **evidence,
                    "candidate_rank": candidate["candidate_rank"],
                    "candidate_title": candidate["title"],
                    "candidate_topics": candidate["topics"],
                }
            )
    return dict(selected), dict(candidates)


def build_decisions(
    seed: dict[str, list[str]],
    corpus: dict[str, list[dict[str, object]]],
    selected_evidence: dict[str, list[dict[str, object]]],
    candidate_evidence: dict[str, list[dict[str, object]]],
    activity_id: str,
    source_id: str,
) -> list[dict[str, object]]:
    selected_titles = {key(title) for titles in seed.values() for title in titles}
    selected_display = {
        key(title): title for titles in seed.values() for title in titles
    }
    candidate_titles = set(candidate_evidence) - selected_titles
    decisions: list[dict[str, object]] = []

    def add_decision(
        title_key: str, status: str, evidence: list[dict[str, object]]
    ) -> None:
        rows = corpus.get(title_key, [])
        if not rows:
            return
        title = rows[0]["title"]
        target_uri = document_uri(str(title))
        decisions.append(
            {
                "decision_id": f"curation-{document_key(str(title))}",
                "status": status,
                "target": {
                    "type": "eli:LegalResource",
                    "uri": target_uri,
                    "document_key": document_key(str(title)),
                    "title": title,
                },
                "annotation": {
                    "type": "oa:Annotation",
                    "oa:hasTarget": target_uri,
                    "oa:hasBody": status,
                    "oa:motivatedBy": "oa:classifying",
                    "prov:wasGeneratedBy": activity_id,
                    "prov:hadPrimarySource": source_id,
                },
                "corpus_matches": rows,
                "manual_evidence": evidence,
            }
        )

    for title_key in sorted(selected_titles & set(corpus)):
        add_decision(
            title_key,
            VALIDATED,
            selected_evidence.get(title_key, []),
        )
    for title_key in sorted(candidate_titles & set(corpus)):
        add_decision(
            title_key,
            NOT_SELECTED,
            candidate_evidence.get(title_key, []),
        )
    return sorted(decisions, key=lambda item: item["target"]["title"].casefold())


def create_output(args: argparse.Namespace) -> dict[str, object]:
    seed = load_seed_titles(Path(args.seed))
    corpus = corpus_index(Path(args.corpus))
    selected_evidence, candidate_evidence = manual_evidence(Path(args.decisions))
    activity_id = args.activity
    source_id = "inst:Dis_arretes_full_corrected"
    decisions = build_decisions(
        seed,
        corpus,
        selected_evidence,
        candidate_evidence,
        activity_id,
        source_id,
    )
    validated = sum(item["status"] == VALIDATED for item in decisions)
    not_selected = sum(item["status"] == NOT_SELECTED for item in decisions)
    output = {
        "schema_version": "1.0",
        "generated_at": date.today().isoformat(),
        "scope": {
            "corpus": str(Path(args.corpus)),
            "manual_decisions": str(Path(args.decisions)),
            "seed": str(Path(args.seed)),
        },
        "activity": {
            "id": activity_id,
            "type": "prov:Activity",
            "rdfs:label": "Maintenance corpus curation",
            "rdfs:comment": (
                "Activity that classifies legal documents according to their "
                "selection for the maintenance corpus."
            ),
            "prov:used": [source_id],
            "prov:startedAtTime": CURATION_ACTIVITY_START,
            "prov:endedAtTime": CURATION_ACTIVITY_END,
        },
        "status_concepts": {
            VALIDATED: {
                "type": "skos:Concept",
                "skos:prefLabel": "Validated for maintenance",
                "skos:definition": (
                    "The legal document has been reviewed and validated as "
                    "relevant to the maintenance corpus."
                ),
            },
            NOT_SELECTED: {
                "type": "skos:Concept",
                "skos:prefLabel": "Not selected for maintenance",
                "skos:definition": (
                    "The document was considered as a candidate and excluded "
                    "after the relevant arrêté for the maintenance corpus had "
                    "been selected."
                ),
            },
        },
        "counts": {
            "corpus_arrete_titles": len(corpus),
            "selected_seed_titles": len(seed),
            "validated_annotations": validated,
            "not_selected_annotations": not_selected,
            "total_annotations": len(decisions),
        },
        "decisions": decisions,
    }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create corpus-only document curation decisions."
    )
    parser.add_argument("--corpus", default="src/data/maintreg_database_clean.csv")
    parser.add_argument(
        "--decisions",
        default="src/data/Dis_arretes_full_corrected.csv",
    )
    parser.add_argument(
        "--seed",
        default="src/data/seed_arrete_set_by_document.json",
    )
    parser.add_argument(
        "--activity",
        default="inst:maintenanceCurationActivity-Dis_arretes_full_corrected",
    )
    parser.add_argument(
        "--output",
        default="src/data/arrete_curation_decisions.json",
    )
    args = parser.parse_args()
    output = create_output(args)
    destination = Path(args.output)
    destination.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output["counts"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
