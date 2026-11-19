import argparse
import ast
import html
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd
from rdflib import BNode, Graph, Literal, Namespace, URIRef
from rdflib.namespace import RDF, RDFS

PALETTE = [
    "#e3f2fd",
    "#fce4ec",
    "#e8f5e9",
    "#fff8e1",
    "#ede7f6",
    "#f3e5f5",
    "#e0f7fa",
    "#fbe9e7",
    "#f1f8e9",
    "#e8eaf6",
]

INSTANCE_NS = Namespace("https://w3id.org/semleg/legimaintlex/resource/")
SEMLEG_NS = Namespace("https://w3id.org/semleg#")
OA = Namespace("http://www.w3.org/ns/oa#")
PROV = Namespace("http://www.w3.org/ns/prov#")
ELI = Namespace("http://data.europa.eu/eli/ontology#")
DCTERMS = Namespace("http://purl.org/dc/terms/")
XSD = Namespace("http://www.w3.org/2001/XMLSchema#")


def parse_triplets(value: Any) -> list[dict[str, Any]]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []

    if isinstance(value, list):
        return [v for v in value if isinstance(v, dict)]

    text = str(value).strip()
    if not text:
        return []

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [v for v in parsed if isinstance(v, dict)]
        if isinstance(parsed, dict) and isinstance(parsed.get("triplets"), list):
            return [v for v in parsed["triplets"] if isinstance(v, dict)]
    except Exception:
        pass

    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [v for v in parsed if isinstance(v, dict)]
        if isinstance(parsed, dict) and isinstance(parsed.get("triplets"), list):
            return [v for v in parsed["triplets"] if isinstance(v, dict)]
    except Exception:
        pass

    return []


def sanitize_css_class(label: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_-]", "_", (label or "Entity").strip())
    clean = re.sub(r"_+", "_", clean).strip("_")
    if not clean:
        clean = "Entity"
    if clean[0].isdigit():
        clean = f"X_{clean}"
    return clean


def slugify(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value or "")
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^A-Za-z0-9]+", "-", ascii_value.lower()).strip("-")
    return slug or "document"


def parse_provenance(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return {}
    text = str(value).strip()
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
    return {}


def get_segments(triplet: dict[str, Any], side: str) -> list[tuple[int, int]]:
    key = f"{side}_spans"
    spans = triplet.get(key)
    if isinstance(spans, list):
        out = []
        for item in spans:
            if not isinstance(item, dict):
                continue
            start = item.get("start")
            end = item.get("end")
            if isinstance(start, int) and isinstance(end, int) and end > start:
                out.append((start, end))
        if out:
            return out

    start = triplet.get(f"{side}_start")
    end = triplet.get(f"{side}_end")
    if isinstance(start, int) and isinstance(end, int) and end > start:
        return [(start, end)]

    return []


def clip_segments(
    segments: list[tuple[int, int]], text_len: int
) -> list[tuple[int, int]]:
    clipped = []
    for s, e in segments:
        s2 = max(0, min(s, text_len))
        e2 = max(0, min(e, text_len))
        if e2 > s2:
            clipped.append((s2, e2))
    return clipped


def extract_text(content: str, segments: list[tuple[int, int]]) -> str:
    return " [...] ".join(content[s:e] for s, e in segments).strip()


def infer_segments(content: str, mention_text: str) -> list[tuple[int, int]]:
    """Find a mention in the article when the source triplet has no spans."""
    mention_text = mention_text.strip()
    if not mention_text:
        return []

    start = content.find(mention_text)
    if start >= 0:
        return [(start, start + len(mention_text))]

    match = re.search(re.escape(mention_text), content, flags=re.IGNORECASE)
    if match:
        return [(match.start(), match.end())]

    # French determiner and simple singular/plural variation, for example
    # "les lieux" vs "des lieux" and "le propriétaire" vs "les propriétaires".
    determinants = (
        "le",
        "la",
        "les",
        "l'",
        "un",
        "une",
        "des",
        "du",
        "de",
        "au",
        "aux",
        "ce",
        "cet",
        "cette",
        "ces",
    )
    first_word = re.match(
        r"(?i)(le|la|les|l'|un|une|des|du|de|au|aux|ce|cet|cette|ces)\s+",
        mention_text,
    )
    if first_word:
        remainder_words = mention_text[first_word.end() :].split()
        if remainder_words:
            remainder_pattern = r"\s+".join(
                word if word.endswith("s") else f"{word}s?"
                for word in (re.escape(item) for item in remainder_words)
            )
            determinant_pattern = "|".join(re.escape(item) for item in determinants)
            match = re.search(
                rf"\b(?:{determinant_pattern})\s+{remainder_pattern}",
                content,
                flags=re.IGNORECASE,
            )
            if match:
                return [(match.start(), match.end())]
    return []


def load_rdf_selectors(
    rdf_path: Path | None,
) -> dict[tuple[str, int, str], list[tuple[int, int]]]:
    """Load existing OA selectors keyed by article id, triplet index and role."""
    if rdf_path is None:
        return {}
    graph = Graph()
    graph.parse(rdf_path)
    selectors = {}
    pattern = re.compile(r"/article/[^/]+/triple/(\d+)/mention/(head|tail)$")
    for annotation in graph.subjects(RDF.type, OA.Annotation):
        match = pattern.search(str(annotation))
        if not match:
            continue
        article_match = re.search(
            r"/extraction/[^/]+/article/([^/]+)/triple/", str(annotation)
        )
        if not article_match:
            continue
        target = next(graph.objects(annotation, OA.hasTarget), None)
        selector = next(graph.objects(target, OA.hasSelector), None)
        if selector is None:
            continue
        start = next(graph.objects(selector, OA.start), None)
        end = next(graph.objects(selector, OA.end), None)
        if start is not None and end is not None:
            selectors[(article_match.group(1), int(match.group(1)), match.group(2))] = [
                (int(start), int(end))
            ]
    return selectors


def format_offsets(segments: list[tuple[int, int]]) -> str:
    return "; ".join(f"{s}-{e}" for s, e in segments)


def choose_triplets_column(df: pd.DataFrame, preferred: str | None) -> str:
    if preferred and preferred in df.columns:
        return preferred
    if "legal_triplets_with_spans" in df.columns:
        return "legal_triplets_with_spans"
    if "legal_triplets" in df.columns:
        return "legal_triplets"
    raise ValueError(
        "No triplets column found. Expected 'legal_triplets_with_spans' or 'legal_triplets'."
    )


def render_content_with_annotations(
    content: str, mentions: list[dict[str, Any]]
) -> str:
    n = len(content)
    boundaries = {0, n}
    for m in mentions:
        for s, e in m["segments"]:
            boundaries.add(s)
            boundaries.add(e)

    points = sorted(boundaries)
    out: list[str] = []

    for i in range(len(points) - 1):
        s = points[i]
        e = points[i + 1]
        if e <= s:
            continue

        chunk = html.escape(content[s:e])
        active = [
            m for m in mentions if any(ms <= s and me >= e for ms, me in m["segments"])
        ]

        if not active:
            out.append(chunk)
            continue

        active = sorted(active, key=lambda x: (x["label"], x["id"]))
        wrapped = chunk
        for m in active:
            cls = sanitize_css_class(m["label"])
            title = html.escape(f"{m['id']} | {m['label']} | {m['offsets']}")
            wrapped = f"<span class=\"ann ann-{cls}\" data-mid=\"{m['id']}\" title=\"{title}\">{wrapped}</span>"
        out.append(wrapped)

    return "".join(out)


def build_doc_html(
    doc_title: str,
    content: str,
    mentions: list[dict[str, Any]],
    relations: list[dict[str, Any]],
    unmatched: list[str],
    color_by_label: dict[str, str],
) -> str:
    legend_items = []
    for label in sorted(color_by_label):
        cls = sanitize_css_class(label)
        legend_items.append(
            f'<span class="legend-item ann ann-{cls}">{html.escape(label)}</span>'
        )

    rel_rows = []
    for r in relations:
        rel_rows.append(
            "<tr>"
            f"<td>{html.escape(r['relation'])}</td>"
            f"<td>{html.escape(r['head'])}</td>"
            f"<td>{html.escape(r['tail'])}</td>"
            f"<td>{html.escape(r['head_offsets'])}</td>"
            f"<td>{html.escape(r['tail_offsets'])}</td>"
            "</tr>"
        )

    unmatched_block = ""
    if unmatched:
        items = "".join(f"<li>{html.escape(u)}</li>" for u in unmatched)
        unmatched_block = f"<h3>Unmatched Mentions</h3><ul>{items}</ul>"

    dynamic_css = []
    for label, color in color_by_label.items():
        cls = sanitize_css_class(label)
        dynamic_css.append(
            f".ann-{cls} {{ background: {color}; border-bottom: 2px solid #555; }}"
        )

    content_html = render_content_with_annotations(content, mentions)

    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>{html.escape(doc_title)}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 20px; color: #1f2937; }}
    h1 {{ margin: 0 0 8px; }}
    .meta {{ color: #6b7280; margin-bottom: 12px; }}
    .legend {{ margin: 14px 0; display: flex; flex-wrap: wrap; gap: 8px; }}
    .legend-item {{ padding: 2px 8px; border-radius: 6px; font-size: 13px; }}
    .content {{ white-space: pre-wrap; line-height: 1.5; border: 1px solid #e5e7eb; padding: 12px; border-radius: 8px; }}
    .ann {{ border-radius: 4px; padding: 0 1px; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 16px; font-size: 14px; }}
    th, td {{ border: 1px solid #e5e7eb; padding: 6px; vertical-align: top; }}
    th {{ background: #f9fafb; text-align: left; }}
    {''.join(dynamic_css)}
  </style>
</head>
<body>
  <h1>{html.escape(doc_title)}</h1>
  <div class=\"meta\">Mentions: {len(mentions)} | Relations: {len(relations)}</div>
  <div class=\"legend\">{''.join(legend_items)}</div>
  <div class=\"content\">{content_html}</div>
  <h3>Relations</h3>
  <table>
    <thead>
      <tr><th>Relation</th><th>Head</th><th>Tail</th><th>Head Offsets</th><th>Tail Offsets</th></tr>
    </thead>
    <tbody>
      {''.join(rel_rows) if rel_rows else '<tr><td colspan="5">No relations with spans</td></tr>'}
    </tbody>
  </table>
  {unmatched_block}
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export CSV triplets to HTML visualization with colored spans and relations."
    )
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--triplets-column", default=None)
    parser.add_argument("--content-column", default="content")
    parser.add_argument("--doc-id-column", default="index")
    parser.add_argument(
        "--no-html",
        action="store_true",
        help="Export only the RDF and text files; do not create HTML files.",
    )
    parser.add_argument(
        "--eli-resources",
        action="store_true",
        help="Export ELI document/article resources with content resources in the RDF.",
    )
    parser.add_argument(
        "--nrows",
        "--max-docs",
        dest="nrows",
        type=int,
        default=None,
        help="Number of articles to export (default: all).",
    )
    parser.add_argument(
        "--rdf-file",
        type=Path,
        default=Path("mentions.ttl"),
        help="RDF/Turtle filename inside output-dir (default: mentions.ttl).",
    )
    parser.add_argument(
        "--selectors-rdf",
        type=Path,
        default=None,
        help="Existing RDF containing oa:TextPositionSelector offsets.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input_csv)

    if args.content_column not in df.columns:
        raise ValueError(f"Missing content column: {args.content_column}")

    triplets_col = choose_triplets_column(df, args.triplets_column)
    selectors = load_rdf_selectors(args.selectors_rdf)

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.nrows is not None and args.nrows < 0:
        raise ValueError("--nrows must be greater than or equal to 0")
    rows = df if args.nrows is None else df.head(args.nrows)

    rdf = Graph()
    rdf.bind("semleg", SEMLEG_NS)
    rdf.bind("inst", INSTANCE_NS)
    rdf.bind("rdfs", RDFS)
    rdf.bind("oa", OA)
    rdf.bind("prov", PROV)
    rdf.bind("eli", ELI)
    rdf.bind("dcterms", DCTERMS)
    rdf.bind("xsd", XSD)
    rdf.bind("semleg", SEMLEG_NS)

    total_docs = 0
    total_relations = 0
    total_unmatched = 0

    index_lines = []
    if not args.no_html:
        index_lines = [
            "<!doctype html><html><head><meta charset='utf-8'><title>Triplets Visualization</title></head><body>",
            "<h1>Triplets Visualization</h1><ul>",
        ]

    for i, row in rows.iterrows():
        doc_id = (
            str(row.get(args.doc_id_column, i))
            if pd.notna(row.get(args.doc_id_column, i))
            else str(i)
        )
        safe_doc_id = re.sub(r"[^A-Za-z0-9._-]", "_", doc_id) or str(i)

        content = (
            "" if pd.isna(row[args.content_column]) else str(row[args.content_column])
        )
        page_title = f"Document {doc_id}"
        provenance_data = parse_provenance(row.get("provenance", ""))
        provenance = str(row.get("provenance", ""))
        article_id = str(provenance_data.get("id_local") or doc_id)
        if article_id == doc_id:
            article_id_match = re.search(
                r"id_local['\"]?\s*:\s*['\"]([^'\"]+)", provenance
            )
            article_id = article_id_match.group(1) if article_id_match else doc_id
        triplets = parse_triplets(row[triplets_col])

        mention_map: dict[
            tuple[str, tuple[tuple[int, int], ...], str], dict[str, Any]
        ] = {}
        relations: list[dict[str, Any]] = []
        unmatched: list[str] = []
        m_counter = 1

        def get_or_create_mention(
            side: str, triplet: dict[str, Any], triplet_index: int
        ) -> dict[str, Any] | None:
            nonlocal m_counter
            label_key = f"{side}_type"
            text_key = side
            label = str(triplet.get(label_key, "") or side.title())
            mention_text = str(triplet.get(text_key, "") or "")
            segments = clip_segments(get_segments(triplet, side), len(content))
            if not segments:
                segments = clip_segments(
                    selectors.get((article_id, triplet_index, side), []), len(content)
                )
            if not segments:
                segments = infer_segments(content, mention_text)
            if not segments:
                unmatched.append(f"{side.upper()} '{mention_text}'")
                return None
            key = (label, tuple(segments), mention_text)
            if key in mention_map:
                return mention_map[key]
            mention = {
                "id": f"M{m_counter}",
                "label": label,
                "text": mention_text or extract_text(content, segments),
                "segments": segments,
                "offsets": format_offsets(segments),
            }
            mention_map[key] = mention
            m_counter += 1
            return mention

        for triplet_index, t in enumerate(triplets):
            source_triplet_index = (
                int(t.get("triplet_index", triplet_index))
                if str(t.get("triplet_index", "")).isdigit()
                else triplet_index
            )
            head = get_or_create_mention("head", t, source_triplet_index)
            tail = get_or_create_mention("tail", t, source_triplet_index)
            if not head or not tail:
                continue
            rel = str(t.get("relation", "relatedTo") or "relatedTo")
            relations.append(
                {
                    "relation": rel,
                    "head": head["text"],
                    "tail": tail["text"],
                    "head_offsets": head["offsets"],
                    "tail_offsets": tail["offsets"],
                }
            )

        mentions = list(mention_map.values())

        # Keep the RDF focused on mentions: no relation triples are exported.
        article_uri = URIRef(INSTANCE_NS[f"article/{safe_doc_id}"])
        if args.eli_resources:
            source_title = str(provenance_data.get("title") or "").strip()
            document_slug = slugify(source_title or article_id)
            document_uri = URIRef(INSTANCE_NS[f"document/{document_slug}"])
            article_resource_uri = URIRef(
                INSTANCE_NS[
                    f"document/{document_slug}/article/article-{slugify(article_id)}-{safe_doc_id}"
                ]
            )
            content_uri = URIRef(f"{article_resource_uri}/content")
            rdf.add((document_uri, RDF.type, ELI.LegalResource))
            if source_title:
                rdf.add(
                    (
                        document_uri,
                        DCTERMS.title,
                        Literal(source_title, datatype=XSD.string),
                    )
                )
            rdf.add((article_resource_uri, RDF.type, ELI.LegalResourceSubdivision))
            if source_title:
                rdf.add(
                    (
                        article_resource_uri,
                        DCTERMS.title,
                        Literal(source_title, datatype=XSD.string),
                    )
                )
            rdf.add(
                (
                    article_resource_uri,
                    ELI.id_local,
                    Literal(article_id, datatype=XSD.string),
                )
            )
            rdf.add((article_resource_uri, ELI.is_part_of, document_uri))
            rdf.add((article_resource_uri, SEMLEG_NS.hasContent, content_uri))
            rdf.add((content_uri, RDF.value, Literal(content, datatype=XSD.string)))
            article_uri = article_resource_uri
        activity_uri = URIRef(INSTANCE_NS["mentionExtractionActivity_viz_sample"])
        rdf.add((activity_uri, RDF.type, PROV.Activity))
        if not args.eli_resources:
            rdf.add((article_uri, RDFS.label, Literal(page_title)))
        for mention in mentions:
            annotation_uri = URIRef(f"{article_uri}/mention/{mention['id']}")
            entity_slug = re.sub(r"[^a-z0-9]+", "-", mention["text"].lower()).strip(
                "-"
            )[:120]
            body_uri = URIRef(
                INSTANCE_NS[f"entity/{mention['label'].lower()}/{entity_slug}"]
            )
            selector_nodes = []
            for start, end in mention["segments"]:
                selector = BNode()
                rdf.add((selector, RDF.type, OA.TextPositionSelector))
                rdf.add((selector, OA.start, Literal(start)))
                rdf.add((selector, OA.end, Literal(end)))
                selector_nodes.append(selector)
            rdf.add((annotation_uri, RDF.type, OA.Annotation))
            rdf.add((annotation_uri, OA.hasBody, body_uri))
            target_uri = BNode()
            rdf.add((annotation_uri, OA.hasTarget, target_uri))
            for selector in selector_nodes:
                rdf.add((target_uri, OA.hasSelector, selector))
            rdf.add((target_uri, OA.hasSource, article_uri))
            rdf.add((annotation_uri, PROV.wasGeneratedBy, activity_uri))

        txt_name = f"{safe_doc_id}.txt"
        if not args.eli_resources:
            (out_dir / txt_name).write_text(content, encoding="utf-8")

        if not args.no_html:
            labels = sorted({m["label"] for m in mentions})
            color_by_label = {
                label: PALETTE[idx % len(PALETTE)] for idx, label in enumerate(labels)
            }

            doc_html = build_doc_html(
                doc_title=page_title,
                content=content,
                mentions=mentions,
                relations=relations,
                unmatched=unmatched,
                color_by_label=color_by_label,
            )

            file_name = f"{safe_doc_id}.html"
            (out_dir / file_name).write_text(doc_html, encoding="utf-8")

            index_lines.append(
                f'<li><a href="{html.escape(file_name)}">{html.escape(page_title)}</a>'
                f" - mentions: {len(mentions)}, relations: {len(relations)}, unmatched: {len(unmatched)}</li>"
            )

        total_docs += 1
        total_relations += len(relations)
        total_unmatched += len(unmatched)

    if not args.no_html:
        index_lines.append("</ul></body></html>")
        (out_dir / "index.html").write_text("\n".join(index_lines), encoding="utf-8")
    rdf_path = out_dir / args.rdf_file
    rdf_path.parent.mkdir(parents=True, exist_ok=True)
    rdf.serialize(destination=rdf_path, format="turtle")

    print(f"Exported docs: {total_docs}")
    print(f"Relations rendered: {total_relations}")
    print(f"Unmatched mentions: {total_unmatched}")
    print(f"RDF mentions: {rdf_path}")
    if not args.no_html:
        print(f"Open: {out_dir / 'index.html'}")


if __name__ == "__main__":
    main()
