import argparse
import ast
import json
import re
import sys
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import pandas as pd


def _configure_utf8_stdio() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def safe_print(text: str) -> None:
    try:
        print(text)
    except UnicodeEncodeError:
        fallback = text.encode("utf-8", errors="replace").decode("utf-8")
        print(fallback)


def parse_triplets(value: Any) -> list[dict[str, Any]]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []

    if isinstance(value, list):
        return [t for t in value if isinstance(t, dict)]

    text = str(value).strip()
    if not text:
        return []

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [t for t in parsed if isinstance(t, dict)]
        if isinstance(parsed, dict) and isinstance(parsed.get("triplets"), list):
            return [t for t in parsed["triplets"] if isinstance(t, dict)]
    except Exception:
        pass

    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [t for t in parsed if isinstance(t, dict)]
        if isinstance(parsed, dict) and isinstance(parsed.get("triplets"), list):
            return [t for t in parsed["triplets"] if isinstance(t, dict)]
    except Exception:
        pass

    return []


def find_span(text: str, mention: str) -> tuple[int, int] | None:
    if not mention:
        return None

    # 1) Exact match
    start = text.find(mention)
    if start != -1:
        return start, start + len(mention)

    # 2) Case-insensitive match
    pattern = re.compile(re.escape(mention), flags=re.IGNORECASE)
    m = pattern.search(text)
    if m:
        return m.start(), m.end()

    # 3) Punctuation-tolerant match at mention boundaries
    # Useful when mention omits trailing punctuation present in content.
    mention_stripped = mention.strip().strip(".,;:!?\"'()[]{}")
    if mention_stripped:
        pattern = re.compile(
            rf"(?<!\w){re.escape(mention_stripped)}(?:[\s\.,;:!?]|$)",
            flags=re.IGNORECASE,
        )
        m = pattern.search(text)
        if m:
            end = m.end()
            # Exclude trailing separator/punctuation if matched by the suffix group.
            while end > m.start() and text[end - 1] in " \t\r\n.,;:!?":
                end -= 1
            return m.start(), end

    return None


def _normalize_char(ch: str) -> str:
    decomposed = unicodedata.normalize("NFKD", ch)
    base = "".join(c for c in decomposed if not unicodedata.combining(c)).lower()
    if not base:
        return ""
    if base.isalnum():
        return base
    return " "


def _normalize_with_map(text: str) -> tuple[str, list[int]]:
    chars: list[str] = []
    idx_map: list[int] = []
    for i, ch in enumerate(text):
        norm = _normalize_char(ch)
        if not norm:
            continue
        for c in norm:
            chars.append(c)
            idx_map.append(i)
    normalized = re.sub(r"\s+", " ", "".join(chars)).strip()

    # Rebuild map after whitespace collapsing by walking original normalized chars.
    # We keep first matching index for each output char.
    out_map: list[int] = []
    if normalized:
        raw = "".join(chars)
        raw_i = 0
        for out_ch in normalized:
            while raw_i < len(raw) and raw[raw_i] != out_ch:
                raw_i += 1
            if raw_i < len(raw):
                out_map.append(idx_map[raw_i])
                raw_i += 1
            else:
                out_map.append(idx_map[-1] if idx_map else 0)
    return normalized, out_map


def find_approx_span(text: str, mention: str) -> tuple[int, int] | None:
    if not mention:
        return None

    text_norm, text_map = _normalize_with_map(text)
    mention_norm, _ = _normalize_with_map(mention)
    if not text_norm or not mention_norm:
        return None

    # 1) Normalized exact
    start_n = text_norm.find(mention_norm)
    if start_n != -1:
        end_n = start_n + len(mention_norm) - 1
        start = text_map[start_n]
        end = text_map[end_n] + 1
        return start, end

    # 2) Fuzzy fallback by longest common match
    match = SequenceMatcher(
        None, mention_norm, text_norm, autojunk=False
    ).find_longest_match(0, len(mention_norm), 0, len(text_norm))
    if match.size <= 0:
        return None

    ratio = match.size / max(1, len(mention_norm))
    if ratio < 0.55:
        return None

    start_n = match.b
    end_n = match.b + match.size - 1
    if start_n < 0 or end_n >= len(text_map):
        return None

    start = text_map[start_n]
    end = text_map[end_n] + 1
    return start, end


def _normalize_token(token: str) -> str:
    decomposed = unicodedata.normalize("NFKD", token)
    base = "".join(c for c in decomposed if not unicodedata.combining(c)).lower()
    return "".join(ch for ch in base if ch.isalnum())


def _tokenize_with_spans(text: str) -> list[tuple[str, int, int]]:
    tokens: list[tuple[str, int, int]] = []
    for match in re.finditer(r"\w+", text, flags=re.UNICODE):
        token = _normalize_token(match.group(0))
        if token:
            tokens.append((token, match.start(), match.end()))
    return tokens


def find_discontinuous_spans(text: str, mention: str) -> list[tuple[int, int]] | None:
    if not mention:
        return None

    text_tokens = _tokenize_with_spans(text)
    mention_tokens = _tokenize_with_spans(mention)
    if not text_tokens or not mention_tokens:
        return None

    def _tokens_match(a: str, b: str) -> bool:
        if a == b:
            return True
        if len(a) >= 5 and len(b) >= 5:
            return SequenceMatcher(None, a, b, autojunk=False).ratio() >= 0.88
        return False

    matched_idx: list[int] = []
    j = 0
    for m_tok, _, _ in mention_tokens:
        found = False
        while j < len(text_tokens):
            t_tok, _, _ = text_tokens[j]
            if _tokens_match(m_tok, t_tok):
                matched_idx.append(j)
                j += 1
                found = True
                break
            j += 1
        if not found:
            return None

    if not matched_idx:
        return None

    spans: list[tuple[int, int]] = []
    run_start = matched_idx[0]
    prev = matched_idx[0]
    for idx in matched_idx[1:]:
        if idx == prev + 1:
            prev = idx
            continue
        spans.append((text_tokens[run_start][1], text_tokens[prev][2]))
        run_start = idx
        prev = idx
    spans.append((text_tokens[run_start][1], text_tokens[prev][2]))
    return spans


def process_file(input_csv: Path, output_csv: Path) -> None:
    df = pd.read_csv(input_csv)

    if "content" not in df.columns or "legal_triplets" not in df.columns:
        raise ValueError(
            "Input CSV must contain columns: 'content' and 'legal_triplets'."
        )

    enriched_rows = []
    missing_count = 0

    for row_idx, row in df.iterrows():
        content = "" if pd.isna(row["content"]) else str(row["content"])
        triplets = parse_triplets(row["legal_triplets"])
        triplets_with_spans = []

        for triplet_idx, triplet in enumerate(triplets):
            head = str(triplet.get("head", "") or "")
            tail = str(triplet.get("tail", "") or "")

            head_span = find_span(content, head)
            tail_span = find_span(content, tail)
            head_spans = [head_span] if head_span else []
            tail_spans = [tail_span] if tail_span else []
            head_mode = "exact" if head_span else None
            tail_mode = "exact" if tail_span else None

            if head and head_span is None:
                head_span = find_approx_span(content, head)
                if head_span is not None:
                    head_mode = "approx"
                    head_spans = [head_span]
            if tail and tail_span is None:
                tail_span = find_approx_span(content, tail)
                if tail_span is not None:
                    tail_mode = "approx"
                    tail_spans = [tail_span]

            if head and head_span is None:
                head_discontinuous = find_discontinuous_spans(content, head)
                if head_discontinuous:
                    head_mode = "discontinuous"
                    head_spans = head_discontinuous
                    head_span = (head_discontinuous[0][0], head_discontinuous[-1][1])
            if tail and tail_span is None:
                tail_discontinuous = find_discontinuous_spans(content, tail)
                if tail_discontinuous:
                    tail_mode = "discontinuous"
                    tail_spans = tail_discontinuous
                    tail_span = (tail_discontinuous[0][0], tail_discontinuous[-1][1])

            if head and head_span is None:
                # safe_print(f"[HEAD_NO_MATCH]\thead={head}\tcontent={content}")
                missing_count += 1
            if tail and tail_span is None:
                # safe_print(f"[TAIL_NO_MATCH]\ttail={tail}\tcontent={content}")
                missing_count += 1

            triplet_out = dict(triplet)
            triplet_out["head_start"] = head_span[0] if head_span else None
            triplet_out["head_end"] = head_span[1] if head_span else None
            triplet_out["tail_start"] = tail_span[0] if tail_span else None
            triplet_out["tail_end"] = tail_span[1] if tail_span else None
            triplet_out["head_spans"] = [
                {"start": s, "end": e}
                for s, e in head_spans
                if s is not None and e is not None
            ]
            triplet_out["tail_spans"] = [
                {"start": s, "end": e}
                for s, e in tail_spans
                if s is not None and e is not None
            ]
            triplet_out["head_span_mode"] = head_mode
            triplet_out["tail_span_mode"] = tail_mode
            triplet_out["triplet_index"] = triplet_idx
            triplets_with_spans.append(triplet_out)

        enriched_rows.append(json.dumps(triplets_with_spans, ensure_ascii=False))

    out_df = df.copy()
    out_df["legal_triplets_with_spans"] = enriched_rows
    out_df.to_csv(output_csv, index=False)

    safe_print(f"Saved: {output_csv}")
    safe_print(f"Missing matches: {missing_count}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Detect head/tail spans (start/end) for each triplet in legal_triplets."
        )
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        required=True,
        help="Path to input CSV with columns content and legal_triplets.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Output CSV path. Default: <input_stem>_with_spans.csv",
    )
    return parser.parse_args()


def main() -> None:
    _configure_utf8_stdio()
    args = parse_args()
    input_csv: Path = args.input_csv

    if args.output_csv is None:
        output_csv = input_csv.with_name(f"{input_csv.stem}_with_spans.csv")
    else:
        output_csv = args.output_csv

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    process_file(input_csv=input_csv, output_csv=output_csv)


if __name__ == "__main__":
    main()
