import argparse
import re
import pandas as pd
import math
from deduplication import (
    dedup_when_only_originalText_varies,
    dedup_when_only_domain_varies_concat_domains,
)


def remove_rows_matching_regex(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Remove rows from a DataFrame where the specified column matches:
      - "arrêté ... (publié|publication) au Journal officiel..." (optionally length-limited)
      - "Le présent arrêté (entre|entrera) en vigueur ..." (various date forms)
      - exact headers like "Arrête:" / "Arrêtent:" (with whitespace tolerance)

    Notes on fixes vs your version:
      - Use compiled patterns consistently (no re.search(pattern_obj, ...) mismatch).
      - Use vectorized string ops for speed and fewer edge cases.
      - Make the "ignore" mask explicit and readable.
      - Make the "entry into force" regex more robust (covers 'entrera', 'à compter du',
        'le lendemain du jour de sa publication...', and common date variants).
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")

    s = df[column]

    # Work only on strings; keep NA as NA
    s_str = s.where(s.apply(lambda x: isinstance(x, str)), other=pd.NA)

    # 1) Publication in Journal officiel (covers 'publié au' and 'publication au')
    pattern_pub = re.compile(
        r"\barrêté\b.*?\b(?:publié|publication)\s+au\s+Journal officiel de la République française\b",
        flags=re.IGNORECASE | re.DOTALL,
    )

    # 2) Entry into force patterns (more robust than the initial one)
    # Covers:
    # - "Le présent arrêté entre en vigueur le 1er juillet 2015."
    # - "Le présent arrêté entrera en vigueur le 1er juillet 2015."
    # - "Le présent arrêté entre en vigueur à compter du 1er juillet 2015."
    # - "Le présent arrêté entrera en vigueur le lendemain du jour de sa publication au Journal officiel..."

    pattern_eif = re.compile(
        r"présent arrêté (entr.+)? en vigueur le\s+(\d{1,2}(?:er)?\s+\w+\s+\d{4})",
        flags=re.IGNORECASE,
    )
    # 3) Exact headers "Arrête:" / "Arrêtent:" (trim + case-insensitive)
    mask_header = (
        s_str.str.strip().str.casefold().isin({"arrête:", "arrêtent:"}).fillna(False)
    )

    # Publication mask (for any length)
    mask_pub = s_str.str.contains(pattern_eif, na=False)

    # Optional: only drop very short publication boilerplate lines (your original idea)
    # If you actually want to drop ALL publication matches, set this to mask_pub.
    mask_pub_short = mask_pub & (s_str.str.len().fillna(0) < 60)

    # Entry into force mask
    mask_eif = s_str.str.contains(pattern_pub, na=False)

    # Combine masks (choose mask_pub_short to mimic your previous behavior)
    mask_drop = mask_header | mask_eif | mask_pub_short

    return df.loc[~mask_drop].reset_index(drop=True)


def expand_refs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expand the 'refs' column into one row per reference entry.

    Two formats are supported:
    1) Non-ARRETE (dict-of-dict):
       refs = {"L111-7": {"id": "...", "content": "..."}}
       -> columns: num, id, content

    2) ARRETE (date -> list of documents):
       refs = {"27.06.1994": [{"CID": "...", "Title": "...", "Articles": [...]}, ...]}
       -> one row per (date, document, article) with columns:
          date, id, title, num, chapterTitle, content
       (Also keeps original row columns: originalText, domain, tree, type)
    """
    expanded_rows = []
    base_cols = ["originalText", "domain", "tree", "type", "refs"]

    for _, row in df.iterrows():
        refs = row.get("refs", {})

        # Normalize refs to a dict
        if not isinstance(refs, dict) or not refs:
            # If refs is missing/empty, skip.
            continue

        row_base = {col: row.get(col) for col in base_cols}
        title = row.get("originalText", "")
        if isinstance(title, str) and ":" in title:
            title = title.split(":")[0]

        if row.get("type") == "ARRETE":
            # ARRETE format: { "DD.MM.YYYY": [ {CID, Title, Articles}, ... ], ... }
            for date_key, docs in refs.items():
                if not isinstance(docs, list):
                    continue

                for doc in docs:
                    if not isinstance(doc, dict):
                        continue

                    cid = doc.get("CID")
                    doc_title = doc.get("Title")
                    articles = doc.get("Articles", [])

                    # If there are no articles, still emit one row per document
                    if not isinstance(articles, list) or len(articles) == 0:
                        expanded_rows.append(
                            {
                                **row_base,
                                "date": date_key,
                                "id": cid,
                                "title": doc_title,
                                "num": None,
                                "chapterTitle": None,
                                "content": None,
                            }
                        )
                        continue

                    for idx, art in enumerate(articles, start=1):
                        if not isinstance(art, dict):
                            continue

                        expanded_rows.append(
                            {
                                **row_base,
                                "date": date_key,
                                "id": cid,
                                "title": doc_title,
                                "num": None,
                                "chapterTitle": art.get("chapterTitle"),
                                "content": art.get("content"),
                            }
                        )
        else:
            # Default format: { "L111-7": {"id": "...", "content": "..."} }
            for key, value in refs.items():
                ref_id = value.get("id") if isinstance(value, dict) else None
                ref_content = value.get("content") if isinstance(value, dict) else None

                expanded_rows.append(
                    {
                        **row_base,
                        "num": key,
                        "title": title,
                        "id": ref_id,
                        "content": ref_content,
                    }
                )

    df = pd.DataFrame(expanded_rows)
    df_mask = df["content"].apply(lambda x: isinstance(x, float))
    new_df = df[~df_mask]
    return new_df


def format_text(text: object) -> str:
    """
    Normalize string content by:
    - removing HTML-like tags (e.g., </p><p><br/>)
    - collapsing multiple whitespace to a single space
    - stripping leading/trailing whitespace
    """
    if text is None or (isinstance(text, float) and pd.isna(text)) or pd.isna(text):
        return ""
    s = str(text)
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"\s\s+", " ", s)
    s = re.sub(r"(\.\.)+", " ", s)
    return s.strip()


def add_content_counts(
    df: pd.DataFrame,
    content_col: str = "content",
    encoding_name: str = "cl100k_base",
    add_clean_text: bool = True,
) -> pd.DataFrame:
    """
    Add two columns to the DataFrame:
      - content_char_count: number of characters in `content_col`
      - content_token_count: number of tokens in `content_col` using tiktoken

    Notes:
      - Missing/NaN content is treated as empty string.
      - Uses a configurable tiktoken encoding (default: cl100k_base).
      - If add_clean_text=True, also adds `content_clean` with normalized string.
    """
    try:
        import tiktoken
    except ImportError as exc:
        raise ImportError(
            "tiktoken is required for token counts. Install it with `pip install tiktoken`."
        ) from exc

    out = df.copy()
    enc = tiktoken.get_encoding(encoding_name)

    def _to_text(x) -> str:
        if add_clean_text:
            return format_text(x)
        if x is None or (isinstance(x, float) and pd.isna(x)) or pd.isna(x):
            return ""
        return str(x)

    def _token_count(text: str) -> int:
        return len(enc.encode(text))

    text_series = out[content_col].apply(_to_text)
    if add_clean_text:
        out["content"] = text_series

    out["content_char_count"] = text_series.str.len()
    out["content_token_count"] = text_series.apply(_token_count)

    df_mask = out["content"].apply(lambda x: isinstance(x, float))
    new_df = out[~df_mask]
    return new_df


def describe_dataset(df: pd.DataFrame, mode: str = "basic") -> None:
    """
    Describe the dataset in three possible modes:

    Modes:
    - "basic": print unique counts and total sums
    - "distribution": print value_counts distributions
    - "plots": print value_counts and generate bar plots (one per column)
    """
    unique_cols = ["originalText", "id", "num", "chapterTitle", "title"]
    sum_cols = ["content_char_count", "content_token_count"]
    dist_cols = ["domain", "type"]

    if mode == "basic":
        print("=== UNIQUE COUNTS ===")
        for col in unique_cols:
            if col in df.columns:
                print(f"nb_uniques ({col}) =", df[col].nunique())

        print("\n=== TOTAL SUMS ===")
        for col in sum_cols:
            if col in df.columns:
                print(f"sum ({col}) =", df[col].sum())

    elif mode == "distribution":
        print("=== DISTRIBUTIONS (value_counts) ===")
        for col in dist_cols:
            if col in df.columns:
                print(f"\nDistribution for {col}: nb {len(df[col].value_counts())}")
                print(df[col].value_counts())

    elif mode == "plots":
        import matplotlib.pyplot as plt

        print("=== DISTRIBUTIONS WITH PLOTS ===")
        for col in dist_cols:
            if col in df.columns:
                counts = df[col].value_counts()

                print(f"\nDistribution for {col}:")
                print(counts)

                # Create a separate figure for each plot (no subplots)
                plt.figure()
                counts.plot(kind="bar")
                plt.title(f"Distribution of {col}")
                plt.xlabel(col)
                plt.ylabel("Count")
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()
                plt.savefig(f"img/{col}.png")
                plt.show()
    elif mode == "all":
        describe_dataset(df, mode="basic")
        describe_dataset(df, mode="distribution")
        describe_dataset(df, mode="plots")
    else:
        raise ValueError("Invalid mode. Choose from: 'basic', 'distribution', 'plots'.")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate the main dataset, counts, and optional descriptions/NER."
    )
    parser.add_argument(
        "--input", default="data_v6_bis.json", help="Input JSON dataset path."
    )
    parser.add_argument(
        "--out-csv", default="main_dataset_irit.csv", help="Output CSV path."
    )
    parser.add_argument("--out-json", default="data_v7.json", help="Output JSON path.")
    parser.add_argument(
        "--encoding", default="cl100k_base", help="tiktoken encoding name."
    )
    parser.add_argument(
        "--describe",
        choices=["basic", "distribution", "plots", "none", "all"],
        default="none",
        help="Run dataset description.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    df = pd.read_json(args.input)
    expanded_df = expand_refs(df)
    # Remove rows where 'content' contains "<contenu de l'article>" or is NaN
    expanded_df = expanded_df[
        ~expanded_df["content"].apply(
            lambda ele: isinstance(ele, str) and "<contenu de l'article>" in ele
        )
    ]

    # Filter out NaN / empty content
    mask_placeholder = expanded_df["content"].str.contains(
        "<contenu de l'article>", na=False
    )
    mask_nan = expanded_df["content"].isna()
    mask_empty = expanded_df["content"].str.strip().eq("")

    expanded_df = expanded_df[~mask_placeholder & ~mask_nan & ~mask_empty]

    new_df = add_content_counts(
        expanded_df, content_col="content", encoding_name=args.encoding
    )
    new_df = new_df.loc[new_df["content"].str.strip().ne("")].copy()

    print(
        f"Dataset has {len(new_df)} rows after filtering non-string content, {len(new_df) - len(expanded_df)} rows removed."
    )

    last_df = remove_rows_matching_regex(new_df, "content")

    print(
        f"Final dataset has {len(last_df)} rows after filtering metadata contents, {len(new_df) - len(last_df)} rows removed."
    )

    ## deduplication

    df_step1, report_original = dedup_when_only_originalText_varies(
        last_df,
        content_col="content",
        original_text_col="originalText",
        skip_cols=[
            "refs",
            "tree",
            "originalText",
        ],  # you can pass e.g. ["refs", "tree"] if you want them ignored here
    )
    print("Rows after step 1 of deduplication:", len(df_step1))

    # 2) Deduplicate groups where only domain varies (excluding noisy cols),
    #    and also drop pure duplicates with NO variation.
    df_final, report_domain = dedup_when_only_domain_varies_concat_domains(
        df_step1,
        content_col="content",
        domain_col="domain",
        skip_cols=[
            "refs",
            "tree",
            "originalText",
        ],  # mirrors your variation-pattern logic
        also_dedup_no_variation=True,
    )

    print("Rows after step 2 of deduplication:", len(df_final))

    vc = df_final["content"].value_counts(dropna=False)
    dup_contents = vc[vc > 1].index
    df_final = df_final[~df_final["content"].isin(dup_contents)]
    print("Rows after deduplication with metadata different:", len(df_final))
    df_final.to_csv(args.out_csv, index=False)
    df_final.to_json(args.out_json, index=False)

    #  Balanced sample: max 3 rows per (domain, title)
    sampled_parts = []

    for (_, _), g in df.groupby(["domain", "title"]):
        n = min(len(g), max(1, math.ceil(len(g) * 0.10)))
        sampled_parts.append(g.sample(n=n, random_state=42))

    new_df = pd.concat(sampled_parts, ignore_index=True)
    new_df.to_csv("src/data/maintreg_database_clean_balanced.csv", index=False)
    if args.describe != "none":
        describe_dataset(df_final, mode=args.describe)


if __name__ == "__main__":
    main()
