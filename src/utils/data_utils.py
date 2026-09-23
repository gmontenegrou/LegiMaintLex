import pandas as pd
import logging

logger = logging.getLogger(__name__)


def load_articles_from_csv(
    csv_path: str = None,
    n_rows: int = 10,
    random=False,
) -> pd.DataFrame:
    """
    Load a CSV containing at least 'input' and 'output' columns and return the head(n_rows).
    """
    if csv_path.endswith(".csv"):
        df = pd.read_csv(csv_path)
        logger.info(f"Nb total rows: {len(df)} in the csv.")
    else:
        # logger.error("Failed to read CSV at %s: %s", csv_path)
        logger.info("Trying to read as JSON.")
        try:
            df = pd.read_json(csv_path, lines=True)
            logger.info(f"Nb total rows: {len(df)} in the json.")
        except Exception as json_exc:
            logger.error("Failed to read JSON at %s: %s", csv_path, json_exc)
            raise ValueError(f"Failed to load data from {csv_path} as CSV or JSON.")

    if random:
        logger.info(f"Sampling...: {n_rows} rows with 42 seed.")
        df.sample(n=n_rows, random_state=42)

    logger.info("Loaded %d rows from %s.", min(n_rows, len(df)), csv_path)
    return df


query_classes = """
PREFIX rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX owl:  <http://www.w3.org/2002/07/owl#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT DISTINCT ?class ?label
WHERE {
  ?class rdf:type owl:Class .
  OPTIONAL { ?class rdfs:label ?label }
}
ORDER BY ?class


"""

query_hierarchy = """
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?child ?parent
WHERE {
  ?child rdfs:subClassOf ?parent .
}
ORDER BY ?parent ?child
"""


def variation_pattern_stats_excluding_cols(
    df: pd.DataFrame, content_col: str = "content", skip_cols: list = None
) -> pd.DataFrame:
    """
    Analyze duplicated content groups and compute statistics of
    column-variation patterns, excluding specific columns.
    example of use:
        skip_cols = ["refs", "tree", "originalText", "num"]

        stats = variation_pattern_stats_excluding_cols(
            df,
            content_col="content",
            skip_cols=skip_cols
        )

    """

    if skip_cols is None:
        skip_cols = []

    def make_hashable(x):
        if isinstance(x, list):
            return tuple(x)
        if isinstance(x, set):
            return tuple(sorted(x))
        if isinstance(x, dict):
            return tuple(sorted(x.items()))
        return x

    pattern_stats = {}

    vc = df[content_col].value_counts(dropna=False)
    dup_contents = vc[vc > 1].index
    df_dups = df[df[content_col].isin(dup_contents)]

    cols_to_check = [c for c in df.columns if c != content_col and c not in skip_cols]

    for content_value, group in df_dups.groupby(content_col, dropna=False):

        varying_cols = [
            c
            for c in cols_to_check
            if group[c].map(make_hashable).nunique(dropna=False) > 1
        ]

        pattern = " | ".join(sorted(varying_cols)) if varying_cols else "NO_VARIATION"

        if pattern not in pattern_stats:
            pattern_stats[pattern] = {"group_count": 0, "total_rows": 0}

        pattern_stats[pattern]["group_count"] += 1
        pattern_stats[pattern]["total_rows"] += len(group)

    stats_df = (
        pd.DataFrame.from_dict(pattern_stats, orient="index")
        .reset_index()
        .rename(columns={"index": "varying_pattern"})
        .sort_values("group_count", ascending=False)
        .reset_index(drop=True)
    )

    stats_df["percentage_of_groups"] = (
        stats_df["group_count"] / stats_df["group_count"].sum() * 100
    ).round(2)

    return stats_df


def make_hashable(value):
    """Convert common unhashable objects into hashable representations."""
    if isinstance(value, list):
        return tuple(value)
    if isinstance(value, set):
        return tuple(sorted(value))
    if isinstance(value, dict):
        return tuple(sorted(value.items()))
    return value


def compute_varying_cols(
    group: pd.DataFrame, content_col: str, skip_cols: list[str]
) -> list[str]:
    """Compute varying columns in a group, excluding content_col and skip_cols."""
    cols_to_check = [
        c for c in group.columns if c != content_col and c not in skip_cols
    ]
    varying = []
    for c in cols_to_check:
        if group[c].map(make_hashable).nunique(dropna=False) > 1:
            varying.append(c)
    return sorted(varying)


def find_contents_by_pattern(
    df: pd.DataFrame,
    target_pattern: str,
    content_col: str = "content",
    skip_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    usage example:
        skip_cols = ["refs", "tree", "originalText", "num"]
        hits = find_contents_by_pattern(df, "id | title", skip_cols=skip_cols)
    Return a DataFrame listing all duplicated content values whose varying pattern
    (excluding skip_cols) matches target_pattern.

    target_pattern format must match what you saw in stats, e.g.:
      - "id | num | title"
      - "date | domain | id | num | title"
      - "NO_VARIATION"
    """
    if skip_cols is None:
        skip_cols = []

    vc = df[content_col].value_counts(dropna=False)
    dup_contents = vc[vc > 1].index
    df_dups = df[df[content_col].isin(dup_contents)].copy()

    rows = []
    for content_value, group in df_dups.groupby(content_col, dropna=False):
        varying_cols = compute_varying_cols(
            group, content_col=content_col, skip_cols=skip_cols
        )
        pattern_str = " | ".join(varying_cols) if varying_cols else "NO_VARIATION"
        if pattern_str == target_pattern:
            rows.append({"content": content_value, "group_size": len(group)})

    out = (
        pd.DataFrame(rows)
        .sort_values("group_size", ascending=False)
        .reset_index(drop=True)
    )
    return out


def inspect_content_group(
    df: pd.DataFrame,
    content_value: str,
    content_col: str = "content",
    skip_cols: list[str] | None = None,
    show_cols: list[str] | None = None,
    sort_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    example of use:
        example_content = hits.loc[0, "content"]
    group_df, summary = inspect_content_group(
        df,
        content_value=example_content,
        skip_cols=skip_cols,
        show_cols=["content", "domain", "id", "num", "title", "date", "chapterTitle", "originalText"],
    )
    Inspect all rows for a specific content_value.
    Returns:
      - group_df (optionally column-filtered and sorted)
      - summary dict with varying columns and per-column nunique
    """
    if skip_cols is None:
        skip_cols = []
    if show_cols is None:
        show_cols = [c for c in df.columns]  # show everything by default
    if sort_cols is None:
        sort_cols = [
            c
            for c in ["id", "num", "title", "date", "chapterTitle", "domain"]
            if c in df.columns
        ]

    group = df[df[content_col] == content_value].copy()

    varying_cols = compute_varying_cols(
        group, content_col=content_col, skip_cols=skip_cols
    )

    nunique_map = {}
    for c in group.columns:
        nunique_map[c] = int(group[c].map(make_hashable).nunique(dropna=False))

    # Sort + select columns for display
    group_view = group.copy()
    if sort_cols:
        # Only keep sort columns that exist
        sort_cols_existing = [c for c in sort_cols if c in group_view.columns]
        if sort_cols_existing:
            group_view = group_view.sort_values(sort_cols_existing)
    show_cols_existing = [c for c in show_cols if c in group_view.columns]
    group_view = group_view[show_cols_existing]

    summary = {
        "group_size": len(group),
        "varying_cols_excluding_skip": varying_cols,
        "nunique_per_col": nunique_map,
    }
    return group_view, summary


def inspect_n_groups_for_pattern(
    df: pd.DataFrame,
    target_pattern: str,
    n_groups: int = 3,
    content_col: str = "content",
    skip_cols: list[str] | None = None,
    show_cols: list[str] | None = None,
) -> list[tuple[str, pd.DataFrame, dict]]:
    """
    Find the first N content groups matching target_pattern and return their full rows + summaries.
    """
    hits = find_contents_by_pattern(
        df, target_pattern, content_col=content_col, skip_cols=skip_cols
    )
    results = []

    for i in range(min(n_groups, len(hits))):
        content_value = hits.loc[i, "content"]
        group_df, summary = inspect_content_group(
            df,
            content_value=content_value,
            content_col=content_col,
            skip_cols=skip_cols,
            show_cols=show_cols,
        )
        results.append((content_value, group_df, summary))

    return results
