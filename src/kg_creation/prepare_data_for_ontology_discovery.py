"""
python src/kg_creation/prepare_data_for_ontology_discovery.py `
  --input "exp/new_ontology/openai/classes_guided_extraction_two_steps_by_openai_gpt-4.1_nrows_1389_reparsed_with_topics_normalized.csv" `
  --clean-output "exp/new_ontology/openai/classes_guided_extraction_two_steps_by_openai_gpt-4.1_nrows_1389_reparsed_with_topics_normalized_cleaned.csv" `
  --errors-output "exp/new_ontology/openai/classes_guided_extraction_two_steps_by_openai_gpt-4.1_nrows_1389_reparsed_with_topics_normalized_errors_removed.csv" `
  --sample-output "exp/new_ontology/openai/classes_guided_extraction_two_steps_by_openai_gpt-4.1_nrows_1389_reparsed_with_topics_normalized_sample_for_ontology_discovery.csv"

"""

import argparse
import math
from pathlib import Path

import pandas as pd

CLASSES = [
    "Action",
    "Actor",
    "Artifact",
    "Condition",
    "Definition",
    "Location",
    "Modality",
    "Reason",
    "Reference",
    "Situation",
    "Time",
]

DEFAULT_INPUT = Path(
    "exp/new_ontology/openai/classes_guided_extraction_two_steps_by_openai_gpt-4.1_nrows_1389_reparsed_with_topics_normalized.csv"
)


def infer_expected_tail_types_from_relation(
    relation: str, class_list: list[str]
) -> list[str]:
    if pd.isna(relation):
        return []

    rel = str(relation).strip().lower()
    return [cls for cls in class_list if cls.lower() in rel]


def has_error_marker(value: object) -> bool:
    if pd.isna(value):
        return False
    return str(value).strip().upper().startswith("ERROR:")


def normalize_group_component(value: object) -> str:
    if pd.isna(value):
        return "Unknown"

    text = str(value).strip()
    if not text:
        return "Unknown"

    sanitized = "".join(ch if ch.isalnum() else "_" for ch in text)
    sanitized = "_".join(part for part in sanitized.split("_") if part)
    return sanitized or "Unknown"


def add_group_id_from_signature(df: pd.DataFrame) -> pd.DataFrame:
    df_out = df.copy()

    required_columns = ["head_type", "relation", "tail_type"]
    missing_columns = [col for col in required_columns if col not in df_out.columns]
    if missing_columns:
        raise ValueError(
            "Cannot create group_id. Missing required columns: "
            + ", ".join(missing_columns)
        )

    df_out["group_id"] = df_out.apply(
        lambda row: " | ".join(
            [
                normalize_group_component(row["head_type"]),
                normalize_group_component(row["relation"]),
                normalize_group_component(row["tail_type"]),
            ]
        ),
        axis=1,
    )

    return df_out


def detect_error_rows(df: pd.DataFrame) -> pd.Series:
    error_mask = pd.Series(False, index=df.index)

    candidate_columns = [
        "step1_raw_output",
        "step2_raw_output",
        "legal_triplets",
        "triples",
    ]
    existing_columns = [col for col in candidate_columns if col in df.columns]

    for col in existing_columns:
        error_mask = error_mask | df[col].apply(has_error_marker)

    if "step1_valid_json" in df.columns:
        error_mask = error_mask | (
            df["step1_valid_json"].astype(str).str.strip().str.lower() != "true"
        )

    return error_mask


def detect_tail_type_mismatch_rows(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not {"relation", "tail_type"}.issubset(df.columns):
        empty = pd.DataFrame(columns=list(df.columns))
        return empty, empty

    check_df = df.copy()
    check_df["expected_tail_types_from_relation"] = check_df["relation"].apply(
        lambda relation: infer_expected_tail_types_from_relation(relation, CLASSES)
    )
    check_df = check_df[
        check_df["expected_tail_types_from_relation"].map(len) > 0
    ].copy()

    if check_df.empty:
        return check_df, check_df.copy()

    check_df["tail_type_matches_expectation"] = check_df.apply(
        lambda row: str(row["tail_type"]).strip()
        in row["expected_tail_types_from_relation"],
        axis=1,
    )
    errors_df = check_df[~check_df["tail_type_matches_expectation"]].copy()
    return check_df, errors_df


def build_sample(
    df: pd.DataFrame, sample_ratio: float, random_state: int
) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    if "group_id" not in df.columns:
        sample_size = max(1, math.ceil(len(df) * sample_ratio))
        return df.sample(
            n=min(sample_size, len(df)), random_state=random_state
        ).reset_index(drop=True)

    return (
        df.groupby("group_id", group_keys=False)
        .apply(
            lambda group: group.sample(
                n=min(len(group), max(1, math.ceil(len(group) * sample_ratio))),
                random_state=random_state,
            )
        )
        .reset_index(drop=True)
    )


def resolve_output_paths(input_path: Path) -> tuple[Path, Path, Path]:
    clean_path = input_path.with_name(f"{input_path.stem}_cleaned.csv")
    errors_path = input_path.with_name(f"{input_path.stem}_errors_removed.csv")
    sample_path = input_path.with_name(
        f"{input_path.stem}_sample_for_ontology_discovery.csv"
    )
    return clean_path, errors_path, sample_path


def run_preparation(
    input_path: Path,
    clean_output_path: Path | None,
    errors_output_path: Path | None,
    sample_output_path: Path | None,
    sample_ratio: float,
    random_state: int,
) -> None:
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} rows from {input_path}")

    error_mask = detect_error_rows(df)
    df_without_model_errors = df[~error_mask].copy()
    removed_model_errors_df = df[error_mask].copy()

    print(f"Rows removed due to explicit errors: {len(removed_model_errors_df)}")

    checked_df, relation_errors_df = detect_tail_type_mismatch_rows(
        df_without_model_errors
    )
    if not relation_errors_df.empty:
        relation_error_indexes = relation_errors_df.index
        df_clean = df_without_model_errors.drop(index=relation_error_indexes).copy()
    else:
        df_clean = df_without_model_errors.copy()

    removed_errors_df = pd.concat(
        [removed_model_errors_df, relation_errors_df], axis=0
    ).sort_index()
    removed_errors_df = removed_errors_df.loc[
        ~removed_errors_df.index.duplicated(keep="first")
    ]

    print(f"Rows checked for relation/type mismatch: {len(checked_df)}")
    print(f"Rows removed due to relation/type mismatch: {len(relation_errors_df)}")
    print(f"Final clean rows: {len(df_clean)}")

    df_clean = add_group_id_from_signature(df_clean)
    print("Created group_id from signature (head_type, relation, tail_type)")

    sample_df = build_sample(
        df=df_clean,
        sample_ratio=sample_ratio,
        random_state=random_state,
    )
    print(f"Sample rows for ontology discovery: {len(sample_df)}")

    default_clean_path, default_errors_path, default_sample_path = resolve_output_paths(
        input_path
    )
    clean_output_path = clean_output_path or default_clean_path
    errors_output_path = errors_output_path or default_errors_path
    sample_output_path = sample_output_path or default_sample_path

    clean_output_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(clean_output_path, index=False)
    removed_errors_df.to_csv(errors_output_path, index=False)
    sample_df.to_csv(sample_output_path, index=False)

    print(f"Saved clean dataset to: {clean_output_path}")
    print(f"Saved removed rows to: {errors_output_path}")
    print(f"Saved ontology discovery sample to: {sample_output_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare normalized triplets for ontology discovery by removing error rows, "
            "creating group_id from (head_type, relation, tail_type), and generating a stratified sample."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="CSV input path.",
    )
    parser.add_argument(
        "--clean-output",
        type=Path,
        default=None,
        help="Output CSV path for cleaned data.",
    )
    parser.add_argument(
        "--errors-output",
        type=Path,
        default=None,
        help="Output CSV path for removed rows.",
    )
    parser.add_argument(
        "--sample-output",
        type=Path,
        default=None,
        help="Output CSV path for ontology discovery sample.",
    )
    parser.add_argument(
        "--sample-ratio",
        type=float,
        default=0.05,
        help="Sample ratio by group_id. Default: 0.05",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed used for sampling.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_preparation(
        input_path=args.input,
        clean_output_path=args.clean_output,
        errors_output_path=args.errors_output,
        sample_output_path=args.sample_output,
        sample_ratio=args.sample_ratio,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
