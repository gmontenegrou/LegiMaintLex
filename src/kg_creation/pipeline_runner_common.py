import argparse
import csv
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

PLACEHOLDER_PATTERN = re.compile(r"\$\{([^}]+)\}")


def deep_merge(base: Any, override: Any) -> Any:
    if isinstance(base, dict) and isinstance(override, dict):
        merged = dict(base)
        for key, value in override.items():
            if key in merged:
                merged[key] = deep_merge(merged[key], value)
            else:
                merged[key] = value
        return merged
    return override


def load_yaml_file(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as fh:
        payload = yaml.safe_load(fh) or {}
    if not isinstance(payload, dict):
        raise ValueError("The pipeline YAML must contain a top-level mapping.")
    return payload


def load_yaml_config(config_path: Path) -> dict[str, Any]:
    payload = load_yaml_file(config_path)

    inherits = payload.pop("inherits", None)
    if not inherits:
        return payload

    parent_refs = inherits if isinstance(inherits, list) else [inherits]
    merged_parent: dict[str, Any] = {}

    for parent_ref in parent_refs:
        parent_path = Path(parent_ref)
        if not parent_path.is_absolute():
            parent_path = (config_path.parent / parent_path).resolve()
        parent_payload = load_yaml_config(parent_path)
        merged_parent = deep_merge(merged_parent, parent_payload)

    return deep_merge(merged_parent, payload)


def get_by_dotted_path(payload: dict[str, Any], dotted_path: str) -> Any:
    current: Any = payload
    for part in dotted_path.split("."):
        if not isinstance(current, dict) or part not in current:
            raise KeyError(f"Unknown config placeholder: {dotted_path}")
        current = current[part]
    return current


def render_value(value: Any, context: dict[str, Any]) -> Any:
    if isinstance(value, str):
        full_match = PLACEHOLDER_PATTERN.fullmatch(value)
        if full_match:
            return render_value(
                get_by_dotted_path(context, full_match.group(1)), context
            )

        def _replace(match: re.Match[str]) -> str:
            resolved = render_value(
                get_by_dotted_path(context, match.group(1)), context
            )
            return str(resolved)

        return PLACEHOLDER_PATTERN.sub(_replace, value)

    if isinstance(value, list):
        return [render_value(item, context) for item in value]

    if isinstance(value, dict):
        return {key: render_value(item, context) for key, item in value.items()}

    return value


def cli_args_from_mapping(args_mapping: dict[str, Any]) -> list[str]:
    cli_args: list[str] = []
    for key, value in args_mapping.items():
        flag = f"--{key}"
        if value is None:
            continue
        if isinstance(value, bool):
            if value:
                cli_args.append(flag)
            continue
        if isinstance(value, list):
            for item in value:
                cli_args.extend([flag, str(item)])
            continue
        cli_args.extend([flag, str(value)])
    return cli_args


def select_stages(
    stages: list[dict[str, Any]],
    only: list[str] | None,
    from_stage: str | None,
    to_stage: str | None,
) -> list[dict[str, Any]]:
    selected = [stage for stage in stages if stage.get("enabled", True)]

    if only:
        only_set = set(only)
        selected = [stage for stage in selected if stage.get("name") in only_set]

    if from_stage:
        stage_names = [stage.get("name") for stage in selected]
        if from_stage not in stage_names:
            raise ValueError(f"Stage not found in enabled stages: {from_stage}")
        selected = selected[stage_names.index(from_stage) :]

    if to_stage:
        stage_names = [stage.get("name") for stage in selected]
        if to_stage not in stage_names:
            raise ValueError(f"Stage not found in enabled stages: {to_stage}")
        selected = selected[: stage_names.index(to_stage) + 1]

    return selected


def build_runtime_context(
    config: dict[str, Any],
    config_path: Path,
    python_executable_override: str | None,
) -> dict[str, Any]:
    runtime = dict(config)
    runtime.setdefault("variables", {})
    runtime.setdefault("paths", {})
    runtime.setdefault("runtime", {})

    raw_input_csv = runtime["paths"].get("raw_input_csv")
    input_nrows = None
    input_subset_nrows = runtime["variables"].get("input_subset_nrows")
    if input_subset_nrows is not None:
        input_subset_nrows = int(input_subset_nrows)
    if raw_input_csv:
        input_path = Path(str(raw_input_csv))
        if input_path.exists():
            with input_path.open("r", encoding="utf-8-sig", newline="") as fh:
                reader = csv.reader(fh)
                next(reader, None)
                input_nrows = sum(1 for _ in reader)

    effective_input_nrows = input_nrows
    if input_subset_nrows is not None:
        if input_nrows is None:
            effective_input_nrows = input_subset_nrows
        else:
            effective_input_nrows = min(input_nrows, input_subset_nrows)

    runtime["runtime"] = {
        "config_path": str(config_path),
        "config_dir": str(config_path.parent.resolve()),
        "repo_root": str(Path.cwd()),
        "python_executable": python_executable_override
        or runtime["variables"].get("python_executable")
        or sys.executable,
        "input_nrows": input_nrows,
        "effective_input_nrows": effective_input_nrows,
    }
    return runtime


def run_stage(
    stage: dict[str, Any],
    context: dict[str, Any],
    dry_run: bool,
) -> None:
    stage_name = stage.get("name", "<unnamed>")
    script = render_value(stage.get("script"), context)
    args_mapping = render_value(stage.get("args", {}), context)
    env_overrides = render_value(stage.get("env", {}), context)
    python_executable = str(context["runtime"]["python_executable"])

    if not script:
        raise ValueError(f"Stage '{stage_name}' is missing 'script'.")
    if not isinstance(args_mapping, dict):
        raise ValueError(f"Stage '{stage_name}' args must be a mapping.")

    command = [python_executable, str(script), *cli_args_from_mapping(args_mapping)]
    printable = subprocess.list2cmdline(command)
    print(f"[RUN] {stage_name}")
    print(f"      {printable}")

    if dry_run:
        return

    env = os.environ.copy()
    env.update({str(key): str(value) for key, value in env_overrides.items()})
    subprocess.run(command, check=True, env=env)


def build_arg_parser(default_config_path: str | None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the dataset creation pipeline from a YAML configuration file."
    )
    config_kwargs: dict[str, Any] = {
        "type": Path,
        "help": "YAML config describing pipeline stages and arguments.",
    }
    if default_config_path is None:
        config_kwargs["required"] = True
    else:
        config_kwargs["default"] = Path(default_config_path)
    parser.add_argument("--config", **config_kwargs)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands without executing them.",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        help="Run only the listed enabled stage names.",
    )
    parser.add_argument(
        "--from-stage",
        default=None,
        help="Start execution from this enabled stage name.",
    )
    parser.add_argument(
        "--to-stage",
        default=None,
        help="Stop execution after this enabled stage name.",
    )
    parser.add_argument(
        "--python",
        dest="python_executable",
        default=None,
        help="Optional Python executable override for all stages.",
    )
    return parser


def main(default_config_path: str | None = "config/default_experiment.yaml") -> None:
    args = build_arg_parser(default_config_path=default_config_path).parse_args()
    config = load_yaml_config(args.config)
    context = build_runtime_context(
        config=config,
        config_path=args.config.resolve(),
        python_executable_override=args.python_executable,
    )

    stages = config.get("stages", [])
    if not isinstance(stages, list):
        raise ValueError("'stages' must be a list in the pipeline YAML.")

    selected_stages = select_stages(
        stages=stages,
        only=args.only,
        from_stage=args.from_stage,
        to_stage=args.to_stage,
    )

    if not selected_stages:
        raise ValueError("No enabled stages matched the requested filters.")

    for stage in selected_stages:
        run_stage(stage=stage, context=context, dry_run=args.dry_run)
