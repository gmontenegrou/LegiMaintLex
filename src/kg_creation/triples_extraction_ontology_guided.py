import argparse
import asyncio
import json
import os
import time
from pathlib import Path
import pandas as pd
import tiktoken
from tqdm import tqdm
from openai import AsyncOpenAI
from mistralai import Mistral
from prompts import (
    build_prompt_topic_classification,
    build_prompt_triplets_extraction,
    build_prompt_entities_extraction_class_restricted,
    build_prompt_relations_extraction_fully_constrained_with_entities,
)


def get_async_client(provider: str):
    provider_normalized = provider.strip().lower()

    if provider_normalized == "openai":
        return AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    if provider_normalized == "mistral":
        # NOTE: Many mistralai SDK versions provide a synchronous client.
        # We'll call it from async via asyncio.to_thread.
        return Mistral(api_key=os.getenv("MISTRAL_API_KEY"))

    raise ValueError("Unsupported provider. Use 'openai' or 'mistral'.")


def extract_json_from_text(text: str):
    """
    Remove markdown fences like ```json ...``` and parse JSON.
    Returns (parsed_or_None, cleaned_text).
    """
    import re

    if text is None:
        return None, ""

    s = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", s, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        s = fence.group(1).strip()

    try:
        return json.loads(s), s
    except Exception:
        pass

    m = re.search(r"(\{.*\})", s, flags=re.DOTALL)
    if m:
        candidate = m.group(1).strip()
        try:
            return json.loads(candidate), candidate
        except Exception:
            return None, candidate

    return None, s


def get_encoding_for_model(model: str):
    try:
        return tiktoken.encoding_for_model(model)
    except Exception:
        return tiktoken.get_encoding("cl100k_base")


def count_tokens(enc, text: str) -> int:
    if not text:
        return 0
    return len(enc.encode(text))


def extract_usage_tokens(usage):
    if usage is None:
        return None, None

    if isinstance(usage, dict):
        prompt_tokens = usage.get("prompt_tokens", usage.get("input_tokens"))
        completion_tokens = usage.get("completion_tokens", usage.get("output_tokens"))
        return prompt_tokens, completion_tokens

    prompt_tokens = getattr(usage, "prompt_tokens", None)
    completion_tokens = getattr(usage, "completion_tokens", None)

    if prompt_tokens is None:
        prompt_tokens = getattr(usage, "input_tokens", None)
    if completion_tokens is None:
        completion_tokens = getattr(usage, "output_tokens", None)

    return prompt_tokens, completion_tokens


async def generate_text(
    client,
    provider: str,
    model: str,
    prompt: str,
    timeout_seconds: float = 120.0,
    max_retries: int = 3,
    retry_wait_seconds: float = 5.0,
):
    return await generate_text_with_retry(
        client=client,
        provider=provider,
        model=model,
        prompt=prompt,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        retry_wait_seconds=retry_wait_seconds,
    )


async def generate_text_with_retry(
    client,
    provider: str,
    model: str,
    prompt: str,
    timeout_seconds: float = 120.0,
    max_retries: int = 3,
    retry_wait_seconds: float = 5.0,
):
    provider_normalized = provider.strip().lower()
    last_exception = None

    for attempt in range(1, max_retries + 1):
        try:
            if provider_normalized == "openai":
                response = await asyncio.wait_for(
                    client.responses.create(
                        model=model,
                        input=prompt,
                        temperature=0,
                        max_output_tokens=4000,
                    ),
                    timeout=timeout_seconds,
                )
                return response.output_text.strip(), getattr(response, "usage", None)

            if provider_normalized == "mistral":
                # Many mistralai SDK versions are synchronous. Run in a thread.
                def _call_mistral():
                    return client.chat.complete(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0,
                        max_tokens=4000,
                    )

                response = await asyncio.wait_for(
                    asyncio.to_thread(_call_mistral),
                    timeout=timeout_seconds,
                )

                # Handle response shape robustly
                try:
                    msg = response.choices[0].message
                    if isinstance(msg, dict):
                        text = str(msg.get("content", "")).strip()
                    else:
                        text = str(getattr(msg, "content", "")).strip()
                    return text, getattr(response, "usage", None)
                except Exception:
                    return str(response).strip(), getattr(response, "usage", None)

            raise ValueError("Unsupported provider. Use 'openai' or 'mistral'.")
        except Exception as exc:
            last_exception = exc
            if attempt >= max_retries:
                break
            print(
                f"[WARN] provider={provider_normalized} model={model} "
                f"attempt={attempt}/{max_retries} failed: {exc}. "
                f"Retrying in {retry_wait_seconds}s..."
            )
            await asyncio.sleep(retry_wait_seconds)

    raise RuntimeError(
        f"LLM request failed after {max_retries} attempts "
        f"(provider={provider_normalized}, model={model}). Last error: {last_exception}"
    )


def _extract_triplets_list(step1_output_text: str):
    parsed, _ = extract_json_from_text(step1_output_text)
    if isinstance(parsed, dict) and isinstance(parsed.get("triples"), list):
        return parsed.get("triples", []), True
    return [], False


def _extract_entities_list(entities_output_text: str):
    parsed, _ = extract_json_from_text(entities_output_text)
    if isinstance(parsed, dict) and isinstance(parsed.get("entities"), list):
        return parsed.get("entities", []), True
    return [], False


def _debug_snippet(text: str, max_len: int = 600) -> str:
    if text is None:
        return ""
    s = str(text).replace("\n", "\\n")
    if len(s) <= max_len:
        return s
    return s[:max_len] + "...(truncated)"


def load_checkpoint_results(checkpoint_path: str | Path) -> tuple[list[dict], set[int]]:
    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        return [], set()

    checkpoint_df = pd.read_csv(checkpoint)
    if checkpoint_df.empty or "index" not in checkpoint_df.columns:
        return [], set()

    checkpoint_df = checkpoint_df.drop_duplicates(subset=["index"], keep="last")
    records = checkpoint_df.to_dict(orient="records")
    processed_indices = {
        int(idx) for idx in checkpoint_df["index"].tolist() if pd.notna(idx)
    }
    return records, processed_indices


def _merge_topics_into_triplets(triplets, topic_payload):
    if not isinstance(triplets, list):
        return []
    out = [dict(t) if isinstance(t, dict) else t for t in triplets]
    if not isinstance(topic_payload, dict):
        return out

    assignments = topic_payload.get("topic_by_triplet_index", [])
    if not isinstance(assignments, list):
        return out

    for item in assignments:
        if not isinstance(item, dict):
            continue
        idx = item.get("i", item.get("index"))
        topic = item.get("topic")
        if isinstance(idx, int) and 0 <= idx < len(out) and isinstance(out[idx], dict):
            if topic is not None:
                out[idx]["topic"] = topic

    return out


def compactify_triplets_for_topic_classification(
    triplets: list[dict],
) -> list[dict[str, str | int]]:
    compact = []
    for idx, triplet in enumerate(triplets):
        if not isinstance(triplet, dict):
            continue
        compact.append(
            {
                "i": idx,
                "h": str(triplet.get("head", "")).strip(),
                "ht": str(triplet.get("head_type", "")).strip(),
                "p": str(triplet.get("relation", "")).strip(),
                "t": str(triplet.get("tail", "")).strip(),
                "tt": str(triplet.get("tail_type", "")).strip(),
            }
        )
    return compact


async def _process_row(
    idx,
    df,
    model,
    provider,
    delay,
    sem,
    client,
    enc,
    constraint_mode="partial",
    constraints="triplets_ont",
    constraints_base=None,
    constraints_full=None,
    debug_outputs=False,
    request_timeout_seconds=120.0,
    max_retries=3,
    retry_wait_seconds=5.0,
):
    async with sem:
        row = df.iloc[idx]
        context = row["content"]

        mode = (constraint_mode or "partial").strip().lower()
        constraints_for_single_step = (
            constraints_base if mode == "partial" else constraints_full
        )
        prompt1 = build_prompt_triplets_extraction(context, constraints_for_single_step)

        parsed_json = {"context_main_topic": "", "triples": []}
        entities_payload = {"entities": []}
        entities_list = []
        entities_valid_json = False
        entities_output = ""
        valid_json = False
        triplets_output = ""
        entities_input_tokens = 0
        entities_output_tokens = 0
        step1_input_tokens = 0
        step1_output_tokens = 0
        step1_5_input_tokens = 0
        step1_5_output_tokens = 0

        try:
            if mode == "two_steps":
                if not constraints_base or not constraints_full:
                    raise ValueError(
                        "constraint_mode='two_steps' requires both constraints_base and constraints_full."
                    )

                # STEP 1A: entity extraction with base constraints (classes)
                prompt_entities = build_prompt_entities_extraction_class_restricted(
                    context, constraints_base
                )
                entities_output, usage_entities = await generate_text(
                    client,
                    provider,
                    model,
                    prompt_entities,
                    timeout_seconds=request_timeout_seconds,
                    max_retries=max_retries,
                    retry_wait_seconds=retry_wait_seconds,
                )
                u_ent_in, u_ent_out = extract_usage_tokens(usage_entities)
                ent_input_tokens = (
                    u_ent_in
                    if u_ent_in is not None
                    else count_tokens(enc, prompt_entities)
                )
                ent_output_tokens = (
                    u_ent_out
                    if u_ent_out is not None
                    else count_tokens(enc, entities_output)
                )
                entities_input_tokens = int(ent_input_tokens)
                entities_output_tokens = int(ent_output_tokens)

                entities_list, entities_valid_json = _extract_entities_list(
                    entities_output
                )
                entities_payload = {"entities": entities_list}
                if debug_outputs:
                    print(
                        f"[row {idx}] entities_output_raw={_debug_snippet(entities_output)}"
                    )
                    print(f"[row {idx}] entities_count={len(entities_list)}")
                entities_json = json.dumps(
                    {"entities": entities_list}, ensure_ascii=False
                )

                # STEP 1B: relation extraction with full constraints and provided entities
                relation_extraction_open = (
                    constraints or ""
                ).strip().lower() == "classes"
                prompt_relations = (
                    build_prompt_relations_extraction_fully_constrained_with_entities(
                        context,
                        entities_json,
                        constraints_full,
                        relation_extraction_open=relation_extraction_open,
                    )
                )
                relations_output, usage_relations = await generate_text(
                    client,
                    provider,
                    model,
                    prompt_relations,
                    timeout_seconds=request_timeout_seconds,
                    max_retries=max_retries,
                    retry_wait_seconds=retry_wait_seconds,
                )
                u_rel_in, u_rel_out = extract_usage_tokens(usage_relations)
                rel_input_tokens = (
                    u_rel_in
                    if u_rel_in is not None
                    else count_tokens(enc, prompt_relations)
                )
                rel_output_tokens = (
                    u_rel_out
                    if u_rel_out is not None
                    else count_tokens(enc, relations_output)
                )

                triplets_output = relations_output
                step1_input_tokens = int(ent_input_tokens) + int(rel_input_tokens)
                step1_output_tokens = int(ent_output_tokens) + int(rel_output_tokens)

                triplets, valid_step1 = _extract_triplets_list(relations_output)
                if debug_outputs:
                    print(
                        f"[row {idx}] relations_output_raw={_debug_snippet(relations_output)}"
                    )
                    print(
                        f"[row {idx}] relations_valid_json={valid_step1} triplets_count={len(triplets)}"
                    )
                valid_json = valid_step1
                if valid_step1:
                    parsed_json = {
                        "context_main_topic": "",
                        "triples": triplets,
                    }
                else:
                    valid_json = False
                    parsed_json = {"context_main_topic": "", "triples": []}
            else:
                # STEP 1 (single prompt mode)
                triplets_output, usage1 = await generate_text(
                    client,
                    provider,
                    model,
                    prompt1,
                    timeout_seconds=request_timeout_seconds,
                    max_retries=max_retries,
                    retry_wait_seconds=retry_wait_seconds,
                )
                usage_prompt_1, usage_completion_1 = extract_usage_tokens(usage1)
                step1_input_tokens = (
                    usage_prompt_1
                    if usage_prompt_1 is not None
                    else count_tokens(enc, prompt1)
                )
                step1_output_tokens = (
                    usage_completion_1
                    if usage_completion_1 is not None
                    else count_tokens(enc, triplets_output)
                )

                triplets, valid_step1 = _extract_triplets_list(triplets_output)
                if debug_outputs:
                    print(
                        f"[row {idx}] step1_output_raw={_debug_snippet(triplets_output)}"
                    )
                    print(
                        f"[row {idx}] step1_valid_json={valid_step1} triplets_count={len(triplets)}"
                    )
                valid_json = valid_step1
                if valid_step1:
                    compact_triplets = compactify_triplets_for_topic_classification(
                        triplets
                    )
                    prompt1_5 = build_prompt_topic_classification(
                        context, json.dumps(compact_triplets, ensure_ascii=False)
                    )
                    topics_output, usage1_5 = await generate_text(
                        client,
                        provider,
                        model,
                        prompt1_5,
                        timeout_seconds=request_timeout_seconds,
                        max_retries=max_retries,
                        retry_wait_seconds=retry_wait_seconds,
                    )
                    u15_in, u15_out = extract_usage_tokens(usage1_5)
                    step1_5_input_tokens = (
                        u15_in if u15_in is not None else count_tokens(enc, prompt1_5)
                    )
                    step1_5_output_tokens = (
                        u15_out
                        if u15_out is not None
                        else count_tokens(enc, topics_output)
                    )
                    topics_payload, _ = extract_json_from_text(topics_output)
                    if debug_outputs:
                        print(
                            f"[row {idx}] topics_output_raw={_debug_snippet(topics_output)}"
                        )
                    enriched_triplets = _merge_topics_into_triplets(
                        triplets, topics_payload
                    )
                    parsed_json = {
                        "context_main_topic": "",
                        "triples": enriched_triplets,
                    }
                else:
                    valid_json = False

        except Exception as e:
            triplets_output = f"ERROR: {str(e)}"
            valid_json = False
            if debug_outputs:
                print(f"[row {idx}] exception_in_process_row={str(e)}")

        # Provenance (use row[...] not df.loc[idx,...])
        provenance = {
            "type_document": row.get("type"),
            "title": row.get("title"),
            "is_about": row.get("domain"),
            "number": row.get("num"),
            "id_local": row.get("id"),
        }

        if delay:
            await asyncio.sleep(delay)

    return {
        "index": idx,  # positional index in this df
        "content": context,
        "context_main_topic": parsed_json.get("context_main_topic", ""),
        "legal_triplets": parsed_json.get("triples", []),
        "entities_output_raw": entities_output,
        "entities_payload": entities_payload,
        "entities_extracted": entities_list,
        "entities_valid_json": entities_valid_json,
        "step1_raw_output": triplets_output,
        "step1_payload": parsed_json,
        "step1_valid_json": valid_json,
        "entities_input_tokens": entities_input_tokens,
        "entities_output_tokens": entities_output_tokens,
        "step1_input_tokens": step1_input_tokens + step1_5_input_tokens,
        "step1_output_tokens": step1_output_tokens + step1_5_output_tokens,
        "total_input_tokens": step1_input_tokens + step1_5_input_tokens,
        "total_output_tokens": step1_output_tokens + step1_5_output_tokens,
        "provenance": provenance,
    }


async def run_two_step_pipeline_async(
    df,
    constraints_BASE,
    constraints_FULL,
    model="gpt-4.1",
    provider="openai",
    delay=0.0,
    max_concurrency=10,
    checkpoint_every=500,
    checkpoint_path="exp/new_ontology/extraction_checkpoint.csv",
    constraint_mode="partial",
    constraints="triplets_ont",
    debug_outputs=False,
    request_timeout_seconds=120.0,
    max_retries=3,
    retry_wait_seconds=5.0,
    resume_from_checkpoint=False,
):
    process_start = time.perf_counter()
    # Ensure checkpoint directory exists
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    print(
        f"Running pipeline with model={model}, provider={provider}, "
        f"delay={delay}s, max_concurrency={max_concurrency}, "
        f"constraint_mode={constraint_mode}, "
        f"constraints={constraints}"
    )

    if not constraints_BASE or not constraints_FULL:
        raise ValueError("Both constraints_BASE and constraints_FULL are required.")

    existing_results = []
    processed_indices: set[int] = set()
    if resume_from_checkpoint:
        existing_results, processed_indices = load_checkpoint_results(checkpoint_path)
        if existing_results:
            print(
                f"Loaded {len(existing_results)} rows from checkpoint. "
                f"Resuming {len(df) - len(processed_indices)} remaining rows."
            )

    client = get_async_client(provider)
    enc = get_encoding_for_model(model)
    sem = asyncio.Semaphore(max_concurrency)

    pending_indices = [idx for idx in range(len(df)) if idx not in processed_indices]
    if not pending_indices:
        result_df = (
            pd.DataFrame(existing_results)
            .sort_values(by="index")
            .reset_index(drop=True)
        )
        print("Checkpoint already contains all requested rows. Nothing left to run.")
        return result_df

    tasks = [
        _process_row(
            idx,
            df,
            model,
            provider,
            delay,
            sem,
            client,
            enc,
            constraint_mode=constraint_mode,
            constraints=constraints,
            constraints_base=constraints_BASE,
            constraints_full=constraints_FULL,
            debug_outputs=debug_outputs,
            request_timeout_seconds=request_timeout_seconds,
            max_retries=max_retries,
            retry_wait_seconds=retry_wait_seconds,
        )
        for idx in pending_indices
    ]

    results = list(existing_results)
    for coro in tqdm(
        asyncio.as_completed(tasks),
        total=len(tasks),
        desc="Processing rows",
        unit="row",
    ):
        results.append(await coro)

        if checkpoint_every and len(results) % checkpoint_every == 0:
            pd.DataFrame(results).to_csv(checkpoint_path, index=False)

    results.sort(key=lambda x: x["index"])
    result_df = pd.DataFrame(results)
    print(
        "Token totals:"
        f" input={int(result_df['total_input_tokens'].sum())},"
        f" output={int(result_df['total_output_tokens'].sum())}"
    )
    elapsed_seconds = time.perf_counter() - process_start
    print(
        "Total processing time:"
        f" {elapsed_seconds:.2f}s ({elapsed_seconds / 60:.2f} min)"
    )
    return result_df


async def _reparse_and_classify_row(
    idx,
    df,
    model,
    provider,
    sem,
    client,
    enc,
    debug_outputs=False,
    request_timeout_seconds=120.0,
    max_retries=3,
    retry_wait_seconds=5.0,
):
    async with sem:
        row = df.iloc[idx]
        context = row.get("content", "")
        raw_output = row.get("step1_raw_output", "")

        triplets, valid_step1 = _extract_triplets_list(raw_output)
        parsed_json = {
            "context_main_topic": "",
            "triples": triplets if valid_step1 else [],
        }

        topic_input_tokens = 0
        topic_output_tokens = 0

        try:
            if valid_step1:
                compact_triplets = compactify_triplets_for_topic_classification(
                    triplets
                )
                prompt_topic = build_prompt_topic_classification(
                    context, json.dumps(compact_triplets, ensure_ascii=False)
                )
                topics_output, usage_topic = await generate_text(
                    client,
                    provider,
                    model,
                    prompt_topic,
                    timeout_seconds=request_timeout_seconds,
                    max_retries=max_retries,
                    retry_wait_seconds=retry_wait_seconds,
                )
                u_in, u_out = extract_usage_tokens(usage_topic)
                topic_input_tokens = (
                    u_in if u_in is not None else count_tokens(enc, prompt_topic)
                )
                topic_output_tokens = (
                    u_out if u_out is not None else count_tokens(enc, topics_output)
                )
                topics_payload, _ = extract_json_from_text(topics_output)
                if debug_outputs:
                    print(
                        f"[row {idx}] reparsed_triplets_count={len(triplets)} "
                        f"topics_output_raw={_debug_snippet(topics_output)}"
                    )
                parsed_json["triples"] = _merge_topics_into_triplets(
                    triplets, topics_payload
                )
            elif debug_outputs:
                print(
                    f"[row {idx}] reparsing_failed raw_output={_debug_snippet(raw_output)}"
                )
        except Exception as exc:
            valid_step1 = False
            parsed_json["triples"] = []
            if debug_outputs:
                print(f"[row {idx}] reparse_topic_exception={str(exc)}")

        result = row.to_dict()
        result["context_main_topic"] = parsed_json.get("context_main_topic", "")
        result["legal_triplets"] = parsed_json.get("triples", [])
        result["step1_payload"] = parsed_json
        result["step1_valid_json"] = valid_step1
        result["topic_input_tokens"] = int(topic_input_tokens)
        result["topic_output_tokens"] = int(topic_output_tokens)
        result["total_input_tokens"] = int(result.get("total_input_tokens", 0)) + int(
            topic_input_tokens
        )
        result["total_output_tokens"] = int(result.get("total_output_tokens", 0)) + int(
            topic_output_tokens
        )
        return result


async def reparse_and_run_topic_classification_async(
    input_csv_path,
    output_csv_path,
    model="gpt-4.1",
    provider="openai",
    max_concurrency=10,
    checkpoint_every=100,
    checkpoint_path=None,
    debug_outputs=False,
    request_timeout_seconds=120.0,
    max_retries=3,
    retry_wait_seconds=5.0,
    resume_from_checkpoint=False,
):
    process_start = time.perf_counter()
    input_path = Path(input_csv_path)
    output_path = Path(output_csv_path)
    checkpoint_target = Path(checkpoint_path) if checkpoint_path else output_path

    checkpoint_target.parent.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    print(
        f"Reparsing step1 outputs and running topic classification with "
        f"model={model}, provider={provider}, rows={len(df)}"
    )

    client = get_async_client(provider)
    enc = get_encoding_for_model(model)
    sem = asyncio.Semaphore(max_concurrency)

    existing_results = []
    processed_indices: set[int] = set()
    if resume_from_checkpoint:
        existing_results, processed_indices = load_checkpoint_results(checkpoint_target)
        if existing_results:
            print(
                f"Loaded {len(existing_results)} rows from checkpoint. "
                f"Resuming {len(df) - len(processed_indices)} remaining rows."
            )

    pending_indices = [idx for idx in range(len(df)) if idx not in processed_indices]
    if not pending_indices:
        result_df = (
            pd.DataFrame(existing_results)
            .sort_values(by="index")
            .reset_index(drop=True)
        )
        result_df.to_csv(output_path, index=False)
        print(
            "Checkpoint already contains all requested rows. Nothing left to reparse."
        )
        return result_df

    tasks = [
        _reparse_and_classify_row(
            idx,
            df,
            model,
            provider,
            sem,
            client,
            enc,
            debug_outputs=debug_outputs,
            request_timeout_seconds=request_timeout_seconds,
            max_retries=max_retries,
            retry_wait_seconds=retry_wait_seconds,
        )
        for idx in pending_indices
    ]

    results = list(existing_results)
    for coro in tqdm(
        asyncio.as_completed(tasks),
        total=len(tasks),
        desc="Reparsing rows",
        unit="row",
    ):
        results.append(await coro)
        if checkpoint_every and len(results) % checkpoint_every == 0:
            pd.DataFrame(results).to_csv(checkpoint_target, index=False)

    results.sort(key=lambda x: int(x.get("index", 0)))
    result_df = pd.DataFrame(results)
    result_df.to_csv(output_path, index=False)

    elapsed_seconds = time.perf_counter() - process_start
    print(
        f"Saved reparsed results to {output_path} in "
        f"{elapsed_seconds:.2f}s ({elapsed_seconds / 60:.2f} min)"
    )
    return result_df


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract ontology-guided triplets from a CSV dataset and optionally "
            "reparse an existing extraction output to recover topics."
        )
    )
    parser.add_argument("--input-csv", type=Path, required=True, help="Input CSV path.")
    parser.add_argument(
        "--output-csv",
        type=Path,
        required=True,
        help="Output CSV path for extracted triplets.",
    )
    parser.add_argument(
        "--output-input-copy-csv",
        type=Path,
        default=None,
        help="Optional output CSV path to save a copy of the input dataset.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help="Checkpoint CSV path. Defaults to <output-csv stem>_checkpoint.csv.",
    )
    parser.add_argument(
        "--constraints-base-path",
        type=Path,
        default=Path("src/data/semleg-triple-ont-auto-classes.ttl"),
        help="Ontology constraints file used for partial/two-step prompting.",
    )
    parser.add_argument(
        "--constraints-full-path",
        type=Path,
        default=Path("src/data/semleg-triple-ont-auto-classes.ttl"),
        help="Ontology constraints file used for full/two-step prompting.",
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "mistral"],
        default="openai",
        help="LLM provider for triplet extraction.",
    )
    parser.add_argument(
        "--model",
        default="gpt-4.1",
        help="Model name used for triplet extraction.",
    )
    parser.add_argument(
        "--constraint-mode",
        choices=["partial", "fully_constrained", "two_steps"],
        default="two_steps",
        help="Prompting strategy for triplet extraction.",
    )
    parser.add_argument(
        "--constraints",
        default="classes",
        help="Constraint family identifier consumed inside the prompts.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Delay between async requests.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=5,
        help="Maximum number of concurrent row-processing tasks.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=50,
        help="Save a checkpoint every N processed rows.",
    )
    parser.add_argument(
        "--debug-outputs",
        action="store_true",
        help="Print raw intermediate LLM outputs for debugging.",
    )
    parser.add_argument(
        "--reparse-existing-outputs",
        action="store_true",
        help="Reparse an existing extraction CSV and re-run topic classification instead of full extraction.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional maximum number of input rows to process from the input CSV.",
    )
    parser.add_argument(
        "--request-timeout-seconds",
        type=float,
        default=120.0,
        help="Timeout in seconds for each individual LLM request.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum number of retries for each LLM request.",
    )
    parser.add_argument(
        "--retry-wait-seconds",
        type=float,
        default=5.0,
        help="Wait time in seconds between retries.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        action="store_true",
        help="Resume from an existing checkpoint CSV by skipping already processed rows.",
    )
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()

    checkpoint_path = args.checkpoint_path or args.output_csv.with_name(
        f"{args.output_csv.stem}_checkpoint.csv"
    )

    with args.constraints_base_path.open("r", encoding="utf-8") as f:
        constraints_BASE = f.read()
    with args.constraints_full_path.open("r", encoding="utf-8") as f:
        constraints_FULL = f.read()

    if args.reparse_existing_outputs:
        asyncio.run(
            reparse_and_run_topic_classification_async(
                input_csv_path=args.input_csv,
                output_csv_path=args.output_csv,
                model=args.model,
                provider=args.provider,
                max_concurrency=args.max_concurrency,
                checkpoint_every=args.checkpoint_every,
                checkpoint_path=checkpoint_path,
                debug_outputs=args.debug_outputs,
                request_timeout_seconds=args.request_timeout_seconds,
                max_retries=args.max_retries,
                retry_wait_seconds=args.retry_wait_seconds,
                resume_from_checkpoint=args.resume_from_checkpoint,
            )
        )
        print(
            "Reparse + topic classification completed. "
            f"Results saved to {args.output_csv}"
        )
    else:
        new_df = pd.read_csv(args.input_csv)
        if args.max_rows is not None:
            new_df = new_df.head(args.max_rows).copy()
        print(f"Loaded dataset with {len(new_df)} rows")

        results_df = asyncio.run(
            run_two_step_pipeline_async(
                new_df,
                constraints_BASE=constraints_BASE,
                constraints_FULL=constraints_FULL,
                model=args.model,
                provider=args.provider,
                delay=args.delay,
                max_concurrency=args.max_concurrency,
                checkpoint_every=args.checkpoint_every,
                checkpoint_path=str(checkpoint_path),
                constraint_mode=args.constraint_mode,
                constraints=args.constraints,
                debug_outputs=args.debug_outputs,
                request_timeout_seconds=args.request_timeout_seconds,
                max_retries=args.max_retries,
                retry_wait_seconds=args.retry_wait_seconds,
                resume_from_checkpoint=args.resume_from_checkpoint,
            )
        )

        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(args.output_csv, index=False)

        if args.output_input_copy_csv:
            args.output_input_copy_csv.parent.mkdir(parents=True, exist_ok=True)
            new_df.to_csv(args.output_input_copy_csv, index=False)

        print(f"Pipeline completed. Results saved to {args.output_csv}")
