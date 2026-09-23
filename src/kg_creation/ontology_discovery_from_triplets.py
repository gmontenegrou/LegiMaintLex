import argparse
import asyncio
import ast
import json
import os
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional
from mistralai import File, Mistral

import pandas as pd
import tiktoken
from openai import AsyncOpenAI

from prompts import prompt_object_properties_candidates, prompt_owl_generation_compact

PROVIDER_BASE_URLS = {
    "gpt": None,
    "mistral": "https://api.mistral.ai/v1",
}


def get_async_client(provider: str):
    provider = provider.lower()

    if provider == "gpt":
        return AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    if provider == "mistral":
        return Mistral(
            api_key=os.getenv("MISTRAL_API_KEY"),
            # base_url=PROVIDER_BASE_URLS["mistral"],
        )

    raise ValueError("Unsupported provider")


def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    import re

    if not text:
        return None

    s = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", s, flags=re.DOTALL)
    if fence:
        s = fence.group(1).strip()

    try:
        return json.loads(s)
    except Exception:
        pass

    match = re.search(r"(\{.*\})", s, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            return None

    return None


def _clean_literal_block(text: str) -> str:
    import re

    if not text:
        return ""
    s = text.strip()
    fence = re.search(r"```(?:json|python|py)?\s*(.*?)\s*```", s, flags=re.DOTALL)
    if fence:
        s = fence.group(1).strip()
    return s


def _normalize_candidate_payload(payload: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(payload, dict):
        return None

    out = {
        "candidates_classes": [],
        "candidates_object_props": [],
        "candidates_datatype_props": [],
        "unmapped": [],
        "conflicts": [],
    }

    for entry in payload.get("candidates_classes", []) or []:
        if isinstance(entry, dict):
            out["candidates_classes"].append(entry)
        elif isinstance(entry, (list, tuple)) and len(entry) >= 4:
            out["candidates_classes"].append(
                {
                    "id": entry[0],
                    "parent": entry[1],
                    "evidence": (
                        list(entry[2]) if isinstance(entry[2], (list, tuple)) else []
                    ),
                    "confidence": entry[3],
                }
            )

    for entry in payload.get("candidates_object_props", []) or []:
        if isinstance(entry, dict):
            out["candidates_object_props"].append(entry)
        elif isinstance(entry, (list, tuple)) and len(entry) >= 6:
            out["candidates_object_props"].append(
                {
                    "id": entry[0],
                    "domain": entry[1],
                    "range": entry[2],
                    "evidence": (
                        list(entry[3]) if isinstance(entry[3], (list, tuple)) else []
                    ),
                    "confidence": entry[4],
                    "reify": bool(entry[5]),
                }
            )

    for entry in payload.get("candidates_datatype_props", []) or []:
        if isinstance(entry, dict):
            out["candidates_datatype_props"].append(entry)
        elif isinstance(entry, (list, tuple)) and len(entry) >= 4:
            out["candidates_datatype_props"].append(
                {
                    "id": entry[0],
                    "domain": entry[1],
                    "datatype": entry[2],
                    "evidence": (
                        list(entry[3]) if isinstance(entry[3], (list, tuple)) else []
                    ),
                }
            )

    for entry in payload.get("unmapped", []) or []:
        if isinstance(entry, dict):
            out["unmapped"].append(entry)
        elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
            out["unmapped"].append({"expr": entry[0], "rejected_element": entry[1:]})

    for entry in payload.get("conflicts", []) or []:
        if isinstance(entry, dict):
            out["conflicts"].append(entry)
        elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
            triplet_ids = entry[0]
            out["conflicts"].append(
                {
                    "triplet_ids": (
                        list(triplet_ids)
                        if isinstance(triplet_ids, (list, tuple))
                        else []
                    ),
                    "rejected_element": entry[1:],
                }
            )

    return out


def extract_step1_candidate_model(text: str) -> Optional[Dict[str, Any]]:
    parsed_json = extract_json_from_text(text)
    normalized_json = _normalize_candidate_payload(parsed_json) if parsed_json else None
    if normalized_json:
        return normalized_json

    s = _clean_literal_block(text)
    if not s:
        return None

    try:
        parsed_literal = ast.literal_eval(s)
        normalized_literal = _normalize_candidate_payload(parsed_literal)
        if normalized_literal:
            return normalized_literal
    except Exception:
        pass

    import re

    match = re.search(r"(\{.*\})", s, flags=re.DOTALL)
    if match:
        try:
            parsed_literal = ast.literal_eval(match.group(1))
            normalized_literal = _normalize_candidate_payload(parsed_literal)
            if normalized_literal:
                return normalized_literal
        except Exception:
            return None

    return None


def get_encoding_for_model(model: str):
    try:
        return tiktoken.encoding_for_model(model)
    except Exception:
        return tiktoken.get_encoding("cl100k_base")


def count_tokens(enc, text: str) -> int:
    if not text:
        return 0
    return len(enc.encode(text))


def truncate_to_token_budget(
    enc, text: str, max_tokens: int, keep_tail: bool = True
) -> str:
    if not text or max_tokens <= 0:
        return ""
    token_ids = enc.encode(text)
    if len(token_ids) <= max_tokens:
        return text
    clipped = token_ids[-max_tokens:] if keep_tail else token_ids[:max_tokens]
    return enc.decode(clipped)


def infer_context_window_tokens(model: str) -> int:
    m = (model or "").lower()
    if "gpt-4.1" in m:
        return 1_000_000
    if "gpt-4o" in m:
        return 128_000
    if "mistral" in m:
        return 128_000
    return 128_000


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


async def llm_response_text(
    client, provider: str, model: str, prompt: str, max_output_tokens: int = 6000
):
    provider_normalized = provider.strip().lower()

    if provider_normalized == "gpt":
        response = await client.responses.create(
            model=model,
            input=prompt,
            temperature=0,
            max_output_tokens=max_output_tokens,
        )
        return response.output_text.strip(), getattr(response, "usage", None)

    if provider_normalized == "mistral":
        # Many mistralai SDK versions are synchronous. Run in a thread.
        def _call_mistral():
            return client.chat.complete(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )

        response = await asyncio.to_thread(_call_mistral)

        # Handle response shape robustly
        try:
            msg = response.choices[0].message
            # msg may be object with .content or a dict
            if isinstance(msg, dict):
                text = str(msg.get("content", "")).strip()
            else:
                text = str(getattr(msg, "content", "")).strip()
            return text, getattr(response, "usage", None)
        except Exception:
            # fallback: stringify response
            return str(response).strip(), getattr(response, "usage", None)

    raise ValueError("Unsupported provider. Use 'gpt' or 'mistral'.")


def _extract_content_from_batch_body(body: Dict[str, Any]) -> str:
    try:
        msg = body["choices"][0]["message"]["content"]
    except Exception:
        return ""

    if isinstance(msg, str):
        return msg.strip()
    if isinstance(msg, list):
        parts = []
        for item in msg:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "".join(parts).strip()
    return str(msg).strip()


async def run_mistral_batch_step1(
    prompts: List[str],
    model: str,
    poll_interval_seconds: float = 2.0,
) -> Dict[str, Dict[str, Any]]:
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("Missing MISTRAL_API_KEY for mistral batch.")

    client = Mistral(api_key=api_key)

    def _create_job():
        buffer = BytesIO()
        for idx, prompt in enumerate(prompts):
            req = {
                "custom_id": str(idx),
                "body": {
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0,
                },
            }
            buffer.write(json.dumps(req, ensure_ascii=False).encode("utf-8"))
            buffer.write(b"\n")

        input_file = client.files.upload(
            file=File(
                file_name="ontology_discovery_step1.jsonl", content=buffer.getvalue()
            ),
            purpose="batch",
        )
        return client.batch.jobs.create(
            input_files=[input_file.id],
            model=model,
            endpoint="/v1/chat/completions",
            metadata={"job_type": "ontology_discovery_step1"},
        )

    job = await asyncio.to_thread(_create_job)
    print(f"Created Mistral batch job: {job.id}")

    while job.status in ["QUEUED", "RUNNING"]:
        await asyncio.sleep(poll_interval_seconds)
        job = await asyncio.to_thread(client.batch.jobs.get, job_id=job.id)
        total = getattr(job, "total_requests", 0) or 0
        succ = getattr(job, "succeeded_requests", 0) or 0
        fail = getattr(job, "failed_requests", 0) or 0
        done_pct = (100.0 * (succ + fail) / total) if total else 0.0
        print(
            f"Batch status={job.status} total={total} "
            f"succeeded={succ} failed={fail} done={done_pct:.1f}%"
        )

    outputs_by_id: Dict[str, Dict[str, Any]] = {}
    if getattr(job, "output_file", None):
        out_file = await asyncio.to_thread(
            client.files.download, file_id=job.output_file
        )
        lines: List[str] = []
        for chunk in out_file.stream:
            lines.extend(chunk.decode("utf-8").splitlines())
        for line in lines:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                cid = str(item.get("custom_id", ""))
                body = item.get("response", {}).get("body", {})
                outputs_by_id[cid] = {
                    "text": _extract_content_from_batch_body(body),
                    "usage": body.get("usage"),
                }
            except Exception:
                continue

    if getattr(job, "error_file", None):
        err_file = await asyncio.to_thread(
            client.files.download, file_id=job.error_file
        )
        err_lines: List[str] = []
        for chunk in err_file.stream:
            err_lines.extend(chunk.decode("utf-8").splitlines())
        for line in err_lines:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                cid = str(item.get("custom_id", ""))
                outputs_by_id[cid] = {
                    "text": f"ERROR: {json.dumps(item, ensure_ascii=False)}",
                    "usage": None,
                }
            except Exception:
                continue

    return outputs_by_id


# async def llm_response_text(
#     client, model: str, prompt: str, max_output_tokens: int = 6000
# ):
#     response = await client.responses.create(
#         model=model,
#         input=prompt,
#         temperature=0,
#         max_output_tokens=max_output_tokens,
#     )
#     return response.output_text.strip(), getattr(response, "usage", None)


def _triplet_cols(df: pd.DataFrame) -> List[str]:
    preferred = [
        "head",
        "head_type",
        "relation",
        "tail",
        "tail_type",
        "evidence",
        # "topic",
        # "text",
    ]
    return [c for c in preferred if c in df.columns]


def build_triplet_batches(
    df: pd.DataFrame, batch_size: int
) -> List[List[Dict[str, Any]]]:
    cols = _triplet_cols(df)
    rows = df[cols].fillna("").to_dict(orient="records")
    batches = []

    for i in range(0, len(rows), batch_size):
        chunk = rows[i : i + batch_size]
        batches.append([dict(r) for r in chunk])

    return batches


def _unique_list_of_dicts(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for x in items or []:
        try:
            key = json.dumps(x, sort_keys=True, ensure_ascii=False)
        except Exception:
            key = str(x)
        if key not in seen:
            seen.add(key)
            out.append(x)
    return out


def compactify_triplets_batch(triplets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert triplet dicts into compact representations with short keys:
    s = head, st = head_type, p = relation, o = tail, ot = tail_type, ev = evidence
    """
    compact = []
    for t in triplets:
        compact.append(
            {
                "s": t.get("head", ""),  # truncate long strings
                "st": t.get("head_type", ""),
                "p": t.get("relation", ""),
                "o": t.get("tail", ""),
                "ot": t.get("tail_type", ""),
                "ev": t.get("evidence", ""),
            }
        )
    return compact


def merge_candidate_models(
    acc: Optional[Dict[str, Any]],
    new: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    if not acc:
        acc = {}
    if not new:
        return acc

    array_keys = [
        "candidates_classes",
        "candidates_object_props",
        "candidates_datatype_props",
        "unmapped",
        "conflicts",
    ]
    for k in array_keys:
        acc[k] = _unique_list_of_dicts((acc.get(k) or []) + (new.get(k) or []))
    return acc


async def run_ontology_discovery_batched(
    df_triplets: pd.DataFrame,
    ontology: str,
    model: str = "gpt-4.1",
    provider: str = "gpt",
    batch_size: int = 15,
    run_step2: bool = False,
    mistral_use_batch: bool = False,
    mistral_batch_poll_interval: float = 2.0,
    include_previous_output_context: bool = False,
    previous_context_max_chars: int = 12000,
    context_window_tokens: Optional[int] = None,
    step1_max_output_tokens: int = 10000,
    step1_safety_margin_tokens: int = 1000,
    step1_retry_missing_batch: bool = True,
) -> Dict[str, Any]:
    client = get_async_client(provider)
    enc = get_encoding_for_model(model)
    batch_size = max(1, int(batch_size))
    batches = build_triplet_batches(df_triplets, batch_size=batch_size)
    batch_source_row_indexes: List[List[Any]] = []
    if "_source_row_index" in df_triplets.columns:
        source_rows = df_triplets["_source_row_index"].tolist()
        for i in range(0, len(source_rows), batch_size):
            batch_source_row_indexes.append(source_rows[i : i + batch_size])
    else:
        source_rows = df_triplets.index.tolist()
        for i in range(0, len(source_rows), batch_size):
            batch_source_row_indexes.append(source_rows[i : i + batch_size])
    context_window_tokens = context_window_tokens or infer_context_window_tokens(model)
    step1_input_budget = max(
        0, context_window_tokens - step1_max_output_tokens - step1_safety_margin_tokens
    )

    batch_results = []
    accumulated_model = {}
    last_step2_raw = ""
    prompt_tokens_total = 0
    completion_tokens_total = 0

    use_mistral_batch_step1 = (
        provider.strip().lower() == "mistral"
        and mistral_use_batch
        and not run_step2
        and not include_previous_output_context
    )
    if mistral_use_batch and not use_mistral_batch_step1:
        print(
            "mistral_use_batch requested but disabled. "
            "Requirements: provider='mistral', run_step2=False, include_previous_output_context=False."
        )

    step1_prompts: List[str] = []
    step1_prompt_tokens: List[int] = []
    batch_step1_outputs: Dict[str, Dict[str, Any]] = {}
    if use_mistral_batch_step1:
        for triplets_batch in batches:
            compact_triplets = compactify_triplets_batch(triplets_batch)
            triplets_json = json.dumps(compact_triplets, ensure_ascii=False)
            prompt1 = prompt_object_properties_candidates(
                current_ontology=ontology, triplets_json=triplets_json
            )
            step1_prompts.append(prompt1)
            step1_prompt_tokens.append(count_tokens(enc, prompt1))
        batch_step1_outputs = await run_mistral_batch_step1(
            prompts=step1_prompts,
            model=model,
            poll_interval_seconds=mistral_batch_poll_interval,
        )

    for batch_idx, triplets_batch in enumerate(batches, start=1):
        previous_context_json = ""
        if include_previous_output_context and accumulated_model:
            prev_json = json.dumps(accumulated_model, ensure_ascii=False)
            if len(prev_json) > previous_context_max_chars:
                prev_json = prev_json[-previous_context_max_chars:]
            previous_context_json = prev_json

        compact_triplets = compactify_triplets_batch(triplets_batch)
        triplets_json = json.dumps(compact_triplets, ensure_ascii=False)

        prompt1_base = prompt_object_properties_candidates(
            current_ontology=ontology, triplets_json=triplets_json
        )
        prompt1 = prompt1_base
        if previous_context_json:
            previous_header = (
                "\n\nAdditional context from previous batches "
                "(reuse/extend, do not duplicate semantically equivalent items):\n"
            )
            base_tokens = count_tokens(enc, prompt1_base)
            header_tokens = count_tokens(enc, previous_header)
            available_for_previous = max(
                0, step1_input_budget - base_tokens - header_tokens
            )
            previous_context_json = truncate_to_token_budget(
                enc, previous_context_json, available_for_previous, keep_tail=True
            )
            if previous_context_json:
                prompt1 += f"{previous_header}{previous_context_json}"

        prompt1_tokens_est = count_tokens(enc, prompt1)
        if prompt1_tokens_est > step1_input_budget:
            print(
                f"Warning: step1 prompt in batch {batch_idx} is {prompt1_tokens_est} "
                f"tokens, above budget {step1_input_budget}."
            )
        print("using provider:", provider)
        if use_mistral_batch_step1:
            cid = str(batch_idx - 1)
            entry = batch_step1_outputs.get(cid, {})
            raw_step1 = entry.get("text", "ERROR: Missing batch output")
            usage1 = entry.get("usage", None)
            if step1_retry_missing_batch and (
                (not raw_step1)
                or raw_step1.startswith("ERROR: Missing batch output")
                or raw_step1.startswith("ERROR:")
            ):
                print(
                    f"Retrying step1 request for batch {batch_idx} "
                    f"(custom_id={cid}) after missing/failed batch output."
                )
                raw_step1, usage1 = await llm_response_text(
                    client,
                    provider=provider,
                    model=model,
                    prompt=step1_prompts[batch_idx - 1],
                    max_output_tokens=step1_max_output_tokens,
                )
        else:
            raw_step1, usage1 = await llm_response_text(
                client,
                model=model,
                prompt=prompt1,
                max_output_tokens=step1_max_output_tokens,
                provider=provider,
            )
        u1_in, u1_out = extract_usage_tokens(usage1)
        if use_mistral_batch_step1:
            step1_input_tokens = (
                u1_in if u1_in is not None else step1_prompt_tokens[batch_idx - 1]
            )
        else:
            step1_input_tokens = (
                u1_in if u1_in is not None else count_tokens(enc, prompt1)
            )
        step1_output_tokens = (
            u1_out if u1_out is not None else count_tokens(enc, raw_step1)
        )
        prompt_tokens_total += int(step1_input_tokens)
        completion_tokens_total += int(step1_output_tokens)
        parsed_step1 = extract_step1_candidate_model(raw_step1)
        parsed_step1_ok = isinstance(parsed_step1, dict)
        if parsed_step1 is None:
            print(
                f"Warning: Failed to parse step 1 structured output in batch {batch_idx}. "
                "Keeping accumulated model in memory for continuity."
            )
        step1_success = parsed_step1_ok

        if parsed_step1_ok:
            accumulated_model = merge_candidate_models(accumulated_model, parsed_step1)

        step2_input_tokens = 0
        step2_output_tokens = 0
        raw_step2 = ""
        step2_success = False

        if parsed_step1_ok:
            step2_input = accumulated_model if accumulated_model else parsed_step1
        elif accumulated_model:
            step2_input = accumulated_model
        else:
            # Last fallback requested: use raw step1 output directly.
            step2_input = raw_step1

        if run_step2 and step2_input:
            if isinstance(step2_input, str):
                candidate_payload = step2_input
            else:
                candidate_payload = json.dumps(
                    step2_input, ensure_ascii=False, indent=2
                )

            prompt2 = prompt_owl_generation_compact(candidate_json=candidate_payload)
            raw_step2, usage2 = await llm_response_text(
                client,
                provider=provider,
                model=model,
                prompt=prompt2,
                max_output_tokens=7000,
            )
            u2_in, u2_out = extract_usage_tokens(usage2)
            step2_input_tokens = (
                u2_in if u2_in is not None else count_tokens(enc, prompt2)
            )
            step2_output_tokens = (
                u2_out if u2_out is not None else count_tokens(enc, raw_step2)
            )
            prompt_tokens_total += int(step2_input_tokens)
            completion_tokens_total += int(step2_output_tokens)
            last_step2_raw = raw_step2
            step2_success = True

        batch_results.append(
            {
                "batch_index": batch_idx,
                "batch_triplets_count": len(triplets_batch),
                "batch_source_row_indexes": batch_source_row_indexes[batch_idx - 1],
                "step1_success": step1_success,
                "step1_input_tokens": int(step1_input_tokens),
                "step1_output_tokens": int(step1_output_tokens),
                "step1_raw": raw_step1,
                "step1_parsed": parsed_step1 if step1_success else {},
                "step2_success": step2_success,
                "step2_input_tokens": int(step2_input_tokens),
                "step2_output_tokens": int(step2_output_tokens),
                "step2_raw": raw_step2,
            }
        )

        print(
            f"[Batch {batch_idx}/{len(batches)}] "
            f"triplets={len(triplets_batch)} step1_success={step1_success} "
            f"tokens_in={int(step1_input_tokens + step2_input_tokens)} "
            f"tokens_out={int(step1_output_tokens + step2_output_tokens)}"
        )

    print(
        "Token totals:"
        f" input={prompt_tokens_total}, output={completion_tokens_total}, "
        f"total={prompt_tokens_total + completion_tokens_total}"
    )

    return {
        "batch_size": batch_size,
        "mistral_use_batch": use_mistral_batch_step1,
        "num_batches": len(batches),
        "batch_results": batch_results,
        "accumulated_candidate_model": accumulated_model,
        "final_owl_ttl": last_step2_raw,
        "token_usage": {
            "prompt_tokens_total": int(prompt_tokens_total),
            "completion_tokens_total": int(completion_tokens_total),
            "total_tokens": int(prompt_tokens_total + completion_tokens_total),
        },
    }


if __name__ == "__main__":

    def build_arg_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description=(
                "Run ontology discovery from triplets in batches and save the "
                "accumulated candidate model and optional OWL output."
            )
        )
        parser.add_argument(
            "--input-csv",
            type=Path,
            required=True,
            help="Input CSV with triplets prepared for ontology discovery.",
        )
        parser.add_argument(
            "--ontology-path",
            type=Path,
            default=Path("src/data/semleg-ontology-filtered.ttl"),
            help="Base ontology Turtle file used as prompt context.",
        )
        parser.add_argument(
            "--provider",
            choices=["gpt", "mistral"],
            default="gpt",
            help="LLM provider for ontology discovery.",
        )
        parser.add_argument(
            "--model",
            default="gpt-4.1",
            help="Model name to use for ontology discovery.",
        )
        parser.add_argument(
            "--batch-size",
            type=int,
            default=15,
            help="Number of triplets per batch.",
        )
        parser.add_argument(
            "--output-dir",
            type=Path,
            default=None,
            help="Directory where outputs will be written. Defaults to exp/new_ontology/<provider>/.",
        )
        parser.add_argument(
            "--batch-results-output",
            type=Path,
            default=None,
            help="Optional explicit path for batch-by-batch JSON output.",
        )
        parser.add_argument(
            "--candidate-model-output",
            type=Path,
            default=None,
            help="Optional explicit path for the accumulated candidate model JSON.",
        )
        parser.add_argument(
            "--final-owl-output",
            type=Path,
            default=None,
            help="Optional explicit path for the final OWL TTL when --run-step2 is enabled.",
        )
        parser.add_argument(
            "--run-step2",
            action="store_true",
            help="Also run the second prompt to generate OWL Turtle output.",
        )
        parser.add_argument(
            "--mistral-use-batch",
            action="store_true",
            help="Use Mistral batch API for step 1 when supported.",
        )
        parser.add_argument(
            "--mistral-batch-poll-interval",
            type=float,
            default=2.0,
            help="Polling interval in seconds for Mistral batch jobs.",
        )
        parser.add_argument(
            "--include-previous-output-context",
            action="store_true",
            help="Include previous accumulated output as extra context in subsequent batches.",
        )
        parser.add_argument(
            "--previous-context-max-chars",
            type=int,
            default=12000,
            help="Max characters kept from previous accumulated context.",
        )
        parser.add_argument(
            "--context-window-tokens",
            type=int,
            default=None,
            help="Optional manual override for the model context window.",
        )
        parser.add_argument(
            "--step1-max-output-tokens",
            type=int,
            default=10000,
            help="Max output tokens for step 1.",
        )
        parser.add_argument(
            "--step1-safety-margin-tokens",
            type=int,
            default=1000,
            help="Safety margin reserved inside the context window.",
        )
        parser.add_argument(
            "--no-step1-retry-missing-batch",
            action="store_true",
            help="Disable retry when a Mistral batch item is missing or malformed.",
        )
        return parser

    args = build_arg_parser().parse_args()

    output_dir = args.output_dir or Path(f"exp/new_ontology/{args.provider}")
    output_dir.mkdir(parents=True, exist_ok=True)

    batch_results_output = (
        args.batch_results_output
        or output_dir / "ontology_discovery_batches_final_two.json"
    )
    candidate_model_output = (
        args.candidate_model_output
        or output_dir / "ontology_discovery_accumulated_candidate_model_final_two.json"
    )
    final_owl_output = (
        args.final_owl_output or output_dir / "ontology_discovery_final.ttl"
    )

    with args.ontology_path.open("r", encoding="utf-8") as f:
        current_ontology = f.read()

    df_triplets = pd.read_csv(args.input_csv)
    print(len(df_triplets), "triplets found")

    result = asyncio.run(
        run_ontology_discovery_batched(
            df_triplets=df_triplets,
            ontology=current_ontology,
            model=args.model,
            provider=args.provider,
            batch_size=args.batch_size,
            run_step2=args.run_step2,
            mistral_use_batch=args.mistral_use_batch,
            mistral_batch_poll_interval=args.mistral_batch_poll_interval,
            include_previous_output_context=args.include_previous_output_context,
            previous_context_max_chars=args.previous_context_max_chars,
            context_window_tokens=args.context_window_tokens,
            step1_max_output_tokens=args.step1_max_output_tokens,
            step1_safety_margin_tokens=args.step1_safety_margin_tokens,
            step1_retry_missing_batch=not args.no_step1_retry_missing_batch,
        )
    )

    batch_results_output.parent.mkdir(parents=True, exist_ok=True)
    with batch_results_output.open("w", encoding="utf-8") as f:
        json.dump(result["batch_results"], f, ensure_ascii=False, indent=2)

    candidate_model_output.parent.mkdir(parents=True, exist_ok=True)
    with candidate_model_output.open("w", encoding="utf-8") as f:
        json.dump(
            result["accumulated_candidate_model"], f, ensure_ascii=False, indent=2
        )

    if args.run_step2:
        final_owl_output.parent.mkdir(parents=True, exist_ok=True)
        with final_owl_output.open("w", encoding="utf-8") as f:
            f.write(result["final_owl_ttl"])

    print("Ontology discovery completed.")
    print(f"Batches: {result['num_batches']} (batch_size={result['batch_size']})")
    print(
        "Token usage:"
        f" input={result['token_usage']['prompt_tokens_total']},"
        f" output={result['token_usage']['completion_tokens_total']},"
        f" total={result['token_usage']['total_tokens']}"
    )
    print("Saved:")
    print(f"- Input dataset: {args.input_csv}")
    print(f"- Batch results: {batch_results_output}")
    print(f"- Candidate model: {candidate_model_output}")
    if args.run_step2:
        print(f"- Final OWL: {final_owl_output}")
