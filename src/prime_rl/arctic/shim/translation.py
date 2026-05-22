"""Pure OAI <-> DSS /generate translation functions.

Kept free of tokenizer / network / FastAPI dependencies so they can be
unit-tested in isolation.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from typing import Any

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def oai_sampling_params(body: dict) -> dict:
    """Extract sampling parameters from an OpenAI chat-completions body
    and reshape them into DSS `/generate`'s `sampling_params` dict.

    Forwards only keys DSS's sampling engine accepts. `extra_body` is
    merged on top so vLLM-style extras (e.g. repetition_penalty, min_tokens)
    flow through.

    Note on transport: the OpenAI Python SDK flattens `extra_body` into
    top-level JSON before sending, so keys the caller stuffed under
    `sampling_args["extra_body"]` arrive on the wire at the top level of
    `body`. We look there first and fall back to a nested `extra_body` for
    direct (non-SDK) HTTP callers.
    """
    params: dict[str, Any] = {}

    if (t := body.get("temperature")) is not None:
        params["temperature"] = float(t)
    if (p := body.get("top_p")) is not None:
        params["top_p"] = float(p)
    if (k := body.get("top_k")) is not None:
        params["top_k"] = int(k)
    if (mt := body.get("max_tokens") or body.get("max_completion_tokens")) is not None:
        params["max_tokens"] = int(mt)
    if (stop := body.get("stop")) is not None:
        params["stop"] = stop
    if (n := body.get("n")) is not None:
        params["n"] = int(n)
    if (seed := body.get("seed")) is not None:
        params["seed"] = int(seed)
    # OAI carries "logprobs: bool" + "top_logprobs: int"; DSS takes a single int.
    if body.get("logprobs") is True and (tl := body.get("top_logprobs")) is not None:
        params["logprobs"] = int(tl)
    elif body.get("logprobs") is True:
        params["logprobs"] = 1

    nested = body.get("extra_body") or {}
    for key in ("min_tokens", "repetition_penalty"):
        if key in body:
            params[key] = body[key]
        elif key in nested:
            params[key] = nested[key]

    return params


def extract_routing_metadata(body: dict) -> tuple[str | None, bool]:
    """Pull the optional routing annotation from an OAI request body.

    Returns ``(routing_key, strict)``. ``routing_key`` is the opaque
    affinity key stamped by the orchestrator (typically ``str(group_id)``);
    the ArcticInference scheduler hashes it to pick a replica, so every
    turn of one multi-turn rollout lands on the replica that holds the
    KV-cache for the earlier turns. ``strict`` opts into hard pinning
    instead of the default soft-affinity (which rings to neighbouring
    replicas under load).

    The OpenAI Python SDK flattens ``extra_body`` into top-level JSON, so
    we look there first and fall back to a nested ``extra_body`` for
    direct HTTP callers.
    """
    nested = body.get("extra_body") or {}
    routing_key = body.get("routing_key")
    if routing_key is None:
        routing_key = nested.get("routing_key")
    if routing_key is not None and not isinstance(routing_key, str):
        routing_key = str(routing_key)
    strict = bool(body.get("routing_strict") or nested.get("routing_strict"))
    return routing_key, strict


def _serialize_tool_arguments(arguments: Any) -> str | None:
    if arguments is None:
        return "{}"
    if isinstance(arguments, str):
        return arguments
    if isinstance(arguments, dict):
        return json.dumps(arguments)
    return None


def _tool_call_from_payload(payload: dict[str, Any], index: int) -> dict[str, Any] | None:
    function = payload.get("function")
    if isinstance(function, dict):
        name = function.get("name")
        arguments = function.get("arguments")
    else:
        name = payload.get("name")
        arguments = payload.get("arguments")

    if not isinstance(name, str) or not name:
        return None

    serialized_arguments = _serialize_tool_arguments(arguments)
    if serialized_arguments is None:
        return None

    tool_call_id = payload.get("id")
    if not isinstance(tool_call_id, str) or not tool_call_id:
        tool_call_id = f"call_{index}"

    return {
        "id": tool_call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": serialized_arguments,
        },
    }


def parse_tool_calls(text: str) -> list[dict[str, Any]] | None:
    matches = TOOL_CALL_RE.findall(text)
    if not matches:
        return None

    tool_calls: list[dict[str, Any]] = []
    for raw_payload in matches:
        try:
            parsed_payload = json.loads(raw_payload)
        except json.JSONDecodeError:
            return None

        payloads = parsed_payload if isinstance(parsed_payload, list) else [parsed_payload]
        for payload in payloads:
            if not isinstance(payload, dict):
                return None
            tool_call = _tool_call_from_payload(payload, len(tool_calls))
            if tool_call is None:
                return None
            tool_calls.append(tool_call)

    return tool_calls or None


def dss_result_to_oai_choice(result: dict, index: int) -> dict:
    """Convert one DSS /generate result to one OpenAI choice dict.

    DSS result shape: {"text": str, "token_ids": [int], "finish_reason": str,
                       "logprobs": [...]}. OAI choice shape matches vLLM's
    OAI server output so verifiers' parser reads it unchanged.
    """
    text = result.get("text", "")
    tool_calls = parse_tool_calls(text) if isinstance(text, str) else None
    message: dict[str, Any] = {
        "role": "assistant",
        "content": text,
    }
    finish_reason = result.get("finish_reason", "stop")
    if tool_calls is not None:
        message = {
            "role": "assistant",
            "content": None,
            "tool_calls": tool_calls,
        }
        finish_reason = "tool_calls"

    choice: dict[str, Any] = {
        "index": index,
        "message": message,
        "finish_reason": finish_reason,
    }
    if (lp := result.get("logprobs")) is not None:
        choice["logprobs"] = {"content": lp}
    return choice


def dss_results_to_oai_response(results: list[dict], model: str, request_id: str | None = None) -> dict:
    """Shape a full OpenAI chat.completion response from a list of DSS results."""
    return {
        "id": request_id or f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [dss_result_to_oai_choice(r, i) for i, r in enumerate(results)],
    }
