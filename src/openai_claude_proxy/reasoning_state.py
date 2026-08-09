"""Stateless transport for opaque OpenAI Responses continuation state."""

from __future__ import annotations

import base64
import binascii
import json
from dataclasses import dataclass
from typing import Any

_REASONING_MAGIC = b"openai-claude-proxy:responses-reasoning:v1\x00"
_OUTPUT_MAGIC = b"openai-claude-proxy:responses-output:v1\x00"
_OUTPUT_MAGIC_V2 = b"openai-claude-proxy:responses-output:v2\x00"


@dataclass(frozen=True, slots=True)
class ResponsesOutputState:
    """Carry exact provider output and the upstream model it belongs to."""

    output: list[dict[str, Any]]
    model: str | None


def _encode(value: object, magic: bytes) -> str:
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode()
    return base64.b64encode(magic + serialized).decode("ascii")


def _decode(data: object, magic: bytes, error_message: str) -> object | None:
    if not isinstance(data, str):
        return None
    try:
        decoded = base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError):
        return None
    if not decoded.startswith(magic):
        return None
    try:
        return json.loads(decoded[len(magic) :])
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(error_message) from exc


def encode_responses_reasoning(item: dict[str, Any]) -> str:
    """Encode one OpenAI reasoning item as opaque Anthropic block data."""
    if item.get("type") != "reasoning":
        raise ValueError("Only Responses reasoning items can be encoded")
    return _encode(item, _REASONING_MAGIC)


def decode_responses_reasoning(data: object) -> dict[str, Any] | None:
    """Decode proxy-owned state, or return None for another provider's block."""
    item = _decode(
        data,
        _REASONING_MAGIC,
        "Invalid proxy Responses reasoning state",
    )
    if item is None:
        return None
    if not isinstance(item, dict) or item.get("type") != "reasoning":
        raise ValueError("Invalid proxy Responses reasoning state")
    return item


def encode_responses_output(
    items: list[dict[str, Any]],
    *,
    model: str | None = None,
) -> str:
    """Encode the exact ordered Responses output used for a later stateless turn."""
    if not all(isinstance(item, dict) and isinstance(item.get("type"), str) for item in items):
        raise ValueError("Responses output state must contain typed objects")
    if model is not None:
        if not model:
            raise ValueError("Responses output state model must be non-empty")
        return _encode({"model": model, "output": items}, _OUTPUT_MAGIC_V2)
    return _encode(items, _OUTPUT_MAGIC)


def decode_responses_output_state(data: object) -> ResponsesOutputState | None:
    """Decode model-bound output state while accepting legacy unbound carriers."""
    version_two = _decode(
        data,
        _OUTPUT_MAGIC_V2,
        "Invalid proxy Responses output state",
    )
    if version_two is not None:
        if not isinstance(version_two, dict):
            raise ValueError("Invalid proxy Responses output state")
        model = version_two.get("model")
        items = version_two.get("output")
        if not isinstance(model, str) or not model:
            raise ValueError("Invalid proxy Responses output state")
        if not isinstance(items, list) or not all(
            isinstance(item, dict) and isinstance(item.get("type"), str) for item in items
        ):
            raise ValueError("Invalid proxy Responses output state")
        return ResponsesOutputState(output=items, model=model)

    items = _decode(
        data,
        _OUTPUT_MAGIC,
        "Invalid proxy Responses output state",
    )
    if items is None:
        return None
    if not isinstance(items, list) or not all(
        isinstance(item, dict) and isinstance(item.get("type"), str) for item in items
    ):
        raise ValueError("Invalid proxy Responses output state")
    return ResponsesOutputState(output=items, model=None)


def decode_responses_output(data: object) -> list[dict[str, Any]] | None:
    """Decode proxy-owned complete output items, or ignore another carrier."""
    state = decode_responses_output_state(data)
    return None if state is None else state.output
