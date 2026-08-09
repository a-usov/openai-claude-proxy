"""Validate representative proxy output with the reviewed Anthropic SDK."""

from __future__ import annotations

import json
from typing import Any

import pytest
from anthropic import __version__ as anthropic_version
from anthropic.pagination import SyncPage
from anthropic.types import (
    ErrorResponse,
    Message,
    MessageTokensCount,
    ModelInfo,
    RawMessageStreamEvent,
    Usage,
)
from pydantic import TypeAdapter

from openai_claude_proxy.conversion import openai_to_anthropic
from openai_claude_proxy.errors import anthropic_error
from openai_claude_proxy.models import model_page
from openai_claude_proxy.streaming import anthropic_message_stream

REVIEWED_ANTHROPIC_VERSION = "0.121.0"


def _message() -> dict[str, Any]:
    return openai_to_anthropic(
        {
            "id": "chatcmpl_conformance",
            "model": "upstream-model",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": "Checking.",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "lookup",
                                    "arguments": '{"query":"synthetic"}',
                                },
                            },
                        ],
                    },
                },
            ],
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 5,
                "prompt_tokens_details": {"cached_tokens": 3},
                "completion_tokens_details": {"reasoning_tokens": 2},
            },
        },
        "claude-conformance",
    )


async def _stream_payloads(message: dict[str, Any]) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    async for encoded in anthropic_message_stream(message):
        data_line = next(
            line for line in encoded.decode().splitlines() if line.startswith("data: ")
        )
        decoded = json.loads(data_line.removeprefix("data: "))
        if not isinstance(decoded, dict):
            raise TypeError("Anthropic SSE data must be an object")
        payloads.append(decoded)
    return payloads


@pytest.mark.anyio
async def test_anthropic_sdk_conformance() -> None:
    assert anthropic_version == REVIEWED_ANTHROPIC_VERSION

    message = _message()
    Message.model_validate(message)
    Usage.model_validate(message["usage"])
    MessageTokensCount.model_validate({"input_tokens": 42})

    stream_adapter = TypeAdapter(RawMessageStreamEvent)
    for event in await _stream_payloads(message):
        stream_adapter.validate_python(event)

    error_payloads = [
        anthropic_error(
            "slow down",
            provider_type="rate_limit_exceeded",
            status_code=429,
            request_id="req_123",
        ),
        anthropic_error("try later", code="server_error"),
        anthropic_error("Token counting is not supported", status_code=501),
    ]
    for payload in error_payloads:
        ErrorResponse.model_validate(payload)

    model_payload = model_page(
        {"claude-gpt-5-6-sol": "GPT-5.6 Sol"},
        {"limit": "1000"},
    )
    SyncPage[ModelInfo].model_validate(model_payload)
