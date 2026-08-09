"""Exercise proxy pass-through wires with the reviewed OpenAI Python SDK."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import httpx
import pytest
from openai import AsyncOpenAI, RateLimitError
from openai import __version__ as openai_version
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.responses import Response, ResponseStreamEvent
from pydantic import TypeAdapter

from openai_claude_proxy.app import create_app
from openai_claude_proxy.config import Settings
from openai_claude_proxy.conversion import anthropic_to_openai, anthropic_to_responses
from tests.support.real_http import LoopbackServer

if TYPE_CHECKING:
    from collections.abc import Sequence

REVIEWED_OPENAI_VERSION = "2.53.0"
RATE_LIMIT_STATUS = 429


def _validate_translated_request_wires() -> None:
    source = {
        "model": "claude-conformance",
        "max_tokens": 128,
        "stream": True,
        "system": "Use tools.",
        "output_config": {"effort": "high"},
        "messages": [{"role": "user", "content": "Synthetic"}],
        "tools": [
            {
                "name": "lookup",
                "description": "Lookup synthetic data",
                "input_schema": {
                    "type": "object",
                    "properties": {"key": {"type": "string"}},
                    "required": ["key"],
                    "additionalProperties": False,
                },
            },
        ],
    }
    settings = Settings(model_map={"claude-conformance": "gpt-conformance"})
    schema = source["tools"][0]["input_schema"]
    if anthropic_to_openai(source, settings) != {
        "model": "gpt-conformance",
        "messages": [
            {"role": "system", "content": "Use tools."},
            {"role": "user", "content": "Synthetic"},
        ],
        "max_tokens": 128,
        "stream": True,
        "reasoning_effort": "high",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "Lookup synthetic data",
                    "parameters": schema,
                },
            },
        ],
        "stream_options": {"include_usage": True},
    }:
        raise AssertionError("Chat Completions translated request wire drifted")
    if anthropic_to_responses(source, settings) != {
        "model": "gpt-conformance",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Synthetic"}],
            },
        ],
        "max_output_tokens": 128,
        "stream": True,
        "store": False,
        "instructions": "Use tools.",
        "reasoning": {"effort": "high"},
        "tools": [
            {
                "type": "function",
                "name": "lookup",
                "description": "Lookup synthetic data",
                "parameters": schema,
            },
        ],
    }:
        raise AssertionError("Responses translated request wire drifted")


def _response_payload(*, status: str, output: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "id": "resp_conformance",
        "created_at": 1_786_000_000.0,
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "metadata": {},
        "model": "gpt-conformance",
        "object": "response",
        "output": output,
        "parallel_tool_calls": True,
        "status": status,
        "tool_choice": "auto",
        "tools": [],
        "usage": {
            "input_tokens": 4,
            "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
            "output_tokens": 1,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 5,
        },
    }


def _output_message() -> dict[str, Any]:
    return {
        "type": "message",
        "id": "msg_conformance",
        "role": "assistant",
        "status": "completed",
        "phase": "final_answer",
        "content": [
            {
                "type": "output_text",
                "text": "Synthetic answer.",
                "annotations": [],
                "logprobs": [],
            },
        ],
    }


def _sse(events: Sequence[dict[str, Any] | str]) -> bytes:
    return "".join(
        f"data: {event if isinstance(event, str) else json.dumps(event)}\n\n" for event in events
    ).encode()


def _chat_payload() -> dict[str, Any]:
    return {
        "id": "chatcmpl_conformance",
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "logprobs": None,
                "message": {"content": "Synthetic answer.", "role": "assistant"},
            },
        ],
        "created": 1_786_000_000,
        "model": "gpt-conformance",
        "object": "chat.completion",
        "usage": {"completion_tokens": 1, "prompt_tokens": 4, "total_tokens": 5},
    }


def _chat_chunks() -> list[dict[str, Any] | str]:
    return [
        {
            "id": "chatcmpl_conformance",
            "choices": [
                {
                    "delta": {"content": "Synthetic answer.", "role": "assistant"},
                    "finish_reason": None,
                    "index": 0,
                },
            ],
            "created": 1_786_000_000,
            "model": "gpt-conformance",
            "object": "chat.completion.chunk",
        },
        {
            "id": "chatcmpl_conformance",
            "choices": [{"delta": {}, "finish_reason": "stop", "index": 0}],
            "created": 1_786_000_000,
            "model": "gpt-conformance",
            "object": "chat.completion.chunk",
        },
        "[DONE]",
    ]


def _response_events() -> list[dict[str, Any]]:
    output = _output_message()
    return [
        {
            "type": "response.created",
            "sequence_number": 0,
            "response": _response_payload(status="in_progress", output=[]),
        },
        {
            "type": "response.output_text.delta",
            "sequence_number": 1,
            "output_index": 0,
            "content_index": 0,
            "item_id": "msg_conformance",
            "delta": "Synthetic answer.",
            "logprobs": [],
        },
        {
            "type": "response.output_text.done",
            "sequence_number": 2,
            "output_index": 0,
            "content_index": 0,
            "item_id": "msg_conformance",
            "text": "Synthetic answer.",
            "logprobs": [],
        },
        {
            "type": "response.completed",
            "sequence_number": 3,
            "response": _response_payload(status="completed", output=[output]),
        },
    ]


async def _upstream(request: httpx.Request) -> httpx.Response:
    if request.url.path == "/v1/models":
        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": [
                    {
                        "id": "gpt-conformance",
                        "created": 1_786_000_000,
                        "object": "model",
                        "owned_by": "synthetic",
                    },
                ],
            },
        )
    body = json.loads(request.content)
    if request.url.path == "/v1/responses":
        if body.get("model") == "error-conformance":
            return httpx.Response(
                RATE_LIMIT_STATUS,
                headers={"x-request-id": "req_error_conformance"},
                json={
                    "error": {
                        "message": "Synthetic rate limit",
                        "type": "rate_limit_error",
                        "param": None,
                        "code": "rate_limit_exceeded",
                    },
                },
            )
        if body.get("stream"):
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=_sse(_response_events()),
            )
        return httpx.Response(
            200, json=_response_payload(status="completed", output=[_output_message()])
        )
    if request.url.path == "/v1/chat/completions":
        if body.get("stream"):
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=_sse(_chat_chunks()),
            )
        return httpx.Response(200, json=_chat_payload())
    if request.url.path == "/v1/embeddings":
        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": [{"object": "embedding", "embedding": [0.0, 1.0], "index": 0}],
                "model": "embedding-conformance",
                "usage": {"prompt_tokens": 1, "total_tokens": 1},
            },
        )
    return httpx.Response(404, json={"error": {"message": "Unknown synthetic route"}})


async def _exercise_client_calls(client: AsyncOpenAI) -> None:
    response = await client.responses.create(model="gpt-conformance", input="Synthetic")
    if response.output_text != "Synthetic answer.":
        raise AssertionError("Unexpected buffered Responses text")

    response_stream = await client.responses.create(
        model="gpt-conformance",
        input="Synthetic",
        stream=True,
    )
    response_event_types = [event.type async for event in response_stream]
    if response_event_types[-1] != "response.completed":
        raise AssertionError("Responses stream did not complete")

    chat = await client.chat.completions.create(
        model="gpt-conformance",
        messages=[{"role": "user", "content": "Synthetic"}],
    )
    if chat.choices[0].message.content != "Synthetic answer.":
        raise AssertionError("Unexpected buffered Chat text")

    chat_stream = await client.chat.completions.create(
        model="gpt-conformance",
        messages=[{"role": "user", "content": "Synthetic"}],
        stream=True,
    )
    chat_chunks = [chunk async for chunk in chat_stream]
    if chat_chunks[-1].choices[0].finish_reason != "stop":
        raise AssertionError("Chat stream did not stop")

    models = await client.models.list()
    if models.data[0].id != "gpt-conformance":
        raise AssertionError("Model pass-through failed")
    embedding = await client.embeddings.create(
        model="embedding-conformance",
        input="Synthetic",
    )
    if embedding.data[0].embedding != [0.0, 1.0]:
        raise AssertionError("Generic pass-through failed")

    try:
        await client.responses.create(model="error-conformance", input="Synthetic")
    except RateLimitError as exc:
        if exc.status_code != RATE_LIMIT_STATUS or exc.request_id != "req_error_conformance":
            raise AssertionError("OpenAI error metadata was not preserved") from exc
        if exc.code != "rate_limit_exceeded" or exc.type != "rate_limit_error":
            raise AssertionError("OpenAI error body was not parsed as expected") from exc
    else:
        raise AssertionError("Expected the synthetic OpenAI rate-limit response")


def test_openai_sdk_schema_conformance() -> None:
    assert openai_version == REVIEWED_OPENAI_VERSION
    ChatCompletion.model_validate(_chat_payload())
    chat_chunk_adapter = TypeAdapter(ChatCompletionChunk)
    for chunk in _chat_chunks()[:-1]:
        chat_chunk_adapter.validate_python(chunk)
    Response.model_validate(_response_payload(status="completed", output=[_output_message()]))
    response_event_adapter = TypeAdapter(ResponseStreamEvent)
    for event in _response_events():
        response_event_adapter.validate_python(event)
    _validate_translated_request_wires()


@pytest.mark.anyio
async def test_openai_sdk_client_conformance_over_real_http() -> None:
    app = create_app(transport=httpx.MockTransport(_upstream))
    try:
        async with (
            LoopbackServer(app) as server,
            AsyncOpenAI(
                api_key="synthetic-conformance-key",
                base_url=f"{server.base_url}/v1",
                max_retries=0,
            ) as client,
        ):
            await _exercise_client_calls(client)
    except PermissionError:
        raise pytest.skip.Exception(
            "The execution sandbox does not permit loopback listening sockets",
        ) from None
