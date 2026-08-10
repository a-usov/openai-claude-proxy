from __future__ import annotations

import asyncio
import base64
import json
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

import httpx
import pytest

from openai_claude_proxy.app import create_app
from openai_claude_proxy.config import Settings
from openai_claude_proxy.models import automatic_model_alias
from openai_claude_proxy.reasoning_state import (
    encode_responses_output,
    encode_responses_reasoning,
)
from openai_claude_proxy.streaming import (
    openai_stream_to_anthropic,
    responses_stream_to_anthropic,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


def event_stream(chunks: list[dict[str, object] | str]) -> bytes:
    lines = []
    for chunk in chunks:
        data = chunk if isinstance(chunk, str) else json.dumps(chunk)
        lines.append(f"data: {data}\n\n")
    return "".join(lines).encode()


class SlowResponseStream(httpx.AsyncByteStream):
    """Delay the first byte so heartbeat behavior can be observed."""

    async def __aiter__(self) -> AsyncIterator[bytes]:
        """Yield one terminal event after a short artificial pause."""
        await asyncio.sleep(0.02)
        yield event_stream(
            [
                {
                    "type": "response.completed",
                    "response": {
                        "id": "resp_slow",
                        "status": "completed",
                        "output": [
                            {
                                "type": "message",
                                "content": [{"type": "output_text", "text": "done"}],
                            },
                        ],
                        "usage": {"input_tokens": 1, "output_tokens": 1},
                    },
                },
            ],
        )


class IncrementalChatToolStream(httpx.AsyncByteStream):
    """Expose whether the translator consumed the terminal upstream chunk."""

    def __init__(self) -> None:
        """Initialize with an unread terminal chunk."""
        self.terminal_chunk_read = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        """Yield tool arguments before exposing the terminal chunk."""
        yield event_stream(
            [
                {
                    "id": "chatcmpl_live_tool",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_1",
                                        "function": {
                                            "name": "read",
                                            "arguments": '{"path":"README.md"}',
                                        },
                                    },
                                ],
                            },
                        },
                    ],
                },
            ],
        )
        self.terminal_chunk_read = True
        yield event_stream(
            [
                {
                    "id": "chatcmpl_live_tool",
                    "choices": [
                        {"index": 0, "delta": {}, "finish_reason": "tool_calls"},
                    ],
                },
                "[DONE]",
            ],
        )


class DisconnectingResponseStream(httpx.AsyncByteStream):
    """Yield optional bytes and then simulate an upstream socket failure."""

    def __init__(self, chunks: list[bytes]) -> None:
        """Store chunks that precede the disconnect."""
        self.chunks = chunks
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        """Yield configured chunks and then fail like an httpx transport."""
        for chunk in self.chunks:
            yield chunk
        raise httpx.ReadError("sensitive transport detail")

    async def aclose(self) -> None:
        """Record response cleanup."""
        self.closed = True


class BlockingResponseStream(httpx.AsyncByteStream):
    """Wait forever so downstream cancellation can interrupt an upstream read."""

    def __init__(self) -> None:
        """Initialize lifecycle observations."""
        self.read_started = asyncio.Event()
        self.never_release = asyncio.Event()
        self.cancelled = False
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        """Block on the first read and observe its cancellation."""
        self.read_started.set()
        try:
            await self.never_release.wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        yield b""

    async def aclose(self) -> None:
        """Record response cleanup."""
        self.closed = True


@pytest.mark.anyio
async def test_messages_forwards_helper_key_as_bearer_and_maps_response() -> None:
    seen: dict[str, Any] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["authorization"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            headers={
                "x-ratelimit-limit-requests": "100",
                "anthropic-ratelimit-tokens-remaining": "9000",
                "retry-after-ms": "250",
                "x-should-retry": "false",
                "set-cookie": "must-not-forward=1",
            },
            json={
                "id": "chatcmpl_test",
                "model": "work-model",
                "choices": [
                    {"message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"},
                ],
                "usage": {"prompt_tokens": 4, "completion_tokens": 1},
            },
        )

    app = create_app(
        Settings(
            upstream_base_url="https://foundry.example/openai/v1",
            auth_mode="bearer",
            model_override="work-model",
            upstream_query={"api-version": "2025-01-01"},
        ),
        transport=httpx.MockTransport(handler),
    )
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://proxy",
        ) as client,
    ):
        response = await client.post(
            "/v1/messages",
            headers={"x-api-key": "from-helper"},
            json={
                "model": "claude-anything",
                "max_tokens": 100,
                "output_config": {"effort": "high"},
                "messages": [{"role": "user", "content": "hi"}],
            },
        )

    assert response.status_code == 200
    assert seen["authorization"] == "Bearer from-helper"
    assert (
        seen["url"] == "https://foundry.example/openai/v1/chat/completions?api-version=2025-01-01"
    )
    assert seen["body"]["model"] == "work-model"
    assert seen["body"]["reasoning_effort"] == "high"
    assert response.json()["content"] == [{"type": "text", "text": "hello"}]
    assert response.headers["x-ratelimit-limit-requests"] == "100"
    assert response.headers["anthropic-ratelimit-tokens-remaining"] == "9000"
    assert response.headers["retry-after-ms"] == "250"
    assert response.headers["x-should-retry"] == "false"
    assert "set-cookie" not in response.headers


@pytest.mark.anyio
async def test_disagreeing_helper_credentials_return_sanitized_authentication_error() -> None:
    async def unexpected_handler(_request: httpx.Request) -> httpx.Response:
        msg = "Ambiguous credentials must not reach the upstream"
        raise AssertionError(msg)

    app = create_app(
        Settings(auth_mode="bearer"),
        transport=httpx.MockTransport(unexpected_handler),
    )
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
            base_url="http://proxy",
        ) as client,
    ):
        response = await client.post(
            "/v1/messages",
            headers={
                "authorization": "Bearer first-sensitive-key",
                "x-api-key": "second-sensitive-key",
            },
            json={"model": "claude", "max_tokens": 10, "messages": []},
        )

    assert response.status_code == 401
    assert response.json() == {
        "type": "error",
        "error": {
            "type": "authentication_error",
            "message": "Credential headers disagree",
        },
    }
    assert "first-sensitive-key" not in response.text
    assert "second-sensitive-key" not in response.text


@pytest.mark.anyio
async def test_streaming_text_and_tool_call_become_anthropic_events() -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={
                "content-type": "text/event-stream",
                "x-ratelimit-remaining-requests": "99",
            },
            content=event_stream(
                [
                    {
                        "id": "chatcmpl_stream",
                        "model": "work-model",
                        "choices": [{"delta": {"role": "assistant"}, "finish_reason": None}],
                    },
                    {
                        "id": "chatcmpl_stream",
                        "model": "work-model",
                        "choices": [{"delta": {"content": "Let me "}, "finish_reason": None}],
                    },
                    {
                        "id": "chatcmpl_stream",
                        "model": "work-model",
                        "choices": [
                            {
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "id": "call_7",
                                            "function": {"name": "shell", "arguments": '{"cmd":'},
                                        },
                                    ],
                                },
                                "finish_reason": None,
                            },
                        ],
                    },
                    {
                        "id": "chatcmpl_stream",
                        "choices": [
                            {
                                "delta": {
                                    "tool_calls": [
                                        {"index": 0, "function": {"arguments": '"ls"}'}}
                                    ],
                                },
                                "finish_reason": "tool_calls",
                            },
                        ],
                    },
                    {
                        "choices": [],
                        "usage": {
                            "prompt_tokens": 12,
                            "completion_tokens": 5,
                            "completion_tokens_details": {"reasoning_tokens": 3},
                        },
                    },
                    "[DONE]",
                ],
            ),
        )

    app = create_app(Settings(), transport=httpx.MockTransport(handler))
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://proxy",
        ) as client,
    ):
        response = await client.post(
            "/v1/messages",
            json={
                "model": "claude",
                "max_tokens": 100,
                "stream": True,
                "messages": [{"role": "user", "content": "go"}],
            },
        )

    assert response.status_code == 200
    assert response.headers["x-ratelimit-remaining-requests"] == "99"
    body = response.text
    assert "event: message_start" in body
    assert '"type":"text_delta","text":"Let me "' in body
    assert '"id":"call_7","name":"shell"' in body
    assert '"partial_json":"{\\"cmd\\":\\"ls\\"}"' in body
    assert '"stop_reason":"tool_use"' in body
    assert '"output_tokens":5' in body
    assert '"output_tokens_details":{"thinking_tokens":3}' in body
    assert body.endswith('event: message_stop\ndata: {"type":"message_stop"}\n\n')


@pytest.mark.anyio
async def test_chat_nonparallel_tool_arguments_stream_before_upstream_finishes() -> None:
    upstream_stream = IncrementalChatToolStream()
    response = httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        stream=upstream_stream,
    )
    translated = openai_stream_to_anthropic(
        response,
        "claude-test",
        request_payload={"parallel_tool_calls": False},
    )

    early_chunks = [await anext(translated) for _index in range(3)]
    early_body = b"".join(early_chunks).decode()

    assert '"id":"call_1","name":"read"' in early_body
    assert '"partial_json":"{\\"path\\":\\"README.md\\"}"' in early_body
    assert upstream_stream.terminal_chunk_read is False

    remaining_body = b"".join([chunk async for chunk in translated]).decode()
    assert upstream_stream.terminal_chunk_read is True
    assert '"stop_reason":"tool_use"' in remaining_body
    assert "event: message_stop" in remaining_body


@pytest.mark.anyio
async def test_stream_translators_close_upstream_when_downstream_is_cancelled() -> None:
    for converter in (openai_stream_to_anthropic, responses_stream_to_anthropic):
        upstream_stream = BlockingResponseStream()
        response = httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=upstream_stream,
        )
        translated = converter(response, "claude-test", ping_interval=60)
        read_task = asyncio.ensure_future(anext(translated))
        await upstream_stream.read_started.wait()

        read_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await read_task

        assert upstream_stream.cancelled
        assert upstream_stream.closed
        assert response.is_closed


@pytest.mark.anyio
@pytest.mark.parametrize("protocol", ["chat", "responses"])
@pytest.mark.parametrize("after_start", [False, True])
async def test_stream_disconnects_emit_safe_errors_and_close_response(
    protocol: str,
    *,
    after_start: bool,
) -> None:
    if protocol == "chat":
        chunks = (
            [
                event_stream(
                    [
                        {
                            "id": "chatcmpl_disconnect",
                            "choices": [
                                {
                                    "delta": {"content": "partial"},
                                    "finish_reason": None,
                                },
                            ],
                        },
                    ],
                ),
            ]
            if after_start
            else []
        )
        converter = openai_stream_to_anthropic
    else:
        chunks = (
            [
                event_stream(
                    [
                        {
                            "type": "response.created",
                            "response": {"id": "resp_disconnect"},
                        },
                    ],
                ),
            ]
            if after_start
            else []
        )
        converter = responses_stream_to_anthropic
    upstream_stream = DisconnectingResponseStream(chunks)
    response = httpx.Response(200, stream=upstream_stream)

    body = b"".join([chunk async for chunk in converter(response, "claude-test")]).decode()

    assert ("event: message_start" in body) is after_start
    assert "event: error" in body
    assert "upstream stream disconnected" in body
    assert "sensitive transport detail" not in body
    assert "event: message_stop" not in body
    assert upstream_stream.closed
    assert response.is_closed


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("protocol", "raw_event", "expected_message"),
    [
        ("chat", b"data: {not-json}\n\n", "Malformed JSON in Chat Completions"),
        ("responses", b"data: {not-json}\n\n", "Malformed JSON in Responses"),
        (
            "responses",
            event_stream([{"type": "response.future_event", "secret": "must-not-appear"}]),
            "Unsupported Responses stream event type",
        ),
    ],
)
async def test_malformed_and_unknown_stream_events_are_controlled_errors(
    protocol: str,
    raw_event: bytes,
    expected_message: str,
) -> None:
    upstream_stream = DisconnectingResponseStream([raw_event])
    response = httpx.Response(200, stream=upstream_stream)
    converter = openai_stream_to_anthropic if protocol == "chat" else responses_stream_to_anthropic

    body = b"".join([chunk async for chunk in converter(response, "claude-test")]).decode()

    assert "event: error" in body
    assert expected_message in body
    assert "must-not-appear" not in body
    assert upstream_stream.closed
    assert response.is_closed


@pytest.mark.anyio
async def test_responses_empty_completed_lifecycle_is_reported_as_an_error() -> None:
    response = httpx.Response(
        200,
        content=event_stream(
            [
                {"type": "response.queued", "response": {"id": "resp_lifecycle"}},
                {"type": "response.in_progress", "response": {"id": "resp_lifecycle"}},
                {"type": "response.created", "response": {"id": "resp_lifecycle"}},
                {
                    "type": "response.completed",
                    "response": {
                        "id": "resp_lifecycle",
                        "status": "completed",
                        "output": [],
                        "usage": {"input_tokens": 1, "output_tokens": 1},
                    },
                },
            ],
        ),
    )

    body = b"".join(
        [chunk async for chunk in responses_stream_to_anthropic(response, "claude-test")],
    ).decode()

    assert "event: error" in body
    assert "event: message_start" in body
    assert "Responses API completed without assistant text or a function call" in body
    assert "event: message_stop" not in body
    assert response.is_closed


@pytest.mark.anyio
async def test_openai_route_is_transparently_forwarded() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "catalog.example"
        assert request.url.path == "/v1/models"
        assert request.headers["x-api-key"] == "helper-key"
        return httpx.Response(
            200,
            headers={"x-ratelimit-limit-requests": "100"},
            json={"object": "list", "data": []},
        )

    app = create_app(
        Settings(
            upstream_base_url="https://gateway.example/v1/proxy",
            upstream_models_base_url="https://catalog.example/v1",
            auth_mode="passthrough",
        ),
        transport=httpx.MockTransport(handler),
    )
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://proxy",
        ) as client,
    ):
        response = await client.get("/v1/models", headers={"x-api-key": "helper-key"})

    assert response.status_code == 200
    assert response.headers["x-ratelimit-limit-requests"] == "100"
    assert response.json() == {"object": "list", "data": []}


@pytest.mark.anyio
async def test_model_discovery_rewrites_x_api_key_helper_credential() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/models"
        assert request.headers["authorization"] == "Bearer model-helper-key"
        assert "x-api-key" not in request.headers
        return httpx.Response(200, json={"object": "list", "data": []})

    app = create_app(
        Settings(
            upstream_base_url="https://gateway.example/v1",
            auth_mode="bearer",
        ),
        transport=httpx.MockTransport(handler),
    )
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://proxy",
        ) as client,
    ):
        response = await client.get("/v1/models", headers={"x-api-key": "model-helper-key"})

    assert response.status_code == 200


@pytest.mark.anyio
async def test_automatic_model_discovery_aliases_and_routes_upstream_models() -> None:
    catalog_requests = 0
    luna_id = "gateway-gpt-5-6-luna"
    kimi_id = "vendor/kimi-k2.5"
    luna_alias = automatic_model_alias(luna_id)
    kimi_alias = automatic_model_alias(kimi_id)

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal catalog_requests
        assert request.headers["authorization"] == "Bearer helper-key"
        if request.url.host == "catalog.example":
            catalog_requests += 1
            assert request.method == "GET"
            assert request.url.path == "/v1/models"
            assert dict(request.url.params) == {"api-version": "test-version"}
            return httpx.Response(
                200,
                headers={"x-request-id": "catalog-request-id"},
                json={
                    "object": "list",
                    "data": [
                        {
                            "id": luna_id,
                            "model_name": "gpt-5.6-luna",
                            "object": "model",
                            "display_name": "Gateway Luna",
                        },
                        {
                            "id": kimi_id,
                            "model_name": "kimi-k2.5",
                            "object": "model",
                            "display_name": "Kimi K2.5",
                        },
                        {
                            "id": "text-embedding-3-large",
                            "object": "model",
                        },
                    ],
                },
            )
        assert request.url.host == "gateway.example"
        assert request.url.path == "/v1/proxy/responses"
        body = json.loads(request.content)
        assert body["model"] == kimi_id
        return httpx.Response(
            200,
            json={
                "id": "resp_auto_model",
                "model": kimi_id,
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "id": "msg_auto_model",
                        "role": "assistant",
                        "status": "completed",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "Automatic model routing works.",
                                "annotations": [],
                            },
                        ],
                    },
                ],
                "usage": {"input_tokens": 3, "output_tokens": 4},
            },
        )

    app = create_app(
        Settings(
            upstream_base_url="https://gateway.example/v1/proxy",
            upstream_models_base_url="https://catalog.example/v1",
            upstream_models_path="/models",
            upstream_responses_path="/responses",
            upstream_query={"api-version": "test-version"},
            openai_api="responses",
            auth_mode="bearer",
            model_discovery_mode="auto",
            model_discovery_include=("gateway-gpt-*", "vendor/*"),
            model_discovery_exclude=("text-embedding-*",),
            model_discovery={luna_alias: "GPT-5.6 Luna"},
            model_map={luna_alias: luna_id},
        ),
        transport=httpx.MockTransport(handler),
    )
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://proxy",
        ) as client,
    ):
        models = await client.get(
            "/v1/models",
            params={"limit": 100},
            headers={"x-api-key": "helper-key"},
        )
        retrieved = await client.get(
            f"/v1/models/{quote(kimi_alias, safe='')}",
            headers={"x-api-key": "helper-key"},
        )
        completion = await client.post(
            "/v1/messages",
            headers={"x-api-key": "helper-key"},
            json={
                "model": kimi_alias,
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

    assert models.status_code == 200
    assert models.headers["x-request-id"] == "catalog-request-id"
    assert models.json()["data"] == [
        {
            "id": luna_alias,
            "display_name": "GPT-5.6 Luna",
            "type": "model",
            "created_at": "1970-01-01T00:00:00Z",
        },
        {
            "id": kimi_alias,
            "display_name": "Kimi K2.5",
            "type": "model",
            "created_at": "1970-01-01T00:00:00Z",
        },
    ]
    assert retrieved.status_code == 200
    assert retrieved.json()["id"] == kimi_alias
    assert completion.status_code == 200
    assert completion.json()["model"] == kimi_alias
    assert (
        next(block["text"] for block in completion.json()["content"] if block["type"] == "text")
        == "Automatic model routing works."
    )
    assert catalog_requests == 2


@pytest.mark.anyio
async def test_automatic_model_discovery_rejects_malformed_upstream_list() -> None:
    upstream: httpx.Response | None = None

    async def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal upstream
        upstream = httpx.Response(
            200,
            json={
                "object": "list",
                "data": [{"id": 7, "secret": "must-not-appear"}],
            },
        )
        return upstream

    app = create_app(
        Settings(model_discovery_mode="auto"),
        transport=httpx.MockTransport(handler),
    )
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://proxy",
        ) as client,
    ):
        response = await client.get("/v1/models")

    assert response.status_code == 502
    assert response.json()["error"]["message"] == ("Invalid response from upstream model discovery")
    assert "must-not-appear" not in response.text
    assert upstream is not None
    assert upstream.is_closed


@pytest.mark.anyio
async def test_automatic_model_discovery_preserves_upstream_errors() -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            headers={"retry-after": "4", "x-request-id": "catalog-error-id"},
            json={
                "error": {
                    "type": "rate_limit_error",
                    "message": "Model catalog rate limit reached",
                },
            },
        )

    app = create_app(
        Settings(model_discovery_mode="auto"),
        transport=httpx.MockTransport(handler),
    )
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://proxy",
        ) as client,
    ):
        response = await client.get("/v1/models")

    assert response.status_code == 429
    assert response.headers["retry-after"] == "4"
    assert response.json() == {
        "type": "error",
        "error": {
            "type": "rate_limit_error",
            "message": "Model catalog rate limit reached",
        },
        "request_id": "catalog-error-id",
    }


@pytest.mark.anyio
async def test_claude_code_helper_request_with_thinking_reaches_prefixed_upstream() -> None:
    seen: dict[str, Any] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl_title",
                "model": "gpt-test",
                "choices": [
                    {
                        "message": {"role": "assistant", "content": '{"title":"Test"}'},
                        "finish_reason": "stop",
                    },
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 4},
            },
        )

    app = create_app(
        Settings(
            upstream_base_url="https://gateway.example/v1/proxy",
            model_override="gpt-test",
        ),
        transport=httpx.MockTransport(handler),
    )
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://proxy",
        ) as client,
    ):
        response = await client.post(
            "/v1/messages?beta=true",
            json={
                "model": "claude-opus-4-6",
                "messages": [{"role": "user", "content": "Create a title"}],
                "system": [{"type": "text", "text": "Return a JSON title"}],
                "max_tokens": 64000,
                "thinking": {"type": "disabled"},
                "temperature": 1,
                "output_config": {
                    "effort": "high",
                    "format": {
                        "type": "json_schema",
                        "schema": {
                            "type": "object",
                            "properties": {"title": {"type": "string"}},
                            "required": ["title"],
                            "additionalProperties": False,
                        },
                    },
                },
            },
        )

    assert response.status_code == 200
    assert seen["url"] == "https://gateway.example/v1/proxy/chat/completions"
    assert seen["body"]["reasoning_effort"] == "high"
    assert "thinking" not in seen["body"]


@pytest.mark.anyio
async def test_claude_code_cached_system_request_reaches_responses_upstream() -> None:
    seen: dict[str, Any] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "resp_claude_code",
                "model": "gpt-test",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "id": "msg_claude_code",
                        "role": "assistant",
                        "status": "completed",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "Ready.",
                                "annotations": [],
                            },
                        ],
                    },
                ],
                "usage": {"input_tokens": 10, "output_tokens": 2},
            },
        )

    app = create_app(
        Settings(
            upstream_base_url="https://gateway.example/v1/proxy",
            openai_api="responses",
            model_override="gpt-test",
        ),
        transport=httpx.MockTransport(handler),
    )
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://proxy",
        ) as client,
    ):
        response = await client.post(
            "/v1/messages?beta=true",
            json={
                "model": "claude-opus-4-6",
                "messages": [{"role": "user", "content": "Hello"}],
                "system": [
                    {"type": "text", "text": "Billing metadata"},
                    {
                        "type": "text",
                        "text": "Stable system prompt",
                        "cache_control": {"type": "ephemeral"},
                    },
                ],
                "max_tokens": 64000,
                "thinking": {"type": "adaptive", "display": "omitted"},
                "context_management": {
                    "edits": [{"type": "clear_thinking_20251015", "keep": "all"}],
                },
                "output_config": {"effort": "high"},
            },
        )

    assert response.status_code == 200
    assert seen["url"] == "https://gateway.example/v1/proxy/responses"
    assert seen["body"]["input"][0] == {
        "type": "message",
        "role": "developer",
        "content": [
            {"type": "input_text", "text": "Billing metadata"},
            {
                "type": "input_text",
                "text": "Stable system prompt",
                "prompt_cache_breakpoint": {"mode": "explicit"},
            },
        ],
    }
    assert seen["body"]["prompt_cache_options"] == {"mode": "explicit", "ttl": "30m"}


@pytest.mark.anyio
async def test_claude_code_hello_probe_is_handled_locally() -> None:
    upstream_called = False

    async def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal upstream_called
        upstream_called = True
        return httpx.Response(500)

    app = create_app(Settings(), transport=httpx.MockTransport(handler))
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://proxy",
        ) as client,
    ):
        head_response = await client.head("/api/hello")
        get_response = await client.get("/api/hello")

    assert head_response.status_code == 200
    assert head_response.headers["content-type"] == "application/json"
    assert head_response.headers["content-length"] == "20"
    assert head_response.content == b""
    assert get_response.status_code == 200
    assert get_response.json() == {"message": "hello"}
    assert upstream_called is False


@pytest.mark.anyio
async def test_anthropic_upstream_is_passthrough() -> None:
    original = {
        "model": "claude",
        "messages": [
            {"role": "system", "content": "Mid-conversation policy"},
            {"role": "user", "content": "Hello"},
        ],
        "max_tokens": 10,
        "stream": False,
        "thinking": {"type": "adaptive"},
        "context_management": {"edits": []},
        "service_tier": "standard_only",
        "container": "container_1",
        "inference_geo": "us",
        "anthropic-user-profile-id": "profile_1",
    }
    original_bytes = json.dumps(original, indent=2).encode()

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/anthropic/v1/messages"
        assert request.content == original_bytes
        assert request.headers["anthropic-future-feature"] == "enabled"
        assert request.headers["x-claude-code-future-context"] == "context"
        return httpx.Response(200, json={"type": "message", "content": []})

    app = create_app(
        Settings(
            upstream_base_url="https://gateway.example/anthropic/v1",
            upstream_protocol="anthropic",
        ),
        transport=httpx.MockTransport(handler),
    )
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://proxy",
        ) as client,
    ):
        response = await client.post(
            "/v1/messages",
            headers={
                "anthropic-future-feature": "enabled",
                "content-type": "application/json",
                "x-claude-code-future-context": "context",
            },
            content=original_bytes,
        )

    assert response.status_code == 200
    assert response.json() == {"type": "message", "content": []}


@pytest.mark.anyio
async def test_auto_protocol_routes_mapped_claude_and_non_claude_models() -> None:
    seen_protocols: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if request.url.path == "/v1/proxy/v1/messages":
            seen_protocols.append("anthropic")
            assert body["model"] == "gateway-claude-opus"
            assert body["future_anthropic_field"] == {"enabled": True}
            assert dict(request.url.params) == {"beta": "true", "tenant": "example"}
            assert request.headers["anthropic-version"] == "2023-06-01"
            assert request.headers["anthropic-beta"] == "future-beta"
            assert "openai-organization" not in request.headers
            return httpx.Response(
                200,
                json={
                    "id": "msg_auto_anthropic",
                    "type": "message",
                    "role": "assistant",
                    "model": "gateway-claude-opus",
                    "content": [{"type": "text", "text": "Anthropic route"}],
                    "stop_reason": "end_turn",
                    "stop_sequence": None,
                    "usage": {"input_tokens": 3, "output_tokens": 2},
                },
            )

        assert request.url.path == "/v1/proxy/v1/responses"
        seen_protocols.append("openai")
        assert body["model"] == "gateway-gpt-reasoning"
        assert dict(request.url.params) == {"tenant": "example"}
        assert request.headers["openai-organization"] == "org_example"
        assert "anthropic-version" not in request.headers
        assert "anthropic-beta" not in request.headers
        return httpx.Response(
            200,
            json={
                "id": "resp_auto_openai",
                "model": "gateway-gpt-reasoning",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "OpenAI route"}],
                    },
                ],
                "usage": {"input_tokens": 3, "output_tokens": 2},
            },
        )

    app = create_app(
        Settings(
            upstream_base_url="https://gateway.example/v1/proxy",
            upstream_protocol="auto",
            openai_api="responses",
            upstream_messages_path="/v1/messages",
            upstream_responses_path="/v1/responses",
            upstream_query={"tenant": "example"},
            auth_mode="bearer",
            model_map={
                "claude-opus-*": "gateway-claude-opus",
                "claude-sonnet-*": "gateway-gpt-reasoning",
            },
        ),
        transport=httpx.MockTransport(handler),
    )
    headers = {
        "x-api-key": "helper-secret",
        "anthropic-version": "2023-06-01",
        "anthropic-beta": "future-beta",
        "openai-organization": "org_example",
    }
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://proxy",
        ) as client,
    ):
        anthropic = await client.post(
            "/v1/messages?beta=true",
            headers=headers,
            json={
                "model": "claude-opus-5",
                "max_tokens": 32,
                "messages": [{"role": "user", "content": "Hello"}],
                "future_anthropic_field": {"enabled": True},
            },
        )
        translated = await client.post(
            "/v1/messages?beta=true",
            headers=headers,
            json={
                "model": "claude-sonnet-4-6",
                "max_tokens": 32,
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

    assert anthropic.status_code == translated.status_code == 200
    assert anthropic.json()["content"] == [{"type": "text", "text": "Anthropic route"}]
    assert translated.json()["content"][1] == {"type": "text", "text": "OpenAI route"}
    assert seen_protocols == ["anthropic", "openai"]


@pytest.mark.anyio
async def test_auto_protocol_streams_mapped_anthropic_models_without_translation() -> None:
    upstream_response: httpx.Response | None = None
    sse = (
        b'event: message_start\ndata: {"type":"message_start","message":{"model":"team-claude"}}\n\n'
        b'event: message_stop\ndata: {"type":"message_stop"}\n\n'
    )

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal upstream_response
        assert request.url.path == "/v1/messages"
        assert json.loads(request.content)["model"] == "team-claude"
        upstream_response = httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=sse,
        )
        return upstream_response

    app = create_app(
        Settings(
            upstream_base_url="https://gateway.example",
            upstream_protocol="auto",
            upstream_messages_path="/v1/messages",
            model_map={"claude-opus-*": "team-claude"},
        ),
        transport=httpx.MockTransport(handler),
    )
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://proxy",
        ) as client,
    ):
        response = await client.post(
            "/v1/messages",
            json={
                "model": "claude-opus-5",
                "max_tokens": 32,
                "stream": True,
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

    assert response.status_code == 200
    assert response.content == sse
    assert upstream_response is not None
    assert upstream_response.is_closed


@pytest.mark.anyio
@pytest.mark.parametrize("openai_api", ["chat_completions", "responses"])
async def test_malformed_anthropic_requests_return_400_before_upstream(openai_api: str) -> None:
    async def unexpected_handler(_request: httpx.Request) -> httpx.Response:
        msg = "Malformed client input must not reach the upstream"
        raise AssertionError(msg)

    app = create_app(
        Settings(openai_api=openai_api),
        transport=httpx.MockTransport(unexpected_handler),
    )
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://proxy",
        ) as client,
    ):
        response = await client.post(
            "/v1/messages",
            json={
                "model": "claude-test",
                "max_tokens": 100,
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "image", "source": {"type": []}}],
                    },
                ],
            },
        )

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert response.json()["error"]["message"] == (
        "messages[0].content[0].source.type must be 'base64' or 'url'"
    )


@pytest.mark.anyio
@pytest.mark.parametrize("upstream_protocol", ["openai", "anthropic"])
async def test_oversized_message_requests_return_413_before_upstream(
    upstream_protocol: str,
) -> None:
    async def unexpected_handler(_request: httpx.Request) -> httpx.Response:
        msg = "Oversized client input must not reach the upstream"
        raise AssertionError(msg)

    async def request_chunks() -> AsyncIterator[bytes]:
        yield b'{"model":"claude","max_tokens":10,"messages":[{"role":"user","content":"'
        yield b"x" * 100
        yield b'"}]}'

    app = create_app(
        Settings(upstream_protocol=upstream_protocol, max_request_body_bytes=64),
        transport=httpx.MockTransport(unexpected_handler),
    )
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://proxy",
        ) as client,
    ):
        response = await client.post(
            "/v1/messages",
            headers={"content-type": "application/json"},
            content=request_chunks(),
        )

    assert response.status_code == 413
    assert response.json() == {
        "type": "error",
        "error": {
            "type": "invalid_request_error",
            "message": "Request body exceeds the configured size limit",
        },
    }


@pytest.mark.anyio
async def test_oversized_buffered_upstream_response_returns_502_and_is_closed() -> None:
    seen: dict[str, httpx.Response] = {}

    async def handler(_request: httpx.Request) -> httpx.Response:
        response = httpx.Response(
            200,
            json={
                "id": "chatcmpl_large",
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "x" * 500},
                        "finish_reason": "stop",
                    },
                ],
            },
        )
        seen["response"] = response
        return response

    app = create_app(
        Settings(max_response_body_bytes=64),
        transport=httpx.MockTransport(handler),
    )
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://proxy",
        ) as client,
    ):
        response = await client.post(
            "/v1/messages",
            json={"model": "claude", "max_tokens": 10, "messages": []},
        )

    assert response.status_code == 502
    assert response.json()["error"]["type"] == "api_error"
    assert response.json()["error"]["message"] == (
        "Invalid response from upstream: Upstream response exceeded the response size limit"
    )
    assert seen["response"].is_closed


@pytest.mark.anyio
async def test_streamed_upstream_response_is_not_buffered_to_apply_total_size_limit() -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=event_stream(
                [
                    {
                        "id": "chatcmpl_stream_size",
                        "choices": [
                            {
                                "delta": {"content": "streamed despite one-byte limit"},
                                "finish_reason": "stop",
                            },
                        ],
                    },
                    "[DONE]",
                ],
            ),
        )

    app = create_app(
        Settings(max_response_body_bytes=1),
        transport=httpx.MockTransport(handler),
    )
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://proxy",
        ) as client,
    ):
        response = await client.post(
            "/v1/messages",
            json={
                "model": "claude",
                "max_tokens": 10,
                "stream": True,
                "messages": [],
            },
        )

    assert response.status_code == 200
    assert "streamed despite one-byte limit" in response.text
    assert "event: message_stop" in response.text


@pytest.mark.anyio
async def test_upstream_error_uses_anthropic_error_envelope() -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            headers={
                "retry-after": "3",
                "retry-after-ms": "3000",
                "request-id": "req_123",
                "x-ratelimit-remaining-requests": "0",
                "x-should-retry": "true",
            },
            json={"error": {"type": "rate_limit_exceeded", "message": "slow down"}},
        )

    app = create_app(Settings(), transport=httpx.MockTransport(handler))
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://proxy",
        ) as client,
    ):
        response = await client.post(
            "/v1/messages",
            json={"model": "claude", "max_tokens": 2, "messages": []},
        )

    assert response.status_code == 429
    assert response.headers["retry-after"] == "3"
    assert response.headers["retry-after-ms"] == "3000"
    assert response.headers["x-ratelimit-remaining-requests"] == "0"
    assert response.headers["x-should-retry"] == "true"
    assert response.json() == {
        "type": "error",
        "error": {"type": "rate_limit_error", "message": "slow down"},
        "request_id": "req_123",
    }


@pytest.mark.anyio
async def test_upstream_error_redacts_sensitive_structured_details() -> None:
    sensitive_values = (
        "internal.example",
        "query-secret",
        "bearer-secret",
        "key-secret",
        "proprietary prompt",
        "request-id-secret",
    )

    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            headers={"x-request-id": "token=request-id-secret"},
            json={
                "error": {
                    "type": "invalid_request_error",
                    "message": (
                        "Gateway https://internal.example/path?sig=query-secret "
                        "Authorization: Bearer bearer-secret api_key=key-secret "
                        "prompt: proprietary prompt"
                    ),
                },
            },
        )

    app = create_app(Settings(), transport=httpx.MockTransport(handler))
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://proxy",
        ) as client,
    ):
        response = await client.post(
            "/v1/messages",
            json={"model": "claude", "max_tokens": 10, "messages": []},
        )

    assert response.status_code == 400
    assert "[redacted" in response.text
    for sensitive in sensitive_values:
        assert sensitive not in response.text


@pytest.mark.anyio
async def test_oversized_upstream_error_body_is_not_returned_and_response_is_closed() -> None:
    seen: dict[str, httpx.Response] = {}

    async def handler(_request: httpx.Request) -> httpx.Response:
        response = httpx.Response(
            500,
            content=b'{"error":{"message":"' + b"sensitive-error-body" * 100 + b'"}}',
        )
        seen["response"] = response
        return response

    app = create_app(
        Settings(max_error_body_bytes=64),
        transport=httpx.MockTransport(handler),
    )
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://proxy",
        ) as client,
    ):
        response = await client.post(
            "/v1/messages",
            json={"model": "claude", "max_tokens": 10, "messages": []},
        )

    assert response.status_code == 500
    assert response.json()["error"] == {
        "type": "api_error",
        "message": "Upstream request failed",
    }
    assert "sensitive-error-body" not in response.text
    assert seen["response"].is_closed


@pytest.mark.anyio
async def test_upstream_error_maps_quota_code_to_billing_error() -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            json={
                "error": {
                    "type": "server_error",
                    "code": "insufficient_quota",
                    "message": "quota exhausted",
                },
            },
        )

    app = create_app(Settings(), transport=httpx.MockTransport(handler))
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://proxy",
        ) as client,
    ):
        response = await client.post(
            "/v1/messages",
            json={"model": "claude", "max_tokens": 2, "messages": []},
        )

    assert response.status_code == 429
    assert response.json()["error"] == {
        "type": "billing_error",
        "message": "quota exhausted",
    }


@pytest.mark.anyio
async def test_token_count_unsupported_uses_standard_anthropic_error_type() -> None:
    app = create_app(Settings(token_count_mode="unsupported"))  # noqa: S106
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://proxy",
        ) as client,
    ):
        response = await client.post("/v1/messages/count_tokens", json={})

    assert response.status_code == 501
    assert response.json()["error"]["type"] == "api_error"


@pytest.mark.anyio
async def test_responses_token_count_uses_exact_upstream_endpoint() -> None:
    seen: dict[str, Any] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            headers={"x-request-id": "req_count", "x-ratelimit-remaining-tokens": "900"},
            json={"object": "response.input_tokens", "input_tokens": 42},
        )

    app = create_app(
        Settings(
            openai_api="responses",
            model_map={"claude-test": "gpt-5.6-sol"},
        ),
        transport=httpx.MockTransport(handler),
    )
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://proxy",
        ) as client,
    ):
        response = await client.post(
            "/v1/messages/count_tokens",
            json={
                "model": "claude-test",
                "messages": [
                    {"role": "user", "content": "Question"},
                    {"role": "assistant", "content": "Partial answer"},
                ],
                "output_config": {"effort": "high"},
                "tools": [
                    {
                        "name": "lookup",
                        "input_schema": {"type": "object"},
                    },
                ],
            },
        )

    assert response.status_code == 200
    assert response.json() == {"input_tokens": 42}
    assert response.headers["x-request-id"] == "req_count"
    assert response.headers["x-ratelimit-remaining-tokens"] == "900"
    assert seen == {
        "path": "/v1/responses/input_tokens",
        "body": {
            "model": "gpt-5.6-sol",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Question"}],
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "input_text", "text": "Partial answer"}],
                },
            ],
            "reasoning": {"effort": "high"},
            "tools": [
                {
                    "type": "function",
                    "name": "lookup",
                    "description": "",
                    "parameters": {"type": "object"},
                },
            ],
        },
    }


@pytest.mark.anyio
async def test_auto_protocol_routes_token_counts_by_mapped_model() -> None:
    seen_protocols: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if request.url.path == "/v1/proxy/v1/messages/count_tokens":
            seen_protocols.append("anthropic")
            assert body["model"] == "gateway-claude-opus"
            assert body["future_anthropic_field"] is True
            assert dict(request.url.params) == {"beta": "true"}
            return httpx.Response(200, json={"input_tokens": 31})

        assert request.url.path == "/v1/proxy/v1/responses/input_tokens"
        seen_protocols.append("openai")
        assert body == {
            "model": "gateway-gpt-reasoning",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Hello"}],
                },
            ],
        }
        assert not request.url.query
        return httpx.Response(
            200,
            json={"object": "response.input_tokens", "input_tokens": 29},
        )

    app = create_app(
        Settings(
            upstream_base_url="https://gateway.example/v1/proxy",
            upstream_protocol="auto",
            openai_api="responses",
            upstream_messages_path="/v1/messages",
            upstream_responses_input_tokens_path="/v1/responses/input_tokens",
            model_map={
                "claude-opus-*": "gateway-claude-opus",
                "claude-sonnet-*": "gateway-gpt-reasoning",
            },
        ),
        transport=httpx.MockTransport(handler),
    )
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://proxy",
        ) as client,
    ):
        anthropic = await client.post(
            "/v1/messages/count_tokens?beta=true",
            json={
                "model": "claude-opus-5",
                "messages": [{"role": "user", "content": "Hello"}],
                "future_anthropic_field": True,
            },
        )
        translated = await client.post(
            "/v1/messages/count_tokens?beta=true",
            json={
                "model": "claude-sonnet-4-6",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

    assert anthropic.status_code == translated.status_code == 200
    assert anthropic.json() == {"input_tokens": 31}
    assert translated.json() == {"input_tokens": 29}
    assert seen_protocols == ["anthropic", "openai"]


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("fallback", "expected_status"),
    [("unsupported", 501), ("estimate", 200)],
)
async def test_responses_token_count_has_configurable_unsupported_endpoint_fallback(
    fallback: str,
    expected_status: int,
) -> None:
    seen_response: dict[str, httpx.Response] = {}

    async def handler(_request: httpx.Request) -> httpx.Response:
        response = httpx.Response(404, json={"error": {"message": "Not implemented"}})
        seen_response["response"] = response
        return response

    app = create_app(
        Settings(openai_api="responses", token_count_fallback=fallback),
        transport=httpx.MockTransport(handler),
    )
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://proxy",
        ) as client,
    ):
        response = await client.post(
            "/v1/messages/count_tokens",
            json={"model": "claude-test", "messages": [{"role": "user", "content": "Hi"}]},
        )

    assert response.status_code == expected_status
    assert seen_response["response"].is_closed
    if fallback == "estimate":
        assert response.json()["input_tokens"] >= 1
    else:
        assert response.json()["error"]["type"] == "api_error"


@pytest.mark.anyio
async def test_exact_token_count_preserves_upstream_error_metadata() -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            headers={"retry-after": "2", "x-request-id": "req_count_limit"},
            json={"error": {"type": "rate_limit_error", "message": "Slow down"}},
        )

    app = create_app(
        Settings(openai_api="responses", token_count_mode="exact"),  # noqa: S106
        transport=httpx.MockTransport(handler),
    )
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://proxy",
        ) as client,
    ):
        response = await client.post(
            "/v1/messages/count_tokens",
            json={"model": "claude-test", "messages": []},
        )

    assert response.status_code == 429
    assert response.headers["retry-after"] == "2"
    assert response.headers["x-request-id"] == "req_count_limit"
    assert response.json() == {
        "type": "error",
        "error": {"type": "rate_limit_error", "message": "Slow down"},
        "request_id": "req_count_limit",
    }


@pytest.mark.anyio
async def test_chat_token_count_prefers_client_fallback_unless_estimate_is_enabled() -> None:
    request = {"model": "claude-test", "messages": [{"role": "user", "content": "Hello"}]}

    for mode, expected_status in (("auto", 501), ("estimate", 200)):
        app = create_app(Settings(openai_api="chat_completions", token_count_mode=mode))
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://proxy",
            ) as client,
        ):
            response = await client.post("/v1/messages/count_tokens", json=request)

        assert response.status_code == expected_status
        if mode == "estimate":
            assert response.json()["input_tokens"] >= 1


@pytest.mark.anyio
async def test_upstream_timeout_does_not_echo_transport_details() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("credential=must-not-appear", request=request)

    app = create_app(Settings(), transport=httpx.MockTransport(handler))
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://proxy",
        ) as client,
    ):
        response = await client.post(
            "/v1/messages",
            json={"model": "claude", "max_tokens": 2, "messages": []},
        )

    assert response.status_code == 504
    assert response.json()["error"] == {
        "type": "timeout_error",
        "message": "Upstream request timed out",
    }
    assert "must-not-appear" not in response.text


@pytest.mark.anyio
async def test_buffered_response_is_wrapped_when_upstream_ignores_stream() -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl_buffered",
                "model": "work-model",
                "choices": [{"message": {"content": "buffered text"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 7, "completion_tokens": 2},
            },
        )

    app = create_app(Settings(), transport=httpx.MockTransport(handler))
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://proxy",
        ) as client,
    ):
        response = await client.post(
            "/v1/messages",
            json={
                "model": "claude",
                "max_tokens": 10,
                "stream": True,
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

    assert response.headers["content-type"].startswith("text/event-stream")
    assert '"type":"text_delta","text":"buffered text"' in response.text
    assert '"stop_reason":"end_turn"' in response.text
    assert '"output_tokens":2' in response.text


@pytest.mark.anyio
async def test_buffered_invalid_tool_arguments_return_sanitized_protocol_error() -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl_invalid_tool",
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "function": {
                                        "name": "read",
                                        "arguments": "must-not-appear",
                                    },
                                },
                            ],
                        },
                        "finish_reason": "tool_calls",
                    },
                ],
            },
        )

    app = create_app(Settings(), transport=httpx.MockTransport(handler))
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://proxy",
        ) as client,
    ):
        response = await client.post(
            "/v1/messages",
            json={"model": "claude", "max_tokens": 2, "messages": []},
        )

    assert response.status_code == 502
    assert response.json()["error"]["type"] == "api_error"
    assert "invalid JSON" in response.json()["error"]["message"]
    assert "must-not-appear" not in response.text


@pytest.mark.anyio
async def test_responses_backend_translates_request_and_buffered_response() -> None:
    seen: dict[str, Any] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        seen["session"] = request.headers.get("x-claude-code-session-id")
        seen["agent"] = request.headers.get("x-claude-code-agent-id")
        return httpx.Response(
            200,
            json={
                "id": "resp_buffered",
                "model": "gpt-5.6-sol",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "done"}],
                    },
                ],
                "usage": {"input_tokens": 9, "output_tokens": 2},
            },
        )

    app = create_app(
        Settings(
            upstream_base_url="https://gateway.example/v1",
            openai_api="responses",
            model_map={"claude-opus-5": "gpt-5.6-sol"},
        ),
        transport=httpx.MockTransport(handler),
    )
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://proxy",
        ) as client,
    ):
        response = await client.post(
            "/v1/messages",
            headers={
                "x-api-key": "helper-key",
                "x-claude-code-session-id": "session-1",
                "x-claude-code-agent-id": "agent-1",
            },
            json={
                "model": "claude-opus-5",
                "max_tokens": 100,
                "output_config": {"effort": "high"},
                "messages": [{"role": "user", "content": "work"}],
            },
        )

    assert response.status_code == 200
    assert seen == {
        "path": "/v1/responses",
        "body": {
            "model": "gpt-5.6-sol",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "work"}],
                },
            ],
            "max_output_tokens": 100,
            "stream": False,
            "store": False,
            "reasoning": {"effort": "high"},
            "include": ["reasoning.encrypted_content"],
        },
        "session": "session-1",
        "agent": "agent-1",
    }
    assert response.json()["model"] == "claude-opus-5"
    assert [block for block in response.json()["content"] if block["type"] == "text"] == [
        {"type": "text", "text": "done"}
    ]


@pytest.mark.anyio
async def test_responses_stream_translates_text_tools_and_usage() -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=event_stream(
                [
                    {
                        "type": "response.created",
                        "response": {"id": "resp_stream", "model": "gpt-5.6-sol"},
                    },
                    {
                        "type": "response.output_text.delta",
                        "output_index": 0,
                        "content_index": 0,
                        "delta": "Working",
                    },
                    {
                        "type": "response.output_text.done",
                        "output_index": 0,
                        "content_index": 0,
                        "text": "Working",
                    },
                    {
                        "type": "response.output_item.added",
                        "output_index": 1,
                        "item": {
                            "type": "function_call",
                            "id": "fc_1",
                            "call_id": "call_1",
                            "name": "shell",
                        },
                    },
                    {
                        "type": "response.function_call_arguments.delta",
                        "output_index": 1,
                        "delta": '{"cmd":"ls"}',
                    },
                    {
                        "type": "response.output_item.done",
                        "output_index": 1,
                        "item": {"type": "function_call"},
                    },
                    {
                        "type": "response.completed",
                        "response": {
                            "id": "resp_stream",
                            "status": "completed",
                            "usage": {
                                "input_tokens": 40,
                                "input_tokens_details": {
                                    "cached_tokens": 10,
                                    "cache_write_tokens": 5,
                                },
                                "output_tokens": 8,
                                "output_tokens_details": {"reasoning_tokens": 3},
                            },
                        },
                    },
                ],
            ),
        )

    app = create_app(
        Settings(openai_api="responses"),
        transport=httpx.MockTransport(handler),
    )
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://proxy",
        ) as client,
    ):
        response = await client.post(
            "/v1/messages",
            json={
                "model": "claude-opus-5",
                "max_tokens": 100,
                "stream": True,
                "messages": [{"role": "user", "content": "go"}],
            },
        )

    body = response.text
    assert '"model":"claude-opus-5"' in body
    assert '"type":"text_delta","text":"Working"' in body
    assert '"id":"call_1","name":"shell"' in body
    assert '"partial_json":"{\\"cmd\\":\\"ls\\"}"' in body
    assert '"stop_reason":"tool_use"' in body
    assert '"input_tokens":25' in body
    assert '"cache_read_input_tokens":10' in body
    assert '"cache_creation_input_tokens":5' in body
    assert '"thinking_tokens":3' in body


@pytest.mark.anyio
async def test_responses_standard_stream_error_preserves_message_and_normalizes_code() -> None:
    response = httpx.Response(
        200,
        content=event_stream(
            [
                {
                    "type": "error",
                    "sequence_number": 1,
                    "code": "rate_limit_exceeded",
                    "message": "slow down",
                    "param": None,
                },
            ],
        ),
    )

    body = b"".join(
        [chunk async for chunk in responses_stream_to_anthropic(response, "claude-test")]
    ).decode()

    assert '"type":"rate_limit_error","message":"slow down"' in body
    assert "Responses API stream failed" not in body


@pytest.mark.anyio
async def test_responses_stream_error_after_message_start_keeps_valid_prefix() -> None:
    response = httpx.Response(
        200,
        content=event_stream(
            [
                {
                    "type": "response.created",
                    "response": {"id": "resp_failed", "model": "gpt-test"},
                },
                {
                    "type": "error",
                    "sequence_number": 2,
                    "code": "server_error",
                    "message": "try later",
                    "param": None,
                },
            ],
        ),
    )

    body = b"".join(
        [chunk async for chunk in responses_stream_to_anthropic(response, "claude-test")]
    ).decode()

    assert body.index("event: message_start") < body.index("event: error")
    assert '"type":"api_error","message":"try later"' in body


@pytest.mark.anyio
async def test_responses_failed_stream_event_normalizes_nested_error() -> None:
    response = httpx.Response(
        200,
        content=event_stream(
            [
                {
                    "type": "response.failed",
                    "response": {
                        "id": "resp_failed",
                        "error": {"code": "server_error", "message": "try later"},
                    },
                },
            ],
        ),
    )

    body = b"".join(
        [chunk async for chunk in responses_stream_to_anthropic(response, "claude-test")]
    ).decode()

    assert '"type":"api_error","message":"try later"' in body


@pytest.mark.anyio
async def test_chat_stream_error_normalizes_provider_type() -> None:
    response = httpx.Response(
        200,
        content=event_stream(
            [
                {
                    "error": {
                        "type": "insufficient_quota",
                        "message": "quota exhausted",
                    },
                },
            ],
        ),
    )

    body = b"".join(
        [chunk async for chunk in openai_stream_to_anthropic(response, "claude-test")]
    ).decode()

    assert '"type":"billing_error","message":"quota exhausted"' in body


@pytest.mark.anyio
async def test_chat_stream_refusal_preserves_text_and_stop_reason() -> None:
    response = httpx.Response(
        200,
        content=event_stream(
            [
                {
                    "id": "chatcmpl_refusal",
                    "choices": [
                        {"index": 0, "delta": {"refusal": "I cannot help."}},
                    ],
                },
                {
                    "id": "chatcmpl_refusal",
                    "choices": [
                        {"index": 0, "delta": {}, "finish_reason": "stop"},
                    ],
                },
                "[DONE]",
            ],
        ),
    )

    body = b"".join(
        [chunk async for chunk in openai_stream_to_anthropic(response, "claude-test")]
    ).decode()

    assert '"type":"text_delta","text":"I cannot help."' in body
    assert '"stop_reason":"refusal"' in body
    assert '"stop_details":{"type":"refusal","explanation":"I cannot help."}' in body


@pytest.mark.anyio
async def test_responses_stream_refusal_preserves_text_and_stop_reason() -> None:
    refusal_item = {
        "type": "message",
        "id": "msg_refusal",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "refusal", "refusal": "I cannot help."}],
    }
    response = httpx.Response(
        200,
        content=event_stream(
            [
                {
                    "type": "response.created",
                    "response": {"id": "resp_refusal", "model": "gpt-test"},
                },
                {
                    "type": "response.refusal.delta",
                    "output_index": 0,
                    "content_index": 0,
                    "delta": "I cannot help.",
                },
                {
                    "type": "response.refusal.done",
                    "output_index": 0,
                    "content_index": 0,
                    "refusal": "I cannot help.",
                },
                {
                    "type": "response.completed",
                    "response": {
                        "id": "resp_refusal",
                        "status": "completed",
                        "output": [refusal_item],
                        "usage": {"input_tokens": 4, "output_tokens": 3},
                    },
                },
            ],
        ),
    )

    body = b"".join(
        [chunk async for chunk in responses_stream_to_anthropic(response, "claude-test")]
    ).decode()

    assert '"type":"text_delta","text":"I cannot help."' in body
    assert '"stop_reason":"refusal"' in body
    assert '"stop_details":{"type":"refusal","explanation":"I cannot help."}' in body


@pytest.mark.anyio
async def test_chat_stream_rejects_non_object_function_arguments() -> None:
    response = httpx.Response(
        200,
        content=event_stream(
            [
                {
                    "id": "chatcmpl_invalid_tool",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_1",
                                        "function": {"name": "read", "arguments": "[]"},
                                    },
                                ],
                            },
                        },
                    ],
                },
                "[DONE]",
            ],
        ),
    )

    body = b"".join(
        [chunk async for chunk in openai_stream_to_anthropic(response, "claude-test")]
    ).decode()

    assert "event: error" in body
    assert '"type":"api_error"' in body
    assert "must be a JSON object" in body
    assert '"value"' not in body


@pytest.mark.anyio
async def test_responses_stream_rejects_non_object_function_arguments() -> None:
    response = httpx.Response(
        200,
        content=event_stream(
            [
                {
                    "type": "response.created",
                    "response": {"id": "resp_invalid_tool", "model": "gpt-test"},
                },
                {
                    "type": "response.output_item.added",
                    "output_index": 0,
                    "item": {
                        "type": "function_call",
                        "id": "fc_1",
                        "call_id": "call_1",
                        "name": "read",
                        "arguments": "",
                    },
                },
                {
                    "type": "response.function_call_arguments.delta",
                    "output_index": 0,
                    "delta": "[]",
                },
                {
                    "type": "response.output_item.done",
                    "output_index": 0,
                    "item": {
                        "type": "function_call",
                        "id": "fc_1",
                        "call_id": "call_1",
                        "name": "read",
                        "arguments": "[]",
                    },
                },
            ],
        ),
    )

    body = b"".join(
        [chunk async for chunk in responses_stream_to_anthropic(response, "claude-test")]
    ).decode()

    assert "event: error" in body
    assert '"type":"api_error"' in body
    assert "must be a JSON object" in body
    assert '"value"' not in body


@pytest.mark.anyio
async def test_chat_stream_rejects_unknown_finish_reason() -> None:
    response = httpx.Response(
        200,
        content=event_stream(
            [
                {
                    "id": "chatcmpl_unknown_stop",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": "partial"},
                            "finish_reason": "future_reason",
                        },
                    ],
                },
                "[DONE]",
            ],
        ),
    )

    body = b"".join(
        [chunk async for chunk in openai_stream_to_anthropic(response, "claude-test")]
    ).decode()

    assert "event: error" in body
    assert "Unsupported OpenAI finish reason" in body
    assert '"stop_reason":"end_turn"' not in body


@pytest.mark.anyio
async def test_responses_stream_maps_context_window_exhaustion() -> None:
    response = httpx.Response(
        200,
        content=event_stream(
            [
                {
                    "type": "response.created",
                    "response": {"id": "resp_context", "model": "gpt-test"},
                },
                {
                    "type": "response.incomplete",
                    "response": {
                        "id": "resp_context",
                        "status": "incomplete",
                        "output": [],
                        "incomplete_details": {"reason": "context_window_exceeded"},
                        "usage": {"input_tokens": 100, "output_tokens": 0},
                    },
                },
            ],
        ),
    )

    body = b"".join(
        [chunk async for chunk in responses_stream_to_anthropic(response, "claude-test")]
    ).decode()

    assert '"stop_reason":"model_context_window_exceeded"' in body
    assert "event: message_stop" in body


@pytest.mark.anyio
async def test_responses_stream_recovers_arguments_from_authoritative_done_event() -> None:
    function_item = {
        "type": "function_call",
        "id": "fc_1",
        "call_id": "call_1",
        "name": "read",
        "arguments": '{"path":"README.md"}',
    }
    response = httpx.Response(
        200,
        content=event_stream(
            [
                {
                    "type": "response.created",
                    "response": {"id": "resp_tool", "model": "gpt-test"},
                },
                {
                    "type": "response.output_item.added",
                    "output_index": 0,
                    "item": {**function_item, "arguments": ""},
                },
                {
                    "type": "response.function_call_arguments.done",
                    "output_index": 0,
                    "item_id": "fc_1",
                    "name": "read",
                    "arguments": '{"path":"README.md"}',
                },
                {
                    "type": "response.output_item.done",
                    "output_index": 0,
                    "item": function_item,
                },
                {
                    "type": "response.completed",
                    "response": {
                        "id": "resp_tool",
                        "status": "completed",
                        "output": [function_item],
                    },
                },
            ],
        ),
    )

    body = b"".join(
        [chunk async for chunk in responses_stream_to_anthropic(response, "claude-test")]
    ).decode()

    encoded_arguments = '"partial_json":"{\\"path\\":\\"README.md\\"}"'
    assert body.count(encoded_arguments) == 1
    assert '"stop_reason":"tool_use"' in body


@pytest.mark.anyio
async def test_responses_stream_recovers_text_from_authoritative_done_event() -> None:
    message_item = {
        "type": "message",
        "id": "msg_1",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": "Recovered", "annotations": []}],
    }
    response = httpx.Response(
        200,
        content=event_stream(
            [
                {
                    "type": "response.created",
                    "response": {"id": "resp_text", "model": "gpt-test"},
                },
                {
                    "type": "response.output_text.done",
                    "output_index": 0,
                    "content_index": 0,
                    "item_id": "msg_1",
                    "text": "Recovered",
                },
                {
                    "type": "response.completed",
                    "response": {
                        "id": "resp_text",
                        "status": "completed",
                        "output": [message_item],
                    },
                },
            ],
        ),
    )

    body = b"".join(
        [chunk async for chunk in responses_stream_to_anthropic(response, "claude-test")]
    ).decode()

    assert body.count('"type":"text_delta","text":"Recovered"') == 1
    assert '"stop_reason":"end_turn"' in body


@pytest.mark.anyio
@pytest.mark.parametrize("sparse_source", ["terminal", "output_item_done"])
async def test_responses_stream_recovers_text_from_sparse_completion(
    sparse_source: str,
) -> None:
    message_item = {
        "type": "message",
        "id": "msg_sparse",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": "Hello!", "annotations": []}],
    }
    events: list[dict[str, object] | str] = [
        {
            "type": "response.created",
            "response": {"id": "resp_sparse", "model": "gpt-test"},
        },
    ]
    if sparse_source == "output_item_done":
        events.append(
            {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": message_item,
            },
        )
    events.append(
        {
            "type": "response.completed",
            "response": {
                "id": "resp_sparse",
                "status": "completed",
                "output": [message_item],
            },
        },
    )
    response = httpx.Response(200, content=event_stream(events))

    body = b"".join(
        [chunk async for chunk in responses_stream_to_anthropic(response, "claude-test")]
    ).decode()

    assert body.count('"type":"text_delta","text":"Hello!"') == 1
    assert body.count('"content_block":{"type":"text","text":""}') == 1
    assert body.index('"type":"text_delta","text":"Hello!"') < body.index(
        '"stop_reason":"end_turn"',
    )


@pytest.mark.anyio
async def test_responses_stream_recovers_unstreamed_text_suffix_from_completion() -> None:
    message_item = {
        "type": "message",
        "id": "msg_partial",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": "Hello!", "annotations": []}],
    }
    response = httpx.Response(
        200,
        content=event_stream(
            [
                {
                    "type": "response.output_text.delta",
                    "output_index": 0,
                    "content_index": 0,
                    "delta": "Hel",
                },
                {
                    "type": "response.completed",
                    "response": {
                        "id": "resp_partial",
                        "status": "completed",
                        "output": [message_item],
                    },
                },
            ],
        ),
    )

    body = b"".join(
        [chunk async for chunk in responses_stream_to_anthropic(response, "claude-test")]
    ).decode()

    assert '"type":"text_delta","text":"Hel"' in body
    assert '"type":"text_delta","text":"lo!"' in body
    assert body.count('"content_block":{"type":"text","text":""}') == 1


@pytest.mark.anyio
async def test_responses_stream_does_not_reopen_text_for_duplicate_done_events() -> None:
    message_item = {
        "type": "message",
        "id": "msg_done",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": "Done", "annotations": []}],
    }
    response = httpx.Response(
        200,
        content=event_stream(
            [
                {
                    "type": "response.output_text.done",
                    "output_index": 0,
                    "content_index": 0,
                    "text": "Done",
                },
                {
                    "type": "response.content_part.done",
                    "output_index": 0,
                    "content_index": 0,
                    "part": message_item["content"][0],
                },
                {
                    "type": "response.output_item.done",
                    "output_index": 0,
                    "item": message_item,
                },
                {
                    "type": "response.completed",
                    "response": {
                        "id": "resp_done",
                        "status": "completed",
                        "output": [message_item],
                    },
                },
            ],
        ),
    )

    body = b"".join(
        [chunk async for chunk in responses_stream_to_anthropic(response, "claude-test")]
    ).decode()

    assert body.count('"content_block":{"type":"text","text":""}') == 1
    assert body.count('"type":"text_delta","text":"Done"') == 1
    assert body.count('"type":"content_block_stop","index":0') == 1


@pytest.mark.anyio
async def test_responses_stream_recovers_tool_call_from_sparse_completion() -> None:
    function_item = {
        "type": "function_call",
        "id": "fc_sparse",
        "call_id": "call_sparse",
        "name": "read",
        "arguments": '{"path":"README.md"}',
    }
    response = httpx.Response(
        200,
        content=event_stream(
            [
                {
                    "type": "response.completed",
                    "response": {
                        "id": "resp_sparse_tool",
                        "status": "completed",
                        "output": [function_item],
                    },
                },
            ],
        ),
    )

    body = b"".join(
        [chunk async for chunk in responses_stream_to_anthropic(response, "claude-test")]
    ).decode()

    assert '"id":"call_sparse","name":"read"' in body
    assert '"partial_json":"{\\"path\\":\\"README.md\\"}"' in body
    assert '"stop_reason":"tool_use"' in body


@pytest.mark.anyio
async def test_responses_stream_rejects_mismatched_authoritative_arguments() -> None:
    response = httpx.Response(
        200,
        content=event_stream(
            [
                {
                    "type": "response.output_item.added",
                    "output_index": 0,
                    "item": {
                        "type": "function_call",
                        "id": "fc_1",
                        "call_id": "call_1",
                        "name": "read",
                        "arguments": "",
                    },
                },
                {
                    "type": "response.function_call_arguments.delta",
                    "output_index": 0,
                    "delta": '{"path":',
                },
                {
                    "type": "response.function_call_arguments.done",
                    "output_index": 0,
                    "item_id": "fc_1",
                    "name": "read",
                    "arguments": '{"other":"README.md"}',
                },
            ],
        ),
    )

    body = b"".join(
        [chunk async for chunk in responses_stream_to_anthropic(response, "claude-test")]
    ).decode()

    assert "event: error" in body
    assert "do not match streamed deltas" in body
    assert "event: content_block_stop" not in body


@pytest.mark.anyio
async def test_corrupt_proxy_reasoning_state_is_rejected_without_upstream_call() -> None:
    encoded = encode_responses_reasoning(
        {
            "type": "reasoning",
            "id": "rs_sensitive",
            "encrypted_content": "must-not-appear",
        },
    )
    corrupt_data = base64.b64encode(base64.b64decode(encoded)[:-1]).decode()

    async def unexpected_handler(_request: httpx.Request) -> httpx.Response:
        msg = "Malformed state must be rejected before an upstream request"
        raise AssertionError(msg)

    app = create_app(
        Settings(openai_api="responses"),
        transport=httpx.MockTransport(unexpected_handler),
    )
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://proxy",
        ) as client,
    ):
        response = await client.post(
            "/v1/messages",
            json={
                "model": "claude-test",
                "max_tokens": 100,
                "messages": [
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "redacted_thinking", "data": corrupt_data},
                        ],
                    },
                    {"role": "user", "content": "Continue"},
                ],
            },
        )

    assert response.status_code == 400
    assert response.json() == {
        "type": "error",
        "error": {
            "type": "invalid_request_error",
            "message": "Invalid proxy Responses reasoning state",
        },
    }
    assert "must-not-appear" not in response.text


@pytest.mark.anyio
async def test_corrupt_proxy_output_state_is_rejected_without_upstream_call() -> None:
    encoded = encode_responses_output(
        [
            {
                "type": "message",
                "id": "msg_sensitive",
                "content": [{"type": "output_text", "text": "must-not-appear"}],
            },
        ],
    )
    corrupt_data = base64.b64encode(base64.b64decode(encoded)[:-1]).decode()

    async def unexpected_handler(_request: httpx.Request) -> httpx.Response:
        msg = "Malformed state must be rejected before an upstream request"
        raise AssertionError(msg)

    app = create_app(
        Settings(openai_api="responses"),
        transport=httpx.MockTransport(unexpected_handler),
    )
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://proxy",
        ) as client,
    ):
        response = await client.post(
            "/v1/messages",
            json={
                "model": "claude-test",
                "max_tokens": 100,
                "messages": [
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "redacted_thinking", "data": corrupt_data},
                        ],
                    },
                    {"role": "user", "content": "Continue"},
                ],
            },
        )

    assert response.status_code == 400
    assert response.json() == {
        "type": "error",
        "error": {
            "type": "invalid_request_error",
            "message": "Invalid proxy Responses output state",
        },
    }
    assert "must-not-appear" not in response.text


@pytest.mark.anyio
async def test_model_bound_output_state_cannot_be_replayed_after_model_remapping() -> None:
    output = [
        {
            "type": "message",
            "id": "msg_old",
            "role": "assistant",
            "status": "completed",
            "content": [
                {"type": "output_text", "text": "Old answer", "annotations": []},
            ],
        },
    ]
    encoded = encode_responses_output(output, model="upstream-model-a")

    async def unexpected_handler(_request: httpx.Request) -> httpx.Response:
        msg = "Mismatched state must be rejected before an upstream request"
        raise AssertionError(msg)

    app = create_app(
        Settings(
            openai_api="responses",
            model_map={"claude-test": "upstream-model-b"},
        ),
        transport=httpx.MockTransport(unexpected_handler),
    )
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://proxy",
        ) as client,
    ):
        response = await client.post(
            "/v1/messages",
            json={
                "model": "claude-test",
                "max_tokens": 100,
                "messages": [
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "redacted_thinking", "data": encoded},
                            {"type": "text", "text": "Old answer"},
                        ],
                    },
                    {"role": "user", "content": "Continue"},
                ],
            },
        )

    assert response.status_code == 400
    assert response.json()["error"] == {
        "type": "invalid_request_error",
        "message": "Proxy Responses output state belongs to a different upstream model",
    }


@pytest.mark.anyio
async def test_legacy_unbound_output_state_is_not_replayed_upstream() -> None:
    output = [
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Old answer", "annotations": []}],
        },
    ]
    encoded = encode_responses_output(output)

    async def unexpected_handler(_request: httpx.Request) -> httpx.Response:
        msg = "Unbound legacy state must not reach any upstream model"
        raise AssertionError(msg)

    app = create_app(
        Settings(openai_api="responses"),
        transport=httpx.MockTransport(unexpected_handler),
    )
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://proxy",
        ) as client,
    ):
        response = await client.post(
            "/v1/messages",
            json={
                "model": "claude-test",
                "max_tokens": 100,
                "messages": [
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "redacted_thinking", "data": encoded},
                            {"type": "text", "text": "Old answer"},
                        ],
                    },
                    {"role": "user", "content": "Continue"},
                ],
            },
        )

    assert response.status_code == 400
    assert response.json()["error"]["message"] == (
        "Legacy proxy Responses output state is not bound to an upstream model"
    )


@pytest.mark.anyio
async def test_model_discovery_returns_configured_claude_aliases() -> None:
    app = create_app(
        Settings(
            model_discovery={
                "claude-gpt-5-6-sol": "GPT-5.6 Sol",
                "anthropic-gpt-5-6-terra": "GPT-5.6 Terra",
                "claude-gpt-5-6-luna": "GPT-5.6 Luna",
            },
            model_map={"claude-*": "gpt-5.6-sol", "anthropic-*": "gpt-5.6-terra"},
        ),
    )
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://proxy",
        ) as client,
    ):
        response = await client.get(
            "/v1/models",
            params={"limit": 1000},
            headers={"x-api-key": "helper-key"},
        )
        after = await client.get(
            "/v1/models",
            params={"after_id": "claude-gpt-5-6-sol", "limit": 1},
        )
        before = await client.get(
            "/v1/models",
            params={"before_id": "claude-gpt-5-6-luna", "limit": 1},
        )
        empty = await client.get(
            "/v1/models",
            params={"after_id": "claude-gpt-5-6-luna"},
        )
        retrieved = await client.get("/v1/models/anthropic-gpt-5-6-terra")

    assert response.json() == {
        "data": [
            {
                "id": "claude-gpt-5-6-sol",
                "display_name": "GPT-5.6 Sol",
                "type": "model",
                "created_at": "1970-01-01T00:00:00Z",
            },
            {
                "id": "anthropic-gpt-5-6-terra",
                "display_name": "GPT-5.6 Terra",
                "type": "model",
                "created_at": "1970-01-01T00:00:00Z",
            },
            {
                "id": "claude-gpt-5-6-luna",
                "display_name": "GPT-5.6 Luna",
                "type": "model",
                "created_at": "1970-01-01T00:00:00Z",
            },
        ],
        "has_more": False,
        "first_id": "claude-gpt-5-6-sol",
        "last_id": "claude-gpt-5-6-luna",
    }
    assert [model["id"] for model in after.json()["data"]] == ["anthropic-gpt-5-6-terra"]
    assert after.json()["has_more"] is True
    assert [model["id"] for model in before.json()["data"]] == ["anthropic-gpt-5-6-terra"]
    assert before.json()["has_more"] is True
    assert empty.json() == {
        "data": [],
        "has_more": False,
        "first_id": None,
        "last_id": None,
    }
    assert retrieved.json() == {
        "id": "anthropic-gpt-5-6-terra",
        "display_name": "GPT-5.6 Terra",
        "type": "model",
        "created_at": "1970-01-01T00:00:00Z",
    }


@pytest.mark.anyio
async def test_model_discovery_rejects_invalid_pagination_and_unknown_model() -> None:
    app = create_app(
        Settings(
            model_discovery={"claude-test": "Claude Test"},
            model_map={"claude-test": "claude-test"},
        ),
    )
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://proxy",
        ) as client,
    ):
        invalid_limit = await client.get("/v1/models", params={"limit": 0})
        invalid_cursor = await client.get(
            "/v1/models",
            params={"after_id": "claude-missing"},
        )
        missing_model = await client.get("/v1/models/claude-missing")

    assert invalid_limit.status_code == 400
    assert invalid_limit.json()["error"]["type"] == "invalid_request_error"
    assert invalid_cursor.status_code == 400
    assert invalid_cursor.json()["error"]["type"] == "invalid_request_error"
    assert missing_model.status_code == 404
    assert missing_model.json()["error"]["type"] == "not_found_error"


@pytest.mark.anyio
async def test_responses_stream_emits_ping_during_upstream_silence() -> None:
    upstream = httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        stream=SlowResponseStream(),
    )

    chunks = [
        chunk
        async for chunk in responses_stream_to_anthropic(
            upstream,
            "claude-opus-5",
            ping_interval=0.005,
        )
    ]
    body = b"".join(chunks).decode()

    assert "event: ping" in body
    assert "event: message_stop" in body
    assert upstream.is_closed
