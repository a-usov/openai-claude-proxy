"""Targeted tests across real Uvicorn and TCP boundaries."""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse

from openai_claude_proxy.app import create_app
from openai_claude_proxy.config import Settings
from tests.support.real_http import LoopbackServer

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


def _sse(payload: dict[str, Any]) -> bytes:
    return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n".encode()


def _created() -> bytes:
    return _sse(
        {
            "type": "response.created",
            "response": {"id": "resp_real", "model": "gpt-real", "status": "in_progress"},
        },
    )


def _terminal(text: str = "done") -> bytes:
    item = {
        "type": "message",
        "id": "msg_real",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }
    return b"".join(
        [
            _sse(
                {
                    "type": "response.output_text.done",
                    "output_index": 0,
                    "content_index": 0,
                    "item_id": "msg_real",
                    "text": text,
                },
            ),
            _sse({"type": "response.output_item.done", "output_index": 0, "item": item}),
            _sse(
                {
                    "type": "response.completed",
                    "response": {
                        "id": "resp_real",
                        "model": "gpt-real",
                        "status": "completed",
                        "output": [item],
                        "usage": {"input_tokens": 2, "output_tokens": 1},
                    },
                },
            ),
        ],
    )


def _anthropic_request() -> dict[str, Any]:
    return {
        "model": "claude-real",
        "max_tokens": 32,
        "stream": True,
        "messages": [{"role": "user", "content": "Synthetic"}],
    }


@asynccontextmanager
async def _proxy_pair(
    upstream: FastAPI,
    *,
    ping_interval: float = 0,
    request_timeout: float = 2,
) -> AsyncIterator[str]:
    try:
        async with LoopbackServer(upstream) as upstream_server:
            proxy = create_app(
                Settings(
                    upstream_base_url=f"{upstream_server.base_url}/v1",
                    openai_api="responses",
                    model_map={"claude-real": "gpt-real"},
                    stream_ping_interval=ping_interval,
                    request_timeout=request_timeout,
                    connect_timeout=min(request_timeout, 1),
                ),
            )
            async with LoopbackServer(proxy) as proxy_server:
                yield proxy_server.base_url
    except PermissionError:
        raise pytest.skip.Exception(
            "The execution sandbox does not permit loopback listening sockets",
        ) from None


async def _read_until(lines: AsyncIterator[str], needle: str) -> list[str]:
    consumed: list[str] = []
    async with asyncio.timeout(2):
        async for line in lines:
            consumed.append(line)
            if needle in line:
                return consumed
    raise AssertionError(f"Stream ended before {needle!r}")


@pytest.mark.anyio
async def test_sse_chunk_reaches_client_before_upstream_completes() -> None:
    upstream = FastAPI()
    release = asyncio.Event()

    @upstream.post("/v1/responses")
    async def responses(_request: Request) -> StreamingResponse:
        async def body() -> AsyncIterator[bytes]:
            yield _created()
            yield _sse(
                {
                    "type": "response.output_text.delta",
                    "output_index": 0,
                    "content_index": 0,
                    "item_id": "msg_real",
                    "delta": "first",
                },
            )
            await release.wait()
            yield _terminal("first")

        return StreamingResponse(body(), media_type="text/event-stream")

    async with (
        _proxy_pair(upstream) as proxy_url,
        httpx.AsyncClient(timeout=3) as client,
        client.stream("POST", f"{proxy_url}/v1/messages", json=_anthropic_request()) as response,
    ):
        assert response.status_code == 200
        lines = response.aiter_lines()
        await _read_until(lines, '"type":"text_delta"')
        assert not release.is_set()
        release.set()
        remaining = [line async for line in lines]

    assert any('"type":"message_stop"' in line for line in remaining)


@pytest.mark.anyio
async def test_keepalive_ping_crosses_real_connection_during_upstream_pause() -> None:
    upstream = FastAPI()
    release = asyncio.Event()

    @upstream.post("/v1/responses")
    async def responses(_request: Request) -> StreamingResponse:
        async def body() -> AsyncIterator[bytes]:
            yield _created()
            await release.wait()
            yield _terminal()

        return StreamingResponse(body(), media_type="text/event-stream")

    async with (
        _proxy_pair(upstream, ping_interval=0.03) as proxy_url,
        httpx.AsyncClient(timeout=3) as client,
        client.stream("POST", f"{proxy_url}/v1/messages", json=_anthropic_request()) as response,
    ):
        lines = response.aiter_lines()
        await _read_until(lines, '"type":"ping"')
        assert not release.is_set()
        release.set()
        await _read_until(lines, '"type":"message_stop"')


@pytest.mark.anyio
async def test_client_cancellation_closes_paused_upstream_stream() -> None:
    upstream = FastAPI()
    closed = asyncio.Event()
    never = asyncio.Event()

    @upstream.post("/v1/responses")
    async def responses(_request: Request) -> StreamingResponse:
        async def body() -> AsyncIterator[bytes]:
            try:
                yield _created()
                yield _sse(
                    {
                        "type": "response.output_text.delta",
                        "output_index": 0,
                        "content_index": 0,
                        "item_id": "msg_real",
                        "delta": "started",
                    },
                )
                await never.wait()
            finally:
                closed.set()

        return StreamingResponse(body(), media_type="text/event-stream")

    async with _proxy_pair(upstream) as proxy_url:
        client = httpx.AsyncClient(timeout=3)
        try:
            async with client.stream(
                "POST",
                f"{proxy_url}/v1/messages",
                json=_anthropic_request(),
            ) as response:
                await _read_until(response.aiter_lines(), '"type":"text_delta"')
        finally:
            await client.aclose()
        async with asyncio.timeout(2):
            await closed.wait()


@pytest.mark.anyio
async def test_socket_failures_timeouts_and_partial_sse_are_sanitized() -> None:
    slow_upstream = FastAPI()

    @slow_upstream.post("/v1/responses")
    async def slow(_request: Request) -> Response:
        await asyncio.sleep(0.2)
        return Response(b"too late", media_type="application/json")

    async with (
        _proxy_pair(slow_upstream, request_timeout=0.03) as proxy_url,
        httpx.AsyncClient(timeout=3) as client,
    ):
        timeout_response = await client.post(
            f"{proxy_url}/v1/messages",
            json=_anthropic_request(),
        )
    assert timeout_response.status_code == 504
    assert timeout_response.json()["error"]["message"] == "Upstream request timed out"

    unreachable_proxy = create_app(
        Settings(
            upstream_base_url="http://127.0.0.1:1/v1",
            openai_api="responses",
            model_map={"claude-real": "gpt-real"},
            request_timeout=0.2,
            connect_timeout=0.2,
        ),
    )
    async with (
        LoopbackServer(unreachable_proxy) as proxy_server,
        httpx.AsyncClient(timeout=3) as client,
    ):
        connect_response = await client.post(
            f"{proxy_server.base_url}/v1/messages",
            json=_anthropic_request(),
        )
    assert connect_response.status_code == 502
    assert connect_response.json()["error"]["message"] == "Could not reach upstream"

    partial_upstream = FastAPI()

    @partial_upstream.post("/v1/responses")
    async def partial(_request: Request) -> StreamingResponse:
        async def body() -> AsyncIterator[bytes]:
            yield _created()
            yield b'data: {"type":"response.output_text.delta"'

        return StreamingResponse(body(), media_type="text/event-stream")

    async with (
        _proxy_pair(partial_upstream) as proxy_url,
        httpx.AsyncClient(timeout=3) as client,
    ):
        partial_response = await client.post(
            f"{proxy_url}/v1/messages",
            json=_anthropic_request(),
        )
    assert partial_response.status_code == 200
    assert '"type":"error"' in partial_response.text
    assert "Malformed JSON in Responses stream event" in partial_response.text


@pytest.mark.anyio
async def test_real_passthrough_preserves_duplicate_query_content_type_status_and_retry() -> None:
    upstream = FastAPI()
    seen: dict[str, Any] = {}

    @upstream.post("/v1/embeddings")
    async def embeddings(request: Request) -> Response:
        seen["tags"] = request.query_params.getlist("tag")
        seen["request_id"] = request.headers.get("x-request-id")
        return Response(
            b'{"error":{"message":"Synthetic limit"}}',
            status_code=429,
            headers={
                "content-type": "application/problem+json; charset=utf-8",
                "retry-after": "3",
                "x-request-id": "req_real",
            },
        )

    async with (
        _proxy_pair(upstream) as proxy_url,
        httpx.AsyncClient(timeout=3) as client,
    ):
        response = await client.post(
            f"{proxy_url}/v1/embeddings?tag=a&tag=b",
            headers={"x-request-id": "client-real"},
            json={"model": "embedding-test", "input": "Synthetic"},
        )

    assert seen == {"tags": ["a", "b"], "request_id": "client-real"}
    assert response.status_code == 429
    assert response.headers["content-type"] == "application/problem+json; charset=utf-8"
    assert response.headers["retry-after"] == "3"
    assert response.headers["x-request-id"] == "req_real"
