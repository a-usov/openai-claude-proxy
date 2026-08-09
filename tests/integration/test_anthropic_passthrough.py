"""Multi-turn Anthropic pass-through preserves provider wire semantics."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import httpx
import pytest

from openai_claude_proxy.app import create_app
from openai_claude_proxy.config import Settings
from tests.support.scenario import ExpectedRequest, ScenarioTransport

if TYPE_CHECKING:
    from collections.abc import Callable


def _encoded(value: dict[str, Any]) -> bytes:
    return json.dumps(value, separators=(",", ":")).encode()


@pytest.mark.anyio
async def test_multiturn_anthropic_passthrough_preserves_bodies_headers_query_and_sse() -> None:
    first_body = _encoded(
        {
            "model": "claude-upstream",
            "max_tokens": 64,
            "stream": True,
            "messages": [{"role": "user", "content": "Synthetic turn one"}],
        },
    )
    second_body = _encoded(
        {
            "model": "claude-upstream",
            "max_tokens": 64,
            "stream": True,
            "messages": [
                {"role": "user", "content": "Synthetic turn one"},
                {"role": "assistant", "content": "Synthetic answer one"},
                {"role": "user", "content": "Synthetic turn two"},
            ],
        },
    )
    error_body = _encoded(
        {
            "model": "claude-upstream",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "Synthetic rate-limit probe"}],
        },
    )
    first_sse = b'event: message_start\ndata: {"type":"message_start"}\n\n'
    second_sse = b'event: message_stop\ndata: {"type":"message_stop"}\n\n'
    upstream_error = _encoded(
        {
            "type": "error",
            "error": {"type": "rate_limit_error", "message": "Synthetic limit"},
            "request_id": "req_synthetic",
        },
    )

    def validate(
        body_expected: bytes,
    ) -> Callable[[httpx.Request, dict[str, Any]], None]:
        def validator(request: httpx.Request, _body: dict[str, Any]) -> None:
            assert request.content == body_expected
            assert request.headers["x-api-key"] == "helper-secret"
            assert request.headers["anthropic-version"] == "2023-06-01"
            assert request.headers["anthropic-beta"] == "synthetic-beta"
            assert dict(request.url.params) == {
                "client-query": "kept",
                "api-version": "configured",
            }

        return validator

    scenario = ScenarioTransport(
        [
            ExpectedRequest(
                "POST",
                "/v1/messages",
                validate(first_body),
                first_sse,
            ),
            ExpectedRequest(
                "POST",
                "/v1/messages",
                validate(second_body),
                second_sse,
            ),
            ExpectedRequest(
                "POST",
                "/v1/messages",
                validate(error_body),
                upstream_error,
                response_status=429,
                response_headers={
                    "content-type": "application/json",
                    "retry-after": "2",
                    "request-id": "req_synthetic",
                },
            ),
        ],
    )
    app = create_app(
        Settings(
            upstream_base_url="https://anthropic.example/v1",
            upstream_protocol="anthropic",
            upstream_query={"api-version": "configured"},
        ),
        transport=scenario,
    )
    common_headers = {
        "content-type": "application/json",
        "x-api-key": "helper-secret",
        "anthropic-version": "2023-06-01",
        "anthropic-beta": "synthetic-beta",
    }

    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://proxy",
        ) as client,
    ):
        first = await client.post(
            "/v1/messages?client-query=kept",
            headers=common_headers,
            content=first_body,
        )
        second = await client.post(
            "/v1/messages?client-query=kept",
            headers=common_headers,
            content=second_body,
        )
        error = await client.post(
            "/v1/messages?client-query=kept",
            headers=common_headers,
            content=error_body,
        )

    assert first.status_code == second.status_code == 200
    assert first.content == first_sse
    assert second.content == second_sse
    assert error.status_code == 429
    assert error.content == upstream_error
    assert error.headers["retry-after"] == "2"
    assert error.headers["request-id"] == "req_synthetic"
    scenario.assert_complete()
