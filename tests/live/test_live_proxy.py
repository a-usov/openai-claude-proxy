"""Credential-gated live checks using synthetic prompts and bounded output."""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qsl

import httpx
import pytest

from openai_claude_proxy.app import create_app
from openai_claude_proxy.config import Settings
from tests.support.conversation import AnthropicConversation

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

pytestmark = pytest.mark.live
LIVE_DIALECTS = ("chat", "responses", "anthropic")
MAX_OUTPUT_TOKENS = 64
LIVE_TIMEOUT = 60.0
SYNTHETIC_TOOLS = [
    {
        "name": "lookup_synthetic_value",
        "description": "Return a synthetic test value for a synthetic key",
        "input_schema": {
            "type": "object",
            "properties": {"key": {"type": "string"}},
            "required": ["key"],
            "additionalProperties": False,
        },
    },
]


@dataclass(frozen=True, slots=True)
class _LiveTarget:
    dialect: str
    base_url: str
    model: str
    credential: str = field(repr=False)
    auth_mode: str
    upstream_query: dict[str, str]
    probe_features: bool

    @property
    def client_model(self) -> str:
        return self.model if self.dialect == "anthropic" else f"claude-live-{self.dialect}"


def _skip(message: str) -> None:
    raise pytest.skip.Exception(message)


def _target(dialect: str) -> _LiveTarget:
    prefix = f"LIVE_{dialect.upper()}_"
    values = {
        "base_url": os.environ.get(prefix + "BASE_URL"),
        "model": os.environ.get(prefix + "MODEL"),
        "credential": os.environ.get(prefix + "API_KEY"),
    }
    missing = [name for name, value in values.items() if not value]
    if missing:
        _skip(f"{dialect} live target is missing: {', '.join(missing)}")
    default_auth = "passthrough" if dialect == "anthropic" else "bearer"
    return _LiveTarget(
        dialect=dialect,
        base_url=str(values["base_url"]),
        model=str(values["model"]),
        credential=str(values["credential"]),
        auth_mode=os.environ.get(prefix + "AUTH_MODE", default_auth),
        upstream_query=dict(parse_qsl(os.environ.get(prefix + "UPSTREAM_QUERY", ""))),
        probe_features=os.environ.get(prefix + "PROBES") == "1",
    )


def _settings(target: _LiveTarget) -> Settings:
    protocol = "anthropic" if target.dialect == "anthropic" else "openai"
    openai_api = "responses" if target.dialect == "responses" else "chat_completions"
    model_map = {} if protocol == "anthropic" else {target.client_model: target.model}
    return Settings(
        upstream_base_url=target.base_url,
        upstream_protocol=protocol,
        openai_api=openai_api,
        auth_mode=target.auth_mode,
        upstream_query=target.upstream_query,
        model_map=model_map,
        request_timeout=LIVE_TIMEOUT,
        connect_timeout=10,
        passthrough_enabled=False,
    )


@asynccontextmanager
async def _client(target: _LiveTarget) -> AsyncIterator[httpx.AsyncClient]:
    app = create_app(_settings(target))
    headers = {
        "x-api-key": target.credential,
        "anthropic-version": "2023-06-01",
        "x-client-request-id": f"synthetic-live-{target.dialect}",
    }
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://proxy",
            headers=headers,
            timeout=LIVE_TIMEOUT + 5,
        ) as client,
    ):
        yield client


def _require_success(response: httpx.Response, target: _LiveTarget, operation: str) -> None:
    if response.status_code >= 400:
        msg = f"{target.dialect}/{target.model} {operation} failed with HTTP {response.status_code}"
        raise AssertionError(msg)


def _text_content(message: dict[str, Any]) -> str:
    return "".join(
        block["text"]
        for block in message.get("content", [])
        if block.get("type") == "text" and isinstance(block.get("text"), str)
    )


@pytest.mark.anyio
@pytest.mark.parametrize("dialect", LIVE_DIALECTS)
async def test_live_minimal_text_generation(dialect: str) -> None:
    target = _target(dialect)
    async with _client(target) as client:
        response = await client.post(
            "/v1/messages",
            json={
                "model": target.client_model,
                "max_tokens": MAX_OUTPUT_TOKENS,
                "messages": [
                    {"role": "user", "content": "Reply with the single word synthetic."},
                ],
            },
        )
    _require_success(response, target, "text smoke test")
    if not _text_content(response.json()):
        raise AssertionError(f"{target.dialect}/{target.model} returned no text")


@pytest.mark.anyio
@pytest.mark.parametrize("dialect", LIVE_DIALECTS)
async def test_live_streamed_tool_round_trip(dialect: str) -> None:
    target = _target(dialect)
    async with _client(target) as client:
        conversation = AnthropicConversation(
            client,
            model=target.client_model,
            tools=SYNTHETIC_TOOLS,
            max_tokens=MAX_OUTPUT_TOKENS,
        )
        first = await conversation.ask(
            "Call lookup_synthetic_value exactly once with key alpha before answering.",
        )
        if len(first.tool_uses) != 1:
            raise AssertionError(
                f"{target.dialect}/{target.model} did not produce one synthetic tool call",
            )
        tool_id = first.tool_uses[0]["id"]
        final = await conversation.submit_tool_results({tool_id: "synthetic-value"})

    if final.message.get("stop_reason") != "end_turn":
        raise AssertionError(
            f"{target.dialect}/{target.model} did not complete the synthetic tool round trip",
        )


@pytest.mark.anyio
async def test_live_responses_reasoning_state_round_trip() -> None:
    target = _target("responses")
    async with _client(target) as client:
        conversation = AnthropicConversation(
            client,
            model=target.client_model,
            tools=SYNTHETIC_TOOLS,
            max_tokens=MAX_OUTPUT_TOKENS,
        )
        first = await conversation.ask(
            "Call lookup_synthetic_value exactly once with key state before answering.",
        )
        state_blocks = [
            block for block in first.message["content"] if block.get("type") == "redacted_thinking"
        ]
        if not state_blocks:
            _skip(f"responses/{target.model} returned no replayable reasoning state")
        tool_id = first.tool_uses[0]["id"]
        final = await conversation.submit_tool_results({tool_id: "synthetic-state-value"})

    if final.message.get("stop_reason") != "end_turn":
        raise AssertionError(f"responses/{target.model} did not complete reasoning replay")


@pytest.mark.anyio
@pytest.mark.parametrize("effort", ["low", "medium", "high", "xhigh", "max"])
async def test_live_responses_reasoning_effort_probe(effort: str) -> None:
    target = _target("responses")
    if not target.probe_features:
        _skip(f"responses/{target.model} extended probes are disabled")
    async with _client(target) as client:
        response = await client.post(
            "/v1/messages",
            json={
                "model": target.client_model,
                "max_tokens": MAX_OUTPUT_TOKENS,
                "output_config": {"effort": effort},
                "messages": [{"role": "user", "content": "Reply with synthetic."}],
            },
        )
    if response.status_code >= 400:
        _skip(
            f"responses/{target.model} reports effort {effort} unsupported "
            f"with HTTP {response.status_code}",
        )


@pytest.mark.anyio
@pytest.mark.parametrize("dialect", LIVE_DIALECTS)
async def test_live_optional_structured_cache_mapping_and_attribution_probe(dialect: str) -> None:
    target = _target(dialect)
    if not target.probe_features:
        _skip(f"{target.dialect}/{target.model} extended probes are disabled")
    schema = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
        "additionalProperties": False,
    }
    async with _client(target) as client:
        response = await client.post(
            "/v1/messages",
            json={
                "model": target.client_model,
                "max_tokens": MAX_OUTPUT_TOKENS,
                "cache_control": {"type": "ephemeral"},
                "output_config": {
                    "format": {"type": "json_schema", "name": "synthetic", "schema": schema},
                },
                "messages": [
                    {
                        "role": "user",
                        "content": "Return JSON whose value is synthetic.",
                    },
                ],
            },
        )
    _require_success(response, target, "extended feature probe")
    message = response.json()
    try:
        structured = json.loads(_text_content(message))
    except json.JSONDecodeError as exc:
        raise AssertionError(
            f"{target.dialect}/{target.model} returned invalid structured output",
        ) from exc
    if not isinstance(structured, dict) or structured.get("value") != "synthetic":
        raise AssertionError(
            f"{target.dialect}/{target.model} returned the wrong structured output shape",
        )
    if message.get("model") != target.client_model:
        raise AssertionError(f"{target.dialect}/{target.model} model mapping was not preserved")
    assert isinstance(message.get("usage"), dict), (
        f"{target.dialect}/{target.model} returned no usage metadata"
    )
    cache_fields = {
        "cache_creation_input_tokens",
        "cache_read_input_tokens",
    }.intersection(message["usage"])
    if not cache_fields:
        _skip(f"{target.dialect}/{target.model} exposed no prompt-cache usage fields")
