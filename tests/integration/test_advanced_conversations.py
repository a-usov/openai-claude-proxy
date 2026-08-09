"""Deterministic multi-step Responses scenarios beyond the basic happy path."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from openai_claude_proxy.app import create_app
from openai_claude_proxy.config import Settings
from tests.support.conversation import AnthropicConversation
from tests.support.scenario import ExpectedRequest, ScenarioTransport

FIXTURES = Path(__file__).parents[1] / "fixtures"
TOOLS = [
    {
        "name": "read_file",
        "description": "Read a workspace file",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
            "additionalProperties": False,
        },
    },
]


def _fixture(relative_path: str) -> bytes:
    return (FIXTURES / relative_path).read_bytes()


def _event_stream(events: list[dict[str, Any]]) -> bytes:
    return "".join(
        f"data: {json.dumps(event, separators=(',', ':'))}\n\n" for event in events
    ).encode()


def _tool_call_stream(
    *,
    response_id: str,
    item_id: str,
    call_id: str,
    arguments: str,
) -> bytes:
    item = {
        "type": "function_call",
        "id": item_id,
        "call_id": call_id,
        "name": "read_file",
        "arguments": arguments,
    }
    return _event_stream(
        [
            {
                "type": "response.created",
                "response": {
                    "id": response_id,
                    "model": "gpt-test",
                    "status": "in_progress",
                },
            },
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {**item, "arguments": ""},
            },
            {
                "type": "response.function_call_arguments.delta",
                "output_index": 0,
                "item_id": item_id,
                "delta": arguments,
            },
            {"type": "response.output_item.done", "output_index": 0, "item": item},
            {
                "type": "response.completed",
                "response": {
                    "id": response_id,
                    "model": "gpt-test",
                    "status": "completed",
                    "output": [item],
                    "usage": {"input_tokens": 10, "output_tokens": 5},
                },
            },
        ],
    )


def _text_stream(text: str, *, response_id: str = "resp_final") -> bytes:
    item = {
        "type": "message",
        "id": f"msg_{response_id}",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }
    return _event_stream(
        [
            {
                "type": "response.created",
                "response": {
                    "id": response_id,
                    "model": "gpt-test",
                    "status": "in_progress",
                },
            },
            {
                "type": "response.output_text.delta",
                "output_index": 0,
                "content_index": 0,
                "item_id": item["id"],
                "delta": text,
            },
            {
                "type": "response.output_text.done",
                "output_index": 0,
                "content_index": 0,
                "item_id": item["id"],
                "text": text,
            },
            {"type": "response.output_item.done", "output_index": 0, "item": item},
            {
                "type": "response.completed",
                "response": {
                    "id": response_id,
                    "model": "gpt-test",
                    "status": "completed",
                    "output": [item],
                    "usage": {"input_tokens": 20, "output_tokens": 5},
                },
            },
        ],
    )


def _refusal_stream(text: str) -> bytes:
    item = {
        "type": "message",
        "id": "msg_refusal",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "refusal", "refusal": text}],
    }
    return _event_stream(
        [
            {
                "type": "response.created",
                "response": {
                    "id": "resp_refusal",
                    "model": "gpt-test",
                    "status": "in_progress",
                },
            },
            {
                "type": "response.refusal.delta",
                "output_index": 0,
                "content_index": 0,
                "item_id": "msg_refusal",
                "delta": text,
            },
            {
                "type": "response.refusal.done",
                "output_index": 0,
                "content_index": 0,
                "item_id": "msg_refusal",
                "refusal": text,
            },
            {"type": "response.output_item.done", "output_index": 0, "item": item},
            {
                "type": "response.completed",
                "response": {
                    "id": "resp_refusal",
                    "model": "gpt-test",
                    "status": "completed",
                    "output": [item],
                    "usage": {"input_tokens": 20, "output_tokens": 3},
                },
            },
        ],
    )


def _buffered_refusal(text: str) -> bytes:
    return json.dumps(
        {
            "id": "resp_refusal",
            "model": "gpt-test",
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "id": "msg_refusal",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "refusal", "refusal": text}],
                },
            ],
            "usage": {"input_tokens": 20, "output_tokens": 3},
        },
        separators=(",", ":"),
    ).encode()


def _termination_stream(reason: str) -> bytes:
    return _event_stream(
        [
            {
                "type": "response.created",
                "response": {
                    "id": f"resp_{reason}",
                    "model": "gpt-test",
                    "status": "in_progress",
                },
            },
            {
                "type": "response.incomplete",
                "response": {
                    "id": f"resp_{reason}",
                    "model": "gpt-test",
                    "status": "incomplete",
                    "incomplete_details": {"reason": reason},
                    "output": [],
                    "usage": {"input_tokens": 20, "output_tokens": 0},
                },
            },
        ],
    )


def _reasoning_tool_stream(*, tag: str, model: str) -> bytes:
    call_id = f"call_{tag}"
    reasoning = {
        "type": "reasoning",
        "id": f"rs_{tag}",
        "summary": [],
        "encrypted_content": f"opaque-{tag}",
    }
    tool = {
        "type": "function_call",
        "id": f"fc_{tag}",
        "call_id": call_id,
        "name": "read_file",
        "arguments": json.dumps({"path": f"{tag}.txt"}, separators=(",", ":")),
    }
    return _event_stream(
        [
            {
                "type": "response.created",
                "response": {"id": f"resp_{tag}", "model": model, "status": "in_progress"},
            },
            {"type": "response.output_item.added", "output_index": 0, "item": reasoning},
            {"type": "response.output_item.done", "output_index": 0, "item": reasoning},
            {"type": "response.output_item.added", "output_index": 1, "item": tool},
            {"type": "response.output_item.done", "output_index": 1, "item": tool},
            {
                "type": "response.completed",
                "response": {
                    "id": f"resp_{tag}",
                    "model": model,
                    "status": "completed",
                    "output": [reasoning, tool],
                    "usage": {"input_tokens": 10, "output_tokens": 5},
                },
            },
        ],
    )


def _validate_initial(_request: httpx.Request, body: dict[str, Any]) -> None:
    assert body["model"] == "gpt-test"
    assert body["input"] == [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "Start"}],
        },
    ]


@pytest.mark.anyio
async def test_tool_failure_can_recover_with_another_tool_call() -> None:
    def validate_failure(_request: httpx.Request, body: dict[str, Any]) -> None:
        assert body["input"][-1] == {
            "type": "function_call_output",
            "call_id": "call_read",
            "output": "[Tool error]\npermission denied",
        }

    def validate_recovery(_request: httpx.Request, body: dict[str, Any]) -> None:
        assert body["input"][-1] == {
            "type": "function_call_output",
            "call_id": "call_recover",
            "output": "README contents",
        }

    scenario = ScenarioTransport(
        [
            ExpectedRequest(
                "POST",
                "/v1/responses",
                _validate_initial,
                _tool_call_stream(
                    response_id="resp_read",
                    item_id="fc_read",
                    call_id="call_read",
                    arguments='{"path":"README.md"}',
                ),
            ),
            ExpectedRequest(
                "POST",
                "/v1/responses",
                validate_failure,
                _tool_call_stream(
                    response_id="resp_recover",
                    item_id="fc_recover",
                    call_id="call_recover",
                    arguments='{"path":"README.md.bak"}',
                ),
            ),
            ExpectedRequest(
                "POST",
                "/v1/responses",
                validate_recovery,
                _text_stream("Recovered."),
            ),
        ],
    )
    app = create_app(
        Settings(openai_api="responses", model_map={"claude-test": "gpt-test"}),
        transport=scenario,
    )

    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://proxy",
        ) as client,
    ):
        conversation = AnthropicConversation(client, model="claude-test", tools=TOOLS)
        first = await conversation.ask("Start")
        assert first.message["stop_reason"] == "tool_use"
        second = await conversation.submit_tool_results(
            {"call_read": "permission denied"},
            error_ids={"call_read"},
        )
        assert second.tool_uses[0]["id"] == "call_recover"
        final = await conversation.submit_tool_results(
            {"call_recover": "README contents"},
        )

    assert final.message["stop_reason"] == "end_turn"
    assert final.message["content"][0] == {"type": "text", "text": "Recovered."}
    scenario.assert_complete()


@pytest.mark.anyio
@pytest.mark.parametrize("upstream_mode", ["streamed", "buffered"])
async def test_refusal_on_a_later_conversation_turn(upstream_mode: str) -> None:
    streamed_upstream = upstream_mode == "streamed"

    def validate_second_turn(_request: httpx.Request, body: dict[str, Any]) -> None:
        assert body["input"][-1]["content"][0]["text"] == "Now do the unsafe thing"

    scenario = ScenarioTransport(
        [
            ExpectedRequest("POST", "/v1/responses", _validate_initial, _text_stream("Okay.")),
            ExpectedRequest(
                "POST",
                "/v1/responses",
                validate_second_turn,
                (
                    _refusal_stream("I cannot help with that.")
                    if streamed_upstream
                    else _buffered_refusal("I cannot help with that.")
                ),
                response_headers={
                    "content-type": (
                        "text/event-stream" if streamed_upstream else "application/json"
                    ),
                },
            ),
        ],
    )
    app = create_app(
        Settings(openai_api="responses", model_map={"claude-test": "gpt-test"}),
        transport=scenario,
    )

    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://proxy",
        ) as client,
    ):
        conversation = AnthropicConversation(client, model="claude-test", tools=TOOLS)
        await conversation.ask("Start")
        refusal = await conversation.ask("Now do the unsafe thing")

    assert refusal.message["stop_reason"] == "refusal"
    assert refusal.message["stop_details"] == {
        "type": "refusal",
        "explanation": "I cannot help with that.",
    }
    scenario.assert_complete()


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("provider_reason", "anthropic_reason"),
    [
        ("max_output_tokens", "max_tokens"),
        ("content_filter", "refusal"),
        ("context_window_exceeded", "model_context_window_exceeded"),
    ],
)
async def test_termination_after_a_tool_result(
    provider_reason: str,
    anthropic_reason: str,
) -> None:
    def validate_tool_result(_request: httpx.Request, body: dict[str, Any]) -> None:
        assert body["input"][-1]["type"] == "function_call_output"

    scenario = ScenarioTransport(
        [
            ExpectedRequest(
                "POST",
                "/v1/responses",
                _validate_initial,
                _fixture("responses/tool_call.sse"),
            ),
            ExpectedRequest(
                "POST",
                "/v1/responses",
                validate_tool_result,
                _termination_stream(provider_reason),
            ),
        ],
    )
    app = create_app(
        Settings(openai_api="responses", model_map={"claude-test": "gpt-test"}),
        transport=scenario,
    )

    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://proxy",
        ) as client,
    ):
        conversation = AnthropicConversation(client, model="claude-test", tools=TOOLS)
        await conversation.ask("Start")
        terminal = await conversation.submit_tool_results({"call_read": "README contents"})

    assert terminal.message["stop_reason"] == anthropic_reason
    scenario.assert_complete()


@pytest.mark.anyio
async def test_structured_json_output_survives_a_tool_round_trip() -> None:
    expected_format = {
        "type": "json_schema",
        "name": "answer",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": False,
        },
    }

    def validate_format(_request: httpx.Request, body: dict[str, Any]) -> None:
        assert body["text"]["format"] == expected_format

    def validate_format_after_tool(_request: httpx.Request, body: dict[str, Any]) -> None:
        validate_format(_request, body)
        assert body["input"][-1]["type"] == "function_call_output"

    scenario = ScenarioTransport(
        [
            ExpectedRequest(
                "POST",
                "/v1/responses",
                validate_format,
                _fixture("responses/tool_call.sse"),
            ),
            ExpectedRequest(
                "POST",
                "/v1/responses",
                validate_format_after_tool,
                _text_stream('{"answer":"done"}'),
            ),
        ],
    )
    app = create_app(
        Settings(openai_api="responses", model_map={"claude-test": "gpt-test"}),
        transport=scenario,
    )

    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://proxy",
        ) as client,
    ):
        conversation = AnthropicConversation(client, model="claude-test", tools=TOOLS)
        conversation.request_overrides = {
            "output_config": {
                "format": {
                    "type": "json_schema",
                    "name": "answer",
                    "schema": expected_format["schema"],
                },
            },
        }
        await conversation.ask("Start")
        final = await conversation.submit_tool_results({"call_read": "README contents"})

    assert final.message["content"][0]["text"] == '{"answer":"done"}'
    scenario.assert_complete()


@pytest.mark.anyio
async def test_prompt_cache_breakpoint_tracks_growing_tool_history() -> None:
    def validate_initial_cache(_request: httpx.Request, body: dict[str, Any]) -> None:
        assert body["prompt_cache_options"] == {"mode": "explicit", "ttl": "30m"}
        assert body["input"][0]["content"][0]["prompt_cache_breakpoint"] == {
            "mode": "explicit",
        }

    def validate_grown_cache(_request: httpx.Request, body: dict[str, Any]) -> None:
        assert [item["type"] for item in body["input"]] == [
            "message",
            "function_call",
            "function_call_output",
        ]
        assert body["prompt_cache_options"] == {"mode": "explicit", "ttl": "30m"}
        assert body["input"][-1]["output"] == [
            {
                "type": "input_text",
                "text": "README contents",
                "prompt_cache_breakpoint": {"mode": "explicit"},
            },
        ]

    scenario = ScenarioTransport(
        [
            ExpectedRequest(
                "POST",
                "/v1/responses",
                validate_initial_cache,
                _fixture("responses/tool_call.sse"),
            ),
            ExpectedRequest(
                "POST",
                "/v1/responses",
                validate_grown_cache,
                _text_stream("Cached."),
            ),
        ],
    )
    app = create_app(
        Settings(openai_api="responses", model_map={"claude-test": "gpt-test"}),
        transport=scenario,
    )

    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://proxy",
        ) as client,
    ):
        conversation = AnthropicConversation(client, model="claude-test", tools=TOOLS)
        conversation.request_overrides = {"cache_control": {"type": "ephemeral"}}
        await conversation.ask("Start")
        final = await conversation.submit_tool_results({"call_read": "README contents"})

    assert final.message["stop_reason"] == "end_turn"
    scenario.assert_complete()


@pytest.mark.anyio
async def test_identical_retries_are_stateless_and_idempotent() -> None:
    upstream_bodies: list[dict[str, Any]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        decoded = json.loads(request.content)
        assert isinstance(decoded, dict)
        upstream_bodies.append(decoded)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=_fixture("responses/reasoning_tool_call.sse"),
        )

    app = create_app(
        Settings(openai_api="responses", model_map={"claude-test": "gpt-test"}),
        transport=httpx.MockTransport(handler),
    )
    payload = {
        "model": "claude-test",
        "max_tokens": 100,
        "stream": True,
        "messages": [{"role": "user", "content": "Start"}],
        "tools": TOOLS,
    }

    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://proxy",
        ) as client,
    ):
        first, retry = await asyncio.gather(
            client.post("/v1/messages", json=payload),
            client.post("/v1/messages", json=payload),
        )

    assert first.status_code == retry.status_code == 200
    assert first.content == retry.content
    assert upstream_bodies[0] == upstream_bodies[1]


class _IsolationUpstream:
    def __init__(self) -> None:
        self.seen: list[tuple[str, str, str, str]] = []

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert isinstance(body, dict)
        model = body["model"]
        effort = body["reasoning"]["effort"]
        agent = request.headers["x-claude-code-agent-id"]
        input_items = body["input"]
        reasoning_items = [item for item in input_items if item["type"] == "reasoning"]
        if reasoning_items:
            tag = reasoning_items[0]["id"].removeprefix("rs_")
            assert reasoning_items == [
                {
                    "type": "reasoning",
                    "id": f"rs_{tag}",
                    "summary": [],
                    "encrypted_content": f"opaque-{tag}",
                },
            ]
            assert input_items[-1] == {
                "type": "function_call_output",
                "call_id": f"call_{tag}",
                "output": f"result-{tag}",
            }
            response = _text_stream(f"done-{tag}", response_id=f"resp_done_{tag}")
        else:
            tag = input_items[0]["content"][0]["text"]
            response = _reasoning_tool_stream(tag=tag, model=model)
        self.seen.append((tag, model, effort, agent))
        await asyncio.sleep(0)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=response,
        )


@pytest.mark.anyio
async def test_concurrent_sessions_agents_models_efforts_and_state_remain_isolated() -> None:
    upstream = _IsolationUpstream()
    app = create_app(
        Settings(
            openai_api="responses",
            model_map={"claude-a": "gpt-a", "claude-b": "gpt-b"},
        ),
        transport=httpx.MockTransport(upstream),
    )
    cases = [
        ("alpha", "session-a", "parent", "claude-a", "low"),
        ("beta", "session-b", "agent-b", "claude-b", "high"),
        ("gamma", "shared", "parent", "claude-a", "medium"),
        ("delta", "shared", "subagent", "claude-b", "xhigh"),
        ("epsilon", "racing", "same-agent", "claude-a", "low"),
        ("zeta", "racing", "same-agent", "claude-b", "high"),
    ]

    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://proxy",
        ) as client,
    ):
        conversations: list[AnthropicConversation] = []
        for _tag, session, agent, model, effort in cases:
            conversation = AnthropicConversation(
                client,
                model=model,
                tools=TOOLS,
                headers={
                    "x-claude-code-session-id": session,
                    "x-claude-code-agent-id": agent,
                },
            )
            conversation.request_overrides = {"output_config": {"effort": effort}}
            conversations.append(conversation)

        first_turns = await asyncio.gather(
            *(
                conversation.ask(case[0])
                for conversation, case in zip(conversations, cases, strict=True)
            )
        )
        final_turns = await asyncio.gather(
            *(
                conversation.submit_tool_results({f"call_{case[0]}": f"result-{case[0]}"})
                for conversation, case in zip(conversations, cases, strict=True)
            ),
        )

    assert all(turn.message["stop_reason"] == "tool_use" for turn in first_turns)
    assert [turn.message["content"][0]["text"] for turn in final_turns] == [
        f"done-{tag}" for tag, *_rest in cases
    ]
    expected = {
        (tag, "gpt-a" if model == "claude-a" else "gpt-b", effort, agent)
        for tag, _session, agent, model, effort in cases
    }
    assert set(upstream.seen) == expected
    assert len(upstream.seen) == len(cases) * 2


@pytest.mark.anyio
async def test_model_remapping_change_rejects_state_from_another_upstream_model() -> None:
    first_scenario = ScenarioTransport(
        [
            ExpectedRequest(
                "POST",
                "/v1/responses",
                _validate_initial,
                _text_stream("Model A answer."),
            ),
        ],
    )
    first_app = create_app(
        Settings(openai_api="responses", model_map={"claude-test": "gpt-test"}),
        transport=first_scenario,
    )

    async with (
        first_app.router.lifespan_context(first_app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=first_app),
            base_url="http://proxy",
        ) as client,
    ):
        first_conversation = AnthropicConversation(client, model="claude-test", tools=TOOLS)
        await first_conversation.ask("Start")
        transcript = first_conversation.messages.copy()
    first_scenario.assert_complete()

    async def unexpected_handler(_request: httpx.Request) -> httpx.Response:
        msg = "State bound to model A must not reach remapped model B"
        raise AssertionError(msg)

    second_app = create_app(
        Settings(openai_api="responses", model_map={"claude-test": "gpt-model-b"}),
        transport=httpx.MockTransport(unexpected_handler),
    )
    async with (
        second_app.router.lifespan_context(second_app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=second_app),
            base_url="http://proxy",
        ) as client,
    ):
        second_conversation = AnthropicConversation(client, model="claude-test", tools=TOOLS)
        second_conversation.messages = transcript
        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            await second_conversation.ask("Continue after remapping")

    assert exc_info.value.response.status_code == 400
    assert exc_info.value.response.json()["error"]["message"] == (
        "Proxy Responses output state belongs to a different upstream model"
    )
