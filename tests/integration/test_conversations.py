"""End-to-end multi-turn conversations through a stateful fake upstream."""

from __future__ import annotations

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
    {
        "name": "summarize_text",
        "description": "Summarize text",
        "input_schema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
            "additionalProperties": False,
        },
    },
]


def _fixture(relative_path: str) -> bytes:
    return (FIXTURES / relative_path).read_bytes()


def _assert_session_header(request: httpx.Request) -> None:
    assert request.headers["x-claude-code-session-id"] == "session-integration"


def _validate_chat_initial(request: httpx.Request, body: dict[str, Any]) -> None:
    _assert_session_header(request)
    assert body["model"] == "gpt-test"
    assert body["stream"] is True
    assert body["messages"] == [{"role": "user", "content": "Read the README"}]
    assert body["tools"][0]["function"]["name"] == "read_file"


def _validate_chat_follow_up(request: httpx.Request, body: dict[str, Any]) -> None:
    _assert_session_header(request)
    messages = body["messages"]
    assert [message["role"] for message in messages] == ["user", "assistant", "tool"]
    assert messages[1]["tool_calls"] == [
        {
            "id": "call_read",
            "type": "function",
            "function": {"name": "read_file", "arguments": '{"path":"README.md"}'},
        },
    ]
    assert messages[2] == {
        "role": "tool",
        "tool_call_id": "call_read",
        "content": "README contents",
    }


def _validate_responses_initial(request: httpx.Request, body: dict[str, Any]) -> None:
    _assert_session_header(request)
    assert body["model"] == "gpt-test"
    assert body["stream"] is True
    assert body["store"] is False
    assert body["input"] == [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "Read the README"}],
        },
    ]
    assert body["tools"][0]["name"] == "read_file"


def _validate_responses_buffered_initial(
    request: httpx.Request,
    body: dict[str, Any],
) -> None:
    _assert_session_header(request)
    assert body["stream"] is False
    assert body["input"] == [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "Read the README"}],
        },
    ]


def _validate_responses_follow_up(request: httpx.Request, body: dict[str, Any]) -> None:
    _assert_session_header(request)
    assert [item["type"] for item in body["input"]] == [
        "message",
        "function_call",
        "function_call_output",
    ]
    assert body["input"][1] == {
        "type": "function_call",
        "id": "fc_read",
        "call_id": "call_read",
        "name": "read_file",
        "arguments": '{"path":"README.md"}',
    }
    assert body["input"][2] == {
        "type": "function_call_output",
        "call_id": "call_read",
        "output": "README contents",
    }


def _validate_responses_second_follow_up(
    request: httpx.Request,
    body: dict[str, Any],
) -> None:
    _assert_session_header(request)
    assert [item["type"] for item in body["input"]] == [
        "message",
        "function_call",
        "function_call_output",
        "function_call",
        "function_call_output",
    ]
    assert body["input"][3] == {
        "type": "function_call",
        "id": "fc_summarize",
        "call_id": "call_summarize",
        "name": "summarize_text",
        "arguments": '{"text":"README contents"}',
    }
    assert body["input"][4] == {
        "type": "function_call_output",
        "call_id": "call_summarize",
        "output": "Summary result",
    }


def _validate_reasoning_follow_up(request: httpx.Request, body: dict[str, Any]) -> None:
    _assert_session_header(request)
    if body.get("previous_response_id"):
        assert body["previous_response_id"] == "resp_reasoning_1"
        return

    reasoning_items = [item for item in body["input"] if item.get("type") == "reasoning"]
    expected = [
        {
            "type": "reasoning",
            "id": "rs_1",
            "summary": [],
            "encrypted_content": "opaque-reasoning-state",
        },
    ]
    if reasoning_items != expected:
        msg = "The continuation did not replay the encrypted reasoning item"
        raise AssertionError(msg)
    item_types = [item["type"] for item in body["input"]]
    if not (
        item_types.index("reasoning")
        < item_types.index("function_call")
        < item_types.index("function_call_output")
    ):
        msg = "The replayed reasoning and function items were out of order"
        raise AssertionError(msg)


def _validate_parallel_reasoning_follow_up(
    request: httpx.Request,
    body: dict[str, Any],
) -> None:
    _assert_session_header(request)
    assert [item["type"] for item in body["input"]] == [
        "message",
        "reasoning",
        "function_call",
        "function_call",
        "function_call_output",
        "function_call_output",
    ]
    assert body["input"][1] == {
        "type": "reasoning",
        "id": "rs_parallel",
        "summary": [],
        "encrypted_content": "opaque-parallel-state",
    }
    assert [item.get("call_id") for item in body["input"][2:]] == [
        "call_read",
        "call_summarize",
        "call_read",
        "call_summarize",
    ]


def _validate_responses_final_answer_follow_up(
    request: httpx.Request,
    body: dict[str, Any],
) -> None:
    _assert_session_header(request)
    assert body["input"] == [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "Read the README"}],
        },
        {
            "type": "message",
            "id": "msg_final",
            "role": "assistant",
            "status": "completed",
            "content": [
                {
                    "type": "output_text",
                    "text": "Read complete.",
                    "annotations": [],
                },
            ],
        },
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "What next?"}],
        },
    ]


def _buffered_final_answer() -> bytes:
    return json.dumps(
        {
            "id": "resp_final",
            "model": "gpt-test",
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "id": "msg_final",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "Read complete.",
                            "annotations": [],
                        },
                    ],
                },
            ],
            "usage": {"input_tokens": 10, "output_tokens": 3},
        },
        separators=(",", ":"),
    ).encode()


@pytest.mark.anyio
async def test_chat_completions_tool_loop_reaches_a_final_answer() -> None:
    scenario = ScenarioTransport(
        [
            ExpectedRequest(
                method="POST",
                path="/v1/chat/completions",
                validate=_validate_chat_initial,
                response_body=_fixture("chat/tool_call.sse"),
            ),
            ExpectedRequest(
                method="POST",
                path="/v1/chat/completions",
                validate=_validate_chat_follow_up,
                response_body=_fixture("chat/final_answer.sse"),
            ),
        ],
    )
    app = create_app(
        Settings(model_map={"claude-test": "gpt-test"}),
        transport=scenario,
    )

    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://proxy",
        ) as client,
    ):
        conversation = AnthropicConversation(
            client,
            model="claude-test",
            tools=TOOLS,
            headers={"x-claude-code-session-id": "session-integration"},
        )
        first = await conversation.ask("Read the README")
        assert first.message["stop_reason"] == "tool_use"
        assert first.tool_uses == [
            {
                "type": "tool_use",
                "id": "call_read",
                "name": "read_file",
                "input": {"path": "README.md"},
            },
        ]

        final = await conversation.submit_tool_results({"call_read": "README contents"})

    assert final.message["stop_reason"] == "end_turn"
    assert [block for block in final.message["content"] if block["type"] == "text"] == [
        {"type": "text", "text": "Read complete."}
    ]
    scenario.assert_complete()


@pytest.mark.anyio
async def test_responses_final_answer_is_replayed_on_a_later_user_turn() -> None:
    scenario = ScenarioTransport(
        [
            ExpectedRequest(
                method="POST",
                path="/v1/responses",
                validate=_validate_responses_initial,
                response_body=_fixture("responses/final_answer.sse"),
            ),
            ExpectedRequest(
                method="POST",
                path="/v1/responses",
                validate=_validate_responses_final_answer_follow_up,
                response_body=_fixture("responses/final_answer.sse"),
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
        conversation = AnthropicConversation(
            client,
            model="claude-test",
            tools=TOOLS,
            headers={"x-claude-code-session-id": "session-integration"},
        )
        first = await conversation.ask("Read the README")
        assert first.message["stop_reason"] == "end_turn"
        assert first.message["content"][-1]["type"] == "redacted_thinking"
        second = await conversation.ask("What next?")

    assert second.message["stop_reason"] == "end_turn"
    scenario.assert_complete()


@pytest.mark.anyio
async def test_buffered_responses_final_answer_is_replayed_on_a_later_user_turn() -> None:
    scenario = ScenarioTransport(
        [
            ExpectedRequest(
                method="POST",
                path="/v1/responses",
                validate=_validate_responses_buffered_initial,
                response_body=_buffered_final_answer(),
                response_headers={"content-type": "application/json"},
            ),
            ExpectedRequest(
                method="POST",
                path="/v1/responses",
                validate=_validate_responses_final_answer_follow_up,
                response_body=_buffered_final_answer(),
                response_headers={"content-type": "application/json"},
            ),
        ],
    )
    app = create_app(
        Settings(openai_api="responses", model_map={"claude-test": "gpt-test"}),
        transport=scenario,
    )
    headers = {"x-claude-code-session-id": "session-integration"}

    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://proxy",
        ) as client,
    ):
        first = await client.post(
            "/v1/messages",
            headers=headers,
            json={
                "model": "claude-test",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "Read the README"}],
            },
        )
        first.raise_for_status()
        first_message = first.json()
        assert first_message["content"][0]["type"] == "redacted_thinking"

        second = await client.post(
            "/v1/messages",
            headers=headers,
            json={
                "model": "claude-test",
                "max_tokens": 1024,
                "messages": [
                    {"role": "user", "content": "Read the README"},
                    {"role": "assistant", "content": first_message["content"]},
                    {"role": "user", "content": "What next?"},
                ],
            },
        )

    second.raise_for_status()
    assert second.json()["stop_reason"] == "end_turn"
    scenario.assert_complete()


@pytest.mark.anyio
async def test_responses_tool_loop_reaches_a_final_answer() -> None:
    scenario = ScenarioTransport(
        [
            ExpectedRequest(
                method="POST",
                path="/v1/responses",
                validate=_validate_responses_initial,
                response_body=_fixture("responses/tool_call.sse"),
            ),
            ExpectedRequest(
                method="POST",
                path="/v1/responses",
                validate=_validate_responses_follow_up,
                response_body=_fixture("responses/second_tool_call.sse"),
            ),
            ExpectedRequest(
                method="POST",
                path="/v1/responses",
                validate=_validate_responses_second_follow_up,
                response_body=_fixture("responses/final_answer.sse"),
            ),
        ],
    )
    app = create_app(
        Settings(
            openai_api="responses",
            model_map={"claude-test": "gpt-test"},
        ),
        transport=scenario,
    )

    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://proxy",
        ) as client,
    ):
        conversation = AnthropicConversation(
            client,
            model="claude-test",
            tools=TOOLS,
            headers={"x-claude-code-session-id": "session-integration"},
        )
        first = await conversation.ask("Read the README")
        assert first.message["stop_reason"] == "tool_use"
        assert first.tool_uses[0]["input"] == {"path": "README.md"}

        second = await conversation.submit_tool_results({"call_read": "README contents"})
        assert second.message["stop_reason"] == "tool_use"
        assert second.tool_uses[0] == {
            "type": "tool_use",
            "id": "call_summarize",
            "name": "summarize_text",
            "input": {"text": "README contents"},
        }

        final = await conversation.submit_tool_results({"call_summarize": "Summary result"})

    assert final.message["stop_reason"] == "end_turn"
    assert [block for block in final.message["content"] if block["type"] == "text"] == [
        {"type": "text", "text": "Read complete."}
    ]
    scenario.assert_complete()


@pytest.mark.anyio
async def test_responses_reasoning_state_survives_a_proxy_restart() -> None:
    first_scenario = ScenarioTransport(
        [
            ExpectedRequest(
                method="POST",
                path="/v1/responses",
                validate=_validate_responses_initial,
                response_body=_fixture("responses/reasoning_tool_call.sse"),
            ),
        ],
    )
    first_app = create_app(
        Settings(
            openai_api="responses",
            model_map={"claude-test": "gpt-test"},
        ),
        transport=first_scenario,
    )

    async with (
        first_app.router.lifespan_context(first_app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=first_app),
            base_url="http://proxy",
        ) as client,
    ):
        first_conversation = AnthropicConversation(
            client,
            model="claude-test",
            tools=TOOLS,
            headers={"x-claude-code-session-id": "session-integration"},
        )
        first = await first_conversation.ask("Read the README")
        assert first.message["stop_reason"] == "tool_use"
        assert first.message["content"][0]["type"] == "redacted_thinking"
        messages = first_conversation.messages.copy()

    first_scenario.assert_complete()

    second_scenario = ScenarioTransport(
        [
            ExpectedRequest(
                method="POST",
                path="/v1/responses",
                validate=_validate_reasoning_follow_up,
                response_body=_fixture("responses/final_answer.sse"),
            ),
        ],
    )
    second_app = create_app(
        Settings(
            openai_api="responses",
            model_map={"claude-test": "gpt-test"},
        ),
        transport=second_scenario,
    )
    async with (
        second_app.router.lifespan_context(second_app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=second_app),
            base_url="http://proxy",
        ) as client,
    ):
        second_conversation = AnthropicConversation(
            client,
            model="claude-test",
            tools=TOOLS,
            headers={"x-claude-code-session-id": "session-integration"},
        )
        second_conversation.messages = messages
        final = await second_conversation.submit_tool_results(
            {"call_read": "README contents"},
        )

    assert final.message["stop_reason"] == "end_turn"
    second_scenario.assert_complete()


@pytest.mark.anyio
async def test_responses_reasoning_state_precedes_parallel_tool_calls() -> None:
    scenario = ScenarioTransport(
        [
            ExpectedRequest(
                method="POST",
                path="/v1/responses",
                validate=_validate_responses_initial,
                response_body=_fixture("responses/reasoning_parallel_tool_call.sse"),
            ),
            ExpectedRequest(
                method="POST",
                path="/v1/responses",
                validate=_validate_parallel_reasoning_follow_up,
                response_body=_fixture("responses/final_answer.sse"),
            ),
        ],
    )
    app = create_app(
        Settings(
            openai_api="responses",
            model_map={"claude-test": "gpt-test"},
        ),
        transport=scenario,
    )

    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://proxy",
        ) as client,
    ):
        conversation = AnthropicConversation(
            client,
            model="claude-test",
            tools=TOOLS,
            headers={"x-claude-code-session-id": "session-integration"},
        )
        first = await conversation.ask("Read the README")
        assert [block["type"] for block in first.message["content"]] == [
            "redacted_thinking",
            "tool_use",
            "tool_use",
            "redacted_thinking",
        ]
        final = await conversation.submit_tool_results(
            {
                "call_read": "README contents",
                "call_summarize": "Summary result",
            },
        )

    assert final.message["stop_reason"] == "end_turn"
    scenario.assert_complete()
