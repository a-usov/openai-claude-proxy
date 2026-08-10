from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

from openai_claude_proxy.config import ConfigError, Settings
from openai_claude_proxy.conversion import (
    ConversionError,
    anthropic_to_openai,
    anthropic_to_responses,
    finish_reason_to_anthropic,
    openai_to_anthropic,
    responses_stop_reason,
    responses_to_anthropic,
)
from openai_claude_proxy.reasoning_state import (
    decode_responses_output,
    decode_responses_reasoning,
)


def test_anthropic_request_converts_system_tools_and_tool_results() -> None:
    source = {
        "model": "claude-sonnet-4-5",
        "system": [{"type": "text", "text": "Be useful", "cache_control": {"type": "ephemeral"}}],
        "max_tokens": 1000,
        "stream": True,
        "messages": [
            {"role": "user", "content": "Inspect the repo"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "I'll inspect it."},
                    {"type": "tool_use", "id": "toolu_1", "name": "shell", "input": {"cmd": "ls"}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_1", "content": "README.md"},
                ],
            },
        ],
        "tools": [
            {
                "name": "shell",
                "description": "Run a command",
                "input_schema": {
                    "type": "object",
                    "properties": {"cmd": {"type": "string"}},
                    "required": ["cmd"],
                },
            },
        ],
        "tool_choice": {"type": "auto"},
    }
    settings = Settings(model_map={"claude-*": "company-coding-model"})

    result = anthropic_to_openai(source, settings)

    assert result["model"] == "company-coding-model"
    assert result["messages"][0] == {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": "Be useful",
                "prompt_cache_breakpoint": {"mode": "explicit"},
            },
        ],
    }
    assert result["prompt_cache_options"] == {"mode": "explicit", "ttl": "30m"}
    assert result["messages"][2]["tool_calls"][0]["function"]["arguments"] == '{"cmd":"ls"}'
    assert result["messages"][3] == {
        "role": "tool",
        "tool_call_id": "toolu_1",
        "content": "README.md",
    }
    assert result["tools"][0]["function"]["parameters"]["required"] == ["cmd"]
    assert result["stream_options"] == {"include_usage": True}


def test_chat_translates_structured_output_and_parallel_tool_preference() -> None:
    result = anthropic_to_openai(
        {
            "model": "claude-sonnet-4-6",
            "max_tokens": 100,
            "messages": [],
            "tools": [
                {
                    "name": "lookup",
                    "input_schema": {"type": "object", "additionalProperties": False},
                    "strict": True,
                },
            ],
            "tool_choice": {"type": "auto", "disable_parallel_tool_use": True},
            "output_config": {
                "format": {
                    "type": "json_schema",
                    "name": "answer",
                    "schema": {"type": "object", "additionalProperties": False},
                },
            },
        },
        Settings(),
    )

    assert result["parallel_tool_calls"] is False
    assert result["tools"][0]["function"]["strict"] is True
    assert result["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "answer",
            "schema": {"type": "object", "additionalProperties": False},
            "strict": True,
        },
    }


def test_model_passes_through_by_default_and_can_be_explicitly_overridden() -> None:
    assert Settings().map_model("claude-selected-model") == "claude-selected-model"
    assert (
        Settings(model_override="deployment-name").map_model("claude-selected-model")
        == "deployment-name"
    )
    assert Settings.from_env({"DEFAULT_MODEL": "legacy-name"}).model_override == "legacy-name"
    assert (
        Settings.from_env(
            {"MODEL_OVERRIDE": "new-name", "DEFAULT_MODEL": "legacy-name"},
        ).model_override
        == "new-name"
    )


def test_tool_errors_and_configurable_token_field_are_preserved() -> None:
    result = anthropic_to_openai(
        {
            "model": "claude",
            "max_tokens": 50,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_error",
                            "is_error": True,
                            "content": "command failed",
                        },
                    ],
                },
            ],
        },
        Settings(max_tokens_field="max_completion_tokens"),
    )

    assert "max_tokens" not in result
    assert result["max_completion_tokens"] == 50
    assert result["messages"][0]["content"] == "[Tool error]\ncommand failed"


def test_anthropic_effort_is_translated_for_gpt_56_models() -> None:
    base = {
        "model": "claude-opus-5",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "Solve this"}],
    }
    settings = Settings(model_map={"claude-opus-*": "gpt-5.6-sol"})

    for effort in ("low", "medium", "high", "xhigh", "max"):
        source = {**base, "output_config": {"effort": effort}}
        result = anthropic_to_openai(source, settings)
        assert result["model"] == "gpt-5.6-sol"
        assert result["reasoning_effort"] == effort


@pytest.mark.parametrize(
    "thinking",
    [
        {"type": "adaptive"},
        {"type": "adaptive", "display": "omitted"},
        {"type": "adaptive", "display": "summarized"},
        {"type": "disabled"},
    ],
)
@pytest.mark.parametrize(
    ("converter", "reasoning_field"),
    [
        (anthropic_to_openai, "reasoning_effort"),
        (anthropic_to_responses, "reasoning"),
    ],
)
def test_claude_code_thinking_modes_preserve_explicit_effort(
    thinking: dict[str, str],
    converter: Callable[[dict[str, Any], Settings], dict[str, Any]],
    reasoning_field: str,
) -> None:
    result = converter(
        {
            "model": "claude-opus-4-6",
            "max_tokens": 64000,
            "messages": [{"role": "user", "content": "Solve this"}],
            "thinking": thinking,
            "output_config": {"effort": "high"},
        },
        Settings(),
    )

    expected = "high" if reasoning_field == "reasoning_effort" else {"effort": "high"}
    assert result[reasoning_field] == expected


@pytest.mark.parametrize("converter", [anthropic_to_openai, anthropic_to_responses])
def test_manual_thinking_budget_is_rejected_without_guessing_effort(
    converter: Callable[[dict[str, Any], Settings], dict[str, Any]],
) -> None:
    with pytest.raises(ConversionError, match="fixed token budget"):
        converter(
            {
                "model": "claude-opus-4-6",
                "max_tokens": 4096,
                "messages": [{"role": "user", "content": "Solve this"}],
                "thinking": {"type": "enabled", "budget_tokens": 2048},
            },
            Settings(),
        )


@pytest.mark.parametrize("converter", [anthropic_to_openai, anthropic_to_responses])
def test_claude_code_keep_all_thinking_context_directive_is_a_safe_noop(
    converter: Callable[[dict[str, Any], Settings], dict[str, Any]],
) -> None:
    result = converter(
        {
            "model": "claude-opus-4-6",
            "max_tokens": 64000,
            "messages": [{"role": "user", "content": "Solve this"}],
            "thinking": {"type": "adaptive", "display": "omitted"},
            "context_management": {
                "edits": [{"type": "clear_thinking_20251015", "keep": "all"}],
            },
            "output_config": {"effort": "high"},
        },
        Settings(),
    )

    assert result["model"] == "claude-opus-4-6"


@pytest.mark.parametrize("converter", [anthropic_to_openai, anthropic_to_responses])
def test_context_management_that_changes_history_remains_rejected(
    converter: Callable[[dict[str, Any], Settings], dict[str, Any]],
) -> None:
    with pytest.raises(ConversionError, match="changes conversation context"):
        converter(
            {
                "model": "claude-opus-4-6",
                "max_tokens": 64000,
                "messages": [{"role": "user", "content": "Solve this"}],
                "context_management": {
                    "edits": [
                        {
                            "type": "clear_thinking_20251015",
                            "keep": {"type": "thinking_turns", "value": 2},
                        },
                    ],
                },
            },
            Settings(),
        )


def test_effort_mapping_can_remap_or_omit_provider_specific_levels() -> None:
    base = {
        "model": "claude-opus-5",
        "max_tokens": 100,
        "messages": [],
    }
    settings = Settings(reasoning_effort_map={"max": "high", "xhigh": None})

    remapped = anthropic_to_openai({**base, "output_config": {"effort": "max"}}, settings)
    omitted = anthropic_to_openai({**base, "output_config": {"effort": "xhigh"}}, settings)

    assert remapped["reasoning_effort"] == "high"
    assert "reasoning_effort" not in omitted


def test_effort_map_environment_overlays_safe_defaults() -> None:
    settings = Settings.from_env({"REASONING_EFFORT_MAP": '{"xhigh":"high","max":null}'})

    assert settings.reasoning_effort_map["low"] == "low"
    assert settings.reasoning_effort_map["xhigh"] == "high"
    assert settings.reasoning_effort_map["max"] is None

    with pytest.raises(ConfigError, match="values must be strings or null"):
        Settings.from_env({"REASONING_EFFORT_MAP": '{"max":5}'})


def test_legacy_top_level_effort_is_supported_but_conflicts_are_rejected() -> None:
    base = {
        "model": "claude-sonnet-4-6",
        "max_tokens": 100,
        "messages": [],
    }

    result = anthropic_to_openai({**base, "effort": "high"}, Settings())
    assert result["reasoning_effort"] == "high"

    with pytest.raises(ConversionError, match="must match"):
        anthropic_to_openai(
            {
                **base,
                "effort": "low",
                "output_config": {"effort": "max"},
            },
            Settings(),
        )


def test_standard_effort_null_and_legacy_none_have_explicit_behavior() -> None:
    base = {
        "model": "claude-sonnet-4-6",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "Solve this"}],
    }

    assert "reasoning_effort" not in anthropic_to_openai(
        {**base, "output_config": {"effort": None}},
        Settings(),
    )
    assert anthropic_to_openai({**base, "effort": "none"}, Settings())["reasoning_effort"] == (
        "none"
    )
    with pytest.raises(ConversionError, match=r"output_config\.effort"):
        anthropic_to_openai(
            {**base, "output_config": {"effort": "none"}},
            Settings(),
        )


def test_effort_is_omitted_when_absent_or_translation_is_disabled() -> None:
    base = {
        "model": "claude-sonnet-4-6",
        "max_tokens": 100,
        "messages": [],
    }

    assert "reasoning_effort" not in anthropic_to_openai(base, Settings())
    assert "reasoning_effort" not in anthropic_to_openai(
        {**base, "output_config": {"effort": "high"}},
        Settings(reasoning_effort_enabled=False),
    )


def test_static_extra_body_has_final_precedence_over_dynamic_effort() -> None:
    result = anthropic_to_openai(
        {
            "model": "claude-opus-5",
            "max_tokens": 100,
            "messages": [],
            "output_config": {"effort": "max"},
        },
        Settings(extra_openai_body={"reasoning_effort": "low"}),
    )

    assert result["reasoning_effort"] == "low"


def test_image_blocks_convert_to_openai_data_urls() -> None:
    result = anthropic_to_openai(
        {
            "model": "claude",
            "max_tokens": 10,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is this?"},
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": "image/png", "data": "abc"},
                        },
                    ],
                },
            ],
        },
        Settings(),
    )

    content = result["messages"][0]["content"]
    assert content[1]["image_url"]["url"] == "data:image/png;base64,abc"


def test_openai_tool_response_converts_to_anthropic() -> None:
    source = {
        "id": "chatcmpl_1",
        "model": "deployment-name",
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": "Checking.",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "read", "arguments": '{"path":"a.py"}'},
                        },
                    ],
                },
            },
        ],
        "usage": {
            "prompt_tokens": 20,
            "completion_tokens": 8,
            "completion_tokens_details": {"reasoning_tokens": 5},
        },
    }

    result = openai_to_anthropic(source, "claude")

    assert result["id"] == "chatcmpl_1"
    assert result["model"] == "claude"
    assert result["stop_reason"] == "tool_use"
    assert result["content"][1] == {
        "type": "tool_use",
        "id": "call_1",
        "name": "read",
        "input": {"path": "a.py"},
    }
    assert result["usage"] == {
        "input_tokens": 20,
        "output_tokens": 8,
        "output_tokens_details": {"thinking_tokens": 5},
    }


def test_anthropic_request_converts_to_typed_responses_items() -> None:
    result = anthropic_to_responses(
        {
            "model": "claude-opus-5",
            "system": [{"type": "text", "text": "Be precise"}],
            "max_tokens": 500,
            "stream": True,
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Checking."},
                        {
                            "type": "tool_use",
                            "id": "toolu_1",
                            "name": "shell",
                            "input": {"cmd": "pwd"},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_1",
                            "content": "/workspace",
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": "abc",
                            },
                        },
                    ],
                },
            ],
            "tools": [
                {
                    "name": "shell",
                    "description": "Run a command",
                    "input_schema": {"type": "object"},
                    "strict": True,
                },
            ],
            "tool_choice": {"type": "any", "disable_parallel_tool_use": True},
            "output_config": {
                "effort": "max",
                "format": {
                    "type": "json_schema",
                    "schema": {
                        "type": "object",
                        "properties": {"answer": {"type": "string"}},
                        "required": ["answer"],
                        "additionalProperties": False,
                    },
                },
            },
        },
        Settings(model_map={"claude-opus-5": "gpt-5.6-sol"}),
    )

    assert result["model"] == "gpt-5.6-sol"
    assert result["instructions"] == "Be precise"
    assert result["max_output_tokens"] == 500
    assert result["store"] is False
    assert result["reasoning"] == {"effort": "max"}
    assert result["parallel_tool_calls"] is False
    assert result["tool_choice"] == "required"
    assert result["tools"][0]["strict"] is True
    assert result["text"]["format"]["name"] == "anthropic_output"
    assert result["text"]["format"]["type"] == "json_schema"
    assert result["input"] == [
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "input_text", "text": "Checking."}],
        },
        {
            "type": "function_call",
            "call_id": "toolu_1",
            "name": "shell",
            "arguments": '{"cmd":"pwd"}',
        },
        {"type": "function_call_output", "call_id": "toolu_1", "output": "/workspace"},
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_image", "image_url": "data:image/png;base64,abc"}],
        },
    ]


def test_responses_result_maps_tools_model_and_cache_usage() -> None:
    reasoning_item = {"type": "reasoning", "id": "rs_1", "summary": []}
    result = responses_to_anthropic(
        {
            "id": "resp_1",
            "model": "gpt-5.6-sol",
            "status": "completed",
            "output": [
                reasoning_item,
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "I will inspect it."}],
                },
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "read",
                    "arguments": '{"path":"README.md"}',
                },
            ],
            "usage": {
                "input_tokens": 100,
                "input_tokens_details": {
                    "cached_tokens": 30,
                    "cache_write_tokens": 20,
                },
                "output_tokens": 12,
                "output_tokens_details": {"reasoning_tokens": 7},
            },
        },
        "claude-opus-5",
    )

    assert result["model"] == "claude-opus-5"
    assert result["stop_reason"] == "tool_use"
    assert result["content"][0]["type"] == "redacted_thinking"
    assert decode_responses_output(result["content"][0]["data"]) == [
        reasoning_item,
        {
            "type": "message",
            "content": [{"type": "output_text", "text": "I will inspect it."}],
        },
        {
            "type": "function_call",
            "call_id": "call_1",
            "name": "read",
            "arguments": '{"path":"README.md"}',
        },
    ]
    assert decode_responses_reasoning(result["content"][1]["data"]) == reasoning_item
    assert result["content"][2] == {"type": "text", "text": "I will inspect it."}
    assert result["content"][3] == {
        "type": "tool_use",
        "id": "call_1",
        "name": "read",
        "input": {"path": "README.md"},
    }
    assert result["usage"] == {
        "input_tokens": 50,
        "cache_read_input_tokens": 30,
        "cache_creation_input_tokens": 20,
        "output_tokens": 12,
        "output_tokens_details": {"thinking_tokens": 7},
    }


def test_chat_completion_refusal_preserves_text_and_stop_reason() -> None:
    result = openai_to_anthropic(
        {
            "id": "chatcmpl_refusal",
            "choices": [
                {
                    "message": {"role": "assistant", "refusal": "I cannot help."},
                    "finish_reason": "stop",
                },
            ],
        },
        "claude-test",
    )

    assert result["content"] == [{"type": "text", "text": "I cannot help."}]
    assert result["stop_reason"] == "refusal"
    assert result["stop_details"] == {
        "type": "refusal",
        "explanation": "I cannot help.",
    }


def test_chat_content_filter_without_refusal_text_uses_refusal_stop_reason() -> None:
    result = openai_to_anthropic(
        {
            "id": "chatcmpl_filtered",
            "choices": [
                {
                    "message": {"role": "assistant", "content": None},
                    "finish_reason": "content_filter",
                },
            ],
        },
        "claude-test",
    )

    assert result["stop_reason"] == "refusal"
    assert result["stop_details"] == {"type": "refusal", "explanation": None}


def test_responses_refusal_preserves_text_and_stop_reason() -> None:
    result = responses_to_anthropic(
        {
            "id": "resp_refusal",
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "id": "msg_refusal",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "refusal", "refusal": "I cannot help."}],
                },
            ],
        },
        "claude-test",
    )

    assert {"type": "text", "text": "I cannot help."} in result["content"]
    assert result["stop_reason"] == "refusal"
    assert result["stop_details"] == {
        "type": "refusal",
        "explanation": "I cannot help.",
    }


def test_responses_content_filter_without_refusal_text_uses_refusal_stop_reason() -> None:
    result = responses_to_anthropic(
        {
            "id": "resp_filtered",
            "status": "incomplete",
            "incomplete_details": {"reason": "content_filter"},
            "output": [],
        },
        "claude-test",
    )

    assert result["stop_reason"] == "refusal"
    assert result["stop_details"] == {"type": "refusal", "explanation": None}


@pytest.mark.parametrize("arguments", ["not-json", "[]", '"scalar"', "42", False])
def test_chat_completion_rejects_non_object_function_arguments(arguments: object) -> None:
    with pytest.raises(ConversionError, match="Upstream function arguments"):
        openai_to_anthropic(
            {
                "id": "chatcmpl_invalid_tool",
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "function": {"name": "read", "arguments": arguments},
                                },
                            ],
                        },
                        "finish_reason": "tool_calls",
                    },
                ],
            },
            "claude-test",
        )


@pytest.mark.parametrize("arguments", ["not-json", "[]", '"scalar"', "42", False])
def test_responses_rejects_non_object_function_arguments(arguments: object) -> None:
    with pytest.raises(ConversionError, match="Upstream function arguments"):
        responses_to_anthropic(
            {
                "id": "resp_invalid_tool",
                "status": "completed",
                "output": [
                    {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "read",
                        "arguments": arguments,
                    },
                ],
            },
            "claude-test",
        )


def test_empty_function_arguments_map_to_an_empty_object() -> None:
    chat = openai_to_anthropic(
        {
            "id": "chatcmpl_empty_tool",
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "function": {"name": "ping", "arguments": ""},
                            },
                        ],
                    },
                    "finish_reason": "tool_calls",
                },
            ],
        },
        "claude-test",
    )
    responses = responses_to_anthropic(
        {
            "id": "resp_empty_tool",
            "status": "completed",
            "output": [
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "ping",
                    "arguments": "",
                },
            ],
        },
        "claude-test",
    )

    assert chat["content"][0]["input"] == {}
    assert responses["content"][-1]["input"] == {}


def test_responses_reasoning_state_round_trips_in_order() -> None:
    reasoning_item = {
        "type": "reasoning",
        "id": "rs_1",
        "summary": [],
        "encrypted_content": "opaque-state",
    }
    translated = responses_to_anthropic(
        {
            "id": "resp_1",
            "status": "completed",
            "output": [
                reasoning_item,
                {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "call_1",
                    "name": "read",
                    "arguments": '{"path":"README.md"}',
                },
            ],
        },
        "claude-test",
        {"model": "claude-test"},
    )

    request = anthropic_to_responses(
        {
            "model": "claude-test",
            "max_tokens": 100,
            "messages": [
                {"role": "user", "content": "Read the README"},
                {"role": "assistant", "content": translated["content"]},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "call_1",
                            "content": "README contents",
                        },
                    ],
                },
            ],
        },
        Settings(),
    )

    assert [item["type"] for item in request["input"]] == [
        "message",
        "reasoning",
        "function_call",
        "function_call_output",
    ]
    assert request["input"][1] == reasoning_item
    assert request["input"][2] == {
        "type": "function_call",
        "id": "fc_1",
        "call_id": "call_1",
        "name": "read",
        "arguments": '{"path":"README.md"}',
    }


def test_responses_complete_output_state_round_trips_exactly() -> None:
    output = [
        {
            "type": "message",
            "id": "msg_1",
            "role": "assistant",
            "status": "completed",
            "phase": "final_answer",
            "content": [
                {
                    "type": "output_text",
                    "text": "I checked it.",
                    "annotations": [],
                },
            ],
        },
        {
            "type": "function_call",
            "id": "fc_1",
            "call_id": "call_1",
            "name": "read",
            "arguments": '{"path":"README.md"}',
            "status": "completed",
            "caller": "tool",
            "namespace": "workspace",
        },
    ]
    translated = responses_to_anthropic(
        {"id": "resp_1", "status": "completed", "output": output},
        "claude-test",
        {"model": "claude-test"},
    )

    request = anthropic_to_responses(
        {
            "model": "claude-test",
            "max_tokens": 100,
            "messages": [
                {"role": "assistant", "content": translated["content"]},
                {"role": "user", "content": "Continue"},
            ],
        },
        Settings(),
    )

    assert request["input"][:-1] == output


def test_responses_complete_output_state_rejects_visible_content_changes() -> None:
    translated = responses_to_anthropic(
        {
            "id": "resp_1",
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "id": "msg_1",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {"type": "output_text", "text": "Original", "annotations": []},
                    ],
                },
            ],
        },
        "claude-test",
        {"model": "claude-test"},
    )
    translated["content"][1]["text"] = "Modified"

    with pytest.raises(ConversionError, match="does not match"):
        anthropic_to_responses(
            {
                "model": "claude-test",
                "max_tokens": 100,
                "messages": [
                    {"role": "assistant", "content": translated["content"]},
                    {"role": "user", "content": "Continue"},
                ],
            },
            Settings(),
        )


def test_responses_rejects_stop_sequences_instead_of_dropping_them() -> None:
    with pytest.raises(ConversionError, match="stop_sequences"):
        anthropic_to_responses(
            {
                "model": "claude",
                "max_tokens": 10,
                "messages": [],
                "stop_sequences": ["STOP"],
            },
            Settings(),
        )


@pytest.mark.parametrize("converter", [anthropic_to_openai, anthropic_to_responses])
@pytest.mark.parametrize(
    "field",
    [
        "anthropic-user-profile-id",
        "container",
        "inference_geo",
        "service_tier",
        "top_k",
    ],
)
def test_unsupported_anthropic_capability_fields_are_rejected(
    converter: Callable[[dict[str, Any], Settings], dict[str, Any]],
    field: str,
) -> None:
    values = {
        "anthropic-user-profile-id": "profile_1",
        "container": "container_1",
        "inference_geo": "us",
        "service_tier": "standard_only",
        "top_k": 10,
    }
    request = {
        "model": "claude-test",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "Hello"}],
        field: values[field],
    }

    with pytest.raises(ConversionError, match=field):
        converter(request, Settings())


@pytest.mark.parametrize("converter", [anthropic_to_openai, anthropic_to_responses])
@pytest.mark.parametrize(
    ("patch", "error"),
    [
        (None, "max_tokens"),
        ({"max_tokens": True}, "max_tokens"),
        ({"stream": "true"}, "stream"),
        ({"messages": "hello"}, "messages"),
        ({"messages": [{"role": [], "content": "hello"}]}, "role"),
        ({"messages": [{"role": "user", "content": 7}]}, "content"),
        (
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "image", "source": {"type": []}}],
                    },
                ],
            },
            "source.type",
        ),
        ({"tools": [{"name": "lookup", "input_schema": []}]}, "input_schema"),
        ({"tool_choice": {"type": []}}, "tool_choice.type"),
        ({"future_capability": True}, "future_capability"),
    ],
)
def test_malformed_requests_raise_conversion_errors_for_each_backend(
    converter: Callable[[dict[str, Any], Settings], dict[str, Any]],
    patch: dict[str, object] | None,
    error: str,
) -> None:
    request: dict[str, Any] = {
        "model": "claude-test",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "Hello"}],
    }
    if patch is None:
        request.pop("max_tokens")
    else:
        request.update(patch)

    with pytest.raises(ConversionError, match=error):
        converter(request, Settings())


def test_metadata_and_mid_conversation_system_roles_translate_for_both_backends() -> None:
    source = {
        "model": "claude-test",
        "max_tokens": 100,
        "metadata": {"user_id": "opaque-user-123"},
        "messages": [
            {"role": "user", "content": "Start"},
            {"role": "system", "content": [{"type": "text", "text": "New policy"}]},
            {"role": "user", "content": "Continue"},
        ],
    }

    chat = anthropic_to_openai(source, Settings())
    responses = anthropic_to_responses(source, Settings())

    assert chat["safety_identifier"] == "opaque-user-123"
    assert chat["messages"][1] == {"role": "developer", "content": "New policy"}
    assert responses["safety_identifier"] == "opaque-user-123"
    assert responses["input"][1] == {
        "type": "message",
        "role": "developer",
        "content": [{"type": "input_text", "text": "New policy"}],
    }


def test_responses_maps_tool_callers_and_deferred_loading() -> None:
    source = {
        "model": "claude-test",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "Use a tool"}],
        "tools": [
            {
                "type": "custom",
                "name": "lookup",
                "input_schema": {"type": "object"},
                "allowed_callers": ["direct", "code_execution_20260521"],
                "defer_loading": True,
            },
        ],
    }

    result = anthropic_to_responses(source, Settings())

    assert result["tools"][0]["allowed_callers"] == ["direct", "programmatic"]
    assert result["tools"][0]["defer_loading"] is True
    with pytest.raises(ConversionError, match="allowed_callers"):
        anthropic_to_openai(source, Settings())


@pytest.mark.parametrize(
    "field",
    ["cache_control", "eager_input_streaming", "input_examples"],
)
def test_tool_fields_without_openai_equivalents_are_rejected(field: str) -> None:
    values: dict[str, object] = {
        "cache_control": {"type": "ephemeral"},
        "eager_input_streaming": True,
        "input_examples": [{"query": "example"}],
    }
    request = {
        "model": "claude-test",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "Use a tool"}],
        "tools": [
            {
                "name": "lookup",
                "input_schema": {"type": "object"},
                field: values[field],
            },
        ],
    }

    for converter in (anthropic_to_openai, anthropic_to_responses):
        with pytest.raises(ConversionError, match=field):
            converter(request, Settings())


def test_prompt_cache_breakpoints_translate_without_mutating_the_request() -> None:
    source = {
        "model": "claude-test",
        "max_tokens": 100,
        "cache_control": {"type": "ephemeral", "ttl": "1h"},
        "messages": [{"role": "user", "content": "Cache through here"}],
    }
    original = {
        "model": "claude-test",
        "max_tokens": 100,
        "cache_control": {"type": "ephemeral", "ttl": "1h"},
        "messages": [{"role": "user", "content": "Cache through here"}],
    }

    chat = anthropic_to_openai(source, Settings())
    responses = anthropic_to_responses(source, Settings())

    assert source == original
    assert chat["messages"][0]["content"][0]["prompt_cache_breakpoint"] == {
        "mode": "explicit",
    }
    assert responses["input"][0]["content"][0]["prompt_cache_breakpoint"] == {
        "mode": "explicit",
    }
    assert chat["prompt_cache_options"] == {"mode": "explicit", "ttl": "30m"}
    assert responses["prompt_cache_options"] == {"mode": "explicit", "ttl": "30m"}


def test_rich_tool_result_images_are_preserved_only_by_responses() -> None:
    source = {
        "model": "claude-test",
        "max_tokens": 100,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_1",
                        "cache_control": {"type": "ephemeral"},
                        "content": [
                            {"type": "text", "text": "Screenshot"},
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": "abc",
                                },
                            },
                        ],
                    },
                ],
            },
        ],
    }

    result = anthropic_to_responses(source, Settings())

    assert result["input"][0]["output"] == [
        {"type": "input_text", "text": "Screenshot"},
        {
            "type": "input_image",
            "image_url": "data:image/png;base64,abc",
            "prompt_cache_breakpoint": {"mode": "explicit"},
        },
    ]
    with pytest.raises(ConversionError, match="Chat Completions"):
        anthropic_to_openai(source, Settings())


@pytest.mark.parametrize("converter", [anthropic_to_openai, anthropic_to_responses])
def test_final_assistant_prefill_is_rejected(
    converter: Callable[[dict[str, Any], Settings], dict[str, Any]],
) -> None:
    with pytest.raises(ConversionError, match="assistant-prefill"):
        converter(
            {
                "model": "claude-test",
                "max_tokens": 100,
                "messages": [
                    {"role": "user", "content": "Choose A or B"},
                    {"role": "assistant", "content": "The answer is ("},
                ],
            },
            Settings(),
        )


def test_foreign_redacted_reasoning_is_not_silently_discarded() -> None:
    with pytest.raises(ConversionError, match="proxy-owned"):
        anthropic_to_responses(
            {
                "model": "claude-test",
                "max_tokens": 100,
                "messages": [
                    {
                        "role": "assistant",
                        "content": [{"type": "redacted_thinking", "data": "foreign-state"}],
                    },
                    {"role": "user", "content": "Continue"},
                ],
            },
            Settings(),
        )


@pytest.mark.parametrize(
    ("reason", "expected"),
    [
        (None, None),
        ("stop", "end_turn"),
        ("length", "max_tokens"),
        ("tool_calls", "tool_use"),
        ("function_call", "tool_use"),
        ("content_filter", "refusal"),
    ],
)
def test_chat_finish_reason_mapping_is_exhaustive(
    reason: str | None,
    expected: str | None,
) -> None:
    assert finish_reason_to_anthropic(reason) == expected


def test_chat_unknown_finish_reason_is_rejected() -> None:
    with pytest.raises(ConversionError, match="Unsupported OpenAI finish reason"):
        finish_reason_to_anthropic("future_reason")


def test_buffered_chat_response_requires_terminal_finish_reason() -> None:
    with pytest.raises(ConversionError, match="did not include a finish reason"):
        openai_to_anthropic(
            {
                "id": "chatcmpl_missing_stop",
                "choices": [{"message": {"content": "partial"}, "finish_reason": None}],
            },
            "claude-test",
        )


@pytest.mark.parametrize(
    ("status", "incomplete_reason", "has_tool_call", "has_refusal", "expected"),
    [
        ("completed", None, False, False, "end_turn"),
        ("completed", None, True, False, "tool_use"),
        ("completed", None, False, True, "refusal"),
        ("incomplete", "max_output_tokens", False, False, "max_tokens"),
        ("incomplete", "content_filter", False, False, "refusal"),
        (
            "incomplete",
            "context_window_exceeded",
            False,
            False,
            "model_context_window_exceeded",
        ),
    ],
)
def test_responses_terminal_mapping_is_exhaustive(
    status: str,
    incomplete_reason: str | None,
    *,
    has_tool_call: bool,
    has_refusal: bool,
    expected: str,
) -> None:
    assert (
        responses_stop_reason(
            status=status,
            incomplete_reason=incomplete_reason,
            has_tool_call=has_tool_call,
            has_refusal=has_refusal,
        )
        == expected
    )


@pytest.mark.parametrize("status", ["failed", "cancelled", "queued", "in_progress"])
def test_responses_non_completed_status_is_rejected(status: str) -> None:
    with pytest.raises(ConversionError, match="non-completed status"):
        responses_stop_reason(
            status=status,
            incomplete_reason=None,
            has_tool_call=False,
            has_refusal=False,
        )


def test_responses_unknown_incomplete_reason_is_rejected() -> None:
    with pytest.raises(ConversionError, match="incomplete reason"):
        responses_stop_reason(
            status="incomplete",
            incomplete_reason="future_reason",
            has_tool_call=False,
            has_refusal=False,
        )


def test_responses_settings_are_loaded_and_validated() -> None:
    settings = Settings.from_env(
        {
            "OPENAI_API": "responses",
            "UPSTREAM_RESPONSES_PATH": "/custom/responses",
            "MODEL_DISCOVERY": '{"claude-gpt-5-6":"GPT-5.6"}',
            "MODEL_MAP": '{"claude-gpt-5-6":"gpt-5.6"}',
            "STREAM_PING_INTERVAL": "9.5",
        },
    )

    assert settings.openai_api == "responses"
    assert settings.upstream_responses_path == "/custom/responses"
    assert settings.model_discovery == {"claude-gpt-5-6": "GPT-5.6"}
    assert settings.stream_ping_interval == 9.5

    with pytest.raises(ConfigError, match="must begin with 'claude' or 'anthropic'"):
        Settings(model_discovery={"gpt-5.6": "GPT-5.6"})
    with pytest.raises(ConfigError, match="must begin with 'claude' or 'anthropic'"):
        Settings(model_discovery={"gateway-claude-model": "Hidden Claude model"})
    with pytest.raises(ConfigError, match="must match MODEL_MAP or MODEL_OVERRIDE"):
        Settings(model_discovery={"claude-unmapped": "Unmapped"})

    passthrough = Settings(
        model_discovery={"claude-upstream-id": "Upstream model"},
        model_map={"claude-upstream-id": "claude-upstream-id"},
    )
    assert passthrough.map_model("claude-upstream-id") == "claude-upstream-id"
