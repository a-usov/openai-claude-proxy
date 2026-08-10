"""Validate Anthropic Messages requests before OpenAI translation."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Literal

from .exceptions import ConversionError

Backend = Literal["chat_completions", "responses"]

STANDARD_EFFORTS = frozenset({"low", "medium", "high", "xhigh", "max"})
LEGACY_EFFORTS = STANDARD_EFFORTS | {"none"}
_TOP_LEVEL_FIELDS = frozenset(
    {
        "anthropic-user-profile-id",
        "cache_control",
        "container",
        "context_management",
        "effort",
        "inference_geo",
        "max_tokens",
        "messages",
        "metadata",
        "model",
        "output_config",
        "output_format",
        "service_tier",
        "stop_sequences",
        "stream",
        "system",
        "temperature",
        "thinking",
        "tool_choice",
        "tools",
        "top_k",
        "top_p",
    },
)
_UNSUPPORTED_TOP_LEVEL_FIELDS = (
    "anthropic-user-profile-id",
    "container",
    "inference_geo",
    "service_tier",
    "top_k",
)
_TOOL_FIELDS = frozenset(
    {
        "allowed_callers",
        "cache_control",
        "defer_loading",
        "description",
        "eager_input_streaming",
        "input_examples",
        "input_schema",
        "name",
        "strict",
        "type",
    },
)
_CACHEABLE_BLOCK_TYPES = frozenset({"image", "text", "tool_result", "tool_use"})
_IMAGE_MEDIA_TYPES = frozenset({"image/gif", "image/jpeg", "image/png", "image/webp"})
_PROGRAMMATIC_CALLERS = frozenset(
    {
        "code_execution_20250825",
        "code_execution_20260120",
        "code_execution_20260521",
    },
)
MAX_MESSAGES = 100_000
MAX_PROMPT_CACHE_BREAKPOINTS = 4
MIN_THINKING_BUDGET_TOKENS = 1024


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _reject_unknown_fields(value: dict[str, Any], allowed: frozenset[str], path: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        joined = ", ".join(unknown)
        raise ConversionError(f"{path} contains unsupported field(s): {joined}")


def _validate_cache_control(value: object, path: str) -> None:
    if value is None:
        return
    if not isinstance(value, dict):
        raise ConversionError(f"{path} must be an object or null")
    _reject_unknown_fields(value, frozenset({"ttl", "type"}), path)
    if value.get("type") != "ephemeral":
        raise ConversionError(f"{path}.type must be 'ephemeral'")
    ttl = value.get("ttl", "5m")
    if not isinstance(ttl, str) or ttl not in {"5m", "1h"}:
        raise ConversionError(f"{path}.ttl must be '5m' or '1h'")


def _validate_image(block: dict[str, Any], path: str) -> None:
    _reject_unknown_fields(block, frozenset({"cache_control", "source", "type"}), path)
    source = block.get("source")
    if not isinstance(source, dict):
        raise ConversionError(f"{path}.source must be an object")
    source_type = source.get("type")
    if source_type == "base64":
        _reject_unknown_fields(source, frozenset({"data", "media_type", "type"}), f"{path}.source")
        if source.get("media_type") not in _IMAGE_MEDIA_TYPES:
            raise ConversionError(f"{path}.source.media_type is not a supported image type")
        if not isinstance(source.get("data"), str):
            raise ConversionError(f"{path}.source.data must be a base64 string")
    elif source_type == "url":
        _reject_unknown_fields(source, frozenset({"type", "url"}), f"{path}.source")
        if not isinstance(source.get("url"), str) or not source["url"]:
            raise ConversionError(f"{path}.source.url must be a non-empty string")
    else:
        raise ConversionError(f"{path}.source.type must be 'base64' or 'url'")


def _validate_text(block: dict[str, Any], path: str) -> None:
    _reject_unknown_fields(block, frozenset({"cache_control", "text", "type"}), path)
    if not isinstance(block.get("text"), str):
        raise ConversionError(f"{path}.text must be a string")


def _validate_tool_use(block: dict[str, Any], path: str, backend: Backend) -> None:
    _reject_unknown_fields(
        block,
        frozenset({"cache_control", "caller", "id", "input", "name", "type"}),
        path,
    )
    for field in ("id", "name"):
        if not isinstance(block.get(field), str) or not block[field]:
            raise ConversionError(f"{path}.{field} must be a non-empty string")
    if not isinstance(block.get("input"), dict):
        raise ConversionError(f"{path}.input must be an object")
    if block.get("caller") is not None:
        raise ConversionError(f"{path}.caller cannot be represented by the {backend} backend")
    if block.get("cache_control") is not None:
        raise ConversionError(
            f"{path}.cache_control cannot be represented on an OpenAI function call",
        )


def _validate_tool_result_content(value: object, path: str, backend: Backend) -> None:
    if value is None or isinstance(value, str):
        return
    if not isinstance(value, list):
        raise ConversionError(f"{path} must be a string or an array of content blocks")
    for index, item in enumerate(value):
        item_path = f"{path}[{index}]"
        if not isinstance(item, dict):
            raise ConversionError(f"{item_path} must be an object")
        item_type = item.get("type")
        if item_type == "text":
            _validate_text(item, item_path)
        elif item_type == "image" and backend == "responses":
            _validate_image(item, item_path)
        elif item_type == "image":
            raise ConversionError(f"{item_path} cannot be represented by Chat Completions")
        else:
            raise ConversionError(f"Unsupported tool-result content block: {item_type!r}")


def _validate_tool_result(block: dict[str, Any], path: str, backend: Backend) -> None:
    _reject_unknown_fields(
        block,
        frozenset({"cache_control", "content", "is_error", "tool_use_id", "type"}),
        path,
    )
    if not isinstance(block.get("tool_use_id"), str) or not block["tool_use_id"]:
        raise ConversionError(f"{path}.tool_use_id must be a non-empty string")
    if "is_error" in block and not isinstance(block["is_error"], bool):
        raise ConversionError(f"{path}.is_error must be a boolean")
    _validate_tool_result_content(block.get("content"), f"{path}.content", backend)


def _validate_block(block: dict[str, Any], path: str, role: str, backend: Backend) -> None:
    block_type = block.get("type")
    if not isinstance(block_type, str):
        raise ConversionError(f"{path}.type must be a string")
    if block_type == "text":
        _validate_text(block, path)
    elif block_type == "image" and role == "user":
        _validate_image(block, path)
    elif block_type == "tool_use" and role == "assistant":
        _validate_tool_use(block, path, backend)
    elif block_type == "tool_result" and role == "user":
        _validate_tool_result(block, path, backend)
    elif block_type == "redacted_thinking" and role == "assistant" and backend == "responses":
        _reject_unknown_fields(block, frozenset({"data", "type"}), path)
        if not isinstance(block.get("data"), str) or not block["data"]:
            raise ConversionError(f"{path}.data must be a non-empty string")
    elif block_type in {"thinking", "redacted_thinking"}:
        raise ConversionError(
            f"{path} contains Anthropic reasoning state that cannot be translated safely",
        )
    else:
        raise ConversionError(f"Unsupported {role} content block: {block_type!r}")
    _validate_cache_control(block.get("cache_control"), f"{path}.cache_control")


def _validate_message(message: object, index: int, backend: Backend) -> None:
    path = f"messages[{index}]"
    if not isinstance(message, dict):
        raise ConversionError(f"{path} must be an object")
    _reject_unknown_fields(message, frozenset({"content", "role"}), path)
    role = message.get("role")
    if not isinstance(role, str) or role not in {"assistant", "system", "user"}:
        raise ConversionError(f"{path}.role must be 'user', 'assistant', or 'system'")
    content = message.get("content")
    if isinstance(content, str):
        return
    if not isinstance(content, list):
        raise ConversionError(f"{path}.content must be a string or an array of blocks")
    for block_index, block in enumerate(content):
        block_path = f"{path}.content[{block_index}]"
        if not isinstance(block, dict):
            raise ConversionError(f"{block_path} must be an object")
        if role == "system" and block.get("type") != "text":
            raise ConversionError(f"{block_path} must be a text block for a system message")
        _validate_block(block, block_path, role, backend)


def _validate_system(value: object, backend: Backend) -> None:
    if value is None or isinstance(value, str):
        return
    if not isinstance(value, list):
        raise ConversionError("system must be a string or an array of text blocks")
    for index, block in enumerate(value):
        path = f"system[{index}]"
        if not isinstance(block, dict) or block.get("type") != "text":
            raise ConversionError(f"{path} must be a text block")
        _validate_text(block, path)
        _validate_cache_control(block.get("cache_control"), f"{path}.cache_control")
        if backend == "responses" and block.get("cache_control") is not None:
            raise ConversionError("Responses instructions do not support prompt-cache breakpoints")


def _validate_output_config(payload: dict[str, Any]) -> None:
    value = payload.get("output_config")
    if value is not None:
        if not isinstance(value, dict):
            raise ConversionError("output_config must be an object or null")
        _reject_unknown_fields(value, frozenset({"effort", "format"}), "output_config")
        effort = value.get("effort")
        if effort is not None and (not isinstance(effort, str) or effort not in STANDARD_EFFORTS):
            raise ConversionError("output_config.effort is not a supported Anthropic effort")
    legacy_effort = payload.get("effort")
    if legacy_effort is not None and (
        not isinstance(legacy_effort, str) or legacy_effort not in LEGACY_EFFORTS
    ):
        raise ConversionError("legacy effort must be low, medium, high, xhigh, max, or none")


def _validate_thinking(value: object, max_tokens: object) -> None:
    if value is None:
        return
    if not isinstance(value, dict):
        raise ConversionError("thinking must be an object or null")
    thinking_type = value.get("type")
    if thinking_type == "adaptive":
        _reject_unknown_fields(value, frozenset({"display", "type"}), "thinking")
    elif thinking_type == "disabled":
        _reject_unknown_fields(value, frozenset({"type"}), "thinking")
    elif thinking_type == "enabled":
        _reject_unknown_fields(
            value,
            frozenset({"budget_tokens", "display", "type"}),
            "thinking",
        )
        budget_tokens = value.get("budget_tokens")
        if (
            not isinstance(budget_tokens, int)
            or isinstance(budget_tokens, bool)
            or budget_tokens < MIN_THINKING_BUDGET_TOKENS
        ):
            raise ConversionError("thinking.budget_tokens must be an integer of at least 1024")
        if (
            isinstance(max_tokens, int)
            and not isinstance(max_tokens, bool)
            and budget_tokens >= max_tokens
        ):
            raise ConversionError("thinking.budget_tokens must be less than max_tokens")
        raise ConversionError(
            "thinking.type 'enabled' uses a fixed token budget with no safe OpenAI translation; "
            "use adaptive thinking with output_config.effort",
        )
    else:
        raise ConversionError("thinking.type must be 'adaptive', 'disabled', or 'enabled'")
    display = value.get("display")
    if display is not None and display not in {"omitted", "summarized"}:
        raise ConversionError("thinking.display must be 'omitted' or 'summarized'")


def _validate_context_management(value: object) -> None:
    if value is None:
        return
    if not isinstance(value, dict):
        raise ConversionError("context_management must be an object or null")
    _reject_unknown_fields(value, frozenset({"edits"}), "context_management")
    edits = value.get("edits")
    if not isinstance(edits, list):
        raise ConversionError("context_management.edits must be an array")
    for index, edit in enumerate(edits):
        path = f"context_management.edits[{index}]"
        if not isinstance(edit, dict):
            raise ConversionError(f"{path} must be an object")
        _reject_unknown_fields(edit, frozenset({"keep", "type"}), path)
        if edit.get("type") != "clear_thinking_20251015" or edit.get("keep") != "all":
            raise ConversionError(
                f"{path} changes conversation context and has no safe OpenAI translation",
            )


def _validate_tools(value: object, backend: Backend) -> None:
    if value is None:
        return
    if not isinstance(value, list):
        raise ConversionError("tools must be an array")
    for index, tool in enumerate(value):
        path = f"tools[{index}]"
        if not isinstance(tool, dict):
            raise ConversionError(f"{path} must be an object")
        _reject_unknown_fields(tool, _TOOL_FIELDS, path)
        tool_type = tool.get("type")
        if tool_type is not None and tool_type != "custom":
            raise ConversionError(f"{path}.type is not a translatable client tool")
        if not isinstance(tool.get("name"), str) or not tool["name"]:
            raise ConversionError(f"{path}.name must be a non-empty string")
        if not isinstance(tool.get("input_schema"), dict):
            raise ConversionError(f"{path}.input_schema must be an object")
        if "description" in tool and not isinstance(tool["description"], str):
            raise ConversionError(f"{path}.description must be a string")
        if "strict" in tool and not isinstance(tool["strict"], bool):
            raise ConversionError(f"{path}.strict must be a boolean")
        _validate_cache_control(tool.get("cache_control"), f"{path}.cache_control")
        if tool.get("cache_control") is not None:
            raise ConversionError(f"{path}.cache_control has no OpenAI function-tool equivalent")
        _validate_tool_capabilities(tool, path, backend)


def _validate_tool_capabilities(tool: dict[str, Any], path: str, backend: Backend) -> None:
    callers = tool.get("allowed_callers")
    if callers is not None:
        if not isinstance(callers, list) or not callers:
            raise ConversionError(f"{path}.allowed_callers must be a non-empty array")
        if not all(
            isinstance(caller, str) and (caller == "direct" or caller in _PROGRAMMATIC_CALLERS)
            for caller in callers
        ):
            raise ConversionError(f"{path}.allowed_callers contains an unsupported caller")
        if backend == "chat_completions" and any(caller != "direct" for caller in callers):
            raise ConversionError(
                f"{path}.allowed_callers cannot be represented by Chat Completions"
            )
    defer_loading = tool.get("defer_loading")
    if defer_loading is not None and not isinstance(defer_loading, bool):
        raise ConversionError(f"{path}.defer_loading must be a boolean")
    if backend == "chat_completions" and defer_loading is True:
        raise ConversionError(f"{path}.defer_loading cannot be represented by Chat Completions")
    if tool.get("eager_input_streaming") is not None:
        raise ConversionError(f"{path}.eager_input_streaming has no OpenAI equivalent")
    examples = tool.get("input_examples")
    if examples is not None and (not isinstance(examples, list) or examples):
        raise ConversionError(f"{path}.input_examples has no OpenAI equivalent")


def _validate_tool_choice(value: object) -> None:
    if value is None:
        return
    if not isinstance(value, dict):
        raise ConversionError("tool_choice must be an object")
    _reject_unknown_fields(
        value,
        frozenset({"disable_parallel_tool_use", "name", "type"}),
        "tool_choice",
    )
    choice_type = value.get("type")
    if not isinstance(choice_type, str) or choice_type not in {"any", "auto", "none", "tool"}:
        raise ConversionError("tool_choice.type must be 'auto', 'any', 'tool', or 'none'")
    if choice_type == "tool":
        if not isinstance(value.get("name"), str) or not value["name"]:
            raise ConversionError("tool_choice.name must be a non-empty string")
    elif "name" in value:
        raise ConversionError("tool_choice.name is only valid when type is 'tool'")
    if "disable_parallel_tool_use" in value and not isinstance(
        value["disable_parallel_tool_use"],
        bool,
    ):
        raise ConversionError("tool_choice.disable_parallel_tool_use must be a boolean")
    if choice_type == "none" and "disable_parallel_tool_use" in value:
        raise ConversionError("disable_parallel_tool_use is not valid for tool_choice type 'none'")


def _mark_top_level_cache_control(payload: dict[str, Any]) -> None:
    cache_control = payload.get("cache_control")
    if cache_control is None:
        return
    messages = payload["messages"]
    for message in reversed(messages):
        content = message["content"]
        if isinstance(content, str):
            message["content"] = [
                {"type": "text", "text": content, "cache_control": deepcopy(cache_control)},
            ]
            return
        for block in reversed(content):
            block_type = block.get("type")
            if isinstance(block_type, str) and block_type in _CACHEABLE_BLOCK_TYPES:
                block["cache_control"] = deepcopy(cache_control)
                return
    system = payload.get("system")
    if isinstance(system, str):
        payload["system"] = [
            {"type": "text", "text": system, "cache_control": deepcopy(cache_control)},
        ]
        return
    if isinstance(system, list) and system:
        system[-1]["cache_control"] = deepcopy(cache_control)
        return
    tools = payload.get("tools")
    if isinstance(tools, list) and tools:
        tools[-1]["cache_control"] = deepcopy(cache_control)
        return
    raise ConversionError("cache_control requires at least one cacheable request block")


def _count_cache_controls(value: object) -> int:
    if isinstance(value, dict):
        own = int(value.get("cache_control") is not None)
        return own + sum(_count_cache_controls(item) for item in value.values())
    if isinstance(value, list):
        return sum(_count_cache_controls(item) for item in value)
    return 0


def prepare_anthropic_request(
    payload: dict[str, Any],
    backend: Backend,
    *,
    token_count: bool = False,
) -> dict[str, Any]:
    """Return a validated copy with top-level cache control made explicit."""
    _reject_unknown_fields(payload, _TOP_LEVEL_FIELDS, "request")
    requested_model = payload.get("model")
    if not isinstance(requested_model, str) or not requested_model:
        raise ConversionError("model must be a non-empty string")
    max_tokens = payload.get("max_tokens")
    if (not token_count or max_tokens is not None) and (
        not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens <= 0
    ):
        raise ConversionError("max_tokens must be a positive integer for OpenAI translation")
    if "stream" in payload and not isinstance(payload["stream"], bool):
        raise ConversionError("stream must be a boolean")
    messages = payload.get("messages")
    if not isinstance(messages, list):
        raise ConversionError("messages must be an array")
    if len(messages) > MAX_MESSAGES:
        raise ConversionError("messages cannot contain more than 100000 entries")
    for index, message in enumerate(messages):
        _validate_message(message, index, backend)
    if not token_count and messages and messages[-1].get("role") == "assistant":
        raise ConversionError(
            "Final assistant-prefill messages cannot be represented faithfully by OpenAI",
        )
    _validate_system(payload.get("system"), backend)
    _validate_output_config(payload)
    _validate_thinking(payload.get("thinking"), max_tokens)
    _validate_context_management(payload.get("context_management"))
    _validate_tools(payload.get("tools"), backend)
    _validate_tool_choice(payload.get("tool_choice"))
    for field in _UNSUPPORTED_TOP_LEVEL_FIELDS:
        if payload.get(field) is not None:
            raise ConversionError(f"{field} has no safe OpenAI translation")
    metadata = payload.get("metadata")
    if metadata is not None:
        if not isinstance(metadata, dict):
            raise ConversionError("metadata must be an object or null")
        _reject_unknown_fields(metadata, frozenset({"user_id"}), "metadata")
        if metadata.get("user_id") is not None and not isinstance(metadata["user_id"], str):
            raise ConversionError("metadata.user_id must be a string or null")
    for field in ("temperature", "top_p"):
        if field in payload and (not _is_number(payload[field]) or not 0 <= payload[field] <= 1):
            raise ConversionError(f"{field} must be a number between 0 and 1")
    stop_sequences = payload.get("stop_sequences")
    if stop_sequences is not None and (
        not isinstance(stop_sequences, list)
        or not all(isinstance(item, str) for item in stop_sequences)
    ):
        raise ConversionError("stop_sequences must be an array of strings")
    _validate_cache_control(payload.get("cache_control"), "cache_control")

    prepared = deepcopy(payload)
    _mark_top_level_cache_control(prepared)
    prepared.pop("cache_control", None)
    if _count_cache_controls(prepared) > MAX_PROMPT_CACHE_BREAKPOINTS:
        raise ConversionError("OpenAI supports at most four explicit prompt-cache breakpoints")
    # Validate the location selected by top-level cache control as well.
    _validate_system(prepared.get("system"), backend)
    _validate_tools(prepared.get("tools"), backend)
    for index, message in enumerate(prepared["messages"]):
        _validate_message(message, index, backend)
    return prepared


def map_tool_callers(callers: list[str]) -> list[str]:
    """Map Anthropic caller identifiers to the Responses caller union."""
    mapped = ["direct" if caller == "direct" else "programmatic" for caller in callers]
    return list(dict.fromkeys(mapped))
