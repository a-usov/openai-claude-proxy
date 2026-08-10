"""Buffered Anthropic Messages and OpenAI Chat Completions translation."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any

from .exceptions import ConversionError
from .reasoning_state import (
    decode_responses_output_state,
    decode_responses_reasoning,
    encode_responses_output,
    encode_responses_reasoning,
)
from .request_validation import map_tool_callers, prepare_anthropic_request

if TYPE_CHECKING:
    from .config import Settings


_PROMPT_CACHE_BREAKPOINT = {"mode": "explicit"}
_MAX_SAFETY_IDENTIFIER_LENGTH = 64


def _safety_identifier(user_id: str) -> str:
    if len(user_id) <= _MAX_SAFETY_IDENTIFIER_LENGTH:
        return user_id
    return hashlib.sha256(user_id.encode()).hexdigest()


def _with_cache_breakpoint(
    target: dict[str, Any],
    source: dict[str, Any],
) -> dict[str, Any]:
    if source.get("cache_control") is not None:
        target["prompt_cache_breakpoint"] = _PROMPT_CACHE_BREAKPOINT.copy()
    return target


def _has_cache_breakpoint(value: object) -> bool:
    if isinstance(value, dict):
        return value.get("cache_control") is not None or any(
            _has_cache_breakpoint(item) for item in value.values()
        )
    if isinstance(value, list):
        return any(_has_cache_breakpoint(item) for item in value)
    return False


def _openai_content(blocks: list[dict[str, Any]]) -> str | list[dict[str, Any]] | None:
    content: list[dict[str, Any]] = []
    for block in blocks:
        kind = block.get("type")
        if kind == "text":
            content.append(
                _with_cache_breakpoint(
                    {"type": "text", "text": block["text"]},
                    block,
                ),
            )
        elif kind == "image":
            source = block["source"]
            source_type = source["type"]
            if source_type == "base64":
                url = f"data:{source['media_type']};base64,{source['data']}"
            else:
                url = source["url"]
            content.append(
                _with_cache_breakpoint(
                    {"type": "image_url", "image_url": {"url": url}},
                    block,
                ),
            )
        elif kind in {"tool_use", "tool_result"}:
            continue
        else:
            raise ConversionError(f"Unsupported Anthropic content block: {kind!r}")
    if not content:
        return None
    if all(item["type"] == "text" and "prompt_cache_breakpoint" not in item for item in content):
        return "".join(str(item["text"]) for item in content)
    return content


def _chat_tool_result_content(block: dict[str, Any]) -> str | list[dict[str, Any]]:
    value = block.get("content", "")
    prefix = "[Tool error]\n" if block.get("is_error") else ""
    if isinstance(value, str):
        if block.get("cache_control") is None:
            return prefix + value
        return [
            _with_cache_breakpoint(
                {"type": "text", "text": prefix + value},
                block,
            ),
        ]
    parts = [_with_cache_breakpoint({"type": "text", "text": item["text"]}, item) for item in value]
    if prefix:
        parts.insert(0, {"type": "text", "text": prefix.rstrip("\n")})
    if not parts:
        parts.append({"type": "text", "text": ""})
    if block.get("cache_control") is not None:
        parts[-1]["prompt_cache_breakpoint"] = _PROMPT_CACHE_BREAKPOINT.copy()
    return parts


def _assistant_message(blocks: list[dict[str, Any]]) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant"}
    content = _openai_content(blocks)
    message["content"] = content
    tool_calls = [
        {
            "id": block.get("id", ""),
            "type": "function",
            "function": {
                "name": block.get("name", ""),
                "arguments": json.dumps(block.get("input", {}), separators=(",", ":")),
            },
        }
        for block in blocks
        if block.get("type") == "tool_use"
    ]
    if tool_calls:
        message["tool_calls"] = tool_calls
    return message


def _user_messages(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    ordinary: list[dict[str, Any]] = []

    def flush_ordinary() -> None:
        if ordinary:
            messages.append({"role": "user", "content": _openai_content(ordinary)})
            ordinary.clear()

    for block in blocks:
        if block.get("type") != "tool_result":
            ordinary.append(block)
            continue
        flush_ordinary()
        result: dict[str, Any] = {
            "role": "tool",
            "tool_call_id": block["tool_use_id"],
            "content": _chat_tool_result_content(block),
        }
        messages.append(result)
    flush_ordinary()
    return messages


def _output_config(payload: dict[str, Any]) -> dict[str, Any]:
    output_config = payload.get("output_config")
    if output_config is None:
        return {}
    if not isinstance(output_config, dict):
        raise ConversionError("output_config must be an object")
    return output_config


def _mapped_effort(payload: dict[str, Any], settings: Settings) -> str | None:
    configured_effort = _output_config(payload).get("effort")
    legacy_effort = payload.get("effort")
    if (
        configured_effort is not None
        and legacy_effort is not None
        and configured_effort != legacy_effort
    ):
        raise ConversionError("effort and output_config.effort must match when both are set")
    effort = configured_effort if configured_effort is not None else legacy_effort
    if effort is None:
        return None
    if not isinstance(effort, str) or not effort:
        raise ConversionError("effort must be a non-empty string")
    if not settings.reasoning_effort_enabled:
        return None
    if effort not in settings.reasoning_effort_map:
        raise ConversionError(f"No OpenAI reasoning effort mapping is configured for {effort!r}")
    return settings.reasoning_effort_map[effort]


def _structured_output_format(payload: dict[str, Any]) -> dict[str, Any] | None:
    output_config = _output_config(payload)
    configured_format = output_config.get("format")
    legacy_format = payload.get("output_format")
    if (
        configured_format is not None
        and legacy_format is not None
        and configured_format != legacy_format
    ):
        raise ConversionError(
            "output_format and output_config.format must match when both are set",
        )
    output_format = configured_format if configured_format is not None else legacy_format
    if output_format is None:
        return None
    if not isinstance(output_format, dict) or output_format.get("type") != "json_schema":
        raise ConversionError("Only output_config.format type 'json_schema' is supported")
    schema = output_format.get("schema")
    if not isinstance(schema, dict):
        raise ConversionError("output_config.format.schema must be an object")
    return {
        "name": str(output_format.get("name") or "anthropic_output"),
        "schema": schema,
        "strict": True,
    }


def _chat_tool(tool: dict[str, Any]) -> dict[str, Any]:
    function: dict[str, Any] = {
        "name": tool["name"],
        "description": tool.get("description", ""),
        "parameters": tool.get("input_schema", {"type": "object", "properties": {}}),
    }
    if isinstance(tool.get("strict"), bool):
        function["strict"] = tool["strict"]
    return {"type": "function", "function": function}


def _responses_tool(tool: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "type": "function",
        "name": tool["name"],
        "description": tool.get("description", ""),
        "parameters": tool.get("input_schema", {"type": "object", "properties": {}}),
    }
    if isinstance(tool.get("strict"), bool):
        result["strict"] = tool["strict"]
    if callers := tool.get("allowed_callers"):
        result["allowed_callers"] = map_tool_callers(callers)
    if "defer_loading" in tool:
        result["defer_loading"] = tool["defer_loading"]
    return result


def _tool_choice(payload: dict[str, Any]) -> tuple[object | None, bool | None]:
    choice = payload.get("tool_choice")
    if not choice:
        return None, None
    choice_type = choice.get("type") if isinstance(choice, dict) else choice
    if choice_type == "any":
        translated: object = "required"
    elif choice_type in {"auto", "none"}:
        translated = choice_type
    elif choice_type == "tool" and isinstance(choice, dict):
        translated = {"type": "function", "name": choice.get("name", "")}
    else:
        raise ConversionError(f"Unsupported Anthropic tool_choice: {choice_type!r}")
    disable_parallel = choice.get("disable_parallel_tool_use") if isinstance(choice, dict) else None
    if disable_parallel is not None and not isinstance(disable_parallel, bool):
        raise ConversionError("tool_choice.disable_parallel_tool_use must be a boolean")
    return translated, None if disable_parallel is None else not disable_parallel


def anthropic_to_openai(payload: dict[str, Any], settings: Settings) -> dict[str, Any]:
    """Translate one Anthropic Messages request into OpenAI Chat Completions."""
    payload = prepare_anthropic_request(payload, "chat_completions")
    requested_model = payload["model"]

    messages: list[dict[str, Any]] = []
    system = payload.get("system")
    if system:
        system_blocks = [{"type": "text", "text": system}] if isinstance(system, str) else system
        messages.append({"role": "system", "content": _openai_content(system_blocks)})

    for source_message in payload["messages"]:
        role = source_message["role"]
        raw_content = source_message["content"]
        blocks = (
            [{"type": "text", "text": raw_content}] if isinstance(raw_content, str) else raw_content
        )
        if role == "assistant":
            messages.append(_assistant_message(blocks))
        elif role == "user":
            messages.extend(_user_messages(blocks))
        else:
            messages.append({"role": "developer", "content": _openai_content(blocks)})

    result: dict[str, Any] = {
        "model": settings.map_model(requested_model),
        "messages": messages,
        settings.max_tokens_field: max(payload["max_tokens"], settings.min_output_tokens),
        "stream": payload.get("stream", False),
    }
    for source, target in (
        ("temperature", "temperature"),
        ("top_p", "top_p"),
        ("stop_sequences", "stop"),
    ):
        if source in payload:
            result[target] = payload[source]

    if mapped_effort := _mapped_effort(payload, settings):
        result["reasoning_effort"] = mapped_effort
    if output_format := _structured_output_format(payload):
        result["response_format"] = {
            "type": "json_schema",
            "json_schema": output_format,
        }
    if user_id := (payload.get("metadata") or {}).get("user_id"):
        result["safety_identifier"] = _safety_identifier(user_id)

    if tools := payload.get("tools"):
        result["tools"] = [_chat_tool(tool) for tool in tools]
    translated_choice, parallel_tool_calls = _tool_choice(payload)
    if translated_choice is not None:
        if isinstance(translated_choice, dict):
            result["tool_choice"] = {
                "type": "function",
                "function": {"name": translated_choice["name"]},
            }
        else:
            result["tool_choice"] = translated_choice
    if parallel_tool_calls is not None:
        result["parallel_tool_calls"] = parallel_tool_calls
    if result["stream"]:
        result["stream_options"] = {"include_usage": True}
    if _has_cache_breakpoint(payload):
        result["prompt_cache_options"] = {"mode": "explicit", "ttl": "30m"}

    # Explicit administrator configuration wins over client-derived fields.
    result.update(settings.extra_openai_body)
    return result


def _responses_content(
    block: dict[str, Any],
    *,
    assistant: bool,
) -> dict[str, Any] | None:
    kind = block.get("type")
    if kind == "text":
        return _with_cache_breakpoint(
            {
                "type": "input_text",
                "text": block["text"],
            },
            block,
        )
    if kind == "image" and not assistant:
        source = block["source"]
        if source["type"] == "base64":
            image_url = f"data:{source['media_type']};base64,{source['data']}"
        else:
            image_url = source["url"]
        return _with_cache_breakpoint(
            {"type": "input_image", "image_url": image_url},
            block,
        )
    raise ConversionError(f"Unsupported Anthropic content block: {kind!r}")


def _responses_assistant_items(
    blocks: list[dict[str, Any]],
    expected_model: str,
) -> list[dict[str, Any]]:
    complete_output: list[list[dict[str, Any]]] = []
    visible_blocks: list[dict[str, Any]] = []
    for block in blocks:
        decoded_output = None
        if block.get("type") == "redacted_thinking":
            try:
                state = decode_responses_output_state(block.get("data"))
            except ValueError as exc:
                raise ConversionError(str(exc)) from exc
            if state is not None:
                if state.model is None:
                    raise ConversionError(
                        "Legacy proxy Responses output state is not bound to an upstream model",
                    )
                if state.model != expected_model:
                    raise ConversionError(
                        "Proxy Responses output state belongs to a different upstream model",
                    )
                decoded_output = state.output
        if decoded_output is None:
            visible_blocks.append(block)
        else:
            complete_output.append(decoded_output)

    if complete_output:
        if len(complete_output) != 1:
            raise ConversionError(
                "Assistant message contains multiple proxy Responses output states"
            )
        output_items = complete_output[0]
        if _responses_replay_content(visible_blocks) != _responses_replay_content(
            _responses_visible_blocks(output_items),
        ):
            raise ConversionError(
                "Proxy Responses output state does not match the assistant content",
            )
        return output_items

    items: list[dict[str, Any]] = []
    content: list[dict[str, Any]] = []

    def flush_content() -> None:
        if content:
            items.append({"type": "message", "role": "assistant", "content": content.copy()})
            content.clear()

    for block in visible_blocks:
        if block.get("type") == "redacted_thinking":
            try:
                reasoning_item = decode_responses_reasoning(block.get("data"))
            except ValueError as exc:
                raise ConversionError(str(exc)) from exc
            if reasoning_item is not None:
                raise ConversionError(
                    "Legacy proxy Responses reasoning state is not bound to an upstream model",
                )
            raise ConversionError(
                "Only proxy-owned Responses reasoning state can be replayed upstream",
            )
        if block.get("type") == "tool_use":
            flush_content()
            items.append(
                {
                    "type": "function_call",
                    "call_id": block.get("id", ""),
                    "name": block.get("name", ""),
                    "arguments": json.dumps(block.get("input", {}), separators=(",", ":")),
                },
            )
            continue
        translated = _responses_content(block, assistant=True)
        if translated is not None:
            content.append(translated)
    flush_content()
    return items


def _responses_replay_content(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the user-visible semantics protected by a complete output carrier."""
    content: list[dict[str, Any]] = []
    for block in blocks:
        block_type = block.get("type")
        if block_type == "text":
            text = block.get("text", "")
            if content and content[-1]["type"] == "text":
                content[-1]["text"] += text
            else:
                content.append({"type": "text", "text": text})
            continue
        if block_type == "tool_use":
            content.append(
                {
                    "type": "tool_use",
                    "id": block.get("id", ""),
                    "name": block.get("name", ""),
                    "input": block.get("input", {}),
                },
            )
            continue
        if block_type == "redacted_thinking":
            try:
                reasoning_item = decode_responses_reasoning(block.get("data"))
            except ValueError as exc:
                raise ConversionError(str(exc)) from exc
            if reasoning_item is not None:
                # The complete, model-bound output carrier is authoritative. A
                # stream may expose a different metadata projection of the same
                # opaque reasoning item in response.output_item.done.
                continue
            raise ConversionError(
                "Only proxy-owned Responses reasoning state can be replayed upstream",
            )
        raise ConversionError(f"Unsupported Anthropic content block: {block_type!r}")
    return content


def _responses_visible_blocks(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for item in items:
        item_type = item.get("type")
        if item_type == "message":
            for part in item.get("content") or []:
                if not isinstance(part, dict):
                    raise ConversionError("Responses message content parts must be objects")
                part_type = part.get("type")
                if part_type == "output_text" and isinstance(part.get("text"), str):
                    blocks.append({"type": "text", "text": part["text"]})
                elif part_type == "refusal" and isinstance(part.get("refusal"), str):
                    blocks.append({"type": "text", "text": part["refusal"]})
                else:
                    raise ConversionError(
                        f"Unsupported Responses message content part: {part_type!r}",
                    )
        elif item_type == "function_call":
            blocks.append(
                {
                    "type": "tool_use",
                    "id": item.get("call_id") or item.get("id", ""),
                    "name": item.get("name", ""),
                    "input": parse_function_arguments(item.get("arguments")),
                },
            )
        elif item_type == "reasoning":
            blocks.append(
                {
                    "type": "redacted_thinking",
                    "data": encode_responses_reasoning(item),
                },
            )
        else:
            raise ConversionError(f"Unsupported Responses output item: {item_type!r}")
    return blocks


def _responses_user_items(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    content: list[dict[str, Any]] = []

    def flush_content() -> None:
        if content:
            items.append({"type": "message", "role": "user", "content": content.copy()})
            content.clear()

    for block in blocks:
        if block.get("type") == "tool_result":
            flush_content()
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": block["tool_use_id"],
                    "output": _responses_tool_result_output(block),
                },
            )
            continue
        translated = _responses_content(block, assistant=False)
        if translated is not None:
            content.append(translated)
    flush_content()
    return items


def _responses_tool_result_output(block: dict[str, Any]) -> str | list[dict[str, Any]]:
    value = block.get("content", "")
    prefix = "[Tool error]\n" if block.get("is_error") else ""
    if isinstance(value, str):
        if block.get("cache_control") is None:
            return prefix + value
        return [
            _with_cache_breakpoint(
                {"type": "input_text", "text": prefix + value},
                block,
            ),
        ]
    output: list[dict[str, Any]] = []
    if prefix:
        output.append({"type": "input_text", "text": prefix.rstrip("\n")})
    for item in value:
        translated = _responses_content(item, assistant=False)
        if translated is not None:
            output.append(translated)
    if not output:
        output.append({"type": "input_text", "text": ""})
    if block.get("cache_control") is not None:
        output[-1]["prompt_cache_breakpoint"] = _PROMPT_CACHE_BREAKPOINT.copy()
    return output


_RESPONSES_INPUT_TOKEN_FIELDS = frozenset(
    {
        "input",
        "instructions",
        "model",
        "parallel_tool_calls",
        "reasoning",
        "text",
        "tool_choice",
        "tools",
    },
)


def _anthropic_to_responses(
    payload: dict[str, Any],
    settings: Settings,
    *,
    token_count: bool,
) -> dict[str, Any]:
    payload = prepare_anthropic_request(payload, "responses", token_count=token_count)
    requested_model = payload["model"]
    mapped_model = settings.map_model(requested_model)
    if payload.get("stop_sequences"):
        raise ConversionError("stop_sequences cannot be represented by the Responses API")

    input_items: list[dict[str, Any]] = []
    system = payload.get("system")
    system_has_cache_breakpoint = isinstance(system, list) and _has_cache_breakpoint(system)
    if system_has_cache_breakpoint:
        input_items.append(
            {
                "type": "message",
                "role": "developer",
                "content": [
                    translated
                    for block in system
                    if (translated := _responses_content(block, assistant=False)) is not None
                ],
            },
        )
    for source_message in payload["messages"]:
        role = source_message["role"]
        raw_content = source_message["content"]
        blocks = (
            [{"type": "text", "text": raw_content}] if isinstance(raw_content, str) else raw_content
        )
        if role == "assistant":
            input_items.extend(_responses_assistant_items(blocks, mapped_model))
        elif role == "user":
            input_items.extend(_responses_user_items(blocks))
        else:
            input_items.append(
                {
                    "type": "message",
                    "role": "developer",
                    "content": [
                        translated
                        for block in blocks
                        if (translated := _responses_content(block, assistant=False)) is not None
                    ],
                },
            )

    result: dict[str, Any] = {
        "model": mapped_model,
        "input": input_items,
    }
    if not token_count:
        result.update(
            {
                "max_output_tokens": max(payload["max_tokens"], settings.min_output_tokens),
                "stream": payload.get("stream", False),
                "store": False,
            },
        )
    if system and not system_has_cache_breakpoint:
        result["instructions"] = (
            system if isinstance(system, str) else "".join(block["text"] for block in system)
        )
    for field in ("temperature", "top_p"):
        if field in payload:
            result[field] = payload[field]
    if mapped_effort := _mapped_effort(payload, settings):
        result["reasoning"] = {"effort": mapped_effort}
    if output_format := _structured_output_format(payload):
        result["text"] = {"format": {"type": "json_schema", **output_format}}
    if user_id := (payload.get("metadata") or {}).get("user_id"):
        result["safety_identifier"] = _safety_identifier(user_id)
    if tools := payload.get("tools"):
        result["tools"] = [_responses_tool(tool) for tool in tools]
    translated_choice, parallel_tool_calls = _tool_choice(payload)
    if translated_choice is not None:
        result["tool_choice"] = translated_choice
    if parallel_tool_calls is not None:
        result["parallel_tool_calls"] = parallel_tool_calls
    if _has_cache_breakpoint(payload):
        result["prompt_cache_options"] = {"mode": "explicit", "ttl": "30m"}

    result.update(settings.extra_openai_body)
    if (
        not token_count
        and result.get("store") is False
        and "reasoning" in result
        and "include" not in result
    ):
        result["include"] = ["reasoning.encrypted_content"]
    return result


def anthropic_to_responses(payload: dict[str, Any], settings: Settings) -> dict[str, Any]:
    """Translate one Anthropic Messages request into an OpenAI Responses request."""
    return _anthropic_to_responses(payload, settings, token_count=False)


def anthropic_to_responses_token_count(
    payload: dict[str, Any],
    settings: Settings,
) -> dict[str, Any]:
    """Translate an Anthropic token-count input into Responses input-token fields."""
    converted = _anthropic_to_responses(payload, settings, token_count=True)
    return {key: value for key, value in converted.items() if key in _RESPONSES_INPUT_TOKEN_FIELDS}


def validate_anthropic_chat_token_count(
    payload: dict[str, Any],
    _settings: Settings,
) -> dict[str, Any]:
    """Validate a count request using the Chat translation capability policy."""
    prepare_anthropic_request(payload, "chat_completions", token_count=True)
    return {}


def parse_function_arguments(arguments: object) -> dict[str, Any]:
    """Parse provider function arguments without changing their JSON meaning."""
    if isinstance(arguments, dict):
        return arguments
    if arguments is None:
        return {}
    if isinstance(arguments, (str, bytes, bytearray)) and len(arguments) == 0:
        return {}
    if not isinstance(arguments, (str, bytes, bytearray)):
        raise ConversionError("Upstream function arguments must be a JSON object")
    try:
        decoded = json.loads(arguments)
    except (json.JSONDecodeError, TypeError, UnicodeDecodeError) as exc:
        raise ConversionError("Upstream function arguments contain invalid JSON") from exc
    if not isinstance(decoded, dict):
        raise ConversionError("Upstream function arguments must be a JSON object")
    return decoded


def refusal_stop_details(explanation: object = None) -> dict[str, object]:
    """Build Anthropic's structured refusal details from provider text."""
    return {
        "type": "refusal",
        "explanation": explanation if isinstance(explanation, str) and explanation else None,
    }


def openai_to_anthropic(
    payload: dict[str, Any],
    requested_model: str,
    _request_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Translate one buffered OpenAI Chat Completions response into Anthropic."""
    choices = payload.get("choices") or []
    if not choices:
        raise ConversionError("Upstream response did not contain a choice")
    choice = choices[0]
    message = choice.get("message") or {}
    content: list[dict[str, Any]] = []
    text = message.get("content")
    if isinstance(text, str) and text:
        content.append({"type": "text", "text": text})
    elif isinstance(text, list):
        content.extend(
            {"type": "text", "text": part.get("text", "")}
            for part in text
            if isinstance(part, dict) and part.get("type") in {"text", "output_text"}
        )
    refusal = message.get("refusal")
    has_refusal = isinstance(refusal, str) and bool(refusal)
    if has_refusal:
        content.append({"type": "text", "text": refusal})
    tool_calls = list(message.get("tool_calls") or [])
    if legacy_call := message.get("function_call"):
        tool_calls.append(
            {
                "id": payload.get("id", "call_proxy"),
                "function": legacy_call,
            },
        )
    for call in tool_calls:
        function = call.get("function") or {}
        content.append(
            {
                "type": "tool_use",
                "id": call.get("id", ""),
                "name": function.get("name", ""),
                "input": parse_function_arguments(function.get("arguments")),
            },
        )
    usage = payload.get("usage") or {}
    finish_reason = choice.get("finish_reason")
    if not isinstance(finish_reason, str):
        raise ConversionError("Chat Completions response did not include a finish reason")
    stop_reason = finish_reason_to_anthropic(finish_reason)
    if has_refusal and not tool_calls:
        stop_reason = "refusal"
    result = {
        "id": payload.get("id", "msg_proxy"),
        "type": "message",
        "role": "assistant",
        "model": requested_model,
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": anthropic_usage(usage),
    }
    if stop_reason == "refusal":
        result["stop_details"] = refusal_stop_details(refusal)
    return result


def responses_to_anthropic(
    payload: dict[str, Any],
    requested_model: str,
    request_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Translate one buffered OpenAI Responses result into an Anthropic message."""
    if payload.get("type") == "error":
        message = payload.get("message")
        raise ConversionError(
            message if isinstance(message, str) else "Responses API returned an error event",
        )
    if payload.get("error"):
        error = payload["error"]
        message = error.get("message", "Responses API failed") if isinstance(error, dict) else error
        raise ConversionError(str(message))

    raw_output = payload.get("output") or []
    if not isinstance(raw_output, list) or not all(isinstance(item, dict) for item in raw_output):
        raise ConversionError("Responses output must be an array of objects")
    output: list[dict[str, Any]] = raw_output
    visible_content = _responses_visible_blocks(output)
    content = [
        {
            "type": "redacted_thinking",
            "data": encode_responses_output(
                output,
                model=(request_payload or {}).get("model"),
            ),
        },
        *visible_content,
    ]
    has_tool_call = any(item.get("type") == "function_call" for item in output)
    has_refusal = any(
        part.get("type") == "refusal"
        for item in output
        if item.get("type") == "message"
        for part in (item.get("content") or [])
        if isinstance(part, dict)
    )
    refusal_text = next(
        (
            part["refusal"]
            for item in output
            if item.get("type") == "message"
            for part in (item.get("content") or [])
            if isinstance(part, dict)
            and part.get("type") == "refusal"
            and isinstance(part.get("refusal"), str)
        ),
        None,
    )

    status = payload.get("status")
    has_visible_text = any(
        block.get("type") == "text" and bool(block.get("text")) for block in visible_content
    )
    if status != "incomplete" and not (has_tool_call or has_visible_text):
        raise ConversionError(
            "Responses API completed without assistant text or a function call",
        )
    incomplete_reason = (payload.get("incomplete_details") or {}).get("reason")
    stop_reason = responses_stop_reason(
        status=status,
        incomplete_reason=incomplete_reason,
        has_tool_call=has_tool_call,
        has_refusal=has_refusal,
    )
    result = {
        "id": payload.get("id", "msg_proxy"),
        "type": "message",
        "role": "assistant",
        "model": requested_model,
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": anthropic_usage(payload.get("usage") or {}),
    }
    if stop_reason == "refusal":
        result["stop_details"] = refusal_stop_details(refusal_text)
    return result


def finish_reason_to_anthropic(reason: str | None) -> str | None:
    """Map an OpenAI finish reason to its closest Anthropic stop reason."""
    mapped = {
        None: None,
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
        "function_call": "tool_use",
        "content_filter": "refusal",
    }.get(reason)
    if mapped is None and reason is not None:
        raise ConversionError(f"Unsupported OpenAI finish reason: {reason!r}")
    return mapped


def responses_stop_reason(
    *,
    status: object,
    incomplete_reason: object,
    has_tool_call: bool,
    has_refusal: bool,
) -> str:
    """Map a terminal Responses status without disguising nonterminal states."""
    if status in {"failed", "cancelled", "queued", "in_progress"}:
        raise ConversionError(f"Responses returned non-completed status: {status!r}")
    if status not in {None, "completed", "incomplete"}:
        raise ConversionError(f"Unsupported Responses status: {status!r}")
    if has_tool_call:
        return "tool_use"
    if status != "incomplete":
        return "refusal" if has_refusal else "end_turn"
    if incomplete_reason == "max_output_tokens":
        return "max_tokens"
    if has_refusal or incomplete_reason == "content_filter":
        return "refusal"
    if incomplete_reason in {
        "context_length_exceeded",
        "context_window_exceeded",
        "model_context_window_exceeded",
    }:
        return "model_context_window_exceeded"
    raise ConversionError(f"Unsupported Responses incomplete reason: {incomplete_reason!r}")


def anthropic_usage(usage: dict[str, Any]) -> dict[str, Any]:
    """Map OpenAI usage counters into the Anthropic usage schema."""
    details = usage.get("prompt_tokens_details") or usage.get("input_tokens_details") or {}
    cached = int(details.get("cached_tokens") or 0)
    cache_write = int(details.get("cache_write_tokens") or 0)
    total_input = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
    result: dict[str, Any] = {
        "input_tokens": max(0, total_input - cached - cache_write),
        "output_tokens": int(usage.get("completion_tokens") or usage.get("output_tokens") or 0),
    }
    if "cached_tokens" in details:
        result["cache_read_input_tokens"] = cached
    if "cache_write_tokens" in details:
        result["cache_creation_input_tokens"] = cache_write
    completion_details = (
        usage.get("completion_tokens_details") or usage.get("output_tokens_details") or {}
    )
    reasoning_tokens = completion_details.get("reasoning_tokens")
    if reasoning_tokens is not None:
        result["output_tokens_details"] = {"thinking_tokens": int(reasoning_tokens)}
    return result


def estimate_anthropic_tokens(payload: dict[str, Any]) -> int:
    """Estimate tokens conservatively when the upstream cannot count Claude tokens."""
    strings = [
        json.dumps(payload[key], ensure_ascii=False, separators=(",", ":"))
        for key in ("system", "messages", "tools")
        if key in payload
    ]
    text = "".join(strings)
    # Roughly four UTF-8 bytes per token, with a small structural overhead.
    return max(1, (len(text.encode("utf-8")) + 3) // 4)
