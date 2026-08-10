"""Server-sent event parsing and streaming protocol translation."""

from __future__ import annotations

import asyncio
import json
from contextlib import suppress
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import httpx

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

from .conversion import (
    ConversionError,
    anthropic_usage,
    finish_reason_to_anthropic,
    parse_function_arguments,
    refusal_stop_details,
    responses_stop_reason,
)
from .errors import anthropic_error
from .reasoning_state import encode_responses_output, encode_responses_reasoning

_IGNORED_RESPONSES_EVENT_TYPES = frozenset(
    {
        "response.in_progress",
        "response.output_text.annotation.added",
        "response.queued",
        "response.reasoning_summary_part.added",
        "response.reasoning_summary_part.done",
        "response.reasoning_summary_text.delta",
        "response.reasoning_summary_text.done",
        "response.reasoning_text.delta",
        "response.reasoning_text.done",
    },
)
_RESPONSES_EVENT_TYPES = _IGNORED_RESPONSES_EVENT_TYPES | {
    "error",
    "response.completed",
    "response.content_part.added",
    "response.content_part.done",
    "response.created",
    "response.failed",
    "response.function_call_arguments.delta",
    "response.function_call_arguments.done",
    "response.incomplete",
    "response.output_item.added",
    "response.output_item.done",
    "response.output_text.delta",
    "response.output_text.done",
    "response.refusal.delta",
    "response.refusal.done",
}


def sse(event: str, data: dict[str, Any]) -> bytes:
    """Encode one named Anthropic server-sent event."""
    encoded = json.dumps(data, separators=(",", ":"), ensure_ascii=False)
    return f"event: {event}\ndata: {encoded}\n\n".encode()


def _error_event(
    detail: object,
    fallback_message: str,
    *,
    request_id: object = None,
) -> bytes:
    if isinstance(detail, dict):
        message = detail.get("message")
        provider_type = detail.get("type")
        code = detail.get("code")
        request_id = detail.get("request_id") or request_id
    else:
        message = detail
        provider_type = None
        code = None
    return sse(
        "error",
        anthropic_error(
            message if message is not None else fallback_message,
            provider_type=provider_type,
            code=code,
            request_id=request_id,
        ),
    )


def _responses_error_event(event: dict[str, Any]) -> bytes:
    if event.get("type") == "error":
        return _error_event(
            event,
            "Responses API stream failed",
            request_id=event.get("request_id"),
        )
    response = event.get("response") or {}
    return _error_event(
        response.get("error") or {},
        "Responses API stream failed",
        request_id=response.get("request_id"),
    )


def _decode_sse_object(raw_data: str, protocol: str) -> dict[str, Any]:
    try:
        decoded = json.loads(raw_data)
    except json.JSONDecodeError as exc:
        raise ConversionError(f"Malformed JSON in {protocol} stream event") from exc
    if not isinstance(decoded, dict):
        raise ConversionError(f"{protocol} stream events must be JSON objects")
    return decoded


def _chat_stream_choice(chunk: dict[str, Any]) -> dict[str, Any] | None:
    if "choices" not in chunk:
        raise ConversionError("Chat Completions stream event did not contain choices")
    choices = chunk.get("choices")
    if not isinstance(choices, list):
        raise ConversionError("Chat Completions stream choices must be an array")
    if not choices:
        return None
    choice = choices[0]
    if not isinstance(choice, dict):
        raise ConversionError("Chat Completions stream choices must be objects")
    return choice


def _chat_stream_delta(choice: dict[str, Any]) -> dict[str, Any]:
    delta = choice.get("delta") or {}
    if not isinstance(delta, dict):
        raise ConversionError("Chat Completions stream delta must be an object")
    return delta


def _chat_tool_deltas(delta: dict[str, Any], message_id: str) -> list[dict[str, Any]]:
    raw_tool_calls = delta.get("tool_calls") or []
    if not isinstance(raw_tool_calls, list) or not all(
        isinstance(tool_call, dict) for tool_call in raw_tool_calls
    ):
        raise ConversionError("Chat Completions tool-call deltas must be objects")
    tool_calls: list[dict[str, Any]] = raw_tool_calls
    if legacy_call := delta.get("function_call"):
        if not isinstance(legacy_call, dict):
            raise ConversionError("Chat Completions function-call delta must be an object")
        tool_calls.append(
            {
                "index": 0,
                "id": f"call_{message_id}",
                "function": legacy_call,
            },
        )
    return tool_calls


def _validate_responses_stream_event(event: dict[str, Any]) -> str:
    event_type = event.get("type")
    if not isinstance(event_type, str):
        raise ConversionError("Responses stream event type must be a string")
    if event_type not in _RESPONSES_EVENT_TYPES:
        raise ConversionError("Unsupported Responses stream event type")
    if event_type in {
        "response.created",
        "response.completed",
        "response.incomplete",
    } and not isinstance(
        event.get("response"),
        dict,
    ):
        raise ConversionError("Responses lifecycle event must contain a response object")
    if event_type in {"response.output_item.added", "response.output_item.done"}:
        item = event.get("item")
        if not isinstance(item, dict):
            raise ConversionError("Responses output item must be an object")
        if item.get("type") not in {"function_call", "message", "reasoning"}:
            raise ConversionError("Unsupported Responses output item in stream")
    if event_type in {"response.output_text.delta", "response.refusal.delta"} and not isinstance(
        event.get("delta"),
        str,
    ):
        raise ConversionError("Responses text delta must be a string")
    if event_type in {"response.content_part.added", "response.content_part.done"}:
        part = event.get("part")
        if not isinstance(part, dict) or part.get("type") not in {"output_text", "refusal"}:
            raise ConversionError("Unsupported Responses content part in stream")
    return event_type


async def anthropic_message_stream(message: dict[str, Any]) -> AsyncIterator[bytes]:
    """Emit a valid stream when a compatible upstream ignored stream=true."""
    usage = message.get("usage") or {}
    yield sse(
        "message_start",
        {
            "type": "message_start",
            "message": {
                **message,
                "content": [],
                "stop_details": None,
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {
                    "input_tokens": int(usage.get("input_tokens", 0)),
                    "output_tokens": 0,
                },
            },
        },
    )
    for index, block in enumerate(message.get("content") or []):
        kind = block.get("type")
        if kind == "text":
            start = {"type": "text", "text": ""}
            delta = {"type": "text_delta", "text": block.get("text", "")}
        elif kind == "tool_use":
            start = {
                "type": "tool_use",
                "id": block.get("id", ""),
                "name": block.get("name", ""),
                "input": {},
            }
            delta = {
                "type": "input_json_delta",
                "partial_json": json.dumps(
                    block.get("input", {}),
                    separators=(",", ":"),
                    ensure_ascii=False,
                ),
            }
        elif kind == "redacted_thinking":
            start = {
                "type": "redacted_thinking",
                "data": block.get("data", ""),
            }
            delta = None
        else:
            continue
        yield sse(
            "content_block_start",
            {"type": "content_block_start", "index": index, "content_block": start},
        )
        if delta is not None:
            yield sse(
                "content_block_delta",
                {"type": "content_block_delta", "index": index, "delta": delta},
            )
        yield sse(
            "content_block_stop",
            {"type": "content_block_stop", "index": index},
        )
    delta = {
        "stop_reason": message.get("stop_reason"),
        "stop_sequence": message.get("stop_sequence"),
    }
    if stop_details := message.get("stop_details"):
        delta["stop_details"] = stop_details
    yield sse(
        "message_delta",
        {
            "type": "message_delta",
            "delta": delta,
            "usage": usage,
        },
    )
    yield sse("message_stop", {"type": "message_stop"})


async def _lines_with_ping(
    response: httpx.Response,
    ping_interval: float,
) -> AsyncIterator[str | None]:
    if ping_interval == 0:
        async for line in response.aiter_lines():
            yield line
        return

    async def next_line() -> str:
        return await anext(lines)

    lines = response.aiter_lines()
    line_task = asyncio.create_task(next_line())
    try:
        while True:
            done, _pending = await asyncio.wait({line_task}, timeout=ping_interval)
            if not done:
                yield None
                continue
            try:
                line = line_task.result()
            except StopAsyncIteration:
                break
            yield line
            line_task = asyncio.create_task(next_line())
    finally:
        if not line_task.done():
            line_task.cancel()
            with suppress(asyncio.CancelledError):
                await line_task


async def iter_sse_data(
    response: httpx.Response,
    ping_interval: float = 0,
) -> AsyncIterator[str | None]:
    """Yield complete data payloads from an upstream SSE response."""
    data_lines: list[str] = []
    async for line in _lines_with_ping(response, ping_interval):
        if line is None:
            yield None
            continue
        if line == "":
            if data_lines:
                yield "\n".join(data_lines)
                data_lines.clear()
            continue
        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
        elif line.startswith(":"):
            yield None
    if data_lines:
        yield "\n".join(data_lines)


@dataclass(slots=True)
class _ChatToolStreamState:
    """Track Chat tool blocks and stream them when nonparallel order is guaranteed."""

    stream_live: bool
    tools: dict[int, dict[str, Any]] = field(default_factory=dict)
    next_content_index: int = 0
    text_index: int | None = None
    live_openai_index: int | None = None
    live_anthropic_index: int | None = None
    live_argument_count: int = 0

    def record_delta(self, tool_call: dict[str, Any], message_id: str) -> int:
        """Accumulate one OpenAI tool delta and return its provider index."""
        openai_index = int(tool_call.get("index", 0))
        function = tool_call.get("function") or {}
        buffered = self.tools.setdefault(
            openai_index,
            {"id": "", "name": "", "arguments": []},
        )
        if tool_call.get("id"):
            buffered["id"] = tool_call["id"]
        if function.get("name"):
            buffered["name"] += function["name"]
        if function.get("arguments"):
            buffered["arguments"].append(function["arguments"])
        if not buffered["id"] and tool_call.get("function_call"):
            buffered["id"] = f"call_{message_id}"
        return openai_index

    def ensure_text_allowed(self) -> None:
        """Reject provider text that would reopen a live tool block."""
        if self.live_anthropic_index is not None:
            raise ConversionError("Upstream emitted text after starting a function call")

    async def emit_live(self, openai_index: int) -> AsyncIterator[bytes]:
        """Emit newly available pieces of a guaranteed-single tool call."""
        if not self.stream_live:
            return
        if self.live_openai_index is not None and self.live_openai_index != openai_index:
            raise ConversionError("Upstream emitted parallel tools when they were disabled")
        tool = self.tools[openai_index]
        if not tool["name"]:
            return
        self.live_openai_index = openai_index
        if self.text_index is not None:
            yield sse(
                "content_block_stop",
                {"type": "content_block_stop", "index": self.text_index},
            )
            self.text_index = None
        if self.live_anthropic_index is None:
            self.live_anthropic_index = self.next_content_index
            self.next_content_index += 1
            yield sse(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": self.live_anthropic_index,
                    "content_block": {
                        "type": "tool_use",
                        "id": tool["id"] or f"tool_{openai_index}",
                        "name": tool["name"],
                        "input": {},
                    },
                },
            )
        arguments = tool["arguments"]
        for part in arguments[self.live_argument_count :]:
            yield sse(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": self.live_anthropic_index,
                    "delta": {"type": "input_json_delta", "partial_json": part},
                },
            )
        self.live_argument_count = len(arguments)

    async def emit_terminal(self) -> AsyncIterator[bytes]:
        """Validate and finish all buffered or live Chat tool calls."""
        for openai_index in sorted(self.tools):
            tool = self.tools[openai_index]
            arguments = "".join(tool["arguments"])
            parse_function_arguments(arguments)
            if self.stream_live:
                async for encoded in self.emit_live(openai_index):
                    yield encoded
                if self.live_anthropic_index is None:
                    raise ConversionError("Upstream function call did not include a name")
                yield sse(
                    "content_block_stop",
                    {"type": "content_block_stop", "index": self.live_anthropic_index},
                )
                continue
            anthropic_index = self.next_content_index
            self.next_content_index += 1
            yield sse(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": anthropic_index,
                    "content_block": {
                        "type": "tool_use",
                        "id": tool["id"] or f"tool_{openai_index}",
                        "name": tool["name"],
                        "input": {},
                    },
                },
            )
            if arguments:
                yield sse(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": anthropic_index,
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": arguments,
                        },
                    },
                )
            yield sse(
                "content_block_stop",
                {"type": "content_block_stop", "index": anthropic_index},
            )


def _require_chat_finish_reason(reason: str | None) -> str:
    if reason is None:
        raise ConversionError("Chat Completions stream ended without a finish reason")
    return reason


async def openai_stream_to_anthropic(
    response: httpx.Response,
    requested_model: str,
    ping_interval: float = 0,
    request_payload: dict[str, Any] | None = None,
) -> AsyncIterator[bytes]:
    """Translate an OpenAI Chat Completions stream into Anthropic event order."""
    message_id = "msg_proxy"
    input_tokens = 0
    output_tokens = 0
    output_tokens_details: dict[str, int] | None = None
    converted_usage: dict[str, Any] = {}
    finish_reason: str | None = None
    has_refusal = False
    refusal_parts: list[str] = []
    started = False
    tool_state = _ChatToolStreamState(
        stream_live=bool(request_payload and request_payload.get("parallel_tool_calls") is False),
    )

    try:
        async for raw_data in iter_sse_data(response, ping_interval):
            if raw_data is None:
                yield sse("ping", {"type": "ping"})
                continue
            if raw_data == "[DONE]":
                break
            chunk = _decode_sse_object(raw_data, "Chat Completions")
            if "error" in chunk:
                yield _error_event(
                    chunk["error"],
                    "Chat Completions stream failed",
                    request_id=chunk.get("request_id"),
                )
                return

            message_id = chunk.get("id") or message_id
            usage = chunk.get("usage")
            if usage:
                converted_usage = anthropic_usage(usage)
                input_tokens = converted_usage["input_tokens"]
                output_tokens = converted_usage["output_tokens"]
                output_tokens_details = converted_usage.get("output_tokens_details")

            if not started:
                started = True
                yield sse(
                    "message_start",
                    {
                        "type": "message_start",
                        "message": {
                            "id": message_id,
                            "type": "message",
                            "role": "assistant",
                            "model": requested_model,
                            "content": [],
                            "stop_reason": None,
                            "stop_sequence": None,
                            "usage": {"input_tokens": input_tokens, "output_tokens": 0},
                        },
                    },
                )

            choice = _chat_stream_choice(chunk)
            if choice is None:
                continue
            if choice.get("finish_reason") is not None:
                finish_reason = choice["finish_reason"]
            delta = _chat_stream_delta(choice)
            refusal = delta.get("refusal")
            has_refusal = has_refusal or isinstance(refusal, str)
            if isinstance(refusal, str):
                refusal_parts.append(refusal)
            text_parts = [delta.get("content"), refusal]
            for text in (part for part in text_parts if isinstance(part, str) and part):
                tool_state.ensure_text_allowed()
                if tool_state.text_index is None:
                    tool_state.text_index = tool_state.next_content_index
                    tool_state.next_content_index += 1
                    yield sse(
                        "content_block_start",
                        {
                            "type": "content_block_start",
                            "index": tool_state.text_index,
                            "content_block": {"type": "text", "text": ""},
                        },
                    )
                yield sse(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": tool_state.text_index,
                        "delta": {"type": "text_delta", "text": text},
                    },
                )

            for tool_call in _chat_tool_deltas(delta, message_id):
                openai_index = tool_state.record_delta(tool_call, message_id)
                async for encoded in tool_state.emit_live(openai_index):
                    yield encoded

        if not started:
            yield sse(
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "id": message_id,
                        "type": "message",
                        "role": "assistant",
                        "model": requested_model,
                        "content": [],
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {"input_tokens": input_tokens, "output_tokens": 0},
                    },
                },
            )
        if tool_state.text_index is not None:
            yield sse(
                "content_block_stop",
                {"type": "content_block_stop", "index": tool_state.text_index},
            )
            tool_state.text_index = None
        async for encoded in tool_state.emit_terminal():
            yield encoded
        final_usage: dict[str, Any] = converted_usage or {"output_tokens": output_tokens}
        if output_tokens_details is not None and "output_tokens_details" not in final_usage:
            final_usage["output_tokens_details"] = output_tokens_details
        stop_reason = finish_reason_to_anthropic(_require_chat_finish_reason(finish_reason))
        if has_refusal and not tool_state.tools:
            stop_reason = "refusal"
        final_delta: dict[str, Any] = {
            "stop_reason": stop_reason,
            "stop_sequence": None,
        }
        if stop_reason == "refusal":
            final_delta["stop_details"] = refusal_stop_details("".join(refusal_parts))
        yield sse(
            "message_delta",
            {
                "type": "message_delta",
                "delta": final_delta,
                "usage": final_usage,
            },
        )
        yield sse("message_stop", {"type": "message_stop"})
    except ConversionError as exc:
        yield _error_event(str(exc), "Invalid Chat Completions stream event")
    except httpx.HTTPError:
        yield _error_event(None, "Chat Completions upstream stream disconnected")
    except (AttributeError, TypeError, ValueError):
        yield _error_event(None, "Malformed Chat Completions stream event")
    finally:
        await response.aclose()


def _responses_message_start(message_id: str, requested_model: str) -> bytes:
    return sse(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": message_id,
                "type": "message",
                "role": "assistant",
                "model": requested_model,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        },
    )


def _record_responses_refusal_delta(
    event_type: object,
    delta: object,
    refusal_parts: list[str],
) -> None:
    if event_type == "response.refusal.delta" and isinstance(delta, str):
        refusal_parts.append(delta)


def _require_open_responses_block(index: int, open_blocks: set[int]) -> None:
    if index not in open_blocks:
        raise ConversionError("Responses delta arrived after content completed")


def _responses_function_argument_suffix(
    arguments: object,
    emitted_parts: list[str],
) -> str:
    emitted = "".join(emitted_parts)
    if arguments is None:
        arguments = emitted
    parse_function_arguments(arguments)
    if not isinstance(arguments, str) or not arguments.startswith(emitted):
        raise ConversionError("Upstream finalized function arguments do not match streamed deltas")
    return arguments[len(emitted) :]


def _responses_argument_event(
    event: dict[str, Any],
    index: int,
    output_index: int,
    argument_parts: dict[int, list[str]],
) -> bytes:
    parts = argument_parts.setdefault(output_index, [])
    if event.get("type") == "response.function_call_arguments.delta":
        delta = event.get("delta")
        if not isinstance(delta, str):
            raise ConversionError("Upstream function argument delta must be text")
        parts.append(delta)
        suffix = delta
    else:
        suffix = _responses_function_argument_suffix(
            event.get("arguments"),
            parts,
        )
        parts.append(suffix)
    if not suffix:
        return b""
    return sse(
        "content_block_delta",
        {
            "type": "content_block_delta",
            "index": index,
            "delta": {"type": "input_json_delta", "partial_json": suffix},
        },
    )


def _responses_text_done_events(
    event: dict[str, Any],
    block_indices: dict[tuple[str, int, int], int],
    open_blocks: set[int],
    text_parts: dict[tuple[int, int], list[str]],
    refusal_parts: list[str],
) -> bytes:
    output_index = int(event.get("output_index", 0))
    content_index = int(event.get("content_index", 0))
    key = (output_index, content_index)
    emitted = "".join(text_parts.get(key, []))
    event_type = event.get("type")
    if event_type == "response.output_text.done":
        complete = event.get("text")
    elif event_type == "response.refusal.done":
        complete = event.get("refusal")
    else:
        part = event.get("part") or {}
        complete = part.get("text") or part.get("refusal") or emitted
    if not isinstance(complete, str) or not complete.startswith(emitted):
        raise ConversionError("Upstream finalized text does not match streamed deltas")

    index = block_indices.get(("text", output_index, content_index))
    if index is None:
        return b""
    if index not in open_blocks:
        if complete != emitted:
            raise ConversionError("Upstream finalized text changed after it was completed")
        return b""
    suffix = complete[len(emitted) :]
    events: list[bytes] = []
    if suffix:
        text_parts.setdefault(key, []).append(suffix)
        if event_type == "response.refusal.done":
            refusal_parts.append(suffix)
        events.append(
            sse(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {"type": "text_delta", "text": suffix},
                },
            ),
        )
    open_blocks.remove(index)
    events.append(
        sse(
            "content_block_stop",
            {"type": "content_block_stop", "index": index},
        ),
    )
    return b"".join(events)


def _responses_message_content(
    item: dict[str, Any],
) -> list[tuple[int, str, str]]:
    content = item.get("content")
    if content is None:
        return []
    if not isinstance(content, list) or not all(isinstance(part, dict) for part in content):
        raise ConversionError("Responses message content must be an array of objects")

    completed: list[tuple[int, str, str]] = []
    for content_index, part in enumerate(content):
        part_type = part.get("type")
        if part_type == "output_text":
            text = part.get("text")
        elif part_type == "refusal":
            text = part.get("refusal")
        else:
            raise ConversionError("Unsupported Responses message content in stream")
        if not isinstance(text, str):
            raise ConversionError("Responses finalized message content must be text")
        completed.append((content_index, part_type, text))
    return completed


def _responses_output_item_done_events(
    event: dict[str, Any],
    next_index: int,
    block_indices: dict[tuple[str, int, int], int],
    open_blocks: set[int],
    argument_parts: dict[int, list[str]],
) -> tuple[bytes, int]:
    item = event.get("item") or {}
    item_type = item.get("type")
    if item_type == "reasoning":
        events = [
            sse(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": next_index,
                    "content_block": {
                        "type": "redacted_thinking",
                        "data": encode_responses_reasoning(item),
                    },
                },
            ),
            sse(
                "content_block_stop",
                {"type": "content_block_stop", "index": next_index},
            ),
        ]
        return b"".join(events), next_index + 1
    if item_type != "function_call":
        if item_type == "message":
            return b"", next_index
        raise ConversionError("Unsupported Responses output item in stream")

    output_index = int(event.get("output_index", 0))
    parts = argument_parts.setdefault(output_index, [])
    suffix = _responses_function_argument_suffix(
        item.get("arguments"),
        parts,
    )
    index = block_indices.get(("tool", output_index, 0))
    if index is None:
        return b"", next_index
    if index not in open_blocks:
        if suffix:
            raise ConversionError("Upstream finalized function arguments changed after completion")
        return b"", next_index
    events: list[bytes] = []
    if suffix:
        parts.append(suffix)
        events.append(
            sse(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {"type": "input_json_delta", "partial_json": suffix},
                },
            ),
        )
    open_blocks.remove(index)
    events.append(
        sse(
            "content_block_stop",
            {"type": "content_block_stop", "index": index},
        ),
    )
    return b"".join(events), next_index


def _responses_output_state_events(
    output: object,
    index: int,
    model: str | None,
) -> tuple[list[bytes], int]:
    if not isinstance(output, list) or not all(
        isinstance(item, dict) and isinstance(item.get("type"), str) for item in output
    ):
        return [], index
    return [
        sse(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": index,
                "content_block": {
                    "type": "redacted_thinking",
                    "data": encode_responses_output(output, model=model),
                },
            },
        ),
        sse(
            "content_block_stop",
            {"type": "content_block_stop", "index": index},
        ),
    ], index + 1


@dataclass(frozen=True, slots=True)
class _ResponsesTerminalContext:
    has_tool_call: bool
    has_refusal: bool
    refusal_text: str
    upstream_model: str | None


def _responses_terminal_events(
    terminal: dict[str, Any],
    index: int,
    context: _ResponsesTerminalContext,
) -> tuple[bytes, int]:
    state_events, next_index = _responses_output_state_events(
        terminal.get("output"),
        index,
        context.upstream_model,
    )
    usage = anthropic_usage(terminal.get("usage") or {})
    incomplete_reason = (terminal.get("incomplete_details") or {}).get("reason")
    stop_reason = responses_stop_reason(
        status=terminal.get("status"),
        has_tool_call=context.has_tool_call,
        has_refusal=context.has_refusal,
        incomplete_reason=incomplete_reason,
    )
    delta: dict[str, Any] = {"stop_reason": stop_reason, "stop_sequence": None}
    if stop_reason == "refusal":
        delta["stop_details"] = refusal_stop_details(context.refusal_text)
    events = [
        *state_events,
        sse(
            "message_delta",
            {
                "type": "message_delta",
                "delta": delta,
                "usage": usage,
            },
        ),
        sse("message_stop", {"type": "message_stop"}),
    ]
    return b"".join(events), next_index


@dataclass(slots=True)
class _ResponsesStreamState:
    requested_model: str
    upstream_model: str | None
    message_id: str = "msg_proxy"
    started: bool = False
    next_content_index: int = 0
    block_indices: dict[tuple[str, int, int], int] = field(default_factory=dict)
    open_blocks: set[int] = field(default_factory=set)
    tool_items: dict[int, dict[str, Any]] = field(default_factory=dict)
    tool_argument_parts: dict[int, list[str]] = field(default_factory=dict)
    text_parts: dict[tuple[int, int], list[str]] = field(default_factory=dict)
    has_tool_call: bool = False
    has_refusal: bool = False
    refusal_parts: list[str] = field(default_factory=list)

    def ensure_started(self) -> bytes:
        if self.started:
            return b""
        self.started = True
        return _responses_message_start(self.message_id, self.requested_model)

    def start_text(self, output_index: int, content_index: int) -> bytes:
        key = ("text", output_index, content_index)
        if key in self.block_indices:
            return b""
        index = self.next_content_index
        self.next_content_index += 1
        self.block_indices[key] = index
        self.open_blocks.add(index)
        return sse(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": index,
                "content_block": {"type": "text", "text": ""},
            },
        )

    def start_tool(self, output_index: int) -> bytes:
        key = ("tool", output_index, 0)
        if key in self.block_indices:
            return b""
        index = self.next_content_index
        self.next_content_index += 1
        self.block_indices[key] = index
        tool = self.tool_items.get(output_index, {})
        self.open_blocks.add(index)
        return sse(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": index,
                "content_block": {
                    "type": "tool_use",
                    "id": tool.get("call_id") or tool.get("id", ""),
                    "name": tool.get("name", ""),
                    "input": {},
                },
            },
        )

    def reconcile_message_item(self, item: dict[str, Any], output_index: int) -> bytes:
        events: list[bytes] = []
        for content_index, part_type, complete in _responses_message_content(item):
            is_refusal = part_type == "refusal"
            self.has_refusal = self.has_refusal or is_refusal
            events.append(self.start_text(output_index, content_index))
            done_event: dict[str, Any] = {
                "type": "response.refusal.done" if is_refusal else "response.output_text.done",
                "output_index": output_index,
                "content_index": content_index,
                "refusal" if is_refusal else "text": complete,
            }
            events.append(
                _responses_text_done_events(
                    done_event,
                    self.block_indices,
                    self.open_blocks,
                    self.text_parts,
                    self.refusal_parts,
                ),
            )
        return b"".join(events)

    def reconcile_output_item(self, event: dict[str, Any]) -> bytes:
        item = event["item"]
        if item.get("type") == "message":
            return self.reconcile_message_item(item, int(event.get("output_index", 0)))
        events, self.next_content_index = _responses_output_item_done_events(
            event,
            self.next_content_index,
            self.block_indices,
            self.open_blocks,
            self.tool_argument_parts,
        )
        return events

    def reconcile_terminal_output(self, output: object) -> bytes:
        if not isinstance(output, list):
            return b""
        events: list[bytes] = []
        for output_index, item in enumerate(output):
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type == "message":
                events.append(self.reconcile_message_item(item, output_index))
            elif item_type == "function_call":
                self.tool_items[output_index] = item
                self.has_tool_call = True
                events.append(self.start_tool(output_index))
                events.append(
                    self.reconcile_output_item(
                        {
                            "type": "response.output_item.done",
                            "output_index": output_index,
                            "item": item,
                        },
                    ),
                )
        return b"".join(events)

    def has_visible_output(self) -> bool:
        return self.has_tool_call or any("".join(parts) for parts in self.text_parts.values())

    def finish(self, terminal: dict[str, Any], event_type: str) -> bytes:
        events = [self.reconcile_terminal_output(terminal.get("output"))]
        if event_type == "response.completed" and not self.has_visible_output():
            raise ConversionError(
                "Responses API completed without assistant text or a function call",
            )
        events.extend(
            sse(
                "content_block_stop",
                {"type": "content_block_stop", "index": index},
            )
            for index in sorted(self.open_blocks)
        )
        self.open_blocks.clear()
        terminal_events, self.next_content_index = _responses_terminal_events(
            terminal,
            self.next_content_index,
            _ResponsesTerminalContext(
                has_tool_call=self.has_tool_call,
                has_refusal=self.has_refusal,
                refusal_text="".join(self.refusal_parts),
                upstream_model=self.upstream_model,
            ),
        )
        events.append(terminal_events)
        return b"".join(events)


async def responses_stream_to_anthropic(
    response: httpx.Response,
    requested_model: str,
    ping_interval: float = 0,
    request_payload: dict[str, Any] | None = None,
) -> AsyncIterator[bytes]:
    """Translate typed OpenAI Responses events into Anthropic event order."""
    raw_upstream_model = (request_payload or {}).get("model")
    state = _ResponsesStreamState(
        requested_model,
        raw_upstream_model if isinstance(raw_upstream_model, str) else None,
    )

    try:
        async for raw_data in iter_sse_data(response, ping_interval):
            if raw_data is None:
                yield sse("ping", {"type": "ping"})
                continue
            event = _decode_sse_object(raw_data, "Responses")
            event_type = _validate_responses_stream_event(event)
            if event_type == "response.created":
                created = event["response"]
                state.message_id = created.get("id") or state.message_id
                yield state.ensure_started()
            elif event_type == "response.output_item.added":
                item = event["item"]
                if item.get("type") == "function_call":
                    output_index = int(event.get("output_index", 0))
                    state.tool_items[output_index] = item
                    state.has_tool_call = True
                    yield state.ensure_started()
                    yield state.start_tool(output_index)
            elif event_type in {"response.output_text.delta", "response.refusal.delta"}:
                state.has_refusal = state.has_refusal or event_type == "response.refusal.delta"
                output_index = int(event.get("output_index", 0))
                content_index = int(event.get("content_index", 0))
                yield state.ensure_started()
                yield state.start_text(output_index, content_index)
                index = state.block_indices[("text", output_index, content_index)]
                _require_open_responses_block(index, state.open_blocks)
                delta = event["delta"]
                _record_responses_refusal_delta(event_type, delta, state.refusal_parts)
                if delta:
                    state.text_parts.setdefault((output_index, content_index), []).append(delta)
                    yield sse(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": index,
                            "delta": {"type": "text_delta", "text": delta},
                        },
                    )
            elif event_type in {
                "response.function_call_arguments.delta",
                "response.function_call_arguments.done",
            }:
                output_index = int(event.get("output_index", 0))
                yield state.ensure_started()
                yield state.start_tool(output_index)
                index = state.block_indices[("tool", output_index, 0)]
                argument_events = _responses_argument_event(
                    event,
                    index,
                    output_index,
                    state.tool_argument_parts,
                )
                yield argument_events
            elif event_type in {
                "response.output_text.done",
                "response.refusal.done",
                "response.content_part.done",
            }:
                output_index = int(event.get("output_index", 0))
                content_index = int(event.get("content_index", 0))
                yield state.ensure_started()
                yield state.start_text(output_index, content_index)
                done_events = _responses_text_done_events(
                    event,
                    state.block_indices,
                    state.open_blocks,
                    state.text_parts,
                    state.refusal_parts,
                )
                yield done_events
            elif event_type == "response.output_item.done":
                yield state.ensure_started()
                yield state.reconcile_output_item(event)
            elif event_type in {"response.completed", "response.incomplete"}:
                terminal = event["response"]
                state.message_id = terminal.get("id") or state.message_id
                yield state.ensure_started()
                yield state.finish(terminal, event_type)
                return
            elif event_type in {"error", "response.failed"}:
                yield _responses_error_event(event)
                return
        yield sse(
            "error",
            {
                "type": "error",
                "error": {
                    "type": "api_error",
                    "message": "Responses API stream ended before a terminal event",
                },
            },
        )
    except ConversionError as exc:
        yield _error_event(str(exc), "Invalid Responses stream event")
    except httpx.HTTPError:
        yield _error_event(None, "Responses upstream stream disconnected")
    except (AttributeError, TypeError, ValueError):
        yield _error_event(None, "Malformed Responses stream event")
    finally:
        await response.aclose()


async def passthrough_stream(response: httpx.Response) -> AsyncIterator[bytes]:
    """Relay upstream bytes while guaranteeing that the response is closed."""
    try:
        async for chunk in response.aiter_bytes():
            yield chunk
    finally:
        await response.aclose()
