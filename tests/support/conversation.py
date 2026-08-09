"""A small Claude-side conversation driver for proxy integration tests."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import httpx


@dataclass(frozen=True, slots=True)
class AnthropicTurn:
    """Contain one accumulated Anthropic message and its raw SSE events."""

    message: dict[str, Any]
    events: list[dict[str, Any]]

    @property
    def tool_uses(self) -> list[dict[str, Any]]:
        """Return tool-use content blocks in response order."""
        return [block for block in self.message["content"] if block.get("type") == "tool_use"]


def _parse_sse(body: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    event_name: str | None = None
    data_lines: list[str] = []

    for line in [*body.splitlines(), ""]:
        if line == "":
            if data_lines:
                payload = json.loads("\n".join(data_lines))
                assert isinstance(payload, dict)
                if event_name is not None:
                    assert payload.get("type") == event_name
                events.append(payload)
            event_name = None
            data_lines.clear()
        elif line.startswith("event:"):
            event_name = line[6:].strip()
        elif line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
    return events


def _accumulate_message(events: list[dict[str, Any]]) -> dict[str, Any]:
    message: dict[str, Any] | None = None
    blocks: dict[int, dict[str, Any]] = {}
    partial_json: dict[int, list[str]] = {}
    stopped = False

    for event in events:
        event_type = event.get("type")
        if event_type == "message_start":
            assert message is None
            message = deepcopy(event["message"])
        elif event_type == "content_block_start":
            index = int(event["index"])
            assert index not in blocks
            blocks[index] = deepcopy(event["content_block"])
        elif event_type == "content_block_delta":
            index = int(event["index"])
            delta = event["delta"]
            if delta.get("type") == "text_delta":
                blocks[index]["text"] += delta.get("text", "")
            elif delta.get("type") == "input_json_delta":
                partial_json.setdefault(index, []).append(delta.get("partial_json", ""))
        elif event_type == "content_block_stop":
            index = int(event["index"])
            if index in partial_json:
                blocks[index]["input"] = json.loads("".join(partial_json[index]) or "{}")
        elif event_type == "message_delta":
            assert message is not None
            message.update(event.get("delta") or {})
            if usage := event.get("usage"):
                message["usage"] = usage
        elif event_type == "message_stop":
            stopped = True

    assert message is not None
    assert stopped
    message["content"] = [blocks[index] for index in sorted(blocks)]
    return message


class AnthropicConversation:
    """Drive a multi-turn Anthropic Messages conversation through the proxy."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        model: str,
        tools: list[dict[str, Any]],
        headers: dict[str, str] | None = None,
        max_tokens: int = 1024,
    ) -> None:
        """Initialize an empty conversation with stable request configuration."""
        self._client = client
        self._model = model
        self._tools = deepcopy(tools)
        self._headers = headers or {}
        self._max_tokens = max_tokens
        self.request_overrides: dict[str, Any] = {}
        self.messages: list[dict[str, Any]] = []

    async def ask(self, text: str) -> AnthropicTurn:
        """Append a user message and run the next assistant turn."""
        self.messages.append({"role": "user", "content": text})
        return await self._run_turn()

    async def submit_tool_results(
        self,
        results: dict[str, str],
        *,
        error_ids: set[str] | None = None,
    ) -> AnthropicTurn:
        """Append tool results and run the next assistant turn."""
        assert self.messages
        assert self.messages[-1]["role"] == "assistant"
        failures = error_ids or set()
        self.messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": output,
                        **({"is_error": True} if tool_id in failures else {}),
                    }
                    for tool_id, output in results.items()
                ],
            },
        )
        return await self._run_turn()

    async def _run_turn(self) -> AnthropicTurn:
        payload = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "stream": True,
            "messages": self.messages,
            "tools": self._tools,
        }
        payload.update(deepcopy(self.request_overrides))
        response = await self._client.post(
            "/v1/messages",
            headers=self._headers,
            json=payload,
        )
        response.raise_for_status()
        events = _parse_sse(response.text)
        message = _accumulate_message(events)
        self.messages.append({"role": "assistant", "content": deepcopy(message["content"])})
        return AnthropicTurn(message=message, events=events)
