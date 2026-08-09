"""Stateful scripted HTTP transport for multi-request protocol tests."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any

import httpx

JsonObject = dict[str, Any]
RequestValidator = Callable[[httpx.Request, JsonObject], None]


class _TrackedStream(httpx.AsyncByteStream):
    def __init__(self, body: bytes) -> None:
        self._body = body
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield self._body

    async def aclose(self) -> None:
        self.closed = True


@dataclass(frozen=True, slots=True)
class ExpectedRequest:
    """Describe one ordered upstream request and its scripted response."""

    method: str
    path: str
    validate: RequestValidator
    response_body: bytes
    response_status: int = 200
    response_headers: dict[str, str] = field(
        default_factory=lambda: {"content-type": "text/event-stream"},
    )


class ScenarioTransport(httpx.AsyncBaseTransport):
    """Execute expected requests in order and reject unexpected traffic."""

    def __init__(self, steps: list[ExpectedRequest]) -> None:
        """Initialize the scenario with an ordered copy of its steps."""
        self._steps = steps.copy()
        self._position = 0
        self._response_streams: list[_TrackedStream] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        """Validate one request and return the corresponding scripted response."""
        if self._position >= len(self._steps):
            msg = f"Unexpected upstream request: {request.method} {request.url.path}"
            raise AssertionError(msg)

        step = self._steps[self._position]
        self._position += 1
        assert request.method == step.method
        assert request.url.path == step.path

        decoded = json.loads(await request.aread())
        assert isinstance(decoded, dict)
        step.validate(request, decoded)
        stream = _TrackedStream(step.response_body)
        self._response_streams.append(stream)
        return httpx.Response(
            step.response_status,
            headers=step.response_headers,
            stream=stream,
            request=request,
        )

    def assert_complete(self) -> None:
        """Assert that every scripted upstream step was consumed."""
        assert self._position == len(self._steps), (
            f"Consumed {self._position} of {len(self._steps)} expected upstream requests"
        )
        assert all(stream.closed for stream in self._response_streams), (
            "One or more scripted upstream responses were not closed"
        )
