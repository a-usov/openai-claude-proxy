"""OpenAI protocol adapters selected once during application construction."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from fnmatch import fnmatchcase
from typing import Any

import httpx

from .config import Settings
from .conversion import (
    anthropic_to_openai,
    anthropic_to_responses,
    anthropic_to_responses_token_count,
    openai_to_anthropic,
    responses_to_anthropic,
    validate_anthropic_chat_token_count,
)
from .exceptions import ConversionError
from .streaming import openai_stream_to_anthropic, responses_stream_to_anthropic

RequestConverter = Callable[[dict[str, Any], Settings], dict[str, Any]]
ResponseConverter = Callable[[dict[str, Any], str, dict[str, Any]], dict[str, Any]]
StreamConverter = Callable[
    [httpx.Response, str, float, dict[str, Any]],
    AsyncIterator[bytes],
]


@dataclass(frozen=True, slots=True)
class OpenAIBackend:
    """Bind one OpenAI wire protocol's path and translation functions."""

    path: str
    request_converter: RequestConverter
    response_converter: ResponseConverter
    stream_converter: StreamConverter
    token_count_path: str | None
    token_count_converter: RequestConverter

    def convert_request(
        self,
        payload: dict[str, Any],
        settings: Settings,
    ) -> dict[str, Any]:
        """Convert an Anthropic request into this backend's wire format."""
        return self.request_converter(payload, settings)

    def convert_response(
        self,
        payload: dict[str, Any],
        requested_model: str,
        request_payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Convert this backend's buffered result into an Anthropic message."""
        return self.response_converter(payload, requested_model, request_payload)

    def convert_stream(
        self,
        response: httpx.Response,
        requested_model: str,
        ping_interval: float,
        request_payload: dict[str, Any],
    ) -> AsyncIterator[bytes]:
        """Convert this backend's event stream into Anthropic events."""
        return self.stream_converter(
            response,
            requested_model,
            ping_interval,
            request_payload,
        )

    def convert_token_count_request(
        self,
        payload: dict[str, Any],
        settings: Settings,
    ) -> dict[str, Any]:
        """Validate and convert a token-count request for this backend."""
        return self.token_count_converter(payload, settings)


@dataclass(frozen=True, slots=True)
class MessagesBackend:
    """Describe one request's selected upstream Messages-compatible route."""

    protocol: str
    path: str
    token_count_path: str | None
    mapped_model: str
    openai: OpenAIBackend | None = None
    rewrite_anthropic_model: bool = False

    @property
    def translates_openai(self) -> bool:
        """Return whether this route translates through an OpenAI wire protocol."""
        return self.openai is not None

    def convert_request(
        self,
        payload: dict[str, Any],
        settings: Settings,
    ) -> dict[str, Any]:
        """Convert or minimally rewrite one Messages request for this route."""
        if self.openai is not None:
            return self.openai.convert_request(payload, settings)
        if self.rewrite_anthropic_model:
            return {**payload, "model": self.mapped_model}
        return payload

    def convert_token_count_request(
        self,
        payload: dict[str, Any],
        settings: Settings,
    ) -> dict[str, Any]:
        """Convert or minimally rewrite one token-count request for this route."""
        if self.openai is not None:
            return self.openai.convert_token_count_request(payload, settings)
        if self.rewrite_anthropic_model:
            return {**payload, "model": self.mapped_model}
        return payload


def select_openai_backend(settings: Settings) -> OpenAIBackend:
    """Select the configured backend once, before serving any requests."""
    backends = {
        "chat_completions": OpenAIBackend(
            path=settings.upstream_chat_path,
            request_converter=anthropic_to_openai,
            response_converter=openai_to_anthropic,
            stream_converter=openai_stream_to_anthropic,
            token_count_path=None,
            token_count_converter=validate_anthropic_chat_token_count,
        ),
        "responses": OpenAIBackend(
            path=settings.upstream_responses_path,
            request_converter=anthropic_to_responses,
            response_converter=responses_to_anthropic,
            stream_converter=responses_stream_to_anthropic,
            token_count_path=settings.upstream_responses_input_tokens_path,
            token_count_converter=anthropic_to_responses_token_count,
        ),
    }
    return backends[settings.openai_api]


def select_messages_backend(
    settings: Settings,
    requested_model: object,
    openai_backend: OpenAIBackend,
) -> MessagesBackend:
    """Select one upstream protocol after resolving the mapped upstream model."""
    if not isinstance(requested_model, str) or not requested_model:
        raise ConversionError("model must be a non-empty string")

    mapped_model = settings.map_model(requested_model)
    protocol = settings.upstream_protocol
    rewrite_anthropic_model = False
    if protocol == "auto":
        normalized_model = mapped_model.lower()
        protocol = (
            "anthropic"
            if any(
                fnmatchcase(normalized_model, pattern)
                for pattern in settings.anthropic_model_patterns
            )
            else "openai"
        )
        rewrite_anthropic_model = protocol == "anthropic"

    if protocol == "anthropic":
        messages_path = settings.upstream_messages_path
        return MessagesBackend(
            protocol="anthropic",
            path=messages_path,
            token_count_path=f"{messages_path.rstrip('/')}/count_tokens",
            mapped_model=mapped_model,
            rewrite_anthropic_model=rewrite_anthropic_model,
        )
    return MessagesBackend(
        protocol="openai",
        path=openai_backend.path,
        token_count_path=openai_backend.token_count_path,
        mapped_model=mapped_model,
        openai=openai_backend,
    )
