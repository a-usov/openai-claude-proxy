"""Anthropic-compatible error normalization for translated protocols."""

from __future__ import annotations

import re
from typing import Literal, TypeAlias, cast

AnthropicErrorType: TypeAlias = Literal[
    "invalid_request_error",
    "authentication_error",
    "permission_error",
    "not_found_error",
    "rate_limit_error",
    "timeout_error",
    "overloaded_error",
    "api_error",
    "billing_error",
]

_ALLOWED_ERROR_TYPES: frozenset[str] = frozenset(
    {
        "invalid_request_error",
        "authentication_error",
        "permission_error",
        "not_found_error",
        "rate_limit_error",
        "timeout_error",
        "overloaded_error",
        "api_error",
        "billing_error",
    },
)
_TYPE_ALIASES: dict[str, AnthropicErrorType] = {
    "bad_request": "invalid_request_error",
    "conflict_error": "invalid_request_error",
    "invalid_api_key": "authentication_error",
    "invalid_prompt": "invalid_request_error",
    "invalid_request": "invalid_request_error",
    "permission_denied": "permission_error",
    "rate_limit_exceeded": "rate_limit_error",
    "server_error": "api_error",
    "service_unavailable": "api_error",
    "insufficient_quota": "billing_error",
    "billing_hard_limit_reached": "billing_error",
}
_STATUS_ERROR_TYPES: dict[int, AnthropicErrorType] = {
    401: "authentication_error",
    402: "billing_error",
    403: "permission_error",
    404: "not_found_error",
    408: "timeout_error",
    429: "rate_limit_error",
    504: "timeout_error",
    529: "overloaded_error",
}
_CLIENT_ERROR_MIN = 400
_SERVER_ERROR_MIN = 500
_MAX_ERROR_MESSAGE_LENGTH = 2048
_MAX_REQUEST_ID_LENGTH = 256
_URL = re.compile(r"\bhttps?://[^\s<>\"']+", re.IGNORECASE)
_BEARER = re.compile(r"\bbearer\s+[^\s,;]+", re.IGNORECASE)
_SECRET_ASSIGNMENT = re.compile(
    r"\b(authorization|api[-_ ]?key|access[-_ ]?token|token|secret|password|credential|sig)"
    r"\s*[:=]\s*(?:\"[^\"]*\"|'[^']*'|[^\s,;]+)",
    re.IGNORECASE,
)
_SENSITIVE_PAYLOAD = re.compile(
    r"\b(request\s+body|tool\s+result|prompt)\s*[:=]\s*.+$",
    re.IGNORECASE,
)


def normalize_error_type(
    provider_type: object = None,
    *,
    code: object = None,
    status_code: int | None = None,
) -> AnthropicErrorType:
    """Map provider-specific discriminators and HTTP status to Anthropic's union."""
    if isinstance(provider_type, str):
        normalized = provider_type.strip().lower()
        if normalized in _ALLOWED_ERROR_TYPES:
            return cast("AnthropicErrorType", normalized)

    for candidate in (code, provider_type):
        if isinstance(candidate, str):
            normalized = candidate.strip().lower()
        else:
            continue
        if mapped := _TYPE_ALIASES.get(normalized):
            return mapped

    if status_code in _STATUS_ERROR_TYPES:
        return _STATUS_ERROR_TYPES[status_code]
    if status_code is not None and _CLIENT_ERROR_MIN <= status_code < _SERVER_ERROR_MIN:
        return "invalid_request_error"
    return "api_error"


def safe_error_message(value: object, fallback: str) -> str:
    """Return bounded single-line provider text suitable for a client envelope."""
    if not isinstance(value, str) or not value.strip():
        value = fallback
    cleaned = " ".join(value.split())
    cleaned = _URL.sub("[redacted URL]", cleaned)
    cleaned = _BEARER.sub("Bearer [redacted]", cleaned)
    cleaned = _SECRET_ASSIGNMENT.sub(lambda match: f"{match.group(1)}=[redacted]", cleaned)
    cleaned = _SENSITIVE_PAYLOAD.sub(lambda match: f"{match.group(1)}=[redacted]", cleaned)
    if len(cleaned) <= _MAX_ERROR_MESSAGE_LENGTH:
        return cleaned
    return cleaned[: _MAX_ERROR_MESSAGE_LENGTH - 1] + "…"


def anthropic_error(
    message: object,
    *,
    provider_type: object = None,
    code: object = None,
    status_code: int | None = None,
    request_id: object = None,
) -> dict[str, object]:
    """Build an Anthropic SDK-compatible error envelope."""
    body: dict[str, object] = {
        "type": "error",
        "error": {
            "type": normalize_error_type(
                provider_type,
                code=code,
                status_code=status_code,
            ),
            "message": safe_error_message(message, "Request failed"),
        },
    }
    if isinstance(request_id, str) and request_id:
        safe_request_id = safe_error_message(request_id, "")[:_MAX_REQUEST_ID_LENGTH]
        if safe_request_id:
            body["request_id"] = safe_request_id
    return body
