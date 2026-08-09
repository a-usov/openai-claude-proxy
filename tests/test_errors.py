from __future__ import annotations

import pytest

from openai_claude_proxy.errors import (
    anthropic_error,
    normalize_error_type,
    safe_error_message,
)


@pytest.mark.parametrize(
    ("provider_type", "code", "status_code", "expected"),
    [
        ("rate_limit_error", None, 500, "rate_limit_error"),
        ("server_error", "rate_limit_exceeded", 500, "rate_limit_error"),
        ("insufficient_quota", None, 429, "billing_error"),
        ("unknown", None, 401, "authentication_error"),
        ("unknown", None, 402, "billing_error"),
        ("unknown", None, 403, "permission_error"),
        ("unknown", None, 404, "not_found_error"),
        ("unknown", None, 408, "timeout_error"),
        ("unknown", None, 429, "rate_limit_error"),
        ("unknown", None, 529, "overloaded_error"),
        ("unknown", None, 422, "invalid_request_error"),
        ("unknown", None, 500, "api_error"),
    ],
)
def test_normalize_error_type_uses_only_anthropic_discriminators(
    provider_type: object,
    code: object,
    status_code: int,
    expected: str,
) -> None:
    assert normalize_error_type(provider_type, code=code, status_code=status_code) == expected


def test_anthropic_error_includes_optional_request_id() -> None:
    assert anthropic_error(
        "slow down",
        provider_type="rate_limit_exceeded",
        request_id="req_123",
    ) == {
        "type": "error",
        "error": {"type": "rate_limit_error", "message": "slow down"},
        "request_id": "req_123",
    }


def test_safe_error_message_is_single_line_and_bounded() -> None:
    message = safe_error_message("first\nsecond " + "x" * 3000, "fallback")

    assert message.startswith("first second ")
    assert "\n" not in message
    assert len(message) == 2048
    assert message.endswith("…")


def test_safe_error_message_redacts_credentials_urls_and_sensitive_payloads() -> None:
    message = safe_error_message(
        "Failed at https://internal.example/path?sig=url-secret "
        "Authorization: Bearer bearer-secret api_key=key-secret "
        'request body: {"prompt":"proprietary prompt"}',
        "fallback",
    )

    assert message == (
        "Failed at [redacted URL] Authorization=[redacted] [redacted] "
        "api_key=[redacted] request body=[redacted]"
    )
    for secret in ("internal.example", "url-secret", "bearer-secret", "key-secret", "proprietary"):
        assert secret not in message


def test_anthropic_error_sanitizes_untrusted_request_id() -> None:
    result = anthropic_error(
        "failed",
        request_id="token=request-id-secret",
    )

    assert result["request_id"] == "token=[redacted]"
