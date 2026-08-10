from __future__ import annotations

import pytest

from openai_claude_proxy.auth import CredentialError, upstream_headers
from openai_claude_proxy.config import ConfigError, Settings


@pytest.mark.parametrize("mode", ["bearer", "api-key", "x-api-key"])
@pytest.mark.parametrize(
    "incoming",
    [
        {"authorization": "Bearer helper-secret"},
        {"authorization": "helper-secret"},
        {"x-api-key": "helper-secret"},
        {"api-key": "helper-secret"},
        {
            "authorization": "Bearer helper-secret",
            "x-api-key": "helper-secret",
        },
    ],
)
def test_rewrite_modes_accept_each_helper_credential_location(
    mode: str,
    incoming: dict[str, str],
) -> None:
    result = upstream_headers(incoming, Settings(auth_mode=mode))

    expected_name = "authorization" if mode == "bearer" else mode
    expected_value = "Bearer helper-secret" if mode == "bearer" else "helper-secret"
    assert result[expected_name] == expected_value
    assert set(result) & {"authorization", "api-key", "x-api-key"} == {expected_name}


@pytest.mark.parametrize("mode", ["bearer", "api-key", "x-api-key"])
def test_rewrite_modes_reject_disagreeing_or_missing_credentials(mode: str) -> None:
    with pytest.raises(CredentialError, match="disagree"):
        upstream_headers(
            {"authorization": "Bearer first-secret", "x-api-key": "second-secret"},
            Settings(auth_mode=mode),
        )
    with pytest.raises(CredentialError, match="No helper credential"):
        upstream_headers({}, Settings(auth_mode=mode))


def test_passthrough_preserves_disagreeing_credentials_and_none_strips_them() -> None:
    incoming = {
        "authorization": "Bearer first-secret",
        "x-api-key": "second-secret",
        "api-key": "third-secret",
    }

    preserved = upstream_headers(incoming, Settings(auth_mode="passthrough"))
    stripped = upstream_headers(incoming, Settings(auth_mode="none"))

    for name, value in incoming.items():
        assert preserved[name] == value
        assert name not in stripped


def test_static_auth_ignores_helper_headers_and_static_headers_have_final_precedence() -> None:
    result = upstream_headers(
        {
            "authorization": "Bearer helper-secret",
            "x-api-key": "different-helper-secret",
        },
        Settings(
            auth_mode="static",
            upstream_api_key="configured-secret",
            upstream_api_key_header="authorization",
            upstream_api_key_scheme="Bearer",
            static_headers={"authorization": "Gateway final-secret", "x-tenant": "work"},
        ),
    )

    assert result["authorization"] == "Gateway final-secret"
    assert result["x-tenant"] == "work"
    assert "x-api-key" not in result


def test_provider_header_policies_are_separate_and_hop_by_hop_headers_are_stripped() -> None:
    incoming = {
        "anthropic-version": "2023-06-01",
        "anthropic-beta": "future-beta",
        "openai-organization": "org_work",
        "x-correlation-id": "corr_1",
        "connection": "keep-alive",
        "cookie": "secret-cookie",
    }

    openai = upstream_headers(incoming, Settings(upstream_protocol="openai"))
    anthropic = upstream_headers(incoming, Settings(upstream_protocol="anthropic"))

    assert openai["openai-organization"] == "org_work"
    assert openai["x-correlation-id"] == "corr_1"
    assert "anthropic-version" not in openai
    assert "anthropic-beta" not in openai
    assert anthropic["anthropic-version"] == "2023-06-01"
    assert anthropic["anthropic-beta"] == "future-beta"
    assert anthropic["x-correlation-id"] == "corr_1"
    assert "openai-organization" not in anthropic
    for result in (openai, anthropic):
        assert "connection" not in result
        assert "cookie" not in result


def test_auto_protocol_uses_the_selected_route_header_policy() -> None:
    incoming = {
        "anthropic-version": "2023-06-01",
        "anthropic-beta": "future-beta",
        "openai-organization": "org_work",
    }
    settings = Settings(upstream_protocol="auto")

    anthropic = upstream_headers(incoming, settings, protocol="anthropic")
    openai = upstream_headers(incoming, settings, protocol="openai")

    assert anthropic["anthropic-version"] == "2023-06-01"
    assert anthropic["anthropic-beta"] == "future-beta"
    assert "openai-organization" not in anthropic
    assert openai["openai-organization"] == "org_work"
    assert "anthropic-version" not in openai
    assert "anthropic-beta" not in openai


def test_anthropic_mandatory_headers_cannot_be_removed_and_openai_opt_in_is_explicit() -> None:
    incoming = {
        "anthropic-version": "2023-06-01",
        "anthropic-beta": "future-beta",
    }

    anthropic = upstream_headers(
        incoming,
        Settings(
            upstream_protocol="anthropic",
            forward_headers=(),
            anthropic_forward_headers=(),
        ),
    )
    openai = upstream_headers(
        incoming,
        Settings(
            upstream_protocol="openai",
            openai_forward_headers=("anthropic-beta",),
        ),
    )

    assert anthropic == {
        "accept": "application/json",
        "content-type": "application/json",
        "anthropic-version": "2023-06-01",
        "anthropic-beta": "future-beta",
    }
    assert openai["anthropic-beta"] == "future-beta"
    assert "anthropic-version" not in openai


@pytest.mark.parametrize(
    "environment",
    [
        {"FORWARD_HEADERS": "authorization"},
        {"OPENAI_FORWARD_HEADERS": "connection"},
        {"ANTHROPIC_FORWARD_HEADERS": "bad header"},
    ],
)
def test_configured_forward_policies_reject_auth_hop_by_hop_and_invalid_names(
    environment: dict[str, str],
) -> None:
    with pytest.raises(ConfigError, match="forbidden header"):
        Settings.from_env(environment)


@pytest.mark.parametrize(
    "setting_name",
    ["MAX_REQUEST_BODY_BYTES", "MAX_RESPONSE_BODY_BYTES", "MAX_ERROR_BODY_BYTES"],
)
def test_body_size_limits_must_be_positive(setting_name: str) -> None:
    with pytest.raises(ConfigError, match="positive integer"):
        Settings.from_env({setting_name: "0"})
