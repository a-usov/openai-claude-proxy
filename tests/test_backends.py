from __future__ import annotations

import pytest

from openai_claude_proxy.backends import select_messages_backend, select_openai_backend
from openai_claude_proxy.config import ConfigError, Settings
from openai_claude_proxy.exceptions import ConversionError
from openai_claude_proxy.models import automatic_model_alias


def test_auto_protocol_classifies_the_mapped_upstream_model() -> None:
    settings = Settings(
        upstream_protocol="auto",
        openai_api="responses",
        upstream_messages_path="/v1/messages",
        model_map={
            "claude-opus-*": "gateway-claude-opus",
            "claude-sonnet-*": "gateway-gpt-reasoning",
        },
    )
    openai = select_openai_backend(settings)

    anthropic = select_messages_backend(settings, "claude-opus-5", openai)
    translated = select_messages_backend(settings, "claude-sonnet-4-6", openai)

    assert anthropic.protocol == "anthropic"
    assert anthropic.path == "/v1/messages"
    assert anthropic.token_count_path == f"{anthropic.path}/count_tokens"
    assert anthropic.mapped_model == "gateway-claude-opus"
    assert not anthropic.translates_openai
    assert translated.protocol == "openai"
    assert translated.path == settings.upstream_responses_path
    assert translated.mapped_model == "gateway-gpt-reasoning"
    assert translated.translates_openai


def test_auto_anthropic_route_rewrites_only_the_model() -> None:
    settings = Settings(
        upstream_protocol="auto",
        model_map={"claude-picker-alias": "team-claude-deployment"},
    )
    selected = select_messages_backend(
        settings,
        "claude-picker-alias",
        select_openai_backend(settings),
    )
    payload = {
        "model": "claude-picker-alias",
        "messages": [{"role": "user", "content": "hello"}],
        "future_anthropic_field": {"enabled": True},
    }

    assert selected.convert_request(payload, settings) == {
        **payload,
        "model": "team-claude-deployment",
    }
    assert payload["model"] == "claude-picker-alias"


def test_auto_protocol_patterns_are_case_insensitive_and_configurable() -> None:
    settings = Settings.from_env(
        {
            "UPSTREAM_PROTOCOL": "auto",
            "ANTHROPIC_MODEL_PATTERNS": "team-opus-*,TEAM-SONNET-*",
        },
    )
    openai = select_openai_backend(settings)

    assert select_messages_backend(settings, "TEAM-OPUS-1", openai).protocol == "anthropic"
    assert select_messages_backend(settings, "team-sonnet-2", openai).protocol == "anthropic"
    assert select_messages_backend(settings, "team-gpt-1", openai).protocol == "openai"


def test_auto_protocol_classifies_decoded_discovery_aliases() -> None:
    settings = Settings(upstream_protocol="auto", model_discovery_mode="auto")
    openai = select_openai_backend(settings)

    anthropic_alias = automatic_model_alias("gateway-claude-opus")
    openai_alias = automatic_model_alias("gateway-gpt-reasoning")

    assert select_messages_backend(settings, anthropic_alias, openai).protocol == "anthropic"
    assert select_messages_backend(settings, openai_alias, openai).protocol == "openai"


def test_auto_protocol_requires_a_model_and_anthropic_pattern() -> None:
    settings = Settings(upstream_protocol="auto")
    with pytest.raises(ConversionError, match="model must be a non-empty string"):
        select_messages_backend(settings, None, select_openai_backend(settings))
    with pytest.raises(ConfigError, match="ANTHROPIC_MODEL_PATTERNS"):
        Settings(upstream_protocol="auto", anthropic_model_patterns=())
