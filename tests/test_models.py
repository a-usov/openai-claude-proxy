from __future__ import annotations

import pytest

from openai_claude_proxy.exceptions import ConversionError
from openai_claude_proxy.models import (
    automatic_model_alias,
    automatic_upstream_model,
    openai_model_aliases,
)


@pytest.mark.parametrize(
    "model_id",
    [
        "gateway-gpt-5-6-luna",
        "vendor/kimi-k2.5:latest",
        "unicode-模型",
    ],
)
def test_automatic_model_alias_is_claude_compatible_and_reversible(model_id: str) -> None:
    alias = automatic_model_alias(model_id)

    assert alias.startswith("claude-proxy-")
    assert automatic_upstream_model(alias) == model_id


@pytest.mark.parametrize(
    "alias",
    [
        "claude-proxy-",
        "claude-proxy-%67ateway-model",
        "claude-proxy-%FF",
        "other-model",
    ],
)
def test_automatic_model_alias_rejects_invalid_or_noncanonical_values(alias: str) -> None:
    assert automatic_upstream_model(alias) is None


def test_openai_model_aliases_filter_and_preserve_display_names() -> None:
    aliases = openai_model_aliases(
        {
            "object": "list",
            "data": [
                {"id": "gateway-gpt-5-6-sol", "object": "model"},
                {
                    "id": "gateway-gpt-5-6-luna",
                    "object": "model",
                    "display_name": "GPT-5.6 Luna",
                },
                {"id": "gateway-gpt-5-6-terra", "object": "model"},
                {"id": "kimi-k2.5", "object": "model"},
                {"id": "text-embedding-3-large", "object": "model"},
            ],
        },
        include=("gateway-gpt-*", "kimi-*"),
        exclude=("*-terra",),
    )

    assert aliases == {
        "claude-proxy-gateway-gpt-5-6-sol": "gateway-gpt-5-6-sol",
        "claude-proxy-gateway-gpt-5-6-luna": "GPT-5.6 Luna",
        "claude-proxy-kimi-k2.5": "kimi-k2.5",
    }


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {"object": "model", "data": []},
        {"object": "list"},
        {"object": "list", "data": ["model-id"]},
        {"object": "list", "data": [{}]},
        {"object": "list", "data": [{"id": ""}]},
    ],
)
def test_openai_model_aliases_reject_invalid_upstream_schemas(payload: object) -> None:
    with pytest.raises(ConversionError):
        openai_model_aliases(payload, include=("*",), exclude=())
