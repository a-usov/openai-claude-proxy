from __future__ import annotations

import base64

import pytest

from openai_claude_proxy.reasoning_state import (
    decode_responses_output,
    decode_responses_output_state,
    decode_responses_reasoning,
    encode_responses_output,
    encode_responses_reasoning,
)


def test_reasoning_state_codec_preserves_the_complete_item() -> None:
    item = {
        "type": "reasoning",
        "id": "rs_1",
        "status": "completed",
        "summary": [{"type": "summary_text", "text": "Safe summary"}],
        "encrypted_content": "opaque-state",
    }

    assert decode_responses_reasoning(encode_responses_reasoning(item)) == item


def test_reasoning_state_codec_ignores_foreign_redacted_thinking() -> None:
    assert decode_responses_reasoning("anthropic-owned-state") is None


def test_reasoning_state_codec_rejects_corrupt_proxy_state() -> None:
    encoded = encode_responses_reasoning({"type": "reasoning", "id": "rs_1"})
    decoded = base64.b64decode(encoded)
    corrupted = base64.b64encode(decoded[:-1]).decode()

    with pytest.raises(ValueError, match="Invalid proxy Responses reasoning state"):
        decode_responses_reasoning(corrupted)


def test_output_state_codec_preserves_all_ordered_items() -> None:
    items = [
        {
            "type": "message",
            "id": "msg_1",
            "role": "assistant",
            "status": "completed",
            "phase": "final_answer",
            "content": [
                {
                    "type": "output_text",
                    "text": "Done.",
                    "annotations": [{"type": "custom", "value": "kept"}],
                },
            ],
        },
        {
            "type": "function_call",
            "id": "fc_1",
            "call_id": "call_1",
            "name": "read",
            "arguments": '{"path":"README.md"}',
            "status": "completed",
            "caller": "tool",
            "namespace": "workspace",
        },
    ]

    assert decode_responses_output(encode_responses_output(items)) == items


def test_output_state_codec_preserves_the_bound_upstream_model() -> None:
    items = [{"type": "message", "role": "assistant", "content": []}]

    state = decode_responses_output_state(
        encode_responses_output(items, model="upstream-deployment-a"),
    )

    assert state is not None
    assert state.output == items
    assert state.model == "upstream-deployment-a"


def test_output_state_codec_ignores_foreign_redacted_thinking() -> None:
    assert decode_responses_output("anthropic-owned-state") is None


def test_output_state_codec_rejects_corrupt_proxy_state() -> None:
    encoded = encode_responses_output([{"type": "message"}])
    decoded = base64.b64decode(encoded)
    corrupted = base64.b64encode(decoded[:-1]).decode()

    with pytest.raises(ValueError, match="Invalid proxy Responses output state"):
        decode_responses_output(corrupted)


def test_output_state_codec_rejects_untyped_items() -> None:
    with pytest.raises(ValueError, match="typed objects"):
        encode_responses_output([{}])
