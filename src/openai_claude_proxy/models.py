"""Anthropic Models API shapes for configured Claude-visible aliases."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .exceptions import ConversionError

if TYPE_CHECKING:
    from collections.abc import Mapping

UNKNOWN_MODEL_CREATED_AT = "1970-01-01T00:00:00Z"
_DEFAULT_PAGE_LIMIT = 20
_MIN_PAGE_LIMIT = 1
_MAX_PAGE_LIMIT = 1000


def model_info(model_id: str, display_name: str) -> dict[str, object]:
    """Build one Anthropic SDK-compatible model object with unknown release date."""
    return {
        "id": model_id,
        "display_name": display_name,
        "type": "model",
        "created_at": UNKNOWN_MODEL_CREATED_AT,
    }


def _page_limit(query: Mapping[str, str]) -> int:
    raw_limit = query.get("limit")
    if raw_limit is None:
        return _DEFAULT_PAGE_LIMIT
    try:
        limit = int(raw_limit)
    except ValueError as exc:
        raise ConversionError("Model list limit must be an integer") from exc
    if not _MIN_PAGE_LIMIT <= limit <= _MAX_PAGE_LIMIT:
        raise ConversionError("Model list limit must be between 1 and 1000")
    return limit


def _cursor_index(model_ids: list[str], cursor: str, name: str) -> int:
    try:
        return model_ids.index(cursor)
    except ValueError as exc:
        raise ConversionError(f"Unknown model pagination {name}: {cursor!r}") from exc


def model_page(
    models: Mapping[str, str],
    query: Mapping[str, str],
) -> dict[str, object]:
    """Apply Anthropic cursor pagination to configured model aliases."""
    after_id = query.get("after_id")
    before_id = query.get("before_id")
    if after_id and before_id:
        raise ConversionError("after_id and before_id cannot be used together")

    limit = _page_limit(query)
    model_ids = list(models)
    if before_id:
        end = _cursor_index(model_ids, before_id, "before_id")
        start = max(0, end - limit)
        has_more = start > 0
    else:
        start = _cursor_index(model_ids, after_id, "after_id") + 1 if after_id else 0
        end = min(len(model_ids), start + limit)
        has_more = end < len(model_ids)

    selected_ids = model_ids[start:end]
    return {
        "data": [model_info(model_id, models[model_id]) for model_id in selected_ids],
        "has_more": has_more,
        "first_id": selected_ids[0] if selected_ids else None,
        "last_id": selected_ids[-1] if selected_ids else None,
    }
