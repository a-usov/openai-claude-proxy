"""Credential extraction and allowlisted upstream header rewriting."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

    from .config import Settings


class CredentialError(ValueError):
    """Reject missing or ambiguous helper credentials without exposing them."""


def _authorization_key(value: str) -> str:
    stripped = value.strip()
    if stripped.lower().startswith("bearer "):
        key = stripped[7:].strip()
    elif " " not in stripped:
        key = stripped
    else:
        raise CredentialError("Authorization must contain a Bearer or raw API key")
    if not key:
        raise CredentialError("The helper credential is empty")
    return key


def _incoming_key(headers: Mapping[str, str]) -> str | None:
    credentials: list[str] = []
    if authorization := headers.get("authorization"):
        credentials.append(_authorization_key(authorization))
    if value := headers.get("x-api-key"):
        credentials.append(value.strip())
    if value := headers.get("api-key"):
        credentials.append(value.strip())
    if not credentials:
        return None
    if any(not credential for credential in credentials):
        raise CredentialError("The helper credential is empty")
    if len(set(credentials)) != 1:
        raise CredentialError("Credential headers disagree")
    return credentials[0]


def upstream_headers(
    incoming: Mapping[str, str],
    settings: Settings,
    *,
    json_body: bool = True,
) -> dict[str, str]:
    """Build a small, explicit upstream header set and apply auth rewriting."""
    headers = {"accept": incoming.get("accept", "application/json")}
    if json_body:
        headers["content-type"] = "application/json"
    elif content_type := incoming.get("content-type"):
        headers["content-type"] = content_type
    forward_names = set(settings.forward_headers)
    if settings.upstream_protocol == "anthropic":
        forward_names.update(settings.anthropic_forward_headers)
        forward_names.update(
            name for name in incoming if name.startswith(("anthropic-", "x-claude-code-"))
        )
    else:
        forward_names.update(settings.openai_forward_headers)
    for name in forward_names:
        if value := incoming.get(name):
            headers[name] = value

    mode = settings.auth_mode
    if mode == "passthrough":
        for name in ("authorization", "x-api-key", "api-key"):
            if value := incoming.get(name):
                headers[name] = value
    elif mode != "none":
        key = settings.upstream_api_key if mode == "static" else _incoming_key(incoming)
        if not key:
            raise CredentialError("No helper credential was provided")
        if mode == "bearer":
            headers["authorization"] = f"Bearer {key}"
        elif mode in {"api-key", "x-api-key"}:
            headers[mode] = key
        else:
            prefix = (
                f"{settings.upstream_api_key_scheme} " if settings.upstream_api_key_scheme else ""
            )
            headers[settings.upstream_api_key_header] = f"{prefix}{key}"

    headers.update(settings.static_headers)
    return headers
