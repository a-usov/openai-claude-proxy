"""Environment-backed configuration and deterministic model mapping."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from fnmatch import fnmatchcase
from typing import TYPE_CHECKING
from urllib.parse import parse_qsl

if TYPE_CHECKING:
    from collections.abc import Mapping

DEFAULT_REASONING_EFFORT_MAP: dict[str, str | None] = {
    "none": "none",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "xhigh",
    "max": "max",
}
FORBIDDEN_FORWARD_HEADERS = frozenset(
    {
        "accept",
        "api-key",
        "authorization",
        "connection",
        "content-length",
        "content-type",
        "cookie",
        "host",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "set-cookie",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
        "x-api-key",
    },
)


class ConfigError(ValueError):
    """Raised when proxy configuration is invalid."""


def _boolean(value: str | None, *, default: bool) -> bool:
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ConfigError(f"Expected a boolean value, got {value!r}")


def _json_object(value: str | None, name: str) -> dict[str, object]:
    if not value:
        return {}
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{name} must be a JSON object: {exc}") from exc
    if not isinstance(decoded, dict):
        raise ConfigError(f"{name} must be a JSON object")
    return decoded


def _csv(value: str | None, default: tuple[str, ...]) -> tuple[str, ...]:
    if value is None:
        return default
    return tuple(part.strip().lower() for part in value.split(",") if part.strip())


def _reasoning_effort_map(value: str | None) -> dict[str, str | None]:
    result = DEFAULT_REASONING_EFFORT_MAP.copy()
    if value is None:
        return result
    configured = _json_object(value, "REASONING_EFFORT_MAP")
    for source, target in configured.items():
        if not isinstance(source, str) or not source:
            raise ConfigError("REASONING_EFFORT_MAP keys must be non-empty strings")
        if target is not None and not isinstance(target, str):
            raise ConfigError("REASONING_EFFORT_MAP values must be strings or null")
        result[source] = target
    return result


@dataclass(slots=True)
class Settings:
    """Hold validated proxy settings loaded directly or from the environment."""

    upstream_base_url: str = "https://api.openai.com/v1"
    upstream_protocol: str = "openai"
    openai_api: str = "chat_completions"
    upstream_chat_path: str = "/chat/completions"
    upstream_responses_path: str = "/responses"
    upstream_responses_input_tokens_path: str = "/responses/input_tokens"
    upstream_messages_path: str = "/messages"
    auth_mode: str = "passthrough"
    upstream_api_key: str | None = None
    upstream_api_key_header: str = "authorization"
    upstream_api_key_scheme: str = "Bearer"
    forward_headers: tuple[str, ...] = (
        "x-client-request-id",
        "x-claude-code-session-id",
        "x-claude-code-agent-id",
        "x-claude-code-parent-agent-id",
        "x-request-id",
        "x-correlation-id",
        "traceparent",
        "tracestate",
    )
    openai_forward_headers: tuple[str, ...] = (
        "idempotency-key",
        "openai-organization",
        "openai-project",
    )
    anthropic_forward_headers: tuple[str, ...] = (
        "anthropic-beta",
        "anthropic-version",
        "anthropic-workspace-id",
        "idempotency-key",
    )
    static_headers: dict[str, str] = field(default_factory=dict)
    upstream_query: dict[str, str] = field(default_factory=dict)
    model_map: dict[str, str] = field(default_factory=dict)
    model_discovery: dict[str, str] = field(default_factory=dict)
    model_override: str | None = None
    max_tokens_field: str = "max_tokens"
    reasoning_effort_enabled: bool = True
    reasoning_effort_map: dict[str, str | None] = field(
        default_factory=DEFAULT_REASONING_EFFORT_MAP.copy,
    )
    extra_openai_body: dict[str, object] = field(default_factory=dict)
    request_timeout: float = 600.0
    connect_timeout: float = 10.0
    max_request_body_bytes: int = 32 * 1024 * 1024
    max_response_body_bytes: int = 32 * 1024 * 1024
    max_error_body_bytes: int = 64 * 1024
    stream_ping_interval: float = 15.0
    verify_ssl: bool = True
    passthrough_enabled: bool = True
    passthrough_strip_prefix: str = "/v1"
    token_count_mode: str = "auto"  # noqa: S105
    token_count_fallback: str = "unsupported"  # noqa: S105
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        """Normalize and validate configuration after initialization."""
        self.upstream_base_url = self.upstream_base_url.rstrip("/")
        self.upstream_protocol = self.upstream_protocol.lower()
        self.openai_api = self.openai_api.lower()
        self.auth_mode = self.auth_mode.lower()
        self.upstream_api_key_header = self.upstream_api_key_header.lower()
        self.passthrough_strip_prefix = self.passthrough_strip_prefix.rstrip("/")
        self.forward_headers = tuple(name.lower() for name in self.forward_headers)
        self.openai_forward_headers = tuple(name.lower() for name in self.openai_forward_headers)
        self.anthropic_forward_headers = tuple(
            name.lower() for name in self.anthropic_forward_headers
        )

        if self.upstream_protocol not in {"openai", "anthropic"}:
            raise ConfigError("UPSTREAM_PROTOCOL must be 'openai' or 'anthropic'")
        if self.openai_api not in {"chat_completions", "responses"}:
            raise ConfigError("OPENAI_API must be 'chat_completions' or 'responses'")
        if self.auth_mode not in {
            "passthrough",
            "bearer",
            "api-key",
            "x-api-key",
            "static",
            "none",
        }:
            raise ConfigError(
                "AUTH_MODE must be passthrough, bearer, api-key, x-api-key, static, or none",
            )
        if self.auth_mode == "static" and not self.upstream_api_key:
            raise ConfigError("UPSTREAM_API_KEY is required when AUTH_MODE=static")
        if self.token_count_mode not in {"auto", "exact", "estimate", "unsupported"}:
            raise ConfigError(
                "TOKEN_COUNT_MODE must be 'auto', 'exact', 'estimate', or 'unsupported'",
            )
        if self.token_count_fallback not in {"estimate", "unsupported"}:
            raise ConfigError("TOKEN_COUNT_FALLBACK must be 'estimate' or 'unsupported'")
        if self.max_tokens_field not in {"max_tokens", "max_completion_tokens"}:
            raise ConfigError("MAX_TOKENS_FIELD must be 'max_tokens' or 'max_completion_tokens'")
        if self.stream_ping_interval < 0:
            raise ConfigError("STREAM_PING_INTERVAL must be zero or greater")
        for setting_name, size in (
            ("MAX_REQUEST_BODY_BYTES", self.max_request_body_bytes),
            ("MAX_RESPONSE_BODY_BYTES", self.max_response_body_bytes),
            ("MAX_ERROR_BODY_BYTES", self.max_error_body_bytes),
        ):
            if size <= 0:
                raise ConfigError(f"{setting_name} must be a positive integer")
        for policy_name, names in (
            ("FORWARD_HEADERS", self.forward_headers),
            ("OPENAI_FORWARD_HEADERS", self.openai_forward_headers),
            ("ANTHROPIC_FORWARD_HEADERS", self.anthropic_forward_headers),
        ):
            for name in names:
                if (
                    not name
                    or name in FORBIDDEN_FORWARD_HEADERS
                    or any(character.isspace() or character == ":" for character in name)
                ):
                    raise ConfigError(f"{policy_name} contains forbidden header {name!r}")
        for model_id, display_name in self.model_discovery.items():
            if not model_id or not display_name:
                raise ConfigError("MODEL_DISCOVERY keys and values must be non-empty strings")
            if not model_id.lower().startswith(("claude", "anthropic")):
                raise ConfigError(
                    "MODEL_DISCOVERY IDs must begin with 'claude' or 'anthropic' for Claude Code",
                )
            if not self.model_override and not any(
                model_id == pattern or fnmatchcase(model_id, pattern) for pattern in self.model_map
            ):
                raise ConfigError(
                    f"MODEL_DISCOVERY alias {model_id!r} must match MODEL_MAP or MODEL_OVERRIDE",
                )
        for source, target in self.reasoning_effort_map.items():
            if not isinstance(source, str) or not source:
                raise ConfigError("REASONING_EFFORT_MAP keys must be non-empty strings")
            if target is not None and (not isinstance(target, str) or not target):
                raise ConfigError("REASONING_EFFORT_MAP values must be non-empty strings or null")

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> Settings:
        """Build settings from an environment mapping without mutating it."""
        env = os.environ if environ is None else environ
        defaults = cls()
        query = dict(parse_qsl(env.get("UPSTREAM_QUERY", ""), keep_blank_values=True))
        query.update(
            {
                str(k): str(v)
                for k, v in _json_object(
                    env.get("UPSTREAM_QUERY_JSON"),
                    "UPSTREAM_QUERY_JSON",
                ).items()
            },
        )
        static_headers = {
            str(k).lower(): str(v)
            for k, v in _json_object(
                env.get("UPSTREAM_HEADERS_JSON"),
                "UPSTREAM_HEADERS_JSON",
            ).items()
        }
        model_map = {
            str(k): str(v) for k, v in _json_object(env.get("MODEL_MAP"), "MODEL_MAP").items()
        }
        model_discovery = {
            str(k): str(v)
            for k, v in _json_object(
                env.get("MODEL_DISCOVERY"),
                "MODEL_DISCOVERY",
            ).items()
        }
        return cls(
            upstream_base_url=env.get("UPSTREAM_BASE_URL", defaults.upstream_base_url),
            upstream_protocol=env.get("UPSTREAM_PROTOCOL", defaults.upstream_protocol),
            openai_api=env.get("OPENAI_API", defaults.openai_api),
            upstream_chat_path=env.get("UPSTREAM_CHAT_PATH", defaults.upstream_chat_path),
            upstream_responses_path=env.get(
                "UPSTREAM_RESPONSES_PATH",
                defaults.upstream_responses_path,
            ),
            upstream_responses_input_tokens_path=env.get(
                "UPSTREAM_RESPONSES_INPUT_TOKENS_PATH",
                defaults.upstream_responses_input_tokens_path,
            ),
            upstream_messages_path=env.get(
                "UPSTREAM_MESSAGES_PATH",
                defaults.upstream_messages_path,
            ),
            auth_mode=env.get("AUTH_MODE", defaults.auth_mode),
            upstream_api_key=env.get("UPSTREAM_API_KEY"),
            upstream_api_key_header=env.get(
                "UPSTREAM_API_KEY_HEADER",
                defaults.upstream_api_key_header,
            ),
            upstream_api_key_scheme=env.get(
                "UPSTREAM_API_KEY_SCHEME",
                defaults.upstream_api_key_scheme,
            ),
            forward_headers=_csv(env.get("FORWARD_HEADERS"), defaults.forward_headers),
            openai_forward_headers=_csv(
                env.get("OPENAI_FORWARD_HEADERS"),
                defaults.openai_forward_headers,
            ),
            anthropic_forward_headers=_csv(
                env.get("ANTHROPIC_FORWARD_HEADERS"),
                defaults.anthropic_forward_headers,
            ),
            static_headers=static_headers,
            upstream_query=query,
            model_map=model_map,
            model_discovery=model_discovery,
            model_override=env.get("MODEL_OVERRIDE") or env.get("DEFAULT_MODEL"),
            max_tokens_field=env.get("MAX_TOKENS_FIELD", defaults.max_tokens_field),
            reasoning_effort_enabled=_boolean(
                env.get("REASONING_EFFORT_ENABLED"),
                default=defaults.reasoning_effort_enabled,
            ),
            reasoning_effort_map=_reasoning_effort_map(env.get("REASONING_EFFORT_MAP")),
            extra_openai_body=_json_object(env.get("EXTRA_OPENAI_BODY"), "EXTRA_OPENAI_BODY"),
            request_timeout=float(env.get("REQUEST_TIMEOUT", defaults.request_timeout)),
            connect_timeout=float(env.get("CONNECT_TIMEOUT", defaults.connect_timeout)),
            max_request_body_bytes=int(
                env.get("MAX_REQUEST_BODY_BYTES", defaults.max_request_body_bytes),
            ),
            max_response_body_bytes=int(
                env.get("MAX_RESPONSE_BODY_BYTES", defaults.max_response_body_bytes),
            ),
            max_error_body_bytes=int(
                env.get("MAX_ERROR_BODY_BYTES", defaults.max_error_body_bytes),
            ),
            stream_ping_interval=float(
                env.get("STREAM_PING_INTERVAL", defaults.stream_ping_interval),
            ),
            verify_ssl=_boolean(env.get("VERIFY_SSL"), default=defaults.verify_ssl),
            passthrough_enabled=_boolean(
                env.get("PASSTHROUGH_ENABLED"),
                default=defaults.passthrough_enabled,
            ),
            passthrough_strip_prefix=env.get(
                "PASSTHROUGH_STRIP_PREFIX",
                defaults.passthrough_strip_prefix,
            ),
            token_count_mode=env.get("TOKEN_COUNT_MODE", defaults.token_count_mode),
            token_count_fallback=env.get(
                "TOKEN_COUNT_FALLBACK",
                defaults.token_count_fallback,
            ),
            log_level=env.get("LOG_LEVEL", defaults.log_level),
        )

    def map_model(self, requested: str) -> str:
        """Map a client model using override, exact, glob, then pass-through order."""
        if self.model_override:
            return self.model_override
        if requested in self.model_map:
            return self.model_map[requested]
        for pattern, replacement in self.model_map.items():
            if fnmatchcase(requested, pattern):
                return replacement
        return requested

    def upstream_url(self, path: str) -> str:
        """Join a configured relative API path to the fixed upstream origin."""
        return f"{self.upstream_base_url}/{path.lstrip('/')}"
