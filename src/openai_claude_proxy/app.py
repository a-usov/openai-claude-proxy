"""FastAPI application, upstream lifecycle, and protocol-aware HTTP errors."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from .auth import CredentialError, upstream_headers
from .backends import select_openai_backend
from .config import Settings
from .conversion import (
    ConversionError,
    estimate_anthropic_tokens,
)
from .errors import anthropic_error
from .models import model_info, model_page
from .streaming import (
    anthropic_message_stream,
    passthrough_stream,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable

    from .backends import OpenAIBackend

HTTP_CLIENT_ERROR = 400
UNSUPPORTED_TOKEN_COUNT_STATUSES = frozenset({404, 405, 501})


class RequestBodyTooLargeError(ValueError):
    """Signal that a client body exceeded the configured deployment limit."""


def _error(
    message: str,
    status_code: int = 400,
    error_type: str | None = None,
    request_id: str | None = None,
) -> JSONResponse:
    return JSONResponse(
        anthropic_error(
            message,
            provider_type=error_type,
            status_code=status_code,
            request_id=request_id,
        ),
        status_code=status_code,
    )


def _response_headers(response: httpx.Response) -> dict[str, str]:
    allowed = {
        "request-id",
        "retry-after",
        "retry-after-ms",
        "x-request-id",
        "x-should-retry",
    }
    prefixes = ("anthropic-ratelimit-", "x-ratelimit-")
    return {
        name: value
        for name, value in response.headers.items()
        if name.lower() in allowed or name.lower().startswith(prefixes)
    }


async def _read_request_body(request: Request, max_bytes: int) -> bytes:
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            parsed_content_length = int(content_length)
        except ValueError:
            parsed_content_length = None
        if parsed_content_length is not None and parsed_content_length > max_bytes:
            raise RequestBodyTooLargeError
    body = bytearray()
    async for chunk in request.stream():
        if len(body) + len(chunk) > max_bytes:
            raise RequestBodyTooLargeError
        body.extend(chunk)
    return bytes(body)


async def _read_response_body(response: httpx.Response, max_bytes: int) -> bytes | None:
    body = bytearray()
    async for chunk in response.aiter_bytes():
        if len(body) + len(chunk) > max_bytes:
            return None
        body.extend(chunk)
    return bytes(body)


def _require_response_body(value: bytes | None, message: str) -> bytes:
    if value is None:
        raise ConversionError(message)
    return value


async def _read_json(request: Request, max_bytes: int) -> dict[str, Any]:
    try:
        body = json.loads(await _read_request_body(request, max_bytes))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ConversionError("Request body must be valid JSON") from exc
    if not isinstance(body, dict):
        raise ConversionError("Request body must be a JSON object")
    return body


def _decode_responses_token_count(raw: bytes) -> int:
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ConversionError("Responses input-token result is not valid JSON") from exc
    if not isinstance(decoded, dict):
        raise ConversionError("Responses input-token result must be an object")
    input_tokens = decoded.get("input_tokens")
    if (
        decoded.get("object") != "response.input_tokens"
        or not isinstance(input_tokens, int)
        or isinstance(input_tokens, bool)
        or input_tokens < 0
    ):
        raise ConversionError("Responses input-token result has an invalid schema")
    return input_tokens


def _token_count_fallback(
    settings: Settings,
    payload: dict[str, Any],
    *,
    exact: bool,
) -> JSONResponse:
    if not exact and settings.token_count_fallback == "estimate":  # noqa: S105
        return JSONResponse({"input_tokens": estimate_anthropic_tokens(payload)})
    return _error(
        "Exact token counting is not supported by this OpenAI-compatible upstream",
        501,
    )


async def _count_openai_tokens(
    request: Request,
    settings: Settings,
    backend: OpenAIBackend,
    send_json: Callable[[Request, str, dict[str, Any]], Awaitable[httpx.Response]],
    upstream_error: Callable[[httpx.Response], Awaitable[JSONResponse]],
) -> Response:
    if settings.token_count_mode == "unsupported":  # noqa: S105
        return _error(
            "Token counting is not supported by this OpenAI-compatible upstream",
            501,
        )
    try:
        payload = await _read_json(request, settings.max_request_body_bytes)
        count_body = backend.convert_token_count_request(payload, settings)
    except ConversionError as exc:
        return _error(str(exc))
    if settings.token_count_mode == "estimate":  # noqa: S105
        return JSONResponse({"input_tokens": estimate_anthropic_tokens(payload)})
    if backend.token_count_path is None:
        return _token_count_fallback(
            settings,
            payload,
            exact=settings.token_count_mode == "exact",  # noqa: S105
        )

    upstream = await send_json(request, backend.token_count_path, count_body)
    if upstream.status_code >= HTTP_CLIENT_ERROR:
        if (
            settings.token_count_mode == "auto"  # noqa: S105
            and upstream.status_code in UNSUPPORTED_TOKEN_COUNT_STATUSES
        ):
            await upstream.aclose()
            return _token_count_fallback(settings, payload, exact=False)
        return await upstream_error(upstream)
    try:
        raw = _require_response_body(
            await _read_response_body(upstream, settings.max_response_body_bytes),
            "Responses input-token result exceeded the response size limit",
        )
        input_tokens = _decode_responses_token_count(raw)
    except ConversionError:
        await upstream.aclose()
        return _error("Invalid response from upstream token counter", 502, "api_error")
    headers = _response_headers(upstream)
    await upstream.aclose()
    return JSONResponse({"input_tokens": input_tokens}, headers=headers)


def create_app(
    settings: Settings | None = None,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    """Create a configured proxy application with an optional test transport."""
    configured = settings or Settings.from_env()
    openai_backend = select_openai_backend(configured)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        timeout = httpx.Timeout(configured.request_timeout, connect=configured.connect_timeout)
        app.state.client = httpx.AsyncClient(
            timeout=timeout,
            verify=configured.verify_ssl,
            transport=transport,
            follow_redirects=False,
        )
        yield
        await app.state.client.aclose()

    app = FastAPI(title="OpenAI Claude Proxy", version="0.1.0", lifespan=lifespan)
    app.state.settings = configured

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {
            "status": "ok",
            "upstream_protocol": configured.upstream_protocol,
            "openai_api": configured.openai_api,
        }

    @app.api_route("/api/hello", methods=["GET", "HEAD"])
    async def claude_code_hello() -> Response:
        return Response('{"message": "hello"}', media_type="application/json")

    @app.get("/v1/models")
    async def models(request: Request) -> Response:
        if configured.model_discovery:
            try:
                page = model_page(configured.model_discovery, request.query_params)
            except ConversionError as exc:
                return _error(str(exc))
            return JSONResponse(page)
        return await forward_json(request, "/models")

    @app.get("/v1/models/{model_id}")
    async def retrieve_model(request: Request, model_id: str) -> Response:
        if configured.model_discovery:
            display_name = configured.model_discovery.get(model_id)
            if display_name is None:
                return _error("Model not found", 404)
            return JSONResponse(model_info(model_id, display_name))
        return await forward_json(request, f"/models/{model_id}")

    @app.post("/v1/messages/count_tokens")
    async def count_tokens(request: Request) -> Response:
        if configured.upstream_protocol == "anthropic":
            count_path = configured.upstream_messages_path.rstrip("/") + "/count_tokens"
            return await forward_json(request, count_path)
        return await _count_openai_tokens(
            request,
            configured,
            openai_backend,
            send_json,
            upstream_error,
        )

    @app.post("/v1/messages")
    async def messages(request: Request) -> Response:
        if configured.upstream_protocol == "anthropic":
            return await forward_json(request, configured.upstream_messages_path)
        try:
            anthropic_body = await _read_json(request, configured.max_request_body_bytes)
            openai_body = openai_backend.convert_request(anthropic_body, configured)
        except ConversionError as exc:
            return _error(str(exc))

        upstream = await send_json(request, openai_backend.path, openai_body)
        if upstream.status_code >= HTTP_CLIENT_ERROR:
            return await upstream_error(upstream)
        upstream_content_type = upstream.headers.get("content-type", "").lower()
        if openai_body.get("stream") and upstream_content_type.startswith("text/event-stream"):
            stream = openai_backend.convert_stream(
                upstream,
                str(anthropic_body.get("model", "")),
                configured.stream_ping_interval,
                openai_body,
            )
            return StreamingResponse(
                stream,
                media_type="text/event-stream",
                headers={
                    **_response_headers(upstream),
                    "cache-control": "no-cache",
                    "x-accel-buffering": "no",
                },
            )
        try:
            raw = _require_response_body(
                await _read_response_body(upstream, configured.max_response_body_bytes),
                "Upstream response exceeded the response size limit",
            )
            decoded = json.loads(raw)
            converted = openai_backend.convert_response(
                decoded,
                str(anthropic_body["model"]),
                openai_body,
            )
        except (json.JSONDecodeError, ConversionError, TypeError, AttributeError, KeyError) as exc:
            await upstream.aclose()
            return _error(f"Invalid response from upstream: {exc}", 502, "api_error")
        headers = _response_headers(upstream)
        await upstream.aclose()
        if openai_body.get("stream"):
            return StreamingResponse(
                anthropic_message_stream(converted),
                media_type="text/event-stream",
                headers={
                    **headers,
                    "cache-control": "no-cache",
                    "x-accel-buffering": "no",
                },
            )
        return JSONResponse(converted, headers=headers)

    async def send_json(request: Request, path: str, body: dict[str, Any]) -> httpx.Response:
        headers = upstream_headers(request.headers, configured)
        outbound = request.app.state.client.build_request(
            "POST",
            configured.upstream_url(path),
            headers=headers,
            params=configured.upstream_query,
            json=body,
        )
        try:
            return await request.app.state.client.send(outbound, stream=True)
        except httpx.TimeoutException as exc:
            raise UpstreamError("Upstream request timed out", 504) from exc
        except httpx.HTTPError as exc:
            raise UpstreamError("Could not reach upstream", 502) from exc

    async def upstream_error(upstream: httpx.Response) -> JSONResponse:
        raw = await _read_response_body(upstream, configured.max_error_body_bytes)
        headers = _response_headers(upstream)
        await upstream.aclose()
        message: object = None
        provider_type: object = None
        code: object = None
        request_id: object = upstream.headers.get("request-id") or upstream.headers.get(
            "x-request-id"
        )
        try:
            decoded = json.loads(raw) if raw is not None else None
            detail = decoded.get("error", decoded) if isinstance(decoded, dict) else decoded
            if isinstance(detail, dict):
                message = detail.get("message")
                provider_type = detail.get("type")
                code = detail.get("code")
            if isinstance(decoded, dict):
                request_id = decoded.get("request_id") or request_id
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass
        return JSONResponse(
            anthropic_error(
                message if message is not None else "Upstream request failed",
                provider_type=provider_type,
                code=code,
                status_code=upstream.status_code,
                request_id=request_id,
            ),
            status_code=upstream.status_code,
            headers=headers,
        )

    async def forward_json(request: Request, path: str) -> Response:
        body = await _read_request_body(request, configured.max_request_body_bytes)
        headers = upstream_headers(request.headers, configured, json_body=False)
        forwarded_query = [
            (name, value)
            for name, value in request.query_params.multi_items()
            if name not in configured.upstream_query
        ]
        forwarded_query.extend(configured.upstream_query.items())
        outbound = request.app.state.client.build_request(
            request.method,
            configured.upstream_url(path),
            headers=headers,
            params=forwarded_query,
            content=body,
        )
        try:
            upstream = await request.app.state.client.send(outbound, stream=True)
        except httpx.TimeoutException:
            return _error("Upstream request timed out", 504)
        except httpx.HTTPError:
            return _error("Could not reach upstream", 502)
        headers_out = _response_headers(upstream)
        content_type = upstream.headers.get("content-type", "application/json")
        if content_type.startswith("text/event-stream"):
            return StreamingResponse(
                passthrough_stream(upstream),
                status_code=upstream.status_code,
                headers={
                    **headers_out,
                    "cache-control": "no-cache",
                    "content-type": content_type,
                    "x-accel-buffering": "no",
                },
            )
        content = await _read_response_body(upstream, configured.max_response_body_bytes)
        if content is None:
            await upstream.aclose()
            return _error("Upstream response exceeded the response size limit", 502, "api_error")
        await upstream.aclose()
        return Response(
            content,
            status_code=upstream.status_code,
            headers={**headers_out, "content-type": content_type},
        )

    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
    async def passthrough(request: Request, path: str) -> Response:
        if not configured.passthrough_enabled:
            return _error("Pass-through routes are disabled", 404, "not_found_error")
        incoming_path = "/" + path
        prefix = configured.passthrough_strip_prefix
        if prefix and (incoming_path == prefix or incoming_path.startswith(prefix + "/")):
            incoming_path = incoming_path[len(prefix) :] or "/"
        return await forward_json(request, incoming_path)

    @app.exception_handler(UpstreamError)
    async def handle_upstream_failure(_request: Request, exc: UpstreamError) -> JSONResponse:
        return _error(exc.message, exc.status_code)

    @app.exception_handler(CredentialError)
    async def handle_credential_failure(_request: Request, exc: CredentialError) -> JSONResponse:
        return _error(str(exc), 401, "authentication_error")

    @app.exception_handler(RequestBodyTooLargeError)
    async def handle_large_request(
        _request: Request,
        _exc: RequestBodyTooLargeError,
    ) -> JSONResponse:
        return _error("Request body exceeds the configured size limit", 413)

    return app


class UpstreamError(RuntimeError):
    """Represent a safe client-facing failure to contact an upstream service."""

    def __init__(self, message: str, status_code: int) -> None:
        """Initialize the error with sanitized text and its downstream status."""
        self.message = message
        self.status_code = status_code
        super().__init__(message)
