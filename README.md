# OpenAI Claude Proxy

A small Python proxy that lets Claude Code use an OpenAI-compatible service. It
accepts Anthropic Messages API requests and translates messages, images, tools,
tool results, structured outputs, reasoning effort, errors, token usage, and SSE
streams to either OpenAI Chat Completions or Responses. It can also transparently
proxy Anthropic APIs and other OpenAI-compatible routes.

This is intended for internal gateways and Azure/Microsoft Foundry deployments as
well as the public OpenAI or Anthropic APIs. It uses raw HTTP rather than a
provider SDK, so the upstream only needs to implement the relevant wire protocol.

## Quick start with `uv`

```bash
uv sync --locked

export UPSTREAM_BASE_URL="https://gateway.example.com/openai/v1"
export UPSTREAM_PROTOCOL="openai"
export AUTH_MODE="passthrough"

uv run openai-claude-proxy
```

Point Claude Code at the local proxy:

```bash
export ANTHROPIC_BASE_URL="http://127.0.0.1:8080"
claude
```

The health endpoint is `GET /health`.

Copy `.env.example` for a complete, secret-free configuration template. See
`CONTRIBUTING.md` for the development workflow, `SECURITY.md` for deployment and
reporting guidance, and `AGENTS.md` for repository conventions used by coding
agents.

## Claude Code and `apiKeyHelper`

Configure the helper in your user or managed Claude Code settings. For example:

```json
{
  "apiKeyHelper": "/absolute/path/to/get-work-token.sh"
}
```

The script must print only the credential. Claude Code sends helper output in both
`X-Api-Key` and `Authorization: Bearer` on model requests. With the default
`AUTH_MODE=passthrough`, the proxy preserves both headers. To normalize the
credential for an upstream, choose one of:

```bash
# Send only Authorization: Bearer <helper output>
export AUTH_MODE=bearer

# Send only api-key: <helper output> (common for Azure key auth)
export AUTH_MODE=api-key

# Send only x-api-key: <helper output>
export AUTH_MODE=x-api-key
```

The proxy never logs or returns credentials. It forwards only explicitly allowed
metadata headers plus the selected authentication headers. Rewrite modes accept a
helper credential from `Authorization`, `X-Api-Key`, or `Api-Key`; if multiple
credential headers are present they must resolve to the same value, otherwise the
proxy returns a sanitized HTTP 401 before contacting the upstream. `passthrough`
intentionally preserves disagreeing headers, `none` strips them, and `static`
ignores them in favor of configured credentials. `UPSTREAM_HEADERS_JSON` has final
precedence for enterprise gateways that require a static auth header.

## Microsoft Foundry examples

For the current OpenAI-compatible v1 endpoint:

```bash
export UPSTREAM_BASE_URL="https://YOUR-RESOURCE.openai.azure.com/openai/v1"
export UPSTREAM_PROTOCOL=openai
export AUTH_MODE=api-key
export MODEL_OVERRIDE="YOUR-DEPLOYMENT-NAME"
uv run openai-claude-proxy
```

To use the newer Responses protocol, add:

```bash
export OPENAI_API=responses
```

The default remains Chat Completions for compatibility with existing gateways.

If your company gateway accepts the helper token as bearer auth instead, use
`AUTH_MODE=bearer`. If it expects the headers exactly as Claude Code sends them,
use `AUTH_MODE=passthrough`.

Older Azure deployment URLs can be represented by putting the deployment in the
base URL and changing the completion path:

```bash
export UPSTREAM_BASE_URL="https://RESOURCE.openai.azure.com/openai/deployments/DEPLOYMENT"
export UPSTREAM_CHAT_PATH="/chat/completions"
export UPSTREAM_QUERY="api-version=2024-10-21"
export AUTH_MODE=api-key
```

For an Azure AI Model Inference-style endpoint whose route is
`/models/chat/completions`, use a base URL ending in `/models`.

## Configuration

All configuration is via environment variables.

| Variable | Default | Purpose |
| --- | --- | --- |
| `UPSTREAM_BASE_URL` | `https://api.openai.com/v1` | Fixed upstream base URL |
| `UPSTREAM_PROTOCOL` | `openai` | `openai` translates Messages; `anthropic` passes them through |
| `OPENAI_API` | `chat_completions` | Translation backend: `chat_completions` or `responses` |
| `UPSTREAM_CHAT_PATH` | `/chat/completions` | OpenAI Chat Completions path relative to the base |
| `UPSTREAM_RESPONSES_PATH` | `/responses` | OpenAI Responses path relative to the base |
| `UPSTREAM_RESPONSES_INPUT_TOKENS_PATH` | `/responses/input_tokens` | Responses exact input-token count path relative to the base |
| `UPSTREAM_MESSAGES_PATH` | `/messages` | Anthropic Messages path relative to the base |
| `AUTH_MODE` | `passthrough` | `passthrough`, `bearer`, `api-key`, `x-api-key`, `static`, or `none` |
| `UPSTREAM_API_KEY` | unset | Credential used only by `AUTH_MODE=static` |
| `UPSTREAM_API_KEY_HEADER` | `authorization` | Header used by static auth |
| `UPSTREAM_API_KEY_SCHEME` | `Bearer` | Static auth prefix; set empty for a raw value |
| `MODEL_OVERRIDE` | unset | Replace every client model with this deployment/model name |
| `MODEL_MAP` | `{}` | JSON exact/glob model mapping, evaluated in insertion order |
| `MODEL_DISCOVERY` | `{}` | Static JSON map of Claude-visible model IDs to display names |
| `MAX_TOKENS_FIELD` | `max_tokens` | Use `max_completion_tokens` for upstreams that require the newer name |
| `REASONING_EFFORT_ENABLED` | `true` | Translate Claude request effort into OpenAI reasoning effort |
| `REASONING_EFFORT_MAP` | identity map | JSON overrides for individual effort levels; `null` omits a level |
| `UPSTREAM_QUERY` | unset | URL-encoded query parameters appended to upstream calls |
| `UPSTREAM_QUERY_JSON` | `{}` | Query parameters as JSON; overrides duplicate URL-encoded keys |
| `UPSTREAM_HEADERS_JSON` | `{}` | Static headers as JSON; these have final precedence |
| `FORWARD_HEADERS` | see source | Common non-auth request headers to preserve for either provider |
| `OPENAI_FORWARD_HEADERS` | OpenAI organization/project and idempotency | Additional headers for translated and pass-through OpenAI routes |
| `ANTHROPIC_FORWARD_HEADERS` | Anthropic version/beta/workspace and idempotency | Additional headers for Anthropic pass-through routes |
| `EXTRA_OPENAI_BODY` | `{}` | JSON fields merged last into translated OpenAI requests |
| `REQUEST_TIMEOUT` | `600` | Overall/read timeout in seconds |
| `CONNECT_TIMEOUT` | `10` | Connection timeout in seconds |
| `MAX_REQUEST_BODY_BYTES` | `33554432` | Maximum buffered client request body (32 MiB) |
| `MAX_RESPONSE_BODY_BYTES` | `33554432` | Maximum buffered successful upstream response (32 MiB) |
| `MAX_ERROR_BODY_BYTES` | `65536` | Maximum upstream error body inspected for normalization (64 KiB) |
| `STREAM_PING_INTERVAL` | `15` | Seconds of upstream silence before emitting an Anthropic `ping`; `0` disables it |
| `VERIFY_SSL` | `true` | Verify the upstream TLS certificate |
| `PASSTHROUGH_ENABLED` | `true` | Forward routes other than the explicit Messages routes |
| `PASSTHROUGH_STRIP_PREFIX` | `/v1` | Prefix removed before joining a pass-through path to the base URL |
| `TOKEN_COUNT_MODE` | `auto` | `auto`, `exact`, `estimate`, or `unsupported` for OpenAI upstreams |
| `TOKEN_COUNT_FALLBACK` | `unsupported` | In `auto` mode, use `estimate` or return unsupported when an exact endpoint is unavailable |
| `HOST` / `PORT` | `127.0.0.1` / `8080` | Listen address |

Model mapping example:

```bash
export MODEL_MAP='{
  "claude-opus-*": "work-reasoning-deployment",
  "claude-sonnet-*": "work-coding-deployment",
  "claude-haiku-*": "work-fast-deployment"
}'
```

With neither setting, the client-selected model passes through unchanged.
`MODEL_OVERRIDE` takes precedence over `MODEL_MAP`; the legacy `DEFAULT_MODEL`
environment name remains an alias. Exact map keys take precedence over glob patterns.

### Model and reasoning mapping

Claude Code can keep using its recognized Anthropic model choices while the proxy
routes those choices to differently named deployments:

```bash
export MODEL_MAP='{
  "claude-opus-5*": "gpt-5.6-sol",
  "claude-sonnet-4-6*": "gpt-5.6-terra",
  "claude-haiku-*": "gpt-5.6-luna"
}'
```

The proxy returns the Claude-visible alias in its Anthropic response while sending
the mapped deployment upstream. When Claude sends `output_config.effort`, the proxy
dynamically sends Chat Completions `reasoning_effort` or Responses
`reasoning.effort`. The Anthropic-standard values are `low`, `medium`, `high`,
`xhigh`, and `max`; null or an omitted value sends no effort. A legacy top-level
`effort` is accepted as a Claude Code compatibility extension and additionally
accepts `none`. The default identity map covers all six values. If an upstream
supports fewer levels, override only the differences:

```bash
# Cap Anthropic max/xhigh choices at the highest level this deployment accepts.
export REASONING_EFFORT_MAP='{"xhigh":"high","max":"high"}'
```

An override value of `null` omits effort for that source level. Set
`REASONING_EFFORT_ENABLED=false` to disable dynamic translation entirely. A
matching value supplied through `EXTRA_OPENAI_BODY` deliberately wins over the
dynamic value. For Chat-only gateways that require it, set
`MAX_TOKENS_FIELD=max_completion_tokens`; Responses always uses
`max_output_tokens`.

OpenAI recommends the Responses API for GPT-5.6 reasoning, tool-calling, and
multi-turn workflows. Select it with `OPENAI_API=responses`. Requests default to
`store=false`, avoiding server-side response retention unless explicitly overridden
with `EXTRA_OPENAI_BODY`. For stateless conversations, the proxy encodes the exact
ordered `response.output` array in a proxy-marked Anthropic `redacted_thinking`
block. Claude Code returns that block with later history, allowing the proxy to
replay provider message IDs, statuses, annotations, reasoning, and function-call
metadata without process-local conversation storage. Manually authored Anthropic
assistant text uses schema-valid Responses easy-input messages instead.

Visible text and tool calls remain normal Anthropic blocks. On replay, the proxy
validates those blocks against the carried output and sends each original OpenAI
output item only once. This works across proxy workers and restarts; clients must
preserve response content blocks. The carrier is base64-encoded transport data, not
encryption, and can contain copies of visible output text and tool arguments. Treat
the Claude transcript as sensitive. Only proxy-marked blocks are decoded, and
malformed or mismatched proxy state is rejected without including its contents in
the error. Complete carriers are bound to the mapped upstream model; older unbound
carrier formats and carriers from another model are rejected rather than replayed
unsafely. Verify which effort levels and function-tool features your Foundry
deployment or company gateway supports; incompatibilities are returned to Claude
Code rather than silently weakening the request. See OpenAI's
[Responses migration guide](https://developers.openai.com/api/docs/guides/migrate-to-responses).

### Model discovery

Claude Code filters gateway discovery results to IDs beginning with `claude` or
`anthropic`. Publish Claude-shaped aliases and map those aliases to upstream model
or deployment names:

```bash
export MODEL_DISCOVERY='{
  "claude-gpt-5-6-sol": "GPT-5.6 Sol",
  "claude-gpt-5-6-terra": "GPT-5.6 Terra",
  "claude-gpt-5-6-luna": "GPT-5.6 Luna"
}'
export MODEL_MAP='{
  "claude-gpt-5-6-sol": "gpt-5.6-sol",
  "claude-gpt-5-6-terra": "gpt-5.6-terra",
  "claude-gpt-5-6-luna": "gpt-5.6-luna"
}'
export CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY=1
```

When `MODEL_DISCOVERY` is empty, `GET /v1/models` continues to pass through to
the configured upstream without normalizing its OpenAI response shape. When static
discovery is configured, list and retrieve responses follow Anthropic's Models API:
entries include `type: "model"`, unknown release dates use the Unix epoch, and
`limit`, `after_id`, and `before_id` cursor pagination are supported. Discovery does
not expose credentials or upstream URLs. Every published alias must be covered by
`MODEL_OVERRIDE` or an exact/glob `MODEL_MAP` entry, preventing a display-only alias
from being sent accidentally as an upstream model ID. To declare intentional
pass-through, add an explicit identity entry such as
`{"claude-upstream-id":"claude-upstream-id"}`.

Static auth is useful when clients must not supply upstream credentials:

```bash
export AUTH_MODE=static
export UPSTREAM_API_KEY="..."
export UPSTREAM_API_KEY_HEADER="api-key"
export UPSTREAM_API_KEY_SCHEME=""
```

Prefer injecting `UPSTREAM_API_KEY` from a secrets manager rather than an `.env`
file or image layer.

## Other upstream modes

To proxy an Anthropic-compatible service without translation:

```bash
export UPSTREAM_BASE_URL="https://api.anthropic.com/v1"
export UPSTREAM_PROTOCOL=anthropic
export AUTH_MODE=passthrough
```

`POST /v1/messages` and `POST /v1/messages/count_tokens` are then relayed as-is,
including streaming responses. The generic pass-through route also makes calls
such as `GET /v1/models` or `POST /v1/embeddings` available to OpenAI-compatible
clients. The upstream host is fixed by configuration; request paths cannot choose
another host.

## Translation coverage and limitations

Supported Anthropic inputs include top-level and mid-conversation system prompts,
text, base64/URL images, multiple and strict function tools, tool results,
structured JSON output, reasoning effort, `metadata.user_id` attribution, sampling
settings, prompt-cache breakpoints, and streaming. Mid-conversation Anthropic
`system` roles become ordered OpenAI `developer` messages. Metadata user IDs become
OpenAI `safety_identifier`; they are never logged. Responses also preserves text
and image tool-result content and maps deferred tools plus direct/programmatic tool
caller restrictions. Chat Completions supports text-only tool results. OpenAI
content and function calls are translated back into Anthropic content blocks and
correctly ordered event streams.
During an upstream pause, the proxy emits Anthropic `ping` events so Claude Code's
stream watchdog remains active. Claude Code attribution headers are forwarded;
Anthropic pass-through additionally preserves future `anthropic-*` and
`x-claude-code-*` headers regardless of custom forwarding lists, so required
Anthropic version/beta headers cannot be accidentally removed. Translated OpenAI
routes do not receive Anthropic headers unless they are explicitly added to
`OPENAI_FORWARD_HEADERS` or static headers. Authentication, cookie, host,
content-framing, and hop-by-hop names are rejected in configurable forwarding
lists rather than bypassing the dedicated policies.

Translated failures use Anthropic's standard error discriminators even when an
OpenAI-compatible service returns provider-specific values. Buffered errors retain
their upstream HTTP status, safe retry/rate-limit headers, and optional request ID.
The same allowlisted metadata is preserved on successful buffered, streamed, and
pass-through responses; unrelated headers such as cookies remain stripped. Provider
messages are whitespace-normalized and bounded; transport exception details are not
returned to the client. URL values, bearer credentials, common secret assignments,
and labeled prompts/request bodies/tool results are redacted from structured
provider messages and request IDs. Non-JSON and oversized upstream errors become a
generic message. Anthropic pass-through bodies remain unmodified when they fit the
configured deployment limits.
OpenAI refusal text remains visible and is accompanied by Anthropic
`stop_reason: "refusal"` and structured `stop_details` in buffered and streamed
responses.
Function-call arguments must be empty or a JSON object. Invalid JSON and scalar or
array values terminate translation with a sanitized protocol error; the proxy never
rewrites them into a different object. In a stream, an error terminates the open
tool block without emitting a misleading `content_block_stop`.
Responses argument/text/refusal `done` events recover deltas omitted by compatible
gateways and are checked against any pieces already emitted. Chat tool arguments
stream immediately when `parallel_tool_calls=false`; parallel calls remain buffered
to preserve valid Anthropic block ordering. Malformed JSON, unknown typed Responses
events, and upstream disconnects terminate with a sanitized Anthropic stream error.
Downstream cancellation cancels an outstanding upstream read, and every terminal,
error, disconnect, and cancellation path closes the streamed upstream response.
Client requests, buffered success responses, and inspected error bodies have
separate configurable byte limits. Oversized client input returns HTTP 413 and an
oversized buffered upstream result returns a sanitized HTTP 502. SSE response
streams remain incremental and are not accumulated merely to impose a total-size
limit.

Chat Completions `stop` maps to Anthropic `end_turn` with no `stop_sequence` because
the OpenAI response does not distinguish a natural stop from a matched custom
sequence. Its `length` value maps to `max_tokens` because it likewise does not
distinguish output limits from context exhaustion. Responses maps standard output
limits and content filtering explicitly, and maps context-window exhaustion when a
compatible gateway supplies that reason. Failed, cancelled, queued, unknown, or
unterminated results produce protocol errors rather than a false `end_turn`.

There are unavoidable semantic differences between providers:

- Anthropic `cache_control` on translatable text, image, and tool-result content
  becomes OpenAI `prompt_cache_breakpoint: {"mode":"explicit"}`. Chat system
  content supports the same mapping. A top-level marker is first applied to the
  last cacheable Anthropic block. OpenAI currently exposes one request-wide `30m`
  lifetime, so Anthropic's per-marker `5m` and `1h` TTLs cannot be preserved
  exactly. Cache-marked Anthropic system arrays become Responses developer input
  messages so their text-block breakpoints remain in the same prompt position.
  Responses function calls and function-tool schemas do not accept breakpoints in
  the reviewed OpenAI SDK; those locations are rejected instead of silently moved.
  Reported cache reads and writes are mapped into Anthropic usage fields. See
  OpenAI's [prompt-caching guide](https://developers.openai.com/api/docs/guides/prompt-caching).
- Anthropic adaptive and disabled `thinking` modes are accepted while explicit
  `output_config.effort` continues to control OpenAI reasoning effort. The
  `thinking.display` setting needs no OpenAI field because translated responses do
  not expose raw reasoning. Fixed `thinking.type: enabled` token budgets have no
  equivalent effort level and return HTTP 400 instead of being guessed. Claude
  Code's `clear_thinking_20251015` context edit with `keep: all` is accepted as a
  no-op; context edits that actually remove history remain unsupported.
- Other context-management strategies, service-tier, container, inference-geo,
  top-k, user-profile attribution, hosted/server tools, eager tool-input streaming,
  and tool input examples have no sufficiently close portable meaning and return
  HTTP 400 on translated routes. Anthropic pass-through sends these fields and
  future fields without rewriting the JSON body.
- In the default `auto` mode, the Responses backend converts the full count request
  and calls `/responses/input_tokens`, preserving upstream errors, request IDs, and
  retry metadata. Chat Completions has no exact count endpoint, and compatible
  gateways may omit the Responses endpoint, so the default returns HTTP 501 and
  lets Claude Code use its local fallback. Set `TOKEN_COUNT_FALLBACK=estimate` to
  retain the old byte-based fallback in auto mode, or set
  `TOKEN_COUNT_MODE=estimate` to avoid all upstream count calls. `exact` requires
  the configured Responses endpoint; `unsupported` disables counting explicitly.
- Provider-specific hosted tools, citations, and PDF/document blocks are not
  translated. Proxy-marked OpenAI Responses continuation state is the exception:
  complete output-item arrays and provider-encrypted reasoning items are transported
  through `redacted_thinking` blocks. Other encrypted or signed reasoning formats
  are not translated. A non-representable input returns HTTP 400 instead of being
  silently corrupted.
- A final Anthropic assistant message requests assistant-prefill continuation.
  Neither reviewed OpenAI endpoint preserves that behavior, so translated routes
  reject it. Assistant history followed by a user or system turn remains supported.
- Responses preserves image tool results. Chat Completions only permits text in a
  tool message, so image, document, citation, search-result, and tool-reference
  results are rejected there rather than replaced with placeholder text.
- Anthropic `stop_sequences` map to Chat Completions `stop`; Responses has no direct
  equivalent, so selecting Responses rejects requests that contain them.
- Neither protocol returns a portable monetary cost. The proxy maps token and cache
  usage, but pricing remains provider/deployment configuration outside the wire API.

Because Claude Code depends heavily on tool calling, confirm that the selected
upstream model supports parallel function calls and follows JSON schemas reliably.
Representative sanitized multi-turn bodies are collected in
[`docs/wire-fixtures.md`](docs/wire-fixtures.md).

## Testing provider gateways

The required test suite is credential-free and excludes tests marked `live`. To run
the opt-in live checks, configure one or more dialect-specific targets and select the
marker explicitly:

```bash
export LIVE_RESPONSES_BASE_URL="https://gateway.example.com/openai/v1"
export LIVE_RESPONSES_MODEL="deployment-name"
export LIVE_RESPONSES_API_KEY="..."
export LIVE_RESPONSES_AUTH_MODE="bearer"
export LIVE_RESPONSES_UPSTREAM_QUERY="api-version=2026-01-01"

uv run pytest -m live -q
```

Use the same suffixes with `LIVE_CHAT_` or `LIVE_ANTHROPIC_`. OpenAI dialects
default to `AUTH_MODE=bearer`; Anthropic defaults to `passthrough`. Set a target's
`PROBES=1` variable, such as `LIVE_RESPONSES_PROBES=1`, to additionally probe
reasoning-effort levels, structured output, prompt-cache usage, model mapping, and
request attribution.

Live tests send only fixed synthetic prompts and tool results, cap each response at
64 output tokens, use a 60-second upstream timeout, and make a small bounded number
of calls. They never print credentials, complete request bodies, tool results, or
model output. Failures identify only the configured dialect, model, operation, and
HTTP status. Keep live tests out of ordinary CI unless they run in a dedicated,
secret-scoped environment with an explicit cost budget.

## Docker

```bash
docker build -t openai-claude-proxy .
docker run --rm -p 127.0.0.1:8080:8080 \
  -e UPSTREAM_BASE_URL="https://gateway.example.com/openai/v1" \
  -e AUTH_MODE=passthrough \
  openai-claude-proxy
```

## Development

```bash
uv sync --locked
uv run ruff format --check .
uv run ruff check .
uv run ty check
uv run pytest -q
uv build
```

The pytest suite includes the pinned OpenAI and Anthropic SDK conformance checks and
real loopback HTTP coverage. No live provider credential is needed; provider
upstreams are mocked. See [`docs/sdk-conformance.md`](docs/sdk-conformance.md) for
the reviewed SDK revisions and update policy.
