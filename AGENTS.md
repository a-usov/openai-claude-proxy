# AGENTS.md

## Project purpose

This repository provides a small, security-conscious compatibility proxy for
Claude Code. Its primary path translates Anthropic Messages API requests into
OpenAI Chat Completions or Responses requests. It also supports transparent
Anthropic and generic OpenAI-compatible pass-through routes.

## Repository map

- `src/openai_claude_proxy/app.py`: FastAPI routes, upstream HTTP lifecycle, and errors.
- `src/openai_claude_proxy/backends.py`: selected-once OpenAI protocol adapters.
- `src/openai_claude_proxy/auth.py`: credential extraction and upstream auth rewriting.
- `src/openai_claude_proxy/config.py`: environment configuration and model mapping.
- `src/openai_claude_proxy/conversion.py`: buffered request/response translation.
- `src/openai_claude_proxy/streaming.py`: SSE parsing and streaming translation.
- `tests/`: mocked-upstream unit and integration tests.
- `README.md`: operator-facing configuration and deployment instructions.

## Required workflow

Use `uv`; do not create `requirements.txt` files or invoke `pip` directly.

```bash
uv sync --locked
uv run ruff format --check .
uv run ruff check .
uv run ty check
uv run pytest -q
uv build
```

Use `uv add <package>` for runtime dependencies and `uv add --dev <package>` for
development dependencies. Commit `pyproject.toml` and `uv.lock` together. Never
edit `uv.lock` manually.

Run `uv run ruff format .` after modifying Python files. All checks above must pass
before handoff. Add focused tests for every protocol, routing, or auth behavior
change; tests must not require a live provider or real credential.

## Implementation invariants

- Never log credentials, authorization headers, complete request bodies, or tool
  results. Error messages must not echo secrets.
- The upstream origin comes only from configuration. Never accept an arbitrary
  upstream URL from a client request.
- Keep forwarded headers allowlisted and strip hop-by-hop headers.
- Always close streamed `httpx.Response` objects, including cancellation and
  error paths.
- Preserve HTTP status, retry metadata, and Anthropic-compatible error envelopes
  where possible.
- Anthropic streams must preserve valid event order: `message_start`, content
  block start/delta/stop events, `message_delta`, then `message_stop`.
- Keep provider-specific behavior behind `Settings`; do not hard-code company
  hostnames, deployment names, tokens, or API versions.
- Reject content that cannot be represented safely instead of silently changing
  its meaning. Document unavoidable lossy translations in `README.md`.
- `apiKeyHelper` credentials may arrive in both `Authorization` and `X-Api-Key`.
  Changes to auth precedence or rewriting require tests for every affected mode.
- Exact model mappings take precedence over glob mappings; `MODEL_OVERRIDE` takes
  precedence over all mappings.
- Preserve request-level reasoning effort dynamically. Do not silently reduce an
  effort level or remove tools to hide an upstream compatibility error.
- Select OpenAI protocol behavior through `backends.py`; do not add protocol-name
  conditionals throughout request routing.

## Style and compatibility

- Support Python 3.11 and newer.
- Use modern type annotations and keep `ty check` clean without blanket ignores.
- Let Ruff own Python formatting and import ordering.
- Prefer small protocol helpers over provider SDK dependencies.
- Keep the public configuration environment-based and update `.env.example` and
  the README whenever a setting is added or changed.
- Avoid breaking existing defaults. If a breaking change is necessary, explain it
  in the pull request and provide a migration example.

## Pull-request expectations

Describe the user-visible behavior, security implications, configuration changes,
and verification performed. Include representative wire-format fixtures for new
provider compatibility. Do not include internal endpoints, tenant identifiers,
credentials, captured work prompts, or proprietary model output in commits.
