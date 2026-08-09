# Contributing

## Development setup

Install `uv`, then create the locked development environment:

```bash
uv sync --locked
```

Run the service locally with placeholder configuration:

```bash
UPSTREAM_BASE_URL="https://gateway.example.com/openai/v1" \
uv run openai-claude-proxy
```

Copy `.env.example` only as a reference. The application reads environment
variables directly; it does not load `.env` files itself.

## Making changes

- Keep changes focused and provider-neutral.
- Add mocked tests for new request fields, response shapes, auth modes, errors,
  and streaming events.
- Never use a production endpoint or credential in tests or documentation.
- Add dependencies with `uv add` or `uv add --dev` and commit the updated lockfile.
- Update `README.md` and `.env.example` for configuration changes.

Format Python before running the complete checks:

```bash
uv run ruff format .
uv run ruff format --check .
uv run ruff check .
uv run ty check
uv run pytest -q
uv build
```

## Release validation

Before a release, configure the opt-in live tests for every intended gateway
dialect and run:

```bash
uv run pytest -m live -q
```

Use only synthetic prompts and tool results. Supply gateway URLs, deployment names,
and credentials through the environment; never commit them or captured work
traffic. Enable the optional feature probes to confirm the selected deployments'
reasoning levels, structured output, prompt caching, model mapping, and request
attribution behavior. The mocked regression suite and all required verification
commands above must also pass on the release commit.

## Pull requests

A pull request should explain:

- The behavior being added or corrected.
- Which client and upstream protocol combinations are affected.
- Any security or credential-handling implications.
- New or changed environment variables.
- The tests and manual checks performed.

Keep compatibility changes backed by minimal, sanitized wire examples. Do not
paste internal URLs, request traces, prompts, API keys, or proprietary responses.
