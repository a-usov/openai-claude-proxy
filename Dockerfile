FROM ghcr.io/astral-sh/uv:0.12.3 AS uv

FROM python:3.12-slim

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
COPY --from=uv /uv /uvx /bin/
COPY pyproject.toml uv.lock README.md /app/
COPY src /app/src
RUN uv sync --locked --no-dev

USER nobody
EXPOSE 8080
ENV HOST=0.0.0.0 PORT=8080
CMD ["/app/.venv/bin/openai-claude-proxy"]
