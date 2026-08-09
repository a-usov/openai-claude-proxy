"""Console entry point for running the proxy with Uvicorn."""

from __future__ import annotations

import os

import uvicorn


def main() -> None:
    """Run the proxy using host, port, and logging environment settings."""
    uvicorn.run(
        "openai_claude_proxy.app:create_app",
        factory=True,
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8080")),
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
    )


if __name__ == "__main__":
    main()
