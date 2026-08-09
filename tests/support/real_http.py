"""Bounded loopback Uvicorn lifecycle helpers for real-HTTP integration tests."""

from __future__ import annotations

import asyncio
import socket
from contextlib import AbstractAsyncContextManager
from typing import TYPE_CHECKING, Self

import uvicorn

if TYPE_CHECKING:
    from types import TracebackType

    from fastapi import FastAPI

STARTUP_TIMEOUT = 5.0
SHUTDOWN_TIMEOUT = 5.0


class LoopbackServer(AbstractAsyncContextManager["LoopbackServer"]):
    """Run one ASGI application on an ephemeral loopback port."""

    def __init__(self, app: FastAPI) -> None:
        """Prepare a server without allocating a socket until entry."""
        self._app = app
        self._socket: socket.socket | None = None
        self._server: uvicorn.Server | None = None
        self._task: asyncio.Task[None] | None = None
        self.base_url = ""

    async def __aenter__(self) -> Self:
        """Bind a free loopback port and wait for bounded server startup."""
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            server_socket.bind(("127.0.0.1", 0))
            server_socket.listen()
            server_socket.setblocking(False)  # noqa: FBT003 - socket API is positional-only.
        except BaseException:
            server_socket.close()
            raise
        self._socket = server_socket
        port = server_socket.getsockname()[1]
        self.base_url = f"http://127.0.0.1:{port}"
        self._server = uvicorn.Server(
            uvicorn.Config(
                self._app,
                host="127.0.0.1",
                port=port,
                lifespan="on",
                log_level="error",
                access_log=False,
            ),
        )
        self._task = asyncio.create_task(self._server.serve(sockets=[server_socket]))
        try:
            async with asyncio.timeout(STARTUP_TIMEOUT):
                while not self._server.started:
                    if self._task.done():
                        await self._task
                    await asyncio.sleep(0.01)
        except BaseException:
            await self._shutdown()
            raise
        return self

    async def __aexit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        """Request bounded shutdown and release the listening socket."""
        await self._shutdown()

    async def _shutdown(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._task is not None:
            try:
                async with asyncio.timeout(SHUTDOWN_TIMEOUT):
                    await self._task
            except TimeoutError:
                self._task.cancel()
                await asyncio.gather(self._task, return_exceptions=True)
        if self._socket is not None:
            self._socket.close()
        self._task = None
        self._server = None
        self._socket = None
