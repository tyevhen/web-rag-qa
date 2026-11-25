import uuid

import structlog
from starlette.datastructures import MutableHeaders
from starlette.requests import Request
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from api.errors import unhandled_error_handler


class RequestContextMiddleware:
    """One id per request: bound into every log line, returned as X-Request-ID, used by /ask as its run_id.

    Also the catch-all for errors no handler maps. An app-level Exception handler would log twice, since
    Starlette's ServerErrorMiddleware re-raises after calling it and uvicorn logs the traceback again.
    Pure ASGI rather than @app.middleware("http"): no extra task per request, and the route runs in this context.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        run_id = str(uuid.uuid4())
        scope.setdefault("state", {})["run_id"] = run_id  # what request.state.run_id reads
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(run_id=run_id)
        started = False

        async def send_with_id(message: Message) -> None:
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
                MutableHeaders(scope=message).append("X-Request-ID", run_id)
            await send(message)

        try:
            await self.app(scope, receive, send_with_id)
        except Exception as exc:
            if started:  # headers are already out; nothing left to send but the error
                raise
            response = await unhandled_error_handler(Request(scope), exc)
            await response(scope, receive, send_with_id)
