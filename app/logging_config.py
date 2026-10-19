"""
Structured logging and request correlation.

Log lines are emitted as logfmt so Promtail/Loki can extract labels with the
built-in `logfmt` stage. The previous format interpolated values into the
message with f-strings and left msg= unquoted, so any space in a message
split it into spurious keys at parse time. Here every value is escaped and
quoted when needed, and structured fields are passed through `extra=` rather
than formatted into the message, which keeps messages stable and greppable.

Each HTTP request is tagged with a request ID (taken from an inbound
X-Request-ID header or generated) that is attached to every log line written
while serving it, including lines from the inference worker thread, and is
echoed back in the response headers so clients can quote it in bug reports.
"""

import logging
import re
import time
import uuid
from contextvars import ContextVar
from datetime import UTC, datetime

from starlette.types import ASGIApp, Message, Receive, Scope, Send

request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)

# Attributes every LogRecord carries. Anything else on a record was supplied
# through `extra=` and is rendered as a logfmt field.
_RESERVED_ATTRS = frozenset(
    vars(logging.LogRecord("", 0, "", 0, "", None, None)).keys()
    | {"message", "asctime", "taskName"}
)

_NEEDS_QUOTING = re.compile(r'[\s="\\]')


def _logfmt_value(value: object) -> str:
    text = str(value)
    if text and not _NEEDS_QUOTING.search(text):
        return text
    escaped = (
        text.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )
    return f'"{escaped}"'


class LogfmtFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.fromtimestamp(record.created, tz=UTC)
        pairs: list[tuple[str, object]] = [
            ("ts", ts.isoformat(timespec="milliseconds").replace("+00:00", "Z")),
            ("level", record.levelname.lower()),
            ("logger", record.name),
            ("msg", record.getMessage()),
        ]

        request_id = request_id_var.get()
        if request_id is not None:
            pairs.append(("request_id", request_id))

        for key, value in record.__dict__.items():
            if key not in _RESERVED_ATTRS and not key.startswith("_"):
                pairs.append((key, value))

        # Tracebacks are kept on the same line (newlines escaped) so a single
        # failure is a single log entry rather than dozens of orphaned lines.
        if record.exc_info:
            pairs.append(("exc", self.formatException(record.exc_info)))

        return " ".join(f"{key}={_logfmt_value(value)}" for key, value in pairs)


def configure_logging(level: int = logging.INFO) -> None:
    """
    Installs the logfmt formatter on the root logger and routes uvicorn's
    loggers through it, so the process emits one consistent format.

    uvicorn configures its own handlers before importing the application;
    clearing them here (at import time) takes precedence over that setup.
    """
    handler = logging.StreamHandler()
    handler.setFormatter(LogfmtFormatter())

    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)

    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uv_logger = logging.getLogger(name)
        uv_logger.handlers.clear()
        uv_logger.propagate = True


# Inbound IDs are echoed into logs and response headers, so only a bounded,
# header-safe charset is accepted; anything else is replaced, not trusted.
_VALID_REQUEST_ID = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")

# Probe and scrape traffic arrives every few seconds per pod. Logging it at
# INFO would drown the request log, so it is demoted to DEBUG.
_QUIET_PATHS = frozenset({"/health", "/health/live", "/health/ready", "/metrics"})

_access_logger = logging.getLogger("vortex.access")


class RequestContextMiddleware:
    """
    Assigns a request ID, exposes it as X-Request-ID, and writes one access
    log line per request with status and duration.

    Implemented as plain ASGI middleware rather than BaseHTTPMiddleware so the
    context variable is set in the same task that runs the endpoint, with no
    extra task or response buffering in between.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        inbound = dict(scope["headers"]).get(b"x-request-id", b"").decode("latin-1")
        request_id = inbound if _VALID_REQUEST_ID.match(inbound) else uuid.uuid4().hex
        token = request_id_var.set(request_id)

        status_code = 500
        start = time.monotonic()

        async def send_with_request_id(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
                headers = list(message.get("headers", []))
                headers.append((b"x-request-id", request_id.encode("latin-1")))
                message = {**message, "headers": headers}
            await send(message)

        try:
            await self.app(scope, receive, send_with_request_id)
        finally:
            path = scope.get("path", "")
            level = logging.DEBUG if path in _QUIET_PATHS else logging.INFO
            _access_logger.log(
                level,
                "Request completed.",
                extra={
                    "method": scope.get("method", ""),
                    "path": path,
                    "status": status_code,
                    "duration_seconds": round(time.monotonic() - start, 4),
                },
            )
            request_id_var.reset(token)
