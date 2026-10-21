"""
The worker's HTTP endpoint: Prometheus metrics and its Kubernetes probes.

    /metrics       Prometheus exposition
    /health/ready  200 once the model is loaded and the worker is consuming
    /health/live   200 unless a restart would fix something

The server runs in a thread of its own, so answering a request proves only
that the process exists. /health/live therefore checks the two failures a
restart recovers from, as the API's liveness probe does: an event loop that
has stopped running (its heartbeat goes stale, and no job can make progress)
and an inference thread wedged inside the backend long past its deadline
(the job it belongs to can never finish, and its slot is never freed).
"""

import asyncio
import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.backends import InferenceBackend

logger = logging.getLogger("vortex.worker")

HEARTBEAT_INTERVAL_SECONDS = 5.0
# Several missed heartbeats before the loop counts as stuck. Large enough
# that a busy loop (JSON parsing a big result, a slow Redis round trip) is
# never mistaken for a dead one.
HEARTBEAT_STALE_SECONDS = 30.0
# Margin past the inference deadline before a generation counts as wedged
# rather than slow to stop, as in app/main.py.
STALL_GRACE_SECONDS = 60.0


class WorkerHealth:
    """State the probes report on, shared between the event loop and the server."""

    def __init__(self, backend: InferenceBackend, timeout_seconds: float) -> None:
        self._backend = backend
        self._timeout = timeout_seconds
        self._heartbeat: float | None = None
        self.consuming = False

    def beat(self) -> None:
        self._heartbeat = time.monotonic()

    async def run_heartbeat(self) -> None:
        """Updates the heartbeat for as long as the event loop is running."""
        while True:
            self.beat()
            await asyncio.sleep(HEARTBEAT_INTERVAL_SECONDS)

    def ready(self) -> bool:
        return self.consuming

    def liveness_problem(self) -> str | None:
        """Why the worker should be restarted, or None while it is healthy."""
        if (
            self._heartbeat is not None
            and time.monotonic() - self._heartbeat > HEARTBEAT_STALE_SECONDS
        ):
            return "The event loop has stopped running."
        if self._backend.busy_seconds() > self._timeout + STALL_GRACE_SECONDS:
            return "An inference thread is stalled past its deadline."
        return None


def serve(
    health: WorkerHealth,
    port: int,
    host: str = "0.0.0.0",  # noqa: S104
) -> ThreadingHTTPServer:
    """
    Starts the endpoint in a daemon thread. All interfaces by default: the
    kubelet and Prometheus reach it on the pod IP. It has no write
    operations and exposes no job data.
    """

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path == "/metrics":
                self._reply(200, generate_latest(), CONTENT_TYPE_LATEST)
            elif self.path == "/health/ready":
                if health.ready():
                    self._reply(200, b"ready\n")
                else:
                    self._reply(503, b"not consuming\n")
            elif self.path == "/health/live":
                problem = health.liveness_problem()
                if problem is None:
                    self._reply(200, b"alive\n")
                else:
                    self._reply(503, f"{problem}\n".encode())
            else:
                self._reply(404, b"not found\n")

        def _reply(
            self, status: int, body: bytes, content_type: str = "text/plain"
        ) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            # Probes and scrapes every few seconds would drown the job logs;
            # failures are visible as probe events on the pod instead.
            pass

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    threading.Thread(
        target=server.serve_forever, name="health-server", daemon=True
    ).start()
    return server
