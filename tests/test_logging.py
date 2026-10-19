import logging
import re
import sys

import pytest

from app.logging_config import LogfmtFormatter, request_id_var

# Minimal logfmt reader: key=value pairs where a value is either bare (no
# spaces) or double-quoted with backslash escapes. Mirrors what Loki's
# logfmt stage accepts, so a line that parses here parses there.
_PAIR = re.compile(r'(\w+)=("(?:[^"\\]|\\.)*"|\S*)')


def parse_logfmt(line: str) -> dict[str, str]:
    pairs = {}
    consumed = 0
    for match in _PAIR.finditer(line):
        key, raw = match.groups()
        if raw.startswith('"'):
            raw = re.sub(r"\\(.)", lambda m: {"n": "\n"}.get(m[1], m[1]), raw[1:-1])
        pairs[key] = raw
        consumed += len(match[0]) + 1
    assert consumed - 1 == len(line), f"unparsed content in: {line}"
    return pairs


def _record(msg: str, exc_info=None, **extra) -> logging.LogRecord:
    record = logging.LogRecord(
        "vortex.test", logging.INFO, __file__, 1, msg, None, exc_info
    )
    record.__dict__.update(extra)
    return record


class _CaptureHandler(logging.Handler):
    """Formats at emit time, as StreamHandler does, so context vars apply."""

    def __init__(self) -> None:
        super().__init__()
        self.setFormatter(LogfmtFormatter())
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(self.format(record))


@pytest.fixture
def log_lines():
    handler = _CaptureHandler()
    root = logging.getLogger()
    root.addHandler(handler)
    yield handler.lines
    root.removeHandler(handler)


def test_messages_with_spaces_and_quotes_round_trip():
    line = LogfmtFormatter().format(_record('model said "hi" = done'))

    assert parse_logfmt(line)["msg"] == 'model said "hi" = done'


def test_extra_fields_become_keys():
    line = LogfmtFormatter().format(
        _record("Loaded.", duration_seconds=1.5, path="/a b")
    )

    fields = parse_logfmt(line)
    assert fields["duration_seconds"] == "1.5"
    assert fields["path"] == "/a b"
    assert fields["level"] == "info"
    assert fields["logger"] == "vortex.test"


def test_traceback_stays_on_one_line():
    try:
        raise ValueError("first\nsecond")
    except ValueError:
        line = LogfmtFormatter().format(_record("Failed.", exc_info=sys.exc_info()))

    assert "\n" not in line
    assert "ValueError: first\nsecond" in parse_logfmt(line)["exc"]


def test_request_id_is_attached_when_set():
    token = request_id_var.set("req-1")
    try:
        line = LogfmtFormatter().format(_record("Inside a request."))
    finally:
        request_id_var.reset(token)

    assert parse_logfmt(line)["request_id"] == "req-1"


def test_request_id_is_omitted_outside_requests():
    line = LogfmtFormatter().format(_record("Startup."))

    assert "request_id" not in parse_logfmt(line)


def test_inbound_request_id_is_echoed(client):
    response = client.get("/health", headers={"X-Request-ID": "abc-123"})

    assert response.headers["x-request-id"] == "abc-123"


@pytest.mark.parametrize("inbound", ["has spaces", "x" * 129, 'bad"quote'])
def test_malformed_request_id_is_replaced(client, inbound):
    response = client.get("/health", headers={"X-Request-ID": inbound})

    assert re.fullmatch(r"[0-9a-f]{32}", response.headers["x-request-id"])


def test_request_id_reaches_the_inference_thread(client, log_lines):
    client.post(
        "/v1/completions",
        json={"prompt": "hello"},
        headers={"X-Request-ID": "trace-7"},
    )

    worker_lines = [
        parse_logfmt(line) for line in log_lines if "tests.fake_llama" in line
    ]
    assert worker_lines, "fake model did not log from the worker thread"
    assert all(fields["request_id"] == "trace-7" for fields in worker_lines)


def test_access_log_line_per_request(client, log_lines):
    client.post(
        "/v1/completions", json={"prompt": "hello"}, headers={"X-Request-ID": "r1"}
    )

    access = [
        parse_logfmt(line) for line in log_lines if "logger=vortex.access" in line
    ]
    assert len(access) == 1
    assert access[0]["path"] == "/v1/completions"
    assert access[0]["status"] == "200"
    assert access[0]["request_id"] == "r1"


def test_probe_traffic_is_not_logged_at_info(client, log_lines):
    client.get("/health/live")
    client.get("/metrics")

    assert not [line for line in log_lines if "logger=vortex.access" in line]
