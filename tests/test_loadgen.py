"""
scripts/loadgen.py against a local HTTP stub.

The stub writes server-sent events with real pauses between them, so the
client-side timings the generator reports can be checked against known gaps.
"""

import importlib.util
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "loadgen", Path(__file__).resolve().parents[1] / "scripts" / "loadgen.py"
)
loadgen = importlib.util.module_from_spec(_SPEC)
# Registered before execution: dataclasses resolve their module through
# sys.modules, and scripts/ is not an importable package.
sys.modules["loadgen"] = loadgen
_SPEC.loader.exec_module(loadgen)

GAP = 0.1  # seconds between streamed chunks


class Stub(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    behaviour = "ok"
    seen: list[dict] = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Stub.seen.append(
            {"path": self.path, "body": body, "auth": self.headers.get("Authorization")}
        )
        if Stub.behaviour == "overloaded":
            self._json(503, {"detail": "overloaded"})
        elif not body.get("stream"):
            self._json(
                200,
                {
                    "choices": [{"message": {"content": "hi"}, "text": "hi"}],
                    "usage": {"prompt_tokens": 4, "completion_tokens": 2},
                },
            )
        else:
            self._stream(body)

    def _json(self, status, payload):
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _stream(self, body):
        chat = self.path.endswith("/chat/completions")
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close")
        self.end_headers()

        def send(payload):
            self.wfile.write(f"data: {json.dumps(payload)}\n\n".encode())
            self.wfile.flush()

        def text(value):
            if chat:
                return {"choices": [{"delta": {"content": value}}]}
            return {"choices": [{"text": value}]}

        if chat:
            send({"choices": [{"delta": {"role": "assistant", "content": ""}}]})
        time.sleep(GAP)
        send(text("a"))
        if Stub.behaviour == "stream-error":
            send({"error": {"message": "decode failed", "code": 500}})
            return
        for value in ("b", "c"):
            time.sleep(GAP)
            send(text(value))
        send({"choices": [], "usage": {"prompt_tokens": 4, "completion_tokens": 3}})
        self.wfile.write(b"data: [DONE]\n\n")

    def log_message(self, *args):
        pass


@pytest.fixture
def server():
    Stub.behaviour = "ok"
    Stub.seen = []
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Stub)
    thread = threading.Thread(
        target=httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
    )
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def config(url, **overrides):
    values = {
        "url": url,
        "endpoint": "chat",
        "stream": True,
        "concurrency": 2,
        "requests": 4,
        "max_tokens": 8,
        "temperature": 0.0,
        "timeout_s": 10.0,
        "prompts": ["one", "two"],
    }
    values.update(overrides)
    return loadgen.Config(**values)


def test_streamed_timings_are_measured_from_the_client(server):
    results, wall = loadgen.run(config(server), api_key="secret")

    assert [r.ok for r in results] == [True] * 4
    for r in results:
        assert r.ttft_s == pytest.approx(GAP, abs=0.08)
        assert r.tpot_s == pytest.approx(GAP, abs=0.08)
        assert (r.prompt_tokens, r.completion_tokens) == (4, 3)
    # Two workers, four requests of ~3 gaps each: two rounds.
    assert wall == pytest.approx(2 * 3 * GAP, abs=0.3)
    # Two workers race, so arrival order is not fixed; prompts cycle by index.
    assert all(
        s["body"]["stream_options"] == {"include_usage": True} for s in Stub.seen
    )
    assert sorted(s["body"]["messages"][0]["content"] for s in Stub.seen) == [
        "one",
        "one",
        "two",
        "two",
    ]
    assert {s["auth"] for s in Stub.seen} == {"Bearer secret"}


def test_completions_endpoint_and_buffered_mode(server):
    results, _ = loadgen.run(
        # One worker keeps arrival order fixed for the per-request checks.
        config(server, endpoint="completions", stream=False, concurrency=1),
        api_key=None,
    )

    assert all(r.ok and r.ttft_s is None for r in results)
    assert Stub.seen[0]["path"] == "/v1/completions"
    assert Stub.seen[0]["body"]["prompt"] == "one"
    assert "stream_options" not in Stub.seen[0]["body"]
    assert Stub.seen[0]["auth"] is None


def test_streamed_completions_endpoint(server):
    results, _ = loadgen.run(config(server, endpoint="completions"), api_key=None)

    assert all(r.ok and r.completion_tokens == 3 for r in results)


def test_http_errors_are_counted_by_status(server):
    Stub.behaviour = "overloaded"

    results, wall = loadgen.run(config(server), api_key=None)
    summary = loadgen.summarize(results, wall)

    assert summary["failed"] == 4
    assert summary["failures_by_status"] == {"503": 4}
    assert summary["latency_seconds"] is None


def test_error_event_inside_a_stream_is_a_failure(server):
    Stub.behaviour = "stream-error"

    results, _ = loadgen.run(config(server), api_key=None)

    assert all(not r.ok and r.status == 200 for r in results)
    assert "decode failed" in results[0].error


def test_unreachable_server_is_reported():
    results, _ = loadgen.run(
        config("http://127.0.0.1:9", requests=1, timeout_s=0.5), api_key=None
    )

    assert results[0].ok is False
    assert results[0].status == 0


def test_percentile_matches_linear_interpolation():
    values = [1.0, 2.0, 3.0, 4.0]

    assert loadgen.percentile(values, 50) == 2.5
    assert loadgen.percentile(values, 90) == pytest.approx(3.7)
    assert loadgen.percentile([5.0], 99) == 5.0


def test_main_writes_a_json_report(server, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("LOADGEN_TEST_KEY", "from-env")
    report_path = tmp_path / "run.json"

    code = loadgen.main(
        [
            "--url",
            server,
            "--stream",
            "--requests",
            "2",
            "--api-key-env",
            "LOADGEN_TEST_KEY",
            "--json",
            str(report_path),
        ]
    )

    assert code == 0
    report = json.loads(report_path.read_text())
    assert report["summary"]["succeeded"] == 2
    assert report["config"]["stream"] is True
    assert "prompts" not in report["config"]
    assert len(report["results"]) == 2
    assert Stub.seen[0]["auth"] == "Bearer from-env"
    assert "ttft (s)" in capsys.readouterr().out


def test_main_fails_only_when_nothing_succeeded(server):
    Stub.behaviour = "overloaded"

    assert loadgen.main(["--url", server, "--requests", "1"]) == 1


@pytest.mark.parametrize(
    "argv",
    [
        ["--url", "ftp://host"],
        ["--url", "http://host", "--concurrency", "0"],
    ],
)
def test_invalid_arguments_exit(argv):
    with pytest.raises(SystemExit):
        loadgen.parse_args(argv)
