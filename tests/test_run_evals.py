"""scripts/run_evals.py: the checks, the dataset, and a run against a stub server."""

import importlib.util
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "run_evals", ROOT / "scripts" / "run_evals.py"
)
evals = importlib.util.module_from_spec(_SPEC)
# Registered before execution: dataclasses resolve their module through
# sys.modules, and scripts/ is not an importable package.
sys.modules["run_evals"] = evals
_SPEC.loader.exec_module(evals)


def passes(check: dict, reply: str) -> bool:
    return evals.CHECKS[check["type"]](reply, check)


@pytest.mark.parametrize(
    "check, reply, expected",
    [
        ({"type": "contains", "value": "Paris"}, "the capital is paris.", True),
        ({"type": "contains", "value": "Paris"}, "Lyon", False),
        ({"type": "contains_any", "values": ["red", "blue"]}, "Blue", True),
        ({"type": "contains_any", "values": ["red", "blue"]}, "green", False),
        ({"type": "not_contains", "value": "9.11"}, "9.9", True),
        ({"type": "not_contains", "value": "9.11"}, "9.11 is larger", False),
        ({"type": "regex", "pattern": r"\b42\b"}, "The answer is 42.", True),
        ({"type": "regex", "pattern": r"\b42\b"}, "420", False),
        ({"type": "regex", "pattern": "^arr", "ignore_case": True}, "Arr, fine", True),
        ({"type": "regex", "pattern": "^HELLO$"}, "hello", False),
        ({"type": "json"}, '["red", "green"]', True),
        ({"type": "json"}, "red, green", False),
        ({"type": "json", "keys": ["name"]}, '```json\n{"name": "Ada"}\n```', True),
        ({"type": "json", "keys": ["name"]}, '["name"]', False),
        ({"type": "max_words", "value": 1}, "Red", True),
        ({"type": "max_words", "value": 1}, "Red, obviously", False),
        ({"type": "line_count", "min": 3, "max": 3}, "a\n\nb\nc\n", True),
        ({"type": "line_count", "min": 3, "max": 3}, "a\nb", False),
    ],
)
def test_checks(check, reply, expected):
    assert passes(check, reply) is expected


def test_repository_dataset_is_valid():
    cases = evals.load_dataset(str(ROOT / "evals" / "datasets" / "core.jsonl"))

    assert len(cases) >= 20
    assert all(case.get("tags") for case in cases)


def test_dataset_errors_name_the_line(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text(
        '{"id": "a", "messages": [], "checks": [{"type": "contains", "value": "x"}]}\n'
        '{"id": "b", "messages": [], "checks": [{"type": "vibes"}]}\n'
    )
    with pytest.raises(ValueError, match=r"bad.jsonl:2: unknown check type 'vibes'"):
        evals.load_dataset(str(bad))

    duplicate = tmp_path / "dup.jsonl"
    case = '{"id": "a", "messages": [], "checks": []}\n'
    duplicate.write_text(case + case)
    with pytest.raises(ValueError, match="duplicate case ids"):
        evals.load_dataset(str(duplicate))


REPLIES = {"capital": "Paris", "math": "The answer is 41."}


class Stub(BaseHTTPRequestHandler):
    seen: list[dict] = []

    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Stub.seen.append({"body": body, "auth": self.headers.get("Authorization")})
        question = body["messages"][-1]["content"]
        if question == "boom":
            self.send_response(500)
            self.end_headers()
            self.wfile.write(b'{"detail": "internal"}')
            return
        data = json.dumps(
            {
                "choices": [
                    {"message": {"role": "assistant", "content": REPLIES[question]}}
                ],
                "usage": {"completion_tokens": 3},
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture
def server():
    Stub.seen = []
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Stub)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def write_dataset(tmp_path, *cases) -> str:
    path = tmp_path / "cases.jsonl"
    path.write_text("".join(json.dumps(c) + "\n" for c in cases))
    return str(path)


def case(case_id, question, checks, tags):
    return {
        "id": case_id,
        "tags": tags,
        "messages": [{"role": "user", "content": question}],
        "checks": checks,
    }


def test_run_reports_pass_rate_per_tag_and_failures(
    server, tmp_path, monkeypatch, capsys
):
    monkeypatch.setenv("VORTEX_API_KEY", "secret-token")
    dataset = write_dataset(
        tmp_path,
        case(
            "capital",
            "capital",
            [{"type": "contains", "value": "Paris"}],
            ["knowledge"],
        ),
        case("math", "math", [{"type": "regex", "pattern": r"\b42\b"}], ["reasoning"]),
        case("broken", "boom", [{"type": "contains", "value": "x"}], ["reasoning"]),
    )
    report = tmp_path / "report.json"

    code = evals.main(
        [
            "--url",
            server,
            "--dataset",
            dataset,
            "--json",
            str(report),
            "--min-pass-rate",
            "0.5",
        ]
    )

    assert code == 1  # 1 of 3 passed
    summary = json.loads(report.read_text())["summary"]
    assert (summary["passed"], summary["errors"]) == (1, 1)
    assert summary["by_tag"] == {"knowledge": 1.0, "reasoning": 0.0}
    out = capsys.readouterr().out
    assert "FAIL math" in out and "reply: 'The answer is 41.'" in out
    assert "FAIL broken" in out and "HTTPError" in out
    # Deterministic generation, and the server's bearer token from the env.
    assert all(s["body"]["temperature"] == 0 for s in Stub.seen)
    assert {s["auth"] for s in Stub.seen} == {"Bearer secret-token"}


def test_run_passes_the_gate(server, tmp_path):
    dataset = write_dataset(
        tmp_path,
        case("capital", "capital", [{"type": "contains", "value": "Paris"}], ["k"]),
    )

    assert (
        evals.main(["--url", server, "--dataset", dataset, "--min-pass-rate", "1.0"])
        == 0
    )
