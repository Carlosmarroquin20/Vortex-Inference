"""scripts/bench_llama.py against a stand-in for llama_cpp.Llama."""

import importlib.util
import json
import sys
import time
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "bench_llama", Path(__file__).resolve().parents[1] / "scripts" / "bench_llama.py"
)
bench = importlib.util.module_from_spec(_SPEC)
# Registered before execution: dataclasses resolve their module through
# sys.modules, and scripts/ is not an importable package.
sys.modules["bench_llama"] = bench
_SPEC.loader.exec_module(bench)


class TimedLlama:
    """Evaluates at a fixed cost per token, faster for batches than singles."""

    instances: list["TimedLlama"] = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.evaluated: list[int] = []
        TimedLlama.instances.append(self)

    def tokenize(self, text: bytes) -> list[int]:
        return list(range(len(text.split())))

    def reset(self) -> None:
        self.evaluated.clear()

    def eval(self, tokens) -> None:
        time.sleep(0.0005 * len(tokens) if len(tokens) > 1 else 0.002)
        self.evaluated.extend(tokens)


@pytest.fixture
def timed_llama(monkeypatch):
    import llama_cpp

    TimedLlama.instances = []
    monkeypatch.setattr(llama_cpp, "Llama", TimedLlama)
    return TimedLlama


def test_each_configuration_is_loaded_and_measured(timed_llama, tmp_path, capsys):
    out = tmp_path / "results.json"

    code = bench.main(
        [
            "--model", "/models/m.gguf",
            "--threads", "2",
            "--threads-batch", "0,2",
            "--batch", "512,16",
            "--prompt-tokens", "40",
            "--gen-tokens", "4",
            "--repeat", "1",
            "--json", str(out),
        ]
    )  # fmt: skip

    assert code == 0
    results = json.loads(out.read_text())
    assert len(results) == len(timed_llama.instances) == 4
    # 0 means llama-cpp-python's default and is passed as None.
    assert [i.kwargs["n_threads_batch"] for i in timed_llama.instances] == [
        None, None, 2, 2,
    ]  # fmt: skip
    assert {i.kwargs["n_ubatch"] for i in timed_llama.instances} == {512, 16}
    first = results[0]
    assert first["prompt_tokens"] == 40
    # Batched evaluation is cheaper per token in the stand-in, as in llama.cpp.
    assert first["prefill_tokens_per_second"] > first["decode_tokens_per_second"]
    assert "prefill tok/s" in capsys.readouterr().out


def test_prompt_has_exactly_the_requested_length():
    prompt = bench.build_prompt(TimedLlama(), 123)

    assert len(prompt) == 123


def test_table_lists_one_row_per_configuration():
    row = bench.Result(2, 2, 512, 1.0, 256, 115.2, 38.6)

    table = bench.format_table([row, row])

    assert len(table.splitlines()) == 3
    assert "115.2" in table and "38.6" in table
