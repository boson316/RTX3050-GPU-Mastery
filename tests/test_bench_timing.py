"""
The timing harness must raise GPU clocks before its short measurement window: at idle clocks the
same kernel measured up to 16x slower (docs/ENGINEERING_LOG.md #15).
"""
from __future__ import annotations

import re
import statistics
import time
from pathlib import Path

import pytest
import torch
from conftest import requires_cuda

from benchmarks import bench_utils


@requires_cuda
def test_run_bench_preheats_before_first_launch(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(bench_utils, "preheat_gpu", lambda: calls.append("preheat"))
    bench_utils.run_bench(lambda: calls.append("launch"), warmup=1, repeat=1, rounds=1)
    assert calls[0] == "preheat"
    assert "launch" in calls


@requires_cuda
def test_preheat_lasts_the_requested_time() -> None:
    start = time.perf_counter()
    bench_utils.preheat_gpu(0.2)
    assert time.perf_counter() - start >= 0.2


@requires_cuda
def test_latency_is_stable_after_idle() -> None:
    x = torch.randn(2048, 768, device="cuda", dtype=torch.half)
    samples = []
    for _ in range(6):
        time.sleep(1.5)  # let the clocks drop toward idle
        samples.append(bench_utils.run_bench(lambda: torch.softmax(x, -1)))
    assert max(samples) < 3 * statistics.median(samples), samples


# The roofline once had its own unverified timer and timed torch.randn with some kernels (#16).
def test_roofline_uses_the_shared_harness() -> None:
    from profiling import roofline_analysis

    assert roofline_analysis.run_bench is bench_utils.run_bench


def test_roofline_rejects_a_wrong_kernel_before_timing(monkeypatch) -> None:
    from profiling import roofline_analysis

    monkeypatch.setattr(roofline_analysis, "run_bench", lambda fn: pytest.fail("timed a wrong kernel"))
    ref = torch.zeros(4)
    with pytest.raises(bench_utils.BenchmarkError):
        roofline_analysis._verified_bench("wrong", lambda: ref + 1, ref)


def test_roofline_source_times_no_input_generation_and_drops_no_failures() -> None:
    source = (Path(__file__).resolve().parent.parent / "profiling" / "roofline_analysis.py").read_text(encoding="utf-8")
    assert not re.search(r"lambda:[^\n]*torch\.randn", source)
    assert not re.search(r"except Exception:\s*\n\s*pass", source)


# dashboard charts once came from a script with hard-coded numbers and from unverified timing loops (#17).
def test_charts_do_not_time_or_hard_code_anything() -> None:
    repo = Path(__file__).resolve().parent.parent
    assert not (repo / "cuda" / "plot_matrix_speedup.py").exists()
    source = (repo / "benchmarks" / "generate_charts.py").read_text(encoding="utf-8")
    assert "perf_counter" not in source


def test_chart_values_come_from_the_csv(tmp_path, monkeypatch) -> None:
    pytest.importorskip("matplotlib")
    from benchmarks import generate_charts

    (tmp_path / "in.csv").write_text("implementation,size,latency_ms\nA,1x1,1.5\nB,1x1,2.5\nA,2x2,3.5\n", encoding="utf-8")
    monkeypatch.setattr(generate_charts, "BENCHMARKS_DIR", tmp_path)
    generate_charts._plot_from_csv("in.csv", "size", "out.png", "t")
    assert (tmp_path / "out.png").stat().st_size > 0
