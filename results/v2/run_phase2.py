"""Run a benchmark N times, keeping each run's log and CSV plus a GPU clock/power trace.

Usage: python results/v2/run_phase2.py <run_id> <name> [<name> ...]
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOGS = ROOT / "results" / "v2" / "logs"
CSVS = ROOT / "results" / "v2" / "csv"
QUERY = "timestamp,pstate,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,clocks_event_reasons.active"
SCRIPTS = {
    "matmul": ("benchmarks/benchmark_matmul.py", "benchmark_matmul_results.csv"),
    "conv": ("benchmarks/benchmark_conv.py", "benchmark_conv_results.csv"),
    "attention": ("benchmarks/benchmark_attention.py", "benchmark_attention_results.csv"),
    "transformer": ("benchmarks/benchmark_transformer.py", "benchmark_transformer_results.csv"),
}


def main() -> int:
    run_id, names = sys.argv[1], sys.argv[2:]
    (CSVS / f"run{run_id}").mkdir(parents=True, exist_ok=True)
    status = 0
    for name in names:
        script, csv_name = SCRIPTS[name]
        log = LOGS / f"{name}_run{run_id}.log"
        trace = LOGS / f"{name}_run{run_id}_clocks.csv"
        with open(trace, "w") as t:
            sampler = subprocess.Popen(
                ["nvidia-smi", f"--query-gpu={QUERY}", "--format=csv", "-lms", "500"], stdout=t, stderr=subprocess.STDOUT
            )
        start = time.time()
        with open(log, "w") as f:
            code = subprocess.run([sys.executable, script], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode
        sampler.terminate()
        sampler.wait()
        shutil.copy(ROOT / "benchmarks" / csv_name, CSVS / f"run{run_id}" / csv_name)
        print(f"run{run_id} {name}: exit={code} {time.time() - start:.0f}s", flush=True)
        status = status or code
    return status


if __name__ == "__main__":
    sys.exit(main())
