"""Run the v2.1 benchmarks for one round, keeping each log and CSV plus a GPU clock/power trace.

Usage: python results/v2_1/run_phase2.py <run_id>
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOGS = ROOT / "results" / "v2_1" / "logs"
CSVS = ROOT / "results" / "v2_1" / "csv"
QUERY = "timestamp,pstate,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,clocks_event_reasons.active,memory.used"


def run(name: str, cmd: list[str], run_id: str) -> int:
    log = LOGS / f"{name}_run{run_id}.log"
    trace = LOGS / f"{name}_run{run_id}_clocks.csv"
    with open(trace, "w") as t:
        sampler = subprocess.Popen(["nvidia-smi", f"--query-gpu={QUERY}", "--format=csv", "-lms", "500"], stdout=t, stderr=subprocess.STDOUT)
    start = time.time()
    with open(log, "w") as f:
        code = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode
    sampler.terminate()
    sampler.wait()
    print(f"run{run_id} {name}: exit={code} {time.time() - start:.0f}s", flush=True)
    return code


def main() -> int:
    run_id = sys.argv[1]
    (CSVS / f"run{run_id}").mkdir(parents=True, exist_ok=True)
    status = run("transformer", [sys.executable, "benchmarks/benchmark_transformer.py"], run_id)
    shutil.copy(ROOT / "benchmarks" / "benchmark_transformer_results.csv", CSVS / f"run{run_id}" / "benchmark_transformer_results.csv")
    status = run("longseq", [sys.executable, "benchmarks/benchmark_attention_longseq.py", run_id], run_id) or status
    return status


if __name__ == "__main__":
    sys.exit(main())
