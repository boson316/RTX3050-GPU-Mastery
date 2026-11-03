"""
Run a benchmark binary many times while sampling the GPU's power/clock state, to explain
run-to-run bimodality.

Motivation: in v2 the 1024^3 WMMA GEMM (and occasionally cuBLAS) landed in one of two clusters,
~7.5 or ~10.9 TFLOPS, for a whole process run; a 1 s preheat did not remove it. Kernel bugs do
not produce whole-run clusters, GPU state does. Candidates on a 45 W laptop GeForce:
  - performance state (P0 vs P2: GeForce often runs CUDA work in P2 with a lower memory clock),
  - SW power cap / thermal slowdown (reported by NVML as "throttle reasons"),
  - SM clock at launch time (boost not yet ramped).

For every run this records the binary's KEY=VALUE results plus, sampled every --interval-ms
while the binary runs: P-state, SM and memory clock, power draw, temperature and the union of
NVML throttle reasons. It then splits the runs into a fast and a slow cluster on --metric and
prints the GPU state of each cluster side by side, so the cause can be read off directly.

Usage (repo root, after building the roadmap):
  python tools/gpu_state_probe.py --runs 20
  python tools/gpu_state_probe.py --exe cuda_roadmap/level4_tensor_core/fp16_tensor_core_matmul/fp16_2048 --runs 10
Requires: pip install nvidia-ml-py
Output: results/v2_1/gpu_state_probe_<exe>.csv and a summary on stdout.
"""
from __future__ import annotations

import argparse
import csv
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_EXE = REPO / "cuda_roadmap/level4_tensor_core/fp16_tensor_core_matmul/fp16_matmul_bench"

# NVML clocks-throttle-reason bits (nvml.h: nvmlClocksThrottleReason*)
THROTTLE_BITS = {
    0x1: "gpu_idle",
    0x2: "app_clocks_setting",
    0x4: "sw_power_cap",
    0x8: "hw_slowdown",
    0x10: "sync_boost",
    0x20: "sw_thermal",
    0x40: "hw_thermal",
    0x80: "hw_power_brake",
    0x100: "display_clock",
}


def decode(mask: int) -> list[str]:
    return [name for bit, name in THROTTLE_BITS.items() if mask & bit]


class Sampler(threading.Thread):
    def __init__(self, nvml, handle, interval_s: float):
        super().__init__(daemon=True)
        self.nvml, self.h, self.interval = nvml, handle, interval_s
        self.samples: list[dict] = []
        self.stop = threading.Event()

    def _throttle(self) -> int:
        for fn in ("nvmlDeviceGetCurrentClocksEventReasons", "nvmlDeviceGetCurrentClocksThrottleReasons"):
            if hasattr(self.nvml, fn):
                return int(getattr(self.nvml, fn)(self.h))
        return 0

    def run(self) -> None:
        n = self.nvml
        while not self.stop.is_set():
            try:
                self.samples.append({
                    "pstate": int(n.nvmlDeviceGetPerformanceState(self.h)),
                    "sm_mhz": n.nvmlDeviceGetClockInfo(self.h, n.NVML_CLOCK_SM),
                    "mem_mhz": n.nvmlDeviceGetClockInfo(self.h, n.NVML_CLOCK_MEM),
                    "power_w": n.nvmlDeviceGetPowerUsage(self.h) / 1000.0,
                    "temp_c": n.nvmlDeviceGetTemperature(self.h, n.NVML_TEMPERATURE_GPU),
                    "util": n.nvmlDeviceGetUtilizationRates(self.h).gpu,
                    "throttle": self._throttle(),
                })
            except Exception:  # noqa: BLE001 - a missed sample is not fatal
                pass
            self.stop.wait(self.interval)


def run_once(exe: Path, nvml, handle, interval_s: float) -> dict:
    sampler = Sampler(nvml, handle, interval_s)
    sampler.start()
    t0 = time.perf_counter()
    proc = subprocess.run([str(exe)], cwd=exe.parent, capture_output=True, text=True, timeout=600)
    wall = time.perf_counter() - t0
    sampler.stop.set()
    sampler.join(timeout=2)
    row: dict = {"exit": proc.returncode, "wall_s": round(wall, 2)}
    for line in proc.stdout.splitlines():
        for tok in line.split():
            if "=" in tok:
                k, v = tok.split("=", 1)
                row[k] = v
    busy = [s for s in sampler.samples if s["util"] > 50] or sampler.samples  # samples while the GPU worked
    if busy:
        for key in ("sm_mhz", "mem_mhz", "power_w", "temp_c"):
            row[f"{key}_median"] = round(statistics.median(s[key] for s in busy), 1)
        row["sm_mhz_min"] = min(s["sm_mhz"] for s in busy)
        row["pstates"] = "/".join(f"P{p}" for p in sorted({s["pstate"] for s in busy}))
        mask = 0
        for s in busy:
            mask |= s["throttle"]
        row["throttle"] = "+".join(r for r in decode(mask) if r != "gpu_idle") or "none"
        row["samples"] = len(busy)
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exe", type=Path, default=DEFAULT_EXE, help="benchmark binary (without .exe is fine)")
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--metric", default="WMMA_TILED_GFLOPS", help="output key used to split fast/slow runs")
    ap.add_argument("--interval-ms", type=float, default=20)
    ap.add_argument("--gap-s", type=float, default=2.0, help="idle time between runs (lets clocks drop, as in v2)")
    args = ap.parse_args()

    exe = args.exe if args.exe.is_absolute() else REPO / args.exe
    if not exe.is_file() and exe.with_suffix(".exe").is_file():
        exe = exe.with_suffix(".exe")
    if not exe.is_file():
        sys.exit(f"binary not found: {exe} (build it first: cuda_roadmap/build.bat)")
    try:
        import pynvml as nvml
    except ImportError:
        sys.exit("pip install nvidia-ml-py")
    nvml.nvmlInit()
    handle = nvml.nvmlDeviceGetHandleByIndex(0)
    print(f"{nvml.nvmlDeviceGetName(handle)}: {args.runs} runs of {exe.name}, sampling every {args.interval_ms} ms")

    rows = []
    for i in range(args.runs):
        row = {"run": i + 1, **run_once(exe, nvml, handle, args.interval_ms / 1000)}
        rows.append(row)
        print(f"  run {i + 1:2d}: {args.metric}={row.get(args.metric, '?'):>9}  "
              f"CUBLAS={row.get('CUBLAS_GFLOPS', '?'):>9}  {row.get('pstates', '?'):>5}  "
              f"SM {row.get('sm_mhz_median', '?')} MHz  MEM {row.get('mem_mhz_median', '?')} MHz  "
              f"{row.get('power_w_median', '?')} W  {row.get('temp_c_median', '?')} C  throttle={row.get('throttle', '?')}",
              flush=True)
        time.sleep(args.gap_s)
    nvml.nvmlShutdown()

    out_dir = REPO / "results" / "v2_1"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"gpu_state_probe_{exe.stem}.csv"
    fields = sorted({k for r in rows for k in r}, key=lambda k: (k != "run", k))
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\nCSV: {out}")

    vals = [(float(r[args.metric]), r) for r in rows if args.metric in r and r.get("exit") == 0]
    if len(vals) < 4:
        print("not enough successful runs to compare clusters")
        return
    xs = sorted(v for v, _ in vals)
    # split at the largest gap between consecutive sorted values (1-D two-cluster split)
    gap, cut = max((xs[i + 1] - xs[i], (xs[i + 1] + xs[i]) / 2) for i in range(len(xs) - 1))
    slow = [r for v, r in vals if v < cut]
    fast = [r for v, r in vals if v >= cut]
    if gap < 0.05 * xs[-1] or not slow or not fast:
        print(f"no clear clusters on {args.metric} (largest gap {gap:.0f}, range {xs[0]:.0f}-{xs[-1]:.0f})")
        return

    def col(group, key):
        v = [float(r[key]) for r in group if key in r]
        return f"{statistics.median(v):.1f}" if v else "-"

    print(f"\nclusters on {args.metric} (split at {cut:.0f}):")
    print(f"{'':18s}{'slow':>14s}{'fast':>14s}")
    print(f"{'runs':18s}{len(slow):>14d}{len(fast):>14d}")
    for key in (args.metric, "CUBLAS_GFLOPS", "sm_mhz_median", "sm_mhz_min", "mem_mhz_median", "power_w_median",
                "temp_c_median"):
        print(f"{key:18s}{col(slow, key):>14s}{col(fast, key):>14s}")
    for key in ("pstates", "throttle"):
        s = ", ".join(sorted({r.get(key, '?') for r in slow}))
        f_ = ", ".join(sorted({r.get(key, '?') for r in fast}))
        print(f"{key:18s}  slow: {s}\n{'':18s}  fast: {f_}")
    print("\nRead it as: the row whose values differ between the two columns is the cause "
          "(e.g. mem_mhz lower + P2 in 'slow' -> memory P-state; sw_power_cap only in 'slow' -> power limit).")


if __name__ == "__main__":
    main()
