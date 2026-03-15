"""
Run NVIDIA Nsight profiling for transformer kernel benchmarks.

For each kernel benchmark run:
  1. GPU occupancy (Nsight Compute)
  2. Memory bandwidth (Nsight Compute)
  3. Warp divergence / warp execution efficiency (Nsight Compute)
  4. Kernel execution timeline (Nsight Systems)

Outputs:
  - profiling/nsight_reports/<prefix>_timeline.nsys-rep   (Nsight Systems)
  - profiling/nsight_reports/<prefix>_kernels.ncu-rep     (Nsight Compute binary)
  - profiling/nsight_reports/<prefix>_kernels_metrics.json (parsed metrics for roofline)

Run from repo root:
  python profiling/run_nsight_profiling.py

Requires: NVIDIA Nsight Systems (nsys), Nsight Compute (ncu) on PATH or in CUDA install.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
NSIGHT_REPORTS = REPO_ROOT / "profiling" / "nsight_reports"
BENCHMARK_SCRIPT = REPO_ROOT / "benchmarks" / "transformer_benchmark.py"
QUICK_PROBE_SCRIPT = Path(__file__).resolve().parent / "_ncu_quick_probe.py"
REPORT_PREFIX = "transformer"

# Nsight Compute: use --set basic (occupancy, memory throughput, warp efficiency included).
# Optional: add --metrics for more (can fail on some GPU architectures).
NCU_EXTRA_METRICS = (
    "achieved_occupancy,"
    "dram__throughput.avg.pct_of_peak_sustained_elapsed,"
    "warp_execution_efficiency,gpu__time_duration.sum,dram__bytes.sum"
)


def find_tool(name: str) -> str | None:
    """Find nsys or ncu on PATH, in CUDA install, or in Windows Program Files."""
    exe = name + (".exe" if sys.platform == "win32" else "")
    # 1. PATH
    for path in os.environ.get("PATH", "").split(os.pathsep):
        p = Path(path) / exe
        if p.exists():
            return str(p)
    # 2. CUDA_PATH / CUDA_HOME
    cuda_home = os.environ.get("CUDA_PATH") or os.environ.get("CUDA_HOME")
    if cuda_home:
        for sub in ("Nsight Systems", "Nsight Compute", "bin"):
            d = Path(cuda_home) / sub
            if not d.exists():
                continue
            for f in d.rglob(exe):
                return str(f)
    # 3. Windows: Program Files (Nsight 常安裝於此)
    if sys.platform == "win32":
        for base in [
            Path(os.environ.get("ProgramFiles", "C:\\Program Files")),
            Path(os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)")),
        ]:
            nvidia = base / "NVIDIA Corporation"
            if not nvidia.exists():
                continue
            if name == "nsys":
                for d in nvidia.glob("Nsight Systems*"):
                    for f in (d / "host").rglob(exe) if (d / "host").exists() else d.rglob(exe):
                        return str(f)
            elif name == "ncu":
                for d in nvidia.glob("Nsight Compute*"):
                    for f in (d / "target").rglob(exe) if (d / "target").exists() else d.rglob(exe):
                        return str(f)
    return None


def run_nsight_systems(out_path: Path) -> bool:
    """Run Nsight Systems to capture kernel execution timeline."""
    nsys = find_tool("nsys")
    if not nsys:
        print("Warning: nsys (Nsight Systems) not found. Skip timeline.")
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_file = out_path.with_suffix("")  # nsys adds .nsys-rep
    # 僅 trace CUDA（不加 nvtx）可減少需管理員權限的項目，降低 exit 1 機率
    skip_nsys = os.environ.get("SKIP_NSYS", "").strip().lower() in ("1", "true", "yes")
    if skip_nsys:
        print("Skipping Nsight Systems (SKIP_NSYS=1).")
        return False
    cmd = [
        nsys, "profile",
        "-o", str(out_file),
        "--stats=true",
        "--trace=cuda",
        sys.executable, str(BENCHMARK_SCRIPT),
    ]
    print("Running Nsight Systems (timeline)...")
    try:
        result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=300)
        rep = out_file.with_suffix(".nsys-rep")
        if rep.exists():
            print(f"  Timeline report: {rep}")
            return True
        # nsys 常因「被 profiling 的程式」回傳非 0 而 exit 1，或權限不足
        print(f"  nsys exited with code {result.returncode}.")
        if result.stderr:
            for line in result.stderr.strip().splitlines()[-8:]:
                print(f"    {line}")
        print("  Tip: Set SKIP_NSYS=1 to skip timeline and only run ncu; or run CMD as Administrator.")
        return False
    except subprocess.TimeoutExpired:
        print("  nsys timed out.")
        return False
    except Exception as e:
        print(f"  nsys failed: {e}")
        return False


def run_nsight_compute(out_rep: Path, out_json: Path) -> bool:
    """Run Nsight Compute for occupancy, memory bandwidth, warp divergence."""
    ncu = find_tool("ncu")
    if not ncu:
        print("Warning: ncu (Nsight Compute) not found. Skip kernel metrics.")
        return False
    out_rep.parent.mkdir(parents=True, exist_ok=True)
    use_quick = os.environ.get("NCU_QUICK", "").strip().lower() in ("1", "true", "yes")
    if use_quick:
        script = QUICK_PROBE_SCRIPT
        ncu_kernel_count = 1
        print("Running Nsight Compute (NCU_QUICK=1: one kernel, ~1 min)...")
    else:
        script = BENCHMARK_SCRIPT
        ncu_kernel_count = int(os.environ.get("NCU_KERNEL_COUNT", "3"))
        print("Running Nsight Compute (occupancy, memory BW, warp efficiency)...")
        print(f"  (may take 5-15 min for {ncu_kernel_count} kernels; set NCU_QUICK=1 for ~1 min single-kernel probe)")
    cmd = [
        ncu,
        "-o", str(out_rep.with_suffix("")),
        "-c", str(ncu_kernel_count),
        "--set", "basic",
        "--page", "raw",
        "--csv",
        "--force-overwrite",
        sys.executable, str(script),
    ]
    timeout_sec = 120 if use_quick else 600
    try:
        result = subprocess.run(
            cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=timeout_sec
        )
        rep_file = out_rep.with_suffix(".ncu-rep")
        if not rep_file.exists():
            rep_file = out_rep.with_suffix(".nsight-cuprof")
        if rep_file.exists():
            print(f"  Kernel report: {rep_file}")
            # Re-export raw CSV from report for parsing
            export_raw_csv(ncu, rep_file, out_json)
        else:
            parse_and_save_metrics(result.stdout or "", result.stderr or "", out_json)
        if result.returncode != 0 and "PROF" not in (result.stdout or ""):
            print(f"  ncu stderr: {result.stderr[:500]}")
        return True
    except subprocess.TimeoutExpired:
        print("  ncu timed out (reduce -c or run with smaller benchmark).")
        return False
    except Exception as e:
        print(f"  ncu failed: {e}")
        return False


def export_raw_csv(ncu: str, rep_file: Path, out_json: Path) -> None:
    """Export raw CSV from .ncu-rep and parse into JSON."""
    cmd = [ncu, "--import", str(rep_file), "--page", "raw", "--csv"]
    try:
        result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=120)
        parse_and_save_metrics(result.stdout or "", result.stderr or "", out_json)
    except Exception as e:
        print(f"  Export raw CSV failed: {e}")


def _parse_value(val: str) -> float | str:
    val = val.strip().replace("%", "")
    for suffix, factor in [("G", 1e9), ("M", 1e6), ("K", 1e3)]:
        if suffix in val.upper():
            try:
                return float(val.upper().replace(suffix, "").strip()) * factor
            except ValueError:
                break
    try:
        return float(val)
    except ValueError:
        return val


def parse_and_save_metrics(stdout: str, stderr: str, out_json: Path) -> None:
    """Parse ncu raw CSV output and save per-kernel metrics to JSON."""
    kernels: list[dict] = []
    lines = [s.strip() for s in stdout.splitlines()]
    current: dict | None = None
    for i, line in enumerate(lines):
        if not line or line.startswith("=="):
            continue
        parts = [p.strip().strip('"') for p in line.split(",")]
        if len(parts) >= 3 and current is not None:
            name, unit, val = parts[0], parts[1], parts[2]
            current["metrics"][name] = {"unit": unit, "value": _parse_value(val)}
            continue
        # New kernel: look for "Kernel" / "Name" header or a line that starts kernel block
        if "Kernel" in line or (i > 0 and "Name" in lines[i - 1] and len(parts) >= 1):
            kernel_name = parts[0] if parts else "unknown"
            if kernel_name and not kernel_name.startswith("Metric"):
                current = {"kernel": kernel_name, "metrics": {}}
                kernels.append(current)
    if not kernels:
        kernels = [{"kernel": "open .ncu-rep in Nsight Compute UI", "metrics": {}}]
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"kernels": kernels, "note": "From ncu --page raw --csv"}, f, indent=2)
    print(f"  Metrics JSON: {out_json}")


def main():
    NSIGHT_REPORTS.mkdir(parents=True, exist_ok=True)
    prefix = os.environ.get("NSIGHT_REPORT_PREFIX", REPORT_PREFIX)

    timeline_path = NSIGHT_REPORTS / f"{prefix}_timeline"
    run_nsight_systems(timeline_path)

    kernels_rep = NSIGHT_REPORTS / f"{prefix}_kernels"
    metrics_json = NSIGHT_REPORTS / f"{prefix}_kernels_metrics.json"
    run_nsight_compute(kernels_rep, metrics_json)

    print("\nProfiling results stored in:", NSIGHT_REPORTS)
    print("Generate roofline plot: python profiling/roofline_analysis/plot_roofline.py")
    if not find_tool("nsys") or not find_tool("ncu"):
        print("\nTip: If nsys/ncu were not found, install Nsight Systems & Nsight Compute from")
        print("  https://developer.nvidia.com/nsight-systems  and  https://developer.nvidia.com/nsight-compute")
        print("  or add their bin folders to PATH (e.g. under 'NVIDIA Corporation\\Nsight Compute 20xx\\target\\...').")


if __name__ == "__main__":
    main()
