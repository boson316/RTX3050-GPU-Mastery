"""Generate the README tables for v2.1 from the kept CSVs, so no number is typed by hand.

Writes results/v2_1/tables_en.md and results/v2_1/tables_zh.md. Run from the repo root.
"""
from __future__ import annotations

import csv
import glob
import re
import statistics as st
from pathlib import Path

V21 = Path("results/v2_1")
V20 = Path("results/v2")
IMPLS = ("PyTorch", "Triton", "CUDA")
RUNS = (1, 2, 3)

ZH = {
    "Kernel": "Kernel", "Shape": "形狀", "Bound": "受限於", "memory": "記憶體", "compute": "運算",
    "S": "S", "Implementation": "實作", "Latency (ms)": "延遲（ms）", "Extra VRAM (MiB)": "額外顯存（MiB）",
    "Max error vs FP64": "相對 FP64 最大誤差", "TFLOPS": "TFLOPS",
    "% of NVML-max roof": "佔 NVML 最大時脈屋頂 %", "% of sustained roof": "佔持續時脈屋頂 %",
    "Metric": "指標", "Median": "中位數", "Min": "最小", "Max": "最大", "Runs": "次數",
    "State": "狀態", "Slow runs": "慢群", "Fast runs": "快群",
    "1024³ WMMA tiled (GFLOPS)": "1024³ WMMA tiled（GFLOPS）", "1024³ cuBLAS (GFLOPS)": "1024³ cuBLAS（GFLOPS）",
    "2048³ WMMA tiled (GFLOPS)": "2048³ WMMA tiled（GFLOPS）", "2048³ cuBLAS (GFLOPS)": "2048³ cuBLAS（GFLOPS）",
    "2048³ WMMA % of cuBLAS (%)": "2048³ WMMA 佔 cuBLAS（%）", "1024³ WMMA % of cuBLAS (%)": "1024³ WMMA 佔 cuBLAS（%）",
}


def rows_of(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def picked(base: Path, fname: str, keys: tuple[str, ...]) -> dict:
    """key -> (row of the median-latency round, min latency, median latency, max latency)."""
    runs = [rows_of(base / "csv" / f"run{r}" / fname) for r in RUNS]
    out = {}
    for row in runs[0]:
        key = tuple(row[k] for k in keys)
        same = sorted((next(x for x in run if tuple(x[k] for k in keys) == key) for run in runs), key=lambda x: float(x["latency_ms"]))
        out[key] = (same[1], float(same[0]["latency_ms"]), float(same[1]["latency_ms"]), float(same[2]["latency_ms"]))
    return out


def cell(data: dict, key: tuple[str, ...], metric: str) -> str:
    if key not in data:
        return "—"
    row, lo, med, hi = data[key]
    lat = f"{med:.3f} ({lo:.3f}–{hi:.3f})"
    if metric == "tflops":
        return f"{lat} · {float(row['gflops']) / 1e3:.2f} TFLOPS"
    return f"{lat} · {float(row['memory_bandwidth_gbs']):.0f} GB/s"


SHAPES = (("QKV", "2048×768→2304", "tflops"), ("MLP", "2048×768→3072→768", "tflops"), ("Softmax", "2048×768", "gbs"),
          ("LayerNorm", "2048×768", "gbs"), ("GELU", "1572864", "gbs"), ("Softmax@8192x768", "8192×768", "gbs"),
          ("LayerNorm@8192x768", "8192×768", "gbs"), ("GELU@8192x768", "6291456", "gbs"))


def bench_table(tr, base: Path) -> str:
    data = picked(base, "benchmark_transformer_results.csv", ("kernel", "implementation"))
    out = []
    for kernel, shape, metric in SHAPES:
        if not any(k[0] == kernel for k in data):
            continue
        out.append([kernel.split("@")[0], shape] + [cell(data, (kernel, i), metric) for i in IMPLS])
    return table([tr("Kernel"), tr("Shape"), "PyTorch", "Triton", "CUDA"], out)


def longseq_tables(tr) -> str:
    data = picked(V21, "benchmark_attention_longseq_results.csv", ("S", "precision", "impl"))
    assert all(v[0]["verified"] == "PASS" for v in data.values())
    parts = []
    for precision, label in (("fp16", "FP16"), ("fp32", "IEEE FP32"), ("tf32", "TF32")):
        out = []
        for (s, p, impl), (row, lo, med, hi) in data.items():
            if p == precision:
                out.append([s, impl, f"{med:.3f} ({lo:.3f}–{hi:.3f})", row["tflops"], row["peak_extra_mib"], row["max_abs_err_vs_fp64"]])
        parts.append(f"**{label}**\n\n" + table([tr("S"), tr("Implementation"), tr("Latency (ms)"), tr("TFLOPS"), tr("Extra VRAM (MiB)"), tr("Max error vs FP64")], out))
    return "\n\n".join(parts)


def device() -> dict:
    env = (V20 / "environment.md").read_text(encoding="utf-8")
    clock = float(re.search(r"SM clock ([\d.]+) MHz", env).group(1))
    bw = float(re.search(r"peak DRAM bandwidth : (\d+) GB/s", env).group(1))
    peak = float(re.search(r"peak fp16_tc_fp32acc\s*: ([\d.]+) TFLOPS", env).group(1)) * 1e3
    samples = []
    for f in glob.glob(str(V20 / "logs" / "*_run[123]_clocks.csv")):
        with open(f, newline="") as fh:
            for r in csv.reader(fh):
                if len(r) >= 8 and r[2].strip().split()[0].isdigit() and int(r[6].split()[0]) >= 50:
                    samples.append(int(r[2].split()[0]))
    return {"clock": clock, "bw": bw, "peak": peak, "sustained": peak * st.median(samples) / clock}


def roofline_table(tr, dev: dict) -> str:
    runs = [rows_of(V21 / "csv" / f"run{r}" / "roofline_results.csv") for r in RUNS]
    picks = [("matmul", "2048x2048x2048", "GEMM 2048³"), ("transformer", "QKV", "QKV"), ("transformer", "MLP", "MLP"),
             ("attention", "B2_H8_S512_D64", "Attention S512"), ("conv2d", "B128_1x64_64x64", "Conv B128 64×64"),
             ("conv2d", "B128_1x32_28x28", "Conv B128 28×28"), ("transformer", "Softmax", "Softmax"),
             ("transformer", "LayerNorm", "LayerNorm"), ("transformer", "GELU", "GELU")]
    out = []
    for kernel, config, label in picks:
        cells, cells_s, bound = [], [], ""
        for impl in IMPLS:
            same = [r for run in runs for r in run if (r["kernel"], r["config"], r["implementation"]) == (kernel, config, impl)]
            if not same:
                cells.append("—")
                cells_s.append("—")
                continue
            same.sort(key=lambda r: float(r["latency_ms"]))
            row = same[len(same) // 2]
            ai, g = float(row["arithmetic_intensity"]), float(row["achieved_gflops"])
            bound = bound or (tr("memory") if dev["bw"] * ai < dev["peak"] else tr("compute"))  # from the first (FP16) row
            cells.append(f"{100 * g / min(dev['bw'] * ai, dev['peak']):.0f}")
            cells_s.append(f"{100 * g / min(dev['bw'] * ai, dev['sustained']):.0f}")
        out.append([label, bound, " / ".join(cells), " / ".join(cells_s)])
    head = "PyTorch / Triton / CUDA"
    return table([tr("Kernel"), tr("Bound"), f"{tr('% of NVML-max roof')} ({head})", f"{tr('% of sustained roof')} ({head})"], out)


def wmma_table(tr) -> str:
    small = rows_of(V21 / "gpu_state_probe_fp16_matmul_bench.csv")
    large = rows_of(V21 / "gpu_state_probe_fp16_2048.csv")
    out = []
    for label, rows, key, fmt in (("1024³ WMMA tiled (GFLOPS)", small, "WMMA_TILED_GFLOPS", ".1f"), ("1024³ cuBLAS (GFLOPS)", small, "CUBLAS_GFLOPS", ".1f"),
                                  ("1024³ WMMA % of cuBLAS (%)", small, "WMMA_TILED_PCT_OF_CUBLAS", ".1f"), ("2048³ WMMA tiled (GFLOPS)", large, "WMMA_TILED_GFLOPS", ".1f"),
                                  ("2048³ cuBLAS (GFLOPS)", large, "CUBLAS_GFLOPS", ".1f"), ("2048³ WMMA % of cuBLAS (%)", large, "WMMA_TILED_PCT_OF_CUBLAS", ".1f")):
        vals = [float(r[key]) for r in rows]
        out.append([tr(label), f"{st.median(vals):{fmt}}", f"{min(vals):{fmt}}", f"{max(vals):{fmt}}", str(len(vals))])
    return table([tr("Metric"), tr("Median"), tr("Min"), tr("Max"), tr("Runs")], out)


def cluster_table(tr) -> str:
    rows = rows_of(V21 / "gpu_state_probe_fp16_matmul_bench.csv")
    slow = [r for r in rows if float(r["WMMA_TILED_GFLOPS"]) < 8058]  # the probe's split, gpu_state_probe_1024.log
    fast = [r for r in rows if float(r["WMMA_TILED_GFLOPS"]) >= 8058]
    out = [[tr("Runs"), str(len(slow)), str(len(fast))]]
    for key in ("WMMA_TILED_GFLOPS", "WMMA_NAIVE_GFLOPS", "CUBLAS_GFLOPS", "sm_mhz_median", "mem_mhz_median", "power_w_median", "temp_c_median"):
        out.append([f"`{key}`", f"{st.median(float(r[key]) for r in slow):.1f}", f"{st.median(float(r[key]) for r in fast):.1f}"])
    for key in ("pstates", "throttle"):
        out.append([f"`{key}`", ", ".join(sorted({r[key] for r in slow})), ", ".join(sorted({r[key] for r in fast}))])
    return table([tr("State"), tr("Slow runs"), tr("Fast runs")], out)


def build(tr) -> str:
    return "\n\n".join([
        "## measurements (v2.1)\n\n" + bench_table(tr, V21),
        "## control (v2.0)\n\n" + bench_table(tr, V20),
        "## roofline (v2.1)\n\n" + roofline_table(tr, device()),
        "## longseq (v2.1)\n\n" + longseq_tables(tr),
        "## wmma probe (v2.1)\n\n" + wmma_table(tr),
        "## wmma clusters (v2.1)\n\n" + cluster_table(tr),
    ]) + "\n"


(V21 / "tables_en.md").write_text(build(lambda s: s), encoding="utf-8")
(V21 / "tables_zh.md").write_text(build(lambda s: ZH.get(s, s)), encoding="utf-8")
print("wrote tables")
