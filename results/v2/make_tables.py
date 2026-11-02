"""Generate the README result tables from the kept CSVs and logs, so no number is typed by hand.

Writes results/v2/tables_en.md and results/v2/tables_zh.md. Run from the repo root.
"""
from __future__ import annotations

import csv
import glob
import json
import re
import statistics as st
from pathlib import Path

BASE = Path("results/v2")
FINAL = BASE / "csv" / "final"
RANGES = json.loads((FINAL / "latency_ranges.json").read_text(encoding="utf-8"))
IMPLS = ("PyTorch", "Triton", "CUDA")

ZH = {
    "Kernel": "Kernel", "Shape": "形狀", "Bound": "受限於", "memory": "記憶體", "compute": "運算",
    "S": "S", "Implementation": "實作", "Latency (ms)": "延遲（ms）", "Extra VRAM (MiB)": "額外顯存（MiB）",
    "Max error vs FP64": "相對 FP64 最大誤差", "Matrix-materializing": "傳統（產生完整矩陣）",
    "Metric": "指標", "Median": "中位數", "Min": "最小", "Max": "最大", "Runs": "次數",
    "% of NVML-max roof": "佔 NVML 最大時脈屋頂 %", "% of sustained roof": "佔持續時脈屋頂 %",
    "Launch floor (ms)": "發送下限（ms）", "Value": "數值",
    "Median SM clock under load": "負載下 SM 時脈中位數",
    "Peak at NVML max clock": "NVML 最大時脈的峰值",
    "Peak at the sustained clock": "持續時脈的峰值",
    "Peak DRAM bandwidth": "DRAM 峰值頻寬",
}


def rows_of(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def cell(prefix: str, row: dict | None, key: tuple[str, ...], metric: str) -> str:
    if row is None:
        return "—"
    r = RANGES[f"{prefix}|{'|'.join(key)}"]
    lat = f"{r['median']:.3f} ({r['min']:.3f}–{r['max']:.3f})"
    if metric == "tflops":
        return f"{lat} · {float(row['gflops']) / 1e3:.2f} TFLOPS"
    return f"{lat} · {float(row['memory_bandwidth_gbs']):.0f} GB/s"


def benchmark_table(tr) -> str:
    out = []
    data = rows_of(FINAL / "benchmark_matmul_results.csv")
    for size in ("512x512x512", "1024x1024x1024", "2048x2048x2048"):
        by = {r["implementation"]: r for r in data if r["size"] == size}
        out.append(["GEMM", size.replace("x", "×")] + [cell("matmul", by.get(i), (i, size), "tflops") for i in IMPLS])
    data = rows_of(FINAL / "benchmark_transformer_results.csv")
    shapes = {"QKV": "2048×768→2304", "MLP": "2048×768→3072→768", "Softmax": "2048×768", "LayerNorm": "2048×768", "GELU": "1572864"}
    for kernel, shape in shapes.items():
        by = {r["implementation"]: r for r in data if r["kernel"] == kernel}
        metric = "tflops" if kernel in ("QKV", "MLP") else "gbs"
        out.append([kernel, shape] + [cell("transformer", by.get(i), (kernel, i), metric) for i in IMPLS])
    data = rows_of(FINAL / "benchmark_conv_results.csv")
    for config in dict.fromkeys(r["config"] for r in data):
        by = {r["implementation"]: r for r in data if r["config"] == config}
        out.append(["Conv 3×3", config] + [cell("conv", by.get(i), (i, config), "tflops") for i in IMPLS])
    data = rows_of(FINAL / "benchmark_attention_results.csv")
    for config in dict.fromkeys(r["config"] for r in data):
        by = {r["implementation"]: r for r in data if r["config"] == config}
        out.append(["Attention", config] + [cell("attention", by.get(i), (i, config), "tflops") for i in IMPLS])
    return table([tr("Kernel"), tr("Shape"), "PyTorch", "Triton", "CUDA"], out)


def sustained_clock() -> float:
    samples = []
    for f in glob.glob(str(BASE / "logs" / "*_run[123]_clocks.csv")):
        with open(f, newline="") as fh:
            for r in csv.reader(fh):
                if len(r) >= 8 and r[2].strip().split()[0].isdigit() and int(r[6].split()[0]) >= 50:
                    samples.append(int(r[2].split()[0]))
    return st.median(samples)


def device() -> dict:
    env = (BASE / "environment.md").read_text(encoding="utf-8")
    sms = int(re.search(r"(\d+) SMs", env).group(1))
    clock = float(re.search(r"SM clock ([\d.]+) MHz", env).group(1))
    bw = float(re.search(r"peak DRAM bandwidth : (\d+) GB/s", env).group(1))
    peak = float(re.search(r"peak fp16_tc_fp32acc\s*: ([\d.]+) TFLOPS", env).group(1)) * 1e3
    return {"sms": sms, "clock": clock, "bw": bw, "peak": peak, "sustained": peak * sustained_clock() / clock}


def roofline_table(tr, dev) -> str:
    runs = [rows_of(BASE / "csv" / f"run{r}" / "roofline_results.csv") for r in (1, 2, 3)]
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
            memory = dev["bw"] * ai < dev["peak"]
            bound = bound or (tr("memory") if memory else tr("compute"))  # from the first (FP16) row
            cells.append(f"{100 * g / min(dev['bw'] * ai, dev['peak']):.0f}")
            cells_s.append(f"{100 * g / min(dev['bw'] * ai, dev['sustained']):.0f}")
        out.append([label, bound, " / ".join(cells), " / ".join(cells_s)])
    head = "PyTorch / Triton / CUDA"
    return table([tr("Kernel"), tr("Bound"), f"{tr('% of NVML-max roof')} ({head})", f"{tr('% of sustained roof')} ({head})"], out)


def device_table(tr, dev) -> str:
    return table([tr("Metric"), tr("Value")], [
        [tr("Peak at NVML max clock"), f"{dev['peak'] / 1e3:.1f} TFLOPS @ {dev['clock']:.0f} MHz"],
        [tr("Median SM clock under load"), f"{sustained_clock():.0f} MHz"],
        [tr("Peak at the sustained clock"), f"{dev['sustained'] / 1e3:.1f} TFLOPS"],
        [tr("Peak DRAM bandwidth"), f"{dev['bw']:.0f} GB/s"],
    ])


def longseq_table(tr) -> str:
    lat: dict[tuple[str, str], list[float]] = {}
    mem: dict[tuple[str, str], str] = {}
    for r in (1, 2, 3):
        for row in rows_of(BASE / "csv" / f"run{r}" / "longseq_attention.csv"):
            assert row["verified"] == "PASS", row
            lat.setdefault((row["S"], row["impl"]), []).append(float(row["latency_ms"]))
            mem[(row["S"], row["impl"])] = row["peak_extra_mib"]
    diag = (BASE / "logs" / "diag_sdpa.log").read_text(encoding="utf-8", errors="replace")
    out = []
    for block in re.split(r"(?=S=\d+ _fused)", diag)[1:]:
        S = re.match(r"S=(\d+)", block).group(1)
        for impl in ("SDPA", "Triton", "CUDA"):
            err = re.search(rf"{impl}\s+max_abs_err_vs_FP64=(\S+)", block).group(1)
            v = lat[(S, impl)]
            out.append([S, impl, f"{st.median(v):.3f} ({min(v):.3f}–{max(v):.3f})", mem[(S, impl)], err])
        naive = re.search(r"peak_extra_mib=([\d.]+)", block).group(1)
        out.append([S, tr("Matrix-materializing"), "—", naive, "—"])
    return table([tr("S"), tr("Implementation"), tr("Latency (ms)"), tr("Extra VRAM (MiB)"), tr("Max error vs FP64")], out)


def wmma_table(tr) -> str:
    text = (BASE / "logs" / "wmma_final.log").read_text(encoding="utf-8").splitlines()
    assert all("VERIFY_WMMA_TILED=PASS" in x and "VERIFY_CUBLAS=PASS" in x for x in text)
    out = []
    for key, label, unit in (("WMMA_TILED_GFLOPS", "WMMA tiled", "GFLOPS"), ("CUBLAS_GFLOPS", "cuBLAS", "GFLOPS"),
                             ("WMMA_TILED_PCT_OF_CUBLAS", "WMMA % of cuBLAS", "%")):
        v = [float(re.search(rf"\b{key}=([\d.]+)", x).group(1)) for x in text]
        out.append([f"{label} ({unit})", f"{st.median(v):.1f}", f"{min(v):.1f}", f"{max(v):.1f}", str(len(v))])
    return table([tr("Metric"), tr("Median"), tr("Min"), tr("Max"), tr("Runs")], out)


def floor_table(tr) -> str:
    diag = (BASE / "logs" / "diag_timing.log").read_text(encoding="utf-8", errors="replace")
    block = diag.split("launch floor")[1]
    return table([tr("Implementation"), tr("Launch floor (ms)")],
                 [[i, re.search(rf"{i}\s+median ([\d.]+) ms", block).group(1)] for i in IMPLS])


def main() -> None:
    for lang, tr in (("en", lambda s: s), ("zh", lambda s: ZH.get(s, s))):
        dev = device()
        parts = {"device": device_table(tr, dev), "bench": benchmark_table(tr), "roof": roofline_table(tr, dev),
                 "longseq": longseq_table(tr), "wmma": wmma_table(tr), "floor": floor_table(tr)}
        text = "\n\n".join(f"<!-- {k} -->\n{v}" for k, v in parts.items())
        (BASE / f"tables_{lang}.md").write_text(text + "\n", encoding="utf-8")
    print((BASE / "tables_en.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
