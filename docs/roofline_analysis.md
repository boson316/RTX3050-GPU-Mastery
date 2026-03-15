# Roofline Analysis

This document describes the **roofline model** and how it is used in this repository to classify transformer kernels as **memory-bound** or **compute-bound**, and to plot **achieved performance** vs **arithmetic intensity**. Implementation: `profiling/roofline_analysis/plot_roofline.py`; target GPU: **RTX 3050**.

---

## 1. Algorithm (roofline model)

The roofline model bounds **achievable performance** (e.g. in GFLOPS) by two ceilings:

1. **Memory ceiling:** Performance ≤ (memory bandwidth in GB/s) × (arithmetic intensity in FLOP/byte).  
   So: **perf = BW × AI** when limited by memory.

2. **Compute ceiling:** Performance ≤ peak FLOPS (e.g. FP16 TFLOPS).  
   So: **perf = peak_FLOPS** when limited by compute.

**Roofline curve:** For each arithmetic intensity *AI*, the maximum performance is  
**min(BW × AI, peak_FLOPS)**.

- **Ridge point:** AI_ridge = peak_FLOPS / BW.  
  - If **AI < AI_ridge** → kernel is **memory-bound** (performance lies on the slope).  
  - If **AI > AI_ridge** → kernel is **compute-bound** (performance lies on the flat roof).

**Arithmetic intensity:** AI = (total FLOPs) / (total bytes read + written from global memory).

---

## 2. GPU memory access and roofline

- **Bytes** in the denominator are the **traffic to/from global memory** (or the dominant level you model). Tiling and shared memory **reduce** effective bytes (reuse), so they **increase** AI and can move a kernel from memory-bound to compute-bound.
- **FLOPs** are the useful floating-point operations (e.g. matmul: 2·M·N·K for C = A×B).

So roofline does not describe “access pattern” per se; it uses **total FLOPs and total bytes** to get AI and then compares achieved performance to the roof.

---

## 3. Kernel launch configuration

Roofline plotting does not change kernel launches. The script:

- Uses **theoretical** FLOPs and bytes per kernel (from B, S, H and formulas in the code).
- Optionally uses **measured** duration or achieved GFLOPS from `profiling/nsight_reports/<prefix>_kernels_metrics.json` (from Nsight Compute).
- Plots one point per **logical kernel** (QKV, Softmax, LayerNorm, GELU, MLP), not per launch.

---

## 4. Optimization techniques (interpretation)

- **Below the roofline:** Kernel is not reaching the theoretical ceiling; possible causes: low occupancy, warp divergence, poor coalescing, or wrong FLOP/byte count.
- **On the slope (memory-bound):** Optimize memory: coalescing, tiling, recomputation to reduce bytes.
- **On the flat (compute-bound):** Optimize compute: occupancy, tensor cores, better block size.
- **Ridge point (RTX 3050 FP16):** ~9 TFLOPS / 192 GB/s ≈ **47 FLOP/byte**. Kernels with AI &lt; 47 are memory-bound; AI &gt; 47 compute-bound.

---

## 5. Benchmark results (plots and numbers)

After running:

```bash
python profiling/roofline_analysis/plot_roofline.py
```

Output: **profiling/nsight_reports/roofline_model.png**

- **X-axis:** Arithmetic intensity (FLOP/byte), log scale.
- **Y-axis:** Performance (GFLOPS), log scale.
- **Black curve:** Roofline (slope BW, then flat at peak FLOPS).
- **Vertical dashed line:** Ridge (AI ≈ 46.9).
- **Blue circles:** Memory-bound kernels (e.g. Softmax, LayerNorm, GELU).
- **Orange squares:** Compute-bound kernels (e.g. QKV, MLP).

Theoretical FLOPs/bytes (from `plot_roofline.py`):

| Kernel | FLOPs (approx) | Bytes (approx) |
|--------|----------------|----------------|
| QKV (fused) | 2·M·H·H3 | M·H·2 + H·H3·2 + H3·2 + M·H3·2 |
| Softmax | 5·M·H | 2·M·H·2 |
| LayerNorm | 4·M·H | 2·M·H·2 + H·2·2 |
| GELU | 8·M·H | M·H·2·2 |
| MLP (fused) | 2·M·H·H4 + 2·M·H4·H + 8·M·H4 | (see code) |

M = B×S, H3 = 3*H, H4 = 4*H; 2 = FP16 element size in bytes.

---

## 6. Diagrams / tables

### Roofline curve (conceptual)

| Region | Condition | Performance |
|--------|-----------|-------------|
| Memory-bound | AI ≤ peak_FLOPS/BW | perf = BW × AI |
| Compute-bound | AI > peak_FLOPS/BW | perf = peak_FLOPS |

### RTX 3050 parameters (default in script)

| Parameter | Default | Env override |
|-----------|---------|--------------|
| Peak FP16 (GFLOPS) | 9000 | ROOFLINE_PEAK_GFLOPS |
| Memory BW (GB/s) | 192 | ROOFLINE_PEAK_GBPS |
| Ridge (FLOP/byte) | 9000/192 ≈ 46.9 | — |

### Data flow in script

```
theoretical_flops_bytes(B, S, H)  →  (label, flops, bytes) per kernel
         ↓
load_nsight_metrics(prefix)       →  optional duration/achieved from JSON
         ↓
kernel_to_roofline_point(...)      →  (intensity, perf, label, is_memory_bound)
         ↓
plot_roofline()                    →  roofline curve + scatter points → PNG
```

---

## 7. Code snippets (from this repo)

### Theoretical FLOPs and bytes (QKV, Softmax, LayerNorm, GELU, MLP)

```python
# profiling/roofline_analysis/plot_roofline.py
def theoretical_flops_bytes(B: int, S: int, H: int) -> list[tuple[str, float, float]]:
    M = B * S
    H3 = H * 3
    H4 = H * 4
    results = []
    flops_qkv = 2 * M * H * H3
    bytes_qkv = M * H * 2 + H * H3 * 2 + H3 * 2 + M * H3 * 2
    results.append(("QKV (fused)", flops_qkv, bytes_qkv))
    # ... Softmax, LayerNorm, GELU, MLP
    return results
```

### Roofline point and classification

```python
def kernel_to_roofline_point(name, flops, bytes_total, duration_ns=None, achieved_gflops=None, ...):
    intensity = flops / bytes_total
    ridge = PEAK_GFLOPS_FP16 / PEAK_GBPS
    is_memory_bound = intensity < ridge
    if duration_ns and duration_ns > 0:
        perf = (flops / 1e9) / (duration_ns / 1e9)
    elif achieved_gflops is not None and achieved_gflops > 0:
        perf = achieved_gflops
    else:
        perf = min(PEAK_GBPS * intensity, PEAK_GFLOPS_FP16)
    return (intensity, perf, name, is_memory_bound)
```

### Roofline curve construction

```python
ridge = PEAK_GFLOPS_FP16 / PEAK_GBPS
for x in [0.01, 0.02, ...]:
    y = min(PEAK_GBPS * x, PEAK_GFLOPS_FP16)
    curve.append((x, y))
```

---

## References

- [gpu_memory_hierarchy.md](gpu_memory_hierarchy.md) — why AI and BW matter.
- [optimization_guide.md](optimization_guide.md) — what to do for memory-bound vs compute-bound.
- [nsight_profiling_guide.md](nsight_profiling_guide.md) — how metrics are collected for optional achieved perf.
- [memory_hierarchy_diagrams.md](memory_hierarchy_diagrams.md) — roofline diagram sketch.
