"""
Theoretical peak FLOPS / DRAM bandwidth of the current GPU, derived from the device itself
instead of a hard-coded table.

    peak FLOPS = SMs x (FLOPs / clock / SM for the datapath) x max SM clock
    peak BW    = 2 (DDR) x max memory clock x bus width / 8

SM count comes from torch; clocks and bus width from NVML (pip install nvidia-ml-py).
Per-SM throughput per compute capability (dense, no sparsity):

    datapath                          sm_80 (A100)   sm_86 / sm_89 (GeForce Ampere / Ada)
    FP32 FMA (CUDA cores)               128 FMA          128 FMA          -> x2 FLOPs
    FP16 Tensor Core, FP32 accumulate   1024 FMA         256 FMA          -> x2 FLOPs
    FP16 Tensor Core, FP16 accumulate   1024 FMA         512 FMA          -> x2 FLOPs

(GeForce GA10x runs FP16 MMA with FP32 accumulation at half rate - see the GA102 whitepaper:
RTX 3090 = 71 TFLOPS FP32-acc vs 142 TFLOPS FP16-acc.)

Why it matters: "RTX 3050 Laptop" is several SKUs. The original has 16 SMs and a 128-bit bus;
the 2023 "RTX 3050 6GB Laptop" refresh has 20 SMs and a 96-bit bus. A hard-coded 9 TFLOPS peak
put measured cuBLAS points *above* the roofline in an earlier version of this repo.

Run:  python benchmarks/gpu_specs.py
"""
from __future__ import annotations

import os
from dataclasses import dataclass

# FLOPs per clock per SM (FMA counted as 2 FLOPs)
_FLOPS_PER_CLK_PER_SM = {
    (8, 0): {"fp32": 256, "fp16_tc_fp32acc": 2048, "fp16_tc_fp16acc": 2048},
    (8, 6): {"fp32": 256, "fp16_tc_fp32acc": 512, "fp16_tc_fp16acc": 1024},
    (8, 9): {"fp32": 256, "fp16_tc_fp32acc": 512, "fp16_tc_fp16acc": 1024},
    (7, 5): {"fp32": 128, "fp16_tc_fp32acc": 512, "fp16_tc_fp16acc": 1024},
}


@dataclass
class GpuSpecs:
    name: str
    sm_count: int
    compute_capability: tuple[int, int]
    sm_clock_mhz: float | None
    mem_clock_mhz: float | None
    bus_width_bits: int | None
    l2_bytes: int | None

    def peak_tflops(self, datapath: str = "fp16_tc_fp32acc") -> float | None:
        per = _FLOPS_PER_CLK_PER_SM.get(self.compute_capability, {}).get(datapath)
        if per is None or self.sm_clock_mhz is None:
            return None
        return self.sm_count * per * self.sm_clock_mhz * 1e6 / 1e12

    def peak_dram_gbs(self) -> float | None:
        if self.mem_clock_mhz is None or self.bus_width_bits is None:
            return None
        return 2 * self.mem_clock_mhz * 1e6 * self.bus_width_bits / 8 / 1e9

    def ridge_point(self, datapath: str = "fp16_tc_fp32acc") -> float | None:
        f, b = self.peak_tflops(datapath), self.peak_dram_gbs()
        return f * 1e3 / b if f and b else None


def query(device: int = 0) -> GpuSpecs | None:
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    props = torch.cuda.get_device_properties(device)
    sm_clock = mem_clock = bus = None
    try:
        import pynvml

        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(device)
        sm_clock = float(pynvml.nvmlDeviceGetMaxClockInfo(h, pynvml.NVML_CLOCK_SM))
        mem_clock = float(pynvml.nvmlDeviceGetMaxClockInfo(h, pynvml.NVML_CLOCK_MEM))
        bus = int(pynvml.nvmlDeviceGetMemoryBusWidth(h))
        pynvml.nvmlShutdown()
    except Exception:  # noqa: BLE001 - NVML missing or unsupported: fall back to env overrides
        pass
    sm_clock = float(os.environ.get("GPU_SM_CLOCK_MHZ", sm_clock or 0)) or None
    mem_clock = float(os.environ.get("GPU_MEM_CLOCK_MHZ", mem_clock or 0)) or None
    bus = int(os.environ.get("GPU_BUS_WIDTH_BITS", bus or 0)) or None
    return GpuSpecs(
        name=props.name,
        sm_count=props.multi_processor_count,
        compute_capability=(props.major, props.minor),
        sm_clock_mhz=sm_clock,
        mem_clock_mhz=mem_clock,
        bus_width_bits=bus,
        l2_bytes=getattr(props, "L2_cache_size", None),
    )


def main() -> None:
    s = query()
    if s is None:
        print("No CUDA device.")
        return
    print(f"{s.name}: {s.sm_count} SMs, sm_{s.compute_capability[0]}{s.compute_capability[1]}, "
          f"SM clock {s.sm_clock_mhz} MHz, mem clock {s.mem_clock_mhz} MHz, bus {s.bus_width_bits}-bit, "
          f"L2 {s.l2_bytes and s.l2_bytes // 1024} KB")
    for dp in ("fp32", "fp16_tc_fp32acc", "fp16_tc_fp16acc"):
        t = s.peak_tflops(dp)
        print(f"  peak {dp:17s}: {t:.1f} TFLOPS" if t else f"  peak {dp}: unknown (install nvidia-ml-py)")
    bw = s.peak_dram_gbs()
    print(f"  peak DRAM bandwidth : {bw:.0f} GB/s" if bw else "  peak DRAM bandwidth: unknown")
    r = s.ridge_point()
    if r:
        print(f"  ridge point (FP16 TC, FP32 acc): {r:.0f} FLOP/byte")


if __name__ == "__main__":
    main()
