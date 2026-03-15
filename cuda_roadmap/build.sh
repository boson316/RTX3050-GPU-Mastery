#!/bin/bash
# Build all roadmap CUDA benchmarks. Requires nvcc in PATH.
set -e
NVCC="${NVCC:-nvcc}"
ARCH="${CUDA_ARCH:-sm_86}"

build() {
    local src="$1"
    local dir=$(dirname "$src")
    local name=$(basename "$src" .cu)
    echo "Building $dir/$name ..."
    $NVCC -arch=$ARCH -o "$dir/${name}" "$src" -allow-unsupported-compiler 2>/dev/null || $NVCC -arch=$ARCH -o "$dir/${name}" "$src"
}

build level1_basics/vector_add/vector_add_bench.cu
build level1_basics/reduction/reduction_bench.cu
build level1_basics/naive_matmul/naive_matmul_bench.cu
build level2_memory/tiled_matmul/tiled_matmul_bench.cu
build level2_memory/coalescing/coalescing_bench.cu
build level2_memory/bank_conflict/bank_conflict_bench.cu
build level3_advanced/warp_shuffle_reduction/warp_shuffle_bench.cu
build level3_advanced/fused_ops/fused_ops_bench.cu
build level3_advanced/persistent_kernel/persistent_kernel_bench.cu
# Level 4 requires sm_70+ for WMMA
build level4_tensor_core/fp16_tensor_core_matmul/fp16_matmul_bench.cu
build level4_tensor_core/wmma_example/wmma_example.cu

echo "Done. Run: python run_benchmarks.py"
