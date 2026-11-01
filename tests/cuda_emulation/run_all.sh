#!/usr/bin/env bash
# Build and run every CUDA-kernel emulation test on the CPU (no GPU needed). Used by CI.
# A kernel that deadlocks (e.g. a barrier or full-mask shuffle not reached by every thread)
# is reported as a failure via the timeout.
set -euo pipefail
cd "$(dirname "$0")"
CXX="${CXX:-g++}"
OUT="${TMPDIR:-/tmp}/cuda_emu_tests"
mkdir -p "$OUT"
status=0
for src in test_*_emu.cpp; do
    name="${src%.cpp}"
    echo "=== $name"
    "$CXX" -std=c++20 -O2 -pthread -Wall -Wextra -Wno-unknown-pragmas -Wno-unused-parameter "$src" -o "$OUT/$name"
    if ! timeout 300 "$OUT/$name"; then
        echo "!!! $name failed (or timed out = deadlock)"
        status=1
    fi
done
exit $status
