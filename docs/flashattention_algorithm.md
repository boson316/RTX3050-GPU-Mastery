# FlashAttention Algorithm

This document explains the **FlashAttention** algorithm as implemented in this repository: **CUDA** (`flash_attention_simple/flash_attention_cuda.cu`) and **Triton** (`triton_kernels/flash_attention/flash_attention.py`). Goal: compute **attention(Q,K,V) = softmax(QK^T/√d)V** without materializing the full S×S attention matrix in global memory.

**Tiling diagram:** [images/flashattention_tiling.png](images/flashattention_tiling.png) (from `python docs/scripts/generate_diagrams.py`)

---

## 1. Algorithm

**Standard attention:** Compute P = softmax(QK^T/√d), then O = P V. This requires storing P (S×S), which is **O(S²)** memory and bandwidth.

**FlashAttention (tiled + online softmax):**

1. Partition Q into blocks of **BLOCK_M** rows, K and V into blocks of **BLOCK_N** rows.
2. For each block of Q rows:
   - Initialize running max **m_i** and running sum **l_i** (for online softmax) and accumulator **acc** for the output block.
   - For each block of K/V:
     - Load Q_tile [BLOCK_M, D], K_tile [BLOCK_N, D], V_tile [BLOCK_N, D].
     - Compute scores = Q_tile @ K_tile^T * scale (block of size BLOCK_M×BLOCK_N).
     - **Online softmax:** Update m_i = max(m_i, row_max(scores)), rescale previous acc by exp(m_i_old - m_i_new), add exp(scores - m_i_new) @ V_tile to acc, update l_i.
   - Write acc / l_i as the output block.

3. **No full S×S matrix** is ever written to global memory; only blocks in shared memory (CUDA) or SRAM (Triton) are used.

---

## 2. GPU memory access patterns

- **Q:** Each program loads one block of size BLOCK_M×D once; threads can load in a coalesced way along D.
- **K, V:** Loaded in a loop over key sequence in blocks of BLOCK_N×D; same coalescing along D.
- **Output:** One block BLOCK_M×D written per program; coalesced along D.
- **Scores (BLOCK_M×BLOCK_N):** Kept in shared memory (CUDA) or registers/SRAM (Triton); never written to global.

So the **total global memory** for the attention matrix is **O(S)** (one block at a time), not O(S²).

---

## 3. Kernel launch configuration

### CUDA (`flash_attention_simple/flash_attention_cuda.cu`)

- **Block:** 1D; size chosen to cover BLOCK_M×BLOCK_D and BLOCK_N×BLOCK_D loads (e.g. multiple of 32).
- **Grid:** 1D over (total_blocks_m * B * H), where total_blocks_m = ceil(S / BLOCK_M). Each block handles one (batch, head, Q-block) and loops over K/V blocks.

```c
// flash_attention_simple/flash_attention_cuda.cu
#define BLOCK_M 16
#define BLOCK_N 32
#define BLOCK_D 64
const int total_blocks_m = (S + BLOCK_M - 1) / BLOCK_M;
// blockIdx.x = bh * total_blocks_m + block_m_index
```

### Triton (`triton_kernels/flash_attention/flash_attention.py`)

```python
# Grid: (num_blocks_q, batch * num_heads)
grid = (triton.cdiv(seq_len_q, BLOCK_M), batch * num_heads)
_fwd_kernel[grid](q, k, v, o, ...)
```

- Each program corresponds to one Q block; internally it loops over K/V in steps of BLOCK_N.

---

## 4. Optimization techniques used

1. **Tiling:** Q in BLOCK_M rows, K/V in BLOCK_N rows; working set fits in shared memory (CUDA) or SRAM (Triton).
2. **Online softmax:** Running max and running sum; rescale previous accumulator when max changes so that we never need two passes over the full row. Enables O(1) memory per row instead of O(S).
3. **Kernel fusion:** Load Q,K,V → scores → softmax → multiply by V → accumulate in one kernel; no separate softmax or matmul kernels.
4. **Coalesced loads/stores:** All pointers use strides so that contiguous threads access contiguous elements (along D).
5. **Causal masking (optional):** In Triton, `IS_CAUSAL` sets scores to -inf where row index &lt; col index before softmax.
6. **FP32 accumulation:** Softmax and accumulator in float; output cast to input dtype.

---

## 5. Benchmark results

From `flash_attention_simple/benchmark_flash_attention.py` (B=2, H=8, D=64, float32):

| Seq length | PyTorch (ms) | CUDA (ms) | Triton (ms) |
|------------|--------------|-----------|-------------|
| 128        | —            | —         | —           |
| 256        | —            | —         | —           |
| 512        | —            | —         | —           |
| 1024       | —            | —         | —           |

(Run `python flash_attention_simple/benchmark_flash_attention.py` on RTX 3050 for actual numbers. CUDA requires D≤64 in this implementation.)

---

## 6. Diagrams / tables

### Data flow (one Q block)

| Step | In global | In shared/SRAM | Operation |
|------|-----------|----------------|-----------|
| 1 | Q[block_m, :] | Q_tile | Load |
| 2 | K[block_n, :], V[block_n, :] | K_tile, V_tile | Load |
| 3 | — | scores = Q_tile @ K_tile^T * scale | Compute |
| 4 | — | m_i, l_i updated; acc += softmax(scores) @ V_tile | Online softmax + matmul |
| 5 | — | — | Repeat for next K/V block |
| 6 | O[block_m, :] | acc / l_i | Store |

### Memory complexity

| Approach | Global memory for P | Per-block |
|----------|----------------------|-----------|
| Standard | O(S²) | — |
| FlashAttention | O(1) for P (not stored) | O(BLOCK_M×BLOCK_N + BLOCK_M×D + BLOCK_N×D) in shared |

---

## 7. Code snippets (from this repo)

### Triton: online softmax and accumulate

```python
# triton_kernels/flash_attention/flash_attention.py
m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
for start_n in range(0, seq_len_k_padded, BLOCK_N):
    q = tl.load(q_ptrs, mask=...)
    k = tl.load(k_ptrs, mask=...)
    qk = tl.dot(q, k, allow_tf32=False) * softmax_scale
    if IS_CAUSAL:
        qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk, float("-inf"))
    m_ij = tl.maximum(m_i, tl.max(qk, 1))
    p = tl.exp(qk - m_ij[:, None])
    l_ij = tl.sum(p, 1)
    alpha = tl.exp(m_i - m_ij)
    l_i = l_i * alpha + l_ij
    p = p.to(Q.dtype.element_ty)
    v = tl.load(v_ptrs, mask=...)
    acc = acc * alpha[:, None]
    acc += tl.dot(p, v, allow_tf32=False)
    m_i = m_ij
    k_ptrs += BLOCK_N * stride_kn
    v_ptrs += BLOCK_N * stride_vn
acc = acc / l_i[:, None]
tl.store(o_ptrs, acc.to(...), mask=...)
```

### CUDA: shared memory layout

```c
// flash_attention_simple/flash_attention_cuda.cu
__shared__ float s_Q[BLOCK_M][BLOCK_D];
__shared__ float s_K[BLOCK_N][BLOCK_D];
__shared__ float s_V[BLOCK_N][BLOCK_D];
__shared__ float s_scores[BLOCK_M][BLOCK_N];
__shared__ float s_mi[BLOCK_M];
__shared__ float s_li[BLOCK_M];
```

---

## References

- [triton_kernel_design.md](triton_kernel_design.md) — Triton grid and block design.
- [transformer_gpu_kernels.md](transformer_gpu_kernels.md) — other transformer building blocks.
- FlashAttention paper: Dao et al., “FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.”
