# Tutorial 05: FlashAttention

**FlashAttention** computes attention **without materializing the full S×S score matrix** in high-bandwidth memory (HBM). It uses **tiling** and **online softmax** so that each block of the output is computed from small blocks of Q, K, V in fast memory (SRAM / shared memory), reducing memory traffic and enabling longer sequences.

---

## Concepts

### 1. Standard attention and its cost

Attention is:

- **Scores:** \( S = Q K^T / \sqrt{d} \)  (shape `[B, H, N, M]` or `[..., seq_q, seq_k]`).
- **Weights:** \( P = \mathrm{softmax}(S) \)  (same shape).
- **Output:** \( O = P V \)  (shape `[B, H, N, d]`).

The naive approach stores the full **S** (and often **P**) in HBM: **O(S²)** memory and **O(S²)** bytes moved. For long sequences this is slow and can run out of memory.

### 2. Tiling

- Split the **query** dimension into blocks of size **BLOCK_M** and the **key/value** dimension into blocks of size **BLOCK_N**.
- For each block of **Q** (BLOCK_M rows), iterate over blocks of **K** and **V** (BLOCK_N rows).
- Load **Q_block** (BLOCK_M × d), **K_block** (BLOCK_N × d), **V_block** (BLOCK_N × d) into fast memory.
- Compute the **block** of scores: `S_block = Q_block @ K_block^T` (BLOCK_M × BLOCK_N).
- Apply **online softmax** (see below) and accumulate the contribution to the output block **O_block** (BLOCK_M × d).
- We **never** hold the full S or P in HBM — only small blocks in SRAM/shared memory.

### 3. Online softmax

Softmax over the full row would need the full row of S. We don’t have it all at once. **Online softmax** maintains, for each row, a running **max** and a running **sum of exp** as we process blocks of the row:

- When we see a new block of scores `S_ij`, update:
  - `m_new = max(m_old, max(S_ij))`
  - Rescale previous accumulator: `acc *= exp(m_old - m_new)`; then add `exp(S_ij - m_new)` for the new block.
- After processing all blocks, each row has a correct `max` and `sum(exp)`; we can either output normalized values in a second pass or fuse the normalization into the accumulation (as in the reference implementation).

This is numerically stable and avoids storing the full S.

### 4. Causal mask

For decoder-only (causal) attention, position `i` must not attend to positions `j > i`. When processing a block of keys at `start_n`, set scores for `j > i` to `-inf` before softmax (e.g. with a mask in the Triton/CUDA kernel).

---

## Simplified code (Triton-style idea)

Conceptually, one program per (block of Q rows, batch*head):

```python
# Pseudocode / conceptual
for start_n in range(0, seq_len_k, BLOCK_N):
    q = load(Q_block)   # BLOCK_M x d
    k = load(K_block)   # BLOCK_N x d
    v = load(V_block)   # BLOCK_N x d
    qk = q @ k.T * scale   # BLOCK_M x BLOCK_N
    if causal:
        qk = mask_future(qk, start_n)
    # Online softmax: update m_i, l_i (running max and sum), accumulate into acc
    m_ij = max(qk, dim=1)
    p = exp(qk - m_ij)
    l_ij = sum(p, dim=1)
    # Rescale previous acc by exp(m_i - m_ij), then add p @ v
    acc = acc * exp(m_i - m_ij) + p @ v
    m_i, l_i = m_ij, l_i * exp(m_i - m_ij) + l_ij
# Normalize: acc /= l_i
store(O_block, acc)
```

The real implementation keeps `m_i`, `l_i`, and `acc` in registers/SRAM and does the rescaling and accumulation in one pass.

---

## Implementations in this repository

| Implementation | Location | Description |
|----------------|----------|-------------|
| **Triton FlashAttention** | [triton_kernels/flash_attention/flash_attention.py](../triton_kernels/flash_attention/flash_attention.py) | Full fused kernel: tiled Q,K,V load, online softmax, causal option; benchmark vs SDPA. |
| **FlashAttention (simple)** | [flash_attention_simple/](../flash_attention_simple/) | PyTorch reference + CUDA + Triton wrapper; benchmarks and usage. |
| **CUDA kernel** | [flash_attention_simple/flash_attention_cuda.cu](../flash_attention_simple/flash_attention_cuda.cu) | Tiled CUDA kernel with shared memory and online softmax (D ≤ 64). |
| **PyTorch reference** | [flash_attention_simple/](../flash_attention_simple/) | `reference_pytorch.py` — full S×S matrix for correctness comparison. |

**Build & benchmark (CUDA standalone, optional):**

```bash
cd flash_attention_simple && build.bat
python flash_attention_simple/benchmark_flash_attention.py
```

**Docs:** [docs/flashattention_algorithm.md](../docs/flashattention_algorithm.md), [flash_attention_simple/README.md](../flash_attention_simple/README.md).
