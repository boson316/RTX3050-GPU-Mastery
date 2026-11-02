"""Write results/v2/SUMMARY.md. Tables come from make_tables.py; the commit list comes from git."""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

BASE = Path("results/v2")
text = (BASE / "tables_zh.md").read_text(encoding="utf-8")
parts = re.split(r"<!-- (\w+) -->\n", text)
t = {parts[i]: parts[i + 1].strip() for i in range(1, len(parts), 2)}
log = subprocess.run(["git", "log", "--oneline", "origin/main..HEAD"], capture_output=True, text=True, check=True).stdout.strip()
specs = re.search(r"## python benchmarks/gpu_specs.py\n```\n(.*?)```", (BASE / "environment.md").read_text(encoding="utf-8"), re.S).group(1).strip()

(BASE / "SUMMARY.md").write_text(f"""# RTX3050-GPU-Mastery v2：實測摘要

## 環境

驅動程式 610.62、CUDA 12.4（nvcc 12.4.99）、PyTorch 2.6.0+cu124、Triton 3.2.0、Python 3.11.0、MSVC 14.50（VS 2026）。完整內容見 [environment.md](environment.md)。

```
{specs}
```

## Phase 1：正確性

最終程式碼上的結果：

| 步驟 | 結果 | Log |
|---|---|---|
| ruff | 通過 | [ruff.log](logs/ruff.log) |
| pytest（GPU，含 Triton、autotune、三個 extension、計時工具） | 116 / 116 | [pytest.log](logs/pytest.log) |
| pytest（模擬 CI：CPU 版 torch、最新 Triton；僅 Windows） | 53 通過、17 skip、0 失敗 | [pytest_ci_like.log](logs/pytest_ci_like.log) |
| CUDA CPU 模擬器（MSVC） | 27 / 27 | [emulator_msvc.log](logs/emulator_msvc.log) |
| CUDA roadmap | 11 / 11 PASS | [roadmap.log](logs/roadmap.log) |
| FlashAttention standalone | 8 / 8 PASS | [fa_standalone.log](logs/fa_standalone.log) |
| 三個 PyTorch extension | 全部 OK | [check_extensions.log](logs/check_extensions.log) |

模擬器修改紀錄：`cuda_emu.h` 這次沒有改，所以沒有重跑 g++ 和 clang。

## 關鍵數字

延遲格式為中位數（3 輪最小值～最大值），單位 ms。來源：`csv/final/benchmark_*_results.csv`、`csv/final/latency_ranges.json`。

{t['bench']}

裝置峰值。來源：`environment.md`、`logs/*_run[123]_clocks.csv`。

{t['device']}

Roofline。來源：`csv/run[123]/roofline_results.csv`。

{t['roof']}

長序列 FlashAttention。來源：`csv/run[123]/longseq_attention.csv`、`logs/diag_sdpa.log`。

{t['longseq']}

WMMA 與 cuBLAS。來源：`logs/wmma_final.log`、`logs/phase4_wmma_2048_baseline.log`。

{t['wmma']}

發送下限。來源：`logs/diag_timing.log`。

{t['floor']}

CUDA roadmap 全表見 [`cuda_roadmap/results.md`](../../cuda_roadmap/results.md)。

## 新發現並修正的 bug

詳見 [`docs/ENGINEERING_LOG.md`](../../docs/ENGINEERING_LOG.md)。

- **#13** FlashAttention JIT 在目前的 MSVC 上編譯失敗（少了 `-allow-unsupported-compiler`）。
- **#14** 照 README 安裝會把 CUDA 版 PyTorch 換成 CPU 版。只修了文件，沒有自動化測試。
- **#15** 計時前沒有拉高時脈，同一個 kernel 最多差 16 倍。修正前的資料隔離在 `csv/pre_fix_run*`，沒有採用。
- **#16** Roofline 沒有驗證、默默丟掉失敗項目，還把 `torch.randn` 算進延遲。
- **#17** Dashboard 用寫死的「~520x」舊數字蓋掉圖；另外兩張圖的計時也沒有驗證。

## Phase 4：優化嘗試

三項都沒有改動 kernel，結論見 [`CHANGELOG.md`](../../CHANGELOG.md)。

- **A. WMMA：** 2048³ 已達 cuBLAS 的 79.1～86.5%，超過 60% 目標，所以沒有嘗試 `cp.async` 和 128×128 tile。
- **B. Triton softmax：** 單次讀取最多快約 1%，已 revert。Log：`logs/phase4_softmax_run*.log`。
- **C. Triton FlashAttention：** 沒有候選設定明顯勝過現有 autotune。Log：`logs/phase4_attention_run*.log`。

## 沒做完或有疑慮的地方

- **WMMA 在 1024³ 的分群現象原因不明。** 整次執行會落在約 7.5 或約 10.9 TFLOPS，cuBLAS 也偶爾偏低。1 秒預熱無法消除。
- **0.1 ms 以下的 kernel 仍有雜訊。** 修正後仍可能相差超過 5%，最終 3 輪比較有 24 筆超過 5%（`logs/compare_runs.log`）。這些項目都有跑第 3 輪並附上範圍。
- **C++ 計時沒有預熱。** CUDA roadmap 和 standalone 的數字仍用原本的 warm-up，因為加了預熱也沒有改善。FlashAttention standalone 的時間也可能因此偏慢。
- **編譯器超出支援範圍。** MSVC 14.50 超出 CUDA 12.4 的支援範圍，靠 `-allow-unsupported-compiler` 編譯。所有驗證都通過，但這不是官方支援的組合。
- **Triton 版本取決於環境。** GPU 環境要用 3.2 才符合 PyTorch 2.6；沒有 GPU 時 3.2 會在 import 階段失敗，需要較新版本。
- **CI 尚未實際跑過。** CI 用 Linux 和不鎖版本的 Triton，這裡只在 Windows 上模擬過 pytest 步驟。
- **Nsight 沒有執行**（`ncu` 不在 PATH）。瓶頸解釋來自 roofline，不是硬體計數器。
- **`mnist_acc_loss.png` 沒有重新產生**（全程使用 `--skip-mnist`），README 也沒有引用。
- **`docs/benchmarks.md` 保留舊數字**，頁首已標明是歷史紀錄。
- **`profiling/roofline_analysis/plot_roofline.py` 只更新了 fallback 值**，這次沒有執行。
- **量測條件。** 插電和最佳效能模式有確認；是否有其他程式佔用 GPU，只在開始時檢查過一次。

## 尚未推送的 commit

`git log --oneline origin/main..HEAD`（不含加入本檔的最後一個 commit）：

```
{log}
```
""", encoding="utf-8")
