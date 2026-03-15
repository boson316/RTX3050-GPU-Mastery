# 上傳到 GitHub

## 目前專案對應你想要的結構

| 你要的目錄 | 本 repo 實際路徑 |
|------------|------------------|
| **cuda/** | `cuda_roadmap/`、`gpu_kernels/`（vector_add, reduction, matrix_mul, conv2d） |
| **triton/** | `triton_kernels/` |
| **transformer/** | `gpu_kernels/transformer/`、`flash_attention_simple/` |
| **pytorch_extensions/** | `extension/`、`pytorch_extensions/` |
| **benchmarks/** | `benchmarks/` |
| **profiling/** | `profiling/` |
| **docs/** | `docs/` |
| **tutorials/** | `tutorials/` |
| **notebooks/** | `notebooks/` |
| **tools/** | `tools/` |
| **README.md** | `README.md` |

*不需改資料夾名稱也能上傳；README 已說明實際結構。*

---

## 步驟 1：加入變更並提交

在專案根目錄（`RTX3050-GPU-Mastery`）執行：

```bash
# 加入所有新檔與修改（.gitignore 會排除 .venv、build、*.exe 等）
git add .

# 檢查即將提交的檔案
git status

# 提交
git commit -m "Add tutorials, tools, benchmarks, docs; improve README for portfolio"
```

---

## 步驟 2：在 GitHub 建立 repo（若尚未建立）

1. 登入 https://github.com
2. 右上角 **+** → **New repository**
3. Repository name 填：`RTX3050-GPU-Mastery`（或你要的名稱）
4. 選 **Public**
5. **不要**勾選 "Add a README"（你本地已有）
6. 按 **Create repository**

---

## 步驟 3：設定 remote 並推送

若這是**第一次**推送到這個 GitHub repo：

```bash
# 若已有 origin 但想換網址，可先移除再加：
# git remote remove origin

git remote add origin https://github.com/你的帳號/RTX3050-GPU-Mastery.git

git branch -M main
git push -u origin main
```

若 **origin 已存在**（例如之前就設過），只要推送：

```bash
git push -u origin main
```

---

## 步驟 4：之後再更新

改完程式後：

```bash
git add .
git commit -m "簡短說明這次改動"
git push
```

---

## 若不想上傳某些檔案

先從暫存移除，再寫進 `.gitignore`：

```bash
git reset HEAD 某個路徑或檔案
# 然後在 .gitignore 加上該路徑
```

目前 `.gitignore` 已排除：`.venv/`、`build/`、`*.exe`、`*.nsys-rep`、`*.ncu-rep`、`data/` 等。

---

## Rebase 衝突排除（push 被拒時）

若執行 `git pull --rebase origin main` 後出現 **"You must edit all merge conflicts"**，而實際上 README 等檔案的衝突已手動解決，常見原因是：**有檔案「已暫存但工作區還有修改」**（例如 `gpu_kernels/transformer/transformer_kernels.cu` 的換行符 CRLF），Git 仍視為未完全解決。

**處理方式（CMD）：**

```cmd
cd /d c:\Users\User\Documents\code\cursor\RTX3050-GPU-Mastery

git add -A

set GIT_EDITOR=true
git rebase --continue

git push origin main
```

- **`git add -A`** — 把所有變更（含該檔案）加入暫存，讓 Git 認定衝突已處理完。
- **`set GIT_EDITOR=true`** — 讓 `git rebase --continue` 不跳出編輯器。
- 再執行 **`git rebase --continue`** 與 **`git push origin main`**。
