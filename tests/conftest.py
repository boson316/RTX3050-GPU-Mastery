"""
Test configuration.

Without a CUDA GPU (e.g. GitHub Actions), Triton kernels run in the Triton *interpreter*
(TRITON_INTERPRET=1): the same kernel source is executed with NumPy on the CPU, so the
indexing, masking and math are checked exactly as written. Performance is irrelevant there.
With a GPU the same tests run the compiled kernels.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _has_cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        return False


HAS_CUDA = _has_cuda()
if not HAS_CUDA:
    # Must be set before `import triton`.
    os.environ.setdefault("TRITON_INTERPRET", "1")

import pytest  # noqa: E402

DEVICE = "cuda" if HAS_CUDA else "cpu"


@pytest.fixture
def device() -> str:
    return DEVICE


requires_cuda = pytest.mark.skipif(not HAS_CUDA, reason="needs a CUDA GPU")
