"""
Build transformer CUDA extension (FP16 kernels).
Requires: PyTorch, CUDA toolkit, C++ compiler (VS on Windows), Ninja.

  pip install --no-build-isolation -e .
  or: python setup.py build_ext --inplace
"""
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="transformer_cuda",
    ext_modules=[
        CUDAExtension(
            "_transformer_cuda_native",
            ["transformer_kernels.cu", "transformer_cuda.cpp"],
            extra_compile_args={
                "nvcc": ["-O3", "--use_fast_math", "-arch=sm_86", "--allow-unsupported-compiler"],
                "cxx": ["/O2"] if __import__("sys").platform == "win32" else ["-O3"],
            },
            extra_link_args=["cublas.lib"] if __import__("sys").platform == "win32" else ["-lcublas"],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
