"""
PyTorch CUDA Extension: custom 3x3 conv2d, C_in=1, C_out=32 (FP32/FP16 storage, FP32 accumulate, sm_86).
Build: pip install --no-build-isolation .  (requires CUDA + C++ compiler)
"""
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="custom_conv",
    ext_modules=[
        CUDAExtension(
            "custom_conv",
            ["conv_kernel.cu"],
            extra_compile_args={
                "nvcc": [
                    "-allow-unsupported-compiler",
                    "-gencode=arch=compute_86,code=sm_86",
                    "-O3",
                ],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
