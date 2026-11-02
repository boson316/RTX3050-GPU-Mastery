@echo off
cd /d "%~dp0..\..\cuda_roadmap\level4_tensor_core\fp16_tensor_core_matmul"
nvcc -O3 -std=c++17 -arch=sm_86 -DGEMM_SIZE=2048 -o fp16_2048 fp16_matmul_bench.cu -lcublas -allow-unsupported-compiler
exit /b %errorlevel%
