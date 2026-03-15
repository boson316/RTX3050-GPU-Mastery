@echo off
cd /d "%~dp0"
:: 直接跑 benchmark（預設已是低負載、不跑自訂 CUDA）。若尚未建置 extension 請先執行 run_transformer_benchmark.bat
python benchmarks/transformer_benchmark.py
