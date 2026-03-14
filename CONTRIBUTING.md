# Contributing to RTX3050-GPU-Mastery

Thanks for your interest. Contributions that keep benchmarks reproducible and docs clear are welcome.

## How to contribute

1. **Fork** the repo and create a branch (`git checkout -b feature/your-idea`).
2. **Keep style** consistent: Python (e.g. Ruff), CUDA (comment key kernels).
3. **Update docs** if you add benchmarks or change structure: `README.md`, `docs/benchmarks.md`.
4. **Open a Pull Request** with a short description and, if applicable, benchmark comparison (device + batch sizes).

## CI

- Push to `main`/`master` runs **lint** (Ruff) and a minimal **Python test** (no CUDA build in GitHub Actions).
- Building the CUDA extension requires a machine with CUDA Toolkit + C++ compiler (see `extension/README.md`).

This project is MIT licensed.
