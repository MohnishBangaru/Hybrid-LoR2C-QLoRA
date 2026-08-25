# Changelog

## 0.1.0 - 2026-08-25

- Restructured the repository into an installable `lor2c` package with domain, application,
  settings, infrastructure and CLI layers.
- Replaced the patched `peft==0.5.0` fork with an in-repo `AdapterBank` + `ResidualRouter`
  implementation of LoR2C running on upstream `peft`.
- Added YAML-driven configuration validated by Pydantic, a `lor2c` CLI, JSON logging,
  pluggable experiment tracking, and Git LFS for checkpoints.
- Fixed zero-initialised adapters in the SmolVLM prototype (adapters could never train) and
  QAT conversion without quant/dequant stubs.
- Added class-based unit tests, ruff, mypy strict, pre-commit and GitHub Actions CI.
- Moved prototypes to `legacy/`, notebooks to `notebooks/`, figures to `docs/assets/`, and the
  trained TinyLlama adapter to `artifacts/adapters/tinyllama-qat/`.
