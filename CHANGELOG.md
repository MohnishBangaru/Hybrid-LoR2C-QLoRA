# Changelog

## Unreleased

- Add `adapter.mode: residual` (LoR2C residual adapters without attention LoRA), the paper's
  own configuration, for parameter-matched comparisons against LoRA.
- Fix: the residual `AdapterBank` is now registered on the host model while hooks are attached,
  so the trainer's optimizer actually updates the LoR2C adapters.
- Add the paper's variants: ShareLoR2C (`adapter.shared`), MergeLoR2C and InjectLoR2C
  (`adaptation.merges` / `adaptation.injections`) driven by the SFS spectral score.
- Add QLoRA base quantization (`model.quantization: nf4|int8`), rsLoRA scaling
  (`adapter.scaling: stabilized`), LoRA+ learning-rate ratio (`adapter.ratio`) and DoRA on the
  attention LoRA (`adapter.dora`).
- Add `configs/tinyllama_im.yaml` and `configs/tinyllama_qlora.yaml`.
- Add `lor2c evaluate`: rebuilds a trained run (peft adapter + residual bank from a
  self-describing `manifest.json`) and scores it with EleutherAI lm-eval (`lor2c[evaluate]`).
- Verified end-to-end on a Colab T4 (transformers 5.x); fixes from that run: Hugging Face
  datasets accepted by `Split`, transformers v4/v5 compatibility (`group_by_length`, `dtype`,
  `AutoModelForImageTextToText`), residual bank moved to the model device on attach, injected
  adapters kept in float32 for AMP, adapters-only saving for non-peft models.
- Fix SmolVLM captioning: the assistant caption was passed as a plain string and rendered as
  empty by the chat template (the model learned to emit nothing); label masking now uses the
  prompt token length; evaluation generates from the user prompt instead of the full sample.

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
