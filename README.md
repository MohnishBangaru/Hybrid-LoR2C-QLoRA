# Hybrid LoR2C + QLoRA

Parameter-efficient fine-tuning that combines **LoR2C residual adapters** (one low-rank branch
bridging each decoder layer) with attention-level **LoRA/QLoRA**, optional **quantization-aware
training** of the residual adapters, and a vision-language variant for **SmolVLM** captioning.

The codebase is a small, typed Python package with a strict layered architecture; see
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"                          # core + tooling (CPU tests)
pip install -e ".[huggingface,vision,tracking]"  # real training runs
```

Requires Python 3.11+. No patched `peft` fork is needed anymore; upstream `peft>=0.10` is used.

## Run

```bash
lor2c causal configs/tinyllama.yaml       # LoR2C + LoRA on TinyLlama / Alpaca
lor2c causal configs/tinyllama_qat.yaml   # same, with QAT on the residual adapters
lor2c vision configs/smolvlm.yaml         # low-rank adapters on SmolVLM captions
lor2c -v causal configs/tinyllama.yaml    # debug logging
```

Every run is described by one YAML file validated against `lor2c.settings.schema`. Unknown keys
are rejected. Outputs land in `outputs/<run>/` (`model/` and `residual/`).

Docker:

```bash
docker build -t lor2c .
docker run --gpus all -v $PWD/outputs:/workspace/outputs lor2c causal configs/tinyllama.yaml
```

## Develop

```bash
make check     # ruff format --check, ruff check, mypy --strict, pytest --cov
make format    # apply formatting and safe fixes
pre-commit install
```

Tests are class-based and mirror `src/` under `tests/`; the domain and application layers are
tested on CPU with in-memory fakes, no model downloads.

## Repository layout

```
src/lor2c/           package (domain / application / settings / infrastructure / cli)
tests/               mirrors src/
configs/             run definitions (YAML)
docs/                architecture notes and result figures
artifacts/adapters/  trained adapter checkpoints (Git LFS)
notebooks/           research notebooks (provenance only)
legacy/              original prototype scripts and the patched peft fork they need
```

## Results (research phase)

7B vs 1B comparison from the prototype runs:

![metrics table](docs/assets/metrics_table.png)

![7B vs 1B metrics](docs/assets/metrics_7b_vs_1b.png)

LoR2C adapter preview:

![LoR2C preview](docs/assets/lor2c_preview.png)

## License

MIT, see [LICENSE](LICENSE).
