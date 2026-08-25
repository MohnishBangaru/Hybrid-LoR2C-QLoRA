# Hybrid LoR2C + QLoRA

[![CI](https://github.com/MohnishBangaru/Hybrid-LoR2C-QLoRA/actions/workflows/ci.yml/badge.svg)](https://github.com/MohnishBangaru/Hybrid-LoR2C-QLoRA/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Parameter-efficient fine-tuning that combines **LoR2C residual adapters** (one low-rank branch
bridging each decoder layer) with attention-level **LoRA/QLoRA**, optional **quantization-aware
training** of the residual adapters, and a vision-language variant for **SmolVLM** captioning.

`lor2c` is a small, fully typed Python package with a strict layered (hexagonal) architecture.
Architecture notes live in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md); contribution rules in
[CONTRIBUTING.md](CONTRIBUTING.md); release notes in [CHANGELOG.md](CHANGELOG.md).

## How LoR2C works

```
decoder layer i:   x_i ──► block_i ──► h_i ──(+)──► h'_i
                   │                          ▲
                   └── floor{i+1}: up(down(x_i)) * alpha / r
```

For every decoder layer a rank-`r` adapter reads the layer *input* and adds its scaled delta to
the layer *output*, so each block gains a trainable low-rank residual path without touching the
frozen weights. Attention projections (`q_proj`, `v_proj`) additionally receive standard LoRA via
`peft`. With `quantization.enabled: true` the residual adapters are trained with fake
quantization and converted to int8 kernels at the end. The rank is chosen automatically from the
model width (`r=4, alpha=8` up to 2048 hidden units, `r=16, alpha=32` above) unless fixed in the
config.

The paper's variants ([arXiv 2503.00572](https://arxiv.org/abs/2503.00572)) are all available:

| Variant | Config | Mechanism |
|---|---|---|
| ShareLoR2C | `adapter.shared: true` | one down projection `A` shared by every layer, per-layer `B` |
| MergeLoR2C | `adaptation.merges: N` | the adjacent pair with the lowest information content (minimum SFS) is merged into one adapter spanning both layers, keeping the richer member's weights |
| InjectLoR2C | `adaptation.injections: N` | the single-layer adapter with the lowest SFS is removed and that layer's half-rank attention LoRA is unfrozen |
| IMLoR2C | both | merge + inject, spread over the first quarter of training |

Modern LoRA practice is layered on top: `model.quantization: nf4` loads the frozen base in 4-bit
(QLoRA), `adapter.scaling: stabilized` applies rsLoRA scaling `alpha/sqrt(r)`, `adapter.ratio`
gives the up/`B` projections a higher learning rate (LoRA+), and `adapter.dora: true` switches the
attention LoRA to DoRA.

## Install

Python 3.11 or newer.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"                          # package + lint/type/test tooling (CPU)
pip install -e ".[huggingface,vision,tracking]"  # transformers/peft/datasets, PIL/nltk, wandb
pip install -e ".[evaluate]"                     # lm-eval for `lor2c evaluate`
```

Upstream `peft>=0.10` is used; the patched fork the original prototypes required is no longer
needed (it is archived under `legacy/`).

## Run

```bash
lor2c causal configs/tinyllama.yaml       # LoR2C + LoRA on TinyLlama / Alpaca-cleaned
lor2c causal configs/tinyllama_qat.yaml   # same, with QAT on the residual adapters
lor2c causal configs/tinyllama_im.yaml    # IMLoR2C: merge + inject events, rsLoRA, LoRA+, DoRA
lor2c causal configs/tinyllama_qlora.yaml # NF4 4-bit base (QLoRA) + LoR2C + DoRA
lor2c evaluate configs/evaluate.yaml      # lm-eval benchmarks on a trained run -> scores.json
lor2c vision configs/smolvlm.yaml         # low-rank adapters on SmolVLM anime captions
lor2c -v causal configs/tinyllama.yaml    # debug logging
lor2c --help
```

Docker (CUDA runtime image):

```bash
docker build -t lor2c .
docker run --gpus all -v "$PWD/outputs:/workspace/outputs" lor2c causal configs/tinyllama.yaml
```

Set `WANDB_API_KEY` (see `.env.example`) when a config uses `tracking.kind: wandb`.

## Configuration

Each run is one YAML file validated against `lor2c.settings.schema`; unknown keys are rejected
with a clear error. The main sections:

| Section | Keys | Notes |
|---|---|---|
| top level | `name`, `seed`, `output`, `resume` | `output` receives `checkpoints/`, `model/`, `residual/` |
| `model` | `name`, `revision`, `precision`, `device`, `quantization` | `quantization`: `none` / `int8` / `nf4` (QLoRA) |
| `data` (causal) | `path`, `validation`, `cutoff`, `inputs`, `eos` | hub dataset or local `.json`/`.jsonl` |
| `data` (vision) | `path`, `samples`, `holdout`, `length`, `image`, `instruction` | |
| `adapter` | `mode`, `automatic`, `rank`, `alpha`, `dropout`, `targets`, `scaling`, `shared`, `dora`, `ratio` | `mode`: `base` (LoRA only), `residual` (LoR2C only, the paper's setting), `lor2c` (both); `scaling: stabilized` = rsLoRA; `shared` = ShareLoR2C; `ratio` = LoRA+ |
| `adaptation` (causal) | `merges`, `injections`, `top`, `span` | MergeLoR2C / InjectLoR2C events (IMLoR2C) |
| `train` | `epochs`, `batch`, `micro`, `rate`, `warmup`, ... | `batch` must be a multiple of `micro` |
| `quantization` | `enabled`, `backend` | `fbgemm` (x86) or `qnnpack` (ARM) |
| `evaluation` (vision) | `beams`, `tokens`, `patience` | BLEU-based early stopping |
| `tracking` | `kind`, `project`, `run` | `none` (JSON logs) or `wandb` |
| `benchmark` (evaluate) | `tasks`, `shots`, `limit`, `batch` | lm-eval task names |

See [configs/](configs/) for complete, commented examples.

## Outputs

```
outputs/<run>/
  checkpoints/   transformers.Trainer checkpoints (causal runs)
  model/         peft adapter (adapter_config.json + weights) or trainable state dict
  residual/      adapters.pt (AdapterBank state) + manifest.json (specs, final schedule)
  scores.json    written by `lor2c evaluate`
```

Logs are JSON lines on stderr; metric records carry `step` and a `values` map.

## Develop

```bash
make check       # ruff format --check, ruff check, mypy --strict, pytest --cov
make format      # apply formatting and safe fixes
pre-commit install
```

Tests are class-based and mirror `src/` under `tests/`. The domain and application layers are
exercised on CPU with in-memory fakes (no model downloads); CI runs the full suite on
Python 3.11 and 3.12.

## Repository layout

```
src/lor2c/
  domain/          adapter maths, residual routing, prompt/label/caption rules, early stopping
  application/     CausalTrainingService, VisionTrainingService, ports
  settings/        Pydantic settings schema + YAML loader
  infrastructure/  hooks, injection, quantization, persistence, logging, tracking, huggingface/
  cli/             `lor2c` entry point and composition root
tests/             mirrors src/
configs/           run definitions (YAML)
docs/              architecture notes and result figures
artifacts/         released adapter checkpoints (tinyllama-qat)
notebooks/         research notebooks (provenance only)
legacy/            original prototype scripts and the patched peft fork they need
```

## Status

- Domain, application, settings, hooks, injection, quantization, persistence and CLI are covered
  by unit tests and type-checked in CI.
- The Hugging Face adapters (`src/lor2c/infrastructure/huggingface/`) have not yet been executed
  end-to-end on a GPU host; running `lor2c causal configs/tinyllama.yaml` there is the next
  validation step.

## Results

First controlled comparison (TinyLlama-1.1B, LoRA vs LoR2C vs ShareLoR2C vs IMLoR2C, Colab T4):
all LoR2C variants converge about twice as fast and reach 3.5% lower validation loss, and
IMLoR2C does so at LoRA's parameter budget. Full tables, figures and caveats in
[docs/RESULTS.md](docs/RESULTS.md).

![validation loss vs parameters](docs/assets/results/efficiency.png)

## Results (research phase)

7B vs 1B comparison from the prototype runs:

![metrics table](docs/assets/metrics_table.png)

![7B vs 1B metrics](docs/assets/metrics_7b_vs_1b.png)

LoR2C adapter preview:

![LoR2C preview](docs/assets/lor2c_preview.png)

## License

MIT, see [LICENSE](LICENSE).
