# Reproducing the results

Two reproduction targets exist:

1. **This repository's own results** (`docs/assets/`): TinyLlama-1.1B vs LLaMA-2-7B with LoR2C
   on Alpaca-cleaned, plus the QAT variant.
2. **The paper's Table II** (arXiv 2503.00572): LLaMA-2-7B, LoR2C r=8 vs LoRA r=8, IMLoR2C with
   4 merges + 4 injections, lr 3e-4, batch 128, 3 epochs.

Every run is fully described by one YAML file, so the commands below are the whole recipe.

## 1. Host

- Linux with a CUDA GPU. Memory guide (bf16 base, micro batch 4, cutoff 256):
  TinyLlama-1.1B: 8 GB; LLaMA-2-7B: 24 GB with `quantization: nf4`, 40 GB without.
- Python 3.11 or 3.12, `git`, and a Hugging Face token for gated models (`meta-llama/*`).

## 2. Environment

```bash
git clone https://github.com/MohnishBangaru/Hybrid-LoR2C-QLoRA.git
cd Hybrid-LoR2C-QLoRA
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu121   # match your CUDA
pip install -e ".[huggingface,vision,tracking,dev]"
export HF_TOKEN=...            # gated models only
export WANDB_API_KEY=...       # only if tracking.kind is wandb; otherwise set kind: none
make check                     # CPU sanity: lint, types, 108 unit tests
```

Or the container: `docker build -t lor2c . && docker run --gpus all -e HF_TOKEN -v $PWD/outputs:/workspace/outputs lor2c causal configs/tinyllama.yaml`.

## 3. Smoke test first (5 minutes)

Before any multi-hour run, prove the whole path works on your host with a tiny slice:

```bash
cat > /tmp/smoke.yaml <<'YAML'
name: smoke
output: outputs/smoke
model: {name: TinyLlama/TinyLlama-1.1B-Chat-v0.6, precision: bfloat16}
data: {path: yahma/alpaca-cleaned, validation: 64, cutoff: 128}
adapter: {mode: lor2c, automatic: false, rank: 4, alpha: 8}
adaptation: {merges: 1, injections: 1}
train: {epochs: 0.02, batch: 8, micro: 4, warmup: 2, evaluation: 20, logging: 1}
YAML
lor2c -v causal /tmp/smoke.yaml
ls outputs/smoke/model outputs/smoke/residual
```

Expected: JSON log lines including `"Applied LoRA"`, `"Merged adapters"`, `"Injected attention
LoRA"`, `"Training finished"`, and `outputs/smoke/residual/manifest.json` listing the surviving
adapters (one `floorX+floorY` entry, one floor missing).

## 4. Runs

| Goal | Command | Notes |
|---|---|---|
| Repo baseline, LoR2C on TinyLlama | `lor2c causal configs/tinyllama.yaml` | r=4/alpha=8 chosen automatically for hidden 2048 |
| Repo QAT variant | `lor2c causal configs/tinyllama_qat.yaml` | int8 kernels for the residual adapters; `backend: qnnpack` on ARM |
| Paper IMLoR2C on LLaMA-2-7B | `lor2c causal configs/llama2_7b_im.yaml` | paper hyper-parameters; set `quantization: none` for an exact match |
| Paper LoRA baseline | same config with `adapter.mode: base` and `adaptation: {}` | r=8 on q/v |
| Paper plain LoR2C | same config with `adaptation: {}` | |
| ShareLoR2C | add `adapter.shared: true` | |
| Modern stack | `lor2c causal configs/tinyllama_qlora.yaml` | nf4 + rsLoRA + LoRA+ + DoRA |
| SmolVLM captions | `lor2c vision configs/smolvlm.yaml` | BLEU early stopping |

Seeds are fixed (`seed: 42`); rerunning a config reproduces the same adapter initialisation
and data order. Outputs land in `outputs/<name>/` (`model/` = peft adapter, `residual/` =
LoR2C bank + manifest, `checkpoints/` = Trainer state). Training loss and eval loss are in the
logs and, with `tracking.kind: wandb`, in the W&B project.

## 5. Evaluating

```bash
pip install -e ".[evaluate]"            # EleutherAI lm-eval
lor2c evaluate configs/evaluate.yaml    # point `run:` at the training output directory
cat outputs/tinyllama-lor2c/scores.json
```

`lor2c evaluate` loads the base model, re-attaches the peft attention adapter from
`<run>/model/`, rebuilds the residual bank and its final routing from
`<run>/residual/manifest.json` (including merged spans and injected layers), installs the hooks,
and runs `lm_eval.simple_evaluate` on the configured tasks. One score per task (`acc_norm` when
available, else `acc`) is written to `<run>/scores.json` and logged to the tracker.

To compare variants, evaluate each run directory with the same `benchmark` block and the same
`seed`; keep `limit: null` for reportable numbers. Runs trained with `quantization.enabled: true`
(int8-converted residual adapters) cannot be evaluated this way yet.

## 6. Known differences from the paper

- The paper trains on fp16 weights; `nf4` is optional here and changes absolute numbers slightly.
- SFS scoring follows the paper's definition (see `docs/ARCHITECTURE.md`); the authors' public
  code merges in the opposite direction, so numbers from that code are not directly comparable.
- Alpaca-cleaned on the Hub is periodically updated; pin a dataset revision if exact row counts
  matter.
