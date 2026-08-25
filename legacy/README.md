# Legacy prototypes

Original research scripts kept verbatim for reference and reproducibility. They are **not**
part of the `lor2c` package, are excluded from linting/typing, and depend on a patched
`peft==0.5.0` fork shipped as `peft-0.5.0-fork.zip` (exports `MSLoraConfig`, `LlamaLoRALayer`).

| File | Purpose | Replaced by |
|------|---------|-------------|
| `prototype.py` | LoR2C on TinyLlama via forward hooks | `lor2c causal configs/tinyllama.yaml` |
| `quantization.py` | Same, with QAT on the residual adapters | `lor2c causal configs/tinyllama_qat.yaml` |
| `vlm/lor2c.py` | Low-rank adapters on SmolVLM | `lor2c vision configs/smolvlm.yaml` |
| `vlm/lora.py` | peft LoRA baseline on SmolVLM | `adapter.mode: base` in a vision config |

Known defects in the legacy code that the package fixes:

- `vlm/lor2c.py` zero-initialises both projections, so gradients are identically zero and the
  adapter never trains. The package uses Kaiming-uniform on the down projection.
- `quantization.py` converts QAT modules without quant/dequant stubs, producing kernels that
  cannot accept floating point input. The package wraps adapters in `QuantStub`/`DeQuantStub`.
- Hooks pinned tensors to `cuda:0`, breaking multi-GPU device maps and CPU runs.
- `templates/alpaca.json` had to exist on disk; the template is now a typed constant.
- `vlm/*.py` passed the assistant caption as a plain string, which current SmolVLM chat templates
  render as empty, so every training target was `Assistant: <end_of_utterance>`; label masking by
  decode/re-encode swallowed the caption; and BLEU evaluation generated from the full sample
  (prompt + reference) rather than from the prompt, so scores were always 0.

Original run instructions (Python 3.10):

```bash
pip install fsspec==2025.3.2 accelerate==0.22.0 transformers==4.31.0 evaluate appdirs \
    bitsandbytes datasets fire sentencepiece scipy scikit-learn wandb
pip install -q --force-reinstall numpy==1.24.4 pandas==1.5.3 pyarrow==10.0.1
unzip peft-0.5.0-fork.zip && cd peft-0.5.0 && pip install -e . && cd ..
python prototype.py --mode lor2c
```
