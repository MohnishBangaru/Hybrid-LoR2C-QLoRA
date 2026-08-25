# Contributing & Engineering Standards

This is a research codebase, but changes should still meet a consistent bar so
experiments stay reproducible and reviewable.

## Code style

- Python 3.10+. Lint with [ruff](https://docs.astral.sh/ruff/) before pushing:
  `ruff check .` (configuration lives in `pyproject.toml`; CI enforces it).
- No unused imports or dead code — delete it rather than commenting it out.
- Avoid mutable default arguments; use `Optional[...] = None` and resolve
  inside the function.
- Script entry points go behind `if __name__ == "__main__":`.
- Validate user-facing options early and fail with a clear `ValueError`
  (e.g. an unknown `--mode`), instead of letting an undefined variable crash
  later.
- Prefer explicit guards over numeric hacks (no `x + 1e-9` to dodge
  division by zero).

## Reproducibility

- Every training script must expose (or fix) a random seed and log its full
  hyperparameter configuration at startup.
- Pin new dependencies in `requirements.txt` when a specific version matters.
- The Llama prototypes depend on the **vendored** `peft-0.5.0` fork in
  `archive.zip` (`MSLoraConfig`, `LlamaLoRALayer`); do not swap it for
  upstream `peft` without migrating both prototypes.

## Repository hygiene

- Do not commit experiment outputs (`wandb/`, checkpoints, adapter dumps),
  editor folders, or OS junk (`.DS_Store`) — `.gitignore` covers these.
  The exception is a deliberately released artifact such as
  `lora-alpaca-qat/`.
- Images used in docs live in `assets/`; notebooks live in `notebooks/`.
- Keep large binaries out of git history unless they are release artifacts.

## Commits and branches

- Develop on a feature branch; keep `main` releasable.
- Write imperative, descriptive commit messages
  (`Add QAT conversion for LoR2C adapters`, not `fix stuff`).
- One logical change per commit where practical.

## Notebooks

- Notebooks are for exploration. Once an experiment stabilizes, port it to a
  script under version control with a CLI (see `prototype.py`) so it can be
  run headless and diffed.
