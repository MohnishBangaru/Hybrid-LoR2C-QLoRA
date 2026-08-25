# Contributing

## Architecture rules

The package follows a strict layered (hexagonal) architecture described in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). Non-negotiables:

- `lor2c.domain` imports only `torch`, `pydantic` and the standard library. No `transformers`,
  `peft`, `datasets`, `wandb`, `PIL`, file system, environment reads or logging.
- `lor2c.application` depends on the domain and on `Protocol` ports only; it never imports an
  infrastructure module.
- Every external system is reached through an adapter under `lor2c.infrastructure`; Hugging Face
  SDKs are imported only in `lor2c.infrastructure.huggingface`.
- Anything crossing a boundary is a Pydantic model (`settings/schema.py`, `application/schema.py`,
  `domain/schema.py`). No raw `dict`/`tuple` payloads.
- Behaviour lives in classes; module-level functions are reserved for pure, stateless helpers.
- Internal calls use keyword arguments. Private members use `__name`, protected members `_name`.
- No magic numbers or strings: add them to `domain/constants.py` as enums or `Final` constants.
- Every class and public method carries a one-line docstring stating intent.

## Code style

- Python 3.11+. Run `make check` before pushing: `ruff format --check`, `ruff check`,
  `mypy --strict`, `pytest --cov`. CI enforces all four.
- Type hints on every signature; `Any` only at the outermost boundary.
- Catch the most specific exception, name it `exception`, and re-raise as a
  `lor2c.domain.exceptions` type with an actionable message. Never swallow errors.
- Log only from infrastructure and CLI layers, through `logging` with `ctx_*` extras so the JSON
  formatter promotes them to fields. Never log secrets.

## Tests

- Tests mirror `src/` one-to-one under `tests/` and are class-based (`class TestX:`).
- Domain and application tests run on CPU with in-memory fakes (`tests/application/fakes.py`) and
  never download models.
- Cover real behaviour: the mathematical contract of an adapter, a routing edge case, a settings
  validation rule. Avoid tests that only assert a mock was called.

## Configuration and reproducibility

- Runs are described by YAML files in `configs/` validated by `lor2c.settings.schema`; unknown keys
  are rejected. Add new options to the schema with a `Field(description=...)`.
- Seeds are injected (`seed:` in the config); adapter initialisation takes an explicit
  `torch.Generator`. Do not read environment variables from domain or application code.
- Do not commit experiment outputs (`outputs/`, `wandb/`, checkpoints). Released adapters go under
  `artifacts/adapters/` and should use Git LFS once they grow beyond a few MB.

## Branches, commits and pull requests

- Branch from `main` as `<type>/<short-kebab-description>` with `<type>` one of `feat`, `fix`,
  `chore`, `docs`, `refactor`, `exp`.
- Imperative, descriptive commit messages; one logical change per commit.
- A PR must pass CI and describe what was verified and what was not (for example "not executed on
  GPU"). Keep `main` releasable.

## Legacy code and notebooks

- `legacy/` and `notebooks/` are frozen provenance. Do not extend them; port behaviour into the
  package with tests instead.
