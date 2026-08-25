"""Named collection of residual adapters."""

import torch
from torch import nn

from lor2c.domain.adapter import Branch, LowRankAdapter
from lor2c.domain.exceptions import AdapterError
from lor2c.domain.schema import AdapterSpec


class AdapterBank(nn.Module):
    """Registry of named `LowRankAdapter` modules sharing one input/output width."""

    def __init__(self, *, width: int) -> None:
        super().__init__()
        self.width = width
        self.adapters = nn.ModuleDict()

    @property
    def names(self) -> tuple[str, ...]:
        """Registered adapter names in insertion order."""
        return tuple(self.adapters.keys())

    def register(
        self, *, name: str, spec: AdapterSpec, generator: torch.Generator | None = None
    ) -> LowRankAdapter:
        """Create and store a new adapter; duplicate names are rejected."""
        if name in self.adapters:
            raise AdapterError(f"Adapter '{name}' is already registered.")
        adapter = LowRankAdapter(
            input_width=self.width, output_width=self.width, spec=spec, generator=generator
        )
        self.adapters[name] = adapter
        return adapter

    def get(self, *, name: str) -> Branch:
        """Return the branch registered under `name`."""
        try:
            adapter = self.adapters[name]
        except KeyError as exception:
            raise AdapterError(f"Adapter '{name}' is not registered.") from exception
        if not isinstance(adapter, Branch):
            raise AdapterError(f"Adapter '{name}' has unexpected type {type(adapter).__name__}.")
        return adapter

    def replace(self, *, name: str, adapter: Branch) -> None:
        """Swap the module stored under an existing name (used after quantization conversion)."""
        if name not in self.adapters:
            raise AdapterError(f"Adapter '{name}' is not registered and cannot be replaced.")
        self.adapters[name] = adapter
