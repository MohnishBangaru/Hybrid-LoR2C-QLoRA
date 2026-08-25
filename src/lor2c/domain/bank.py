"""Named collection of residual adapters."""

import torch
from torch import nn

from lor2c.domain.adapter import Branch, LowRankAdapter
from lor2c.domain.constants import KAIMING_NEGATIVE_SLOPE
from lor2c.domain.exceptions import AdapterError
from lor2c.domain.schema import AdapterSpec


class AdapterBank(nn.Module):
    """Registry of named `LowRankAdapter` modules sharing one input/output width."""

    def __init__(self, *, width: int, shared: AdapterSpec | None = None) -> None:
        super().__init__()
        self.width = width
        self.adapters = nn.ModuleDict()
        self.down: nn.Linear | None = None
        if shared is not None:
            self.down = nn.Linear(width, shared.rank, bias=False)

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
        if self.down is not None and self.down.out_features != spec.rank:
            raise AdapterError(
                f"Adapter '{name}' has rank {spec.rank} but the shared projection has rank "
                f"{self.down.out_features}."
            )
        adapter = LowRankAdapter(
            input_width=self.width,
            output_width=self.width,
            spec=spec,
            generator=generator,
            down=self.down,
        )
        if self.down is not None and len(self.adapters) == 0:
            nn.init.kaiming_uniform_(
                self.down.weight, a=KAIMING_NEGATIVE_SLOPE, generator=generator
            )
        self.adapters[name] = adapter
        return adapter

    def rename(self, *, old: str, new: str) -> None:
        """Re-key an adapter (used when two schedule entries merge into one)."""
        adapter = self.get(name=old)
        if new in self.adapters:
            raise AdapterError(f"Adapter '{new}' already exists.")
        del self.adapters[old]
        self.adapters[new] = adapter

    def remove(self, *, name: str) -> None:
        """Drop an adapter from the bank."""
        self.get(name=name)
        del self.adapters[name]

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
