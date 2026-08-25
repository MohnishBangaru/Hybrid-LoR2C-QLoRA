"""Replaces target linear layers with adapter-augmented equivalents."""

from collections.abc import Sequence

import torch
from torch import nn

from lor2c.domain.adapter import AdaptedLinear, LowRankAdapter
from lor2c.domain.exceptions import AdapterError
from lor2c.domain.schema import AdapterSpec


class LinearInjector:
    """Wraps every `nn.Linear` whose name ends with a target suffix in an `AdaptedLinear`."""

    def inject(
        self,
        *,
        model: nn.Module,
        targets: Sequence[str],
        spec: AdapterSpec,
        generator: torch.Generator | None = None,
    ) -> int:
        """Mutate `model` in place and return how many layers were wrapped."""
        matches = [
            (name, module)
            for name, module in model.named_modules()
            if isinstance(module, nn.Linear) and any(name.endswith(target) for target in targets)
        ]
        if not matches:
            raise AdapterError(f"No nn.Linear modules matched targets {list(targets)}.")
        for name, module in matches:
            adapter = LowRankAdapter(
                input_width=module.in_features,
                output_width=module.out_features,
                spec=spec,
                generator=generator,
            ).to(device=module.weight.device, dtype=module.weight.dtype)
            self.__replace(
                model=model, name=name, replacement=AdaptedLinear(base=module, adapter=adapter)
            )
        return len(matches)

    @staticmethod
    def __replace(*, model: nn.Module, name: str, replacement: nn.Module) -> None:
        parent_path, _, attribute = name.rpartition(".")
        parent = model.get_submodule(parent_path) if parent_path else model
        setattr(parent, attribute, replacement)
