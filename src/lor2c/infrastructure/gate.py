"""Per-layer control of attention LoRA trainability by parameter-name pattern."""

import re

from torch import nn

from lor2c.domain.exceptions import AdapterError


class PatternAttentionGate:
    """Freezes/releases LoRA parameters whose names match `layers.<index>.` and a LoRA marker."""

    def __init__(self, *, marker: str = "lora_", layers: str = "layers") -> None:
        self.__marker = marker
        self.__layers = layers

    def freeze(self, *, model: nn.Module) -> None:
        """Disable gradients for every parameter carrying the LoRA marker."""
        for name, parameter in model.named_parameters():
            if self.__marker in name:
                parameter.requires_grad_(False)

    def release(self, *, model: nn.Module, layer: int) -> None:
        """Enable gradients for LoRA parameters belonging to decoder layer `layer`."""
        pattern = re.compile(rf"(^|\.){re.escape(self.__layers)}\.{layer}\.")
        released = 0
        for name, parameter in model.named_parameters():
            if self.__marker in name and pattern.search(name):
                parameter.requires_grad_(True)
                released += 1
        if released == 0:
            raise AdapterError(f"No LoRA parameters found for decoder layer {layer}.")
