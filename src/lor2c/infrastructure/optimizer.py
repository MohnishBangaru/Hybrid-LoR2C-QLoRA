"""Optimizer construction with LoRA+ style learning-rate groups."""

from collections.abc import Iterable

import torch
from torch import nn


class GroupedOptimizerFactory:
    """Builds AdamW with a higher learning rate for `up`/`lora_B` projections (LoRA+)."""

    UP_MARKERS = (".up.", "lora_B")

    def __init__(self, *, rate: float, ratio: float, weight_decay: float = 0.0) -> None:
        self.__rate = rate
        self.__ratio = ratio
        self.__weight_decay = weight_decay

    def build(self, *, model: nn.Module) -> torch.optim.Optimizer:
        """Return AdamW over trainable parameters split into base and up-projection groups."""
        base, up = self.__split(parameters=model.named_parameters())
        groups: list[dict[str, object]] = []
        if base:
            groups.append({"params": base, "lr": self.__rate})
        if up:
            groups.append({"params": up, "lr": self.__rate * self.__ratio})
        return torch.optim.AdamW(groups, lr=self.__rate, weight_decay=self.__weight_decay)

    def __split(
        self, *, parameters: Iterable[tuple[str, nn.Parameter]]
    ) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
        base: list[nn.Parameter] = []
        up: list[nn.Parameter] = []
        for name, parameter in parameters:
            if not parameter.requires_grad:
                continue
            if any(marker in name for marker in self.UP_MARKERS):
                up.append(parameter)
            else:
                base.append(parameter)
        return base, up
