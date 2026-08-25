"""Shared fixtures: a tiny decoder stack standing in for a transformer host model."""

import pytest
import torch
from torch import Tensor, nn


class TinyDecoderLayer(nn.Module):
    """One residual block returning a tuple like Hugging Face decoder layers do."""

    def __init__(self, *, width: int) -> None:
        super().__init__()
        self.q_proj = nn.Linear(width, width)
        self.v_proj = nn.Linear(width, width)
        self.o_proj = nn.Linear(width, width)

    def forward(self, hidden: Tensor) -> tuple[Tensor]:
        return (hidden + self.o_proj(torch.tanh(self.q_proj(hidden) + self.v_proj(hidden))),)


class TinyDecoder(nn.Module):
    """A stack of `TinyDecoderLayer` exposed through `model.layers`."""

    def __init__(self, *, width: int, depth: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(TinyDecoderLayer(width=width) for _ in range(depth))

    def forward(self, hidden: Tensor) -> Tensor:
        for layer in self.layers:
            hidden = layer(hidden)[0]
        return hidden


@pytest.fixture
def width() -> int:
    return 8


@pytest.fixture
def depth() -> int:
    return 3


@pytest.fixture
def decoder(width: int, depth: int) -> TinyDecoder:
    torch.manual_seed(0)
    return TinyDecoder(width=width, depth=depth)


@pytest.fixture
def generator() -> torch.Generator:
    return torch.Generator().manual_seed(7)
