"""Low-rank adapter modules."""

from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn

from lor2c.domain.constants import KAIMING_NEGATIVE_SLOPE
from lor2c.domain.schema import AdapterSpec


class Branch(nn.Module, ABC):
    """A residual branch mapping a hidden state to an additive delta of the same shape."""

    @property
    @abstractmethod
    def dtype(self) -> torch.dtype:
        """Floating dtype the branch expects its input in."""

    @abstractmethod
    def forward(self, hidden: Tensor) -> Tensor:
        """Return the delta for `hidden`."""


class LowRankAdapter(Branch):
    """Residual branch computing `scaling * up(down(dropout(x)))` with a rank-limited bottleneck."""

    def __init__(
        self,
        *,
        input_width: int,
        output_width: int,
        spec: AdapterSpec,
        generator: torch.Generator | None = None,
        down: nn.Linear | None = None,
    ) -> None:
        super().__init__()
        self.spec = spec
        self.shared = down is not None
        self.down = down if down is not None else nn.Linear(input_width, spec.rank, bias=False)
        self.up = nn.Linear(spec.rank, output_width, bias=False)
        self.dropout: nn.Module = nn.Dropout(spec.dropout) if spec.dropout > 0 else nn.Identity()
        self.__reset(generator=generator)

    def __reset(self, *, generator: torch.Generator | None) -> None:
        if not self.shared:
            nn.init.kaiming_uniform_(
                self.down.weight, a=KAIMING_NEGATIVE_SLOPE, generator=generator
            )
        nn.init.zeros_(self.up.weight)

    @property
    def product(self) -> Tensor:
        """The effective update matrix `up @ down` scaled, shape (output, input)."""
        return (self.up.weight @ self.down.weight) * self.spec.scaling

    @property
    def dtype(self) -> torch.dtype:
        """Dtype of the projection weights."""
        return self.down.weight.dtype

    def forward(self, hidden: Tensor) -> Tensor:
        """Return the scaled low-rank delta for `hidden`."""
        projected: Tensor = self.up(self.down(self.dropout(hidden)))
        return projected * self.spec.scaling


class AdaptedLinear(nn.Module):
    """A frozen linear layer whose output is augmented by a trainable residual branch."""

    def __init__(self, *, base: nn.Linear, adapter: Branch) -> None:
        super().__init__()
        self.base = base
        self.adapter = adapter
        self.base.requires_grad_(False)

    def forward(self, hidden: Tensor) -> Tensor:
        """Return `base(hidden) + adapter(hidden)` in the dtype of the base output."""
        base_output: Tensor = self.base(hidden)
        delta: Tensor = self.adapter(hidden.to(self.adapter.dtype))
        return base_output + delta.to(base_output.dtype)
