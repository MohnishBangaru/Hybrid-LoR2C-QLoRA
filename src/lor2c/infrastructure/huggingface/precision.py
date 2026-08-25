"""Maps precision settings onto torch dtypes."""

from typing import ClassVar

import torch

from lor2c.domain.constants import Precision


class PrecisionMapper:
    """Translates the `Precision` enum into a `torch.dtype`."""

    __TABLE: ClassVar[dict[Precision, torch.dtype]] = {
        Precision.FLOAT16: torch.float16,
        Precision.BFLOAT16: torch.bfloat16,
        Precision.FLOAT32: torch.float32,
    }

    def dtype(self, *, precision: Precision) -> torch.dtype:
        """Return the dtype for `precision`."""
        return self.__TABLE[precision]
