"""Quantization-aware training of residual adapters using torch eager-mode quantization."""

import torch
from torch import Tensor, nn
from torch.ao.quantization import (
    DeQuantStub,
    QuantStub,
    convert,
    get_default_qat_qconfig,
    prepare_qat,
)

from lor2c.domain.adapter import Branch, LowRankAdapter
from lor2c.domain.bank import AdapterBank
from lor2c.domain.constants import QuantizationBackend
from lor2c.domain.exceptions import AdapterError


class QuantizedBranch(Branch):
    """Wraps a `LowRankAdapter` between quant/dequant stubs; scaling stays in floating point."""

    def __init__(self, *, adapter: LowRankAdapter) -> None:
        super().__init__()
        self.quant = QuantStub()  # type: ignore[no-untyped-call]
        self.dropout = adapter.dropout
        self.down = adapter.down
        self.up = adapter.up
        self.dequant = DeQuantStub()  # type: ignore[no-untyped-call]
        self.scaling = adapter.spec.scaling
        self.__dtype = adapter.dtype

    @property
    def dtype(self) -> torch.dtype:
        """Input dtype expected before quantization."""
        return self.__dtype

    def forward(self, hidden: Tensor) -> Tensor:
        """Quantize, project, dequantize and scale."""
        projected = self.up(self.down(self.dropout(self.quant(hidden))))
        delta: Tensor = self.dequant(projected)
        return delta * self.scaling


class FakeQuantizer:
    """Prepares adapters for QAT and converts them to integer kernels after training."""

    def __init__(self, *, backend: QuantizationBackend) -> None:
        if backend.value not in torch.backends.quantized.supported_engines:
            raise AdapterError(
                f"Quantization backend '{backend.value}' is not supported here; available: "
                f"{torch.backends.quantized.supported_engines}."
            )
        self.__backend = backend
        self.__qconfig = get_default_qat_qconfig(backend.value)  # type: ignore[no-untyped-call]

    def prepare(self, *, bank: AdapterBank) -> None:
        """Replace each adapter with a fake-quantized `QuantizedBranch` (in place)."""
        torch.backends.quantized.engine = self.__backend.value
        for name in bank.names:
            adapter = bank.get(name=name)
            if not isinstance(adapter, LowRankAdapter):
                raise AdapterError(f"Adapter '{name}' is already quantized or of unknown type.")
            wrapped = QuantizedBranch(adapter=adapter.float())
            wrapped.qconfig = self.__qconfig
            prepared = prepare_qat(wrapped.train(), inplace=True)  # type: ignore[no-untyped-call]
            bank.replace(name=name, adapter=prepared)

    def convert(self, *, bank: AdapterBank) -> None:
        """Replace fake-quantized branches with quantized kernels on CPU (in place)."""
        for name in bank.names:
            adapter = bank.get(name=name)
            if not isinstance(adapter, QuantizedBranch):
                raise AdapterError(f"Adapter '{name}' was not prepared for quantization.")
            converted: nn.Module = convert(adapter.cpu().eval(), inplace=True)  # type: ignore[no-untyped-call]
            if not isinstance(converted, Branch):
                raise AdapterError(f"Conversion of '{name}' produced {type(converted).__name__}.")
            bank.replace(name=name, adapter=converted)
