import pytest
import torch

from lor2c.domain.bank import AdapterBank
from lor2c.domain.constants import QuantizationBackend
from lor2c.domain.exceptions import AdapterError
from lor2c.domain.schema import AdapterSpec
from lor2c.infrastructure.quantization import FakeQuantizer, QuantizedBranch


def _backend() -> QuantizationBackend:
    for backend in QuantizationBackend:
        if backend.value in torch.backends.quantized.supported_engines:
            return backend
    pytest.skip("no quantized engine available on this platform")


class TestFakeQuantizer:
    def __bank(self) -> AdapterBank:
        bank = AdapterBank(width=8)
        adapter = bank.register(name="floor1", spec=AdapterSpec(rank=2, alpha=4))
        with torch.no_grad():
            adapter.up.weight.normal_()
        return bank

    def test_prepare_wraps_adapters_and_keeps_them_trainable(self) -> None:
        bank = self.__bank()
        FakeQuantizer(backend=_backend()).prepare(bank=bank)
        branch = bank.get(name="floor1")
        assert isinstance(branch, QuantizedBranch)
        output = branch(torch.randn(3, 8))
        assert output.shape == (3, 8)
        output.sum().backward()
        assert branch.down.weight.grad is not None

    def test_convert_produces_integer_kernels_close_to_float(self) -> None:
        bank = self.__bank()
        quantizer = FakeQuantizer(backend=_backend())
        quantizer.prepare(bank=bank)
        branch = bank.get(name="floor1")
        hidden = torch.randn(16, 8)
        branch.train()
        for _ in range(5):
            branch(hidden)
        reference = branch.eval()(hidden)
        quantizer.convert(bank=bank)
        converted = bank.get(name="floor1")
        assert type(converted.down).__module__.startswith("torch.ao.nn.quantized")
        assert torch.allclose(converted(hidden), reference, atol=0.2)

    def test_convert_before_prepare_is_an_error(self) -> None:
        with pytest.raises(AdapterError, match="not prepared"):
            FakeQuantizer(backend=_backend()).convert(bank=self.__bank())
