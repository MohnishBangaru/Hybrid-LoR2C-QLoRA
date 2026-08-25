import pytest
import torch

from lor2c.domain.bank import AdapterBank
from lor2c.domain.exceptions import ScheduleError
from lor2c.domain.schedule import ResidualRouter
from lor2c.domain.schema import AdapterSpec, ResidualSchedule
from lor2c.infrastructure.hooks import HookRouter, ModuleListLocator
from tests.conftest import TinyDecoder


class TestModuleListLocator:
    def test_finds_layers(self, decoder: TinyDecoder, depth: int) -> None:
        assert len(ModuleListLocator().locate(model=decoder)) == depth

    def test_missing_attribute_is_an_error(self, decoder: TinyDecoder) -> None:
        with pytest.raises(ScheduleError, match="blocks"):
            ModuleListLocator(attribute="blocks").locate(model=decoder)


class TestHookRouter:
    def __router(self, *, width: int, depth: int, fill: float) -> ResidualRouter:
        bank = AdapterBank(width=width)
        for entry in ResidualSchedule.per_layer(depth=depth).entries:
            adapter = bank.register(name=entry.name, spec=AdapterSpec(rank=2, alpha=2))
            with torch.no_grad():
                adapter.up.weight.fill_(fill)
        return ResidualRouter(bank=bank, schedule=ResidualSchedule.per_layer(depth=depth))

    def test_zero_adapters_leave_output_unchanged(
        self, decoder: TinyDecoder, width: int, depth: int
    ) -> None:
        hidden = torch.randn(2, width)
        baseline = decoder(hidden)
        with HookRouter(locator=ModuleListLocator()).attach(
            model=decoder, router=self.__router(width=width, depth=depth, fill=0.0)
        ):
            assert torch.allclose(decoder(hidden), baseline)

    def test_nonzero_adapters_change_output_only_while_attached(
        self, decoder: TinyDecoder, width: int, depth: int
    ) -> None:
        hidden = torch.randn(2, width)
        baseline = decoder(hidden)
        with HookRouter(locator=ModuleListLocator()).attach(
            model=decoder, router=self.__router(width=width, depth=depth, fill=0.5)
        ):
            assert not torch.allclose(decoder(hidden), baseline)
        assert torch.allclose(decoder(hidden), baseline)

    def test_gradients_flow_into_adapters(
        self, decoder: TinyDecoder, width: int, depth: int
    ) -> None:
        router = self.__router(width=width, depth=depth, fill=0.5)
        with HookRouter(locator=ModuleListLocator()).attach(model=decoder, router=router):
            decoder(torch.randn(2, width)).sum().backward()
        for name in router.bank.names:
            grad = router.bank.get(name=name).down.weight.grad  # type: ignore[union-attr]
            assert grad is not None and torch.any(grad != 0)

    def test_depth_mismatch_is_rejected(self, decoder: TinyDecoder, width: int, depth: int) -> None:
        router = self.__router(width=width, depth=depth + 1, fill=0.0)
        hooks = HookRouter(locator=ModuleListLocator())
        with (
            pytest.raises(ScheduleError, match="expects"),
            hooks.attach(model=decoder, router=router),
        ):
            pass
