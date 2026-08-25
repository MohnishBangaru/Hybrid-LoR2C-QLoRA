import pytest
import torch

from lor2c.domain.bank import AdapterBank
from lor2c.domain.exceptions import ScheduleError
from lor2c.domain.schedule import ResidualRouter
from lor2c.domain.schema import AdapterSpec, ResidualSchedule, ScheduleEntry


class TestResidualRouter:
    def __bank(self, *, names: tuple[str, ...]) -> AdapterBank:
        bank = AdapterBank(width=4)
        for name in names:
            adapter = bank.register(name=name, spec=AdapterSpec(rank=2, alpha=2))
            with torch.no_grad():
                adapter.down.weight.fill_(1.0)
                adapter.up.weight.fill_(1.0)
        return bank

    def test_rejects_schedule_with_unknown_adapter(self) -> None:
        with pytest.raises(ScheduleError, match="missing from the bank"):
            ResidualRouter(
                bank=self.__bank(names=("a",)), schedule=ResidualSchedule.per_layer(depth=2)
            )

    def test_same_layer_entry_adds_delta_from_input(self) -> None:
        router = ResidualRouter(
            bank=self.__bank(names=("floor1",)), schedule=ResidualSchedule.per_layer(depth=1)
        )
        hidden_in = torch.ones(1, 4)
        hidden_out = torch.zeros(1, 4)
        routed = router.route(layer=0, hidden_in=hidden_in, hidden_out=hidden_out)
        # down: 4 ones -> each of 2 dims = 4; up: 2 dims -> each output = 8; scaling alpha/rank = 1
        assert torch.allclose(routed, torch.full((1, 4), 8.0))

    def test_spanning_entry_defers_delta_to_end_layer(self) -> None:
        schedule = ResidualSchedule(depth=3, entries=(ScheduleEntry(name="span", start=0, end=2),))
        router = ResidualRouter(bank=self.__bank(names=("span",)), schedule=schedule)
        zeros = torch.zeros(1, 4)
        assert torch.equal(
            router.route(layer=0, hidden_in=torch.ones(1, 4), hidden_out=zeros), zeros
        )
        assert torch.equal(router.route(layer=1, hidden_in=zeros, hidden_out=zeros), zeros)
        assert torch.allclose(
            router.route(layer=2, hidden_in=zeros, hidden_out=zeros), torch.full((1, 4), 8.0)
        )

    def test_applying_without_capture_is_an_error(self) -> None:
        schedule = ResidualSchedule(depth=2, entries=(ScheduleEntry(name="span", start=0, end=1),))
        router = ResidualRouter(bank=self.__bank(names=("span",)), schedule=schedule)
        with pytest.raises(ScheduleError, match="no delta was captured"):
            router.route(layer=1, hidden_in=torch.zeros(1, 4), hidden_out=torch.zeros(1, 4))

    def test_reset_clears_pending(self) -> None:
        schedule = ResidualSchedule(depth=2, entries=(ScheduleEntry(name="span", start=0, end=1),))
        router = ResidualRouter(bank=self.__bank(names=("span",)), schedule=schedule)
        router.route(layer=0, hidden_in=torch.ones(1, 4), hidden_out=torch.zeros(1, 4))
        router.reset()
        with pytest.raises(ScheduleError):
            router.route(layer=1, hidden_in=torch.zeros(1, 4), hidden_out=torch.zeros(1, 4))
