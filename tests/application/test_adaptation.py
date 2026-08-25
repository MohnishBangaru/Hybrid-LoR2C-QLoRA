import torch
from torch import nn

from lor2c.application.adaptation import AdaptationController
from lor2c.domain.adaptation import AdaptationPlanner, AdaptationTimeline
from lor2c.domain.bank import AdapterBank
from lor2c.domain.schedule import ResidualRouter
from lor2c.domain.schema import AdapterSpec, ResidualSchedule
from lor2c.domain.spectrum import FeatureSpaceShape
from tests.application.fakes import RecordingGate


class TestAdaptationController:
    def __router(self, *, depth: int, spreads: list[float]) -> ResidualRouter:
        """Each adapter gets singular values (1, spread): larger spread means higher SFS."""
        bank = AdapterBank(width=6)
        schedule = ResidualSchedule.per_layer(depth=depth)
        for entry, spread in zip(schedule.entries, spreads, strict=True):
            adapter = bank.register(name=entry.name, spec=AdapterSpec(rank=2, alpha=2))
            with torch.no_grad():
                adapter.down.weight.zero_()
                adapter.up.weight.zero_()
                adapter.down.weight[0, 0] = adapter.down.weight[1, 1] = 1.0
                adapter.up.weight[0, 0] = 1.0
                adapter.up.weight[1, 1] = spread
        return ResidualRouter(bank=bank, schedule=schedule)

    def __controller(
        self, *, router: ResidualRouter, merges: int, injections: int, gate: RecordingGate
    ) -> AdaptationController:
        return AdaptationController(
            router=router,
            model=nn.Linear(1, 1),
            gate=gate,
            planner=AdaptationPlanner(span=4),
            spectrum=FeatureSpaceShape(top=1),
            merges=AdaptationTimeline(epochs=4.0, count=merges),
            injections=AdaptationTimeline(epochs=4.0, count=injections),
        )

    def test_nothing_happens_before_first_interval(self) -> None:
        router = self.__router(depth=3, spreads=[0.5, 0.5, 0.5])
        controller = self.__controller(router=router, merges=1, injections=0, gate=RecordingGate())
        controller.observe(epoch=0.5)
        assert controller.merged == 0
        assert router.bank.names == ("floor1", "floor2", "floor3")

    def test_merge_joins_lowest_sfs_pair_and_keeps_richer_member(self) -> None:
        router = self.__router(depth=3, spreads=[0.9, 0.05, 0.3])
        kept = router.bank.get(name="floor3")
        controller = self.__controller(router=router, merges=1, injections=0, gate=RecordingGate())
        controller.observe(epoch=1.0)
        assert controller.merged == 1
        assert set(router.bank.names) == {"floor1", "floor2+floor3"}
        joined = router.schedule.entry(name="floor2+floor3")
        assert (joined.start, joined.end) == (1, 2)
        assert router.bank.get(name="floor2+floor3") is kept

    def test_inject_removes_lowest_sfs_adapter_and_releases_attention(self) -> None:
        router = self.__router(depth=3, spreads=[0.5, 0.01, 0.5])
        gate = RecordingGate()
        controller = self.__controller(router=router, merges=0, injections=1, gate=gate)
        controller.observe(epoch=1.0)
        assert controller.injected == 1
        assert "floor2" not in router.bank.names
        assert gate.released == [1]

    def test_routing_still_works_after_merge(self) -> None:
        router = self.__router(depth=2, spreads=[0.5, 0.5])
        self.__controller(router=router, merges=1, injections=0, gate=RecordingGate()).observe(
            epoch=1.0
        )
        hidden = torch.ones(1, 6)
        out0 = router.route(layer=0, hidden_in=hidden, hidden_out=torch.zeros(1, 6))
        out1 = router.route(layer=1, hidden_in=hidden, hidden_out=torch.zeros(1, 6))
        assert torch.equal(out0, torch.zeros(1, 6))
        assert torch.any(out1 != 0)
