"""Runtime controller applying MergeLoR2C / InjectLoR2C decisions during training."""

import logging

from torch import nn

from lor2c.application.ports import AttentionGate
from lor2c.domain.adaptation import AdaptationPlanner, AdaptationTimeline
from lor2c.domain.bank import AdapterBank
from lor2c.domain.schedule import ResidualRouter
from lor2c.domain.spectrum import FeatureSpaceShape

LOGGER = logging.getLogger(__name__)


class AdaptationController:
    """Observes training progress and mutates the residual schedule when events fall due."""

    def __init__(
        self,
        *,
        router: ResidualRouter,
        model: nn.Module,
        gate: AttentionGate,
        planner: AdaptationPlanner,
        spectrum: FeatureSpaceShape,
        merges: AdaptationTimeline,
        injections: AdaptationTimeline,
    ) -> None:
        self.__router = router
        self.__model = model
        self.__gate = gate
        self.__planner = planner
        self.__spectrum = spectrum
        self.__merges = merges
        self.__injections = injections
        self.__merged = 0
        self.__injected = 0

    @property
    def merged(self) -> int:
        """Merges performed so far."""
        return self.__merged

    @property
    def injected(self) -> int:
        """Injections performed so far."""
        return self.__injected

    def observe(self, *, epoch: float) -> None:
        """Fire every merge or injection that has become due at `epoch`."""
        merges = self.__merges.due(epoch=epoch)
        injections = self.__injections.due(epoch=epoch)
        if merges == 0 and injections == 0:
            return
        scores = self.__spectrum.score(bank=self.__router.bank)
        for _ in range(merges):
            self.__merge(scores=scores)
        for _ in range(injections):
            self.__inject(scores=scores)

    def __merge(self, *, scores: dict[str, float]) -> None:
        choice = self.__planner.merge(schedule=self.__router.schedule, scores=scores)
        if choice is None:
            LOGGER.warning("No adjacent adapters left to merge")
            return
        bank: AdapterBank = self.__router.bank
        schedule = self.__router.schedule.merged(first=choice.first, second=choice.second)
        joined = schedule.entry(name=f"{choice.first}+{choice.second}").name
        bank.remove(name=choice.second)
        bank.rename(old=choice.first, new=joined)
        self.__router.reschedule(schedule=schedule)
        self.__merged += 1
        LOGGER.info(
            "Merged adapters", extra={"ctx_first": choice.first, "ctx_second": choice.second}
        )

    def __inject(self, *, scores: dict[str, float]) -> None:
        entry = self.__planner.inject(schedule=self.__router.schedule, scores=scores)
        if entry is None:
            LOGGER.warning("No single-layer adapter left to inject")
            return
        self.__router.bank.remove(name=entry.name)
        self.__router.reschedule(schedule=self.__router.schedule.without(name=entry.name))
        self.__gate.release(model=self.__model, layer=entry.start)
        self.__injected += 1
        LOGGER.info(
            "Injected attention LoRA", extra={"ctx_layer": entry.start, "ctx_adapter": entry.name}
        )
