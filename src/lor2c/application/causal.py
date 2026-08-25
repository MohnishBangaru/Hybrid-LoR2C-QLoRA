"""Use-case: fine-tune a causal language model with LoR2C residual adapters."""

import torch

from lor2c.application.ports import (
    CausalDataPort,
    CausalModelPort,
    CausalTrainerPort,
    Quantizer,
    Repository,
    Router,
    Seeder,
    Tracker,
)
from lor2c.application.schema import Outcome
from lor2c.domain.bank import AdapterBank
from lor2c.domain.constants import AdapterMode
from lor2c.domain.rank import RankPolicy
from lor2c.domain.schedule import ResidualRouter
from lor2c.domain.schema import ResidualSchedule
from lor2c.settings.schema import CausalSettings


class CausalTrainingService:
    """Coordinates model loading, adapter planning, training, quantization and persistence."""

    def __init__(
        self,
        *,
        models: CausalModelPort,
        data: CausalDataPort,
        trainer: CausalTrainerPort,
        router: Router,
        quantizer: Quantizer,
        repository: Repository,
        tracker: Tracker,
        seeder: Seeder,
        policy: RankPolicy,
    ) -> None:
        self.__models = models
        self.__data = data
        self.__trainer = trainer
        self.__router = router
        self.__quantizer = quantizer
        self.__repository = repository
        self.__tracker = tracker
        self.__seeder = seeder
        self.__policy = policy

    def run(self, *, settings: CausalSettings) -> Outcome:
        """Execute the full fine-tuning workflow described by `settings`."""
        self.__seeder.seed(value=settings.seed)
        self.__tracker.start(name=settings.name, parameters=settings.model_dump(mode="json"))
        try:
            return self.__execute(settings=settings)
        finally:
            self.__tracker.finish()

    def __execute(self, *, settings: CausalSettings) -> Outcome:
        bundle = self.__models.load(settings=settings.model)
        spec = self.__policy.resolve(hidden_size=bundle.hidden)
        bundle = self.__models.adapt(bundle=bundle, spec=spec, settings=settings.adapter)
        split = self.__data.load(settings=settings.data, seed=settings.seed)

        bank = self.__build_bank(bundle_hidden=bundle.hidden, depth=bundle.depth, settings=settings)
        if bank is None:
            report = self.__trainer.fit(bundle=bundle, split=split, settings=settings)
            self.__repository.save(model=bundle.model, bank=None, output=settings.output)
            return Outcome(name=settings.name, output=settings.output, steps=report.steps)

        schedule = ResidualSchedule.per_layer(depth=bundle.depth)
        router = ResidualRouter(bank=bank, schedule=schedule)
        if settings.quantization.enabled:
            self.__quantizer.prepare(bank=bank)
        with self.__router.attach(model=bundle.model, router=router):
            report = self.__trainer.fit(bundle=bundle, split=split, settings=settings)
        if settings.quantization.enabled:
            self.__quantizer.convert(bank=bank)
        self.__repository.save(model=bundle.model, bank=bank, output=settings.output)
        return Outcome(name=settings.name, output=settings.output, steps=report.steps)

    def __build_bank(
        self, *, bundle_hidden: int, depth: int, settings: CausalSettings
    ) -> AdapterBank | None:
        if settings.adapter.mode is not AdapterMode.LOR2C:
            return None
        spec = self.__policy.resolve(hidden_size=bundle_hidden)
        generator = torch.Generator().manual_seed(settings.seed)
        bank = AdapterBank(width=bundle_hidden)
        for entry in ResidualSchedule.per_layer(depth=depth).entries:
            bank.register(name=entry.name, spec=spec, generator=generator)
        return bank
