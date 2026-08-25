"""Use-case: fine-tune a vision-language model with low-rank adapters and BLEU early stopping."""

from lor2c.application.ports import (
    Evaluator,
    Repository,
    Seeder,
    Tracker,
    VisionDataPort,
    VisionModelPort,
    VisionTrainerPort,
)
from lor2c.application.schema import Metrics, Outcome
from lor2c.domain.rank import RankPolicy
from lor2c.domain.stopping import EarlyStopping
from lor2c.settings.schema import VisionSettings


class VisionTrainingService:
    """Coordinates epoch-wise training, validation, early stopping and adapter persistence."""

    def __init__(
        self,
        *,
        models: VisionModelPort,
        data: VisionDataPort,
        trainer: VisionTrainerPort,
        evaluator: Evaluator,
        repository: Repository,
        tracker: Tracker,
        seeder: Seeder,
        policy: RankPolicy,
    ) -> None:
        self.__models = models
        self.__data = data
        self.__trainer = trainer
        self.__evaluator = evaluator
        self.__repository = repository
        self.__tracker = tracker
        self.__seeder = seeder
        self.__policy = policy

    def run(self, *, settings: VisionSettings) -> Outcome:
        """Execute the captioning fine-tuning workflow described by `settings`."""
        self.__seeder.seed(value=settings.seed)
        self.__tracker.start(name=settings.name, parameters=settings.model_dump(mode="json"))
        try:
            return self.__execute(settings=settings)
        finally:
            self.__tracker.finish()

    def __execute(self, *, settings: VisionSettings) -> Outcome:
        bundle = self.__models.load(settings=settings.model)
        spec = self.__policy.resolve(hidden_size=bundle.hidden)
        self.__models.adapt(bundle=bundle, spec=spec, settings=settings.adapter)
        loaders = self.__data.load(settings=settings.data, batch=settings.train.batch)
        stopping = EarlyStopping(patience=settings.evaluation.patience)

        steps = 0
        best = 0.0
        best_epoch = 0
        for index in range(1, settings.train.epochs + 1):
            report = self.__trainer.epoch(bundle=bundle, loaders=loaders, index=index)
            steps = report.step
            evaluation = self.__evaluator.evaluate(
                bundle=bundle, loaders=loaders, settings=settings.evaluation
            )
            self.__tracker.log(
                metrics=Metrics(
                    step=steps,
                    values={
                        "epoch": float(index),
                        "epoch.loss": report.loss,
                        "epoch.seconds": report.seconds,
                        "epoch.memory": float(report.memory),
                        "validation.bleu": evaluation.bleu,
                    },
                )
            )
            decision = stopping.observe(score=evaluation.bleu, epoch=index)
            best, best_epoch = decision.best, decision.epoch
            if decision.stop:
                break

        self.__repository.save(model=bundle.model, bank=None, output=settings.output)
        self.__tracker.log(
            metrics=Metrics(step=steps, values={"best.bleu": best, "best.epoch": float(best_epoch)})
        )
        return Outcome(
            name=settings.name, output=settings.output, steps=steps, best=best, epoch=best_epoch
        )
