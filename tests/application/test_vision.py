from lor2c.application.vision import VisionTrainingService
from lor2c.domain.rank import FixedRankPolicy
from lor2c.domain.schema import AdapterSpec
from lor2c.settings.schema import VisionSettings
from tests.application.fakes import (
    FakeSeeder,
    FakeTracker,
    FakeVisionDataPort,
    FakeVisionModelPort,
    FakeVisionTrainer,
    RecordingRepository,
    ScriptedEvaluator,
)


class TestVisionTrainingService:
    def __settings(self, *, epochs: int, patience: int) -> VisionSettings:
        return VisionSettings.model_validate(
            {
                "name": "vision",
                "output": "out/vision",
                "model": {"name": "tiny"},
                "data": {"path": "rows"},
                "train": {"epochs": epochs},
                "evaluation": {"patience": patience},
            }
        )

    def __run(self, *, scores: list[float], epochs: int, patience: int) -> tuple:
        trainer = FakeVisionTrainer()
        tracker = FakeTracker()
        repository = RecordingRepository()
        service = VisionTrainingService(
            models=FakeVisionModelPort(width=8, depth=2),
            data=FakeVisionDataPort(),
            trainer=trainer,
            evaluator=ScriptedEvaluator(scores=scores),
            repository=repository,
            tracker=tracker,
            seeder=FakeSeeder(),
            policy=FixedRankPolicy(spec=AdapterSpec(rank=2, alpha=4)),
        )
        outcome = service.run(settings=self.__settings(epochs=epochs, patience=patience))
        return outcome, trainer, tracker, repository

    def test_stops_early_when_bleu_stalls(self) -> None:
        outcome, trainer, tracker, repository = self.__run(
            scores=[0.3, 0.2, 0.1, 0.05, 0.9], epochs=5, patience=2
        )
        assert trainer.epochs == [1, 2, 3]
        assert (outcome.best, outcome.epoch) == (0.3, 1)
        assert repository.outputs[0].name == "vision"
        assert tracker.metrics[-1].values["best.bleu"] == 0.3

    def test_runs_all_epochs_when_improving(self) -> None:
        outcome, trainer, tracker, _ = self.__run(scores=[0.1, 0.2, 0.3], epochs=3, patience=1)
        assert trainer.epochs == [1, 2, 3]
        assert (outcome.best, outcome.epoch, outcome.steps) == (0.3, 3, 30)
        assert [m.values["validation.bleu"] for m in tracker.metrics[:3]] == [0.1, 0.2, 0.3]
