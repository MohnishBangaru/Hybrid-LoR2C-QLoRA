"""Use-case: rebuild a trained run and score it on downstream benchmarks."""

import json
import logging

from lor2c.application.ports import Benchmark, CausalModelPort, Repository, Router, Seeder, Tracker
from lor2c.application.schema import EvaluationOutcome, Metrics
from lor2c.domain.exceptions import ConfigurationError
from lor2c.domain.schedule import ResidualRouter
from lor2c.infrastructure.repository import DiskRepository
from lor2c.settings.schema import EvaluationRunSettings

LOGGER = logging.getLogger(__name__)


class EvaluationService:
    """Loads base + attention adapter + residual bank, attaches hooks, runs the benchmark."""

    SCORES_FILE = "scores.json"

    def __init__(
        self,
        *,
        models: CausalModelPort,
        repository: Repository,
        router: Router,
        benchmark: Benchmark,
        tracker: Tracker,
        seeder: Seeder,
    ) -> None:
        self.__models = models
        self.__repository = repository
        self.__router = router
        self.__benchmark = benchmark
        self.__tracker = tracker
        self.__seeder = seeder

    def run(self, *, settings: EvaluationRunSettings) -> EvaluationOutcome:
        """Evaluate the run at `settings.run` and write `scores.json` next to it."""
        self.__seeder.seed(value=settings.seed)
        self.__tracker.start(name=settings.name, parameters=settings.model_dump(mode="json"))
        try:
            return self.__execute(settings=settings)
        finally:
            self.__tracker.finish()

    def __execute(self, *, settings: EvaluationRunSettings) -> EvaluationOutcome:
        bundle = self.__models.load(settings=settings.model)
        bundle = self.__models.restore(
            bundle=bundle, path=settings.run / DiskRepository.MODEL_DIRECTORY
        )
        manifest = self.__repository.manifest(output=settings.run)
        if manifest is not None and manifest.quantized:
            raise ConfigurationError(
                "Evaluation of int8-converted residual adapters is not supported; rerun training "
                "with quantization.enabled: false to benchmark."
            )
        bundle.model.eval()
        adapters = self.__count(model=bundle.model)
        if manifest is None:
            report = self.__benchmark.run(
                model=bundle.model, settings=settings.benchmark, seed=settings.seed
            )
        else:
            bank = self.__repository.restore(output=settings.run, manifest=manifest)
            adapters += sum(parameter.numel() for parameter in bank.parameters())
            router = ResidualRouter(bank=bank, schedule=manifest.schedule)
            with self.__router.attach(model=bundle.model, router=router):
                report = self.__benchmark.run(
                    model=bundle.model, settings=settings.benchmark, seed=settings.seed
                )
        self.__tracker.log(metrics=Metrics(step=0, values=report.scores))
        output = settings.run / self.SCORES_FILE
        output.write_text(
            json.dumps(
                {"name": settings.name, "adapters": adapters, "scores": report.scores}, indent=2
            ),
            encoding="utf-8",
        )
        LOGGER.info(
            "Evaluation complete", extra={"ctx_scores": report.scores, "ctx_adapters": adapters}
        )
        return EvaluationOutcome(
            name=settings.name, output=output, scores=report.scores, adapters=adapters
        )

    @staticmethod
    def __count(*, model: object) -> int:
        """Parameters belonging to attention adapters (peft names them with `lora_`)."""
        named = getattr(model, "named_parameters", None)
        if named is None:
            return 0
        return sum(parameter.numel() for name, parameter in named() if "lora_" in name)
