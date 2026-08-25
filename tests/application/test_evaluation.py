import json
from pathlib import Path

import pytest

from lor2c.application.evaluation import EvaluationService
from lor2c.application.schema import ResidualManifest
from lor2c.domain.exceptions import ConfigurationError
from lor2c.domain.schema import AdapterSpec, ResidualSchedule
from lor2c.settings.schema import EvaluationRunSettings
from tests.application.fakes import (
    FakeBenchmark,
    FakeModelPort,
    FakeSeeder,
    FakeTracker,
    RecordingRepository,
    RecordingRouter,
)


class TestEvaluationService:
    def __settings(self, *, run: Path) -> EvaluationRunSettings:
        return EvaluationRunSettings.model_validate(
            {
                "name": "eval",
                "run": str(run),
                "model": {"name": "tiny"},
                "benchmark": {"tasks": ["arc_easy"]},
            }
        )

    def __manifest(self, *, quantized: bool = False) -> ResidualManifest:
        spec = AdapterSpec(rank=2, alpha=4)
        schedule = ResidualSchedule.per_layer(depth=3).merged(first="floor1", second="floor2")
        return ResidualManifest(
            width=8,
            specs={"floor1+floor2": spec, "floor3": spec},
            schedule=schedule,
            quantized=quantized,
        )

    def __service(self, *, repository: RecordingRepository) -> tuple[EvaluationService, dict]:
        router = RecordingRouter()
        benchmark = FakeBenchmark()
        benchmark.probe = router
        models = FakeModelPort(width=8, depth=3)
        tracker = FakeTracker()
        service = EvaluationService(
            models=models,
            repository=repository,
            router=router,
            benchmark=benchmark,
            tracker=tracker,
            seeder=FakeSeeder(),
        )
        return service, {
            "router": router,
            "benchmark": benchmark,
            "models": models,
            "tracker": tracker,
        }

    def test_rebuilds_bank_attaches_hooks_and_writes_scores(self, tmp_path: Path) -> None:
        repository = RecordingRepository()
        repository.stored = self.__manifest()
        service, ports = self.__service(repository=repository)

        outcome = service.run(settings=self.__settings(run=tmp_path))

        assert ports["models"].restored_from == tmp_path / "model"
        assert repository.restored is not None
        assert repository.restored.names == ("floor1+floor2", "floor3")
        assert ports["benchmark"].attached is True
        assert ports["router"].routers[0].schedule == repository.stored.schedule
        assert outcome.scores == {"arc_easy/acc_norm": 0.5}
        written = json.loads((tmp_path / "scores.json").read_text())
        assert written["scores"] == {"arc_easy/acc_norm": 0.5}
        assert ports["tracker"].metrics[0].values == {"arc_easy/acc_norm": 0.5}

    def test_run_without_residual_bank_evaluates_attention_adapter_only(
        self, tmp_path: Path
    ) -> None:
        repository = RecordingRepository()
        service, ports = self.__service(repository=repository)
        service.run(settings=self.__settings(run=tmp_path))
        assert ports["router"].routers == []
        assert ports["benchmark"].attached is False

    def test_quantized_bank_is_rejected(self, tmp_path: Path) -> None:
        repository = RecordingRepository()
        repository.stored = self.__manifest(quantized=True)
        service, _ = self.__service(repository=repository)
        with pytest.raises(ConfigurationError, match="int8"):
            service.run(settings=self.__settings(run=tmp_path))
