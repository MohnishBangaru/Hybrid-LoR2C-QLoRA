from pathlib import Path

import pytest

from lor2c.application.causal import CausalTrainingService
from lor2c.domain.exceptions import ConfigurationError
from lor2c.domain.rank import FixedRankPolicy
from lor2c.domain.schema import AdapterSpec
from lor2c.settings.schema import CausalSettings
from tests.application.fakes import (
    FakeDataPort,
    FakeModelPort,
    FakeSeeder,
    FakeTracker,
    FakeTrainer,
    RecordingGate,
    RecordingQuantizer,
    RecordingRepository,
    RecordingRouter,
)


class TestCausalTrainingService:
    def __settings(
        self, *, mode: str, quantization: bool = False, adaptation: dict | None = None
    ) -> CausalSettings:
        return CausalSettings.model_validate(
            {
                "name": "unit",
                "seed": 3,
                "output": "out/unit",
                "model": {"name": "tiny"},
                "data": {"path": "rows"},
                "adapter": {"mode": mode, "automatic": False, "rank": 2, "alpha": 4},
                "quantization": {"enabled": quantization},
                "adaptation": adaptation or {},
            }
        )

    def __service(self, *, ports: dict) -> CausalTrainingService:
        return CausalTrainingService(
            models=ports["models"],
            data=FakeDataPort(),
            trainer=ports["trainer"],
            router=ports["router"],
            quantizer=ports["quantizer"],
            repository=ports["repository"],
            tracker=ports["tracker"],
            seeder=ports["seeder"],
            policy=FixedRankPolicy(spec=AdapterSpec(rank=2, alpha=4)),
            gate=ports["gate"],
        )

    def __ports(self) -> dict:
        router = RecordingRouter()
        trainer = FakeTrainer()
        trainer.router_probe = router
        return {
            "models": FakeModelPort(width=8, depth=3),
            "trainer": trainer,
            "router": router,
            "quantizer": RecordingQuantizer(),
            "repository": RecordingRepository(),
            "tracker": FakeTracker(),
            "seeder": FakeSeeder(),
            "gate": RecordingGate(),
        }

    def test_lor2c_mode_attaches_one_adapter_per_layer_during_training(self) -> None:
        ports = self.__ports()
        outcome = self.__service(ports=ports).run(settings=self.__settings(mode="lor2c"))

        assert ports["trainer"].attached_during_fit is True
        assert ports["router"].attached is False
        router = ports["router"].routers[0]
        assert router.bank.names == ("floor1", "floor2", "floor3")
        assert ports["repository"].banks == [router.bank]
        assert outcome.steps == 7
        assert outcome.output == Path("out/unit")

    def test_base_mode_skips_residual_adapters(self) -> None:
        ports = self.__ports()
        self.__service(ports=ports).run(settings=self.__settings(mode="base"))

        assert ports["router"].routers == []
        assert ports["repository"].banks == [None]
        assert ports["quantizer"].calls == []

    def test_quantization_prepares_before_and_converts_after_training(self) -> None:
        ports = self.__ports()
        self.__service(ports=ports).run(settings=self.__settings(mode="lor2c", quantization=True))
        assert ports["quantizer"].calls == ["prepare", "convert"]

    def test_seeds_and_tracks_lifecycle_even_on_failure(self) -> None:
        ports = self.__ports()

        class ExplodingTrainer(FakeTrainer):
            def fit(self, **kwargs):  # type: ignore[no-untyped-def]
                raise RuntimeError("boom")

        ports["trainer"] = ExplodingTrainer()
        service = self.__service(ports=ports)
        with pytest.raises(RuntimeError, match="boom"):
            service.run(settings=self.__settings(mode="base"))
        assert ports["seeder"].values == [3]
        assert ports["tracker"].events == ["start:unit", "finish"]

    def test_adaptation_installs_observer_and_freezes_attention_for_injection(self) -> None:
        ports = self.__ports()
        self.__service(ports=ports).run(
            settings=self.__settings(mode="lor2c", adaptation={"merges": 1, "injections": 1})
        )
        assert ports["trainer"].observer is not None
        assert ports["gate"].frozen == 1

    def test_merge_only_does_not_freeze_attention(self) -> None:
        ports = self.__ports()
        self.__service(ports=ports).run(
            settings=self.__settings(mode="lor2c", adaptation={"merges": 2})
        )
        assert ports["trainer"].observer is not None
        assert ports["gate"].frozen == 0

    def test_adaptation_with_quantization_is_rejected(self) -> None:
        ports = self.__ports()
        with pytest.raises(ConfigurationError, match="cannot be combined"):
            self.__service(ports=ports).run(
                settings=self.__settings(mode="lor2c", quantization=True, adaptation={"merges": 1})
            )
