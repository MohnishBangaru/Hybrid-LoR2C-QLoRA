from pathlib import Path

import pytest
from pydantic import ValidationError

from lor2c.domain.constants import AdapterMode
from lor2c.settings.schema import CausalSettings, CausalTrainSettings, VisionSettings


class TestCausalSettings:
    def test_minimal_document_applies_defaults(self) -> None:
        settings = CausalSettings.model_validate(
            {"name": "run", "output": "out", "model": {"name": "m"}, "data": {"path": "d"}}
        )
        assert settings.adapter.mode is AdapterMode.LOR2C
        assert settings.accumulation == 32
        assert settings.output == Path("out")

    def test_unknown_keys_are_rejected(self) -> None:
        with pytest.raises(ValidationError, match="extra"):
            CausalSettings.model_validate(
                {
                    "name": "run",
                    "output": "o",
                    "model": {"name": "m"},
                    "data": {"path": "d"},
                    "lr": 1,
                }
            )

    def test_batch_must_divide_by_micro(self) -> None:
        with pytest.raises(ValidationError, match="multiple"):
            CausalTrainSettings(batch=10, micro=4)


class TestVisionSettings:
    def test_adapter_defaults_to_explicit_rank(self) -> None:
        settings = VisionSettings.model_validate(
            {"name": "v", "output": "o", "model": {"name": "m"}, "data": {"path": "d"}}
        )
        assert settings.adapter.automatic is False
        assert settings.evaluation.patience == 3
