from pathlib import Path

import pytest

from lor2c.domain.exceptions import ConfigurationError
from lor2c.settings.loader import SettingsLoader
from lor2c.settings.schema import CausalSettings, EvaluationRunSettings


class TestSettingsLoader:
    def test_loads_valid_yaml(self, tmp_path: Path) -> None:
        path = tmp_path / "run.yaml"
        path.write_text("name: run\noutput: out\nmodel: {name: m}\ndata: {path: d}\n")
        settings = SettingsLoader().load(path=path, schema=CausalSettings)
        assert settings.name == "run"

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigurationError, match="not found"):
            SettingsLoader().load(path=tmp_path / "nope.yaml", schema=CausalSettings)

    def test_invalid_document(self, tmp_path: Path) -> None:
        path = tmp_path / "run.yaml"
        path.write_text("name: run\n")
        with pytest.raises(ConfigurationError, match="invalid"):
            SettingsLoader().load(path=path, schema=CausalSettings)

    def test_non_mapping_document(self, tmp_path: Path) -> None:
        path = tmp_path / "run.yaml"
        path.write_text("- a\n- b\n")
        with pytest.raises(ConfigurationError, match="mapping"):
            SettingsLoader().load(path=path, schema=CausalSettings)

    def test_repository_configs_are_valid(self) -> None:
        root = Path(__file__).resolve().parents[2] / "configs"
        SettingsLoader().load(path=root / "tinyllama.yaml", schema=CausalSettings)
        SettingsLoader().load(path=root / "tinyllama_qat.yaml", schema=CausalSettings)
        SettingsLoader().load(path=root / "tinyllama_im.yaml", schema=CausalSettings)
        SettingsLoader().load(path=root / "tinyllama_qlora.yaml", schema=CausalSettings)
        SettingsLoader().load(path=root / "llama2_7b_im.yaml", schema=CausalSettings)
        SettingsLoader().load(path=root / "evaluate.yaml", schema=EvaluationRunSettings)
