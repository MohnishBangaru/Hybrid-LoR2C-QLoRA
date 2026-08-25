import json
from pathlib import Path

import torch
from torch import nn

from lor2c.domain.bank import AdapterBank
from lor2c.domain.schema import AdapterSpec
from lor2c.infrastructure.repository import DiskRepository


class TestDiskRepository:
    def test_saves_trainable_parameters_and_bank(self, tmp_path: Path) -> None:
        model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
        model[0].requires_grad_(False)
        bank = AdapterBank(width=4)
        bank.register(name="floor1", spec=AdapterSpec(rank=1, alpha=1))

        DiskRepository().save(model=model, bank=bank, output=tmp_path / "run")

        weights = torch.load(tmp_path / "run" / "model" / "adapters.pt")
        assert set(weights) == {"1.weight", "1.bias"}
        manifest = json.loads((tmp_path / "run" / "residual" / "manifest.json").read_text())
        assert manifest == {"width": 4, "names": ["floor1"]}
        restored = AdapterBank(width=4)
        restored.register(name="floor1", spec=AdapterSpec(rank=1, alpha=1))
        restored.load_state_dict(torch.load(tmp_path / "run" / "residual" / "adapters.pt"))
        assert torch.equal(
            restored.get(name="floor1").down.weight, bank.get(name="floor1").down.weight
        )  # type: ignore[union-attr]

    def test_uses_save_pretrained_when_available(self, tmp_path: Path) -> None:
        class Pretrained(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.saved: str | None = None

            def save_pretrained(self, save_directory: str) -> None:
                self.saved = save_directory

        model = Pretrained()
        DiskRepository().save(model=model, bank=None, output=tmp_path)
        assert model.saved == str(tmp_path / "model")
        assert not (tmp_path / "residual").exists()
