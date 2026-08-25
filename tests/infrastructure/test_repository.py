from pathlib import Path

import pytest
import torch
from torch import nn

from lor2c.application.schema import ResidualManifest
from lor2c.domain.bank import AdapterBank
from lor2c.domain.exceptions import ConfigurationError
from lor2c.domain.schema import AdapterSpec, ResidualSchedule
from lor2c.infrastructure.repository import DiskRepository


class TestDiskRepository:
    def __bank_and_manifest(self, *, shared: bool = False) -> tuple[AdapterBank, ResidualManifest]:
        spec = AdapterSpec(rank=2, alpha=4)
        schedule = ResidualSchedule.per_layer(depth=2)
        bank = AdapterBank(width=4, shared=spec if shared else None)
        for entry in schedule.entries:
            adapter = bank.register(name=entry.name, spec=spec)
            with torch.no_grad():
                adapter.up.weight.normal_()
        manifest = ResidualManifest(
            width=4,
            shared=spec if shared else None,
            specs={name: spec for name in bank.names},
            schedule=schedule,
        )
        return bank, manifest

    def test_save_then_manifest_and_restore_round_trip(self, tmp_path: Path) -> None:
        model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
        model[0].requires_grad_(False)
        bank, manifest = self.__bank_and_manifest()
        repository = DiskRepository()

        repository.save(model=model, bank=bank, output=tmp_path / "run", manifest=manifest)

        weights = torch.load(tmp_path / "run" / "model" / "adapters.pt")
        assert set(weights) == {"1.weight", "1.bias"}
        assert repository.manifest(output=tmp_path / "run") == manifest
        restored = repository.restore(output=tmp_path / "run", manifest=manifest)
        assert restored.names == bank.names
        hidden = torch.randn(3, 4)
        for name in bank.names:
            assert torch.equal(restored.get(name=name)(hidden), bank.get(name=name)(hidden))

    def test_shared_bank_round_trip(self, tmp_path: Path) -> None:
        bank, manifest = self.__bank_and_manifest(shared=True)
        repository = DiskRepository()
        repository.save(model=nn.Linear(1, 1), bank=bank, output=tmp_path, manifest=manifest)
        restored = repository.restore(output=tmp_path, manifest=manifest)
        assert restored.down is not None
        assert torch.equal(restored.down.weight, bank.down.weight)  # type: ignore[union-attr]

    def test_manifest_is_none_without_residual_adapters(self, tmp_path: Path) -> None:
        DiskRepository().save(model=nn.Linear(1, 1), bank=None, output=tmp_path)
        assert DiskRepository().manifest(output=tmp_path) is None
        assert not (tmp_path / "residual").exists()

    def test_bank_without_manifest_is_rejected(self, tmp_path: Path) -> None:
        bank, _ = self.__bank_and_manifest()
        with pytest.raises(ConfigurationError, match="manifest"):
            DiskRepository().save(model=nn.Linear(1, 1), bank=bank, output=tmp_path)

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
