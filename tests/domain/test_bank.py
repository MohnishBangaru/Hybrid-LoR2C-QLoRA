import pytest
import torch

from lor2c.domain.bank import AdapterBank
from lor2c.domain.exceptions import AdapterError
from lor2c.domain.schema import AdapterSpec


class TestAdapterBank:
    def test_register_and_get(self, generator: torch.Generator) -> None:
        bank = AdapterBank(width=8)
        adapter = bank.register(
            name="floor1", spec=AdapterSpec(rank=2, alpha=4), generator=generator
        )
        assert bank.get(name="floor1") is adapter
        assert bank.names == ("floor1",)

    def test_duplicate_registration_fails(self) -> None:
        bank = AdapterBank(width=8)
        bank.register(name="floor1", spec=AdapterSpec(rank=2, alpha=4))
        with pytest.raises(AdapterError, match="already registered"):
            bank.register(name="floor1", spec=AdapterSpec(rank=2, alpha=4))

    def test_missing_adapter_fails(self) -> None:
        with pytest.raises(AdapterError, match="not registered"):
            AdapterBank(width=8).get(name="ghost")

    def test_parameters_are_tracked_for_optimisers(self) -> None:
        bank = AdapterBank(width=8)
        bank.register(name="floor1", spec=AdapterSpec(rank=2, alpha=4))
        bank.register(name="floor2", spec=AdapterSpec(rank=2, alpha=4))
        assert sum(p.numel() for p in bank.parameters()) == 2 * (8 * 2 + 2 * 8)

    def test_shared_down_projection_is_one_module(self) -> None:
        spec = AdapterSpec(rank=2, alpha=4)
        bank = AdapterBank(width=8, shared=spec)
        first = bank.register(name="floor1", spec=spec)
        second = bank.register(name="floor2", spec=spec)
        assert first.down is second.down is bank.down  # type: ignore[union-attr]
        assert sum(p.numel() for p in bank.parameters()) == 8 * 2 + 2 * (2 * 8)

    def test_shared_bank_rejects_mismatched_rank(self) -> None:
        bank = AdapterBank(width=8, shared=AdapterSpec(rank=2, alpha=4))
        with pytest.raises(AdapterError, match="shared projection"):
            bank.register(name="floor1", spec=AdapterSpec(rank=4, alpha=4))

    def test_rename_and_remove(self) -> None:
        bank = AdapterBank(width=8)
        adapter = bank.register(name="floor1", spec=AdapterSpec(rank=2, alpha=4))
        bank.register(name="floor2", spec=AdapterSpec(rank=2, alpha=4))
        bank.rename(old="floor1", new="floor1+floor2")
        bank.remove(name="floor2")
        assert bank.names == ("floor1+floor2",)
        assert bank.get(name="floor1+floor2") is adapter
