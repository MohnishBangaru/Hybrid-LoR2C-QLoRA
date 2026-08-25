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
