import torch

from lor2c.domain.adapter import LowRankAdapter
from lor2c.domain.bank import AdapterBank
from lor2c.domain.schema import AdapterSpec
from lor2c.domain.spectrum import FeatureSpaceShape


class TestFeatureSpaceShape:
    def __adapter(self, *, diagonal: list[float]) -> LowRankAdapter:
        adapter = LowRankAdapter(input_width=4, output_width=4, spec=AdapterSpec(rank=2, alpha=2))
        with torch.no_grad():
            adapter.down.weight.zero_()
            adapter.up.weight.zero_()
            for index, value in enumerate(diagonal):
                adapter.down.weight[index, index] = 1.0
                adapter.up.weight[index, index] = value
        return adapter

    def test_sfs_is_mass_outside_top_k(self) -> None:
        adapter = self.__adapter(diagonal=[3.0, 1.0])
        assert FeatureSpaceShape(top=1).score(adapter=adapter) == 0.25

    def test_k_defaults_to_half_rank(self) -> None:
        adapter = self.__adapter(diagonal=[3.0, 1.0])  # rank 2 -> k = 1
        assert FeatureSpaceShape(top=None).score(adapter=adapter) == 0.25

    def test_concentrated_update_has_lower_sfs(self) -> None:
        spread = FeatureSpaceShape(top=1).score(adapter=self.__adapter(diagonal=[1.0, 1.0]))
        peaked = FeatureSpaceShape(top=1).score(adapter=self.__adapter(diagonal=[9.0, 1.0]))
        assert peaked < spread

    def test_scores_every_adapter_in_bank(self) -> None:
        bank = AdapterBank(width=4)
        bank.register(name="a", spec=AdapterSpec(rank=2, alpha=2))
        bank.register(name="b", spec=AdapterSpec(rank=2, alpha=2))
        assert set(FeatureSpaceShape(top=1).scores(bank=bank)) == {"a", "b"}
