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

    def test_top_k_concentration(self) -> None:
        adapter = self.__adapter(diagonal=[3.0, 1.0])
        assert FeatureSpaceShape(top=1).concentration(adapter=adapter) == 0.75

    def test_mean_singular_value_when_top_unset(self) -> None:
        adapter = self.__adapter(diagonal=[3.0, 1.0])
        # singular values of a 4x4 rank-2 product: 3, 1, 0, 0 -> mean 1.0
        assert FeatureSpaceShape(top=None).concentration(adapter=adapter) == 1.0

    def test_scores_every_adapter_in_bank(self) -> None:
        bank = AdapterBank(width=4)
        bank.register(name="a", spec=AdapterSpec(rank=2, alpha=2))
        bank.register(name="b", spec=AdapterSpec(rank=2, alpha=2))
        assert set(FeatureSpaceShape(top=1).score(bank=bank)) == {"a", "b"}
