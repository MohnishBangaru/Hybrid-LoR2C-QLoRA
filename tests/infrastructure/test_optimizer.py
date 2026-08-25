from lor2c.domain.schema import AdapterSpec
from lor2c.infrastructure.injection import LinearInjector
from lor2c.infrastructure.optimizer import GroupedOptimizerFactory
from tests.conftest import TinyDecoder


class TestGroupedOptimizerFactory:
    def test_up_projections_get_scaled_learning_rate(
        self, decoder: TinyDecoder, depth: int
    ) -> None:
        LinearInjector().inject(
            model=decoder, targets=("q_proj",), spec=AdapterSpec(rank=2, alpha=2)
        )
        optimizer = GroupedOptimizerFactory(rate=1e-3, ratio=16).build(model=decoder)
        rates = sorted(group["lr"] for group in optimizer.param_groups)
        assert rates == [1e-3, 1.6e-2]
        up_group = next(g for g in optimizer.param_groups if g["lr"] == 1.6e-2)
        assert len(up_group["params"]) == depth

    def test_ratio_one_yields_single_group(self, decoder: TinyDecoder) -> None:
        LinearInjector().inject(
            model=decoder, targets=("q_proj",), spec=AdapterSpec(rank=2, alpha=2)
        )
        optimizer = GroupedOptimizerFactory(rate=1e-3, ratio=1).build(model=decoder)
        assert {group["lr"] for group in optimizer.param_groups} == {1e-3}
