from lor2c.domain.rank import FixedRankPolicy, WidthRankPolicy
from lor2c.domain.schema import AdapterSpec


class TestWidthRankPolicy:
    def test_tinyllama_width_gets_small_rank(self) -> None:
        spec = WidthRankPolicy(dropout=0.05).resolve(hidden_size=2048)
        assert (spec.rank, spec.alpha, spec.dropout) == (4, 8, 0.05)

    def test_llama_7b_width_gets_large_rank(self) -> None:
        spec = WidthRankPolicy(dropout=0.0).resolve(hidden_size=4096)
        assert (spec.rank, spec.alpha) == (16, 32)


class TestFixedRankPolicy:
    def test_ignores_width(self) -> None:
        spec = AdapterSpec(rank=3, alpha=6)
        policy = FixedRankPolicy(spec=spec)
        assert policy.resolve(hidden_size=1) is spec
        assert policy.resolve(hidden_size=10_000) is spec
