import pytest
import torch

from lor2c.domain.adapter import AdaptedLinear
from lor2c.domain.exceptions import AdapterError
from lor2c.domain.schema import AdapterSpec
from lor2c.infrastructure.injection import LinearInjector
from tests.conftest import TinyDecoder


class TestLinearInjector:
    def test_wraps_only_matching_projections(self, decoder: TinyDecoder, depth: int) -> None:
        count = LinearInjector().inject(
            model=decoder, targets=("q_proj", "v_proj"), spec=AdapterSpec(rank=2, alpha=4)
        )
        assert count == 2 * depth
        first = decoder.layers[0]
        assert isinstance(first.q_proj, AdaptedLinear)
        assert isinstance(first.v_proj, AdaptedLinear)
        assert not isinstance(first.o_proj, AdaptedLinear)

    def test_output_is_unchanged_immediately_after_injection(
        self, decoder: TinyDecoder, width: int
    ) -> None:
        hidden = torch.randn(2, width)
        baseline = decoder(hidden)
        LinearInjector().inject(
            model=decoder, targets=("q_proj",), spec=AdapterSpec(rank=2, alpha=4)
        )
        assert torch.allclose(decoder(hidden), baseline)

    def test_no_match_is_an_error(self, decoder: TinyDecoder) -> None:
        with pytest.raises(AdapterError, match=r"No nn\.Linear"):
            LinearInjector().inject(
                model=decoder, targets=("k_proj",), spec=AdapterSpec(rank=2, alpha=4)
            )
