import torch
from torch import nn

from lor2c.domain.adapter import AdaptedLinear, LowRankAdapter
from lor2c.domain.schema import AdapterSpec


class TestLowRankAdapter:
    def test_initial_output_is_zero_because_up_is_zero(self, generator: torch.Generator) -> None:
        adapter = LowRankAdapter(
            input_width=8, output_width=8, spec=AdapterSpec(rank=2, alpha=4), generator=generator
        )
        assert torch.all(adapter(torch.randn(3, 8)) == 0)
        assert torch.any(adapter.down.weight != 0)

    def test_gradient_reaches_down_projection(self, generator: torch.Generator) -> None:
        adapter = LowRankAdapter(
            input_width=8, output_width=8, spec=AdapterSpec(rank=2, alpha=4), generator=generator
        )
        with torch.no_grad():
            adapter.up.weight.fill_(0.5)
        adapter(torch.randn(3, 8)).sum().backward()
        assert adapter.down.weight.grad is not None
        assert torch.any(adapter.down.weight.grad != 0)

    def test_scaling_multiplies_output(self, generator: torch.Generator) -> None:
        adapter = LowRankAdapter(
            input_width=4, output_width=4, spec=AdapterSpec(rank=2, alpha=8), generator=generator
        )
        with torch.no_grad():
            adapter.up.weight.fill_(1.0)
        hidden = torch.ones(1, 4)
        raw = adapter.up(adapter.down(hidden))
        assert torch.allclose(adapter(hidden), raw * 4.0)

    def test_same_seed_gives_same_weights(self) -> None:
        spec = AdapterSpec(rank=2, alpha=4)
        first = LowRankAdapter(
            input_width=4, output_width=4, spec=spec, generator=torch.Generator().manual_seed(1)
        )
        second = LowRankAdapter(
            input_width=4, output_width=4, spec=spec, generator=torch.Generator().manual_seed(1)
        )
        assert torch.equal(first.down.weight, second.down.weight)


class TestAdaptedLinear:
    def test_matches_base_before_training(self, generator: torch.Generator) -> None:
        base = nn.Linear(4, 6)
        adapter = LowRankAdapter(
            input_width=4, output_width=6, spec=AdapterSpec(rank=2, alpha=2), generator=generator
        )
        wrapped = AdaptedLinear(base=base, adapter=adapter)
        hidden = torch.randn(2, 4)
        assert torch.allclose(wrapped(hidden), base(hidden))

    def test_base_is_frozen_and_adapter_trainable(self, generator: torch.Generator) -> None:
        wrapped = AdaptedLinear(
            base=nn.Linear(4, 4),
            adapter=LowRankAdapter(
                input_width=4,
                output_width=4,
                spec=AdapterSpec(rank=1, alpha=1),
                generator=generator,
            ),
        )
        assert not wrapped.base.weight.requires_grad
        assert wrapped.adapter.down.weight.requires_grad
