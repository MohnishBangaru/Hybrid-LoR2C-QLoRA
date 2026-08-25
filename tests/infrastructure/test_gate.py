import pytest

from lor2c.domain.exceptions import AdapterError
from lor2c.domain.schema import AdapterSpec
from lor2c.infrastructure.gate import PatternAttentionGate
from lor2c.infrastructure.injection import LinearInjector
from tests.conftest import TinyDecoder


class TestPatternAttentionGate:
    def test_freeze_then_release_single_layer(self, decoder: TinyDecoder) -> None:
        LinearInjector().inject(
            model=decoder, targets=("q_proj", "v_proj"), spec=AdapterSpec(rank=2, alpha=2)
        )
        gate = PatternAttentionGate(marker=".adapter.")
        gate.freeze(model=decoder)
        assert not any(p.requires_grad for n, p in decoder.named_parameters() if ".adapter." in n)
        gate.release(model=decoder, layer=1)
        trainable = {
            n for n, p in decoder.named_parameters() if p.requires_grad and ".adapter." in n
        }
        assert trainable == {
            "layers.1.q_proj.adapter.down.weight",
            "layers.1.q_proj.adapter.up.weight",
            "layers.1.v_proj.adapter.down.weight",
            "layers.1.v_proj.adapter.up.weight",
        }

    def test_release_without_match_is_an_error(self, decoder: TinyDecoder) -> None:
        with pytest.raises(AdapterError, match="layer 7"):
            PatternAttentionGate().release(model=decoder, layer=7)
