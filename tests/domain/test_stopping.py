import pytest

from lor2c.domain.stopping import EarlyStopping


class TestEarlyStopping:
    def test_stops_after_patience_epochs_without_improvement(self) -> None:
        stopping = EarlyStopping(patience=2)
        assert stopping.observe(score=0.10, epoch=1).improved
        assert not stopping.observe(score=0.09, epoch=2).stop
        decision = stopping.observe(score=0.08, epoch=3)
        assert decision.stop
        assert (decision.best, decision.epoch) == (0.10, 1)

    def test_improvement_resets_patience(self) -> None:
        stopping = EarlyStopping(patience=2)
        stopping.observe(score=0.1, epoch=1)
        stopping.observe(score=0.05, epoch=2)
        assert stopping.observe(score=0.2, epoch=3).improved
        assert not stopping.observe(score=0.1, epoch=4).stop

    def test_rejects_zero_patience(self) -> None:
        with pytest.raises(ValueError):
            EarlyStopping(patience=0)
