import pytest
from pydantic import ValidationError

from lor2c.domain.schema import AdapterSpec, ResidualSchedule, ScheduleEntry


class TestAdapterSpec:
    def test_scaling_is_alpha_over_rank(self) -> None:
        assert AdapterSpec(rank=4, alpha=8).scaling == 2.0

    def test_rejects_zero_rank(self) -> None:
        with pytest.raises(ValidationError):
            AdapterSpec(rank=0, alpha=8)

    def test_is_immutable(self) -> None:
        spec = AdapterSpec(rank=4, alpha=8)
        with pytest.raises(ValidationError):
            spec.rank = 2  # type: ignore[misc]


class TestScheduleEntry:
    def test_rejects_end_before_start(self) -> None:
        with pytest.raises(ValidationError, match="ends at 1 before it starts at 2"):
            ScheduleEntry(name="a", start=2, end=1)


class TestResidualSchedule:
    def test_per_layer_bridges_each_layer_to_itself(self) -> None:
        schedule = ResidualSchedule.per_layer(depth=3)
        assert [(e.name, e.start, e.end) for e in schedule.entries] == [
            ("floor1", 0, 0),
            ("floor2", 1, 1),
            ("floor3", 2, 2),
        ]

    def test_rejects_duplicate_names(self) -> None:
        entries = (ScheduleEntry(name="a", start=0, end=0), ScheduleEntry(name="a", start=1, end=1))
        with pytest.raises(ValidationError, match="unique"):
            ResidualSchedule(depth=2, entries=entries)

    def test_rejects_entry_beyond_depth(self) -> None:
        with pytest.raises(ValidationError, match="depth is 2"):
            ResidualSchedule(depth=2, entries=(ScheduleEntry(name="a", start=0, end=2),))

    def test_lookups_by_start_and_end(self) -> None:
        schedule = ResidualSchedule(
            depth=4,
            entries=(
                ScheduleEntry(name="span", start=1, end=3),
                ScheduleEntry(name="self", start=1, end=1),
            ),
        )
        assert [e.name for e in schedule.starting_at(layer=1)] == ["span", "self"]
        assert [e.name for e in schedule.ending_at(layer=3)] == ["span"]
        assert schedule.ending_at(layer=0) == ()


class TestAdapterSpecScaling:
    def test_rank_stabilized_scaling_uses_sqrt_rank(self) -> None:
        from lor2c.domain.constants import Scaling

        spec = AdapterSpec(rank=16, alpha=8, mode=Scaling.STABILIZED)
        assert spec.scaling == 2.0

    def test_halved_keeps_other_fields(self) -> None:
        spec = AdapterSpec(rank=8, alpha=16, dropout=0.1).halved()
        assert (spec.rank, spec.alpha, spec.dropout) == (4, 16, 0.1)


class TestResidualScheduleMutation:
    def test_merged_spans_both_layers_and_drops_originals(self) -> None:
        schedule = ResidualSchedule.per_layer(depth=3).merged(first="floor1", second="floor2")
        names = {entry.name for entry in schedule.entries}
        assert names == {"floor1+floor2", "floor3"}
        joined = schedule.entry(name="floor1+floor2")
        assert (joined.start, joined.end) == (0, 1)

    def test_merge_requires_adjacency(self) -> None:
        from lor2c.domain.exceptions import ScheduleError

        with pytest.raises(ScheduleError, match="not adjacent"):
            ResidualSchedule.per_layer(depth=3).merged(first="floor1", second="floor3")

    def test_without_removes_entry(self) -> None:
        schedule = ResidualSchedule.per_layer(depth=2).without(name="floor1")
        assert [entry.name for entry in schedule.entries] == ["floor2"]
