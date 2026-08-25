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
