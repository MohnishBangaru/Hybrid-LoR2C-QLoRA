import pytest

from lor2c.domain.adaptation import AdaptationPlanner, AdaptationTimeline
from lor2c.domain.schema import ResidualSchedule


class TestAdaptationPlanner:
    def test_merge_picks_lowest_combined_score_among_adjacent(self) -> None:
        schedule = ResidualSchedule.per_layer(depth=4)
        scores = {"floor1": 0.9, "floor2": 0.1, "floor3": 0.1, "floor4": 0.9}
        choice = AdaptationPlanner(span=2).merge(schedule=schedule, scores=scores)
        assert choice is not None
        assert (choice.first, choice.second) == ("floor2", "floor3")

    def test_merge_keeps_member_with_higher_sfs(self) -> None:
        schedule = ResidualSchedule.per_layer(depth=2)
        choice = AdaptationPlanner(span=2).merge(
            schedule=schedule, scores={"floor1": 0.1, "floor2": 0.3}
        )
        assert choice is not None and choice.keep == "floor2"

    def test_merge_respects_span_limit(self) -> None:
        schedule = ResidualSchedule.per_layer(depth=3).merged(first="floor1", second="floor2")
        scores = {"floor1+floor2": 0.0, "floor3": 0.0}
        assert AdaptationPlanner(span=2).merge(schedule=schedule, scores=scores) is None
        assert AdaptationPlanner(span=3).merge(schedule=schedule, scores=scores) is not None

    def test_inject_picks_lowest_sfs_single_layer_adapter(self) -> None:
        schedule = ResidualSchedule.per_layer(depth=4).merged(first="floor1", second="floor2")
        scores = {"floor1+floor2": 0.01, "floor3": 0.4, "floor4": 0.2}
        entry = AdaptationPlanner(span=4).inject(schedule=schedule, scores=scores)
        assert entry is not None and entry.name == "floor4"

    def test_rejects_span_below_two(self) -> None:
        with pytest.raises(ValueError):
            AdaptationPlanner(span=1)


class TestAdaptationTimeline:
    def test_events_spread_over_first_quarter(self) -> None:
        timeline = AdaptationTimeline(epochs=4.0, count=2)
        assert timeline.due(epoch=0.4) == 0
        assert timeline.due(epoch=0.5) == 1
        assert timeline.due(epoch=0.9) == 0
        assert timeline.due(epoch=1.0) == 1
        assert timeline.due(epoch=3.9) == 0
        assert timeline.remaining == 0

    def test_skipped_intervals_are_caught_up(self) -> None:
        timeline = AdaptationTimeline(epochs=4.0, count=2)
        assert timeline.due(epoch=2.0) == 2

    def test_zero_count_never_fires(self) -> None:
        timeline = AdaptationTimeline(epochs=3.0, count=0)
        assert timeline.due(epoch=10.0) == 0
