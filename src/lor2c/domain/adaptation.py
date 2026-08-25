"""Merge and inject decisions for IMLoR2C."""

from itertools import pairwise

from pydantic import BaseModel, ConfigDict, Field

from lor2c.domain.constants import Adaptation
from lor2c.domain.schema import ResidualSchedule, ScheduleEntry


class MergeChoice(BaseModel):
    """Two adjacent adapters selected to be merged."""

    model_config = ConfigDict(frozen=True)

    first: str = Field(description="Adapter covering the lower layers.")
    second: str = Field(description="Adapter covering the higher layers.")
    keep: str = Field(description="Member whose weights survive (higher information content).")


class AdaptationPlanner:
    """Chooses adapters by SFS: merge the pair with the least information, inject the single
    adapter with the least information (paper Sections MergeLoR2C / InjectLoR2C)."""

    def __init__(self, *, span: int) -> None:
        if span < 2:
            raise ValueError(f"span must allow at least two layers, got {span}.")
        self.__span = span

    def merge(self, *, schedule: ResidualSchedule, scores: dict[str, float]) -> MergeChoice | None:
        """Adjacent pair with the minimum combined SFS whose union fits within `span` layers."""
        ordered = schedule.ordered
        best: tuple[float, MergeChoice] | None = None
        for lower, upper in pairwise(ordered):
            if lower.end + 1 != upper.start:
                continue
            if (upper.end - lower.start + 1) > self.__span:
                continue
            if lower.name not in scores or upper.name not in scores:
                continue
            total = scores[lower.name] + scores[upper.name]
            keep = lower.name if scores[lower.name] >= scores[upper.name] else upper.name
            if best is None or total < best[0]:
                best = (total, MergeChoice(first=lower.name, second=upper.name, keep=keep))
        return best[1] if best else None

    def inject(
        self, *, schedule: ResidualSchedule, scores: dict[str, float]
    ) -> ScheduleEntry | None:
        """Single-layer adapter with the minimum SFS, to be replaced by attention LoRA."""
        candidates = [
            entry for entry in schedule.entries if entry.start == entry.end and entry.name in scores
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda entry: scores[entry.name])


class AdaptationTimeline:
    """Spreads `count` events evenly over the first quarter-per-event window of training."""

    def __init__(self, *, epochs: float, count: int) -> None:
        self.__count = count
        active = epochs / Adaptation.ACTIVE_FRACTION_DENOMINATOR
        self.__interval = active / count if count > 0 else float("inf")
        self.__fired = 0

    @property
    def remaining(self) -> int:
        """Events not yet fired."""
        return self.__count - self.__fired

    def due(self, *, epoch: float) -> int:
        """Number of events that should fire now given training progress `epoch`."""
        if self.__fired >= self.__count:
            return 0
        allowed = min(self.__count, int(epoch // self.__interval) if self.__interval > 0 else 0)
        pending = max(0, allowed - self.__fired)
        self.__fired += pending
        return pending
