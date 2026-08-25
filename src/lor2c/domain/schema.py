"""Immutable value objects describing adapters and residual routing."""

import math

from pydantic import BaseModel, ConfigDict, Field, model_validator

from lor2c.domain.constants import MERGED_NAME_SEPARATOR, RESIDUAL_ADAPTER_PREFIX, Scaling
from lor2c.domain.exceptions import ScheduleError


class AdapterSpec(BaseModel):
    """Hyper-parameters of one low-rank adapter."""

    model_config = ConfigDict(frozen=True)

    rank: int = Field(gt=0, description="Inner dimension of the low-rank factorisation.")
    alpha: float = Field(gt=0, description="Numerator of the residual scaling factor.")
    dropout: float = Field(ge=0, lt=1, default=0.0, description="Dropout applied to the input.")
    mode: Scaling = Field(default=Scaling.STANDARD, description="Standard alpha/r or rsLoRA.")

    @property
    def scaling(self) -> float:
        """Scale applied to the adapter output: alpha/r, or alpha/sqrt(r) when stabilized."""
        if self.mode is Scaling.STABILIZED:
            return self.alpha / math.sqrt(self.rank)
        return self.alpha / self.rank

    def halved(self) -> "AdapterSpec":
        """Specification with half the rank (used when a residual adapter is injected)."""
        return self.model_copy(update={"rank": max(1, self.rank // 2)})


class ScheduleEntry(BaseModel):
    """One adapter that reads the input of layer `start` and writes to the output of layer `end`."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(min_length=1, description="Unique adapter identifier.")
    start: int = Field(ge=0, description="Decoder layer whose input feeds the adapter.")
    end: int = Field(ge=0, description="Decoder layer whose output receives the adapter delta.")

    @model_validator(mode="after")
    def __validate_order(self) -> "ScheduleEntry":
        if self.end < self.start:
            raise ValueError(
                f"Entry '{self.name}' ends at {self.end} before it starts at {self.start}."
            )
        return self


class ResidualSchedule(BaseModel):
    """Complete routing plan of residual adapters across a stack of decoder layers."""

    model_config = ConfigDict(frozen=True)

    depth: int = Field(gt=0, description="Number of decoder layers in the host model.")
    entries: tuple[ScheduleEntry, ...] = Field(description="Adapters and the layers they bridge.")

    @model_validator(mode="after")
    def __validate_entries(self) -> "ResidualSchedule":
        names = [entry.name for entry in self.entries]
        if len(names) != len(set(names)):
            raise ValueError("Schedule entry names must be unique.")
        for entry in self.entries:
            if entry.end >= self.depth:
                raise ValueError(
                    f"Entry '{entry.name}' ends at layer {entry.end} but depth is {self.depth}."
                )
        return self

    @classmethod
    def per_layer(cls, *, depth: int) -> "ResidualSchedule":
        """Build the canonical LoR2C plan: one adapter bridging each decoder layer to itself."""
        entries = tuple(
            ScheduleEntry(name=f"{RESIDUAL_ADAPTER_PREFIX}{index + 1}", start=index, end=index)
            for index in range(depth)
        )
        return cls(depth=depth, entries=entries)

    @property
    def ordered(self) -> tuple[ScheduleEntry, ...]:
        """Entries sorted by start layer."""
        return tuple(sorted(self.entries, key=lambda entry: entry.start))

    def entry(self, *, name: str) -> ScheduleEntry:
        """Return the entry called `name`."""
        for candidate in self.entries:
            if candidate.name == name:
                return candidate
        raise ScheduleError(f"Schedule has no entry named '{name}'.")

    def merged(self, *, first: str, second: str) -> "ResidualSchedule":
        """Schedule where `first` and `second` are replaced by one entry spanning both."""
        one, two = self.entry(name=first), self.entry(name=second)
        if one.end + 1 != two.start:
            raise ScheduleError(f"Entries '{first}' and '{second}' are not adjacent.")
        joined = ScheduleEntry(
            name=f"{first}{MERGED_NAME_SEPARATOR}{second}", start=one.start, end=two.end
        )
        rest = tuple(entry for entry in self.entries if entry.name not in {first, second})
        return ResidualSchedule(depth=self.depth, entries=(*rest, joined))

    def without(self, *, name: str) -> "ResidualSchedule":
        """Schedule with the named entry removed."""
        self.entry(name=name)
        return ResidualSchedule(
            depth=self.depth, entries=tuple(entry for entry in self.entries if entry.name != name)
        )

    def starting_at(self, *, layer: int) -> tuple[ScheduleEntry, ...]:
        """Entries whose delta is computed from the input of `layer`."""
        return tuple(entry for entry in self.entries if entry.start == layer)

    def ending_at(self, *, layer: int) -> tuple[ScheduleEntry, ...]:
        """Entries whose delta is added to the output of `layer`."""
        return tuple(entry for entry in self.entries if entry.end == layer)
