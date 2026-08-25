"""Routing of residual adapter deltas across decoder layers."""

from torch import Tensor

from lor2c.domain.bank import AdapterBank
from lor2c.domain.exceptions import ScheduleError
from lor2c.domain.schema import ResidualSchedule


class ResidualRouter:
    """Computes adapter deltas at schedule start layers and injects them at end layers."""

    def __init__(self, *, bank: AdapterBank, schedule: ResidualSchedule) -> None:
        self.__bank = bank
        self.__schedule = schedule
        self.__pending: dict[str, Tensor] = {}
        self.__validate()

    def __validate(self) -> None:
        missing = [
            entry.name for entry in self.__schedule.entries if entry.name not in self.__bank.names
        ]
        if missing:
            raise ScheduleError(f"Schedule references adapters missing from the bank: {missing}.")

    @property
    def schedule(self) -> ResidualSchedule:
        """The routing plan this router executes."""
        return self.__schedule

    @property
    def bank(self) -> AdapterBank:
        """The adapters this router reads from."""
        return self.__bank

    def route(self, *, layer: int, hidden_in: Tensor, hidden_out: Tensor) -> Tensor:
        """Capture deltas for entries starting at `layer`; apply deltas for entries ending there."""
        self.__capture(layer=layer, hidden_in=hidden_in)
        return self.__apply(layer=layer, hidden_out=hidden_out)

    def reset(self) -> None:
        """Drop any deltas captured but never applied (for example after an interrupted forward)."""
        self.__pending.clear()

    def __capture(self, *, layer: int, hidden_in: Tensor) -> None:
        for entry in self.__schedule.starting_at(layer=layer):
            adapter = self.__bank.get(name=entry.name)
            self.__pending[entry.name] = adapter(hidden_in.to(adapter.dtype))

    def __apply(self, *, layer: int, hidden_out: Tensor) -> Tensor:
        result = hidden_out
        for entry in self.__schedule.ending_at(layer=layer):
            try:
                delta = self.__pending.pop(entry.name)
            except KeyError as exception:
                raise ScheduleError(
                    f"Adapter '{entry.name}' should end at layer {layer} but no delta was captured "
                    f"at layer {entry.start}; check that hooks cover every scheduled layer."
                ) from exception
            result = result + delta.to(dtype=hidden_out.dtype, device=hidden_out.device)
        return result
