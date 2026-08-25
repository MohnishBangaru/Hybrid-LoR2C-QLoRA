"""Policies that decide adapter hyper-parameters from model geometry."""

from typing import Protocol

from lor2c.domain.constants import Rank, Scaling
from lor2c.domain.schema import AdapterSpec


class RankPolicy(Protocol):
    """Strategy resolving an `AdapterSpec` for a model of a given hidden width."""

    def resolve(self, *, hidden_size: int) -> AdapterSpec:
        """Return the adapter specification to use."""
        ...


class FixedRankPolicy:
    """Always returns the same specification regardless of model width."""

    def __init__(self, *, spec: AdapterSpec) -> None:
        self.__spec = spec

    def resolve(self, *, hidden_size: int) -> AdapterSpec:
        """Return the configured specification."""
        return self.__spec


class WidthRankPolicy:
    """Chooses a smaller rank for narrow models and a larger rank for wide models."""

    def __init__(self, *, dropout: float, mode: Scaling = Scaling.STANDARD) -> None:
        self.__dropout = dropout
        self.__mode = mode

    def resolve(self, *, hidden_size: int) -> AdapterSpec:
        """Pick rank/alpha by comparing `hidden_size` with the small-model threshold."""
        if hidden_size <= Rank.SMALL_HIDDEN_LIMIT:
            return AdapterSpec(
                rank=Rank.SMALL_RANK,
                alpha=Rank.SMALL_ALPHA,
                dropout=self.__dropout,
                mode=self.__mode,
            )
        return AdapterSpec(
            rank=Rank.LARGE_RANK, alpha=Rank.LARGE_ALPHA, dropout=self.__dropout, mode=self.__mode
        )
