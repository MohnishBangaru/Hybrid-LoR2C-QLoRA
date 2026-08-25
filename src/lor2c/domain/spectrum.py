"""Singular-value statistics of adapter update matrices."""

import torch

from lor2c.domain.adapter import LowRankAdapter
from lor2c.domain.bank import AdapterBank
from lor2c.domain.constants import Adaptation


class FeatureSpaceShape:
    """Shape of Feature Space (SFS): share of spectral mass outside the top-k singular values.

    Low SFS means the update is concentrated in few directions (low information content);
    high SFS means information is spread across the spectrum.
    """

    def __init__(self, *, top: int | None) -> None:
        self.__top = top

    def score(self, *, adapter: LowRankAdapter) -> float:
        """SFS of `adapter`; k defaults to half the adapter rank when not configured."""
        with torch.no_grad():
            values = torch.linalg.svdvals(adapter.product.float())
        total = float(values.sum())
        if total <= 0:
            return 0.0
        count = (
            self.__top
            if self.__top is not None
            else max(1, adapter.spec.rank // Adaptation.INJECTED_RANK_DIVISOR)
        )
        count = min(count, values.numel())
        return 1.0 - float(values[:count].sum()) / total

    def scores(self, *, bank: AdapterBank) -> dict[str, float]:
        """SFS for every `LowRankAdapter` in the bank."""
        result: dict[str, float] = {}
        for name in bank.names:
            adapter = bank.get(name=name)
            if isinstance(adapter, LowRankAdapter):
                result[name] = self.score(adapter=adapter)
        return result
