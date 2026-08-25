"""Singular-value statistics of adapter update matrices."""

import torch

from lor2c.domain.adapter import LowRankAdapter
from lor2c.domain.bank import AdapterBank


class FeatureSpaceShape:
    """Scores adapters by how concentrated their update matrix spectrum is (SFS from the paper)."""

    def __init__(self, *, top: int | None) -> None:
        self.__top = top

    def concentration(self, *, adapter: LowRankAdapter) -> float:
        """Share of spectral mass in the top-k singular values, or the mean singular value."""
        with torch.no_grad():
            values = torch.linalg.svdvals(adapter.product.float())
        total = float(values.sum())
        if self.__top is None:
            return float(values.mean()) if values.numel() else 0.0
        if total <= 0:
            return 0.0
        count = min(self.__top, values.numel())
        return float(values[:count].sum()) / total

    def score(self, *, bank: AdapterBank) -> dict[str, float]:
        """Concentration for every `LowRankAdapter` in the bank."""
        scores: dict[str, float] = {}
        for name in bank.names:
            adapter = bank.get(name=name)
            if isinstance(adapter, LowRankAdapter):
                scores[name] = self.concentration(adapter=adapter)
        return scores
