"""Early stopping policy."""

from pydantic import BaseModel, ConfigDict, Field


class StoppingDecision(BaseModel):
    """Outcome of observing one validation score."""

    model_config = ConfigDict(frozen=True)

    improved: bool = Field(description="Whether the observed score beat the previous best.")
    stop: bool = Field(description="Whether training should halt now.")
    best: float = Field(description="Best score observed so far.")
    epoch: int = Field(description="Epoch at which the best score was observed.")


class EarlyStopping:
    """Halts training once a maximised metric fails to improve for `patience` epochs."""

    def __init__(self, *, patience: int) -> None:
        if patience < 1:
            raise ValueError(f"patience must be at least 1, got {patience}.")
        self.__patience = patience
        self.__best = float("-inf")
        self.__epoch = 0
        self.__stale = 0

    def observe(self, *, score: float, epoch: int) -> StoppingDecision:
        """Record `score` for `epoch` and decide whether to continue."""
        improved = score > self.__best
        if improved:
            self.__best = score
            self.__epoch = epoch
            self.__stale = 0
        else:
            self.__stale += 1
        return StoppingDecision(
            improved=improved,
            stop=self.__stale >= self.__patience,
            best=self.__best,
            epoch=self.__epoch,
        )
