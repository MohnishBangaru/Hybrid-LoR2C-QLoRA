"""Weights & Biases tracker."""

from lor2c.application.schema import Metrics
from lor2c.domain.exceptions import ConfigurationError


class WandbTracker:
    """Forwards run lifecycle and metrics to a Weights & Biases project."""

    def __init__(self, *, project: str, run: str | None) -> None:
        try:
            import wandb
        except ImportError as exception:
            raise ConfigurationError(
                "tracking.kind is 'wandb' but the package is missing; install lor2c[tracking]."
            ) from exception
        self.__wandb = wandb
        self.__project = project
        self.__run = run

    def start(self, *, name: str, parameters: dict[str, object]) -> None:
        """Initialise a run named `run` (defaults to the settings name)."""
        self.__wandb.init(project=self.__project, name=self.__run or name, config=parameters)

    def log(self, *, metrics: Metrics) -> None:
        """Send metrics for the given step."""
        self.__wandb.log(metrics.values, step=metrics.step)

    def finish(self) -> None:
        """Close the run."""
        self.__wandb.finish()
