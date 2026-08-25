"""Tracker that only writes to the structured log."""

import logging

from lor2c.application.schema import Metrics

LOGGER = logging.getLogger(__name__)


class LogTracker:
    """Records run lifecycle and metrics through the logging subsystem."""

    def start(self, *, name: str, parameters: dict[str, object]) -> None:
        """Log the run parameters."""
        LOGGER.info("Run started", extra={"ctx_run": name, "ctx_parameters": parameters})

    def log(self, *, metrics: Metrics) -> None:
        """Log the metric values."""
        LOGGER.info("Metrics", extra={"ctx_step": metrics.step, "ctx_values": metrics.values})

    def finish(self) -> None:
        """Log run completion."""
        LOGGER.info("Run finished")
