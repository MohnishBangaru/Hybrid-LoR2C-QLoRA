import logging

import pytest

from lor2c.application.schema import Metrics
from lor2c.infrastructure.tracking.noop import LogTracker


class TestLogTracker:
    def test_logs_lifecycle_and_metrics(self, caplog: pytest.LogCaptureFixture) -> None:
        tracker = LogTracker()
        with caplog.at_level(logging.INFO):
            tracker.start(name="run", parameters={"seed": 1})
            tracker.log(metrics=Metrics(step=2, values={"loss": 0.5}))
            tracker.finish()
        messages = [record.getMessage() for record in caplog.records]
        assert messages == ["Run started", "Metrics", "Run finished"]
        assert caplog.records[1].ctx_values == {"loss": 0.5}  # type: ignore[attr-defined]
