"""Structured JSON logging for the outer layers."""

import json
import logging
import sys
from datetime import UTC, datetime


class JsonFormatter(logging.Formatter):
    """Serialises each record as one JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        """Render `record` as JSON including any `extra` fields."""
        payload: dict[str, object] = {
            "time": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        for key, value in record.__dict__.items():
            if key.startswith("ctx_"):
                payload[key[4:]] = value
        return json.dumps(payload, default=str)


class LoggingConfigurator:
    """Installs the JSON formatter on the root logger exactly once."""

    def configure(self, *, level: int = logging.INFO) -> None:
        """Route all logs to stderr as JSON lines at `level`."""
        root = logging.getLogger()
        for handler in list(root.handlers):
            root.removeHandler(handler)
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(JsonFormatter())
        root.addHandler(handler)
        root.setLevel(level)
