import json
import logging

from lor2c.infrastructure.logging import JsonFormatter


class TestJsonFormatter:
    def test_includes_context_fields(self) -> None:
        record = logging.LogRecord("lor2c", logging.INFO, __file__, 1, "Hello %s", ("world",), None)
        record.ctx_step = 3
        payload = json.loads(JsonFormatter().format(record))
        assert payload["message"] == "Hello world"
        assert payload["step"] == 3
        assert payload["level"] == "INFO"
