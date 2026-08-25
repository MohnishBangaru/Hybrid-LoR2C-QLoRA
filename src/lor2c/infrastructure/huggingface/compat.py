"""Smooths over API differences between supported transformers releases (4.4x and 5.x)."""

import dataclasses
import logging

LOGGER = logging.getLogger(__name__)


class TransformersCompatibility:
    """Adapts keyword arguments to whatever the installed transformers version accepts."""

    LEGACY_DTYPE_KEYWORD = "torch_dtype"
    DTYPE_KEYWORD = "dtype"
    DTYPE_KEYWORD_SINCE = (4, 56)

    def training_arguments(self, **arguments: object) -> dict[str, object]:
        """Drop `TrainingArguments` keys unknown to the installed version, logging each drop."""
        from transformers import TrainingArguments

        known = {field.name for field in dataclasses.fields(TrainingArguments)}
        accepted = {key: value for key, value in arguments.items() if key in known}
        for key in arguments.keys() - known:
            LOGGER.warning(
                "TrainingArguments option unsupported by installed transformers; ignored",
                extra={"ctx_option": key},
            )
        return accepted

    def dtype_keyword(self) -> str:
        """`dtype` from transformers 4.56 onwards, `torch_dtype` on older releases."""
        import transformers

        major, minor = (int(part) for part in transformers.__version__.split(".")[:2])
        if (major, minor) >= self.DTYPE_KEYWORD_SINCE:
            return self.DTYPE_KEYWORD
        return self.LEGACY_DTYPE_KEYWORD
