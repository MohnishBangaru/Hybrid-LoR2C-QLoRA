"""Label masking rules for causal language modelling."""

from collections.abc import Sequence

from lor2c.domain.constants import Label


class LabelMasker:
    """Produces loss labels where prompt and padding positions are ignored."""

    def mask_prefix(self, *, labels: Sequence[int], prefix_length: int) -> list[int]:
        """Ignore the first `prefix_length` positions so the loss only covers the response."""
        if prefix_length < 0:
            raise ValueError(f"prefix_length must be non-negative, got {prefix_length}.")
        bounded = min(prefix_length, len(labels))
        return [int(Label.IGNORED)] * bounded + list(labels[bounded:])

    def mask_value(self, *, labels: Sequence[int], value: int) -> list[int]:
        """Ignore every position equal to `value` (typically the pad token)."""
        return [int(Label.IGNORED) if label == value else label for label in labels]
