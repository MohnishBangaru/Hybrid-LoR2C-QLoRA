"""Caption text normalisation."""

from lor2c.domain.constants import CAPTION_NUMERIC_PREFIX


class CaptionNormalizer:
    """Strips dataset artefacts such as leading enumeration numbers from captions."""

    def normalize(self, *, text: str) -> str:
        """Return `text` without a leading numeric prefix and surrounding whitespace."""
        return CAPTION_NUMERIC_PREFIX.sub("", text, count=1).strip()
