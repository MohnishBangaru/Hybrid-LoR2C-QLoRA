"""Shared lazily-loaded tokenizer/processor so data and trainer ports agree on preprocessing."""

from typing import Protocol

from lor2c.domain.exceptions import ConfigurationError
from lor2c.settings.schema import ModelSettings


class TokenizerLike(Protocol):
    """Subset of the tokenizer interface used by this package."""

    eos_token_id: int
    pad_token_id: int
    padding_side: str


class HubContext:
    """Caches the tokenizer or processor for a run so every port reuses one instance."""

    def __init__(self, *, settings: ModelSettings) -> None:
        self.__settings = settings
        self.__tokenizer: object | None = None
        self.__processor: object | None = None

    @property
    def settings(self) -> ModelSettings:
        """Model settings the context was created for."""
        return self.__settings

    def tokenizer(self) -> object:
        """Return the text tokenizer, loading it on first use."""
        if self.__tokenizer is None:
            try:
                from transformers import AutoTokenizer
            except ImportError as exception:
                raise ConfigurationError(
                    "Install lor2c[huggingface] to load models."
                ) from exception
            tokenizer = AutoTokenizer.from_pretrained(
                self.__settings.name, revision=self.__settings.revision
            )
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = tokenizer.unk_token_id or 0
            tokenizer.padding_side = "left"
            self.__tokenizer = tokenizer
        return self.__tokenizer

    def processor(self, *, image: int) -> object:
        """Return the multimodal processor sized for `image` pixels on the longest edge."""
        if self.__processor is None:
            try:
                from transformers import AutoProcessor
            except ImportError as exception:
                raise ConfigurationError(
                    "Install lor2c[huggingface] to load models."
                ) from exception
            self.__processor = AutoProcessor.from_pretrained(
                self.__settings.name,
                revision=self.__settings.revision,
                size={"longest_edge": image},
            )
        return self.__processor
