"""Instruction dataset loading through `datasets`."""

from lor2c.application.schema import Split
from lor2c.domain.exceptions import ConfigurationError
from lor2c.domain.masking import LabelMasker
from lor2c.domain.prompt import PromptTemplate
from lor2c.infrastructure.huggingface.context import HubContext
from lor2c.infrastructure.huggingface.tokenization import InstructionTokenizer
from lor2c.settings.schema import CausalDataSettings


class HubCausalDataPort:
    """Loads a hub or local JSON dataset and tokenises it with the run's template."""

    LOCAL_SUFFIXES = (".json", ".jsonl")

    def __init__(
        self, *, context: HubContext, template: PromptTemplate, masker: LabelMasker
    ) -> None:
        self.__context = context
        self.__template = template
        self.__masker = masker

    def load(self, *, settings: CausalDataSettings, seed: int) -> Split:
        """Return shuffled, tokenised train and optional validation datasets."""
        try:
            from datasets import load_dataset
        except ImportError as exception:
            raise ConfigurationError(
                f"Install lor2c[huggingface] to load datasets ({exception})."
            ) from exception
        if settings.path.endswith(self.LOCAL_SUFFIXES):
            raw = load_dataset("json", data_files=settings.path)
        else:
            raw = load_dataset(settings.path)
        tokenize = InstructionTokenizer(
            tokenizer=self.__context.tokenizer(),
            template=self.__template,
            masker=self.__masker,
            settings=settings,
        )
        source = raw["train"]
        if settings.validation == 0:
            return Split(train=source.shuffle(seed=seed).map(tokenize), validation=None)
        parts = source.train_test_split(test_size=settings.validation, shuffle=True, seed=seed)
        return Split(
            train=parts["train"].shuffle(seed=seed).map(tokenize),
            validation=parts["test"].map(tokenize),
        )
