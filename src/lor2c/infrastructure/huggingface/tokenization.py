"""Turns instruction rows into token/label dictionaries."""

from collections.abc import Mapping

from lor2c.domain.masking import LabelMasker
from lor2c.domain.prompt import PromptTemplate
from lor2c.settings.schema import CausalDataSettings


class InstructionTokenizer:
    """Renders a prompt, tokenises it and masks prompt tokens when `settings.inputs` is false."""

    INSTRUCTION = "instruction"
    INPUT = "input"
    OUTPUT = "output"

    def __init__(
        self,
        *,
        tokenizer: object,
        template: PromptTemplate,
        masker: LabelMasker,
        settings: CausalDataSettings,
    ) -> None:
        self.__tokenizer = tokenizer
        self.__template = template
        self.__masker = masker
        self.__settings = settings

    def __call__(self, row: Mapping[str, str]) -> dict[str, list[int]]:
        """Tokenise one dataset row."""
        full = self.__template.render(
            instruction=row[self.INSTRUCTION],
            context=row.get(self.INPUT),
            response=row[self.OUTPUT],
        )
        encoded = self.__encode(text=full, eos=True)
        if self.__settings.inputs:
            return encoded
        prompt = self.__template.render(
            instruction=row[self.INSTRUCTION], context=row.get(self.INPUT)
        )
        prefix = len(self.__encode(text=prompt, eos=self.__settings.eos)["input_ids"])
        if self.__settings.eos:
            prefix -= 1
        encoded["labels"] = self.__masker.mask_prefix(
            labels=encoded["labels"], prefix_length=prefix
        )
        return encoded

    def __encode(self, *, text: str, eos: bool) -> dict[str, list[int]]:
        tokenizer = self.__tokenizer
        result = tokenizer(  # type: ignore[operator]
            text, truncation=True, max_length=self.__settings.cutoff, padding=False
        )
        ids: list[int] = list(result["input_ids"])
        mask: list[int] = list(result["attention_mask"])
        eos_id: int = tokenizer.eos_token_id  # type: ignore[attr-defined]
        if eos and ids[-1] != eos_id and len(ids) < self.__settings.cutoff:
            ids.append(eos_id)
            mask.append(1)
        return {"input_ids": ids, "attention_mask": mask, "labels": list(ids)}
