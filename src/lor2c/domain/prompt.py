"""Instruction prompt templating."""

from pydantic import BaseModel, ConfigDict, Field

from lor2c.domain.constants import Alpaca
from lor2c.domain.exceptions import TemplateError


class PromptTemplate(BaseModel):
    """Renders instruction/context/response triples into a single training prompt."""

    model_config = ConfigDict(frozen=True)

    description: str = Field(description="Human readable summary of the template.")
    with_input: str = Field(description="Format used when a context input is present.")
    without_input: str = Field(description="Format used when no context input is present.")
    response_split: str = Field(description="Marker separating the prompt from the response.")

    @classmethod
    def alpaca(cls) -> "PromptTemplate":
        """The Alpaca-LoRA default template."""
        return cls(
            description=Alpaca.DESCRIPTION,
            with_input=Alpaca.WITH_INPUT,
            without_input=Alpaca.WITHOUT_INPUT,
            response_split=Alpaca.RESPONSE_SPLIT,
        )

    def render(
        self, *, instruction: str, context: str | None = None, response: str | None = None
    ) -> str:
        """Build the prompt, appending the response when one is supplied."""
        if context:
            prompt = self.with_input.format(instruction=instruction, context=context)
        else:
            prompt = self.without_input.format(instruction=instruction)
        return f"{prompt}{response}" if response else prompt

    def extract_response(self, *, text: str) -> str:
        """Return the generated response portion of a decoded model output."""
        head, separator, tail = text.partition(self.response_split)
        if not separator:
            raise TemplateError(f"Output does not contain response marker '{self.response_split}'.")
        return tail.strip()
