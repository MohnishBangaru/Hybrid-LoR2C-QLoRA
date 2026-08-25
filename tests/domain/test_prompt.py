import pytest

from lor2c.domain.exceptions import TemplateError
from lor2c.domain.prompt import PromptTemplate


class TestPromptTemplate:
    def test_renders_with_context(self) -> None:
        text = PromptTemplate.alpaca().render(
            instruction="Translate to French.", context="Hello", response="Bonjour"
        )
        assert "### Instruction:\nTranslate to French." in text
        assert "### Input:\nHello" in text
        assert text.endswith("### Response:\nBonjour")

    def test_renders_without_context(self) -> None:
        text = PromptTemplate.alpaca().render(instruction="Say hi.")
        assert "### Input:" not in text
        assert text.endswith("### Response:\n")

    def test_extracts_response(self) -> None:
        template = PromptTemplate.alpaca()
        text = template.render(instruction="Say hi.", response="hi there")
        assert template.extract_response(text=text) == "hi there"

    def test_missing_marker_is_an_error(self) -> None:
        with pytest.raises(TemplateError):
            PromptTemplate.alpaca().extract_response(text="no marker here")
