from lor2c.domain.masking import LabelMasker
from lor2c.domain.prompt import PromptTemplate
from lor2c.infrastructure.huggingface.tokenization import InstructionTokenizer
from lor2c.settings.schema import CausalDataSettings


class WordTokenizer:
    """Deterministic stand-in: one token per whitespace-separated word, EOS id 99."""

    eos_token_id = 99

    def __call__(self, text: str, truncation: bool, max_length: int, padding: bool) -> dict:
        ids = [len(word) for word in text.split()][:max_length]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


class TestInstructionTokenizer:
    def __tokenizer(self, *, inputs: bool, eos: bool = False) -> InstructionTokenizer:
        return InstructionTokenizer(
            tokenizer=WordTokenizer(),
            template=PromptTemplate.alpaca(),
            masker=LabelMasker(),
            settings=CausalDataSettings(path="x", cutoff=512, inputs=inputs, eos=eos),
        )

    def test_appends_eos_and_copies_labels_when_training_on_inputs(self) -> None:
        result = self.__tokenizer(inputs=True)(
            {"instruction": "Say hi", "input": "", "output": "hi"}
        )
        assert result["input_ids"][-1] == 99
        assert result["labels"] == result["input_ids"]
        assert len(result["attention_mask"]) == len(result["input_ids"])

    def test_masks_prompt_tokens_when_not_training_on_inputs(self) -> None:
        tokenizer = self.__tokenizer(inputs=False)
        result = tokenizer({"instruction": "Say hi", "input": "", "output": "hi there friend"})
        masked = [label for label in result["labels"] if label == -100]
        unmasked = [label for label in result["labels"] if label != -100]
        assert len(unmasked) == 4  # three response words plus EOS
        assert len(masked) == len(result["input_ids"]) - 4
