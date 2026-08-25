import torch

from lor2c.domain.caption import CaptionNormalizer
from lor2c.infrastructure.huggingface.captions import CaptionDataset
from lor2c.settings.schema import VisionDataSettings


class WordProcessor:
    """Stand-in processor: one token per word, image expands to three tokens, right padding."""

    IMAGE_TOKENS = 3
    PAD = 0

    class Tokenizer:
        pad_token_id = 0

    tokenizer = Tokenizer()

    def apply_chat_template(self, messages, add_generation_prompt, tokenize):
        text = "User: <image> " + messages[0]["content"][1]["text"] + " Assistant:"
        if not add_generation_prompt:
            text += " " + messages[1]["content"] + " <end>"
        return text

    def __call__(
        self, text, images, return_tensors, padding=None, truncation=False, max_length=None
    ):
        ids = []
        for word in text.split():
            ids.extend([1] * self.IMAGE_TOKENS if word == "<image>" else [len(word) + 1])
        mask = [1] * len(ids)
        if padding == "max_length":
            ids += [self.PAD] * (max_length - len(ids))
            mask += [0] * (max_length - len(mask))
        return {"input_ids": torch.tensor([ids]), "attention_mask": torch.tensor([mask])}


class FakeImage:
    width = 8
    height = 8

    def convert(self, mode):
        return self


class TestCaptionDataset:
    def __dataset(self, *, generation: bool) -> CaptionDataset:
        rows = [{"image": FakeImage(), "text": "1. two cats sleeping"}]
        settings = VisionDataSettings(path="x", samples=1, length=20, image=16)
        return CaptionDataset(
            rows=rows,
            processor=WordProcessor(),
            settings=settings,
            normalizer=CaptionNormalizer(),
            generation=generation,
        )

    def test_labels_cover_only_the_caption(self) -> None:
        sample = self.__dataset(generation=False)[0]
        # prompt: "User:" + 3 image tokens + "Describe" "this" "image." + "Assistant:" = 8 tokens
        # caption: "two" "cats" "sleeping" "<end>" = 4 tokens, then padding to 20
        labels = sample["labels"]
        kept = (labels != -100).nonzero().flatten().tolist()
        assert kept == [8, 9, 10, 11]
        assert torch.equal(labels[kept], sample["input_ids"][kept])
        assert "prompt_ids" not in sample

    def test_generation_samples_carry_prompt_tensors(self) -> None:
        sample = self.__dataset(generation=True)[0]
        assert sample["prompt_ids"].shape[-1] == 8
        assert sample["prompt_mask"].sum() == 8
