"""Image-caption dataset and loaders for vision-language fine-tuning."""

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from lor2c.application.schema import Loaders
from lor2c.domain.caption import CaptionNormalizer
from lor2c.domain.constants import Label
from lor2c.domain.exceptions import ConfigurationError
from lor2c.infrastructure.huggingface.context import HubContext
from lor2c.settings.schema import VisionDataSettings


class CaptionDataset(Dataset[dict[str, Tensor]]):
    """Builds chat-formatted, label-masked samples from (image, text) rows."""

    IMAGE = "image"
    TEXT = "text"
    PROMPT_IDS = "prompt_ids"
    PROMPT_MASK = "prompt_mask"

    def __init__(
        self,
        *,
        rows: object,
        processor: object,
        settings: VisionDataSettings,
        normalizer: CaptionNormalizer,
        generation: bool = False,
    ) -> None:
        self.__rows = rows
        self.__processor = processor
        self.__settings = settings
        self.__normalizer = normalizer
        self.__generation = generation

    def __len__(self) -> int:
        return len(self.__rows)  # type: ignore[arg-type]

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        row = self.__rows[index]  # type: ignore[index]
        image = self.__resize(image=row[self.IMAGE].convert("RGB"))
        caption = self.__normalizer.normalize(text=row[self.TEXT])
        messages: list[dict[str, object]] = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": self.__settings.instruction},
                ],
            },
            {"role": "assistant", "content": caption},
        ]
        processor = self.__processor
        prompt = processor.apply_chat_template(  # type: ignore[attr-defined]
            messages, add_generation_prompt=False, tokenize=False
        )
        encoded = processor(  # type: ignore[operator]
            text=prompt,
            images=image,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.__settings.length,
        )
        sample = {key: value.squeeze(0) for key, value in encoded.items()}
        prompt_sample = self.__prompt(messages=messages[:1], image=image)
        sample["labels"] = self.__labels(
            ids=sample["input_ids"],
            mask=sample["attention_mask"],
            prefix=int(prompt_sample[self.PROMPT_IDS].shape[-1]),
        )
        if self.__generation:
            sample.update(prompt_sample)
        return sample

    def __prompt(self, *, messages: list[dict[str, object]], image: object) -> dict[str, Tensor]:
        """Tokenise the user turn alone with the assistant header appended, for generation."""
        processor = self.__processor
        prompt = processor.apply_chat_template(  # type: ignore[attr-defined]
            messages, add_generation_prompt=True, tokenize=False
        )
        encoded = processor(text=prompt, images=image, return_tensors="pt")  # type: ignore[operator]
        return {
            self.PROMPT_IDS: encoded["input_ids"].squeeze(0),
            self.PROMPT_MASK: encoded["attention_mask"].squeeze(0),
        }

    @staticmethod
    def __labels(*, ids: Tensor, mask: Tensor, prefix: int) -> Tensor:
        """Ignore padding and the first `prefix` real tokens (the prompt); keep the caption."""
        labels = ids.clone()
        labels[mask == 0] = int(Label.IGNORED)
        positions = torch.nonzero(mask, as_tuple=False).flatten()
        labels[positions[:prefix]] = int(Label.IGNORED)
        return labels

    def __resize(self, *, image: object) -> object:
        longest = max(image.width, image.height)  # type: ignore[attr-defined]
        if longest <= self.__settings.image:
            return image
        ratio = self.__settings.image / longest
        from PIL import Image

        return image.resize(  # type: ignore[attr-defined]
            (int(image.width * ratio), int(image.height * ratio)),  # type: ignore[attr-defined]
            Image.Resampling.LANCZOS,
        )


class HubVisionDataPort:
    """Loads the head of a hub dataset for training and its tail for validation."""

    def __init__(self, *, context: HubContext, normalizer: CaptionNormalizer, seed: int) -> None:
        self.__context = context
        self.__normalizer = normalizer
        self.__seed = seed

    def load(self, *, settings: VisionDataSettings, batch: int) -> Loaders:
        """Return train/validation loaders; validation uses batch size one for generation."""
        try:
            from datasets import load_dataset
        except ImportError as exception:
            raise ConfigurationError(
                f"Install lor2c[huggingface] to load datasets ({exception})."
            ) from exception
        processor = self.__context.processor(image=settings.image)
        holdout = max(1, int(settings.samples * settings.holdout))
        train_rows = load_dataset(settings.path, split=f"train[:{settings.samples}]")
        validation_rows = load_dataset(settings.path, split=f"train[-{holdout}:]")
        train = CaptionDataset(
            rows=train_rows, processor=processor, settings=settings, normalizer=self.__normalizer
        )
        validation = CaptionDataset(
            rows=validation_rows,
            processor=processor,
            settings=settings,
            normalizer=self.__normalizer,
            generation=True,
        )
        generator = torch.Generator().manual_seed(self.__seed)
        return Loaders(
            train=DataLoader(train, batch_size=batch, shuffle=True, generator=generator),
            validation=DataLoader(validation, batch_size=1, shuffle=False),
        )
