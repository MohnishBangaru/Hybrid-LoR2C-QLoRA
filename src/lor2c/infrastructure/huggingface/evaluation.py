"""Caption generation and BLEU scoring."""

import logging

import torch
from torch import nn
from torch.amp.autocast_mode import autocast

from lor2c.application.schema import Bundle, EvaluationReport, Loaders
from lor2c.domain.constants import Label
from lor2c.domain.exceptions import ConfigurationError
from lor2c.infrastructure.huggingface.context import HubContext
from lor2c.settings.schema import EvaluationSettings

LOGGER = logging.getLogger(__name__)


class BleuEvaluator:
    """Generates a caption per validation sample and computes smoothed corpus BLEU."""

    GENERATION_KEYS = ("input_ids", "attention_mask", "pixel_values")

    def __init__(self, *, context: HubContext, image: int) -> None:
        self.__context = context
        self.__image = image

    def evaluate(
        self, *, bundle: Bundle[nn.Module], loaders: Loaders, settings: EvaluationSettings
    ) -> EvaluationReport:
        """Score generated captions against the references in `loaders.validation`."""
        try:
            from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
        except ImportError as exception:
            raise ConfigurationError("Install lor2c[vision] to evaluate BLEU.") from exception
        tokenizer = self.__context.processor(image=self.__image).tokenizer  # type: ignore[attr-defined]
        model = bundle.model
        device = next(model.parameters()).device
        model.eval()
        references: list[list[list[str]]] = []
        hypotheses: list[list[str]] = []
        with torch.no_grad():
            for batch in loaders.validation:
                labels = batch["labels"][0]
                reference = tokenizer.decode(
                    labels[labels != int(Label.IGNORED)], skip_special_tokens=True
                )
                inputs = {
                    key: batch[key].to(device) for key in self.GENERATION_KEYS if key in batch
                }
                with autocast(
                    device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
                ):
                    generated = model.generate(
                        **inputs, max_new_tokens=settings.tokens, num_beams=settings.beams
                    )
                hypothesis = tokenizer.decode(generated[0], skip_special_tokens=True).strip()
                references.append([reference.split()])
                hypotheses.append(hypothesis.split())
        score = float(
            corpus_bleu(references, hypotheses, smoothing_function=SmoothingFunction().method4)
        )
        LOGGER.info("Evaluated", extra={"ctx_bleu": score, "ctx_samples": len(hypotheses)})
        return EvaluationReport(bleu=score, samples=len(hypotheses))
