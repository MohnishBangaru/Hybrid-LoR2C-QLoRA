"""Causal training loop delegated to `transformers.Trainer`."""

import logging
import sys

import torch
from torch import nn

from lor2c.application.schema import Bundle, Split, TrainingReport
from lor2c.domain.exceptions import ConfigurationError
from lor2c.infrastructure.huggingface.context import HubContext
from lor2c.settings.schema import CausalSettings

LOGGER = logging.getLogger(__name__)


class HubCausalTrainerPort:
    """Configures and runs `transformers.Trainer` for a bundle and split."""

    PAD_MULTIPLE = 8
    OPTIMIZER = "adamw_torch"

    def __init__(self, *, context: HubContext) -> None:
        self.__context = context

    def fit(
        self, *, bundle: Bundle[nn.Module], split: Split, settings: CausalSettings
    ) -> TrainingReport:
        """Train and return the number of optimisation steps performed."""
        try:
            from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments
        except ImportError as exception:
            raise ConfigurationError("Install lor2c[huggingface] to train.") from exception
        validating = split.validation is not None
        arguments = TrainingArguments(
            output_dir=str(settings.output / "checkpoints"),
            per_device_train_batch_size=settings.train.micro,
            gradient_accumulation_steps=settings.accumulation,
            warmup_steps=settings.train.warmup,
            num_train_epochs=settings.train.epochs,
            learning_rate=settings.train.rate,
            logging_steps=settings.train.logging,
            optim=self.OPTIMIZER,
            eval_strategy="steps" if validating else "no",
            eval_steps=settings.train.evaluation if validating else None,
            save_strategy="steps",
            save_steps=settings.train.evaluation,
            save_total_limit=settings.train.checkpoints,
            load_best_model_at_end=validating,
            group_by_length=settings.train.grouping,
            report_to="none",
            seed=settings.seed,
        )
        model = bundle.model
        if settings.train.compile and sys.platform != "win32":
            model = torch.compile(model)  # type: ignore[assignment]
        trainer = Trainer(
            model=model,
            args=arguments,
            train_dataset=split.train,
            eval_dataset=split.validation,
            data_collator=DataCollatorForSeq2Seq(
                self.__context.tokenizer(), pad_to_multiple_of=self.PAD_MULTIPLE, padding=True
            ),
        )
        resume = str(settings.resume) if settings.resume else None
        result = trainer.train(resume_from_checkpoint=resume)
        LOGGER.info("Training finished", extra={"ctx_steps": trainer.state.global_step})
        return TrainingReport(steps=trainer.state.global_step, loss=result.training_loss)
