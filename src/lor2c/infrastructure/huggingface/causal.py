"""Causal language model loading and attention LoRA via `peft`."""

import logging

from torch import nn

from lor2c.application.schema import Bundle
from lor2c.domain.exceptions import ConfigurationError
from lor2c.domain.schema import AdapterSpec
from lor2c.infrastructure.huggingface.context import HubContext
from lor2c.infrastructure.huggingface.precision import PrecisionMapper
from lor2c.settings.schema import AdapterSettings, ModelSettings

LOGGER = logging.getLogger(__name__)


class HubCausalModelPort:
    """Loads `AutoModelForCausalLM` and wraps attention projections with peft LoRA."""

    def __init__(self, *, context: HubContext, precision: PrecisionMapper) -> None:
        self.__context = context
        self.__precision = precision

    def load(self, *, settings: ModelSettings) -> Bundle[nn.Module]:
        """Load the base model in the configured precision and freeze it."""
        try:
            from transformers import AutoModelForCausalLM
        except ImportError as exception:
            raise ConfigurationError("Install lor2c[huggingface] to load models.") from exception
        model = AutoModelForCausalLM.from_pretrained(
            settings.name,
            revision=settings.revision,
            torch_dtype=self.__precision.dtype(precision=settings.precision),
            device_map=settings.device,
        )
        model.config.use_cache = False
        self.__context.tokenizer()
        return Bundle(
            model=model,
            hidden=int(model.config.hidden_size),
            depth=int(model.config.num_hidden_layers),
        )

    def adapt(
        self, *, bundle: Bundle[nn.Module], spec: AdapterSpec, settings: AdapterSettings
    ) -> Bundle[nn.Module]:
        """Return a new bundle whose model carries LoRA on `settings.targets`."""
        try:
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        except ImportError as exception:
            raise ConfigurationError("Install lor2c[huggingface] to apply LoRA.") from exception
        configuration = LoraConfig(
            r=spec.rank,
            lora_alpha=int(spec.alpha),
            lora_dropout=spec.dropout,
            target_modules=list(settings.targets),
            bias="none",
            task_type="CAUSAL_LM",
        )
        prepared = prepare_model_for_kbit_training(bundle.model)  # type: ignore[no-untyped-call]
        adapted = get_peft_model(prepared, configuration)
        adapted.print_trainable_parameters()
        LOGGER.info("Applied LoRA", extra={"ctx_rank": spec.rank, "ctx_targets": settings.targets})
        return Bundle(model=adapted, hidden=bundle.hidden, depth=bundle.depth)
