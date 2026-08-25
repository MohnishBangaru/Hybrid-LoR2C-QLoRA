"""Causal language model loading and attention LoRA via `peft`."""

import logging
from pathlib import Path

from torch import nn

from lor2c.application.schema import Bundle
from lor2c.domain.constants import BaseQuantization
from lor2c.domain.exceptions import ConfigurationError
from lor2c.domain.schema import AdapterSpec
from lor2c.infrastructure.huggingface.compat import TransformersCompatibility
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
        dtype = self.__precision.dtype(precision=settings.precision)
        options: dict[str, object] = {
            "revision": settings.revision,
            "device_map": settings.device,
            "quantization_config": self.__quantization(settings=settings, dtype=dtype),
            TransformersCompatibility().dtype_keyword(): dtype,
        }
        model = AutoModelForCausalLM.from_pretrained(settings.name, **options)
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
            use_rslora=spec.mode.value == "stabilized",
            use_dora=settings.dora,
        )
        prepared = prepare_model_for_kbit_training(  # type: ignore[no-untyped-call]
            bundle.model, use_gradient_checkpointing=False
        )
        adapted = get_peft_model(prepared, configuration)
        adapted.print_trainable_parameters()
        LOGGER.info("Applied LoRA", extra={"ctx_rank": spec.rank, "ctx_targets": settings.targets})
        return Bundle(model=adapted, hidden=bundle.hidden, depth=bundle.depth)

    def restore(self, *, bundle: Bundle[nn.Module], path: Path) -> Bundle[nn.Module]:
        """Wrap the base model with the peft adapter saved at `path`."""
        try:
            from peft import PeftModel
        except ImportError as exception:
            raise ConfigurationError("Install lor2c[huggingface] to load adapters.") from exception
        if not (path / "adapter_config.json").exists():
            raise ConfigurationError(f"No peft adapter found at {path}.")
        restored = PeftModel.from_pretrained(bundle.model, str(path))
        return Bundle(model=restored, hidden=bundle.hidden, depth=bundle.depth)

    @staticmethod
    def __quantization(*, settings: ModelSettings, dtype: object) -> object | None:
        if settings.quantization is BaseQuantization.NONE:
            return None
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as exception:
            raise ConfigurationError("Install lor2c[huggingface] to quantize.") from exception
        if settings.quantization is BaseQuantization.INT8:
            eight: object = BitsAndBytesConfig(load_in_8bit=True)
            return eight
        four: object = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
        return four
