"""Vision-language model loading and adapter injection."""

import logging

import torch
from torch import nn

from lor2c.application.schema import Bundle
from lor2c.domain.exceptions import ConfigurationError
from lor2c.domain.schema import AdapterSpec
from lor2c.infrastructure.huggingface.compat import TransformersCompatibility
from lor2c.infrastructure.huggingface.context import HubContext
from lor2c.infrastructure.huggingface.precision import PrecisionMapper
from lor2c.infrastructure.injection import LinearInjector
from lor2c.settings.schema import AdapterSettings, ModelSettings

LOGGER = logging.getLogger(__name__)


class HubVisionModelPort:
    """Loads `AutoModelForVision2Seq` and injects low-rank adapters into target projections."""

    def __init__(
        self,
        *,
        context: HubContext,
        precision: PrecisionMapper,
        injector: LinearInjector,
        seed: int,
    ) -> None:
        self.__context = context
        self.__precision = precision
        self.__injector = injector
        self.__seed = seed

    def load(self, *, settings: ModelSettings) -> Bundle[nn.Module]:
        """Load and freeze the base model."""
        try:
            from transformers import AutoModelForVision2Seq
        except ImportError as exception:
            raise ConfigurationError("Install lor2c[huggingface] to load models.") from exception
        options: dict[str, object] = {
            "revision": settings.revision,
            "device_map": settings.device,
            TransformersCompatibility().dtype_keyword(): self.__precision.dtype(
                precision=settings.precision
            ),
        }
        model = AutoModelForVision2Seq.from_pretrained(settings.name, **options)
        model.requires_grad_(False)
        text = getattr(model.config, "text_config", model.config)
        return Bundle(
            model=model,
            hidden=int(text.hidden_size),
            depth=int(text.num_hidden_layers),
        )

    def adapt(
        self, *, bundle: Bundle[nn.Module], spec: AdapterSpec, settings: AdapterSettings
    ) -> int:
        """Inject adapters into every linear layer matching `settings.targets`."""
        generator = torch.Generator().manual_seed(self.__seed)
        count = self.__injector.inject(
            model=bundle.model, targets=settings.targets, spec=spec, generator=generator
        )
        trainable = sum(p.numel() for p in bundle.model.parameters() if p.requires_grad)
        LOGGER.info("Injected adapters", extra={"ctx_count": count, "ctx_trainable": trainable})
        return count
