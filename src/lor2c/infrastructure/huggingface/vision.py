"""Vision-language model loading and adapter injection."""

import logging
from typing import Protocol, cast

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


class Geometry(Protocol):
    """Configuration fields describing the language decoder."""

    hidden_size: int
    num_hidden_layers: int


class Loadable(Protocol):
    """Auto classes exposing `from_pretrained`."""

    def from_pretrained(self, name: str, **options: object) -> nn.Module:
        """Instantiate a model from the hub or a local path."""
        ...


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
        loader = self.__loader()
        options: dict[str, object] = {
            "revision": settings.revision,
            "device_map": settings.device,
            TransformersCompatibility().dtype_keyword(): self.__precision.dtype(
                precision=settings.precision
            ),
        }
        model = loader.from_pretrained(settings.name, **options)
        model.requires_grad_(False)
        configuration: object = model.config
        text = cast(Geometry, getattr(configuration, "text_config", configuration))
        return Bundle(model=model, hidden=int(text.hidden_size), depth=int(text.num_hidden_layers))

    @staticmethod
    def __loader() -> Loadable:
        """Image-text-to-text auto class on new transformers, Vision2Seq on older releases."""
        try:
            import transformers
        except ImportError as exception:
            raise ConfigurationError(
                f"Install lor2c[huggingface] to load models ({exception})."
            ) from exception
        for name in ("AutoModelForImageTextToText", "AutoModelForVision2Seq"):
            loader = getattr(transformers, name, None)
            if loader is not None:
                return cast(Loadable, loader)
        raise ConfigurationError("Installed transformers has no image-text-to-text auto class.")

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
