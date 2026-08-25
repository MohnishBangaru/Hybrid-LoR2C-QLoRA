"""Persists and restores trained adapters on the local file system."""

import logging
from pathlib import Path
from typing import Protocol, runtime_checkable

import torch
from torch import nn

from lor2c.application.schema import ResidualManifest
from lor2c.domain.bank import AdapterBank
from lor2c.domain.exceptions import ConfigurationError

LOGGER = logging.getLogger(__name__)


@runtime_checkable
class Pretrained(Protocol):
    """Models able to serialise themselves in the Hugging Face layout."""

    def save_pretrained(self, save_directory: str) -> None:
        """Write weights and configuration to `save_directory`."""
        ...


class DiskRepository:
    """Writes attention adapters under `model/` and residual adapters under `residual/`."""

    MODEL_DIRECTORY = "model"
    RESIDUAL_DIRECTORY = "residual"
    WEIGHTS_FILE = "adapters.pt"
    MANIFEST_FILE = "manifest.json"

    def save(
        self,
        *,
        model: nn.Module,
        bank: AdapterBank | None,
        output: Path,
        manifest: ResidualManifest | None = None,
    ) -> None:
        """Persist `model` (trainable parameters) and, if present, the residual `bank`."""
        output.mkdir(parents=True, exist_ok=True)
        self.__save_model(model=model, directory=output / self.MODEL_DIRECTORY)
        if bank is not None:
            if manifest is None:
                raise ConfigurationError("A residual bank cannot be saved without its manifest.")
            self.__save_bank(
                bank=bank, manifest=manifest, directory=output / self.RESIDUAL_DIRECTORY
            )
        LOGGER.info("Saved adapters", extra={"ctx_output": str(output)})

    def manifest(self, *, output: Path) -> ResidualManifest | None:
        """Read the residual manifest, or `None` when the run has no residual adapters."""
        path = output / self.RESIDUAL_DIRECTORY / self.MANIFEST_FILE
        if not path.exists():
            return None
        return ResidualManifest.model_validate_json(path.read_text(encoding="utf-8"))

    def restore(self, *, output: Path, manifest: ResidualManifest) -> AdapterBank:
        """Rebuild the bank from `manifest` and load its saved weights."""
        bank = AdapterBank(width=manifest.width, shared=manifest.shared)
        for name, spec in manifest.specs.items():
            bank.register(name=name, spec=spec)
        weights = torch.load(
            output / self.RESIDUAL_DIRECTORY / self.WEIGHTS_FILE, map_location="cpu"
        )
        bank.load_state_dict(weights)
        bank.eval()
        return bank

    def __save_model(self, *, model: nn.Module, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        if isinstance(model, Pretrained):
            model.save_pretrained(str(directory))
            return
        trainable = {
            name: parameter.detach().cpu()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        torch.save(trainable, directory / self.WEIGHTS_FILE)

    def __save_bank(
        self, *, bank: AdapterBank, manifest: ResidualManifest, directory: Path
    ) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(bank.state_dict(), directory / self.WEIGHTS_FILE)
        (directory / self.MANIFEST_FILE).write_text(
            manifest.model_dump_json(indent=2), encoding="utf-8"
        )
