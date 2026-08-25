"""Persists trained adapters to the local file system."""

import json
import logging
from pathlib import Path
from typing import Protocol, runtime_checkable

import torch
from torch import nn

from lor2c.domain.bank import AdapterBank

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

    def save(self, *, model: nn.Module, bank: AdapterBank | None, output: Path) -> None:
        """Persist `model` (trainable parameters) and, if present, the residual `bank`."""
        output.mkdir(parents=True, exist_ok=True)
        self.__save_model(model=model, directory=output / self.MODEL_DIRECTORY)
        if bank is not None:
            self.__save_bank(bank=bank, directory=output / self.RESIDUAL_DIRECTORY)
        LOGGER.info("Saved adapters", extra={"ctx_output": str(output)})

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

    def __save_bank(self, *, bank: AdapterBank, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(bank.state_dict(), directory / self.WEIGHTS_FILE)
        manifest = {"width": bank.width, "names": list(bank.names)}
        (directory / self.MANIFEST_FILE).write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
