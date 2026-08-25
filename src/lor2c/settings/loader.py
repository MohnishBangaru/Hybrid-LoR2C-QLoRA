"""Reads YAML files into validated settings models."""

from pathlib import Path
from typing import TypeVar

import yaml
from pydantic import BaseModel, ValidationError

from lor2c.domain.exceptions import ConfigurationError

SettingsType = TypeVar("SettingsType", bound=BaseModel)


class SettingsLoader:
    """Parses a YAML document and validates it against a settings model."""

    def load(self, *, path: Path, schema: type[SettingsType]) -> SettingsType:
        """Return a validated `schema` instance built from the YAML file at `path`."""
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exception:
            raise ConfigurationError(f"Settings file not found: {path}.") from exception
        except yaml.YAMLError as exception:
            raise ConfigurationError(
                f"Settings file {path} is not valid YAML: {exception}"
            ) from exception
        if not isinstance(raw, dict):
            raise ConfigurationError(
                f"Settings file {path} must contain a mapping at the top level."
            )
        try:
            return schema.model_validate(raw)
        except ValidationError as exception:
            raise ConfigurationError(
                f"Settings file {path} is invalid:\n{exception}"
            ) from exception
