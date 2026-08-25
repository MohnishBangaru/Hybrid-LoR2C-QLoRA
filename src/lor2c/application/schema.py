"""Typed records exchanged between the application layer and its ports."""

from pathlib import Path
from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field
from torch import nn
from torch.utils.data import DataLoader, Dataset

from lor2c.domain.schema import AdapterSpec, ResidualSchedule

ModelType = TypeVar("ModelType", bound=nn.Module)


class Bundle(BaseModel, Generic[ModelType]):
    """A loaded model plus the geometry the application needs to plan adapters."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: ModelType = Field(description="The host network.")
    hidden: int = Field(gt=0, description="Residual stream width.")
    depth: int = Field(gt=0, description="Number of decoder layers.")


class Split(BaseModel):
    """Train/validation dataset pair."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    train: Dataset[dict[str, object]] = Field(description="Training examples.")
    validation: Dataset[dict[str, object]] | None = Field(default=None, description="Held-out set.")


class Loaders(BaseModel):
    """Train/validation data loader pair."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    train: DataLoader[dict[str, object]] = Field(description="Training batches.")
    validation: DataLoader[dict[str, object]] = Field(description="Validation batches.")


class Metrics(BaseModel):
    """A set of scalar metrics logged at one step."""

    model_config = ConfigDict(frozen=True)

    step: int = Field(ge=0, description="Global step the metrics belong to.")
    values: dict[str, float] = Field(description="Metric name to value.")


class EpochReport(BaseModel):
    """Outcome of one training epoch."""

    model_config = ConfigDict(frozen=True)

    epoch: int = Field(ge=1, description="1-based epoch index.")
    loss: float = Field(description="Mean training loss.")
    seconds: float = Field(ge=0, description="Wall-clock duration.")
    memory: int = Field(ge=0, description="Peak device memory in bytes.")
    step: int = Field(ge=0, description="Global step after the epoch.")


class EvaluationReport(BaseModel):
    """Outcome of one validation pass."""

    model_config = ConfigDict(frozen=True)

    bleu: float = Field(ge=0, le=1, description="Smoothed corpus BLEU.")
    samples: int = Field(ge=0, description="Number of scored samples.")


class TrainingReport(BaseModel):
    """Summary of a completed training loop."""

    model_config = ConfigDict(frozen=True)

    steps: int = Field(ge=0, description="Total optimisation steps.")
    loss: float | None = Field(default=None, description="Final training loss.")


class Outcome(BaseModel):
    """What a training use-case hands back to its caller."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(description="Run identifier.")
    output: Path = Field(description="Where artefacts were written.")
    steps: int = Field(ge=0, description="Total optimisation steps.")
    best: float | None = Field(default=None, description="Best validation score, if evaluated.")
    epoch: int | None = Field(default=None, description="Epoch of the best score, if evaluated.")


class ResidualManifest(BaseModel):
    """Everything needed to rebuild a saved residual adapter bank and its routing."""

    model_config = ConfigDict(frozen=True)

    width: int = Field(gt=0, description="Residual stream width.")
    shared: AdapterSpec | None = Field(
        default=None, description="Spec of the shared down projection."
    )
    specs: dict[str, AdapterSpec] = Field(description="Adapter name to its specification.")
    schedule: ResidualSchedule = Field(description="Routing plan at the end of training.")
    quantized: bool = Field(default=False, description="Whether adapters were converted to int8.")


class BenchmarkReport(BaseModel):
    """Scores returned by a benchmark harness."""

    model_config = ConfigDict(frozen=True)

    scores: dict[str, float] = Field(description="Metric name (task/metric) to value.")
    samples: int = Field(ge=0, description="Total evaluated samples across tasks.")


class EvaluationOutcome(BaseModel):
    """What the evaluation use-case hands back to its caller."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(description="Run identifier.")
    output: Path = Field(description="Where the scores were written.")
    scores: dict[str, float] = Field(description="Benchmark scores.")
    trainable: int = Field(ge=0, description="Trainable parameters of the loaded adapters.")
