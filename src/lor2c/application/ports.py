"""Interfaces the application layer depends on; infrastructure implements them."""

from contextlib import AbstractContextManager
from pathlib import Path
from typing import Protocol

from torch import nn

from lor2c.application.schema import (
    Bundle,
    EpochReport,
    EvaluationReport,
    Loaders,
    Metrics,
    Split,
    TrainingReport,
)
from lor2c.domain.bank import AdapterBank
from lor2c.domain.schedule import ResidualRouter
from lor2c.domain.schema import AdapterSpec
from lor2c.settings.schema import (
    AdapterSettings,
    CausalDataSettings,
    CausalSettings,
    EvaluationSettings,
    ModelSettings,
    VisionDataSettings,
)


class Tracker(Protocol):
    """Experiment tracking sink."""

    def start(self, *, name: str, parameters: dict[str, object]) -> None:
        """Open a run."""
        ...

    def log(self, *, metrics: Metrics) -> None:
        """Record metrics for a step."""
        ...

    def finish(self) -> None:
        """Close the run."""
        ...


class CausalModelPort(Protocol):
    """Loads a causal language model and applies attention-level LoRA."""

    def load(self, *, settings: ModelSettings) -> Bundle[nn.Module]:
        """Load the frozen base model."""
        ...

    def adapt(
        self, *, bundle: Bundle[nn.Module], spec: AdapterSpec, settings: AdapterSettings
    ) -> Bundle[nn.Module]:
        """Attach trainable LoRA to the configured target modules."""
        ...


class Router(Protocol):
    """Attaches a `ResidualRouter` to a model's decoder layers."""

    def attach(self, *, model: nn.Module, router: ResidualRouter) -> AbstractContextManager[None]:
        """Register hooks; leaving the context removes them."""
        ...


class Quantizer(Protocol):
    """Quantization-aware training of adapter projections."""

    def prepare(self, *, bank: AdapterBank) -> None:
        """Replace projections with fake-quantized modules in place."""
        ...

    def convert(self, *, bank: AdapterBank) -> None:
        """Replace fake-quantized modules with integer kernels in place."""
        ...


class CausalDataPort(Protocol):
    """Loads and tokenises an instruction dataset."""

    def load(self, *, settings: CausalDataSettings, seed: int) -> Split:
        """Return tokenised train/validation splits."""
        ...


class CausalTrainerPort(Protocol):
    """Runs the optimisation loop for a causal model."""

    def fit(
        self, *, bundle: Bundle[nn.Module], split: Split, settings: CausalSettings
    ) -> TrainingReport:
        """Train and return a summary."""
        ...


class Repository(Protocol):
    """Persists trained adapters."""

    def save(self, *, model: nn.Module, bank: AdapterBank | None, output: Path) -> None:
        """Write adapter weights and configuration to `output`."""
        ...


class VisionModelPort(Protocol):
    """Loads a vision-language model and injects adapters into target projections."""

    def load(self, *, settings: ModelSettings) -> Bundle[nn.Module]:
        """Load the frozen base model."""
        ...

    def adapt(
        self, *, bundle: Bundle[nn.Module], spec: AdapterSpec, settings: AdapterSettings
    ) -> int:
        """Inject adapters; return the number of injected modules."""
        ...


class VisionDataPort(Protocol):
    """Builds captioning data loaders."""

    def load(self, *, settings: VisionDataSettings, batch: int) -> Loaders:
        """Return train/validation loaders."""
        ...


class VisionTrainerPort(Protocol):
    """Runs one epoch of the vision training loop."""

    def epoch(self, *, bundle: Bundle[nn.Module], loaders: Loaders, index: int) -> EpochReport:
        """Train for one epoch and return its report."""
        ...


class Evaluator(Protocol):
    """Scores generated captions against references."""

    def evaluate(
        self, *, bundle: Bundle[nn.Module], loaders: Loaders, settings: EvaluationSettings
    ) -> EvaluationReport:
        """Generate captions for the validation loader and score them."""
        ...


class Seeder(Protocol):
    """Makes the run deterministic."""

    def seed(self, *, value: int) -> None:
        """Seed every random source in use."""
        ...
