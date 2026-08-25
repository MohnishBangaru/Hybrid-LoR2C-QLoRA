"""Interfaces the application layer depends on; infrastructure implements them."""

from contextlib import AbstractContextManager
from pathlib import Path
from typing import Protocol

from torch import nn

from lor2c.application.schema import (
    BenchmarkReport,
    Bundle,
    EpochReport,
    EvaluationReport,
    Loaders,
    Metrics,
    ResidualManifest,
    Split,
    TrainingReport,
)
from lor2c.domain.bank import AdapterBank
from lor2c.domain.schedule import ResidualRouter
from lor2c.domain.schema import AdapterSpec
from lor2c.settings.schema import (
    AdapterSettings,
    BenchmarkSettings,
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

    def restore(self, *, bundle: Bundle[nn.Module], path: Path) -> Bundle[nn.Module]:
        """Attach a previously saved attention adapter from `path`."""
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


class Observer(Protocol):
    """Receives training progress so structural adaptations can be scheduled."""

    def observe(self, *, epoch: float) -> None:
        """Called after every optimisation step with fractional epoch progress."""
        ...


class CausalTrainerPort(Protocol):
    """Runs the optimisation loop for a causal model."""

    def fit(
        self,
        *,
        bundle: Bundle[nn.Module],
        split: Split,
        settings: CausalSettings,
        observer: Observer | None = None,
    ) -> TrainingReport:
        """Train and return a summary."""
        ...


class AttentionGate(Protocol):
    """Controls which attention LoRA parameters are trainable, per decoder layer."""

    def freeze(self, *, model: nn.Module) -> None:
        """Freeze every attention LoRA parameter."""
        ...

    def release(self, *, model: nn.Module, layer: int) -> None:
        """Make the attention LoRA parameters of `layer` trainable."""
        ...


class Repository(Protocol):
    """Persists and restores trained adapters."""

    def save(
        self,
        *,
        model: nn.Module,
        bank: AdapterBank | None,
        output: Path,
        manifest: ResidualManifest | None = None,
    ) -> None:
        """Write adapter weights and configuration to `output`."""
        ...

    def manifest(self, *, output: Path) -> ResidualManifest | None:
        """Read the residual manifest saved under `output`, if any."""
        ...

    def restore(self, *, output: Path, manifest: ResidualManifest) -> AdapterBank:
        """Rebuild the residual bank described by `manifest` with its saved weights."""
        ...


class Benchmark(Protocol):
    """Scores a model on downstream tasks."""

    def run(self, *, model: nn.Module, settings: BenchmarkSettings, seed: int) -> BenchmarkReport:
        """Evaluate and return per-task scores."""
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
