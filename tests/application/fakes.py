"""In-memory port implementations used to exercise the application services."""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

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
from tests.conftest import TinyDecoder


class FakeTracker:
    def __init__(self) -> None:
        self.events: list[str] = []
        self.metrics: list[Metrics] = []

    def start(self, *, name: str, parameters: dict[str, object]) -> None:
        self.events.append(f"start:{name}")

    def log(self, *, metrics: Metrics) -> None:
        self.metrics.append(metrics)

    def finish(self) -> None:
        self.events.append("finish")


class FakeSeeder:
    def __init__(self) -> None:
        self.values: list[int] = []

    def seed(self, *, value: int) -> None:
        self.values.append(value)


class FakeModelPort:
    def __init__(self, *, width: int, depth: int) -> None:
        self.width = width
        self.depth = depth
        self.specs: list[AdapterSpec] = []

    def load(self, *, settings: ModelSettings) -> Bundle[nn.Module]:
        return Bundle(
            model=TinyDecoder(width=self.width, depth=self.depth),
            hidden=self.width,
            depth=self.depth,
        )

    def adapt(
        self, *, bundle: Bundle[nn.Module], spec: AdapterSpec, settings: AdapterSettings
    ) -> Bundle[nn.Module]:
        self.specs.append(spec)
        return bundle

    def restore(self, *, bundle: Bundle[nn.Module], path: Path) -> Bundle[nn.Module]:
        self.restored_from = path
        return bundle


class FakeVisionModelPort(FakeModelPort):
    def adapt(
        self, *, bundle: Bundle[nn.Module], spec: AdapterSpec, settings: AdapterSettings
    ) -> int:  # type: ignore[override]
        self.specs.append(spec)
        return 2


class FakeDataPort:
    def load(self, *, settings: CausalDataSettings, seed: int) -> Split:
        return Split(train=TensorDataset(torch.zeros(4, 2)), validation=None)


class FakeVisionDataPort:
    def load(self, *, settings: VisionDataSettings, batch: int) -> Loaders:
        loader = DataLoader(TensorDataset(torch.zeros(2, 2)), batch_size=batch)
        return Loaders(train=loader, validation=loader)


class FakeTrainer:
    def __init__(self) -> None:
        self.attached_during_fit: bool | None = None
        self.router_probe: RecordingRouter | None = None
        self.observer: object | None = None

    def fit(
        self,
        *,
        bundle: Bundle[nn.Module],
        split: Split,
        settings: CausalSettings,
        observer: object | None = None,
    ) -> TrainingReport:
        if self.router_probe is not None:
            self.attached_during_fit = self.router_probe.attached
        self.observer = observer
        return TrainingReport(steps=7, loss=1.5)


class RecordingGate:
    def __init__(self) -> None:
        self.frozen = 0
        self.released: list[int] = []

    def freeze(self, *, model: nn.Module) -> None:
        self.frozen += 1

    def release(self, *, model: nn.Module, layer: int) -> None:
        self.released.append(layer)


class FakeVisionTrainer:
    def __init__(self) -> None:
        self.epochs: list[int] = []

    def epoch(self, *, bundle: Bundle[nn.Module], loaders: Loaders, index: int) -> EpochReport:
        self.epochs.append(index)
        return EpochReport(epoch=index, loss=1.0 / index, seconds=0.1, memory=0, step=index * 10)


class ScriptedEvaluator:
    def __init__(self, *, scores: list[float]) -> None:
        self.scores = list(scores)

    def evaluate(
        self, *, bundle: Bundle[nn.Module], loaders: Loaders, settings: EvaluationSettings
    ) -> EvaluationReport:
        return EvaluationReport(bleu=self.scores.pop(0), samples=1)


class RecordingRouter:
    def __init__(self) -> None:
        self.attached = False
        self.routers: list[ResidualRouter] = []

    @contextmanager
    def attach(self, *, model: nn.Module, router: ResidualRouter) -> Iterator[None]:
        self.routers.append(router)
        self.attached = True
        try:
            yield
        finally:
            self.attached = False


class RecordingQuantizer:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def prepare(self, *, bank: AdapterBank) -> None:
        self.calls.append("prepare")

    def convert(self, *, bank: AdapterBank) -> None:
        self.calls.append("convert")


class RecordingRepository:
    def __init__(self) -> None:
        self.banks: list[AdapterBank | None] = []
        self.outputs: list[Path] = []
        self.manifests: list[ResidualManifest | None] = []
        self.stored: ResidualManifest | None = None
        self.restored: AdapterBank | None = None

    def save(
        self,
        *,
        model: nn.Module,
        bank: AdapterBank | None,
        output: Path,
        manifest: ResidualManifest | None = None,
    ) -> None:
        self.banks.append(bank)
        self.outputs.append(output)
        self.manifests.append(manifest)

    def manifest(self, *, output: Path) -> ResidualManifest | None:
        return self.stored

    def restore(self, *, output: Path, manifest: ResidualManifest) -> AdapterBank:
        bank = AdapterBank(width=manifest.width, shared=manifest.shared)
        for name, spec in manifest.specs.items():
            bank.register(name=name, spec=spec)
        self.restored = bank
        return bank


class FakeBenchmark:
    def __init__(self) -> None:
        self.attached: bool | None = None
        self.probe: RecordingRouter | None = None

    def run(self, *, model: nn.Module, settings: BenchmarkSettings, seed: int) -> BenchmarkReport:
        self.attached = self.probe.attached if self.probe is not None else None
        return BenchmarkReport(scores={"arc_easy/acc_norm": 0.5}, samples=10)
