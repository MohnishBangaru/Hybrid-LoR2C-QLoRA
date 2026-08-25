"""Composition root wiring infrastructure adapters into application services."""

from lor2c.application.causal import CausalTrainingService
from lor2c.application.ports import Quantizer, Tracker
from lor2c.application.vision import VisionTrainingService
from lor2c.domain.bank import AdapterBank
from lor2c.domain.caption import CaptionNormalizer
from lor2c.domain.constants import TrackerKind
from lor2c.domain.masking import LabelMasker
from lor2c.domain.prompt import PromptTemplate
from lor2c.domain.rank import FixedRankPolicy, RankPolicy, WidthRankPolicy
from lor2c.domain.schema import AdapterSpec
from lor2c.infrastructure.hooks import HookRouter, ModuleListLocator
from lor2c.infrastructure.huggingface.captions import HubVisionDataPort
from lor2c.infrastructure.huggingface.causal import HubCausalModelPort
from lor2c.infrastructure.huggingface.context import HubContext
from lor2c.infrastructure.huggingface.dataset import HubCausalDataPort
from lor2c.infrastructure.huggingface.evaluation import BleuEvaluator
from lor2c.infrastructure.huggingface.loop import TorchVisionTrainerPort
from lor2c.infrastructure.huggingface.precision import PrecisionMapper
from lor2c.infrastructure.huggingface.trainer import HubCausalTrainerPort
from lor2c.infrastructure.huggingface.vision import HubVisionModelPort
from lor2c.infrastructure.injection import LinearInjector
from lor2c.infrastructure.quantization import FakeQuantizer
from lor2c.infrastructure.repository import DiskRepository
from lor2c.infrastructure.seeding import TorchSeeder
from lor2c.infrastructure.tracking.noop import LogTracker
from lor2c.infrastructure.tracking.wandb import WandbTracker
from lor2c.settings.schema import AdapterSettings, CausalSettings, TrackingSettings, VisionSettings


class ServiceFactory:
    """Builds fully wired training services from validated settings."""

    def causal(self, *, settings: CausalSettings) -> CausalTrainingService:
        """Assemble the causal language model fine-tuning service."""
        context = HubContext(settings=settings.model)
        precision = PrecisionMapper()
        return CausalTrainingService(
            models=HubCausalModelPort(context=context, precision=precision),
            data=HubCausalDataPort(
                context=context, template=PromptTemplate.alpaca(), masker=LabelMasker()
            ),
            trainer=HubCausalTrainerPort(context=context),
            router=HookRouter(locator=ModuleListLocator()),
            quantizer=self.__quantizer(settings=settings),
            repository=DiskRepository(),
            tracker=self.__tracker(settings=settings.tracking),
            seeder=TorchSeeder(),
            policy=self.__policy(settings=settings.adapter),
        )

    def vision(self, *, settings: VisionSettings) -> VisionTrainingService:
        """Assemble the vision-language fine-tuning service."""
        context = HubContext(settings=settings.model)
        tracker = self.__tracker(settings=settings.tracking)
        return VisionTrainingService(
            models=HubVisionModelPort(
                context=context,
                precision=PrecisionMapper(),
                injector=LinearInjector(),
                seed=settings.seed,
            ),
            data=HubVisionDataPort(
                context=context, normalizer=CaptionNormalizer(), seed=settings.seed
            ),
            trainer=TorchVisionTrainerPort(
                settings=settings.train, epochs=settings.train.epochs, tracker=tracker
            ),
            evaluator=BleuEvaluator(context=context, image=settings.data.image),
            repository=DiskRepository(),
            tracker=tracker,
            seeder=TorchSeeder(),
            policy=self.__policy(settings=settings.adapter),
        )

    @staticmethod
    def __quantizer(*, settings: CausalSettings) -> Quantizer:
        if settings.quantization.enabled:
            return FakeQuantizer(backend=settings.quantization.backend)
        return _InactiveQuantizer()

    @staticmethod
    def __tracker(*, settings: TrackingSettings) -> Tracker:
        if settings.kind is TrackerKind.WANDB:
            return WandbTracker(project=settings.project, run=settings.run)
        return LogTracker()

    @staticmethod
    def __policy(*, settings: AdapterSettings) -> RankPolicy:
        if settings.automatic:
            return WidthRankPolicy(dropout=settings.dropout)
        spec = AdapterSpec(rank=settings.rank, alpha=settings.alpha, dropout=settings.dropout)
        return FixedRankPolicy(spec=spec)


class _InactiveQuantizer:
    """Quantizer used when quantization is disabled; every call is a no-op."""

    def prepare(self, *, bank: AdapterBank) -> None:
        """Do nothing."""

    def convert(self, *, bank: AdapterBank) -> None:
        """Do nothing."""
