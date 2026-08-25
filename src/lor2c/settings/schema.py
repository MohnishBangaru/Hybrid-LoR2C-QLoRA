"""Pydantic settings models for causal and vision fine-tuning runs."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from lor2c.domain.constants import (
    AdapterMode,
    BaseQuantization,
    Precision,
    QuantizationBackend,
    Scaling,
    TrackerKind,
)


class Strict(BaseModel):
    """Base class: immutable and rejects unknown keys so typos fail at the boundary."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class ModelSettings(Strict):
    """Identity and loading options of the base model."""

    name: str = Field(min_length=1, description="Hub identifier or local path of the base model.")
    revision: str | None = Field(default=None, description="Optional git revision on the hub.")
    precision: Precision = Field(default=Precision.FLOAT16, description="Load/compute precision.")
    device: str = Field(default="auto", description="Device map passed to the loader.")
    quantization: BaseQuantization = Field(
        default=BaseQuantization.NONE, description="Frozen-weight quantization (QLoRA uses nf4)."
    )


class AdapterSettings(Strict):
    """Low-rank adapter hyper-parameters."""

    mode: AdapterMode = Field(default=AdapterMode.LOR2C, description="Base LoRA or LoR2C routing.")
    rank: int = Field(default=16, gt=0, description="Bottleneck rank.")
    alpha: float = Field(default=32, gt=0, description="Scaling numerator.")
    dropout: float = Field(default=0.05, ge=0, lt=1, description="Adapter input dropout.")
    targets: tuple[str, ...] = Field(
        default=("q_proj", "v_proj"), description="Linear module suffixes receiving LoRA."
    )
    automatic: bool = Field(
        default=True, description="Derive rank/alpha from model width instead of using rank/alpha."
    )
    scaling: Scaling = Field(
        default=Scaling.STANDARD, description="alpha/r or rsLoRA alpha/sqrt(r)."
    )
    shared: bool = Field(
        default=False, description="ShareLoR2C: one down projection for all layers."
    )
    dora: bool = Field(default=False, description="Use DoRA for the attention LoRA (peft).")
    ratio: float = Field(default=1.0, ge=1, description="LoRA+ learning-rate multiplier for up/B.")


class AdaptationSettings(Strict):
    """IMLoR2C schedule: how many residual adapters to merge or inject during training."""

    merges: int = Field(default=0, ge=0, description="MergeLoR2C: adjacent pairs to merge.")
    injections: int = Field(default=0, ge=0, description="InjectLoR2C: adapters to swap for LoRA.")
    top: int | None = Field(
        default=None, gt=0, description="k for SFS = 1 - top_k/total; defaults to rank // 2."
    )
    span: int = Field(default=4, ge=2, description="Maximum layers one merged adapter may cover.")

    @property
    def active(self) -> bool:
        """Whether any merge or injection is scheduled."""
        return self.merges > 0 or self.injections > 0


class TrackingSettings(Strict):
    """Experiment tracking configuration."""

    kind: TrackerKind = Field(default=TrackerKind.NONE, description="Tracking backend.")
    project: str = Field(default="LoR2C-QLoRA", description="Project name at the tracker.")
    run: str | None = Field(default=None, description="Run display name.")


class QuantizationSettings(Strict):
    """Quantization-aware training of the residual adapters."""

    enabled: bool = Field(
        default=False, description="Wrap adapter projections in fake-quant modules."
    )
    backend: QuantizationBackend = Field(
        default=QuantizationBackend.FBGEMM, description="Quantization engine."
    )


class CausalDataSettings(Strict):
    """Instruction dataset options."""

    path: str = Field(min_length=1, description="Hub dataset name or local json/jsonl path.")
    validation: int = Field(default=2000, ge=0, description="Number of held-out validation rows.")
    cutoff: int = Field(default=256, gt=0, description="Maximum tokens per example.")
    inputs: bool = Field(default=True, description="Include prompt tokens in the loss.")
    eos: bool = Field(default=False, description="Append EOS when masking prompt tokens.")


class CausalTrainSettings(Strict):
    """Optimisation schedule for causal language model fine-tuning."""

    epochs: float = Field(default=3, gt=0, description="Number of passes over the data.")
    batch: int = Field(default=128, gt=0, description="Effective batch size.")
    micro: int = Field(default=4, gt=0, description="Per-device batch size.")
    rate: float = Field(default=3e-4, gt=0, description="Peak learning rate.")
    warmup: int = Field(default=100, ge=0, description="Linear warmup steps.")
    logging: int = Field(default=10, gt=0, description="Steps between metric logs.")
    evaluation: int = Field(default=200, gt=0, description="Steps between evaluations.")
    checkpoints: int = Field(default=3, gt=0, description="Checkpoints retained on disk.")
    grouping: bool = Field(default=False, description="Group samples by length.")
    compile: bool = Field(default=False, description="Compile the model with torch.compile.")

    @model_validator(mode="after")
    def __validate_batches(self) -> "CausalTrainSettings":
        if self.batch % self.micro != 0:
            raise ValueError(f"batch ({self.batch}) must be a multiple of micro ({self.micro}).")
        return self


class CausalSettings(Strict):
    """Complete configuration of a causal language model fine-tuning run."""

    name: str = Field(min_length=1, description="Run identifier used for outputs and tracking.")
    seed: int = Field(default=42, description="Global random seed.")
    output: Path = Field(description="Directory receiving checkpoints and adapters.")
    resume: Path | None = Field(default=None, description="Checkpoint or adapter to resume from.")
    model: ModelSettings
    data: CausalDataSettings
    adapter: AdapterSettings = Field(default_factory=AdapterSettings)
    adaptation: AdaptationSettings = Field(default_factory=AdaptationSettings)
    train: CausalTrainSettings = Field(default_factory=CausalTrainSettings)
    quantization: QuantizationSettings = Field(default_factory=QuantizationSettings)
    tracking: TrackingSettings = Field(default_factory=TrackingSettings)

    @property
    def accumulation(self) -> int:
        """Gradient accumulation steps implied by batch and micro batch."""
        return self.train.batch // self.train.micro


class VisionDataSettings(Strict):
    """Image captioning dataset options."""

    path: str = Field(min_length=1, description="Hub dataset name.")
    samples: int = Field(default=100, gt=0, description="Training rows taken from the head.")
    holdout: float = Field(default=0.1, gt=0, lt=1, description="Fraction used for validation.")
    length: int = Field(default=512, gt=0, description="Maximum token length of a sample.")
    image: int = Field(default=512, gt=0, description="Longest image edge in pixels.")
    instruction: str = Field(default="Describe this image.", description="User turn text.")


class VisionTrainSettings(Strict):
    """Optimisation schedule for vision-language fine-tuning."""

    epochs: int = Field(default=10, gt=0, description="Maximum number of epochs.")
    batch: int = Field(default=4, gt=0, description="Batch size.")
    rate: float = Field(default=2e-4, gt=0, description="Peak learning rate.")
    warmup: float = Field(default=0.1, ge=0, lt=1, description="Fraction of steps for warmup.")
    clip: float = Field(default=1.0, gt=0, description="Gradient norm clipping threshold.")
    logging: int = Field(default=10, gt=0, description="Steps between console logs.")


class EvaluationSettings(Strict):
    """Caption generation and early stopping options."""

    beams: int = Field(default=3, gt=0, description="Beam width for generation.")
    tokens: int = Field(default=64, gt=0, description="Maximum generated tokens.")
    patience: int = Field(default=3, gt=0, description="Epochs without BLEU gain before stopping.")


class VisionSettings(Strict):
    """Complete configuration of a vision-language fine-tuning run."""

    name: str = Field(min_length=1, description="Run identifier used for outputs and tracking.")
    seed: int = Field(default=42, description="Global random seed.")
    output: Path = Field(description="Directory receiving the trained adapter.")
    model: ModelSettings
    data: VisionDataSettings
    adapter: AdapterSettings = Field(default_factory=lambda: AdapterSettings(automatic=False))
    train: VisionTrainSettings = Field(default_factory=VisionTrainSettings)
    evaluation: EvaluationSettings = Field(default_factory=EvaluationSettings)
    tracking: TrackingSettings = Field(default_factory=TrackingSettings)


class BenchmarkSettings(Strict):
    """Which benchmark tasks to run and how."""

    tasks: tuple[str, ...] = Field(
        default=("arc_easy", "hellaswag"), min_length=1, description="lm-eval task names."
    )
    shots: int = Field(default=0, ge=0, description="Few-shot examples per task.")
    limit: int | None = Field(default=None, gt=0, description="Cap on samples per task.")
    batch: int = Field(default=8, gt=0, description="Evaluation batch size.")


class EvaluationRunSettings(Strict):
    """Complete configuration of a benchmark evaluation of a trained run."""

    name: str = Field(min_length=1, description="Evaluation identifier.")
    run: Path = Field(description="Output directory of the training run to evaluate.")
    seed: int = Field(default=42, description="Random seed for the harness.")
    model: ModelSettings
    benchmark: BenchmarkSettings = Field(default_factory=BenchmarkSettings)
    tracking: TrackingSettings = Field(default_factory=TrackingSettings)
