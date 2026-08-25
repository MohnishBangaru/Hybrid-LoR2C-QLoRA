"""Named constants and enumerations shared across the domain."""

import math
import re
from enum import IntEnum, StrEnum
from typing import Final


class AdapterMode(StrEnum):
    """Which low-rank strategy is applied to the base model."""

    BASE = "base"
    LOR2C = "lor2c"


class Precision(StrEnum):
    """Floating point precision used to load and train the base model."""

    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"
    FLOAT32 = "float32"


class QuantizationBackend(StrEnum):
    """Quantization engine used for quantization-aware training of adapters."""

    FBGEMM = "fbgemm"
    QNNPACK = "qnnpack"


class TrackerKind(StrEnum):
    """Experiment tracking backend."""

    NONE = "none"
    WANDB = "wandb"


class Label(IntEnum):
    """Sentinel label values understood by the loss function."""

    IGNORED = -100


class Rank(IntEnum):
    """Default rank/alpha pairs chosen from the base model width."""

    SMALL_HIDDEN_LIMIT = 2048
    SMALL_RANK = 4
    SMALL_ALPHA = 8
    LARGE_RANK = 16
    LARGE_ALPHA = 32


KAIMING_NEGATIVE_SLOPE: Final[float] = math.sqrt(5)
RESIDUAL_ADAPTER_PREFIX: Final[str] = "floor"
CAPTION_NUMERIC_PREFIX: Final[re.Pattern[str]] = re.compile(r"^\s*\d+(\.\d+)*\s*[:.\-\u2013]?\s*")


class Alpaca:
    """Default Alpaca instruction template strings."""

    DESCRIPTION: Final[str] = "Template used by Alpaca-LoRA."
    WITH_INPUT: Final[str] = (
        "Below is an instruction that describes a task, paired with an input that provides "
        "further context. Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{context}\n\n### Response:\n"
    )
    WITHOUT_INPUT: Final[str] = (
        "Below is an instruction that describes a task. Write a response that appropriately "
        "completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n"
    )
    RESPONSE_SPLIT: Final[str] = "### Response:"
