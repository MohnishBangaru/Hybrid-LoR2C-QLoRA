"""Mixed-precision training epoch for vision-language models."""

import logging
import time

import torch
from torch import Tensor, nn
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler

from lor2c.application.ports import Tracker
from lor2c.application.schema import Bundle, EpochReport, Loaders, Metrics
from lor2c.domain.exceptions import ConfigurationError
from lor2c.infrastructure.optimizer import GroupedOptimizerFactory
from lor2c.settings.schema import VisionTrainSettings

LOGGER = logging.getLogger(__name__)


class TorchVisionTrainerPort:
    """Owns optimizer, scheduler and scaler; trains one epoch per call."""

    def __init__(
        self, *, settings: VisionTrainSettings, epochs: int, tracker: Tracker, ratio: float = 1.0
    ) -> None:
        self.__settings = settings
        self.__epochs = epochs
        self.__tracker = tracker
        self.__ratio = ratio
        self.__optimizer: torch.optim.Optimizer | None = None
        self.__scheduler: torch.optim.lr_scheduler.LambdaLR | None = None
        self.__scaler: GradScaler | None = None
        self.__step = 0

    def epoch(self, *, bundle: Bundle[nn.Module], loaders: Loaders, index: int) -> EpochReport:
        """Run one pass over `loaders.train` and return aggregate statistics."""
        model = bundle.model
        device = next(model.parameters()).device
        optimizer, scheduler, scaler = self.__lazy(model=model, steps=len(loaders.train))
        model.train()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        total = 0.0
        for position, batch in enumerate(loaders.train):
            loss = self.__step_once(
                model=model,
                batch={key: value.to(device) for key, value in batch.items()},
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                device=device,
            )
            total += loss
            self.__step += 1
            self.__tracker.log(
                metrics=Metrics(
                    step=self.__step,
                    values={"loss": loss, "rate": scheduler.get_last_lr()[0]},
                )
            )
            if position % self.__settings.logging == 0:
                LOGGER.info(
                    "Step", extra={"ctx_epoch": index, "ctx_position": position, "ctx_loss": loss}
                )
        memory = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        return EpochReport(
            epoch=index,
            loss=total / max(1, len(loaders.train)),
            seconds=time.perf_counter() - started,
            memory=memory,
            step=self.__step,
        )

    def __lazy(
        self, *, model: nn.Module, steps: int
    ) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR, GradScaler]:
        if (
            self.__optimizer is not None
            and self.__scheduler is not None
            and self.__scaler is not None
        ):
            return self.__optimizer, self.__scheduler, self.__scaler
        try:
            from transformers import get_linear_schedule_with_warmup
        except ImportError as exception:
            raise ConfigurationError("Install lor2c[huggingface] to train.") from exception
        optimizer = GroupedOptimizerFactory(rate=self.__settings.rate, ratio=self.__ratio).build(
            model=model
        )
        total = self.__epochs * steps
        scheduler: torch.optim.lr_scheduler.LambdaLR = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=max(1, int(self.__settings.warmup * total)),
            num_training_steps=total,
        )
        device = next(model.parameters()).device
        scaler = GradScaler(device.type, enabled=device.type == "cuda")
        self.__optimizer, self.__scheduler, self.__scaler = optimizer, scheduler, scaler
        return optimizer, scheduler, scaler

    def __step_once(
        self,
        *,
        model: nn.Module,
        batch: dict[str, Tensor],
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LambdaLR,
        scaler: GradScaler,
        device: torch.device,
    ) -> float:
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            loss: Tensor = model(**batch).loss
        scaler.scale(loss).backward()  # type: ignore[no-untyped-call]
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], self.__settings.clip
        )
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        return float(loss.item())
