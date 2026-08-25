"""Downstream benchmark scoring through EleutherAI lm-eval."""

import logging

from torch import nn

from lor2c.application.schema import BenchmarkReport
from lor2c.domain.exceptions import ConfigurationError
from lor2c.infrastructure.huggingface.context import HubContext
from lor2c.settings.schema import BenchmarkSettings

LOGGER = logging.getLogger(__name__)


class LmEvalBenchmark:
    """Runs `lm_eval.simple_evaluate` on an in-memory model and flattens the metrics."""

    PREFERRED = ("acc_norm,none", "acc,none", "exact_match,none", "f1,none")

    def __init__(self, *, context: HubContext) -> None:
        self.__context = context

    def run(self, *, model: nn.Module, settings: BenchmarkSettings, seed: int) -> BenchmarkReport:
        """Evaluate every configured task and return `task/metric` scores."""
        try:
            from lm_eval import simple_evaluate
            from lm_eval.models.huggingface import HFLM
        except ImportError as exception:
            raise ConfigurationError(
                f"Install lor2c[evaluate] to run benchmarks ({exception})."
            ) from exception
        wrapped = HFLM(
            pretrained=model, tokenizer=self.__context.tokenizer(), batch_size=settings.batch
        )
        results = simple_evaluate(
            model=wrapped,
            tasks=list(settings.tasks),
            num_fewshot=settings.shots,
            limit=settings.limit,
            random_seed=seed,
            numpy_random_seed=seed,
            torch_random_seed=seed,
            fewshot_random_seed=seed,
        )
        if results is None:
            raise ConfigurationError("lm-eval returned no results.")
        scores: dict[str, float] = {}
        samples = 0
        for task, metrics in results["results"].items():
            for key in self.PREFERRED:
                if key in metrics:
                    scores[f"{task}/{key.split(',')[0]}"] = float(metrics[key])
                    break
            samples += int(results.get("n-samples", {}).get(task, {}).get("effective", 0))
        LOGGER.info("Benchmark finished", extra={"ctx_scores": scores})
        return BenchmarkReport(scores=scores, samples=samples)
