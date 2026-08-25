"""`lor2c` command-line entry point."""

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from lor2c import __version__
from lor2c.cli.factory import ServiceFactory
from lor2c.domain.exceptions import DomainError
from lor2c.infrastructure.logging import LoggingConfigurator
from lor2c.settings.loader import SettingsLoader
from lor2c.settings.schema import CausalSettings, EvaluationRunSettings, VisionSettings

LOGGER = logging.getLogger(__name__)


class CommandLine:
    """Parses arguments, loads settings and dispatches to the matching training service."""

    CAUSAL = "causal"
    VISION = "vision"
    EVALUATE = "evaluate"

    def __init__(
        self, *, factory: ServiceFactory, loader: SettingsLoader, logging_: LoggingConfigurator
    ) -> None:
        self.__factory = factory
        self.__loader = loader
        self.__logging = logging_

    @classmethod
    def run(cls, argv: Sequence[str] | None = None) -> int:
        """Console-script entry: build the default command line and execute it."""
        command = cls(
            factory=ServiceFactory(), loader=SettingsLoader(), logging_=LoggingConfigurator()
        )
        code = command.execute(argv=list(argv if argv is not None else sys.argv[1:]))
        if argv is None:
            sys.exit(code)
        return code

    def execute(self, *, argv: list[str]) -> int:
        """Run the command described by `argv` and return a process exit code."""
        namespace = self.__parser().parse_args(argv)
        self.__logging.configure(level=logging.DEBUG if namespace.verbose else logging.INFO)
        try:
            outcome: object
            if namespace.command == self.CAUSAL:
                settings = self.__loader.load(path=namespace.config, schema=CausalSettings)
                outcome = self.__factory.causal(settings=settings).run(settings=settings)
            elif namespace.command == self.EVALUATE:
                evaluation = self.__loader.load(path=namespace.config, schema=EvaluationRunSettings)
                outcome = self.__factory.evaluate(settings=evaluation).run(settings=evaluation)
            else:
                vision = self.__loader.load(path=namespace.config, schema=VisionSettings)
                outcome = self.__factory.vision(settings=vision).run(settings=vision)
        except DomainError as exception:
            LOGGER.error("Run failed", extra={"ctx_error": str(exception)})
            return 2
        LOGGER.info(
            "Run complete",
            extra={"ctx_outcome": outcome.model_dump(mode="json")},  # type: ignore[attr-defined]
        )
        return 0

    def __parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(prog="lor2c", description="LoR2C/QLoRA fine-tuning.")
        parser.add_argument("--version", action="version", version=__version__)
        parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging.")
        commands = parser.add_subparsers(dest="command", required=True)
        for name, help_ in (
            (self.CAUSAL, "Fine-tune a causal language model with LoR2C residual adapters."),
            (self.VISION, "Fine-tune a vision-language model with low-rank adapters."),
            (self.EVALUATE, "Score a trained causal run on lm-eval benchmarks."),
        ):
            sub = commands.add_parser(name, help=help_)
            sub.add_argument("config", type=Path, help="Path to a YAML settings file.")
        return parser
