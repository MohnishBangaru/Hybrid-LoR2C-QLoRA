from pathlib import Path

from lor2c.application.schema import Outcome
from lor2c.cli.main import CommandLine
from lor2c.domain.exceptions import ConfigurationError
from lor2c.infrastructure.logging import LoggingConfigurator
from lor2c.settings.loader import SettingsLoader
from lor2c.settings.schema import CausalSettings


class StubService:
    def __init__(self) -> None:
        self.settings: CausalSettings | None = None

    def run(self, *, settings: CausalSettings) -> Outcome:
        self.settings = settings
        return Outcome(name=settings.name, output=Path("out"), steps=1)


class StubFactory:
    def __init__(self) -> None:
        self.service = StubService()

    def causal(self, *, settings: CausalSettings) -> StubService:
        return self.service

    def vision(self, *, settings: object) -> StubService:
        raise AssertionError("vision should not be built for a causal command")

    def evaluate(self, *, settings: object) -> StubService:
        self.evaluated = settings
        return self.service


class TestCommandLine:
    def __command(self, *, factory: StubFactory) -> CommandLine:
        return CommandLine(factory=factory, loader=SettingsLoader(), logging_=LoggingConfigurator())  # type: ignore[arg-type]

    def test_causal_command_loads_settings_and_runs_service(self, tmp_path: Path) -> None:
        config = tmp_path / "run.yaml"
        config.write_text("name: cli\noutput: out\nmodel: {name: m}\ndata: {path: d}\n")
        factory = StubFactory()
        code = self.__command(factory=factory).execute(argv=["causal", str(config)])
        assert code == 0
        assert factory.service.settings is not None
        assert factory.service.settings.name == "cli"

    def test_evaluate_command_dispatches_to_evaluation_service(self, tmp_path: Path) -> None:
        config = tmp_path / "eval.yaml"
        config.write_text("name: e\nrun: outputs/x\nmodel: {name: m}\n")
        factory = StubFactory()
        code = self.__command(factory=factory).execute(argv=["evaluate", str(config)])
        assert code == 0
        assert factory.evaluated.name == "e"

    def test_invalid_settings_return_nonzero(self, tmp_path: Path) -> None:
        config = tmp_path / "run.yaml"
        config.write_text("name: cli\n")
        code = self.__command(factory=StubFactory()).execute(argv=["causal", str(config)])
        assert code == 2

    def test_domain_error_from_service_returns_nonzero(self, tmp_path: Path) -> None:
        config = tmp_path / "run.yaml"
        config.write_text("name: cli\noutput: out\nmodel: {name: m}\ndata: {path: d}\n")

        class Failing(StubService):
            def run(self, *, settings: CausalSettings) -> Outcome:
                raise ConfigurationError("bad")

        factory = StubFactory()
        factory.service = Failing()
        assert self.__command(factory=factory).execute(argv=["causal", str(config)]) == 2
