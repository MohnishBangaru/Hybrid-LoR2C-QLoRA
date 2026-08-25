"""Attaches a `ResidualRouter` to decoder layers through forward hooks."""

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from typing import Protocol

from torch import Tensor, nn
from torch.utils.hooks import RemovableHandle

from lor2c.domain.exceptions import ScheduleError
from lor2c.domain.schedule import ResidualRouter


class LayerLocator(Protocol):
    """Finds the ordered decoder layers of a host model."""

    def locate(self, *, model: nn.Module) -> Sequence[nn.Module]:
        """Return decoder layers in forward order."""
        ...


class ModuleListLocator:
    """Locates the first `nn.ModuleList` reachable under a dotted attribute name."""

    def __init__(self, *, attribute: str = "layers") -> None:
        self.__attribute = attribute

    def locate(self, *, model: nn.Module) -> Sequence[nn.Module]:
        """Depth-first search for a `ModuleList` attribute called `attribute`."""
        for name, module in model.named_modules():
            if name.split(".")[-1] == self.__attribute and isinstance(module, nn.ModuleList):
                return list(module)
        raise ScheduleError(
            f"No nn.ModuleList named '{self.__attribute}' found in {type(model).__name__}."
        )


Output = Tensor | tuple[Tensor, ...]
Hook = Callable[[nn.Module, tuple[Tensor, ...], Output], Output]


class HookRouter:
    """Registers one forward hook per decoder layer that delegates to the router."""

    def __init__(self, *, locator: LayerLocator) -> None:
        self.__locator = locator

    @contextmanager
    def attach(self, *, model: nn.Module, router: ResidualRouter) -> Iterator[None]:
        """Install hooks for the lifetime of the context and remove them afterwards."""
        layers = self.__locator.locate(model=model)
        if len(layers) != router.schedule.depth:
            raise ScheduleError(
                f"Model exposes {len(layers)} decoder layers but the schedule expects "
                f"{router.schedule.depth}."
            )
        handles: list[RemovableHandle] = []
        try:
            for index, layer in enumerate(layers):
                handles.append(layer.register_forward_hook(self.__hook(index=index, router=router)))
            yield
        finally:
            for handle in handles:
                handle.remove()
            router.reset()

    @staticmethod
    def __hook(*, index: int, router: ResidualRouter) -> Hook:
        def forward_hook(module: nn.Module, inputs: tuple[Tensor, ...], output: Output) -> Output:
            hidden_in = inputs[0]
            if isinstance(output, tuple):
                routed = router.route(layer=index, hidden_in=hidden_in, hidden_out=output[0])
                return (routed, *output[1:])
            return router.route(layer=index, hidden_in=hidden_in, hidden_out=output)

        return forward_hook
