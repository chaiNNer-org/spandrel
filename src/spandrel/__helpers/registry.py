from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

from .canonicalize import canonicalize_state_dict
from .model_descriptor import ModelDescriptor, StateDict


class UnsupportedModelError(Exception):
    pass


@dataclass(frozen=True)
class ArchSupport:
    id: str
    """
    The ID of the architecture.

    For built-in architectures, this is the same as the module name. E.g. `spandrel.architectures.RestoreFormer` has the ID `RestoreFormer`.
    """
    detect: Callable[[StateDict], bool]
    """
    Inspects the given state dict and returns True if this architecture is detected.
    """
    load: Callable[[StateDict], ModelDescriptor]
    """
    Loads a model descriptor for this architecture from the given state dict.
    """
    before: tuple[str, ...] = tuple()
    """
    This architecture is detected before the architectures with the given IDs.

    See the documentation of `ArchRegistry` for more information on ordering.
    """


class ArchRegistry:
    """
    A registry of architectures.

    Architectures are detected/loaded in insertion order unless `before` is specified.
    """

    def __init__(self):
        self._architectures: list[ArchSupport] = []
        self._ordered: list[ArchSupport] = []
        self._by_id: dict[str, ArchSupport] = {}

    def copy(self) -> ArchRegistry:
        """
        Returns a copy of the registry.
        """
        new = ArchRegistry()
        new._architectures = self._architectures.copy()
        new._ordered = self._ordered.copy()
        new._by_id = self._by_id.copy()
        return new

    def __contains__(self, id: str) -> bool:
        return id in self._by_id

    def __getitem__(self, id: str) -> ArchSupport:
        return self._by_id[id]

    def get(self, id: str) -> ArchSupport | None:
        return self._by_id.get(id, None)

    def architectures(
        self,
        order: Literal["insertion", "detection"] = "insertion",
    ) -> list[ArchSupport]:
        """
        Returns a new list with all architectures in the registry.

        The order of architectures in the list is either insertion order or the order in which architectures are detected.
        """
        if order == "insertion":
            return list(self._architectures)
        elif order == "detection":
            return list(self._ordered)
        else:
            raise ValueError(f"Invalid order: {order}")

    def add(self, *architectures: ArchSupport):
        """
        Adds the given architectures to the registry.

        Throws an error if an architecture with the same ID already exists.
        Throws an error if a circular dependency of `detect_before` references is detected.

        If an error is thrown, the registry is left unchanged.
        """

        new_architectures = self._architectures.copy()
        new_by_id = self._by_id.copy()
        for arch in architectures:
            if arch.id in new_by_id:
                raise ValueError(f"Duplicate architecture ID: {arch.id}")

            new_architectures.append(arch)
            new_by_id[arch.id] = arch

        new_ordered = ArchRegistry._get_ordered(new_architectures)

        self._architectures = new_architectures
        self._ordered = new_ordered
        self._by_id = new_by_id

    @staticmethod
    def _get_ordered(architectures: list[ArchSupport]) -> list[ArchSupport]:
        inv_before: dict[str, list[str]] = {}
        by_id: dict[str, ArchSupport] = {}
        for arch in architectures:
            by_id[arch.id] = arch
            for before in arch.before:
                if before not in inv_before:
                    inv_before[before] = []
                inv_before[before].append(arch.id)

        ordered: list[ArchSupport] = []
        seen: set[ArchSupport] = set()
        stack: list[str] = []

        def visit(arch: ArchSupport):
            if arch.id in stack:
                raise ValueError(
                    f"Circular dependency in architecture detection: {' -> '.join([*stack, arch.id])}"
                )
            if arch in seen:
                return
            seen.add(arch)
            stack.append(arch.id)

            for before in inv_before.get(arch.id, []):
                visit(by_id[before])

            ordered.append(arch)
            stack.pop()

        for arch in architectures:
            visit(arch)

        return ordered

    def load(self, state_dict: StateDict) -> ModelDescriptor:
        """
        Detects the architecture of the given state dict and loads it.

        This will canonicalize the state dict if it isn't already.

        Throws an `UnsupportedModelError` if the model architecture is not supported.
        """

        state_dict = canonicalize_state_dict(state_dict)

        for arch in self._ordered:
            if arch.detect(state_dict):
                return arch.load(state_dict)

        raise UnsupportedModelError
