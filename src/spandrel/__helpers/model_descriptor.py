from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import Any, Dict, Generic, TypeVar, Union

import torch

T = TypeVar("T", bound=torch.nn.Module)

StateDict = Dict[str, Any]


@dataclass
class SizeRequirements:
    minimum: int | None = None
    """
    The minimum size of the input image in pixels.
    """
    multiple_of: int | None = None
    """
    The width and height of the image must be a multiple of this value.
    """
    square: bool = False
    """
    The image must be square.
    """

    @property
    def none(self) -> bool:
        """
        Returns True if no size requirements are specified.

        If True, then `check` is guaranteed to always return True.
        """
        return self.minimum is None and self.multiple_of is None and not self.square

    def check(self, width: int, height: int) -> bool:
        """
        Checks if the given width and height satisfy the size requirements.
        """
        if self.minimum is not None:
            if width < self.minimum or height < self.minimum:
                return False

        if self.multiple_of is not None:
            if width % self.multiple_of != 0 or height % self.multiple_of != 0:
                return False

        if self.square:
            if width != height:
                return False

        return True


class ModelBase(ABC, Generic[T]):
    def __init__(
        self,
        model: T,
        state_dict: StateDict,
        architecture: str,
        tags: list[str],
        supports_half: bool,
        supports_bfloat16: bool,
        scale: int,
        input_channels: int,
        output_channels: int,
        size: SizeRequirements | None = None,
    ):
        self.model: T = model
        self.state_dict: StateDict = state_dict
        self.architecture: str = architecture
        self.tags: list[str] = tags
        self.supports_half: bool = supports_half
        self.supports_bfloat16: bool = supports_bfloat16

        self.scale: int = scale
        self.input_channels: int = input_channels
        self.output_channels: int = output_channels

        self.size: SizeRequirements = size or SizeRequirements()

        self.model.load_state_dict(state_dict)  # type: ignore

    def to(self, device: torch.device):
        self.model.to(device)
        return self


class SRModelDescriptor(ModelBase[T], Generic[T]):
    pass


class FaceSRModelDescriptor(ModelBase[T], Generic[T]):
    pass


class InpaintModelDescriptor(ModelBase[T], Generic[T]):
    def __init__(
        self,
        model: T,
        state_dict: StateDict,
        architecture: str,
        tags: list[str],
        supports_half: bool,
        supports_bfloat16: bool,
        input_channels: int,
        output_channels: int,
        size: SizeRequirements | None = None,
    ):
        super().__init__(
            model,
            state_dict,
            architecture,
            tags,
            supports_half=supports_half,
            supports_bfloat16=supports_bfloat16,
            scale=1,
            input_channels=input_channels,
            output_channels=output_channels,
            size=size,
        )


class RestorationModelDescriptor(ModelBase[T], Generic[T]):
    def __init__(
        self,
        model: T,
        state_dict: StateDict,
        architecture: str,
        tags: list[str],
        supports_half: bool,
        supports_bfloat16: bool,
        input_channels: int,
        output_channels: int,
        size: SizeRequirements | None = None,
    ):
        super().__init__(
            model,
            state_dict,
            architecture,
            tags,
            supports_half=supports_half,
            supports_bfloat16=supports_bfloat16,
            scale=1,
            input_channels=input_channels,
            output_channels=output_channels,
            size=size,
        )


ModelDescriptor = Union[
    SRModelDescriptor,
    FaceSRModelDescriptor,
    InpaintModelDescriptor,
    RestorationModelDescriptor,
]
