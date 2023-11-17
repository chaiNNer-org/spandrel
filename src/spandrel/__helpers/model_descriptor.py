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
        """
        return self.minimum is None and self.multiple_of is None and not self.square


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
        size_requirements: SizeRequirements | None = None,
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

        self.size_requirements: SizeRequirements = (
            size_requirements or SizeRequirements()
        )

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
        size_requirements: SizeRequirements | None = None,
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
            size_requirements=size_requirements,
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
        size_requirements: SizeRequirements | None = None,
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
            size_requirements=size_requirements,
        )


ModelDescriptor = Union[
    SRModelDescriptor,
    FaceSRModelDescriptor,
    InpaintModelDescriptor,
    RestorationModelDescriptor,
]
