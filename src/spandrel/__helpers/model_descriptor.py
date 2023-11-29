from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, Generic, Literal, TypeVar, Union

import torch
from torch import Tensor

T = TypeVar("T", bound=torch.nn.Module, covariant=True)

StateDict = Dict[str, Any]


@dataclass
class SizeRequirements:
    minimum: int = 0
    """
    The minimum size of the input image in pixels.

    Default/neutral value: `0`
    """
    multiple_of: int = 1
    """
    The width and height of the image must be a multiple of this value.

    Default/neutral value: `1`
    """
    square: bool = False
    """
    The image must be square.

    Default/neutral value: `False`
    """

    @property
    def none(self) -> bool:
        """
        Returns True if no size requirements are specified.

        If True, then `check` is guaranteed to always return True.
        """
        return self.minimum == 0 and self.multiple_of == 1 and not self.square

    def check(self, width: int, height: int) -> bool:
        """
        Checks if the given width and height satisfy the size requirements.
        """
        if width < self.minimum or height < self.minimum:
            return False

        if width % self.multiple_of != 0 or height % self.multiple_of != 0:
            return False

        if self.square and width != height:
            return False

        return True


Purpose = Literal["SR", "FaceSR", "Inpaint", "Restoration"]
"""
A short string describing the purpose of the model.

- `SR`: Super resolution
- `FaceSR`: Face super resolution
- `Inpaint`: Image inpainting
- `Restoration`: Image restoration (denoising, deblurring, JPEG, etc.)
"""


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
        """
        The model itself: a `torch.nn.Module` with weights loaded in.

        The specific subclass of `torch.nn.Module` depends on the model architecture.
        """
        self.architecture: str = architecture
        """
        The name of the model architecture. E.g. "ESRGAN".
        """
        self.tags: list[str] = tags
        """
        A list of tags for the model, usually describing the size or model
        parameters. E.g. "64nf" or "large".

        Tags are specific to the architecture of the model. Some architectures
        may not have any tags.
        """
        self.supports_half: bool = supports_half
        """
        Whether the model supports half precision (fp16).
        """
        self.supports_bfloat16: bool = supports_bfloat16
        """
        Whether the model supports bfloat16 precision.
        """

        self.scale: int = scale
        """
        The output scale of super resolution models. E.g. 4x, 2x, 1x.

        Models that are not super resolution models (e.g. denoisers) have a
        scale of 1.
        """
        self.input_channels: int = input_channels
        """
        The number of input image channels of the model. E.g. 3 for RGB, 1 for grayscale.
        """
        self.output_channels: int = output_channels
        """
        The number of output image channels of the model. E.g. 3 for RGB, 1 for grayscale.
        """

        self.size_requirements: SizeRequirements = (
            size_requirements or SizeRequirements()
        )
        """
        Size requirements for the input image. E.g. minimum size.
        """

        self.model.load_state_dict(state_dict)  # type: ignore

    @property
    @abstractmethod
    def purpose(self) -> Purpose:
        """
        The purpose of this model.
        """
        ...

    def to(self, device: torch.device):
        self.model.to(device)
        return self

    def eval(self):
        self.model.eval()
        return self

    def train(self, mode: bool = True):
        self.model.train(mode)
        return self


class ImageModelDescriptor(ModelBase[T], Generic[T]):
    """
    A model that takes an image as input and returns an image.
    """

    def __init__(
        self,
        model: T,
        state_dict: StateDict,
        architecture: str,
        purpose: Literal["SR", "FaceSR", "Restoration"],
        tags: list[str],
        supports_half: bool,
        supports_bfloat16: bool,
        scale: int,
        input_channels: int,
        output_channels: int,
        size_requirements: SizeRequirements | None = None,
        call_fn: Callable[[T, Tensor], Tensor] | None = None,
    ):
        assert (
            purpose != "Restoration" or scale == 1
        ), "Restoration models must have a scale of 1"

        super().__init__(
            model,
            state_dict,
            architecture,
            tags,
            supports_half=supports_half,
            supports_bfloat16=supports_bfloat16,
            scale=scale,
            input_channels=input_channels,
            output_channels=output_channels,
            size_requirements=size_requirements,
        )

        self._purpose: Literal["SR", "FaceSR", "Restoration"] = purpose

        self._call_fn = call_fn or (lambda model, image: model(image))

    @property
    def purpose(self) -> Literal["SR", "FaceSR", "Restoration"]:
        return self._purpose

    def __call__(self, image: Tensor) -> Tensor:
        output = self._call_fn(self.model, image)
        assert isinstance(
            output, Tensor
        ), f"Expected {type(self.model).__name__} model to returns a tensor, but got {type(output)}"
        return output


class MaskedImageModelDescriptor(ModelBase[T], Generic[T]):
    """
    A model that takes an image and a mask for that image as input and returns an image.
    """

    def __init__(
        self,
        model: T,
        state_dict: StateDict,
        architecture: str,
        purpose: Literal["Inpaint"],
        tags: list[str],
        supports_half: bool,
        supports_bfloat16: bool,
        input_channels: int,
        output_channels: int,
        size_requirements: SizeRequirements | None = None,
        call_fn: Callable[[T, Tensor, Tensor], Tensor] | None = None,
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

        self._purpose: Literal["Inpaint"] = purpose

        self._call_fn = call_fn or (lambda model, image, mask: model(image, mask))

    @property
    def purpose(self) -> Literal["Inpaint"]:
        return self._purpose

    def __call__(self, image: Tensor, mask: Tensor) -> Tensor:
        output = self._call_fn(self.model, image, mask)
        assert isinstance(
            output, Tensor
        ), f"Expected {type(self.model).__name__} model to returns a tensor, but got {type(output)}"
        return output


ModelDescriptor = Union[
    ImageModelDescriptor[torch.nn.Module],
    MaskedImageModelDescriptor[torch.nn.Module],
]
"""
A model descriptor is a loaded model with metadata. Metadata includes the
architecture, purpose, tags, and other information about the model.

The purpose of a model is described by the type of the model descriptor. E.g.
a super resolution model has a descriptor of type `SRModelDescriptor`.
"""
