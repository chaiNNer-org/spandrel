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
        size_requirements: SizeRequirements | None = None,
    ):
        self.model: T = model
        """
        The model itself: a `torch.nn.Module` with weights loaded in.

        The specific subclass of `torch.nn.Module` depends on the model architecture.
        """
        self.state_dict: StateDict = state_dict
        """
        The state dict of the model (weights and biases).
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

    def to(
        self,
        device: int | torch.device | None = torch.device("cpu"),
        dtype: torch.dtype | str | None = None,
        non_blocking: bool = False,
    ):
        self.model.to(device, dtype, non_blocking)
        return self

    def half(self):
        self.model.half()
        return self

    def float(self):
        self.model.float()
        return self

    def cuda(self):
        self.model.cuda()
        return self

    def cpu(self):
        self.model.cpu()
        return self

    def eval(self):
        self.model.eval()
        return self

    def __call__(self, *args, **kwargs):  # noqa: ANN002
        return self.model(*args, **kwargs)


class SRModelDescriptor(ModelBase[T], Generic[T]):
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """
        Upscale the given image by the scale factor of the model.
        """
        return self.model(image)


class FaceSRModelDescriptor(ModelBase[T], Generic[T]):
    def __call__(
        self, image: torch.Tensor, return_rgb: bool = False, weight: float = 0.5
    ) -> torch.Tensor:
        return self.model(image, return_rgb=return_rgb, weight=weight)


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

    def __call__(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Inpaints the given input image in the masked areas.
        """
        return self.model(image, mask)


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

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """
        Runs the restoration model on the given input image.
        """
        return self.model(image)


ModelDescriptor = Union[
    SRModelDescriptor,
    FaceSRModelDescriptor,
    InpaintModelDescriptor,
    RestorationModelDescriptor,
]
"""
A model descriptor is a loaded model with metadata. Metadata includes the
architecture, purpose, tags, and other information about the model.

The purpose of a model is described by the type of the model descriptor. E.g.
a super resolution model has a descriptor of type `SRModelDescriptor`.
"""
