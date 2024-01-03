from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Generic, Literal, TypeVar, Union

import torch
from torch import Tensor

T = TypeVar("T", bound=torch.nn.Module, covariant=True)

StateDict = Dict[str, Any]
"""
Spandrel's type alias for PyTorch state dicts.

See https://pytorch.org/tutorials/recipes/recipes/what_is_state_dict.html
"""


@dataclass
class SizeRequirements:
    """
    A set of requirements for the size of an input image.
    """

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


Purpose = Literal["SR", "FaceSR", "Inpainting", "Restoration"]
"""
A short string describing the purpose of the model.

- `SR`: Super resolution
- `FaceSR`: Face super resolution
- `Inpainting`: Image inpainting
- `Restoration`: Image restoration (denoising, deblurring, JPEG, etc.)
"""


class ModelTiling(Enum):
    """
    Describes whether and how a model supports tiling.
    """

    SUPPORTED = 1
    """
    The model supports tiling.
    """
    DISCOURAGED = 2
    """
    The model supports tiling, but it is not recommended.

    This might be because the model heavily relies and global image information,
    and so tiling will likely cause artifacts.
    """
    INTERNAL = 3
    """
    The model does tiling (or similar) internally.

    This is typically done by models that require global image information to
    work properly. As such, it is recommend to not do any tiling before passing
    the image to the model.
    """


class ModelBase(ABC, Generic[T]):
    """
    The base class of all model descriptors.

    This is mostly intended for `instanceof` checks in user code. Use `ModelDescriptor` for type hints instead.
    """

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
        tiling: ModelTiling = ModelTiling.SUPPORTED,
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

        Requirements are specific to individual models and may be different for models of the same architecture.
        """
        self.tiling: ModelTiling = tiling
        """
        Whether the model supports tiling.

        Technically, all models support tiling. This is simply a recommendation
        on how to best use the model.
        """

        self.model.load_state_dict(state_dict)  # type: ignore

    @property
    @abstractmethod
    def purpose(self) -> Purpose:
        """
        The purpose of this model.
        """
        ...

    @property
    def device(self) -> torch.device:
        """
        The device of the underlying module.

        Use `to` to move the model to a different device.
        """
        # This makes the following assumptions:
        # - The model is on a single device
        # - The model has at least one parameter
        # Both are true for all models implemented in Spandrel.
        # https://stackoverflow.com/questions/58926054/how-to-get-the-device-type-of-a-pytorch-module-conveniently
        return next(self.model.parameters()).device

    def to(self, device: str | torch.device):
        """
        Moves the parameters and buffers of the underlying module to the given device.

        Use `device` to get the current device of the model.
        """
        self.model.to(device)
        return self

    def eval(self):
        """
        Sets the underlying module in evaluation mode.

        Same as `self.train(False)`.
        """
        self.model.eval()
        return self

    def train(self, mode: bool = True):
        """
        Sets the underlying module in training mode.

        Same as `self.model.train(mode)`.
        """
        self.model.train(mode)
        return self


class ImageModelDescriptor(ModelBase[T], Generic[T]):
    """
    A model that takes an image as input and returns an image. See `__call__` for more information.
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
        tiling: ModelTiling = ModelTiling.SUPPORTED,
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
            tiling=tiling,
        )

        self._purpose: Literal["SR", "FaceSR", "Restoration"] = purpose

        self._call_fn = call_fn or (lambda model, image: model(image))

    @property
    def purpose(self) -> Literal["SR", "FaceSR", "Restoration"]:
        return self._purpose

    def __call__(self, image: Tensor) -> Tensor:
        """
        Takes a single image tensor as input and returns a single image tensor as output.

        The `image` tensor must be a 4D tensor with shape `(1, input_channels, H, W)`. The width and height are expected to satisfy the `size_requirements` of the model. The data type (float32, float16, bfloat16) and device of the `image` tensor must be the same as the model. The range of the `image` tensor must be ``[0, 1]``.

        The output tensor will be a 4D tensor with shape `(1, output_channels, H*scale, W*scale)`. The data type and device of the output tensor will be the same as the `image` tensor. The range of the output tensor will be ``[0, 1]``.
        """

        output = self._call_fn(self.model, image)
        assert isinstance(
            output, Tensor
        ), f"Expected {type(self.model).__name__} model to returns a tensor, but got {type(output)}"
        return output


class MaskedImageModelDescriptor(ModelBase[T], Generic[T]):
    """
    A model that takes an image and a mask for that image as input and returns an image. See `__call__` for more information.
    """

    def __init__(
        self,
        model: T,
        state_dict: StateDict,
        architecture: str,
        purpose: Literal["Inpainting"],
        tags: list[str],
        supports_half: bool,
        supports_bfloat16: bool,
        input_channels: int,
        output_channels: int,
        size_requirements: SizeRequirements | None = None,
        tiling: ModelTiling = ModelTiling.SUPPORTED,
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
            tiling=tiling,
        )

        self._purpose: Literal["Inpainting"] = purpose

        self._call_fn = call_fn or (lambda model, image, mask: model(image, mask))

    @property
    def purpose(self) -> Literal["Inpainting"]:
        return self._purpose

    def __call__(self, image: Tensor, mask: Tensor) -> Tensor:
        """
        Takes an image tensor and an image mask tensor as input and returns a single image tensor as output.

        The data type (float32, float16, bfloat16) and device of the `image` and `mask` tensors must be the same as the model.

        The `image` tensor must be a 4D tensor with shape `(1, input_channels, H, W)`. The width and height are expected to satisfy the `size_requirements` of the model. The range of the `image` tensor must be ``[0, 1]``.

        The `mask` tensor must be a 4D tensor with shape `(1, 1, H, W)`. The width and height must be the same as `image` tensor. The values of the `mask` tensor must be either 0 (keep) or 1 (inpaint).

        The output tensor will be a 4D tensor with shape `(1, output_channels, H, W)`. The data type and device of the output tensor will be the same as the `image` tensor. The range of the output tensor will be ``[0, 1]``.
        """

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

The API of a model is described by the type of the model descriptor. E.g.
a SISR model will have a descriptor of type `ImageModelDescriptor`.
"""
