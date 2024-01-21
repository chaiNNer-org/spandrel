from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Generic, Literal, TypeVar, Union, overload

import torch
from torch import Tensor
from typing_extensions import Self

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

    `minimum` is guaranteed to be a multiple of `multiple_of` and to be >= 0.

    On initialization, if `minimum` is not a multiple of `multiple_of`, it will be rounded up to the next multiple of `multiple_of`.

    Default/neutral value: `0`
    """
    multiple_of: int = 1
    """
    The width and height of the image must be a multiple of this value.

    `multiple_of` is guaranteed to be >= 1.

    Default/neutral value: `1`
    """
    square: bool = False
    """
    The image must be square.

    Default/neutral value: `False`
    """

    def __post_init__(self):
        assert self.minimum >= 0, "minimum must be >= 0"
        assert self.multiple_of >= 1, "multiple_of must be >= 1"

        if self.minimum % self.multiple_of != 0:
            self.minimum = (self.minimum // self.multiple_of + 1) * self.multiple_of

    @property
    def none(self) -> bool:
        """
        Returns True if no size requirements are specified.

        If True, then `check` is guaranteed to always return True.
        """
        return self.minimum == 0 and self.multiple_of == 1 and not self.square

    def check(self, width: int, height: int) -> bool:
        """
        Returns whether the given width and height satisfy the size requirements.
        """
        return self.get_padding(width, height) == (0, 0)

    def get_padding(self, width: int, height: int) -> tuple[int, int]:
        """
        Given an image size, this returns the minimum amount of padding necessary to satisfy the size requirements. The returned padding is in the format `(pad_width, pad_height)` and is guaranteed to be non-negative.
        """

        def ceil_modulo(x: int, mod: int) -> int:
            if x % mod == 0:
                return x
            return (x // mod + 1) * mod

        w: int = max(self.minimum, width)
        h: int = max(self.minimum, height)

        w = ceil_modulo(w, self.multiple_of)
        h = ceil_modulo(h, self.multiple_of)

        if self.square:
            w = h = max(w, h)

        return w - width, h - height


def _pad(t: torch.Tensor, req: SizeRequirements):
    w = t.shape[-1]
    h = t.shape[-2]

    pad_w, pad_h = req.get_padding(w, h)

    if pad_w or pad_h:
        # reflect padding only allows a maximum padding of size - 1
        reflect_pad_w = min(pad_w, w - 1)
        reflect_pad_h = min(pad_h, h - 1)
        t = torch.nn.functional.pad(t, (0, reflect_pad_w, 0, reflect_pad_h), "reflect")

        # do the rest of the padding (if any) with replicate, which has no such restrictions
        pad_w -= reflect_pad_w
        pad_h -= reflect_pad_h
        t = torch.nn.functional.pad(t, (0, pad_w, 0, pad_h), "replicate")

        return True, t
    else:
        return False, t


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


class UnsupportedDtypeError(Exception):
    """
    An error that will be thrown by `.to` if the model does not support the given dtype.

    See `ModelBase.to` for more information.
    """

    pass


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

        Users of spandrel's call API can largely ignore size requirements, because the call API will automatically pad the input image to satisfy the requirements. Size requirements might still be useful for user code that tiles images by allowing it to pick an optimal tile size to avoid padding.
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

    @property
    def dtype(self) -> torch.dtype:
        """
        The data type of the underlying module.

        Use `to` to cast the model to a different data type.
        """
        # this makes the same assumptions as `device`
        return next(self.model.parameters()).dtype

    @overload
    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> Self:
        ...

    @overload
    def to(self, dtype: torch.dtype) -> Self:
        ...

    def to(self, *args: object, **kwargs) -> Self:
        """
        Moves and casts the parameters and buffers of the underlying module to the given device and data type.

        For more information, see https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.to.

        Use `device` to get the current device and `dtype` to get the current data type of the model.

        Throws `UnsupportedDtypeError` if the model does not support the given data type. If you want to force a dtype cast, use `.model.to(dtype)` instead.
        """

        # turn positional arguments into keyword arguments
        def set_kw(name: str, value: object):
            if name in kwargs:
                raise TypeError(f"to() got multiple values for keyword argument {name}")
            kwargs[name] = value

        if len(args) == 1:
            arg: object = args[0]
            if isinstance(arg, torch.dtype):
                set_kw("dtype", arg)
            elif isinstance(arg, (torch.device, str)) or arg is None:
                set_kw("device", arg)
            else:
                raise TypeError(
                    f"to() expected a torch.device or torch.dtype, but got {type(arg)}"
                )
        elif len(args) == 2:
            set_kw("device", args[0])
            set_kw("dtype", args[1])
        elif len(args) > 2:
            raise TypeError(
                f"to() expected at most 2 positional arguments, got {len(args)}"
            )

        device: torch.device | str | None = kwargs.pop("device", None)
        dtype: torch.dtype | None = kwargs.pop("dtype", None)

        if len(kwargs) > 0:
            raise TypeError(f"to() got unexpected keyword arguments {list(kwargs)}")

        if dtype is not None:
            if dtype == torch.float16 and not self.supports_half:
                raise UnsupportedDtypeError(
                    f"{self.architecture} does not support half precision (fp16)"
                )
            if dtype == torch.bfloat16 and not self.supports_bfloat16:
                raise UnsupportedDtypeError(
                    f"{self.architecture} does not support bfloat16 precision"
                )

        if isinstance(device, str):
            device = torch.device(device)

        self.model.to(device=device, dtype=dtype)
        return self

    def half(self) -> Self:
        """
        Moves the parameters and buffers of the underlying module to half precision (fp16).

        Same as `self.to(torch.half)`.
        """
        self.to(torch.half)
        return self

    def bfloat16(self) -> Self:
        """
        Moves the parameters and buffers of the underlying module to bfloat16 precision.

        Same as `self.to(torch.bfloat16)`.
        """
        self.to(torch.bfloat16)
        return self

    def float(self) -> Self:
        """
        Moves the parameters and buffers of the underlying module to single precision (fp32).

        Same as `self.to(torch.float)`.
        """
        self.to(torch.float)
        return self

    def cpu(self) -> Self:
        """
        Moves the parameters and buffers of the underlying module to the CPU.

        Same as `self.to(torch.device("cpu"))`.
        """
        self.model.cpu()
        return self

    def cuda(self, device: int | None = None) -> Self:
        """
        Moves the parameters and buffers of the underlying module to the GPU.

        Same as `self.to(torch.device("cuda"))`.
        """
        self.model.cuda(device)
        return self

    def eval(self) -> Self:
        """
        Sets the underlying module in evaluation mode.

        Same as `self.train(False)`.
        """
        self.model.eval()
        return self

    def train(self, mode: bool = True) -> Self:
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

        The `image` tensor must be a 4D tensor with shape `(1, input_channels, H, W)`. The data type (float32, float16, bfloat16) and device of the `image` tensor must be the same as the model. The range of the `image` tensor must be ``[0, 1]``.

        The output tensor will be a 4D tensor with shape `(1, output_channels, H*scale, W*scale)`. The data type and device of the output tensor will be the same as the `image` tensor. The range of the output tensor will be ``[0, 1]``.

        If the width and height of the `image` tensor do not satisfy the `size_requirements` of the model, then the `image` tensor will be padded to satisfy the requirements. The additional padding will be removed from the output tensor before returning it. If the image already satisfies the requirements, then no padding will be added.
        """
        if len(image.shape) != 4:
            raise ValueError(
                f"Expected image tensor to have 4 dimensions, but got {image.shape}"
            )

        _, _, h, w = image.shape

        # satisfy size requirements
        did_pad, image = _pad(image, self.size_requirements)

        # call model
        output = self._call_fn(self.model, image)
        assert isinstance(
            output, Tensor
        ), f"Expected {type(self.model).__name__} model to returns a tensor, but got {type(output)}"

        # guarantee range
        output = output.clamp_(0, 1)

        # remove padding
        if did_pad:
            output = output[..., : h * self.scale, : w * self.scale]

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

        The `image` tensor must be a 4D tensor with shape `(1, input_channels, H, W)`. The range of the `image` tensor must be ``[0, 1]``.

        The `mask` tensor must be a 4D tensor with shape `(1, 1, H, W)`. The width and height must be the same as `image` tensor. The values of the `mask` tensor must be either 0 (keep) or 1 (inpaint).

        The output tensor will be a 4D tensor with shape `(1, output_channels, H, W)`. The data type and device of the output tensor will be the same as the `image` tensor. The range of the output tensor will be ``[0, 1]``.

        If the width and height of the `image` tensor do not satisfy the `size_requirements` of the model, then the `image` tensor will be padded to satisfy the requirements. The additional padding will be removed from the output tensor before returning it. If the image already satisfies the requirements, then no padding will be added.
        """
        if len(image.shape) != 4:
            raise ValueError(
                f"Expected image tensor to have 4 dimensions, but got {image.shape}"
            )
        if len(mask.shape) != 4:
            raise ValueError(
                f"Expected mask tensor to have 4 dimensions, but got {mask.shape}"
            )

        _, _, h, w = image.shape

        # check mask
        mask_shape = torch.Size([1, 1, h, w])
        if mask.shape != mask_shape:
            raise ValueError(
                f"Expected mask shape to be {mask_shape}, but got {mask.shape}"
            )

        # satisfy size requirements
        did_pad, image = _pad(image, self.size_requirements)
        _, mask = _pad(mask, self.size_requirements)

        # call model
        output = self._call_fn(self.model, image, mask)
        assert isinstance(
            output, Tensor
        ), f"Expected {type(self.model).__name__} model to returns a tensor, but got {type(output)}"

        # guarantee range
        output = output.clamp_(0, 1)

        # remove padding
        if did_pad:
            output = output[..., : h * self.scale, : w * self.scale]

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
