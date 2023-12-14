import torch


# Source: https://github.com/kornia/kornia/blob/master/kornia/color/lab.py
# Apache-2.0 license
def linear_rgb_to_lab(image: torch.Tensor) -> torch.Tensor:
    r"""Convert a RGB image to Lab.

    .. image:: _static/img/rgb_to_lab.png

    The input RGB image is assumed to be in the range of :math:`[0, 1]`. Lab
    color is computed using the D65 illuminant and Observer 2.

    Args:
        image: RGB Image to be converted to Lab with shape :math:`(*, 3, H, W)`.

    Returns:
        Lab version of the image with shape :math:`(*, 3, H, W)`.
        The L channel values are in the range 0..100. a and b are in the range -128..127.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = rgb_to_lab(input)  # 2x3x4x5
    """
    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(
            f"Input size must have a shape of (*, 3, H, W). Got {image.shape}"
        )

    xyz_im: torch.Tensor = linear_rgb_to_xyz(image)

    # normalize for D65 white point
    xyz_ref_white = torch.tensor(
        [0.95047, 1.0, 1.08883], device=xyz_im.device, dtype=xyz_im.dtype
    )[..., :, None, None]
    xyz_normalized = torch.div(xyz_im, xyz_ref_white)

    threshold = 0.008856
    power = torch.pow(xyz_normalized.clamp(min=threshold), 1 / 3.0)
    scale = 7.787 * xyz_normalized + 4.0 / 29.0
    xyz_int = torch.where(xyz_normalized > threshold, power, scale)

    x: torch.Tensor = xyz_int[..., 0, :, :]
    y: torch.Tensor = xyz_int[..., 1, :, :]
    z: torch.Tensor = xyz_int[..., 2, :, :]

    L: torch.Tensor = (116.0 * y) - 16.0
    a: torch.Tensor = 500.0 * (x - y)
    b: torch.Tensor = 200.0 * (y - z)

    out: torch.Tensor = torch.stack([L, a, b], dim=-3)

    return out


# Source: https://github.com/kornia/kornia/blob/master/kornia/color/lab.py
# Apache-2.0 license
def lab_to_linear_rgb(image: torch.Tensor, clip: bool = True) -> torch.Tensor:
    r"""Convert a Lab image to RGB.

    The L channel is assumed to be in the range of :math:`[0, 100]`.
    a and b channels are in the range of :math:`[-128, 127]`.

    Args:
        image: Lab image to be converted to RGB with shape :math:`(*, 3, H, W)`.
        clip: Whether to apply clipping to insure output RGB values in range :math:`[0, 1]`.

    Returns:
        Lab version of the image with shape :math:`(*, 3, H, W)`.
        The output RGB image are in the range of :math:`[0, 1]`.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = lab_to_rgb(input)  # 2x3x4x5
    """
    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(
            f"Input size must have a shape of (*, 3, H, W). Got {image.shape}"
        )

    L: torch.Tensor = image[..., 0, :, :]
    a: torch.Tensor = image[..., 1, :, :]
    b: torch.Tensor = image[..., 2, :, :]

    fy = (L + 16.0) / 116.0
    fx = (a / 500.0) + fy
    fz = fy - (b / 200.0)

    # if color data out of range: Z < 0
    fz = fz.clamp(min=0.0)

    fxyz = torch.stack([fx, fy, fz], dim=-3)

    # Convert from Lab to XYZ
    power = torch.pow(fxyz, 3.0)
    scale = (fxyz - 4.0 / 29.0) / 7.787
    xyz = torch.where(fxyz > 0.2068966, power, scale)

    # For D65 white point
    xyz_ref_white = torch.tensor(
        [0.95047, 1.0, 1.08883], device=xyz.device, dtype=xyz.dtype
    )[..., :, None, None]
    xyz_im = xyz * xyz_ref_white

    rgbs_im: torch.Tensor = xyz_to_linear_rgb(xyz_im)

    # https://github.com/richzhang/colorization-pytorch/blob/66a1cb2e5258f7c8f374f582acc8b1ef99c13c27/util/util.py#L107
    #     rgbs_im = torch.where(rgbs_im < 0, torch.zeros_like(rgbs_im), rgbs_im)

    # Clip to 0,1 https://www.w3.org/Graphics/Color/srgb
    if clip:
        rgbs_im = torch.clamp(rgbs_im, min=0.0, max=1.0)

    return rgbs_im


def rgb_to_lab(image: torch.Tensor) -> torch.Tensor:
    return linear_rgb_to_lab(rgb_to_linear_rgb(image))


def lab_to_rgb(image: torch.Tensor, clip: bool = True) -> torch.Tensor:
    return linear_rgb_to_rgb(lab_to_linear_rgb(image, clip=clip))


# Source: https://github.com/kornia/kornia/blob/master/kornia/color/xyz.py
# Apache-2.0 license
def linear_rgb_to_xyz(image: torch.Tensor) -> torch.Tensor:
    r"""Convert a RGB image to XYZ.

    .. image:: _static/img/rgb_to_xyz.png

    Args:
        image: RGB Image to be converted to XYZ with shape :math:`(*, 3, H, W)`.

    Returns:
         XYZ version of the image with shape :math:`(*, 3, H, W)`.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = rgb_to_xyz(input)  # 2x3x4x5
    """

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(
            f"Input size must have a shape of (*, 3, H, W). Got {image.shape}"
        )

    r: torch.Tensor = image[..., 0, :, :]
    g: torch.Tensor = image[..., 1, :, :]
    b: torch.Tensor = image[..., 2, :, :]

    x: torch.Tensor = 0.412453 * r + 0.357580 * g + 0.180423 * b
    y: torch.Tensor = 0.212671 * r + 0.715160 * g + 0.072169 * b
    z: torch.Tensor = 0.019334 * r + 0.119193 * g + 0.950227 * b

    out: torch.Tensor = torch.stack([x, y, z], -3)

    return out


# Source: https://github.com/kornia/kornia/blob/master/kornia/color/xyz.py
# Apache-2.0 license
def xyz_to_linear_rgb(image: torch.Tensor) -> torch.Tensor:
    r"""Convert a XYZ image to RGB.

    Args:
        image: XYZ Image to be converted to RGB with shape :math:`(*, 3, H, W)`.

    Returns:
        RGB version of the image with shape :math:`(*, 3, H, W)`.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = xyz_to_rgb(input)  # 2x3x4x5
    """

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(
            f"Input size must have a shape of (*, 3, H, W). Got {image.shape}"
        )

    x: torch.Tensor = image[..., 0, :, :]
    y: torch.Tensor = image[..., 1, :, :]
    z: torch.Tensor = image[..., 2, :, :]

    r: torch.Tensor = (
        3.2404813432005266 * x + -1.5371515162713185 * y + -0.4985363261688878 * z
    )
    g: torch.Tensor = (
        -0.9692549499965682 * x + 1.8759900014898907 * y + 0.0415559265582928 * z
    )
    b: torch.Tensor = (
        0.0556466391351772 * x + -0.2040413383665112 * y + 1.0573110696453443 * z
    )

    out: torch.Tensor = torch.stack([r, g, b], dim=-3)

    return out


def rgb_to_linear_rgb(image: torch.Tensor) -> torch.Tensor:
    return torch.pow(image, 2.2)


def linear_rgb_to_rgb(image: torch.Tensor) -> torch.Tensor:
    return torch.pow(image, 1 / 2.2)
