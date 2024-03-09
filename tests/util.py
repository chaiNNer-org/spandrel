from __future__ import annotations

import hashlib
import logging
import os
import random
import re
import sys
import zipfile
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from inspect import getsource
from pathlib import Path
from typing import Any, Callable, Protocol, TypeVar, runtime_checkable
from urllib.parse import unquote, urlencode, urlparse

import cv2
import numpy as np
import requests
import safetensors.torch
import torch
from bs4 import BeautifulSoup, Tag
from syrupy.filters import props

from spandrel import (
    MAIN_REGISTRY,
    Architecture,
    ImageModelDescriptor,
    ModelDescriptor,
    ModelLoader,
)
from spandrel_extra_arches import EXTRA_REGISTRY

MAIN_REGISTRY.add(*EXTRA_REGISTRY)

MODEL_DIR = Path("./tests/models/")
ZIP_DIR = Path("./tests/zips/")
IMAGE_DIR = Path("./tests/images/")

IS_CI = os.environ.get("CI") == "true"

logger = logging.getLogger(__name__)


def get_url_file_name(url: str) -> str:
    return Path(unquote(urlparse(url).path)).name


def convert_google_drive_link(url: str) -> str:
    pattern = re.compile(
        r"^https://drive.google.com/file/d/([a-zA-Z0-9_\-]+)/view(?:\?.*)?$"
    )
    m = pattern.match(url)
    if not m:
        return url

    file_id = m.group(1)
    url = "https://drive.google.com/uc?export=download&confirm=1&id=" + file_id

    # confirm=1 doesn't work sometimes, so we check whether we get a file or a
    # website, and then parse the website to get the real download link
    response = requests.head(url, allow_redirects=True)
    response.raise_for_status()
    content_type = response.headers["Content-Type"]
    if content_type == "text/html; charset=utf-8":
        # download and parse the website
        response = requests.get(url)
        response.raise_for_status()

        soup = BeautifulSoup(response.text)
        form = soup.find("form")
        assert isinstance(form, Tag)
        base_url = form.attrs["action"]
        params: dict[str, str] = {}
        for i in form.find_all("input"):
            assert isinstance(i, Tag)
            if "name" in i.attrs and "value" in i.attrs:
                params[i.attrs["name"]] = i.attrs["value"]
        url = base_url + "?" + urlencode(params)

    return url


def download_file(url: str, filename: Path | str) -> None:
    filename = Path(filename)
    filename.parent.mkdir(exist_ok=True)
    url = convert_google_drive_link(url)
    logger.info("Downloading %s to %s", url, filename)
    torch.hub.download_url_to_file(url, str(filename), progress=not IS_CI)


def extract_file_from_zip(
    zip_path: Path | str,
    rel_model_path: str,
    filename: Path | str,
):
    filename = Path(filename)
    filename.parent.mkdir(exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        filename.write_bytes(zip_ref.read(rel_model_path))


def get_test_device() -> torch.device:
    return torch.device(os.environ.get("SPANDREL_TEST_DEVICE") or "cpu")


@dataclass
class ModelFile:
    name: str

    @property
    def path(self) -> Path:
        return MODEL_DIR / self.name

    def exists(self) -> bool:
        return self.path.exists()

    def load_model(self) -> ModelDescriptor:
        return ModelLoader().load_from_file(self.path)

    @staticmethod
    def from_url(url: str, name: str | None = None):
        file = ModelFile(name or get_url_file_name(url))

        if not file.exists():
            download_file(url, file.path)

        return file

    @staticmethod
    def from_url_zip(url: str, rel_model_path: str, name: str | None = None):
        file = ModelFile(name or Path(rel_model_path).name)

        if not file.exists():
            zip_path = ZIP_DIR / f"{hashlib.sha256(url.encode()).hexdigest()}.zip"
            if not zip_path.exists():
                download_file(url, zip_path)
            extract_file_from_zip(zip_path, rel_model_path, file.path)

        return file


disallowed_props = props("model", "state_dict", "device", "dtype")


@contextmanager
def expect_error(snapshot):
    try:
        yield None
        did_error = False
    except Exception as e:
        did_error = True
        assert e == snapshot

    if not did_error:
        raise AssertionError("Expected an error, but none was raised")


def read_image(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return image


def write_image(path: str | Path, image: np.ndarray):
    cv2.imwrite(str(path), image)


def image_to_tensor(img: np.ndarray) -> torch.Tensor:
    img = img.astype(np.float32) / 255.0
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    if img.shape[2] == 1:
        pass
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
    tensor = torch.from_numpy(img)
    return tensor.unsqueeze(0)


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    image = tensor.cpu().squeeze().numpy()
    image = np.transpose(image, (1, 2, 0))
    image = np.clip((image * 255.0).round(), 0, 255)
    image = image.astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image


def image_inference_tensor(
    model: ImageModelDescriptor, tensor: torch.Tensor
) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        return model(tensor)


def image_inference(model: ImageModelDescriptor, image: np.ndarray) -> np.ndarray:
    tensor = image_to_tensor(image).to(get_test_device())
    return tensor_to_image(image_inference_tensor(model, tensor))


def get_h_w_c(image: np.ndarray) -> tuple[int, int, int]:
    if len(image.shape) == 2:
        return image.shape[0], image.shape[1], 1
    return image.shape[0], image.shape[1], image.shape[2]


class TestImage(Enum):
    SR_8 = "8x8.png"
    SR_16 = "16x16.png"
    SR_32 = "32x32.png"
    SR_64 = "64x64.png"
    JPEG_15 = "jpeg-15.jpg"
    GRAY_EINSTEIN = "einstein.png"
    BLURRY_FACE = "blurry-face.jpg"


def assert_image_inference(
    model_file: ModelFile,
    model: ModelDescriptor,
    test_images: list[TestImage],
    tolerance: float = 1,
):
    assert isinstance(model, ImageModelDescriptor)

    test_images.sort(key=lambda image: image.value)

    update_mode = "--snapshot-update" in sys.argv

    outputs_dir = os.environ.get("SPANDREL_TEST_OUTPUTS_DIR") or "outputs"
    model.to(get_test_device())

    for test_image in test_images:
        path = IMAGE_DIR / "inputs" / test_image.value

        image = read_image(path)
        image_h, image_w, image_c = get_h_w_c(image)

        if model.input_channels == 1 and image_c == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image_c = 1

        assert (
            image_c == model.input_channels
        ), f"Expected the input image '{test_image.value}' to have {model.input_channels} channels, but it had {image_c} channels."

        try:
            output = image_inference(model, image)
        except Exception as e:
            raise AssertionError(f"Failed on {test_image.value}") from e
        output_h, output_w, output_c = get_h_w_c(output)

        assert (
            output_c == model.output_channels
        ), f"Expected the output of '{test_image.value}' to have {model.output_channels} channels, but it had {output_c} channels."
        assert (
            output_w == image_w * model.scale and output_h == image_h * model.scale
        ), f"Expected the input image '{test_image.value}' {image_w}x{image_h} to be scaled {model.scale}x, but the output was {output_w}x{output_h}."

        expected_path = (
            IMAGE_DIR / outputs_dir / path.stem / f"{model_file.path.stem}.png"
        )

        if update_mode and not expected_path.exists():
            expected_path.parent.mkdir(exist_ok=True, parents=True)
            write_image(expected_path, output)
            continue

        assert expected_path.exists(), f"Expected {expected_path} to exist."
        expected = read_image(expected_path)

        # Assert that the images are the same within a certain tolerance
        # The CI for some reason has a bit of FP precision loss compared to my local machine
        # Therefore, a tolerance of 1 is fine enough.
        close_enough = np.allclose(output, expected, atol=tolerance)
        if update_mode and not close_enough:
            write_image(expected_path, output)
            continue

        assert close_enough, f"Failed on {test_image.value}"


T = TypeVar("T", bound=torch.nn.Module)


def _get_diff(a: Mapping[str, object], b: Mapping[str, object]) -> str | None:
    lines: list[str] = []

    keys = list(set(a.keys()).union(b.keys()))
    keys.sort()

    is_different = False

    for key in keys:
        a_val = a.get(key, "<missing>")
        b_val = b.get(key, "<missing>")

        # make lists and tuples comparable
        if (
            type(a_val) != type(b_val)
            and isinstance(a_val, (list, tuple))
            and isinstance(b_val, (list, tuple))
        ):
            if isinstance(a_val, tuple):
                a_val = list(a_val)
            if isinstance(b_val, tuple):
                b_val = list(b_val)

        if a_val == b_val:
            lines.append(f"  {key}: {a_val}")
        else:
            lines.append(f"> {key}: {a_val} != {b_val}")
            is_different = True

    if not is_different:
        return None

    return "\n".join(lines)


def assert_loads_correctly(
    arch: Architecture[T],
    *models: Callable[[], T],
    check_safe_tensors: bool = True,
    ignore_parameters: set[str] | None = None,
):
    @runtime_checkable
    class WithHyperparameters(Protocol):
        hyperparameters: dict[str, Any]

    def assert_same(model_name: str, a: T, b: T) -> None:
        assert (
            type(b) == type(a)
        ), f"Expected {model_name} to be loaded correctly, but found a {type(b)} instead."

        assert isinstance(a, WithHyperparameters)
        assert isinstance(b, WithHyperparameters)

        a_hp = {**a.hyperparameters}
        b_hp = {**b.hyperparameters}

        if ignore_parameters is not None:
            for param in ignore_parameters:
                a_hp.pop(param, None)
                b_hp.pop(param, None)

        diff = _get_diff(a_hp, b_hp)
        if diff:
            raise AssertionError(f"Failed condition for {model_name}. Keys:\n\n{diff}")

    for model_fn in models:
        model_name = getsource(model_fn)
        try:
            model = model_fn()
        except Exception as e:
            raise AssertionError(f"Failed to create model: {model_name}") from e

        try:
            state_dict = model.state_dict()
            loaded = arch.load(state_dict)
        except Exception as e:
            raise AssertionError(f"Failed to load: {model_name}") from e

        assert_same(model_name, model, loaded.model)

        if check_safe_tensors:
            try:
                b = safetensors.torch.save(model.state_dict())
            except Exception as e:
                raise AssertionError(
                    f"Failed to save as safetensors: {model_name}"
                ) from e

            try:
                sf_loaded = arch.load(safetensors.torch.load(b))
            except Exception as e:
                raise AssertionError(
                    f"Failed to load from safetensors: {model_name}"
                ) from e

            assert_same(model_name, model, sf_loaded.model)


def seed_rngs(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def assert_size_requirements(
    model: ModelDescriptor,
    max_size: int = 64,
    max_candidates: int = 8,
) -> None:
    assert isinstance(model, ImageModelDescriptor)

    device = get_test_device()

    def test_size(width: int, height: int) -> None:
        try:
            input_tensor = torch.rand(1, model.input_channels, height, width)
            model.to(device).eval()

            with torch.no_grad():
                output_tensor = model(input_tensor.to(device))

            assert output_tensor.shape[1] == model.output_channels, "Incorrect channels"
            assert output_tensor.shape[2] == height * model.scale, "Incorrect height"
            assert output_tensor.shape[3] == width * model.scale, "Incorrect width"
        except Exception as e:
            raise AssertionError(
                f"Failed size requirement test for {width=} {height=}"
            ) from e

    req = model.size_requirements
    candidates: list[int] = []

    # fill candidates
    current = req.minimum // req.multiple_of * req.multiple_of
    while current <= max_size and len(candidates) < max_candidates:
        if req.check(current, current) and current > 0:
            candidates.append(current)
        current += req.multiple_of

    # fast path for non-square models
    failed_non_square = None
    if not req.square:
        # test 2 candidates at once by using one as width and the other as height

        try:
            # make sure the list is even
            if len(candidates) % 2 == 1:
                candidates.append(candidates[-1])

            for i in range(0, len(candidates), 2):
                test_size(candidates[i], candidates[i + 1])
            return
        except Exception as e:  # noqa: E722
            # fall through and let the below code handle it
            failed_non_square = e

    valid: list[int] = []
    invalid: list[tuple[int, Exception]] = []
    for width in candidates:
        try:
            test_size(width, width)
            valid.append(width)
        except Exception as e:
            invalid.append((width, e))

    if len(invalid) > 0:
        raise AssertionError(
            f"Failed size requirement test.\n"
            f"Valid sizes: {valid}\n"
            f"Invalid sizes: {[size for size, _ in invalid]}"
        ) from invalid[0][1]

    if failed_non_square is not None:
        raise AssertionError(
            "Failed size requirement test for non-square models"
        ) from failed_non_square
