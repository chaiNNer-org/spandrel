from __future__ import annotations

import os
import re
import sys
import tempfile
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from inspect import getsource
from pathlib import Path
from typing import Any, Callable, TypeVar
from urllib.parse import unquote, urlparse
from urllib.request import urlretrieve

import cv2
import numpy as np
import torch
from syrupy.filters import props

from spandrel import ModelBase, ModelDescriptor, ModelLoader, StateDict

MODEL_DIR = Path("./tests/models/")
IMAGE_DIR = Path("./tests/images/")


def get_url_file_name(url: str) -> str:
    return Path(unquote(urlparse(url).path)).name


def convert_google_drive_link(url: str) -> str:
    pattern = re.compile(
        r"^https://drive.google.com/file/d/([a-zA-Z0-9\-]+)/view(?:\?.*)?$"
    )
    m = pattern.match(url)
    if not m:
        return url
    file_id = m.group(1)
    return "https://drive.google.com/uc?export=download&confirm=1&id=" + file_id


def download_model(url: str, name: str | None = None) -> str:
    filename = name or get_url_file_name(url)
    print(f"Downloading {filename}...")
    MODEL_DIR.mkdir(exist_ok=True)

    url = convert_google_drive_link(url)

    path, _ = urlretrieve(url, filename=MODEL_DIR / filename)
    return path


def extract_zip(path: str, rel_model_path: Path | str, name: str):
    if not zipfile.is_zipfile(path):
        print(f"Skipping {path} because it is not a zip file.")
        return

    if (MODEL_DIR / name).exists():
        print(f"Skipping {path} because {name} already exists.")
        return

    with zipfile.ZipFile(path, "r") as zip_ref:
        with tempfile.TemporaryDirectory() as tmpdirname:
            zip_ref.extractall(tmpdirname)
            model_path = Path(tmpdirname) / rel_model_path
            assert model_path.exists(), f"Expected {model_path} to exist."
            model_path.rename(MODEL_DIR / name)
            return model_path


@dataclass
class ModelFile:
    name: str

    @property
    def path(self) -> Path:
        return MODEL_DIR / self.name

    @property
    def exists(self) -> bool:
        return self.path.exists()

    def download(self, url: str):
        if not self.exists:
            download_model(url, name=self.name)
        return self

    def load_model(self) -> ModelDescriptor:
        return ModelLoader().load_from_file(self.path)

    @staticmethod
    def from_url(url: str, name: str | None = None):
        name = name or get_url_file_name(url)
        return ModelFile(name).download(url)

    @staticmethod
    def from_url_zip(url: str, rel_model_path: Path | str, name: str | None = None):
        name = os.path.basename(rel_model_path) if name is None else name
        if (MODEL_DIR / name).exists():
            return ModelFile(name)
        path = download_model(url, "temp.zip")
        print(f"Extracting {path}...")
        extract_zip(path, rel_model_path or name, name)
        os.remove(path)
        return ModelFile(name or get_url_file_name(url))


disallowed_props = props("model", "state_dict")


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
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
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
    model: torch.nn.Module, tensor: torch.Tensor
) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        return model(tensor)


def image_inference(model: torch.nn.Module, image: np.ndarray) -> np.ndarray:
    return tensor_to_image(image_inference_tensor(model, image_to_tensor(image)))


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


def assert_image_inference(
    model_file: ModelFile,
    model: ModelDescriptor,
    test_images: list[TestImage],
):
    test_images.sort(key=lambda image: image.value)

    update_mode = "--snapshot-update" in sys.argv

    for test_image in test_images:
        path = IMAGE_DIR / "inputs" / test_image.value

        image = read_image(path)
        image_h, image_w, image_c = get_h_w_c(image)

        assert (
            image_c == model.input_channels
        ), f"Expected the input image '{test_image.value}' to have {model.input_channels} channels, but it had {image_c} channels."

        try:
            output = image_inference(model.model, image)
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
            IMAGE_DIR / "outputs" / path.stem / f"{model_file.path.stem}.png"
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
        close_enough = np.allclose(output, expected, atol=1)
        if update_mode and not close_enough:
            write_image(expected_path, output)
            continue

        assert close_enough, f"Failed on {test_image.value}"


T = TypeVar("T", bound=torch.nn.Module)


def _get_different_keys(a: Any, b: Any, keys: list[str]) -> str:
    lines: list[str] = []

    keys = list(set(dir(a)).intersection(keys))
    keys.sort()

    for key in keys:
        a_val = getattr(a, key)
        b_val = getattr(b, key)
        if a_val == b_val:
            lines.append(f"{key}: {a_val}")
        else:
            lines.append(f"{key}: {a_val} != {b_val}")

    return "\n".join(lines)


def _get_compare_keys(condition: Callable) -> list[str]:
    pattern = re.compile(r"a\.(\w+)")
    return [m.group(1) for m in pattern.finditer(getsource(condition))]


def assert_loads_correctly(
    load: Callable[[StateDict], ModelBase[T]],
    *models: Callable[[], T],
    condition: Callable[[T, T], bool] = lambda _a, _b: True,
):
    for model_fn in models:
        model_name = getsource(model_fn)
        try:
            model = model_fn()
        except Exception as e:
            raise AssertionError(f"Failed to create model: {model_name}") from e

        try:
            state_dict = model.state_dict()
            loaded = load(state_dict)
        except Exception as e:
            raise AssertionError(f"Failed to load: {model_name}") from e

        assert (
            type(loaded.model) == type(model)
        ), f"Expected {model_name} to be loaded correctly, but found a {type(loaded.model)} instead."

        assert condition(model, loaded.model), (
            f"Failed condition for {model_name}."
            f" Keys:\n\n{_get_different_keys(model,loaded.model, _get_compare_keys(condition))}"
        )
