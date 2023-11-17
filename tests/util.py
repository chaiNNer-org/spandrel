from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from urllib.parse import unquote, urlparse
from urllib.request import urlretrieve

import cv2
import numpy as np
import torch
from syrupy.filters import props  # type: ignore

MODEL_DIR = Path("./tests/models/")
IMAGE_DIR = Path("./tests/images/")


def download_model(url: str, name: str | None = None) -> str:
    filename = name or Path(unquote(urlparse(url).path)).name
    print(f"Downloading {filename}...")
    MODEL_DIR.mkdir(exist_ok=True)
    path, _ = urlretrieve(url, filename=MODEL_DIR / filename)
    return path


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

    @staticmethod
    def from_url(url: str):
        name = Path(unquote(urlparse(url).path)).name
        return ModelFile(name).download(url)


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


def simple_image_inference(
    model: torch.nn.Module, tensor: torch.Tensor
) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        return model(tensor)


def simple_infer_from_image_path(
    model: torch.nn.Module, path: str | Path
) -> np.ndarray:
    tensor = image_to_tensor(read_image(path))
    return tensor_to_image(simple_image_inference(model, tensor))


class ImageTestNames(Enum):
    SR_16 = "16x16.png"
    SR_32 = "32x32.png"
    SR_64 = "64x64.png"


def compare_images_to_results(
    model_name: str, model: torch.nn.Module, test_images: list[ImageTestNames]
) -> bool:
    image_paths = sorted((IMAGE_DIR / "inputs").glob("*.png"))
    test_image_values = [image.value for image in test_images]
    image_paths = [path for path in image_paths if path.name in test_image_values]
    for path in image_paths:
        print(f"Comparing {path.name}...")
        result = simple_infer_from_image_path(model, path)
        image_name = path.name
        basename, _ = os.path.splitext(image_name)
        base_model_name, _ = os.path.splitext(model_name)
        gt_path = IMAGE_DIR / "outputs" / basename / f"{base_model_name}.png"
        gt = read_image(gt_path)

        # Assert that the images are the same within a certain tolerance
        # The CI for some reason has a bit of FP precision loss compared to my local machine
        # Therefore, a tolerance of 1 is fine enough.
        if not np.allclose(result, gt, atol=1):
            return False
    return True
