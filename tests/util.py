from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import random
import re
import sys
import zipfile
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from inspect import getsource
from pathlib import Path
from typing import Any, Callable, Protocol, TypeVar, runtime_checkable
from urllib.parse import unquote, urlencode, urlparse

import cv2
import numpy as np
import pytest
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
    SizeRequirements,
    UnsupportedModelError,
)
from spandrel.util import KeyCondition
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

    def load_model(self, expected_arch: Architecture | None = None) -> ModelDescriptor:
        loader = ModelLoader()
        state_dict = loader.load_state_dict_from_file(self.path)
        try:
            model = loader.load_from_state_dict(state_dict)
            if expected_arch is not None and model.architecture.id != expected_arch.id:
                raise AssertionError(
                    f"Expected architecture {expected_arch.id}, but got {model.architecture.id}"
                )
            return model
        except UnsupportedModelError as e:
            if expected_arch is not None:
                if expected_arch.id not in loader.registry:
                    raise AssertionError(
                        f"Expected architecture {expected_arch.id} to be in the registry"
                    ) from e

                cond = expected_arch._detect  # type: ignore
                if isinstance(cond, KeyCondition):
                    kind = cond._kind  # type: ignore
                    keys = cond._keys  # type: ignore
                    if kind == "all":
                        missing: set[str] = set()
                        for key in keys:
                            if isinstance(key, str) and key not in state_dict:
                                missing.add(key)
                        if len(missing) > 0:
                            raise AssertionError(f"Missing keys: {missing}") from e
            raise

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
    HAZE = "haze.jpg"


def assert_image_inference(
    model_file: ModelFile,
    model: ModelDescriptor,
    test_images: list[TestImage],
    max_single_pixel_error: float = 1,
    max_mean_error: float = 0,
):
    # it doesn't make sense to have a max_single_pixel_error less than max_average_error
    max_single_pixel_error = max(max_single_pixel_error, max_mean_error)

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
        close_enough = np.allclose(output, expected, atol=max_single_pixel_error)
        if close_enough:
            # all pixels are close enough, so this is a pass
            continue

        error = cv2.absdiff(
            output.astype(np.int32),
            expected.astype(np.int32),
        ).astype(np.int32)
        mean_error = np.mean(error)

        if mean_error <= max_mean_error:
            # mean error is close enough, so this is a pass
            continue

        if update_mode:
            # update the snapshot
            write_image(expected_path, output)
            continue

        # prepare a useful error message
        error_max = int(np.max(error))
        error_dist = "Error distribution:"
        for i in range(error_max + 1):
            error_dist += f"\n  {i}: {np.sum(error == i)}"

        raise AssertionError(
            f"Failed on {test_image.value}."
            f"\nError mean: {mean_error}"
            f"\nError max: {np.max(error)}"
            f"\nError min: {np.min(error)}"
            f"\n{error_dist}"
        )


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

    def get_candidates(max_size: int, max_candidates: int) -> list[int]:
        candidates: list[int] = []

        # fill candidates
        current = req.minimum // req.multiple_of * req.multiple_of
        while current <= max_size and len(candidates) < max_candidates:
            if req.check(current, current) and current > 0:
                candidates.append(current)
            current += req.multiple_of

        return candidates

    candidates = get_candidates(max_size, max_candidates)

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

    # collect some candidates, just we have some data to work with
    if max(candidates) < 32:
        candidates = get_candidates(max_size=32, max_candidates=32)

    valid: list[int] = []
    invalid: list[tuple[int, Exception]] = []
    for width in candidates:
        try:
            test_size(width, width)
            valid.append(width)
        except Exception as e:
            invalid.append((width, e))

    def guess_size_requirements() -> SizeRequirements | None:
        if len(valid) < 2:
            # not enough data
            return None

        square: bool
        try:
            test_size(valid[0], valid[1])
            square = False
        except:  # noqa: E722
            square = True

        min_valid = min(valid)
        max_valid = max(valid)

        candidate_multiples: set[int] = {
            *range(1, 16),
            *(2**i for i in range(math.floor(math.log2(max_valid)) + 1)),
        }
        for multiple in sorted(candidate_multiples):
            guess = SizeRequirements(
                minimum=min_valid,
                multiple_of=multiple,
                square=square,
            )
            # the condition for success is NOT that all valid sizes pass,
            # but that all invalid sizes fail and any valid sizes pass
            invalid_fail = all(not guess.check(size, size) for size, _ in invalid)
            valid_pass = any(guess.check(size, size) for size in valid)
            if invalid_fail and valid_pass:
                return guess

        return None

    if len(invalid) > 0:
        raise AssertionError(
            f"Failed size requirement test.\n"
            f"Valid sizes: {valid}\n"
            f"Invalid sizes: {[size for size, _ in invalid]}\n\n"
            f"Based on the above data, the following size requirement is suggested:\n{guess_size_requirements() or 'Insufficient data to guess'}\n"
        ) from invalid[0][1]

    if failed_non_square is not None:
        raise AssertionError(
            "Failed size requirement test for non-square models"
        ) from failed_non_square


@lru_cache
def _get_changed_files() -> list[str] | None:
    repository = os.getenv("GITHUB_REPOSITORY") or "chaiNNer-org/spandrel"
    pull_request_ref = os.getenv("GITHUB_REF")

    if not repository or not pull_request_ref:
        logger.warn("Missing required environment variables.")
        return None

    # Extract pull request number from GITHUB_REF
    pull_request_number = pull_request_ref.split("/")[-2]
    logger.info(f"Checking for changes in PR {pull_request_number}")

    try:
        response = requests.get(
            f"https://api.github.com/repos/{repository}/pulls/{pull_request_number}/files"
        )
        response.raise_for_status()

        return [file["filename"] for file in json.loads(response.text)]
    except Exception as e:
        print(f"Error making request: {e}")
        return None


def _did_change(arch_name: str) -> bool:
    changed = _get_changed_files()
    if changed is None:
        # something went wrong, so we'll conservatively assume it changed
        return True

    if any(f"architectures/{arch_name.lower()}/" in file.lower() for file in changed):
        # the architecture was changed
        return True

    if any(f"tests/test_{arch_name}.py" in file for file in changed):
        # the test itself was changed
        return True

    important_files = [
        r"/spandrel/__helpers/(?!main_registry\.py)",
        r"/spandrel/util/",
        r"tests/util.py",
        r"pyproject.toml",
        r"requirements-dev.txt",
    ]
    pattern = re.compile("|".join(important_files))
    if any(pattern.match(file) for file in changed):
        # some important files were changed
        return True

    return False


def skip_if_unchanged(file: str):
    if not IS_CI or os.getenv("GITHUB_EVENT_NAME") != "pull_request":
        # only skip tests on pull requests CI
        return

    match = re.search(re.compile(r"\btest_(\w+)\.py$"), file)
    if not match:
        # not a test file
        return
    arch_name = match.group(1)

    if _did_change(arch_name):
        # test changed, so we have to run it
        return

    pytest.skip("No changes detected in file: " + file, allow_module_level=True)
