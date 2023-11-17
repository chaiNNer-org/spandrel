from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse
from urllib.request import urlretrieve

from syrupy.filters import props  # type: ignore

MODEL_DIR = Path("./tests/models/")


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
