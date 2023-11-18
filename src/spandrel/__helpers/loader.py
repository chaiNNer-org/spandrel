from __future__ import annotations

import os
from pathlib import Path

import torch
from safetensors.torch import load_file

from .main_registry import MAIN_REGISTRY
from .model_descriptor import ModelDescriptor, StateDict
from .registry import ArchRegistry
from .unpickler import RestrictedUnpickle


class ModelLoader:
    """Class for automatically loading a pth file into any architecture"""

    def __init__(
        self,
        device: torch.device | None = None,
        registry: ArchRegistry = MAIN_REGISTRY,
    ):
        self.device: torch.device = device or torch.device("cpu")
        self.registry: ArchRegistry = registry
        """
        The architecture registry to use for loading models.

        *Note:* Unless initialized with a custom registry, this is the global main registry (`MAIN_REGISTRY`).
        Modifying this registry will affect all `ModelLoader` instances without a custom registry.
        """

    def load_from_file(self, path: str | Path) -> ModelDescriptor:
        """
        Load a model from the given file path.

        Throws a `ValueError` if the file extension is not supported.
        Throws an `UnsupportedModelError` if the model architecture is not supported.
        """

        state_dict = self.load_state_dict_from_file(path)
        return self.load_from_state_dict(state_dict)

    def load_state_dict_from_file(self, path: str | Path) -> StateDict:
        """
        Load the state dict of a model from the given file path.

        State dicts are typically only useful to pass them into the `load`
        function of a specific architecture.

        Throws a `ValueError` if the file extension is not supported.
        """

        extension = os.path.splitext(path)[1].lower()

        if extension == ".pt":
            return self._load_torchscript(path)
        elif extension == ".pth":
            return self._load_pth(path)
        elif extension == ".ckpt":
            return self._load_ckpt(path)
        elif extension == ".safetensors":
            return self._load_safetensors(path)
        else:
            raise ValueError(
                f"Unsupported model file extension {extension}. Please try a supported model type."
            )

    def load_from_state_dict(self, state_dict: StateDict) -> ModelDescriptor:
        """
        Load a model from the given state dict.

        Throws an `UnsupportedModelError` if the model architecture is not supported.
        """

        return self.registry.load(state_dict).to(self.device)

    def _load_pth(self, path: str | Path) -> StateDict:
        return torch.load(
            path,
            map_location=self.device,
            pickle_module=RestrictedUnpickle,  # type: ignore
        )

    def _load_torchscript(self, path: str | Path) -> StateDict:
        return torch.jit.load(  # type: ignore
            path, map_location=self.device
        ).state_dict()

    def _load_safetensors(self, path: str | Path) -> StateDict:
        return load_file(path, device=str(self.device))

    def _load_ckpt(self, path: str | Path) -> StateDict:
        checkpoint = torch.load(
            path,
            map_location=self.device,
            pickle_module=RestrictedUnpickle,  # type: ignore
        )
        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        state_dict = {}
        for i, j in checkpoint.items():
            if "netG." in i:
                key = i.replace("netG.", "")
                state_dict[key] = j
            elif "module." in i:
                key = i.replace("module.", "")
                state_dict[key] = j
        return state_dict
