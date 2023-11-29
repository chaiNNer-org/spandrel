import functools
import math
import re
from collections import OrderedDict

from ...__helpers.model_descriptor import ImageModelDescriptor, StateDict
from .arch.RRDB import RRDBNet


def _new_to_old_arch(state: StateDict, state_map: dict, num_blocks: int):
    """Convert a new-arch model state dictionary to an old-arch dictionary."""
    if "params_ema" in state:
        state = state["params_ema"]

    if "conv_first.weight" not in state:
        # model is already old arch, this is a loose check, but should be sufficient
        return state

    # add nb to state keys
    for kind in ("weight", "bias"):
        state_map[f"model.1.sub.{num_blocks}.{kind}"] = state_map[
            f"model.1.sub./NB/.{kind}"
        ]
        del state_map[f"model.1.sub./NB/.{kind}"]

    old_state = OrderedDict()
    for old_key, new_keys in state_map.items():
        for new_key in new_keys:
            if r"\1" in old_key:
                for k, v in state.items():
                    sub = re.sub(new_key, old_key, k)
                    if sub != k:
                        old_state[sub] = v
            else:
                if new_key in state:
                    old_state[old_key] = state[new_key]

    # upconv layers
    max_upconv = 0
    for key in state.keys():
        match = re.match(r"(upconv|conv_up)(\d)\.(weight|bias)", key)
        if match is not None:
            _, key_num, key_type = match.groups()
            old_state[f"model.{int(key_num) * 3}.{key_type}"] = state[key]
            max_upconv = max(max_upconv, int(key_num) * 3)

    # final layers
    for key in state.keys():
        if key in ("HRconv.weight", "conv_hr.weight"):
            old_state[f"model.{max_upconv + 2}.weight"] = state[key]
        elif key in ("HRconv.bias", "conv_hr.bias"):
            old_state[f"model.{max_upconv + 2}.bias"] = state[key]
        elif key in ("conv_last.weight",):
            old_state[f"model.{max_upconv + 4}.weight"] = state[key]
        elif key in ("conv_last.bias",):
            old_state[f"model.{max_upconv + 4}.bias"] = state[key]

    # Sort by first numeric value of each layer
    def compare(item1: str, item2: str):
        parts1 = item1.split(".")
        parts2 = item2.split(".")
        int1 = int(parts1[1])
        int2 = int(parts2[1])
        return int1 - int2

    sorted_keys = sorted(old_state.keys(), key=functools.cmp_to_key(compare))

    # Rebuild the output dict in the right order
    out_dict = OrderedDict((k, old_state[k]) for k in sorted_keys)

    return out_dict


def _get_scale(state: StateDict, min_part: int = 6) -> int:
    n = 0
    for part in list(state):
        parts = part.split(".")[1:]
        if len(parts) == 2:
            part_num = int(parts[0])
            if part_num > min_part and parts[1] == "weight":
                n += 1
    return 2**n


def _get_num_blocks(state: StateDict, state_map: dict) -> int:
    nbs = []
    state_keys = state_map[r"model.1.sub.\1.RDB\2.conv\3.0.\4"] + (
        r"model\.\d+\.sub\.(\d+)\.RDB(\d+)\.conv(\d+)\.0\.(weight|bias)",
    )
    for state_key in state_keys:
        for k in state:
            m = re.search(state_key, k)
            if m:
                nbs.append(int(m.group(1)))
        if nbs:
            break
    return max(*nbs) + 1


def load(state_dict: StateDict) -> ImageModelDescriptor[RRDBNet]:
    state = state_dict
    model_arch = "ESRGAN"

    state_map = {
        # currently supports old, new, and newer RRDBNet arch models
        # ESRGAN, BSRGAN/RealSR, Real-ESRGAN
        "model.0.weight": ("conv_first.weight",),
        "model.0.bias": ("conv_first.bias",),
        "model.1.sub./NB/.weight": ("trunk_conv.weight", "conv_body.weight"),
        "model.1.sub./NB/.bias": ("trunk_conv.bias", "conv_body.bias"),
        r"model.1.sub.\1.RDB\2.conv\3.0.\4": (
            r"RRDB_trunk\.(\d+)\.RDB(\d)\.conv(\d+)\.(weight|bias)",
            r"body\.(\d+)\.rdb(\d)\.conv(\d+)\.(weight|bias)",
        ),
    }
    if "params_ema" in state:
        state = state["params_ema"]
        # model_arch = "RealESRGAN"
    num_blocks = _get_num_blocks(state, state_map)
    plus = any("conv1x1" in k for k in state.keys())
    if plus:
        model_arch = "ESRGAN+"

    state = _new_to_old_arch(state, state_map, num_blocks)

    highest_weight_num = max(int(re.search(r"model.(\d+)", k).group(1)) for k in state)  # type: ignore

    in_nc: int = state["model.0.weight"].shape[1]
    out_nc: int = state[f"model.{highest_weight_num}.bias"].shape[0]

    scale: int = _get_scale(state)
    num_filters: int = state["model.0.weight"].shape[0]

    c2x2 = False
    if state["model.0.weight"].shape[-2] == 2:
        c2x2 = True
        scale = round(math.sqrt(scale / 4))
        model_arch = "ESRGAN-2c2"

    # Detect if pixelunshuffle was used (Real-ESRGAN)
    if in_nc in (out_nc * 4, out_nc * 16) and out_nc in (
        in_nc / 4,
        in_nc / 16,
    ):
        shuffle_factor = int(math.sqrt(in_nc / out_nc))
    else:
        shuffle_factor = None

    model = RRDBNet(
        in_nc=in_nc,
        out_nc=out_nc,
        num_filters=num_filters,
        num_blocks=num_blocks,
        scale=scale,
        c2x2=c2x2,
        shuffle_factor=shuffle_factor,
    )
    tags = [
        f"{num_filters}nf",
        f"{num_blocks}nb",
    ]

    # Adjust these properties for calculations outside of the model
    if shuffle_factor:
        in_nc //= shuffle_factor**2
        scale //= shuffle_factor

    return ImageModelDescriptor(
        model,
        state,
        architecture=model_arch,
        purpose="Restoration" if scale == 1 else "SR",
        tags=tags,
        supports_half=True,
        supports_bfloat16=True,
        scale=scale,
        input_channels=in_nc,
        output_channels=out_nc,
    )
