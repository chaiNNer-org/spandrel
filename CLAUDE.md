# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Spandrel is a library for loading and running pre-trained PyTorch models. It automatically detects model architecture and hyperparameters from state dicts and provides a unified interface. Supports 50+ image super-resolution and restoration architectures.

## Common Commands

### Testing
```bash
pytest tests                                    # Run all tests
pytest tests/test_ESRGAN.py                     # Run tests for one architecture
pytest tests/test_ESRGAN.py --snapshot-update   # Run tests and update snapshots
```

### Linting & Formatting
```bash
ruff check libs tests --fix --unsafe-fixes      # Lint with auto-fix
ruff format libs tests                          # Format code
pyright libs tests                              # Type checking
```

### Development Scripts
```bash
python scripts/dump_dummy.py                    # Dump dummy model state dict to dump.yml
python scripts/dump_state_dict.py /path/to.pth  # Dump actual model state dict to dump.yml
```

### Setup
```bash
pip install -r requirements-dev.txt
pip install -e libs/spandrel -e libs/spandrel_extra_arches
```

## Architecture

### Dual Package Structure
- `libs/spandrel/` — Main library (permissively licensed architectures)
- `libs/spandrel_extra_arches/` — Extra architectures with non-permissive licenses (CodeFormer, MAT, etc.)

### Core Components (`libs/spandrel/spandrel/__helpers/`)
- **`main_registry.py`** — Central `MAIN_REGISTRY` containing all architecture registrations. Detection order matters (insertion order).
- **`loader.py`** — `ModelLoader` class, entry point for loading `.pth`/`.safetensors`/`.ckpt`/`.pt` files
- **`model_descriptor.py`** — `ImageModelDescriptor` and `MaskedImageModelDescriptor` wrappers with metadata (scale, purpose, channels, etc.)
- **`registry.py`** — `ArchRegistry` and `ArchSupport` classes
- **`canonicalize.py`** — State dict normalization for variant formats

### Architecture Pattern (`libs/spandrel/spandrel/architectures/<ARCH>/`)
Each architecture directory contains:
- **`__init__.py`** — `<Name>Arch(Architecture)` class with `detect` (KeyCondition-based) and `load` (state dict → ModelDescriptor) methods
- **`__arch/`** — Original upstream model code (minimally modified) + LICENSE file

Architecture code in `__arch/` is exempt from strict linting and type checking.

### Key Utilities (`spandrel/util/`)
- **`KeyCondition`** — Fluent API for state dict key detection (`has_all`, `has_any`, composable)
- **`get_seq_len(state_dict, key)`** — Detect sequence length from numbered keys (e.g., `body.0`, `body.1`)
- **`get_pixelshuffle_params()`** — Detect upscale factor from pixelshuffle layers
- **`get_scale_and_output_channels()`** — Derive scale from channel multiplier

### Testing (`tests/`)
- One `test_<ARCH>.py` per architecture
- **`tests/util.py`** — Key helpers:
  - `assert_loads_correctly(arch, *lambdas, condition)` — Tests parameter detection with dummy models (no pretrained weights needed)
  - `assert_image_inference(file, model, test_images)` — Tests inference with snapshot comparison
  - `ModelFile.from_url(url)` — Downloads and caches models in `tests/models/`
- Snapshots use the **syrupy** library, stored in `tests/__snapshots__/`
- `skip_if_unchanged(__file__)` — Skips tests when architecture files haven't changed

## Adding New Architectures

See `CONTRIBUTING.md` for the full guide. Summary:

1. Copy model code into `architectures/<ARCH>/__arch/` with LICENSE
2. Create `__init__.py` with an `Architecture` subclass implementing `detect` and `load`
3. Use `scripts/dump_dummy.py` to understand state dict structure and derive parameter detection formulas
4. Register in `__helpers/main_registry.py`
5. Write tests and run `pytest tests/test_<ARCH>.py --snapshot-update`

## Linting Rules

- **Ruff**: Line length 88. Architecture code (`__arch/`) exempt from naming and annotation rules. Test code exempt from naming and annotation rules.
- **PyRight**: Strict mode for library code. Architecture directories are ignored (type checking disabled).
- Allowed external deps in architecture code: `torch`, `torchvision`, `numpy`, `einops`. Use vendored `timm` from `__arch_helpers/timm/` instead of the `timm` package.
