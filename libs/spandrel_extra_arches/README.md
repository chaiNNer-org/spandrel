# Spandrel extra architectures

[![PyPI package](https://img.shields.io/badge/pip%20install-spandrel_extra_arches-brightgreen)](https://pypi.org/project/spandrel_extra_arches/)
[![version number](https://img.shields.io/pypi/v/spandrel_extra_arches?color=green&label=version)](https://github.com/chaiNNer-org/spandrel/releases)
[![PyPi Downloads](https://img.shields.io/pypi/dw/spandrel_extra_arches)](https://pypi.org/project/spandrel_extra_arches/#files)
[![Python Version](https://img.shields.io/pypi/pyversions/spandrel)](https://pypi.org/project/spandrel/#files:~:text=Requires%3A%20Python%20%3C3.12%2C%20%3E%3D3.8)

This library implements various PyTorch architectures with restrictive licenses for [spandrel](https://github.com/chaiNNer-org/spandrel).

If you are working on a private project or non-commercial open-source project, you are free to use this library. If you are working on a commercial or closed-source project, you may need to review the licenses of the individual architectures before using this library.

## Installation

You need to install both `spandrel` and `spandrel_extra_arches`:

```shell
pip install spandrel spandrel_extra_arches
```

## Basic usage

```python
import spandrel
import spandrel_extra_arches

# add extra architectures before `ModelLoader` is used
spandrel_extra_arches.install()

# load a model from disk
model = spandrel.ModelLoader().load_from_file(r"path/to/model.pth")

... # use model
```

## Model Architecture Support

For a full list of all architectures implemented in this library, see [the architectures marked with "(+)" here](https://github.com/chaiNNer-org/spandrel#model-architecture-support).

## License

This library is licensed under the MIT license but contains the source code of architectures with non-commercial and copyleft licenses. You may need to review the licenses of the individual architectures before using this library.
