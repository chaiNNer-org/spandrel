# Spandrel NC

This library implements various PyTorch architectures with non-commercial licenses for [spandrel](https://github.com/chaiNNer-org/spandrel).

## Installation

You need to install both `spandrel` and `spandrel_nc`:

```shell
pip install spandrel spandrel_nc
```

## Basic usage

```python
from spandrel import MAIN_REGISTRY, ModelLoader
from spandrel_nc import NC_REGISTRY

# add nc and cl architectures before `ModelLoader` is used
MAIN_REGISTRY.add(NC_REGISTRY)

# load a model from disk
model = ModelLoader().load_from_file(r"path/to/model.pth")

... # use model
```

## Model Architecture Support

For a full list of all architectures implemented in this library, see the architectures [marked with "(nc)" here](https://github.com/chaiNNer-org/spandrel#model-architecture-support).

## License

This library is licensed under the MIT license but contains the source code of architectures with non-commercial licenses. Consequently, this library cannot be used in commercial products.
