# Spandrel NC CL

This library implements various AI architecture with non-commercial and copyleft licenses for [spandrel](https://github.com/chaiNNer-org/spandrel).

> NOTE: This library contains all architectures implemented by `spandrel_nc`. Consequently, you do *not* need to install `spandrel_nc` if you install `spandrel_nc_cl`.

## Installation

You need to install both `spandrel` and `spandrel_nc_cl`:

```shell
pip install spandrel spandrel_nc_cl
```

## Basic usage

```python
from spandrel import MAIN_REGISTRY, ModelLoader
from spandrel_nc_cl import NC_CL_REGISTRY

# add nc and cl architectures before `ModelLoader` is used
MAIN_REGISTRY.add(NC_CL_REGISTRY)

# load a model from disk
model = ModelLoader().load_from_file(r"path/to/model.pth")

... # use model
```

## Model Architecture Support

For a full list of all architectures implemented in this library, see the architectures [marked with "(nc cl)" here](https://github.com/chaiNNer-org/spandrel#model-architecture-support).

## License

This library is licensed under the MIT license but contains the source code of architectures with non-commercial and copyleft licenses. Consequently, this library cannot be used in commercial and/or closed-source products.
