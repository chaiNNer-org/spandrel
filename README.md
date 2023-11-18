# Spandrel

[![PyPI package](https://img.shields.io/badge/pip%20install-spandrel-brightgreen)](https://pypi.org/project/spandrel/)
[![version number](https://img.shields.io/pypi/v/spandrel?color=green&label=version)](https://github.com/chaiNNer-org/spandrel/releases)
![PyPi Downloads](https://img.shields.io/pypi/dw/spandrel)
![PyPi Wheel](https://img.shields.io/pypi/wheel/spandrel)
![Python Version](https://img.shields.io/pypi/pyversions/spandrel)

[![Actions Status](https://github.com/chaiNNer-org/spandrel/workflows/Test/badge.svg)](https://github.com/chaiNNer-org/spandrel/actions)
[![License](https://img.shields.io/github/license/chaiNNer-org/spandrel)](https://github.com/chaiNNer-org/spandrel/blob/main/LICENSE)
[![Contributors](https://img.shields.io/github/contributors/chaiNNer-org/spandrel)](https://github.com/chaiNNer-org/spandrel/graphs/contributors)

This package ports [chaiNNer](https://github.com/chaiNNer-org/chaiNNer)'s PyTorch architecture support and model loading functionality into its own package, and wraps it into an easy-to-use API.

After seeing many projects extract out chaiNNer's model support into their own projects, I decided it was probably worth the effort of creating a PyPi package that those developers could use instead.

Slightly selfishly, I'm also hoping this will encourage the community to help add support for more models, so I don't have to do it myself. This will ultimately benefit everyone.

This package does not yet have easy inference code for these model types, but porting that code is planned as well.

## Installation

Spandrel is available through pip and can be installed via a simple pip install command:

```shell
pip install spandrel
```

## Usage

**This package is still in early stages of development, and is subject to change at any time.**

To use this package for automatic architecture loading, simply use the ModelLoader class like so:

```python
from spandrel import ModelLoader
import torch

# Initialize the ModelLoader class with an optional preferred torch.device. Defaults to cpu.
model_loader = ModelLoader(torch.device("cuda:0"))

# Load the model from the given path
loaded_model = model_loader.load_from_file(r"/path/to/your/model.pth")
```

And that's it. The model gets loaded into a helper class with various helpful bits of information, as well as the actual model information.

```py
# The model itself (a torch.nn.Module loaded with the weights)
loaded_model.model

# The state dict of the model (the weights)
loaded_model.state_dict

# The architecture of the model (e.g. "ESRGAN")
loaded_model.architecture

# A list of tags for the model, usually describing the size (e.g. ["64nf", "large"])
loaded_model.tags

# A boolean indicating whether the model supports half precision (fp16)
loaded_model.supports_half

# A boolean indicating whether the model supports bfloat16 precision
loaded_model.supports_bfloat16

# The scale of the model (e.g. 4)
loaded_model.scale

# The number of input channels of the model (e.g. 3)
loaded_model.input_channels

# The number of output channels of the model (e.g. 3)
loaded_model.output_channels

# A SizeRequirements object describing the image size requirements of the model
# i.e the minimum size, the multiple of size, and whether the model requires a square input
loaded_model.size_requirements
```

You can also just use this helper class for inference the same way you would with the `model` directly, so for example you could do `result = loaded_model(img)` and it will automatically call the forward method of the model. It also supports moving it to other devices, so you can call `.to` on it just like you would the direct model.

## Model Architecture Support

Spandrel currently supports a limited amount of neural network architectures. It can auto-detect these architectures just from their files alone.

This has only been tested with the models that are linked here, and any unofficial variants (especially if changes are made to their architectures) are not guaranteed to work.

### Pytorch

#### Single Image Super Resolution

- [ESRGAN](https://github.com/xinntao/ESRGAN) (RRDBNet)
  - This includes regular [ESRGAN](https://github.com/xinntao/ESRGAN), [ESRGAN+](https://github.com/ncarraz/ESRGANplus), "new-arch ESRGAN" ([RealSR](https://github.com/jixiaozhong/RealSR), [BSRGAN](https://github.com/cszn/BSRGAN)), [SPSR](https://github.com/Maclory/SPSR), and [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)
  - Models: [Community ESRGAN](https://openmodeldb.info) | [ESRGAN+](https://drive.google.com/drive/folders/1lNky9afqEP-qdxrAwDFPJ1g0ui4x7Sin) | [BSRGAN](https://github.com/cszn/BSRGAN/tree/main/model_zoo) | [RealSR](https://github.com/jixiaozhong/RealSR#pre-trained-models) | [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN/blob/master/docs/model_zoo.md)
- [Real-ESRGAN Compact](https://github.com/xinntao/Real-ESRGAN) (SRVGGNet) | [Models](https://github.com/xinntao/Real-ESRGAN/blob/master/docs/model_zoo.md)
- [Swift-SRGAN](https://github.com/Koushik0901/Swift-SRGAN) | [Models](https://github.com/Koushik0901/Swift-SRGAN/releases/tag/v0.1)
- [SwinIR](https://github.com/JingyunLiang/SwinIR) | [Models](https://github.com/JingyunLiang/SwinIR/releases/tag/v0.0)
- [Swin2SR](https://github.com/mv-lab/swin2sr) | [Models](https://github.com/mv-lab/swin2sr/releases/tag/v0.0.1)
- [HAT](https://github.com/XPixelGroup/HAT) | [Models](https://drive.google.com/drive/folders/1HpmReFfoUqUbnAOQ7rvOeNU3uf_m69w0)
- [Omni-SR](https://github.com/Francis0625/Omni-SR) | [Models](https://github.com/Francis0625/Omni-SR#preparation)
- [SRFormer](https://github.com/HVision-NKU/SRFormer) | [Models](https://github.com/HVision-NKU/SRFormer#pretrain-models)
- [DAT](https://github.com/zhengchen1999/DAT) | [Models](https://github.com/zhengchen1999/DAT#testing)

#### Face Restoration

- [GFPGAN](https://github.com/TencentARC/GFPGAN) | [1.2](https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.2.pth), [1.3](https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth), [1.4](https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth)
- [RestoreFormer](https://github.com/wzhouxiff/RestoreFormer) | [Model](https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/RestoreFormer.pth)
- [CodeFormer](https://github.com/sczhou/CodeFormer) | [Model](https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth)

#### Inpainting

- [LaMa](https://github.com/advimman/lama) | [Model](https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt)
- [MAT](https://github.com/fenglinglwb/MAT) | [Model](https://github.com/Sanster/models/releases/download/add_mat/Places_512_FullData_G.pth)

#### Denoising

- [SCUNet](https://github.com/cszn/SCUNet) | [GAN Model](https://github.com/cszn/KAIR/releases/download/v1.0/scunet_color_real_gan.pth) | [PSNR Model](https://github.com/cszn/KAIR/releases/download/v1.0/scunet_color_real_psnr.pth)

#### DeJPEG

- [FBCNN](https://github.com/jiaxi-jiang/FBCNN) | [Models](https://github.com/jiaxi-jiang/FBCNN/releases/tag/v1.0)


## File type support

Spandrel mainly supports loading .pth files for its supported architectures. This is what you will typically find from official repos and community trained models. However, Spandrel also supports loading TorchScript traced models (.pt) by extracting the weights out of them, as well as loading certain types of .ckpt files, as well as any supported model that has been saved as or converted to a .safetensors file.

## Security

As you may know, loading `.pth` files usually [poses a security risk](https://github.com/pytorch/pytorch/issues/52596) due to python's `pickle` module being unsafe and vulnerable to arbitrary code execution (ACE). Because of this, Spandrel uses a custom unpickler function that only allows loading certain types of data out of a .pth file. This completely prevents ACE and makes loading untrusted files secure.

## Contributing

Feel free to contribute more model architecture support. When I add model support, I usually dig through the .pth file (state dict) keys and weights to find a way to get all the parameters of a model. At some point, I will document that entire process here. For now, there are plenty of example to reference.

If the model arch you're adding does not have any parameter variants (for example, different scales or layer counts) then it should be fine adding it without any of the param detection. At the very least, you will need to find something uniquely identifiable in your model (usually a unique, really long key) that you can then add to `/spandrel/__helpers/main_registry.py` in order to load your model (preferably at the bottom). You will also need to set up the `__init__.py` file for your arch to include a `load` method, returning as ModelDescriptor with the model and some metadata about the model and its parameters.

Like with the parameter detection, there's plenty of examples there. This might seem like a lot of hardcoding (and it very well is), but it's the only way to identify models based on just the .pth file (or any other weight storage format), since these files are just the weights of a model. If anybody can figure out a better way to do this, be my guest, but for now this is the best way and it works well.

## License Notice

This repo is bounded by GPLv3 license. However, all the architectures used in this repository are bound by their own original licenses, which have been included in their respective places in this repo. The state dict parsing (load.py) files are not bound by these original licenses as they are new code.

The original code has also been slightly modified and formatted to fit the needs of this repo. If you want to use these architectures in your own codebase (but why would you if you have this package 😉), I recommend grabbing them from their original sources.
