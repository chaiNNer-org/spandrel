# Spandrel

[![PyPI package](https://img.shields.io/badge/pip%20install-spandrel-brightgreen)](https://pypi.org/project/spandrel/)
[![version number](https://img.shields.io/pypi/v/spandrel?color=green&label=version)](https://github.com/chaiNNer-org/spandrel/releases)
[![PyPi Downloads](https://img.shields.io/pypi/dw/spandrel)](https://pypi.org/project/spandrel/#files)
[![Python Version](https://img.shields.io/pypi/pyversions/spandrel)](https://pypi.org/project/spandrel/#files:~:text=Requires%3A%20Python%20%3C3.12%2C%20%3E%3D3.8)

[![Actions Status](https://github.com/chaiNNer-org/spandrel/workflows/Test/badge.svg)](https://github.com/chaiNNer-org/spandrel/actions)
[![License](https://img.shields.io/github/license/chaiNNer-org/spandrel)](https://github.com/chaiNNer-org/spandrel/blob/main/LICENSE)
[![Contributors](https://img.shields.io/github/contributors/chaiNNer-org/spandrel)](https://github.com/chaiNNer-org/spandrel/graphs/contributors)

This package ports [chaiNNer](https://github.com/chaiNNer-org/chaiNNer)'s PyTorch architecture support and model loading functionality into its own package, and wraps it into an easy-to-use API.

After seeing many projects extract out chaiNNer's model support into their own projects, I decided it was probably worth the effort of creating a PyPi package that those developers could use instead.

I'm also hoping that by having a central package anyone can use, the community will be encouraged [to help add support for more models](CONTRIBUTING.md). This will ultimately benefit everyone.

This package does not yet have easy inference code, but porting that code is planned as well.

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

And that's it. The model gets loaded into a helper class called a ModelDescriptor with various helpful bits of information, as well as the actual model information.

```py
# The model itself (a torch.nn.Module loaded with the weights)
loaded_model.model

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

ModelDescriptors also support basic inference, with per-descriptor parameters to keep everything simple. For example, an `ImageModelDescriptor` (used for super-resolution and restoration) takes in a single image tensor and returns a single image tensor, whereas a `MaskedModelDescriptor` (used for inpainting) takes in an image tensor and a mask tensor and returns a single image tensor.

> **_NOTE: This is not an inference wrapper in the sense that it wil convert an image to a tensor for you. This is purely making the forward passes of these models more convenient to use, since the actual forward passes are not always as simple as image in/image out._**

ModelDescriptors also have a few convenience methods to make them more similar to regular `torch.nn.Module`s: `.to`, `.train`, and `.eval`.

Example:

```py
model = ModelLoader().load_from_file(r"/path/to/your/model.pth")
model.to("cuda:0")
model.eval()
def process(tensor: Tensor) -> Tensor:
    with torch.no_grad():
        return model(tensor)
```

## Model Architecture Support

Spandrel currently supports a limited amount of neural network architectures. It can auto-detect these architectures just from their files alone.

> **_NOTE: By its very nature, Spandrel will never be able to support every model architecture. The goal is just to support as many as is realistically possible._**

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
- [FeMaSR](https://github.com/chaofengc/FeMaSR) | [Models](https://github.com/chaofengc/FeMaSR/releases/tag/v0.1-pretrain_models)
- [GRLIR](https://github.com/ofsoundof/GRL-Image-Restoration) | [Models](https://github.com/ofsoundof/GRL-Image-Restoration/releases/tag/v1.0.0)
- [DITN](https://github.com/yongliuy/DITN) | [Models](https://drive.google.com/drive/folders/1XpHW27H5j2S4IH8t4lccgrgHkIjqrS-X)
- [MM-RealSR](https://github.com/TencentARC/MM-RealSR) | [Models](https://github.com/TencentARC/MM-RealSR/releases/tag/v1.0.0)
- [SPAN](https://github.com/hongyuanyu/SPAN) | [Models](https://drive.google.com/file/d/1iYUA2TzKuxI0vzmA-UXr_nB43XgPOXUg/view?usp=sharing)
- [Real-CUGAN](https://github.com/bilibili/ailab/tree/main/Real-CUGAN) | [Models](https://drive.google.com/drive/folders/1jAJyBf2qKe2povySwsGXsVMnzVyQzqDD), [Pro Models](https://drive.google.com/drive/folders/1hfT4WwnNUaS43ErrgXk0J1R5Ik8s5NVo)

#### Face Restoration

- [GFPGAN](https://github.com/TencentARC/GFPGAN) | [1.2](https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.2.pth), [1.3](https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth), [1.4](https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth)
- [RestoreFormer](https://github.com/wzhouxiff/RestoreFormer) | [Model](https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/RestoreFormer.pth)
- [CodeFormer](https://github.com/sczhou/CodeFormer) | [Model](https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth)

#### Inpainting

- [LaMa](https://github.com/advimman/lama) | [Model](https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt)
- [MAT](https://github.com/fenglinglwb/MAT) | [Model](https://github.com/Sanster/models/releases/download/add_mat/Places_512_FullData_G.pth)

#### Denoising

- [SCUNet](https://github.com/cszn/SCUNet) | [GAN Model](https://github.com/cszn/KAIR/releases/download/v1.0/scunet_color_real_gan.pth) | [PSNR Model](https://github.com/cszn/KAIR/releases/download/v1.0/scunet_color_real_psnr.pth)
- [Uformer](https://github.com/ZhendongWang6/Uformer) | [Denoise SIDD Model](https://mailustceducn-my.sharepoint.com/:u:/g/personal/zhendongwang_mail_ustc_edu_cn/Ea7hMP82A0xFlOKPlQnBJy0B9gVP-1MJL75mR4QKBMGc2w?e=iOz0zz) | [Deblur GoPro Model](https://mailustceducn-my.sharepoint.com/:u:/g/personal/zhendongwang_mail_ustc_edu_cn/EfCPoTSEKJRAshoE6EAC_3YB7oNkbLUX6AUgWSCwoJe0oA?e=jai90x)
- [KBNet](https://github.com/zhangyi-3/KBNet) | [Models](https://mycuhk-my.sharepoint.com/personal/1155135732_link_cuhk_edu_hk/_layouts/15/onedrive.aspx?ga=1&id=%2Fpersonal%2F1155135732%5Flink%5Fcuhk%5Fedu%5Fhk%2FDocuments%2Fshare%2FKBNet%2FDenoising%2Fpretrained%5Fmodels)

#### DeJPEG

- [FBCNN](https://github.com/jiaxi-jiang/FBCNN) | [Models](https://github.com/jiaxi-jiang/FBCNN/releases/tag/v1.0)

#### Colorization

- [DDColor](https://github.com/piddnad/DDColor) | [Models](https://github.com/piddnad/DDColor/blob/master/MODEL_ZOO.md)


## File type support

Spandrel mainly supports loading `.pth` files for all supported architectures. This is what you will typically find from official repos and community trained models. However, Spandrel also supports loading TorchScript traced models (`.pt`), certain types of `.ckpt` files, and `.safetensors` files for any supported architecture saved in one of these formats.

## Security

As you may know, loading `.pth` files usually [poses a security risk](https://github.com/pytorch/pytorch/issues/52596) due to python's `pickle` module being unsafe and vulnerable to arbitrary code execution (ACE). Because of this, Spandrel uses a custom unpickler function that only allows loading certain types of data out of a .pth file. This ideally prevents ACE and makes loading untrusted files more secure. Note that there still could be the possibility of ACE (though we don't expect this to be the case), so if you're still concerned about security, only load .safetensors models.

## License Notice

This repo is bounded by GPLv3 license. However, all the architectures used in this repository are bound by their own original licenses, which have been included in their respective places in this repo. The state dict parsing (load.py) files are not bound by these original licenses as they are new code.

The original code has also been slightly modified and formatted to fit the needs of this repo. If you want to use these architectures in your own codebase (but why would you if you have this package 😉), I recommend grabbing them from their original sources.
