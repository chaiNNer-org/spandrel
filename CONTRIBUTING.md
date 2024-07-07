# Contributing

First of all, thank you so much for taking the time to contribute! If you have trouble or get stuck at any point, please don't hesitate to ask for help. You can do so by opening an issue or talk to us in the _#contrib-dev_ channel on [chaiNNer's discord server](https://discord.gg/pzvAKPKyHM).

This document will explain how to [get started](#getting-started) and give a guide for [adding new architectures](#adding-new-architectures).

## Getting started

Before you can start with the actual development, you need to set up the project.

1. Fork and clone the repository. ([GitHub guide](https://docs.github.com/en/get-started/quickstart/fork-a-repo))
2. Install our dev dependencies: `pip install -r requirements-dev.txt`.
3. Install the packages of this repo as editable installs: `pip install -e libs/spandrel -e libs/spandrel_extra_arches`.

   This will also install their dependencies, so this might take a while. (One of our dependencies, `torch`, is huge (>2GB) and might need a few minutes to download depending on your internet speed.)

We recommend using [Visual Studio Code](https://code.visualstudio.com/) as your IDE. It has great support for Python and is easy to set up. If you do, please install the following extensions: PyLance, Ruff, and Code Spell Checker. VSCode should show you a notification when you first open the project to install these extensions.

### Running tests

Spandrel has a lot of tests that 1) need to download pretrained models and 2) need to do a lot of CPU-heavy computations. Running all tests for the first time might take several minutes depending on your computer and internet speed.

As such, we recommend running only the tests for the architecture you are working on. This can be done in 2 ways:

1. Using the command line: `pytest tests/test_<arch>.py`. E.g. if you are working on ESRGAN, you can run `pytest tests/test_ESRGAN.py` to run only the tests for ESRGAN.
2. Using the [VSCode test runner](https://code.visualstudio.com/docs/python/testing#_run-tests). This also allows you to run individual tests and to easily debug tests.

Updating/adding snapshots has to be done via the command line. Use the command: `pytest tests/test_<arch>.py --snapshot-update`.

### Formatting, linting, type checking

We use [ruff](https://docs.astral.sh/ruff/) for formatting and linting and [pyright](https://microsoft.github.io/pyright/#/type-concepts) for static type checking. If you are using VSCode with the recommended extensions, everything is already set up and configured for you. If you are using a different IDE, you will have to set up these tools yourself or use them through the command line:

- Ruff linting + auto fix: `ruff check libs tests --fix --unsafe-fixes`
- Ruff formatting: `ruff format libs tests`
- PyRight: `pyright libs tests`

#### `pre-commit`

You can also use [pre-commit](https://pre-commit.com/) to automatically run Ruff before committing.
To do so, install pre-commit (`pip install pre-commit`) and run `pre-commit install` in the root directory of the repo.
You can also run `pre-commit run --all-files` to run pre-commit on all files in the repo.

### Project Structure

The project is structured as follows:

- `libs/spandrel/spandrel/`: The code of the library.
- `libs/spandrel/spandrel/__helpers/`: The internal implementation of private and public classes/constants/functions. This includes `ModelLoader`, `ModelDescriptor`, `MAIN_REGISTRY`, and much more.
- `libs/spandrel/spandrel/__init__.py`: This file re-exports classes/constants/functions from `__helpers` to define the public API of the library.
- `libs/spandrel/spandrel/architectures/`: The directory containing all architecture implementations. E.g. ESRGAN, SwinIR, etc.
- `libs/spandrel/spandrel/architectures/<arch>/__init__.py`: The file containing the `load` method for that architecture. (A `load` method takes a state dict, detects the hyperparameters of the architecture, and returns a `ModelDescriptor` variant.)
- `tests/`: The directory containing all tests for the library.
- `scripts/`: The directory for scripts used in the development of the library.

### Useful commands

#### Type checking

- `pyright libs`: Check for type errors.

#### Testing

- `pytest tests`: Run all tests.
- `pytest tests --snapshot-update`: Run all tests and update snapshots.
- `pytest tests/test_<arch>.py`: Run the tests for a specific architecture.
- `pytest tests/test_<arch>.py --snapshot-update`: Run the tests for a specific architecture and update snapshots.

##### Running inference tests with another Torch device

You can set the `SPANDREL_TEST_DEVICE` environment variable to run inference tests using the specified Torch device. The default is `cpu`.

If you use that environment variable, it may be useful to set `SPANDREL_TEST_OUTPUTS_DIR` to (e.g.) `outputs-mps`; this will make the test suite output images to a directory named `outputs-mps` instead of `outputs`, so you can use your favorite comparison tool to compare the outputs of the tests run on different devices.

For instance, to test inference using `mps` (Metal Performance Shaders) on an Apple Silicon chip, you could try:

```
env PYTORCH_ENABLE_MPS_FALLBACK=1 SPANDREL_TEST_DEVICE=mps SPANDREL_TEST_OUTPUTS_DIR=outputs-mps pytest --snapshot-update
```

#### Scripts

- `python scripts/dump_state_dict.py /path/to/model.pth`: Dumps the contents of a state dict into `dump.yml`. Useful for adding architectures. See the documentation in the file for more information.
- `python scripts/dump_dummy.py`: Same as `dump_state_dict.py`, but it dumps the contents of a dummy model instead. See the documentation in the file for more information.

## Adding new architectures

This guide will take you through the process of adding a new architecture to Spandrel. We will use [DITN](https://github.com/yongliuy/DITN) as an example.

Note that adding support for an architecture does **NOT** require understanding that architecture. E.g. you do not need to understand how DITN works to add it to Spandrel. You only need to be able to read, write, and understand Python code.

### Step 0: Prerequisites

Before we can start, we need to find out a few things about the architecture we want to add. We need to know:

1. Whether the architecture is implemented using PyTorch. \
   If it is implemented with anything else (e.g. TensorFlow), we cannot add it to Spandrel. The repos of the architecture should have a `requirements.txt` file that lists all dependencies. If it lists `torch` as a dependency, we are good to go.
2. Where we can get official models. \
   Pretty much all architectures have official models trained by the researchers that made the architecture. Typically, the README of the repo will give you links to these models (often Google Drive). Some also use GitHub releases. Take note of these links for now. We will later use the official models in our tests.
3. The license of the architecture. \
   If the architecture's code is not licensed under a permissive license (e.g. MIT, BSD, Apache), we cannot add it to Spandrel. If there is no license or the license is not clear, you have to ask the authors for permission to copy their code. A permissive license or explicit permission from the authors is required for adding an architecture to Spandrel.

In the case of DITN, the architecture is implemented using PyTorch, a link to the official models is in the README, and it is licensed.

### Step 1: Copy the architecture code

The first step is to copy the code of the architecture into Spandrel.

Of course, the copy this code, we first have to find it! Unfortunately, project layout is not consistent across architecture repos, so you might be to search a bit. You need that file with the class that defines the architecture. In most repos, this file is called `model.py`, `models/<arch name>.py`, or `basicsr/archs/<arch name>_arch.py`. You know that you found the right file when it contains a class that is named after the architecture and inherits from `nn.Module`.

In the case of DITN, the file is called [`models/DITN_Real.py`](https://github.com/yongliuy/DITN/blob/3438e429c0538ee5061a7cfca587df0c4097703f/models/DITN_Real.py#L197).

Once you have found the file, create a new directory `libs/spandrel/architectures/<arch name>/__arch/` and copy the file into it. This directory will contain all code that we will copy. Since we respect the copy right of the original authors, we will also copy the `LICENSE` file of the repo into this directory.

In the case of DITN, we copy `models/DITN_Real.py` to `libs/spandrel/architectures/DITN/__arch/DITN_Real.py` and add the `LICENSE` file of the repo.

The main model file might also reference other files in the repo. You have to copy those files into your `__arch/` directory as well.

In the case of DITN, the main model file doesn't reference any other files, so we are done.

### Step 1.1: Linting, type errors, and cleaning up

You might have already noticed that the code you copied lights up like a Christmas tree in your IDE. In VSCode, just hit Ctrl+S to let ruff fix the formatting and most linting errors.

However, there are a few things that we have to fix manually.

- "Use `super()` instead of `super(__class__, self)`" \
   Most architectures use the old style of calling `super()` to support Python 3.7. Spandrel does not support Python 3.7, so you can just replace `super(__class__, self)` with `super()`.
- Unused variables \
   Researchers are trying to push what is possible and often not too concerned with code quality. You will often find unused variables which our linters do not like. Instead of removing those variables, prefix them with `_`. In general, don't try to clean up the code (yet). You do not want to accidentally change the architecture code and then wonder why the official models aren't working.
- Type errors \
   Some of them can be fixed by adding type annotations, but it's often best to just use a `# type: ignore` comment to suppress the error (for now).
- Removing `**kwargs` \
   The main class of your architecture might have an used parameter `**kwargs` or `**ignorekwargs`. Remove it. This parameter means that typoing a parameter name will not result in an error, and it will just silently ignore the parameter. This is the cause of much frustration and many bugs, so save yourself and remove the parameter. \
   If the parameter is used, leave it for now.

In general, try to change the original code as little as possible in this stage. We can clean up the code after we have fully added support for the architecture and have tests to make sure that we didn't break anything.

Finally, some model files contain a `if __name__ == "__main__":` section to run the model as a script. We do not ever want to run the model as a script, so we can just remove this section.

In the case of DITN, we just ignore a few type errors and remove [this section](https://github.com/yongliuy/DITN/blob/3438e429c0538ee5061a7cfca587df0c4097703f/models/DITN_Real.py#L249).

### Step 1.2: Fixing dependencies

The model files might have dependencies to external packages. We generally try to keep the number of dependencies for spandrel as low as possible, so we want to remove any dependencies that are not strictly necessary.

We allow models to have the following external dependencies: `torch`, `torchvision`, `numpy`, and `einops`.

`timm` is a special dependency for us, because we have vendored the most important code from `timm`. So instead of using the `timm` package, we should use the vendored code from `libs/spandrel/architectures/__arch_helpers/timm/`.

In the case of DITN, its model file has no external dependencies, except for `torch` and `einops`, so we are done.

### Step 2: Adding a `load` method

With the architecture code in place, we can start integrating it into spandrel. The first step is to define a `load` function. This function will take a state dict, detect the hyperparameters of the architecture (= the input arguments of the architecture's Python class), and return a loaded model.

We will worry about parameter detection later, for now, we'll just return a dummy model.

Create a file `libs/spandrel/architectures/<arch name>/__init__.py` and add the following code:

```python
from ...__helpers.model_descriptor import ImageModelDescriptor, StateDict
from .__arch.ARCH_NAME import ARCH_NAME


def load(state_dict: StateDict) -> ImageModelDescriptor[ARCH_NAME]:
    # default values
    param1 = 1
    param2 = 2

    model = ARCH_NAME(
        param1=param1,
        param2=param2,
    )

    return ImageModelDescriptor(
        model,
        state_dict,
        architecture="ARCH_NAME",
        purpose="SR",
        tags=[],
        supports_half=True,
        supports_bfloat16=True,
        scale=1,  # TODO: fix me
        input_channels=3,  # TODO: fix me
        output_channels=3,  # TODO: fix me
    )
```

Next, fill in the template:

- Replace `architecture="ARCH_NAME"` with the name of your architecture.
- Replace the references to `ARCH_NAME` with your architecture model file and class.
- Replace `param1` and `param2` with the parameters of your architecture. Copy the default values of the parameters from the model class and pass them into the constructor (`model = ARCH_NAME(...)`).
- Fill in the missing value for `scale`, `input_channels`, and `output_channels`.
  - `scale`: If your architecture is a super resolution architecture, there should be a `scale`, `upscale`, or `upscale_factor` parameter. Use this value. If your architecture is not a super resolution architecture, set this value to `1`.
  - `input_channels`: The number of input channels of the architecture. Again, there should be a parameter for this. Something like `in_nc`, `in_channels`, `inp_channels`. If there isn't leave it at 3 for RGB images.
  - `output_channels`: The number of output channels of the architecture. Same as for `input_channels`. However, if there is no parameter for output channels, but there is one for input channels, just use the parameter for input channels.
- Set the purpose of the architecture. If you have a restoration architecture (e.g. denoising, deblurring, removing JPEG artifacts), set this to "Restoration". If you have a super resolution architecture, set this to "SR" if the scale of the model is >1 and "Restoration" otherwise. We consider 1x model for SR architectures (e.g. 1x ESRGAN model) to be restoration models. (See DITN example below.)

In the case of DITN, the filled in template looks like this:

<details>
<summary>Click to see code</summary>

```python
from ...__helpers.model_descriptor import ImageModelDescriptor, StateDict
from .__arch.DITN_Real import DITN_Real as DITN


def load(state_dict: StateDict) -> ImageModelDescriptor[DITN]:
    # default values
    inp_channels = 3
    dim = 60
    ITL_blocks = 4
    SAL_blocks = 4
    UFONE_blocks = 1
    ffn_expansion_factor = 2
    bias = False
    LayerNorm_type = "WithBias"
    patch_size = 8
    upscale = 4

    model = DITN(
        inp_channels=inp_channels,
        dim=dim,
        ITL_blocks=ITL_blocks,
        SAL_blocks=SAL_blocks,
        UFONE_blocks=UFONE_blocks,
        ffn_expansion_factor=ffn_expansion_factor,
        bias=bias,
        LayerNorm_type=LayerNorm_type,
        patch_size=patch_size,
        upscale=upscale,
    )

    return ImageModelDescriptor(
        model,
        state_dict,
        architecture="DITN",
        purpose="Restoration" if upscale == 1 else "SR",
        tags=[],
        supports_half=True,
        supports_bfloat16=True,
        scale=upscale,
        input_channels=inp_channels,
        output_channels=inp_channels,
    )
```

Note the `purpose="Restoration" if upscale == 1 else "SR"`. This is a little trick to assign the correct purpose based on the scale of the model.

</details>

Next, we are going to write a test to make sure our `load` function works. Create a new file `tests/test_ARCH.py` and use the following template to write a single test:

```python
from spandrel.architectures.ARCH_NAME import ARCH_NAME, load

from .util import assert_loads_correctly


def test_load():
    assert_loads_correctly(
        load,
        lambda: ARCH_NAME(),
        condition=lambda a, b: (
            a.param1 == b.param1
            and a.param2 == b.param2
        ),
    )
```

The `assert_loads_correctly` function will use the models returned by the given lambdas to verify that the `load` function is implemented correctly. Conceptually, it will save a temporary `.pth` file with the model returned by the lambda, load it again using the `load` function, and then compare the two models. If the two models are the same, the test passes.

This function allows us to test arbitrary parameter configurations without having a pretrained model for each configuration. This is very useful to make sure that the parameter detection code we'll write works correctly.

The function takes 3 arguments:

1. A `load` function.
2. Any number of lambdas returning a model. \
   Those are the model that will be tested.
3. An optional `condition` function. \
   This function compares the fields of the original model (as returned by the lambda(s)) and the loaded model (output of `load`). Since many model class store some of their parameters as fields, we can make sure that `load` correctly detected the parameters here. To know which fields to compare, just look at the code of your model class and see which parameters it assigns to fields in the constructor.

If you [run this test](#running-tests) now, it should pass. Since we are passing in the default values for all parameters in `load`, we should get the same model as just `ARCH_NAME()`.

With this now all setup up, we can start detecting parameters in the next step.

But before that, here's how this test looks like for DITN:

<details>
<summary>Click to see code</summary>

```python
from spandrel.architectures.DITN import DITN, load

from .util import assert_loads_correctly


def test_load():
    assert_loads_correctly(
        load,
        lambda: DITN(),
        condition=lambda a, b: (
            a.patch_size == b.patch_size
            and a.dim == b.dim
            and a.scale == b.scale
            and a.SAL_blocks == b.SAL_blocks
            and a.ITL_blocks == b.ITL_blocks
        ),
    )
```

Notice how `condition` does not cover all parameters. This particular model doesn't store all parameters as fields, so we don't compare them here.

</details>

### Step 3: Detecting parameters

Detecting parameters is an imperfect science. It might not be possible to detect all parameters in all circumstances. The problem is that the state dict does not necessarily contain information about all parameters. We will cover strategies to deal with undetectable parameters later. For now, let's detect those that can be detected.

To see whether a parameter is detectable, and how we can detect it, we will use the `dump_dummy.py` script. This script creates a dummy model (just like the lambdas we passed to `assert_loads_correctly`), and then dumps its state dict into a YAML file. This YAML file contains all information that is available in the state dict presented in a structured and "readable" way.

Open `scripts/dump_dummy.py` in your IDE and edit the `create_dummy` function. Make it return a dummy model of your architecture. Then run the script with `python scripts/dump_dummy.py`. The script will create a `dump.yml` file in the root directory of the repo (next to `README.md`). Open the file in your IDE.

In the case of DITN, the dumped state dict for the dummy model `DITN` look like this:

```yaml
# DITN.DITN()
sft
  sft.weight: Tensor float32 Size([60, 3, 3, 3])
  sft.bias:   Tensor float32 Size([60])
UFONE.0
  ITLs
    0
      attn
        temperature
          UFONE.0.ITLs.0.attn.temperature: Tensor float32 Size([1, 1, 1])
        qkv
          UFONE.0.ITLs.0.attn.qkv.weight: Tensor float32 Size([180, 60])
          UFONE.0.ITLs.0.attn.qkv.bias:   Tensor float32 Size([180])
        project_out
          UFONE.0.ITLs.0.attn.project_out.weight: Tensor float32 Size([60, 60, 1, 1])
      conv1
        UFONE.0.ITLs.0.conv1.weight: Tensor float32 Size([60, 60, 1, 1])
        UFONE.0.ITLs.0.conv1.bias:   Tensor float32 Size([60])
      conv2
        UFONE.0.ITLs.0.conv2.weight: Tensor float32 Size([60, 60, 1, 1])
        UFONE.0.ITLs.0.conv2.bias:   Tensor float32 Size([60])
      ffn
        UFONE.0.ITLs.0.ffn.project_in.weight:  Tensor float32 Size([240, 60, 1, 1])
        UFONE.0.ITLs.0.ffn.dwconv.weight:      Tensor float32 Size([240, 1, 3, 3])
        UFONE.0.ITLs.0.ffn.project_out.weight: Tensor float32 Size([60, 120, 1, 1])
    1
      ...
    2
      ...
    3
      ...
  SALs
    ...
conv_after_body
  conv_after_body.weight: Tensor float32 Size([60, 60, 3, 3])
  conv_after_body.bias:   Tensor float32 Size([60])
upsample.0
  upsample.0.weight: Tensor float32 Size([48, 60, 3, 3])
  upsample.0.bias:   Tensor float32 Size([48])
```

(Shortened for brevity.)

In this YAMl file, we can see all keys in the state dict and their values. Keys are grouped into a tree structure by common prefix. E.g. `sft.weight` and `sft.bias` are grouped under `sft`. This is useful to see which keys belong to the same python object. Tensor values are printed with their data type and shape. The shape of tensors in the main thing we'll use to detect parameters.

Let's see how those keys and grouping relate to the python code of the model class. Here's a section of the `__init__` code of the `DITN` class:

```python
self.sft = nn.Conv2d(inp_channels, dim, 3, 1, 1)

## UFONE Block1
UFONE_body = [
    UFONE(
        dim,
        ffn_expansion_factor,
        bias,
        LayerNorm_type,
        ITL_blocks,
        SAL_blocks,
        patch_size,
    )
    for _ in range(UFONE_blocks)
]
self.UFONE = nn.Sequential(*UFONE_body)

self.conv_after_body = nn.Conv2d(dim, dim, 3, 1, 1)
self.upsample = UpsampleOneStep(upscale, dim, 3)
```

The fact those the field names in the class and those groups in `dump.yml` match is no coincidence. Every key in the state dict is the _path_ to a particular field. E.g. `sft.weight` is the path to the `weight` field of the `sft` field of the model. We could even access this field directly in Python:

```python
model = DITN()
print(model.sft.weight.shape) # prints torch.Size([60, 3, 3, 3])
```

Let's look at `sft` a little more. Here's its python code and dumped state:

```python
self.sft = nn.Conv2d(inp_channels, dim, 3, 1, 1)
```

```yaml
sft
  sft.weight: Tensor float32 Size([60, 3, 3, 3])
  sft.bias:   Tensor float32 Size([60])
```

Looking at the default values, we can see `inp_channels=3` and `dim=60`. We can also see a 60 and some 3s in the shape of `sft.weight`, but which one is which? We can use the `dump_dummy.py` script to find out!

1. Use the IDE's git integration to _stage_ (not commit) `dump.yml`.
2. Modify the `create_dummy` function in `dump_dummy.py` to return a model with different values for `inp_channels` (e.g. `DITN(inp_channels=4)`).
3. Save the script and run it again. `dump.yml` not contains the dumped state dict of the new dummy model.
4. Use the git integration of our IDE to look at the diff of the staged `dump.yml` and compare it with the new `dump.yml` to see what changed.

Here's what changed for `sft`:

```diff
 sft
-  sft.weight: Tensor float32 Size([60, 3, 3, 3])
+  sft.weight: Tensor float32 Size([60, 4, 3, 3])
   sft.bias:   Tensor float32 Size([60])
```

We can see that only the second element in the shape of `sft.weight` changed. So we can conclude that the second element is `inp_channels` and the first element is `dim`. You might not be convinced yet, so feel free to play around with different parameter values some more.

Using this new knowledge, we can now detect the `inp_channels` parameters. So let's update the `load` function use the shape of `sft.weight` to detect `inp_channels`:

```python
def load(state_dict: StateDict) -> ImageModelDescriptor[DITN]:
    # default values
    inp_channels = 3
    dim = 60
    # ...

    inp_channels = state_dict["sft.weight"].shape[1]

    model = DITN(
        inp_channels=inp_channels,
        dim=dim,
        # ...
    )
```

Now, let's see whether this was actually correct. Let's add a few more test cases to `test_DITN_load` and run it:

```python
assert_loads_correctly(
    load,
    lambda: DITN(),
    lambda: DITN(inp_channels=4),
    lambda: DITN(inp_channels=1),
    condition=lambda a, b: (...),
)
```

The test passed, so we can be reasonably confident that we detected `inp_channels` correctly. The same process also applies to `dim`.

Unfortunately, not all parameters can be read directly like this. Let's take a look at DITN's `UFONE_blocks` parameter. In code, it is used like this.

```python
UFONE_body = [
    UFONE(
        dim,
        ffn_expansion_factor,
        bias,
        LayerNorm_type,
        ITL_blocks,
        SAL_blocks,
        patch_size,
    )
    for _ in range(UFONE_blocks) # <--
]
self.UFONE = nn.Sequential(*UFONE_body)
```

So `UFONE_blocks` controls the length of the sequence of the `UFONE`. Its default value is 1, so it's not much a sequence by default. We can still see this in the dumped state dict:

```yaml
UFONE.0
ITLs
...
```

When you see a grouping called `<name>.0`, that's a sequence of length 1. If we modify `dump_dummy.py` to set `UFONE_blocks=3`, we'll get this instead:

```yaml
UFONE
0
ITLs
...
1
ITLs
...
2
TLs
..
```

So we can clearly see the length of sequences using the groupings in `dump.yml`. However, we cannot access this length directly from state dict. To get around this, we'll use a helper function called `get_seq_len`. As the name suggests, it determines the length of a sequence in a state dict. Adding this to DITN's `load` function looks like this:

```python
# ...
from ..__arch_helpers.state import get_seq_len

def load(state_dict: StateDict) -> ImageModelDescriptor[DITN]:
    # ...
    UFONE_blocks = get_seq_len(state_dict, "UFONE")
```

Again, we'll add tests for this to make sure that it works.

As the last example, we'll detect DITN's `ffn_expansion_factor` parameter. As before, we'll modify `dump_dummy.py` to see what the parameter does. Its default value is 2, so we'll set it to 3. Here's the diff:

```diff
       ffn
-        UFONE.0.ITLs.0.ffn.project_in.weight:  Tensor float32 Size([240, 60, 1, 1])
-        UFONE.0.ITLs.0.ffn.dwconv.weight:      Tensor float32 Size([240, 1, 3, 3])
-        UFONE.0.ITLs.0.ffn.project_out.weight: Tensor float32 Size([60, 120, 1, 1])
+        UFONE.0.ITLs.0.ffn.project_in.weight:  Tensor float32 Size([360, 60, 1, 1])
+        UFONE.0.ITLs.0.ffn.dwconv.weight:      Tensor float32 Size([360, 1, 3, 3])
+        UFONE.0.ITLs.0.ffn.project_out.weight: Tensor float32 Size([60, 180, 1, 1])
 ...
       ffn
-        UFONE.0.ITLs.1.ffn.project_in.weight:  Tensor float32 Size([240, 60, 1, 1])
-        UFONE.0.ITLs.1.ffn.dwconv.weight:      Tensor float32 Size([240, 1, 3, 3])
-        UFONE.0.ITLs.1.ffn.project_out.weight: Tensor float32 Size([60, 120, 1, 1])
+        UFONE.0.ITLs.1.ffn.project_in.weight:  Tensor float32 Size([360, 60, 1, 1])
+        UFONE.0.ITLs.1.ffn.dwconv.weight:      Tensor float32 Size([360, 1, 3, 3])
+        UFONE.0.ITLs.1.ffn.project_out.weight: Tensor float32 Size([60, 180, 1, 1])
 ...
       ffn
-        UFONE.0.ITLs.2.ffn.project_in.weight:  Tensor float32 Size([240, 60, 1, 1])
-        UFONE.0.ITLs.2.ffn.dwconv.weight:      Tensor float32 Size([240, 1, 3, 3])
-        UFONE.0.ITLs.2.ffn.project_out.weight: Tensor float32 Size([60, 120, 1, 1])
+        UFONE.0.ITLs.2.ffn.project_in.weight:  Tensor float32 Size([360, 60, 1, 1])
+        UFONE.0.ITLs.2.ffn.dwconv.weight:      Tensor float32 Size([360, 1, 3, 3])
+        UFONE.0.ITLs.2.ffn.project_out.weight: Tensor float32 Size([60, 180, 1, 1])
 ...
```

A lot of things changed, so the diff is shortened for brevity.

We can see that a few `240`s changed to `360`s and a few `120`s changed to `180`s. So these values seem to linearly scale with `ffn_expansion_factor`. So in code, we would expect to see a `some_value * ffn_expansion_factor` somewhere. Since the keys in state dict are just paths, let's follow the path `UFONE.0.ITLs.0.ffn.project_in.weight`. This eventually brings us to this class:

```python
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super().__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(
            hidden_features * 2,
            hidden_features * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_features * 2,
            bias=bias,
        )

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        ...
```

We can see `hidden_features = int(dim * ffn_expansion_factor)`, so that's where the scaling happens. We also see that `hidden_features` is used for `project_in`, `dwconv`, and `project_out`, which are exactly the field that changed in our diff.

`project_in` uses `dim` and `hidden_features * 2` for its shape. Since the default value of `dim` is 60, the shape should include a 60 and a 60\*3\*2 = 360, and that's exactly what we see in the diff.

So we now know that we can calculate `ffn_expansion_factor` using this:

```python
hidden_features = state_dict["UFONE.0.ITLs.0.ffn.project_in.weight"].shape[0] / 2
ffn_expansion_factor = hidden_features / dim
```

We already detected `dim` before, so we can calculate `ffn_expansion_factor` directly.

To conclude this section: this is how we detect parameters. Use `dump_dummy.py` to see what a parameter does, think of a way to detect the changes a parameter causes, and then write code to detect the parameter. The above example are all relatively simple, so some parameters might be a lot harder to detect. If you are stuck, either skip the parameter or ask for help (see the start of this document).

### Step 3: Registering the architecture and testing official models

After we are done with detecting whatever parameters are detectable, we can register the architecture and test official models.

Before we register our architecture, we'll add a test using one of the official models. (If your architecture doesn't have accessible official models, you'll have to skip adding the test.)

Create a new function `tests/test_ARCH.py` like this:

```python
from spandrel.architectures.ARCH import ARCH, load

from .util import (
    ModelFile,
    TestImage,
    assert_image_inference,
    assert_loads_correctly,
    disallowed_props,
    skip_if_unchanged,
)

skip_if_unchanged(__file__)


def test_load():
    ...


def test_ARCH_model_name(snapshot):
    file = ModelFile.from_url(
        "https://example.com/path/to/model_name.pth"
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, ARCH)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_16, TestImage.SR_32, TestImage.SR_64],
    )
```

Fill in the details as usual and run the test. It should download the model successfully and then fail with an `UnsupportedModelError`. Let's fix this error my registering the architecture.

Open `libs\spandrel\__helpers\main_registry.py` and start by importing your architecture in the `from ..architectures import (...)` state. Then get read to scroll all the way down and add a new `ArchSupport` object at the end of `MAIN_REGISTRY.add(...)`. The `ArchSupport` object will tell spandrel how to detect whether an arbitrary state dict (so a state dict of any architecture) is from your architecture. We do this by simply detecting the presence of a particular key. Each architecture has a few keys that are always present in the state dict, and we'll use those for detection.

In the case of DITN, its `ArchSupport` object looks like this:

```python
ArchSupport(
    id="DITN",
    detect=_has_keys(
        "sft.weight",
        "UFONE.0.ITLs.0.attn.temperature",
        "UFONE.0.ITLs.0.ffn.project_in.weight",
        "UFONE.0.ITLs.0.ffn.dwconv.weight",
        "UFONE.0.ITLs.0.ffn.project_out.weight",
        "conv_after_body.weight",
        "upsample.0.weight",
    ),
    load=DITN.load,
)
```

Run the test from before again, it should now fail because there's not snapshot for it yet. So let's run the architecture tests from the command line to fix this: `pytest tests/test_ARCH.py --snapshot-update`. Running this, all tests should pass, and a new snapshot should be created. You'll also see a few new images. Those are the snapshots for our inference tests.

Lastly, your architecture and a link to the official models to `README.md`.

And with this, you are pretty much done. You can add more tests now, or clean up architecture code a little if you want to.

### Step 4: Undetectable parameters

Undetectable parameters are very difficult to deal with. There are a few strategies we can use to deal with them, but none of them are perfect.

1. Look at the official training configuration. \
   Many repos will include the configuration used to train the official models. This configuration will include the values for all (changed) parameters. If all configurations use the same value, it's probably fine to use that value too. If they use different values, then maybe we can detect the parameter by looking at the differences between the configurations. E.g. config 1 has `a=1, b=2, c=3`, config 2 has `a=2, b=2, c=2`, and `c` is undetectable, then we can deduce the value of `c` by looking at `a` if `a` is detectable.
2. Hope that no model changes the default. \
   Sounds like a bad strategy, but it works surprisingly often. Some architectures have dozens of parameters, but only a few are actually used in the official models. So with a bit of luck, the default value will work just fine.

Unfortunately, that's pretty much all we can do.
