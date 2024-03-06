from spandrel_extra_arches.architectures.CodeFormer import CodeFormer, CodeFormerArch

from .util import (
    ModelFile,
    TestImage,
    assert_image_inference,
    assert_loads_correctly,
    assert_size_requirements,
    disallowed_props,
)


def test_load():
    assert_loads_correctly(
        CodeFormerArch(),
        lambda: CodeFormer(),
        lambda: CodeFormer(dim_embd=256),
        lambda: CodeFormer(n_layers=5, codebook_size=512, latent_size=64),
        lambda: CodeFormer(connect_list=["16", "32", "64"]),
    )


def test_size_requirements():
    file = ModelFile.from_url(
        "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth"
    )
    # TODO: this currently doesn't ensure that 1024x1024 is invalid
    assert_size_requirements(file.load_model(), max_size=512)


def test_CodeFormer(snapshot):
    file = ModelFile.from_url(
        "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth"
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, CodeFormer)
    assert_image_inference(
        model_file=file,
        model=model,
        test_images=[TestImage.BLURRY_FACE],
    )
