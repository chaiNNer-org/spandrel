from spandrel_extra_arches.architectures.MPRNet import MPRNet, MPRNetArch

from .util import (
    ModelFile,
    TestImage,
    assert_image_inference,
    assert_loads_correctly,
    assert_size_requirements,
    disallowed_props,
    skip_if_unchanged,
)

skip_if_unchanged(__file__)


def test_load():
    assert_loads_correctly(
        MPRNetArch(),
        lambda: MPRNet(),
        lambda: MPRNet(in_c=4, out_c=1),
        lambda: MPRNet(n_feat=20),
        lambda: MPRNet(kernel_size=5),
        lambda: MPRNet(bias=True),
        lambda: MPRNet(reduction=8),
        lambda: MPRNet(scale_orsnetfeats=32),
        lambda: MPRNet(scale_unetfeats=10),
        lambda: MPRNet(num_cab=4),
        check_safe_tensors=False,
    )


def test_size_requirements():
    file = ModelFile.from_url(
        "https://drive.google.com/file/d/1QwQUVbk6YVOJViCsOKYNykCsdJSVGRtb/view",
        name="MPRNet_model_deblurring.pth",
    )
    assert_size_requirements(file.load_model())


def test_deblurring(snapshot):
    file = ModelFile.from_url(
        "https://drive.google.com/file/d/1QwQUVbk6YVOJViCsOKYNykCsdJSVGRtb/view",
        name="MPRNet_model_deblurring.pth",
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, MPRNet)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_32],
    )


def test_deraining(snapshot):
    file = ModelFile.from_url(
        "https://drive.google.com/file/d/1O3WEJbcat7eTY6doXWeorAbQ1l_WmMnM/view",
        name="MPRNet_model_deraining.pth",
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, MPRNet)


def test_denoising(snapshot):
    file = ModelFile.from_url(
        "https://drive.google.com/file/d/1LODPt9kYmxwU98g96UrRA0_Eh5HYcsRw/view",
        name="MPRNet_model_denoising.pth",
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, MPRNet)
