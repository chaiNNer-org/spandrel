from spandrel.architectures.RealCUGAN import (
    RealCUGANArch,
    UpCunet2x,
    UpCunet2x_fast,
    UpCunet3x,
    UpCunet4x,
)

from .util import (
    ModelFile,
    TestImage,
    assert_image_inference,
    assert_loads_correctly,
    assert_size_requirements,
    assert_training,
    disallowed_props,
    skip_if_unchanged,
)

skip_if_unchanged(__file__)


def test_load():
    assert_loads_correctly(
        RealCUGANArch(),
        lambda: UpCunet2x(in_channels=3, out_channels=3),
        lambda: UpCunet2x(in_channels=1, out_channels=4),
        lambda: UpCunet2x(pro=True),
        lambda: UpCunet3x(in_channels=3, out_channels=3),
        lambda: UpCunet3x(in_channels=1, out_channels=4),
        lambda: UpCunet3x(pro=True),
        lambda: UpCunet4x(in_channels=3, out_channels=3),
        lambda: UpCunet4x(in_channels=1, out_channels=4),
        lambda: UpCunet4x(pro=True),
        lambda: UpCunet2x_fast(in_channels=3, out_channels=3),
        lambda: UpCunet2x_fast(in_channels=1, out_channels=1),
    )


def test_size_requirements():
    file = ModelFile.from_url(
        "https://drive.google.com/file/d/1VtBY4ZEebEiYL-IZRGJ61LUDCSRvkdoC/view?usp=sharing",
        name="up2x-latest-no-denoise.pth",
    )
    assert_size_requirements(file.load_model())

    file = ModelFile.from_url(
        "https://drive.google.com/file/d/1DfB-tMUKU_3NwQuM9Z0ZGYPhfLmzkDHb/view?usp=sharing",
        name="up3x-latest-no-denoise.pth",
    )
    assert_size_requirements(file.load_model())

    file = ModelFile.from_url(
        "https://drive.google.com/file/d/1Y7SGNuivVjPf1g6F3IMvTsqt64p_pTeH/view?usp=sharing",
        name="up4x-latest-no-denoise.pth",
    )
    assert_size_requirements(file.load_model())

    file = ModelFile.from_url(
        "https://drive.google.com/file/d/1BxpM_J-tnGuxXpC61vKfqm-08ofhSnyF/view?usp=sharing",
        name="sudo_shuffle_cugan_9.584.969.pth",
    )
    assert_size_requirements(file.load_model(), max_candidates=128, max_size=128)


def test_train():
    assert_training(RealCUGANArch(), UpCunet2x())
    assert_training(RealCUGANArch(), UpCunet2x_fast())
    assert_training(RealCUGANArch(), UpCunet3x())
    assert_training(RealCUGANArch(), UpCunet4x())


def test_RealCUGAN_2x(snapshot):
    file = ModelFile.from_url(
        "https://drive.google.com/file/d/1VtBY4ZEebEiYL-IZRGJ61LUDCSRvkdoC/view?usp=sharing",
        name="up2x-latest-no-denoise.pth",
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, UpCunet2x)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_32, TestImage.SR_64],
    )


def test_RealCUGAN_3x(snapshot):
    file = ModelFile.from_url(
        "https://drive.google.com/file/d/1DfB-tMUKU_3NwQuM9Z0ZGYPhfLmzkDHb/view?usp=sharing",
        name="up3x-latest-no-denoise.pth",
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, UpCunet3x)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_32, TestImage.SR_64],
    )


def test_RealCUGAN_4x(snapshot):
    file = ModelFile.from_url(
        "https://drive.google.com/file/d/1Y7SGNuivVjPf1g6F3IMvTsqt64p_pTeH/view?usp=sharing",
        name="up4x-latest-no-denoise.pth",
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, UpCunet4x)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_32, TestImage.SR_64],
    )


def test_RealCUGAN_2x_pro(snapshot):
    file = ModelFile.from_url(
        "https://drive.google.com/file/d/10QOxPsGmWyBTLK2ATTR9FzRNaXWSEfrt/view?usp=sharing",
        name="pro-no-denoise-up2x.pth",
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, UpCunet2x)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_32, TestImage.SR_64],
    )


def test_RealCUGAN_3x_pro(snapshot):
    file = ModelFile.from_url(
        "https://drive.google.com/file/d/1jTmhjMwusUsRp2h9lsMZtTM9n8UL6p_V/view?usp=sharing",
        name="pro-no-denoise-up3x.pth",
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, UpCunet3x)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_32, TestImage.SR_64],
    )


def test_RealCUGAN_2x_fast(snapshot):
    file = ModelFile.from_url(
        "https://drive.google.com/file/d/1BxpM_J-tnGuxXpC61vKfqm-08ofhSnyF/view?usp=sharing",
        name="sudo_shuffle_cugan_9.584.969.pth",
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, UpCunet2x_fast)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_32, TestImage.SR_64],
    )
