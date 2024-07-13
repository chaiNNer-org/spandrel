from spandrel.architectures.SeemoRe import SeemoRe, SeemoReArch

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
        SeemoReArch(),
        lambda: SeemoRe(),
        lambda: SeemoRe(in_chans=1, embedding_dim=32, num_layers=2),
        lambda: SeemoRe(scale=2, num_experts=4),
        lambda: SeemoRe(scale=1, global_kernel_size=7),
        lambda: SeemoRe(num_experts=4, lr_space="linear"),
        lambda: SeemoRe(num_experts=1, lr_space="linear"),
        lambda: SeemoRe(num_experts=4, lr_space="double"),
        lambda: SeemoRe(num_experts=2, lr_space="double"),
        lambda: SeemoRe(num_experts=3, lr_space="exp"),
        # detect official configs
        # T
        lambda: SeemoRe(
            scale=2,
            in_chans=3,
            num_experts=3,
            img_range=1,
            num_layers=6,
            embedding_dim=36,
            use_shuffle=True,
            lr_space="exp",
            topk=1,
            recursive=2,
            global_kernel_size=11,
        ),
        # B
        lambda: SeemoRe(
            scale=2,
            in_chans=3,
            num_experts=3,
            img_range=1,
            num_layers=8,
            embedding_dim=48,
            use_shuffle=True,
            lr_space="exp",
            topk=1,
            recursive=2,
            global_kernel_size=11,
        ),
        # L
        lambda: SeemoRe(
            scale=2,
            in_chans=3,
            num_experts=3,
            img_range=1,
            num_layers=16,
            embedding_dim=48,
            use_shuffle=False,
            lr_space="exp",
            topk=1,
            recursive=1,
            global_kernel_size=11,
        ),
    )


def test_size_requirements():
    file = ModelFile.from_url(
        "https://drive.google.com/file/d/1TLq5C2URWCvrI3HKsTxvwN4M5GpYj8rc/view?usp=drive_link",
        name="SeemoRe_T_X2.pth",
    )
    assert_size_requirements(file.load_model())


def test_SeemoRe_T_X2(snapshot):
    file = ModelFile.from_url(
        "https://drive.google.com/file/d/1TLq5C2URWCvrI3HKsTxvwN4M5GpYj8rc/view?usp=drive_link",
        name="SeemoRe_T_X2.pth",
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, SeemoRe)
    assert_image_inference(
        file, model, [TestImage.SR_16, TestImage.SR_32, TestImage.SR_64]
    )


def test_SeemoRe_T_X4(snapshot):
    file = ModelFile.from_url(
        "https://drive.google.com/file/d/1gX6H7jmbE4GwC0OIyRFrbfu5wGSfUpX4/view?usp=drive_link",
        name="SeemoRe_T_X4.pth",
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, SeemoRe)
    assert_image_inference(
        file, model, [TestImage.SR_16, TestImage.SR_32, TestImage.SR_64]
    )
