from spandrel.architectures.SGNet import SGNet, load

from .util import ModelFile, assert_loads_correctly, disallowed_props


def test_SGNet_load():
    assert_loads_correctly(
        load,
        lambda: SGNet(num_feats=80, kernel_size=3, scale=16),
        lambda: SGNet(num_feats=40, kernel_size=3, scale=8),
        lambda: SGNet(num_feats=64, kernel_size=5, scale=4),
        lambda: SGNet(num_feats=20, kernel_size=1, scale=2),
    )


def test_SGNet_Real(snapshot):
    file = ModelFile.from_url(
        "https://drive.google.com/file/d/1AGZRF9CiPj0yRZuJBcCIr-0S_2Wmn3gJ/view?usp=sharing",
        name="SGNet_Real.pth",
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, SGNet)


# def test_SGNet_X4(snapshot):
#     file = ModelFile.from_url(
#         "https://drive.google.com/file/d/1qmWbKm1nrfGc-TNFLJp4IpQ4NWWXxcc6/view?usp=sharing",
#         name="SGNet_X4.pth",
#     )
#     model = file.load_model()
#     assert model == snapshot(exclude=disallowed_props)
#     assert isinstance(model.model, SGNet)


# def test_SGNet_X8(snapshot):
#     file = ModelFile.from_url(
#         "https://drive.google.com/file/d/1Rn9PWxyRdVxdnceo7X_QiUrqpFzrERW4/view?usp=sharing",
#         name="SGNet_X8.pth",
#     )
#     model = file.load_model()
#     assert model == snapshot(exclude=disallowed_props)
#     assert isinstance(model.model, SGNet)


# def test_SGNet_X16(snapshot):
#     file = ModelFile.from_url(
#         "https://drive.google.com/file/d/1KqxhMcP3AJzk1U9TjjZz_XZCm9Kb_Vh9/view?usp=sharing",
#         name="SGNet_X16.pth",
#     )
#     model = file.load_model()
#     assert model == snapshot(exclude=disallowed_props)
#     assert isinstance(model.model, SGNet)
