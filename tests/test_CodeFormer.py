from spandrel.architectures.CodeFormer import CodeFormer, load

from .util import ModelFile, assert_loads_correctly, disallowed_props


def test_CodeFormer_load():
    assert_loads_correctly(
        load,
        lambda: CodeFormer(),
        lambda: CodeFormer(dim_embd=256, n_head=4),
        lambda: CodeFormer(n_layers=5, codebook_size=512, latent_size=64),
        lambda: CodeFormer(connect_list=["16", "32", "64"]),
        condition=lambda a, b: (
            a.connect_list == b.connect_list
            and a.dim_embd == b.dim_embd
            and a.n_layers == b.n_layers
            and a.codebook_size == b.codebook_size
        ),
    )


def test_CodeFormer(snapshot):
    file = ModelFile.from_url(
        "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth"
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, CodeFormer)
