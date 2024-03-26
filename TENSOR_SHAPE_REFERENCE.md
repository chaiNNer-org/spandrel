# Tensor shape reference

This document contains a listing of commonly-used modules, how their parameters affect the shape of their weights and biases, and how to derive their parameters from their the shape of their weights and biases.

## `torch.nn`

### Modules without tensors

The following modules do not have tensors that are stored in the state dict:

- `torch.nn.AdaptiveAvgPool2d`
- `torch.nn.AdaptiveMaxPool2d`
- `torch.nn.CELU`
- `torch.nn.Dropout`
- `torch.nn.ELU`
- `torch.nn.GELU`
- `torch.nn.GLU`
- `torch.nn.Identity`
- `torch.nn.LeakyReLU`
- `torch.nn.PixelShuffle`
- `torch.nn.PixelUnshuffle`
- `torch.nn.ReLU`
- `torch.nn.SELU`
- `torch.nn.Sigmoid`
- `torch.nn.Softmax`
- `torch.nn.Tanh`

### `torch.nn.BatchNorm2d`

All of the listed parameters can be deduced from the state dict.

```py
p = nn.BatchNorm2d(num_features=N, affine=True, track_running_stats=True)

# p.weight:              Tensor Size([N])
# p.bias:                Tensor Size([N])
# p.running_mean:        Tensor Size([N])
# p.running_var:         Tensor Size([N])
# p.num_batches_tracked: Tensor Size([])
```

```py
p = nn.BatchNorm2d(num_features=N, affine=True, track_running_stats=False)

# p.weight: Tensor Size([N])
# p.bias:   Tensor Size([N])
```

```py
p = nn.BatchNorm2d(num_features=N, affine=False, track_running_stats=False)

# nothing is stored in state dict
```

### `torch.nn.Conv2d`

All of the listed parameters can be deduced from the state dict.

```py
p = nn.Conv2d(in_channels=I, out_channels=O, kernel_size=K, bias=True)

# p.weight: Tensor Size([O, I, K, K])
# p.bias:   Tensor Size([O])
```

```py
p = nn.Conv2d(in_channels=I, out_channels=O, kernel_size=K, bias=False)

# p.weight: Tensor Size([O, I, K, K])
```

```py
p = nn.Conv2d(in_channels=I, out_channels=O, kernel_size=(K1, K2), bias=True)

# p.weight: Tensor Size([O, I, K1, K2])
# p.bias:   Tensor Size([O])
```

```py
assert I % G == 0 and O % G == 0
p = nn.Conv2d(in_channels=I, out_channels=O, kernel_size=(K1, K2), group=G, bias=True)

# p.weight: Tensor Size([O, I/G, K1, K2])
# p.bias:   Tensor Size([O])
```

### `torch.nn.Embedding`

All of the listed parameters can be deduced from the state dict.

```py
p = nn.Embedding(num_embeddings=N, embedding_dim=D)

# p.weight: Tensor Size([N, D])
```

### `torch.nn.Linear`

All of the listed parameters can be deduced from the state dict.

```py
p = nn.Linear(in_features=I, out_features=O, bias=True)

# p.weight: Tensor Size([O, I])
# p.bias:   Tensor Size([O])
```

```py
p = p = nn.Linear(in_features=I, out_features=O, bias=False)

# p.weight: Tensor Size([O, I])
```

### `torch.nn.MultiheadAttention`

All of the listed parameters except for `num_heads` can be deduced from the state dict. The only thing known about `num_heads` is that it's a factor of `embed_dim`.

```py
assert D % H == 0
p = nn.MultiheadAttention(embed_dim=D, num_heads=H, bias=True)

# p.in_proj_weight:  Tensor Size([3*D, D])
# p.in_proj_bias:    Tensor Size([3*D])
# p.out_proj.weight: Tensor Size([D, D])
# p.out_proj.bias:   Tensor Size([D])
```

```py
assert D % H == 0
p = nn.MultiheadAttention(embed_dim=D, num_heads=H, bias=False)

# p.in_proj_weight:  Tensor Size([3*D, D])
# p.out_proj.weight: Tensor Size([D, D])
```

```py
assert D % H == 0
p = nn.MultiheadAttention(embed_dim=D, num_heads=H, bias=True, add_bias_kv=True, kdim=K, vdim=V)

# p.q_proj_weight:   Tensor Size([D, D])
# p.k_proj_weight:   Tensor Size([D, K])
# p.v_proj_weight:   Tensor Size([D, V])
# p.in_proj_bias:    Tensor Size([3*D])
# p.bias_k:          Tensor Size([1, 1, D])
# p.bias_v:          Tensor Size([1, 1, D])
# p.out_proj.weight: Tensor Size([D, D])
# p.out_proj.bias:   Tensor Size([D])
```

```py
assert D % H == 0
p = nn.MultiheadAttention(embed_dim=D, num_heads=H, bias=True, add_bias_kv=False, kdim=K, vdim=V)

# p.q_proj_weight:   Tensor Size([D, D])
# p.k_proj_weight:   Tensor Size([D, K])
# p.v_proj_weight:   Tensor Size([D, V])
# p.in_proj_bias:    Tensor Size([3*D])
# p.out_proj.weight: Tensor Size([D, D])
# p.out_proj.bias:   Tensor Size([D])
```

```py
assert D % H == 0
p = nn.MultiheadAttention(embed_dim=D, num_heads=H, bias=False, add_bias_kv=False, kdim=K, vdim=V)

# p.q_proj_weight:   Tensor Size([D, D])
# p.k_proj_weight:   Tensor Size([D, K])
# p.v_proj_weight:   Tensor Size([D, V])
# p.out_proj.weight: Tensor Size([D, D])
```
