import torch
import torch.nn.functional as F


def _next_power_of_two(n: int):
    return 1 if n == 0 else 2 ** (n - 1).bit_length()

def fast_hadamard_transform(x: torch.Tensor):
    original_dim = x.shape[-1]
    padded_dim = _next_power_of_two(original_dim)

    # ✅ FIXED: cast to float32 for numerical stability
    # float16 overflows during butterfly additions at large dims (4096, 14336)
    original_dtype = x.dtype
    x = x.float()

    if padded_dim != original_dim:
        pad = padded_dim - original_dim
        x = F.pad(x, (0, pad))

    d = padded_dim
    orig_shape = x.shape
    x = x.reshape(-1, d)

    h = 1
    while h < d:
        x = x.reshape(-1, d // (2 * h), 2 * h)
        x_l, x_r = x.chunk(2, dim=-1)
        x = torch.cat([x_l + x_r, x_l - x_r], dim=-1)
        h *= 2

    x = x.reshape(orig_shape)
    x = x / (d ** 0.5)

    if padded_dim != original_dim:
        x = x[..., :original_dim]

    # ✅ cast back to original dtype
    return x.to(original_dtype)