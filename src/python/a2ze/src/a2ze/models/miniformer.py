"""
The following code has been modified from https://github.com/lucidrains/enformer-pytorch
"""

from collections import OrderedDict
import math
from typing import Optional

import einops
import einops.layers.torch
import torch


def exponential_linspace_int(start, end, num, divisible_by = 1):
    def _round(x):
        return int(round(x / divisible_by) * divisible_by)

    base = math.exp(math.log(end / start) / (num - 1))
    return [_round(start * base**i) for i in range(num)]


def get_positional_features_exponential(positions, features, seq_len, min_half_life = 3.):
    max_range = math.log(seq_len) / math.log(2.)
    half_life = 2 ** torch.linspace(min_half_life, max_range, features, device = positions.device)
    half_life = half_life[None, ...]
    positions = positions.abs()[..., None]
    return torch.exp(-math.log(2.) / half_life * positions)


def get_positional_features_central_mask(positions, features, seq_len):
    center_widths = 2 ** torch.arange(1, features + 1, device = positions.device).float()
    center_widths = center_widths - 1
    return (center_widths[None, ...] > positions.abs()[..., None]).float()


def gamma_pdf(x, concentration, rate):
    log_unnormalized_prob = torch.xlogy(concentration - 1., x) - rate * x
    log_normalization = (torch.lgamma(concentration) - concentration * torch.log(rate))
    return torch.exp(log_unnormalized_prob - log_normalization)


def get_positional_features_gamma(positions, features, seq_len, stddev = None, start_mean = None, eps = 1e-8):
    stddev = seq_len / (2 * features) if stddev is None else stddev

    start_mean = seq_len / features if start_mean is None else start_mean

    mean = torch.linspace(start_mean, seq_len, features, device = positions.device)
    mean = mean[None, ...]
    concentration = (mean / stddev) ** 2
    rate = mean / stddev ** 2
    probabilities = gamma_pdf(positions.float().abs()[..., None], concentration, rate)
    probabilities = probabilities + eps
    outputs = probabilities / torch.amax(probabilities, dim = -1, keepdim = True)
    return outputs


def get_positional_embed(seq_len, feature_size, device):
    distances = torch.arange(-seq_len + 1, seq_len, device = device)

    feature_functions = [
        get_positional_features_exponential,
        get_positional_features_central_mask,
        get_positional_features_gamma
    ]

    num_components = len(feature_functions) * 2

    if (feature_size % num_components) != 0:
        raise ValueError(f'feature size ({feature_size}) is not divisible by number of components ({num_components})')

    num_basis_per_class = feature_size // num_components

    embeddings = []
    for fn in feature_functions:
        embeddings.append(fn(distances, num_basis_per_class, seq_len))

    embeddings = torch.cat(embeddings, dim = -1)
    embeddings = torch.cat((embeddings, torch.sign(distances)[..., None] * embeddings), dim = -1)
    return embeddings


def relative_shift(x):
    to_pad = torch.zeros_like(x[..., :1])
    x = torch.cat((to_pad, x), dim = -1)
    _, h, t1, t2 = x.shape
    x = x.reshape(-1, h, t2, t1)
    x = x[:, :, 1:, :]
    x = x.reshape(-1, h, t1, t2 - 1)
    return x[..., :((t2 + 1) // 2)]


class Residual(torch.nn.Module):
    def __init__(self, fn):
        super().__init__()

        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class AttentionPool(torch.nn.Module):
    def __init__(self, dim, pool_size = 2):
        super().__init__()

        self.pool_size = pool_size
        self.pool_fn = einops.layers.torch.Rearrange('b d (n p) -> b d n p', p = pool_size)

        self.to_attn_logits = torch.nn.Conv2d(dim, dim, 1, bias = False)

        torch.nn.init.dirac_(self.to_attn_logits.weight)

        with torch.no_grad():
            self.to_attn_logits.weight.mul_(2)

    def forward(self, x):
        b, _, n = x.shape
        remainder = n % self.pool_size
        needs_padding = remainder > 0

        if needs_padding:
            x = torch.nn.functional.pad(x, (0, remainder), value = 0)
            mask = torch.zeros((b, 1, n), dtype = torch.bool, device = x.device)
            mask = torch.nn.functional.pad(mask, (0, remainder), value = True)

        x = self.pool_fn(x)
        logits = self.to_attn_logits(x)

        if needs_padding:
            mask_value = -torch.finfo(logits.dtype).max
            logits = logits.masked_fill(self.pool_fn(mask), mask_value)

        attn = logits.softmax(dim = -1)

        return (x * attn).sum(dim = -1)


class ConvBlock(torch.nn.Module):
    def __init__(self, dim: int, dim_out: Optional[int] = None, kernel_size: int = 1):
        super().__init__()

        self.bn = torch.nn.BatchNorm1d(num_features = dim)
        self.conv = torch.nn.Conv1d(
            in_channels = dim,
            out_channels = dim_out if dim_out is not None else dim,
            kernel_size = kernel_size,
            padding = kernel_size // 2
        )

    def forward(self, x: torch.Tensor):
        x = torch.nn.functional.gelu(self.bn(x))
        return self.conv(x)


class Attention(torch.nn.Module):
    def __init__(
        self,
        dim,
        *,
        num_rel_pos_features,
        heads = 8,
        dim_key = 64,
        dim_value = 64,
        dropout = 0.,
        pos_dropout = 0.
    ):
        super().__init__()
        self.scale = dim_key ** -0.5
        self.heads = heads

        self.to_q = torch.nn.Linear(dim, dim_key * heads, bias = False)
        self.to_k = torch.nn.Linear(dim, dim_key * heads, bias = False)
        self.to_v = torch.nn.Linear(dim, dim_value * heads, bias = False)

        self.to_out = torch.nn.Linear(dim_value * heads, dim)
        torch.nn.init.zeros_(self.to_out.weight)
        torch.nn.init.zeros_(self.to_out.bias)

        # relative positional encoding

        self.num_rel_pos_features = num_rel_pos_features

        self.to_rel_k = torch.nn.Linear(num_rel_pos_features, dim_key * heads, bias = False)
        self.rel_content_bias = torch.nn.Parameter(torch.randn(1, heads, 1, dim_key))
        self.rel_pos_bias = torch.nn.Parameter(torch.randn(1, heads, 1, dim_key))

        # dropouts

        self.pos_dropout = torch.nn.Dropout(pos_dropout)
        self.attn_dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        n, h, device = x.shape[-2], self.heads, x.device

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q, k, v = map(lambda t: einops.rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        q = q * self.scale

        content_logits = torch.einsum('b h i d, b h j d -> b h i j', q + self.rel_content_bias, k)

        positions = get_positional_embed(n, self.num_rel_pos_features, device)
        positions = self.pos_dropout(positions)
        rel_k = self.to_rel_k(positions)

        rel_k = einops.rearrange(rel_k, 'n (h d) -> h n d', h = h)
        rel_logits = torch.einsum('b h i d, h j d -> b h i j', q + self.rel_pos_bias, rel_k)
        rel_logits = relative_shift(rel_logits)

        logits = content_logits + rel_logits
        attn = logits.softmax(dim = -1)
        attn = self.attn_dropout(attn)

        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = einops.rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Miniformer(torch.nn.Module):
    """
    Scaled-down Enformer based on enformer-pytorch implementation, with non-binned output.
    """

    def __init__(
        self,
        dim: int = 120,
        num_downsamples: int = 3,
        depth: int = 3,
        heads: int = 2,
        attn_dim_key: int = 64,
        attn_dropout_rate: float = 0.05,
        pos_dropout_rate: float = 0.01,
        dropout_rate: float = 0.4,
        n_out_features: int = 1
    ):
        if dim % 12 != 0:
            raise ValueError("Model dimensions must be divisible by 12")

        super().__init__()

        self.stem = torch.nn.Sequential(OrderedDict((
            ('conv1', torch.nn.Conv1d(
                in_channels = 4,
                out_channels = dim // 2,
                kernel_size = 15,
                padding = 7
            )),
            ('conv_block1', Residual(ConvBlock(
                dim = dim // 2
            ))),
            ('attn_pool1', AttentionPool(
                dim = dim // 2,
                pool_size = 2
            ))
        )))

        conv_tower_filter_dims = exponential_linspace_int(
            start = dim // 2,
            end = dim,
            num = num_downsamples - 1,
            divisible_by = 6
        )
        conv_tower_filter_dims = [dim // 2, *conv_tower_filter_dims]

        conv_layers = []
        for dim_in, dim_out in zip(conv_tower_filter_dims[:-1], conv_tower_filter_dims[1:]):
            conv_layers.append(torch.nn.Sequential(
                ConvBlock(dim_in, dim_out, kernel_size = 5),
                Residual(ConvBlock(dim_out, dim_out, 1)),
                AttentionPool(dim_out, pool_size = 2)
            ))

        self.conv_tower = torch.nn.Sequential(*conv_layers)

        transformer = []
        for _ in range(depth):
            transformer.append(torch.nn.Sequential(
                Residual(torch.nn.Sequential(
                    torch.nn.LayerNorm(dim),
                    Attention(
                        dim = dim,
                        heads = heads,
                        dim_key = attn_dim_key,
                        dim_value = dim // heads,
                        dropout = attn_dropout_rate,
                        pos_dropout = pos_dropout_rate,
                        num_rel_pos_features = dim // heads
                    ),
                    torch.nn.Dropout(dropout_rate)
                )),
                Residual(torch.nn.Sequential(
                    torch.nn.LayerNorm(dim),
                    torch.nn.Linear(dim, dim * 2),
                    torch.nn.Dropout(dropout_rate),
                    torch.nn.ReLU(),
                    torch.nn.Linear(dim * 2, dim),
                    torch.nn.Dropout(dropout_rate)
                ))
            ))

        self.transformer = torch.nn.Sequential(*transformer)

        self.final_pointwise = torch.nn.Sequential(
            einops.layers.torch.Rearrange('b n d -> b d n'),
            ConvBlock(conv_tower_filter_dims[-1], dim * 2, 1),
            einops.layers.torch.Rearrange('b d n -> b n d'),
            torch.nn.Dropout(dropout_rate / 8),
            torch.nn.GELU()
        )

        self.head = torch.nn.Linear(dim * 2, n_out_features)

        self.trunk = torch.nn.Sequential(
            self.stem,
            self.conv_tower,
            einops.layers.torch.Rearrange('b d n -> b n d'),
            self.transformer,
            self.final_pointwise
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.trunk(x)
        x = self.head(x).sum(dim = -2)

        return x
