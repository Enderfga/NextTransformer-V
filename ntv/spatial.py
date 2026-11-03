"""How a backbone sees the 2D layout of image tokens: patch order, attention position mode, 2D short convolution.

All three act on the image tokens of a batch (12 x 12 merged patches per image) without changing any method's code:

- Patch order: the order in which an image's patches enter the sequence (raster, snake, or a generalized Hilbert
  curve that keeps every pair of consecutive patches spatially adjacent).
- Position mode of the attention layers: `rope1d` (NextTransformer's RoPE over the flattened sequence), `nope`
  (no rotary position), or `rope2d`, the RoPE-TV layout from kexue.fm/archives/10352: text token n sits at (n, n),
  which reduces exactly to RoPE-1D, and patch (i, j) of an image that follows text position L sits at
  (L + (wh - h)/2 + i + 1, L + (wh - w)/2 + j + 1). Even frequency pairs read the first coordinate, odd pairs the
  second.
- 2D short convolution: the causal 1D short convolution of linear-attention mixers lets the online "key -> value"
  learning use the left neighbour (kexue.fm/archives/11320); in raster order an image patch never sees the patch
  above it. At image positions this replaces the convolution with a 3 x 3 depthwise convolution over the patch grid.
  Image positions carry no loss, so the 2D convolution may look at all neighbours of the same image.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

GRID = 12
TOKENS = GRID * GRID
ORDERS = ("raster", "snake", "hilbert")
POSITIONS = ("rope1d", "nope", "rope2d")


def gilbert(width, height):
    """Generalized Hilbert curve over a width x height grid; yields (x, y) cells, each step to a neighbour."""
    def sign(v):
        return (v > 0) - (v < 0)

    def walk(x, y, ax, ay, bx, by):
        w, h = abs(ax + ay), abs(bx + by)
        dax, day, dbx, dby = sign(ax), sign(ay), sign(bx), sign(by)
        if h == 1:
            for _ in range(w):
                yield x, y
                x, y = x + dax, y + day
            return
        if w == 1:
            for _ in range(h):
                yield x, y
                x, y = x + dbx, y + dby
            return
        ax2, ay2, bx2, by2 = ax // 2, ay // 2, bx // 2, by // 2
        w2, h2 = abs(ax2 + ay2), abs(bx2 + by2)
        if 2 * w > 3 * h:
            if w2 % 2 and w > 2:
                ax2, ay2 = ax2 + dax, ay2 + day
            yield from walk(x, y, ax2, ay2, bx, by)
            yield from walk(x + ax2, y + ay2, ax - ax2, ay - ay2, bx, by)
        else:
            if h2 % 2 and h > 2:
                bx2, by2 = bx2 + dbx, by2 + dby
            yield from walk(x, y, bx2, by2, ax2, ay2)
            yield from walk(x + bx2, y + by2, ax, ay, bx - bx2, by - by2)
            yield from walk(x + (ax - dax) + (bx2 - dbx), y + (ay - day) + (by2 - dby),
                            -bx2, -by2, -(ax - ax2), -(ay - ay2))

    if width >= height:
        yield from walk(0, 0, width, 0, 0, height)
    else:
        yield from walk(0, 0, 0, width, height, 0)


def patch_order(name, grid=GRID):
    """order[p] = row-major grid cell placed at sequence slot p of an image."""
    if name == "raster":
        cells = list(range(grid * grid))
    elif name == "snake":
        cells = [r * grid + (c if r % 2 == 0 else grid - 1 - c) for r in range(grid) for c in range(grid)]
    elif name == "hilbert":
        cells = [y * grid + x for x, y in gilbert(grid, grid)]
    else:
        raise ValueError(f"unknown patch order {name!r}")
    return torch.tensor(cells, dtype=torch.long)


def rotary_tables(model, image_mask, order, mode, base=100000):
    """Per-row cos/sin of shape (B, T, 1, head_dim / 2) for `nope` and `rope2d` (`rope1d` keeps the model's)."""
    b, t = image_mask.shape
    half = model.cos.shape[-1]
    device = image_mask.device
    if mode == "nope":
        ones = torch.ones((1, t, 1, half), device=device, dtype=model.cos.dtype)
        return ones, torch.zeros_like(ones)
    inv_freq = 1.0 / (base ** (torch.arange(0, 2 * half, 2, dtype=torch.float32, device=device) / (2 * half)))
    px = torch.arange(t, dtype=torch.float32, device=device).expand(b, t).clone()
    py = px.clone()
    flat = image_mask.reshape(-1).nonzero().squeeze(1)
    if flat.numel():
        slot = torch.arange(flat.numel(), device=device) % TOKENS
        start = (flat - slot) % t                         # sequence index of each image's first patch
        cell = order.to(device)[slot]
        row, col = (cell // GRID).float(), (cell % GRID).float()
        offset = (TOKENS - GRID) / 2                      # (wh - h) / 2 = (wh - w) / 2 for a square grid
        px.view(-1)[flat] = (start - 1).float() + offset + row + 1
        py.view(-1)[flat] = (start - 1).float() + offset + col + 1
    even = (torch.arange(half, device=device) % 2 == 0)
    angle = torch.where(even, px[..., None] * inv_freq, py[..., None] * inv_freq)
    return angle.cos().to(model.cos.dtype)[:, :, None], angle.sin().to(model.cos.dtype)[:, :, None]


class ImageConv2d(nn.Module):
    """Depthwise k x k convolution over the patch grid of each image, same padding (all neighbours)."""

    def __init__(self, width, size=3):
        super().__init__()
        self.size = size
        self.weight = nn.Parameter(torch.empty(width, 1, size, size))

    @torch.no_grad()
    def init_weights(self, generator):
        self.weight.copy_(torch.empty(self.weight.shape).normal_(0, 1 / self.size, generator=generator))

    def forward(self, grid):
        """grid: (N, D, GRID, GRID) -> same shape."""
        return F.conv2d(grid, self.weight.to(grid.dtype), padding=self.size // 2, groups=grid.shape[1])


def _conv_hook(model, conv2d):
    def hook(module, args, output):
        state = model._visual
        if state is None:
            return output
        mask, index = state[0], model._visual_index
        if index.numel() == 0:  # no images: keep the output exactly (layout included) as the language model's
            return output
        x = args[0]
        b, t, d = x.shape
        n = index.numel() // TOKENS
        tokens = x.reshape(b * t, d).index_select(0, index).view(n, TOKENS, d)
        grid = tokens[:, model._inverse_order].view(n, GRID, GRID, d).permute(0, 3, 1, 2)
        mixed = conv2d(grid).permute(0, 2, 3, 1).reshape(n, TOKENS, d)[:, model._order]
        return output.masked_scatter(mask.unsqueeze(-1), F.silu(mixed).to(output.dtype))
    return hook


def attach_spatial(model, order="raster", position="rope1d", conv2d="none", seed=42):
    """Configure patch order, attention position mode and optional 2D short convolution of a multimodal model."""
    if order not in ORDERS or position not in POSITIONS or conv2d not in ("none", "k", "qkv"):
        raise ValueError("unknown spatial setting")
    device = model.lm_head.weight.device
    model.spatial = dict(order=order, position=position, conv2d=conv2d)
    model._order = patch_order(order).to(device)
    model._inverse_order = torch.argsort(model._order)
    model._visual_index = None
    if conv2d != "none":
        from nexttransformer.models.linear import LinearMixer
        names = ("k_conv",) if conv2d == "k" else ("q_conv", "k_conv", "v_conv")
        convs = nn.ModuleDict()
        generator = torch.Generator().manual_seed(seed + 2)
        for i, block in enumerate(model.transformer.h):
            mixer = getattr(block, "attn", None)
            if not isinstance(mixer, LinearMixer) or getattr(mixer, "k_conv", None) is None:
                continue
            for name in names:
                source = getattr(mixer, name)
                with torch.device("meta"):
                    conv = ImageConv2d(source.weight.shape[0])
                conv.to_empty(device=device)
                conv.init_weights(generator)
                convs[f"{i}_{name}"] = conv
                source.register_forward_hook(_conv_hook(model, conv))
        if not len(convs):
            raise ValueError(f"{model.config.method} has no linear-attention short convolutions")
        model.image_convs = convs
    return model


class RawPatches(nn.Module):
    """'Lossless' input: each 32 x 32 RGB patch of the 384 px image is one token (3,072 values in [-1, 1])."""
    dtype = torch.bfloat16

    def __init__(self, path=None, device="cpu", dtype=torch.bfloat16):
        super().__init__()
        self.dtype = dtype

    @torch.no_grad()
    def forward(self, images):
        n = images.shape[0]
        size = images.shape[-1] // GRID
        pixels = images.to(self.dtype).div(127.5).sub(1.)
        patches = pixels.view(n, 3, GRID, size, GRID, size).permute(0, 2, 4, 1, 3, 5)
        return patches.reshape(n * TOKENS, 3 * size * size)


RAW_DIM = 3 * 32 * 32
