"""Visual tokens for unchanged NextTransformer language models.

A multimodal model here is a NextTransformer model of any registered method plus one visual projector.
Every method reads token ids only through `transformer.wte` and the `value_embeds` tables, so two kinds
of forward hook are enough to feed images to all of them without changing their code:

- the token-embedding hook replaces the embedding at image positions with projected encoder features;
- each value-embedding hook zeroes the value embedding at image positions (a placeholder id carries no
  meaning, so it must not select a learned value vector).

With no image positions the hooks change nothing, so a text-only batch reproduces the language model
exactly. The projector always runs, on zero rows when a batch has no images, so every trainable
parameter receives a gradient at every update, as the trainer requires.
"""
from contextlib import contextmanager
from dataclasses import replace

import torch
from torch import nn
from torch._subclasses.fake_tensor import FakeTensor
from torch.nn import functional as F

from nexttransformer.models.backbone import Linear, norm
from nexttransformer.models.lm import build_model
from nexttransformer.protocol.spec import config_path, load_config

# SigLIP2 so400m patch features (1152) after a 2 x 2 spatial merge.
VISUAL_DIM = 4 * 1152


class VisualProjector(nn.Module):
    """Merged encoder features (N, in_dim) -> model width, with the shell FFN's ReLU^2."""

    def __init__(self, in_dim, width):
        super().__init__()
        self.c_fc = Linear(in_dim, width, bias=False)
        self.c_proj = Linear(width, width, bias=False)

    def forward(self, features):
        return self.c_proj(F.relu(self.c_fc(norm(features))).square())

    @torch.no_grad()
    def init_weights(self, generator):
        # The embedding path RMS-normalizes the result, so only the shape of the map matters here. A zero
        # output projection (as in the blocks) would make that normalization divide by ~0, so both
        # matrices use the shell's uniform init with std 1/sqrt(fan_in).
        for linear in (self.c_fc, self.c_proj):
            bound = 3 ** .5 * linear.in_features ** -.5
            linear.weight.copy_(torch.empty(linear.weight.shape).uniform_(-bound, bound, generator=generator))


def _embedding_hook(model):
    def hook(module, args, output):
        state = model._visual
        if state is None:
            return output
        mask, features = state
        projected = model.visual_projector(features).to(output.dtype)
        return output.masked_scatter(mask.unsqueeze(-1), projected)
    return hook


def _value_hook(model):
    def hook(module, args, output):
        state = model._visual
        if state is None:
            return output
        return output.masked_fill(state[0].unsqueeze(-1), 0)
    return hook


def attach_visual(model, visual_dim=VISUAL_DIM, seed=42):
    """Add a projector and the embedding hooks to a model built by NextTransformer's `build_model`.

    The projector is created on the meta device and initialized from its own generator, so attaching it
    does not consume the global RNG stream the language model's own initialization and routing use.
    """
    device = model.lm_head.weight.device
    with torch.device("meta"):
        projector = VisualProjector(visual_dim, model.config.width)
    projector.to_empty(device=device)
    projector.init_weights(torch.Generator().manual_seed(seed + 1))
    model.visual_projector = projector
    model.visual_dim = visual_dim
    model._visual = None
    model.transformer.wte.register_forward_hook(_embedding_hook(model))
    for table in model.value_embeds.values():
        table.register_forward_hook(_value_hook(model))
    return model


def build_mm_model(config, device="cpu", seed=42, visual_dim=VISUAL_DIM, spatial=None):
    """`spatial`: optional dict(order=..., position=..., conv2d=...) for ntv.spatial.attach_spatial."""
    model = attach_visual(build_model(config, device, seed), visual_dim, seed)
    if spatial:
        from ntv.spatial import attach_spatial
        attach_spatial(model, seed=seed, **spatial)
    return model


@contextmanager
def visual_inputs(model, image_mask, features, index=None, grid=None):
    """Make (image_mask, features) visible to the hooks for one forward call.

    image_mask: (B, T) bool, True at image positions; each image is a contiguous run of grid x grid positions.
    features:   (image_mask.sum(), visual_dim): each image's tokens in row-major grid order (the encoder's
                order); a spatial patch order, if configured, is applied here.
    index:      flat indices of the image positions; computed from image_mask when omitted (pass it for
                shape-only audits, where it cannot be computed).
    grid:       patches per image side; the track trains with 12 (144 tokens), larger grids are for
                zero-shot resolution transfer at evaluation.
    `model` is the unwrapped module (pass `ddp.module` when training with DDP).
    """
    if image_mask.dtype != torch.bool or image_mask.ndim != 2:
        raise ValueError("image_mask must be a (B, T) bool tensor")
    if features.ndim != 2 or features.shape[1] != model.visual_dim:
        raise ValueError(f"features must be (N, {model.visual_dim})")
    from ntv.spatial import GRID, grid_orders, rotary_tables
    grid = grid or GRID
    shape_only = isinstance(image_mask, FakeTensor) or isinstance(features, FakeTensor)
    if not shape_only and features.shape[0] != int(image_mask.sum()):
        raise ValueError("one feature row is required per image position")
    if model._visual is not None:
        raise RuntimeError("visual inputs are already set")
    spatial = getattr(model, "spatial", None)
    swapped = None
    if spatial is not None:
        if not shape_only and features.shape[0] % (grid * grid):
            raise ValueError(f"image features must come in whole {grid} x {grid} images")
        order = grid_orders(model, grid)
        features = features.view(-1, grid * grid, features.shape[-1])[:, order[0]].reshape(-1, features.shape[-1])
        model._grid, model._active_order = grid, order
        model._visual_index = image_mask.reshape(-1).nonzero().squeeze(1) if index is None else index
        if spatial["position"] != "rope1d" and not shape_only:
            swapped = (model.cos, model.sin)
            model.cos, model.sin = rotary_tables(model, image_mask, order[0], spatial["position"], grid=grid)
    model._visual = (image_mask, features)
    try:
        yield
    finally:
        model._visual = None
        if spatial is not None:
            model._visual_index = None
            model._grid, model._active_order = GRID, grid_orders(model, GRID)
        if swapped is not None:
            model.cos, model.sin = swapped


def text_only(model, idx):
    """Empty visual inputs for a batch without images (the projector still runs, on zero rows)."""
    p = model.visual_projector.c_fc.weight
    return visual_inputs(model, torch.zeros(idx.shape, dtype=torch.bool, device=idx.device),
                         torch.zeros((0, model.visual_dim), dtype=p.dtype, device=p.device))


def track_config(method, sequence_len=8192, root=None, **overrides):
    """A method's admitted NextTransformer configuration at the track's context length.

    NextTransformer's `scale="formal"` guard pins the context to 2,048. Model code reads `scale` only in
    that validation and in the formal-evaluation guards, so the track uses `scale="debug"` to lift the
    guard; the model's computation is unchanged. Calibration may then change depth and FFN width.
    """
    config = load_config(config_path(method) if root is None else config_path(method, root))
    return replace(config, scale="debug", sequence_len=sequence_len, **overrides)
