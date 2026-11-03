"""Patch orders, attention position modes, the 2D short convolution and raw patches."""
from dataclasses import replace
import math

import pytest
import torch

from nexttransformer.models.config import debug_config
from ntv.cost import Track, audit_mm
from ntv.model import build_mm_model, text_only, visual_inputs
from ntv.spatial import GRID, ORDERS, RAW_DIM, TOKENS, RawPatches, patch_order, rotary_tables

DIM = 20
T = 400


@pytest.fixture(autouse=True)
def reproducible():
    torch.manual_seed(42)
    torch.set_num_threads(1)


def activate(model):
    with torch.no_grad():
        for block in model.transformer.h:
            block.attn.c_proj.weight.normal_(std=.03)
            block.mlp.c_proj.weight.normal_(std=.03)


def build(method, **spatial):
    c = debug_config(method, sequence_len=T)
    model = build_mm_model(c, visual_dim=DIM, spatial=spatial or None)
    activate(model)
    model.set_step(c.burnin_steps + 1)
    return c, model


def twin(method, **spatial):
    """The same language-model weights without and with the spatial settings."""
    c, base = build(method)
    _, other = build(method, **spatial)
    other.load_state_dict({**other.state_dict(), **base.state_dict()})
    return c, base, other


def one_image(start=5, batch=2):
    mask = torch.zeros((batch, T), dtype=torch.bool)
    mask[:, start:start + TOKENS] = True
    return mask, torch.randn(int(mask.sum()), DIM)


@pytest.mark.parametrize("name", ORDERS)
def test_orders_are_permutations_and_curves_step_to_neighbours(name):
    order = patch_order(name)
    assert sorted(order.tolist()) == list(range(TOKENS))
    rows, cols = order // GRID, order % GRID
    steps = (rows.diff().abs() + cols.diff().abs()).tolist()
    if name == "raster":
        assert torch.equal(order, torch.arange(TOKENS))
    else:
        assert all(step == 1 for step in steps)


@pytest.mark.parametrize("method", ["dense_full", "gdn_hybrid"])
def test_rope2d_equals_rope1d_on_text(method):
    c, base, other = twin(method, position="rope2d")
    idx = torch.randint(c.vocab_size, (2, T))
    base.eval(); other.eval()
    with torch.no_grad(), text_only(base, idx):
        expected = base(idx)
    with torch.no_grad(), text_only(other, idx):
        assert torch.equal(other(idx), expected)


def test_nope_changes_attention_but_not_a_pure_linear_model():
    for method, changed in (("dense_full", True), ("gdn", False)):
        c, base, other = twin(method, position="nope")
        idx = torch.randint(c.vocab_size, (2, T))
        base.eval(); other.eval()
        with torch.no_grad(), text_only(base, idx), text_only(other, idx):
            pass
        with torch.no_grad():
            with text_only(base, idx):
                a = base(idx)
            with text_only(other, idx):
                b = other(idx)
        assert (not torch.equal(a, b)) == changed, method


def test_rope2d_coordinates_follow_rope_tv():
    _, model = build("dense_full", position="rope2d")
    mask, _ = one_image(start=5, batch=1)
    cos, sin = rotary_tables(model, mask, patch_order("raster"), "rope2d")
    half = cos.shape[-1]
    inv = 1.0 / (100000 ** (torch.arange(0, 2 * half, 2, dtype=torch.float32) / (2 * half)))
    angle = torch.atan2(sin[0, :, 0], cos[0, :, 0])

    def expect(px, py):
        a = torch.where(torch.arange(half) % 2 == 0, px * inv, py * inv)
        return torch.atan2(a.sin(), a.cos())

    base = 5 - 1 + (TOKENS - GRID) / 2  # L + (wh - h) / 2 with L = 4
    for slot, (i, j) in ((0, (0, 0)), (1, (0, 1)), (13, (1, 1)), (143, (11, 11))):
        torch.testing.assert_close(angle[5 + slot], expect(base + i + 1, base + j + 1), atol=2e-3, rtol=0)
    torch.testing.assert_close(angle[300], expect(300., 300.), atol=2e-3, rtol=0)
    torch.testing.assert_close(angle[2], expect(2., 2.), atol=2e-3, rtol=0)


@pytest.mark.parametrize("conv2d", ["k", "qkv"])
def test_2d_convolution_is_inert_on_text_and_learns_on_images(conv2d):
    c, base, other = twin("gdn_hybrid", conv2d=conv2d)
    idx = torch.randint(c.vocab_size, (2, T))
    base.eval(); other.eval()
    with torch.no_grad():
        with text_only(base, idx):
            expected = base(idx)
        with text_only(other, idx):
            assert torch.equal(other(idx), expected)
    other.train()
    mask, features = one_image()
    labels = torch.randint(c.vocab_size, idx.shape)
    labels[mask] = -1
    with visual_inputs(other, mask, features):
        other(idx, labels).backward()
    for p in other.image_convs.parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0
    other.eval()
    with torch.no_grad():
        with visual_inputs(other, mask, features):
            out = other(idx)
        moved = features.clone()
        moved[-1] += 1.  # the last patch: a 2D convolution may carry it to earlier patches of the same image
        with visual_inputs(other, mask, moved):
            changed = other(idx)
    torch.testing.assert_close(changed[:, :5], out[:, :5], atol=0, rtol=0)  # text before the image
    assert not torch.equal(changed[:, 5 + TOKENS:], out[:, 5 + TOKENS:])


@pytest.mark.parametrize("order", ["snake", "hilbert"])
def test_patch_order_changes_the_sequence_not_the_text_before_it(order):
    c, base, other = twin("gdn_hybrid", order=order)
    idx = torch.randint(c.vocab_size, (2, T))
    mask, features = one_image()
    base.eval(); other.eval()
    with torch.no_grad():
        with visual_inputs(base, mask, features):
            a = base(idx)
        with visual_inputs(other, mask, features):
            b = other(idx)
    torch.testing.assert_close(b[:, :5], a[:, :5], atol=0, rtol=0)
    assert not torch.equal(a[:, 5 + TOKENS:], b[:, 5 + TOKENS:])


def test_raw_patches_are_row_major_32px_tiles():
    images = torch.randint(0, 255, (2, 3, 384, 384), dtype=torch.uint8)
    out = RawPatches(dtype=torch.float32)(images)
    assert out.shape == (2 * TOKENS, RAW_DIM)
    i, j = 3, 7
    tile = images[1, :, 32 * i:32 * (i + 1), 32 * j:32 * (j + 1)].float().div(127.5).sub(1.)
    torch.testing.assert_close(out[TOKENS + i * GRID + j], tile.reshape(-1))


@pytest.mark.parametrize("spatial", [dict(conv2d="k"), dict(conv2d="qkv", order="hilbert", position="rope2d")])
def test_spatial_variants_audit_without_unknown_operators(spatial):
    c = debug_config("gdn_hybrid", sequence_len=T)
    track = Track(sequence_len=T, rows=2, images_per_row=2, visual_dim=DIM)
    base = audit_mm(c, track, optimizer=False)
    other = audit_mm(c, track, optimizer=False, spatial=spatial)
    assert other["unknown"] == {} and base["unknown"] == {}
    # Debug width 48 inflates the share of the 3 x 3 convolution; the formal-size share is checked separately.
    assert 0 < other["total"] - base["total"] < .05 * base["total"]


def test_scrambles_move_tokens_and_keep_their_content():
    from ntv.spatial_eval import Scrambled, scramble_index

    grid = torch.arange(144.).view(1, 12, 12)  # each token's value is its raster position
    rows, cols = grid // 12, grid % 12

    class Encoder(torch.nn.Module):
        def forward(self, images):
            return torch.stack([rows, cols], -1).repeat(images.shape[0], 1, 1, 1).view(-1, 2)

    images = torch.zeros(2, 3, 384, 384, dtype=torch.uint8)
    out = {m: Scrambled(Encoder(), scramble_index(m))(images).view(2, 12, 12, 2) for m in ("vflip", "hflip", "shuffle")}
    assert torch.equal(out["vflip"][..., 0], 11 - rows.expand(2, -1, -1))  # rows reversed, columns kept
    assert torch.equal(out["vflip"][..., 1], cols.expand(2, -1, -1))
    assert torch.equal(out["hflip"][..., 1], 11 - cols.expand(2, -1, -1))
    assert torch.equal(out["hflip"][..., 0], rows.expand(2, -1, -1))
    seen = out["shuffle"][0, ..., 0] * 12 + out["shuffle"][0, ..., 1]
    assert sorted(seen.flatten().tolist()) == list(range(144)) and not torch.equal(seen, grid[0])
    assert torch.equal(scramble_index("shuffle"), scramble_index("shuffle"))


def test_position_probe_reads_position_only_when_tokens_carry_it():
    from ntv.position_probe import fit

    g = torch.Generator().manual_seed(0)
    n, d = 40, 32
    noise = torch.randn(n, 144, d, generator=g)
    pos = torch.arange(144.)
    coords = torch.stack([pos // 12, pos % 12], 1) / 11
    embed = coords @ torch.randn(2, d, generator=g)  # a smooth position embedding, as a ViT's is
    with_position = noise * .1 + embed
    carried = fit(with_position[:30], with_position[30:])
    absent = fit(noise[:30], noise[30:])
    assert carried["row_r2"] > .95 and carried["col_r2"] > .95 and carried["row_acc"] > .9
    assert absent["row_r2"] < .1 and absent["col_acc"] < .2
