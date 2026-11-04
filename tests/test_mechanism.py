"""The layer probe's ridge from sufficient statistics, and the mechanism analysis on a tiny model."""
import numpy as np
import pytest
import torch

from nexttransformer.models.config import debug_config
from ntv.mechanism import Moments, analyse, fit, r2, solve
from ntv.model import build_mm_model
from ntv.spatial import GRID, TOKENS

DIM = 20


@pytest.fixture(autouse=True)
def reproducible():
    torch.manual_seed(0)
    torch.set_num_threads(1)


def test_ridge_from_moments_matches_direct_predictions():
    x = torch.randn(600, 7, dtype=torch.float64) * torch.tensor([1., 2, 3, .5, 1, 4, .1], dtype=torch.float64)
    y = x @ torch.randn(7, 2, dtype=torch.float64) + .3 * torch.randn(600, 2, dtype=torch.float64) + 5
    m = Moments(7, 2, "cpu")
    for part in range(0, 600, 128):  # batch by batch
        m.add(x[part:part + 128], y[part:part + 128])
    w, c = solve(m, .1)
    pred = x @ w + c
    direct = (1 - ((y - pred) ** 2).sum(0) / ((y - y.mean(0)) ** 2).sum(0)).tolist()
    np.testing.assert_allclose(r2(m, w, c), direct, rtol=1e-8)
    a, b = Moments(7, 2, "cpu"), Moments(7, 2, "cpu")
    a.add(x[:300], y[:300])
    b.add(x[300:], y[300:])
    result = fit(a, b, {"all": m})
    assert result["lam"] in (1e-3, 1e-2, 1e-1, 1., 10., 100.) and result["all"][0] > .9


class PositionEncoder(torch.nn.Module):
    """Stub visual tokens that carry their grid position (a smooth embedding) plus a little content noise."""

    def __init__(self):
        super().__init__()
        grid = torch.arange(TOKENS)
        rc = torch.stack([grid // GRID, grid % GRID], 1).float() / GRID
        self.position = torch.cat([rc, torch.sin(3 * rc), torch.cos(3 * rc)], 1) @ torch.randn(6, DIM)

    def forward(self, images):
        noise = images.float().mean((1, 2, 3))[:, None, None] / 255 * torch.randn(1, TOKENS, DIM) * .1
        return (self.position[None] + noise).reshape(-1, DIM)


def test_layer_probe_separates_order_from_content_and_reads_gdn_gates():
    c = debug_config("gdn_hybrid", sequence_len=1 + TOKENS + 48)
    model = build_mm_model(c, visual_dim=DIM, spatial=dict(position="rope1d")).eval()
    n = 48
    images = [np.random.randint(0, 255, (3, 32, 32), dtype=np.uint8) for _ in range(n)]
    captions = [list(np.random.randint(3, 90, 12)) for _ in range(n)]
    result = analyse(model, PositionEncoder(), images, captions, bos=1, device="cpu", train=32, val=16, batch=16)
    first = result["layer_probe"][0]  # the input: projected tokens, which carry the position embedding
    assert first["layer"] == 0 and first["row_r2"] > .9 and first["col_r2"] > .9
    # Reordering moves the content (and its position embedding) with the token: the input encodes content position.
    assert first["vflip_row_r2_content"] > .9 > first["vflip_row_r2_slot"]
    assert first["hflip_col_r2_content"] > .9 > first["hflip_col_r2_slot"]
    assert [row["layer"] for row in result["layer_probe"]] == list(range(len(model.transformer.h) + 1))
    gdn = [i + 1 for i, block in enumerate(model.transformer.h) if hasattr(block.attn, "decay_proj")]
    assert [row["layer"] for row in result["gdn_retention"]] == gdn and gdn
    for row in result["gdn_retention"]:
        for key in ("image_token", "image_row", "image_image", "caption_token", "caption_row"):
            assert all(0 < v <= 1 for v in row[key])
        assert all(a >= b for a, b in zip(row["image_token"], row["image_row"]))
