"""Where in the backbone the 2D position of image tokens lives, and how far the linear layers' state reaches.

Two descriptive analyses of a checkpoint (docs/spatial-addendum.md), on COCO images in rows `[BOS] [image] caption`:

- Layer probe. Ridge regression from the residual stream to each image token's position, at the input (after the
  visual projector) and after every block, fitted on some images and scored (R^2) on others. In the training order a
  token's place in the sequence and its place in the image coincide, so a probe could use either. The images are
  therefore also fed with their tokens in a random order (one fixed permutation per image), where the two are
  independent, and two probes are fitted there: one for the slot a token sits in (position from token order) and one
  for the cell it came from (position from token content).
- GDN retention. For every GDN layer and head, the forget gate alpha at image and at caption positions, as the
  fraction of the state that survives one token, one image row (12 tokens) and one image (144 tokens). With the
  delta rule the state can also be overwritten, so these are upper bounds on what a layer can carry that far.

  python -m ntv.mechanism --checkpoint <run>/checkpoint_00005000 --coco <coco5k dir> --encoder <dir> \
      --resources <dir> --out <file.json>
"""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ntv.data import decode_image
from ntv.spatial import GRID

LAMBDAS = (1e-3, 1e-2, 1e-1, 1., 10., 100.)
SPANS = {"token": 1, "row": GRID, "image": GRID * GRID}


class Moments:
    """Sufficient statistics of a linear regression of y (n x k) on x (n x d), accumulated batch by batch."""

    def __init__(self, d, k, device):
        z = dict(dtype=torch.float64, device=device)
        self.n = 0
        self.sx, self.sy = torch.zeros(d, **z), torch.zeros(k, **z)
        self.xx, self.xy, self.yy = torch.zeros(d, d, **z), torch.zeros(d, k, **z), torch.zeros(k, **z)

    def add(self, x, y):
        x, y = x.double(), y.double()
        self.n += x.shape[0]
        self.sx += x.sum(0)
        self.sy += y.sum(0)
        self.xx += x.T @ x
        self.xy += x.T @ y
        self.yy += (y * y).sum(0)

    def __add__(self, other):
        out = Moments.__new__(Moments)
        out.n = self.n + other.n
        for name in ("sx", "sy", "xx", "xy", "yy"):
            setattr(out, name, getattr(self, name) + getattr(other, name))
        return out


def solve(m, lam):
    """Ridge weights and intercept on standardized features (penalty lam per sample)."""
    mean = m.sx / m.n
    cov = m.xx / m.n - torch.outer(mean, mean)
    scale = cov.diagonal().clamp_min(1e-12).rsqrt()
    a = cov * torch.outer(scale, scale) + lam * torch.eye(len(mean), dtype=cov.dtype, device=cov.device)
    b = (m.xy / m.n - torch.outer(mean, m.sy / m.n)) * scale[:, None]
    w = torch.linalg.solve(a, b) * scale[:, None]
    return w, m.sy / m.n - mean @ w


def r2(m, w, c):
    """R^2 per target of predictions x w + c on the set summarized by m."""
    sse = (m.yy - 2 * (w * m.xy).sum(0) - 2 * c * m.sy + (w * (m.xx @ w)).sum(0) + 2 * c * (m.sx @ w)
           + m.n * c * c)
    sst = m.yy - m.sy ** 2 / m.n
    return (1 - sse / sst).tolist()


def fit(fit_set, val_set, test_sets):
    """Penalty chosen on val_set, refit on fit_set + val_set, R^2 on each test set (dict name -> Moments)."""
    scores = {lam: float(np.mean(r2(val_set, *solve(fit_set, lam)))) for lam in LAMBDAS}
    best = max(scores, key=scores.get)
    w, c = solve(fit_set + val_set, best)
    return dict(lam=best, **{name: r2(m, w, c) for name, m in test_sets.items()})


def permutation(image, grid=GRID):
    """The fixed random token order of one image in the decorrelated probe."""
    return torch.randperm(grid * grid, generator=torch.Generator().manual_seed(10_000 + image))


def targets(index, grid=GRID):
    """For an image fed in token order `index` (slot s holds cell index[s]): (slot row, slot col, cell row, cell col)."""
    slot = torch.arange(grid * grid)
    return torch.stack([slot // grid, slot % grid, index // grid, index % grid], 1).double()


@torch.no_grad()
def collect(model, encoder, images, captions, bos, device, decays, batch=16, grid=GRID):
    """Yield (order, {layer: (B * grid^2, width) states at image positions}, targets, image indices) per batch.

    Layer 0 is the input (token embeddings with the projected image features), layer i the output of block i. Every
    batch is fed in raster order and again with each image's tokens in its own random order ("permuted").
    `decays` collects the GDN log forget gates (per head) at image and caption positions of the raster rows.
    """
    from ntv.evaluate import rows
    from ntv.model import visual_inputs
    states, current, hooks = {}, {}, []
    hooks.append(model.transformer.wte.register_forward_hook(lambda m, a, out: states.__setitem__(0, out)))
    for i, block in enumerate(model.transformer.h, 1):
        hooks.append(block.register_forward_hook(lambda m, a, out, i=i: states.__setitem__(i, out)))
        mixer = getattr(block, "attn", None)
        if getattr(mixer, "decay_proj", None) is not None:
            def gate(m, a, out, i=i, mixer=mixer):  # log alpha = -A softplus(W_a x + dt_bias), as in the mixer
                current[i] = -mixer.log_decay.exp() * F.softplus(out.float() + mixer.dt_bias)
            hooks.append(mixer.decay_proj.register_forward_hook(gate))
    try:
        for first in range(0, len(images), batch):
            chunk = list(range(first, min(first + batch, len(images))))
            x, y, mask = rows([(s, captions[s]) for s in chunk], 1 + grid * grid + 48, bos, True, grid * grid)
            x, mask = x.to(device), mask.to(device)
            caption = torch.zeros_like(mask)
            caption[:, 1:] = (y[:, :-1] >= 0).to(device)  # positions holding caption tokens
            pixels = torch.from_numpy(np.stack([images[s] for s in chunk])).to(device)
            features = encoder(pixels).view(len(chunk), grid * grid, -1)
            for order in ("raster", "permuted"):
                index = [torch.arange(grid * grid) if order == "raster" else permutation(s, grid) for s in chunk]
                fed = torch.stack([features[k, i.to(device)] for k, i in enumerate(index)]).reshape(-1, features.shape[-1])
                with visual_inputs(model, mask, fed, grid=grid):
                    model(x)
                width = states[0].shape[-1]
                y = torch.cat([targets(i, grid) for i in index]).to(device)
                yield order, {k: v[mask].reshape(-1, width).float() for k, v in states.items()}, y, chunk
                if order == "raster":
                    for i, log_alpha in current.items():  # (B, T, heads)
                        d = decays.setdefault(i, dict(image=[], caption=[]))
                        d["image"].append(log_alpha[mask].cpu())
                        d["caption"].append(log_alpha[caption].cpu())
                states.clear()
                current.clear()
    finally:
        for h in hooks:
            h.remove()


def columns(m, cols):
    """The same moments with only some target columns."""
    out = Moments.__new__(Moments)
    out.n, out.sx, out.xx = m.n, m.sx, m.xx
    out.sy, out.xy, out.yy = m.sy[cols], m.xy[:, cols], m.yy[cols]
    return out


def analyse(model, encoder, images, captions, bos, device, train, val, batch=16, grid=GRID):
    """Layer probes and GDN retention; images [0, val) choose the penalty, [val, train) fit, [train, n) test."""
    moments, decays = {}, {}
    for order, layers, y, chunk in collect(model, encoder, images, captions, bos, device, decays, batch, grid):
        if (chunk[0] < val) != (chunk[-1] < val) or (chunk[0] < train) != (chunk[-1] < train):
            raise ValueError("--val and --train must be multiples of --batch")
        part = "val" if chunk[0] < val else "fit" if chunk[0] < train else "test"
        for layer, x in layers.items():
            if (layer, order, part) not in moments:
                moments[(layer, order, part)] = Moments(x.shape[-1], 4, device)
            moments[(layer, order, part)].add(x, y)
    slot, cell = [0, 1], [2, 3]  # target columns: (slot row, slot col, cell row, cell col)
    probe = []
    for layer in sorted({key[0] for key in moments}):
        def probe_on(order, cols):
            m = {part: columns(moments[(layer, order, part)], cols) for part in ("fit", "val", "test")}
            return fit(m["fit"], m["val"], {"test": m["test"]})
        raster, by_order, by_content = probe_on("raster", cell), probe_on("permuted", slot), probe_on("permuted", cell)
        probe.append(dict(layer=layer, row_r2=raster["test"][0], col_r2=raster["test"][1],
                          order_row_r2=by_order["test"][0], order_col_r2=by_order["test"][1],
                          content_row_r2=by_content["test"][0], content_col_r2=by_content["test"][1],
                          lam=[raster["lam"], by_order["lam"], by_content["lam"]]))
    retention = []
    for layer, d in sorted(decays.items()):
        row = dict(layer=layer)
        for where in ("image", "caption"):
            log_alpha = torch.cat(d[where]).median(0).values  # per head
            for name, span in SPANS.items():
                row[f"{where}_{name}"] = (log_alpha * span).exp().tolist()
        retention.append(row)
    return dict(layer_probe=probe, gdn_retention=retention)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--coco", type=Path, required=True)
    parser.add_argument("--encoder", type=Path)
    parser.add_argument("--resources", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--images", type=int, default=1024, help="COCO images used, in file order")
    parser.add_argument("--val", type=int, default=128, help="the first images, to choose the ridge penalty")
    parser.add_argument("--train", type=int, default=768, help="images before this index are not test images")
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()
    from nanochat.tokenizer import RustBPETokenizer
    from ntv.evaluate import IMAGE_DIR, load_model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config, step, encoder, settings = load_model(args.checkpoint, device, args.encoder)
    tok = RustBPETokenizer.from_directory(str(args.resources / "tokenizer"))
    with open(args.coco / "test_5k_mscoco_2014.csv", newline="") as handle:
        records = list(csv.DictReader(handle))[:args.images]
    images = [decode_image((args.coco / IMAGE_DIR / r["filename"]).read_bytes())[0] for r in records]
    captions = tok.encode([json.loads(r["raw"])[0].strip() for r in records], num_threads=8)
    result = dict(method=config.method, step=step, checkpoint=str(args.checkpoint), **settings,
                  images=len(images), fit_images=args.train - args.val, val_images=args.val,
                  test_images=len(images) - args.train,
                  **analyse(model, encoder, images, captions, tok.get_bos_token_id(), device, args.train, args.val,
                            args.batch))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    for row in result["layer_probe"]:
        print(f"layer {row['layer']:2d}  raster row {row['row_r2']:.2f} col {row['col_r2']:.2f}  "
              f"from order: row {row['order_row_r2']:.2f} col {row['order_col_r2']:.2f}  "
              f"from content: row {row['content_row_r2']:.2f} col {row['content_col_r2']:.2f}")
    for row in result["gdn_retention"]:
        print(f"GDN block {row['layer']:2d}  retention (median head) image: row {np.median(row['image_row']):.3f} "
              f"image {np.median(row['image_image']):.3f}  caption: row {np.median(row['caption_row']):.3f}")


if __name__ == "__main__":
    main()
