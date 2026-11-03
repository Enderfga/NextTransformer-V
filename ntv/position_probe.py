"""How much 2D position the visual tokens carry by themselves, before any backbone: ridge regression from each
token's features to its row and to its column in the 12 x 12 token grid, fitted on some COCO images and scored on
others (R^2, and accuracy of the rounded prediction). SigLIP tokens carry the encoder's position embedding; raw pixel
patches carry only what the content implies (sky at the top, ...).

  python -m ntv.position_probe --coco <coco5k dir> --encoder <dir> --visual siglip --out <file.json>
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from ntv.data import decode_image
from ntv.evaluate import IMAGE_DIR
from ntv.spatial import GRID

LAMBDAS = (1e-3, 1e-2, 1e-1, 1., 10., 100.)


def features(encoder, images, device, batch=32):
    """(N, 144, D) float32 token features of uint8 (N, 3, 384, 384) images."""
    out = []
    for first in range(0, len(images), batch):
        pixels = torch.from_numpy(np.stack(images[first:first + batch])).to(device)
        out.append(encoder(pixels).float().view(len(pixels), GRID * GRID, -1))
    return torch.cat(out)


def ridge(x, y, lam):
    """Weights of ridge regression of y on x (both centred), penalty lam per sample."""
    d = x.shape[1]
    return torch.linalg.solve(x.T @ x + lam * x.shape[0] * torch.eye(d, device=x.device, dtype=x.dtype), x.T @ y)


def fit(train, test, val_share=.2):
    """R^2 and rounded accuracy on `test` for row and column; the penalty is chosen on part of `train`."""
    grid = torch.arange(GRID * GRID, device=train.device)
    target = torch.stack([grid // GRID, grid % GRID], 1).double()

    def design(feats):
        return feats.reshape(-1, feats.shape[-1]).double(), target.repeat(feats.shape[0], 1)

    def r2(pred, y):
        return (1 - ((pred - y) ** 2).sum(0) / ((y - y.mean(0)) ** 2).sum(0)).tolist()

    n_val = max(1, int(len(train) * val_share))
    fit_x, fit_y = design(train[n_val:])
    val_x, val_y = design(train[:n_val])
    mean, std = fit_x.mean(0), fit_x.std(0).clamp_min(1e-6)
    y_mean = fit_y.mean(0)
    scores = {}
    for lam in LAMBDAS:
        w = ridge((fit_x - mean) / std, fit_y - y_mean, lam)
        scores[lam] = float(np.mean(r2((val_x - mean) / std @ w + y_mean, val_y)))
    best = max(scores, key=scores.get)
    all_x, all_y = design(train)
    mean, std = all_x.mean(0), all_x.std(0).clamp_min(1e-6)
    y_mean = all_y.mean(0)
    w = ridge((all_x - mean) / std, all_y - y_mean, best)
    test_x, test_y = design(test)
    pred = (test_x - mean) / std @ w + y_mean
    hit = (pred.round().clamp(0, GRID - 1) == test_y).double().mean(0).tolist()
    row_r2, col_r2 = r2(pred, test_y)
    return dict(lam=best, row_r2=row_r2, col_r2=col_r2, row_acc=hit[0], col_acc=hit[1], chance_acc=1 / GRID)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--coco", type=Path, required=True)
    parser.add_argument("--encoder", type=Path)
    parser.add_argument("--visual", default="siglip", choices=("siglip", "raw32"))
    parser.add_argument("--train", type=int, default=1024)
    parser.add_argument("--test", type=int, default=256)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    from ntv.train import visual_encoder
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(args.coco / "test_5k_mscoco_2014.csv", newline="") as handle:
        records = list(csv.DictReader(handle))[:args.train + args.test]
    images = [decode_image((args.coco / IMAGE_DIR / r["filename"]).read_bytes())[0] for r in records]
    encoder, _ = visual_encoder(args.visual, args.encoder, device)
    feats = features(encoder, images, device)
    result = dict(visual=args.visual, train_images=args.train, test_images=len(images) - args.train,
                  **fit(feats[:args.train], feats[args.train:]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
