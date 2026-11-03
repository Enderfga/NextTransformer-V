"""Spatial-relation accuracy of a checkpoint on a probe in the ntv.synth eval format (the synthetic set or
What'sUp via ntv.whatsup), on one GPU.

Every image has four candidate captions that differ only in the relation. Two scores:
- `acc4`: the candidate whose likelihood gains most from seeing the image (loss without the image minus loss with
  it), which cancels the different lengths and priors of the relation phrases;
- `acc2`: the true relation against its opposite (left/right or above/below; same length), by loss given the image.
Both are reported overall, by axis (horizontal / vertical) and by distance in grid cells, with `margin2`, the mean
loss of the opposite relation minus that of the true one (nats; a graded version of acc2 that does not saturate).

Scrambles (secondary analysis): the same scores with the image tokens fed to the backbone in another order while
their content is unchanged. `vflip` reverses the rows of the token grid, `hflip` the columns within each row,
`shuffle` is a fixed random permutation. A backbone that reads position from the token order gets the flipped axis
backwards (acc2 near 0); one that reads it from the token content (e.g. the encoder's position embedding) is unaffected.

  python -m ntv.spatial_eval --checkpoint <run>/checkpoint_00005000 --probe <synth-eval dir> --encoder <dir> \
      --resources <dir> --out <file.json>
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch

from nanochat.tokenizer import RustBPETokenizer
from ntv.data import decode_image
from ntv.evaluate import caption_losses, load_model
from ntv.spatial import GRID

OPPOSITE = {"to the left of": "to the right of", "to the right of": "to the left of", "above": "below",
            "below": "above", "on": "under", "under": "on"}
SCRAMBLES = ("vflip", "hflip", "shuffle")


def scramble_index(mode, grid=GRID, seed=0):
    """Raster positions of the tokens fed to the backbone, in order (see the module docstring)."""
    index = torch.arange(grid * grid).view(grid, grid)
    if mode == "vflip":
        index = index.flip(0)
    elif mode == "hflip":
        index = index.flip(1)
    elif mode == "shuffle":
        index = index.flatten()[torch.randperm(grid * grid, generator=torch.Generator().manual_seed(seed))]
    else:
        raise ValueError(f"unknown scramble {mode!r}")
    return index.flatten()


class Scrambled(torch.nn.Module):
    """An encoder whose tokens come out of each image in the order `index` instead of raster order."""

    def __init__(self, encoder, index):
        super().__init__()
        self.encoder, self.index = encoder, index

    def forward(self, images):
        out = self.encoder(images)
        out = out.view(images.shape[0], len(self.index), out.shape[-1])
        return out[:, self.index.to(out.device)].reshape(-1, out.shape[-1])


def score(items, relations, with_image, without):
    """acc4, acc2, margin2 and counts by group, from (items x relations) losses with and without the image."""
    groups = defaultdict(lambda: [0, 0, 0., 0])  # acc4 hits, acc2 hits, margin sum, count
    for i, item in enumerate(items):
        true, opposite = relations.index(item["relation"]), relations.index(OPPOSITE[item["relation"]])
        hit4 = int(np.argmax(without[i] - with_image[i]) == true)
        hit2 = int(with_image[i, true] < with_image[i, opposite])
        margin = float(with_image[i, opposite] - with_image[i, true])
        for key in ("all", item["axis"], f"{item['axis']}_d{item['distance']}"):
            group = groups[key]
            group[0] += hit4
            group[1] += hit2
            group[2] += margin
            group[3] += 1
    ordered = sorted(groups.items())
    return dict(**{f"{key}_acc4": g[0] / g[3] for key, g in ordered},
                **{f"{key}_acc2": g[1] / g[3] for key, g in ordered},
                **{f"{key}_margin2": g[2] / g[3] for key, g in ordered},
                **{f"{key}_count": g[3] for key, g in ordered})


def probe(model, encoder, directory, tok, device, length=200, batch=64, limit=None, scrambles=()):
    """Scores of a loaded model on the probe in `directory` (eval.jsonl + images), also under each scramble."""
    items = [json.loads(line) for line in open(Path(directory) / "eval.jsonl")][:limit]
    images = [decode_image((Path(directory) / item["image"]).read_bytes())[0] for item in items]
    bos = tok.get_bos_token_id()
    relations = list(items[0]["candidates"])  # the same candidate relations, in the same order, for every item
    if any(list(item["candidates"]) != relations for item in items):
        raise ValueError("every probe item needs the same candidate relations")
    ids = tok.encode([item["candidates"][r] for item in items for r in relations], num_threads=8)
    n, k = len(items), len(relations)
    examples = [(i // k, ids[i]) for i in range(n * k)]
    with_image, _ = caption_losses(model, encoder, examples, images, length, bos, device, batch)
    without, _ = caption_losses(model, encoder, [(None, ids[i]) for i in range(n * k)], images,
                                length, bos, device, batch)
    without = without.reshape(n, k)
    result = dict(images=n, **score(items, relations, with_image.reshape(n, k), without))
    for mode in scrambles:
        scrambled, _ = caption_losses(model, Scrambled(encoder, scramble_index(mode)), examples, images,
                                      length, bos, device, batch)
        result.update({f"{mode}_{key}": value for key, value in
                       score(items, relations, scrambled.reshape(n, k), without).items()
                       if not key.endswith("_count")})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--encoder", type=Path)
    parser.add_argument("--resources", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--length", type=int, default=200)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--scrambles", default=",".join(SCRAMBLES), help="comma-separated; empty for none")
    args = parser.parse_args()
    scrambles = tuple(mode for mode in args.scrambles.split(",") if mode)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tok = RustBPETokenizer.from_directory(str(args.resources / "tokenizer"))
    model, config, step, encoder, settings = load_model(args.checkpoint, device, args.encoder)
    result = dict(method=config.method, step=step, checkpoint=str(args.checkpoint), **settings,
                  **probe(model, encoder, args.probe, tok, device, args.length, args.batch, args.limit,
                        scrambles))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in result.items()}, indent=1))


if __name__ == "__main__":
    main()
