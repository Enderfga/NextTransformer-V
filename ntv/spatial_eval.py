"""Spatial-relation accuracy of a checkpoint on the synthetic probe (ntv.synth eval set), on one GPU.

Every image has four candidate captions that differ only in the relation. Two scores:
- `acc4`: the candidate whose likelihood gains most from seeing the image (loss without the image minus loss with
  it), which cancels the different lengths and priors of the relation phrases;
- `acc2`: the true relation against its opposite (left/right or above/below; same length), by loss given the image.
Both are reported overall, by axis (horizontal / vertical) and by distance in grid cells.

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
from ntv.synth import RELATIONS

OPPOSITE = {"to the left of": "to the right of", "to the right of": "to the left of", "above": "below",
            "below": "above"}


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
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    items = [json.loads(line) for line in open(args.probe / "eval.jsonl")][:args.limit]
    images = [decode_image((args.probe / item["image"]).read_bytes())[0] for item in items]
    tok = RustBPETokenizer.from_directory(str(args.resources / "tokenizer"))
    bos = tok.get_bos_token_id()
    texts = [item["candidates"][r] for item in items for r in RELATIONS]
    ids = tok.encode(texts, num_threads=8)
    model, config, step, encoder, settings = load_model(args.checkpoint, device, args.encoder)
    n, k = len(items), len(RELATIONS)
    with_image, _ = caption_losses(model, encoder, [(i // k, ids[i]) for i in range(n * k)], images,
                                   args.length, bos, device, args.batch)
    without, _ = caption_losses(model, encoder, [(None, ids[i]) for i in range(n * k)], images,
                                args.length, bos, device, args.batch)
    with_image, without = with_image.reshape(n, k), without.reshape(n, k)
    groups = defaultdict(lambda: [0, 0, 0])  # acc4 hits, acc2 hits, count
    for i, item in enumerate(items):
        true, opposite = RELATIONS.index(item["relation"]), RELATIONS.index(OPPOSITE[item["relation"]])
        hit4 = int(np.argmax(without[i] - with_image[i]) == true)
        hit2 = int(with_image[i, true] < with_image[i, opposite])
        for key in ("all", item["axis"], f"{item['axis']}_d{item['distance']}"):
            groups[key][0] += hit4
            groups[key][1] += hit2
            groups[key][2] += 1
    result = dict(method=config.method, step=step, checkpoint=str(args.checkpoint), images=n, **settings,
                  **{f"{key}_acc4": a / c for key, (a, _, c) in sorted(groups.items())},
                  **{f"{key}_acc2": b / c for key, (_, b, c) in sorted(groups.items())},
                  **{f"{key}_count": c for key, (_, _, c) in sorted(groups.items())})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in result.items()}, indent=1))


if __name__ == "__main__":
    main()
