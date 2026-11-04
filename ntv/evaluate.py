"""Evaluate a checkpoint on COCO 2014 5k test (Karpathy split), on one GPU.

- Caption BPB: bits per byte of each image's first caption, given the image.
- 5-way retrieval: each image's first caption against the first captions of four other images (fixed seed).
  `raw` picks the caption with the lowest mean loss given the image; `pmi` picks the largest gain of
  log-likelihood from seeing the image (the loss without the image minus the loss with it).

Each example is one short row, `[BOS] [image x 144] caption` (or `[BOS] caption` without the image), padded to
a fixed length; only caption tokens are scored.

  python -m ntv.evaluate --checkpoint <run>/checkpoint_00005000 --coco <coco5k dir> --encoder <dir> \
      --resources <dir> --out <file.json>
"""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch

from nanochat.tokenizer import RustBPETokenizer
from nexttransformer.models.config import ModelConfig
from ntv.data import IMAGE_TOKENS, decode_image
from ntv.model import build_mm_model, visual_inputs

IMAGE_DIR = "images/images_mscoco_2014_5k_test"


def load_model(checkpoint, device, encoder_path=None):
    """The checkpoint's model, with its visual input and spatial settings, and the matching encoder."""
    from ntv.train import visual_encoder
    saved = torch.load(Path(checkpoint) / "model.pt", map_location="cpu", weights_only=False)
    fields = {k: tuple(v) if isinstance(v, list) else v for k, v in saved["model_config"].items()}
    config = ModelConfig(**fields)
    train = saved.get("train_config", {})
    spatial = {k: train.get(k, d) for k, d in (("order", "raster"), ("position", "rope1d"), ("conv2d", "none"))}
    encoder, visual_dim = visual_encoder(train.get("visual", "siglip"), encoder_path, device)
    model = build_mm_model(config, device, visual_dim=visual_dim, spatial=spatial)
    model.load_state_dict(saved["model"], strict=True)
    model.set_step(saved["state"]["step"])
    model.eval()
    return model, config, saved["state"]["step"], encoder, dict(spatial, visual=train.get("visual", "siglip"))


def rows(examples, length, bos, with_image, image_tokens=IMAGE_TOKENS):
    """examples: (image slot or None, caption ids) -> inputs, targets, image mask."""
    b = len(examples)
    ids = torch.zeros((b, length + 1), dtype=torch.long)
    targets = torch.full((b, length), -1, dtype=torch.long)
    mask = torch.zeros((b, length), dtype=torch.bool)
    for i, (_, caption) in enumerate(examples):
        start = 1 + image_tokens if with_image else 1
        n = min(len(caption), length - start)
        ids[i, 0] = bos
        ids[i, start:start + n] = torch.tensor(caption[:n])
        targets[i, start - 1:start - 1 + n] = ids[i, start:start + n]
        if with_image:
            mask[i, 1:start] = True
    return ids[:, :-1], targets, mask


@torch.no_grad()
def caption_losses(model, encoder, examples, images, length, bos, device, batch, grid=None):
    """Summed loss (nats) and token count of each example's caption. `grid`: patches per image side of the
    encoder's output (12 at the training resolution)."""
    from ntv.spatial import GRID
    grid = grid or GRID
    with_image = examples[0][0] is not None
    nats, counts = [], []
    for first in range(0, len(examples), batch):
        chunk = examples[first:first + batch]
        x, y, mask = rows(chunk, length, bos, with_image, grid * grid)
        x, y, mask = x.to(device), y.to(device), mask.to(device)
        if with_image:
            slots = [slot for slot, _ in chunk]
            unique = sorted(set(slots))
            pixels = torch.from_numpy(np.stack([images[s] for s in unique])).to(device)
            encoded = encoder(pixels).view(len(unique), grid * grid, -1)
            index = torch.tensor([unique.index(s) for s in slots], device=device)
            features = encoded[index].reshape(-1, encoded.shape[-1])
        else:
            features = torch.zeros((0, model.visual_dim), device=device)
        with visual_inputs(model, mask, features, grid=grid):
            loss = model(x, y, loss_reduction="none").view_as(y).double()
        valid = y >= 0
        nats += (loss * valid).sum(1).tolist()
        counts += valid.sum(1).tolist()
    return np.array(nats), np.array(counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--coco", type=Path, required=True)
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--resources", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--length", type=int, default=256)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--limit", type=int, help="first N images only (quick checks)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--whatsup", type=Path, help="What'sUp probe directory (default: <coco>/../whatsup)")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.coco / "test_5k_mscoco_2014.csv", newline="") as handle:
        records = list(csv.DictReader(handle))[:args.limit]
    tok = RustBPETokenizer.from_directory(str(args.resources / "tokenizer"))
    bos = tok.get_bos_token_id()
    captions = tok.encode([json.loads(r["raw"])[0].strip() for r in records], num_threads=8)
    decoded = [decode_image((args.coco / IMAGE_DIR / r["filename"]).read_bytes()) for r in records]
    images = [image for image, _ in decoded]
    token_bytes = torch.load(args.resources / "tokenizer" / "token_bytes.pt", map_location="cpu")
    caption_bytes = np.array([int(token_bytes[torch.tensor(c)].sum()) for c in captions])

    model, config, step, encoder, settings = load_model(args.checkpoint, device, args.encoder)
    n = len(records)
    true_nats, true_counts = caption_losses(model, encoder, [(i, captions[i]) for i in range(n)], images,
                                            args.length, bos, device, args.batch)
    prior_nats, _ = caption_losses(model, encoder, [(None, captions[i]) for i in range(n)], images,
                                   args.length, bos, device, args.batch)
    rng = np.random.default_rng(args.seed)
    distractors = [rng.choice([j for j in range(n) if j != i], 4, replace=False) for i in range(n)]
    pairs = [(i, captions[j]) for i in range(n) for j in distractors[i]]
    wrong_nats, wrong_counts = caption_losses(model, encoder, pairs, images, args.length, bos, device, args.batch)
    wrong_nats, wrong_counts = wrong_nats.reshape(n, 4), wrong_counts.reshape(n, 4)
    wrong_prior = prior_nats[np.stack(distractors)]
    raw = (true_nats / true_counts)[:, None] < (wrong_nats / wrong_counts)
    pmi = (prior_nats - true_nats)[:, None] > (wrong_prior - wrong_nats)
    result = dict(method=config.method, step=step, checkpoint=str(args.checkpoint), images=n, **settings,
                  bad_images=int(sum(bad for _, bad in decoded)),
                  caption_bpb=float(true_nats.sum() / (math.log(2) * caption_bytes.sum())),
                  caption_bpb_without_image=float(prior_nats.sum() / (math.log(2) * caption_bytes.sum())),
                  retrieval_raw=float(raw.all(1).mean()), retrieval_pmi=float(pmi.all(1).mean()), chance=.2)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    whatsup = args.whatsup or args.coco.parent / "whatsup"
    if (whatsup / "eval.jsonl").exists():  # the real-image spatial probe, next to COCO when downloaded
        from ntv.spatial_eval import probe
        spatial = dict(method=config.method, step=step, checkpoint=str(args.checkpoint), **settings,
                       **probe(model, encoder, whatsup, tok, device, batch=args.batch))
        target = args.out.parent / f"whatsup_{args.checkpoint.name}.json"
        target.write_text(json.dumps(spatial, indent=2) + "\n")
        print(json.dumps({k: v for k, v in spatial.items() if "acc" in k}, indent=1))


if __name__ == "__main__":
    main()
