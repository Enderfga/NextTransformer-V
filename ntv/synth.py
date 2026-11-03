"""Synthetic spatial-relation images: a controlled probe of how a backbone reads 2D position.

Each 384 x 384 image holds a 4 x 4 grid of cells. Two target objects (distinct colour-shape pairs) share a row or a
column, so their relation is purely horizontal ("to the left of" / "to the right of") or purely vertical ("above" /
"below"); zero to two distractors occupy other cells. The caption states the relation of the two targets:
"a red circle is above a blue square."

In raster order a horizontal neighbour is the next token but a vertical neighbour is a whole image row away, so the
gap between horizontal and vertical accuracy measures how much 2D structure a backbone recovers.

  python -m ntv.synth train --out <dir> --shards 120 --per-shard 5000 --seed 1
  python -m ntv.synth eval  --out <dir> --count 2000 --seed 2       (images + eval.jsonl with 4 candidates each)
"""
import argparse
from io import BytesIO
import json
from multiprocessing import Pool
from pathlib import Path
import tarfile

import numpy as np
from PIL import Image, ImageDraw

SIZE, GRID = 384, 4
CELL = SIZE // GRID
COLORS = {"red": (220, 40, 40), "green": (40, 170, 60), "blue": (40, 80, 220), "yellow": (235, 210, 40),
          "purple": (140, 60, 180), "orange": (240, 140, 30), "white": (245, 245, 245), "black": (20, 20, 20)}
SHAPES = ("circle", "square", "triangle", "diamond")
RELATIONS = ("to the left of", "to the right of", "above", "below")
HORIZONTAL = {"to the left of", "to the right of"}


def draw_shape(draw, shape, color, cx, cy, r):
    if shape == "circle":
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=color)
    elif shape == "square":
        draw.rectangle((cx - r, cy - r, cx + r, cy + r), fill=color)
    elif shape == "triangle":
        draw.polygon(((cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)), fill=color)
    else:
        draw.polygon(((cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)), fill=color)


def relation(a, b):
    """Relation of cell a to cell b ((row, col)); they share a row or a column."""
    if a[0] == b[0]:
        return "to the left of" if a[1] < b[1] else "to the right of"
    return "above" if a[0] < b[0] else "below"


def sample(rng):
    """One scene: image, the two targets, their relation and their distance in cells."""
    kinds = rng.choice(len(COLORS) * len(SHAPES), size=4, replace=False)
    objects = [(list(COLORS)[k // len(SHAPES)], SHAPES[k % len(SHAPES)]) for k in kinds]
    if rng.random() < .5:  # same row
        row = int(rng.integers(GRID))
        c1, c2 = rng.choice(GRID, size=2, replace=False)
        cells = [(row, int(c1)), (row, int(c2))]
    else:                  # same column
        col = int(rng.integers(GRID))
        r1, r2 = rng.choice(GRID, size=2, replace=False)
        cells = [(int(r1), col), (int(r2), col)]
    free = [(r, c) for r in range(GRID) for c in range(GRID) if (r, c) not in cells]
    distractors = int(rng.integers(0, 3))
    for i in rng.choice(len(free), size=distractors, replace=False):
        cells.append(free[int(i)])
    base = rng.integers(90, 200, size=3)
    pixels = np.clip(base + rng.normal(0, 12, (SIZE, SIZE, 3)), 0, 255).astype(np.uint8)
    image = Image.fromarray(pixels)
    draw = ImageDraw.Draw(image)
    for (color, shape), (r, c) in zip(objects, cells):
        jitter = rng.integers(-12, 13, size=2)
        draw_shape(draw, shape, COLORS[color], c * CELL + CELL // 2 + int(jitter[0]),
                   r * CELL + CELL // 2 + int(jitter[1]), int(rng.integers(22, 34)))
    rel = relation(cells[0], cells[1])
    distance = abs(cells[0][0] - cells[1][0]) + abs(cells[0][1] - cells[1][1])
    return image, objects[0], objects[1], rel, distance


def article(word):
    return "an" if word[0] in "aeiou" else "a"


def caption(a, b, rel):
    return f"{article(a[0])} {a[0]} {a[1]} is {rel} {article(b[0])} {b[0]} {b[1]}."


def jpeg(image):
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


def write_shard(job):
    path, seed, count = job
    rng = np.random.default_rng(seed)
    with tarfile.open(path, "w") as tar:
        for i in range(count):
            image, a, b, rel, _ = sample(rng)
            for name, data in ((f"{i:06d}.jpg", jpeg(image)), (f"{i:06d}.txt", caption(a, b, rel).encode())):
                info = tarfile.TarInfo(name)
                info.size = len(data)
                tar.addfile(info, BytesIO(data))
    return path.name


def write_eval(out, count, seed):
    rng = np.random.default_rng(seed)
    (out / "images").mkdir(parents=True, exist_ok=True)
    with open(out / "eval.jsonl", "w") as handle:
        for i in range(count):
            image, a, b, rel, distance = sample(rng)
            image.save(out / "images" / f"{i:05d}.jpg", quality=90)
            handle.write(json.dumps(dict(image=f"images/{i:05d}.jpg", relation=rel, distance=distance,
                                         axis="horizontal" if rel in HORIZONTAL else "vertical",
                                         candidates={r: caption(a, b, r) for r in RELATIONS})) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("split", choices=("train", "eval"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shards", type=int, default=120)
    parser.add_argument("--per-shard", type=int, default=5000)
    parser.add_argument("--count", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.split == "eval":
        write_eval(args.out, args.count, args.seed)
        return
    jobs = [(args.out / f"synth-train-{i:04d}.tar", args.seed * 100003 + i, args.per_shard) for i in range(args.shards)]
    with Pool(args.workers) as pool:
        for name in pool.imap_unordered(write_shard, jobs):
            print(name, flush=True)


if __name__ == "__main__":
    main()
