"""Offline packing of the track's rows: a fixed number of image-caption documents per row, filled with text.

A row holds `sequence_len + 1` positions (inputs and shifted targets). Each image-caption document is
`[BOS] [image x 144] caption`, never cropped; ClimbMix documents (`[BOS] text`) fill the remaining space by
best fit from a buffer, and the last one is cropped when nothing fits. The documents of a row are placed in a
seeded random order, so images are spread through the row. Every input is read in a fixed order and every
random choice is seeded, so the same inputs always give the same rows.

Outputs (in --out):
  ids.npy        uint16 (rows, T + 1)  token ids; image positions hold 0 (replaced by the model's hooks)
  img_start.npy  int32  (rows, K)      first position of each image, ascending within a row
  cap_len.npy    int16  (rows, K)      caption tokens after each image
  img_ref.npy    int64  (rows, K)      index into the image table
  images.npz     image table: tar index, byte offset and size of each JPEG
  meta.json      inputs, counts and settings

  python -m ntv.pack --tar-root <data dir> --source cc12m/cc12m-train:0:991 --source cc3m/cc3m-train:0:591 \
      --text <climbmix dir> --text-shards 0 71 --resources <resources> --rows 640000 --out <dir>
"""
import argparse
import bisect
import json
from multiprocessing import Pool
from pathlib import Path
import tarfile

import numpy as np
import pyarrow.parquet as pq

from nanochat.tokenizer import RustBPETokenizer

IMAGE_TOKENS = 144
IMAGE_SUFFIXES = {"jpg", "jpeg", "png", "webp"}


def index_tar(path):
    """(byte offset, size, caption) of every complete image-caption sample, in tar order."""
    samples = {}
    with tarfile.open(path, "r:") as tar:
        for member in tar:
            if not member.isfile():
                continue
            key, _, suffix = member.name.rpartition(".")
            item = samples.setdefault(key, {})
            if suffix.lower() in IMAGE_SUFFIXES:
                item["image"] = (member.offset_data, member.size)
            elif suffix == "txt":
                item["caption"] = tar.extractfile(member).read().decode("utf-8", "replace").strip()
    return [(*item["image"], item["caption"]) for item in samples.values()
            if "image" in item and item.get("caption")]


_TOKENIZER = None


def _tokenizer(resources):
    global _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = RustBPETokenizer.from_directory(str(Path(resources) / "tokenizer"))
    return _TOKENIZER


def tokenize_shard(job):
    """BOS-prefixed token ids of every document of one ClimbMix shard, flattened, plus document lengths."""
    path, resources = job
    tok = _tokenizer(resources)
    texts = pq.read_table(path, columns=["text"]).column("text").to_pylist()
    docs = tok.encode(texts, prepend=tok.get_bos_token_id(), num_threads=1)
    lengths = np.fromiter((len(d) for d in docs), dtype=np.int64, count=len(docs))
    return np.concatenate([np.asarray(d, dtype=np.uint16) for d in docs]), lengths


class TextDocuments:
    """Documents in shard order; a best-fit buffer of `size` documents keyed by length."""

    def __init__(self, shards, size=1000):
        self.shards, self.size = shards, size
        self.shard, self.doc, self.offset = 0, 0, 0
        self.keys, self.docs, self.serial = [], {}, 0
        self.used_tokens = self.cropped = 0

    def _next(self):
        while self.shard < len(self.shards):
            tokens, lengths = self.shards[self.shard]
            if self.doc < len(lengths):
                n = int(lengths[self.doc])
                doc = tokens[self.offset:self.offset + n]
                self.doc += 1
                self.offset += n
                return doc
            self.shard, self.doc, self.offset = self.shard + 1, 0, 0
        raise RuntimeError("text documents exhausted; add text shards")

    def take(self, capacity):
        """The longest buffered document that fits. When none fits, the first `capacity` tokens of the longest one;
        its remainder (a continuation without BOS) goes back to the buffer, so no token is discarded and no long
        document can stay in the buffer for ever."""
        while len(self.keys) < self.size:
            doc = self._next()
            if len(doc) < 2:
                continue
            key = (len(doc), self.serial)
            self.serial += 1
            bisect.insort(self.keys, key)
            self.docs[key] = doc
        i = bisect.bisect_right(self.keys, (capacity, float("inf"))) - 1
        if i < 0:
            key = self.keys.pop()
            whole = self.docs.pop(key)
            doc, rest = whole[:capacity], whole[capacity:]
            self.cropped += 1
            if len(rest) >= 2:
                rest_key = (len(rest), self.serial)
                self.serial += 1
                bisect.insort(self.keys, rest_key)
                self.docs[rest_key] = rest
        else:
            key = self.keys.pop(i)
            doc = self.docs.pop(key)
        self.used_tokens += len(doc)
        return doc


def pack(images, captions, text, rows, per_row, sequence_len, bos, seed, out):
    """Write `rows` rows using images in the order given (one image per caption)."""
    width = sequence_len + 1
    if rows * per_row > len(images):
        raise ValueError(f"{rows} rows need {rows * per_row} images; only {len(images)} are available")
    out.mkdir(parents=True, exist_ok=True)
    ids = np.lib.format.open_memmap(out / "ids.npy", mode="w+", dtype=np.uint16, shape=(rows, width))
    img_start = np.zeros((rows, per_row), np.int32)
    cap_len = np.zeros((rows, per_row), np.int16)
    img_ref = np.zeros((rows, per_row), np.int64)
    for r in range(rows):
        refs = images[r * per_row:(r + 1) * per_row]
        documents = [("image", ref) for ref in refs]
        capacity = width - sum(1 + IMAGE_TOKENS + len(captions[ref]) for ref in refs)
        if capacity <= 0:
            raise ValueError("image documents do not fit in a row")
        while capacity > 0:
            doc = text.take(capacity)
            documents.append(("text", doc))
            capacity -= len(doc)
        order = np.random.default_rng([seed, r]).permutation(len(documents))
        row = np.zeros(width, np.uint16)
        position, k = 0, 0
        for i in order:
            kind, value = documents[i]
            if kind == "text":
                row[position:position + len(value)] = value
                position += len(value)
                continue
            caption = captions[value]
            row[position] = bos
            img_start[r, k], cap_len[r, k], img_ref[r, k] = position + 1, len(caption), value
            position += 1 + IMAGE_TOKENS
            row[position:position + len(caption)] = caption
            position += len(caption)
            k += 1
        if position != width or k != per_row:
            raise RuntimeError("packing error")
        ids[r] = row
    ids.flush()
    np.save(out / "img_start.npy", img_start)
    np.save(out / "cap_len.npy", cap_len)
    np.save(out / "img_ref.npy", img_ref)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tar-root", type=Path, required=True)
    parser.add_argument("--source", action="append", required=True,
                        help="<prefix>:<first>:<last>: tars <tar-root>/<prefix>-<first>..<last>.tar (inclusive)")
    parser.add_argument("--text", type=Path, required=True)
    parser.add_argument("--text-shards", type=int, nargs="+", required=True,
                        help="first and last ClimbMix shard (inclusive), or one shard number")
    parser.add_argument("--resources", type=Path, required=True)
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--per-row", type=int, default=12)
    parser.add_argument("--sequence-len", type=int, default=8192)
    parser.add_argument("--caption-max", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()

    names = []
    for source in args.source:
        prefix, first, last = source.rsplit(":", 2)
        names += [f"{prefix}-{i:04d}.tar" for i in range(int(first), int(last) + 1)]
    tars = [args.tar_root / name for name in names]
    first, last = args.text_shards[0], args.text_shards[-1]
    shards = [args.text / f"shard_{i:05d}.parquet" for i in range(first, last + 1)]
    for path in [*tars, *shards]:
        if not path.exists():
            raise FileNotFoundError(path)
    with Pool(args.workers) as pool:
        indexed = pool.map(index_tar, tars)
        text_shards = pool.map(tokenize_shard, [(p, args.resources) for p in shards])
    tar_id = np.concatenate([np.full(len(s), i, np.int32) for i, s in enumerate(indexed)])
    offsets = np.array([s[0] for t in indexed for s in t], np.int64)
    sizes = np.array([s[1] for t in indexed for s in t], np.int64)
    tok = _tokenizer(args.resources)
    encoded = tok.encode([s[2] for t in indexed for s in t], num_threads=args.workers)
    captions = [c[:args.caption_max] for c in encoded]
    usable = np.array([i for i, c in enumerate(captions) if c], np.int64)
    order = usable[np.random.default_rng(args.seed).permutation(len(usable))]

    text = TextDocuments(text_shards)
    pack(order, captions, text, args.rows, args.per_row, args.sequence_len, tok.get_bos_token_id(), args.seed, args.out)
    np.savez(args.out / "images.npz", tar=tar_id, offset=offsets, size=sizes)
    meta = dict(tars=names, text_shards=[p.name for p in shards], rows=args.rows,
                per_row=args.per_row, sequence_len=args.sequence_len, image_tokens=IMAGE_TOKENS,
                caption_max=args.caption_max, seed=args.seed, images_indexed=len(offsets),
                images_with_captions=len(usable), images_used=args.rows * args.per_row,
                text_tokens_used=text.used_tokens, text_documents_cropped=text.cropped,
                text_tokens_available=int(sum(len(t) for t, _ in text_shards)))
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
