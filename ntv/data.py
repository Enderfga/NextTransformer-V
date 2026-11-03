"""Packed rows as training batches: token ids, image masks, target masks and decoded images.

A batch holds `inputs`/`targets` (B, T), `image_mask` (B, T) for the model, `caption`/`text` (B, T) masks over
targets for evaluation, and `images` (N, 3, S, S) uint8 in the row-major order of `image_mask`. Targets are -1
wherever the predicted position is an image position. A JPEG that fails to decode becomes a black image and is
counted in `bad_images`; one bad sample never stops a run.
"""
from io import BytesIO
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, Sampler

IMAGE_TOKENS = 144


def decode_image(data, size=384):
    """JPEG bytes -> uint8 (3, size, size); a black image and True when decoding fails."""
    try:
        image = Image.open(BytesIO(data))
        image.draft("RGB", (size, size))  # JPEG DCT downscaling: faster decode
        image = image.convert("RGB").resize((size, size), Image.BILINEAR)
        return np.asarray(image).transpose(2, 0, 1), False
    except Exception:
        return np.zeros((3, size, size), np.uint8), True


class PackedRows(Dataset):
    def __init__(self, directory, tar_dir, image_size=384):
        directory = Path(directory)
        self.meta = json.loads((directory / "meta.json").read_text())
        if self.meta["image_tokens"] != IMAGE_TOKENS:
            raise ValueError("packed rows use a different image token count")
        self.ids = np.load(directory / "ids.npy", mmap_mode="r")
        self.img_start = np.load(directory / "img_start.npy")
        self.cap_len = np.load(directory / "cap_len.npy")
        self.img_ref = np.load(directory / "img_ref.npy")
        table = np.load(directory / "images.npz")
        self.tar, self.offset, self.size = table["tar"], table["offset"], table["size"]
        self.tar_paths = [str(Path(tar_dir) / name) for name in self.meta["tars"]]
        self.image_size = image_size
        self._files = {}

    def __len__(self):
        return len(self.ids)

    def _image(self, ref):
        path = self.tar_paths[self.tar[ref]]
        handle = self._files.get(path)
        if handle is None:
            handle = self._files[path] = open(path, "rb")
        handle.seek(int(self.offset[ref]))
        return decode_image(handle.read(int(self.size[ref])), self.image_size)

    def __getitem__(self, row):
        decoded = [self._image(ref) for ref in self.img_ref[row]]
        return dict(row=row, ids=np.asarray(self.ids[row], dtype=np.int64), img_start=self.img_start[row],
                    cap_len=self.cap_len[row], images=np.stack([d[0] for d in decoded]),
                    bad_images=sum(d[1] for d in decoded))


def collate(samples):
    ids = torch.from_numpy(np.stack([s["ids"] for s in samples]))
    b, width = ids.shape
    full = torch.zeros((b, width), dtype=torch.bool)       # image positions over the T + 1 row
    caption = torch.zeros((b, width), dtype=torch.bool)    # caption-token positions over the T + 1 row
    for i, s in enumerate(samples):
        for start, n in zip(s["img_start"].tolist(), s["cap_len"].tolist()):
            full[i, start:start + IMAGE_TOKENS] = True
            caption[i, start + IMAGE_TOKENS:start + IMAGE_TOKENS + n] = True
    targets = ids[:, 1:].clone()
    targets[full[:, 1:]] = -1
    caption_targets = caption[:, 1:]
    return dict(rows=torch.tensor([s["row"] for s in samples]), inputs=ids[:, :-1].contiguous(), targets=targets,
                image_mask=full[:, :-1].contiguous(), caption=caption_targets.contiguous(),
                text=(~full[:, 1:] & ~caption_targets).contiguous(),
                images=torch.from_numpy(np.concatenate([s["images"] for s in samples])),
                bad_images=sum(s["bad_images"] for s in samples))


class RankRows(Sampler):
    """Rows of one rank, update after update: update s uses rows [s*R, (s+1)*R), split evenly over ranks."""

    def __init__(self, rows_per_update, rank, world, start_update, updates):
        if rows_per_update % world:
            raise ValueError("rows per update must divide evenly over ranks")
        self.per_rank = rows_per_update // world
        self.rows_per_update, self.rank = rows_per_update, rank
        self.start_update, self.updates = start_update, updates

    def __iter__(self):
        for update in range(self.start_update, self.updates):
            first = update * self.rows_per_update + self.rank * self.per_rank
            yield from range(first, first + self.per_rank)

    def __len__(self):
        return (self.updates - self.start_update) * self.per_rank
