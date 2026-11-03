"""Packing, loading and training run end to end on CPU with synthetic data and a stub encoder."""
from io import BytesIO
import json
import tarfile

import numpy as np
from PIL import Image
import pytest
import torch

import ntv.train
from ntv.data import IMAGE_TOKENS, PackedRows, collate
from ntv.pack import TextDocuments, index_tar, pack

VOCAB = 32768
PER_ROW, T = 3, 600


def make_tar(path, n, rng):
    with tarfile.open(path, "w") as tar:
        for i in range(n):
            image = Image.fromarray(rng.integers(0, 255, (40, 56, 3), dtype=np.uint8))
            buffer = BytesIO()
            image.save(buffer, format="JPEG")
            for name, data in ((f"{i:06d}.jpg", buffer.getvalue()), (f"{i:06d}.txt", f"caption {i}".encode()),
                               (f"{i:06d}.json", b"{}")):
                info = tarfile.TarInfo(name)
                info.size = len(data)
                tar.addfile(info, BytesIO(data))


@pytest.fixture
def packed(tmp_path):
    rng = np.random.default_rng(0)
    tars = tmp_path / "tars"
    tars.mkdir()
    make_tar(tars / "cc12m-train-0000.tar", 40, rng)
    indexed = index_tar(tars / "cc12m-train-0000.tar")
    assert len(indexed) == 40
    captions = [rng.integers(3, 500, rng.integers(1, 30)).tolist() for _ in indexed]
    lengths = rng.integers(2, 300, 400)
    tokens = rng.integers(3, 500, int(lengths.sum())).astype(np.uint16)
    text = TextDocuments([(tokens, lengths)], size=50)
    out = tmp_path / "rows"
    rows = 12
    order = rng.permutation(len(indexed))
    pack(order, captions, text, rows, PER_ROW, T, bos=1, seed=42, out=out)
    np.savez(out / "images.npz", tar=np.zeros(len(indexed), np.int32),
             offset=np.array([s[0] for s in indexed]), size=np.array([s[1] for s in indexed]))
    (out / "meta.json").write_text(json.dumps(dict(tars=["cc12m-train-0000.tar"], image_tokens=IMAGE_TOKENS)))
    return out, tars, captions, order


def test_rows_masks_and_images(packed):
    out, tars, captions, order = packed
    rows = PackedRows(out, tars, image_size=32)
    batch = collate([rows[0], rows[1]])
    x, y, mask = batch["inputs"], batch["targets"], batch["image_mask"]
    assert x.shape == y.shape == mask.shape == (2, T)
    assert int(mask.sum()) == 2 * PER_ROW * IMAGE_TOKENS
    assert batch["images"].shape == (2 * PER_ROW, 3, 32, 32) and batch["bad_images"] == 0
    # A target is ignored exactly when it predicts an image position.
    full = torch.cat((mask, torch.zeros(2, 1, dtype=torch.bool)), 1)
    assert torch.equal(y == -1, full[:, 1:])
    # Caption targets are exactly the caption tokens of the image just before them.
    for i in range(2):
        # A row holds exactly its assigned images, placed in the row's shuffled document order.
        assert sorted(rows.img_ref[i]) == sorted(order[i * PER_ROW:(i + 1) * PER_ROW])
        assert (np.diff(rows.img_start[i]) > 0).all()
        for k in range(PER_ROW):
            start, n, ref = rows.img_start[i, k], rows.cap_len[i, k], rows.img_ref[i, k]
            assert x[i, start - 1] == 1  # BOS opens each image document
            span = slice(start + IMAGE_TOKENS - 1, start + IMAGE_TOKENS - 1 + n)
            assert batch["caption"][i, span].all() and y[i, span].tolist() == captions[ref]
    assert not (batch["caption"] & batch["text"]).any() and not (batch["text"] & (y == -1)).any()
    assert int(batch["caption"].sum()) == int(rows.cap_len[:2].sum())


def test_training_runs_evaluates_and_resumes(packed, tmp_path, monkeypatch):
    out, tars, _, _ = packed

    class StubEncoder(torch.nn.Module):
        def __init__(self, path, device):
            super().__init__()
            self.proj = torch.randn(3, 4608, generator=torch.Generator().manual_seed(0))

        def forward(self, images):
            pooled = images.float().mean((2, 3)) / 255  # (N, 3): depends on the image
            return (pooled @ self.proj).repeat_interleave(IMAGE_TOKENS, 0)

    monkeypatch.setattr(ntv.train, "FrozenSiglip", StubEncoder)
    resources = tmp_path / "resources" / "tokenizer"
    resources.mkdir(parents=True)
    torch.save(torch.ones(VOCAB, dtype=torch.int32), resources / "token_bytes.pt")
    run = tmp_path / "run"
    common = ["--method", "dense_full", "--shape", "width=48,heads=2,layers=2,mlp_width=192",
              "--train-rows", str(out), "--val-rows", str(out), "--tars", str(tars), "--encoder", "unused",
              "--resources", str(resources.parent), "--output", str(run), "--updates", "4",
              "--rows-per-update", "2", "--micro-batch", "1", "--sequence-len", str(T), "--eval-every", "2",
              "--eval-rows", "4", "--checkpoint-every", "2", "--log-every", "1", "--workers", "0"]
    ntv.train.main(common + ["--stop-after", "2"])
    ntv.train.main(common)
    events = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    starts = [e["update"] for e in events if e["event"] == "start"]
    trained = [e["update"] for e in events if e["event"] == "train"]
    assert starts == [0, 2] and trained == [1, 2, 3, 4]
    evals = [e for e in events if e["event"] == "eval"]
    assert [e["update"] for e in evals] == [0, 2, 4]
    for e in evals:
        assert all(np.isfinite(e[k]) for k in ("caption_bpb", "text_bpb", "caption_bpb_shuffled"))
    assert json.loads((run / "latest.json").read_text())["step"] == 4
