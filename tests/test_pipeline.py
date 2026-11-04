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


def test_coco_evaluation_runs_on_a_checkpoint(packed, tmp_path, monkeypatch):
    import sys
    import ntv.evaluate
    out, tars, _, _ = packed
    test_training_runs_evaluates_and_resumes(packed, tmp_path, monkeypatch)
    rng = np.random.default_rng(1)
    coco = tmp_path / "coco"
    (coco / ntv.evaluate.IMAGE_DIR).mkdir(parents=True)
    lines = ["filepath,sentids,filename,imgid,split,cocoid,tokens,raw,sentid"]
    for i in range(7):
        Image.fromarray(rng.integers(0, 255, (30, 30, 3), dtype=np.uint8)).save(coco / ntv.evaluate.IMAGE_DIR / f"{i}.jpg")
        raw = json.dumps([f"a photo number {i}", "other"]).replace('"', '""')
        lines.append(f'val2014,"[]",{i}.jpg,"[]",test,{i},"[]","{raw}","[]"')
    (coco / "test_5k_mscoco_2014.csv").write_text("\n".join(lines) + "\n")

    class FakeTokenizer:
        @staticmethod
        def from_directory(path):
            return FakeTokenizer()

        def get_bos_token_id(self):
            return 1

        def encode(self, texts, num_threads=1):
            return [[3 + (ord(ch) % 200) for ch in t] for t in texts]

    monkeypatch.setattr(ntv.evaluate, "RustBPETokenizer", FakeTokenizer)
    run = tmp_path / "run"
    checkpoint = run / json.loads((run / "latest.json").read_text())["checkpoint"]
    result_path = tmp_path / "coco.json"
    monkeypatch.setattr(sys, "argv", ["evaluate", "--checkpoint", str(checkpoint), "--coco", str(coco),
                                      "--encoder", "unused", "--resources", str(tmp_path / "resources"),
                                      "--out", str(result_path), "--length", "200", "--batch", "4"])
    from ntv.synth import write_eval
    write_eval(tmp_path / "whatsup", 4, seed=3)  # any probe in the eval format, next to the COCO directory
    ntv.evaluate.main()
    result = json.loads(result_path.read_text())
    assert result["images"] == 7 and result["step"] == 4
    whatsup = json.loads((tmp_path / f"whatsup_{checkpoint.name}.json").read_text())
    assert whatsup["images"] == 4 and 0 <= whatsup["all_acc2"] <= 1
    assert 0 <= result["retrieval_raw"] <= 1 and 0 <= result["retrieval_pmi"] <= 1
    assert np.isfinite(result["caption_bpb"]) and np.isfinite(result["caption_bpb_without_image"])


def test_spatial_probe_runs_on_a_spatial_checkpoint(packed, tmp_path, monkeypatch):
    import sys
    import ntv.evaluate
    import ntv.spatial_eval
    from ntv.synth import write_eval
    out, tars, _, _ = packed

    class StubEncoder(torch.nn.Module):
        def __init__(self, path, device):
            super().__init__()
            self.proj = torch.randn(3, 4608, generator=torch.Generator().manual_seed(0))

        def forward(self, images):
            tokens = (images.shape[-1] // 32) ** 2  # 144 at 384 px, more on larger images
            return (images.float().mean((2, 3)) / 255 @ self.proj).repeat_interleave(tokens, 0)

    class FakeTokenizer:
        @staticmethod
        def from_directory(path):
            return FakeTokenizer()

        def get_bos_token_id(self):
            return 1

        def encode(self, texts, num_threads=1):
            return [[3 + (ord(ch) % 200) for ch in t] for t in texts]

    monkeypatch.setattr(ntv.train, "FrozenSiglip", StubEncoder)
    monkeypatch.setattr(ntv.spatial_eval, "RustBPETokenizer", FakeTokenizer)
    resources = tmp_path / "resources" / "tokenizer"
    resources.mkdir(parents=True)
    torch.save(torch.ones(VOCAB, dtype=torch.int32), resources / "token_bytes.pt")
    run = tmp_path / "run"
    ntv.train.main(["--method", "gdn_hybrid", "--shape", "width=48,heads=2,layers=4,mlp_width=192",
                    "--train-rows", str(out), "--val-rows", str(out), "--tars", str(tars), "--encoder", "unused",
                    "--resources", str(resources.parent), "--output", str(run), "--updates", "2",
                    "--rows-per-update", "2", "--micro-batch", "1", "--sequence-len", str(T), "--eval-every", "2",
                    "--eval-rows", "2", "--checkpoint-every", "2", "--log-every", "1", "--workers", "0",
                    "--position", "rope2d", "--order", "hilbert", "--conv2d", "k"])
    probe = tmp_path / "probe"
    write_eval(probe, 6, seed=3)
    checkpoint = run / json.loads((run / "latest.json").read_text())["checkpoint"]
    result_path = tmp_path / "spatial.json"
    monkeypatch.setattr(sys, "argv", ["spatial_eval", "--checkpoint", str(checkpoint), "--probe", str(probe),
                                      "--resources", str(resources.parent), "--out", str(result_path),
                                      "--batch", "8"])
    ntv.spatial_eval.main()
    result = json.loads(result_path.read_text())
    assert result["images"] == 6 and result["conv2d"] == "k" and result["position"] == "rope2d"
    assert 0 <= result["all_acc4"] <= 1 and 0 <= result["all_acc2"] <= 1
    assert result["horizontal_count"] + result["vertical_count"] == 6
    for mode in ("vflip", "hflip", "shuffle"):
        assert 0 <= result[f"{mode}_vertical_acc2"] <= 1 and np.isfinite(result[f"{mode}_all_margin2"])
    assert np.isfinite(result["vertical_margin2"]) and "vflip_all_count" not in result
    assert result["image_size"] == 384 and result["grid"] == 12
    larger = tmp_path / "spatial576.json"
    monkeypatch.setattr(sys, "argv", ["spatial_eval", "--checkpoint", str(checkpoint), "--probe", str(probe),
                                      "--resources", str(resources.parent), "--out", str(larger),
                                      "--batch", "8", "--size", "576"])
    ntv.spatial_eval.main()
    result = json.loads(larger.read_text())
    assert result["grid"] == 18 and result["images"] == 6 and 0 <= result["vflip_vertical_acc2"] <= 1


def test_text_buffer_conserves_tokens_and_drains_long_documents():
    rng = np.random.default_rng(5)
    lengths = np.concatenate([rng.integers(2, 200, 3000), rng.integers(5000, 20000, 300)])
    rng.shuffle(lengths)
    tokens = rng.integers(3, 500, int(lengths.sum())).astype(np.uint16)
    text = TextDocuments([(tokens, lengths)], size=100)
    taken = 0
    for _ in range(400):  # rows with a small text budget: long documents never fit whole
        capacity = 300
        while capacity > 0:
            doc = text.take(capacity)
            assert 0 < len(doc) <= capacity
            capacity -= len(doc)
            taken += len(doc)
    buffered = sum(len(d) for d in text.docs.values())
    consumed = int(lengths[:text.doc].sum())  # documents read so far (one shard)
    assert taken + buffered == consumed == text.used_tokens + buffered
    assert max(len(d) for d in text.docs.values()) < 20000  # long documents are being worked through
