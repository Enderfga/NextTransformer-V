"""Summarize finished runs: one row per run from its metrics and evaluation files.

  python -m ntv.report <run dir> [<run dir> ...]       (each with metrics.jsonl and *.json evaluation files)
"""
import json
from pathlib import Path
import sys


def summarize(directory):
    directory = Path(directory)
    events = [json.loads(line) for line in (directory / "metrics.jsonl").read_text().splitlines()]
    start = next(e for e in events if e["event"] == "start")
    evals = [e for e in events if e["event"] == "eval"]
    trains = [e for e in events if e["event"] == "train"]
    last = evals[-1] if evals else {}
    settings = start["settings"]
    row = dict(run=directory.name, method=settings["method"], visual=settings.get("visual", "siglip"),
               position=settings.get("position", "rope1d"), order=settings.get("order", "raster"),
               conv2d=settings.get("conv2d", "none"), updates=last.get("update"),
               caption_bpb=last.get("caption_bpb"), text_bpb=last.get("text_bpb"),
               image_gain=(last.get("caption_bpb_shuffled", 0) - last.get("caption_bpb", 0)) if last else None,
               seconds_per_update=sorted(t["seconds"] for t in trains)[len(trains) // 2] if trains else None)
    for path in sorted(directory.glob("*.json")):
        data = json.loads(path.read_text())
        if path.name.startswith("coco_"):
            row.update(coco_bpb=data["caption_bpb"], coco_retrieval_pmi=data["retrieval_pmi"])
        elif path.name.startswith(("spatial_", "whatsup_")):
            prefix = "whatsup" if path.name.startswith("whatsup_") else f"synth@{data['step']}"
            for key in ("all_acc4", "horizontal_acc4", "vertical_acc4", "all_acc2", "horizontal_acc2", "vertical_acc2"):
                if key in data:
                    row[f"{prefix}:{key}"] = data[key]
    return row


def main():
    rows = [summarize(d) for d in sys.argv[1:]]
    keys = list(dict.fromkeys(k for r in rows for k in r))
    print("\t".join(keys))
    for r in rows:
        print("\t".join(f"{r[k]:.4f}" if isinstance(r.get(k), float) else str(r.get(k, "")) for k in keys))


if __name__ == "__main__":
    main()
