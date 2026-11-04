"""Training FLOPs per token of every pilot variant, against the Dense anchor (must be within 1%).

Position mode, patch order, the 2D short convolution and the visual input change the audited graph only slightly;
this audits each combination used by configs/pilot/jobs.txt with the calibrated depth and width of its backbone.

  python scripts/audit_variants.py
"""
import json
from dataclasses import replace
from pathlib import Path

import torch

from ntv.cost import Track, audit_mm
from ntv.model import track_config
from ntv.spatial import RAW_DIM

CONFIGS = Path(__file__).resolve().parents[1] / "configs" / "track"
VARIANTS = (
    ("dense_full", {}, "siglip"), ("dense_full", dict(position="nope"), "siglip"),
    ("dense_full", {}, "raw32"), ("dense_full", dict(position="nope"), "raw32"),
    ("gdn_hybrid", {}, "siglip"), ("gdn_hybrid", dict(position="nope"), "siglip"),
    ("gdn_hybrid", dict(position="rope2d"), "siglip"), ("gdn_hybrid", dict(position="nope", conv2d="k"), "siglip"),
    ("gdn_hybrid", dict(conv2d="k"), "siglip"), ("gdn_hybrid", dict(position="rope2d", conv2d="k"), "siglip"),
    ("gdn_hybrid", dict(position="nope", order="hilbert"), "siglip"), ("gdn_hybrid", dict(position="nope"), "raw32"),
    ("gdn_hybrid", dict(position="nope", conv2d="k"), "raw32"), ("gdn_hybrid", {}, "raw32"),
    ("gdn_hybrid", dict(position="rope2d"), "raw32"), ("gdn_hybrid", dict(conv2d="k"), "raw32"),
)


def calibrated(method):
    saved = json.loads((CONFIGS / f"{method}.json").read_text())
    return replace(track_config(method), layers=saved["layers"], mlp_width=saved["mlp_width"])


def main():
    torch.set_num_threads(4)
    anchor = json.loads((CONFIGS / "anchor.json").read_text())["flops_per_token"]
    print(f"anchor {anchor / 1e9:.4f} GFLOPs/token")
    worst = 0.
    for method, spatial, visual in VARIANTS:
        dim = RAW_DIM if visual == "raw32" else 4608
        report = audit_mm(calibrated(method), Track(visual_dim=dim), spatial=spatial or None)
        error = report["flops_per_token"] / anchor - 1
        worst = max(worst, abs(error))
        assert not report["unknown"], report["unknown"]
        print(f"{method:11s} {visual:6s} {json.dumps(spatial):45s} {report['flops_per_token'] / 1e9:.4f} {100 * error:+.3f}%")
    assert worst < .01, worst
    print(f"largest deviation {100 * worst:.3f}%")


if __name__ == "__main__":
    main()
