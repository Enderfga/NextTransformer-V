"""Calibrate methods to the track's compute anchor with NextTransformer's v2 grid rule.

The anchor is the Dense baseline's admitted shell (full attention, as admitted for the 2,048-token protocol)
at the track's context. A method whose admitted structure is already within 1% keeps it. Otherwise every
allowed depth in 16..32 is audited at the starting FFN width and at +128 (the FFN cost is linear in the
width), every FFN width on the [3d, 5d] multiple-of-128 grid is estimated, and the estimated-within-1%
candidates are fully audited in tie-break order: FFN closest to the start, smallest |error|, depth closest
to the start, shallower. The first audited candidate within 1% is selected; the whole table is recorded.
Selection never looks at any training or evaluation result.

  python -m ntv.calibrate --methods dense_full gdn gdn_hybrid nsa qsa mhc --out <dir> --workers 64
"""
import argparse
from dataclasses import replace
import hashlib
import json
from multiprocessing import Pool
from pathlib import Path

import torch

from ntv.cost import Track, audit_mm, stages
from ntv.model import track_config

TOLERANCE = .01
ANCHOR = "dense_full"


def _key(config, track, stage):
    identity = dict(model=config.fingerprint, track=repr(track), stage=stage)
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def _audit(job):
    config, track, stage, cache = job
    path = Path(cache) / (_key(config, track, stage) + ".json")
    if path.exists():
        return json.loads(path.read_text())["total"]
    torch.set_num_threads(1)
    report = audit_mm(config, track, stage=stage)
    if report["unknown"]:
        raise RuntimeError(f"{config.method}: uncovered operators {report['unknown']}")
    path.write_text(json.dumps(dict(total=report["total"], parameters=report["parameters"],
                                    flops_per_token=report["flops_per_token"])))
    return report["total"]


class Calibrator:
    def __init__(self, track, cache, workers):
        self.track, self.cache = track, Path(cache)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.pool = Pool(workers)

    def costs(self, configs):
        """Update-weighted FLOPs per token of each configuration, audited in parallel."""
        jobs, index = [], []
        for i, c in enumerate(configs):
            for stage, depth, updates in stages(c, self.track):
                jobs.append((c, self.track, stage, str(self.cache)))
                index.append((i, updates))
        totals = self.pool.map(_audit, jobs)
        weighted = [0] * len(configs)
        for (i, updates), total in zip(index, totals):
            weighted[i] += updates * total
        return [w / self.track.total_tokens for w in weighted]

    def calibrate(self, start, anchor):
        initial = self.costs([start])[0]
        record = dict(method=start.method, start=dict(layers=start.layers, mlp_width=start.mlp_width),
                      start_flops_per_token=initial, start_error=initial / anchor - 1)
        if abs(initial / anchor - 1) <= TOLERANCE:
            return start, dict(record, rule="admitted structure within tolerance", error=initial / anchor - 1)
        depths = []
        for n in range(16, 33):
            try:
                depths.append(replace(start, layers=n))
            except ValueError:  # not a whole repeating unit of a hybrid layout
                pass
        base = self.costs(depths)
        wider = self.costs([replace(c, mlp_width=c.mlp_width + 128) for c in depths])
        width, candidates, table = start.width, [], []
        for c, cost, more in zip(depths, base, wider):
            slope = (more - cost) / 128
            table.append(dict(layers=c.layers, flops_per_token=cost, ffn_slope=slope))
            for ffn in range(3 * width, 5 * width + 1, 128):
                estimate = (cost + (ffn - c.mlp_width) * slope) / anchor - 1
                if abs(estimate) <= TOLERANCE:
                    candidates.append(dict(layers=c.layers, mlp_width=ffn, estimated_error=estimate))
        candidates.sort(key=lambda t: (abs(t["mlp_width"] - start.mlp_width), abs(t["estimated_error"]),
                                       abs(t["layers"] - start.layers), t["layers"]))
        record.update(rule="v2 grid", depths=table, candidates=candidates)
        for candidate in candidates:
            chosen = replace(start, layers=candidate["layers"], mlp_width=candidate["mlp_width"])
            error = self.costs([chosen])[0] / anchor - 1
            candidate["audited_error"] = error
            if abs(error) <= TOLERANCE:
                return chosen, dict(record, error=error)
        raise RuntimeError(f"{start.method}: no grid candidate within tolerance")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()
    track = Track()
    calibrator = Calibrator(track, args.cache or args.out / "cache", args.workers)
    anchor_config = track_config(ANCHOR, sequence_len=track.sequence_len)
    anchor = calibrator.costs([anchor_config])[0]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "anchor.json").write_text(json.dumps(dict(method=ANCHOR, track=repr(track), flops_per_token=anchor,
                                                          model=anchor_config.serialized()), indent=2) + "\n")
    print(f"anchor {ANCHOR}: {anchor / 1e9:.4f} GFLOPs per token", flush=True)
    for method in args.methods:
        chosen, record = calibrator.calibrate(track_config(method, sequence_len=track.sequence_len), anchor)
        (args.out / f"{method}.json").write_text(json.dumps(chosen.serialized(), indent=2) + "\n")
        (args.out / f"{method}.calibration.json").write_text(json.dumps(record, indent=2) + "\n")
        print(f"{method}: layers {chosen.layers}, FFN {chosen.mlp_width}, error {record['error']:+.4%} "
              f"({record['rule']})", flush=True)


if __name__ == "__main__":
    main()
