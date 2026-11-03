"""The pre-registered decision rule, on hand-made results."""
import json

from ntv.analysis import decide, load_run, render

BASE = dict(caption_bpb=1.0, text_bpb=.9, caption_bpb_shuffled=1.4)


def write(root, name, vertical, horizontal=.9, bpb=None):
    d = root / name
    d.mkdir(parents=True)
    events = [dict(event="start", update=0, settings={}), dict(event="train", update=10, seconds=3.0),
              dict(event="eval", update=5000, **(bpb or BASE))]
    (d / "metrics.jsonl").write_text("\n".join(json.dumps(e) for e in events) + "\n")
    for step in (1000, 3000, 5000):
        (d / f"spatial_checkpoint_{step:08d}.json").write_text(json.dumps(
            dict(step=step, vertical_acc2=vertical * step / 5000, horizontal_acc2=horizontal * step / 5000)))
    return d


def test_decision_rule(tmp_path):
    values = {"d-rope1d": .8, "d-nope": .7, "h-rope1d": .70, "h-rope1d-s2": .72, "h-nope": .60, "h-rope2d": .75,
              "h-nope-conv2d": .70, "h-nope-conv2d-s2": .69, "h-rope1d-conv2d": .73, "r-h-nope": .55}
    for name, value in values.items():
        write(tmp_path, name, value)
    write(tmp_path, "r-h-nope-conv2d", .61, bpb=dict(BASE, text_bpb=.905))
    runs = {d.name: load_run(d) for d in tmp_path.iterdir()}
    decision = decide(runs)
    assert abs(decision["seed_spread"] - .02) < 1e-9 and decision["required_complete"]
    verdicts = {d["mode"]: d["verdict"] for d in decision["decisions"]}
    assert verdicts == {"NoPE": "effective", "RoPE-1D": "negative", "NoPE, raw patches": "inconclusive"}
    html = render(runs, decision)
    assert "有效" in html and "无效" in html and "不确定" in html


def test_decision_waits_for_both_seed_pairs(tmp_path):
    for name in ("h-nope", "h-nope-conv2d", "h-rope1d", "h-rope1d-s2"):
        write(tmp_path, name, .7)
    decision = decide({d.name: load_run(d) for d in tmp_path.iterdir()})
    assert decision["seed_spread"] is None
    assert all(d["verdict"] == "pending" for d in decision["decisions"])


def test_secondary_analyses_render_without_entering_the_decision(tmp_path):
    for name in ("h-rope1d", "h-nope"):
        d = write(tmp_path, name, .999)
        for step in (1000, 5000):
            (d / f"hard_checkpoint_{step:08d}.json").write_text(json.dumps(dict(
                step=step, vertical_acc2=.9, horizontal_acc2=.88, vflip_vertical_acc2=.6, hflip_horizontal_acc2=.3)))
    runs = {d.name: load_run(d) for d in tmp_path.iterdir()}
    assert runs["h-rope1d"]["hard"][5000]["vflip_vertical_acc2"] == .6
    decision = decide(runs)
    position = {"siglip": dict(row_r2=.86, col_r2=.89, row_acc=.36, col_acc=.4, chance_acc=1 / 12, test_images=256)}
    html = render(runs, decision, position)
    assert "R² 0.86 / 0.89" in html and "难版探针" in html and "垂直: 60.0%" in html
