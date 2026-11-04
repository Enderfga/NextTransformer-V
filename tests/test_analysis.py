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


def test_hard_probe_contrasts_against_the_hard_seed_spread(tmp_path):
    hard = {"h-rope1d": .90, "h-rope1d-s2": .91, "h-nope-conv2d": .95, "h-nope-conv2d-s2": .93, "h-nope": .92,
            "h-rope2d": .905}
    for name, value in hard.items():
        d = write(tmp_path, name, .999)
        (d / "hard_checkpoint_00005000.json").write_text(json.dumps(dict(
            step=5000, vertical_acc2=value, horizontal_acc2=.9, vflip_vertical_acc2=.6, hflip_horizontal_acc2=.3)))
    from ntv.analysis import secondary
    runs = {d.name: load_run(d) for d in tmp_path.iterdir()}
    sec = secondary(runs)
    assert abs(sec["hard_seed_spread"]["vertical_acc2"] - .02) < 1e-9
    rows = {r["treated"]: r["diff"] for r in sec["hard_contrasts"]}
    assert abs(rows["h-nope-conv2d"]["vertical_acc2"] - .03) < 1e-9 and rows["h-rope1d-conv2d"]["vertical_acc2"] is None
    html = render(runs, decide(runs), None, sec)
    assert "<td>+3.0</td>" in html          # 3 points is within twice the spread (4 points): not bold
    assert "<td>4.0</td>" in html           # the 2x spread row
    (tmp_path / "h-nope" / "hard_checkpoint_00005000.json").write_text(json.dumps(dict(
        step=5000, vertical_acc2=.85, horizontal_acc2=.9, vflip_vertical_acc2=.6, hflip_horizontal_acc2=.3)))
    runs = {d.name: load_run(d) for d in tmp_path.iterdir()}
    assert "<td><b>+10.0</b></td>" in render(runs, decide(runs))   # 10 points > twice the 2-point spread


def test_conclusion_is_rendered_at_the_top(tmp_path):
    write(tmp_path, "h-rope1d", .999)
    runs = {d.name: load_run(d) for d in tmp_path.iterdir()}
    html = render(runs, decide(runs), conclusion="First **finding**.\n\n- one\n- two <x>")
    assert "<p>First <b>finding</b>.</p>" in html and "<li>two &lt;x&gt;</li>" in html
    assert html.index("结论") < html.index("预注册判定")
    assert "<h2>结论</h2>" not in render(runs, decide(runs))
