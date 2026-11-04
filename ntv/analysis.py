"""The pre-registered decision (docs/spatial-preregistration.md) and a one-page report of the 2D-position runs.

  python -m ntv.analysis <results dir> --out <report.html> [--json <decision.json>]

<results dir> holds one directory per run with `metrics.jsonl` and the evaluation files written by the run's
evaluation (`spatial_checkpoint_*.json`, `coco_*.json`, `whatsup_*.json`, and the secondary `hard_checkpoint_*.json`),
and optionally `position-probe/{siglip,raw32}.json`. Only the pre-registered metric enters the decision; the harder
probe, the token-order scrambles and the position probe are descriptive (docs/spatial-addendum.md).
"""
import argparse
from html import escape
import json
from pathlib import Path

GAIN = .05          # at least 5 percentage points of vertical_acc2
NOISE = 2           # and more than twice the seed spread
BPB_SLACK = .003    # caption and text BPB not worse by more than this
SEED_PAIRS = (("h-rope1d", "h-rope1d-s2"), ("h-nope-conv2d", "h-nope-conv2d-s2"))
DECISIONS = (("NoPE", "h-nope-conv2d", "h-nope"), ("RoPE-1D", "h-rope1d-conv2d", "h-rope1d"),
             ("NoPE, raw patches", "r-h-nope-conv2d", "r-h-nope"))
REQUIRED = ("d-rope1d", "d-nope", "h-rope1d", "h-nope", "h-rope2d", "h-nope-conv2d")
# Secondary, descriptive (docs/spatial-addendum.md): differences on the harder probe, against the seed spread measured
# on that probe with the same two seed pairs.
HARD_KEYS = (("vertical_acc2", "垂直"), ("horizontal_acc2", "水平"), ("vflip_vertical_acc2", "垂直翻转后的垂直"),
             ("hflip_horizontal_acc2", "水平翻转后的水平"))
CONTRASTS = tuple((f"二维卷积 · {mode}", treated, base) for mode, treated, base in DECISIONS) + (
    ("RoPE-2D 对 RoPE-1D（混合）", "h-rope2d", "h-rope1d"), ("NoPE 对 RoPE-1D（混合）", "h-nope", "h-rope1d"),
    ("NoPE 对 RoPE-1D（Dense）", "d-nope", "d-rope1d"), ("混合对 Dense（RoPE-1D）", "h-rope1d", "d-rope1d"),
    ("混合对 Dense（NoPE）", "h-nope", "d-nope"), ("Hilbert 对逐行顺序（混合 · NoPE）", "h-nope-hilbert", "h-nope"),
    ("二维卷积 · RoPE-2D", "h-rope2d-conv2d", "h-rope2d"))
LABELS = {"d-rope1d": "Dense · RoPE-1D", "d-nope": "Dense · NoPE", "h-rope1d": "Hybrid · RoPE-1D",
          "h-rope1d-s2": "Hybrid · RoPE-1D (seed 43)", "h-nope": "Hybrid · NoPE", "h-rope2d": "Hybrid · RoPE-2D",
          "h-nope-conv2d": "Hybrid · NoPE + 2D conv", "h-nope-conv2d-s2": "Hybrid · NoPE + 2D conv (seed 43)",
          "h-rope1d-conv2d": "Hybrid · RoPE-1D + 2D conv", "h-nope-hilbert": "Hybrid · NoPE, Hilbert order",
          "h-rope2d-conv2d": "Hybrid · RoPE-2D + 2D conv", "r-h-nope": "Hybrid · NoPE, raw patches",
          "r-h-nope-conv2d": "Hybrid · NoPE + 2D conv, raw patches", "r-h-rope1d": "Hybrid · RoPE-1D, raw patches",
          "r-h-rope2d": "Hybrid · RoPE-2D, raw patches", "r-h-rope1d-conv2d": "Hybrid · RoPE-1D + 2D conv, raw patches",
          "r-d-rope1d": "Dense · RoPE-1D, raw patches", "r-d-nope": "Dense · NoPE, raw patches"}


def load_run(directory):
    directory = Path(directory)
    events = [json.loads(line) for line in (directory / "metrics.jsonl").read_text().splitlines()]
    evals = [e for e in events if e["event"] == "eval"]
    trains = [e for e in events if e["event"] == "train"]
    run = dict(name=directory.name, events=events, final=evals[-1] if evals else None, probe={}, hard={}, coco=None,
               whatsup=None, seconds=sorted(t["seconds"] for t in trains)[len(trains) // 2] if trains else None)
    for path in directory.glob("spatial_checkpoint_*.json"):
        data = json.loads(path.read_text())
        run["probe"][data["step"]] = data
    for path in directory.glob("hard_checkpoint_*.json"):
        data = json.loads(path.read_text())
        run["hard"][data["step"]] = data
    for path in directory.glob("coco_*.json"):
        run["coco"] = json.loads(path.read_text())
    for path in directory.glob("whatsup_*.json"):
        run["whatsup"] = json.loads(path.read_text())
    run["complete"] = bool(run["final"] and run["final"]["update"] == 5000 and 5000 in run["probe"])
    return run


def final(run, key, source="probe"):
    """`key` at the final checkpoint; source "probe" (pre-registered) or "hard" (secondary)."""
    if not run or not run["complete"]:
        return None
    return run[source].get(5000, {}).get(key)


def decide(runs):
    spreads = []
    for a, b in SEED_PAIRS:
        if a in runs and b in runs and runs[a]["complete"] and runs[b]["complete"]:
            spreads.append(abs(final(runs[a], "vertical_acc2") - final(runs[b], "vertical_acc2")))
    spread = max(spreads) if len(spreads) == len(SEED_PAIRS) else None
    out = []
    for mode, treated, base in DECISIONS:
        t, b = runs.get(treated), runs.get(base)
        row = dict(mode=mode, treated=treated, baseline=base, verdict="pending")
        if t and b and t["complete"] and b["complete"]:
            gain = final(t, "vertical_acc2") - final(b, "vertical_acc2")
            caption = t["final"]["caption_bpb"] - b["final"]["caption_bpb"]
            text = t["final"]["text_bpb"] - b["final"]["text_bpb"]
            row.update(gain=gain, caption_bpb_delta=caption, text_bpb_delta=text)
            if spread is not None:
                if gain >= GAIN and gain > NOISE * spread and caption <= BPB_SLACK and text <= BPB_SLACK:
                    row["verdict"] = "effective"
                elif gain <= NOISE * spread:
                    row["verdict"] = "negative"
                else:
                    row["verdict"] = "inconclusive"
        out.append(row)
    return dict(seed_spread=spread, seed_spreads=spreads, decisions=out,
                required_complete=all(r in runs and runs[r]["complete"] for r in REQUIRED))


def secondary(runs):
    """Descriptive only: differences between runs on the harder probe and the seed spread on that probe."""
    spread = {}
    for key, _ in HARD_KEYS:
        diffs = [abs(final(runs[a], key, "hard") - final(runs[b], key, "hard")) for a, b in SEED_PAIRS
                 if final(runs.get(a), key, "hard") is not None and final(runs.get(b), key, "hard") is not None]
        spread[key] = max(diffs) if len(diffs) == len(SEED_PAIRS) else None
    rows = []
    for label, treated, base in CONTRASTS:
        diff = {}
        for key, _ in HARD_KEYS:
            a, b = final(runs.get(treated), key, "hard"), final(runs.get(base), key, "hard")
            diff[key] = None if a is None or b is None else a - b
        rows.append(dict(label=label, treated=treated, baseline=base, diff=diff))
    return dict(hard_seed_spread=spread, hard_contrasts=rows)


def contrast_table(sec):
    """Hard-probe differences in percentage points; bold where larger than twice the seed spread."""
    spread = sec["hard_seed_spread"]
    rows = []
    for row in sec["hard_contrasts"]:
        if all(v is None for v in row["diff"].values()):
            continue
        cells = []
        for key, _ in HARD_KEYS:
            v, s = row["diff"][key], spread[key]
            text = "–" if v is None else f"{100 * v:+.1f}"
            beyond = v is not None and s is not None and abs(v) > NOISE * s
            cells.append(f"<td><b>{text}</b></td>" if beyond else f"<td>{text}</td>")
        rows.append(f"<tr><td>{escape(row['label'])}</td>" + "".join(cells) + "</tr>")
    if not rows:
        return "<p>等待结果</p>"
    noise = "".join(f"<td>{'–' if spread[k] is None else f'{100 * NOISE * spread[k]:.1f}'}</td>" for k, _ in HARD_KEYS)
    head = "".join(f"<th>{escape(name)}</th>" for _, name in HARD_KEYS)
    return (f'<div class="wide"><table><tr><th>对比（前者减后者，百分点）</th>{head}</tr>{"".join(rows)}'
            f'<tr><td>2× 种子间波动</td>{noise}</tr></table></div>')


def pct(value):
    return "–" if value is None else f"{100 * value:.1f}"


def num(value, digits=4):
    return "–" if value is None else f"{value:.{digits}f}"


def dumbbell(runs, names, keys=("horizontal_acc2", "vertical_acc2"), source="probe", low=.4, note="chance"):
    """One row per run: a horizontal and a vertical value at the final checkpoint, on one low–100% axis."""
    rows = [n for n in names if all(final(runs[n], k, source) is not None for k in keys)]
    left, width, top, step = 270, 480, 28, 30
    height = top + step * len(rows) + 34
    x = lambda v: left + (v - low) / (1 - low) * width
    parts = [f'<svg viewBox="0 0 {left + width + 30} {height}" role="img" aria-label="Horizontal and vertical '
             f'relation accuracy per run">']
    ticks = [t / 10 for t in range(int(round(low * 10)), 11, 1 if low >= .3 else 2)]
    for tick in ticks:
        parts.append(f'<line class="grid" x1="{x(tick):.1f}" x2="{x(tick):.1f}" y1="{top - 10}" y2="{height - 30}"/>'
                     f'<text class="tick" x="{x(tick):.1f}" y="{height - 12}" text-anchor="middle">{int(tick * 100)}%</text>')
    if note:
        parts.append(f'<text class="tick" x="{x(.5) + 4:.1f}" y="{top - 14}">{escape(note)}</text>')
    for i, name in enumerate(rows):
        y = top + step * i + 8
        h, v = (final(runs[name], k, source) for k in keys)
        parts.append(f'<text class="row" x="{left - 12}" y="{y + 4}" text-anchor="end">{escape(LABELS.get(name, name))}</text>'
                     f'<line class="link" x1="{x(min(h, v)):.1f}" x2="{x(max(h, v)):.1f}" y1="{y}" y2="{y}"/>')
        for axis, value, cls in (("水平", h, "s1"), ("垂直", v, "s2")):
            tip = escape(f"{LABELS.get(name, name)} — {axis}: {100 * value:.1f}%")
            parts.append(f'<circle class="dot {cls}" cx="{x(value):.1f}" cy="{y}" r="5" data-tip="{tip}"/>'
                         f'<circle class="hit" cx="{x(value):.1f}" cy="{y}" r="11" data-tip="{tip}"/>')
    parts.append("</svg>")
    return "".join(parts)


def curves(runs, names, key, title, source="probe"):
    """Probe accuracy over checkpoints for up to four runs, one categorical slot each, labelled at the end."""
    names = [n for n in names if n in runs and runs[n][source]][:4]
    if not names:
        return f'<figure class="small"><figcaption>{escape(title)}</figcaption><p>等待结果</p></figure>'
    values = [d[key] for n in names for d in runs[n][source].values()]
    low = min(.4, (int(min(values, default=.4) * 10)) / 10)
    left, width, top, height = 44, 300, 16, 170
    x = lambda s: left + (s - 1000) / 4000 * width
    y = lambda v: top + (1 - (v - low) / (1 - low)) * (height - top - 26)
    swatches = "".join(f'<span><i style="background:var(--series-{slot})"></i>{escape(LABELS.get(n, n))}</span>'
                       for slot, n in enumerate(names, 1))
    parts = [f'<figure class="small"><figcaption>{escape(title)}</figcaption><div class="legend">{swatches}</div>'
             f'<svg viewBox="0 0 {left + width + 150} {height}" role="img" aria-label="{escape(title)}">']
    for tick in [t / 10 for t in range(int(low * 10), 11, 2)]:
        parts.append(f'<line class="grid" x1="{left}" x2="{left + width}" y1="{y(tick):.1f}" y2="{y(tick):.1f}"/>'
                     f'<text class="tick" x="{left - 6}" y="{y(tick) + 4:.1f}" text-anchor="end">{int(tick * 100)}%</text>')
    for step in (1000, 3000, 5000):
        parts.append(f'<text class="tick" x="{x(step):.1f}" y="{height - 6}" text-anchor="middle">{step // 1000}k</text>')
    ends = []
    for slot, name in enumerate(names, 1):
        points = sorted((s, d[key]) for s, d in runs[name][source].items())
        path = " ".join(f"{'M' if i == 0 else 'L'}{x(s):.1f},{y(v):.1f}" for i, (s, v) in enumerate(points))
        parts.append(f'<path class="line c{slot}" d="{path}"/>')
        for s, v in points:
            tip = escape(f"{LABELS.get(name, name)} — {s} updates: {100 * v:.1f}%")
            parts.append(f'<circle class="dot c{slot}" cx="{x(s):.1f}" cy="{y(v):.1f}" r="4" data-tip="{tip}"/>'
                         f'<circle class="hit" cx="{x(s):.1f}" cy="{y(v):.1f}" r="10" data-tip="{tip}"/>')
        ends.append([y(points[-1][1]) + 4, x(points[-1][0]) + 8, LABELS.get(name, name).split(" · ")[-1]])
    ends.sort()
    for i in range(1, len(ends)):  # keep end labels at least one line apart
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 13)
    for ty, tx, label in ends:
        parts.append(f'<text class="end" x="{tx:.1f}" y="{ty:.1f}">{escape(label)}</text>')
    parts.append("</svg></figure>")
    return "".join(parts)


STYLE = """
.viz-root{color-scheme:light;--surface-1:#fcfcfb;--surface-2:#f2f1ee;--text-primary:#0b0b0b;--text-secondary:#52514e;
--text-muted:#7a7975;--grid:#e6e5e1;--series-1:#2a78d6;--series-2:#eb6834;--series-3:#1baf7a;--series-4:#eda100;
--good:#1f7a3a;--bad:#b4402f;font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
background:var(--surface-1);color:var(--text-primary);max-width:980px;margin:0 auto;padding:28px}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])) .viz-root{color-scheme:dark;
--surface-1:#1a1a19;--surface-2:#252523;--text-primary:#f0efec;--text-secondary:#c3c2b7;--text-muted:#8f8e88;
--grid:#33332f;--series-1:#3987e5;--series-2:#d95926;--series-3:#199e70;--series-4:#c98500;--good:#5cb878;--bad:#e2725f}}
body{margin:0;background:var(--surface-1)} h1{font-size:22px;margin:0 0 4px} h2{font-size:17px;margin:28px 0 8px}
p,li{color:var(--text-secondary)} .tiles{display:flex;gap:12px;flex-wrap:wrap;margin:16px 0}
.tile{background:var(--surface-2);border-radius:10px;padding:12px 16px;min-width:220px}
.tile .k{font-size:13px;color:var(--text-muted)} .tile .v{font-size:22px;font-weight:600}
.tile .s{font-size:13px;color:var(--text-secondary)} svg{width:100%;height:auto;display:block}
.grid{stroke:var(--grid);stroke-width:1} .tick,.row,.end{fill:var(--text-secondary);font-size:12px}
.link{stroke:var(--text-muted);stroke-width:2;stroke-linecap:round}
.dot{stroke:var(--surface-1);stroke-width:2} .s1,.c1{fill:var(--series-1)} .s2,.c2{fill:var(--series-2)}
.c3{fill:var(--series-3)} .c4{fill:var(--series-4)} .hit{fill:transparent;cursor:default}
.line{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round} path.c1{stroke:var(--series-1)}
path.c2{stroke:var(--series-2)} path.c3{stroke:var(--series-3)} path.c4{stroke:var(--series-4)}
.legend{display:flex;flex-wrap:wrap;gap:6px 16px;font-size:13px;color:var(--text-secondary);margin:4px 0 8px}
.legend i{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px;vertical-align:-1px}
.smalls{display:flex;gap:16px;flex-wrap:wrap} figure.small{margin:0;flex:1 1 440px}
figcaption{font-size:13px;color:var(--text-secondary);margin-bottom:4px} .wide{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:13px} th,td{padding:5px 8px;text-align:right;
border-bottom:1px solid var(--grid)} th:first-child,td:first-child{text-align:left} th{color:var(--text-muted);font-weight:500}
#tip{position:fixed;pointer-events:none;background:var(--text-primary);color:var(--surface-1);font-size:12px;
padding:4px 8px;border-radius:6px;opacity:0;transition:opacity .1s}
"""

SCRIPT = """
const tip=document.getElementById('tip');
document.querySelectorAll('[data-tip]').forEach(el=>{
 el.addEventListener('mousemove',e=>{tip.textContent=el.dataset.tip;tip.style.left=(e.clientX+12)+'px';
 tip.style.top=(e.clientY+12)+'px';tip.style.opacity=1});
 el.addEventListener('mouseleave',()=>{tip.style.opacity=0});});
"""


def render(runs, decision, position=None, sec=None):
    order = [n for n in LABELS if n in runs] + sorted(n for n in runs if n not in LABELS)
    tiles = []
    for d in decision["decisions"]:
        verdict = {"effective": "有效", "negative": "无效（未超出噪声）", "inconclusive": "不确定",
                   "pending": "等待结果"}[d["verdict"]]
        detail = (f"垂直 acc2 {100 * d['gain']:+.1f} 个百分点；caption BPB {d['caption_bpb_delta']:+.4f}，"
                  f"文本 BPB {d['text_bpb_delta']:+.4f}") if "gain" in d else "两个 run 都完成后判定"
        base = final(runs.get(d["baseline"]), "vertical_acc2")
        if base is not None and base > 1 - GAIN:
            detail += f"。基线已达 {100 * base:.1f}%，这个探针上不可能再提高 5 个百分点"
        tiles.append(f'<div class="tile"><div class="k">二维卷积 · {escape(d["mode"])}</div>'
                     f'<div class="v">{verdict}</div><div class="s">{escape(detail)}</div></div>')
    spread = decision["seed_spread"]
    tiles.append(f'<div class="tile"><div class="k">种子间波动（垂直 acc2）</div><div class="v">'
                 f'{"–" if spread is None else f"{100 * spread:.1f} 个百分点"}</div>'
                 f'<div class="s">判定门槛：≥5 个百分点且 &gt; 2× 波动</div></div>')
    position_tiles = ""
    if position:
        cells = []
        for kind, label in (("siglip", "SigLIP2 特征"), ("raw32", "原始像素 patch")):
            if kind in position:
                q = position[kind]
                cells.append(f'<div class="tile"><div class="k">{label}：线性读出 token 的行 / 列</div>'
                             f'<div class="v">R² {q["row_r2"]:.2f} / {q["col_r2"]:.2f}</div>'
                             f'<div class="s">取整后命中 {100 * q["row_acc"]:.0f}% / {100 * q["col_acc"]:.0f}%'
                             f'（随机 {100 * q["chance_acc"]:.0f}%），COCO {q["test_images"]} 张测试图</div></div>')
        position_tiles = f'<div class="tiles">{"".join(cells)}</div>'
    rows = []
    for n in order:
        r = runs[n]
        f, w, c = r["final"] or {}, r["whatsup"] or {}, r["coco"] or {}
        rows.append("<tr>" + "".join(f"<td>{v}</td>" for v in (
            escape(LABELS.get(n, n)), f.get("update", "–"), num(f.get("caption_bpb")), num(f.get("text_bpb")),
            num((f.get("caption_bpb_shuffled") or 0) - (f.get("caption_bpb") or 0)) if f else "–",
            pct(final(r, "horizontal_acc2")), pct(final(r, "vertical_acc2")),
            pct(final(r, "horizontal_acc2", "hard")), pct(final(r, "vertical_acc2", "hard")),
            pct(final(r, "hflip_horizontal_acc2", "hard")), pct(final(r, "vflip_vertical_acc2", "hard")),
            pct(w.get("horizontal_acc2")), pct(w.get("vertical_acc2")), pct(c.get("retrieval_pmi")),
            num(r["seconds"], 2))) + "</tr>")
    table = ('<div class="wide"><table><tr><th>Run</th><th>步数</th><th>caption BPB</th><th>文本 BPB</th>'
             "<th>看图增益</th><th>合成·水平</th><th>合成·垂直</th><th>难版·水平</th><th>难版·垂直</th>"
             "<th>难版·水平翻转</th><th>难版·垂直翻转</th><th>What'sUp·水平</th><th>What'sUp·垂直</th>"
             "<th>COCO 检索</th><th>秒/步</th></tr>" + "".join(rows) + "</table></div>")
    key = ["h-rope1d", "h-nope", "h-rope2d", "h-nope-conv2d"]
    raw = ["r-h-nope", "r-h-nope-conv2d", "r-h-rope1d", "r-h-rope2d"]
    legend = ('<div class="legend"><span><i style="background:var(--series-1)"></i>水平关系（左/右）</span>'
              '<span><i style="background:var(--series-2)"></i>垂直关系（上/下）</span></div>')
    flip_keys = ("hflip_horizontal_acc2", "vflip_vertical_acc2")
    return f"""<!doctype html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>混合骨干 VLM 的二维位置</title>
<style>{STYLE}</style></head><body><main class="viz-root">
<h1>混合骨干 VLM 里，图像的二维位置从哪来？</h1>
<p>所有 run 算力对齐（训练 FLOPs/token 在 Dense 锚点 ±1% 内），5,000 步、约 52 亿 token、8K 上下文、每行 12 张图。
判定规则在出结果前写死（docs/spatial-preregistration.md）。</p>
<h2>预注册判定（主指标：合成探针垂直关系 acc2）</h2>
<div class="tiles">{''.join(tiles)}</div>
<h2>水平 vs 垂直关系（合成探针，二选一准确率，最终 checkpoint）</h2>{legend}{dumbbell(runs, order)}
<h2>以下为次要分析（不参与判定，见 docs/spatial-addendum.md）</h2>
<h2>位置信息在视觉 token 本身里有多少</h2>{position_tiles}
<h2>更难的探针（8×8 网格、更小的物体、3–6 个干扰物）</h2>{legend}{dumbbell(runs, order, source="hard")}
<h2>打乱 token 顺序：模型靠顺序还是靠内容读位置</h2>
<p>只改变图像 token 送进骨干的顺序，内容不变。水平翻转（每行内倒序）后看水平关系，垂直翻转（行倒序）后看垂直关系。
0% = 判断被完全反转，全靠 token 顺序；100% = 不受影响，全靠 token 内容（如编码器的位置编码）；50% = 两者各半或读不出。
难版探针，最终 checkpoint。</p>
{legend}{dumbbell(runs, order, keys=flip_keys, source="hard", low=0., note="")}
<h2>难版探针上的差异，和种子间波动比较</h2>
<p>每格是两个 run 在难版探针最终 checkpoint 上的差（百分点）。种子间波动取两对换种子 run 在同一指标上差值的较大者；
超过 2 倍波动的格子加粗。只作描述，不参与判定。</p>
{contrast_table(sec or secondary(runs))}
<h2>学习曲线</h2><div class="smalls">{curves(runs, key, "vertical_acc2", "垂直关系 acc2（合成探针）")}
{curves(runs, key, "horizontal_acc2", "水平关系 acc2（合成探针）")}
{curves(runs, key, "vertical_acc2", "垂直关系 acc2（难版探针）", "hard")}
{curves(runs, raw, "vertical_acc2", "垂直关系 acc2（难版探针，原始像素输入）", "hard")}
{curves(runs, key, "vflip_vertical_acc2", "行倒序后的垂直 acc2（难版探针；越低越依赖 token 顺序）", "hard")}
{curves(runs, raw, "vflip_vertical_acc2", "行倒序后的垂直 acc2（难版探针，原始像素输入）", "hard")}</div>
<h2>全部指标</h2>{table}
<p>BPB 越低越好；"看图增益"= 把图像换成别的图像后 caption BPB 的上升；What'sUp 为真实照片（on/under 算垂直，
left/right 算水平）；COCO 检索为 5 选 1（随机 20%）。</p>
</main><div id="tip" role="tooltip"></div><script>{SCRIPT}</script></body></html>"""


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    runs = {d.name: load_run(d) for d in sorted(args.results.iterdir()) if (d / "metrics.jsonl").exists()}
    decision = decide(runs)
    position = {p.stem: json.loads(p.read_text()) for p in (args.results / "position-probe").glob("*.json")}
    sec = secondary(runs)
    args.out.write_text(render(runs, decision, position, sec))
    if args.json:
        args.json.write_text(json.dumps(dict(decision, secondary=sec), indent=2) + "\n")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
