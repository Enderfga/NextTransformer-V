"""Source of the README highlight cards. Writes one standalone HTML file per card next to this script; render each
at 1240 x 420 CSS px with device scale 2 (headless Chromium) and save as WebP in docs/assets/. Numbers on the cards
come from results/tables.md."""
from pathlib import Path

HERE = Path(__file__).parent
BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#1d2433", "#5b6272"

STYLE = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html,body{margin:0;background:transparent}
.card{width:1240px;height:420px;box-sizing:border-box;padding:14px;border-radius:34px;background:var(--frame)}
.inner{width:100%;height:100%;box-sizing:border-box;border-radius:24px;display:flex;align-items:center;gap:34px;
  padding:0 40px 0 46px;background:linear-gradient(115deg,#ffffff 0%,#fbfbff 50%,#f1f3fe 100%);font-family:Inter,sans-serif}
.text{flex:0 0 500px}
.eyebrow{display:flex;align-items:center;gap:16px;font-weight:700;font-size:15px;letter-spacing:.2em;color:var(--accent)}
.icon{width:58px;height:58px;border-radius:17px;background:var(--icon);box-shadow:0 10px 20px var(--glow);display:grid;place-items:center}
h2{font-weight:800;font-size:40px;line-height:1.1;color:#1d2433;margin:20px 0 14px;letter-spacing:-.015em}
p{font-size:19px;line-height:1.5;color:#5b6272;margin:0 0 20px}
.link{font-weight:700;font-size:18px;color:var(--accent)}
.visual{flex:1;min-width:0;height:330px;overflow:hidden;border-radius:24px;background:#f8f9fe;border:1.5px solid #e2e6f4;
  box-shadow:inset 0 1px 0 #fff;display:flex;align-items:center;justify-content:center}
svg text{font-family:Inter,sans-serif}
.visual svg{width:94%;height:auto}
"""


def page(accent, frame, icon, glow, icon_svg, eyebrow, title, body, link, visual):
    return f"""<!doctype html><html><head><meta charset="utf-8"><style>{STYLE}
.card{{--accent:{accent};--frame:{frame};--icon:{icon};--glow:{glow}}}</style></head><body>
<div class="card"><div class="inner"><div class="text">
<div class="eyebrow"><div class="icon">{icon_svg}</div>{eyebrow}</div>
<h2>{title}</h2><p>{body}</p><div class="link">{link} &rarr;</div></div>
<div class="visual">{visual}</div></div></div></body></html>"""


def grid(x, y, n, size, fill, stroke="#ffffff", gap=1.2, rx=1.5):
    cell = size / n
    return "".join(f'<rect x="{x + j * cell + gap / 2:.1f}" y="{y + i * cell + gap / 2:.1f}" width="{cell - gap:.1f}" '
                   f'height="{cell - gap:.1f}" rx="{rx}" fill="{fill(i, j)}" stroke="none"/>'
                   for i in range(n) for j in range(n))


def mix(a, b, t):
    a, b = [int(a[k:k + 2], 16) for k in (1, 3, 5)], [int(b[k:k + 2], 16) for k in (1, 3, 5)]
    return "#" + "".join(f"{round(p + (q - p) * t):02x}" for p, q in zip(a, b))


# 01 · TRACK -------------------------------------------------------------------------------------------------------
def track_visual():
    w, h = 640, 300
    parts = [f'<svg viewBox="0 0 {w} {h}">']
    parts.append(f'<text x="24" y="34" font-size="16.1" font-weight="700" fill="#7a8194" letter-spacing="1.5">ONE ROW · 8,192 TOKENS</text>')
    x, y = 24, 52
    seq = ["t", "t", "t", "img", "c", "c", "t", "t", "img", "c", "c", "t", "t", "t", "img", "c"]
    for kind in seq:
        if kind == "img":
            parts.append(grid(x, y, 6, 46, lambda i, j: mix("#a9c4f5", "#f4b39a", (i + j) / 10)))
            x += 52
        else:
            col = "#cfd5e6" if kind == "t" else "#e8c7b8"
            parts.append(f'<rect x="{x}" y="{y + 15}" width="18" height="16" rx="5" fill="{col}"/>')
            x += 24
    legend = (("text", "#cfd5e6", 4), ("image: 144 frozen SigLIP2 tokens", "#bccff2", 3), ("caption", "#e8c7b8", 4))
    lx = 24
    for label, col, rx in legend:
        parts.append(f'<rect x="{lx}" y="111" width="14" height="12" rx="{rx}" fill="{col}"/>'
                     f'<text x="{lx + 20}" y="122" font-size="14.9" fill="#7a8194">{label}</text>')
        lx += 20 + 7.1 * len(label) + 22
    parts.append(f'<text x="{lx}" y="122" font-size="14.9" fill="#7a8194">12 images per row</text>')
    parts.append(f'<text x="24" y="168" font-size="16.1" font-weight="700" fill="#7a8194" letter-spacing="1.5">SAME TRAINING FLOPs PER TOKEN</text>')
    rows = (("Dense", "23 layers", 0.0, "#5b7cf0"), ("GDN hybrid", "28 layers", -0.254, "#8b6cf0"),
            ("GDN hybrid + 2D conv", "28 layers", -0.245, "#e5607e"))
    for k, (name, layers, err, col) in enumerate(rows):
        yy = 186 + k * 34
        parts.append(f'<text x="24" y="{yy + 17}" font-size="17.2" font-weight="600" fill="#1d2433">{name}</text>'
                     f'<text x="246" y="{yy + 17}" font-size="16.1" fill="#7a8194">{layers}</text>'
                     f'<rect x="334" y="{yy + 4}" width="220" height="18" rx="9" fill="#e9ecf6"/>'
                     f'<rect x="334" y="{yy + 4}" width="{220 * (1 + err / 100):.1f}" height="18" rx="9" fill="{col}"/>'
                     f'<text x="626" y="{yy + 18}" font-size="14.9" font-weight="600" fill="#5b6272" text-anchor="end">'
                     f'{"anchor" if err == 0 else f"{err:+.2f}%"}</text>')
    parts.append('</svg>')
    return "".join(parts)


# 02 · TEST --------------------------------------------------------------------------------------------------------
def test_visual():
    w, h = 640, 300
    parts = [f'<svg viewBox="0 0 {w} {h}">']
    size, n, top = 116, 4, 22
    base = lambda i, j: mix(mix("#dbe6fb", "#c2d5f7", j / 3), "#f6d3c4", i / 3)
    panels = (("original", lambda i, j: (i, j)), ("rows reversed", lambda i, j: (n - 1 - i, j)),
              ("columns reversed", lambda i, j: (i, n - 1 - j)))
    for k, (label, where) in enumerate(panels):
        x = 34 + k * 200
        parts.append(f'<rect x="{x - 8}" y="{top - 8}" width="{size + 16}" height="{size + 16}" rx="12" fill="#ffffff" '
                     f'stroke="#e2e6f4"/>')
        # content travels with the token: draw each source cell (i, j) at its new place
        cell = size / n
        for i in range(n):
            for j in range(n):
                r, c = where(i, j)
                parts.append(f'<rect x="{x + c * cell + 1:.1f}" y="{top + r * cell + 1:.1f}" width="{cell - 2:.1f}" '
                             f'height="{cell - 2:.1f}" rx="3" fill="{base(i, j)}"/>')
        for (i, j, shape) in ((0, 1, "circle"), (2, 1, "square")):
            r, c = where(i, j)
            cx, cy = x + c * cell + cell / 2, top + r * cell + cell / 2
            parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="10" fill="{ORANGE}"/>' if shape == "circle" else
                         f'<rect x="{cx - 9:.1f}" y="{cy - 9:.1f}" width="18" height="18" rx="3" fill="{BLUE}"/>')
        parts.append(f'<text x="{x + size / 2}" y="{top + size + 30}" font-size="16.1" font-weight="600" fill="#1d2433" '
                     f'text-anchor="middle">{label}</text>')
        if k:
            parts.append(f'<path d="M{x - 52} {top + size / 2} h28" stroke="#9aa3b8" stroke-width="2.2" '
                         f'stroke-linecap="round" marker-end="url(#a)"/>')
    parts.insert(1, '<defs><marker id="a" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="5" markerHeight="5" '
                    'orient="auto"><path d="M0 0L10 5L0 10z" fill="#9aa3b8"/></marker></defs>')
    # readout scale with three real runs (vertical acc2 after the row flip, harder probe)
    y0, x0, x1 = 230, 34, 606
    parts.append(f'<defs><linearGradient id="g" x1="0" x2="1"><stop offset="0" stop-color="#f3c9b8"/>'
                 f'<stop offset="1" stop-color="#c9d9f6"/></linearGradient></defs>'
                 f'<rect x="{x0}" y="{y0}" width="{x1 - x0}" height="10" rx="5" fill="url(#g)"/>'
                 f'<text x="{x0}" y="{y0 + 34}" font-size="14.9" fill="#5b6272">0% = token order</text><text x="{(x0 + x1) / 2}" y="{y0 + 58}" font-size="13.8" fill="#9aa3b8" text-anchor="middle">vertical acc2 with the rows reversed, harder probe</text>'
                 f'<text x="{x1}" y="{y0 + 34}" font-size="14.9" fill="#5b6272" text-anchor="end">token content = 100%</text>')
    for value, label, below in ((.286, "Hybrid NoPE + 2D conv 29%", False), (.583, "Hybrid RoPE-1D 58%", True),
                                (.877, "Hybrid NoPE 88%", False)):
        cx = x0 + value * (x1 - x0)
        parts.append(f'<circle cx="{cx:.1f}" cy="{y0 + 5}" r="7" fill="{ORANGE}" stroke="#fff" stroke-width="2.5"/>'
                     f'<text x="{cx:.1f}" y="{y0 + 32 if below else y0 - 12}" font-size="14.4" font-weight="600" '
                     f'fill="#1d2433" text-anchor="middle">{label}</text>')
    parts.append('</svg>')
    return "".join(parts)


# 03 · FINDING -----------------------------------------------------------------------------------------------------
def finding_visual():
    w, h = 640, 300
    parts = [f'<svg viewBox="0 0 {w} {h}">']
    groups = (("VERTICAL ACCURACY, TOKEN ROWS REVERSED", .877, .286, "−59 points · beyond 2× seed spread (15.4)", True),
              ("VERTICAL ACCURACY", .940, .908, "−3.2 points · within 2× seed spread (5.7)", False))
    x0, width = 262, 300
    for g, (title, a, b, note, strong) in enumerate(groups):
        y = 34 + g * 138
        parts.append(f'<text x="24" y="{y}" font-size="15.5" font-weight="700" fill="#7a8194" letter-spacing="1.4">{title}</text>')
        for k, (label, v, col) in enumerate((("Hybrid · NoPE", a, "#f2a88a"), ("Hybrid · NoPE + 2D conv", b, ORANGE))):
            yy = y + 16 + k * 34
            parts.append(f'<text x="24" y="{yy + 17}" font-size="17.2" font-weight="600" fill="#1d2433">{label}</text>'
                         f'<rect x="{x0}" y="{yy + 3}" width="{width}" height="20" rx="10" fill="#e9ecf6"/>'
                         f'<rect x="{x0}" y="{yy + 3}" width="{width * v:.1f}" height="20" rx="10" fill="{col}"/>'
                         f'<text x="{x0 + width * v + 10:.1f}" y="{yy + 18}" font-size="16.1" font-weight="700" '
                         f'fill="#1d2433">{100 * v:.1f}%</text>')
        parts.append(f'<text x="{x0}" y="{y + 102}" font-size="15.5" font-weight="{700 if strong else 500}" '
                     f'fill="{"#c24f22" if strong else "#5b6272"}">{note}</text>')
    parts.append('</svg>')
    return "".join(parts)


ICONS = {
    "track": '<svg width="30" height="30" viewBox="0 0 30 30" fill="none" stroke="#fff" stroke-width="2.4" '
             'stroke-linecap="round"><rect x="3" y="6" width="11" height="11" rx="2"/><path d="M18 9h9M18 14h9M3 23h24"/></svg>',
    "test": '<svg width="30" height="30" viewBox="0 0 30 30" fill="none" stroke="#fff" stroke-width="2.4" '
            'stroke-linecap="round" stroke-linejoin="round"><path d="M8 4v22M8 26l-4-4M8 26l4-4M22 26V4M22 4l-4 4M22 4l4 4"/></svg>',
    "finding": '<svg width="30" height="30" viewBox="0 0 30 30" fill="none" stroke="#fff" stroke-width="2.4" '
               'stroke-linecap="round"><path d="M5 25V14M12 25V6M19 25v-8M26 25V10"/></svg>',
}

CARDS = {
    "card-track": page("#3557c9", "linear-gradient(120deg,#5b7cf0,#7f8ff4 55%,#a7b6fa)",
                       "linear-gradient(135deg,#6a8ff7,#3b5fe0)", "rgba(59,95,224,.35)", ICONS["track"],
                       "01 · TRACK", "Image tokens,<br>same contract",
                       "Frozen SigLIP2 tokens interleaved with text at an 8K context. Every backbone trains on the "
                       "same rows in the same order, at the same training FLOPs per token.",
                       "Read the RFC", track_visual()),
    "card-test": page("#6b49d8", "linear-gradient(120deg,#8b6cf0,#a58bf4 55%,#c9b6fa)",
                      "linear-gradient(135deg,#9d7ff6,#6b49d8)", "rgba(107,73,216,.35)", ICONS["test"],
                      "02 · THE TEST", "Reorder the tokens,<br>keep the content",
                      "Reverse the image's token rows before the backbone reads them. If its answers flip, it read "
                      "position from token order; if they hold, from the tokens' own features.",
                      "How the probes work", test_visual()),
    "card-finding": page("#c24f22", "linear-gradient(120deg,#f08a5d,#f3a07e 55%,#f7c2a8)",
                         "linear-gradient(135deg,#f49a6c,#e0622c)", "rgba(224,98,44,.35)", ICONS["finding"],
                         "03 · FINDING", "2D mixing changes how,<br>not how well",
                         "A 2D short convolution makes the hybrid read vertical position from the token grid instead "
                         "of the encoder's features. Its accuracy stays within seed noise.",
                         "See the results", finding_visual()),
}

if __name__ == "__main__":
    for name, html in CARDS.items():
        (HERE / f"{name}.html").write_text(html)
        print(name)
