"""Source of the README navigation buttons: one HTML file per button, rendered at 274 x 90 CSS px, device scale 2."""
from pathlib import Path

HERE = Path(__file__).parent
ICONS = {
    "results": '<path d="M5 21V12M11 21V5M17 21v-7M23 21V9" />',
    "rfc": '<path d="M7 4h10l5 5v15H7z" /><path d="M17 4v5h5M11 14h7M11 18h7" />',
    "reproduce": '<path d="M5 14a9 9 0 0 1 15.5-6.2M23 14a9 9 0 0 1-15.5 6.2" /><path d="M20.5 3v5h-5M7.5 25v-5h5" />',
}
BUTTONS = (("results", "See the results", "#2f56c8", "#e9efff"), ("rfc", "Read the RFC", "#6b49d8", "#f1ecff"),
           ("reproduce", "Reproduce", "#c24f22", "#fff0e8"))
TEMPLATE = """<!doctype html><html><head><meta charset="utf-8"><style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@700&display=swap');
html,body{{margin:0;background:transparent}}
.b{{box-sizing:border-box;width:274px;height:90px;padding:12px}}
.p{{height:100%;border-radius:33px;background:linear-gradient(180deg,#fff,{bg});border:1.5px solid {line};
  box-shadow:0 6px 14px {glow};display:flex;align-items:center;justify-content:center;gap:12px;
  font:700 20px Inter,sans-serif;color:{fg}}}
svg{{width:26px;height:26px;fill:none;stroke:{fg};stroke-width:2.4;stroke-linecap:round;stroke-linejoin:round}}
</style></head><body><div class="b"><div class="p"><svg viewBox="0 0 28 28">{icon}</svg>{label}</div></div></body></html>"""

if __name__ == "__main__":
    for key, label, fg, bg in BUTTONS:
        html = TEMPLATE.format(bg=bg, line=fg + "40", glow=fg + "26", fg=fg, icon=ICONS[key], label=label)
        (HERE / f"button-{key}.html").write_text(html)
        print(key)
