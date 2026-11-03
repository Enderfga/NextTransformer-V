"""What'sUp controlled images (Kamath et al., 2023) in the spatial-probe format of ntv.synth.

Each file `controlled_images/<object>_<relation>_<reference>.jpeg` shows the object on, under, left of or right of
the reference; the four candidate captions differ only in the relation. `on` / `under` are vertical relations and
`left of` / `right of` horizontal ones.

  python -m ntv.whatsup --root <downloaded dataset dir>     (writes <root>/eval.jsonl)
"""
import argparse
import json
from pathlib import Path
import re

PHRASES = {"left_of": "to the left of", "right_of": "to the right of", "on": "on", "under": "under"}
AXIS = {"to the left of": "horizontal", "to the right of": "horizontal", "on": "vertical", "under": "vertical"}
PATTERN = re.compile(r"^(?P<object>.+?)_(?P<relation>left_of|right_of|on|under)_(?P<reference>.+)\.jpe?g$")


def article(word):
    return "an" if word[0] in "aeiou" else "a"


def caption(obj, phrase, ref):
    return f"{article(obj)} {obj} {phrase} {article(ref)} {ref}."


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for path in sorted((args.root / "controlled_images").glob("*.jp*g")):
        match = PATTERN.match(path.name)
        if not match:
            continue
        obj, ref = (match[k].replace("-", " ").replace("_", " ") for k in ("object", "reference"))
        phrase = PHRASES[match["relation"]]
        rows.append(dict(image=str(path.relative_to(args.root)), relation=phrase, axis=AXIS[phrase], distance=0,
                         candidates={p: caption(obj, p, ref) for p in PHRASES.values()}))
    with open(args.root / "eval.jsonl", "w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    print(f"{len(rows)} images")


if __name__ == "__main__":
    main()
