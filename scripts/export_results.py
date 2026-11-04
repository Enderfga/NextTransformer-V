"""Copy the summaries of finished runs into the repository's `results/` directory, with site paths removed.

<summaries dir> holds one directory per run: either the run directory written by scripts/pilot.sh (the files are read
from its `summary/`) or the summary itself. Each run's summary (metrics.jsonl, latest.json and the evaluation JSON
files) is copied unchanged except for path values: an absolute path becomes the part after the run root
(`runs/<run>/checkpoint_00005000`, `packed/pilot-train`). The export fails if any absolute path is left.

  python scripts/export_results.py <summaries dir> results
"""
import json
import re
import shutil
import sys
from pathlib import Path

KEEP = ("runs/", "packed/")
ABSOLUTE = re.compile(r'"/[^"]*"')


def relative(value):
    if isinstance(value, str) and value.startswith("/"):
        for marker in KEEP:
            if marker in value:
                return marker + value.split(marker, 1)[1]
        raise ValueError(f"absolute path without a known root: {value!r}")
    if isinstance(value, dict):
        return {k: relative(v) for k, v in value.items()}
    if isinstance(value, list):
        return [relative(v) for v in value]
    return value


def export(source, target):
    target.mkdir(parents=True, exist_ok=True)
    for path in sorted(source.iterdir()):
        if path.suffix == ".jsonl":
            lines = [json.dumps(relative(json.loads(line))) for line in path.read_text().splitlines() if line.strip()]
            text = "\n".join(lines) + "\n"
        elif path.suffix == ".json":
            text = json.dumps(relative(json.loads(path.read_text())), indent=2) + "\n"
        else:
            continue
        if ABSOLUTE.search(text):
            raise ValueError(f"absolute path left in {path}")
        (target / path.name).write_text(text)


def main():
    source, target = Path(sys.argv[1]), Path(sys.argv[2])
    runs = []
    for d in sorted(source.iterdir()):
        summary = d / "summary" if (d / "summary" / "metrics.jsonl").exists() else d   # run dir or summaries dir
        if (summary / "metrics.jsonl").exists():
            runs.append((d.name, summary))
    for name, summary in runs:
        if (target / name).exists():
            shutil.rmtree(target / name)
        export(summary, target / name)
    if (source / "position-probe").is_dir():
        export(source / "position-probe", target / "position-probe")
    print(f"exported {len(runs)} runs to {target}")


if __name__ == "__main__":
    main()
