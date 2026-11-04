"""Exported run summaries keep every value except site paths, and refuse unknown absolute paths."""
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("export_results", Path(__file__).parents[1] / "scripts" / "export_results.py")
export_results = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export_results)


def test_paths_become_relative_and_values_are_kept(tmp_path):
    run = tmp_path / "runs" / "h-nope" / "summary"
    run.mkdir(parents=True)
    (run / "metrics.jsonl").write_text(json.dumps(dict(event="start", settings=dict(train_rows="/x/y/packed/pilot-train",
                                                                                     seed=42))) + "\n")
    (run / "spatial_checkpoint_00005000.json").write_text(json.dumps(dict(checkpoint="/x/y/runs/h-nope/checkpoint_00005000",
                                                                          vertical_acc2=.999)))
    export_results.export(run, tmp_path / "out")
    start = json.loads((tmp_path / "out" / "metrics.jsonl").read_text())
    probe = json.loads((tmp_path / "out" / "spatial_checkpoint_00005000.json").read_text())
    assert start["settings"] == dict(train_rows="packed/pilot-train", seed=42)
    assert probe == dict(checkpoint="runs/h-nope/checkpoint_00005000", vertical_acc2=.999)


def test_unknown_absolute_path_is_refused(tmp_path):
    (tmp_path / "latest.json").write_text(json.dumps(dict(encoder="/home/someone/siglip")))
    with pytest.raises(ValueError):
        export_results.export(tmp_path, tmp_path / "out")
