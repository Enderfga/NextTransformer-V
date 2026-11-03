"""The track's audit is NextTransformer's audit plus the visual path, and nothing else."""
from dataclasses import replace

import pytest

from nexttransformer.models.config import debug_config
from nexttransformer.protocol.compute import audit_graph
from ntv.cost import Track, audit_mm

TRACK = Track(sequence_len=16, rows=2, image_tokens=144, images_per_row=0)


@pytest.mark.parametrize("method", ["dense_full", "gdn_hybrid", "nsa", "qsa", "mhc", "gdn"])
def test_text_only_audit_equals_nexttransformer(method):
    c = debug_config(method, sequence_len=16)
    track = replace(TRACK, sequence_len=c.sequence_len)
    ours = audit_mm(c, track, optimizer=False, images=False)
    theirs = audit_graph(c, batch=track.rows, optimizer=False)
    assert ours["unknown"] == theirs["unknown"] == {}
    assert ours["operators"] == theirs["operators"]


def test_images_add_only_the_projector():
    c = debug_config("dense_full", sequence_len=400)
    track = Track(sequence_len=400, rows=2, images_per_row=2)
    with_images = audit_mm(c, track, optimizer=False)
    without = audit_mm(c, track, optimizer=False, images=False)
    assert with_images["unknown"] == {} and with_images["image_positions"] == 2 * 2 * 144
    n, d, w = with_images["image_positions"], 4608, c.width
    # Forward and weight gradients of both matrices; an input gradient only for the second, because the
    # frozen encoder's features need none.
    projector_matmuls = 2 * 2 * n * (d * w + w * w) + 2 * n * w * w
    extra = with_images["matrix_total"] - without["matrix_total"]
    assert extra == projector_matmuls
