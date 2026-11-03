"""Training-cost audit of a multimodal model, with NextTransformer's operator ledger.

The audit is NextTransformer's `audit_graph` with visual inputs: a fake (shape-only) forward, backward and
one optimizer update of the multimodal model, counted operator by operator by the same `Ledger`, attention
rules and optimizer accounting. Image positions follow a fixed nominal layout; the projector and the
embedding hooks are charged like any other operator. The frozen vision encoder is outside this account
(it is identical for every method and is reported separately).
"""
from dataclasses import dataclass
import math

import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from nexttransformer.models.config import CHUNK_BACKENDS, LINEAR
from nexttransformer.models.plugins import stage_start
from nexttransformer.models.registry import PLUGIN_METHODS
from nexttransformer.protocol.compute import FORMULA_VERSION, Ledger, attention_rules, count_optimizer
from ntv.model import VISUAL_DIM, build_mm_model, visual_inputs


class TrackLedger(Ledger):
    """NextTransformer's ledger plus the data-movement operators of the visual hooks.

    `masked_scatter` copies projected features into image positions and its backward selects their
    gradients: data movement, which the ledger keeps outside the FLOP account (like `index_put`).
    """
    ZERO = Ledger.ZERO | {"masked_scatter", "masked_scatter_backward"}


@dataclass(frozen=True)
class Track:
    """Controlled values of the multimodal track that enter the cost account."""
    sequence_len: int = 8192
    rows: int = 128                    # 128 x 8192 = 1,048,576 tokens per update, as in NextTransformer
    total_tokens: int = 5_242_880_000  # pilot budget: 5,000 updates
    image_tokens: int = 144            # per image after the 2 x 2 merge
    images_per_row: int = 12           # nominal audit layout: 1,728 of 8,192 positions (21.1%)
    visual_dim: int = VISUAL_DIM

    @property
    def global_batch_tokens(self):
        return self.rows * self.sequence_len

    @property
    def steps(self):
        if self.total_tokens % self.global_batch_tokens:
            raise ValueError("token budget must contain complete updates")
        return self.total_tokens // self.global_batch_tokens


def nominal_image_mask(track, sequence_len=None, rows=1):
    """Images spread evenly through each row, separated by text."""
    t = sequence_len or track.sequence_len
    n, size = track.images_per_row, track.image_tokens
    if n * size >= t:
        raise ValueError("the nominal image layout does not fit the context")
    gap = (t - n * size) // (n + 1)
    mask = torch.zeros((rows, t), dtype=torch.bool)
    for i in range(n):
        start = gap * (i + 1) + size * i
        mask[:, start:start + size] = True
    return mask


def stage_start_step(config, stage, steps):
    if config.method in PLUGIN_METHODS:
        return stage_start(config, stage)
    if stage == "burnin":
        return 0
    if stage == "gate_only":
        return steps - config.ouro_gate_steps
    return config.burnin_steps + 1


def audit_mm(config, track=Track(), *, stage="joint", depth=None, optimizer=True, images=True):
    """One logical update (track.rows x track.sequence_len tokens), counted with the ledger."""
    t, rows = config.sequence_len, track.rows
    if t != track.sequence_len:
        raise ValueError("audit the configuration at the track's context length")
    if config.method in LINEAR and config.backend not in CHUNK_BACKENDS:
        raise NotImplementedError("scan backends need NextTransformer's affine extrapolation; not supported yet")
    pair = CHUNK_BACKENDS.get(config.backend)
    if pair in (config.width, config.heads, config.head_dim, config.mlp_width,
                config.vocab_size, config.gate_rank, config.conv_size):
        raise ValueError("a model dimension equals the chunk length; C x C tiles would be ambiguous")
    mask = nominal_image_mask(track, t, rows) if images else torch.zeros((rows, t), dtype=torch.bool)
    image_positions = int(mask.sum())
    ledger = TrackLedger(pair)
    with FakeTensorMode(allow_non_fake_inputs=True):
        model = build_mm_model(config, "cpu", visual_dim=track.visual_dim)
        model.set_training_stage(stage_start_step(config, stage, track.steps))
        if config.method == "huginn":
            if depth is None:
                raise ValueError("Huginn cost requires an explicit sampled depth")
            model.accounting_depth = depth
        x = torch.zeros((rows, t), dtype=torch.long)
        features = torch.zeros((image_positions, track.visual_dim))
        with attention_rules(model, ledger, stage), ledger, visual_inputs(model, mask, features):
            loss = model(x, x)
            ledger.phase = "backward"
            loss.backward()
        if optimizer:
            count_optimizer(model, ledger)
        parameters = sum(p.numel() for p in model.parameters())
        projector = sum(p.numel() for p in model.visual_projector.parameters())
    report = ledger.report()
    report.update(method=config.method, model_hash=config.fingerprint, sequence_len=t, rows=rows, stage=stage,
                  depth=depth, parameters=parameters, projector_parameters=projector,
                  image_positions=image_positions, flops_per_token=report["total"] / (rows * t),
                  formula_version=FORMULA_VERSION)
    return report


def stages(config, track=Track()):
    """(stage, depth, updates) with the method's registered stage lengths (as in NextTransformer)."""
    steps = track.steps
    if config.method == "ouro":
        return [("joint", None, steps - config.ouro_gate_steps), ("gate_only", None, config.ouro_gate_steps)]
    if config.has_sparse_burnin:
        return [("burnin", None, config.burnin_steps), ("joint", None, steps - config.burnin_steps)]
    if config.method == "huginn" or config.method in PLUGIN_METHODS:
        raise NotImplementedError(f"{config.method}: stage weighting not supported yet")
    return [("joint", None, steps)]


def flops_per_token(config, track=Track(), audit=audit_mm):
    """Update-weighted average training FLOPs per input token over the whole budget."""
    weighted = 0
    for stage, depth, updates in stages(config, track):
        weighted += updates * audit(config, track, stage=stage, depth=depth)["total"]
    return weighted / track.total_tokens


def relative_error(value, anchor):
    return value / anchor - 1


def is_close(value, anchor, tolerance=.01):
    return math.isfinite(value) and abs(relative_error(value, anchor)) <= tolerance
