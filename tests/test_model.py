"""Visual injection must leave every NextTransformer method intact and feed images correctly."""
import pytest
import torch

from nexttransformer.models.config import METHODS, debug_config
from nexttransformer.models.lm import build_model
from nexttransformer.training.optimizer import parameter_groups
from ntv.model import build_mm_model, text_only, visual_inputs

VISUAL_DIM = 20


@pytest.fixture(autouse=True)
def reproducible():
    torch.manual_seed(42)
    torch.set_num_threads(1)


def activate(model):
    # Shell output projections start at zero; give them weight so every branch affects the output.
    with torch.no_grad():
        for block in model.transformer.h:
            block.attn.c_proj.weight.normal_(std=.03)
            block.mlp.c_proj.weight.normal_(std=.03)


def pair(method):
    """The plain language model and its multimodal twin with identical language-model weights."""
    c = debug_config(method, sequence_len=16)
    plain = build_model(c)
    activate(plain)
    mm = build_mm_model(c, visual_dim=VISUAL_DIM)
    mm.load_state_dict({**mm.state_dict(), **plain.state_dict()})
    for m in (plain, mm):
        m.set_step(c.burnin_steps + 1)  # the sparse path, not only the dense burn-in
    return c, plain, mm


def images(c, batch=2, spans=((3, 7), (10, 12))):
    mask = torch.zeros((batch, c.sequence_len), dtype=torch.bool)
    for start, stop in spans:
        mask[:, start:stop] = True
    return mask, torch.randn(int(mask.sum()), VISUAL_DIM)


def lm_grads(model):
    return {n: p.grad for n, p in model.named_parameters()
            if p.requires_grad and not n.startswith("visual_projector.")}


@pytest.mark.parametrize("method", METHODS)
def test_text_only_batches_reproduce_the_language_model(method):
    c, plain, mm = pair(method)
    idx = torch.randint(c.vocab_size, (2, c.sequence_len))
    labels = torch.randint(c.vocab_size, idx.shape)
    plain(idx, labels).backward()
    with text_only(mm, idx):
        mm(idx, labels).backward()
    expected = lm_grads(plain)
    for name, grad in lm_grads(mm).items():
        assert torch.equal(grad, expected[name]), name
    # The projector ran on zero rows: it has (zero) gradients, as the trainer requires.
    for p in mm.visual_projector.parameters():
        assert p.grad is not None and not p.grad.any()
    plain.eval(); mm.eval()
    with torch.no_grad():
        reference = plain(idx)
        assert torch.equal(mm(idx), reference)  # no visual state: hooks are inert
        with text_only(mm, idx):
            assert torch.equal(mm(idx), reference)


@pytest.mark.parametrize("method", METHODS)
def test_image_positions_ignore_the_placeholder_id_and_stay_causal(method):
    c, _, mm = pair(method)
    mask, features = images(c)
    idx = torch.randint(c.vocab_size, (2, c.sequence_len))
    other = idx.clone()
    other[mask] = (other[mask] + 5) % c.vocab_size
    mm.eval()
    with torch.no_grad():
        with visual_inputs(mm, mask, features):
            out = mm(idx)
        with visual_inputs(mm, mask, features):
            torch.testing.assert_close(mm(other), out, atol=0, rtol=0)
        changed = features.clone()
        changed[:4] += 1.  # the first image of the first row only
        with visual_inputs(mm, mask, changed):
            moved = mm(idx)
    first = 3
    torch.testing.assert_close(moved[:, :first], out[:, :first], atol=0, rtol=0)
    assert not torch.equal(moved[0, first:], out[0, first:])


@pytest.mark.parametrize("method", ["dense_full", "gdn_hybrid", "nsa", "mhc", "ouro"])
def test_projector_learns_from_text_targets(method):
    c, _, mm = pair(method)
    mask, features = images(c)
    idx = torch.randint(c.vocab_size, (2, c.sequence_len))
    labels = torch.randint(c.vocab_size, idx.shape)
    labels[mask] = -1
    with visual_inputs(mm, mask, features):
        loss = mm(idx, labels)
    loss.backward()
    assert torch.isfinite(loss)
    for p in mm.visual_projector.parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0


def test_projector_is_owned_by_one_muon_group():
    c = debug_config("dense_full", sequence_len=16)
    mm = build_mm_model(c, visual_dim=VISUAL_DIM)
    owners = {}
    for group in parameter_groups(mm):
        for name in group["parameter_names"]:
            owners[name] = group["kind"]
    assert owners["visual_projector.c_fc.weight"] == "muon"
    assert owners["visual_projector.c_proj.weight"] == "muon"


def test_visual_inputs_reject_mismatched_features():
    c = debug_config("dense_full", sequence_len=16)
    mm = build_mm_model(c, visual_dim=VISUAL_DIM)
    mask, features = images(c)
    with pytest.raises(ValueError):
        with visual_inputs(mm, mask, features[:-1]):
            pass
    assert mm._visual is None
