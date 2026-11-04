# RFC (draft, not filed): a multimodal track for NextTransformer

Status: draft for discussion; the results section is updated as pilot runs finish.

## Summary

Add a second protocol next to `nextt-reference-50b-v1` in which every row of the training stream carries image tokens
from a frozen vision encoder, at an 8,192-token context. The track keeps NextTransformer's contract (fixed data and
order, matched theoretical training FLOPs per token audited by the operator ledger, fixed evaluation events) and asks
the same question for multimodal pretraining: how should a backbone spend a fixed training budget when part of the
sequence is an image?

A prototype lives in a separate repository that depends on NextTransformer at a pinned commit and changes none of its
files. This RFC lists what the prototype had to work around, so the track could later be merged as a new protocol.

## Why it is worth a separate protocol

- At 2,048 tokens of text, attention is about 9% of the Dense baseline's training FLOPs, so sparse and linear
  attention can save at most a few layers' worth of compute and their main benefits are not exercised. At 8,192 tokens
  attention is about 29%; GDN saves about 22% and NSA about 20% of the Dense cost, which calibration turns into 7 and
  6 extra layers.
- Current production backbones (Qwen's GDN + sparse-attention preview of Qwen4, Kimi K3's KDA + MLA without RoPE) are
  hybrids of exactly the mixers in the zoo, and they are natively multimodal. How these mixers handle image tokens is
  not covered by the text protocol.
- Multimodal position encoding research (M-RoPE, MRoPE-Interleave, RoPE-TV) is built on RoPE attention; in hybrids
  most layers are linear and have no position encoding at all.

## Design of the prototype

- Visual tokenizer: frozen SigLIP2 so400m/16 at 384 px, 2 x 2 merge, 144 tokens per image; its cost is reported
  separately, like the text tokenizer. A raw-patch variant (32 px patches, no encoder) is available.
- Injection: a projector (two linear layers, ReLU^2) and forward hooks on `transformer.wte` and the `value_embeds`
  tables replace the embedding at image positions and zero the value embedding there. Every registered method reads
  token ids only through these modules, so all methods work unchanged; with no image positions the model is bitwise
  identical to NextTransformer's.
- Data: CC12M + CC3M image-caption pairs and ClimbMix text, packed offline into 8,193-position rows with 12 image
  documents (`[BOS] [image] caption`) per row and best-fit text; seeded and deterministic. The dev shard (06541) is
  the text evaluation; the validation shard (06542) and CORE are untouched.
- Budget of the pilot: 5,000 updates of 1,048,576 tokens (128 rows x 8,192).
- Compute: NextTransformer's `Ledger`, attention rules and optimizer accounting, applied to the multimodal model with a
  nominal layout of 12 images per row. The Dense shell at 8,192 tokens is the anchor (5.9956 GFLOPs per token); every
  method is calibrated to within 1% with the v2 grid rule.
- Evaluation: caption BPB (and with shuffled images), text BPB, COCO 5k caption BPB and 5-way retrieval by likelihood,
  a synthetic spatial-relation probe and What'sUp.

## What the prototype had to work around (candidate hooks)

1. `scale="formal"` pins the context to 2,048; a new protocol has no legal formal configuration, so the prototype uses
   `scale="debug"` (which only lifts validation guards). A protocol-bound validation would remove the workaround.
2. No embedding hook: multimodal input needs a supported way to replace token embeddings (and value embeddings) at
   given positions.
3. The ledger has no entries for `masked_scatter` / `masked_scatter_backward` / `index_select` (data movement) and
   `index_add` (accumulation); the prototype subclasses `Ledger`. A patch with these four rules and a test is in
   `upstream/0001-ledger-data-movement-ops.patch`. It passes `tests/test_compute.py` and `checks.golden` (every method
   is bitwise identical), but because `compute.py` is fingerprinted, it marks every published cost proof as stale until
   a maintainer re-runs the full audit. A plugin-level way to register extra ledger rules would avoid that.
4. `parameter_groups` validates the training configuration against the 50B protocol, so another budget cannot use it
   directly (the prototype passes the default configuration, whose learning rates and batch match).
5. `audit_graph` builds its own model, so a wrapped model cannot be audited without re-implementing the audit loop.
6. Preparation requires all 6,543 ClimbMix shards, while the tokenizer needs only the first 2B characters; a partial
   download reproduces the pinned tokenizer hash exactly.

## Preliminary results (pilot, 2026-11-04; updated as runs finish)

All runs: 5,000 updates of 128 x 8,192 tokens (5.24B tokens), frozen SigLIP2, seed 42 unless stated, training FLOPs
per token within 1% of the Dense anchor (`scripts/audit_variants.py`). Files: `results/`; tables: `results/tables.md`.

**Backbones at 8,192 tokens.**

| backbone | layers / FFN | parameters | caption BPB | text BPB (dev) | COCO caption BPB | seconds per update (8 x H200) |
|---|---|---:|---:|---:|---:|---:|
| Dense (anchor) | 23 / 6144 | 1.37B | 0.7728 | 0.7184 | 0.8472 | 2.90 |
| GDN hybrid | 28 / 6016 | 1.65B | 0.7613 | 0.7152 | 0.8419 | 3.16 |
| GDN, QSA hybrid, mHC | | | queued | | | |

At matched compute, the GDN hybrid gets five more layers and lower loss on captions and text. It is about 9% slower
per update with the current kernels.

**2D position of image tokens** (pre-registered in `docs/spatial-preregistration.md`; secondary analyses in
`docs/spatial-addendum.md`; the README has the full summary).

- **Pre-registered decision.** The 2D short convolution is not effective in either position mode. The vertical acc2
  gain is +0.1 points with NoPE and with RoPE-1D. Caption BPB changes by +0.0025 and +0.0022, text BPB by +0.0001 and
  -0.0002, and the seed spread is 0.1 points. The primary metric is saturated at 99.9–100%. SigLIP2 tokens carry their
  position: ridge regression recovers a token's row and column with R^2 0.86 / 0.89, against 0.03 / 0.00 for raw
  pixel patches.
- **Mechanism.** Reordering the image tokens while keeping their content shows where each backbone reads position
  from. The table gives acc2 on the harder probe after the token order is reversed along one axis: near 0 means
  position is read from token order, near 100% from content. The seed spread, measured with two seed pairs, is 3.1
  points for the horizontal column and 7.7 for the vertical one.

  | run | horizontal, columns reversed | vertical, rows reversed |
  |---|---:|---:|
  | Dense, RoPE-1D | 80.5% | 31.9% |
  | Dense, NoPE | 82.0% | 51.7% |
  | Hybrid, RoPE-1D (seeds 42 / 43) | 27.7% / 30.8% | 58.3% / 65.5% |
  | Hybrid, RoPE-2D | 36.9% | 61.2% |
  | Hybrid, NoPE | 19.0% | 87.7% |
  | Hybrid, NoPE + 2D conv (seeds 42 / 43) | 19.5% / 22.1% | 28.6% / 20.9% |
  | Hybrid, RoPE-1D + 2D conv | 20.0% | 21.0% |

  - The recurrence of the linear layers reads horizontal order.
  - Without RoPE, the hybrid reads vertical position from the encoder's features.
  - With the 2D convolution, it reads vertical position from the token grid, in both position modes.
  - RoPE-2D does not change where vertical position is read from, and shifts horizontal reading only slightly
    towards content (+9.2 points, against a 6.3-point threshold).
- **Accuracy.** No run differs from another by more than twice the seed spread on either axis of the harder probe
  (88–96%). COCO retrieval is 98% for every run, and What'sUp is near chance at this scale.
- **Resolution transfer** (exploratory, `docs/spatial-addendum.md`). Final checkpoints are scored without retraining on
  the same images at 576 and 768 px, that is 18 x 18 and 24 x 24 tokens per image instead of 12 x 12. At 768 px,
  vertical acc2 changes as follows:

  | backbone and position mode | vertical acc2 change at 768 px |
  |---|---:|
  | RoPE-1D (Dense, and the hybrid with two seeds) | -13 to -20 points |
  | NoPE | -5 points at most |
  | NoPE hybrid with the 2D convolution (two seeds) | +1.8 / -1.1 points |

  The seed spread of the change is 7.0 points. The track can therefore separate position designs by how they
  generalize to new resolutions, even when they tie at the training resolution.
- **Layer probe.** Every backbone derives a token's row from token order within a few layers (R² 0.83–0.98), but
  almost none of its column (≤ 0.15). The hybrids keep the encoder's position in every layer; Dense with RoPE-1D loses
  much of it in the middle layers. GDN forget gates retain more state over image tokens than over text.
- **Pending.** The raw-patch replication pair, where the encoder supplies no position. Without the convolution, the
  NoPE hybrid reads vertical relations from token order (97.6%) but not horizontal ones (51%, chance).

## Open questions

- Encoder cost: report only (as now), or charge it to the budget?
- Image share and resolution as protocol variables (12 x 144 tokens per row in the pilot).
- Evaluation suite for the formal version (multimodal CORE analogue).
