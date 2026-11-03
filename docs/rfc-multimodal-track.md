# RFC (draft, not filed): a multimodal track for NextTransformer

Status: draft for discussion; results section to be completed from the pilot runs.

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
   `index_add` (accumulation); the prototype subclasses `Ledger`.
4. `parameter_groups` validates the training configuration against the 50B protocol, so another budget cannot use it
   directly (the prototype passes the default configuration, whose learning rates and batch match).
5. `audit_graph` builds its own model, so a wrapped model cannot be audited without re-implementing the audit loop.
6. Preparation requires all 6,543 ClimbMix shards, while the tokenizer needs only the first 2B characters; a partial
   download reproduces the pinned tokenizer hash exactly.

## Preliminary results

To be completed: benchmark of the calibrated backbones at 8,192 tokens, the 2D-position study (pre-registered in
`docs/spatial-preregistration.md`), throughput per method.

## Open questions

- Encoder cost: report only (as now), or charge it to the budget?
- Image share and resolution as protocol variables (12 x 144 tokens per row in the pilot).
- Evaluation suite for the formal version (multimodal CORE analogue).
