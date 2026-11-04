# NextTransformer-V

A prototype multimodal track for [NextTransformer](https://github.com/Interactive-Active-AI/NextTransformer): the same
contract of fixed data, fixed data order and matched training compute, applied to sequences that interleave image tokens
with text at an 8,192-token context.

Its first study asks where a hybrid backbone (Gated DeltaNet layers with a few attention layers) gets the 2D position of
image tokens, and whether making its linear mixing 2D-aware helps. It was pre-registered before the runs started
([pre-registration](docs/spatial-preregistration.md), [addendum](docs/spatial-addendum.md)).

**Status (2026-11-04).** The diagnostic grid has finished. The seed pairs and the raw-pixel replication are still
running. Every number below comes from the files in [`results/`](results/), and the tables are generated from them
([`results/tables.md`](results/tables.md)).

## The 2D-position study

The six diagnostic runs share the same budget: about 1.4–1.6B parameters, 5,000 updates of 128 rows x 8,192 tokens
(5.24B tokens), 12 images per row, and training FLOPs per token within 1% of the Dense anchor. They differ in three
things:

- **Backbone:** Dense, or the GDN hybrid, where three of every four layers are GDN.
- **Attention position encoding:** RoPE-1D over the flattened sequence, NoPE, or RoPE-2D.
- **2D short convolution:** with or without a 3 x 3 depthwise convolution over each image's token grid, replacing the
  1D short convolution of the linear layers at image positions.

Image tokens are frozen SigLIP2 features (so400m/16 at 384 px, merged 2 x 2 into 12 x 12 tokens) fed in raster order.

### Findings so far

- **The pre-registered metric is saturated.** All six runs reach 99.9–100% vertical acc2 on the synthetic
  spatial-relation probe. The training mix contains images with the same layout, and SigLIP2 tokens carry their
  position (a linear probe recovers a token's row and column with R^2 0.86 / 0.89). A 5-point gain is therefore
  impossible, and the pre-registered rule can only call the 2D convolution ineffective in this setting.
- **Each backbone reads position from a different source.** The runs are scored again on a harder probe, with
  the image tokens reordered while their content is unchanged. Reversing the columns of each token row inverts the
  horizontal answers of a backbone that reads horizontal position from token order. Reversing the token rows does the
  same for vertical position.

  | run | horizontal, columns reversed | vertical, rows reversed |
  |---|---:|---:|
  | Dense, RoPE-1D | 80.5% (content) | 31.9% (order) |
  | Hybrid, RoPE-1D | 27.7% (order) | 58.3% (both) |
  | Hybrid, NoPE | 19.0% (order) | 87.7% (content) |
  | Hybrid, NoPE + 2D conv | 19.5% (order) | 28.6% (order, through the convolution) |

  The GDN recurrence supplies horizontal position from token order. In the hybrid without RoPE, vertical position
  comes from the encoder's features. The 2D convolution makes the hybrid read vertical position from the token grid
  instead.

  ![Where each backbone reads position from](docs/figures/token-order.svg)

  With the 2D convolution, vertical position is read from the token grid from the first checkpoint on. Without
  it, the NoPE hybrid relies more and more on the encoder's features as training goes on:

  ![Vertical accuracy with the token rows reversed, during training](docs/figures/token-order-curves.svg)

- **Accuracy and loss barely differ at this scale.** On the harder probe, all runs score 91–95% on both axes. The
  run with the 2D convolution is 3.2 points lower on vertical acc2; whether that exceeds seed noise is pending. COCO
  caption BPB (0.84–0.86), COCO 5-way retrieval (98%) and What'sUp (about 50%, near chance) barely separate the runs.

  ![Relation accuracy on the harder probe](docs/figures/accuracy.svg)

Still running: the two seed pairs, which set the noise level for every comparison above, and the raw-pixel runs. Raw
32 px patches carry no position embedding, so they test the 2D mechanisms when the encoder does not supply position.

## Design of the track

- Visual tokens: frozen SigLIP2, 144 tokens per image, with a projector of two linear layers. Forward hooks on the
  token and value embeddings feed the images, so every NextTransformer method runs unchanged. With no images the
  model is bitwise identical to NextTransformer's. A raw-patch input (32 px patches, no encoder) is also available.
- Data: CC12M + CC3M image-caption pairs, plus synthetic spatial-relation images, packed with ClimbMix text into
  8,193-position rows. There are 12 documents of the form `[BOS] [image] caption` per row, and the packing is
  deterministic. The text evaluation uses the dev shard (06541). The validation shard (06542) and CORE are not used.
- Compute: NextTransformer's operator ledger audits the multimodal model. Every backbone is calibrated to within 1% of
  the Dense anchor at 8,192 tokens (5.9956 GFLOPs per token) with the v2 grid rule
  ([`configs/track`](configs/track)). [`scripts/audit_variants.py`](scripts/audit_variants.py) checks every pilot
  variant against the anchor; the largest deviation is 0.29% ([`results/compute-audit.txt`](results/compute-audit.txt)).
- Evaluation:
  - caption and text BPB;
  - COCO 5k caption BPB and retrieval by likelihood;
  - the synthetic probe, its harder version and the token-order scrambles;
  - What'sUp.

What the prototype had to work around in NextTransformer, and could become upstream hooks, is listed in the
[RFC draft](docs/rfc-multimodal-track.md).

## Reproduce

Python 3.10 (`.python-version`) with [uv](https://docs.astral.sh/uv/). NextTransformer is a submodule pinned at the commit this was built
on, and none of its files are changed. Everything below writes under `$NTV_ROOT`.

```bash
git clone --recursive <this repository> && cd NextTransformer-V
export NTV_ROOT=<a large local disk>
bash scripts/bootstrap.sh                                   # environment (.venv), torch 2.9.1 + fla

# Inputs, at pinned revisions
bash scripts/download.sh climbmix 128 && bash scripts/download.sh siglip2
bash scripts/download.sh cc12m 0 995 && bash scripts/download.sh cc3m 0 575
bash scripts/download.sh coco5k && bash scripts/download.sh whatsup
python -m ntv.prepare_text --data-dir $NTV_ROOT/data/climbmix --resources $NTV_ROOT/resources   # pinned tokenizer
python -m ntv.whatsup --root $NTV_ROOT/data/whatsup

# Synthetic spatial-relation images: training shards, the probe, and the harder probe
python -m ntv.synth train --out $NTV_ROOT/data/synth --shards 120 --per-shard 5000 --seed 1
python -m ntv.synth eval --out $NTV_ROOT/data/synth-eval --count 2000 --seed 2
python -m ntv.synth eval --out $NTV_ROOT/data/synth-hard --count 2000 --seed 3 --grid 8 --radius 12,18 \
    --distractors 3,7 --jitter 6

# Packed rows (640,000 training rows; 256 validation rows on the dev shard)
python -m ntv.pack --tar-root $NTV_ROOT/data --source cc12m/cc12m-train:0:991 --source cc3m/cc3m-train:0:575 \
    --source synth/synth-train:0:119 --text $NTV_ROOT/data/climbmix --text-shards 0 79 \
    --resources $NTV_ROOT/resources --rows 640000 --out $NTV_ROOT/packed/pilot-train
python -m ntv.pack --tar-root $NTV_ROOT/data --source cc12m/cc12m-train:992:995 --text $NTV_ROOT/data/climbmix \
    --text-shards 6541 --resources $NTV_ROOT/resources --rows 256 --out $NTV_ROOT/packed/pilot-val

# Train and evaluate the pilot runs on an 8-GPU node (about 4.5 hours per run on H200)
bash scripts/pilot.sh configs/pilot/jobs.txt

# Report, pre-registered decision and tables
python scripts/export_results.py $NTV_ROOT/runs results    # summaries without site paths
python -m ntv.analysis results --out report.html --json results/decision.json --markdown results/tables.md \
    --figures docs/figures
```

Calibration is already in [`configs/track`](configs/track). To redo it:
`python -m ntv.calibrate --methods dense_full gdn gdn_hybrid nsa qsa qsa_hybrid mhc --out configs/track`.
The linear position probe of the visual tokens is `python -m ntv.position_probe`.

## Layout

| Path | Contents |
|---|---|
| `ntv/model.py`, `ntv/spatial.py` | visual inputs for any NextTransformer method; patch order, RoPE-2D, 2D short convolution |
| `ntv/encoder.py`, `ntv/data.py`, `ntv/pack.py` | frozen encoder, packed rows, deterministic packing |
| `ntv/cost.py`, `ntv/calibrate.py` | compute audit and calibration with NextTransformer's ledger |
| `ntv/train.py` | trainer (NextTransformer's optimizer, schedule and checkpoint format) |
| `ntv/evaluate.py`, `ntv/spatial_eval.py`, `ntv/synth.py`, `ntv/whatsup.py`, `ntv/position_probe.py` | evaluations |
| `ntv/analysis.py` | pre-registered decision, report and tables |
| `configs/` | calibrated backbones, the pilot job list |
| `docs/` | pre-registration, addendum, RFC draft |
| `results/` | summaries of every finished run (metrics and evaluation files) |
