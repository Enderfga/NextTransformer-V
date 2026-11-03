# Pre-registered analysis: 2D position in hybrid VLM backbones

Written before any run of this phase finished; it is not changed after results arrive.

## Question

Where does a hybrid backbone (linear-attention layers plus a few full-attention layers) get the 2D position of image
tokens, and does 2D-aware linear mixing (a 2D short convolution at image positions) help?

## Runs

All runs use the same packed data (CC12M + CC3M + synthetic spatial images, ClimbMix text), 5,000 updates of
1,048,576 tokens, 8,192-token rows with 12 images of 144 tokens each, frozen SigLIP2 features unless stated, and
configurations calibrated to within 1% of the Dense anchor's training FLOPs per token. Seed 42 unless stated.

| Run | Backbone | Attention position | Patch order | 2D short conv | Input | Seed |
|---|---|---|---|---|---|---|
| d-rope1d | Dense | RoPE-1D | raster | none | SigLIP2 | 42 |
| d-nope | Dense | NoPE | raster | none | SigLIP2 | 42 |
| h-rope1d | GDN hybrid | RoPE-1D | raster | none | SigLIP2 | 42 |
| h-rope1d-s2 | GDN hybrid | RoPE-1D | raster | none | SigLIP2 | 43 |
| h-nope | GDN hybrid | NoPE | raster | none | SigLIP2 | 42 |
| h-rope2d | GDN hybrid | RoPE-2D (RoPE-TV) | raster | none | SigLIP2 | 42 |
| h-nope-conv2d | GDN hybrid | NoPE | raster | K | SigLIP2 | 42 |
| h-nope-conv2d-s2 | GDN hybrid | NoPE | raster | K | SigLIP2 | 43 |
| h-rope1d-conv2d | GDN hybrid | RoPE-1D | raster | K | SigLIP2 | 42 |
| r-h-nope | GDN hybrid | NoPE | raster | none | raw 32 px patches | 42 |
| r-h-nope-conv2d | GDN hybrid | NoPE | raster | K | raw 32 px patches | 42 |

Further runs (Hilbert order, RoPE-2D + 2D conv, benchmark backbones) are exploratory and do not enter the decision.

## Primary metric

Two-way accuracy (true relation against its opposite, by loss given the image) on vertical relations of the
synthetic spatial probe (2,000 images, seed 2), at the final checkpoint: `vertical_acc2`.

## Seed spread

`spread = max(|vertical_acc2(h-rope1d) - vertical_acc2(h-rope1d-s2)|,
              |vertical_acc2(h-nope-conv2d) - vertical_acc2(h-nope-conv2d-s2)|)`

## Decision rule for the 2D short convolution

The 2D short convolution is called effective in a position mode if, against the baseline with the same position
mode (h-nope-conv2d vs h-nope; h-rope1d-conv2d vs h-rope1d), all of the following hold:

1. `vertical_acc2` improves by at least 5 percentage points;
2. the improvement exceeds `2 * spread`;
3. neither the caption BPB nor the text BPB of the final evaluation is worse by more than 0.003.

Otherwise the result is reported as negative (no improvement beyond noise) or inconclusive. The raw-patch pair
(r-h-nope-conv2d vs r-h-nope) is reported as a replication under the same rule.

## Secondary, descriptive analyses

Horizontal vs vertical accuracy (acc2 and acc4) per run; accuracy by distance; What'sUp; COCO caption BPB and 5-way
retrieval; learning curves of the probe over the five checkpoints. None of these changes the decision.

## Reserved data

The NextTransformer validation shard (06542) and CORE are not used for any choice. The text evaluation uses the dev
shard (06541).
