### Pre-registered decision (primary metric: synthetic-probe vertical acc2, final checkpoint)

| 2D convolution, position mode | treated | baseline | gain (points) | caption BPB change | text BPB change | verdict |
|---|---|---|---:|---:|---:|---|
| NoPE | h-nope-conv2d | h-nope | +0.1 | +0.0025 | +0.0001 | negative |
| RoPE-1D | h-rope1d-conv2d | h-rope1d | +0.1 | +0.0022 | -0.0002 | negative |
| NoPE, raw patches | r-h-nope-conv2d | r-h-nope | – | – | – | pending |

Seed spread of the primary metric: 0.1 points (larger of the two seed pairs).

### All runs (final checkpoint)

Accuracies in %. `col flip` / `row flip`: horizontal accuracy with the columns of each token row reversed, vertical accuracy with the token rows reversed (harder probe).

| run | method | position | 2D conv | visual | seed | caption BPB | text BPB | synth H | synth V | hard H | hard V | hard H, col flip | hard V, row flip | What'sUp H | What'sUp V | COCO BPB | COCO retrieval | s/update |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| d-rope1d | dense_full | rope1d | none | siglip | 42 | 0.7728 | 0.7184 | 100.0 | 100.0 | 94.8 | 94.1 | 80.5 | 31.9 | 51.4 | 53.8 | 0.8472 | 98.2 | 2.90 |
| d-nope | dense_full | nope | none | siglip | 42 | 0.7934 | 0.7375 | 100.0 | 100.0 | 93.5 | 94.1 | 82.0 | 51.7 | 50.5 | 52.4 | 0.8556 | 98.1 | 2.87 |
| h-rope1d | gdn_hybrid | rope1d | none | siglip | 42 | 0.7613 | 0.7152 | 100.0 | 99.9 | 90.8 | 93.3 | 27.7 | 58.3 | 49.5 | 55.3 | 0.8419 | 98.3 | 3.16 |
| h-rope1d-s2 | gdn_hybrid | rope1d | none | siglip | 43 | 0.7620 | 0.7152 | 100.0 | 100.0 | 92.3 | 96.1 | 30.8 | 65.5 | 51.9 | 51.0 | 0.8425 | 98.4 | 3.16 |
| h-nope | gdn_hybrid | nope | none | siglip | 42 | 0.7692 | 0.7190 | 100.0 | 99.9 | 91.3 | 94.0 | 19.0 | 87.7 | 51.0 | 52.9 | 0.8434 | 98.1 | 3.19 |
| h-rope2d | gdn_hybrid | rope2d | none | siglip | 42 | 0.7604 | 0.7150 | 100.0 | 100.0 | 92.0 | 91.4 | 36.9 | 61.2 | 50.0 | 51.9 | 0.8386 | 98.1 | 3.16 |
| h-nope-conv2d | gdn_hybrid | nope | k | siglip | 42 | 0.7717 | 0.7190 | 99.9 | 100.0 | 92.5 | 90.8 | 19.5 | 28.6 | 45.7 | 51.9 | 0.8471 | 98.2 | 3.40 |
| h-nope-conv2d-s2 | gdn_hybrid | nope | k | siglip | 43 | 0.7727 | 0.7196 | 99.9 | 99.9 | 89.9 | 91.4 | 22.1 | 20.9 | 49.5 | 52.9 | 0.8428 | 98.2 | 3.40 |
| h-rope1d-conv2d | gdn_hybrid | rope1d | k | siglip | 42 | 0.7635 | 0.7150 | 99.9 | 100.0 | 87.7 | 93.0 | 20.0 | 21.0 | 51.4 | 51.4 | 0.8460 | 98.3 | 3.41 |
| r-h-nope | gdn_hybrid | nope | none | raw32 | 42 | 0.9241 | 0.7200 | 50.8 | 97.6 | 52.0 | 79.5 | 47.8 | 21.6 | 50.5 | 50.0 | 1.0230 | 76.3 | 2.92 |

### Harder-probe contrasts (descriptive)

Differences in points, treated minus baseline. **Bold**: larger than twice the seed spread on that metric.

| treated | baseline | V | H | V, row flip | H, col flip |
|---|---|---:|---:|---:|---:|
| h-nope-conv2d | h-nope | -3.2 | +1.2 | **-59.1** | +0.5 |
| h-rope1d-conv2d | h-rope1d | -0.3 | -3.1 | **-37.3** | **-7.7** |
| h-rope2d | h-rope1d | -1.9 | +1.2 | +3.0 | **+9.2** |
| h-nope | h-rope1d | +0.7 | +0.5 | **+29.5** | **-8.7** |
| d-nope | d-rope1d | +0.0 | -1.3 | **+19.8** | +1.5 |
| h-rope1d | d-rope1d | -0.8 | -4.0 | **+26.3** | **-52.8** |
| h-nope | d-nope | -0.1 | -2.2 | **+36.0** | **-63.0** |
| 2x seed spread | | 5.7 | 5.3 | 15.4 | 6.3 |

### Zero-shot resolution transfer (harder probe, final checkpoint, no retraining)

The same probe images at 384 px (12 x 12 tokens, the training resolution), 576 px (18 x 18) and 768 px (24 x 24). Accuracies in %; change in points from 384 px.

| run | V 384 | V 576 | V 768 | V change at 768 | H 384 | H 576 | H 768 | H change at 768 | V 768, rows reversed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| d-rope1d | 94.1 | 89.3 | 73.8 | -20.3 | 94.8 | 90.9 | 78.4 | -16.5 | 56.2 |
| d-nope | 94.1 | 96.4 | 95.4 | +1.3 | 93.5 | 94.9 | 93.7 | +0.2 | 56.9 |
| h-rope1d | 93.3 | 87.8 | 73.0 | -20.3 | 90.8 | 93.0 | 82.2 | -8.6 | 54.8 |
| h-rope1d-s2 | 96.1 | 92.1 | 82.9 | -13.3 | 92.3 | 89.7 | 74.8 | -17.5 | 62.9 |
| h-nope | 94.0 | 92.3 | 89.1 | -4.8 | 91.3 | 89.8 | 86.8 | -4.6 | 77.2 |
| h-rope2d | 91.4 | 92.9 | 86.3 | -5.1 | 92.0 | 87.0 | 80.6 | -11.4 | 54.8 |
| h-nope-conv2d | 90.8 | 94.8 | 92.6 | +1.8 | 92.5 | 93.6 | 95.0 | +2.5 | 34.4 |
| h-nope-conv2d-s2 | 91.4 | 93.3 | 90.3 | -1.1 | 89.9 | 91.8 | 89.6 | -0.3 | 23.3 |
| h-rope1d-conv2d | 93.0 | 91.6 | 84.6 | -8.4 | 87.7 | 89.4 | 86.6 | -1.1 | 36.3 |

Seed spread of the change at 768 px (larger of the two seed pairs): vertical 7.0 points, horizontal 8.9 points.

### Layer probe (COCO images, R² of ridge regression from the residual stream)

`order`: the row (column) of the slot a token sits in, `content`: the row (column) of the cell it came from, both fitted on images fed in a random token order. Input = after the visual projector; middle and last = after the middle and the last block.

| run | input content row / col | middle: order row / col | middle: content row / col | last: order row / col | last: content row / col |
|---|---:|---:|---:|---:|---:|
| d-rope1d | 0.67 / 0.60 | 0.85 / 0.03 | 0.37 / 0.28 | 0.84 / 0.03 | 0.48 / 0.39 |
| d-nope | 0.70 / 0.55 | 0.90 / 0.03 | 0.60 / 0.32 | 0.84 / 0.01 | 0.64 / 0.38 |
| h-rope1d | 0.72 / 0.63 | 0.92 / 0.03 | 0.71 / 0.60 | 0.86 / 0.03 | 0.67 / 0.51 |
| h-rope1d-s2 | 0.72 / 0.61 | 0.92 / 0.03 | 0.72 / 0.58 | 0.87 / 0.03 | 0.69 / 0.50 |
| h-nope | 0.75 / 0.62 | 0.91 / 0.03 | 0.74 / 0.59 | 0.87 / 0.03 | 0.73 / 0.55 |
| h-rope2d | 0.71 / 0.62 | 0.87 / 0.15 | 0.70 / 0.59 | 0.83 / 0.10 | 0.66 / 0.51 |
| h-nope-conv2d | 0.76 / 0.64 | 0.86 / 0.05 | 0.74 / 0.60 | 0.84 / 0.04 | 0.72 / 0.56 |
| h-nope-conv2d-s2 | 0.76 / 0.65 | 0.91 / 0.04 | 0.75 / 0.62 | 0.90 / 0.06 | 0.74 / 0.59 |
| h-rope1d-conv2d | 0.74 / 0.62 | 0.94 / 0.07 | 0.73 / 0.59 | 0.88 / 0.06 | 0.69 / 0.51 |
| r-h-nope | 0.03 / 0.01 | 0.98 / 0.05 | 0.11 / 0.02 | 0.98 / 0.05 | 0.10 / 0.02 |

### Position carried by the visual tokens (ridge regression, before any backbone)

| visual input | row R^2 | column R^2 | row accuracy | column accuracy | chance |
|---|---:|---:|---:|---:|---:|
| raw32 | 0.03 | 0.00 | 9% | 8% | 8% |
| siglip | 0.86 | 0.89 | 36% | 40% | 8% |
