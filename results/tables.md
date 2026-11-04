### Pre-registered decision (primary metric: synthetic-probe vertical acc2, final checkpoint)

| 2D convolution, position mode | treated | baseline | gain (points) | caption BPB change | text BPB change | verdict |
|---|---|---|---:|---:|---:|---|
| NoPE | h-nope-conv2d | h-nope | +0.1 | +0.0025 | +0.0001 | negative |
| RoPE-1D | h-rope1d-conv2d | h-rope1d | – | – | – | pending |
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

### Harder-probe contrasts (descriptive)

Differences in points, treated minus baseline. **Bold**: larger than twice the seed spread on that metric.

| treated | baseline | V | H | V, row flip | H, col flip |
|---|---|---:|---:|---:|---:|
| h-nope-conv2d | h-nope | -3.2 | +1.2 | **-59.1** | +0.5 |
| h-rope2d | h-rope1d | -1.9 | +1.2 | +3.0 | **+9.2** |
| h-nope | h-rope1d | +0.7 | +0.5 | **+29.5** | **-8.7** |
| d-nope | d-rope1d | +0.0 | -1.3 | **+19.8** | +1.5 |
| h-rope1d | d-rope1d | -0.8 | -4.0 | **+26.3** | **-52.8** |
| h-nope | d-nope | -0.1 | -2.2 | **+36.0** | **-63.0** |
| 2x seed spread | | 5.7 | 5.3 | 15.4 | 6.3 |

### Position carried by the visual tokens (ridge regression, before any backbone)

| visual input | row R^2 | column R^2 | row accuracy | column accuracy | chance |
|---|---:|---:|---:|---:|---:|
| raw32 | 0.03 | 0.00 | 9% | 8% | 8% |
| siglip | 0.86 | 0.89 | 36% | 40% | 8% |
