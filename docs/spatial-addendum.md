# Addendum to the pre-registered analysis (secondary analyses)

Written on 2026-11-04 after the first run (h-rope1d) was evaluated. The pre-registration
(`docs/spatial-preregistration.md`) is unchanged: its primary metric, seed spread and decision rule still decide the
result. Nothing below enters that decision.

## Why

On the first finished run (GDN hybrid, RoPE-1D, frozen SigLIP2), the pre-registered probe is at its ceiling:
`vertical_acc2` is 99.9% at the final checkpoint and already 98.9% at update 1,000. A 5-point gain is impossible from
there, so for the SigLIP runs the decision rule can only return "negative". What'sUp is near chance (52%) and also
does not separate configurations. The analyses below were added to describe where position comes from with a measure
that has room to move. They were specified before they were run on any checkpoint, and none of their settings was
tuned on results.

## Added analyses

1. **Graded margin.** For every probe item, `margin2` is the loss of the opposite relation minus the loss of the true
   one (nats), averaged by group. It is a non-saturating version of acc2.
2. **Token-order scrambles.** The probe is scored again with the image tokens fed to the backbone in another order,
   with their content unchanged: `vflip` reverses the rows of the 12 x 12 token grid, `hflip` reverses the columns
   within each row, and `shuffle` applies one fixed random permutation. If a backbone reads position from token order,
   flipping an axis inverts its answers on that axis (acc2 near 0). If it reads position from token content, for
   example the encoder's position embedding, its answers are unaffected (acc2 near 100%).
3. **Harder synthetic probe.** This probe uses the same captions and relations with a layout that differs from
   training: an 8 x 8 grid instead of 4 x 4, object radius 12–18 px instead of 22–34, 3–6 distractors instead of 0–2,
   and a jitter of 6 px. It has 2,000 images (seed 3) and is built with
   `python -m ntv.synth eval --count 2000 --seed 3 --grid 8 --radius 12,18 --distractors 3,7 --jitter 6`.
   The parameters were fixed before its first evaluation. It is scored at every checkpoint, together with the
   scrambles.
4. **Linear position probe of the visual tokens.** Ridge regression predicts each token's row and column from its
   features, before any backbone. It is fitted on 1,024 COCO images and scored on 256 others; the penalty is chosen
   on a held-out part of the training images. It is run for SigLIP2 features and for raw 32 px patches.
5. **Raw-patch runs beyond the replication pair.** These are r-h-rope1d, r-h-rope2d, r-h-rope1d-conv2d, r-d-rope1d
   and r-d-nope. Raw patches carry no position embedding, so the backbone has to supply position itself. They are
   exploratory and were queued after all runs required by the pre-registration.

## Reading

All of these analyses are descriptive. Differences between runs on the harder probe are compared with the seed
spread measured on that probe (the same two seed pairs), not with the spread of the primary metric.

The report lists these harder-probe contrasts, fixed on 2026-11-04 before either seed pair had finished:

- the three pre-registered pairs (2D convolution against the same position mode);
- RoPE-2D against RoPE-1D, and NoPE against RoPE-1D, in the hybrid;
- NoPE against RoPE-1D in Dense;
- hybrid against Dense, with RoPE-1D and with NoPE;
- Hilbert against raster order (hybrid, NoPE);
- 2D convolution with RoPE-2D.

Each contrast is reported on four metrics: vertical acc2, horizontal acc2, vertical acc2 after the row flip, and
horizontal acc2 after the column flip. The seed spread of a metric is the larger difference between the two seed
pairs on that metric. A difference is marked when it exceeds twice that spread.

## Second addendum: resolution transfer and mechanism (2026-11-04)

These three analyses were added after the SigLIP2 results were in. They are exploratory, run on existing final
checkpoints without new training, and do not enter the decision.

6. **Zero-shot resolution transfer.** The harder probe, with the same images resized to 576 px and 768 px, so each
   image becomes 18 x 18 or 24 x 24 tokens instead of 12 x 12. Vertical neighbours are then 18 or 24 tokens apart in
   the sequence instead of 12. SigLIP2 interpolates its position embedding at the new size. Raw patches stay
   32 x 32 pixels. Run with `python -m ntv.spatial_eval --size 576` (or 768). At 384 px the code reproduces the
   archived probe results exactly, on all 222 values of a checkpoint.
7. **Layer probe.** Ridge regression from the residual stream to each image token's row and column, at the input and
   after every block, on COCO images. Fitted on 640 images, penalty chosen on 128, scored on 256. In the training order
   a token's place in the sequence and its place in the image coincide, so the images are also fed with their tokens in
   a fixed random order per image. There, separate probes are fitted for the slot a token sits in (position from
   order) and for the cell it came from (position from content). `python -m ntv.mechanism`.
8. **GDN retention.** For every GDN layer and head, the median forget gate at image and at caption positions, given
   as the fraction of the state that survives one token, one image row (12 tokens) and one image (144 tokens). Because
   the delta rule can also overwrite the state, these are upper bounds on what a layer can carry that far.
