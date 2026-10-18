# X3I merge and outputs

X3I is Sigma's Quattro Super Fine Detail container. The implementation reads
complete X3F frames through bounded file windows and merges them in frame 0
camera units. The [format reference](./format.md#x3i-super-fine-detail-container)
describes the measured container layout. The
[x3f-docs archive](https://github.com/sagwaco/x3f-docs/tree/main/spp-rgb-trace/followup-x3i-sfd)
contains the sample inspection and corrections. The Rust merge follows the
measured exposure structure and noise weighting.

## Convert and inspect

```sh
x3f_extract -dng -no-denoise capture.X3I
x3f_extract -tiff -color sRGB capture.X3I
x3f_extract -ppm capture.X3I
x3f_extract -meta capture.X3I
x3f_extract -dng -x3i-report capture.X3I
x3f_extract -dng -x3i-ghost-map -o output capture.X3I
x3f_extract -ppm -color none -no-denoise -x3i-frame 0 capture.X3I
x3f_extract -x3i-extract -o frames capture.X3I
```

DNG retains extended highlight range. TIFF and PPM apply frame 0's camera
matrix and the requested output encoding. Values outside the output range
clip at zero and one. These outputs do not run the RGB chroma LUT or
single-frame highlight recovery. `-color none` writes normalized camera BMT
without an RGB matrix or gamma. `-no-crop` retains the full sensor extent;
otherwise TIFF and PPM use the active area. TIFF embeds the matching processed
ICC profile and frame 0's orientation. PPM carries no ICC or orientation tags.
`-ppm-ascii` selects P3 instead of binary P6. `-cineon` applies the existing
log transfer to TIFF. It still clips values above one.

`-color-mode` uses frame 0's unprocessed calibration before merging. DNG embeds
profiles. TIFF applies the selected look. dp Quattro modes are supported;
sd Quattro and sd Quattro H modes return the existing calibration error.
`-spp-pipeline` is Merrill-only and returns an error for X3I. Use
`-x3i-frame <n>` for an embedded JPEG, RAW dump or unprocessed top layer.
X3I inputs run serially in a CLI batch to limit simultaneous raster storage.
X3F inputs retain parallel batch processing.

`-x3i-ghost-map` writes one binary 8-bit PGM per non-reference frame next to
the merge output, for example `capture.X3I.ghost5.pgm`. It supports DNG, TIFF
and PPM merges and honours `-o`. Maps use the full merge resolution before
output cropping. They store the final ghost weight after the 3x3 minimum:
255 means weight one, 0 means fully rejected, and 128 marks invalid or
uncovered pixels. Valid weights that round to 128 use 127. With ghost rejection
disabled, every valid pixel is 255. These maps do not include saturation or
inverse-variance weights.

Library callers can set `MergeOptions::ghost_maps` and read
`MergedScene::ghost_maps`. `convert_file` callers can set the research option
`ProcessOptions::x3i_ghost_maps` and read `ConversionReport::ghost_maps`.
Maps add one byte per pixel per non-reference frame. Disabled maps allocate
no storage and do not change the merged samples. These research controls and
results add public Rust struct fields. Complete struct literals must add the
fields or use `..Default::default()` where available.

## Memory

The merge keeps all of its data in memory and writes no temporary files. Both
passes process bands of 128 output rows, so the scaled samples, variances,
saturation, validity, warped copies and ghost weights of a frame exist only for
one band at a time. For the complete raster, the merge keeps the weighted sums,
the weights and the ghost estimate (33 bytes per pixel) and the frame that it
processes (15 bytes per pixel). Thus the peak memory increases with the merge
raster, which is the full sensor extent with its masked borders, not the
active area.

Peak memory of `x3f_extract -dng -no-denoise` with global alignment, measured
with `/usr/bin/time -l` on macOS aarch64:

| Camera | Merge raster | Peak RSS | Peak live allocations |
| --- | --- | ---: | ---: |
| dp0 Quattro | 5888 x 3672 (21.6 MP) | 2.0 GB | 1.40 GB |
| sd Quattro H | 6656 x 4480 (29.8 MP) | 2.7 GB | 1.88 GB |

Peak RSS changes by about 0.1 GB between runs. Local alignment
(`X3F_X3I_ALIGN=local`) measured 2.0 GB and 2.4 GB. The peak occurs while the
producer prepares a frame for the accumulation pass. Peak live allocations is
the largest total of allocations of 64 KiB or more at one time, counted in an
instrumented build. Peak RSS is larger because the macOS allocator keeps freed
large blocks resident for later allocations.

For other sizes, plan for about 95 bytes per pixel of the merge raster, for
example 2.9 GB for 30 MP and 3.8 GB for 40 MP. About 65 bytes per pixel are
live allocations.

The merge keeps fallback candidates only for the pixels that the reference
frame does not cover. These are 2.6 percent of the pixels on the dp0 sample
and 0.4 percent on the sd Quattro H sample. When most of the reference frame
is clipped or invalid, the merge keeps them for all pixels. This is the worst
case: it adds about 26 bytes per pixel. With this layout, the dp0 sample
measured 2.5 GB and the sd Quattro H sample 3.5 GB.

These options add memory (measured on the dp0 sample):

| Option | Peak RSS |
| --- | ---: |
| Default | 2.0 GB |
| `-sgain` (spatial gain, off by default on Quattro) | 3.5 GB |
| `-color-mode Vivid` | 2.1 GB |
| `X3F_X3I_DENOISE_SCALE=0.3` with `-denoise-backend cpu` | 2.4 GB |

Spatial gain keeps 12 bytes of gain per pixel with each frame that the
producer prepares. `-x3i-ghost-map`
adds one byte per pixel for each non-reference frame (6 bytes per pixel for
seven frames). `-tiff` measured 2.0 GB on the dp0 sample, and
`prepare_scene_linear` measured 2.8 GB on the sd Quattro H sample.

Each merge that runs at the same time needs its own memory. The CLI runs the
X3I inputs of a batch one after the other. A library caller that runs merges
in parallel needs the sum of their peaks.

## Middle exposure only

The middle exposure is the frame with the median `CaptureShutter`. With an
even frame count, it is the shorter of the two middle exposures. Equal
shutters keep file order. `X3iContainer::middle_exposure_frame` returns its
index. It is frame 0 on both sample captures.

```sh
x3f_extract -dng -x3i-middle capture.X3I
x3f_extract -jpg -x3i-middle capture.X3I
```

`-x3i-middle` is `-x3i-frame <n>` with that index, so the output has the same
bytes and the same `_<n>` file name. It cannot combine with `-x3i-frame`,
`-x3i-extract`, `-x3i-report` or `-x3i-ghost-map`. `-meta` dumps the selected frame. Like
`-x3i-frame`, it runs serially in a CLI batch.

In the library, set `x3i_source: X3iSource::MiddleExposure` in
`ProcessOptions` or `SceneLinearOptions`. The default is `X3iSource::Merge`.
X3F input ignores the field.

- `convert_file` converts the frame as a single X3F file. This includes
  `OutputFormat::Jpeg` (the frame's embedded JPEG), color modes, the Merrill
  pipeline and highlight recovery. It adds the report line
  `X3I middle exposure: frame <n> shutter_us=<us>`.
- `prepare_scene_linear` returns the frame's own scene. It honors
  `highlight_recovery` and denoise as for an X3F. It has no merged range.
- `read_as_shot_crop` always reads frame 0.

## Editor source

`prepare_scene_linear` detects `FOVi` and returns the merged active area as
extended-linear sRGB. It retains finite negative RGB and values above one
when post-merge denoise is off. Frame 0 supplies the RGB matrix, white balance,
orientation and optional Color Mode calibration. The calibration is read before
preparation consumes its RAW buffer. The renderer applies orientation once.
`read_as_shot_crop` uses frame 0's camera framing.

`convert_file` accepts `OutputFormat::Tiff`, `Ppm` and `PpmAscii` for these
outputs. Callers with exhaustive matches on `OutputFormat` must add the PPM
variants.

The API emits `Read` once and `Decode` and `Process` once per frame.
`convert_file` also emits `Write`. Callers own cancellation. A failed or
cancelled file conversion removes its reserved output and preserves any
existing destination. Scene preparation writes no output. X3I support is
currently in the native Rust API and CLI; the C ABI and WASM have no X3I entry.

## Experimental denoise

Each frame is prepared with denoise off. Post-merge denoise is also off unless
`X3F_X3I_DENOISE_SCALE` is set above zero. The CLI's default denoise intensity
alone does not enable it. To evaluate the research strength:

```sh
X3F_X3I_DENOISE_SCALE=0.3 x3f_extract -dng -denoise-backend cpu capture.X3I
```

The selected NLM or BM3D runs once on the encoded merged camera raster.
The effective strength is override times intensity/10, divided by the shared
DNG encoding scale `S`. The bridge compresses all channels by four to keep
the Quattro YUV transform in range. It adds the filtered residual to the
original codes in multiples of four. This preserves full-range highlights,
but deep shadows can show bands. At `S=11.4`, one encoded code is about
0.00017 reference units and can be close to one noise standard deviation.
Denoised scene preparation uses this unsigned quantized raster and loses
negative camera samples. `-no-denoise` skips this pass and keeps the previous
X3I DNG bytes. Keep the override unset for normal use.

The research strength was 0.3. The dark dp0 patch was 64x64 pixels at
(5120,2240). Its B/M code noise increased from 1.095/1.241 to 1.771/1.861.
The sd Quattro H patch at (4928,3904) increased it from
1.662/1.524 to 3.041/3.172. T was unchanged with default luma denoise off.
These are detrended population standard deviations at S=11.3862 and S=2.7893.
All residual changes were multiples of four. Adobe DNG SDK 1.7.1 renders
show stronger chroma variation in the denoised patches. These measurements
use the phase 1 merge and do not establish a useful default strength.

## Phase 2 statistics

The statistics pass aligns each frame to frame 0. A skipped alignment keeps
that frame at its original position and reports possible misregistration.
Global alignment uses one translation or bounded similarity transform.
`X3F_X3I_ALIGN=local` adds a smooth displacement field from 256-pixel tiles
at half resolution. The estimator rejects weak peaks and shifts that disagree
with nearby tiles or the global transform. It fills missing tiles by neighbour
regularisation. Bilinear interpolation evaluates the field between tile centres;
Catmull-Rom still samples the image. All 16 source taps must be valid, and
their maximum saturation value remains the roll-off input.
The local field has a bounded gradient to prevent folds. A weak local estimate
keeps the global transform and reports the reason. Global remains the default
because local does not improve every frame on both tested samples.
Reports keep the global `retained_tiles` and `max_tile_residual` fields and add
local tile counts, field range, fit residual and post-warp global residual.
The weak-alignment rule uses the post-warp retained-tile count and residual
when a local field is applied. A missing post-warp estimate is weak.
Frames without a local field use the original global estimate.
`applied_residual` is the largest trusted post-warp shift on a grid offset
by half a tile step from the fitting grid. It is absent with fewer than six
trusted tiles. This maximum includes subject motion, not only camera motion.
The local search covers eight full-resolution pixels around
the global transform; the global search still covers 64 pixels.
The field uses tile-sized storage. The merge warps one band of output rows at
a time, from the source rows that the taps of that band read.
A scalar exposure fit uses the median of clean, supported bin medians.
An affine fit supplies a black-offset diagnostic only. Near-black bias is
nonlinear, so using the affine slope would bias the scale. Short exposures
use reference levels from 5 to 60 percent of the ceiling. Long exposures use 5 to 25 percent and must stay below 40 percent
of their own ceiling. Frame T must exceed ten read sigmas. A bin needs at
least 1000 pairs or 0.1 percent of pixels. A clean bin also needs a mean frame
signal above thirty read sigmas, with reference samples below the roll-off
knee. At least two clean bins must qualify. A trend above 2 percent across
those bins, or a scale change above 0.3 EV, keeps the nominal shutter ratio.

Before the scalar, the statistics pass fits an exposure field for each frame:
`scale = ratio * (a + b*u + c*v)`, with `u` and `v` from -1 to 1 across the
frame's own width and height. B and M share one field; T has its own. The fit
uses 8x8 blocks of raw pairs, paired through the alignment transform. A block
needs every sample below the roll-off knee in both frames, both raw means above
ten read sigmas and above 0.02 of white, a reference mean within 3 percent of
all eight neighbours, and a ratio within 0.3 EV of the shutter ratio. The ratio
uses raw samples times their spatial gain, which are the units of the samples
that the field multiplies. Thus a known gain map, also when the frames are
shifted, does not become an exposure gradient. Tiles of about 14 per shorter
side (256 pixels on Quattro frames) take the median of at least 20 blocks. The
seed is the least median of squares plane through sampled tile triples. Tiles
more than 2.5 robust sigmas (at least 1.25 percent, because the plane does not
model the curvature of a real field) from the seed are dropped, and a weighted
least-squares fit of the remaining tiles gives the field. The refit keeps a
tile by its distance from the plane only, so a kept tile is not proven static:
coherent motion that changes the ratio by less than the threshold stays in the
fit and can bend it. Thus the field has a damage bound, not a motion test. Per
group, the block-weighted rms of the relative difference between the refit and
the seed over the kept tiles must stay within `field_shift_limit` (0.45
percent). The field applies when each group keeps at least 30 tiles in three or
more quadrants, with a residual rms below 1.5 percent, the damage bound,
gradient terms below 0.1 and every corner within 0.3 EV. A small tile rms does
not bound a response error across levels, so the field also needs a flat T
level trend. The trend uses the static blocks, which are the flat T blocks of
the tiles that both the seed and the refit keep, with the field applied, in the
level bins and clean rules of the scalar fit. The blocks of a bin must cover
the pixel-pair minimum of 1000 pixels or 0.1 percent of the pixels. The trend
must stay within the 2 percent trend limit. A response error far from the plane
looks like motion to the seed, which then removes most of its level range from
the static blocks. Thus each level bin that is clean for all flat T blocks of
the field tiles must also be clean for the static blocks, and the static blocks
must hold at least half of its blocks. A frame with fewer than two clean static
bins has no measured trend, so its field has no response check. Pixel pairs do
not gate the field, because residual misregistration increases their trend.
Otherwise the frame uses the clean-bin scalar, then the shutter ratio, and a
warning gives the reason. The applied scale enters the samples, the variance
(squared), the +1 EV noise fit, the effective-variance calibration and the
ghost residuals. `-x3i-report` prints the source, the scalar and the field
coefficients of each frame, the refit-to-seed rms, the flat-block trend, the
lost bins, the static fraction of each clean bin and the pixel-pair trend with
the field applied.

Known limits of the field gates: coherent motion that bends the refit by less
than the damage bound still passes. In the 512-pixel test scene, a 384-pixel
square that moves by 128 pixels passes up to 0.9 percent contrast, and then
makes a static corner up to 0.46 percent too dark. A large loss of static
blocks in a level bin also rejects correct fields: in the same scene, a 3
percent square gives a correct field, but the merge uses the shutter ratio.

Raw reference/+1 EV frame pairs provide separate B, M and T shot-noise fits
from flat 8x8 blocks. Raw pairs use the image warp's inverse transform and
the nearest source sample. Half-resolution pairs use the centres of their
2x2 blocks. Each fit needs 50 blocks; otherwise its default or
explicit override remains. Effective variance fixes `c_ref=1` and calibrates
separate T and B/M corrections from at least 50 flat blocks above ten read
sigmas in both frames. Corrections stay in `[0.5,4]` by default. A frame is
weak when its alignment was skipped, retained at most four tiles, or has a
maximum retained-tile residual above 1.5 pixels. With an applied local field,
these limits use the post-warp global estimate. A missing estimate or a
post-warp median shift above 1.5 pixels also marks local alignment weak.
Disabled alignment keeps the ordinary threshold; it does not mark frames weak.
A reference model term more than eight times the frame term marks the calibration
as weak. Pixels below ten sigmas use the uncorrected model. This overestimates
variance after the intermediate zero clamp and conservatively under-rejects dark noise.

The accumulation pass compares each frame with the reference estimate.
Ghost weights are `exp(-max_channel(z²)/sigma²)`, followed by a 3x3 minimum.
The default sigma is 3. The weak threshold defaults to the ordinary threshold.
Weak frames use the smaller of the ordinary and weak sigma controls.
Reported ghost fractions count valid weights below 0.1. At the default
threshold, this corresponds to a channel residual above 4.55 sigma before
the minimum.
Summary and alignment rows remain on stderr. Ratio and ghost bins are
written only with `-x3i-report`.

Every frame is prepared once in each pass, for `2n` merge-loader calls.
Decode events occur in statistics and Process events in accumulation.
Statistics rasters are released before accumulation. See [Memory](#memory)
for the measured peak memory, and the
[pipeline description](./pipeline.md#x3i-merge-stage) for the model and fallbacks.

dp0 exposure field: the dp0 frames read 2.5 to 9 percent off their shutter
ratios, and the error changes smoothly across the sensor. On the 1/1024 s frame
it goes from 0.913 of nominal at the top left to about 1.04 at the bottom
right. The error is the same in B and M and flat across signal level, so it is
an exposure difference, not a response curve. The CAMF raw LUTs
(`LUT_HI_RR_x1`, `LUT_LOW_RR_x1`) apply only to the dp Quattro highlight-pixel
grid, and the sd Quattro engine does not apply its `LUT_RR_x*` tables. With the
field, flat static regions agree with the reference within 1 percent, against
up to 10.7 percent with shutter ratios. In a hold-out test, a field fitted
without two static regions predicted them within 1.1 percent. All six frames
use their fields. With the field applied, the flat-block T trend is 0.18
percent or less on frames 2 to 4. On frame 1, every level bin is below thirty
read sigmas, and frames 5 and 6 have only one clean bin, so no trend is
measured there and these fields have no response check. The refit differs from
the seed by at most 0.29 percent rms on dp0 (frame 1) and 0.11 percent on H,
and every clean bin keeps at least 98 percent of its blocks static. The
pixel-pair trend with the field is 3.6 and 11.5 percent on frames 2 and 3.
Residual misregistration at texture causes it, because flat blocks, which a
sub-pixel misregistration cannot bias, do not show it. The sd Quattro H fields
have centre values within 0.4 percent of nominal and gradient terms below 0.6
percent. Its two shortest frames keep the clean-bin scalar, because too few
flat blocks reach 0.02 of white. Near black, a per-frame black error of about
one raw count remains; the 0.02 floor keeps it out of the fit, but the merge
does not correct it.

## Research overrides

| Variable | Effect |
| --- | --- |
| `X3F_X3I_NOISE=sigma,k,q` | Replace the read, linear and quadratic noise terms. An empty sigma keeps per-frame shield estimates. |
| `X3F_X3I_MAX_S=value` | Cap the shared encoding scale. Unset keeps all finite headroom. Report clipped samples. |
| `X3F_X3I_SATURATION=b,m,t` | Override the measured physical ceiling of each plane in normalized raw-range units. |
| `X3F_X3I_ROLLOFF=knee,width,margin` | Set knee, width and validity margin relative to the ceiling. Defaults: `0.75,0.02,0.15`. |
| `X3F_X3I_NO_ALIGN=1` | Disable frame alignment. Merge frames at their original positions. A misregistered frame can fail the exposure field gates and use the fallback. |
| `X3F_X3I_ALIGN=global\|local` | Select the global transform or a smooth local correction. Default: `global`. `NO_ALIGN` takes precedence. |
| `X3F_X3I_FIXED_RATIOS=1` | Keep nominal shutter ratios. Report fitted ratios without applying them. |
| `X3F_X3I_RATIO_FIELD=0` | Skip the exposure field. Use the clean-bin scalar, then the shutter ratio. |
| `X3F_X3I_NO_GHOST=1` | Disable ghost rejection and effective-variance calibration. |
| `X3F_X3I_GHOST_SIGMA=value` | Set the positive finite ghost threshold in sigma. Default: `3`. |
| `X3F_X3I_WEAK_GHOST_SIGMA=value` | Set the positive finite weak-alignment threshold in sigma. Default: GHOST_SIGMA; bounded by GHOST_SIGMA. |
| `X3F_X3I_C_MAX=value` | Set the effective-variance upper clamp. Default: `4`; must be finite and at least C_MIN. Caps of one or more have no effect on the two measured samples. |
| `X3F_X3I_C_MIN=value` | Set the effective-variance calibration floor. Default: `0.5`; range: above zero through one. |
| `X3F_X3I_RATIO_TREND=value` | Set the maximum relative trend across clean supported ratio bins. Default: `0.02` (2 percent). Larger trends keep the nominal ratio. |
| `X3F_X3I_DENOISE_SCALE=value` | Enable experimental post-merge denoise when positive. Unset or zero disables it. |

See the [merge stage](./pipeline.md#x3i-merge-stage) for weighting, validity,
fallbacks and the DNG scale. See the [corpus checks](./contributing.md#x3i-corpus-checks)
for parity and reader validation. The current producer still expands integer
Quattro planes before merging. A merge before expansion, iterative ghost pass,
float DNG and tiled accumulators remain research work.
