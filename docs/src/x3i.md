# X3I merge and outputs

X3I is Sigma's Quattro Super Fine Detail container. The implementation reads
complete X3F frames through bounded file windows and merges them in frame 0
camera units. The [format reference](./format.md#x3i-super-fine-detail-container)
describes the measured container layout. The
[x3f-docs archive](https://github.com/sagwaco/x3f-docs/tree/main/spp-rgb-trace/followup-x3i-sfd)
contains the sample inspection, SPP compose reverse engineering and corrections.
The Rust merge follows the measured exposure structure and noise weighting.
It does not claim byte parity with SPP.

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
  `OutputFormat::Jpeg` (the frame's embedded JPEG), color modes, SPP and
  highlight recovery. It adds the report line
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
`applied_residual` is the largest trusted post-warp shift on a grid offset
by half a tile step from the fitting grid. It is absent with fewer than six
trusted tiles. This maximum includes subject motion, not only camera motion.
The local search covers eight full-resolution pixels around
the global transform; the global search still covers 64 pixels.
The field uses tile-sized storage. `warp_rows` can produce an output row range
from a complete source frame; the merge still allocates full warped frames.
A scalar exposure fit uses the median of clean, supported bin medians.
An affine fit supplies a black-offset diagnostic only. Near-black bias is
nonlinear, so using the affine slope would bias the scale. Short exposures
use reference levels from 5 to 60 percent of the ceiling. Long exposures use 5 to 25 percent and must stay below 40 percent
of their own ceiling. Frame T must exceed ten read sigmas. A bin needs at
least 1000 pairs or 0.1 percent of pixels. A clean bin also needs a mean frame
signal above thirty read sigmas, with reference samples below the roll-off
knee. At least two clean bins must qualify. A trend above 2 percent across
those bins, or a scale change above 0.3 EV, keeps the nominal shutter ratio.

Raw reference/+1 EV frame pairs provide separate B, M and T shot-noise fits
from flat 8x8 blocks. Raw pairs use the image warp's inverse transform and
the nearest source sample. Half-resolution pairs use the centres of their
2x2 blocks. Each fit needs 50 blocks; otherwise its default or
explicit override remains. Effective variance fixes `c_ref=1` and calibrates
separate T and B/M corrections from at least 50 flat blocks above ten read
sigmas in both frames. Corrections stay in `[0.5,4]` by default. A frame is
weak when its alignment was skipped, retained at most four tiles, or has a
maximum retained-tile residual above 1.5 pixels. Disabled alignment keeps the
ordinary threshold; it does not mark frames weak. A reference model term
more than eight times the frame term marks the calibration as weak. Pixels
below ten sigmas use the uncorrected model. This overestimates variance after
the intermediate zero clamp and conservatively under-rejects dark noise.

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
Statistics rasters are released before accumulation. An aligned H conversion
measured 43 seconds and 4.39 GB peak RSS in the final uncontended run. The
1.4 GB target still needs row-band processing. See the
[pipeline description](./pipeline.md#x3i-merge-stage) for the model and fallbacks.

Known dp0 limit: the frames are not mutually linear across the bracket.
On a static region, short frames read 2 to 8 percent high at mid levels,
growing with the exposure difference, and 6 to 10 percent low near the
reference cut. Ghost rejection keeps the reference there, and short frames
carry only the range above the reference cut; a hand-off step of up to about
10 percent is possible on this body. One
hypothesis is that SPP uses a response-linearisation LUT from the CAMF
`OECF_*` entries before composing; this has not been verified, and this
pipeline does not apply such a LUT. Investigate it separately without
changing the merge model here.

## Research overrides

| Variable | Effect |
| --- | --- |
| `X3F_X3I_NOISE=sigma,k,q` | Replace the read, linear and quadratic noise terms. An empty sigma keeps per-frame shield estimates. |
| `X3F_X3I_MAX_S=value` | Cap the shared encoding scale. Unset keeps all finite headroom. Report clipped samples. |
| `X3F_X3I_SATURATION=b,m,t` | Override the measured physical ceiling of each plane in normalized raw-range units. |
| `X3F_X3I_ROLLOFF=knee,width,margin` | Set knee, width and validity margin relative to the ceiling. Defaults: `0.75,0.02,0.15`. |
| `X3F_X3I_NO_ALIGN=1` | Disable frame alignment. Merge frames at their original positions. |
| `X3F_X3I_ALIGN=global\|local` | Select the global transform or a smooth local correction. Default: `global`. `NO_ALIGN` takes precedence. |
| `X3F_X3I_FIXED_RATIOS=1` | Keep nominal shutter ratios. Report fitted ratios without applying them. |
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
