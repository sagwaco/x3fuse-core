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

## Research overrides

| Variable | Effect |
| --- | --- |
| `X3F_X3I_NOISE=sigma,k,q` | Replace the read, linear and quadratic noise terms. An empty sigma keeps per-frame shield estimates. |
| `X3F_X3I_MAX_S=value` | Cap the shared encoding scale. Unset keeps all finite headroom. Report clipped samples. |
| `X3F_X3I_SATURATION=b,m,t` | Override the measured physical ceiling of each plane in normalized raw-range units. |
| `X3F_X3I_ROLLOFF=knee,width,margin` | Set knee, width and validity margin relative to the ceiling. Defaults: `0.75,0.02,0.15`. |
| `X3F_X3I_NO_ALIGN=1` | Disable frame alignment. Merge frames at their original positions. |
| `X3F_X3I_FIXED_RATIOS=1` | Keep nominal shutter ratios. Report fitted ratios without applying them. |
| `X3F_X3I_NO_GHOST=1` | Disable ghost rejection and effective-variance calibration. |
| `X3F_X3I_C_MIN=value` | Set the effective-variance calibration floor. Default: `0.1`; range: above zero through one. |
| `X3F_X3I_RATIO_TREND=value` | Set the maximum relative trend across supported ratio bins. Default: `0.02` (2 percent). Larger trends keep the nominal ratio. |
| `X3F_X3I_DENOISE_SCALE=value` | Enable experimental post-merge denoise when positive. Unset or zero disables it. |

See the [merge stage](./pipeline.md#x3i-merge-stage) for weighting, validity,
fallbacks and the DNG scale. See the [corpus checks](./contributing.md#x3i-corpus-checks)
for parity and reader validation. The current producer still expands integer
Quattro planes before merging. A merge before expansion, iterative ghost pass,
float DNG and tiled accumulators remain research work.
