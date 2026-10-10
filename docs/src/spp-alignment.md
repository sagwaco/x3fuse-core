# Merrill rendering alignment

The reference is the ARM64 Merrill engine shipped in SIGMA Photo Pro 6.9.0.0,
not the legacy x3f C converter. The independently maintained evidence is in
[x3f-docs](https://github.com/sagwaco/x3f-docs/tree/main/spp-rgb-trace): decompiler
records, original instructions, reconstructed kernels, and native differential
tests. The library SHA-256 is
`037419dc04bb1afe86dddad7a14888a1b36d6c00a0231ec50a6974a9963ce5e6`.

The implementation now uses the recovered calibration and output stages for
ordinary Merrill sRGB, Adobe RGB and ProPhoto RGB rendering. It remains a
partial pipeline reconstruction. DNG, scene-linear, Cineon and Quattro rendering
retain their existing processing. DNG's embedded preview retains its existing
calibration too.

| Stage | Previous behavior | Verified Sigma behavior and implementation |
| --- | --- | --- |
| Black level | Mean of several shielded regions | Merrill RGB uses the channel-wise 1% trimmed signed mean of `DarkShieldTop`. CAMF right/bottom are inclusive: native `GetRectFromCamfile` adds one before its half-open loops. |
| WB calibration | Sensor, temperature, aperture factors; no temperature validation | Render gain uses aperture, validated temperature, then sensor factors. Any temperature component outside `[0.5,1.5]`, or nonfinite, resets the entire temperature vector to ones. |
| ISO | Double-precision CaptureISO/SensorISO | Float32 division and native zero fallbacks; nonzero `ISOrange` remaps exact 800/1600/3200/6400 to 666/1200/2133/4266. |
| Camera color | WB matrix through XYZ | Selected WB correction, ISO, then right-multiplied `ColorModeCompensations`; header color mode supplies the CLI's as-shot policy. Sigma's editable adjustment enum is a separate input. |
| Working units | Per-channel saturation ratios | Carry original `maxRaw-black` across preprocessing; reconstruct gained counts using `WB/0.55` for SD1 Merrill and `2*WB` for DP Merrill, followed by calibration. Existing intermediate quantization still limits equivalence. |
| ColorDQ | Absent | Native signed LUT construction and two-pass chroma correction, after camera color. Strength is `ColorDQCamRGB * min(noise74, ISOgain)`; the verified corpus has constant ISO/slider columns, so no estimated noise level is guessed. |
| Tone | No Sigma tone curve | `TCColorModeSettings`, mode contrast, ImageDepth saturation, logistic/Bézier curve and inverse-sRGB normalization. `X3F_EV` changes tone exposure. Other creative controls are neutral. |
| Spatial gain | Lens-position interpolation and double bilinear lookup | Native DP aperture/focus and SD1 chief-ray-angle table construction, float32 vertical interpolation over `y/(height-1)`, and incremental double horizontal interpolation over `width/(columns-1)`. Native skip bit 4 and the caller’s spatial-gain option disable it. Application remains at the existing conversion stage. |
| Output color | One combined floating-point matrix before gamma; assumed sRGB source | Camera color precedes tone. EXIF ColorSpace 1 selects the sRGB source; other present values select Adobe. Native target conversion right-multiplies the Adobe matrix inverse for that source; matching source/target skips conversion. Q12 signed rounding and integer arithmetic follow. |
| Gamma | 1024 interpolated floating-point samples | Native 65,536-entry rounded integer LUT sampled at `i/65536`; sRGB transfer or pure 2.2/1.8 power. |
| TIFF interpretation | Ordinary exports lacked ICC and orientation | Embed a profile matching the actual transfer and preserve camera orientation in the TIFF tag. Stored pixels remain in sensor orientation; Sigma physically rotates its export. |

`crates/x3f-sys/src/merrill.rs` contains the scalar output kernels;
`process.rs` supplies CAMF calibration and count-domain inputs. The shared
`jpeg_exif.rs` reader supplies the embedded EXIF F-number and source color space;
DNG metadata uses the same parser. The processing
code preserves per-conversion cancellation and does not load the Sigma library
at runtime. Native execution is used only by the external verification harness.

Spatial aperture follows the managed request: embedded EXIF FNumber takes
precedence, with PROP APERTURE used only when that rational is absent or has
a zero denominator. Decimal PROP values use a float32 product by 100000 and
ties-to-even integer rounding. Native processing divides the resulting unsigned
rational in double precision, then converts to float32. Direct PROP values differ
from CAMF CaptureAperture on 36 of the 53 files; EXIF FNumber agrees with the
CAMF value at float32 precision on all 53. Missing source ColorSpace selects
legacy rendering because the managed missing-tag default is not fully verified.
The parser currently accepts a single SHORT ColorSpace value, as present in the
corpus; other native tag forms also conservatively select legacy rendering.

The native setup is validated before preprocessing changes any samples. Unsupported
calibration or tone settings produce a warning and select the existing renderer;
cancellation remains an error. This also preserves `X3F_EV` values outside the
native tone implementation's resource bound. The standalone C `convert_data`
entry point retains its existing normalized-input contract.

For X3F 2.3 and 3.0, Sigma selects process 2.0.x for SD1/SD1 Merrill and
2.1.x for DP Merrill; `GatherParams` divides WB by 0.55 for the former and
doubles it for the latter. The shipped Mono AOT string references and process
constants were decoded to verify the model mapping. The new RGB path accepts
the four explicitly named Merrill models at those file versions. X3F 4+ stores
a process version separately and is not selected from model alone. Folding the
count multiplier into the current matrix also differs from native floating-point
operation order; this is calibrated count alignment, not exact preprocessing.

The source's signed16 output conversion alone is not a valid integration
boundary. Passing negative values from the old color pipeline into it produced
bright colored shadow artifacts. `ApplyColorResidual` clamps finite results to
`[0,32766]` before the later geometry/output steps; the integration uses those
bounds with neutral residual gain. The native tile residual calculation itself
is still missing. The pure output kernel retains its exact signed-wrap behavior
and has a separate regression test for the conditioned integration input.

## Verification

Direct calls into the supplied library verified twelve complete tone LUTs,
five complete gamma LUTs, and 774 RGB48 output pixels, including signed indices
and integer overflow witnesses. The Rust tests pin the native results. These
kernel checks are distinct from whole-image comparisons.

ColorDQ has separate native checks for 18 complete signed LUTs and 6,168
float RGB outputs, covering signed indices, half ties, and integer overflow.
All 53 inventoried Merrill files have ISO noise column 7 equal to four and
slider column 7 equal to one. Three files select Monochrome and retain legacy
processing; the ordinary color reconstruction does not cover that branch.
All 53 pass the full-resolution RGB16 render check with the expected ICC and
orientation metadata, unchanged source hashes, and no unexpected setup fallback.
This is a conversion check; the two retained Sigma exports measure RGB error.

Native spatial-table validation covers all 53 files: 2,486,389 float32 entries
match the supplied library exactly. A separate image-overload ApplyGains oracle
matches 7,947 channel samples and row padding across eleven cases. Production
uses its spatial interpolation, while float black/gain arithmetic remains future
work. Rust tests retain native table hashes and three spatial pixel-row hashes.

The two real Sigma references are `_P2M0358.X3F` (DP2 Merrill) and
`sigma_sd1_merrill_15.x3f` (SD1 Merrill), exported in RAW adjustment mode,
original size, ProPhoto RGB, TIFF16. Each comparison covers 14,751,744 oriented
pixels, without resizing, color fitting, or histogram matching. Both candidates
use the same source, profile and size. The CPU column enables x3fuse’s default
NLM; it does not claim the same denoise algorithm or strength as Sigma.

| Reference / renderer | MAE | PSNR |
| --- | --- | --- |
| DP2, previous renderer, no denoise | 26.70% | 10.76 dB |
| DP2, recovered renderer, no denoise | 1.15% | 35.43 dB |
| DP2, recovered renderer, CPU NLM | 0.46% | 40.68 dB |
| SD1, previous renderer, no denoise | 13.65% | 15.12 dB |
| SD1, recovered renderer, no denoise | 5.21% | 22.38 dB |
| SD1, recovered renderer, CPU NLM | 1.07% | 36.42 dB |

These results identify candidate binary SHA-256
`f854e77821bb05c6a2c355f9d73284620bbda52046dbe9404566c003d827bba0`.
The earlier DP2 34.15 dB checkpoint assumed an sRGB source despite its Adobe
EXIF tag. Its earlier 34.90 dB intermediate also used fallback tone parameters
because CAMF dimensions were reversed. Both are superseded; source agreement,
rather than the best score from an intermediate configuration, governs changes.

SD1’s visible Sigma noise controls were at the middle of five detents, with
Highlight Control 1.0. Their native numeric request values remain unknown.
CPU NLM removes most shadow speckle, but one severe isolated wheel-pixel defect
remains. At full sensor coordinate (4171,2349), raw C0 is 203 versus neighboring
values 12–23; that site is absent from `BadPixelsF20`. All three packed
`BadClustersF20` expanded regions also exclude it, verified from native geometry
and a direct library execution with synthetic pixels. Native raster despeckling
remains a candidate for its correction; actual Sigma stage pixels are unobserved.
These are two measured scenes, not a corpus-wide color-accuracy claim.

All three DNG reference hashes remain unchanged. Quattro TIFF is byte-identical
to the preceding alignment checkpoint; its pixel bytes also match the original
baseline. The updated Merrill TIFF pins are in the contributor guide.

The release workspace suite passes 296 tests, with six intentionally ignored,
using the local corpus. Strict workspace Clippy, formatting,
and the documentation build. The portable WASM/Wasmtime run passes 24 Merrill
kernel tests; its external corpus is not mounted. Independent Opus review
verified source-color dispatch, matrix composition, shared EXIF parsing and
error propagation. Missing/invalid JPEG EXIF falls back before rendering, and
spatial-row tests check the exact output length.

## Remaining differences

- The front end still stores normalized uint16 intermediates. Sigma uses float
  gained counts; reversing normalization cannot restore lost precision.
- Highlight restoration still uses the existing normalized LUT/gate model,
  after denoise. Sigma restores camera-marked C2 and blends toward original C0
  before denoise, using CAMF saturation maps and different donor rules.
- Rust NLM is not Sigma's tiled bilateral/residual pipeline. Native residual
  mixing, sharpening, hue/fringe filters and nonzero retinex adjustments remain
  reconstruction work. Do not describe the current output as pixel-identical.
- MED/LOW black-rectangle overrides and the separate monochrome/sepia branches
  are not covered by the full-resolution ordinary-color reference.
- ColorDQ setups requiring an unknown native EstimatedISO or an unresolved
  slider input fall back to legacy rendering. Missing or unreadable ISO tables
  now use the verified ordinary HI/MED/LOW fallback column (four at every row).
  Readable malformed tables and the `SkipStepFlags` denoise-bypass branch still
  fall back rather than assuming ordinary range conditioning applies.
- The exact Bézier implementation currently accepts effective tone saturation
  in `[1,16384]` to bound its cubic work. For 12-bit Merrill this includes the
  ordinary -2 EV setting; larger extents select legacy rendering with a warning.
  This is an implementation resource limit, not a restriction discovered in
  Sigma's kernel.

The independent B/M chroma-table validity correction in `highlight.rs` fixes a
separate helper bug. It is not evidence that the normalized highlight algorithm
matches Sigma's native restoration.
