# Camera Color Modes for TIFF and DNG

Status: implementation in progress, 2026-10-13. RGBTables failed the Lightroom
rendering tests. The C3-C5 classic-profile controls passed, including JPEG
export checks for cast, exposure and hue. Full mode profiles are implemented
and pass the combined Lightroom 9.5.1 Local check on two files. Source review
and workspace checks pass. Full SPP image and WB/ISO agreement remain open;
this feature is an unreleased approximation.

## Outcome and scope

Expose camera Color Modes through `ProcessOptions.color_mode` and the legacy
CLI flag `-color-mode`. TIFF applies the selected mode to RGB pixels. DNG keeps
mode-independent raw samples and embeds selectable camera profiles. An omitted
option preserves existing output byte for byte.

| Camera family | Ordinary modes |
| --- | --- |
| Supported full-resolution Merrill | Standard, Vivid, Neutral, Portrait, Landscape, FCBlue |
| Supported dp Quattro | The Merrill names plus SunsetRed, ForestGreen, FCYellow, Cinema, TealAndOrange |

Availability depends on valid camera calibration. sd Quattro and sd Quattro H
remain unsupported because their native sensor calibration and resolution
paths have not been ported. An explicit unsupported request returns an error.
Do not substitute an identity look for missing required calibration.

Normal Quattro strength is effect 0, alpha 1. Initial mode rendering uses Sigma
HP correction OFF. This is an x3fuse baseline, not a claim about SPP's default
or saved setting. It is separate from x3fuse highlight recovery. Strength, HP
controls, monochrome, sepia, HDR, SFD and DRX are outside this change. Existing
synthetic monochrome DNG profiles retain their names. Classic-profile exports
must preserve their as-shot rendering and document changed WB semantics.

## API and CLI contract

`ColorMode` is shared by sys and core. It contains `AsShot` and the eleven named
modes. `Reader::available_color_modes()` uses the same resolver as export and
loads CAMF and optional PROP without decoding RAW or JPEG pixels.

| Selection | Behavior |
| --- | --- |
| `None` | Keep the existing renderer and DNG profile set. |
| `AsShot` | Resolve the recorded camera mode at normal strength. |
| Named mode | Render TIFF with that mode; select it as the DNG primary profile. |
| Unsupported mode, sensor or calibration | Return an actionable error. |

Merrill records its mode in the X3F header. Quattro records PictureMode in the
embedded JPEG's Sigma MakerNote tag 0x003d. The bounded EXIF parser accepts both
byte orders and variable ASCII counts, including 12 and 16. Named capability
queries do not require that JPEG metadata. Unknown or missing As Shot values
fall back to Standard and report a warning. Recorded Monochrome and Sepia
return an unsupported-mode error. Explicit unknown CLI names are errors.

```sh
x3f_extract -tiff -color sRGB -color-mode Vivid input.X3F
x3f_extract -dng -color-mode as-shot input.X3F
x3f_extract -dng -color-mode TealAndOrange input.X3F
```

`-color` remains the output encoding. Accept `-color-mode` only for color TIFF
and DNG exports. Reject Cineon, unprocessed, Qtop, PPM, histogram and embedded
JPEG output with this option. The reusable `get_image` API also renders the
selected mode. Its legacy `get_preview` method does not apply this option;
the DNG writer builds its selected-mode preview separately.

The option is per conversion and does not change the source X3F header. Rust
struct literals need `color_mode` or `..Default::default()`. C ABI layouts and
legacy defaults stay unchanged. No new C selection API is part of this work.

## Native evidence and corrections

The original SPP 6.9.0.0 ARM64 trace is retained in
`../_local/spp-profile-trace-20261012/` and in the published
[x3f-docs archive](https://github.com/sagwaco/x3f-docs/tree/main/spp-rgb-trace/followup-color-modes).
New implementation evidence is in `../_local/color-modes-native-tone/`.
Retain the original archive and label later corrections rather than rewriting
its evidence records.

The implementation must follow these corrected rules:

- Keep the canonical mode, enum ID and Merrill contrast-list ordinal distinct.
  Merrill selects its tone row by the contrast-list ordinal. Preserve its
  FCBlue matrix and tone compatibility rules.
- Native matrix order is ISO * CC * C_mode * diag(WB). Creative C is
  right-multiplied into physical correction CC. Preserve the traced Standard
  absent-list and zero-sentinel skip only where native evidence allows it.
- Merrill tone uses domain gamma 1/2.2. Quattro uses 1/2.2222. Final sRGB
  normalization uses exponent 2.4 and offset 0.055 as a separate step.
  The shared tone LUT is already linear before the output transfer.
- For a valid Quattro TCColorModeSettings table, keep the enum only when
  row_count > enum. Otherwise reset the index to zero before the built-in
  switch. Cinema enum 12 with seven rows therefore uses the default
  Landscape-shaped curve. Test equality and the first valid row count too.
- The ordinary Quattro switch overwrites every tone-row field, including
  scale = 1. Only the CAMF row count affects that switch. Preserve the
  TealAndOrange matrix override under the traced list condition.
- Quattro CreateBasicCurve at 0x118c80 is followed by quarter-resolution
  blending, every-50th-point Bezier smoothing in FP32, fourfold interpolation,
  black/white normalization and inverse sRGB transfer. Do not substitute the
  Merrill generator. Eighteen complete native LUT oracle cases pass.
- The optional Quattro k2 -= 0.7f branch is controlled by Sigma HP correction.
  It stays off for the initial baseline. Existing highlight recovery is not
  that switch.

Shared calibration maps DNG normalized samples through legacy shield black
levels, Merrill trimmed-black offsets, count spans, WB gains and ISO gain.
The Merrill ColorDQ stage and native count rounding also matter. A bare
matrix conjugation is not sufficient to derive a DNG look.

## TIFF implementation

The shared resolver supplies both the ordinary Merrill renderer and the
explicit Merrill SPP pipeline. The dp Quattro ordinary color path is a port
of native calibration and tone stages, not a claim of an existing complete
Quattro SPP reconstruction. Apply the mode before the output color encoding.

Keep crop, orientation, WB, ICC, cancellation, denoise and error handling.
The TIFF serializer needs no camera-mode logic. Use physical, mode-independent
calibration for the new Quattro recovery stage. The existing Merrill TIFF
recovery behavior remains mode-dependent and is a separate comparison limit.

Native kernels, parameter branches, real As Shot resolution and all three
no-option TIFF pins have passed the delegated checks. Whole-image SPP color
agreement across modes, WB and ISO remains a separate release gate.

## DNG design and application evidence

DNG raw samples, levels, crop and recovery scale remain independent of the
selected mode. The selected mode becomes the primary profile and
AsShotProfileName; other supported modes use ExtraCameraProfiles. Names use
the `x3fuse ` prefix. No-option exports retain the existing bytes.

The replacement design uses shared AsShotWhiteXY and per-profile ColorMatrix
and ForwardMatrix values. These explicit-mode profile matrices include the
creative transform. A PCS-neutral ProfileToneCurve and a ProfileLookTable
approximate the remaining native tone and color operations. The look table
runs before Adobe's hue-preserving tone operation. Its targets therefore use
the inverse of that operation, not a copied native one-dimensional curve.

Use the absolute native-derived matrix basis. A transform relative to the
legacy ForwardMatrix would carry incorrect native chroma into the Merrill
Adobe RGB source-space case. Let N be the native linear count matrix, O the
native Q12 output matrix, n0 the serialized physical neutral, and q one common
Standard neutral-count scale. In D50 XYZ coordinates:

```text
A_mode = XYZ_from_sRGB * O * N_mode * diag(n0) / q
d = inverse(A_mode) * D50
CM_mode = diag(d) * CM_physical
FM_mode = A_mode * diag(d)
AsShotWhiteXY = xy(inverse(CM_physical) * n0)
```

O defines the profile coordinate basis here. The residual target keeps its
native order: counts and affine offset, ColorDQ, bounds and count index,
per-channel tone, then O. Invert the serialized profile transform back to the
same DNG raster before evaluating that target. Preserve count spans, ISO and
WB gains, and restore recovery headroom once. Fit in the post-exposure domain
using the actual serialized BaselineExposure and BaselineExposureOffset.

Adobe rescales and rounds the matrices, normalizes ForwardMatrix rows, rounds
WhiteXY on read, and pins normalized camera white to 0.001..1. Audit these
operations before fitting. The first audit passes all modes on four cameras:
minimum camera-white components are 0.269, 0.274, 0.733 and 0.670, with no pin
clipping and maximum relative matrix error 0.000191. This matrix audit does
not establish full color agreement. Measure tone, look-table interpolation,
near-neutral error, input clipping, and the Merrill post-tone O case separately.

The gray eyedropper can cancel a creative cast. Max-white normalization adds
a WB-dependent scalar. It is constant only where the original and scaled
camera whites have the same maximum channel. A fixed exposure offset cannot
correct every WB setting. The legacy synthetic profiles also need a matrix
correction for as-shot rendering under WhiteXY. That correction changes their
custom-WB semantics. Measure and document both limits. C4 confirms that
Lightroom applies per-profile exposure offsets and retains the darker result
in JPEG export. Its total exposures are +3 and +2; the negative-exposure tone
branch remains outside that check.

Classic profiles use DNGVersion 1.4 and DNGBackwardVersion 1.4. Ordinary mode
extras use IIRC, matching the C4 and C5 controls. Their look tables have
dimensions 72x17x65 and encoding 1. Tone curves have 4097 uniform knots with
exact (0,0) and (1,1) endpoints. Zero-saturation table entries keep value
scale 1. These constraints can leave a measured native endpoint residual.
See the [DNG development notes](../crates/x3f-core/src/output/dng/COLOR-MODES.md)
for current source details and measurements.

The initial experiment used a 32-cubed RGB16 RGBTables payload and identity
ProfileToneCurve. RGBTables was introduced in DNG 1.6 for IFD0; its use in
camera-profile IFDs requires 1.7. Those experimental files set both version
fields to 1.7 under the
[Adobe DNG 1.7.1 specification](https://helpx.adobe.com/content/dam/help/en/camera-raw/digital-negative/jcr_content/root/content/flex/items/position/position-par/download_section_733958301/download-1/DNG_Spec_1_7_1_0.pdf).
The SDK validated them and rendered distinct looks. Lightroom did not:

- The first test on both camera families showed all expected names, editable
  adjustments and persistent selection, but no visual change for the A/B or
  named modes. Evidence: `../_local/color-modes-lightroom-first-test/`.
- C1/C2 showed no linear-light inversion from primary or extra RGBTables.
  A classic tone control worked. C1 JPEG export kept the purple car and black
  tire, so the failure also affects export. Evidence:
  `../_local/color-modes-lightroom-second-test/`.
- C3 passed on one Merrill file. Base and Identity II match, Tone Bright II
  brightens, and Cast Magenta II tints the image and exported JPEG. All three
  extras use IIRC. Evidence: `../_local/color-modes-lightroom-third-test/`.
- C4 passed on the same Merrill source. An identity extra profile with BEO -1
  is darker in Lightroom and JPEG export than the BEO 0 primary. All other
  profile fields match. Both version fields are 1.4. Evidence:
  `../_local/color-modes-lightroom-fourth-test/`.
- C5 passed on the same Merrill source. The 72x17x65 look table changes hue
  and respects encoding 1. The 4097-point tone curve brightens the image.
  The hue change remains in JPEG export. Extras use IIRC and both version
  fields are 1.4. Evidence: `../_local/color-modes-lightroom-fifth-test/`.

- The final combined check passed on dp2 Merrill _SDI8040 and dp2 Quattro
  _SDI8284. Both default to Standard and show all six or eleven names. The
  tested modes change visibly; JPEG export retains the look, reopen retains
  selection, and WB/exposure remain editable. Evidence:
  `../_local/color-modes-lightroom-full-test/`.

These are Sang's Lightroom cloud 9.5.1 Local observations. They prove the
listed profile application behaviors on those files, not quantitative SPP
matching or support in untested applications. Do not describe the failed
RGBTables route as compatible.

Static Lightroom evidence confirms RGBTables parsing and copying into profile
objects, but does not prove runtime activation. Native tracing is closed
without an identified public activation setting. Do not add guessed flags,
omit neutral tints, bake selected looks into raw samples, or add sidecars.
Adobe input-gamut clipping and spatial processing remain comparison limits
for a global profile even after application compatibility is established.

## Completed implementation checks and remaining release gates

Source review has no remaining scoped objections. The final release workspace
suite passes 406 tests with 15 ignored; DP2M0981 is absent from the local
corpus and is reported as a skip. Formatting, workspace Clippy, browser WASM
and WASI C bindings pass. Prior CPU-only sys and native tone checks pass.

The delegated 31-export suite preserves decoded raw, levels and crop across
modes, compression and recovery settings. All three no-option DNG and TIFF
pins pass without updates. Serial and concurrent TIFF mode outputs match.
Cancellation and unsupported-output checks pass. Eighteen SDK legacy-profile
comparisons differ by at most one RGB8 code at as-shot. The final source
regenerates all four application fixtures byte for byte. Retained evidence:
`../_local/color-modes-classic-validation/` and
`../_local/color-modes-classic-profile/`.

The profile fit is approximate. The sampled in-gamut p99 dE2000 is at most
1.20 across four cameras, but input clipping can cause much larger local
errors. WB edits keep the profile fixed while SPP can change native camera
calibration. The sampled new-profile WB sweep has much larger errors than
as-shot; it is not a promise of SPP WB parity. The DNG development notes
separate those limits from the mode-specific casts and interpolation residuals.

The following release gates remain:

1. Capture SPP references for all modes with matched WB, exposure, ISO and
   creative settings. Define color/tone tolerances before claiming agreement.
2. Measure whole-image, WB/ISO, clipping and spatial-processing limits against
   those references. The current numerical audits are not SPP image captures.
3. Validate Lightroom Classic and standalone Camera Raw when available. The
   cloud Local result is specific to the tested version and two files.

The source is unstaged and unreleased. No branch integration, commit, push or
release action is part of these implementation checks. Keep camera files and
local application images separate from redistributable source tests.
