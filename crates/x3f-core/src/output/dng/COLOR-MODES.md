# DNG Color Modes development notes

Explicit Color Modes use classic DNG camera profiles. The as-shot result is
an approximation with measured interpolation, endpoint, shadow and clipping
residuals. It is not SPP whole-image parity. Release gates remain in
[the plan](../../../../../docs/COLOR-MODES-PLAN.md).

## Selection, raw data and preview

`ProcessOptions.color_mode = None` keeps the original writer and preview.
With a requested mode, the writer resolves one list before processing consumes
RAW. The selected mode is the primary profile and `AsShotProfileName`. Other
modes use `ExtraCameraProfiles`. Names match the in-camera labels, such as
`Standard`, `Vivid` and `FOV Classic Blue`, without a prefix. The six legacy
synthetic names stay available.

For DNG only, `AsShot` with a recorded `Monochrome` mode makes the existing
synthetic `Monochrome` profile the primary profile and `AsShotProfileName`.
Merrill reads the header mode; Quattro v4 reads EXIF `PictureMode`. All native
color profiles, including Standard, remain selectable alongside the other
synthetic profiles. Standard supplies the shared white point, color preview
and baseline exposure. The primary Monochrome profile keeps its synthetic
matrices and default tone behavior; it does not inherit Standard's native
tone, LookTable or exposure offset. Camera filter selection is not applied.
TIFF still rejects as-shot Monochrome. Sepia still returns an error.

The writer clears `color_mode` before raw processing. Mode selection changes
profile metadata and preview pixels. Decoded samples, levels, crop, recovery
and the source X3F header stay invariant. The same processed sensor image
supplies raw strips and the preview. The preview averages sensor samples in
each thumbnail block, then uses the selected native renderer. It includes
ColorDQ, native count bounds and rounding, integer tone, Q12 output conversion
and output gamma. It can differ from the raw reader's approximate profile
render. There is no second reconstruction or denoise pass.

Merrill supports six modes. Supported dp Quattro, sd Quattro and sd Quattro H
files support eleven. Missing calibration does not produce an identity look.
Sigma HP correction is OFF for this native baseline. DNG highlight recovery
is a separate setting. The real dp2 Quattro count-12 PictureMode resolves
AsShot to Standard without fallback; the two DNG selections are byte-equal.

## sd Quattro and sd Quattro H

The RAW type and CAMERAID must agree: type 0x10025 with CAMERAID 40 is
sd Quattro, and type 0x10027 with CAMERAID 41 is sd Quattro H. A mismatch is
an error. SPP processes sd Quattro with lib95 (Quattro) and sd Quattro H
with lib92 (Tiger). Static native evidence for these rules:

- Preprocess gives counts
  `min(DIG * (raw - RawOffset) * 100 / SensorISO, 65535)`. DIG is the CAMF
  `DigitalISOGain`. Only lib92 reads it; it is unity for sd Quattro.
  A missing RawOffset is 2048 for sd Quattro and 256 for sd Quattro H. A
  missing or zero SensorISO is 100. The port maps the normalized sample `x` to
  `g * (x * (max - black) + black - RawOffset)`, where `g = DIG * 100 /
  SensorISO` and `black` is the measured black. The small count offset is
  nonzero. DIG does not go into the WB or ISO gains. A gain whose sensor
  range exceeds the 65535 clamp is rejected, because the linear matrix
  cannot express the clamp. The exact boundary is accepted.
- The color path uses the GatherParams WB copy B: TempGainFact * WB gains,
  normalized by green. It does not use copy A, which also contains the
  `_HI`/`_BR_3COL_HI` sensor adjustment. Fluorescent applies the native
  F-number table (sd Quattro) or CAMF FnumberGainTable (sd Quattro H). A
  missing sd Quattro H table is unity. A missing EXIF FNumber or a malformed
  table is an error; the port does not copy any native fallback.
- dp Quattro keeps its pinned convention: minimum-normalized WB, measured
  black as the native black and unit scale. lib95 also subtracts RawOffset
  there. Changing this would change the pinned outputs.
- Tone, per-mode tone rows, contrast compensation and the TealAndOrange
  override use the same operations and constants in lib92 and lib95. The
  comparison normalizes relocated addresses, branch targets, assert paths
  and line numbers. The only extra lib92 tone
  gate (priMakeToneLUT +0x5d8) selects progress reporting, not tone math.

These are static results. The managed C# bodies are stubs, so a managed
pre-pass cannot be excluded. There are no SPP image captures for these
bodies, so runtime SPP parity is not established.

Limits:

- Only the full-resolution (HI) layout of the corpus is validated. No
  LOW or MED file was available. The port has no layout gate.
- All local sd Quattro and H files are ISO 100.
- SDQH5085 records Auto WB gains [0.839935, 1.04999, 1.54372]. Through the
  physical ColorMatrix, its neutral gives negative Z, outside the classic
  profile model. Explicit modes return "invalid classic profile as-shot
  white". There is no clamp or WB substitution. The corpus audits use its
  Sunlight preset. The no-option DNG is not affected.
- Existing x3fuse names for CAMF WB codes 3 and 6 are "Shadow" and
  "Florescent". The CAMF keys are "Shade" and "Fluorescent", so an as-shot
  Shade or Fluorescent file fails the calibration lookup. A Custom WB
  without a custom CC matrix also fails. This affects dp Quattro as well.
- Merged X3I DNGs resolve with frame calibration. The profile fit
  extrapolates values above the native range. This is not validated
  against SPP.

## Native coordinates and the classic matrices

Let `x` be the stored normalized sensor sample and `h` its retained highlight
scale. The shared calibration gives the native pre-tone mapping:

```text
N_m = iso * CC_w * C_m * diag(wb_gain) * diag(normalized_to_counts)
b_m = iso * CC_w * C_m * diag(wb_gain) * count_offset
y_m = N_m * (h*x) + b_m
```

`CC_w` is the CAMF calibration for the selected WB. `C_m` is the native
row-major creative matrix. A missing optional matrix means the traced native
skip. Spans, WB gains, scalar ISO and black offsets have separate roles.
Merrill applies its validated per-pixel ColorDQ after this affine mapping.
Native bounds and tone follow, then output conversion `O`, then output gamma.
Merrill EXIF ColorSpace other than sRGB uses the Adobe-to-sRGB output inverse.
Quattro uses identity for the sRGB target. `O` is quantized to native Q12.

`CM0` is the physical DNG ColorMatrix, including the existing diagonal camera
calibration. `n0` is its serialized, max-normalized as-shot neutral. The writer
computes one image-level white point and an absolute pre-tone basis:

```text
xy0 = xy(inverse(CM0) * n0)
A_m = XYZ_from_sRGB * O * N_m * diag(n0) / q
d_m = inverse(A_m) * D50
CM_m = diag(d_m) * CM0
FM_m = A_m * diag(d_m)
```

`q` is one positive Standard neutral count scale. Its common scale cancels
from normalized camera white and the ForwardMatrix. The basis includes the
native WB/span mismatch. Using a relative look on the old physical FM would
retain a large chroma error on Adobe-source Merrill. `O` in `A_m` defines
coordinates; the residual target still applies `O` after per-channel tone.
It is not moved across the nonlinear native operation.

Every `d_m` must be finite and positive. The writer models stored rational
rounding, SDK ColorMatrix normalization, ForwardMatrix white normalization,
six-decimal WhiteXY and the SDK camera-white pin [0.001,1]. Invalid or
ill-conditioned coordinates fail explicitly. Across four cameras and every
mode, the smallest normalized camera white is 0.269. Relative matrix error
after serialization is at most 0.000191. Standard `d` differs from one by
at most about 0.00033 in this corpus.

The image uses AsShotWhiteXY, not AsShotNeutral. Each profile computes its
camera white from its own CM. CM/FM tags intentionally differ by mode.
Their positive diagonal construction permits a neutral cast despite Adobe's
ForwardMatrix white normalization. The WB eyedropper cancels that cast.
That behavior differs from SPP's persistent creative tint.

## Tone and one LookTable

The ordinary profiles carry one ProfileLookTable, with dimensions 72 hue,
17 saturation and 65 value samples. Saturation varies fastest and value is
outermost. Encoding 1 uses sRGB for the value coordinate and value scale.
All zero-saturation value scales are exactly 1, as the specification requires.
Near-neutral saturation scales are capped at 8 and their affected nodes are
reported. Ordinary profiles carry no RGBTables and no additional HueSatMap.

The curve has 4097 uniform linear input points, `i/4096`. Its first point is
(0,0) and its last is (1,1). All outputs are finite and monotone. The SDK's
4096-interval tone table samples these exact knots with `subSample=false`,
so the test model reads their Y values directly and interpolates them.
Arbitrary spline support is not needed for this fixed grid.

Native FP32 sampling can produce small local decreases near white. The DNG
fit clamps finite samples to [0,1] and raises each decrease to the previous
output before fitting the LookTable. Valid curves keep their exact values;
non-finite samples still fail. On dp0 Quattro `DP0Q0130.X3F`, Cinema had one
decreasing knot at input 2990/4096. The repair raises it by 0.0000006556511.
This approximation keeps all eleven profiles available. Native tone, TIFF
pixels, DNG raw samples and the embedded preview are unchanged.

The fit follows the SDK order:

1. Clamp each sensor channel to the selected profile's camera white.
2. Apply FM/balance, then pin ProPhoto components to [0,1].
3. Apply the exposure ramp, then the LookTable, then Adobe's hue-stabilized
   RGB tone. It maps the minimum and maximum through the curve and
   interpolates the middle channel between them.

The profile curve samples the native result on the PCS-neutral axis. For
each LookTable node, the sampler inverts the serialized profile coordinates,
restores `h` once, evaluates `N_m*x+b_m`, ColorDQ, count bounds, native tone
and post-tone `O`, then inverts Adobe's RGB tone to obtain the table target.
The reference uses native integer quantizers; table fitting uses a smooth
clamped count-domain extension of the same unquantized linear tone. No output
gamma is removed from that tone.

The stored BE and per-profile BEO sum to zero after rational serialization.
The fit uses that serialized sum. Generated non-CFA LinearRaw files have
SDK Stage3Gain=1; multi-channel CFA merge behavior is outside this scope.
Total EV is finite and nonnegative. A negative total requires Adobe's
nonlinear exposure-tone stage and is rejected by the fit. User exposure
edits remain normal raw-reader edits, not a claim of native SPP agreement.
Explicit DNG looks use native tone exposure 0 and do not apply `X3F_EV`.
The no-option path keeps its previous environment behavior.

## Legacy profiles and version fields

Under WhiteXY, an unchanged legacy CM would shift its as-shot camera white.
Each synthetic profile therefore gets a diagonal CM row correction that
restores the physical as-shot white. Its ForwardMatrix and default tone
behavior stay unchanged. Eighteen SDK comparisons against three retained
pins cover every legacy name: the full-raster maximum difference is one
8-bit RGB code, with mean differences about 0.0015 to 0.0422 codes.
A separate four-camera check tests all 24 serialized legacy camera whites;
maximum component error is 0.000079 after rational rounding.

This correction is exact only at as-shot before matrix quantization. At other
white points, the legacy profile has a fixed camera-space diagonal difference
from its no-option counterpart. Its custom-WB semantics have changed in
explicit mode exports. No-option legacy profiles keep their original bytes.

Ordinary mode extras use little-endian IIRC, with numeric magic 0x4352.
Legacy extras remain big-endian MMCR. Independent parsers check byte order,
magic, field types, payload offsets and values. Version and BackwardVersion
are both 1.4 for explicit modes. The required tag set includes WhiteXY,
ForwardMatrix, LookTable, LookTableEncoding, BEO and DefaultBlackRender.
None exports keep their previous version fields. Merrill explicit profiles
now declare CalibrationIlluminant1 D65; no-option Merrill omits it.

## Measured residuals and WB limits

The audit samples 8192 active pixels and a 1024-point scene-neutral ramp per
mode and file. It separates sensor-white crossings, ProPhoto clipping,
in-gamut colors, shadows, near-neutrals, saturated colors and hue error.
CIEDE2000 uses clipped linear sRGB converted to Lab D65. The formula passes
published Sharma reference pairs. The worst per-mode as-shot values are:

| Source | In-gamut p99 dE2000 | Scene-neutral maximum dE2000 |
| --- | ---: | ---: |
| SD1 Merrill 15 | 1.20 | 0.49 |
| dp2 Merrill 8040, Adobe source | 0.91 | 0.33 |
| dp2 Quattro 8284 | 0.34 | 0.14 |
| dp0 Quattro 0010 | 0.18 | 1.99 |
| sd Quattro SDIM0002 | 0.24 | 2.55 |
| sd Quattro H SDQH5085, explicit Sunlight WB | 0.98 | 1.50 |
| sd Quattro H _DQH1429 | 0.34 | 1.82 |

SDQH5085 uses Sunlight because its as-shot Auto white is outside the classic
profile model. These values compare the profile with the native renderer on
the same raster. They are not SPP image captures.

The grid/cap choices are unchanged on the held-out dp2 Merrill and dp0
Quattro files. Tables derive from calibration and native math, not fitted
photographic pixels. Adobe-source Merrill has 1576 capped nodes out of
79560; its in-gamut p99 remains below 0.91 in this audit.

Several limits are separate from interpolation:

- Adobe clips sensor samples to camera white before FM. The thresholds
  change per mode under WhiteXY. A partial crossing can change hue, and
  information is lost before the LookTable. One sampled 8040 pixel reaches
  about 30.52 dE; dp0 FCBlue reaches about 27.01. The native preview does not
  reproduce this reader clip.
- Negative or over-range ProPhoto components are also clipped before the
  table. This includes visible saturated colors, not only black noise.
- The required (0,0)/(1,1) curve endpoints cannot match a native neutral
  endpoint below one. Zero-saturation value scales cannot compensate it.
  On dp0, Standard reaches about 0.97 dE and Neutral about 1.37 at the top
  of the ramp. SunsetRed's 1.99 and Cinema's 1.13 maxima include bright
  sensor-white crossings.
- Merrill post-tone `O`, ColorDQ and the affine black shift leave residuals.
  A constant black shift on the DNG's baked-gain raster is not spatially
  equal to native trimmed-black subtraction before gain. Merrill TIFF
  recovery also retains its prior mode-composed spatial reconstruction.
  Whole-TIFF/SPP agreement is a separate gate.

WB results require three distinct statements. First, reader WB edits keep
this profile and its tables fixed. Second, native SPP changes `CC_w` and
WB gains with the CAMF preset, whereas a single-illuminant DNG profile cannot
switch that calibration. This limitation also affects no-option exports.
These numerical measurements concern the new profiles, not the legacy FM.
Much of the measured WB error is already present in the new Standard profile.
Third, mode-specific casts and max-white normalization add further error.
The fit absorbs `r` at the as-shot WB. The scalar formula
`r(n)=max(d*n)/max(n)` is exact everywhere. It is constant only where
`argmax(d*n)` and `argmax(n)` coincide. A fixed BEO is exact only within
that region.

The bounded Sunlight/Overcast/Incandescent sweep has worst in-gamut p99 dE
about 27.1 (SD1), 14.8 (8040), 14.1 (8284) and 10.6 (dp0). Standard itself
reaches about 18.2 for SD1 Incandescent, 11.8 for 8040 Incandescent and 11.0
for 8284 Sunlight/Overcast. Mode increments above Standard reach about 8.8
on SD1 Incandescent and at most about 4.3 in the other tested groups.
These increments compare per-mode p99 with Standard p99 at the same preset;
they are not paired-pixel estimates. These are same-raster numerical
comparisons, not Lightroom or SPP WB tests.
Native WB parity is open. Dual-illuminant work is outside this task.

## Compatibility evidence

The official Adobe DNG SDK 1.7.1 build 2724 is a temporary validator, not a
runtime dependency. Four final ordinary files pass SDK validation without
warnings. Selected-profile floating ProPhoto probes agree with the independent
model within about 0.000105 maximum component error across 16 file/mode cases;
8040 orientation is applied to probe coordinates. This checks renderer order
and serialization, not native image parity.

Sang performed bounded manual tests in Lightroom cloud 9.5.1 Local:

- C1 primary RGBTables inversion and C2 extra inversion had no effect,
  including a C1 JPEG export. C2's classic bright tone did work. A bounded
  native trace found parsing and a render gate, but no documented enable flag.
- C3 WhiteXY plus a per-profile CM produced a strong magenta cast, and the
  JPEG export retained it. Identity and bright IIRC controls behaved as expected.
- C4 per-profile BEO -1 was darker in Lightroom and JPEG. It tests positive
  total EV +3 versus +2, not the negative-exposure-tone branch.
- C5's 72x17x65 LookTable, value encoding 1 and 4097-point curve all worked.
  The hue export retained its change, and encoding matched the expected
  readable-plate reference rather than the encoding-0 control.

Sang's combined Lightroom cloud 9.5.1 Local check passed on both immutable
1791915230810 ordinary mode files, 8040 and 8284, before the label change.
Both had the then-current `x3fuse Standard` default, all six Merrill or eleven
dp Quattro names, visible differences for the tested mode choices, selected
looks retained in JPEG exports, selection retained after reopen, and editable
WB/exposure.
Full SPP captures across modes/WB/ISO and the remaining application gates
stay open. Lightroom Classic and standalone Camera Raw are not installed.

Session-local SDK references and reports are under
`/private/tmp/x3f-dng-classic/`. They include `matrix-audit.tsv`,
`residual-audit.tsv`, `wb-audit.tsv`, `legacy-sdk-comparison.json`,
`sdk-validation.json`, `sdk-pixel-comparison.json` and exact profile names in
`profiles.tsv`. The parent retains durable evidence under `_local/` before
release. Older RGBTables diagnostic source is retained only as private
session evidence; it is not part of the ordinary writer.
