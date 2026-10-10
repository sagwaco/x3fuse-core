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
| Output color | One combined floating-point matrix before gamma | Camera color precedes tone; the native profile matrix follows tone, quantized to Q12 with native signed rounding and integer arithmetic. |
| Gamma | 1024 interpolated floating-point samples | Native 65,536-entry rounded integer LUT sampled at `i/65536`; sRGB transfer or pure 2.2/1.8 power. |
| TIFF interpretation | Ordinary exports lacked ICC and orientation | Embed a profile matching the actual transfer and preserve camera orientation in the TIFF tag. Stored pixels remain in sensor orientation; Sigma physically rotates its export. |

`crates/x3f-sys/src/merrill.rs` contains the scalar output kernels;
`process.rs` supplies CAMF calibration and count-domain inputs. The processing
code preserves per-conversion cancellation and does not load the Sigma library
at runtime. Native execution is used only by the external verification harness.

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
This is a conversion check; only the retained Sigma reference measures RGB error.

Validation also passes 287 release workspace tests with the corpus (six ignored),
the eight native-kernel tests under WASM/Wasmtime, workspace Clippy, formatting,
and the documentation build. All three DNG reference hashes are unchanged;
Quattro TIFF pixel bytes are unchanged, with an intentional ICC metadata update.

The `_P2M0358.X3F` reference is a real Sigma export in RAW adjustment mode,
original size, ProPhoto RGB, TIFF16. Comparison covers all 14,751,744 oriented
pixels, without resizing, color fitting, or histogram matching. The candidate
uses `-tiff -color ProPhotoRGB -no-denoise`; Sigma's native denoise request has
not been captured, so this is a rendering comparison rather than matched
denoise parity. Final measurements and reproducible commands are retained in
the x3f-docs alignment evidence.

| `_P2M0358` ProPhoto RGB16 | Mean absolute error | PSNR |
| --- | --- | --- |
| Previous renderer | 26.70% | 10.76 dB |
| Recovered calibration, ColorDQ, tone and output stages | 1.48% | 34.15 dB |

The initial 34.90 dB intermediate result used fallback tone parameters because
the CAMF dimensions were read in the wrong order. The corrected reader uses
`[rows, 7]` tone tables and `[rows, 13]` noise tables. Source agreement governs
the implementation; the better intermediate score is not a reason to retain
the incorrect table interpretation. This is one measured image, not a corpus
color-accuracy claim.

A second reference for SD1 Merrill is prepared but has not been exported:
the locked macOS session prevented access to Sigma's window. Its count gain is
verified from shipped code and exercised by the corpus sweep; no SD1 RGB-error
measurement is claimed.

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
  slider input fall back to legacy rendering. Missing ISO calibration and the
  `SkipStepFlags` denoise-bypass branch also fall back, rather than assuming
  they share the verified ordinary path's range conditioning.
- The exact Bézier implementation currently accepts effective tone saturation
  in `[1,16384]` to bound its cubic work. For 12-bit Merrill this includes the
  ordinary -2 EV setting; larger extents select legacy rendering with a warning.
  This is an implementation resource limit, not a restriction discovered in
  Sigma's kernel.

The independent B/M chroma-table validity correction in `highlight.rs` fixes a
separate helper bug. It is not evidence that the normalized highlight algorithm
matches Sigma's native restoration.
