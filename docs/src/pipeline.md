# Conversion pipeline

A single `x3f_extract -dng input.X3F` invocation walks the file through
the stages below. Each stage is now native Rust unless explicitly noted.

## 1. Argument parse

[`crates/x3f-cli/src/main.rs`](../../crates/x3f-cli/src/main.rs).
Hand-rolled to preserve the legacy single-dash flag syntax (`-dng`,
`-tiff`, `-color sRGB`, `-no-denoise`, …) so existing scripts and the
test corpus keep working through the port. Resolves a `FileType`
(Dng / Tiff / Ppm / Meta / Jpeg / Raw / Histogram) and a
`ProcessOptions` struct. In batch mode the file list is processed in
parallel via `rayon::par_iter` (added in M7b).

## 2. Open + directory walk

[`crates/x3f-sys/src/io.rs`](../../crates/x3f-sys/src/io.rs).
`Reader::open` (or `Reader::from_bytes` for in-memory input — used by
the WASM/iOS/Android bindings) `fopen`s the file and calls
`x3f_new_from_file`, which parses the X3F header, footer, and
directory entries. The X3F format is a section-table container
(similar in spirit to TIFF) with sections for the embedded JPEG, RAW,
CAMF (camera metadata), property list, and EXIF. See the
[format reference](./format.md) for byte-level layout.

## 3. Section loads

The Rust `Reader` exposes `load_camf` / `load_property_list` /
`load_thumbnail_jpeg` / `load_raw` / `load_unconverted_raw`. Each
resolves a directory entry and dispatches by section type:

- **PROP** — the camera's property list (key-value strings encoded as
  UTF-16LE; converted to UTF-8 in
  [`crates/x3f-sys/src/load.rs`](../../crates/x3f-sys/src/load.rs)).
- **CAMF** — camera metadata blob. Three encodings exist (type-2 stream
  cipher, type-4 zigzag predictor + Huffman, type-5 1-D predictor +
  Huffman); all three are decoded in
  [`crates/x3f-sys/src/load.rs`](../../crates/x3f-sys/src/load.rs) and
  then walked by
  [`crates/x3f-sys/src/meta.rs`](../../crates/x3f-sys/src/meta.rs) for
  named-entry lookup.
- **IMA / IMA2** — the heavy section. Runs the entropy decoder
  ([`crates/x3f-sys/src/entropy.rs`](../../crates/x3f-sys/src/entropy.rs))
  — Huffman (X530 / 10BIT raw + thumbnail), simple_decode
  (uncompressed variant), or TRUE (Sigma's predictor + Huffman) — and,
  on Quattro sensors, the 2×2 expansion that lifts the half-resolution
  M and B planes up to T's resolution
  ([`crates/x3f-sys/src/quattro.rs`](../../crates/x3f-sys/src/quattro.rs)).
  TRUE decoding is parallel across the three planes (M7d).

## 4. Image processing

[`crates/x3f-sys/src/process.rs`](../../crates/x3f-sys/src/process.rs)
+ [`crates/x3f-sys/src/highlight.rs`](../../crates/x3f-sys/src/highlight.rs).
The ordinary rendered-image path runs in this order. The
[Merrill alignment notes](./spp-alignment.md) distinguish the recovered Sigma
stages from the remaining legacy algorithms.

1. **Black level** — ordinary Merrill RGB uses a 1% trimmed mean of
   `DarkShieldTop`. Other paths retain the combined shield means and camera
   firmware workarounds. Stored CAMF rectangle endpoints are inclusive.
2. **Intermediate encoding** — write `round(scale*(raw-black)+bias)` to the
   uint16 buffer, with channel ranges proportional to WB and applicable digital
   gain. Original sensor spans are retained separately for Merrill RGB so its
   later tone input can be reconstructed in sensor-count units.
3. **WB-dependent radial shading** — Merrill C0/C2 corrections from
   `WhiteBalanceColorShadingFactor`, still applied in the integer intermediate.
4. **Bad pixels** — interpolate the camera-specific bad-pixel and cluster lists.
5. **Denoise or Quattro expansion**: portable/GPU NLM or BM3D according to the caller's
   options. This is not Sigma's native tiled filter pipeline.
6. **RGB highlight handling and spatial gain** — build the normalized chroma
   LUT; per pixel, recover highlights, optionally run RepairPix, apply the
   matrix-pathology gate, and evaluate spatial gain. Merrill uses the recovered
   aperture/focus or chief-ray-angle tables and native pixel interpolation;
   EXIF FNumber supplies aperture, with the managed PROP fallback. This ordering
   still differs from Sigma's preprocessing-time gain and highlight restoration.
7. **Camera color** — Merrill RGB restores gained counts and applies selected
   WB correction, float32 ISO scaling/remaps and color-mode compensation.
   It then applies native ColorDQ chroma correction and the residual stage's
   `[0,32766]` bounds with neutral residual gain.
   Remaining render paths retain `get_conv`'s sensor-to-XYZ-to-output matrix
   and per-channel digital ISO column scaling.
8. **Tone and output color** — Merrill applies its recovered tone LUT, Q12
   output matrix selected by EXIF source ColorSpace and requested target, then
   the 65,536-entry integer gamma LUT. Adobe-source conversion includes the
   native Adobe matrix inverse. Other paths
   retain the interpolated gamma/Cineon LUT. The tone stage is excluded from
   DNG and scene-linear output.
9. **File output** — TIFF embeds the actual RGB transfer profile and camera
   orientation. Raw/undefined encodings have no named-space profile. PPM stores
   samples without TIFF metadata.

RGB chroma-LUT recovery remains an explicit option: `-chroma-lut` enables it
and `-no-chroma-lut` disables it for rendered sRGB, Adobe RGB and ProPhoto RGB
TIFF, PPM and histograms through the original x3fuse pipeline. It does not
select SPP processing or impose SPP's denoise limits. Both override
`X3F_NO_CHROMA_LUT`; omitting the flags preserves the existing default (on unless
that environment variable is present). Rust callers set
`ProcessOptions::rgb_chroma_lut` to `Some(true)`, `Some(false)`, or `None` for
that default. The choice belongs to each conversion, including concurrent calls.
Disabling this LUT retains prior-based recovery and the matrix-pathology gate;
it does not select Sigma's native highlight routine. `-spp-pipeline` selects
the separate SPP pipeline and cannot combine with `-chroma-lut`. It uses
camera-based highlight recovery with an
omitted LUT flag or `-no-chroma-lut`. DNG, scene-linear,
`-color none`, unprocessed and Cineon output ignore the RGB override.
`X3F_NO_CHROMA_LUT` still controls the separate DNG-recovery LUT, including
camera-space (`-color none`) output. The RGB LUT remains available as an
alternative while Sigma's native pipeline is reconstructed.

An explicit `ProcessOptions::color_mode` uses the shared validated resolver
for TIFF and DNG. Merrill tone rows use the contrast-list ordinal. dp Quattro
uses its native enum bounds, built-in tone shapes and quarter-resolution
curve reconstruction. TIFF applies the mode before its output ICC encoding.
The option defaults to `None`; the stages above keep their existing behavior
when it is absent. sd Quattro and sd Quattro H modes remain unsupported.

The experimental DNG mode path clears the creative selection before raw
processing, then builds profile metadata and a selected-mode preview from
that same image. Raw samples, levels and recovery scale stay independent of
the selected mode. The replacement classic profiles use creative CM/FM values
and a shared AsShotWhiteXY. Its separate cast, exposure, look-table and long-curve
controls pass in Lightroom. The combined Merrill and dp Quattro files also
pass profile switching, export, reopen and editable-control checks. The
embedded preview uses the native renderer and can differ from the profile fit. The
[Color Modes plan](../COLOR-MODES-PLAN.md) records evidence and remaining gates.

The DNG path replaces stages 6–9 with highlight handling in sensor BMT
space. Recovery is opt-in through `-dng-highlight-recovery`. On Merrill
and older sensors, reliable nearby layer ratios propose reconstruction of
one or two clipped layers. Local proposals must also agree with the
established reconstruction's rendered chroma after both BMT vectors pass
through the color matrix. Donor confidence alone cannot justify a large
color change. When local color is uncertain, retain the established chroma
and recover scalar brightness variation from surviving layers, applying it
to the coherent BMT vector. This conservative separation retains measured
texture without introducing unrelated highlight colors. Fully clipped
pixels cannot supply missing texture themselves.

Nominal channel headroom alone does not guarantee trustworthy Foveon color:
the overlapping BMT responses are not independent display RGB channels.
Even a near-clipped sample below its saturation threshold can produce
severe opposing channel contributions in the color matrix. A matrix-aware
highlight safeguard therefore checks the coherent BMT vector after local
reconstruction, including near-clipped samples. Pathological vectors are
projected onto a neutral ray, with amplitude estimated from smoothly
weighted surviving layers rather than the clipped maximum. This preserves
available brightness variation while suppressing green/magenta highlight
artifacts. It can change nominally unclipped layers inside those
pathological highlights; normal valid unclipped areas retain their samples.

Recovered values and spatial gain stay in floating point until the final
shared 16-bit encoding. The new `-dng-highlight-mapping linear|shoulder`
selector controls how their headroom is stored; it does not enable recovery.
The Rust API exposes the same choice as `DngHighlightMapping`.

- **`linear` (default):** divide every recovered sample by one image-wide
  factor `S >= 1`, measured from the active image area, and encode once
  into `[0, 65535]`. Add `log2(S)` to the existing `BaselineExposure` and
  publish `LinearResponseLimit = 1`. This preserves proportional highlight
  differences for later exposure edits. Readers that ignore
  `BaselineExposure` display the image darker by `log2(S)` stops; applying
  that exposure restores its brightness. Preview generation compensates
  for `S` and applies spatial gain only once.
- **`shoulder`:** use the same recovered samples, then apply the existing
  smooth per-pixel highlight shoulder. Its default knee is `0.85`, adjusted
  by `X3F_DNG_SHOULDER_KNEE`. Uniform scaling within each pixel preserves
  layer ratios while compressing the range above the knee below white.
  `LinearResponseLimit` records the knee when compression is needed and
  otherwise stays `1`; there is no headroom exposure adjustment. The tag
  describes the nonlinear range and does not undo the baked compression.

Both mappings retain uniform `BlackLevel = 0` and `WhiteLevel = 65535`.
With recovery disabled, selecting either mapping leaves the existing
recovery-off conversion unchanged. TIFF/PPM recovery is unchanged.

On Quattro, recovery runs after expansion and retains the existing
CLUT/RepairPix/neutral-fallback reconstruction. The new local BMT method
does not treat the interpolated lower layers as independent full-resolution
measurements. Both final headroom mappings are available on this path.
The previous unconditional Quattro bypass made the recovery option
ineffective. Its recovery-off path retains the expanded sensor samples;
unequal channel saturation can then produce lime/yellow highlights even
with correct color metadata. Recovery remains an explicit processing
choice because it estimates missing color and can change bright scene
colors.

The DNG's `BaselineExposure` carries the ISO brightening that is
deliberately kept out of the raster:
`log2(CaptureISO/SensorISO) + log2(geometric mean of DigitalISOGain)`,
plus `log2(S)` for linear highlight recovery.
Baking `DigitalISOGain` into the raw samples instead (as an earlier
revision did in `preprocess_data`) falsely clipped the top
`log2(gain)` of a stop of real highlight data at `WhiteLevel`; readers
that ignore BE render those files a fraction of a stop darker, which
matches the existing (much larger) BE reliance for Merrill ISO
brightness.

`DigitalISOGain` can differ substantially by channel: the sd Quattro H
samples use `[4, 4, 2]`. Keeping only its geometric mean loses that color
balance and produces a severe cast. The DNG writer computes
`h = DigitalISOGain / geometric_mean(DigitalISOGain)` and publishes
`AsShotNeutral = 1 / (WhiteBalanceGains * h)`. The diagonal
`1 / (OvercastGains * h)` is folded into every profile's ColorMatrix,
so CameraCalibration can be omitted (its default is identity).
ForwardMatrix is unchanged. This preserves the relative gain through
white-balance editing and readers that ignore CameraCalibration, without
clipping the normalized samples. Quattro's intermediate encoding must use
the same relative gain: otherwise adding T-plane detail equally to B, M,
and T produces lime bright detail and purple dark detail after conversion.
The working-range correction intentionally changes reconstructed Quattro
pixels; it is separate from the algebraic metadata correction.

Quattro profiles identify their calibration illuminant as D65, matching
the basis of their ColorMatrix. Hue/saturation maps use `[21,2,1]` and
explicit linear encoding. The DNG specification makes the encoding
inapplicable when there is only one value division; encoding 0 also avoids
an extra inverse sRGB curve in RawTherapee 5.12 and Adobe SDK 1.7.1, which
otherwise darken even an identity map.

The writer then equalizes the three channels into one shared
encoding range (`output::dng::equalize_levels`): the per-channel
saturation points (e.g. Merrill `[16383, 7695, 4829]`) are baked
into the raster and the tags become a uniform `BlackLevel = 0` /
`WhiteLevel = 65535`. Adobe and LibRaw normalize per-channel
WhiteLevel correctly, but Apple's RAW engine and Capture One do not
(single-level normalization destroys the channel ratios — the
historical magenta/green casts in those apps), and baking the
normalization is loss-free since each channel gains range.

The hot loop is parallelised by row via `rayon::par_chunks_mut`
(M7a/c) and decode is parallelised by plane (M7d). Single-image
end-to-end speedups vs. the M0 baseline are 1.5–4× depending on
sensor; batch mode adds another ~4× via top-level `par_iter` over the
file list.

## 5. Output

### Editor source

`x3f_core::prepare_scene_linear` opens a fresh reader and runs the same sensor
normalization, repair, Quattro expansion and optional NLM/BM3D used by conversion.
It returns an owned, active-area-cropped, unrotated RGB `Vec<f32>` in extended
linear sRGB/D65. As-shot WB and capture/digital ISO gains are already applied.
Orientation and camera WB calibration accompany the pixels. The renderer must
apply orientation once and preserve negative channels and values above one until
its chosen output transform.

`SceneLinearImage::color_mode_calibration` captures native Color Mode calibration
from the same decoded RAW before sensor processing changes it. It uses the
as-shot WB and returns `None` for unsupported sensors or invalid native
calibration. Cancellation and other errors still stop preparation. Editors can
reuse this value without decoding RAW a second time.

For a prepared pixel, `inverse(white_balance.camera_to_rgb) * scene_rgb` gives
the normalized BMT input to this calibration. Spatial gain, sensor denoise,
repair and any requested highlight recovery are already in that input. Do not
apply spatial gain again. Black, white and WB normalization match DNG samples,
but spatial gain follows `SceneLinearOptions::apply_sgain` and can differ from
DNG export. DNG output scaling and final integer encoding are absent; the
earlier sensor intermediates still use integers.

Multiply by the matrix from `render_matrix` and add its offset to obtain native
counts. Then apply `color_dq` when present before the sensor's count bounds and
tone lookup. The calibration is fixed at as-shot WB; it does not reproduce a
native WB preset change. Scene
preparation uses different filters and spatial-gain tables from native SPP.
Merrill's black-offset and spatial-gain order also differ. This mapping does
not undo sensor processing or establish whole-image SPP parity.

`read_as_shot_crop` reads camera framing without decoding RAW pixels. It uses
the embedded JPEG aspect and CAMF active area, matching DNG `DefaultUserCrop`,
and returns normalized `[x, y, width, height]` in EXIF-oriented coordinates.
Use it only to initialize new recipes; saved user crops remain authoritative.

The editor handoff evaluates the existing floating sensor-recovery stage before
the DNG pixel normalization, image-maximum scaling or 16-bit encoding. It does
not apply a gamma curve, creative EV or display clamp. Sensor preparation still
uses the established integer intermediates; this is not a replacement decoder.
Editor options are per call; legacy research environment variables continue to
control legacy conversion, not editor exposure or recovery options.

Because sensor preparation mutates its buffers, each denoise/repair change uses
a fresh preparation. Color/tone changes reuse the immutable returned image.
The API reports Read/Decode/Process and accepts caller-owned cancellation.

`output::tiff::write_rgb16` writes already transfer-encoded RGB16 with cancellation
and without a pixel copy. `color_icc_profile` supplies matching sRGB, Adobe RGB
or ProPhoto RGB ICC profiles. These describe actual nonlinear export pixels;
the older Cineon identity-TRC profile remains exclusive to that legacy mode.

One of:

- **DNG** — pure-Rust IFD writer in
  [`crates/x3f-core/src/output/dng/`](../../crates/x3f-core/src/output/dng/).
  Emits standard TIFF, EXIF, and DNG tags, including ForwardMatrix1
  and DefaultBlackRender. Without `-color-mode`, Standard is the default
  camera profile, with five monochrome variants and Unconverted as extras.
  The legacy CMCM picture modes remain disabled. Explicit Color Modes use the
  separate profile path described above. The writer embeds
  Sigma's flat-fielding `OpcodeList3` blobs (lens-correction GainMaps)
  when `-opcodes-dir opcodes` is passed. `-compress` encodes the raw
  plane as lossless JPEG (`Compression = 7`, the pure-Rust LJ92
  encoder in
  [`crates/x3f-core/src/output/dng/ljpeg.rs`](../../crates/x3f-core/src/output/dng/ljpeg.rs)),
  written as a single full-height strip. The legacy Adobe-Deflate
  32-row-strip path was dropped: the DNG spec reserves Deflate for
  floating-point/32-bit data and Apple's RAW engine rejects deflated
  integer raws wholesale (no Finder/Quick Look previews), while the
  dcraw-lineage decoders stop after the first strip of a multi-strip
  lossless-JPEG plane — single-strip LJ92 is the layout everything
  decodes. Required UniqueCameraModel and LinearResponseLimit live in
  IFD0; DefaultUserCrop is RATIONAL[4] in the raw IFD. Legacy hue/saturation
  maps use `[21, 2, 1]` dimensions, with saturation varying fastest and
  identical corrections at both saturation endpoints.
- **16-bit TIFF** —
  [`crates/x3f-core/src/output/tiff.rs`](../../crates/x3f-core/src/output/tiff.rs)
  via the `tiff` crate.
- **PPM (P3 / P6)** —
  [`crates/x3f-core/src/output/ppm.rs`](../../crates/x3f-core/src/output/ppm.rs).
- **Embedded JPEG thumbnail / raw-block dump** — byte-blob copy in
  [`crates/x3f-core/src/lib.rs`](../../crates/x3f-core/src/lib.rs)
  `write_blob`.
- **Metadata text dump** —
  [`crates/x3f-sys/src/print_meta.rs`](../../crates/x3f-sys/src/print_meta.rs).
  Uses `libc::printf` / `fprintf` for byte-identical `%g` / `%9f`
  formatting (tier-2-MD5-pinned).
- **CSV histogram** —
  [`crates/x3f-sys/src/histogram.rs`](../../crates/x3f-sys/src/histogram.rs).

## What's still C

Only two tiny C files remain in the cc-rs build:

| File | Role |
|------|------|
| [`csrc/x3f_printf.c`](../../crates/x3f-sys/csrc/x3f_printf.c) | logging shim (function-pointer hook calls into a Rust callback for mobile/WASM) |
| [`csrc/x3f_version.c`](../../crates/x3f-sys/csrc/x3f_version.c) | version string |

The remaining host pipeline is Rust. The default denoiser is Non-Local Means in
[`crates/x3f-sys/src/denoise.rs`](../../crates/x3f-sys/src/denoise.rs), called
directly by [`run_denoising`](../../crates/x3f-sys/src/process.rs) and the
Quattro upsampler on every target. It mirrors the algorithm of the original
OpenCV pass — fixed-point weight LUT, `BORDER_REFLECT_101` windows, per-channel
`h`, L1 patch distance, V-channel median, and the low-frequency refinement — but
is *not* byte-identical to that old output, and no parity baseline pins it:
the original tier-2/tier-3 tests use `-no-denoise`. (OpenCV / opencv-mobile was removed
entirely; there is no longer a C/C++ denoise path or a build-time download.) On
`wasm32-*` cc-rs is skipped entirely and the variadic `x3f_printf` shim is
satisfied by
[`crates/x3f-sys/src/wasm_c_shims.rs`](../../crates/x3f-sys/src/wasm_c_shims.rs).

Denoise strength is a `0`–`10` intensity (`-denoise <0-10>`, default `10`;
`-no-denoise` is the alias for `0`). The CLI value rides through the
pipeline's `denoise` arg and `x3f-core`'s `ProcessOptions::denoise_intensity`;
[`run_denoising`](../../crates/x3f-sys/src/process.rs) /
[`expand_quattro`](../../crates/x3f-sys/src/process.rs) map it to a
`scale = intensity / 10` that the Rust NLM kernels
([`denoise_area`](../../crates/x3f-sys/src/denoise.rs) /
[`denoise_active_area`](../../crates/x3f-sys/src/denoise.rs)) multiply onto each
sensor's base NLM sigma. `10` → `scale = 1.0` is full-strength denoise; `0`
gates the pass out entirely. (Denoise output is not part of the byte-parity
gate — every tier-2/3 test runs with `-no-denoise`.)

`-spp-pipeline` selects a separate Merrill float path before normalized sensor
processing. `ProcessOptions::spp_denoise` carries this choice per conversion.
The path runs float gains and ordered repairs, native highlight restoration,
despeckle, row/column correction, tiled bilateral and residual denoise,
sharpening, active-area crop, and the shared Merrill tone and output stages.
It accepts HI color TIFF, PPM and histogram output with neutral noise sliders.
It uses CPU execution for `Auto` or `Cpu`; forced `Gpu` returns an error.
It requires default `DenoiseOptions` and rejects BM3D or custom NLM controls.
Unsupported cameras, resolution, calibration and output modes return errors.
The [SPP alignment notes](./spp-alignment.md) describe its verification and limits.

Native CLI builds also include wgpu implementations of NLM and BM3D in
[`denoise/`](../../crates/x3f-sys/src/denoise/). `-denoise-backend auto` (default)
tries Metal/Vulkan/DX12 and retries on CPU if initialization or execution fails.
`cpu` forces the existing Rust path; `gpu` requires acceleration and fails
explicitly. Backend selection travels through `ProcessOptions` or
`SceneLinearOptions` and `Control`, so concurrent conversions can choose
different backends. `DenoiseOptions` travels in the same per-call control and
selects NLM/BM3D, luminance/chroma, strength, and NLM patch/search sizes.
BM3D replaces the NLM/median/low-frequency stages with collaborative hard
thresholding and Wiener filtering in sensor YUV; it estimates noise there
rather than reusing the NLM profile as a standard deviation. See the
[algorithm controls and profile](performance.md#algorithm-and-advanced-controls).
Legacy C callers keep CPU NLM defaults; no C struct layout changes.

The NLM GPU backend uses globally reflected halos around 1024×1024 tiles, the same L1
distances and quantized weights, and FP32 accumulation. Subgroup operations
share patch-distance work when supported; a shared-memory shader covers other
compute adapters. Median and resize operations use Rayon on native targets,
preserving the existing per-pixel arithmetic order. WASM remains serial CPU
with no GPU imports. See [performance notes](performance.md#gpu-denoising) for
numerical tolerances, benchmarks, and the BM3D implementation.

Legacy C headers and the two log/version shims are in
[`crates/x3f-sys/csrc/`](../../crates/x3f-sys/csrc/). Bindgen consumes the
headers for ABI types. The other C function bodies have been ported to Rust.
