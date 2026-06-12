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
The `convert_data` function is the hot loop. In order:

1. **black-level subtraction** — driven by `DarkShieldTop` /
   `DarkShieldBottom` and the left/right `DarkShieldColRange` columns;
   per-camera firmware-bug workarounds (DP2 → skip Bottom; sd Quattro
   H → skip Bottom; Merrill family → skip Right) are preserved
   verbatim.
2. **white-level normalisation** + per-pixel `out = scale*(raw -
   black) + bias` clamp.
3. **spatial gain (lens shading) correction** —
   [`crates/x3f-sys/src/spatial_gain.rs`](../../crates/x3f-sys/src/spatial_gain.rs).
   Reads `IncludeBlocks` and the four nearest neighbours in
   (1/aperture, lens-position) space, builds bilinear weights, and
   assembles per-channel `mgain[]` tables.
4. **bad-pixel interpolation** — iterative neighbour-averaging from
   `BadPixels` / `BadPixelsF20` / `Jpeg_BadClusters` /
   `HighlightPixelsInfo` / `BadPixelsLumaF23` /
   `BadPixelsChromaF23` / the hardcoded sd Quattro AF pixel grid.
5. **WB-conditional radial color shading** (Merrill only) —
   `pix[B] *= 1 + a·rr⁴ + b·rr²` and `pix[T] *= 1 + c·rr⁴ + d·rr²`
   from `WhiteBalanceColorShadingFactor`.
6. **white balance** — multiplies from `WhiteBalanceGains` /
   `DP1_WhiteBalanceGains`.
7. **color-matrix transform** — sensor RGB → sRGB / AdobeRGB /
   ProPhoto / linear via the M6a matrix kernel.
8. **highlight recovery** — the chroma LUT, RepairPix, and
   matrix-pathology gate ported in M6e4. Active research area: see
   the project memory entries on Foveon highlight handling.
9. **gamma LUT application** — sRGB / AdobeRGB-2.2 / ProPhoto-1.8 (D65
   → D50 Bradford for ProPhoto).

The DNG path replaces stages 7–9 with `apply_highlight_clip_dng`
(M6e9): per-pixel CLUT/RepairPix/`L*p`/matrix-pathology gate
(sg-amplified preview) → bake-sg-into-raw → baked soft highlight
shoulder. With `-dng-highlight-recovery` the CLUT step uses the
generalized BMT apply (`chroma_lut_apply_pixel_bmt`): three
scene-derived tables reconstruct whichever single channel clipped
(T from B,M; B from M,T; M from B,T) so recovered highlights keep
scene chroma, and only multi-channel clips fall back to the neutral
`L*p` snap. Recovered values overshoot the sensor white point, so a
global-max scan drives a per-pixel soft shoulder (`L' = knee +
(1-knee)·(1-(1-t/s)^s)`, knee 0.85 via `X3F_DNG_SHOULDER_KNEE`,
slope-1/C1 at the knee, chroma-preserving uniform per-pixel scale)
that compresses `[knee, global_max]` into `[knee, WhiteLevel]`. The
knee is published as `LinearResponseLimit`. An earlier design
divided the whole raster down and compensated via a
`BaselineExposure = log2(scale)` nudge instead — that rendered
correctly only in readers that honour BE (it is an optional hint in
the DNG spec), so it was replaced by the baked shoulder.

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

One of:

- **DNG** — pure-Rust IFD writer in
  [`crates/x3f-core/src/output/dng/`](../../crates/x3f-core/src/output/dng/).
  Emits the standard TIFF + EXIF chrome, the Sigma-private DNG tags
  (50964 ForwardMatrix1, 51110 DefaultBlackRender, …), all 11
  in-camera Sigma color modes as `ExtraCameraProfiles`, and embeds
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
  decodes.
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

Everything else — including the NLM denoise pass — is native Rust. Denoise is
the pure-Rust Non-Local Means in
[`crates/x3f-sys/src/denoise.rs`](../../crates/x3f-sys/src/denoise.rs), called
directly by [`run_denoising`](../../crates/x3f-sys/src/process.rs) and the
Quattro upsampler on every target. It mirrors the algorithm of the original
OpenCV pass — fixed-point weight LUT, `BORDER_REFLECT_101` windows, per-channel
`h`, L1 patch distance, V-channel median, and the low-frequency refinement — but
is *not* byte-identical to that old output, and no parity baseline pins it:
every tier-2/tier-3 test uses `-no-denoise`. (OpenCV / opencv-mobile was removed
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

Everything else under [`src/`](../../src/) is the legacy archive —
header files are still consumed by bindgen for typedefs, but the `.c`
function bodies have all been deleted or migrated.
