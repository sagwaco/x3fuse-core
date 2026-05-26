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
(sg-amplified preview) → bake-sg-into-raw → global-max scale +
uniform divide-down so all channels fit within `WhiteLevel`. The
divide-down factor is published as `BaselineExposure = log2(scale)`
in the DNG, so Lightroom restores brightness on import without
clamping recovered chroma.

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
  when `-opcodes-dir opcodes` is passed. Strips are zlib-compressed in
  parallel (M7d).
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

Two C files remain in the cc-rs build:

| File | Role |
|------|------|
| [`src/x3f_printf.c`](../../src/x3f_printf.c) | logging shim (function-pointer hook calls into a Rust callback for mobile/WASM) |
| [`src/x3f_version.c`](../../src/x3f_version.c) | version string |

Plus the OpenCV NLM denoise path
([`src/x3f_denoise.cpp`](../../src/x3f_denoise.cpp) +
[`x3f_denoise_utils.cpp`](../../src/x3f_denoise_utils.cpp)) which links a
pinned prebuilt
[`opencv-mobile`](https://github.com/nihui/opencv-mobile) when the
target has a matching asset (Apple, Linux, Windows, Android). On
`wasm32-*` cc-rs is skipped entirely; `x3f_denoise` /
`x3f_denoise_active` / `x3f_set_use_opencl` are satisfied by
`#[no_mangle]` Rust no-op shims in
[`crates/x3f-sys/src/wasm_c_shims.rs`](../../crates/x3f-sys/src/wasm_c_shims.rs).
For any non-wasm target without an opencv-mobile prebuilt, the C stub
in
[`crates/x3f-sys/csrc/denoise_stub.c`](../../crates/x3f-sys/csrc/denoise_stub.c)
provides the same no-op resolution.

Denoise strength is a `0`–`10` intensity (`-denoise <0-10>`, default `10`;
`-no-denoise` is the alias for `0`). The CLI value rides through the
pipeline's `denoise` arg and `x3f-core`'s `ProcessOptions::denoise_intensity`;
[`run_denoising`](../../crates/x3f-sys/src/process.rs) /
[`expand_quattro`](../../crates/x3f-sys/src/process.rs) map it to a
`scale = intensity / 10` that the C kernels (`x3f_denoise`,
`x3f_denoise_active`) multiply onto each sensor's base NLM sigma. `10` →
`scale = 1.0` reproduces the legacy full-strength denoise byte-for-byte (so
the tier-2 parity baselines are unaffected); `0` gates the pass out entirely.

Everything else under [`src/`](../../src/) is the legacy archive —
header files are still consumed by bindgen for typedefs, but the `.c`
function bodies have all been deleted or migrated.
