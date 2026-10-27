# x3fuse-core

Command-line converter for Sigma Foveon **X3F** raw files. Decodes Merrill, classic (SD9/SD14-era), and Quattro sensors and writes **DNG**, **TIFF**, **PPM**, embedded **JPEG** thumbnails, **metadata** dumps, and **histogram** CSVs.

Built to power [X3Fuse](https://github.com/sagwaco/x3fuse), which provides a GUI for converting X3F files to DNG, TIFF, and JPEG. The codebase is a Rust Cargo workspace with two tiny C log/version shims. OpenCV is not required.

## Quick start

Prerequisites: Rust 1.88 or newer ([rustup](https://rustup.rs)), a C compiler and standard headers, and libclang for bindgen. No external TIFF or JPEG library is required.

```sh
cargo build --release
target/release/x3f_extract -dng photo.X3F
```

This produces `target/release/x3f_extract`. Multiple input files are processed in parallel.

Native CLI builds include GPU NLM denoising through wgpu (Metal, Vulkan,
or DirectX 12), with automatic CPU fallback. To build a CPU-only CLI:
`cargo build --release -p x3f-cli --no-default-features`. WASM always uses the
portable CPU denoiser, even when the `gpu` feature is enabled.

## Rust library

Add the safe API crate to your application's `Cargo.toml`:

```toml
[dependencies]
x3f-core = "0.4.1"
```

[`convert_file`](crates/x3f-core/src/conversion.rs)
synchronously converts one file to DNG, TIFF, or its embedded JPEG. Pass an
`AtomicBool` for cooperative cancellation and a callback for the four
`ConversionStage` updates. Each call owns its reader and processing options,
so a desktop application can run independent conversions on blocking workers.
The destination must not already exist; failed or cancelled conversions remove
the output they created.

For native GPU denoising in this checkout, enable the library's optional `gpu` Cargo feature.
`ProcessOptions::denoise_backend` and `SceneLinearOptions::denoise_backend`
accept `DenoiseBackend::{Auto, Cpu, Gpu}`. `Auto` is the default; `Gpu` requires
acceleration and returns `Error::Gpu` when unavailable. Default NLM now copies
luminance without filtering, so CPU output can differ slightly from earlier
releases in nearly constant regions. Output with `-no-denoise` remains
byte-identical. GPU floating-point accumulation can also differ slightly.
See [denoising performance and validation](docs/src/performance.md#gpu-denoising).

Select `DenoiseOptions::algorithm` (`Nlm` or `Bm3d`) through either options
struct’s `denoise_options` field. NLM remains the default. Advanced controls
include independent luminance/chroma toggles, strength, and NLM patch/search
sizes. Both algorithms support native GPU execution and CPU/WASM fallback.
Enable the optional `serde` feature to persist these settings.

Version **0.2.0** adds `denoise_backend` and
`denoise_options` to `ProcessOptions` and `SceneLinearOptions`, and GPU errors to the Rust error
enums. This branch also adds `spp_denoise` and `color_mode` to `ProcessOptions`.
`SceneLinearImage` adds optional `color_mode_calibration` for rendering camera profiles.
Downstream exhaustive struct literals and error matches must be updated;
use `..Default::default()` when constructing options. These source changes
must not be published as a 0.1.x patch. The C ABI remains unchanged.

Build C-library artifacts separately with `cargo build --release -p x3f-ffi-c`
to exclude GPU dependencies. A whole-workspace build unifies the CLI's `gpu`
feature into the libraries too; legacy C processing entry points still select
CPU explicitly so their output does not depend on that feature unification.

## Usage

```sh
x3f_extract <switches> <file1.X3F> [file2.X3F ...]
```

### Output format (choose one)

| Flag         | Output                          |
| ------------ | ------------------------------- |
| `-dng`       | DNG `LinearRaw` (**default**). Merge all frames for X3I input. Process X3I inputs serially; keep X3F batch conversion parallel.   |
| `-tiff`      | 3×16-bit TIFF, including merged X3I RGB |
| `-ppm`       | 3×16-bit PPM/P6 (binary), including merged X3I RGB |
| `-ppm-ascii` | 3×16-bit PPM/P3 (ASCII), including merged X3I RGB |
| `-jpg`       | embedded JPEG preview           |
| `-raw`       | RAW area, undecoded             |
| `-meta`      | metadata dump                   |
| `-x3i-frame <n>` | Convert frame `n` of an X3I input (zero based). Append `_<n>` before the output extension. |
| `-x3i-middle` | Convert only the middle exposure of an X3I input, as `-x3i-frame <n>` does for that frame. Append `_<n>` before the output extension. Excludes `-x3i-frame`, `-x3i-extract` and `-x3i-report`. |
| `-x3i-ghost-map` | With an X3I DNG, TIFF or PPM merge, write each non-reference ghost-weight image as `<input>.ghost<n>.pgm`. Honour `-o`. |
| `-x3i-report` | With X3I `-dng`, write estimated sensor ceilings, raw-plane ratios in eight reference-level bins, noise, clipping, merge counts and peak RSS to `<input>.x3i-report`. |
| `-x3i-extract` | Copy all X3I frames to new `<stem>_<n>.X3F` files. Use `-o` to select a directory. Excludes output flags. |
| `-histogram` | histogram CSV                   |
| `-loghist`   | histogram CSV with log exposure |

### Modifiers

| Flag                      | Effect                                                                             |
| ------------------------- | ---------------------------------------------------------------------------------- |
| `-o <DIR>`                | write output to `<DIR>`                                                            |
| `-v` / `-q`               | verbose / quiet (errors only)                                                      |
| `-color <SPACE>`          | RGB color space: `none`, `sRGB`, `AdobeRGB`, `ProPhotoRGB` (does not affect DNG)   |
| `-color-mode <MODE>` | experimental Merrill/dp Quattro camera look for color TIFF or DNG; see the compatibility limit below |
| `-compress`               | lossless compression (DNG: lossless JPEG, TIFF: Deflate/ZIP)                       |
| `-denoise <0-10>`         | Denoise intensity: `0` = off, `10` = full strength (**default**); intermediate values scale filter strength |
| `-no-denoise`             | disable denoising (alias for `-denoise 0`)                               |
| `-denoiser <nlm\|bm3d>` | algorithm (default: NLM); see [advanced controls](docs/src/performance.md#algorithm-and-advanced-controls) |
| `-denoise-backend <auto\|cpu\|gpu>` | execution backend; `auto` falls back to CPU, `gpu` fails if acceleration is unavailable |
| `-spp-pipeline` | Merrill SPP recovery and denoise on CPU with neutral noise sliders; full-resolution color TIFF, PPM or histogram |
| `-no-crop`                | do not crop to the active image area                                               |
| `-no-sgain` / `-sgain`    | disable / force spatial-gain (lens color) compensation                             |
| `-no-fix-bad`             | do not fix bad pixels                                                              |
| `-chroma-lut` / `-no-chroma-lut` | enable / disable scene-derived RGB highlight LUT; override `X3F_NO_CHROMA_LUT` |
| `-wb <PRESET>`            | select a white-balance preset                                                      |
| `-unprocessed`            | dump RAW with no preprocessing                                                     |
| `-qtop`                   | dump the Quattro top layer, unprocessed                                            |
| `-opcodes-dir <DIR>`      | embed pre-rendered DNG `OpcodeList3` flat-fielding blobs (see [opcodes/](opcodes)) |
| `-dng-highlight-recovery` | Foveon highlight recovery for DNG (see [DNG output](#dng-output))                |
| `-cineon`                 | 16-bit TIFF with a Cineon-style log tone curve baked in (requires `-tiff`)         |
| `-offset <OFF>`           | RAW offset for SD14 and older (automatic if omitted)                               |
| `-matrixmax <M>`          | max matrix elements in metadata dump (default 100)                                 |

A few `X3F_*` environment variables tune specific stages, e.g. `X3F_CINEON_SCALE` (Cineon log-curve scale; default 100) and `X3F_DNG_SHOULDER_KNEE` (highlight-recovery knee; default `0.85`).

X3I merges support DNG, TIFF and PPM. DNG preserves extended highlight range.
TIFF and PPM clip output above one and skip the RGB chroma LUT and single-frame
highlight recovery. X3I post-merge denoise stays off unless
`X3F_X3I_DENOISE_SCALE` is positive, even with the default CLI denoise intensity.
Residual quantization can cause deep-shadow bands.
See [X3I merge and outputs](docs/src/x3i.md) and the
[denoise measurements](docs/src/x3i.md#experimental-denoise).
A merge of the tested Quattro captures needs 2 to 3 GB of memory; see
[X3I memory](docs/src/x3i.md#memory).

| X3I environment override | Effect |
| --- | --- |
| `X3F_X3I_NOISE=sigma,k,q` | Set normalized read, linear and quadratic noise terms. |
| `X3F_X3I_MAX_S=value` | Cap highlight encoding scale; report clipped samples. |
| `X3F_X3I_SATURATION=b,m,t` | Override per-plane physical sensor ceilings. |
| `X3F_X3I_ROLLOFF=knee,width,margin` | Set knee, width and validity margin relative to the ceiling. Defaults: `0.75,0.02,0.15`. |
| `X3F_X3I_NO_ALIGN=1` | Disable frame alignment. Merge frames at their original positions. |
| `X3F_X3I_FIXED_RATIOS=1` | Keep nominal shutter ratios. Report fitted ratios without applying them. |
| `X3F_X3I_NO_GHOST=1` | Disable ghost rejection and effective-variance calibration. |
| `X3F_X3I_GHOST_SIGMA=value` | Set the positive ghost threshold in sigma. Default: `3`. |
| `X3F_X3I_WEAK_GHOST_SIGMA=value` | Set the weak-alignment ghost threshold. Default: GHOST_SIGMA; see [controls](docs/src/x3i.md#research-overrides). |
| `X3F_X3I_C_MAX=value` | Set the finite effective-variance upper clamp. Default: `4`; at least C_MIN. |
| `X3F_X3I_C_MIN=value` | Set the effective-variance calibration floor. Default: `0.5`; range: above zero through one. |
| `X3F_X3I_RATIO_TREND=value` | Set the maximum relative trend across clean supported ratio bins. Default: `0.02` (2 percent). Larger trends keep the nominal ratio. |
| `X3F_X3I_DENOISE_SCALE=value` | Positive enables experimental post-merge NLM or BM3D; unset or zero disables it. |

### Examples

```sh
# DNG (LinearRaw) — the default
x3f_extract -dng input.X3F

# 16-bit sRGB TIFF
x3f_extract -tiff -color sRGB input.X3F

# Explicitly retain scene-derived chroma-LUT highlight recovery for RGB
x3f_extract -tiff -chroma-lut input.X3F

# Deflate-compressed TIFF into an output directory
x3f_extract -tiff -compress -o out/ input.X3F

# DNG with a lighter denoise pass (half the default strength)
x3f_extract -dng -denoise 5 input.X3F

# Require GPU denoising (use -v to report the adapter and stage timings)
x3f_extract -v -dng -denoise-backend gpu input.X3F

# Merrill SPP processing with neutral noise sliders
x3f_extract -tiff -color ProPhotoRGB -spp-pipeline input.X3F

# Metadata dump
x3f_extract -meta input.X3F
```

## RGB highlight recovery

For rendered RGB, `-chroma-lut` and `-no-chroma-lut` explicitly enable or disable
the scene-derived LUT, overriding `X3F_NO_CHROMA_LUT`. The default remains on
unless that environment variable is present. Disabling the LUT retains the
prior recovery and matrix gate; it does not select native Sigma recovery. These
RGB options do not affect DNG or Cineon output. The LUT uses the original
x3fuse pipeline and does not select SPP. `-chroma-lut` cannot combine with
`-spp-pipeline`; select one pipeline. `-no-chroma-lut` is accepted with SPP
and does not disable SPP's camera-based highlight recovery.

`-spp-pipeline` selects the complete Merrill float path, including camera-based
highlight repair, tiled denoise and sharpening. It requires an HI Merrill file,
color output and denoise intensity 10. DNG, Cineon, monochrome, MED/LOW and a
forced GPU backend return an error. Rust callers set `ProcessOptions::spp_denoise`.
SPP requires the default `DenoiseOptions`; BM3D and custom NLM controls return
an error because SPP uses a separate filter pipeline.
The default is false. See the [SPP alignment notes](docs/src/spp-alignment.md)
for supported settings and native verification.

## Camera Color Modes (development)

Set `-color-mode Vivid`, another named mode, or `-color-mode as-shot` to use
the new camera-mode path. TIFF applies the look to its pixels. Merrill has
Standard, Vivid, Neutral, Portrait, Landscape and FOV Classic Blue. dp Quattro
adds Sunset Red, Forest Green, FOV Classic Yellow, Cinema and Teal and Orange.
sd Quattro and sd Quattro H need a further calibration port and return an error.

DNG profile names match these camera labels without a prefix. Quote names
with spaces in the CLI, for example `-color-mode "FOV Classic Blue"`.
Existing short names such as `FCBlue` and `TealAndOrange` remain accepted.

Rust callers set `ProcessOptions::color_mode` and can query
`Reader::available_color_modes()`. This new public field requires updates to
exhaustive Rust struct literals; `..Default::default()` retains the existing
output. The C ABI is unchanged. The option is separate from output color
encoding and is accepted for TIFF and DNG exports only.

The DNG implementation is not ready for release. It preserves raw samples and
embeds named profiles. Lightroom 9.5.1 Local ignored the initial RGBTables
looks. The replacement classic-profile controls pass for cast, exposure, look
tables and long tone curves, with JPEG export checks for cast, exposure and hue.
The full Merrill and dp Quattro mode files now pass the combined Lightroom
check: defaults, visible switching, JPEG export, reopen and editable controls.
Source review and workspace checks pass. Full SPP image parity is unverified.
DNG looks approximate native rendering at the export white balance. Later
WB edits keep the profile fixed and can differ from SPP. Highlights outside
the DNG profile gamut can also differ.
See the [implementation status and remaining gates](docs/COLOR-MODES-PLAN.md).
Omitting `-color-mode` retains the existing TIFF and DNG behavior.

## DNG output

Foveon sensors have no demosaicing step, so DNGs are written as **Linear DNGs** (`PhotometricInterpretation = LinearRaw`). The writer bakes per-channel saturation into the raster and tags a uniform `BlackLevel = 0` / `WhiteLevel = 65535`. `BaselineExposure` carries ISO brightening and any linear highlight-headroom compensation; readers that ignore it can show a darker image.

- **`-compress`**: lossless compression. TIFF uses Deflate/ZIP; DNG uses **lossless JPEG** (`Compression = 7`), the only 16-bit integer raw compression the spec allows and the one every engine decodes. Compressed output is bit-identical to uncompressed.
- **`-dng-highlight-recovery`**: reconstructs clipped channels and preserves recovered headroom linearly by default. Add `-dng-highlight-mapping shoulder` to bake the legacy soft shoulder into the raster.

See the [conversion pipeline](docs/src/pipeline.md) chapter for the full DNG writer design.

## Project layout

```
x3f-cli  ──▶  x3f-core  ──▶  x3f-sys  ──FFI──▶  C (log/version shims)
```

| Path | Role |
| ---- | ---- |
| [crates/x3f-cli](crates/x3f-cli) | `x3f_extract` binary |
| [crates/x3f-core](crates/x3f-core) | safe Rust API for reading and converting X3F images |
| [crates/x3f-sys](crates/x3f-sys) | low-level layer, bindgen FFI, pure-Rust NLM denoise |
| [crates/x3f-ffi-c](crates/x3f-ffi-c) | C ABI for iOS / Android / WASM consumers |
| [opcodes/](opcodes) | pre-rendered DNG `OpcodeList3` flat-fielding blobs (Merrill) |
| [docs/](docs) | [mdbook](https://rust-lang.github.io/mdBook/) — pipeline, format reference, FFI, contributor guide |

Browse the book locally with `mdbook serve docs --open` (install via `cargo install mdbook`).

## License

Licensed under the Apache License, Version 2.0 — see [LICENSE](LICENSE).

This project is based on the work from the [Kalpanika x3f project](https://github.com/Kalpanika/x3f). It is **not endorsed by nor affiliated with** that project. Attribution for the original BSD-licensed work it derives from is retained in [NOTICE](NOTICE).
