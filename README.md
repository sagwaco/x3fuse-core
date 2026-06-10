# x3fuse-core

Command-line converter for Sigma Foveon **X3F** raw files. Decodes Merrill, classic (SD9/SD14-era), and Quattro sensors and writes **DNG**, **TIFF**, **PPM**, embedded **JPEG** thumbnails, **metadata** dumps, and **histogram** CSVs.

Built to power [X3Fuse](https://github.com/sagwaco/x3fuse), which provides a GUI for converting X3F files to DNG, TIFF, and JPEG. The codebase is a pure-Rust Cargo workspace (only two tiny C log/version shims remain) so it builds on every target (including `wasm32`) with no external dependencies.

## Quick start

Prerequisite: a Rust toolchain ([rustup](https://rustup.rs)).

```sh
cargo build --release
target/release/x3f_extract -dng photo.X3F
```

This produces `target/release/x3f_extract`. Multiple input files are processed in parallel.

## Usage

```sh
x3f_extract <switches> <file1.X3F> [file2.X3F ...]
```

### Output format (choose one)

| Flag         | Output                          |
| ------------ | ------------------------------- |
| `-dng`       | DNG `LinearRaw` (**default**)   |
| `-tiff`      | 3×16-bit TIFF                   |
| `-ppm`       | 3×16-bit PPM/P6 (binary)        |
| `-ppm-ascii` | 3×16-bit PPM/P3 (ASCII)         |
| `-jpg`       | embedded JPEG preview           |
| `-raw`       | RAW area, undecoded             |
| `-meta`      | metadata dump                   |
| `-histogram` | histogram CSV                   |
| `-loghist`   | histogram CSV with log exposure |

### Modifiers

| Flag                      | Effect                                                                             |
| ------------------------- | ---------------------------------------------------------------------------------- |
| `-o <DIR>`                | write output to `<DIR>`                                                            |
| `-v` / `-q`               | verbose / quiet (errors only)                                                      |
| `-color <SPACE>`          | RGB color space: `none`, `sRGB`, `AdobeRGB`, `ProPhotoRGB` (does not affect DNG)   |
| `-compress`               | lossless compression (DNG: lossless JPEG, TIFF: Deflate/ZIP)                       |
| `-denoise <0-10>`         | NLM denoise intensity: `0` = off, `10` = full strength (**default**); intermediate values linearly scale the NLM sigma |
| `-no-denoise`             | disable the NLM denoise pass (alias for `-denoise 0`)                               |
| `-no-crop`                | do not crop to the active image area                                               |
| `-no-sgain` / `-sgain`    | disable / force spatial-gain (lens color) compensation                             |
| `-no-fix-bad`             | do not fix bad pixels                                                              |
| `-wb <PRESET>`            | select a white-balance preset                                                      |
| `-unprocessed`            | dump RAW with no preprocessing                                                     |
| `-qtop`                   | dump the Quattro top layer, unprocessed                                            |
| `-opcodes-dir <DIR>`      | embed pre-rendered DNG `OpcodeList3` flat-fielding blobs (see [opcodes/](opcodes)) |
| `-dng-highlight-recovery` | Foveon highlight recovery for DNG (see [DNG output](#dng-output))                |
| `-cineon`                 | 16-bit TIFF with a Cineon-style log tone curve baked in (requires `-tiff`)         |
| `-offset <OFF>`           | RAW offset for SD14 and older (automatic if omitted)                               |
| `-matrixmax <M>`          | max matrix elements in metadata dump (default 100)                                 |

A few `X3F_*` environment variables tune specific stages, e.g. `X3F_CINEON_SCALE` (Cineon log-curve scale; default 100) and `X3F_DNG_SHOULDER_KNEE` (highlight-recovery knee; default `0.85`).

### Examples

```sh
# DNG (LinearRaw) — the default
x3f_extract -dng input.X3F

# 16-bit sRGB TIFF
x3f_extract -tiff -color sRGB input.X3F

# Deflate-compressed TIFF into an output directory
x3f_extract -tiff -compress -o out/ input.X3F

# DNG with a lighter denoise pass (half the default strength)
x3f_extract -dng -denoise 5 input.X3F

# Metadata dump
x3f_extract -meta input.X3F
```

## DNG output

Foveon sensors have no demosaicing step, so DNGs are written as **Linear DNGs** (`PhotometricInterpretation = LinearRaw`). To render consistently across RAW engines (Adobe Camera Raw / Lightroom, LibRaw / RawTherapee, Capture One, and Apple's RAW engine) the writer bakes per-channel saturation into the raster and tags a uniform `BlackLevel = 0` / `WhiteLevel = 65535`, and never relies on optional hints like `BaselineExposure`.

- **`-compress`**: lossless compression. TIFF uses Deflate/ZIP; DNG uses **lossless JPEG** (`Compression = 7`), the only 16-bit integer raw compression the spec allows and the one every engine decodes. Compressed output is bit-identical to uncompressed.
- **`-dng-highlight-recovery`**: reconstructs clipped channels from a scene-derived chroma LUT and folds recovered highlights back under `WhiteLevel` via a soft shoulder baked into the raster (published as `LinearResponseLimit`).

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
