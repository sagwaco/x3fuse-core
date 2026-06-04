# x3fuse-core

A command-line converter for Sigma Foveon **X3F** raw files. It decodes X3F images and writes **DNG**, **TIFF**, **PPM**, embedded **JPEG** thumbnails, **metadata** dumps, and **histogram** CSVs. It supports the Merrill, classic (SD9/SD14-era), and Quattro sensor
generations.

The converter is written in Rust as a Cargo workspace. The only non-Rust component is an optional OpenCV-backed denoise pass (see [Build](#build)) — and even that has a portable pure-Rust fallback, so denoise works on every target (including `wasm32`); everything else (container parsing, entropy decode, the processing pipeline, and all output writers) is native Rust.

Built to power [X3Fuse](https://github.com/sagwaco/x3fuse), which provides a GUI for converting X3F files to DNG, TIFF, and JPEG.

## Build

The only prerequisite is a Rust toolchain ([rustup](https://rustup.rs)):

```sh
cargo build --release
```

This produces `target/release/x3f_extract`.

**Denoise (optional, on by default).** The non-local-means denoise pass links against a prebuilt [opencv-mobile](https://github.com/nihui/opencv-mobile) static library, fetched automatically at build time for supported targets (macOS, iOS, Linux, Windows, Android). If no prebuilt is available or the download fails, the build falls back to a no-op denoise stub — the binary still builds and runs; pass `-no-denoise` (or `-denoise 0`) to skip the pass explicitly. Denoise strength is adjustable on a `0`–`10` scale with `-denoise <0-10>` (see [Modifiers](#modifiers)).

## Usage

```sh
x3f_extract <switches> <file1.X3F> [file2.X3F ...]
```

Multiple input files are processed in parallel. The legacy single-dash flag syntax is used.

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
| `-compress`               | Deflate/ZIP compression for DNG and TIFF                                           |
| `-denoise <0-10>`         | NLM denoise intensity: `0` = off, `10` = full strength (**default**); intermediate values linearly scale the NLM sigma |
| `-no-denoise`             | disable the NLM denoise pass (alias for `-denoise 0`)                               |
| `-no-crop`                | do not crop to the active image area                                               |
| `-no-sgain` / `-sgain`    | disable / force spatial-gain (lens color) compensation                             |
| `-no-fix-bad`             | do not fix bad pixels                                                              |
| `-wb <PRESET>`            | select a white-balance preset                                                      |
| `-unprocessed`            | dump RAW with no preprocessing                                                     |
| `-qtop`                   | dump the Quattro top layer, unprocessed                                            |
| `-opcodes-dir <DIR>`      | embed pre-rendered DNG `OpcodeList3` flat-fielding blobs (see [opcodes/](opcodes)) |
| `-dng-highlight-recovery` | Foveon highlight recovery for DNG (see below)                                      |
| `-cineon`                 | 16-bit TIFF with a Cineon-style log tone curve baked in (requires `-tiff`)         |
| `-offset <OFF>`           | RAW offset for SD14 and older (automatic if omitted)                               |
| `-matrixmax <M>`          | max matrix elements in metadata dump (default 100)                                 |

A few `X3F_*` environment variables tune specific stages, e.g. `X3F_CINEON_SCALE` (Cineon log-curve scale; default 100).

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

## Foveon highlight recovery (DNG)

Pass `-dng-highlight-recovery` to enable the Foveon highlight-recovery pipeline. Highlight recovery is not compatible with some RAW engines, and is only tested in Adobe Camera Raw (Lightroom) and RawTherapee.

## RAW Compression

Pass `-compress` to enable RAW zip compression from TIFF and DNG outputs. Note that DNG compression is not compatible with all RAW engines, and is only tested in Adobe Camera Raw (Lightroom).

## RAW Compatibility

Because Foveon sensors do not have a traditional demosaicing step, output DNGs are written as **Linear DNGs** (`PhotometricInterpretation = LinearRaw`) — spec-compliant, but a less common path that many RAW engines (notably Apple RAW) handle poorly, since their decoders are tuned for mosaiced sensors and may misapply the embedded color matrices, levels, or
highlight logic. For best results, prefer Adobe Camera Raw / Lightroom or RawTherapee. For use with other RAW engines, consider exporting as 16-bit TIFF with a flat `-cineon` gamma curve.

## Workspace layout

```
x3f-cli  ──▶  x3f-core  ──▶  x3f-sys  ──FFI──▶  C/C++ (denoise + log/version shims)
```

- [crates/x3f-cli](crates/x3f-cli) — the `x3f_extract` binary.
- [crates/x3f-core](crates/x3f-core) — safe Rust API for reading and converting X3F images.
- [crates/x3f-sys](crates/x3f-sys) — low-level layer; bindgen FFI and the small remaining
  C/C++ ([crates/x3f-sys/csrc/](crates/x3f-sys/csrc)) for the OpenCV denoise pass, plus a
  portable pure-Rust NLM denoise ([src/denoise.rs](crates/x3f-sys/src/denoise.rs)) used where
  OpenCV isn't linked (wasm, offline builds).
- [crates/x3f-ffi-c](crates/x3f-ffi-c) — C ABI wrapper for iOS / Android / WASM consumers.
- [opcodes/](opcodes) — pre-rendered DNG `OpcodeList3` flat-fielding blobs for Merrill bodies.
- [docs/](docs) — the [mdbook](https://rust-lang.github.io/mdBook/) (pipeline, format
  reference, FFI surface, performance, contributor guide).

## License

Licensed under the Apache License, Version 2.0 — see [LICENSE](LICENSE).

This project is based on the work from the [Kalpanika x3f project](https://github.com/Kalpanika/x3f). It is **not endorsed by nor affiliated with** that project. Attribution for the original BSD-licensed work it derives from is retained in [NOTICE](NOTICE).
