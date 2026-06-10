# x3fuse-core

A command-line converter for Sigma Foveon **X3F** raw files. It decodes X3F images and writes **DNG**, **TIFF**, **PPM**, embedded **JPEG** thumbnails, **metadata** dumps, and **histogram** CSVs. It supports the Merrill, classic (SD9/SD14-era), and Quattro sensor
generations.

The converter is written in Rust as a Cargo workspace. The only non-Rust component is two tiny C log/version shims; everything else — container parsing, entropy decode, the processing pipeline, the Non-Local Means denoise pass, and all output writers — is native Rust, so it builds on every target (including `wasm32`) with no external dependencies.

Built to power [X3Fuse](https://github.com/sagwaco/x3fuse), which provides a GUI for converting X3F files to DNG, TIFF, and JPEG.

## Build

The only prerequisite is a Rust toolchain ([rustup](https://rustup.rs)):

```sh
cargo build --release
```

This produces `target/release/x3f_extract`.

**Denoise (on by default).** The non-local-means denoise pass is a pure-Rust implementation ([crates/x3f-sys/src/denoise.rs](crates/x3f-sys/src/denoise.rs)) that runs on every target with no external dependency or build-time download. Pass `-no-denoise` (or `-denoise 0`) to skip the pass; denoise strength is adjustable on a `0`–`10` scale with `-denoise <0-10>` (see [Modifiers](#modifiers)).

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

Pass `-dng-highlight-recovery` to enable the Foveon highlight-recovery pipeline: clipped channels are reconstructed per-channel from a scene-derived chroma LUT (with a neutral fallback for fully blown pixels), and the recovered overshoot is folded back under `WhiteLevel` by a soft highlight shoulder baked into the raster (knee tunable via `X3F_DNG_SHOULDER_KNEE`, default `0.85`; published as `LinearResponseLimit`). The output is self-contained — no reliance on `BaselineExposure` or other optional-to-honour DNG hints — so recovered DNGs render consistently in Adobe Camera Raw / Lightroom, LibRaw / RawTherapee, Capture One, and Apple's RAW engine.

## RAW Compression

Pass `-compress` to compress TIFF and DNG outputs losslessly. TIFF uses Deflate/ZIP. DNG uses **lossless JPEG** (`Compression = 7`, ITU-T T.81 process 14) — the only compression the DNG spec allows for 16-bit integer raw data, and the one every RAW engine decodes. (An earlier version used Adobe Deflate for the DNG raw plane; the spec reserves Deflate for floating-point/32-bit data, and Apple's RAW engine rejects such files outright — compressed DNGs had no Finder/Quick Look previews on macOS. The lossless-JPEG raw plane is written as a single full-height strip, the layout the dcraw-lineage decoders require.) Compressed DNGs decode bit-identically to uncompressed ones and are verified against Adobe Camera Raw / Lightroom, LibRaw, and Apple's RAW engine.

## RAW Compatibility

Because Foveon sensors do not have a traditional demosaicing step, output DNGs are written as **Linear DNGs** (`PhotometricInterpretation = LinearRaw`) — spec-compliant, but a less common path some RAW engines exercise poorly. The writer therefore avoids every LinearRaw feature known to be mishandled outside Adobe: the per-channel saturation points are baked into the raster so the tags are a uniform `BlackLevel = 0` / `WhiteLevel = 65535` (Apple RAW and Capture One mis-normalize per-channel `WhiteLevel` on 3-sample LinearRaw), and highlight recovery never depends on `BaselineExposure` being honoured. Output is tested against Adobe Camera Raw / Lightroom, LibRaw / RawTherapee, and Apple's RAW engine.

## Workspace layout

```
x3f-cli  ──▶  x3f-core  ──▶  x3f-sys  ──FFI──▶  C (log/version shims)
```

- [crates/x3f-cli](crates/x3f-cli) — the `x3f_extract` binary.
- [crates/x3f-core](crates/x3f-core) — safe Rust API for reading and converting X3F images.
- [crates/x3f-sys](crates/x3f-sys) — low-level layer; bindgen FFI and the two tiny remaining
  C shims ([crates/x3f-sys/csrc/](crates/x3f-sys/csrc)), plus the pure-Rust Non-Local Means
  denoise ([src/denoise.rs](crates/x3f-sys/src/denoise.rs)) used on every target.
- [crates/x3f-ffi-c](crates/x3f-ffi-c) — C ABI wrapper for iOS / Android / WASM consumers.
- [opcodes/](opcodes) — pre-rendered DNG `OpcodeList3` flat-fielding blobs for Merrill bodies.
- [docs/](docs) — the [mdbook](https://rust-lang.github.io/mdBook/) (pipeline, format
  reference, FFI surface, performance, contributor guide).

## License

Licensed under the Apache License, Version 2.0 — see [LICENSE](LICENSE).

This project is based on the work from the [Kalpanika x3f project](https://github.com/Kalpanika/x3f). It is **not endorsed by nor affiliated with** that project. Attribution for the original BSD-licensed work it derives from is retained in [NOTICE](NOTICE).
