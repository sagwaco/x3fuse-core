# x3fuse-core

`x3fuse-core` decodes Sigma's Foveon X3F raw photo format and writes DNG, 16-bit
TIFF, PPM, JPEG thumbnails, and metadata dumps. It targets every Foveon
sensor Sigma has shipped — the original SD9/SD10 generation, the DP- and
SD1-class Merrill bodies, and the Quattro family (DP\*Q / SDQ / SDQH).

This book is the long-form companion to the source tree. The
[`README.md`](https://github.com/sagwaco/x3fuse-core/blob/master/README.md)
in the repository root is the 30-second overview; this book is what to
read once you actually want to change something.

## Status

The original ~10K-LOC C/C++ codebase has been fully ported to a Cargo
workspace; the only non-Rust code left is an optional OpenCV-backed
denoise pass — and even that has a portable pure-Rust Non-Local Means
fallback ([`crates/x3f-sys/src/denoise.rs`](../../crates/x3f-sys/src/denoise.rs))
that takes over on every target without an opencv-mobile prebuilt (wasm,
offline/docs.rs, unsupported triples), so denoise works everywhere. The
[port plan](./port-plan.md) is retained as a historical record of how the
port proceeded.

## Build

```sh
cargo build --release
target/release/x3f_extract -dng input.X3F
```

Only prerequisite is a Rust toolchain (rustup). Output formats (PPM,
TIFF, DNG, JPEG thumbnail extract, metadata text dump, histogram CSV)
are pure Rust as of M3 — no system `libtiff` / `libjpeg` / `zlib`
needed. The denoise path links a pinned prebuilt `opencv-mobile`
(auto-fetched by `build.rs`) where one exists, and otherwise falls back to
the portable pure-Rust NLM — so there are still no mandatory system deps,
and the rest of the pipeline is libc-only.

The binary preserves the legacy single-dash flag syntax (`-tiff`,
`-no-denoise`, `-color sRGB`, …) so existing scripts and the test
corpus continue to work.

## Test

```sh
cargo test --workspace                  # all three test tiers
cargo test --workspace --lib            # tier 1 only (no corpus needed)
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
```

Tier-2 (MD5-pinned bit-exact) and tier-3 (perceptual diff) tests need
an X3F corpus that is **not committed** to the repo. They auto-skip
with a one-line notice when the corpus is absent — so a clean
checkout passes CI without setup. See the
[contributor guide](./contributing.md#test-corpus) for how to point
the harness at a corpus you have locally.

## Where to read next

- [Conversion pipeline](./pipeline.md) — how an X3F file becomes a
  DNG/TIFF/PPM, stage by stage.
- [Workspace layout](./workspace.md) — the four-crate split and which
  C source files survive.
- [X3F format reference](./format.md) — the on-disk layout: header,
  directory, sections, CAMF, entropy.
- [FFI surface](./ffi.md) — the C ABI used by iOS / Android / WASM
  consumers; xcframework and wasm cdylib build instructions.
- [Performance notes](./performance.md) — what the hot loops are and
  what was parallelised, vectorised, or left scalar.
- [Contributor guide](./contributing.md) — port conventions, parity
  gates, env-var deprecation policy, test corpus.
- [Port plan](./port-plan.md) — milestone breakdown. Read before
  opening a non-trivial PR so your work lands in the right milestone.
