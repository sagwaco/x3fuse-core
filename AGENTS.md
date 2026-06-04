# AGENTS.md

This file provides guidance to Agents like Claude Code (claude.ai/code) when working with code in this repository.

## Project: x3fuse-core

`x3fuse-core` converts Sigma Foveon X3F raw files to DNG / TIFF / PPM / JPEG / metadata. It began as a ~10K-LOC C/C++ codebase and has been **fully ported to a Rust Cargo workspace**; the only non-Rust code left is an optional OpenCV-backed denoise pass (in [crates/x3f-sys/csrc/](crates/x3f-sys/csrc/)) — which now has a portable pure-Rust Non-Local Means fallback ([crates/x3f-sys/src/denoise.rs](crates/x3f-sys/src/denoise.rs)) that takes over on every target without an opencv-mobile prebuilt (wasm, offline/docs.rs, unsupported triples), so denoise works everywhere. Read [ARCHITECTURE.md](ARCHITECTURE.md) (pipeline + workspace) before non-trivial work; [docs/PORT-PLAN.md](docs/PORT-PLAN.md) is retained as a historical record of the port.

### Build, test, lint

```sh
cargo build --release                                   # produces target/release/x3f_extract
target/release/x3f_extract -dng input.X3F               # legacy single-dash flag syntax preserved

cargo test --workspace                                  # all three test tiers
cargo test --workspace --lib                            # tier 1 only (no corpus needed)
cargo test --workspace -- --nocapture                   # show which tier-2/3 tests skipped

cargo fmt --all --check                                 # CI gate
cargo clippy --workspace --all-targets -- -D warnings   # CI gate
```

### Workspace shape

```
x3f-cli  ──depends-on──▶  x3f-core  ──depends-on──▶  x3f-sys  ──FFI──▶  csrc/ (OpenCV denoise + shims)
                                                              └── src/denoise.rs (portable Rust NLM fallback)
```

`x3f-sys` is the low-level layer where the `cc-rs` build of the remaining C/C++ lives ([crates/x3f-sys/build.rs](crates/x3f-sys/build.rs), sources in [crates/x3f-sys/csrc/](crates/x3f-sys/csrc/)). `x3f-core` is the safe Rust API outside consumers depend on. `x3f-cli` is the binary.

**Finding the right file.** The `x3f-sys` modules use milestone-prefixed doc comments (`M4d`, `M5b`, `M6e`…) that hide what they do — use the per-module map in [docs/src/workspace.md](docs/src/workspace.md) (each Rust module is mapped to the original C file it replaced) rather than grepping. The pipeline runs through `x3f-core`'s [`Reader`](crates/x3f-core/src/lib.rs): `Reader::open`/`from_bytes` → load sections → `dump_dng`/`dump_tiff`/`dump_ppm`/`dump_meta`/`dump_histogram` (1:1 with the legacy C entry points). Output writers are pure Rust under [crates/x3f-core/src/output/](crates/x3f-core/src/output/) (the DNG writer is the largest subsystem). See [docs/src/pipeline.md](docs/src/pipeline.md) for the stage-by-stage flow.

### Conventions

- **Highlight-recovery is the actively-iterated hot loop.** The chroma LUT, RepairPix, and matrix-pathology gate live in [crates/x3f-sys/src/highlight.rs](crates/x3f-sys/src/highlight.rs) and [process.rs](crates/x3f-sys/src/process.rs). Don't refactor while changing behavior; pair-review changes there.
- **Byte-identical parity is the gate.** Tier-2 MD5s on three baselines (SD1M `dcaa9929…`, older raw `_SDI8040` `41a80ce6…`, Quattro `_SDI8284` `c2f70f35…`) must match unless a change is an _intentional_ algorithm change.
- **Legacy CLI flag syntax is preserved.** Single-dash flags (`-dng`, `-tiff`, `-color sRGB`, `-no-denoise`, …) are kept verbatim so existing scripts and the test corpus continue to work. A modern subcommand interface is deferred.
- **`X3F_*` env vars are preserved.** Tunables like `X3F_NO_CHROMA_LUT`, `X3F_REPAIR_PIX`, `X3F_EV`, `X3F_GATE_THR`, `X3F_GATE_WIDTH`, `X3F_CHROMA_LUT_TRACE` keep working.
- **FFI ABI for the denoise boundary stays stable.** The one struct that still crosses into C++ (`x3f_area16_t`) is mirrored with `#[repr(C)]`; bindgen generates the rest of the layouts from the headers in `csrc/`. Heap allocations made on one side and freed on the other use `libc::malloc`/`free`.

### Test corpus

Tier-2 (MD5) and tier-3 (perceptual) tests need an X3F corpus that is **not committed** (large + some files non-redistributable). They look for it at `$X3F_TEST_FILES` (env var) or `<workspace_root>/x3f_test_files/` (default). Tests skip silently with a one-line notice when corpus is missing — `cargo test --workspace` works on a clean checkout. See [crates/x3f-cli/tests/README.md](crates/x3f-cli/tests/README.md) for the three-tier design.
