# Workspace layout

```
x3f/
├── Cargo.toml                      # workspace
├── rust-toolchain.toml             # pinned to stable, components: rustfmt + clippy
├── crates/
│   ├── x3f-sys/                    # FFI bindings + native Rust pipeline + csrc/ (denoise, shims, headers)
│   ├── x3f-core/                   # safe Rust API (RAII Reader, ProcessOptions, output writers)
│   ├── x3f-cli/                    # `x3f_extract` binary
│   └── x3f-ffi-c/                  # cbindgen-generated C ABI for iOS / Android / WASM
├── opcodes/                        # pre-rendered DNG OpcodeList3 lens-correction blobs
├── docs/                           # this book + the port plan
└── x3f_test_files/                 # corpus (gitignored; user-supplied)
```

Dependency direction:

```
x3f-cli  ──▶  x3f-core  ──▶  x3f-sys  ──FFI──▶  csrc/x3f_printf.c, csrc/x3f_version.c
                  ▲                              (denoise is pure Rust: src/denoise.rs)
                  │
              x3f-ffi-c  (cbindgen → libx3f.{a,dylib,so,wasm})
```

`x3f-sys` is the low-level layer. Outside Rust consumers depend on
`x3f-core`. Outside C/iOS/Android/WASM consumers depend on
`x3f-ffi-c`.

## `crates/x3f-sys`

Started life as the cc-rs + bindgen wrapper around the original C
sources. Every function body has since been ported into Rust modules
under [`crates/x3f-sys/src/`](../../crates/x3f-sys/src/). The remaining
C lives in [`crates/x3f-sys/csrc/`](../../crates/x3f-sys/csrc/),
linked but not central:

- [`csrc/x3f_printf.c`](../../crates/x3f-sys/csrc/x3f_printf.c) — logging
  shim. Routes C-side log output through a Rust callback (set by
  `x3f_core::globals::set_log_callback`) so mobile/WASM consumers can
  redirect to NSLog / `__android_log_print` / `console.log`.
- [`csrc/x3f_version.c`](../../crates/x3f-sys/csrc/x3f_version.c) — version string.

Denoise is the pure-Rust Non-Local Means in
[`src/denoise.rs`](../../crates/x3f-sys/src/denoise.rs), called directly by the
pipeline on every target — no C/C++ is compiled for it and there is no
build-time download. (OpenCV / opencv-mobile was removed; earlier revisions
linked a prebuilt opencv-mobile static library on supported targets and fell
back to this Rust implementation elsewhere.)

The Rust modules under
[`crates/x3f-sys/src/`](../../crates/x3f-sys/src/) are a thin native
mirror of what used to be in `src/`:

| Module | Replaces | Milestone |
|--------|----------|-----------|
| `entropy.rs` | TRUE / Huffman / simple decoders in `x3f_io.c` | M5b–d |
| `highlight.rs` | reconstruct_highlights, chroma LUT, RepairPix, sat-map, matrix gate | M6e4 |
| `histogram.rs` | `x3f_histogram.c` | M6b |
| `image.rs` | `x3f_image.c` (image-area accessors, cropping) | M6c |
| `io.rs` | `x3f_io.c` directory walker, header parse, cleanup, getters | M4c, M5e |
| `load.rs` | `x3f_load_data` + section loaders + CAMF decoders | M4d |
| `matrix.rs` | `x3f_matrix.c` (3×3 mul, color-space matrices, gamma LUT) | M6a |
| `meta.rs` | `x3f_meta.c` (CAMF / property accessors) | M4a |
| `print_meta.rs` | `x3f_print_meta.c` (tier-2-MD5-pinned text dump) | M4b |
| `process.rs` | `x3f_process.c` master pipeline (preprocess, convert, expand, denoise) | M6e1–10 |
| `quattro.rs` | Quattro 2×2 expansion (replaces M0 `exit(2)` stub) | M5a |
| `denoise.rs` | pure-Rust NLM denoise (replaced the OpenCV `x3f_denoise.cpp`) | — |
| `spatial_gain.rs` | `x3f_spatial_gain.c` | M6d |
| `sysabi.rs` | wasm32-unknown-unknown libc shim (allocator + no-op file I/O) | M8d-α |
| `wasm_c_shims.rs` | Rust no-op shim for the variadic `x3f_printf` on wasm32 | M8d-α-2 |
| `histogram_wasm_stub.rs`, `print_meta_wasm_stub.rs` | wasm32 fallbacks (no variadics) | M8d-α |

Most of the ported modules export `#[no_mangle] extern "C"` symbols so
the bindgen forward declarations resolve at link time without churn
in callsites — even partially-ported files compile cleanly because
the Rust definitions are blocklisted out of bindgen and re-exported
through `lib.rs`.

## `crates/x3f-core`

Safe, idiomatic Rust API. The headline type is
[`Reader`](../../crates/x3f-core/src/lib.rs), which RAII-owns the
`x3f_t*` and the underlying `FILE*`. Two constructors:

- `Reader::open(path)` — host filesystem path.
- `Reader::from_bytes(&[u8])` — in-memory buffer (uses libc's
  `fmemopen` on Unix-likes, an internal `MemFile` cursor in the
  wasm32 sysabi shim). The same Rust call sequence runs on host and
  on wasm.

`Reader::dump_*` methods (`dump_dng`, `dump_tiff`, `dump_ppm`,
`dump_jpeg`, `dump_meta`, `dump_histogram`) correspond 1:1 with the
legacy C entry points and produce the conversion outputs.

Submodules:

- [`output/`](../../crates/x3f-core/src/output/) — pure-Rust DNG, TIFF,
  and PPM writers (M3).
- [`globals.rs`](../../crates/x3f-core/src/globals.rs) — wraps C-side
  mutable globals (`x3f_printf_level`, `legacy_offset`,
  `max_printed_matrix_elements`) in safe setters, plus
  `set_log_callback` for mobile/WASM logging.
- [`image.rs`](../../crates/x3f-core/src/image.rs) — `Image` struct
  that snapshots `DNG_HIGHLIGHT_SCALE` immediately after
  `x3f_get_image` returns (M5e fix for batch determinism).

## `crates/x3f-cli`

Hand-rolled argument parser plus a `convert_one` per file. Single-dash
flag syntax is preserved verbatim so existing test corpora and shell
scripts keep working through the port. Batch mode parallelises across
the file list via `rayon::par_iter` (M7b).

A modern subcommand interface (`x3f extract`, `x3f info`, `x3f
config`) and a typed `ProcessingConfig` derived via clap is planned
post-port; for now the legacy interface is preserved.

## `crates/x3f-ffi-c`

The C ABI for non-Rust consumers. `crate-type = ["staticlib",
"cdylib", "rlib"]` produces `libx3f.{a,dylib,so,dll,wasm}`; a
`build.rs` invokes `cbindgen` to emit `x3f.h` into both `OUT_DIR` and
`target/<profile>/include/`. See the [FFI surface](./ffi.md) chapter
for the symbol list and per-platform build instructions.
