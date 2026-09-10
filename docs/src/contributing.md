# Contributor guide

This codebase is mid-port from C/C++ to Rust. Most patches land in the
Rust modules, but the way they need to land is shaped by the port —
**byte-identical parity** is the gate for stable surfaces, and
**verbatim-first, tidy-second** is the rule for anything in the
highlight-recovery family.

If you're new to the project, also read:

- [`README.md`](https://github.com/sagwaco/x3fuse-core/blob/master/README.md)
  in the repo root — 30-second project overview and build.
- [`ARCHITECTURE.md`](https://github.com/sagwaco/x3fuse-core/blob/master/ARCHITECTURE.md)
  — the slim entry pointer; the full architecture lives across the
  [pipeline](./pipeline.md) and [workspace](./workspace.md) chapters
  of this book.
- [Port plan](./port-plan.md) — milestone breakdown. Land changes in
  the right milestone.

## Style + tooling

Standard Rust hygiene is enforced in CI:

```sh
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo build --workspace --all-targets
cargo test --workspace
```

Match existing style — even when you'd write it differently — to keep
the diffs small enough that parity-validation MD5s stay
interpretable. Don't refactor adjacent code unless your change makes
it dead.

## Port conventions

A handful of conventions show up over and over in the milestone
notes. Internalising them is the difference between a one-comment PR
and a multi-round review.

### Port verbatim first; tidy in a separate PR

Especially in
[`crates/x3f-sys/src/highlight.rs`](../../crates/x3f-sys/src/highlight.rs)
and the surrounding `process.rs` orchestrator. Highlight-recovery
research lives there — chroma LUT, RepairPix, matrix-pathology gate
— and is actively iterated. A "while I'm here" cleanup that reorders
operations or coalesces branches will silently change the gate
ordering and break parity in ways MD5s catch but ΔE doesn't (or vice
versa).

The C bug in
[`crates/x3f-sys/src/spatial_gain.rs`](../../crates/x3f-sys/src/spatial_gain.rs)
(`x3f_calc_spatial_gain`'s `ci<0` branch missing an `else if`) is
preserved verbatim with a comment. So is the firmware bug in
`Jpeg_BadClusters` (the row/col swap). Don't fix.

### Byte-identical parity is the gate

Two layers of MD5 baselines, one automated and one manual:

**Automated (tier-2,
[`crates/x3f-cli/tests/tier2_md5.rs`](../../crates/x3f-cli/tests/tier2_md5.rs)):**
metadata dumps, the embedded JPEG thumbnail, and PPM rasters are pinned
to exact hashes. Processed TIFF/DNG output is deliberately *not* in
tier-2 — it shifts whenever the highlight-recovery work iterates;
tier-3 perceptual diffs cover it.

**Manual (run before merging anything that touches the pipeline):**
three reference baselines, produced with

```sh
x3f_extract -dng  -no-denoise <input>   # DNG column
x3f_extract -tiff -no-denoise <input>   # TIFF column
```

| Input | DNG | TIFF |
| ----- | --- | ---- |
| SD1M (`sigma_sd1_merrill_15.x3f`) | `a2427c1db46066d7201f7499d46bb02d` | `277cf4b4691652bd57c96b15ba03d47f` |
| DP2 Merrill (`_SDI8040.X3F`) | `94c64ebc524077d4f51e96df868d9271` | `b4cc09aa1c8d127274056660a92ffc0d` |
| Quattro (`_SDI8284.X3F`) | `7259dd2e0b4c5cde7e4e5c2531defa9f` | `661df021b16de5164b03624776fd5507` |

These must match across a change unless the change is an _intentional_
algorithm change — in which case re-pin this table in the same commit
and call the change out in the commit message.

The September 2026 DNG compatibility correction intentionally changes
these hashes. Relative `DigitalISOGain` is retained in the color metadata,
calibration is folded into each profile's ColorMatrix, and invalid or
misplaced tags are corrected. Quattro's intermediate ranges now include
relative digital gain so its reconstructed neutral detail stays neutral.
That processing correction changes pixels in files with unequal digital
gains, including the supplied sd Quattro H images. Uniform digital gains
preserve the previous range arithmetic exactly, so all three TIFF hashes
above remain unchanged. Compare image payloads separately from whole-file
hashes, and distinguish actual reader color checks from structural DNG
validation.

(History: the DNG hashes cited by the port-plan milestones —
`dcaa9929…` / `41a80ce6…` / `c2f70f35…` — are milestone-era values
that no longer reproduce: intentional DNG-writer changes since then
(highlight-recovery iterations, active-area cropping, and most
recently the per-channel level-equalization bake) each moved the
bytes without re-pinning the docs. The table above is the current
re-pin. The "denoise on, M9" TIFF hashes `89a447e6…` / `3b24cdc3…`
are also historical: they were produced by the old opencv-mobile NLM,
and denoise output is not part of the byte-parity gate.)

**The denoise output is _not_ part of the byte-parity gate.** Every
tier-2/tier-3 test runs with `-no-denoise`, so the MD5 baselines above
don't constrain the denoise kernels at all. Denoise is the pure-Rust NLM in
[`crates/x3f-sys/src/denoise.rs`](../../crates/x3f-sys/src/denoise.rs),
used on every target. It is a faithful but deliberately _not_ byte-identical
reimplementation of the original opencv-mobile `fastNlMeansDenoising`
(floating-point `exp`, INTER_AREA / INTER_CUBIC rounding differ); on real
images it tracked OpenCV to ~99.98% of bytes before OpenCV was removed.

### Legacy CLI flag syntax is preserved

Single-dash flags (`-dng`, `-tiff`, `-color sRGB`, `-no-denoise`, …)
are kept verbatim through the port so existing scripts and the test
corpus continue to work. A modern subcommand interface is deferred
until post-port.

### `X3F_*` env vars are preserved through one deprecation cycle

Tunables like `X3F_NO_CHROMA_LUT`, `X3F_REPAIR_PIX`, `X3F_EV`,
`X3F_GATE_THR`, `X3F_GATE_WIDTH`, `X3F_CHROMA_LUT_TRACE`,
`X3F_RUST_DECODE` keep working when ported to a typed config. Read
existing names through `from_env()`-style adapters; don't break them
without a deprecation cycle.

### FFI ABI stability across half-ported modules

Symbols moving from C to Rust use `#[no_mangle] extern "C"` with
`#[used]` anchors so cross-crate dead-code elimination doesn't strip
them before C call sites in remaining `.c` files link. Bindgen
blocklists the C name; the Rust definition is re-exported through
[`crates/x3f-sys/src/lib.rs`](../../crates/x3f-sys/src/lib.rs) under
the same `x3f_sys::x3f_*` path so call-site code in `x3f-core`
doesn't churn.

Layouts of any struct that crosses the boundary use `#[repr(C)]`
plus size and alignment asserts (look for `const _: () =
assert!(size_of::<T>() == ...)` in the modules). Heap allocations
made on one side and freed on the other use `libc::malloc` /
`libc::free` so the partial-port pairing stays valid even when
allocation site and free site are in different languages.

By the end of M5e the cleanup machinery is also in Rust, so new
`Vec<u16>` / `Box<[T]>` allocations are safe — but be careful when
porting code whose buffer is still freed by a leftover C path
(grep for the symbol's call sites before changing allocators).

### No "improving" adjacent code

Touch only what you must. Clean up orphan imports / variables
introduced by your change; leave pre-existing dead code alone unless
removing it is the request. The diff that should land is "every
changed line traces directly to the request."

### Tier-3 perceptual diff for processed TIFF/DNG

Tier-3 cases use `image_diff` from
[`crates/x3f-cli/tests/common/mod.rs`](../../crates/x3f-cli/tests/common/mod.rs)
and assert on `max_abs_diff` + `samples_over_{8,64,512,4096}`. The
tightest cases (zero divergence) are _self-consistency_ checks —
running the same input twice produces byte-identical output. Looser
cases use ΔE-shaped per-channel epsilon thresholds.

When highlight-recovery research lands, expect the tight bounds to
loosen for affected images. Document the loosening in the commit
message; don't quietly bump the threshold to the new max.

The highlight mapping tests in
[`tier3_highlight_recovery.rs`](../../crates/x3f-cli/tests/tier3_highlight_recovery.rs)
decode the **raw SubIFD**, not the IFD0 preview. They check recovered
headroom, exposure compensation, shared-scale midtone precision, preserved
highlight layer ratios, and distinct recovered intensity levels on
`CLIPPED_IMAGE_MERRILL.X3F` and `DP2M0981.X3F`. Recovery-off output must
remain byte-identical when the mapping selector changes, including SD1
Merrill, DP2 Merrill, and Quattro controls. Each child process clears inherited
`X3F_*` research tunables; corpus discovery happens before that cleanup.

```sh
cargo test -p x3f-cli --release --test tier3_highlight_recovery -- --nocapture
```

These are structural and numerical invariants, not proof of improved
appearance relative to Sigma Photo Pro. For a reference comparison, render
the DNG raw data with a fixed reader/profile, export at 0 EV and -2 EV,
align crop/orientation, and convert both that output and the SPP TIFF's
embedded ICC profile into the same linear color space. Compare fixed
highlight regions for surviving texture, color continuity, and edge halos;
also inspect unclipped regions. Neither an embedded-preview comparison nor
a direct BMT-versus-rendered-RGB error establishes recovery quality.
Keep the SPP exports outside the repository alongside the private corpus.
The supplied reference pairs use `CLIPPED_IMAGE_MERRILL_SPP_0EV.tif` /
`CLIPPED_IMAGE_MERRILL_SPP_-2EV.tif` and `DP2M0981-SPP_0EV.tif` /
`DP2M0981-SPP_-2EV.tif`.

The local reconstruction and linear headroom mapping intentionally change
recovery-enabled DNG pixels. Quattro keeps its existing reconstruction;
its final mapping can still change the encoded pixels. Do not update the
manual baseline table from guesses or from recovery-enabled runs: measure
the exact documented recovery-off commands, and record intentional changes
separately from stale historical file hashes.

Include matrix-pathology cases in highlight validation. Foveon layers can
retain nominal headroom while their combined color projection is already
unreliable, so a hard-clipping mask alone is insufficient. Exercise
near-clipped BMT vectors with severe color-matrix cancellation as well as
normal consistent vectors. Also check local proposals against the established
reconstruction in rendered chroma: good donor support must not permit a
large hue change. When chroma remains uncertain, survivor-derived scalar
texture can improve brightness structure while retaining established color.
The safeguard may change all layers of a
pathological highlight together to preserve coherent color and available
brightness detail; preserving each nominally unclipped layer is not an
invariant in that case. Check that normal valid unclipped areas remain
unchanged, and inspect real sky and cloud crops for green/magenta artifacts.

## Test corpus

Tier-2 (MD5) and tier-3 (perceptual) tests need an X3F corpus that is
**not committed** to the repo (the files are large and some are not
redistributable). The harness looks for it in:

1. `$X3F_TEST_FILES` env var, if set.
2. `<workspace_root>/x3f_test_files/` otherwise.

If the corpus is missing **or** a specific input file isn't present,
the affected test prints a one-line "skip" notice and returns
successfully — `cargo test --workspace` works on a clean checkout.
Run with `cargo test -- --nocapture` to see which tests skipped.

Minimum viable corpus, by sensor class:

- **Merrill** (DP\* / SD1) — the tier-2 tests pin
  `sigma_sd1_merrill_10.x3f` and `sigma_sd1_merrill_15.x3f`. Fully
  exercised.
- **Older raw** (SD9 / SD10 / SD14 / SD15 / DP1 / DP2) — not covered by
  the three manual reference files above: the supplied `_SDI8040.X3F`
  identifies as DP2 Merrill. SD9 / SD10 entropy paths are ported but not
  sensor-validated; adding even one SD9/SD10 file would extend the M5b
  differential test to cover them.
- **Quattro** (DP\* / SDQ / SDQH) — `_SDI8284.X3F` covers the SDQH
  path; DP0Q files cover the DP-class Quattro.

See
[`crates/x3f-cli/tests/README.md`](../../crates/x3f-cli/tests/README.md)
for the three-tier test design and how to add a new test case.

## Known regressions / follow-ups

Tracked in the [port plan](./port-plan.md) under "Known regressions".
The current open item is **DNG picture profiles not visible to
downstream consumers** (reported 2026-04-27): the Rust DNG writer
emits all six camera profiles and exiftool confirms the bytes are
present, but Lightroom's profile dropdown no longer shows them
post-port. Suspected cause: subtle structural difference in our
hand-rolled MMCR mini-TIFFs vs libtiff's output. Marked **not
critical** by the user, but should land before the M9 crates.io
publish.

## DNG reader compatibility checks

The September 2026 writer correction preserves per-channel digital ISO
gain in AsShotNeutral and ColorMatrix, folds CameraCalibration into all
enabled profiles, and fixes hue/saturation map dimensions and encoding,
crop types, camera identity, and IFD placement. The accompanying Quattro
working-range correction changes reconstructed pixels to remove false
color in bright and dark detail. The public Rust API, C ABI, and
single-dash CLI flags are unchanged. Applications bundling `x3f_extract`
or linking this library must rebuild with the corrected core.

Use an independent decoder in addition to the Rust tests. With the Adobe
SDK available locally, validate both uncompressed and lossless-JPEG output:

```sh
_local/dng_sdk_1_7_1/dng_sdk/targets/mac/debug64/dng_validate -d 5 output.dng
```

For clipped Foveon highlights, generate a recovery-enabled comparison:

```sh
target/release/x3f_extract -dng -compress -dng-highlight-recovery \
  -dng-highlight-mapping linear input.X3F
```

`linear` is the default mapping when recovery is enabled. Compare against
`-dng-highlight-mapping shoulder` in a separate output directory so both
artifacts remain available. Linear output preserves recovered contrast by
adding its shared encoding scale to `BaselineExposure`; verify exposure
edits and metadata handling in each reader. Readers that ignore this tag
need the corresponding positive exposure adjustment. Shoulder output bakes
the compression into its raster and is the compatibility alternative.
Recovery remains opt-in, with the existing Quattro reconstruction running
after expansion. Turning recovery off can retain lime/yellow clipped
highlights even when the color matrices and reconstructed detail are correct.

Inspect warnings and assertion messages as well as the exit status: the
old malformed DNGs could exit successfully. Exercise the default and each
enabled profile with the validator's `-profile` and `-tif` options. Compare
raw and preview strip payloads before and after any application metadata
transfer, such as X3Fuse's ExifTool `-tagsFromFile INPUT.X3F -all:all` step.

On macOS, the included Core Image check renders actual RAW pixels and
writes JPEG previews and a JSON report with decoder versions and dimensions:

```sh
swift scripts/validate_apple_dng.swift --output target/apple-validation \
  --scale 1 --edits output.dng
```

Use `--scale 0.125` for a batch smoke test. To test a particular decoder,
pass an exact supported raw value such as `--decoder 8.dng`; an unsupported
request fails without falling back. A successful RAW 8 render does not
establish RAW 9 support.

SDK success alone does not establish Lightroom compatibility. Verify the
rebuilt output in Lightroom's Local browser, including metadata display,
full-resolution rendering, exposure/white-balance edits, and export.
For RawTherapee, check both its default input profile and the explicitly
embedded profile with Camera white balance. Check darktable separately:
its ColorMatrix path does not exercise Adobe's ForwardMatrix transform.
Use scenes with blown white highlights, neutral shadow detail, and
saturated colors; successful decoding or an acceptable midtone average
does not establish correct color throughout the image. Default exposure
and tone can differ between readers; these checks do not establish
calibrated color accuracy or identical JPEG appearance.

## Filing a PR

- Reference the milestone (`M6e10`, `M7d`, etc.) in the commit
  subject — the port plan tracks milestones, and a PR that lands the
  wrong milestone is the most common source of churn.
- If the change is intended to be byte-identical, mention which
  baselines you've verified (MD5s).
- If it's intended to change output, call out the new tier-2 / tier-3
  expectations and _why_ they're an improvement, not a regression.

The full behaviour spec for AI-assisted work in this repo is in
[`AGENTS.md`](https://github.com/sagwaco/x3fuse-core/blob/master/AGENTS.md);
human contributors are welcome to follow the same conventions.
