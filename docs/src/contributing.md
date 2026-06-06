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

Tier-2 MD5s on three baselines must match across a port step unless
the step is an _intentional_ algorithm change:

- SD1M (`sigma_sd1_merrill_15.x3f`) DNG `dcaa9929…`
- older raw (`_SDI8040.X3F`) DNG `41a80ce6…`
- Quattro (`_SDI8284.X3F`) DNG `c2f70f35…`

TIFF baselines:

- SD1M `277cf4b4…`, `_SDI8040` `b4cc09aa…`, `_SDI8284` `661df021…`
  (no-denoise) → `89a447e6…` / — / `3b24cdc3…` (denoise on, M9).

Most milestones in [the port plan](./port-plan.md) document the
expected MD5s. When a hash _should_ change (you're shipping an
intentional algorithm tweak), call it out in the commit message and
update the test expectation as part of the same change.

**The denoise output is _not_ part of the byte-parity gate.** Every
tier-2/tier-3 test runs with `-no-denoise`, so the MD5 baselines above
don't constrain the denoise kernels at all. The portable pure-Rust NLM in
[`crates/x3f-sys/src/denoise.rs`](../../crates/x3f-sys/src/denoise.rs)
(used wherever opencv-mobile isn't linked — wasm, offline builds, …) is a
faithful but deliberately _not_ byte-identical reimplementation of
opencv-mobile's `fastNlMeansDenoising` (floating-point `exp`, INTER_AREA /
INTER_CUBIC rounding differ). On real images it tracks OpenCV to ~99.98%
of bytes. If you need to A/B the two on a platform that has both, set
`X3F_PORTABLE_DENOISE=1` to force the Rust path on an OpenCV build.

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
- **Older raw** (SD9 / SD10 / SD14 / SD15 / DP1 / DP2) — `_SDI8040.X3F`
  is in the tier-2 expectations. SD9 / SD10 entropy paths are ported
  but not sensor-validated; adding even one SD9/SD10 file would
  extend the M5b differential test to cover them.
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
