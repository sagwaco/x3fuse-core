# Test harness

The x3f Rust port is mid-migration: highlight-recovery research lives in
[../../../src/x3f_process.c](../../../src/x3f_process.c) and the user iterates
on it actively, while the rest of the pipeline is being ported module by
module. A naive bit-exact regression suite would conflate "intentional
algorithm tweak" with "port broke a parser" — the legacy MD5-only
behave harness retired in M2 had exactly that problem.

This directory holds a three-tier replacement that separates **stable
surfaces** (must be bit-exact) from **unstable surfaces** (must stay within a
perceptual tolerance) from **stage-local invariants** (must hold without
needing a real X3F file at all).

## Tier 1 — unit tests

In-crate `#[cfg(test)]` modules; no corpus needed. They cover everything
that can be exercised with hand-rolled inputs:

- [x3f-core](../../x3f-core/src/lib.rs) — `LibraryError::from_raw` mapping,
  `ColorEncoding::to_raw` mapping, `wb_cstring` interior-NUL handling.
- [x3f-cli](../src/main.rs) — argument parser (the legacy single-dash flag
  Z-macro behaviour, file-list semantics), `make_paths` table, `FileType`
  extension table.

As the port progresses, each ported module adds its own tier-1 tests:

- M3 — `x3f-core::matrix` — vectors ported from the original `x3f_matrix_test.c`.
- M3 — `x3f-format::ppm` — header parser, ascii vs binary edge cases.
- M4 — `x3f-format::container` — directory walker, property-list decoder.
- M5 — `x3f-format::huffman` — table builds, decode round-trips.
- M6 — `x3f-core::pipeline` — black-level math, gamma LUT generation.

Run with: `cargo test --workspace --lib` (or just `cargo test --workspace`,
which runs all three tiers).

## Tier 2 — MD5 of stable surfaces

[tier2_md5.rs](tier2_md5.rs) runs the freshly-built `x3f_extract` against
real X3F files and compares the output against an inline expected MD5.

Tier 2 is **deliberately not used for processed TIFF/DNG output**. Those
sit downstream of the highlight-recovery code, which the user iterates
on; a bit-exact gate would make every algorithm tweak look like a
regression. Stable surfaces are:

| Switch                      | Output           | Why it's stable                                   |
|-----------------------------|------------------|----------------------------------------------------|
| `-meta`                     | metadata dump    | parser-only, no pixel processing                   |
| `-jpg`                      | JPEG thumbnail   | byte-copy of an embedded JPEG                      |
| `-ppm -color none` + crop   | linear PPM       | RAW decode → no colorspace/gamma/highlight passes  |
| `-ppm-ascii -color none`    | ASCII PPM        | as above                                            |

When a hash changes intentionally (bug fix in metadata writer, JPEG path
upgrade), regenerate it once with `md5 <output>` and paste the new value
into the `assert_eq!` in `tier2_md5.rs`.

## Tier 3 — perceptual diff for TIFF/DNG

[tier3_perceptual.rs](tier3_perceptual.rs) decodes 16-bit RGB rasters and
compares them with `image_diff` from [common/mod.rs](common/mod.rs):

- `max_abs_diff` — biggest single-channel deviation.
- `samples_over_{8,64,512,4096}` — count of channel samples whose abs
  diff exceeds the threshold.

Today's tier-3 cases are tight bounds (zero divergence) used for
**self-consistency** properties — determinism, lossless compression,
structural sanity. They look like:

```rust
let a = run_extract(input, switches, ".tif");
let b = run_extract(input, switches, ".tif");
let diff = image_diff(&read_tiff_rgb16(&a), &read_tiff_rgb16(&b));
assert_eq!(diff.max_abs_diff, 0);
```

When M5/M6 introduce committed-side goldens (TIFF/DNG outputs from the
legacy C binary, stored alongside the input X3F), tier-3 grows
additional cases of the form:

```rust
let golden = read_tiff_rgb16(&corpus_dir.join(format!("{name}.golden.tif")));
let diff = image_diff(&actual, &golden);
assert!(diff.max_abs_diff < 256, "{diff:?}");
assert!(diff.samples_over_8 < img.total_samples() / 1000, "{diff:?}");
```

ΔE2000 in CIELAB will be added when we have a colorimetric reference to
compare against — for the linear-light intermediate stages tested today,
per-channel epsilon catches the same regressions without dragging in a
CIE conversion.

## Corpus discovery

All tier-2 and tier-3 tests need an X3F corpus that is **not committed**
to this repo (the files are large and some are not redistributable). The
tests look for it in:

1. `$X3F_TEST_FILES` env var, if set.
2. `<workspace_root>/x3f_test_files/` otherwise.

If the corpus is missing **or** a specific input file isn't present, the
affected test prints a one-line "skip" notice and returns successfully.
This means `cargo test --workspace` works on a clean checkout for CI;
add the corpus locally to get full coverage. Run with
`cargo test -- --nocapture` to see which tests skipped.

The current minimum viable corpus:

- **Merrill** (DP\* / SD1) — full coverage. The tests pin
  `sigma_sd1_merrill_10.x3f`.
- **Quattro** (DP\* / SDQH) — meta + jpeg only in M0; tests pin
  `_SDI8284.X3F`. RAW-path coverage returns in M5.

## Adding a new test case

1. Pick an input with a property worth pinning (e.g. clipped highlights, a
   particular sensor model, a metadata edge case).
2. Decide the tier:
   - **Tier 2** if the output should be byte-identical (metadata, jpeg,
     unprocessed PPM).
   - **Tier 3** if the output is downstream of the highlight code or the
     color matrix.
3. Run `x3f_extract <flags> <input>` once locally to capture the
   expected hash or to write the golden image, then commit the test.
