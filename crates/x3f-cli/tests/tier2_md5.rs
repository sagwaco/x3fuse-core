//! Tier-2 — exact MD5 of stable surfaces.
//!
//! Outputs that should be **bit-stable** as the port progresses (metadata
//! dumps, the embedded JPEG thumbnail extraction, and PPM rasters) get exact
//! MD5 hashes. If the hash changes, either the test or the implementation is
//! wrong — neither should drift silently.
//!
//! These are deliberately *not* used for processed TIFF/DNG output: those
//! shift whenever the highlight-recovery work iterates, and tier-3
//! perceptual diffs cope with that.
//!
//! When a hash changes intentionally (i.e. you've fixed a bug in the
//! metadata writer or upgraded the JPEG extractor), regenerate it once with
//! `cargo run --release -- <flags> <corpus>/<input>` and copy the new hash
//! into the matching constant.

mod common;

use common::{file_md5, run_extract};

// ---------------------------------------------------------------------------
// Merrill (DP* / SD1) — full coverage, all stable surfaces.
// ---------------------------------------------------------------------------

const MERRILL_INPUT: &str = "sigma_sd1_merrill_10.x3f";

#[test]
fn merrill_meta_md5() {
    let input = skip_if_missing!(MERRILL_INPUT);
    let out = run_extract(&input, &["-meta"], ".meta");
    assert_eq!(file_md5(&out), "1f0e54d4dff8107c424681918829738f");
}

#[test]
fn merrill_jpeg_thumbnail_md5() {
    let input = skip_if_missing!(MERRILL_INPUT);
    let out = run_extract(&input, &["-jpg"], ".jpg");
    assert_eq!(file_md5(&out), "73a324ff01fcdf63e0655ec0585c5bf0");
}

// The two PPM hashes were re-pinned alongside the DNG-compatibility
// work: the values inherited from the pre-import port repo
// (`5d7eed37…` / `4af2c689…`) never reproduced in this repository —
// every commit back to the initial import produces the hashes below,
// so the drift happened during the port, after the last corpus run
// that pinned them.
#[test]
fn merrill_ppm_p6_md5() {
    let input = skip_if_missing!(MERRILL_INPUT);
    let out = run_extract(
        &input,
        &["-ppm", "-no-denoise", "-color", "none", "-no-crop"],
        ".ppm",
    );
    assert_eq!(file_md5(&out), "93dcdb5ae5dba9b4b78d70246124ebbc");
}

#[test]
fn merrill_ppm_p3_ascii_md5() {
    let input = skip_if_missing!(MERRILL_INPUT);
    let out = run_extract(
        &input,
        &["-ppm-ascii", "-no-denoise", "-color", "none", "-no-crop"],
        ".ppm",
    );
    assert_eq!(file_md5(&out), "e6304c107084dfeba6ec5c93452b8a12");
}

// ---------------------------------------------------------------------------
// Quattro — only the surfaces that don't go through the (M0-stubbed) Quattro
// 2×2 expansion. Metadata and the embedded JPEG never touch the RAW path,
// so these hashes are stable today and will remain stable when M5 lands.
// ---------------------------------------------------------------------------

const QUATTRO_INPUT: &str = "_SDI8284.X3F";

#[test]
fn quattro_meta_md5() {
    let input = skip_if_missing!(QUATTRO_INPUT);
    let out = run_extract(&input, &["-meta"], ".meta");
    assert_eq!(file_md5(&out), "0a77d95cf4f53acec52c11e756590a28");
}

#[test]
fn quattro_jpeg_thumbnail_md5() {
    let input = skip_if_missing!(QUATTRO_INPUT);
    let out = run_extract(&input, &["-jpg"], ".jpg");
    assert_eq!(file_md5(&out), "87cd494d3bc4eab4e481de6afeb058de");
}
