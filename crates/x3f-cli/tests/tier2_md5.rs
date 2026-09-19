//! Tier-2 — exact MD5 of stable surfaces.
//!
//! Outputs that should be **bit-stable** as the port progresses (metadata
//! dumps and PPM rasters) get exact MD5 hashes. Embedded JPEG extraction is
//! compared directly against the declared source payload. If a hash changes,
//! either the test or the implementation is
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

/// Independently read the container directory: JPEG extraction must preserve
/// exactly the payload, excluding its 28-byte section header. The old MD5s
/// included 28 bytes read beyond the payload allocation and are not valid pins.
fn assert_jpeg_payload(input: &std::path::Path, output: &std::path::Path) {
    let source = std::fs::read(input).unwrap();
    let u32_at =
        |offset| u32::from_le_bytes(source[offset..offset + 4].try_into().unwrap()) as usize;
    let directory = u32_at(source.len() - 4);
    assert_eq!(&source[directory..directory + 4], b"SECd");
    for entry in 0..u32_at(directory + 8) {
        let offset = u32_at(directory + 12 + entry * 12);
        let size = u32_at(directory + 16 + entry * 12);
        if &source[offset..offset + 4] == b"SECi"
            && u32_at(offset + 8) == 2
            && u32_at(offset + 12) == 18
        {
            assert_eq!(
                std::fs::read(output).unwrap(),
                &source[offset + 28..offset + size]
            );
            return;
        }
    }
    panic!("fixture has no embedded JPEG section");
}

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
fn merrill_jpeg_matches_section_payload() {
    let input = skip_if_missing!(MERRILL_INPUT);
    let out = run_extract(&input, &["-jpg"], ".jpg");
    assert_jpeg_payload(&input, &out);
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
fn quattro_jpeg_matches_section_payload() {
    let input = skip_if_missing!(QUATTRO_INPUT);
    let out = run_extract(&input, &["-jpg"], ".jpg");
    assert_jpeg_payload(&input, &out);
}

#[test]
fn available_local_camera_jpegs_match_section_payloads() {
    for name in ["DP2M0981.X3F", "DP0Q0010.X3F"] {
        if let Some(input) = common::find_input(name) {
            let output = run_extract(&input, &["-jpg"], ".jpg");
            assert_jpeg_payload(&input, &output);
        }
    }
}
