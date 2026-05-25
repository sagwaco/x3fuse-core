// `tests/common/mod.rs` is compiled into every integration-test binary, but
// each test file only uses some of these helpers. Without this, every binary
// triggers dead-code warnings for the helpers it doesn't reach.
#![allow(dead_code)]

//! Shared scaffolding for the tier-2 and tier-3 integration tests.
//!
//! All tests run the freshly-built `x3f_extract` binary against an external
//! corpus that is *not* committed to the repository (raw files are large and
//! some are not redistributable). The corpus directory is discovered at
//! runtime; if it is absent the tests skip with a one-line note rather than
//! failing, so `cargo test --workspace` works on a clean checkout.
//!
//! Corpus discovery:
//!   1. `X3F_TEST_FILES` env var, if set.
//!   2. `<workspace_root>/x3f_test_files/`.

use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Absolute path to the freshly-built `x3f_extract` binary. Cargo guarantees
/// the binary is built before this test crate compiles.
pub const X3F_EXTRACT: &str = env!("CARGO_BIN_EXE_x3f_extract");

pub fn corpus_dir() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("X3F_TEST_FILES") {
        let p = PathBuf::from(p);
        return p.is_dir().then_some(p);
    }
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest.parent()?.parent()?.to_path_buf();
    let candidate = workspace_root.join("x3f_test_files");
    candidate.is_dir().then_some(candidate)
}

/// Resolve a corpus filename to an absolute path. Returns `None` if either
/// the corpus is missing or the named file isn't present — callers should
/// emit a skip notice and early-return.
pub fn find_input(name: &str) -> Option<PathBuf> {
    let path = corpus_dir()?.join(name);
    path.is_file().then_some(path)
}

/// Macro to skip an integration test when its required input isn't present.
/// Prints a one-line notice (visible with `cargo test -- --nocapture` or in
/// CI logs) so missing corpus is observable, then `return`s from the caller.
#[macro_export]
macro_rules! skip_if_missing {
    ($name:expr) => {{
        match $crate::common::find_input($name) {
            Some(p) => p,
            None => {
                eprintln!(
                    "  (skip: corpus file `{}` not found; set X3F_TEST_FILES to a directory containing it)",
                    $name
                );
                return;
            }
        }
    }};
}

/// Run `x3f_extract` with the given switches against a single input file in a
/// scratch directory. Returns the path to the produced output (whose
/// extension is determined by the format flag, e.g. `.dng` for `-dng`).
///
/// The output is written next to a *copied* input rather than into the
/// corpus, so concurrent test runs don't race on the same `.tmp` file and
/// we don't pollute the user's corpus with stale outputs.
pub fn run_extract(input: &Path, switches: &[&str], expected_ext: &str) -> PathBuf {
    run_extract_with_env(input, switches, &[], expected_ext)
}

/// As [`run_extract`], but also sets `(name, value)` env vars on the child
/// process. Used by the M5b differential decoder test which dispatches on
/// `X3F_RUST_DECODE`. The env vars are folded into the scratch-dir
/// fingerprint so two tests that differ only in env do not race.
pub fn run_extract_with_env(
    input: &Path,
    switches: &[&str],
    env: &[(&str, &str)],
    expected_ext: &str,
) -> PathBuf {
    let scratch = scratch_for_with_env(input, switches, env);
    let _ = fs::remove_dir_all(&scratch);
    fs::create_dir_all(&scratch).expect("create scratch dir");

    let staged_input = scratch.join(input.file_name().unwrap_or_else(|| OsStr::new("input.X3F")));
    fs::copy(input, &staged_input).expect("stage input file");

    let mut cmd = Command::new(X3F_EXTRACT);
    cmd.args(switches).arg(&staged_input);
    for (k, v) in env {
        cmd.env(k, v);
    }
    let status = cmd.status().expect("spawn x3f_extract");
    assert!(
        status.success(),
        "x3f_extract {switches:?} env={env:?} {} exited {status:?}",
        input.display()
    );

    let mut out = staged_input.into_os_string();
    out.push(expected_ext);
    let out = PathBuf::from(out);
    assert!(
        out.is_file(),
        "expected output `{}` to exist",
        out.display()
    );
    out
}

/// Hex-encoded MD5 of a file's full contents.
pub fn file_md5(path: &Path) -> String {
    use md5::{Digest, Md5};
    let bytes = fs::read(path).unwrap_or_else(|e| panic!("read `{}`: {e}", path.display()));
    let digest = Md5::digest(&bytes);
    digest.iter().map(|b| format!("{b:02x}")).collect()
}

/// A scratch directory for one test run, derived from the input filename and
/// a switch-string fingerprint. Living under `target/` means cargo cleans up
/// across `cargo clean` and concurrent test cases don't clobber each other.
fn scratch_for(input: &Path, switches: &[&str]) -> PathBuf {
    scratch_for_with_env(input, switches, &[])
}

fn scratch_for_with_env(input: &Path, switches: &[&str], env: &[(&str, &str)]) -> PathBuf {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut h = DefaultHasher::new();
    for s in switches {
        s.hash(&mut h);
    }
    for (k, v) in env {
        k.hash(&mut h);
        v.hash(&mut h);
    }
    let target = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    let stem = input
        .file_stem()
        .unwrap_or_else(|| OsStr::new("test"))
        .to_string_lossy();
    target.join(format!("{stem}-{:016x}", h.finish()))
}

/// Decode a 16-bit RGB TIFF (or uncompressed linear DNG) into an interleaved
/// planar buffer. Used by the perceptual-diff tests.
pub fn read_tiff_rgb16(path: &Path) -> RgbImage {
    use tiff::decoder::{Decoder, DecodingResult};
    let f = fs::File::open(path).unwrap_or_else(|e| panic!("open `{}`: {e}", path.display()));
    let mut dec = Decoder::new(f).expect("tiff decoder");
    let (w, h) = dec.dimensions().expect("tiff dimensions");
    let pixels = match dec.read_image().expect("tiff read_image") {
        DecodingResult::U16(v) => v,
        DecodingResult::U8(v) => v.into_iter().map(|x| u16::from(x) << 8).collect(),
        other => panic!("unsupported TIFF sample format: {other:?}"),
    };
    RgbImage {
        width: w,
        height: h,
        pixels,
    }
}

/// 16-bit interleaved RGB image. Pixel order is `[r0, g0, b0, r1, g1, b1, …]`.
pub struct RgbImage {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u16>,
}

/// Per-channel 16-bit difference statistics between two equally-sized images.
#[derive(Debug)]
pub struct ImageDiff {
    pub width: u32,
    pub height: u32,
    /// Maximum absolute per-channel difference seen anywhere.
    pub max_abs_diff: u16,
    /// Number of channel samples with abs diff > threshold, for several
    /// thresholds we care about. Index = log2 bucket.
    pub samples_over_8: usize,
    pub samples_over_64: usize,
    pub samples_over_512: usize,
    pub samples_over_4096: usize,
}

impl ImageDiff {
    /// Total channel samples in the image (width * height * 3).
    pub fn total_samples(&self) -> usize {
        (self.width as usize) * (self.height as usize) * 3
    }

    /// Fraction of channel samples exceeding the given threshold (0..1).
    pub fn fraction_over_8(&self) -> f64 {
        self.samples_over_8 as f64 / self.total_samples().max(1) as f64
    }
}

/// Per-channel diff between two 16-bit RGB images. Panics if dimensions
/// differ — that's a hard test failure, not a near-match.
pub fn image_diff(a: &RgbImage, b: &RgbImage) -> ImageDiff {
    assert_eq!(
        (a.width, a.height),
        (b.width, b.height),
        "image dimension mismatch"
    );
    assert_eq!(
        a.pixels.len(),
        b.pixels.len(),
        "pixel buffer length mismatch"
    );

    let mut max_abs = 0u16;
    let mut over_8 = 0usize;
    let mut over_64 = 0usize;
    let mut over_512 = 0usize;
    let mut over_4096 = 0usize;
    for (&x, &y) in a.pixels.iter().zip(&b.pixels) {
        let d = x.abs_diff(y);
        if d > max_abs {
            max_abs = d;
        }
        if d > 8 {
            over_8 += 1;
            if d > 64 {
                over_64 += 1;
                if d > 512 {
                    over_512 += 1;
                    if d > 4096 {
                        over_4096 += 1;
                    }
                }
            }
        }
    }
    ImageDiff {
        width: a.width,
        height: a.height,
        max_abs_diff: max_abs,
        samples_over_8: over_8,
        samples_over_64: over_64,
        samples_over_512: over_512,
        samples_over_4096: over_4096,
    }
}
