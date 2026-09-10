//! Corpus checks for recovered highlight data, decoded from the raw SubIFD.
//!
//! These test encoding and reconstruction invariants, not visual equivalence
//! to an SPP export. Such exports require color-managed rendering and matching
//! exposure/crop before a perceptual comparison is meaningful.

mod common;

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};

#[test]
fn clipped_merrill_retains_linear_highlight_detail() {
    let input = skip_if_missing!("CLIPPED_IMAGE_MERRILL.X3F");
    check_recovered_mapping(&input);
}

#[test]
fn dp2m0981_retains_linear_highlight_detail() {
    let input = skip_if_missing!("DP2M0981.X3F");
    check_recovered_mapping(&input);
}

#[test]
fn mapping_selection_does_not_enable_recovery() {
    for name in ["sigma_sd1_merrill_15.x3f", "_SDI8040.X3F", "_SDI8284.X3F"] {
        let Some(input) = common::find_input(name) else {
            eprintln!("  (skip: corpus file `{name}` not found; set X3F_TEST_FILES)");
            continue;
        };
        let default = extract(&input, &[]);
        let shoulder = extract(&input, &["-dng-highlight-mapping", "shoulder"]);
        assert!(
            fs::read(default).expect("read default DNG")
                == fs::read(shoulder).expect("read recovery-off DNG"),
            "mapping selection changed recovery-off output for {name}"
        );
    }
}

fn check_recovered_mapping(input: &Path) {
    let off = read_raw(&extract(input, &[]));
    // Omitting the mapping selector deliberately exercises its linear default.
    let linear = read_raw(&extract(input, &["-dng-highlight-recovery"]));
    let shoulder = read_raw(&extract(
        input,
        &[
            "-dng-highlight-recovery",
            "-dng-highlight-mapping",
            "shoulder",
        ],
    ));
    assert_eq!((linear.width, linear.height), (off.width, off.height));
    assert_eq!(
        (linear.width, linear.height),
        (shoulder.width, shoulder.height)
    );
    assert_eq!(linear.linear_response_limit, 1.0);
    assert_eq!(off.linear_response_limit, 1.0);
    assert!(
        (shoulder.linear_response_limit - 0.85).abs() < 1e-6,
        "clipped fixture must exercise the default 0.85 shoulder knee"
    );
    assert!(
        (shoulder.baseline_exposure - off.baseline_exposure).abs() < 1e-6,
        "shoulder mapping must not add linear headroom exposure compensation"
    );
    let scale = (linear.baseline_exposure - off.baseline_exposure).exp2();
    assert!(
        scale.is_finite() && scale > 1.001,
        "clipped fixture has no recovered encoding headroom: scale={scale}"
    );

    let knee = shoulder.linear_response_limit * f64::from(u16::MAX);
    let mut midtone_samples = 0;
    let mut highlight_pixels = 0;
    let mut highlight_levels = vec![false; usize::from(u16::MAX) + 1];
    let mut maximum_midtone_error = 0.0_f64;
    let mut maximum_chroma_error = 0.0_f64;
    for (lp, sp) in linear
        .pixels
        .chunks_exact(3)
        .zip(shoulder.pixels.chunks_exact(3))
    {
        let lmax = f64::from(*lp.iter().max().unwrap());
        let smax = f64::from(*sp.iter().max().unwrap());
        // Both outputs are linear below the knee. Undoing the global scale
        // must recover the same samples to their final quantization precision.
        // Stay clear of the knee so rounding cannot select opposite branches.
        if smax > 256.0 && smax < knee - 32.0 && lmax * scale < knee - 32.0 {
            for (&l, &s) in lp.iter().zip(sp) {
                let error = (f64::from(l) * scale - f64::from(s)).abs();
                maximum_midtone_error = maximum_midtone_error.max(error);
                midtone_samples += 1;
            }
        }
        if lmax * scale > f64::from(u16::MAX) * 1.001 {
            highlight_pixels += 1;
            highlight_levels[*lp.iter().max().unwrap() as usize] = true;
            // Cross products compare chroma without dividing by dark channels.
            // Three output code values allow for rounding in both mappings.
            for (&l, &s) in lp.iter().zip(sp) {
                let error = (f64::from(l) * smax - f64::from(s) * lmax).abs();
                maximum_chroma_error = maximum_chroma_error.max(error / (lmax + smax));
            }
        }
    }
    assert!(
        midtone_samples > 1_000,
        "fixture lacks shared linear samples"
    );
    assert!(
        maximum_midtone_error <= scale + 2.0,
        "headroom compensation loses midtone precision: error={maximum_midtone_error}, scale={scale}"
    );
    assert!(
        highlight_pixels > 64,
        "fixture lacks recovered highlight pixels"
    );
    let distinct_levels = highlight_levels.iter().filter(|&&used| used).count();
    assert!(
        distinct_levels >= 8,
        "recovered highlights collapsed to {distinct_levels} intensity levels"
    );
    assert!(
        maximum_chroma_error <= 3.0,
        "mapping changed reconstructed highlight chroma: error={maximum_chroma_error} code values"
    );
    eprintln!(
        "  {}: scale={scale:.6}, highlight_pixels={highlight_pixels}, highlight_levels={distinct_levels}, midtone_error={maximum_midtone_error:.3}, chroma_error={maximum_chroma_error:.3}",
        input.file_name().unwrap().to_string_lossy()
    );
}

/// Give every run an independent scratch directory and remove inherited
/// research tunables from the child, without mutating this test's environment.
fn extract(input: &Path, switches: &[&str]) -> PathBuf {
    static RUN: AtomicUsize = AtomicUsize::new(0);
    let scratch = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!(
        "highlight-recovery-{}-{}",
        std::process::id(),
        RUN.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir_all(&scratch).expect("create recovery scratch directory");
    let staged = scratch.join(input.file_name().expect("input filename"));
    fs::copy(input, &staged).expect("stage recovery input");
    let mut cmd = Command::new(common::X3F_EXTRACT);
    for (name, _) in std::env::vars_os() {
        if name.to_string_lossy().starts_with("X3F_") {
            cmd.env_remove(name);
        }
    }
    let result = cmd
        .args(["-dng", "-no-denoise"])
        .args(switches)
        .arg(&staged)
        .output()
        .expect("run highlight conversion");
    assert!(
        result.status.success(),
        "highlight conversion {switches:?} failed: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    let mut output = staged.into_os_string();
    output.push(".dng");
    PathBuf::from(output)
}

struct RawDng {
    width: u32,
    height: u32,
    baseline_exposure: f64,
    linear_response_limit: f64,
    pixels: Vec<u16>,
}

struct Entry {
    tag: u16,
    typ: u16,
    count: usize,
    payload: usize,
}

fn read_raw(path: &Path) -> RawDng {
    let bytes = fs::read(path).expect("read recovered DNG");
    assert_eq!(&bytes[..2], b"II", "expected little-endian TIFF");
    assert_eq!(u16_at(&bytes, 2), 42);
    let ifd0 = entries(&bytes, u32_at(&bytes, 4) as usize);
    let raw_offset = numbers(&bytes, &ifd0, 330)[0] as usize;
    let raw = entries(&bytes, raw_offset);
    let width = numbers(&bytes, &raw, 256)[0] as u32;
    let height = numbers(&bytes, &raw, 257)[0] as u32;
    assert_eq!(numbers(&bytes, &raw, 262), vec![34892.0]);
    assert_eq!(numbers(&bytes, &raw, 277), vec![3.0]);
    assert!(numbers(&bytes, &raw, 258).iter().all(|&v| v == 16.0));
    assert_eq!(numbers(&bytes, &raw, 259), vec![1.0]);
    assert!(numbers(&bytes, &raw, 50714).iter().all(|&v| v == 0.0));
    assert!(numbers(&bytes, &raw, 50717).iter().all(|&v| v == 65535.0));
    let exposure = ifd0
        .iter()
        .find(|e| e.tag == 50730)
        .expect("IFD0 BaselineExposure");
    assert_eq!((exposure.typ, exposure.count), (10, 1));
    let limit = ifd0
        .iter()
        .find(|e| e.tag == 50734)
        .expect("IFD0 LinearResponseLimit");
    assert_eq!((limit.typ, limit.count), (5, 1));
    let offsets = numbers(&bytes, &raw, 273);
    let counts = numbers(&bytes, &raw, 279);
    assert_eq!(offsets.len(), counts.len());
    let mut pixels = Vec::with_capacity(width as usize * height as usize * 3);
    for (offset, count) in offsets.into_iter().zip(counts) {
        let offset = offset as usize;
        let count = count as usize;
        assert_eq!(count % 2, 0);
        pixels.extend(
            bytes[offset..offset + count]
                .chunks_exact(2)
                .map(|c| u16::from_le_bytes([c[0], c[1]])),
        );
    }
    assert_eq!(pixels.len(), width as usize * height as usize * 3);
    RawDng {
        width,
        height,
        baseline_exposure: numbers(&bytes, &ifd0, 50730)[0],
        linear_response_limit: numbers(&bytes, &ifd0, 50734)[0],
        pixels,
    }
}

fn entries(bytes: &[u8], offset: usize) -> Vec<Entry> {
    (0..usize::from(u16_at(bytes, offset)))
        .map(|i| {
            let p = offset + 2 + i * 12;
            let typ = u16_at(bytes, p + 2);
            let count = u32_at(bytes, p + 4) as usize;
            let size = match typ {
                1 | 2 | 6 | 7 => 1,
                3 | 8 => 2,
                4 | 9 | 11 | 13 => 4,
                5 | 10 | 12 => 8,
                other => panic!("unsupported TIFF field type {other}"),
            };
            Entry {
                tag: u16_at(bytes, p),
                typ,
                count,
                payload: if count * size <= 4 {
                    p + 8
                } else {
                    u32_at(bytes, p + 8) as usize
                },
            }
        })
        .collect()
}

fn numbers(bytes: &[u8], ifd: &[Entry], tag: u16) -> Vec<f64> {
    let e = ifd
        .iter()
        .find(|e| e.tag == tag)
        .unwrap_or_else(|| panic!("missing tag {tag}"));
    (0..e.count)
        .map(|i| match e.typ {
            3 => f64::from(u16_at(bytes, e.payload + i * 2)),
            4 | 13 => f64::from(u32_at(bytes, e.payload + i * 4)),
            5 | 10 => {
                let numerator = u32_at(bytes, e.payload + i * 8);
                let denominator = u32_at(bytes, e.payload + i * 8 + 4);
                if e.typ == 10 {
                    assert!(
                        (denominator as i32) > 0,
                        "invalid signed rational denominator"
                    );
                    f64::from(numerator as i32) / f64::from(denominator as i32)
                } else {
                    assert!(denominator > 0, "invalid rational denominator");
                    f64::from(numerator) / f64::from(denominator)
                }
            }
            other => panic!("unexpected numeric type {other} for tag {tag}"),
        })
        .collect()
}

fn u16_at(bytes: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes(bytes[offset..offset + 2].try_into().unwrap())
}

fn u32_at(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
}
