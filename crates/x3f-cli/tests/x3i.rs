//! Windowed frame output must equal output from an extracted X3F file.

mod common;

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Mutex;
use x3f_core::X3iContainer;

// Each full merge holds about 2.6 GB. Run corpus merges one at a time.
static X3I_MERGE_LOCK: Mutex<()> = Mutex::new(());

struct Scratch(PathBuf);
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn run(input: &Path, outdir: &Path, flags: &[&str]) -> std::process::Output {
    let output = Command::new(common::X3F_EXTRACT)
        .args(flags)
        .arg("-o")
        .arg(outdir)
        .arg(input)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    output
}

#[test]
fn corpus_frames_match_extracted_metadata_and_ppm() {
    for name in ["DP0Q0002.X3I", "_DQH0978.X3I"] {
        let Some(input) = common::find_input(name) else {
            eprintln!("skip: {name} is absent");
            continue;
        };
        let scratch =
            Scratch(std::env::temp_dir().join(format!("x3i-parity-{}-{name}", std::process::id())));
        std::fs::create_dir(&scratch.0).unwrap();
        let container = X3iContainer::open(&input).unwrap();
        let extracted = scratch.0.join("extracted_0.X3F");
        container.extract_frame(0, &extracted).unwrap();
        for (format, extension) in [("-meta", ".meta"), ("-ppm", ".ppm")] {
            run(
                &input,
                &scratch.0,
                &[format, "-x3i-frame", "0", "-color", "none", "-no-denoise"],
            );
            run(
                &extracted,
                &scratch.0,
                &[format, "-color", "none", "-no-denoise"],
            );
            let windowed = scratch.0.join(format!("{name}_0{extension}"));
            let direct = scratch.0.join(format!("extracted_0.X3F{extension}"));
            let windowed = std::fs::read(windowed).unwrap();
            let direct = std::fs::read(direct).unwrap();
            assert_eq!(windowed.len(), direct.len(), "{name} {format}");
            assert!(windowed == direct, "{name} {format} differs");
        }
        run(&input, &scratch.0, &["-meta"]);
        let metadata = std::fs::read_to_string(scratch.0.join(format!("{name}.meta"))).unwrap();
        assert!(metadata.starts_with("X3I frames\nindex offset length exposure_us EV\n"));
        let reference = std::fs::read_to_string(scratch.0.join("extracted_0.X3F.meta")).unwrap();
        assert!(metadata.ends_with(&reference));
        let extract_dir = scratch.0.join("frames");
        std::fs::create_dir(&extract_dir).unwrap();
        run(&input, &extract_dir, &["-x3i-extract"]);
        for (index, frame) in container.frames().iter().enumerate() {
            let out = extract_dir.join(format!(
                "{}_{index}.X3F",
                input.file_stem().unwrap().to_string_lossy()
            ));
            assert_eq!(std::fs::metadata(out).unwrap().len(), frame.length);
        }
        let output = Command::new(common::X3F_EXTRACT)
            .args(["-x3i-extract", "-o"])
            .arg(&extract_dir)
            .arg(&input)
            .output()
            .unwrap();
        assert!(!output.status.success());
        assert_eq!(
            std::fs::read(extract_dir.join(format!(
                "{}_0.X3F",
                input.file_stem().unwrap().to_string_lossy()
            )))
            .unwrap(),
            std::fs::read(extracted).unwrap()
        );
    }
}

#[test]
fn missing_shutter_keeps_metadata_and_cli_errors_give_guidance() {
    let scratch =
        Scratch(std::env::temp_dir().join(format!("x3i-missing-shutter-{}", std::process::id())));
    std::fs::create_dir(&scratch.0).unwrap();
    let input = scratch.0.join("input.bin");
    let mut frame = b"FOVb".to_vec();
    frame.extend(0x40000u32.to_le_bytes());
    frame.extend([0; 16]);
    for word in [u32::from_le_bytes(*b"SECc"), 0x20000, 2, 0, 0, 0, 0] {
        frame.extend(word.to_le_bytes());
    }
    // Type-2 encrypted zero padding, with key 0.
    frame.push(0x36);
    let directory = frame.len() as u32;
    frame.extend(b"SECd");
    for word in [
        0x20000u32,
        1,
        24,
        29,
        u32::from_le_bytes(*b"CAMF"),
        directory,
    ] {
        frame.extend(word.to_le_bytes());
    }
    let length = frame.len() as u32;
    let mut bytes = vec![0; 0x5a0];
    bytes[..4].copy_from_slice(b"FOVi");
    bytes.extend(frame);
    let directory = bytes.len() as u32;
    bytes.extend(b"SECf");
    for word in [
        0x100u32,
        1,
        0x5a0,
        length,
        u32::from_le_bytes(*b"FRAM"),
        directory,
    ] {
        bytes.extend(word.to_le_bytes());
    }
    std::fs::write(&input, bytes).unwrap();
    run(&input, &scratch.0, &["-meta"]);
    let metadata = std::fs::read_to_string(scratch.0.join("input.bin.meta")).unwrap();
    assert!(metadata.starts_with(&format!(
        "X3I frames\nindex offset length exposure_us EV\n0 1440 {length} - -\n\n"
    )));
    let output = Command::new(common::X3F_EXTRACT)
        .arg("-jpg")
        .arg(&input)
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr)
        .contains("X3I input: use -dng, -tiff, -ppm, -x3i-frame <n>, -x3i-extract or -meta"));
    let missing = scratch.0.join("missing.bin");
    let output = Command::new(common::X3F_EXTRACT)
        .arg(&missing)
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr)
        .contains(&format!("could not open `{}`:", missing.display())));
}

#[test]
fn corpus_dng_merge_writes_seven_frame_report() {
    let _merge = X3I_MERGE_LOCK.lock().unwrap();
    for name in ["DP0Q0002.X3I", "_DQH0978.X3I"] {
        let Some(input) = common::find_input(name) else {
            eprintln!("skip: {name} is absent");
            continue;
        };
        let scratch =
            Scratch(std::env::temp_dir().join(format!("x3i-report-{}-{name}", std::process::id())));
        std::fs::create_dir(&scratch.0).unwrap();
        let output = run(
            &input,
            &scratch.0,
            &[
                "-dng",
                "-no-denoise",
                "-x3i-report",
                "-x3i-ghost-map",
                "-q",
                "-dng-highlight-recovery",
                "-dng-highlight-mapping",
                "linear",
            ],
        );
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(!stderr.contains("X3I frame "));
        assert!(stderr.contains("X3I merge: frames=7"));
        assert!(stderr.contains("have no effect on X3I merges"));
        assert!(scratch.0.join(format!("{name}.dng")).is_file());
        assert!(!scratch.0.join(format!("{name}.dng.tmp")).exists());
        let report = std::fs::read_to_string(scratch.0.join(format!("{name}.x3i-report"))).unwrap();
        assert!(
            report.contains("sigma_B sigma_M sigma_T clip_B clip_M clip_T ratio_B ratio_M ratio_T")
        );
        let container = X3iContainer::open(&input).unwrap();
        for frame in 1..container.frames().len() {
            let map = std::fs::read(scratch.0.join(format!("{name}.ghost{frame}.pgm"))).unwrap();
            let mut lines = map.splitn(4, |&v| v == b'\n');
            assert_eq!(lines.next().unwrap(), b"P5");
            let dimensions: Vec<usize> = std::str::from_utf8(lines.next().unwrap())
                .unwrap()
                .split_whitespace()
                .map(|v| v.parse().unwrap())
                .collect();
            assert_eq!(dimensions.len(), 2);
            let (width, height) = (dimensions[0], dimensions[1]);
            let sensor = container.sensor_size();
            assert!(width >= sensor.0 as usize && height >= sensor.1 as usize);
            let header = format!("P5\n{width} {height}\n255\n");
            assert!(map.starts_with(header.as_bytes()));
            assert_eq!(map.len(), header.len() + width * height);
            assert!(map[header.len()..].contains(&128));
            assert!(!scratch
                .0
                .join(format!("{name}.ghost{frame}.pgm.tmp"))
                .exists());
        }
        assert!(!scratch.0.join(format!("{name}.ghost0.pgm")).exists());
        let rows: Vec<_> = report
            .lines()
            .filter(|line| line.starts_with(|c: char| c.is_ascii_digit()))
            .collect();
        assert_eq!(rows.len(), 7);
        for (i, row) in rows.iter().enumerate() {
            let values: Vec<_> = row.split_whitespace().collect();
            assert_eq!(values.len(), 12);
            assert_eq!(values[0].parse::<usize>().unwrap(), i);
            assert_eq!(
                values[1].parse::<u32>().unwrap(),
                container.frame_exposure_us(i).unwrap()
            );
            let expected = f64::from(container.frame_exposure_us(0).unwrap())
                / f64::from(container.frame_exposure_us(i).unwrap());
            assert!((values[2].parse::<f64>().unwrap() - expected).abs() < 1e-9);
            for fraction in &values[6..9] {
                assert!((0.0..=1.0).contains(&fraction.parse::<f64>().unwrap()));
            }
        }
        assert!(report.contains("X3I noise: k_B="));
        let statistics: Vec<_> = report
            .lines()
            .filter(|line| line.contains(": statistics nominal="))
            .collect();
        let alignments: Vec<_> = report
            .lines()
            .filter(|line| line.contains(": alignment dx="))
            .collect();
        assert_eq!(statistics.len(), 7);
        assert_eq!(alignments.len(), 7);
        for line in &alignments {
            assert!(line.contains("retained_tiles="));
            assert!(line.contains("max_tile_residual="));
            assert!(line.contains("weak="));
        }
        assert_eq!(
            report
                .lines()
                .filter(|line| line.contains(": aligned-ratio-bin "))
                .count(),
            56
        );
        assert_eq!(
            report
                .lines()
                .filter(|line| line.contains(": ghost-bin "))
                .count(),
            70
        );
        for (i, line) in statistics.iter().enumerate() {
            assert!(line.starts_with(&format!("X3I frame {i}:")));
            let number = |key: &str| {
                line.split_whitespace()
                    .find_map(|v| v.strip_prefix(key))
                    .unwrap()
                    .parse::<f64>()
                    .unwrap()
            };
            assert!((number("refined=") / number("nominal=")).log2().abs() <= 0.3);
            for key in ["c=", "c_chroma="] {
                assert!((0.5..=4.0).contains(&number(key)));
            }
            assert!(line.contains("black_offset="));
            assert!(line.contains("calibration_weak="));
            assert!(line.contains("median_z2_flat="));
            assert!(line.contains("ghost_threshold=0.1 ghost_status=preliminary"));
            assert!((0.0..=1.0).contains(&number("ghost_fraction=")));
        }
        assert!(report.contains("X3I merge: frames=7 s="));
        assert!(report.contains("saturation: B="));
        assert_eq!(
            report
                .lines()
                .filter(|line| line.starts_with("ratio-bin "))
                .count(),
            56
        );
        assert!(report.contains("have no effect on X3I merges"));
        #[cfg(unix)]
        {
            let rss = report
                .lines()
                .find_map(|line| line.strip_prefix("peak_rss_bytes="))
                .unwrap();
            assert!(
                rss.split_whitespace()
                    .next()
                    .unwrap()
                    .parse::<u64>()
                    .unwrap()
                    > 0
            );
        }
        if name == "DP0Q0002.X3I" {
            let dng = scratch.0.join(format!("{name}.dng"));
            let expected = common::file_md5(&dng);
            // This float output hash was recorded on macOS aarch64.
            #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
            assert_eq!(expected, "c80db90d70032be445b0a0a0af81b876");
            let report_path = scratch.0.join(format!("{name}.x3i-report"));
            std::fs::remove_file(&report_path).unwrap();
            std::fs::create_dir(&report_path).unwrap();
            let output = Command::new(common::X3F_EXTRACT)
                .args(["-dng", "-no-denoise", "-x3i-report", "-q", "-o"])
                .arg(&scratch.0)
                .arg(&input)
                .output()
                .unwrap();
            assert!(!output.status.success());
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(stderr.contains("X3I report failed:"));
            assert!(stderr.contains("DNG saved to"));
            assert!(!stderr.contains("X3I frame "));
            assert!(!stderr.contains("have no effect on X3I merges"));
            assert_eq!(
                common::file_md5(&dng),
                expected,
                "ignored highlight flags changed DNG bytes"
            );
            assert!(!scratch.0.join(format!("{name}.dng.tmp")).exists());
            assert!(!scratch.0.join(format!("{name}.x3i-report.tmp")).exists());
        } else {
            run(
                &input,
                &scratch.0,
                &["-dng", "-no-denoise", "-x3i-frame", "0"],
            );
            run(
                &input,
                &scratch.0,
                &["-dng", "-no-denoise", "-x3i-frame", "3"],
            );
            let (merged, exposure) = raw_dng(&scratch.0.join(format!("{name}.dng")));
            let (reference, reference_exposure) = raw_dng(&scratch.0.join(format!("{name}_0.dng")));
            let (short, short_exposure) = raw_dng(&scratch.0.join(format!("{name}_3.dng")));
            assert_eq!(merged.len(), reference.len());
            assert_eq!(merged.len(), short.len());
            let scale = (exposure - reference_exposure).exp2();
            let short_scale = (short_exposure - reference_exposure).exp2()
                * f64::from(container.frame_exposure_us(0).unwrap())
                / f64::from(container.frame_exposure_us(3).unwrap());
            assert!(scale > 1.0);
            let mut values: [Vec<u16>; 3] = Default::default();
            let mut refs: [Vec<u16>; 3] = Default::default();
            let mut shorts: [Vec<u16>; 3] = Default::default();
            for ((pixel, reference), short) in merged
                .as_chunks::<3>()
                .0
                .iter()
                .zip(reference.as_chunks::<3>().0)
                .zip(short.as_chunks::<3>().0)
            {
                // This band contains the curtain pixels affected by the soft shoulder.
                if (0.08..=0.11).contains(&(f64::from(reference[0]) / 65535.0)) {
                    for c in 0..3 {
                        values[c].push(pixel[c]);
                        refs[c].push(reference[c]);
                        shorts[c].push(short[c]);
                    }
                }
            }
            assert!(values[0].len() > 1000, "missing curtain region");
            let median = |values: &mut [u16]| {
                let n = values.len();
                let (left, centre, _) = values.select_nth_unstable(n / 2);
                if n.is_multiple_of(2) {
                    (f64::from(*centre) + f64::from(*left.iter().max().unwrap())) * 0.5
                } else {
                    f64::from(*centre)
                }
            };
            for c in 0..3 {
                let merged = scale * median(&mut values[c]);
                let ratio = merged / median(&mut refs[c]);
                assert!((ratio - 1.0).abs() < 0.01, "curtain channel {c}: {ratio}");
                let short_ratio = merged / (short_scale * median(&mut shorts[c]));
                // The old defaults miss this limit by more than five times in every channel.
                assert!(
                    (short_ratio - 1.0).abs() < 0.003,
                    "curtain channel {c} against -1 EV: {short_ratio}"
                );
            }
        }
    }
}

fn raw_dng(path: &Path) -> (Vec<u16>, f64) {
    use tiff::decoder::{ifd::Value, Decoder};
    use tiff::tags::Tag;

    let bytes = std::fs::read(path).unwrap();
    assert_eq!(&bytes[..2], b"II");
    let mut decoder = Decoder::new(std::io::Cursor::new(&bytes)).unwrap();
    let Value::SRational(numerator, denominator) = decoder.get_tag(Tag::Unknown(50730)).unwrap()
    else {
        panic!("expected signed rational BaselineExposure");
    };
    let pointer = decoder
        .get_tag(Tag::SubIfd)
        .unwrap()
        .into_ifd_pointer()
        .unwrap();
    let directory = decoder.read_directory(pointer).unwrap();
    let mut tags = decoder.read_directory_tags(&directory);
    assert_eq!(tags.get_tag_u32(Tag::Compression).unwrap(), 1);
    assert_eq!(tags.get_tag_u16_vec(Tag::BitsPerSample).unwrap(), [16; 3]);
    let width = tags.get_tag_u32(Tag::ImageWidth).unwrap() as usize;
    let height = tags.get_tag_u32(Tag::ImageLength).unwrap() as usize;
    let offsets = tags.get_tag_u32_vec(Tag::StripOffsets).unwrap();
    let counts = tags.get_tag_u32_vec(Tag::StripByteCounts).unwrap();
    assert_eq!(offsets.len(), counts.len());
    let mut pixels = Vec::with_capacity(width * height * 3);
    for (offset, count) in offsets.into_iter().zip(counts) {
        let strip = &bytes[offset as usize..offset as usize + count as usize];
        assert!(strip.len().is_multiple_of(2));
        pixels.extend(
            strip
                .as_chunks::<2>()
                .0
                .iter()
                .map(|&v| u16::from_le_bytes(v)),
        );
    }
    assert_eq!(pixels.len(), width * height * 3);
    (pixels, f64::from(numerator) / f64::from(denominator))
}

fn read_rgb_tiff(path: &Path) -> ((u32, u32), Vec<u16>) {
    let mut decoder = tiff::decoder::Decoder::new(std::fs::File::open(path).unwrap()).unwrap();
    let dimensions = decoder.dimensions().unwrap();
    let tiff::decoder::DecodingResult::U16(data) = decoder.read_image().unwrap() else {
        panic!("expected 16-bit TIFF");
    };
    (dimensions, data)
}

fn assert_midtones(reference: &[u16], actual: impl Fn(usize) -> f64, label: &str) {
    for channel in 0..3 {
        let mut before = Vec::new();
        let mut after = Vec::new();
        for i in (channel..reference.len()).step_by(3 * 16) {
            let value = f64::from(reference[i]) / 65535.0;
            if (0.05..=0.5).contains(&value) {
                before.push(value);
                after.push(actual(i));
            }
        }
        assert!(before.len() > 1000, "{label}: insufficient midtones");
        before.sort_unstable_by(f64::total_cmp);
        after.sort_unstable_by(f64::total_cmp);
        let expected = before[before.len() / 2];
        let measured = after[after.len() / 2];
        eprintln!("{label} channel {channel}: reference median={expected}, actual={measured}");
        assert!(
            (measured / expected - 1.0).abs() <= 0.02,
            "{label}: channel {channel}"
        );
    }
}

fn assert_zero_shift(reference: &[u16], actual: &[u16], width: usize, height: usize) {
    let luma = |data: &[u16], x, y| {
        let i = (y * width + x) * 3;
        0.2126 * f64::from(data[i])
            + 0.7152 * f64::from(data[i + 1])
            + 0.0722 * f64::from(data[i + 2])
    };
    let mut peak = (f64::NEG_INFINITY, 99, 99);
    for dy in -2isize..=2 {
        for dx in -2isize..=2 {
            let (mut n, mut a, mut b, mut aa, mut bb, mut ab) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
            for y in (2..height - 2).step_by(16) {
                for x in (2..width - 2).step_by(16) {
                    let p = luma(reference, x, y);
                    if !(0.05 * 65535.0..=0.5 * 65535.0).contains(&p) {
                        continue;
                    }
                    let q = luma(
                        actual,
                        x.checked_add_signed(dx).unwrap(),
                        y.checked_add_signed(dy).unwrap(),
                    );
                    n += 1.0;
                    a += p;
                    b += q;
                    aa += p * p;
                    bb += q * q;
                    ab += p * q;
                }
            }
            let correlation = (n * ab - a * b) / ((n * aa - a * a) * (n * bb - b * b)).sqrt();
            if correlation > peak.0 {
                peak = (correlation, dx, dy);
            }
        }
    }
    eprintln!("luma correlation peak: {peak:?}");
    assert_eq!((peak.1, peak.2), (0, 0));
}

fn check_merged_color(name: &str, check_position: bool) {
    let _merge = X3I_MERGE_LOCK.lock().unwrap();
    use std::sync::atomic::AtomicBool;
    use x3f_core::{prepare_scene_linear, SceneLinearOptions};

    let Some(input) = common::find_input(name) else {
        eprintln!("skip: {name} is absent");
        return;
    };
    let scratch = Scratch(std::env::temp_dir().join(format!(
        "x3i-color-{}-{name}-{check_position}",
        std::process::id()
    )));
    std::fs::create_dir(&scratch.0).unwrap();
    run(
        &input,
        &scratch.0,
        &["-tiff", "-color", "sRGB", "-no-denoise", "-x3i-frame", "0"],
    );
    run(
        &input,
        &scratch.0,
        &["-tiff", "-color", "sRGB", "-no-denoise"],
    );
    let (dimensions, reference) = read_rgb_tiff(&scratch.0.join(format!("{name}_0.tif")));
    let (merged_dimensions, merged) = read_rgb_tiff(&scratch.0.join(format!("{name}.tif")));
    assert_eq!(dimensions, merged_dimensions);
    if check_position {
        assert_zero_shift(
            &reference,
            &merged,
            dimensions.0 as usize,
            dimensions.1 as usize,
        );
        return;
    }
    assert_midtones(&reference, |i| f64::from(merged[i]) / 65535.0, name);
    drop(merged);
    let scene = prepare_scene_linear(
        &input,
        &SceneLinearOptions {
            denoise_intensity: 0,
            ..Default::default()
        },
        &AtomicBool::new(false),
        |_| {},
    )
    .unwrap();
    assert_eq!((scene.width, scene.height), dimensions);
    assert_midtones(
        &reference,
        |i| {
            let value = f64::from(scene.data[i]).clamp(0.0, 1.0);
            if value <= 0.0031308 {
                12.92 * value
            } else {
                1.055 * value.powf(1.0 / 2.4) - 0.055
            }
        },
        &format!("{name} scene"),
    );
}

#[test]
fn dp0_merged_color_matches_reference_midtones() {
    check_merged_color("DP0Q0002.X3I", false);
}

#[test]
fn quattro_h_merged_color_matches_reference_midtones() {
    check_merged_color("_DQH0978.X3I", false);
}

#[test]
fn dp0_merged_position_matches_reference() {
    check_merged_color("DP0Q0002.X3I", true);
}

#[test]
fn quattro_h_merged_position_matches_reference() {
    check_merged_color("_DQH0978.X3I", true);
}
