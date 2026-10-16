//! Windowed frame output must equal output from an extracted X3F file.

mod common;

use std::path::{Path, PathBuf};
use std::process::Command;
use x3f_core::X3iContainer;

struct Scratch(PathBuf);
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn run(input: &Path, outdir: &Path, flags: &[&str]) {
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
        .arg("-dng")
        .arg(&input)
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr)
        .contains("X3I input: use -x3i-frame <n> or -x3i-extract"));
    let missing = scratch.0.join("missing.bin");
    let output = Command::new(common::X3F_EXTRACT)
        .arg(&missing)
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr)
        .contains(&format!("could not open `{}`:", missing.display())));
}
