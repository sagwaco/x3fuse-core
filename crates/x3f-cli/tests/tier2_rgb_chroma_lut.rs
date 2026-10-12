//! RGB recovery remains selectable without process-global environment changes.
mod common;

use std::process::Command;

#[test]
fn rgb_lut_flags_override_environment_and_leave_other_paths_unchanged() {
    let input = skip_if_missing!("CLIPPED_IMAGE_MERRILL.X3F");
    let scratch = std::env::temp_dir().join(format!("x3f-rgb-lut-{}", std::process::id()));
    std::fs::create_dir_all(&scratch).unwrap();
    let render = |format: &[&str], flags: &[&str], disabled_by_env: bool| {
        let mut cmd = Command::new(common::X3F_EXTRACT);
        cmd.args(format)
            .args(flags)
            .args(["-denoise-backend", "cpu", "-o"])
            .arg(&scratch)
            .env_remove("X3F_NO_CHROMA_LUT");
        if disabled_by_env {
            cmd.env("X3F_NO_CHROMA_LUT", "1");
        }
        let output = cmd.arg(&input).output().unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let suffix = match format[0] {
            "-dng" => "dng",
            "-ppm" => "ppm",
            "-histogram" => "csv",
            _ => "tif",
        };
        common::file_md5(&scratch.join(format!(
            "{}.{suffix}",
            input.file_name().unwrap().to_string_lossy()
        )))
    };
    let enabled = render(&["-tiff"], &["-no-denoise"], false);
    let disabled = render(&["-tiff"], &["-no-denoise"], true);
    assert_ne!(
        enabled, disabled,
        "fixture must exercise legacy LUT recovery"
    );
    assert_eq!(
        render(&["-tiff"], &["-no-denoise", "-no-chroma-lut"], false),
        disabled
    );
    let native = render(&["-tiff"], &["-spp-denoise", "-no-chroma-lut"], false);
    assert_eq!(render(&["-tiff"], &["-spp-denoise"], false), native);
    let refined = render(&["-tiff"], &["-chroma-lut"], false);
    assert_ne!(refined, native, "fixture must exercise SPP LUT refinement");
    assert_eq!(render(&["-tiff"], &["-chroma-lut"], true), refined);
    assert_eq!(
        render(&["-tiff"], &["-spp-denoise", "-chroma-lut"], false),
        refined
    );
    for format in [&["-ppm"][..], &["-histogram"][..]] {
        assert_eq!(
            render(format, &["-chroma-lut"], true),
            render(format, &["-spp-denoise", "-chroma-lut"], false),
            "implicit SPP routing was lost in {format:?}"
        );
    }
    for format in [
        &["-dng"][..],
        &["-dng", "-dng-highlight-recovery"][..],
        &["-tiff", "-color", "none", "-dng-highlight-recovery"][..],
        &["-tiff", "-cineon"][..],
    ] {
        assert_eq!(
            render(format, &["-no-denoise", "-chroma-lut"], false),
            render(format, &["-no-denoise", "-no-chroma-lut"], false),
            "RGB LUT override affected {format:?}"
        );
    }
    for flags in [&["-no-denoise"][..], &["-denoise-backend", "gpu"][..]] {
        let output = Command::new(common::X3F_EXTRACT)
            .args(["-tiff", "-chroma-lut", "-o"])
            .arg(&scratch)
            .args(flags)
            .arg(&input)
            .output()
            .unwrap();
        assert!(
            !output.status.success(),
            "unsupported SPP options succeeded: {flags:?}"
        );
        assert!(String::from_utf8_lossy(&output.stderr).contains("SPP denoise"));
    }
    std::fs::remove_dir_all(scratch).unwrap();
}

#[test]
fn non_merrill_lut_selection_retains_legacy_processing() {
    let input = skip_if_missing!("_SDI8284.X3F");
    let scratch = std::env::temp_dir().join(format!("x3f-quattro-lut-{}", std::process::id()));
    std::fs::create_dir_all(&scratch).unwrap();
    let render = |flags: &[&str], disabled_by_env: bool| {
        let mut cmd = Command::new(common::X3F_EXTRACT);
        cmd.args(["-tiff", "-no-denoise", "-o"])
            .arg(&scratch)
            .args(flags)
            .env_remove("X3F_NO_CHROMA_LUT");
        if disabled_by_env {
            cmd.env("X3F_NO_CHROMA_LUT", "1");
        }
        let output = cmd.arg(&input).output().unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        common::file_md5(&scratch.join("_SDI8284.X3F.tif"))
    };
    assert_eq!(render(&["-chroma-lut"], true), render(&[], false));
    assert_eq!(render(&["-no-chroma-lut"], false), render(&[], true));
    std::fs::remove_dir_all(scratch).unwrap();
}
