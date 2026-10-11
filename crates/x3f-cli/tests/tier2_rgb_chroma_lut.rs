//! RGB recovery remains selectable without process-global environment changes.
mod common;

use std::process::Command;

#[test]
fn rgb_lut_flags_override_environment_and_leave_other_paths_unchanged() {
    let input = skip_if_missing!("CLIPPED_IMAGE_MERRILL.X3F");
    let scratch = std::env::temp_dir().join(format!("x3f-rgb-lut-{}", std::process::id()));
    std::fs::create_dir_all(&scratch).unwrap();
    let render = |format: &[&str], flag: Option<&str>, disabled_by_env: bool| {
        let mut cmd = Command::new(common::X3F_EXTRACT);
        cmd.args(format)
            .args(["-no-denoise", "-o"])
            .arg(&scratch)
            .env_remove("X3F_NO_CHROMA_LUT");
        if let Some(flag) = flag {
            cmd.arg(flag);
        }
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
    let enabled = render(&["-tiff"], None, false);
    let disabled = render(&["-tiff"], None, true);
    assert_ne!(enabled, disabled, "fixture must exercise LUT recovery");
    assert_eq!(render(&["-tiff"], Some("-chroma-lut"), false), enabled);
    assert_eq!(render(&["-tiff"], Some("-chroma-lut"), true), enabled);
    assert_eq!(render(&["-tiff"], Some("-no-chroma-lut"), false), disabled);
    for format in [&["-ppm"][..], &["-histogram"][..]] {
        let enabled = render(format, Some("-chroma-lut"), true);
        let disabled = render(format, Some("-no-chroma-lut"), false);
        assert_ne!(enabled, disabled, "override was lost in {format:?}");
        assert_eq!(render(format, None, true), disabled);
    }
    for format in [
        &["-dng"][..],
        &["-dng", "-dng-highlight-recovery"][..],
        &["-tiff", "-color", "none", "-dng-highlight-recovery"][..],
        &["-tiff", "-cineon"][..],
    ] {
        assert_eq!(
            render(format, Some("-chroma-lut"), false),
            render(format, Some("-no-chroma-lut"), false),
            "RGB LUT override affected {format:?}"
        );
    }
    std::fs::remove_dir_all(scratch).unwrap();
}
