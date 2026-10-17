//! Middle-exposure X3I source: single-frame output, not the merge.

use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;

use x3f_core::{
    convert_file, prepare_scene_linear, OutputFormat, ProcessOptions, SceneLinearOptions,
    X3iContainer, X3iSource,
};

fn corpus_file(name: &str) -> Option<PathBuf> {
    let root = std::env::var_os("X3F_TEST_FILES")
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("../../x3f_test_files"));
    let path = root.join(name);
    if path.is_file() {
        Some(path)
    } else {
        eprintln!("skip: missing {name} for middle exposure coverage");
        None
    }
}

#[test]
fn default_source_is_merge() {
    assert_eq!(X3iSource::default(), X3iSource::Merge);
    assert_eq!(ProcessOptions::default().x3i_source, X3iSource::Merge);
    assert_eq!(SceneLinearOptions::default().x3i_source, X3iSource::Merge);
}

#[cfg(feature = "serde")]
#[test]
fn source_serializes_as_snake_case() {
    for (value, text) in [
        (X3iSource::Merge, "\"merge\""),
        (X3iSource::MiddleExposure, "\"middle_exposure\""),
    ] {
        assert_eq!(serde_json::to_string(&value).unwrap(), text);
        assert_eq!(serde_json::from_str::<X3iSource>(text).unwrap(), value);
    }
}

#[test]
fn middle_exposure_matches_the_extracted_frame_bytes() {
    let Some(input) = corpus_file("DP0Q0002.X3I") else {
        return;
    };
    let dir = std::env::temp_dir().join(format!("x3i-middle-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let container = X3iContainer::open(&input).unwrap();
    let index = container.middle_exposure_frame().unwrap();
    assert_eq!(index, 0);
    let extracted = dir.join("frame.X3F");
    container.extract_frame(index, &extracted).unwrap();
    let cancel = AtomicBool::new(false);
    for (format, name) in [(OutputFormat::Dng, "dng"), (OutputFormat::Jpeg, "jpg")] {
        let options = ProcessOptions {
            denoise_intensity: 0,
            x3i_source: X3iSource::MiddleExposure,
            ..Default::default()
        };
        let middle = dir.join(format!("middle.{name}"));
        let report = convert_file(&input, &middle, format, &options, &cancel, |_| {}).unwrap();
        assert!(report
            .warnings
            .iter()
            .any(|line| line.starts_with("X3I middle exposure: frame 0 shutter_us=")));
        let single = dir.join(format!("single.{name}"));
        let options = ProcessOptions {
            x3i_source: X3iSource::Merge,
            ..options
        };
        convert_file(&extracted, &single, format, &options, &cancel, |_| {}).unwrap();
        assert_eq!(
            std::fs::read(&middle).unwrap(),
            std::fs::read(&single).unwrap(),
            "{name}"
        );
    }
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn middle_scene_equals_the_extracted_frame_scene_not_the_merge() {
    let Some(input) = corpus_file("DP0Q0002.X3I") else {
        return;
    };
    let dir = std::env::temp_dir().join(format!("x3i-middle-scene-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let container = X3iContainer::open(&input).unwrap();
    let index = container.middle_exposure_frame().unwrap();
    let extracted = dir.join("frame.X3F");
    container.extract_frame(index, &extracted).unwrap();
    let cancel = AtomicBool::new(false);
    let options = SceneLinearOptions {
        denoise_intensity: 0,
        highlight_recovery: true,
        ..Default::default()
    };
    let middle = prepare_scene_linear(
        &input,
        &SceneLinearOptions {
            x3i_source: X3iSource::MiddleExposure,
            ..options
        },
        &cancel,
        |_| {},
    )
    .unwrap();
    let single = prepare_scene_linear(&extracted, &options, &cancel, |_| {}).unwrap();
    assert_eq!((middle.width, middle.height), (single.width, single.height));
    assert_eq!(middle.orientation, single.orientation);
    assert_eq!(
        middle.white_balance.camera_to_rgb,
        single.white_balance.camera_to_rgb
    );
    assert_eq!(
        middle.white_balance.xyz_to_camera,
        single.white_balance.xyz_to_camera
    );
    assert_eq!(
        middle.white_balance.as_shot_neutral,
        single.white_balance.as_shot_neutral
    );
    assert_eq!(middle.data, single.data);
    // The merge extends the range past a single frame and differs from it.
    let merged = prepare_scene_linear(
        &input,
        &SceneLinearOptions {
            denoise_intensity: 0,
            ..Default::default()
        },
        &cancel,
        |_| {},
    )
    .unwrap();
    assert_ne!(merged.data, middle.data);
    std::fs::remove_dir_all(dir).unwrap();
}
