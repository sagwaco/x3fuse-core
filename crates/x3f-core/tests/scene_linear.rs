//! Optional real-camera coverage for the immutable editor handoff.
use std::{
    hash::{Hash, Hasher},
    path::PathBuf,
    sync::atomic::AtomicBool,
};
use x3f_core::{prepare_scene_linear, read_as_shot_crop, ConversionStage, SceneLinearOptions};

#[test]
fn camera_sources_preserve_headroom_and_repeat_from_fresh_readers() {
    let corpus = std::env::var_os("X3F_TEST_FILES")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../x3f_test_files"));
    let options = SceneLinearOptions {
        denoise_intensity: 0,
        ..SceneLinearOptions::default()
    };
    let cancel = AtomicBool::new(false);
    let mut checked = 0;
    let mut extended = false;
    for name in [
        "sigma_sd1_merrill_10.x3f",
        "_SDI8040.X3F",
        "_SDI8284.X3F",
        "SDQH5085.X3F",
    ] {
        let path = corpus.join(name);
        if !path.is_file() {
            continue;
        }
        let crop = read_as_shot_crop(&path).unwrap().unwrap();
        assert!(crop
            .iter()
            .all(|v| v.is_finite() && (0.0..=1.0).contains(v)));
        assert!(crop[2] > 0.0 && crop[3] > 0.0);
        assert!((2.0 * crop[0] + crop[2] - 1.0).abs() < 1e-6);
        assert!((2.0 * crop[1] + crop[3] - 1.0).abs() < 1e-6);
        assert_eq!(read_as_shot_crop(&path).unwrap(), Some(crop));
        eprintln!("{name}: as-shot crop {crop:?}");
        for recovery in [false, true] {
            let options = SceneLinearOptions {
                highlight_recovery: recovery,
                ..options
            };
            let mut previous = None;
            for _ in 0..2 {
                let mut stages = Vec::new();
                let image =
                    prepare_scene_linear(&path, &options, &cancel, |s| stages.push(s)).unwrap();
                assert_eq!(
                    stages,
                    [
                        ConversionStage::Read,
                        ConversionStage::Decode,
                        ConversionStage::Process
                    ]
                );
                assert_eq!(
                    image.data.len(),
                    image.width as usize * image.height as usize * 3
                );
                assert!((1..=8).contains(&image.orientation));
                assert!(image.data.iter().all(|v| v.is_finite()));
                let low = image.data.iter().copied().fold(f32::INFINITY, f32::min);
                let high = image.data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                extended |= low < 0.0 || high > 1.0;
                let mut hash = std::collections::hash_map::DefaultHasher::new();
                for value in &image.data {
                    value.to_bits().hash(&mut hash);
                }
                let digest = hash.finish();
                if let Some(previous) = previous {
                    assert_eq!(digest, previous, "{name}, recovery={recovery}");
                }
                previous = Some(digest);
                eprintln!(
                    "{name}, recovery={recovery}: {}x{}, range {low}..{high}",
                    image.width, image.height
                );
            }
        }
        checked += 1;
    }
    if checked == 0 {
        eprintln!("skip: set X3F_TEST_FILES for scene-linear camera coverage");
    } else {
        assert!(extended, "camera sources unexpectedly all display-clipped");
    }
}
