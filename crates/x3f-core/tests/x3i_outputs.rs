//! Corpus coverage for merged X3I scene, TIFF and PPM output.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

use x3f_core::{
    convert_file, prepare_scene_linear, ConversionStage, Error, OutputFormat, ProcessOptions,
    SceneLinearOptions,
};

#[test]
fn merged_scene_and_rendered_outputs_preserve_the_public_contract() {
    let corpus = std::env::var_os("X3F_TEST_FILES")
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("../../x3f_test_files"));
    let directory = std::env::temp_dir().join(format!("x3i-outputs-{}", std::process::id()));
    std::fs::create_dir_all(&directory).unwrap();
    let cancel = AtomicBool::new(false);
    for name in ["DP0Q0002.X3I", "_DQH0978.X3I"] {
        let input = corpus.join(name);
        if !input.is_file() {
            eprintln!("skip: missing {name} for X3I outputs");
            continue;
        }
        let mut stages = Vec::new();
        let scene = prepare_scene_linear(
            &input,
            &SceneLinearOptions {
                denoise_intensity: 0,
                ..Default::default()
            },
            &cancel,
            |stage| stages.push(stage),
        )
        .unwrap();
        assert_eq!(
            scene.data.len(),
            scene.width as usize * scene.height as usize * 3
        );
        assert!(scene.data.iter().all(|v| v.is_finite()));
        assert!((1..=8).contains(&scene.orientation));
        assert_eq!(
            stages
                .iter()
                .filter(|&&v| v == ConversionStage::Process)
                .count(),
            7
        );
        assert_eq!(
            stages
                .iter()
                .filter(|&&v| v == ConversionStage::Decode)
                .count(),
            7
        );
        let merge_stages: Vec<_> = stages
            .iter()
            .copied()
            .filter(|s| matches!(s, ConversionStage::Decode | ConversionStage::Process))
            .collect();
        assert_eq!(merge_stages[..7], [ConversionStage::Decode; 7]);
        assert_eq!(merge_stages[7..], [ConversionStage::Process; 7]);
        assert!(!stages.contains(&ConversionStage::Write));
        if name == "DP0Q0002.X3I" {
            assert!(scene.data.iter().copied().fold(f32::NEG_INFINITY, f32::max) > 1.0);
        }
        assert!(scene.color_mode_calibration.is_some(), "{name}");
        let dimensions = (scene.width, scene.height);
        let expected_dimensions = if name == "DP0Q0002.X3I" {
            (5424, 3616)
        } else {
            (6192, 4128)
        };
        assert_eq!(dimensions, expected_dimensions);
        eprintln!("{name} active dimensions: {dimensions:?}");
        drop(scene);
        let options = ProcessOptions {
            denoise_intensity: 0,
            ..Default::default()
        };
        for format in [OutputFormat::Tiff, OutputFormat::Ppm] {
            let output = directory.join(format!("{name}.{format:?}"));
            convert_file(&input, &output, format, &options, &cancel, |_| {}).unwrap();
            if format == OutputFormat::Tiff {
                let mut decoder =
                    tiff::decoder::Decoder::new(std::fs::File::open(&output).unwrap()).unwrap();
                assert_eq!(decoder.dimensions().unwrap(), dimensions);
                assert!(!decoder
                    .get_tag_u8_vec(tiff::tags::Tag::Unknown(34675))
                    .unwrap()
                    .is_empty());
                assert!(matches!(
                    decoder.read_image().unwrap(),
                    tiff::decoder::DecodingResult::U16(_)
                ));
            } else {
                let bytes = std::fs::read(&output).unwrap();
                let header = format!("P6\n{} {}\n65535\n", dimensions.0, dimensions.1);
                assert!(bytes.starts_with(header.as_bytes()));
                assert_eq!(
                    bytes.len(),
                    header.len() + dimensions.0 as usize * dimensions.1 as usize * 6
                );
            }
            assert!(
                convert_file(&input, &output, format, &options, &cancel, |_| panic!(
                    "existing output"
                ))
                .is_err()
            );
            std::fs::remove_file(&output).unwrap();
            for cancel_at in [
                ConversionStage::Read,
                ConversionStage::Decode,
                ConversionStage::Process,
                ConversionStage::Write,
            ] {
                let result = convert_file(&input, &output, format, &options, &cancel, |stage| {
                    if stage == cancel_at {
                        cancel.store(true, Ordering::Relaxed);
                    }
                });
                assert!(
                    matches!(result, Err(Error::Cancelled)),
                    "{name}: {format:?} {cancel_at:?}"
                );
                assert!(!output.exists());
                cancel.store(false, Ordering::Relaxed);
            }
        }
        let result = prepare_scene_linear(
            &input,
            &SceneLinearOptions {
                denoise_intensity: 0,
                ..Default::default()
            },
            &cancel,
            |stage| {
                if stage == ConversionStage::Process {
                    cancel.store(true, Ordering::Relaxed);
                }
            },
        );
        assert!(matches!(result, Err(Error::Cancelled)));
        cancel.store(false, Ordering::Relaxed);
    }
    std::fs::remove_dir_all(directory).unwrap();
}
