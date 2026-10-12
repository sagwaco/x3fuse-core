//! Optional camera corpus coverage for the desktop API and shared Rayon pool.

use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

use rayon::prelude::*;
use x3f_core::{
    convert_file, ConversionStage, DngHighlightMapping, Error, OutputFormat, ProcessOptions,
};

fn fixtures() -> Vec<PathBuf> {
    let directory = std::env::var_os("X3F_TEST_FILES")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../x3f_test_files"));
    let mut files: Vec<_> = std::fs::read_dir(directory)
        .into_iter()
        .flatten()
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.extension()
                .is_some_and(|ext| ext.eq_ignore_ascii_case("x3f"))
        })
        .collect();
    files.sort();
    files.truncate(2);
    files
}

fn assert_same_bytes(left: &Path, right: &Path) {
    let mut left = File::open(left).unwrap();
    let mut right = File::open(right).unwrap();
    assert_eq!(
        left.metadata().unwrap().len(),
        right.metadata().unwrap().len()
    );
    let mut a = [0; 64 * 1024];
    let mut b = [0; 64 * 1024];
    loop {
        let size = left.read(&mut a).unwrap();
        if size == 0 {
            break;
        }
        right.read_exact(&mut b[..size]).unwrap();
        assert_eq!(&a[..size], &b[..size]);
    }
}

fn loaded_reader(file: &Path) -> x3f_core::Reader {
    let mut reader = x3f_core::Reader::open(file).unwrap();
    reader.load_property_list().unwrap();
    reader.load_camf().unwrap();
    reader.load_raw().unwrap();
    reader
}

#[cfg(not(feature = "gpu"))]
#[test]
fn forced_gpu_errors_reach_every_public_processing_path() {
    use x3f_core::{prepare_scene_linear, DenoiseBackend, SceneLinearOptions};
    let files = fixtures();
    if files.is_empty() {
        eprintln!("skip: set X3F_TEST_FILES for forced-GPU API error coverage");
        return;
    }
    let output = std::env::temp_dir().join(format!("x3f-gpu-errors-{}", std::process::id()));
    std::fs::create_dir_all(&output).unwrap();
    let options = ProcessOptions {
        denoise_backend: DenoiseBackend::Gpu,
        ..ProcessOptions::default()
    };
    let cancel = AtomicBool::new(false);
    let assert_gpu_error = |error: Error| {
        assert!(
            matches!(error, Error::Gpu(ref message) if message.contains("no GPU denoiser")),
            "{error:?}"
        );
    };
    for (index, file) in files.iter().enumerate() {
        match loaded_reader(file).get_image(&options) {
            Ok(_) => panic!("forced GPU succeeded in a CPU-only build"),
            Err(error) => assert_gpu_error(error),
        }
        match prepare_scene_linear(
            file,
            &SceneLinearOptions {
                denoise_backend: DenoiseBackend::Gpu,
                ..SceneLinearOptions::default()
            },
            &cancel,
            |_| {},
        ) {
            Ok(_) => panic!("forced GPU succeeded in a CPU-only build"),
            Err(error) => assert_gpu_error(error),
        }
        assert_gpu_error(
            loaded_reader(file)
                .dump_histogram(
                    output.join(format!("histogram-{index}.csv")),
                    &options,
                    false,
                )
                .unwrap_err(),
        );
        for format in [OutputFormat::Dng, OutputFormat::Tiff] {
            let path = output.join(format!("{index}-{format:?}"));
            let error = convert_file(file, &path, format, &options, &cancel, |_| {}).unwrap_err();
            assert_gpu_error(error);
            assert!(!path.exists(), "failed conversion published an output");
        }
    }
    std::fs::remove_dir_all(output).unwrap();
}

#[test]
fn histogram_ignores_dng_highlight_and_tiff_cineon_options() {
    let files = fixtures();
    if files.is_empty() {
        eprintln!("skip: set X3F_TEST_FILES for histogram compatibility coverage");
        return;
    }
    let output = std::env::temp_dir().join(format!("x3f-histogram-options-{}", std::process::id()));
    std::fs::create_dir_all(&output).unwrap();
    let base = ProcessOptions {
        color_encoding: x3f_core::ColorEncoding::None,
        denoise_backend: x3f_core::DenoiseBackend::Cpu,
        denoise_intensity: 0,
        ..ProcessOptions::default()
    };
    for (index, file) in files.iter().enumerate() {
        let reference = output.join(format!("{index}-reference.csv"));
        loaded_reader(file)
            .dump_histogram(&reference, &base, false)
            .unwrap();
        for (name, options) in [
            (
                "recovery",
                ProcessOptions {
                    dng_highlight_recovery: true,
                    dng_highlight_mapping: DngHighlightMapping::Shoulder,
                    ..base.clone()
                },
            ),
            (
                "cineon",
                ProcessOptions {
                    cineon: true,
                    ..base.clone()
                },
            ),
        ] {
            let path = output.join(format!("{index}-{name}.csv"));
            loaded_reader(file)
                .dump_histogram(&path, &options, false)
                .unwrap();
            assert_same_bytes(&reference, &path);
        }
    }
    std::fs::remove_dir_all(output).unwrap();
}

#[test]
fn concurrent_conversions_match_serial_for_per_file_options() {
    let files = fixtures();
    if files.is_empty() {
        eprintln!("skip: set X3F_TEST_FILES for camera conversion parity");
        return;
    }
    let output = std::env::temp_dir().join(format!("x3f-concurrent-{}", std::process::id()));
    std::fs::create_dir_all(&output).unwrap();
    let mut jobs = Vec::new();
    for (index, file) in files.iter().enumerate() {
        let normal = ProcessOptions {
            denoise_intensity: 0,
            ..ProcessOptions::default()
        };
        jobs.push((file, OutputFormat::Dng, normal.clone()));
        jobs.push((file, OutputFormat::Tiff, normal.clone()));
        jobs.push((
            file,
            OutputFormat::Dng,
            ProcessOptions {
                compress: true,
                dng_highlight_recovery: true,
                dng_highlight_mapping: if index == 0 {
                    DngHighlightMapping::Linear
                } else {
                    DngHighlightMapping::Shoulder
                },
                ..normal.clone()
            },
        ));
        jobs.push((
            file,
            OutputFormat::Tiff,
            ProcessOptions {
                compress: true,
                cineon: true,
                ..normal
            },
        ));
    }
    // Append so the four baseline cases above keep their existing indices.
    // This fixture actually changes with the LUT; the first two Quattros do not.
    let clipped = files[0].with_file_name("CLIPPED_IMAGE_MERRILL.X3F");
    let lut_start = jobs.len();
    if clipped.is_file() {
        for enabled in [true, false] {
            jobs.push((
                &clipped,
                OutputFormat::Tiff,
                ProcessOptions {
                    denoise_intensity: 0,
                    rgb_chroma_lut: Some(enabled),
                    ..ProcessOptions::default()
                },
            ));
        }
    } else {
        eprintln!("skip: CLIPPED_IMAGE_MERRILL.X3F missing for concurrent LUT overrides");
    }
    let cancel = AtomicBool::new(false);
    for (index, (file, format, options)) in jobs.iter().enumerate() {
        convert_file(
            file,
            &output.join(format!("serial-{index}")),
            *format,
            options,
            &cancel,
            |_| {},
        )
        .unwrap();
    }
    if clipped.is_file() {
        assert!(
            std::fs::read(output.join(format!("serial-{lut_start}"))).unwrap()
                != std::fs::read(output.join(format!("serial-{}", lut_start + 1))).unwrap(),
            "concurrent LUT fixture must exercise different pixels"
        );
    }
    // The inner row-parallel processing must safely share these same workers
    // with another Reader using different highlight and Cineon options.
    rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap()
        .install(|| {
            jobs.par_iter()
                .enumerate()
                .for_each(|(index, (file, format, options))| {
                    let parallel = output.join(format!("parallel-{index}"));
                    convert_file(file, &parallel, *format, options, &cancel, |_| {}).unwrap();
                    assert_same_bytes(&output.join(format!("serial-{index}")), &parallel);
                });
        });
    // An optional independent pre-change CLI baseline checks default pixels
    // and DNG metadata, rather than comparing only two current API calls.
    if let Some(baseline) = std::env::var_os("X3F_BASELINE_DIR") {
        for (index, file) in files.iter().enumerate() {
            for (offset, extension) in [(0, "dng"), (1, "tif")] {
                let name = format!(
                    "{}.{}",
                    file.file_name().unwrap().to_string_lossy(),
                    extension
                );
                assert_same_bytes(
                    &PathBuf::from(&baseline).join(name),
                    &output.join(format!("serial-{}", index * 4 + offset)),
                );
            }
        }
    }
    std::fs::remove_dir_all(output).unwrap();
}

#[test]
fn cancellation_interrupts_camera_processing_and_encoding() {
    use std::time::{Duration, Instant};
    let files = fixtures();
    if files.is_empty() {
        eprintln!("skip: set X3F_TEST_FILES for camera cancellation");
        return;
    }
    let output = std::env::temp_dir().join(format!("x3f-cancel-{}.dng", std::process::id()));
    for file in files {
        for stage in [
            ConversionStage::Decode,
            ConversionStage::Process,
            ConversionStage::Write,
        ] {
            let cancel = AtomicBool::new(false);
            let (sender, receiver) = std::sync::mpsc::channel();
            let options = ProcessOptions {
                compress: true,
                denoise_intensity: if stage == ConversionStage::Process {
                    10
                } else {
                    0
                },
                ..ProcessOptions::default()
            };
            std::thread::scope(|scope| {
                let cancel_ref = &cancel;
                let canceller = scope.spawn(move || {
                    receiver.recv_timeout(Duration::from_secs(30)).unwrap();
                    std::thread::sleep(Duration::from_millis(10));
                    cancel_ref.store(true, Ordering::Relaxed);
                    Instant::now()
                });
                let result = convert_file(
                    &file,
                    &output,
                    OutputFormat::Dng,
                    &options,
                    &cancel,
                    |current| {
                        if current == stage {
                            sender.send(()).unwrap();
                        }
                    },
                );
                drop(sender);
                assert!(matches!(result, Err(Error::Cancelled)), "{result:?}");
                assert!(!output.exists());
                assert!(canceller.join().unwrap().elapsed() < Duration::from_secs(5));
            });
        }
    }
}

#[test]
fn spp_errors_and_cancellation_do_not_publish_output() {
    use std::time::{Duration, Instant};
    use x3f_core::{ColorEncoding, DenoiseBackend};

    let directory = std::env::var_os("X3F_TEST_FILES")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../x3f_test_files"));
    let options = ProcessOptions {
        spp_denoise: true,
        denoise_backend: DenoiseBackend::Cpu,
        ..ProcessOptions::default()
    };
    let output = std::env::temp_dir().join(format!("x3f-spp-errors-{}.tif", std::process::id()));
    let cancel = AtomicBool::new(false);
    let mut checked = 0;
    for name in ["_P2M0358.X3F", "sigma_sd1_merrill_15.x3f", "_SDI8284.X3F"] {
        let file = directory.join(name);
        if !file.is_file() {
            continue;
        }
        checked += 1;
        if name == "_SDI8284.X3F" {
            let error = convert_file(
                &file,
                &output,
                OutputFormat::Tiff,
                &options,
                &cancel,
                |_| {},
            )
            .unwrap_err();
            assert!(matches!(error, Error::InvalidData(_)), "{error:?}");
            assert!(!output.exists());
            continue;
        }
        for format in [OutputFormat::Dng, OutputFormat::Jpeg] {
            let error =
                convert_file(&file, &output, format, &options, &cancel, |_| {}).unwrap_err();
            assert!(matches!(error, Error::InvalidData(_)), "{error:?}");
            assert!(!output.exists());
        }
        let reader = loaded_reader(&file);
        for invalid in [
            ProcessOptions {
                rgb_chroma_lut: Some(true),
                ..options.clone()
            },
            ProcessOptions {
                denoise_intensity: 0,
                ..options.clone()
            },
            ProcessOptions {
                cineon: true,
                ..options.clone()
            },
            ProcessOptions {
                color_encoding: ColorEncoding::None,
                ..options.clone()
            },
            ProcessOptions {
                denoise_backend: DenoiseBackend::Gpu,
                ..options.clone()
            },
        ] {
            match reader.get_image(&invalid) {
                Ok(_) => panic!("invalid SPP options succeeded"),
                Err(error) => assert!(
                    matches!(error, Error::InvalidData(_) | Error::Gpu(_)),
                    "{error:?}"
                ),
            }
            for existing in [false, true] {
                if existing {
                    std::fs::write(&output, b"existing histogram").unwrap();
                }
                assert!(reader.dump_histogram(&output, &invalid, false).is_err());
                if existing {
                    assert_eq!(std::fs::read(&output).unwrap(), b"existing histogram");
                    std::fs::remove_file(&output).unwrap();
                } else {
                    assert!(!output.exists());
                }
            }
        }
        drop(reader);
        let (sender, receiver) = std::sync::mpsc::channel();
        cancel.store(false, Ordering::Relaxed);
        std::thread::scope(|scope| {
            let cancel_ref = &cancel;
            let canceller = scope.spawn(move || {
                receiver.recv_timeout(Duration::from_secs(30)).unwrap();
                std::thread::sleep(Duration::from_millis(25));
                cancel_ref.store(true, Ordering::Relaxed);
                Instant::now()
            });
            let result = convert_file(
                &file,
                &output,
                OutputFormat::Tiff,
                &options,
                &cancel,
                |stage| {
                    if stage == ConversionStage::Process {
                        sender.send(()).unwrap();
                    }
                },
            );
            drop(sender);
            assert!(matches!(result, Err(Error::Cancelled)), "{result:?}");
            assert!(!output.exists());
            assert!(canceller.join().unwrap().elapsed() < Duration::from_secs(5));
        });
        cancel.store(false, Ordering::Relaxed);
    }
    if checked == 0 {
        eprintln!("skip: set X3F_TEST_FILES for SPP API errors and cancellation");
    }
}
