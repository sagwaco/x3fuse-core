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
