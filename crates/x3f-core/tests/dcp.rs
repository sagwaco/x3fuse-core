//! DCP file handling and optional camera-corpus embedding checks.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::{Cursor, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;

use tiff::decoder::{ifd::Value, Decoder};
use tiff::tags::{IfdPointer, Tag};
use x3f_core::{convert_file, dcp::DcpLook, Error, OutputFormat, ProcessOptions, Reader};

#[path = "dcp_fixture.rs"]
mod fixture;

struct Scratch(PathBuf);

impl Scratch {
    fn new(name: &str) -> Self {
        let path = std::env::temp_dir().join(format!("x3f-dcp-{name}-{}", std::process::id()));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[test]
fn file_loading_handles_unicode_missing_directories_and_oversize() {
    let scratch = Scratch::new("loading");
    let path = scratch.0.join("色 ' look.dcp");
    fs::write(&path, fixture::profile("Test camera", false)).unwrap();
    assert_eq!(DcpLook::open(&path).unwrap().name(), "Test look");
    assert!(DcpLook::open(scratch.0.join("missing.dcp")).is_err());
    assert!(DcpLook::open(&scratch.0).is_err());
    let mut file = File::create(&path).unwrap();
    file.write_all(b"II").unwrap();
    file.set_len(16 * 1024 * 1024 + 1).unwrap();
    assert!(DcpLook::open(&path).is_err());
}

#[test]
fn non_dng_requests_fail_before_opening_input_or_output() {
    let scratch = Scratch::new("format");
    let source = scratch.0.join("missing.X3F");
    let output = scratch.0.join("result");
    let options = ProcessOptions {
        dng_look: Some(scratch.0.join("look.dcp")),
        ..ProcessOptions::default()
    };
    for format in [OutputFormat::Tiff, OutputFormat::Jpeg] {
        let error = convert_file(
            &source,
            &output,
            format,
            &options,
            &AtomicBool::new(false),
            |_| {},
        )
        .unwrap_err();
        assert!(matches!(error, Error::InvalidData(_)));
        assert!(!output.exists());
    }
    fs::write(&output, b"keep").unwrap();
    assert!(convert_file(
        &source,
        &output,
        OutputFormat::Tiff,
        &options,
        &AtomicBool::new(false),
        |_| {}
    )
    .is_err());
    assert_eq!(fs::read(&output).unwrap(), b"keep");
}

fn camera_fixture() -> Option<PathBuf> {
    let directory = std::env::var_os("X3F_TEST_FILES")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../x3f_test_files"));
    let mut files: Vec<_> = fs::read_dir(directory)
        .into_iter()
        .flatten()
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.extension()
                .is_some_and(|extension| extension.eq_ignore_ascii_case("x3f"))
        })
        .collect();
    files.sort();
    if files.is_empty() {
        eprintln!("skip: set X3F_TEST_FILES for DCP camera conversion checks");
    }
    files.into_iter().next()
}

type Fields = BTreeMap<u16, Value>;

struct Dng {
    bytes: Vec<u8>,
    root: Fields,
    raw: Fields,
}

impl Dng {
    fn read(path: &Path) -> Self {
        let bytes = fs::read(path).unwrap();
        let mut decoder = Decoder::new(Cursor::new(&bytes)).unwrap();
        let raw_offset = decoder.get_tag_u32_vec(Tag::Unknown(330)).unwrap()[0];
        let root = decoder
            .tag_iter()
            .map(|result| {
                let (tag, value) = result.unwrap();
                (tag.to_u16(), value)
            })
            .collect();
        let directory = decoder
            .read_directory(IfdPointer(raw_offset.into()))
            .unwrap();
        let raw = decoder
            .read_directory_tags(&directory)
            .tag_iter()
            .map(|result| {
                let (tag, value) = result.unwrap();
                (tag.to_u16(), value)
            })
            .collect();
        Self { bytes, root, raw }
    }

    fn strips(&self, fields: &Fields) -> Vec<&[u8]> {
        let offsets = fields[&273].clone().into_u32_vec().unwrap();
        let counts = fields[&279].clone().into_u32_vec().unwrap();
        offsets
            .into_iter()
            .zip(counts)
            .map(|(offset, count)| &self.bytes[offset as usize..offset as usize + count as usize])
            .collect()
    }
}

fn convert(source: &Path, output: &Path, options: &ProcessOptions) {
    convert_file(
        source,
        output,
        OutputFormat::Dng,
        options,
        &AtomicBool::new(false),
        |_| {},
    )
    .unwrap();
}

#[test]
fn look_preserves_raw_preview_calibration_and_recovery_output() {
    let Some(source) = camera_fixture() else {
        return;
    };
    let scratch = Scratch::new("embedding");
    for compress in [false, true] {
        for recovery in [false, true] {
            let options = ProcessOptions {
                compress,
                denoise_intensity: 0,
                dng_highlight_recovery: recovery,
                ..ProcessOptions::default()
            };
            let label = format!("{compress}-{recovery}");
            let baseline = scratch.0.join(format!("baseline-{label}.dng"));
            let profiled = scratch.0.join(format!("look-{label}.dng"));
            convert(&source, &baseline, &options);
            let before = Dng::read(&baseline);
            assert_ne!(
                before.root[&50708].clone().into_string().unwrap(),
                "Another camera"
            );
            let path = scratch.0.join("look.dcp");
            fs::write(&path, fixture::profile("Another camera", compress)).unwrap();
            convert(
                &source,
                &profiled,
                &ProcessOptions {
                    dng_look: Some(path),
                    ..options
                },
            );
            let after = Dng::read(&profiled);
            assert_eq!(before.raw, after.raw);
            assert_eq!(before.strips(&before.raw), after.strips(&after.raw));
            assert_eq!(before.strips(&before.root), after.strips(&after.root));
            for (tag, value) in &before.root {
                if ![50707, 50940].contains(tag) {
                    assert_eq!(after.root.get(tag), Some(value), "changed tag {tag}");
                }
            }
            let mut expected_tags: BTreeSet<_> = before.root.keys().copied().collect();
            expected_tags.extend([50940, 50981, 50982, 51108]);
            assert_eq!(
                after.root.keys().copied().collect::<BTreeSet<_>>(),
                expected_tags
            );
            assert_eq!(
                after.root[&50981].clone().into_u32_vec().unwrap(),
                [2, 2, 2]
            );
            assert_eq!(
                after.root[&50982].clone().into_f32_vec().unwrap(),
                fixture::TABLE
            );
            assert_eq!(
                after.root[&50940].clone().into_f32_vec().unwrap(),
                fixture::TONE
            );
            assert_eq!(after.root[&51108].clone().into_u32().unwrap(), 1);
            assert_eq!(
                after.root[&50707].clone().into_u32_vec().unwrap(),
                [1, 4, 0, 0]
            );
        }
    }
}

#[test]
fn invalid_look_leaves_no_conversion_output_or_recovery_mask() {
    let Some(source) = camera_fixture() else {
        return;
    };
    let scratch = Scratch::new("rejection");
    let look = scratch.0.join("look.dcp");
    let output = scratch.0.join("output.dng");
    let mask = scratch.0.join("mask.tif");
    let options = ProcessOptions {
        dng_look: Some(look.clone()),
        dng_highlight_recovery: true,
        dng_recovery_mask: Some(mask.clone()),
        denoise_intensity: 0,
        ..ProcessOptions::default()
    };
    let truncated = fixture::profile("Any camera", false)[..16].to_vec();
    for bytes in [b"not a DCP".to_vec(), truncated] {
        fs::write(&look, bytes).unwrap();
        let result = convert_file(
            &source,
            &output,
            OutputFormat::Dng,
            &options,
            &AtomicBool::new(false),
            |_| {},
        );
        assert!(matches!(result, Err(Error::InvalidData(_))));
        assert!(!output.exists());
        assert!(!mask.exists());
    }
    // The lower-level writer must also reject the look before opening its output.
    fs::write(&output, b"keep").unwrap();
    let mut reader = Reader::open(&source).unwrap();
    reader.load_property_list().unwrap();
    reader.load_camf().unwrap();
    reader.load_raw().unwrap();
    assert!(reader.dump_dng(&output, &options).is_err());
    assert_eq!(fs::read(&output).unwrap(), b"keep");
    assert!(!mask.exists());
}
