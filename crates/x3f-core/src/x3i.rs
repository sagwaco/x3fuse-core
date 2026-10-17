//! X3I containers with independent, bounded X3F frame readers.

use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use crate::{Error, Reader};

const HEADER_SIZE: u64 = 0x5a0;

/// Byte range of a complete X3F frame in an X3I container.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameEntry {
    /// Byte offset from the start of the container.
    pub offset: u64,
    /// Length of the complete X3F file in bytes.
    pub length: u64,
}

/// Validated X3I directory and sensor dimensions.
#[derive(Debug)]
pub struct X3iContainer {
    path: PathBuf,
    frames: Vec<FrameEntry>,
    sensor: (u32, u32),
}

fn invalid(message: &str) -> Error {
    Error::InvalidData(format!("X3I: {message}"))
}

fn io_error(path: &Path, source: std::io::Error) -> Error {
    Error::Io {
        path: path.display().to_string(),
        source,
    }
}

fn input_error(path: &Path, source: std::io::Error) -> Error {
    Error::OpenInput {
        path: path.display().to_string(),
        source,
    }
}

fn word(file: &mut File) -> std::io::Result<u32> {
    let mut bytes = [0; 4];
    file.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

/// Return true when the file starts with the X3I magic, regardless of its name.
pub fn is_x3i(path: impl AsRef<Path>) -> Result<bool, Error> {
    let path = path.as_ref();
    let mut file = File::open(path).map_err(|e| input_error(path, e))?;
    let mut magic = [0; 4];
    match file.read_exact(&mut magic) {
        Ok(()) => Ok(&magic == b"FOVi"),
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => Ok(false),
        Err(e) => Err(input_error(path, e)),
    }
}

/// Exposure difference in stops, computed as log2(frame / reference).
/// Both exposure values must be positive.
pub fn frame_ev(reference_us: u32, frame_us: u32) -> f64 {
    (f64::from(frame_us) / f64::from(reference_us)).log2()
}

impl X3iContainer {
    /// Parse the outer directory and validate each frame's X3F directory.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, Error> {
        let path = path.as_ref();
        let mut file = File::open(path).map_err(|e| input_error(path, e))?;
        let parsed = (|| -> Result<Self, Error> {
            let size = file.metadata().map_err(|e| input_error(path, e))?.len();
            if size < HEADER_SIZE + 16 {
                return Err(invalid("truncated container"));
            }
            let read = |file: &mut File| word(file).map_err(|e| input_error(path, e));
            if read(&mut file)? != u32::from_le_bytes(*b"FOVi") {
                return Err(invalid("invalid container magic"));
            }
            file.seek(SeekFrom::Start(0x28))
                .map_err(|e| input_error(path, e))?;
            let sensor = (read(&mut file)?, read(&mut file)?);
            file.seek(SeekFrom::Start(size - 4))
                .map_err(|e| input_error(path, e))?;
            let directory = u64::from(read(&mut file)?);
            if directory < HEADER_SIZE || directory > size - 16 {
                return Err(invalid("directory outside container data"));
            }
            file.seek(SeekFrom::Start(directory))
                .map_err(|e| input_error(path, e))?;
            if read(&mut file)? != u32::from_le_bytes(*b"SECf") {
                return Err(invalid("invalid directory magic"));
            }
            let _version = read(&mut file)?;
            let count = read(&mut file)?;
            if u64::from(count) > (size - 4 - directory - 12) / 12 {
                return Err(invalid("directory count exceeds input"));
            }
            let mut frames = Vec::new();
            for _ in 0..count {
                let offset = read(&mut file)?;
                let length = read(&mut file)?;
                let kind = read(&mut file)?;
                let end = offset
                    .checked_add(length)
                    .ok_or_else(|| invalid("entry range overflow"))?;
                if u64::from(offset) < HEADER_SIZE || u64::from(end) > directory {
                    return Err(invalid("entry outside container data"));
                }
                if kind == u32::from_le_bytes(*b"FRAM") {
                    frames
                        .try_reserve(1)
                        .map_err(|_| invalid("cannot allocate frame directory"))?;
                    frames.push(FrameEntry {
                        offset: offset.into(),
                        length: length.into(),
                    });
                }
            }
            if frames.is_empty() {
                return Err(invalid("no FRAM entries"));
            }
            Ok(Self {
                path: path.to_path_buf(),
                frames,
                sensor,
            })
        })()?;
        for i in 0..parsed.frames.len() {
            parsed.open_frame(i)?;
        }
        Ok(parsed)
    }

    /// Return frame ranges in directory order.
    pub fn frames(&self) -> &[FrameEntry] {
        &self.frames
    }

    /// Return the sensor width and height from the outer header.
    pub fn sensor_size(&self) -> (u32, u32) {
        self.sensor
    }

    fn frame(&self, index: usize) -> Result<FrameEntry, Error> {
        self.frames
            .get(index)
            .copied()
            .ok_or_else(|| invalid("frame index out of range"))
    }

    /// Open a frame on its own stream with all offsets relative to that frame.
    pub fn open_frame(&self, index: usize) -> Result<Reader, Error> {
        self.open_frame_with_control(index, x3f_sys::Control::none())
    }

    pub(crate) fn open_frame_with_control(
        &self,
        index: usize,
        control: x3f_sys::Control<'_>,
    ) -> Result<Reader, Error> {
        let frame = self.frame(index)?;
        Reader::open_window(&self.path, Some((frame.offset, frame.length)), control)
    }

    /// Return the index of the middle exposure: the frame with the median
    /// CaptureShutter. An even count takes the shorter of the two middle
    /// exposures. Equal shutters keep file order.
    pub fn middle_exposure_frame(&self) -> Result<usize, Error> {
        let exposures = (0..self.frames.len())
            .map(|index| self.frame_exposure_us(index))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(median_index(&exposures))
    }

    /// Load the frame's CaptureShutter value in microseconds.
    pub fn frame_exposure_us(&self, index: usize) -> Result<u32, Error> {
        let mut reader = self.open_frame(index)?;
        reader.load_camf()?;
        let us = reader
            .dng_camf_float("CaptureShutter")
            .filter(|us| us.is_finite() && *us >= 1.0 && *us <= f64::from(u32::MAX))
            .ok_or_else(|| invalid("missing or invalid CaptureShutter"))?;
        Ok(us.round() as u32)
    }

    /// Copy a complete frame to a new file. An existing file is never replaced.
    pub fn extract_frame(&self, index: usize, out: &Path) -> Result<(), Error> {
        let frame = self.frame(index)?;
        let mut source = File::open(&self.path).map_err(|e| input_error(&self.path, e))?;
        source
            .seek(SeekFrom::Start(frame.offset))
            .map_err(|e| input_error(&self.path, e))?;
        let mut destination = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(out)
            .map_err(|e| io_error(out, e))?;
        let result = (|| {
            let copied = std::io::copy(&mut source.take(frame.length), &mut destination)?;
            if copied != frame.length {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "truncated X3I frame",
                ));
            }
            destination.sync_all()
        })();
        drop(destination);
        if let Err(error) = result {
            let _ = std::fs::remove_file(out);
            return Err(io_error(out, error));
        }
        Ok(())
    }
}

/// Index of the median value. A stable sort keeps file order for ties, and
/// `(n - 1) / 2` takes the lower middle of an even count.
fn median_index(exposures: &[u32]) -> usize {
    let mut order: Vec<usize> = (0..exposures.len()).collect();
    order.sort_by_key(|&index| exposures[index]);
    order[(order.len() - 1) / 2]
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    #[test]
    fn middle_exposure_is_the_lower_median_with_stable_ties() {
        assert_eq!(median_index(&[500]), 0);
        assert_eq!(median_index(&[100, 400, 1600]), 1);
        assert_eq!(median_index(&[1600, 100, 400]), 2);
        // Even count: the shorter of the two middle exposures.
        assert_eq!(median_index(&[1600, 100, 400, 800]), 2);
        // Ties keep file order.
        assert_eq!(median_index(&[200, 200, 200]), 1);
        assert_eq!(median_index(&[200, 200, 200, 200]), 1);
        assert_eq!(median_index(&[300, 100, 100, 300]), 2);
    }

    struct Scratch(PathBuf);
    impl Scratch {
        fn new() -> Self {
            static NEXT: AtomicUsize = AtomicUsize::new(0);
            let path = std::env::temp_dir().join(format!(
                "x3i-core-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::create_dir(&path).unwrap();
            Self(path)
        }
        fn write(&self, data: &[u8]) -> PathBuf {
            let path = self.0.join("container.bin");
            std::fs::write(&path, data).unwrap();
            path
        }
    }
    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn put(bytes: &mut [u8], offset: usize, value: u32) {
        bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }

    // Use the minimal X3F layout from the low-level directory tests.
    fn frame(version: u32) -> Vec<u8> {
        let mut bytes = b"FOVb".to_vec();
        bytes.extend(version.to_le_bytes());
        bytes.extend([0; 16]);
        for word in [u32::from_le_bytes(*b"SECp"), version, 0, 0, 0, 0] {
            bytes.extend(word.to_le_bytes());
        }
        bytes.extend(b"SECd");
        for word in [0x20000u32, 1, 24, 24, u32::from_le_bytes(*b"PROP"), 48] {
            bytes.extend(word.to_le_bytes());
        }
        bytes
    }

    fn container() -> Vec<u8> {
        let mut bytes = vec![0; HEADER_SIZE as usize];
        bytes[..4].copy_from_slice(b"FOVi");
        put(&mut bytes, 0x28, 5424);
        put(&mut bytes, 0x2c, 3616);
        let mut entries = Vec::new();
        for version in [0x40000, 0x40001] {
            let frame = frame(version);
            entries.push((bytes.len() as u32, frame.len() as u32));
            bytes.extend(frame);
        }
        let directory = bytes.len() as u32;
        bytes.extend(b"SECf");
        bytes.extend(0x100u32.to_le_bytes());
        bytes.extend(2u32.to_le_bytes());
        for (offset, length) in entries {
            bytes.extend(offset.to_le_bytes());
            bytes.extend(length.to_le_bytes());
            bytes.extend(b"FRAM");
        }
        bytes.extend(directory.to_le_bytes());
        bytes
    }

    #[test]
    fn frames_have_independent_windows_and_extract_exact_bytes() {
        let scratch = Scratch::new();
        let bytes = container();
        let path = scratch.write(&bytes);
        assert!(is_x3i(&path).unwrap());
        let container = X3iContainer::open(&path).unwrap();
        assert_eq!(container.sensor_size(), (5424, 3616));
        assert_eq!(container.frames().len(), 2);
        let mut first = container.open_frame(0).unwrap();
        let mut second = container.open_frame(1).unwrap();
        assert_eq!(first.header_version(), 0x40000);
        assert_eq!(second.header_version(), 0x40001);
        second.load_property_list().unwrap();
        first.load_property_list().unwrap();
        unsafe {
            assert_eq!(
                (*x3f_sys::x3f_get_prop(first.x3f.as_ptr())).header.version,
                0x40000
            );
            assert_eq!(
                (*x3f_sys::x3f_get_prop(second.x3f.as_ptr())).header.version,
                0x40001
            );
        }
        assert!(container.open_frame(2).is_err());
        assert!(container.frame_exposure_us(2).is_err());
        let out = scratch.0.join("frame.X3F");
        container.extract_frame(0, &out).unwrap();
        assert_eq!(std::fs::read(&out).unwrap(), frame(0x40000));
        assert!(!is_x3i(&out).unwrap());
        assert!(container.extract_frame(1, &out).is_err());
        assert_eq!(std::fs::read(&out).unwrap(), frame(0x40000));
        let error = Reader::open(&path).err().unwrap().to_string();
        assert!(
            error.contains("X3I") && error.contains("X3iContainer"),
            "{error}"
        );
    }

    #[test]
    fn truncated_source_does_not_leave_a_partial_extract() {
        let scratch = Scratch::new();
        let path = scratch.write(&container());
        let container = X3iContainer::open(&path).unwrap();
        let frame = container.frames()[0];
        OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len(frame.offset + frame.length - 1)
            .unwrap();
        let out = scratch.0.join("truncated.X3F");
        assert!(container.extract_frame(0, &out).is_err());
        assert!(!out.exists());
    }

    #[test]
    fn malformed_outer_and_inner_directories_fail() {
        let scratch = Scratch::new();
        let good = container();
        let directory = good.len() - 40;
        for (offset, value) in [
            (0, 0),
            (directory, 0),
            (directory + 8, u32::MAX),
            (directory + 8, 0),
            (directory + 12, 0),
            (directory + 12, u32::MAX),
            (directory + 16, u32::MAX),
            (directory + 16, 4),
            (good.len() - 4, 0),
            (good.len() - 4, u32::MAX),
            (HEADER_SIZE as usize, 0),
            // The inner section must not escape into the next frame.
            (HEADER_SIZE as usize + 64, 1000),
            (HEADER_SIZE as usize + 72, 0),
        ] {
            let mut bad = good.clone();
            put(&mut bad, offset, value);
            assert!(
                X3iContainer::open(scratch.write(&bad)).is_err(),
                "offset {offset}"
            );
        }
        let mut missing = good.clone();
        put(&mut missing, directory + 20, 0);
        put(&mut missing, directory + 32, 0);
        assert!(X3iContainer::open(scratch.write(&missing)).is_err());
        for length in [0, 4, HEADER_SIZE as usize, good.len() - 4, good.len() - 1] {
            assert!(X3iContainer::open(scratch.write(&good[..length])).is_err());
        }
    }

    #[test]
    fn exposure_tables_produce_the_expected_stops() {
        for table in [
            [7813, 977, 1953, 3906, 15625, 31250, 62500],
            [324210, 40530, 81050, 162100, 648420, 1296840, 2593680],
        ] {
            for (us, ev) in table
                .into_iter()
                .zip([0.0, -3.0, -2.0, -1.0, 1.0, 2.0, 3.0])
            {
                assert!((frame_ev(table[0], us) - ev).abs() < 0.001);
            }
        }
    }

    #[test]
    fn conversion_errors_preserve_destinations_and_remove_reserved_output() {
        let scratch = Scratch::new();
        let path = scratch.write(&container());
        let out = scratch.0.join("output.dng");
        let cancel = AtomicBool::new(false);
        let result = crate::convert_file(
            &path,
            &out,
            crate::OutputFormat::Dng,
            &crate::ProcessOptions::default(),
            &cancel,
            |_| {},
        );
        assert!(matches!(result, Err(Error::InvalidData(_))));
        assert!(!out.exists());
        std::fs::write(&out, b"keep").unwrap();
        assert!(crate::convert_file(
            &path,
            &out,
            crate::OutputFormat::Dng,
            &crate::ProcessOptions::default(),
            &cancel,
            |_| {}
        )
        .is_err());
        assert_eq!(std::fs::read(&out).unwrap(), b"keep");
        std::fs::remove_file(&out).unwrap();
        cancel.store(true, Ordering::Relaxed);
        assert!(matches!(
            crate::convert_file(
                &path,
                &out,
                crate::OutputFormat::Dng,
                &crate::ProcessOptions::default(),
                &cancel,
                |_| {}
            ),
            Err(Error::Cancelled)
        ));
        assert!(!out.exists());
        cancel.store(false, Ordering::Relaxed);
        assert!(matches!(
            crate::convert_file(
                &path,
                &out,
                crate::OutputFormat::Dng,
                &crate::ProcessOptions::default(),
                &cancel,
                |_| cancel.store(true, Ordering::Relaxed),
            ),
            Err(Error::Cancelled)
        ));
        assert!(!out.exists());
    }

    #[test]
    fn corpus_exposures_use_capture_shutter() {
        let corpus = std::env::var_os("X3F_TEST_FILES")
            .map(PathBuf::from)
            .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("../../x3f_test_files"));
        for (name, expected) in [
            ("DP0Q0002.X3I", [7813, 977, 1953, 3906, 15625, 31250, 62500]),
            (
                "_DQH0978.X3I",
                [324210, 40530, 81050, 162100, 648420, 1296840, 2593680],
            ),
        ] {
            let path = corpus.join(name);
            if !path.is_file() {
                eprintln!("skip: {name} is absent");
                continue;
            }
            let container = X3iContainer::open(path).unwrap();
            assert_eq!(container.frames().len(), expected.len());
            for (index, expected) in expected.into_iter().enumerate() {
                let mut reader = container.open_frame(index).unwrap();
                reader.load_camf().unwrap();
                eprintln!(
                    "{name} frame {index}: CaptureExpTime={:?}, CaptureShutter={:?}",
                    reader.dng_camf_float("CaptureExpTime"),
                    reader.dng_camf_float("CaptureShutter")
                );
                assert_eq!(container.frame_exposure_us(index).unwrap(), expected);
            }
        }
    }
}
