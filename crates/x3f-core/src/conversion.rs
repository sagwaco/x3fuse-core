//! One synchronous conversion with caller-owned cancellation and stage reporting.

use std::fs::{File, OpenOptions};
use std::io::{self, BufWriter, Write};
use std::path::Path;
use std::sync::atomic::AtomicBool;

use x3f_sys::{self as sys, Control};

use crate::{Error, ProcessOptions, Reader};

/// File formats supported by the desktop conversion API.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFormat {
    /// Linear raw Digital Negative.
    Dng,
    /// Processed 16-bit TIFF.
    Tiff,
    /// The camera's embedded JPEG, without re-encoding.
    Jpeg,
}

/// Coarse pipeline stages; these are not percentage estimates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConversionStage {
    /// Open and validate the X3F container.
    Read,
    /// Load metadata and decode the required image sections.
    Decode,
    /// Process sensor samples with the requested options.
    Process,
    /// Encode and write the output file.
    Write,
}

/// Nonfatal issues associated with this conversion alone.
#[derive(Debug, Default)]
pub struct ConversionReport {
    /// Human-readable warnings, for example a missing matching DNG opcode.
    pub warnings: Vec<String>,
}

/// Convert one image into exactly `output`, which must not already exist.
///
/// The caller chooses a temporary output and publishes it after any metadata
/// postprocessing. This function never renames files or changes global settings.
/// On error/cancellation it removes the temporary file it created; existing
/// destinations are never overwritten. Stage callbacks run on the calling thread.
pub fn convert_file(
    input: &Path,
    output: &Path,
    format: OutputFormat,
    options: &ProcessOptions,
    cancel: &AtomicBool,
    mut on_stage: impl FnMut(ConversionStage),
) -> Result<ConversionReport, Error> {
    let control = Control::new(cancel);
    control.check()?;
    // Reserve ownership before any writer can truncate a caller's file.
    let reserved = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)
        .map_err(|source| output_error(output, source))?;
    drop(reserved);

    let result = (|| {
        on_stage(ConversionStage::Read);
        control.check()?;
        let reader = Reader::open_with_control(input, control)?;
        on_stage(ConversionStage::Decode);
        control.check()?;
        let mut report = ConversionReport::default();

        // SAFETY: the reader uniquely owns its parsed section directory for
        // this entire call. load_data checks ranges and frees failed allocations.
        unsafe {
            let x3f = reader.x3f.as_ptr();
            let jpeg = sys::x3f_get_thumb_jpeg(x3f);
            if !jpeg.is_null() {
                sys::load_data(x3f, jpeg, control)?;
            } else if format == OutputFormat::Jpeg {
                return Err(Error::InvalidData("embedded JPEG is missing".into()));
            }
            if format != OutputFormat::Jpeg {
                let property = sys::x3f_get_prop(x3f);
                if !property.is_null() {
                    sys::load_data(x3f, property, control)?;
                }
                sys::load_data(x3f, sys::x3f_get_camf(x3f), control)?;
                sys::load_data(x3f, sys::x3f_get_raw(x3f), control)?;
            }
        }
        on_stage(ConversionStage::Process);
        control.check()?;
        match format {
            OutputFormat::Dng => crate::output::dng::write_controlled(
                &reader,
                output,
                options,
                control,
                &mut || on_stage(ConversionStage::Write),
                &mut report.warnings,
            )?,
            OutputFormat::Tiff => {
                let image = reader.get_image_with_control(options, control)?;
                let profile = options
                    .cineon
                    .then(|| crate::icc::cineon_log_profile(options.color_encoding))
                    .flatten();
                on_stage(ConversionStage::Write);
                control.check()?;
                crate::output::tiff::write_controlled(
                    &image,
                    output,
                    options.compress,
                    profile.as_deref(),
                    control,
                )
                .map_err(|error| output_error(output, error))?;
            }
            OutputFormat::Jpeg => {
                on_stage(ConversionStage::Write);
                control.check()?;
                // SAFETY: the section was loaded above and belongs to reader.
                let bytes = unsafe {
                    let de = sys::x3f_get_thumb_jpeg(reader.x3f.as_ptr());
                    let image = &(*de).header.data_subsection.image_data;
                    if image.data.is_null() || image.data_size == 0 {
                        return Err(Error::InvalidData("embedded JPEG is empty".into()));
                    }
                    std::slice::from_raw_parts(image.data.cast::<u8>(), image.data_size as usize)
                };
                let mut writer = BufWriter::new(
                    File::create(output).map_err(|error| output_error(output, error))?,
                );
                for chunk in bytes.chunks(64 * 1024) {
                    control.check()?;
                    writer
                        .write_all(chunk)
                        .map_err(|error| output_error(output, error))?;
                }
                writer
                    .flush()
                    .map_err(|error| output_error(output, error))?;
            }
        }
        control.check()?;
        Ok(report)
    })();

    if result.is_err() {
        let _ = std::fs::remove_file(output);
    }
    // Writers propagate cancellation through their I/O interfaces. Preserve
    // the public Cancelled variant for every stage.
    if result.is_err() && control.check().is_err() {
        return Err(Error::Cancelled);
    }
    result
}

pub(crate) fn check_io(control: Control<'_>) -> io::Result<()> {
    control
        .check()
        // Interrupted is retried by Write::write_all and would never terminate.
        .map_err(io::Error::other)
}

fn output_error(path: &Path, source: io::Error) -> Error {
    Error::Io {
        path: path.display().to_string(),
        source,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    /// Minimal valid container with one JPEG section, independent of the
    /// nonredistributable camera corpus. Payload copying must not append the
    /// section header length or bytes from the directory that follows it.
    fn jpeg_container(jpeg: &[u8]) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"FOVb");
        bytes.extend_from_slice(&0x0002_0000_u32.to_le_bytes());
        bytes.extend_from_slice(&[0; 32]);
        bytes.extend_from_slice(b"SECi");
        for value in [0x0002_0000_u32, 2, 18, 1, 1, 0] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes.extend_from_slice(jpeg);
        let directory = bytes.len() as u32;
        bytes.extend_from_slice(b"SECd");
        for value in [0x0002_0000_u32, 1, 40, 28 + jpeg.len() as u32, 0] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes.extend_from_slice(&directory.to_le_bytes());
        bytes
    }

    #[test]
    fn conversion_stages_cancellation_unicode_paths_and_owned_outputs() {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "x3f-convert-{}-{}-한글 space",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let input = dir.join("사진.X3F");
        let output = dir.join("출력.jpg");
        let jpeg = [0xff, 0xd8, 0xff, 0xd9];
        std::fs::write(&input, jpeg_container(&jpeg)).unwrap();
        let cancel = AtomicBool::new(false);
        let mut stages = Vec::new();
        convert_file(
            &input,
            &output,
            OutputFormat::Jpeg,
            &ProcessOptions::default(),
            &cancel,
            |stage| stages.push(stage),
        )
        .unwrap();
        assert_eq!(
            stages,
            [
                ConversionStage::Read,
                ConversionStage::Decode,
                ConversionStage::Process,
                ConversionStage::Write
            ]
        );
        assert_eq!(std::fs::read(&output).unwrap(), jpeg);

        // Existing files belong to the caller, including the input itself.
        assert!(convert_file(
            &input,
            &output,
            OutputFormat::Jpeg,
            &ProcessOptions::default(),
            &cancel,
            |_| {}
        )
        .is_err());
        assert_eq!(std::fs::read(&output).unwrap(), jpeg);
        assert!(convert_file(
            &input,
            &input,
            OutputFormat::Jpeg,
            &ProcessOptions::default(),
            &cancel,
            |_| {}
        )
        .is_err());
        std::fs::remove_file(&output).unwrap();

        for cancel_at in [
            ConversionStage::Read,
            ConversionStage::Decode,
            ConversionStage::Process,
            ConversionStage::Write,
        ] {
            cancel.store(false, Ordering::Relaxed);
            let result = convert_file(
                &input,
                &output,
                OutputFormat::Jpeg,
                &ProcessOptions::default(),
                &cancel,
                |stage| {
                    if stage == cancel_at {
                        cancel.store(true, Ordering::Relaxed);
                    }
                },
            );
            assert!(
                matches!(result, Err(Error::Cancelled)),
                "stage {cancel_at:?}: {result:?}"
            );
            assert!(!output.exists());
        }
        cancel.store(false, Ordering::Relaxed);
        std::fs::write(&input, b"corrupt").unwrap();
        assert!(matches!(
            convert_file(
                &input,
                &output,
                OutputFormat::Dng,
                &ProcessOptions::default(),
                &cancel,
                |_| {}
            ),
            Err(Error::InvalidData(_))
        ));
        assert!(!output.exists());
        std::fs::remove_dir_all(dir).unwrap();
    }
}
