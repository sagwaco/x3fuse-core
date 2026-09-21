//! Pure-Rust TIFF writer (16-bit RGB or grayscale, optional ZIP compression).
//!
//! Replaces the libtiff-based `x3f_output_tiff.c`. We match the legacy tag
//! choices exactly so consumers (Lightroom, Capture One, the in-tree tier-3
//! decoder) keep behaving the same:
//!
//! - 3×16-bit RGB (or 1×16-bit MINISBLACK if the processed image is single-
//!   channel — used for `-qtop`/`-unprocessed` Quattro top-layer dumps).
//! - 32 rows per strip, contiguous planar config, top-left orientation.
//! - 72 DPI in both axes (RESUNIT_INCH).
//! - When `compress=true`, deflate with horizontal predictor=2.
//!
//! The on-disk byte layout differs from libtiff's (different IFD ordering,
//! different strip byte counts under deflate), so tier-2 MD5 stability is
//! intentionally *not* asserted for TIFF — tier-3 perceptual diffs decode
//! both files via the `tiff` crate and compare pixels, which is what callers
//! actually care about.

use std::fs::File;
use std::io::{self, BufWriter, Seek, SeekFrom, Write};
use std::path::Path;
use std::sync::atomic::AtomicBool;

use tiff::encoder::colortype::{Gray16, RGB16};
use tiff::encoder::compression::DeflateLevel;
use tiff::encoder::{Compression, TiffEncoder};
use tiff::tags::{Predictor, ResolutionUnit, Tag};

use crate::Image;

/// Numeric value of the standard `ICCProfile` tag (TIFF tech-note TN1).
/// `tiff::tags::Tag` doesn't expose it as an enum constant, so we use the
/// raw u16 the spec assigns.
const ICC_PROFILE_TAG: u16 = 34675;

/// Write `image` as a 16-bit TIFF. Channels must be 1 (grayscale) or 3 (RGB).
/// `icc_profile`, when supplied, is embedded as the standard `ICCProfile`
/// tag (34675) so colour-managed readers interpret the pixels in the
/// correct space — the flat-TIFF writer uses this to tag linear-light
/// output with linear sRGB / Adobe RGB / ProPhoto RGB primaries.
pub fn write(
    image: &Image,
    out: impl AsRef<Path>,
    compress: bool,
    icc_profile: Option<&[u8]>,
) -> io::Result<()> {
    write_controlled(image, out, compress, icc_profile, x3f_sys::Control::none())
}

pub(crate) fn write_controlled(
    image: &Image,
    out: impl AsRef<Path>,
    compress: bool,
    icc_profile: Option<&[u8]>,
    control: x3f_sys::Control<'_>,
) -> io::Result<()> {
    write_samples(
        ImageView {
            data: &image.data,
            rows: image.rows,
            columns: image.columns,
            channels: image.channels,
            row_stride: image.row_stride,
        },
        out.as_ref(),
        compress,
        icc_profile,
        control,
    )
}

/// Write an existing image with caller-owned cancellation.
/// The caller owns the output path and handles temporary-file publication.
pub fn write_cancellable(
    image: &Image,
    out: impl AsRef<Path>,
    compress: bool,
    icc_profile: Option<&[u8]>,
    cancel: &AtomicBool,
) -> io::Result<()> {
    write_controlled(
        image,
        out.as_ref(),
        compress,
        icc_profile,
        x3f_sys::Control::new(cancel),
    )
}

/// Write packed transfer-encoded RGB16 without copying the input pixels.
/// No gamma, matrix, orientation or other image adjustment is applied here.
/// The caller owns the output path and handles temporary-file publication.
pub fn write_rgb16(
    data: &[u16],
    width: u32,
    height: u32,
    out: impl AsRef<Path>,
    compress: bool,
    icc_profile: Option<&[u8]>,
    cancel: &AtomicBool,
) -> io::Result<()> {
    let stride = width
        .checked_mul(3)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "RGB width overflow"))?;
    write_samples(
        ImageView {
            data,
            rows: height,
            columns: width,
            channels: 3,
            row_stride: stride,
        },
        out.as_ref(),
        compress,
        icc_profile,
        x3f_sys::Control::new(cancel),
    )
}

#[derive(Clone, Copy)]
struct ImageView<'a> {
    data: &'a [u16],
    rows: u32,
    columns: u32,
    channels: u32,
    row_stride: u32,
}

fn write_samples(
    image: ImageView<'_>,
    out: &Path,
    compress: bool,
    icc_profile: Option<&[u8]>,
    control: x3f_sys::Control<'_>,
) -> io::Result<()> {
    crate::conversion::check_io(control)?;
    let samples = (image.columns as usize).checked_mul(image.channels as usize);
    let length = (image.row_stride as usize).checked_mul(image.rows as usize);
    if image.rows == 0
        || image.columns == 0
        || !matches!(image.channels, 1 | 3)
        || samples.is_none_or(|n| n > image.row_stride as usize)
        || length.is_none_or(|n| n != image.data.len())
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "invalid TIFF image dimensions or samples",
        ));
    }
    let mut f = ControlledWriter {
        inner: BufWriter::new(File::create(out)?),
        control,
    };
    let mut enc = TiffEncoder::new(&mut f).map_err(to_io)?;
    if compress {
        enc = enc
            .with_predictor(Predictor::Horizontal)
            .with_compression(Compression::Deflate(DeflateLevel::default()));
    }

    let result = match image.channels {
        3 => write_image::<RGB16>(&mut enc, image, 3, icc_profile, control),
        1 => write_image::<Gray16>(&mut enc, image, 1, icc_profile, control),
        n => Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("TIFF writer needs 1 or 3 channels, got {n}"),
        )),
    };
    result?;
    f.flush()
}

/// The encoder calls this writer for each 32-row strip. Keep cancellation above
/// buffering so even small strips are checked before they reach the filesystem.
struct ControlledWriter<'a, W> {
    inner: W,
    control: x3f_sys::Control<'a>,
}

impl<W: Write> Write for ControlledWriter<'_, W> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        crate::conversion::check_io(self.control)?;
        self.inner.write(bytes)
    }

    fn flush(&mut self) -> io::Result<()> {
        crate::conversion::check_io(self.control)?;
        self.inner.flush()
    }
}

impl<W: Seek> Seek for ControlledWriter<'_, W> {
    fn seek(&mut self, position: SeekFrom) -> io::Result<u64> {
        crate::conversion::check_io(self.control)?;
        self.inner.seek(position)
    }
}

fn write_image<C>(
    enc: &mut TiffEncoder<impl Write + Seek>,
    image: ImageView<'_>,
    channels: u32,
    icc_profile: Option<&[u8]>,
    control: x3f_sys::Control<'_>,
) -> io::Result<()>
where
    C: tiff::encoder::colortype::ColorType<Inner = u16>,
    [u16]: tiff::encoder::TiffValue,
{
    let mut img = enc
        .new_image::<C>(image.columns, image.rows)
        .map_err(to_io)?;
    img.rows_per_strip(32).map_err(to_io)?;
    img.encoder()
        .write_tag(Tag::ResolutionUnit, ResolutionUnit::Inch.to_u16())
        .map_err(to_io)?;
    for tag in [Tag::XResolution, Tag::YResolution] {
        img.encoder()
            .write_tag(tag, tiff::encoder::Rational { n: 72, d: 1 })
            .map_err(to_io)?;
    }

    if let Some(profile) = icc_profile {
        // The TIFF spec stores the ICC profile as type UNDEFINED (= byte
        // array). The `tiff` crate's `encoder_image::encoder` exposes the
        // raw `Tag::Unknown(...)` route via `encoder().write_tag()`, which
        // accepts any byte slice that implements `TiffValue`.
        img.encoder()
            .write_tag(Tag::Unknown(ICC_PROFILE_TAG), profile)
            .map_err(to_io)?;
    }

    let pixel_samples = (image.columns as usize) * (channels as usize);
    let stride = image.row_stride as usize;

    // write_data enables the crate's compression before calling write_strip.
    // Calling write_strip directly would emit uncompressed bytes with ZIP tags.
    if stride == pixel_samples {
        crate::conversion::check_io(control)?;
        img.write_data(image.data).map_err(to_io)?;
    } else {
        let mut packed = Vec::with_capacity(image.rows as usize * pixel_samples);
        for row in 0..image.rows as usize {
            crate::conversion::check_io(control)?;
            let offset = row * stride;
            packed.extend_from_slice(&image.data[offset..offset + pixel_samples]);
        }
        img.write_data(&packed).map_err(to_io)?;
    }
    crate::conversion::check_io(control)?;
    Ok(())
}

fn to_io(e: tiff::TiffError) -> io::Error {
    match e {
        tiff::TiffError::IoError(e) => e,
        other => io::Error::other(other),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tiff::decoder::{Decoder, DecodingResult};

    fn fake_image(cols: u32, rows: u32, chans: u32, stride_pad: u32) -> Image {
        let row_stride = cols * chans + stride_pad;
        let mut data = vec![0u16; (rows * row_stride) as usize];
        for r in 0..rows {
            for c in 0..cols {
                let off = (r * row_stride + c * chans) as usize;
                for k in 0..chans {
                    data[off + k as usize] = ((r * 7 + c * 13) as u16).wrapping_add(k as u16 * 100);
                }
            }
        }
        Image {
            data,
            rows,
            columns: cols,
            channels: chans,
            row_stride,
            levels: crate::ImageLevels::default(),
            dng_highlight_scale: 1.0,
            dng_shoulder_ceiling: 1.0,
        }
    }

    fn round_trip(img: &Image, compress: bool) -> Vec<u16> {
        let dir = std::env::temp_dir().join(format!(
            "x3f-tiff-test-{}-{}",
            std::process::id(),
            counter()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("a.tif");
        write(img, &path, compress, None).unwrap();
        let f = File::open(&path).unwrap();
        let mut dec = Decoder::new(f).unwrap();
        match dec.read_image().unwrap() {
            DecodingResult::U16(v) => v,
            other => panic!("unexpected decode result: {other:?}"),
        }
    }

    #[test]
    fn rgb16_packed_roundtrips() {
        let img = fake_image(4, 3, 3, 0);
        let decoded = round_trip(&img, false);
        assert_eq!(decoded, img.data);
    }

    #[test]
    fn editor_rgb16_preserves_samples_and_profile_and_checks_before_truncating() {
        let path = std::env::temp_dir().join(format!(
            "x3f-editor-tiff-{}-{}.tif",
            std::process::id(),
            counter()
        ));
        let cancel = AtomicBool::new(false);
        let pixels = [0, 32768, 65535, 60000, 40, 500];
        let profile = crate::srgb_icc_profile();
        write_rgb16(&pixels, 2, 1, &path, true, Some(&profile), &cancel).unwrap();
        let original = std::fs::read(&path).unwrap();
        let mut decoder = Decoder::new(File::open(&path).unwrap()).unwrap();
        assert_eq!(decoder.dimensions().unwrap(), (2, 1));
        assert_eq!(
            decoder
                .get_tag_u8_vec(Tag::Unknown(ICC_PROFILE_TAG))
                .unwrap(),
            profile
        );
        match decoder.read_image().unwrap() {
            DecodingResult::U16(data) => assert_eq!(data, pixels),
            _ => panic!("expected RGB16"),
        }
        assert!(write_rgb16(&pixels, 3, 1, &path, false, None, &cancel).is_err());
        assert_eq!(std::fs::read(&path).unwrap(), original);
        assert!(write_rgb16(&pixels, 2, 1, &path, false, None, &AtomicBool::new(true)).is_err());
        assert_eq!(std::fs::read(&path).unwrap(), original);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn rgb16_with_stride_padding_strips_padding() {
        // pad each row with 5 extra u16 elements that must NOT appear in the
        // decoded TIFF.
        let img = fake_image(2, 5, 3, 5);
        let decoded = round_trip(&img, false);
        assert_eq!(decoded.len(), 2 * 5 * 3);
        // pixel(2,1) channel 2 → row_stride*2 + 1*3 + 2 in the source layout
        let expected_g = ((2u32 * 7 + 13) as u16).wrapping_add(100);
        assert_eq!(decoded[(2 * 2 + 1) * 3 + 1], expected_g);
    }

    #[test]
    fn rgb16_compressed_roundtrips_losslessly() {
        let img = fake_image(8, 70, 3, 0);
        let decoded = round_trip(&img, true);
        assert_eq!(decoded, img.data);
    }

    #[test]
    fn cancellation_stops_between_tiff_strips() {
        use std::io::Cursor;
        use std::sync::atomic::{AtomicBool, Ordering};

        struct CancelOnStrip<'a> {
            bytes: Cursor<Vec<u8>>,
            cancel: &'a AtomicBool,
            strips: usize,
        }
        impl Write for CancelOnStrip<'_> {
            fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
                // Headers/tags are small; the first raster write cancels the
                // conversion before the next strip, even inside write_all.
                if bytes.len() > 1000 {
                    self.strips += 1;
                    self.cancel.store(true, Ordering::Relaxed);
                }
                self.bytes.write(bytes)
            }
            fn flush(&mut self) -> io::Result<()> {
                Ok(())
            }
        }
        impl Seek for CancelOnStrip<'_> {
            fn seek(&mut self, position: SeekFrom) -> io::Result<u64> {
                self.bytes.seek(position)
            }
        }

        for compressed in [false, true] {
            let cancel = AtomicBool::new(false);
            let control = x3f_sys::Control::new(&cancel);
            let mut writer = ControlledWriter {
                inner: CancelOnStrip {
                    bytes: Cursor::new(Vec::new()),
                    cancel: &cancel,
                    strips: 0,
                },
                control,
            };
            let mut enc = TiffEncoder::new(&mut writer).unwrap();
            if compressed {
                enc = enc.with_compression(Compression::Deflate(DeflateLevel::default()));
            }
            let image = fake_image(128, 100, 3, 0);
            let result = write_image::<RGB16>(
                &mut enc,
                ImageView {
                    data: &image.data,
                    rows: image.rows,
                    columns: image.columns,
                    channels: image.channels,
                    row_stride: image.row_stride,
                },
                3,
                None,
                control,
            );
            assert!(result.is_err());
            assert_eq!(writer.inner.strips, 1);
        }
    }

    #[test]
    fn gray16_roundtrips() {
        let img = fake_image(4, 3, 1, 0);
        let decoded = round_trip(&img, false);
        assert_eq!(decoded, img.data);
    }

    #[test]
    fn unsupported_channel_count_errors() {
        let mut img = fake_image(1, 1, 3, 0);
        img.channels = 2;
        let dir = std::env::temp_dir().join(format!("x3f-tiff-bad-{}", counter()));
        std::fs::create_dir_all(&dir).unwrap();
        let err = write(&img, dir.join("a.tif"), false, None).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
    }

    fn counter() -> u64 {
        use std::sync::atomic::{AtomicU64, Ordering};
        static N: AtomicU64 = AtomicU64::new(0);
        N.fetch_add(1, Ordering::Relaxed)
    }
}
