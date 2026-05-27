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
use std::io::{self, BufWriter};
use std::path::Path;

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
    let path = out.as_ref();
    let f = BufWriter::new(File::create(path)?);
    let mut enc = TiffEncoder::new(f).map_err(to_io)?;
    if compress {
        enc = enc
            .with_predictor(Predictor::Horizontal)
            .with_compression(Compression::Deflate(DeflateLevel::default()));
    }

    match image.channels {
        3 => write_image::<RGB16>(&mut enc, image, 3, icc_profile),
        1 => write_image::<Gray16>(&mut enc, image, 1, icc_profile),
        n => Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("TIFF writer needs 1 or 3 channels, got {n}"),
        )),
    }
}

fn write_image<C>(
    enc: &mut TiffEncoder<BufWriter<File>>,
    image: &Image,
    channels: u32,
    icc_profile: Option<&[u8]>,
) -> io::Result<()>
where
    C: tiff::encoder::colortype::ColorType<Inner = u16>,
    [u16]: tiff::encoder::TiffValue,
{
    let mut img = enc
        .new_image::<C>(image.columns, image.rows)
        .map_err(to_io)?;
    img.rows_per_strip(32).map_err(to_io)?;
    img.resolution(
        ResolutionUnit::Inch,
        tiff::encoder::Rational { n: 72, d: 1 },
    );

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

    // The tiff crate's write_data handles compression + predictor + striping
    // in one shot, but only on a tightly-packed buffer. When the C-side image
    // has stride padding (rows are wider than columns*channels — common after
    // crop) we materialise a packed copy. The transient ~2× memory bump is
    // acceptable for now; revisit if it shows up in profiling.
    if stride == pixel_samples {
        img.write_data(&image.data).map_err(to_io)?;
    } else {
        let total = pixel_samples * (image.rows as usize);
        let mut packed = Vec::with_capacity(total);
        for r in 0..image.rows as usize {
            let off = r * stride;
            packed.extend_from_slice(&image.data[off..off + pixel_samples]);
        }
        img.write_data(&packed).map_err(to_io)?;
    }
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
        let img = fake_image(8, 8, 3, 0);
        let decoded = round_trip(&img, true);
        assert_eq!(decoded, img.data);
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
