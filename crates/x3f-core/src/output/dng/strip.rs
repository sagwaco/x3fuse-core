//! Strip encoding for the raw image plane: uncompressed or lossless JPEG
//! (DNG `Compression = 7`).
//!
//! - Uncompressed: one strip per `ROWS_PER_STRIP` rows; pixels written as
//!   little-endian u16 in interleaved BGR-ish (whatever channel order the
//!   processed buffer holds — the writer is colour-agnostic).
//! - Compressed: same strip layout, each strip a standalone lossless-JPEG
//!   stream (see [`super::ljpeg`]). The legacy C writer used Adobe Deflate
//!   here, but the DNG spec only allows Deflate for floating-point/32-bit
//!   data and Apple's RAW engine enforces that — it refused the whole
//!   file, so `-compress` DNGs had no Finder/Quick Look previews on macOS.

use std::io;

use super::ljpeg;
use crate::Image;

/// One encoded strip ready to be written to file.
pub(crate) struct EncodedStrip {
    pub bytes: Vec<u8>,
}

/// Sub-rectangle of `image` to encode. `(top_row, left_col, rows, cols)`
/// in pixel units; `top_row + rows <= image.rows`, similarly for columns.
/// Used to crop the masked-pixel border off the raw IFD before writing —
/// see [`super::write`] for why this matters (RawTherapee crash).
pub(crate) type CropWindow = (u32, u32, u32, u32);

/// Encode `image` into one or more strips of `rows_per_strip` rows.
/// `compress=true` encodes each strip as lossless JPEG.
/// `crop` selects a sub-rectangle to emit; `None` writes the full frame.
///
/// M7d — strips are independent (each is a standalone byte stream with no
/// cross-strip state), so the per-strip work runs on rayon. Output order
/// is preserved by `into_par_iter()` over an indexed range +
/// `collect::<Result<Vec<_>>>`.
pub(crate) fn encode_strips(
    image: &Image,
    rows_per_strip: u32,
    compress: bool,
    crop: Option<CropWindow>,
) -> io::Result<Vec<EncodedStrip>> {
    use rayon::prelude::*;

    let (top, left, out_rows, out_cols) = crop.unwrap_or((0, 0, image.rows, image.columns));
    let cpp = image.channels as usize;
    let samples_per_row = (out_cols as usize) * cpp;
    let stride = image.row_stride as usize;
    let left_off = (left as usize) * cpp;
    let num_full_strips = out_rows / rows_per_strip;
    let remainder = out_rows % rows_per_strip;
    let total_strips = num_full_strips + u32::from(remainder > 0);

    (0..total_strips)
        .into_par_iter()
        .map(|strip_idx| {
            let start_row = (strip_idx * rows_per_strip) as usize;
            let end_row = ((strip_idx + 1) * rows_per_strip).min(out_rows) as usize;
            let n_rows = end_row - start_row;

            let mut packed: Vec<u16> = Vec::with_capacity(n_rows * samples_per_row);
            for r in start_row..end_row {
                let off = (top as usize + r) * stride + left_off;
                packed.extend_from_slice(&image.data[off..off + samples_per_row]);
            }

            let bytes = if compress {
                ljpeg::encode(&packed, out_cols as usize, n_rows, cpp)
            } else {
                let mut payload: Vec<u8> = Vec::with_capacity(packed.len() * 2);
                for &v in &packed {
                    payload.extend_from_slice(&v.to_le_bytes());
                }
                payload
            };
            Ok(EncodedStrip { bytes })
        })
        .collect()
}

/// Encode a single 8-bit preview as one uncompressed strip (the legacy DNG
/// writer used a single strip for the preview thumbnail).
pub(crate) fn encode_preview_strip(preview: &crate::Preview) -> Vec<u8> {
    let samples_per_row = (preview.columns as usize) * (preview.channels as usize);
    let stride = preview.row_stride as usize;
    if stride == samples_per_row {
        return preview.data.clone();
    }
    let mut out = Vec::with_capacity(samples_per_row * preview.rows as usize);
    for r in 0..preview.rows as usize {
        let off = r * stride;
        out.extend_from_slice(&preview.data[off..off + samples_per_row]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ImageLevels;

    fn fake_image(cols: u32, rows: u32, chans: u32) -> Image {
        let row_stride = cols * chans;
        let mut data = vec![0u16; (rows * row_stride) as usize];
        for r in 0..rows {
            for c in 0..cols {
                let off = (r * row_stride + c * chans) as usize;
                for k in 0..chans {
                    data[off + k as usize] = (r * 31 + c * 17 + k * 5) as u16;
                }
            }
        }
        Image {
            data,
            rows,
            columns: cols,
            channels: chans,
            row_stride,
            levels: ImageLevels::default(),
            dng_highlight_scale: 1.0,
        }
    }

    #[test]
    fn uncompressed_strip_count_matches_rows_per_strip() {
        let img = fake_image(4, 100, 3);
        let strips = encode_strips(&img, 32, false, None).unwrap();
        assert_eq!(strips.len(), 4); // 32 + 32 + 32 + 4
                                     // First three strips: 32 rows × 4 cols × 3 chan × 2 bytes = 768
        for s in &strips[..3] {
            assert_eq!(s.bytes.len(), 32 * 4 * 3 * 2);
        }
        assert_eq!(strips[3].bytes.len(), 4 * 4 * 3 * 2);
    }

    #[test]
    fn cropped_strips_emit_subregion_only() {
        // 8×8 image, channels=3. Crop to rows 2..6, cols 1..5 (4 rows × 4 cols).
        let img = fake_image(8, 8, 3);
        let strips = encode_strips(&img, 32, false, Some((2, 1, 4, 4))).unwrap();
        assert_eq!(strips.len(), 1);
        assert_eq!(strips[0].bytes.len(), 4 * 4 * 3 * 2);
        // First emitted pixel is image (row=2, col=1). Channel-0 value
        // from fake_image: r*31 + c*17 = 2*31 + 1*17 = 79.
        let first = u16::from_le_bytes([strips[0].bytes[0], strips[0].bytes[1]]);
        assert_eq!(first, 79);
    }

    #[test]
    fn compressed_strips_are_standalone_jpeg_streams() {
        // 100 rows / 32 per strip → 4 independent SOI..EOI streams, each
        // sized for its own row count (the last strip is short).
        let img = fake_image(8, 100, 3);
        let strips = encode_strips(&img, 32, true, None).unwrap();
        assert_eq!(strips.len(), 4);
        for s in &strips {
            assert_eq!(&s.bytes[..2], &[0xFF, 0xD8], "strip must start with SOI");
            assert_eq!(
                &s.bytes[s.bytes.len() - 2..],
                &[0xFF, 0xD9],
                "strip must end with EOI"
            );
        }
    }

    #[test]
    fn compressed_crop_matches_uncompressed_crop_dimensions() {
        // Cropped compressed strips carry the crop dimensions in SOF3:
        // height 4 (rows), width 4 (cols) for the (2,1,4,4) window.
        let img = fake_image(8, 8, 3);
        let strips = encode_strips(&img, 32, true, Some((2, 1, 4, 4))).unwrap();
        assert_eq!(strips.len(), 1);
        let b = &strips[0].bytes;
        let sof = b
            .windows(2)
            .position(|w| w == [0xFF, 0xC3])
            .expect("SOF3 present");
        // SOF3 layout: marker(2) len(2) precision(1) height(2) width(2).
        let height = u16::from_be_bytes([b[sof + 5], b[sof + 6]]);
        let width = u16::from_be_bytes([b[sof + 7], b[sof + 8]]);
        assert_eq!((height, width), (4, 4));
    }
}
