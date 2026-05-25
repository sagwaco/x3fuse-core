//! Strip encoding for the raw image plane: uncompressed or Deflate +
//! horizontal predictor.
//!
//! Matches the C path:
//! - Uncompressed: one strip per `ROWS_PER_STRIP` rows; pixels written as
//!   little-endian u16 in interleaved BGR-ish (whatever channel order the
//!   processed buffer holds — the writer is colour-agnostic).
//! - Compressed: same strip layout, with a per-strip horizontal predictor
//!   applied per-channel, then zlib-encoded (`COMPRESSION_ADOBE_DEFLATE`).

use std::io::{self, Write};

use flate2::write::ZlibEncoder;
use flate2::Compression;

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
/// `compress=true` enables horizontal-predictor + zlib (`Deflate`).
/// `crop` selects a sub-rectangle to emit; `None` writes the full frame.
///
/// M7d — strips are independent (no cross-strip state in the predictor
/// or the deflate stream; each strip is a standalone zlib payload), so
/// the per-strip work runs on rayon. Output order is preserved by
/// `into_par_iter()` over an indexed range + `collect::<Result<Vec<_>>>`.
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

            if compress {
                for row in packed.chunks_mut(samples_per_row) {
                    apply_horizontal_predictor_u16(row, cpp);
                }
            }

            let mut payload: Vec<u8> = Vec::with_capacity(packed.len() * 2);
            for &v in &packed {
                payload.extend_from_slice(&v.to_le_bytes());
            }

            let bytes = if compress {
                zlib_encode(&payload)?
            } else {
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

fn apply_horizontal_predictor_u16(row: &mut [u16], samples_per_pixel: usize) {
    if row.len() <= samples_per_pixel {
        return;
    }
    // Reverse iteration so we compute (current - previous) using the
    // original previous-pixel value, not an already-differenced one.
    for i in (samples_per_pixel..row.len()).rev() {
        row[i] = row[i].wrapping_sub(row[i - samples_per_pixel]);
    }
}

fn zlib_encode(data: &[u8]) -> io::Result<Vec<u8>> {
    let mut enc = ZlibEncoder::new(Vec::new(), Compression::default());
    enc.write_all(data)?;
    enc.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ImageLevels;
    use flate2::read::ZlibDecoder;
    use std::io::Read;

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
    fn predictor_first_pixel_unchanged_rest_differenced() {
        let mut row = [10u16, 20, 30, 11, 21, 32];
        apply_horizontal_predictor_u16(&mut row, 3);
        assert_eq!(row[0], 10);
        assert_eq!(row[1], 20);
        assert_eq!(row[2], 30);
        // Per-channel diff: 11-10=1, 21-20=1, 32-30=2.
        assert_eq!(row[3], 1);
        assert_eq!(row[4], 1);
        assert_eq!(row[5], 2);
    }

    #[test]
    fn predictor_round_trips_via_cumulative_sum() {
        let original = vec![10u16, 20, 30, 11, 21, 32, 100, 50, 60];
        let mut row = original.clone();
        apply_horizontal_predictor_u16(&mut row, 3);
        // Restore by cumulative sum per channel.
        for i in 3..row.len() {
            row[i] = row[i].wrapping_add(row[i - 3]);
        }
        assert_eq!(row, original);
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
    fn compressed_strips_round_trip_to_original_bytes() {
        let img = fake_image(8, 32, 3);
        let strips = encode_strips(&img, 32, true, None).unwrap();
        assert_eq!(strips.len(), 1);
        let mut decoded = Vec::new();
        ZlibDecoder::new(&strips[0].bytes[..])
            .read_to_end(&mut decoded)
            .unwrap();
        // Decoded payload is predictor-encoded LE u16. Reverse the
        // predictor per row, then convert back to u16 pixels and compare
        // against the source.
        let cpp = 3usize;
        let cols = 8usize;
        let row_bytes = cols * cpp * 2;
        let mut undone = Vec::with_capacity(decoded.len() / 2);
        for row_chunk in decoded.chunks(row_bytes) {
            let mut row: Vec<u16> = row_chunk
                .chunks(2)
                .map(|c| u16::from_le_bytes([c[0], c[1]]))
                .collect();
            for i in cpp..row.len() {
                row[i] = row[i].wrapping_add(row[i - cpp]);
            }
            undone.extend(row);
        }
        assert_eq!(undone, img.data);
    }
}
