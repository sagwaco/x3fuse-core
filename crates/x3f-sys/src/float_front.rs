//! Native float-image ApplyGains kernel for the ordered Merrill float
//! frontend. Verified bit-for-bit against the supplied Merrill library
//! (ARM64 0x39b38..0x3a42c) through the archived differential oracle.
//! Evidence: x3f-docs `spp-rgb-trace/followup-float-preprocess/`.
//!
//! The operation is
//! `out = f32((f64(sample) - black) * (gain * spatial_value))`
//! with no display clamp and no gamma. It works on sensor-count float
//! samples before any normalization. The ordered frontend follow-up
//! wires this kernel into the conversion path; production still uses
//! the normalized uint16 frontend.

/// One channel of the spatial gain table. `values` is row-major with
/// `rows` rows and `columns` knots per row. The CAMF builders in
/// `spatial_gain.rs` produce this layout.
#[derive(Clone, Debug)]
pub struct GainTable {
    pub values: Vec<f32>,
    pub columns: usize,
    pub rows: usize,
}

impl GainTable {
    pub fn new(values: Vec<f32>, columns: usize, rows: usize) -> Self {
        assert_eq!(values.len(), columns * rows, "gain table shape mismatch");
        assert!(
            columns >= 2 && rows >= 2,
            "gain table needs two knots per axis"
        );
        Self {
            values,
            columns,
            rows,
        }
    }
}

/// Apply black offsets, channel gains and spatial tables in one pass.
///
/// `data` holds interleaved RGB float samples with `row_stride` floats
/// per row. `row_stride` may exceed `width * 3`; the padding floats are
/// never read or written. When any entry of `tables` is `None`, the
/// native all-three gate applies and every channel uses `gain` alone.
///
/// The spatial path requires `height >= 2`. The native operation divides
/// by `height - 1` and produces NaN for a single row.
pub fn apply_gains(
    data: &mut [f32],
    width: usize,
    height: usize,
    row_stride: usize,
    black: [f64; 3],
    gain: [f64; 3],
    tables: [Option<&GainTable>; 3],
) {
    assert!(
        row_stride >= width * 3,
        "row stride smaller than the pixel row"
    );
    assert!(
        data.len() >= row_stride * (height - 1) + width * 3,
        "buffer smaller than the image"
    );
    let spatial = tables.iter().all(Option::is_some);
    for y in 0..height {
        // Per-channel interpolated gain row. The vertical coordinate is
        // float32, and each knot is blended with one float32 fma.
        let mut row = [Vec::new(), Vec::new(), Vec::new()];
        let mut edge = [0.0_f64; 3];
        let mut span = [0.0_f64; 3];
        let mut value = [0.0_f64; 3];
        let mut slope = [0.0_f64; 3];
        let mut knot = [0_usize; 3];
        if spatial {
            for c in 0..3 {
                let table = tables[c].expect("spatial path requires all tables");
                // Native 0x39f74: (f32(y) / f32(height-1)) * f32(rows-1).
                let position = (y as f32 / (height - 1) as f32) * (table.rows - 1) as f32;
                let lo = position.floor() as usize;
                let hi = position.ceil() as usize;
                let fraction = position - lo as f32;
                row[c].resize(table.columns, 0.0);
                for x in 0..table.columns {
                    let a = table.values[lo * table.columns + x];
                    let b = table.values[hi * table.columns + x];
                    row[c][x] = fraction.mul_add(b - a, a);
                }
                // Native horizontal span: width / (columns - 1), not
                // width - 1. The slope numerator is a rounded float32
                // difference before the double conversion.
                span[c] = width as f64 / (table.columns - 1) as f64;
                edge[c] = span[c];
                value[c] = row[c][0] as f64;
                slope[c] = (row[c][1] - row[c][0]) as f64 / span[c];
            }
        }
        let pixels = &mut data[y * row_stride..y * row_stride + width * 3];
        for x in 0..width {
            for c in 0..3 {
                if spatial && x as f64 >= edge[c] {
                    knot[c] += 1;
                    value[c] = row[c][knot[c]] as f64;
                    slope[c] = (row[c][knot[c] + 1] - row[c][knot[c]]) as f64 / span[c];
                    edge[c] += span[c];
                }
                let adjusted = if spatial { gain[c] * value[c] } else { gain[c] };
                let index = 3 * x + c;
                pixels[index] = ((pixels[index] as f64 - black[c]) * adjusted) as f32;
                if spatial {
                    value[c] += slope[c];
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Archived native results: (width, height, spatial, fnv1a hash).
    /// Source: followup-float-preprocess/apply-gains-results.txt, an
    /// independent repeat confirmed every hash.
    const NATIVE_CASES: &[(usize, usize, bool, u64)] = &[
        (1, 5, false, 0x138328c54eef5e5f),
        (2, 5, false, 0xe74038c3d75ccf3e),
        (3, 5, false, 0xcce4f8b3ddcaf15b),
        (4, 5, false, 0xaf74bebb544d557c),
        (5, 5, false, 0x5f4802a8b6ca169d),
        (9, 5, false, 0xe98c4c467bd4bc02),
        (9, 13, true, 0x975b9335a4245ddf),
        (37, 5, false, 0x3ea06b0be3260cbb),
        (37, 13, true, 0x7951b2f5927245f1),
        (97, 5, false, 0x595afa5a2e4e4c46),
        (97, 13, true, 0xb63b750844e59ce2),
    ];

    fn fnv1a(values: &[f32]) -> u64 {
        let bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(values.as_ptr() as _, values.len() * 4) };
        let mut hash = 0xcbf29ce484222325_u64;
        for byte in bytes {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x100000001b3);
        }
        hash
    }

    fn synthetic_tables() -> [GainTable; 3] {
        std::array::from_fn(|c| {
            let columns = 3 + 2 * c;
            let rows = 3 + c;
            let values = (0..columns * rows)
                .map(|i| 0.8_f32 + (((i * 7 + c * 3) % 17) as f32) * 0.071_f32)
                .collect();
            GainTable::new(values, columns, rows)
        })
    }

    /// Port of the archived differential oracle. The padding floats
    /// (-12345.25) must stay untouched, and the kernel must reproduce
    /// every archived native hash bit-for-bit.
    #[test]
    fn apply_gains_matches_native_oracle() {
        let black = [128.125_f64, 255.3, -64.75];
        let gain = [2.21234567_f64, 0.98765432, 4.4433322];
        let tables = synthetic_tables();
        let mut cases = 0;
        let mut samples = 0;
        for &(width, height, spatial, expected_hash) in NATIVE_CASES {
            let stride = width * 3 + 7;
            let mut data = vec![-12345.25_f32; stride * height];
            for y in 0..height {
                for x in 0..width {
                    for c in 0..3 {
                        let raw = ((x * 191 + y * 131 + c * 977) % 8192) as i32 - 128;
                        data[y * stride + 3 * x + c] = (raw as f32) * 0.75_f32;
                    }
                }
            }
            // The archived no-spatial cases pass table 0 and table 2
            // while table 1 is absent: the all-three gate must select
            // the global gain alone.
            let refs: [Option<&GainTable>; 3] = [
                Some(&tables[0]),
                if spatial { Some(&tables[1]) } else { None },
                Some(&tables[2]),
            ];
            apply_gains(&mut data, width, height, stride, black, gain, refs);
            for y in 0..height {
                for pad in width * 3..stride {
                    assert_eq!(data[y * stride + pad], -12345.25_f32, "padding changed");
                }
            }
            assert_eq!(
                fnv1a(&data),
                expected_hash,
                "case {width}x{height} spatial={spatial}"
            );
            cases += 1;
            samples += width * height * 3;
        }
        assert_eq!(cases, 11);
        assert_eq!(samples, 7947);
    }

    #[test]
    fn missing_any_table_selects_global_gain_only() {
        let tables = synthetic_tables();
        let mut with_all = vec![100.5_f32; 3 * 2 * 2];
        let mut without_middle = with_all.clone();
        let black = [1.0_f64; 3];
        let gain = [2.0_f64; 3];
        apply_gains(
            &mut with_all,
            2,
            2,
            6,
            black,
            gain,
            [Some(&tables[0]), Some(&tables[1]), Some(&tables[2])],
        );
        apply_gains(
            &mut without_middle,
            2,
            2,
            6,
            black,
            gain,
            [Some(&tables[0]), None, Some(&tables[2])],
        );
        // The all-three gate makes the missing-middle case equal to the
        // pure global gain, different from the spatial result.
        let pure = vec![((100.5_f64 - 1.0) * 2.0) as f32; 12];
        assert_eq!(without_middle, pure);
        assert_ne!(with_all, pure);
    }
}
