//! `MatchColOffsets` (0x2f284), the `Flt32Mono` + `Flt32RGB` overload: the
//! column-offset correction ("BCS") that `DoMonoBCS` (0x2fc38) runs over one
//! `DoThreadedPix` slice of rows inside `MonochromeDenoise`.
//!
//! The correction runs 16 columns at a time with NEON while 16 remain; the
//! NEON `fmls` and the scalar `fmsub` give a NaN offset opposite signs, and
//! the port reproduces both.
//!
//! `DoMonoBCS` passes the column noise scale (`NoiseParams+0xac`), the ISO
//! (`NoiseParams+0x4`), the mono and RGB pixmaps, the slice `first..=last`
//! (inclusive: `DoThreadedPix` stores `end - 1`), the BCS filter size
//! (`NoiseParams+0xb0`), `false`, an uninitialized register, the
//! `FastBilateral_F20` built by `MonochromeDenoise` (by value) and the scale
//! index `NoiseParams+0xb4`, then `false`. Four of these are never read: the
//! noise scale (forwarded to `DirectionalFastBF` in `s1`, which ignores it),
//! both flags and the uint between them. The port does not take them.
//!
//! `FastBilateral_F20` is 0xa0 bytes. AAPCS64 passes it as a pointer to a
//! caller copy (x7). Only `+0x00` (`float *weights[0x1000]`) and `+0x08`
//! (`int limits[0x1000]`) are read here. `MonochromeDenoise` also stores the
//! BCS noise scale (`+0x10`, double), the noise scaling (`+0x18`), the
//! quadratic coefficients twice (`+0x20..+0x50`) and `CalculateCoeff`
//! outputs, WB gains and mixing weights after them. [`BcsWeights`] holds the
//! two tables.
//!
//! The kernel, per slice:
//! 1. `DirectionalFastBF` (0x35794), mono to RGB: with filter size `f` and
//!    `r = f / 2`, for rows `first..=last` and `i in 0..width - 2r`, the
//!    center `c = mono[r + i]` picks the table row `clamp(trunc(c), 0, 4095)`.
//!    The taps `mono[i..i + 2r]` (2r taps: one fewer right of the center) with
//!    `k = trunc(|c - tap|) <= limit` add `weight[k] * tap` (fused) and
//!    `weight[k]`. The quotient goes to all three channels of RGB column
//!    `r + i`. A non-finite quotient falls back to `mono[r + 2i]` (the native
//!    indexes an already advanced pointer, so this can read the next row).
//!    `f <= 1` copies `mono[r + 2i]` instead. Null tables skip this step.
//! 2. When `last - first >= 40`, blocks of 40 rows (the last one runs to
//!    `last` inclusive): the column means of mono and of RGB channel 0, both
//!    divided by `rows + 1`, give `d = mono_mean - rgb_mean`. With `R = (f -
//!    1) / 2`, every mono pixel `m` of the block in columns `R..width - R`
//!    becomes `m - d` when `|t * m| > |d|`, else stays `m`, then is clamped to
//!    `0..=32766`. For even `f`, `R = r - 1`, so the outer columns compare
//!    against RGB the filter did not write.
//! 3. The threshold `t` is 0.05 up to ISO 100, otherwise interpolated
//!    linearly between ISO points 100, 200, ..., 12800 and the thresholds 0.05,
//!    0.055, 0.07, 0.12, 0.17, 0.25, 0.33. Above 6400 the native reads past the
//!    thresholds into the ISO points (and past those above 12800); the port
//!    keeps the same contiguous table. `t` is then scaled by
//!    `[0.2, 0.3, 0.4, 0.5, 1.0][scale_index]`; indices 5..=8 read NaN, which
//!    disables the correction.
//!
//! Checked bit for bit against the native library through
//! `scripts/spp-native/match_col_offsets_mono.cpp`
//! (fixture `tests/fixtures/spp-mono-bcs-native.txt`).

// Wired into the monochrome pipeline later; only the tests call it for now.
#![cfg_attr(not(test), allow(dead_code))]

use crate::{Error, Result};

/// Rows of the `FastBilateral_F20` weight tables.
pub(crate) const BCS_TABLE_ROWS: usize = 0x1000;

/// `__TEXT,__const` words 0xbbff0..0xbc050, read by contiguous offsets: 7
/// thresholds, 8 ISO points from 0xbc00c, 5 scales from 0xbc02c, then 4 NaN.
const THRESHOLD_TABLE: [u32; 24] = [
    0x3d4c_cccd, // 0.05
    0x3d61_47ae, // 0.055
    0x3d8f_5c29, // 0.07
    0x3df5_c28f, // 0.12
    0x3e2e_147b, // 0.17
    0x3e80_0000, // 0.25
    0x3ea8_f5c3, // 0.33
    0x42c8_0000, // ISO 100
    0x4348_0000, // 200
    0x43c8_0000, // 400
    0x4448_0000, // 800
    0x44c8_0000, // 1600
    0x4548_0000, // 3200
    0x45c8_0000, // 6400
    0x4648_0000, // 12800
    0x3e4c_cccd, // scale 0.2
    0x3e99_999a, // 0.3
    0x3ecc_cccd, // 0.4
    0x3f00_0000, // 0.5
    0x3f80_0000, // 1.0
    0x7fc0_0000,
    0x7fc0_0000,
    0x7fc0_0000,
    0x7fc0_0000,
];
const ISO_POINTS: usize = 7;
const SCALES: usize = 15;

/// Upper clamp of the corrected pixels, 32766.
const PIXEL_MAX: f32 = f32::from_bits(0x46ff_fc00);

/// The two tables of `FastBilateral_F20` read by `DirectionalFastBF`: row `i`
/// holds the range weights for a center of intensity `i`, indexed by the
/// truncated absolute difference up to `limits[i]` inclusive.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct BcsWeights {
    pub rows: Vec<Vec<f32>>,
    pub limits: Vec<i32>,
}

impl BcsWeights {
    /// The tables `MonochromeDenoise` builds at 0x30100..0x302a8 from the BCS
    /// noise scale `NoiseParams+0xac`. With `v = (c0 + c1 i + c2 i^2) / scale`
    /// (`c1 i + c0` and then `(c2 i) i + ...` fused), the limit is
    /// `trunc(3 sqrt(v))` and the weights are `exp(-k^2 / v)` rounded to f32
    /// for `k in 0..=limit`.
    ///
    /// The native skips the build when the scale is 0 and leaves the table
    /// pointers uninitialized; this returns `None` there. Non-finite or
    /// negative scales are rejected.
    pub(crate) fn monochrome_denoise(bcs_scale: f32) -> Result<Option<Self>> {
        if bcs_scale == 0.0 {
            return Ok(None);
        }
        if !bcs_scale.is_finite() || bcs_scale < 0.0 {
            return Err(Error::InvalidData("invalid SPP BCS noise scale"));
        }
        let c0 = f64::from_bits(0x4023_b531_4cd7_1e93);
        let c1 = f64::from_bits(0x3fa5_eea1_5ee6_741a);
        let c2 = f64::from_bits(0x3ef2_288a_0e9f_8189);
        let scale = f64::from(bcs_scale);
        let mut rows = Vec::with_capacity(BCS_TABLE_ROWS);
        let mut limits = Vec::with_capacity(BCS_TABLE_ROWS);
        for i in 0..BCS_TABLE_ROWS {
            let x = f64::from(i as u32);
            let variance = (c2 * x).mul_add(x, c1.mul_add(x, c0)) / scale;
            let limit = (variance.sqrt() * 3.0) as i32;
            limits.push(limit);
            rows.push(
                (0..=limit)
                    .map(|k| (-f64::from(k * k) / variance).exp() as f32)
                    .collect(),
            );
        }
        Ok(Some(Self { rows, limits }))
    }

    fn valid(&self) -> bool {
        self.rows.len() == BCS_TABLE_ROWS
            && self.limits.len() == BCS_TABLE_ROWS
            && self
                .rows
                .iter()
                .zip(&self.limits)
                .all(|(row, &limit)| limit < 0 || row.len() > limit as usize)
    }
}

fn plane_len(size: [usize; 2], stride: usize, channels: usize) -> Option<usize> {
    let [cols, rows] = size;
    let row = cols.checked_mul(channels)?;
    if cols == 0 || rows == 0 || stride < row {
        return None;
    }
    (rows - 1).checked_mul(stride)?.checked_add(row)
}

fn invalid() -> Error {
    Error::InvalidData("invalid SPP mono BCS parameters")
}

/// The native column-offset correction of one row slice, see the module doc.
///
/// `mono` (stride `mono_stride` floats) is corrected in place on rows
/// `first_row..=last_row`; `rgb` (stride `rgb_stride`, three channels) is
/// scratch: `DirectionalFastBF` writes those rows, the correction reads their
/// channel 0. `weights: None` stands for null table pointers.
///
/// Layouts, rows, filter size (`1..=2^24`, radius `(f - 1) / 2` at most the
/// width), scale index (`0..=8`) and tables are checked before any write. The
/// one data-dependent read, the non-finite fallback `mono[r + 2i]`, is checked
/// when it happens: past the end of `mono` the native reads outside the image,
/// and the port returns an error with `mono` untouched and the earlier RGB
/// pixels already written.
#[allow(clippy::too_many_arguments)]
pub(crate) fn match_col_offsets_mono(
    mono: &mut [f32],
    mono_stride: usize,
    rgb: &mut [f32],
    rgb_stride: usize,
    size: [usize; 2],
    first_row: usize,
    last_row: usize,
    iso: f32,
    filter_size: u32,
    weights: Option<&BcsWeights>,
    scale_index: u32,
) -> Result<()> {
    let [cols, rows] = size;
    let match_radius = filter_size.wrapping_sub(1) as usize >> 1;
    if plane_len(size, mono_stride, 1).is_none_or(|needed| mono.len() < needed)
        || plane_len(size, rgb_stride, 3).is_none_or(|needed| rgb.len() < needed)
        || first_row > last_row
        || last_row >= rows
        || cols > i32::MAX as usize
        || !(1..=1 << 24).contains(&filter_size)
        || match_radius > cols
        || scale_index > 8
        || weights.is_some_and(|w| !w.valid())
    {
        return Err(invalid());
    }
    // The strided copy for filter size 1 reads unconditionally.
    if weights.is_some()
        && filter_size == 1
        && (last_row * mono_stride + 2 * (cols - 1)) >= mono.len()
    {
        return Err(invalid());
    }
    if let Some(weights) = weights {
        directional_fast_bf(
            mono,
            mono_stride,
            rgb,
            rgb_stride,
            cols,
            first_row,
            last_row,
            filter_size,
            weights,
        )?;
    }

    let block_count = (last_row - first_row) / 40;
    if block_count == 0 {
        return Ok(());
    }
    let threshold = threshold(iso) * f32::from_bits(THRESHOLD_TABLE[SCALES + scale_index as usize]);
    let mut mono_mean = vec![0.0_f32; cols];
    let mut rgb_mean = vec![0.0_f32; cols];
    for block in 0..block_count {
        let start = first_row + 40 * block;
        let end = if block + 1 == block_count {
            last_row + 1
        } else {
            start + 40
        };
        mono_mean.fill(0.0);
        rgb_mean.fill(0.0);
        for y in start..end {
            let row = &mono[y * mono_stride..][..cols];
            for (sum, &value) in mono_mean.iter_mut().zip(row) {
                *sum = value + *sum;
            }
        }
        let divisor = ((end - start) as u32 + 1) as f32;
        for sum in &mut mono_mean {
            *sum /= divisor;
        }
        for y in start..end {
            let row = &rgb[y * rgb_stride..][..3 * cols];
            for (sum, pixel) in rgb_mean.iter_mut().zip(row.chunks_exact(3)) {
                *sum = pixel[0] + *sum;
            }
        }
        for sum in &mut rgb_mean {
            *sum /= divisor;
        }
        if match_radius >= cols - match_radius {
            continue;
        }
        // The native runs 16 columns at a time with NEON while 16 remain,
        // then one at a time. The two forms only differ in the sign of a NaN
        // offset: `fmls` negates the offset, scalar `fmsub` the 0/1 factor.
        // Optimized builds merge the two algebraically equal arms and lose
        // that sign; `black_box` keeps the vector arm's negated offset.
        let count = cols - 2 * match_radius;
        let vector_end = match_radius + if count >= 16 { count & !15 } else { 0 };
        for y in start..end {
            let row = &mut mono[y * mono_stride..][..cols];
            for x in match_radius..cols - match_radius {
                let value = row[x];
                let offset = mono_mean[x] - rgb_mean[x];
                let use_offset: f32 = if (threshold * value).abs() > offset.abs() {
                    1.0
                } else {
                    0.0
                };
                let corrected = if x < vector_end {
                    core::hint::black_box(-offset).mul_add(use_offset, value)
                } else {
                    (-use_offset).mul_add(offset, value)
                };
                let corrected = if corrected < 0.0 { 0.0 } else { corrected };
                row[x] = if corrected > PIXEL_MAX {
                    PIXEL_MAX
                } else {
                    corrected
                };
            }
        }
    }
    Ok(())
}

/// The ISO threshold of 0x2f4ac..0x2f53c and 0x2f8fc..0x2f960, before the
/// scale index factor.
fn threshold(iso: f32) -> f32 {
    let mut upper = if iso <= 12800.0 { 7 } else { 8 };
    for (point, index) in [
        (6400.0, 6),
        (3200.0, 5),
        (1600.0, 4),
        (800.0, 3),
        (400.0, 2),
    ] {
        if iso <= point {
            upper = index;
        }
    }
    if iso <= 200.0 {
        upper = 1;
    }
    if iso <= 100.0 {
        return f32::from_bits(0x3d4c_cccd);
    }
    let value = |i: usize| f32::from_bits(THRESHOLD_TABLE[i]);
    let point = |i: usize| f32::from_bits(THRESHOLD_TABLE[ISO_POINTS + i]);
    let lower_point = point(upper - 1);
    let upper_point = point(upper);
    let weighted_lower = value(upper - 1) * (upper_point - iso);
    value(upper).mul_add(iso - lower_point, weighted_lower) / (upper_point - lower_point)
}

/// `FPixelProc2_F20::DirectionalFastBF` (0x35794), mono to RGB, on rows
/// `first..=last` with non-null tables.
#[allow(clippy::too_many_arguments)]
fn directional_fast_bf(
    mono: &[f32],
    mono_stride: usize,
    rgb: &mut [f32],
    rgb_stride: usize,
    cols: usize,
    first: usize,
    last: usize,
    filter_size: u32,
    weights: &BcsWeights,
) -> Result<()> {
    let size = filter_size as f32 as i32;
    let radius = (size / 2) as usize;
    let count = cols as i64 - 2 * radius as i64;
    if count < 1 {
        return Ok(());
    }
    let count = count as usize;
    for y in first..=last {
        let base = y * mono_stride;
        let out = &mut rgb[y * rgb_stride + 3 * radius..][..3 * count];
        for (i, pixel) in out.chunks_exact_mut(3).enumerate() {
            // The center pointer has already advanced by i.
            let strided = base + radius + 2 * i;
            let value = if size <= 1 {
                mono[strided]
            } else {
                let center = mono[base + radius + i];
                let index = (center as i32).min(0xfff).max(0) as usize;
                let row = &weights.rows[index];
                let limit = weights.limits[index];
                let mut sum = 0.0_f32;
                let mut weight_sum = 0.0_f32;
                for &tap in &mono[base + i..base + i + 2 * radius] {
                    let k = (center - tap).abs() as i32;
                    if k <= limit {
                        let weight = row[k as usize];
                        sum = tap.mul_add(weight, sum);
                        weight_sum += weight;
                    }
                }
                let value = sum / weight_sum;
                if value.is_finite() {
                    value
                } else {
                    *mono.get(strided).ok_or_else(invalid)?
                }
            };
            pixel.fill(value);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::float_front::fnv1a_f32;

    const NATIVE: &str = include_str!("../../../tests/fixtures/spp-mono-bcs-native.txt");

    fn mix32(mut x: u32) -> u32 {
        x ^= x >> 16;
        x = x.wrapping_mul(0x7feb_352d);
        x ^= x >> 15;
        x = x.wrapping_mul(0x846c_a68b);
        x ^ (x >> 16)
    }

    /// The generator of `match_col_offsets_mono.cpp`.
    fn pixel(seed: u32, x: u32, y: u32, pattern: u32) -> f32 {
        let h = mix32(
            seed.wrapping_mul(0x9e37_79b9)
                ^ x.wrapping_mul(0x10001)
                    .wrapping_add(y.wrapping_mul(0x3c6e_f372)),
        );
        let h2 = mix32(h ^ 0x5bd1_e995);
        if pattern == 1 {
            return ((h % 40500) as i32 - 500) as f32 + (h2 % 1024) as f32 * 0.000_976_562_5;
        }
        if pattern == 2 && (h2 >> 24) % 8 == 0 {
            const SPECIALS: [u32; 10] = [
                0x7f80_0000,
                0xff80_0000,
                0x7fc0_0000,
                0x8000_0000,
                0x7149_f2ca,
                0x0000_0001,
                0x46ff_fe00,
                0xf149_f2ca,
                0x457f_f800,
                0x4580_0000,
            ];
            return f32::from_bits(SPECIALS[((h2 >> 8) % 10) as usize]);
        }
        let level = 600 + ((x.wrapping_mul(37).wrapping_add(y.wrapping_mul(11))) % 1800) as i32;
        let noise = (h % 129) as i32 - 64;
        let stripe = (mix32(seed.wrapping_mul(0x85eb_ca6b) ^ x) % 61) as i32 - 30;
        (level + noise) as f32 + stripe as f32 * 0.75
    }

    fn plane(len: usize, stride: usize, seed: u32, pattern: u32) -> Vec<f32> {
        (0..len)
            .map(|i| pixel(seed, (i % stride) as u32, (i / stride) as u32, pattern))
            .collect()
    }

    fn synthetic_weights() -> BcsWeights {
        let limits: Vec<i32> = (0..BCS_TABLE_ROWS as i32).map(|i| i % 7 - 1).collect();
        let rows = limits
            .iter()
            .enumerate()
            .map(|(i, &limit)| {
                (0..=limit)
                    .map(|k| {
                        if i % 3 == 0 {
                            0.0
                        } else {
                            1.0 / (k + 1 + (i % 5) as i32) as f32
                        }
                    })
                    .collect()
            })
            .collect();
        BcsWeights { rows, limits }
    }

    fn hex32(field: &str) -> u32 {
        u32::from_str_radix(field, 16).unwrap()
    }

    fn hex64(field: &str) -> u64 {
        u64::from_str_radix(field, 16).unwrap()
    }

    fn fixture(tag: &str) -> impl Iterator<Item = (Vec<&'static str>, Vec<&'static str>)> + '_ {
        NATIVE
            .lines()
            .filter(move |line| line.split_whitespace().next() == Some(tag))
            .map(|line| {
                let (inputs, outputs) = line.split_once('|').unwrap();
                (
                    inputs.split_whitespace().skip(1).collect(),
                    outputs.split_whitespace().collect(),
                )
            })
    }

    /// Runs every fixture case; tables of kind 0 need the system libm `exp`
    /// of the native build, so those cases only run on aarch64 macOS.
    #[test]
    fn match_col_offsets_mono_matches_native_battery() {
        let synthetic = synthetic_weights();
        let mut checked = 0;
        for (inputs, outputs) in fixture("BCS") {
            let name = inputs[0];
            let n = |i: usize| inputs[i].parse::<usize>().unwrap();
            let (cols, rows, mono_pad, rgb_pad, first, last) = (n(1), n(2), n(3), n(4), n(5), n(6));
            let filter_size = inputs[7].parse::<u32>().unwrap();
            let iso = f32::from_bits(hex32(inputs[8]));
            let scale_index = inputs[12].parse::<u32>().unwrap();
            let kind = inputs[14].parse::<u32>().unwrap();
            let table_scale = f32::from_bits(hex32(inputs[15]));
            let seed = inputs[16].parse::<u32>().unwrap();
            let pattern = inputs[17].parse::<u32>().unwrap();
            let built;
            let weights = match kind {
                0 => {
                    if !cfg!(all(target_arch = "aarch64", target_os = "macos")) {
                        continue;
                    }
                    built = BcsWeights::monochrome_denoise(table_scale).unwrap();
                    built.as_ref()
                }
                1 => Some(&synthetic),
                _ => None,
            };
            let mono_stride = cols + mono_pad;
            let rgb_stride = 3 * cols + rgb_pad;
            let mut mono = plane(mono_stride * rows, mono_stride, seed, pattern);
            let mut rgb = plane(rgb_stride * rows, rgb_stride, seed ^ 0xa5a5_a5a5, 0);
            match_col_offsets_mono(
                &mut mono,
                mono_stride,
                &mut rgb,
                rgb_stride,
                [cols, rows],
                first,
                last,
                iso,
                filter_size,
                weights,
                scale_index,
            )
            .unwrap();
            assert_eq!(mono.len(), outputs[0].parse::<usize>().unwrap(), "{name}");
            assert_eq!(fnv1a_f32(&mono), hex64(outputs[1]), "{name} mono");
            assert_eq!(rgb.len(), outputs[2].parse::<usize>().unwrap(), "{name}");
            assert_eq!(fnv1a_f32(&rgb), hex64(outputs[3]), "{name} rgb");
            checked += 1;
        }
        let expected = if cfg!(all(target_arch = "aarch64", target_os = "macos")) {
            48
        } else {
            4
        };
        assert_eq!(checked, expected);
    }

    /// The harness builds the kind-0 tables like `MonochromeDenoise`; the
    /// port must produce the same bits (system libm `exp`).
    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    #[test]
    fn monochrome_denoise_weights_match_harness() {
        let mut checked = 0;
        for (inputs, outputs) in fixture("TABLE") {
            let scale = f32::from_bits(hex32(inputs[0]));
            let weights = BcsWeights::monochrome_denoise(scale).unwrap().unwrap();
            let limits: Vec<f32> = weights
                .limits
                .iter()
                .map(|&l| f32::from_bits(l as u32))
                .collect();
            let flat: Vec<f32> = weights.rows.concat();
            assert_eq!(fnv1a_f32(&limits), hex64(outputs[0]), "{scale} limits");
            assert_eq!(flat.len(), outputs[1].parse::<usize>().unwrap());
            assert_eq!(fnv1a_f32(&flat), hex64(outputs[2]), "{scale} weights");
            checked += 1;
        }
        assert_eq!(checked, 4);
    }

    #[test]
    fn threshold_reads_the_contiguous_native_table() {
        assert_eq!(threshold(100.0), 0.05);
        // Past 6400 the thresholds run into the ISO points.
        assert_eq!(threshold(12800.0), 100.0);
        assert!(f32::from_bits(THRESHOLD_TABLE[SCALES + 5]).is_nan());
    }

    #[test]
    fn invalid_inputs_do_not_mutate() {
        let synthetic = synthetic_weights();
        let (cols, rows) = (20, 50);
        let mono0 = plane(cols * rows, cols, 1, 0);
        let rgb0 = plane(3 * cols * rows, 3 * cols, 2, 0);
        let mut broken = synthetic.clone();
        broken.rows[5].clear();
        let cases: [(usize, usize, usize, usize, u32, u32, Option<&BcsWeights>); 9] = [
            (cols - 1, 3 * cols, 0, 49, 9, 4, Some(&synthetic)),
            (cols, 3 * cols - 1, 0, 49, 9, 4, Some(&synthetic)),
            (cols, 3 * cols, 10, 9, 9, 4, Some(&synthetic)),
            (cols, 3 * cols, 0, 50, 9, 4, Some(&synthetic)),
            (cols, 3 * cols, 0, 49, 0, 4, Some(&synthetic)),
            (cols, 3 * cols, 0, 49, 43, 4, Some(&synthetic)),
            (cols, 3 * cols, 0, 49, 9, 9, Some(&synthetic)),
            (cols, 3 * cols, 0, 49, 9, 4, Some(&broken)),
            // Filter size 1 reads mono[2i] of the last row, past the image.
            (cols, 3 * cols, 0, 49, 1, 4, Some(&synthetic)),
        ];
        for (k, &(ms, rs, first, last, filter, scale, weights)) in cases.iter().enumerate() {
            let mut mono = mono0.clone();
            let mut rgb = rgb0.clone();
            let result = match_col_offsets_mono(
                &mut mono,
                ms,
                &mut rgb,
                rs,
                [cols, rows],
                first,
                last,
                800.0,
                filter,
                weights,
                scale,
            );
            assert!(result.is_err(), "case {k}");
            assert_eq!(mono, mono0, "case {k}");
            assert_eq!(rgb, rgb0, "case {k}");
        }
        assert!(BcsWeights::monochrome_denoise(0.0).unwrap().is_none());
        assert!(BcsWeights::monochrome_denoise(-1.0).is_err());
        assert!(BcsWeights::monochrome_denoise(f32::NAN).is_err());
    }
}
