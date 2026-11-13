//! Despeckle and sharpness kernels of the monochrome Merrill pipeline.
//!
//! `DoDespSharp` (0x2fb7c), the last worker of `MonochromeDenoise`, runs on one
//! row band `[first, last]` (inclusive, as `DoThreadedPix` splits the image):
//! - [`despeckle3_mono`]: `Despeckle3(Flt32Mono*, Flt32RGB*, u32, u32, f32,
//!   f32)` at 0x2ed50, with `NoiseParams+0x44` and `NoiseParams+0x4`.
//! - [`sharpen_bnw`]: `FPixelProc2_F20::SharpenBnW(Flt32Mono*, 16SMono*, u32,
//!   u32, f32, f32)` at 0x81768, with the same two floats. The 16SMono plane is
//!   `MonochromeDenoise`'s second argument (worker context `+0x38`), which is
//!   the Stage2 request's `m_pOutLImage` (`+0x108`), that is the manager's
//!   `m_pStage2OutSharpLImageObj` (`IPManager2_F20+0x268`).
//!
//! Stage2B (`priProcessF20Stage2B`, 0xaa784) then reads that same plane back
//! (Stage2B request `+0x70`, taken from `m_pStage2OutSharpLImageObj`) in
//! [`apply_sharp_l_bnw`]: `FPixelProc2_F20::ApplySharpLBnW` at 0x810b0. When
//! grain insertion is off (`request+0x475 == 0`), Stage2B finally copies
//! channel 0 into channel 2 of every pixel: [`copy_channel0_to_channel2`].
//!
//! Sizes are `[columns, rows]`; strides count elements (f32 or i16). Each slice
//! must hold `(rows - 1) * stride + columns * channels` elements. Invalid
//! layouts and parameters return an error before any write.
//!
//! Every kernel is checked bit for bit against the native library through
//! `scripts/spp-native/desp_sharp_bnw.cpp`
//! (fixture `tests/fixtures/spp-mono-desp-sharp-native.txt`).

use crate::{Error, Result};

/// `0.35_f32` as the native literal (0x3eb33333).
const BOUND: f32 = f32::from_bits(0x3eb3_3333);
/// Native `SharpenBnW` limit table: limits at the ISO knots below.
const LIMITS: [f32; 4] = [500.0, 800.0, 1100.0, 1500.0];
const ISO_KNOTS: [f32; 4] = [100.0, 125.0, 160.0, 200.0];
/// Upper clamp of the 16-bit sharpness value.
const SHARP_MAX: f32 = 8191.0;
/// Upper clamp of `ApplySharpLBnW` (0x46fffe00).
const PIXEL_MAX: f32 = 32767.0;

fn invalid() -> Error {
    Error::InvalidData("invalid SPP mono sharpen layout")
}

/// Check one plane: non-empty, sizes and row bytes representable in the
/// native `i32` fields, and the slice long enough.
fn check_plane(
    len: usize,
    size: [usize; 2],
    stride: usize,
    channels: usize,
    element: usize,
) -> Result<()> {
    let [cols, rows] = size;
    let max = i32::MAX as usize;
    let row = cols.checked_mul(channels).ok_or_else(invalid)?;
    let row_bytes = stride.checked_mul(element).ok_or_else(invalid)?;
    if cols == 0 || rows == 0 || cols > max || rows > max || stride < row || row_bytes > max {
        return Err(invalid());
    }
    let needed = (rows - 1)
        .checked_mul(stride)
        .and_then(|n| n.checked_add(row))
        .ok_or_else(invalid)?;
    if len < needed {
        return Err(invalid());
    }
    Ok(())
}

/// `fcsel gt`: `a` when `a > b`, else `b`.
fn gt(a: f32, b: f32) -> f32 {
    if a > b {
        a
    } else {
        b
    }
}

/// `fcsel mi`: `a` when `a < b`, else `b`.
fn mi(a: f32, b: f32) -> f32 {
    if a < b {
        a
    } else {
        b
    }
}

/// Despeckle one band of the mono plane from channel 2 of the RGB image.
///
/// Native `Despeckle3` overload at 0x2ed50 (not the in-place mono overload at
/// 0x7bfd0 that `residual::despeckle3` mirrors). Only `rgb` channel 2 is read;
/// `mono` is only written, at the pixels that are replaced; every other mono
/// sample keeps its value (`MonochromeDenoise` fills it with channel 2 first).
///
/// The threshold is `(64 / (4096 / s))^3` with `s = max(floor, min(level /
/// 100, 64))`, all in f32. Centers run over rows `[max(first, 4) - 2,
/// min(last + 2, rows - 3)]` (i32 arithmetic, wrapping like the native) and
/// columns `[2, columns - 4]`; nothing happens below 6 columns. With `cmax`
/// and `cmin` the max and min of a 3-row column of channel 2, a center `c`
/// equal to `cmax[x]` and above `cmax[x - 1]` or `cmax[x + 1]` is replaced by
/// the max `m` of its 8 neighbors when `c - m > threshold`; otherwise (also
/// when that test fails) a center equal to `cmin[x]` and below `cmin[x - 1]`
/// or `cmin[x + 1]` is replaced by the 8-neighbor min `m` when
/// `c - m < threshold`. A true dark speckle has `c - m <= 0`, so for any
/// positive threshold dark speckles are always clamped (native asymmetry).
///
/// The native keeps a rolling window of column extrema seeded with columns 0
/// and 1 while the first center is column 2, so the first two centers of each
/// row compare against shifted columns (`x - 2`, `x - 1` and `x + 1` for
/// `x = 2`; `x - 2`, `x` and `x + 1` for `x = 3`). The port keeps that
/// window exactly.
#[allow(clippy::too_many_arguments)]
pub fn despeckle3_mono(
    mono: &mut [f32],
    mono_stride: usize,
    rgb: &[f32],
    rgb_stride: usize,
    size: [usize; 2],
    rows: [u32; 2],
    floor: f32,
    level: f32,
) -> Result<()> {
    check_plane(mono.len(), size, mono_stride, 1, 4)?;
    check_plane(rgb.len(), size, rgb_stride, 3, 4)?;
    let [cols, height] = size;

    let scaled = level / 100.0;
    let scaled = if scaled > 64.0 { 64.0 } else { scaled };
    let s = if scaled < floor { floor } else { scaled };
    let threshold = (64.0_f32 / (4096.0_f32 / s)).powf(3.0);

    let first = rows[0] as i32;
    let last = rows[1] as i32;
    let start = if first > 4 { first } else { 4 } - 2;
    let bottom = (height as i32).wrapping_sub(3);
    let after = last.wrapping_add(2);
    let end = if bottom < after { bottom } else { after };
    if start > end || (cols as i32) < 6 {
        return Ok(());
    }

    let at = |row: usize, x: usize| rgb[row * rgb_stride + 3 * x + 2];
    for r in start as usize..=end as usize {
        let (up, down) = (r - 1, r + 1);
        let max1 = gt(at(r, 1), at(up, 1));
        let min1 = mi(at(r, 1), at(up, 1));
        let max0 = gt(at(r, 0), at(up, 0));
        let min0 = mi(at(r, 0), at(up, 0));
        // Register names follow the native loop (0x2eee0).
        let mut s1 = gt(at(down, 0), max0);
        let mut s3 = gt(at(down, 1), max1);
        let mut s2 = mi(at(down, 0), min0);
        let mut s4 = mi(at(down, 1), min1);
        for x in 2..=cols - 4 {
            let s7 = s1;
            s1 = s3;
            let s5 = s2;
            s2 = s4;
            let (p, c) = (at(up, x + 1), at(r, x + 1));
            let high = gt(c, p);
            let low = mi(c, p);
            let n = at(down, x + 1);
            s3 = gt(n, high);
            s4 = mi(n, low);

            let center = at(r, x);
            let out = r * mono_stride + x;
            if (center > s3 || center > s7) && center == s1 {
                let m = gt(at(down, x), gt(at(up, x), gt(s3, s7)));
                if center - m > threshold {
                    mono[out] = m;
                    continue;
                }
            }
            if (center < s4 || center < s5) && center == s2 {
                let m = mi(at(down, x), mi(at(up, x), mi(s4, s5)));
                if center - m < threshold {
                    mono[out] = m;
                }
            }
        }
    }
    Ok(())
}

/// Interpolated `SharpenBnW` limit for an ISO value (native 0x81798..0x81880).
fn sharpen_limit(iso: f32) -> f32 {
    if iso >= 200.0 {
        return 1500.0;
    }
    if iso <= 100.0 {
        return 500.0;
    }
    let i = if iso <= 125.0 {
        1
    } else if iso <= 160.0 {
        2
    } else {
        3
    };
    let (x0, x1) = (ISO_KNOTS[i - 1], ISO_KNOTS[i]);
    LIMITS[i].mul_add(iso - x0, LIMITS[i - 1] * (x1 - iso)) / (x1 - x0)
}

/// Fill one band of the 16-bit sharpness plane from the mono plane.
///
/// Native `FPixelProc2_F20::SharpenBnW` at 0x81768 (the 16SMono overload; the
/// Flt32Mono overload at 0x813b8 is a fixed-limit float variant not used
/// here). Reads `mono` only, writes `sharp_l` only. Centers run over rows
/// `[max(first, 2) - 1, min(last + 1, rows - 2)]` (i32, wrapping like the
/// native) and columns `[1, columns - 2]`; nothing happens below 3 columns.
/// Other samples of `sharp_l` keep their value.
///
/// Per center `c` with 3 by 3 sum `S` (added row by row, left to right, from
/// `0.0`): `v = (c + S / -9) * amount`, zeroed unless `|v| > 0`, clamped to
/// `+-limit(iso)` (500 up to ISO 100, 1500 from ISO 200, linear between the
/// knots 100/125/160/200 -> 500/800/1100/1500), then to `+-0.35 * c * amount`,
/// and stored as `i16(i32(min(4 v + 0.5, 8191)))` (fused, truncating toward
/// zero; the low 16 bits of the saturated i32).
///
/// The native scalar and NEON paths agree for every input, NaN included.
/// `iso` must not be NaN: the native then reads past its limit table.
#[allow(clippy::too_many_arguments)]
pub fn sharpen_bnw(
    mono: &[f32],
    mono_stride: usize,
    sharp_l: &mut [i16],
    sharp_l_stride: usize,
    size: [usize; 2],
    rows: [u32; 2],
    amount: f32,
    iso: f32,
) -> Result<()> {
    check_plane(mono.len(), size, mono_stride, 1, 4)?;
    check_plane(sharp_l.len(), size, sharp_l_stride, 1, 2)?;
    if iso.is_nan() {
        return Err(Error::InvalidData("invalid SPP mono sharpen ISO"));
    }
    let [cols, height] = size;
    let limit = sharpen_limit(iso);

    let first = rows[0] as i32;
    let last = rows[1] as i32;
    let start = if first > 2 { first } else { 2 } - 1;
    let bottom = (height as i32).wrapping_sub(2);
    let end = if bottom <= last {
        bottom
    } else {
        last.wrapping_add(1)
    };
    if start > end || (cols as i32) < 3 {
        return Ok(());
    }

    for r in start as usize..=end as usize {
        let up = &mono[(r - 1) * mono_stride..];
        let row = &mono[r * mono_stride..];
        let down = &mono[(r + 1) * mono_stride..];
        let out = &mut sharp_l[r * sharp_l_stride..];
        for x in 1..cols - 1 {
            let mut sum = up[x - 1] + 0.0;
            sum += up[x];
            sum += up[x + 1];
            sum += row[x - 1];
            sum += row[x];
            sum += row[x + 1];
            sum += down[x - 1];
            sum += down[x];
            sum += down[x + 1];
            let center = row[x];
            let v = (center + sum / -9.0) * amount;
            let magnitude = if v > 0.0 { v } else { -v };
            let t = v * if magnitude > 0.0 { 1.0 } else { 0.0 };
            let t = if limit < t { limit } else { t };
            let t = if t < -limit { -limit } else { t };
            let scaled = center * amount;
            let upper = scaled * BOUND;
            let lower = -scaled * BOUND;
            let t = if t > lower { t } else { lower };
            let t = if upper < t { upper } else { t };
            let q = t.mul_add(4.0, 0.5);
            let q = if q > SHARP_MAX { SHARP_MAX } else { q };
            out[x] = q as i32 as i16;
        }
    }
    Ok(())
}

/// Add the 16-bit sharpness plane back to every RGB channel (Stage2B, mono).
///
/// Native `FPixelProc2_F20::ApplySharpLBnW` at 0x810b0, called by
/// `priProcessF20Stage2B` instead of the color `Sharpen` (0x80b34). For every
/// pixel: `d = f32(f64(sharp_l * 0.25_f32) * amount)`, then each channel
/// becomes `rgb + d` clamped to `[0, 32767]` (NaN and `-0.0` pass through).
/// Covers the whole image. The native counts rows and columns in 16 bits and
/// never terminates beyond 65535, so larger sizes are rejected.
pub fn apply_sharp_l_bnw(
    rgb: &mut [f32],
    rgb_stride: usize,
    sharp_l: &[i16],
    sharp_l_stride: usize,
    size: [usize; 2],
    amount: f64,
) -> Result<()> {
    check_plane(rgb.len(), size, rgb_stride, 3, 4)?;
    check_plane(sharp_l.len(), size, sharp_l_stride, 1, 2)?;
    if size[0] > 0xffff || size[1] > 0xffff {
        return Err(invalid());
    }
    let [cols, rows] = size;
    for y in 0..rows {
        let deltas = &sharp_l[y * sharp_l_stride..][..cols];
        let pixels = &mut rgb[y * rgb_stride..][..3 * cols];
        for (pixel, &s) in pixels.chunks_exact_mut(3).zip(deltas) {
            let delta = (f64::from(f32::from(s) * 0.25) * amount) as f32;
            for value in pixel {
                let v = *value + delta;
                let v = if v > PIXEL_MAX { PIXEL_MAX } else { v };
                *value = if v < 0.0 { 0.0 } else { v };
            }
        }
    }
    Ok(())
}

/// Copy channel 0 into channel 2 of every pixel (Stage2B 0xaab10, mono path
/// without grain insertion).
pub fn copy_channel0_to_channel2(
    rgb: &mut [f32],
    rgb_stride: usize,
    size: [usize; 2],
) -> Result<()> {
    check_plane(rgb.len(), size, rgb_stride, 3, 4)?;
    let [cols, rows] = size;
    for y in 0..rows {
        for pixel in rgb[y * rgb_stride..][..3 * cols].chunks_exact_mut(3) {
            pixel[2] = pixel[0];
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::float_front::fnv1a_f32;

    const NATIVE: &str = include_str!("../../../tests/fixtures/spp-mono-desp-sharp-native.txt");
    const S16_SENTINEL: i16 = 0x7e57;

    fn mix32(mut x: u32) -> u32 {
        x ^= x >> 16;
        x = x.wrapping_mul(0x7feb_352d);
        x ^= x >> 15;
        x = x.wrapping_mul(0x846c_a68b);
        x ^ (x >> 16)
    }

    /// Float generator of `desp_sharp_bnw.cpp` (exact values only).
    fn pixel(seed: u32, index: u32, pattern: u32) -> f32 {
        let h = mix32(seed.wrapping_mul(0x9e37_79b9) ^ index);
        let h2 = mix32(h ^ 0x5bd1_e995);
        let r = h2 >> 24;
        let wide = ((h % 16384) * 256 + (h2 & 255)) as f32 * 0.003_906_25;
        match pattern {
            0 => wide,
            1 => ((h % 4096) as i32 - 512) as f32 * 0.000_488_281_25,
            2 if r < 10 => (1000 + h % 8192) as f32,
            2 if r < 20 => (h % 1000) as f32 * 0.5,
            2 => 1000.0 + (h2 % 3) as f32 * 0.5,
            3 if h2 % 4 == 0 => [
                8191.0, 8191.5, 16383.0, 16383.75, 32767.0, 32766.5, 0.0, 8190.5,
            ][((h >> 8) % 8) as usize],
            3 => (7000 + h % 9500) as f32,
            4 => (index.wrapping_mul(37) % 16384) as f32 + (h % 16) as f32 * 0.0625,
            5 if r % 8 == 0 => f32::from_bits(
                [
                    0x7f80_0000,
                    0xff80_0000,
                    0x7fc0_0000,
                    0x8000_0000,
                    0x0000_0005,
                    0x46ff_fe00,
                    0x46ff_fe01,
                    0x7f7f_ffff,
                ][(h2 % 8) as usize],
            ),
            5 if h2 & 15 == 0 => -wide,
            5 => wide,
            _ if r < 16 => 0.5 + (h % 1024) as f32 * 0.000_976_562_5,
            _ if r < 32 => (h % 256) as f32 * 0.000_976_562_5,
            _ => 0.5 + (h2 % 3) as f32 * 0.000_244_140_625,
        }
    }

    fn sharp(seed: u32, index: u32, pattern: u32) -> i16 {
        let h = mix32(seed.wrapping_mul(0x85eb_ca6b) ^ index ^ 0x27d4_eb2f);
        match pattern {
            0 => h as u16 as i16,
            1 => ((h % 12001) as i32 - 6000) as i16,
            _ => ((h % 41) as i32 - 20) as i16,
        }
    }

    fn floats(len: usize, seed: u32, pattern: u32) -> Vec<f32> {
        (0..len as u32).map(|i| pixel(seed, i, pattern)).collect()
    }

    fn mono_init(rgb: &[f32], size: [usize; 2], rs: usize, ms: usize, init: u32) -> Vec<f32> {
        let mut mono: Vec<f32> = (0..ms * size[1] + 2)
            .map(|i| -0.5 * (1 + i % 97) as f32)
            .collect();
        if init == 1 {
            for y in 0..size[1] {
                for x in 0..size[0] {
                    mono[y * ms + x] = rgb[y * rs + 3 * x + 2];
                }
            }
        }
        mono
    }

    fn fnv1a_i16(values: &[i16]) -> u64 {
        values
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .fold(0xcbf2_9ce4_8422_2325, |h, b| {
                (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3)
            })
    }

    struct Line<'a> {
        name: &'a str,
        fields: Vec<&'a str>,
        hashes: Vec<u64>,
    }

    impl Line<'_> {
        fn u(&self, i: usize) -> usize {
            self.fields[i].parse().unwrap()
        }
        fn u32(&self, i: usize) -> u32 {
            self.fields[i].parse().unwrap()
        }
        fn f(&self, i: usize) -> f32 {
            f32::from_bits(u32::from_str_radix(self.fields[i], 16).unwrap())
        }
        fn d(&self, i: usize) -> f64 {
            f64::from_bits(u64::from_str_radix(self.fields[i], 16).unwrap())
        }
    }

    fn lines(kind: &str) -> Vec<Line<'static>> {
        NATIVE
            .lines()
            .filter(|line| line.starts_with(kind) && line.as_bytes().get(kind.len()) == Some(&b' '))
            .map(|line| {
                let (params, hashes) = line.split_once('|').unwrap();
                let mut fields: Vec<_> = params.split_whitespace().collect();
                fields.remove(0);
                let name = fields.remove(0);
                let hashes = hashes
                    .split_whitespace()
                    .map(|h| u64::from_str_radix(h, 16).unwrap())
                    .collect();
                Line {
                    name,
                    fields,
                    hashes,
                }
            })
            .collect()
    }

    fn despeckle_inputs(l: &Line<'_>, rs: usize, ms: usize, init: u32) -> (Vec<f32>, Vec<f32>) {
        let size = [l.u(0), l.u(1)];
        let rgb = floats(rs * size[1] + 3, l.u32(6), l.u32(7));
        let mono = mono_init(&rgb, size, rs, ms, init);
        (rgb, mono)
    }

    #[test]
    fn despeckle3_mono_matches_native() {
        let cases = lines("DS");
        assert!(cases.len() > 200);
        let mut changed = 0;
        for l in &cases {
            // w h rgb_stride mono_stride first last seed pattern init floor level
            let size = [l.u(0), l.u(1)];
            let (rs, ms) = (l.u(2), l.u(3));
            let (rgb, mut mono) = despeckle_inputs(l, rs, ms, l.u32(8));
            despeckle3_mono(
                &mut mono,
                ms,
                &rgb,
                rs,
                size,
                [l.u32(4), l.u32(5)],
                l.f(9),
                l.f(10),
            )
            .unwrap();
            assert_eq!(
                fnv1a_f32(&mono),
                l.hashes[0],
                "DS {} {:?}",
                l.name,
                l.fields
            );
            let (_, untouched) = despeckle_inputs(l, rs, ms, l.u32(8));
            changed += usize::from(fnv1a_f32(&untouched) != l.hashes[0]);
        }
        // The battery must exercise replacements, not only no-op bands.
        assert!(changed > cases.len() / 2, "{changed} of {}", cases.len());
    }

    #[test]
    fn sharpen_bnw_matches_native() {
        let cases = lines("SB");
        assert!(cases.len() > 200);
        let mut written = 0;
        for l in &cases {
            // w h mono_stride sl_stride first last seed pattern amount iso
            let size = [l.u(0), l.u(1)];
            let (ms, ss) = (l.u(2), l.u(3));
            let mono = floats(ms * size[1] + 2, l.u32(6), l.u32(7));
            let mut sl = vec![S16_SENTINEL; ss * size[1] + 1];
            sharpen_bnw(
                &mono,
                ms,
                &mut sl,
                ss,
                size,
                [l.u32(4), l.u32(5)],
                l.f(8),
                l.f(9),
            )
            .unwrap();
            assert_eq!(fnv1a_i16(&sl), l.hashes[0], "SB {} {:?}", l.name, l.fields);
            written += usize::from(sl.iter().any(|&v| v != S16_SENTINEL));
        }
        assert!(written > cases.len() / 2, "{written} of {}", cases.len());
    }

    #[test]
    fn desp_sharp_worker_matches_native() {
        let cases = lines("DD");
        assert!(cases.len() > 20);
        for l in &cases {
            // w h rgb_stride mono_stride sl_stride first last seed pattern amount iso
            let size = [l.u(0), l.u(1)];
            let (rs, ms, ss) = (l.u(2), l.u(3), l.u(4));
            let rows = [l.u32(5), l.u32(6)];
            let (amount, iso) = (l.f(9), l.f(10));
            let rgb = floats(rs * size[1] + 3, l.u32(7), l.u32(8));
            let mut mono = mono_init(&rgb, size, rs, ms, 1);
            let mut sl = vec![S16_SENTINEL; ss * size[1] + 1];
            despeckle3_mono(&mut mono, ms, &rgb, rs, size, rows, amount, iso).unwrap();
            sharpen_bnw(&mono, ms, &mut sl, ss, size, rows, amount, iso).unwrap();
            assert_eq!(
                fnv1a_f32(&mono),
                l.hashes[0],
                "DD mono {} {:?}",
                l.name,
                l.fields
            );
            assert_eq!(
                fnv1a_i16(&sl),
                l.hashes[1],
                "DD sl {} {:?}",
                l.name,
                l.fields
            );
        }
    }

    #[test]
    fn apply_sharp_l_bnw_matches_native() {
        let cases = lines("AS");
        assert!(cases.len() > 100);
        for l in &cases {
            // w h rgb_stride sl_stride seed pattern sl_pattern amount
            let size = [l.u(0), l.u(1)];
            let (rs, ss) = (l.u(2), l.u(3));
            let seed = l.u32(4);
            let mut rgb = floats(rs * size[1] + 3, seed, l.u32(5));
            let sl: Vec<i16> = (0..(ss * size[1] + 1) as u32)
                .map(|i| sharp(seed, i, l.u32(6)))
                .collect();
            apply_sharp_l_bnw(&mut rgb, rs, &sl, ss, size, l.d(7)).unwrap();
            assert_eq!(fnv1a_f32(&rgb), l.hashes[0], "AS {} {:?}", l.name, l.fields);
        }
    }

    #[test]
    fn sharpen_limit_knots() {
        assert_eq!(sharpen_limit(100.0), 500.0);
        assert_eq!(sharpen_limit(125.0), 800.0);
        assert_eq!(sharpen_limit(160.0), 1100.0);
        assert_eq!(sharpen_limit(200.0), 1500.0);
        assert_eq!(sharpen_limit(f32::INFINITY), 1500.0);
        assert_eq!(sharpen_limit(f32::NEG_INFINITY), 500.0);
    }

    #[test]
    fn copy_channel0_to_channel2_copies_every_pixel() {
        let mut rgb: Vec<f32> = (0..2 * 8).map(|i| i as f32).collect();
        copy_channel0_to_channel2(&mut rgb, 8, [2, 2]).unwrap();
        assert_eq!(
            rgb,
            [0.0, 1.0, 0.0, 3.0, 4.0, 3.0, 6.0, 7.0, 8.0, 9.0, 8.0, 11.0, 12.0, 11.0, 14.0, 15.0]
        );
    }

    #[test]
    fn invalid_inputs_do_not_mutate() {
        let rgb = vec![1.0_f32; 3 * 8 * 8];
        let mut mono = vec![2.0_f32; 8 * 8];
        let mut sl = vec![3_i16; 8 * 8];
        // Short RGB, short mono, stride below width, zero size.
        assert!(
            despeckle3_mono(&mut mono, 8, &rgb[..100], 24, [8, 8], [0, 7], 0.0, 100.0).is_err()
        );
        assert!(despeckle3_mono(&mut mono[..60], 8, &rgb, 24, [8, 8], [0, 7], 0.0, 100.0).is_err());
        assert!(despeckle3_mono(&mut mono, 7, &rgb, 24, [8, 8], [0, 7], 0.0, 100.0).is_err());
        assert!(sharpen_bnw(&mono, 8, &mut sl, 8, [0, 8], [0, 7], 1.0, 100.0).is_err());
        assert!(sharpen_bnw(&mono, 8, &mut sl[..63], 8, [8, 8], [0, 7], 1.0, 100.0).is_err());
        assert!(sharpen_bnw(&mono, 8, &mut sl, 8, [8, 8], [0, 7], 1.0, f32::NAN).is_err());
        let mut rgb2 = rgb.clone();
        assert!(apply_sharp_l_bnw(&mut rgb2, 23, &sl, 8, [8, 8], 1.0).is_err());
        assert!(apply_sharp_l_bnw(&mut rgb2, 24, &sl[..10], 8, [8, 8], 1.0).is_err());
        assert!(copy_channel0_to_channel2(&mut rgb2[..10], 24, [8, 8]).is_err());
        assert!(mono.iter().all(|&v| v == 2.0));
        assert!(sl.iter().all(|&v| v == 3));
        assert_eq!(rgb2, rgb);
    }
}
