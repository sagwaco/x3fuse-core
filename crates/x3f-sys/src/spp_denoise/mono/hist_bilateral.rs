//! The native `FPixelProc2_F20::ApplyHistBilateralFilter` mono overload
//! (0x3301c): `Flt32Mono` source, `Flt32RGB` destination, a row range and a
//! `FastBilateral_F20` passed by value. `DoFastBilateralFilter` (0x2fbd0)
//! calls it from `MonochromeDenoise` (0x2fd3c).
//!
//! The filter is a histogram bilateral over a 19 x 19 window. A 1024-bin
//! histogram holds a u16 count and an f32 sum per bin. A sample goes to bin
//! `clamp(trunc(v), 0, 4095) >> 2` (`fcvtzs` saturates and maps NaN to 0).
//! The window walks the image in vertical stripes, one column at a time:
//! down the first column, up the next, and so on. Each step removes the row
//! or column that leaves the window and then adds the one that enters, in
//! the native order, so every bin sum rounds exactly like the native. At each
//! center the output is
//! `sum_j fma(w_j, sum_j) / sum_j fma(w_j, f32(count_j))`, taken over bins
//! `j` in `low[b]..high[b]` of the center bin `b` that hold samples. The
//! weights `w` are `weights[b][j - low[b]]` (`CalculateCoeff`, 0x30940;
//! [`crate::spp_denoise::fast_bilateral_coeffs`]). Both sums are fused in
//! f32. A non-finite quotient falls back to a mono sample.
//!
//! Only channel 2 of the RGB destination changes, and only at columns
//! `9..width - 9` and rows `first'..=last' - 9` (but see the quirks). Here
//! `first' = max(first, 9)` and `last' = min(last + 9, height - 1), both as
//! signed 32-bit values like the native. Row `last` itself is written when
//! it is at most `height - 10`. The float argument (the noise scaling
//! `NoiseParams+0x28`) and the two trailing unsigned arguments are not read.
//! Of the `FastBilateral_F20` only the +0x50/+0x58/+0x60 tables are read.
//!
//! Native quirks kept:
//! - The fallback for a non-finite quotient reads the mono sample at
//!   `row * stride + (c + 9) + x` of the flat plane, where `x` is the
//!   written column (normally `c + 9`), not the center `(row, c + 9)`. Past
//!   the row end it reads the next row. It only fires when the window holds
//!   NaN or infinite samples, or when the bin counts are inconsistent.
//! - When `first' >= last' - 9` there is no vertical walk. Each column then
//!   produces one value, at row `last' - 9` going down and `first'` going up.
//!   An upward column never sets its column register (x30), so it writes one
//!   column to the left, over the previous downward result. With
//!   `first' = last' - 9` (a one-row range, or 19 rows), only the downward
//!   column of a pair keeps its value, and the last column stays unwritten
//!   when the column count is even.
//! - The counts are u16 and wrap. If `first' > last' - 9`, the column steps
//!   remove rows they never added. The native and this port then keep going
//!   on wrapped counts.
//!
//! Not reproduced: after the last column the native also steps the window
//! once more and reads one column past the image (the last row's read can
//! pass the end of the buffer). The result is discarded, so the port skips
//! that step.

use crate::spp_denoise::FastBilateralCoeffs;
use crate::{Error, Result};

const RADIUS: usize = 9;
const SPAN: usize = 2 * RADIUS + 1;
const BINS: usize = 1024;

fn invalid() -> Error {
    Error::InvalidData("invalid SPP hist bilateral layout")
}

fn bin(value: f32) -> usize {
    // fcvtzs, then min(0xfff), then max(0), then a logical >> 2.
    ((value as i32).min(0xfff).max(0) >> 2) as usize
}

struct Histogram {
    count: [u16; BINS],
    sum: [f32; BINS],
}

impl Histogram {
    fn add(&mut self, value: f32) {
        let b = bin(value);
        self.sum[b] += value;
        self.count[b] = self.count[b].wrapping_add(1);
    }

    fn remove(&mut self, value: f32) {
        let b = bin(value);
        self.sum[b] -= value;
        self.count[b] = self.count[b].wrapping_sub(1);
    }

    // The weighted mean around the center bin, or None when it is not finite
    // (the native also takes the fallback for an empty bin range).
    fn mean(&self, center: f32, coeffs: &FastBilateralCoeffs) -> Option<f32> {
        let b = bin(center);
        let (low, high) = (coeffs.low[b] as usize, coeffs.high[b] as usize);
        if coeffs.low[b] >= coeffs.high[b] {
            return None;
        }
        let (mut num, mut den) = (0.0_f32, 0.0_f32);
        for (j, &w) in (low..high).zip(&coeffs.weights[b]) {
            let count = self.count[j];
            if count != 0 {
                num = w.mul_add(self.sum[j], num);
                den = w.mul_add(f32::from(count), den);
            }
        }
        let mean = num / den;
        mean.is_finite().then_some(mean)
    }
}

/// Run the native `ApplyHistBilateralFilter` mono overload.
///
/// `mono` is the source plane with `mono_stride` f32 samples per row. `rgb`
/// is the interleaved destination with `rgb_stride` f32 samples per row. Both
/// are `size = [columns, rows]`. `first` and `last` are the native row
/// arguments; see the module docs for the rows they select. Images up to
/// 18 columns wide are left unchanged.
///
/// Returns an error before any write when a plane is too short for `size`,
/// when a coefficient bin is out of range or holds too few weights, or when
/// the window rows `first' - 9 ..= first' + 9` and `last' - 18 ..= last'`
/// are not all inside the image.
#[allow(clippy::too_many_arguments)]
pub(crate) fn apply_hist_bilateral_filter(
    mono: &[f32],
    mono_stride: usize,
    rgb: &mut [f32],
    rgb_stride: usize,
    size: [usize; 2],
    first: u32,
    last: u32,
    coeffs: &FastBilateralCoeffs,
) -> Result<()> {
    let [width, height] = size;
    let plane_ok = |len: usize, stride: usize, row: usize| {
        width > 0
            && height > 0
            && stride >= row
            && (height - 1)
                .checked_mul(stride)
                .and_then(|n| n.checked_add(row))
                .is_some_and(|n| n <= len)
    };
    let rows = i32::try_from(height).map_err(|_| invalid())?;
    if !plane_ok(mono.len(), mono_stride, width)
        || !plane_ok(
            rgb.len(),
            rgb_stride,
            width.checked_mul(3).ok_or_else(invalid)?,
        )
    {
        return Err(invalid());
    }
    for b in 0..BINS {
        let (low, high) = (coeffs.low[b], coeffs.high[b]);
        if low < high
            && (low < 0
                || high > BINS as i32
                || coeffs.weights.get(b).map_or(0, Vec::len) < (high - low) as usize)
        {
            return Err(invalid());
        }
    }
    // csel gt on signed 32-bit values, like the native.
    let first = (first as i32).max(RADIUS as i32);
    let last9 = (last as i32).wrapping_add(RADIUS as i32);
    let last = if rows > last9 { last9 } else { rows - 1 };
    if first + RADIUS as i32 > rows - 1 || last < 2 * RADIUS as i32 {
        return Err(invalid());
    }
    if width <= 2 * RADIUS {
        return Ok(());
    }
    let (first, last) = (first as usize, last as usize);
    let top = first - RADIUS;
    let bottom = last - 2 * RADIUS;
    let walk = (top as i32) < (bottom as i32);
    // The fallback reads stay inside the plane: the highest row is at most
    // height - 10, so row * stride + 2 * (width - 10) < (height - 1) * stride
    // + width.
    let at = |y: usize, x: usize| mono[y * mono_stride + x];

    let mut hist = Histogram {
        count: [0; BINS],
        sum: [0.0; BINS],
    };
    for y in top..top + SPAN {
        for x in 0..SPAN {
            hist.add(at(y, x));
        }
    }

    // `x` is the column the native writes and offsets the fallback by; it
    // is the window center `c + 9` except in one quirk below.
    let mut output = |hist: &Histogram, y: usize, c: usize, x: usize| {
        let center = c + RADIUS;
        let value = hist
            .mean(at(y, center), coeffs)
            .unwrap_or_else(|| mono[y * mono_stride + center + x]);
        rgb[y * rgb_stride + 3 * x + 2] = value;
    };
    let columns = width - 2 * RADIUS;
    for c in 0..columns {
        let down = c % 2 == 0;
        // Rows the column step moves across.
        let span_top = if down {
            if walk {
                for r in top..bottom {
                    output(&hist, r + RADIUS, c, c + RADIUS);
                    for x in c..c + SPAN {
                        hist.remove(at(r, x));
                    }
                    for x in c..c + SPAN {
                        hist.add(at(r + SPAN, x));
                    }
                }
            }
            output(&hist, last - RADIUS, c, c + RADIUS);
            bottom
        } else {
            if walk {
                let mut r = bottom;
                loop {
                    output(&hist, r + RADIUS, c, c + RADIUS);
                    for x in c..c + SPAN {
                        hist.remove(at(r + 2 * RADIUS, x));
                    }
                    r -= 1;
                    for x in c..c + SPAN {
                        hist.add(at(r, x));
                    }
                    if r <= top {
                        break;
                    }
                }
            }
            // Without the vertical walk the native never refreshes its
            // column register (x30) in an upward column. It still holds the
            // previous column's center, so the result lands one column to
            // the left, over the value the downward column wrote there.
            output(
                &hist,
                first,
                c,
                if walk { c + RADIUS } else { c + RADIUS - 1 },
            );
            top
        };
        if c + 1 == columns {
            break;
        }
        for y in span_top..span_top + SPAN {
            hist.remove(at(y, c));
        }
        for y in span_top..span_top + SPAN {
            hist.add(at(y, c + SPAN));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::float_front::fnv1a_f32;
    use crate::spp_denoise::fast_bilateral_coeffs;

    const FIXTURE: &str =
        include_str!("../../../tests/fixtures/spp-mono-hist-bilateral-native.txt");

    // The noise model MonochromeDenoise stores at FastBilateral_F20 +0x38,
    // +0x40 and +0x48 before CalculateCoeff.
    const MODEL: [f64; 3] = [9.85389175535173, 0.0428362301201803, 1.73171583027778e-05];

    fn mix32(mut x: u32) -> u32 {
        x ^= x >> 16;
        x = x.wrapping_mul(0x7feb_352d);
        x ^= x >> 15;
        x = x.wrapping_mul(0x846c_a68b);
        x ^ (x >> 16)
    }

    fn gradient(x: u32, y: u32, u: u32) -> f32 {
        let v = 2000.0 + (x * 3 + y * 2) as f32;
        v + ((u >> 20) as i32 - 2048) as f32 * 0.1
    }

    // The harness generator, one rounding per operation.
    fn sample(pattern: u32, seed: u32, x: u32, y: u32) -> f32 {
        let u = mix32(seed ^ (y.wrapping_mul(65537).wrapping_add(x)).wrapping_mul(0x9e37_79b9));
        match pattern {
            0 => (u >> 8) as f32 * (16383.0 / 16_777_216.0),
            1 => ((u % 81) as i32 - 20) as f32 + (u >> 24) as f32 * 0.003_906_25,
            2 => gradient(x, y, u),
            3 => 8191.0 - (u >> 23) as f32 * 0.5,
            4 => 3900.0 + (u >> 23) as f32 * 0.75,
            5 => {
                let base = (((x / 7 + y / 5) % 4) * 1100 + 300) as f32;
                base + ((u >> 24) as f32 * 0.5 - 64.0)
            }
            6 => match u % 37 {
                0 => f32::from_bits(0x7fc1_2345),
                1 => f32::INFINITY,
                2 => f32::NEG_INFINITY,
                3 => 1e30,
                4 => -1e30,
                5 => -0.0,
                _ => gradient(x, y, u),
            },
            7 => (u % 4096) as f32,
            _ => 16383.0 - (u >> 23) as f32,
        }
    }

    // Radius ((7 i + 3 k) mod 13) + k, weight 1 / (1 + |d| / 4).
    fn synthetic_coeffs(variant: i32) -> FastBilateralCoeffs {
        let mut low = [0_i32; 1024];
        let mut high = [0_i32; 1024];
        let mut weights = Vec::with_capacity(1024);
        for i in 0..1024_i32 {
            let r = (i * 7 + variant * 3) % 13 + variant;
            let (lo, hi) = ((i - r).max(0), (i + r).min(1023) + 1);
            low[i as usize] = lo;
            high[i as usize] = hi;
            weights.push(
                (lo..hi)
                    .map(|j| 1.0 / (1.0 + (i - j).abs() as f32 * 0.25))
                    .collect(),
            );
        }
        FastBilateralCoeffs { low, high, weights }
    }

    fn coeff_hash(coeffs: &FastBilateralCoeffs) -> u64 {
        let mut bytes = Vec::new();
        for v in coeffs.low.iter().chain(&coeffs.high) {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        for w in coeffs.weights.iter().flatten() {
            bytes.extend_from_slice(&w.to_bits().to_le_bytes());
        }
        let mut hash = 0xcbf2_9ce4_8422_2325_u64;
        for b in bytes {
            hash ^= u64::from(b);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
        hash
    }

    fn coeffs_for(mode: u32, param: u64) -> Option<FastBilateralCoeffs> {
        match mode {
            // libm exp: pinned to the platform the fixture was recorded on.
            0 if cfg!(all(target_arch = "aarch64", target_os = "macos")) => Some(
                fast_bilateral_coeffs(f64::from_bits(param), MODEL[0], MODEL[1], MODEL[2]),
            ),
            0 => None,
            _ => Some(synthetic_coeffs(param as i32)),
        }
    }

    fn hex(s: &str) -> u64 {
        u64::from_str_radix(s, 16).unwrap()
    }

    #[test]
    fn coefficient_sets_match_native() {
        let mut checked = 0;
        for line in FIXTURE.lines().filter(|l| l.starts_with("COEF ")) {
            let f: Vec<_> = line.split_whitespace().collect();
            if let Some(coeffs) = coeffs_for(f[1].parse().unwrap(), hex(f[2])) {
                assert_eq!(coeff_hash(&coeffs), hex(f[3]), "{line}");
                checked += 1;
            }
        }
        let expected = if cfg!(all(target_arch = "aarch64", target_os = "macos")) {
            9
        } else {
            3
        };
        assert_eq!(checked, expected);
    }

    #[test]
    fn hist_bilateral_matches_native_battery() {
        let mut cache: Vec<((u32, u64), FastBilateralCoeffs)> = Vec::new();
        let (mut checked, mut total) = (0, 0);
        for line in FIXTURE.lines().filter(|l| l.starts_with("HBF ")) {
            total += 1;
            let f: Vec<_> = line.split_whitespace().skip(1).collect();
            let n = |i: usize| f[i].parse::<u32>().unwrap();
            let [w, h, mono_pad, rgb_pad, first, last, pattern] = std::array::from_fn(|i| n(i));
            let (seed, mode, param) = (hex(f[7]) as u32, n(8), hex(f[9]));
            let key = (mode, param);
            if !cache.iter().any(|(k, _)| *k == key) {
                match coeffs_for(mode, param) {
                    Some(c) => cache.push((key, c)),
                    None => continue,
                }
            }
            let coeffs = &cache.iter().find(|(k, _)| *k == key).unwrap().1;
            let (w, h) = (w as usize, h as usize);
            let (ms, rs) = (w + mono_pad as usize, 3 * w + rgb_pad as usize);
            let mut mono = vec![-777.25_f32; ms * h];
            for y in 0..h {
                for x in 0..w {
                    mono[y * ms + x] = sample(pattern, seed, x as u32, y as u32);
                }
            }
            assert_eq!(fnv1a_f32(&mono), hex(f[10]), "mono input {line}");
            let mut rgb: Vec<f32> = (0..rs * h).map(|i| -1000.0 - (i % 97) as f32).collect();
            apply_hist_bilateral_filter(&mono, ms, &mut rgb, rs, [w, h], first, last, coeffs)
                .unwrap();
            assert_eq!(fnv1a_f32(&rgb), hex(f[11]), "{line}");
            checked += 1;
        }
        assert_eq!(total, 630);
        if cfg!(all(target_arch = "aarch64", target_os = "macos")) {
            assert_eq!(checked, total);
        } else {
            assert!(checked > 200);
        }
    }

    #[test]
    fn rejects_bad_layouts_before_writing() {
        let coeffs = synthetic_coeffs(0);
        let mono = vec![1.0_f32; 30 * 30];
        let mut rgb = vec![5.0_f32; 90 * 30];
        let run = |rgb: &mut [f32], size, first, last, c: &FastBilateralCoeffs| {
            apply_hist_bilateral_filter(&mono, 30, rgb, 90, size, first, last, c)
        };
        // Too few rows for the window, a short plane, rows past the bottom.
        assert!(run(&mut rgb, [30, 18], 0, 30, &coeffs).is_err());
        assert!(run(&mut rgb, [30, 31], 0, 30, &coeffs).is_err());
        assert!(run(&mut rgb, [30, 30], 21, 30, &coeffs).is_err());
        assert!(run(&mut rgb, [30, 30], 0, 8, &coeffs).is_err());
        let mut bad = coeffs.clone();
        bad.high[7] = 1025;
        assert!(run(&mut rgb, [30, 30], 0, 30, &bad).is_err());
        let mut bad = coeffs.clone();
        bad.weights[3].pop();
        assert!(run(&mut rgb, [30, 30], 0, 30, &bad).is_err());
        assert!(rgb.iter().all(|&v| v == 5.0));
        // A flat image filters to itself in channel 2 only.
        run(&mut rgb, [30, 30], 0, 30, &coeffs).unwrap();
        for (i, &v) in rgb.iter().enumerate() {
            let (y, x, ch) = (i / 90, (i % 90) / 3, i % 3);
            let inside = ch == 2 && (9..21).contains(&x) && (9..21).contains(&y);
            assert_eq!(v, if inside { 1.0 } else { 5.0 }, "{i}");
        }
    }
}
