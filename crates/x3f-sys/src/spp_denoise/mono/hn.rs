//! `FPixelProc2_F20::Sigma_HN` (0x42434), the `Flt32Mono <- Flt32RGB`
//! overload: RGB to one intensity plane, the kernel of `DoHNMixing` in
//! `MonochromeDenoise`.
//!
//! Per pixel `(r, g, b)` of the selected rows:
//!
//! 1. Only when `highlight_mix > 0` (a NaN mix counts as not positive), a
//!    highlight desaturation in `f64`:
//!    `t = A·rgb`, `w = clamp((t - start) / range, 0, 1)`, `y = B·rgb`, and
//!    each channel becomes `c' = (c * (1 - mix)) as f32 + (((c * (1 - w) +
//!    y * w) as f32) * mix) as f32`, the final sum in `f32`.
//! 2. Each channel is clamped to `[0, 32767]` in `f32` (NaN passes through).
//! 3. The output is `w_r * r + w_g * g + w_b * b` in `f32`, fused as
//!    `fma(w_b, b, fma(w_r, r, w_g * g))`.
//!
//! The two `FColorFloat&` arguments are read-only inputs: `A` (the highlight
//! detector weights) and `B` (the gray the highlights are pulled to). The
//! `FColorFloat32` is the output channel weighting.
//!
//! `DoHNMixing` (0x2fcac) calls it with `A = (1/32, 1/32, 15/16)`,
//! `B = (1/4, 1/2, 1/4)`, `start = 2500`, `range = 950`, `mix = 0` (so only
//! steps 2 and 3 run in production) and the weights `m_ColorMixing` read from
//! `NoiseParams+0x188/0x190/0x198` as `f64` and rounded to `f32`; its rows come
//! from `DoThreadedPix` as an inclusive band `[first, last]`.

use crate::{Error, Result};

/// `32767.0`, the native clamp (`0x46fffe00`).
const PIXEL_MAX: f32 = 32767.0;

/// Parameters of [`sigma_hn`], in native argument order.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct HnParams {
    /// First `FColorFloat&`: weights of the highlight detector `t = A·rgb`.
    pub highlight_weights: [f64; 3],
    /// Second `FColorFloat&`: weights of the gray `y = B·rgb` highlights move to.
    pub gray_weights: [f64; 3],
    /// First `double`: detector value where the ramp starts.
    pub highlight_start: f64,
    /// Second `double`: detector span over which the ramp reaches 1.
    pub highlight_range: f64,
    /// Third `double`: strength of the highlight blend; `<= 0` or NaN skips it.
    pub highlight_mix: f64,
    /// The `FColorFloat32`: output weights of `r`, `g`, `b`.
    pub color_mixing: [f32; 3],
}

impl HnParams {
    /// The arguments `DoHNMixing` passes, from `m_ColorMixing` as stored in
    /// `NoiseParams` (doubles, rounded to `f32` as native `fcvt` does).
    pub(crate) fn monochrome_denoise(color_mixing: [f64; 3]) -> Self {
        Self {
            highlight_weights: [0.03125, 0.03125, 0.9375],
            gray_weights: [0.25, 0.5, 0.25],
            highlight_start: 2500.0,
            highlight_range: 950.0,
            highlight_mix: 0.0,
            color_mixing: color_mixing.map(|value| value as f32),
        }
    }
}

fn clamp_pixel(value: f32) -> f32 {
    let value = if value > PIXEL_MAX { PIXEL_MAX } else { value };
    if value < 0.0 {
        0.0
    } else {
        value
    }
}

fn check_rows(len: usize, stride: usize, row_len: usize, last_row: usize) -> Result<()> {
    let needed = last_row
        .checked_mul(stride)
        .and_then(|offset| offset.checked_add(row_len));
    if stride < row_len || needed.is_none_or(|needed| len < needed) {
        return Err(Error::InvalidData("invalid SPP Sigma_HN image layout"));
    }
    Ok(())
}

/// Port of `FPixelProc2_F20::Sigma_HN` (0x42434): write
/// `output[y * output_stride + x]` from the interleaved RGB
/// `input[y * input_stride + 3 * x ..]` for `x < width` and the inclusive row
/// range `first_row..=last_row` (empty, and a no-op, when `first_row >
/// last_row`, as native). Strides count `f32`s. Nothing outside those rows and
/// columns is written. See the module docs for the formula.
pub(crate) fn sigma_hn(
    output: &mut [f32],
    output_stride: usize,
    input: &[f32],
    input_stride: usize,
    width: usize,
    first_row: usize,
    last_row: usize,
    params: &HnParams,
) -> Result<()> {
    if first_row > last_row || width == 0 {
        return Ok(());
    }
    let input_row = width
        .checked_mul(3)
        .ok_or(Error::InvalidData("invalid SPP Sigma_HN image layout"))?;
    check_rows(input.len(), input_stride, input_row, last_row)?;
    check_rows(output.len(), output_stride, width, last_row)?;

    let [wr, wg, wb] = params.color_mixing;
    let mix = params.highlight_mix;
    let blend = mix > 0.0;
    let keep = 1.0 - mix;
    let [ar, ag, ab] = params.highlight_weights;
    let [br, bg, bb] = params.gray_weights;

    for y in first_row..=last_row {
        let src = &input[y * input_stride..][..input_row];
        let dst = &mut output[y * output_stride..][..width];
        for (out, rgb) in dst.iter_mut().zip(src.chunks_exact(3)) {
            let mut c = [rgb[0], rgb[1], rgb[2]];
            if blend {
                let [r, g, b] = c.map(f64::from);
                let detector = ab.mul_add(b, ar.mul_add(r, ag * g));
                let gray = bb.mul_add(b, br.mul_add(r, bg * g));
                let ramp = (detector - params.highlight_start) / params.highlight_range;
                let ramp = if ramp > 1.0 { 1.0 } else { ramp };
                let ramp = if ramp < 0.0 { 0.0 } else { ramp };
                let rest = 1.0 - ramp;
                let pulled = gray * ramp;
                c = [r, g, b].map(|value| {
                    let desaturated = value.mul_add(rest, pulled) as f32;
                    let kept = (value * keep) as f32;
                    kept + (f64::from(desaturated) * mix) as f32
                });
            }
            let [r, g, b] = c.map(clamp_pixel);
            *out = wb.mul_add(b, wr.mul_add(r, wg * g));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::float_front::fnv1a_f32;

    const NATIVE: &str = include_str!("../../../tests/fixtures/spp-mono-hn-native.txt");
    const SENTINEL: f32 = -12345.5;

    fn mix32(mut x: u32) -> u32 {
        x ^= x >> 16;
        x = x.wrapping_mul(0x7feb_352d);
        x ^= x >> 15;
        x = x.wrapping_mul(0x846c_a68b);
        x ^ (x >> 16)
    }

    /// Same generator as `scripts/spp-native/sigma_hn.cpp`.
    fn pixel(seed: u32, index: u32, pattern: u32) -> f32 {
        let h = mix32(seed.wrapping_mul(0x9e37_79b9) ^ index);
        let h2 = mix32(h ^ 0x5bd1_e995);
        let mantissa = h & 0x7f_ffff;
        if pattern == 1 {
            return f32::from_bits((127 + 9 + h2 % 5) << 23 | mantissa);
        }
        if pattern == 2 && (h2 >> 24) % 8 == 0 {
            const SPECIALS: [u32; 8] = [
                0x7f80_0000,
                0xff80_0000,
                0x7fc0_0000,
                0x8000_0000,
                0x0000_0005,
                0x46ff_fe00,
                0x46ff_fe01,
                0x7f7f_ffff,
            ];
            return f32::from_bits(SPECIALS[(h2 % 8) as usize]);
        }
        let sign = if h2 >> 28 == 0 { 0x8000_0000 } else { 0 };
        f32::from_bits(sign | (120 + h2 % 25) << 23 | mantissa)
    }

    struct Case {
        name: String,
        width: usize,
        height: usize,
        in_stride: usize,
        out_stride: usize,
        first: usize,
        last: usize,
        seed: u32,
        pattern: u32,
        params: HnParams,
        first_out: u32,
        hash: u64,
    }

    fn cases() -> Vec<Case> {
        NATIVE
            .lines()
            .filter(|line| line.starts_with("HN "))
            .map(|line| {
                let (lhs, rhs) = line.split_once(" | ").unwrap();
                let f: Vec<&str> = lhs.split_whitespace().collect();
                let int = |i: usize| f[i].parse::<usize>().unwrap();
                let dbl = |i: usize| f64::from_bits(u64::from_str_radix(f[i], 16).unwrap());
                let flt = |i: usize| f32::from_bits(u32::from_str_radix(f[i], 16).unwrap());
                let (first_out, hash) = rhs.split_once(' ').unwrap();
                Case {
                    name: f[1].to_owned(),
                    width: int(2),
                    height: int(3),
                    in_stride: int(4),
                    out_stride: int(5),
                    first: int(6),
                    last: int(7),
                    seed: f[8].parse().unwrap(),
                    pattern: f[9].parse().unwrap(),
                    params: HnParams {
                        highlight_weights: [dbl(10), dbl(11), dbl(12)],
                        gray_weights: [dbl(13), dbl(14), dbl(15)],
                        highlight_start: dbl(16),
                        highlight_range: dbl(17),
                        highlight_mix: dbl(18),
                        color_mixing: [flt(19), flt(20), flt(21)],
                    },
                    first_out: u32::from_str_radix(first_out, 16).unwrap(),
                    hash: u64::from_str_radix(hash, 16).unwrap(),
                }
            })
            .collect()
    }

    #[test]
    fn matches_native_battery() {
        let cases = cases();
        assert_eq!(cases.len(), 133);
        for case in &cases {
            let input: Vec<f32> = (0..case.in_stride * case.height + 3)
                .map(|i| pixel(case.seed, i as u32, case.pattern))
                .collect();
            let mut output = vec![SENTINEL; case.out_stride * case.height];
            sigma_hn(
                &mut output,
                case.out_stride,
                &input,
                case.in_stride,
                case.width,
                case.first,
                case.last,
                &case.params,
            )
            .unwrap();
            if case.first <= case.last {
                assert_eq!(
                    output[case.first * case.out_stride].to_bits(),
                    case.first_out,
                    "case {} first value",
                    case.name
                );
            }
            assert_eq!(
                fnv1a_f32(&output),
                case.hash,
                "case {} seed {}",
                case.name,
                case.seed
            );
        }
    }

    #[test]
    fn production_parameters() {
        let params = HnParams::monochrome_denoise([0.3, 0.59, 0.11]);
        assert_eq!(params.highlight_mix, 0.0);
        assert_eq!(params.color_mixing, [0.3_f32, 0.59, 0.11]);
        // Plain path: clamp then weighted sum.
        let input = [40000.0, -1.0, 100.0, 1.0, 2.0, 3.0];
        let mut output = [SENTINEL; 2];
        sigma_hn(&mut output, 2, &input, 6, 2, 0, 0, &params).unwrap();
        let [wr, wg, wb] = params.color_mixing;
        assert_eq!(output[0], wb.mul_add(100.0, wr.mul_add(32767.0, 0.0)));
        assert_eq!(output[1], wb.mul_add(3.0, wr.mul_add(1.0, wg * 2.0)));
    }

    #[test]
    fn rejects_bad_layout_before_writing() {
        let params = HnParams::monochrome_denoise([0.3, 0.59, 0.11]);
        let input = vec![1.0_f32; 3 * 4 * 3];
        let mut output = vec![SENTINEL; 4 * 3];
        // Last row past the input.
        assert!(sigma_hn(&mut output, 4, &input, 12, 4, 0, 3, &params).is_err());
        // Input stride shorter than a row.
        assert!(sigma_hn(&mut output, 4, &input, 11, 4, 0, 2, &params).is_err());
        // Output too short.
        assert!(sigma_hn(&mut output[..11], 4, &input, 12, 4, 0, 2, &params).is_err());
        // Output stride shorter than a row.
        assert!(sigma_hn(&mut output, 3, &input, 12, 4, 0, 2, &params).is_err());
        assert!(output.iter().all(|value| *value == SENTINEL));
        // Empty range or width is a no-op, as native.
        assert!(sigma_hn(&mut output, 4, &input, 12, 4, 3, 2, &params).is_ok());
        assert!(sigma_hn(&mut output, 4, &[], 12, 0, 0, 2, &params).is_ok());
        assert!(output.iter().all(|value| *value == SENTINEL));
    }
}
