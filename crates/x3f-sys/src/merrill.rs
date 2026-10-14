//! Reconstructed Merrill tone and RGB16 output math, verified against SPP's
//! supplied ARM64 library. This does not reconstruct its upstream image filters.
//! Evidence: `_local/spp-alignment/tone` and x3f-docs' `spp-rgb-trace` archive.

use crate::control::{Error, Result};
use crate::Control;

const LUT_SIZE: usize = 65536;
// Resource guard, not a native limit: includes 12-bit Merrill data at -2 EV.
// The exact native Bézier evaluation costs O(N * (N/50)^2).
const MAX_CURVE_EXTENT: f64 = 16384.0;

/// GetTanhLUT @0x5d99c, with ProcessTiles' fixed B={0.8,0.8,0.8}.
#[derive(Clone, Debug)]
pub struct ColorDq {
    tables: [Vec<i16>; 3],
    mask: usize,
}

impl ColorDq {
    #[allow(clippy::approx_constant)] // Native literals are not exact pi/2*pi.
    pub(crate) fn new(strengths: [f64; 3]) -> Result<Self> {
        let ratios = strengths.map(|v| v / 0.8);
        if !ratios.iter().all(|v| v.is_finite() && *v > 0.0) {
            return Err(Error::InvalidData("invalid Merrill ColorDQ strength"));
        }
        let step = (0.5 / ratios.into_iter().fold(0.0_f64, f64::max)) * 0.5;
        if !step.is_finite() || step <= 0.0 {
            return Err(Error::InvalidData("unsupported Merrill ColorDQ extent"));
        }
        // Rust's saturating cast agrees with ARM FCVTZS for this positive input.
        let requested = (6.2853 / step) as i32;
        let size = match requested {
            ..=512 => 512,
            ..=1024 => 1024,
            ..=2048 => 2048,
            ..=4096 => 4096,
            _ => 8192,
        };
        let mut tables = std::array::from_fn(|_| vec![0_i16; size]);
        for i in 0..size / 2 {
            let phase = step * i as f64;
            let window = if phase < 3.1416 {
                (phase.cos() + 1.0) * 0.5
            } else {
                0.0
            };
            for (table, strength) in tables.iter_mut().zip(strengths) {
                table[i] = ((strength * window).mul_add((0.8 * i as f64 / strength).tanh(), 0.5)
                    as i32) as i16;
            }
        }
        for table in &mut tables {
            for i in 1..size / 2 {
                table[size - i] = table[i].wrapping_neg();
            }
            table[size / 2] = table[size / 2 + 1];
        }
        Ok(Self {
            tables,
            mask: size - 1,
        })
    }

    /// Return the three signed lookup tables in RGB channel order.
    /// All tables have the same power-of-two length. The index mask is `len() - 1`.
    pub fn tables(&self) -> [&[i16]; 3] {
        self.tables.each_ref().map(Vec::as_slice)
    }

    /// ApplyMatrixAndColorDQ @0x73204..0x732c4, after its matrix multiply.
    /// The residual step that follows this operation bounds RGB to [0,32766].
    pub fn apply(&self, q: [f64; 3]) -> [f32; 3] {
        let residual = std::array::from_fn::<_, 3, _>(|c| {
            let i = ((q[c] + 1.5e-8) + 6755399441055744.0).to_bits() as i32;
            i.wrapping_sub(self.tables[c][i as usize & self.mask] as i32)
        });
        let mean = residual[0]
            .wrapping_add(residual[1].wrapping_shl(1))
            .wrapping_add(residual[2])
            >> 2;
        std::array::from_fn(|c| {
            (q[c] - self.tables[c][residual[c].wrapping_sub(mean) as usize & self.mask] as f64)
                as f32
        })
    }
}

/// NoiseParams+0x74: ISONoiseSettings column 7, then DenoisingSlidersStep
/// column 7. EstimatedISO is deliberately optional only for constant columns;
/// CaptureISO is not a replacement for the native estimated noise level.
pub(crate) fn color_dq_noise_scale(
    iso_rows: &[[f32; 13]],
    slider_rows: Option<&[[f32; 13]]>,
    estimated_iso: Option<f32>,
    luma_noise: f64,
) -> Result<f32> {
    let invalid = || Error::InvalidData("unsupported Merrill ColorDQ noise settings");
    if iso_rows.len() >= 20
        || !luma_noise.is_finite()
        || !iso_rows
            .iter()
            .all(|r| r[0].is_finite() && r[7].is_finite())
        || !iso_rows.windows(2).all(|r| r[0][0] < r[1][0])
    {
        return Err(invalid());
    }
    let base = if iso_rows.is_empty() {
        // Ordinary HI/MED/LOW fallback tables all have column 7 = 4 (@0xbf798/0xbf904/0xbfa70).
        4.0
    } else if iso_rows.iter().all(|r| r[7] == iso_rows[0][7]) {
        iso_rows[0][7]
    } else {
        let iso = estimated_iso
            .filter(|v| v.is_finite())
            .ok_or_else(invalid)?;
        let iso = iso.clamp(iso_rows[0][0], iso_rows.last().unwrap()[0]);
        let upper = iso_rows.iter().position(|r| iso <= r[0]).unwrap().max(1);
        let lo = iso_rows[upper - 1];
        let hi = iso_rows[upper];
        let weight = (iso - lo[0]) / (hi[0] - lo[0]);
        (lo[7] as f64).mul_add(1.0 - weight as f64, (weight * hi[7]) as f64) as f32
    };
    // The native fallback has 1.0 in column 7 for each of its five rows.
    let multiplier = match slider_rows {
        None => 1.0,
        Some(rows) if rows.len() == 5 => {
            let index = ((luma_noise.clamp(0.0, 1.0) as f32) * 4.0) as usize;
            rows[index][7]
        }
        Some(_) => return Err(invalid()),
    };
    let result = base * multiplier;
    if !result.is_finite() || result <= 0.0 {
        return Err(invalid());
    }
    Ok(result)
}

#[derive(Clone, Copy, Debug)]
pub struct ToneParams {
    pub start: f64,
    pub end: f64,
    pub steepness: f64,
    pub steepness2: f64,
    pub breakpoint: f64,
    pub scale: f64,
}

impl ToneParams {
    /// `TCColorModeSettings` rows: scale, start, end, k1, old gamma, break, k2.
    /// SetUpToneParamsF20 ignores the old gamma column and recognizes sentinel 8.
    pub(crate) fn from_color_mode_rows(rows: &[[f32; 7]], mode: usize) -> Self {
        if rows.is_empty() || rows.last().unwrap()[0] == 8.0 {
            return Self {
                start: -1.17,
                end: 1.55,
                steepness: 3.0,
                steepness2: 1.85,
                breakpoint: 0.1,
                scale: 1.0,
            };
        }
        let r = rows.get(mode).unwrap_or(&rows[0]);
        Self {
            start: r[1] as f64,
            end: r[2] as f64,
            steepness: r[3] as f64,
            steepness2: r[6] as f64,
            breakpoint: r[5] as f64,
            scale: r[0] as f64,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct ToneSettings {
    /// CAMF ImageDepth (default 12) supplies (1 << depth) - 1, not white level.
    pub pixel_saturation: f64,
    pub exposure: f64,
    pub shadows: f64,
    /// Includes ColorModeContrastCompensations on the ordinary color branch.
    pub contrast: f64,
    pub highlights: f64,
    pub alternate: bool,
}

fn invalid_tone() -> Error {
    Error::InvalidData("unsupported or degenerate Merrill tone parameters")
}

/// GenerateF20ToneCurve @0x57c64, FillToneCurve @0x57178. All three native
/// channel LUTs are identical, so retain one 65536-entry array per conversion.
pub(crate) fn generate_tone_lut(
    p: ToneParams,
    settings: ToneSettings,
    control: Control<'_>,
) -> Result<Vec<u16>> {
    let (curve, scale) = generate_merrill_tone_curve(p, settings, control)?;
    Ok(quantize_tone_curve(&curve, scale))
}

fn generate_merrill_tone_curve(
    p: ToneParams,
    settings: ToneSettings,
    control: Control<'_>,
) -> Result<(Vec<f32>, f64)> {
    control.check()?;
    if ![
        p.start,
        p.end,
        p.steepness,
        p.steepness2,
        p.breakpoint,
        p.scale,
        settings.pixel_saturation,
        settings.exposure,
        settings.shadows,
        settings.contrast,
        settings.highlights,
    ]
    .iter()
    .all(|v| v.is_finite())
        || p.start >= p.end
        || p.steepness <= 0.0
        || p.steepness2 <= 0.0
        || p.scale < 0.0
        || settings.pixel_saturation <= 0.0
    {
        return Err(invalid_tone());
    }
    let saturation = settings.pixel_saturation * (-settings.exposure).exp2();
    if !(1.0..=MAX_CURVE_EXTENT).contains(&saturation) {
        return Err(Error::InvalidData(
            "Merrill tone extent exceeds supported resource bound (1..16384)",
        ));
    }
    let extent = saturation.max(4095.0);
    let n = extent as usize;
    let (mut start, mut end, mut k1, mut k2) = if settings.alternate {
        (-1.17, 1.55, 3.25, 2.46)
    } else {
        (p.start, p.end, p.steepness, p.steepness2)
    };
    let contrast_scale = (0.5 * settings.contrast.clamp(-2.0, 2.0)).exp2();
    k1 *= contrast_scale;
    k2 *= contrast_scale;
    let shadow = (-settings.shadows).clamp(-2.0, 2.0);
    let mut highlight = settings.highlights.clamp(-2.0, 2.0);
    if !settings.alternate {
        start *= (shadow / 8.0).exp2();
        end *= (highlight / 8.0).exp2();
    }
    let inv_gamma = 1.0 / if settings.alternate { 2.4 } else { 2.2 };
    let domain = |x: f64| x.powf(inv_gamma).mul_add(end - start, start);
    let sigmoid = |x: f64, k: f64| 1.0 / (1.0 + (-k * x).exp());
    let mut curve = vec![0.0_f32; n + 1];
    if p.breakpoint < 1.0 {
        let mut i = 0;
        let mut x = 0.0;
        let step = 1.0 / saturation;
        loop {
            curve[i] = sigmoid(domain(x), k1) as f32;
            x += step;
            if x >= p.breakpoint || i >= n {
                break;
            }
            i += 1;
        }
        let at_break = domain(x);
        let correction = sigmoid(at_break, k1) - sigmoid(at_break, k2);
        for sample in &mut curve[i + 1..] {
            *sample = (sigmoid(domain(x), k2) + correction) as f32;
            x += step;
        }
    } else {
        let norm = sigmoid(domain(extent / saturation), k1);
        for (i, sample) in curve.iter_mut().enumerate() {
            *sample = (sigmoid(domain(i as f64 / saturation), k1) / norm) as f32;
        }
    }
    if settings.alternate && shadow != 0.0 {
        let threshold = curve[n] * 0.25;
        let stop = curve
            .iter()
            .position(|v| *v >= threshold)
            .ok_or_else(invalid_tone)?;
        let base = curve[0];
        let range = curve[stop] - base;
        if range <= 0.0 {
            return Err(invalid_tone());
        }
        let exponent = (2.0 * shadow).exp2() as f32;
        for sample in &mut curve[..=stop] {
            *sample = ((*sample - base) / range)
                .powf(exponent)
                .mul_add(range, base);
        }
    }
    if settings.alternate && highlight != 0.0 {
        let threshold = curve[n] * 0.75;
        let begin = curve
            .iter()
            .rposition(|v| *v <= threshold)
            .ok_or_else(invalid_tone)?;
        let base = curve[begin];
        let range = curve[n] - base;
        if range <= 0.0 {
            return Err(invalid_tone());
        }
        if highlight < 0.0 {
            highlight *= 0.25;
        }
        let exponent = (-2.0 * highlight).exp2() as f32;
        for sample in &mut curve[begin..] {
            *sample = ((*sample - base) / range)
                .powf(exponent)
                .mul_add(range, base);
        }
    }

    // Float32 De Casteljau, every 50th sample plus the final sample. Preserve
    // duplicate endpoints at multiples of 50 and the native FMA evaluation order.
    let mut controls: Vec<f32> = (0..=(extent / 50.0) as usize)
        .map(|i| curve[i * 50])
        .collect();
    controls.push(curve[n]);
    let mut work = controls.clone();
    let reciprocal = 1.0_f32 / extent as f32;
    for (i, sample) in curve.iter_mut().enumerate() {
        control.check()?;
        work.copy_from_slice(&controls);
        let t = i as f32 * reciprocal;
        let omt = 1.0 - t;
        for level in 1..work.len() {
            for j in (level..work.len()).rev() {
                work[j] = omt.mul_add(work[j - 1], t * work[j]);
            }
        }
        *sample = work[work.len() - 1];
    }

    let lo = curve[0] as f64;
    let range = curve[n] as f64 - lo;
    let at_sat = (curve[saturation as usize] as f64 - lo) / range;
    let normalization = saturation / at_sat;
    if !normalization.is_finite() || range <= 0.0 || at_sat <= 0.0 {
        return Err(invalid_tone());
    }
    for sample in &mut curve {
        let value = (*sample as f64 - lo) / range;
        let linear = if value < 0.04045 {
            value / 12.92
        } else {
            ((value + 0.055) / 1.055).powf(2.4)
        };
        *sample = (normalization * linear) as f32;
    }
    let scale = p.scale * 65535.0 / curve[n] as f64;
    if !scale.is_finite() || !curve.iter().all(|v| v.is_finite()) {
        return Err(invalid_tone());
    }
    Ok((curve, scale))
}

fn quantize_tone_curve(curve: &[f32], scale: f64) -> Vec<u16> {
    let n = curve.len() - 1;
    let mut lut = vec![0; LUT_SIZE];
    for i in 0..n {
        lut[i] = (curve[i] as f64).mul_add(scale, 0.5).clamp(0.0, 65535.0) as u16;
    }
    // The native builder normalizes with sample n, but repeats sample n-1.
    let tail = lut[n - 1];
    lut[n..].fill(tail);
    lut
}

fn color_mode_curve(
    params: &crate::ColorModeParameters,
    exposure: f64,
    control: Control<'_>,
) -> Result<(Vec<f32>, f64)> {
    match params.sensor {
        crate::ColorModeSensor::Merrill => generate_merrill_tone_curve(
            params.tone,
            ToneSettings {
                pixel_saturation: params.pixel_saturation,
                exposure,
                shadows: 0.0,
                contrast: params.contrast,
                highlights: 0.0,
                alternate: false,
            },
            control,
        ),
        crate::ColorModeSensor::Quattro => generate_quattro_tone_curve(params, exposure, control),
    }
}

pub(crate) fn generate_color_mode_tone_lut(
    params: &crate::ColorModeParameters,
    exposure: f64,
    control: Control<'_>,
) -> Result<Vec<u16>> {
    let (curve, scale) = color_mode_curve(params, exposure, control)?;
    Ok(quantize_tone_curve(&curve, scale))
}

pub(crate) fn generate_color_mode_linear_tone_lut(
    params: &crate::ColorModeParameters,
    exposure: f64,
    control: Control<'_>,
) -> Result<Vec<f64>> {
    let (curve, scale) = color_mode_curve(params, exposure, control)?;
    let n = curve.len() - 1;
    Ok((0..LUT_SIZE)
        .map(|i| (f64::from(curve[i.min(n - 1)]) * scale / 65535.0).clamp(0.0, 1.0))
        .collect())
}

/// Normal alpha 1 makes the fixed curve's blend coefficient zero.
fn generate_quattro_tone_curve(
    params: &crate::ColorModeParameters,
    exposure: f64,
    control: Control<'_>,
) -> Result<(Vec<f32>, f64)> {
    control.check()?;
    let p = params.tone;
    let saturation = params.pixel_saturation * (-exposure).exp2();
    if ![
        p.start,
        p.end,
        p.steepness,
        p.steepness2,
        p.breakpoint,
        p.scale,
        params.contrast,
        exposure,
    ]
    .iter()
    .all(|v| v.is_finite())
        || p.start >= p.end
        || p.steepness <= 0.0
        || p.steepness2 <= 0.0
        || p.scale <= 0.0
        || !(4.0..=MAX_CURVE_EXTENT).contains(&saturation)
    {
        return Err(Error::InvalidData(
            "unsupported or degenerate Quattro tone parameters",
        ));
    }
    let n = saturation as usize;
    let contrast = (0.5 * params.contrast.clamp(-2.0, 2.0)).exp2();
    let (k1, k2) = (p.steepness * contrast, p.steepness2 * contrast);
    let domain = |x: f64| {
        x.powf(1.0 / 2.2222222222222223)
            .mul_add(p.end - p.start, p.start)
    };
    let sigmoid = |x: f64, k: f64| 1.0 / (1.0 + (-k * x).exp());
    let mut curve = vec![0.0_f32; n + 1];
    let mut x = 0.0;
    let step = 1.0 / saturation;
    let mut i = 0;
    loop {
        curve[i] = sigmoid(domain(x), k1) as f32;
        x += step;
        if x >= p.breakpoint || i >= n {
            break;
        }
        i += 1;
    }
    let at_break = domain(x);
    let correction = sigmoid(at_break, k1) - sigmoid(at_break, k2);
    for sample in &mut curve[i + 1..] {
        *sample = (sigmoid(domain(x), k2) + correction) as f32;
        x += step;
    }
    // CombineToneCurve blends one sample per four counts, then smooths that grid.
    let quarter_extent = saturation * 0.25;
    let quarter_n = n / 4;
    let mut controls: Vec<f32> = (0..=(saturation / 50.0 * 0.25) as usize)
        .map(|i| curve[i * 200])
        .collect();
    controls.push(curve[quarter_n * 4]);
    let mut work = controls.clone();
    let reciprocal = 1.0_f32 / quarter_extent as f32;
    let mut quarter = vec![0.0_f32; quarter_n + 1];
    for (i, sample) in quarter.iter_mut().enumerate() {
        control.check()?;
        work.copy_from_slice(&controls);
        let t = i as f32 * reciprocal;
        let omt = 1.0 - t;
        for level in 1..work.len() {
            for j in (level..work.len()).rev() {
                work[j] = omt.mul_add(work[j - 1], t * work[j]);
            }
        }
        *sample = work[work.len() - 1];
    }
    curve[0] = quarter[0];
    for (i, pair) in quarter.windows(2).enumerate() {
        let base = curve[i * 4];
        let delta = (pair[1] - pair[0]) * 0.25;
        curve[i * 4 + 1] = base + delta;
        curve[i * 4 + 2] = delta.mul_add(2.0, base);
        curve[i * 4 + 3] = delta.mul_add(3.0, base);
        curve[i * 4 + 4] = pair[1];
    }
    curve[quarter_n * 4 + 1..].fill(quarter[quarter_n]);
    let lo = f64::from(curve[0]);
    let range = f64::from(curve[n]) - lo;
    let at_sat = (f64::from(curve[saturation as usize]) - lo) / range;
    let normalization = saturation / at_sat;
    if !normalization.is_finite() || range <= 0.0 || at_sat <= 0.0 {
        return Err(Error::InvalidData("degenerate Quattro tone normalization"));
    }
    for sample in &mut curve {
        let v = (f64::from(*sample) - lo) / range;
        let linear = if v < 0.04045 {
            v / 12.92
        } else {
            ((v + 0.055) / 1.055).powf(2.4)
        };
        *sample = (normalization * linear) as f32;
    }
    let scale = p.scale * 65535.0 / f64::from(curve[n]);
    if !scale.is_finite() || curve.iter().any(|v| !v.is_finite()) {
        return Err(Error::InvalidData("invalid Quattro tone curve"));
    }
    Ok((curve, scale))
}

/// CreateGammaLUT_F20 @0x561f4; manager @0x9efe4 supplies mask65535,size65536.
fn generate_gamma_lut(native_target: u32) -> Vec<u16> {
    let (threshold, slope, offset, power): (f64, f64, f64, f64) = match native_target {
        1 | 3 | 5 | 7 | 9 | 10 => (0.0, 1.0, 0.0, 1.0 / 2.2),
        4 | 6 | 8 | 11 => (0.0, 1.0, 0.0, 1.0 / 1.8),
        _ => (0.0031308, 12.92, 0.055, 1.0 / 2.4),
    };
    (0..LUT_SIZE)
        .map(|i| {
            let value = i as f64 / LUT_SIZE as f64;
            let encoded = if value < threshold {
                slope * value
            } else {
                (1.0 + offset).mul_add(value.powf(power), -offset)
            };
            encoded.mul_add(65535.0, 0.5) as u16
        })
        .collect()
}

/// Native integer tone, output matrix and transfer lookup tables.
pub struct RenderLuts {
    tone: Vec<u16>,
    gamma: Vec<u16>,
    fixed: [i32; 9],
}

impl RenderLuts {
    /// Targets: 2 for sRGB, 3 for Adobe RGB, and 11 for ProPhoto RGB.
    pub fn from_color_mode(
        params: &crate::ColorModeParameters,
        exposure: f64,
        output_matrix: [f64; 9],
        native_target: u32,
        control: Control<'_>,
    ) -> Result<Self> {
        if !matches!(native_target, 2 | 3 | 11) {
            return Err(Error::InvalidData("unsupported color mode output target"));
        }
        let mut fixed = [0; 9];
        for (dest, value) in fixed.iter_mut().zip(output_matrix) {
            let scaled = value.mul_add(4096.0, 0.5);
            if !scaled.is_finite() || scaled < i32::MIN as f64 || scaled > i32::MAX as f64 {
                return Err(Error::InvalidData("invalid color mode output matrix"));
            }
            *dest = scaled as i32;
        }
        Ok(Self {
            tone: generate_color_mode_tone_lut(params, exposure, control)?,
            gamma: generate_gamma_lut(native_target),
            fixed,
        })
    }

    pub(crate) fn new(
        params: ToneParams,
        settings: ToneSettings,
        output_matrix: [f64; 9],
        native_target: u32,
        control: Control<'_>,
    ) -> Result<Self> {
        let mut fixed = [0; 9];
        for (dest, value) in fixed.iter_mut().zip(output_matrix) {
            let scaled = value.mul_add(4096.0, 0.5);
            if !scaled.is_finite() || scaled < i32::MIN as f64 || scaled > i32::MAX as f64 {
                return Err(Error::InvalidData("invalid Merrill output matrix"));
            }
            *dest = scaled as i32;
        }
        Ok(Self {
            tone: generate_tone_lut(params, settings, control)?,
            gamma: generate_gamma_lut(native_target),
            fixed,
        })
    }

    /// Unit-scale Stage3 conversion followed by ordinary RGB48 output, with
    /// channel gains1 and retinex gain256. Inputs are finite native count units.
    pub(crate) fn render(&self, working: [f32; 3]) -> [u16; 3] {
        let indices = working.map(|v| {
            // CropSizeRotateF20 @0x51570: asymmetric +0.5, truncate, low16.
            ((v as f64 + 0.5) as i64 as u16) as usize
        });
        self.render_indices(indices)
    }

    /// Apply tone, Q12 output matrix and output transfer.
    /// Each index must be in 0..65536.
    pub fn render_indices(&self, indices: [usize; 3]) -> [u16; 3] {
        self.render_linear_indices(indices)
            .map(|v| self.gamma[v as usize])
    }

    /// Apply tone and Q12 output matrix before output transfer.
    /// Each index must be in 0..65536. Results use the 0..65535 linear range.
    pub fn render_linear_indices(&self, indices: [usize; 3]) -> [u16; 3] {
        let tone = indices.map(|i| self.tone[i] as i32);
        std::array::from_fn(|r| {
            // MUL/MADD W in output48row wrap at 32 bits before signed shift.
            let mut sum = 2048_i32;
            for (c, sample) in tone.iter().enumerate() {
                sum = sum.wrapping_add(self.fixed[3 * r + c].wrapping_mul(*sample));
            }
            (sum >> 12).clamp(0, 65535) as u16
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quattro_tone_matches_native_complete_arrays() {
        let shapes = [
            [-1.19, 2.85, 3.0, 0.078, 1.55],
            [-1.20, 2.85, 3.55, 0.070, 1.60],
            [-1.19, 2.85, 2.80, 0.080, 1.56],
            [-1.05, 2.65, 3.10, 0.030, 1.97],
            [-1.17, 2.85, 3.0, 0.1, 1.4],
            [-1.20, 2.85, 3.50, 0.070, 1.90],
        ];
        // lib95 ARM64 SHA256 a0f64d51, GenerateF20ToneCurve with alpha 1.
        let expected = [
            0x4d960d5fc12a5ac0,
            0x1728f75d86db08df,
            0x22acafaa3b8b0039,
            0x8cb526d93c724e87,
            0x641b6affc2c35f4d,
            0xfa327e6114870943,
            0x68526ce16cdd9f56,
            0xf3243f8894a4edf8,
            0x8bafb9bbfb182d42,
            0x772ff05d95a5b3e3,
            0x02a33202963a09d6,
            0x803fca0b1b0029bb,
            0xd1cbc17ec8033faf,
            0x66b5e74162288a41,
            0x90542ec120ac0b83,
            0x688d30788a13fdd5,
            0xecf840129b7da72b,
            0x9ccd0033b66cabb1,
        ];
        for (shape, [start, end, steepness, breakpoint, steepness2]) in
            shapes.into_iter().enumerate()
        {
            for setting in 0..3 {
                let params = crate::ColorModeParameters {
                    mode: crate::ColorMode::Standard,
                    sensor: crate::ColorModeSensor::Quattro,
                    as_shot_fallback: false,
                    creative_matrix: None,
                    contrast: if setting == 2 { 0.3 } else { 0.0 },
                    tone: ToneParams {
                        start,
                        end,
                        steepness,
                        steepness2,
                        breakpoint,
                        scale: 1.0,
                    },
                    inverse_gamma: 1.0 / 2.2222222222222223,
                    pixel_saturation: 16383.0,
                };
                let exposure = if setting == 1 { 1.0 } else { 0.0 };
                let lut = generate_color_mode_tone_lut(&params, exposure, Control::none()).unwrap();
                assert_eq!(
                    hash(&lut),
                    expected[shape * 3 + setting],
                    "shape={shape} setting={setting}"
                );
                let linear =
                    generate_color_mode_linear_tone_lut(&params, exposure, Control::none())
                        .unwrap();
                assert!(lut
                    .iter()
                    .zip(linear)
                    .all(
                        |(&encoded, linear)| (f64::from(encoded) - linear * 65535.0).abs()
                            <= 0.50000001
                    ));
                if shape == 0 && setting == 0 {
                    // The retained native HP-on branch changes k2 by float32 -0.7.
                    let mut hp_on = params;
                    hp_on.tone.steepness2 += f64::from(-0.7_f32);
                    assert_eq!(
                        hash(&generate_color_mode_tone_lut(&hp_on, 0.0, Control::none()).unwrap()),
                        0x022fdfe0496c2d4c
                    );
                }
            }
        }
    }

    #[test]
    fn color_dq_matches_native_luts_and_pixel_rows() {
        // Native GetTanhLUT + ApplyMatrixAndColorDQ(identity), original dylib.
        // FNV-1a of every LE i16 LUT entry and every output f32 byte.
        let cases = [
            (
                [0.1, 0.1, 0.1],
                512,
                [0x51d88627df287325; 3],
                0x3edd03b65ae6b39a,
            ),
            (
                [8.0, 8.0, 8.0],
                512,
                [0x6ace97c9069d120e; 3],
                0xffced507afc39209,
            ),
            (
                [8.0, 0.25, 32.0],
                1024,
                [0xe0f9a9d3b03087b4, 0x28c31cf8df2ec325, 0x0572ee70065b366f],
                0xcd449795454c7354,
            ),
            (
                [64.0, 20.0, 2.0],
                2048,
                [0x04064d770e4ec5bd, 0x9645753ca5f4124e, 0x9ed39b3f35a66d13],
                0x521eaad3364ec7ba,
            ),
            (
                [512.0, 256.0, 128.0],
                8192,
                [0xfcd3a9b679c930a0, 0x1e859ea80de5958b, 0x3221fd64d64fcf57],
                0x2246c3b3ce9a951d,
            ),
            (
                [1000.0, 1.0, 2.0],
                8192,
                [0xd1f93f7a21daa51a, 0x46f925813f214324, 0xb45291539499ea97],
                0x5c417d05f279a851,
            ),
        ];
        let mut pixels: Vec<[f32; 3]> = (0..1024)
            .map(|i| {
                [
                    (i - 512) as f32 * 0.5,
                    (i % 71 - 30) as f32 * 0.25,
                    (i % 31 - 15) as f32 * 1.5,
                ]
            })
            .collect();
        pixels.extend([
            [1e9, -1e9, 65536.0],
            [2147483648.0, 2147483648.0, -2147483648.0],
            [-0.5, 0.5, 1.5],
            [-0.50000006, 0.49999997, 32766.0],
        ]);
        for (strengths, size, lut_hashes, pixel_hash) in cases {
            let dq = ColorDq::new(strengths).unwrap();
            assert_eq!(dq.mask + 1, size);
            assert!(size.is_power_of_two());
            for (table, expected) in dq.tables().into_iter().zip(lut_hashes) {
                assert_eq!(table.len(), size);
                assert_eq!(
                    hash(&table.iter().map(|v| *v as u16).collect::<Vec<_>>()),
                    expected
                );
            }
            let actual = pixels
                .iter()
                .flat_map(|p| dq.apply(p.map(f64::from)))
                .flat_map(f32::to_le_bytes)
                .fold(0xcbf29ce484222325_u64, |h, b| {
                    (h ^ b as u64).wrapping_mul(0x100000001b3)
                });
            assert_eq!(actual, pixel_hash, "strengths={strengths:?}");
        }
    }

    #[test]
    fn color_dq_noise_scale_uses_linear_iso_and_discrete_slider() {
        let mut rows = [[0.0_f32; 13]; 2];
        rows[0][0] = 100.0;
        rows[1][0] = 400.0;
        rows[0][7] = 2.0;
        rows[1][7] = 8.0;
        assert_eq!(
            color_dq_noise_scale(&rows, None, Some(250.0), 0.5).unwrap(),
            5.0
        );
        assert_eq!(
            color_dq_noise_scale(&rows, None, Some(0.0), 0.5).unwrap(),
            2.0
        );
        assert_eq!(
            color_dq_noise_scale(&rows, None, Some(800.0), 0.5).unwrap(),
            8.0
        );
        assert!(color_dq_noise_scale(&rows, None, None, 0.5).is_err());
        rows[0][7] = 4.0;
        rows[1][7] = 4.0;
        assert_eq!(color_dq_noise_scale(&rows, None, None, 0.5).unwrap(), 4.0);
        let mut sliders = [[0.0; 13]; 5];
        for (i, row) in sliders.iter_mut().enumerate() {
            row[7] = (i + 1) as f32;
        }
        assert_eq!(
            color_dq_noise_scale(&rows, Some(&sliders), None, 0.499).unwrap(),
            8.0
        );
        assert_eq!(
            color_dq_noise_scale(&rows, Some(&sliders), None, 0.5).unwrap(),
            12.0
        );
        assert_eq!(color_dq_noise_scale(&[], None, None, 0.5).unwrap(), 4.0);
        assert_eq!(
            color_dq_noise_scale(&[], Some(&sliders), None, 0.5).unwrap(),
            12.0
        );
        assert!(color_dq_noise_scale(&rows, Some(&sliders[..4]), None, 0.5).is_err());
        rows[1][0] = 50.0;
        assert!(color_dq_noise_scale(&rows, None, None, 0.5).is_err());
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(ColorDq::new([bad, 8.0, 8.0]).is_err());
        }
    }

    fn params() -> ToneParams {
        ToneParams::from_color_mode_rows(&[[1.0, -1.17, 1.65, 3.0, 0.45, 0.1, 1.7]], 0)
    }

    fn settings() -> ToneSettings {
        ToneSettings {
            pixel_saturation: 4095.0,
            exposure: 0.0,
            shadows: 0.0,
            contrast: 0.0,
            highlights: 0.0,
            alternate: false,
        }
    }

    fn hash(values: &[u16]) -> u64 {
        values.iter().fold(0xcbf29ce484222325_u64, |h, value| {
            value
                .to_le_bytes()
                .into_iter()
                .fold(h, |h, byte| (h ^ byte as u64).wrapping_mul(0x100000001b3))
        })
    }

    #[test]
    fn tone_matches_native_complete_arrays() {
        // Native ARM64 GenerateF20ToneCurve; FNV-1a over LE u16 array bytes.
        // All 65536 entries in each native R/G/B table agree with this fixture.
        let expected = [
            0x0eab876ff87980c5,
            0xfcbbabcb185ceafa,
            0x2ece517c157a7f9c,
            0xd864507c4eae949e,
            0xa479268cb82d2ab1,
            0x290914fde5b9e5cf,
            0x4b4c6d560fc02971,
            0x7a476fc3550af55a,
        ];
        let controls = [
            [0.0; 4],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.3, -1.0],
            [0.0, -1.0, -0.3, 1.0],
        ];
        for (i, expected) in expected.into_iter().enumerate() {
            let [exposure, shadows, contrast, highlights] = controls[i % 4];
            let lut = generate_tone_lut(
                params(),
                ToneSettings {
                    exposure,
                    shadows,
                    contrast,
                    highlights,
                    alternate: i >= 4,
                    ..settings()
                },
                Control::none(),
            )
            .unwrap();
            assert_eq!(hash(&lut), expected, "native vector {i}");
        }
    }

    #[test]
    fn gamma_matches_native_complete_arrays() {
        for (target, expected) in [
            (2, 0xfff5136dc4ac67cd),
            (3, 0x111d54a5242d4078),
            (4, 0x7f9c9ce9a81623aa),
            (8, 0x7f9c9ce9a81623aa),
            (11, 0x7f9c9ce9a81623aa),
        ] {
            assert_eq!(
                hash(&generate_gamma_lut(target)),
                expected,
                "profile {target}"
            );
        }
    }

    #[test]
    fn native_tone_fallback_breakpoint_and_fractional_extent() {
        let expected = [
            0x6ceeb886ca0a6d99,
            0x0d3ed344d2bde039,
            0x645a496c647d61dc,
            0xfe024992875c26bc,
        ];
        for (i, expected) in expected.into_iter().enumerate() {
            let mut p = params();
            let mut s = settings();
            match i {
                0 => {
                    p = ToneParams::from_color_mode_rows(&[], 0);
                    s.exposure = -2.0;
                }
                1 => {
                    p.breakpoint = 1.0;
                    s.pixel_saturation = 5000.0;
                }
                2 => {
                    p.breakpoint = 1.2;
                    s.exposure = -0.25;
                }
                _ => {
                    p.scale = 0.5;
                    s.exposure = 0.5;
                }
            }
            let lut = generate_tone_lut(p, s, Control::none()).unwrap();
            assert_eq!(hash(&lut), expected, "native edge vector {i}");
        }
    }

    #[test]
    fn signed_wrap_and_q12_output() {
        let luts = RenderLuts::new(
            params(),
            settings(),
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            11,
            Control::none(),
        )
        .unwrap();
        assert_eq!(
            luts.render([-2.0, -1.0, 1.0]),
            [
                luts.gamma[luts.tone[65535] as usize],
                0,
                luts.gamma[luts.tone[1] as usize]
            ]
        );
        assert_eq!(
            luts.render([32768.0, 65536.0, 0.0]),
            [luts.gamma[luts.tone[32768] as usize], 0, 0]
        );
        let identity: Vec<u16> = (0..65536).map(|v| v as u16).collect();
        let luts = RenderLuts {
            tone: identity.clone(),
            gamma: identity,
            fixed: [-4095, 8192, 0, 0, 4096, 0, 0, 0, 4096],
        };
        assert_eq!(luts.render([100.0, 30.0, 500.0]), [0, 30, 500]);
    }

    #[test]
    fn output_matches_native_rgb48_rows() {
        let matrices = [
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            [-1.0, 2.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 1.0],
            [30000.0, -30000.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        ];
        let expected = [0x2c7f4ef248577150, 0xf513cdbdbfcfdf00, 0x2175f888181854ef];
        for (matrix, expected) in matrices.into_iter().zip(expected) {
            let luts = RenderLuts::new(params(), settings(), matrix, 11, Control::none()).unwrap();
            let mut output = Vec::new();
            for i in 0..258 {
                let working = match i {
                    256 => [-2.0, -1.0, 32768.0],
                    257 => [65536.0, 1.0, 4094.0],
                    _ => [
                        (i % 33 - 2) as f32 * 128.25,
                        (i % 47) as f32 * 96.5,
                        (i % 71 - 4) as f32 * 64.25,
                    ],
                };
                output.extend(luts.render(working));
            }
            assert_eq!(hash(&output), expected);
        }
    }

    #[test]
    fn invalid_and_cancelled_inputs_fail_before_expensive_work() {
        let cancel = std::sync::atomic::AtomicBool::new(true);
        assert!(matches!(
            generate_tone_lut(params(), settings(), Control::new(&cancel)),
            Err(Error::Cancelled)
        ));
        for saturation in [f64::NAN, f64::INFINITY, 0.0, 65535.0] {
            assert!(generate_tone_lut(
                params(),
                ToneSettings {
                    pixel_saturation: saturation,
                    ..settings()
                },
                Control::none()
            )
            .is_err());
        }
        let fallback = ToneParams::from_color_mode_rows(&[], 0);
        assert_eq!(fallback.end, 1.55);
        assert_eq!(
            ToneParams::from_color_mode_rows(&[[8.0; 7]], 0).steepness2,
            1.85
        );
    }
}
