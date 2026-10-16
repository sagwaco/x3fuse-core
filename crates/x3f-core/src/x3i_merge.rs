//! Merge expanded X3I exposures in reference-frame camera units.
//!
//! Phase 1 does not exclude bad pixels because MergeFrame has no bad-pixel mask.
//! The expansion quantises and clamps BMT to u16. It uses Y = min(4T, 65535)
//! and floors the intermediate bias at zero. The dark roll-off is the only
//! mitigation for these limits in this phase.

use rayon::prelude::*;
use x3f_sys::{Control, MergeFrame};

use crate::{Error, Image, ImageLevels, ProcessOptions};

/// Research controls for the phase 1 exposure merge.
#[derive(Clone, Debug)]
pub struct MergeOptions {
    /// Processing settings for frame preparation and DNG output.
    pub process: ProcessOptions,
    /// Normalised raw value at the saturation roll-off midpoint.
    pub saturation_knee: f32,
    /// Width of the logistic saturation roll-off.
    pub saturation_width: f32,
    /// Reject raw samples at or above one minus this margin.
    pub validity_margin: f32,
    /// Read-noise multiplier for the quadratic near-black roll-off.
    pub dark_sigma: f32,
    /// Placeholder coefficients. Phase 2 will fit these values from frame pairs.
    pub noise_k: [f32; 3],
    /// Quadratic raw-noise coefficients per plane.
    pub noise_q: [f32; 3],
    /// Override the measured read noise in all planes, in normalised raw units.
    pub noise_sigma: Option<f32>,
    /// Minimum accumulated weight before a fallback. The default is 1e-3.
    pub w_min: f32,
    /// Optional safety limit in reference units. None preserves all headroom.
    pub max_s: Option<f32>,
}

impl Default for MergeOptions {
    fn default() -> Self {
        Self {
            process: ProcessOptions::default(),
            saturation_knee: 0.9,
            saturation_width: 0.03,
            validity_margin: 0.02,
            dark_sigma: 3.0,
            noise_k: [4e-5; 3],
            noise_q: [0.0; 3],
            noise_sigma: None,
            w_min: 1e-3,
            max_s: None,
        }
    }
}

impl MergeOptions {
    /// Read research overrides once. The merge does not read environment variables.
    pub fn from_env() -> Result<Self, Error> {
        Self::parse_env(
            std::env::var("X3F_X3I_NOISE").ok().as_deref(),
            std::env::var("X3F_X3I_MAX_S").ok().as_deref(),
        )
    }

    fn parse_env(noise: Option<&str>, max_s: Option<&str>) -> Result<Self, Error> {
        let mut options = Self::default();
        let parse = |value: &str| {
            value
                .trim()
                .parse::<f32>()
                .map_err(|_| Error::InvalidData("invalid X3I noise or scale override".into()))
        };
        if let Some(noise) = noise {
            let parts: Vec<_> = noise.split(',').collect();
            let [sigma, k, q] = parts.as_slice() else {
                return Err(Error::InvalidData(
                    "X3F_X3I_NOISE requires sigma,k,q".into(),
                ));
            };
            options.noise_sigma = if sigma.trim().is_empty() {
                None
            } else {
                Some(parse(sigma)?)
            };
            options.noise_k = [parse(k)?; 3];
            options.noise_q = [parse(q)?; 3];
        }
        options.max_s = max_s.map(parse).transpose()?;
        options.validate()?;
        Ok(options)
    }

    fn validate(&self) -> Result<(), Error> {
        if !self.saturation_knee.is_finite()
            || !self.saturation_width.is_finite()
            || self.saturation_width <= 0.0
            || !self.validity_margin.is_finite()
            || !(0.0..1.0).contains(&self.validity_margin)
            || !self.dark_sigma.is_finite()
            || self.dark_sigma < 0.0
            || !self.w_min.is_finite()
            || self.w_min <= 0.0
            || self
                .noise_k
                .iter()
                .chain(&self.noise_q)
                .any(|v| !v.is_finite() || *v < 0.0)
            || self.noise_sigma.is_some_and(|v| !v.is_finite() || v < 0.0)
            || self.max_s.is_some_and(|v| !v.is_finite() || v < 1.0)
        {
            return Err(Error::InvalidData("invalid X3I merge options".into()));
        }
        if self.process.spp_denoise {
            return Err(Error::InvalidData(
                "SPP denoise does not support DNG output".into(),
            ));
        }
        Ok(())
    }
}

/// Exposure scales and sample statistics from one merge.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MergeReport {
    /// Number of loaded and accumulated frames.
    pub frames_used: usize,
    /// Reference exposure divided by each frame exposure, in load order.
    pub scales: Vec<f64>,
    /// Source frame indices, in load order.
    pub frame_indices: Vec<usize>,
    /// CAMF CaptureShutter exposure times in microseconds, in load order.
    pub exposure_us: Vec<u32>,
    /// Fallback counts are sample counts, not pixel counts.
    pub fallback_best_snr: usize,
    /// Samples taken from the shortest exposure because all frames are invalid.
    pub fallback_shortest: usize,
    /// Samples with no available fallback. Valid input frames make this zero.
    pub fallback_zero: usize,
    /// Samples clipped by the optional safety limit.
    pub clipped_samples: usize,
    /// Linear encoding scale. This is at least one and covers the raster maximum.
    pub s: f64,
    /// Minimum merged sample after the optional safety clip.
    pub minimum: f32,
    /// Maximum merged sample after the optional safety clip.
    pub maximum: f32,
}

impl MergeReport {
    /// Describe fallback and safety clipping counts for a conversion report.
    pub fn warnings(&self) -> Vec<String> {
        let mut warnings = Vec::new();
        if self.fallback_best_snr + self.fallback_shortest + self.fallback_zero > 0 {
            warnings.push(format!(
                "X3I fallback samples: best SNR {}, shortest exposure {}, zero {}",
                self.fallback_best_snr, self.fallback_shortest, self.fallback_zero
            ));
        }
        if self.clipped_samples > 0 {
            warnings.push(format!(
                "X3I safety limit clipped {} samples",
                self.clipped_samples
            ));
        }
        warnings
    }
}

/// Extended-range camera raster in reference-frame units.
#[derive(Clone, Debug)]
pub struct MergedScene {
    /// Three camera samples per pixel, row-major and interleaved.
    pub bmt: Vec<f32>,
    /// Expanded raster width in pixels.
    pub width: u32,
    /// Expanded raster height in pixels.
    pub height: u32,
    /// Linear encoding scale. This is at least one and covers the raster maximum.
    pub s: f64,
    /// Reference-frame camera-to-RGB matrix, row-major.
    pub camera_to_rgb: [f64; 9],
    /// Exposure and range statistics for this raster.
    pub report: MergeReport,
}

fn validate_frame(frame: &MergeFrame, control: Control<'_>) -> Result<(), Error> {
    let pixels = (frame.width as usize).checked_mul(frame.height as usize);
    let samples = pixels.and_then(|n| n.checked_mul(3));
    if frame.width == 0
        || frame.height == 0
        || frame.raw_half_width.checked_mul(2) != Some(frame.width)
        || frame.raw_half_height.checked_mul(2) != Some(frame.height)
        || pixels != Some(frame.raw_top.len())
        || samples != Some(frame.bmt.len())
        || pixels.map(|n| n / 2) != Some(frame.raw_half.len())
        || frame
            .gain
            .as_ref()
            .is_some_and(|g| Some(g.len()) != samples)
        || frame.shutter_us == 0
        || (0..3).any(|c| {
            !frame.raw_black[c].is_finite()
                || !frame.raw_white[c].is_finite()
                || frame.raw_white[c] <= frame.raw_black[c]
                || !frame.sigma[c].is_finite()
                || frame.sigma[c] < 0.0
                || !normalise(0, frame, c).is_finite()
                || !normalise(u16::MAX, frame, c).is_finite()
        })
        || frame.camera_to_rgb.iter().any(|v| !v.is_finite())
    {
        return Err(Error::InvalidData(
            "invalid X3I frame layout or calibration".into(),
        ));
    }
    for row in frame.bmt.chunks(frame.width as usize * 3) {
        control.check()?;
        if row.iter().any(|v| !v.is_finite()) {
            return Err(Error::InvalidData("non-finite X3I frame sample".into()));
        }
    }
    if let Some(gain) = &frame.gain {
        for row in gain.chunks(frame.width as usize * 3) {
            control.check()?;
            if row.iter().any(|v| !v.is_finite() || *v <= 0.0) {
                return Err(Error::InvalidData("invalid X3I spatial gain".into()));
            }
        }
    }
    Ok(())
}

fn normalise(value: u16, frame: &MergeFrame, channel: usize) -> f32 {
    ((value as f64 - frame.raw_black[channel])
        / (frame.raw_white[channel] - frame.raw_black[channel])) as f32
}

fn taps(output: usize, size: usize) -> [usize; 4] {
    // Quattro uses floor((o + 0.5) / 2 - 0.5), four taps and BORDER_REFLECT_101.
    let centre = (output as isize - 1).div_euclid(2);
    std::array::from_fn(|k| {
        if size == 1 {
            return 0;
        }
        let period = 2 * (size as isize - 1);
        let reflected = (centre + k as isize - 1).rem_euclid(period);
        reflected.min(period - reflected) as usize
    })
}

/// Maximum normalised raw value over the exact expansion footprint.
fn footprint_max(frame: &MergeFrame, control: Control<'_>) -> Result<Vec<f32>, Error> {
    let width = frame.width as usize;
    let half_width = frame.raw_half_width as usize;
    let mut horizontal = vec![0.0f32; half_width * frame.raw_half_height as usize * 2];
    horizontal
        .par_chunks_mut(width)
        .enumerate()
        .try_for_each(|(y, row)| -> Result<(), Error> {
            control.check()?;
            for (x, value) in row.iter_mut().enumerate() {
                *value = f32::NEG_INFINITY;
                for sx in taps(x, half_width) {
                    for c in 0..2 {
                        *value = value.max(normalise(
                            frame.raw_half[(y * half_width + sx) * 2 + c],
                            frame,
                            c,
                        ));
                    }
                    // The half-resolution T value sums this top-layer 2x2 block.
                    for ty in y * 2..y * 2 + 2 {
                        for tx in sx * 2..sx * 2 + 2 {
                            *value = value.max(normalise(frame.raw_top[ty * width + tx], frame, 2));
                        }
                    }
                }
            }
            Ok(())
        })?;
    let mut output = vec![0.0; frame.raw_top.len()];
    output
        .par_chunks_mut(width)
        .enumerate()
        .try_for_each(|(y, row)| -> Result<(), Error> {
            control.check()?;
            for (x, value) in row.iter_mut().enumerate() {
                *value = normalise(frame.raw_top[y * width + x], frame, 2);
                for sy in taps(y, frame.raw_half_height as usize) {
                    *value = value.max(horizontal[sy * width + x]);
                }
            }
            Ok(())
        })?;
    Ok(output)
}

/// Erode validity and extend saturation maxima over the same 3x3 window.
fn saturation_max(frame: &MergeFrame, control: Control<'_>) -> Result<Vec<f32>, Error> {
    let footprint = footprint_max(frame, control)?;
    let width = frame.width as usize;
    let height = frame.height as usize;
    let mut output = vec![0.0; footprint.len()];
    output
        .par_chunks_mut(width)
        .enumerate()
        .try_for_each(|(y, row)| -> Result<(), Error> {
            control.check()?;
            for (x, value) in row.iter_mut().enumerate() {
                *value = f32::NEG_INFINITY;
                // Clip the erosion window at the raster edge. Expansion taps reflect separately.
                for sy in y.saturating_sub(1)..=(y + 1).min(height - 1) {
                    for sx in x.saturating_sub(1)..=(x + 1).min(width - 1) {
                        *value = value.max(footprint[sy * width + sx]);
                    }
                }
            }
            Ok(())
        })?;
    Ok(output)
}

/// Load and merge one exposure at a time. Frame zero defines reference units.
pub fn merge(
    frame_count: usize,
    mut load: impl FnMut(usize) -> Result<MergeFrame, Error>,
    options: &MergeOptions,
    control: Control<'_>,
) -> Result<MergedScene, Error> {
    control.check()?;
    options.validate()?;
    if frame_count == 0 {
        return Err(Error::InvalidData(
            "X3I merge requires a reference frame".into(),
        ));
    }
    let reference = load(0)?;
    validate_frame(&reference, control)?;
    let (width, height) = (reference.width, reference.height);
    let camera_to_rgb = reference.camera_to_rgb;
    let reference_us = reference.shutter_us;
    let pixels = reference.raw_top.len();
    let mut sum = vec![0.0f32; pixels * 3];
    let mut weights = vec![[0.0f32; 2]; pixels];
    // shortcut: five fallback planes increase peak memory, use row-band preparation for large rasters.
    let mut fallback = vec![[0.0f32; 3]; pixels];
    let mut best_snr = vec![[-1.0f32; 2]; pixels];
    let mut shortest_us = u32::MAX;
    let mut report = MergeReport::default();
    let mut first = Some(reference);
    for i in 0..frame_count {
        control.check()?;
        let frame = if i == 0 {
            first.take().unwrap()
        } else {
            load(i)?
        };
        validate_frame(&frame, control)?;
        if (frame.width, frame.height) != (width, height) {
            return Err(Error::InvalidData(
                "X3I frame dimensions differ from the reference".into(),
            ));
        }
        let scale = reference_us as f32 / frame.shutter_us as f32;
        let d_max = saturation_max(&frame, control)?;
        let sigma = options.noise_sigma.map_or(frame.sigma, |s| [s; 3]);
        let shortest = i == 0 || frame.shutter_us < shortest_us;
        sum.par_chunks_mut(width as usize * 3)
            .zip(weights.par_chunks_mut(width as usize))
            .zip(fallback.par_chunks_mut(width as usize))
            .zip(best_snr.par_chunks_mut(width as usize))
            .enumerate()
            .try_for_each(
                |(y, (((sum, weights), fallback), best_snr))| -> Result<(), Error> {
                    control.check()?;
                    for x in 0..width as usize {
                        let p = y * width as usize + x;
                        let half = (y / 2 * frame.raw_half_width as usize + x / 2) * 2;
                        let r = [
                            normalise(frame.raw_half[half], &frame, 0),
                            normalise(frame.raw_half[half + 1], &frame, 1),
                            normalise(frame.raw_top[p], &frame, 2),
                        ];
                        let variance: [f32; 3] = std::array::from_fn(|c| {
                            (sigma[c] * sigma[c]
                                + options.noise_k[c] * r[c]
                                + (options.noise_q[c] * r[c]).powi(2))
                            .max(1e-12)
                        });
                        let gain = frame
                            .gain
                            .as_ref()
                            .map_or([1.0; 3], |g| [g[p * 3], g[p * 3 + 1], g[p * 3 + 2]]);
                        // Sum the gained plane variances to keep one shared chroma weight.
                        let var = [
                            scale.powi(2)
                                * (gain[0].powi(2) * variance[0] + gain[1].powi(2) * variance[1]),
                            scale.powi(2) * gain[2].powi(2) * variance[2],
                        ];
                        if var.iter().any(|v| !v.is_finite() || *v <= 0.0) {
                            return Err(Error::InvalidData("non-finite X3I variance".into()));
                        }
                        let signal = [(r[0] + r[1]) * 0.5, r[2]];
                        let expected_signal = [
                            ((gain[0] * r[0].max(0.0)).powi(2) + (gain[1] * r[1].max(0.0)).powi(2))
                                * 0.5,
                            (gain[2] * r[2].max(0.0)).powi(2),
                        ];
                        let read_noise =
                            [(sigma[0].powi(2) + sigma[1].powi(2)).sqrt() * 0.5, sigma[2]];
                        let valid = d_max[p] < 1.0 - options.validity_margin;
                        let sat = 1.0
                            / (1.0
                                + ((d_max[p] - options.saturation_knee)
                                    / options.saturation_width)
                                    .exp());
                        for group in 0..2 {
                            let dark_limit = options.dark_sigma * read_noise[group];
                            let dark = if dark_limit > 0.0 {
                                (signal[group] / dark_limit).clamp(0.0, 1.0).powi(2)
                            } else {
                                1.0
                            };
                            let mut w = if valid { sat * dark / var[group] } else { 0.0 };
                            if i == 0 && valid {
                                w = w.max(1e-3);
                            }
                            let expected = scale.powi(2) * expected_signal[group] / var[group];
                            let better = valid && expected > best_snr[x][group];
                            for c in if group == 0 { 0..2 } else { 2..3 } {
                                let value = scale * frame.bmt[p * 3 + c];
                                sum[x * 3 + c] += w * value;
                                if better || (shortest && best_snr[x][group] < 0.0) {
                                    fallback[x][c] = value;
                                }
                            }
                            if better {
                                best_snr[x][group] = expected;
                            }
                            weights[x][group] += w;
                        }
                    }
                    Ok(())
                },
            )?;
        shortest_us = shortest_us.min(frame.shutter_us);
        report.frames_used += 1;
        report.frame_indices.push(frame.frame_index);
        report
            .scales
            .push(reference_us as f64 / frame.shutter_us as f64);
        report.exposure_us.push(frame.shutter_us);
    }
    report.minimum = f32::INFINITY;
    report.maximum = f32::NEG_INFINITY;
    for (y, row) in sum.chunks_mut(width as usize * 3).enumerate() {
        control.check()?;
        for (j, value) in row.iter_mut().enumerate() {
            let p = y * width as usize + j / 3;
            let c = j % 3;
            let group = usize::from(c == 2);
            let weight = weights[p][group];
            if !weight.is_finite() || !value.is_finite() {
                return Err(Error::InvalidData("non-finite X3I accumulation".into()));
            }
            if weight >= options.w_min {
                *value /= weight;
            } else if best_snr[p][group] >= 0.0 {
                *value = fallback[p][c];
                report.fallback_best_snr += 1;
            } else if report.frames_used > 0 {
                *value = fallback[p][c];
                report.fallback_shortest += 1;
            } else {
                *value = 0.0;
                report.fallback_zero += 1;
            }
            if !value.is_finite() {
                return Err(Error::InvalidData("non-finite X3I merged sample".into()));
            }
            if let Some(limit) = options.max_s {
                if *value > limit {
                    *value = limit;
                    report.clipped_samples += 1;
                }
            }
            report.minimum = report.minimum.min(*value);
            report.maximum = report.maximum.max(*value);
        }
    }
    report.s = f64::from(report.maximum).max(1.0);
    Ok(MergedScene {
        bmt: sum,
        width,
        height,
        s: report.s,
        camera_to_rgb,
        report,
    })
}

/// Encode camera samples as unsigned linear DNG samples. Negative values become zero.
pub fn to_image(scene: &MergedScene) -> Result<Image, Error> {
    let stride = scene.width.checked_mul(3);
    if scene.width == 0
        || scene.height == 0
        || stride.is_none()
        || (scene.width as usize)
            .checked_mul(scene.height as usize)
            .and_then(|n| n.checked_mul(3))
            != Some(scene.bmt.len())
        || !scene.s.is_finite()
        || scene.s < 1.0
        || scene
            .bmt
            .iter()
            .any(|v| !v.is_finite() || f64::from(*v) > scene.s)
    {
        return Err(Error::InvalidData("invalid X3I merged raster".into()));
    }
    Ok(Image {
        data: scene
            .bmt
            .iter()
            .map(|v| {
                (f64::from(*v) / scene.s * 65535.0)
                    .round()
                    .clamp(0.0, 65535.0) as u16
            })
            .collect(),
        rows: scene.height,
        columns: scene.width,
        channels: 3,
        row_stride: stride.unwrap(),
        levels: ImageLevels {
            black: [0.0; 3],
            white: [65535; 3],
        },
        dng_highlight_scale: scene.s,
        dng_shoulder_ceiling: 1.0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicBool;

    fn frame(exposure: u32, truth: f32) -> MergeFrame {
        let value = (truth * exposure as f32 / 1000.0).min(1.0);
        MergeFrame {
            shutter_us: exposure,
            width: 16,
            height: 16,
            bmt: vec![value; 16 * 16 * 3],
            gain: None,
            raw_top: vec![(value * 10000.0).round() as u16; 16 * 16],
            raw_half: vec![(value * 10000.0).round() as u16; 8 * 8 * 2],
            raw_half_width: 8,
            raw_half_height: 8,
            raw_black: [0.0; 3],
            raw_white: [10000.0; 3],
            sigma: [0.001; 3],
            camera_to_rgb: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            ..MergeFrame::default()
        }
    }

    #[test]
    fn two_and_seven_exposures_preserve_headroom_and_constant_scenes() {
        for exposures in [
            &[1000, 125][..],
            &[1000, 125, 250, 500, 2000, 4000, 8000][..],
        ] {
            for truth in [0.25, 4.0] {
                let mut loaded = Vec::new();
                let scene = merge(
                    exposures.len(),
                    |i| {
                        loaded.push(i);
                        let mut f = frame(exposures[i], truth);
                        if truth < 1.0 {
                            for (j, value) in f.bmt.iter_mut().enumerate() {
                                *value += if j % 2 == 0 { 1e-5 } else { -1e-5 };
                            }
                        }
                        Ok(f)
                    },
                    &MergeOptions::default(),
                    Control::none(),
                )
                .unwrap();
                assert_eq!(loaded, (0..exposures.len()).collect::<Vec<_>>());
                assert!(scene.bmt.iter().all(|v| (*v - truth).abs() < 0.0001));
                assert!((scene.s - f64::from(truth.max(1.0))).abs() < 0.0001);
                assert_eq!(scene.report.frames_used, exposures.len());
                assert_eq!(scene.report.scales[1], 8.0);
                assert!(scene.report.warnings().is_empty());
                let image = to_image(&scene).unwrap();
                assert_eq!(image.levels.white, [65535; 3]);
                assert_eq!(image.row_stride, 48);
                assert_eq!(image.dng_shoulder_ceiling, 1.0);
                for (code, value) in image.data.iter().zip(&scene.bmt) {
                    assert!(
                        (f64::from(*code) / 65535.0 * scene.s - f64::from(*value)).abs()
                            <= scene.s / 65535.0
                    );
                }
            }
        }
    }

    #[test]
    fn fallbacks_use_best_snr_then_shortest_and_count_samples() {
        let options = MergeOptions {
            w_min: f32::MAX,
            ..MergeOptions::default()
        };
        let scene = merge(
            2,
            |i| {
                let mut f = frame(if i == 0 { 1000 } else { 2000 }, 0.25);
                f.bmt.fill(if i == 0 { 0.2 } else { 0.6 });
                Ok(f)
            },
            &options,
            Control::none(),
        )
        .unwrap();
        assert!(scene.bmt.iter().all(|v| (*v - 0.3).abs() < 1e-6));
        assert_eq!(scene.report.fallback_best_snr, 768);
        assert_eq!(scene.report.fallback_shortest, 0);
        let clipped = merge(
            3,
            |i| Ok(frame([1000, 125, 2000][i], 16.0)),
            &MergeOptions::default(),
            Control::none(),
        )
        .unwrap();
        assert_eq!(clipped.bmt, vec![8.0; 768]);
        assert_eq!(clipped.report.fallback_shortest, 768);
        assert_eq!(clipped.report.fallback_zero, 0);
        assert_eq!(clipped.report.warnings().len(), 1);
    }

    #[test]
    fn weights_use_measured_noise_gain_and_sigma_override() {
        for sigma_override in [None, Some(0.01)] {
            let options = MergeOptions {
                noise_k: [0.0; 3],
                noise_sigma: sigma_override,
                ..MergeOptions::default()
            };
            let scene = merge(
                2,
                |i| {
                    let mut f = frame(1000, 0.25);
                    f.sigma = [if i == 0 { 0.01 } else { 0.02 }; 3];
                    f.gain = Some(vec![if i == 0 { 1.0 } else { 2.0 }; 768]);
                    f.bmt.fill(if i == 0 { 0.2 } else { 0.6 });
                    f.frame_index = i + 10;
                    Ok(f)
                },
                &options,
                Control::none(),
            )
            .unwrap();
            let ratio = if sigma_override.is_some() { 4.0 } else { 16.0 };
            let expected = (0.2 * ratio + 0.6) / (ratio + 1.0);
            assert!(scene.bmt.iter().all(|v| (*v - expected).abs() < 1e-6));
            assert_eq!(scene.report.frame_indices, [10, 11]);
        }
    }

    #[test]
    fn chroma_variance_uses_each_planes_gain() {
        let options = MergeOptions {
            noise_k: [0.0; 3],
            ..MergeOptions::default()
        };
        let scene = merge(
            2,
            |i| {
                let mut f = frame(1000, 0.25);
                f.sigma = if i == 0 {
                    [0.01, 0.02, 0.01]
                } else {
                    [0.02, 0.01, 0.02]
                };
                let gain = if i == 0 {
                    [2.0, 1.0, 1.0]
                } else {
                    [1.0, 3.0, 1.0]
                };
                f.gain = Some(gain.repeat(256));
                f.bmt.fill(if i == 0 { 0.2 } else { 0.6 });
                Ok(f)
            },
            &options,
            Control::none(),
        )
        .unwrap();
        let chroma = (0.2 * 1.625 + 0.6) / 2.625;
        for pixel in scene.bmt.chunks(3) {
            assert!((pixel[0] - chroma).abs() < 1e-6);
            assert_eq!(pixel[0], pixel[1]);
            assert!((pixel[2] - 0.28).abs() < 1e-6);
        }
    }

    #[test]
    fn raw_noise_floor_scales_with_exposure() {
        let options = MergeOptions {
            noise_k: [0.0; 3],
            noise_sigma: Some(0.0),
            ..MergeOptions::default()
        };
        let scene = merge(
            2,
            |i| {
                let mut f = frame([1000, 2000][i], 0.25);
                f.bmt.fill(if i == 0 { 0.2 } else { 0.6 });
                Ok(f)
            },
            &options,
            Control::none(),
        )
        .unwrap();
        assert!(scene.bmt.iter().all(|v| (*v - 0.28).abs() < 1e-6));
    }

    #[test]
    fn safety_clip_is_optional_and_reported() {
        let options = MergeOptions {
            max_s: Some(2.0),
            ..MergeOptions::default()
        };
        let scene = merge(
            2,
            |i| Ok(frame([1000, 125][i], 4.0)),
            &options,
            Control::none(),
        )
        .unwrap();
        assert_eq!(scene.bmt, vec![2.0; 768]);
        assert_eq!(scene.s, 2.0);
        assert_eq!(scene.report.clipped_samples, 768);
        assert_eq!(scene.report.minimum, 2.0);
    }

    #[test]
    fn footprint_and_erosion_follow_bicubic_taps_and_reflected_borders() {
        for (source, low, high) in [(4, 5, 12), (7, 11, 15), (0, 0, 4)] {
            let mut f = frame(1000, 0.25);
            f.raw_half[(source * 8 + source) * 2] = 10000;
            let footprint = footprint_max(&f, Control::none()).unwrap();
            let eroded = saturation_max(&f, Control::none()).unwrap();
            for y in 0..16 {
                for x in 0..16 {
                    assert_eq!(
                        footprint[y * 16 + x] >= 0.98,
                        (low..=high).contains(&x) && (low..=high).contains(&y),
                        "source {source}, ({x},{y})"
                    );
                    let range = low.saturating_sub(1)..=(high + 1).min(15);
                    assert_eq!(
                        eroded[y * 16 + x] >= 0.98,
                        range.contains(&x) && range.contains(&y)
                    );
                }
            }
        }
        let mut f = frame(1000, 0.25);
        f.raw_top[8 * 16 + 8] = 10000;
        let eroded = saturation_max(&f, Control::none()).unwrap();
        let mut half_invalid = frame(1000, 0.25);
        half_invalid.raw_half[(4 * 8 + 4) * 2] = 10000;
        assert_eq!(
            eroded,
            saturation_max(&half_invalid, Control::none()).unwrap()
        );
    }

    #[test]
    fn noise_overrides_parse_without_mutating_the_environment() {
        let options = MergeOptions::parse_env(Some("0.002, 0.0001, 0.01"), Some("8")).unwrap();
        assert_eq!(options.noise_sigma, Some(0.002));
        assert_eq!(options.noise_k, [0.0001; 3]);
        assert_eq!(options.noise_q, [0.01; 3]);
        let measured = MergeOptions::parse_env(Some(",0.0001,0"), None).unwrap();
        assert_eq!(measured.noise_sigma, None);
        assert_eq!(options.max_s, Some(8.0));
        for value in ["1,2", "1,2,3,4", "nan,0,0", "-1,0,0", "0,-1,0", "x,0,0"] {
            assert!(MergeOptions::parse_env(Some(value), None).is_err());
        }
        for value in ["nan", "inf", "0.5", "x"] {
            assert!(MergeOptions::parse_env(None, Some(value)).is_err());
        }
    }

    #[test]
    fn invalid_frames_options_and_cancellation_fail() {
        let options = MergeOptions::default();
        assert!(merge(0, |_| unreachable!(), &options, Control::none()).is_err());
        for field in 0..7 {
            assert!(merge(
                1,
                |_| {
                    let mut f = frame(1000, 0.25);
                    match field {
                        0 => f.shutter_us = 0,
                        1 => f.raw_top.clear(),
                        2 => f.bmt[0] = f32::NAN,
                        3 => f.raw_white[0] = 0.0,
                        4 => f.gain = Some(vec![0.0; 768]),
                        5 => f.sigma[0] = -1.0,
                        _ => f.width = 0,
                    }
                    Ok(f)
                },
                &options,
                Control::none()
            )
            .is_err());
        }
        assert!(merge(
            2,
            |i| {
                let mut f = frame(1000, 0.25);
                if i == 1 {
                    f.width = 2;
                    f.height = 2;
                    f.raw_half_width = 1;
                    f.raw_half_height = 1;
                    f.bmt.truncate(12);
                    f.raw_top.truncate(4);
                    f.raw_half.truncate(2);
                }
                Ok(f)
            },
            &options,
            Control::none()
        )
        .is_err());
        let cancelled = AtomicBool::new(true);
        assert!(matches!(
            merge(1, |_| unreachable!(), &options, Control::new(&cancelled)),
            Err(Error::Cancelled)
        ));
        let bad_options = MergeOptions {
            process: ProcessOptions {
                spp_denoise: true,
                ..ProcessOptions::default()
            },
            ..options
        };
        assert!(merge(1, |_| unreachable!(), &bad_options, Control::none()).is_err());
    }

    #[test]
    fn reference_floor_preserves_zero_signal_and_channel_weights_are_shared() {
        let scene = merge(
            1,
            |_| Ok(frame(1000, 0.0)),
            &MergeOptions::default(),
            Control::none(),
        )
        .unwrap();
        assert_eq!(scene.bmt, vec![0.0; 768]);
        assert_eq!(scene.report.fallback_best_snr, 0);
        assert_eq!(scene.s, 1.0);
        let scene = merge(
            2,
            |i| {
                let mut f = frame([1000, 2000][i], 0.2);
                f.gain = Some(vec![2.0; 768]);
                for value in &mut f.bmt {
                    *value *= 2.0;
                }
                for pixel in f.bmt.chunks_mut(3) {
                    pixel[1] *= 3.0;
                }
                Ok(f)
            },
            &MergeOptions::default(),
            Control::none(),
        )
        .unwrap();
        assert!(scene
            .bmt
            .chunks(3)
            .all(|p| (p[1] - p[0] * 3.0).abs() < 1e-6));
    }
}
