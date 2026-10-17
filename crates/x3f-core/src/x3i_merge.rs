//! Merge expanded X3I exposures in reference-frame camera units.
//!
//! With fix_bad enabled, the producer repairs samples before the footprint
//! reads them. The snapshot uses the intermediate black and white levels.
//! With the real producer, r is bounded below by minus the intermediate bias
//! fraction, so zero sigma cannot produce a negative variance.
//! Noise uses r = (sample - raw_black) / (raw_white - raw_black).
//! Validity and saturation roll-off use d = r / saturation[plane].
//! A measured sensor ceiling below the CAMF white changes only d, not noise units.
//! A flat wall in an unsaturated frame can also pass the ceiling pile test.
//! shortcut: no cross-frame ceiling check, phase 2 must require the second-longest
//! frame's ceiling to agree within 2 percent before accepting the estimate.
//! The intermediate encoding clamps at zero, so it truncates negative noise. Expansion
//! also quantises and clamps BMT to u16, including Y = min(4T, 65535).
//! The merge applies no signal-dependent dark weight. Unsigned DNG encoding
//! clamps negative merged samples to zero.
//!
//! The expansion builds its half-resolution top sum from unrepaired samples.
//! A defective top pixel can contaminate chroma in its 8x8 expanded block.
//! Repaired raw_top does not expose this defect to the merge. This existing
//! single-frame pipeline limit remains in phase 1.
//!
//! Buffer budget for N expanded pixels. Producer scratch is additional:
//! - Persistent: 3 weighted-sum, 2 weight, 3 fallback and 2 SNR f32 planes,
//!   or 40N bytes. SNR stays f32 to preserve fallback candidate ordering.
//! - One frame: 3 BMT and up to 3 gain f32 planes, or 24N bytes; N top and
//!   N/2 half-plane u16 samples, or 3N bytes.
//! - Footprint peak: one full and one half f32 plane, or 6N bytes. Erosion
//!   reuses the full plane and one row. Accumulation retains one full plane.
//!
//! Gain is released after accumulation. Integration measures RSS on Quattro H.

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
    /// Physical ceilings in raw-range units. None uses one in all planes.
    pub saturation: Option<[f32; 3]>,
    /// Placeholder coefficients. Phase 2 will fit these values from frame pairs.
    pub noise_k: [f32; 3],
    /// Quadratic raw-noise coefficients per plane.
    pub noise_q: [f32; 3],
    /// Override the measured read noise in all planes, in normalised raw units.
    pub noise_sigma: Option<f32>,
    /// Minimum accumulated weight in inverse-variance units. The default is 1e-3.
    /// Typical weights are 1e4 to 1e6. Fallback requires all weights near zero.
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
            saturation: None,
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
            std::env::var("X3F_X3I_SATURATION").ok().as_deref(),
        )
    }

    /// Use an override or estimate the raw ceilings from the longest exposure.
    /// Preparation finishes before the merge allocates its accumulation planes.
    pub fn saturation_for(
        &self,
        container: &crate::X3iContainer,
        control: Control<'_>,
    ) -> Result<[f32; 3], Error> {
        self.saturation_for_with_warnings(container, control, &mut Vec::new())
    }

    /// Estimate container ceilings with the default processing control.
    pub fn saturation_for_container(
        &self,
        container: &crate::X3iContainer,
    ) -> Result<[f32; 3], Error> {
        self.saturation_for(container, Control::none())
    }

    pub(crate) fn saturation_for_with_warnings(
        &self,
        container: &crate::X3iContainer,
        control: Control<'_>,
        warnings: &mut Vec<String>,
    ) -> Result<[f32; 3], Error> {
        control.check()?;
        self.validate()?;
        if let Some(saturation) = self.saturation {
            return Ok(saturation);
        }
        let mut longest = (0, 0);
        for i in 0..container.frames().len() {
            control.check()?;
            let exposure = container.frame_exposure_us(i)?;
            if exposure > longest.1 {
                longest = (i, exposure);
            }
        }
        let mut options = self.process.clone();
        options.denoise_intensity = 0;
        let frame = container
            .open_frame(longest.0)?
            .merge_frame(&options, control)?;
        estimate_saturation(&frame, longest.0, control, warnings)
    }

    fn parse_env(
        noise: Option<&str>,
        max_s: Option<&str>,
        saturation: Option<&str>,
    ) -> Result<Self, Error> {
        let mut options = Self::default();
        let parse = |value: &str| {
            value.trim().parse::<f32>().map_err(|_| {
                Error::InvalidData("invalid X3I noise, scale or saturation override".into())
            })
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
        if let Some(saturation) = saturation {
            let parts: Vec<_> = saturation.split(',').collect();
            let [b, m, t] = parts.as_slice() else {
                return Err(Error::InvalidData(
                    "X3F_X3I_SATURATION requires b,m,t".into(),
                ));
            };
            options.saturation = Some([parse(b)?, parse(m)?, parse(t)?]);
        }
        options.validate()?;
        Ok(options)
    }

    fn validate(&self) -> Result<(), Error> {
        if !self.saturation_knee.is_finite()
            || !self.saturation_width.is_finite()
            || self.saturation_width <= 0.0
            || !self.validity_margin.is_finite()
            || !(0.0..1.0).contains(&self.validity_margin)
            || !self.w_min.is_finite()
            || self.w_min <= 0.0
            || self
                .noise_k
                .iter()
                .chain(&self.noise_q)
                .any(|v| !v.is_finite() || *v < 0.0)
            || self.saturation.is_some_and(|values| {
                values
                    .iter()
                    .any(|v| !v.is_finite() || *v <= 0.0 || *v > 1.0)
            })
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

fn estimate_saturation(
    frame: &MergeFrame,
    index: usize,
    control: Control<'_>,
    warnings: &mut Vec<String>,
) -> Result<[f32; 3], Error> {
    let mut saturation = [1.0; 3];
    for (c, value) in saturation.iter_mut().enumerate() {
        let (ceiling, suspicious) = if c == 2 {
            estimate_plane_saturation(
                frame.raw_top.iter().copied(),
                frame.raw_black[c],
                frame.raw_white[c],
                control,
            )?
        } else {
            estimate_plane_saturation(
                frame.raw_half.iter().skip(c).step_by(2).copied(),
                frame.raw_black[c],
                frame.raw_white[c],
                control,
            )?
        };
        *value = ceiling;
        if suspicious {
            warnings.push(format!(
                "X3I saturation warning: frame={index} plane={} falls back to 1.0 despite a pile below 0.9 of the maximum; check X3F_X3I_SATURATION",
                ["B", "M", "T"][c]
            ));
        }
    }
    Ok(saturation)
}

fn estimate_plane_saturation(
    values: impl Iterator<Item = u16>,
    black: f64,
    white: f64,
    control: Control<'_>,
) -> Result<(f32, bool), Error> {
    let mut histogram = vec![0usize; usize::from(u16::MAX) + 1];
    let mut count = 0usize;
    let mut maximum = 0u16;
    for (i, value) in values.enumerate() {
        if i % 65536 == 0 {
            control.check()?;
        }
        histogram[usize::from(value)] += 1;
        count += 1;
        maximum = maximum.max(value);
    }
    let maximum = usize::from(maximum);
    let mut tail = 0usize;
    let support = count.div_ceil(10000).max(300);
    let Some(mut candidate) = (0..=maximum).rev().find(|&value| {
        tail += histogram[value];
        tail >= support
    }) else {
        return Ok((1.0, false));
    };
    let lower = |upper: usize| {
        (black + 0.99 * (upper as f64 - black))
            .ceil()
            .clamp(0.0, upper as f64) as usize
    };
    let enough = |pile: usize| pile >= 1000 && pile as f64 >= count as f64 * 0.0005;
    // Keep the measured edge when the maximum already has a supported pile.
    if enough(histogram[lower(maximum)..=maximum].iter().sum()) {
        candidate = maximum;
    }
    let ceiling = (candidate as f64 - black) / (white - black);
    if !ceiling.is_finite() || ceiling <= 0.0 {
        return Ok((1.0, false));
    }
    if enough(histogram[lower(candidate)..=candidate].iter().sum()) {
        return Ok((ceiling.clamp(0.3, 1.0) as f32, false));
    }
    // Prefix counts let the fallback diagnostic scan all lower piles once.
    for i in 1..histogram.len() {
        histogram[i] += histogram[i - 1];
    }
    let below = (black + 0.9 * (maximum as f64 - black))
        .ceil()
        .clamp(0.0, maximum as f64) as usize;
    let suspicious = (0..below).any(|upper| {
        let start = lower(upper);
        upper as f64 > black
            && enough(histogram[upper] - start.checked_sub(1).map_or(0, |i| histogram[i]))
    });
    Ok((1.0, suspicious))
}

/// Exposure scales and sample statistics from one merge.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MergeReport {
    /// Number of loaded and accumulated frames.
    pub frames_used: usize,
    /// Physical raw ceilings used by validity and roll-off, in B, M, T order.
    pub saturation: [f32; 3],
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

fn validate_frame(frame: &MergeFrame, control: Control<'_>) -> Result<[f64; 3], Error> {
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
        })
        || frame.camera_to_rgb.iter().any(|v| !v.is_finite())
    {
        return Err(Error::InvalidData(
            "invalid X3I frame layout or calibration".into(),
        ));
    }
    let factors = std::array::from_fn(|c| 1.0 / (frame.raw_white[c] - frame.raw_black[c]));
    if (0..3).any(|c| {
        !factors[c].is_finite()
            || !normalise(0, frame.raw_black[c], factors[c]).is_finite()
            || !normalise(u16::MAX, frame.raw_black[c], factors[c]).is_finite()
    }) {
        return Err(Error::InvalidData("invalid X3I normalisation range".into()));
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
    Ok(factors)
}

fn normalise(value: u16, black: f64, factor: f64) -> f32 {
    ((value as f64 - black) * factor) as f32
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
fn footprint_max(
    frame: &MergeFrame,
    factors: [f64; 3],
    control: Control<'_>,
) -> Result<Vec<f32>, Error> {
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
                    for (c, factor) in factors.iter().enumerate().take(2) {
                        *value = value.max(normalise(
                            frame.raw_half[(y * half_width + sx) * 2 + c],
                            frame.raw_black[c],
                            *factor,
                        ));
                    }
                    // The half-resolution T value sums this top-layer 2x2 block.
                    for ty in y * 2..y * 2 + 2 {
                        for tx in sx * 2..sx * 2 + 2 {
                            *value = value.max(normalise(
                                frame.raw_top[ty * width + tx],
                                frame.raw_black[2],
                                factors[2],
                            ));
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
                *value = normalise(frame.raw_top[y * width + x], frame.raw_black[2], factors[2]);
                for sy in taps(y, frame.raw_half_height as usize) {
                    *value = value.max(horizontal[sy * width + x]);
                }
            }
            Ok(())
        })?;
    Ok(output)
}

/// Erode validity and extend saturation maxima over the same 3x3 window.
fn saturation_max(
    frame: &MergeFrame,
    factors: [f64; 3],
    control: Control<'_>,
) -> Result<Vec<f32>, Error> {
    let mut footprint = footprint_max(frame, factors, control)?;
    let width = frame.width as usize;
    footprint
        .par_chunks_mut(width)
        .try_for_each(|row| -> Result<(), Error> {
            control.check()?;
            let mut left = row[0];
            for x in 0..width {
                let current = row[x];
                row[x] = left.max(current).max(row[(x + 1).min(width - 1)]);
                left = current;
            }
            Ok(())
        })?;
    let mut previous = vec![f32::NEG_INFINITY; width];
    for y in 0..frame.height as usize {
        control.check()?;
        let (processed, remaining) = footprint.split_at_mut((y + 1) * width);
        let row = &mut processed[y * width..];
        for x in 0..width {
            let current = row[x];
            row[x] = previous[x]
                .max(current)
                .max(remaining.get(x).copied().unwrap_or(current));
            previous[x] = current;
        }
    }
    Ok(footprint)
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
    let reference_factors = validate_frame(&reference, control)?;
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
    let mut report = MergeReport {
        saturation: options.saturation.unwrap_or([1.0; 3]),
        ..Default::default()
    };
    let mut first = Some(reference);
    for i in 0..frame_count {
        control.check()?;
        let (mut frame, factors) = if i == 0 {
            (first.take().unwrap(), reference_factors)
        } else {
            let frame = load(i)?;
            let factors = validate_frame(&frame, control)?;
            (frame, factors)
        };
        if (frame.width, frame.height) != (width, height) {
            return Err(Error::InvalidData(
                "X3I frame dimensions differ from the reference".into(),
            ));
        }
        let scale = reference_us as f32 / frame.shutter_us as f32;
        let ceilings = options.saturation.unwrap_or([1.0; 3]);
        let saturation_factors = std::array::from_fn(|c| factors[c] / f64::from(ceilings[c]));
        let d_max = saturation_max(&frame, saturation_factors, control)?;
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
                            normalise(frame.raw_half[half], frame.raw_black[0], factors[0]),
                            normalise(frame.raw_half[half + 1], frame.raw_black[1], factors[1]),
                            normalise(frame.raw_top[p], frame.raw_black[2], factors[2]),
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
                        let expected_signal = [
                            ((gain[0] * r[0].max(0.0)).powi(2) + (gain[1] * r[1].max(0.0)).powi(2))
                                * 0.5,
                            (gain[2] * r[2].max(0.0)).powi(2),
                        ];
                        let valid = d_max[p] < 1.0 - options.validity_margin;
                        let sat = 1.0
                            / (1.0
                                + ((d_max[p] - options.saturation_knee)
                                    / options.saturation_width)
                                    .exp());
                        for group in 0..2 {
                            let mut w = if valid { sat / var[group] } else { 0.0 };
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
        drop(frame.gain.take());
        drop(d_max);
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

/// Exclude unwritten borders from the encoding range. Invalid crops use the full raster.
pub(crate) fn mask_to_active_area(
    scene: &mut MergedScene,
    [top, left, bottom, right]: [u32; 4],
    control: Control<'_>,
) -> Result<(), Error> {
    if bottom <= top || right <= left || bottom > scene.height || right > scene.width {
        return Ok(());
    }
    let mut minimum = f32::INFINITY;
    let mut maximum = f32::NEG_INFINITY;
    for (y, row) in scene.bmt.chunks_mut(scene.width as usize * 3).enumerate() {
        control.check()?;
        if y < top as usize || y >= bottom as usize {
            row.fill(0.0);
        } else {
            row[..left as usize * 3].fill(0.0);
            row[right as usize * 3..].fill(0.0);
        }
        for &value in row.iter() {
            minimum = minimum.min(value);
            maximum = maximum.max(value);
        }
    }
    scene.s = f64::from(maximum).max(1.0);
    scene.report.s = scene.s;
    scene.report.minimum = minimum;
    scene.report.maximum = maximum;
    Ok(())
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
    fn ceiling_estimator_requires_a_pile_and_ignores_an_isolated_hot_pixel() {
        let mut values: Vec<u16> = (0..10000).collect();
        assert_eq!(
            estimate_plane_saturation(values.iter().copied(), 100.0, 10100.0, Control::none())
                .unwrap(),
            (1.0, false)
        );
        let mut isolated = values.clone();
        isolated.push(65535);
        assert_eq!(
            estimate_plane_saturation(isolated.into_iter(), 100.0, 10100.0, Control::none())
                .unwrap(),
            (1.0, false)
        );
        values.extend(std::iter::repeat_n(10000, 5000));
        let (ceiling, suspicious) =
            estimate_plane_saturation(values.iter().copied(), 100.0, 10100.0, Control::none())
                .unwrap();
        assert!((ceiling - 0.99).abs() < 1e-6);
        assert!(!suspicious);
        let pile = std::iter::repeat_n(4600, 5000);
        assert!(
            (estimate_plane_saturation(pile, 100.0, 10100.0, Control::none())
                .unwrap()
                .0
                - 0.45)
                .abs()
                < 1e-6
        );
        let cancelled = AtomicBool::new(true);
        assert!(matches!(
            estimate_plane_saturation([1].into_iter(), 0.0, 10000.0, Control::new(&cancelled)),
            Err(Error::Cancelled)
        ));
    }

    #[test]
    fn ceiling_estimator_keeps_a_plateau_below_an_outlier() {
        let values = std::iter::repeat_n(4500, 5000).chain([9000]);
        assert_eq!(
            estimate_plane_saturation(values, 0.0, 10000.0, Control::none()).unwrap(),
            (0.45, false)
        );
    }

    #[test]
    fn ceiling_estimator_clamps_accepted_low_ceilings_at_point_three() {
        assert_eq!(
            estimate_plane_saturation(
                std::iter::repeat_n(2100, 5000),
                100.0,
                10100.0,
                Control::none()
            )
            .unwrap(),
            (0.3, false)
        );
    }

    #[test]
    fn ceiling_fallback_warns_for_each_plane_with_a_hidden_pile() {
        let values: Vec<u16> = std::iter::repeat_n(4500, 5000)
            .chain(std::iter::repeat_n(9000, 400))
            .collect();
        let frame = MergeFrame {
            raw_half: values.iter().flat_map(|&v| [v; 2]).collect(),
            raw_top: values,
            raw_black: [0.0; 3],
            raw_white: [10000.0; 3],
            ..Default::default()
        };
        let mut warnings = Vec::new();
        assert_eq!(
            estimate_saturation(&frame, 6, Control::none(), &mut warnings).unwrap(),
            [1.0; 3]
        );
        assert_eq!(warnings.len(), 3);
        for (warning, plane) in warnings.iter().zip(["B", "M", "T"]) {
            assert!(warning.starts_with(&format!(
                "X3I saturation warning: frame=6 plane={plane} falls back to 1.0"
            )));
            assert!(warning.contains("pile below 0.9 of the maximum"));
            assert!(warning.contains("X3F_X3I_SATURATION"));
        }
    }

    #[test]
    fn corpus_sensor_ceilings_match_both_camera_families() {
        let corpus = std::env::var_os("X3F_TEST_FILES")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| {
                std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../x3f_test_files")
            });
        for name in ["DP0Q0002.X3I", "_DQH0978.X3I"] {
            let path = corpus.join(name);
            if !path.is_file() {
                eprintln!("skip: missing {name}");
                continue;
            }
            let container = crate::X3iContainer::open(path).unwrap();
            let options = MergeOptions::default();
            let ceilings = options.saturation_for_container(&container).unwrap();
            eprintln!("{name}: saturation={ceilings:?}");
            if name == "DP0Q0002.X3I" {
                assert_eq!(ceilings, [1.0; 3]);
            } else {
                assert_eq!(ceilings, [0.44519424, 0.42977452, 0.5464723]);
            }
            let explicit = MergeOptions {
                saturation: Some([0.4, 0.5, 0.6]),
                ..options
            };
            assert_eq!(
                explicit
                    .saturation_for(&container, Control::none())
                    .unwrap(),
                [0.4, 0.5, 0.6]
            );
        }
    }

    #[test]
    fn a_subwhite_sensor_plateau_does_not_suppress_bright_samples() {
        let load = |i: usize| {
            let mut f = frame([1000, 125, 8000][i], 0.3);
            let value = (0.3 * f.shutter_us as f32 / 1000.0).min(0.45);
            f.bmt.fill(value);
            f.raw_top.fill((value * 10000.0).round() as u16);
            f.raw_half.fill((value * 10000.0).round() as u16);
            Ok(f)
        };
        let options = MergeOptions {
            saturation: Some([0.45; 3]),
            ..Default::default()
        };
        let fixed = merge(3, load, &options, Control::none()).unwrap();
        assert!(fixed.bmt.iter().all(|v| (*v - 0.3).abs() < 1e-5));
        let capped = merge(3, load, &MergeOptions::default(), Control::none()).unwrap();
        assert!(capped.bmt.iter().all(|v| *v < 0.15));
        // One is the old ceiling. It must preserve both signal and noise weighting.
        let explicit_default = merge(
            3,
            load,
            &MergeOptions {
                saturation: Some([1.0; 3]),
                ..Default::default()
            },
            Control::none(),
        )
        .unwrap();
        assert_eq!(capped.bmt, explicit_default.bmt);
    }

    #[test]
    fn masked_border_cannot_set_the_written_encoding_scale() {
        let mut scene = merge(
            1,
            |_| Ok(frame(1000, 2.0)),
            &MergeOptions::default(),
            Control::none(),
        )
        .unwrap();
        scene.bmt.fill(2.0);
        scene.bmt[..16 * 3].fill(100.0);
        scene.s = 100.0;
        mask_to_active_area(&mut scene, [1, 2, 14, 13], Control::none()).unwrap();
        assert_eq!(scene.s, 2.0);
        assert_eq!(scene.report.s, 2.0);
        assert_eq!(scene.report.maximum, 2.0);
        let image = to_image(&scene).unwrap();
        assert!(image.data[..16 * 3].iter().all(|&v| v == 0));
        assert_eq!(image.data[(2 * 16 + 3) * 3], 65535);
        let before = scene.bmt.clone();
        mask_to_active_area(&mut scene, [0, 0, 17, 16], Control::none()).unwrap();
        assert_eq!(scene.bmt, before);
        let cancel = std::sync::atomic::AtomicBool::new(true);
        assert!(matches!(
            mask_to_active_area(&mut scene, [1, 2, 14, 13], Control::new(&cancel)),
            Err(Error::Cancelled)
        ));
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

    fn gaussian_zero_scene_means(options: &MergeOptions) -> ([f64; 3], f64) {
        let exposures = [1000, 125, 250, 500, 2000, 4000, 8000];
        let pixels = 256 * 256;
        let sigma = 0.01;
        let mut seed = 42u64;
        let mut uniform = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((seed >> 11) as f64 + 0.5) / ((1u64 << 53) as f64)
        };
        let scene = merge(
            exposures.len(),
            |i| {
                let mut f = frame(exposures[i], 0.0);
                f.width = 256;
                f.height = 256;
                f.raw_half_width = 128;
                f.raw_half_height = 128;
                f.raw_black = [10000.0; 3];
                f.raw_white = [50000.0; 3];
                f.sigma = [sigma; 3];
                f.raw_top = Vec::with_capacity(pixels);
                f.bmt = Vec::with_capacity(pixels * 3);
                for _ in 0..pixels {
                    let gaussian =
                        (-2.0 * uniform().ln()).sqrt() * (std::f64::consts::TAU * uniform()).cos();
                    let value = gaussian as f32 * sigma;
                    f.raw_top.push((10000.0 + value * 40000.0).round() as u16);
                    // Keep signed floats to isolate weighting bias from the producer's zero clamp.
                    f.bmt.extend([value; 3]);
                }
                f.raw_half = Vec::with_capacity(pixels / 2);
                for y in 0..128 {
                    for x in 0..128 {
                        f.raw_half.extend([f.raw_top[y * 2 * 256 + x * 2]; 2]);
                    }
                }
                Ok(f)
            },
            options,
            Control::none(),
        )
        .unwrap();
        let means = std::array::from_fn(|c| {
            scene
                .bmt
                .as_chunks::<3>()
                .0
                .iter()
                .map(|p| f64::from(p[c]))
                .sum::<f64>()
                / pixels as f64
        });
        let inverse_scale_squared: f64 = exposures
            .iter()
            .map(|us| (*us as f64 / 1000.0).powi(2))
            .sum();
        let standard_error = f64::from(sigma) / (inverse_scale_squared * pixels as f64).sqrt();
        (means, standard_error)
    }

    #[test]
    fn gaussian_zero_scene_has_no_positive_selection_bias() {
        let ([_, _, mean], standard_error) = gaussian_zero_scene_means(&MergeOptions {
            noise_k: [0.0; 3],
            ..MergeOptions::default()
        });
        eprintln!("zero scene: mean {mean}, standard error {standard_error}");
        assert!(
            mean.abs() < 3.0 * standard_error,
            "mean {mean}, standard error {standard_error}"
        );
    }

    #[test]
    fn default_noise_model_zero_scene_has_bounded_negative_bias() {
        let options = MergeOptions::default();
        let (means, standard_error) = gaussian_zero_scene_means(&options);
        let (read_noise_means, _) = gaussian_zero_scene_means(&MergeOptions {
            noise_k: [0.0; 3],
            ..options.clone()
        });
        for c in 0..3 {
            // Sample-dependent variance adds a small negative bias of order k in frame units.
            let bias = means[c] - read_noise_means[c];
            // The largest exposure scale is 8, which converts this bound to reference units.
            let reference_bias_bound = 8.0 * f64::from(options.noise_k[c]);
            eprintln!(
                "default zero scene: channel {c}, mean {}, bias {bias}",
                means[c]
            );
            assert!(bias < 0.0 && bias > -reference_bias_bound);
            assert!(means[c] > -reference_bias_bound - 3.0 * standard_error);
            assert!(means[c] < 3.0 * standard_error);
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
        for (source_x, source_y, x_range, y_range) in [
            (4, 4, 5usize..=12, 5usize..=12),
            (7, 7, 11..=15, 11..=15),
            (0, 0, 0..=4, 0..=4),
            (2, 6, 0..=8, 9..=15),
        ] {
            let mut f = frame(1000, 0.25);
            f.raw_half[(source_y * 8 + source_x) * 2] = 10000;
            let footprint = footprint_max(&f, [0.0001; 3], Control::none()).unwrap();
            let eroded = saturation_max(&f, [0.0001; 3], Control::none()).unwrap();
            let eroded_x = x_range.start().saturating_sub(1)..=(x_range.end() + 1).min(15);
            let eroded_y = y_range.start().saturating_sub(1)..=(y_range.end() + 1).min(15);
            for y in 0..16 {
                for x in 0..16 {
                    assert_eq!(
                        footprint[y * 16 + x] >= 0.98,
                        x_range.contains(&x) && y_range.contains(&y),
                        "source ({source_x},{source_y}), output ({x},{y})"
                    );
                    assert_eq!(
                        eroded[y * 16 + x] >= 0.98,
                        eroded_x.contains(&x) && eroded_y.contains(&y)
                    );
                }
            }
        }
        let mut f = frame(1000, 0.25);
        f.raw_top[8 * 16 + 8] = 10000;
        let eroded = saturation_max(&f, [0.0001; 3], Control::none()).unwrap();
        let mut half_invalid = frame(1000, 0.25);
        half_invalid.raw_half[(4 * 8 + 4) * 2] = 10000;
        assert_eq!(
            eroded,
            saturation_max(&half_invalid, [0.0001; 3], Control::none()).unwrap()
        );
    }

    #[test]
    fn noise_overrides_parse_without_mutating_the_environment() {
        let options =
            MergeOptions::parse_env(Some("0.002, 0.0001, 0.01"), Some("8"), None).unwrap();
        assert_eq!(options.noise_sigma, Some(0.002));
        assert_eq!(options.noise_k, [0.0001; 3]);
        assert_eq!(options.noise_q, [0.01; 3]);
        let measured = MergeOptions::parse_env(Some(",0.0001,0"), None, None).unwrap();
        assert_eq!(measured.noise_sigma, None);
        assert_eq!(options.max_s, Some(8.0));
        for value in ["1,2", "1,2,3,4", "nan,0,0", "-1,0,0", "0,-1,0", "x,0,0"] {
            assert!(MergeOptions::parse_env(Some(value), None, None).is_err());
        }
        for value in ["nan", "inf", "0.5", "x"] {
            assert!(MergeOptions::parse_env(None, Some(value), None).is_err());
        }
    }

    #[test]
    fn saturation_override_requires_three_positive_finite_fractions() {
        assert_eq!(MergeOptions::default().saturation, None);
        let options = MergeOptions::parse_env(None, None, Some("0.44, 0.43, 0.55")).unwrap();
        assert_eq!(options.saturation, Some([0.44, 0.43, 0.55]));
        for value in [
            "",
            "0.4,0.5",
            "0.4,0.5,0.6,0.7",
            "0,1,1",
            "-1,1,1",
            "nan,1,1",
            "inf,1,1",
            "1.1,1,1",
            "x,1,1",
        ] {
            assert!(
                MergeOptions::parse_env(None, None, Some(value)).is_err(),
                "{value}"
            );
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
