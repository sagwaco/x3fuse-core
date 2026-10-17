//! Merge expanded X3I exposures in reference-frame camera units.
//!
//! With fix_bad enabled, the producer repairs samples before the footprint
//! reads them. The snapshot uses the intermediate black and white levels.
//! With the real producer, r is bounded below by minus the intermediate bias
//! fraction, so zero sigma cannot produce a negative variance.
//! Noise uses r = (sample - raw_black) / (raw_white - raw_black).
//! Validity and saturation roll-off use d = r / saturation[plane].
//! A measured sensor ceiling below the CAMF white changes only d, not noise units.
//! Samples near the physical ceiling can already have a compressed response.
//! The default knee of 0.75 and width of 0.02 reduce their weight before the
//! validity cut at 0.85. At the cut, the logistic weight is 0.006693.
//! Even a 64-fold exposure weight gives only 0.4284 of the reference weight.
//! The hard cut excludes the compressed pile near 0.94 to 0.97 of the ceiling.
//! Accept raw agreement within 1 percent per bin.
//! For bins outside that tolerance, allow estimated weighted bias up to 0.2 percent.
//! Ratio refinement accepts frame-wide exposure offsets within 0.3 EV.
//! The zero clamp raises the mean near black. Short frames can then show a level
//! trend without a response-curve change. Ratio fits and their bins require the
//! integer-paired raw T sample above ten times that frame's read noise.
//! Use the median of supported bin medians whose mean frame signal exceeds
//! thirty read sigmas and whose reference is below the roll-off knee. Require
//! two clean bins and apply the 2 percent trend gate only to these bins.
//! Near-black bias is nonlinear, so an affine fit can bend the slope. Keep
//! its offset only as a diagnostic. It does not change the scale or pixels.
//! Keep the nominal scale when clean support is too small.
//! At d = 0.825, the measured +1 EV frame has estimated weighted bias of 0.144 percent.
//! X3F_X3I_ROLLOFF=knee,width,margin overrides these research settings.
//! A flat wall in an unsaturated frame can also pass the ceiling pile test.
//! shortcut: no cross-frame ceiling check, phase 2 must require the second-longest
//! frame's ceiling to agree within 2 percent before accepting the estimate.
//! The intermediate encoding clamps at zero, so it truncates negative noise. Expansion
//! also quantises and clamps BMT to u16, including Y = min(4T, 65535).
//! Clamping near black truncates noise. Its variance can be below the shield
//! estimate. Fix c_ref at one and fit each c_i from at least 50 flat blocks
//! with both raw T samples above ten read sigmas. Clamp c_i to [0.5, 4]
//! by default. Weak alignment can use a separate ghost threshold.
//! Below ten sigmas, use the untruncated model without correction. It overestimates
//! near-black variance and conservatively under-rejects these samples.
//! Flag calibration when the median reference model term exceeds eight times
//! the frame term. Such a small frame term poorly determines c_i.
//! The raw model used for merge weights stays unchanged.
//! The merge applies no signal-dependent dark weight. Unsigned DNG encoding
//! clamps negative merged samples to zero.
//!
//! The expansion builds its half-resolution top sum from unrepaired samples.
//! A defective top pixel can contaminate chroma in its 8x8 expanded block.
//! Repaired raw_top does not expose this defect to the merge. This existing
//! single-frame pipeline limit remains.
//!
//! Buffer budget for N expanded pixels. Producer scratch is additional:
//! - Persistent: 3 weighted-sum, 2 weight, 3 fallback and 2 SNR f32 planes,
//!   or 40N bytes. SNR stays f32 to preserve fallback candidate ordering.
//! - One frame: 3 BMT and up to 3 gain f32 planes, or 24N bytes; N top and
//!   N/2 half-plane u16 samples, or 3N bytes.
//! - Footprint peak: one full and one half f32 plane, or 6N bytes. Erosion
//!   reuses the full plane and one row. Accumulation retains one full plane.
//!
//! Phase 2 loads each frame in two passes. Statistics retains reference planes
//! and raw samples while it examines each other frame.
//! Materialised variance adds 8N transient bytes. The ghost estimate adds 12N
//! bytes for RGB and 8N for its two effective-variance groups.
//! The reported ghost fraction counts valid weights below 0.1 after the 3x3
//! minimum. At sigma 3, this corresponds to |z| above 4.55 before the minimum.
//! Ghost diagnostics are preliminary when alignment is weak or skipped.
//! Validity adds N per frame; ghost weights add 4N. Alignment can also hold
//! an input and output frame. Reference bins and the current noise mask add 2N bytes.
//! Raw reference samples and T-only gain (4N bytes) remain during fitting.
//! All statistics rasters are released before accumulation reloads frame zero.
//! Reference variance and saturation planes are dropped after luma construction.
//! The block summaries are small relative to the image planes. No full sequence
//! is cached. This trades a second frame preparation pass for bounded memory.
//! shortcut: the 1.4 GB budget needs row-band processing, measure RSS before that change.

use rayon::prelude::*;
use x3f_sys::{Control, MergeFrame};

use crate::x3i_align::{self, AlignOptions, Alignment, FramePlanes};
use crate::{Error, Image, ImageLevels, ProcessOptions};

/// Research controls for the exposure merge.
#[derive(Clone, Debug)]
pub struct MergeOptions {
    /// Processing settings for frame preparation and DNG output.
    pub process: ProcessOptions,
    /// Alignment settings for each non-reference frame.
    pub align: AlignOptions,
    /// Keep shutter ratios without data refinement.
    pub fixed_ratios: bool,
    /// Maximum relative spread of clean ratio bins. The default is 0.02.
    pub ratio_trend_limit: f32,
    /// Estimate shot noise from the reference and the +1 EV frame.
    pub fit_noise: bool,
    /// Lower effective-variance clamp. The default is 0.5.
    pub variance_floor: f32,
    /// Upper effective-variance clamp. The default is 4.
    pub variance_ceiling: f32,
    /// Ghost threshold in sigma. The default is 3.
    pub ghost_sigma: f32,
    /// Ghost threshold for weak alignment. None uses the ordinary threshold.
    pub weak_ghost_sigma: Option<f32>,
    /// Return research ghost maps. Each non-reference frame adds one byte per pixel.
    pub ghost_maps: bool,
    /// Reject samples that disagree with the reference estimate.
    pub ghost_rejection: bool,
    /// Fraction of the physical ceiling at the saturation roll-off midpoint.
    pub saturation_knee: f32,
    /// Width of the logistic saturation roll-off, as a fraction of the ceiling.
    pub saturation_width: f32,
    /// Reject samples at or above one minus this margin of the ceiling.
    pub validity_margin: f32,
    /// Physical ceilings in raw-range units. None uses one in all planes.
    pub saturation: Option<[f32; 3]>,
    /// Default shot-noise coefficients when the pair fit has too few blocks.
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
            align: AlignOptions::default(),
            fixed_ratios: false,
            ratio_trend_limit: 0.02,
            fit_noise: true,
            variance_floor: 0.5,
            variance_ceiling: 4.0,
            ghost_sigma: 3.0,
            weak_ghost_sigma: None,
            ghost_maps: false,
            ghost_rejection: true,
            saturation_knee: 0.75,
            saturation_width: 0.02,
            validity_margin: 0.15,
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
            std::env::var("X3F_X3I_ROLLOFF").ok().as_deref(),
            [
                std::env::var("X3F_X3I_NO_ALIGN").ok().as_deref(),
                std::env::var("X3F_X3I_FIXED_RATIOS").ok().as_deref(),
                std::env::var("X3F_X3I_NO_GHOST").ok().as_deref(),
                std::env::var("X3F_X3I_RATIO_TREND").ok().as_deref(),
                std::env::var("X3F_X3I_C_MIN").ok().as_deref(),
                std::env::var("X3F_X3I_C_MAX").ok().as_deref(),
                std::env::var("X3F_X3I_GHOST_SIGMA").ok().as_deref(),
                std::env::var("X3F_X3I_WEAK_GHOST_SIGMA").ok().as_deref(),
            ],
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
        rolloff: Option<&str>,
        phase2: [Option<&str>; 8],
    ) -> Result<Self, Error> {
        let mut options = Self::default();
        let parse = |value: &str| {
            value
                .trim()
                .parse::<f32>()
                .map_err(|_| Error::InvalidData("invalid X3I numeric override".into()))
        };
        if let Some(noise) = noise {
            options.fit_noise = false;
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
        if let Some(rolloff) = rolloff {
            let parts: Vec<_> = rolloff.split(',').collect();
            let [knee, width, margin] = parts.as_slice() else {
                return Err(Error::InvalidData(
                    "X3F_X3I_ROLLOFF requires knee,width,margin".into(),
                ));
            };
            options.saturation_knee = parse(knee)?;
            options.saturation_width = parse(width)?;
            options.validity_margin = parse(margin)?;
        }
        options.align.enabled = phase2[0] != Some("1");
        options.fixed_ratios = phase2[1] == Some("1");
        options.ghost_rejection = phase2[2] != Some("1");
        if let Some(limit) = phase2[3] {
            options.ratio_trend_limit = parse(limit)?;
        }
        if let Some(floor) = phase2[4] {
            options.variance_floor = parse(floor)?;
        }
        if let Some(ceiling) = phase2[5] {
            options.variance_ceiling = parse(ceiling)?;
        }
        if let Some(sigma) = phase2[6] {
            options.ghost_sigma = parse(sigma)?;
        }
        if let Some(sigma) = phase2[7] {
            options.weak_ghost_sigma = Some(parse(sigma)?);
        }
        options.validate()?;
        Ok(options)
    }

    fn validate(&self) -> Result<(), Error> {
        if !self.variance_floor.is_finite()
            || !(0.0 < self.variance_floor && self.variance_floor <= 1.0)
            || !self.variance_ceiling.is_finite()
            || self.variance_ceiling < self.variance_floor
            || [
                self.ghost_sigma,
                self.weak_ghost_sigma.unwrap_or(self.ghost_sigma),
            ]
            .iter()
            .any(|s| !s.is_finite() || *s <= 0.0 || !s.powi(2).is_normal())
            || !self.ratio_trend_limit.is_finite()
            || self.ratio_trend_limit < 0.0
            || !self.saturation_knee.is_finite()
            || !(0.0 < self.saturation_knee && self.saturation_knee <= 1.0)
            || !self.saturation_width.is_finite()
            || self.saturation_width <= 0.0
            || !self.validity_margin.is_finite()
            || !(0.0..1.0).contains(&self.validity_margin)
            || self.saturation_knee >= 1.0 - self.validity_margin
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
        if upper as f64 <= black {
            return false;
        }
        let start = lower(upper);
        let width = upper - start + 1;
        let Some(previous_start) = start.checked_sub(width) else {
            return false;
        };
        let pile = histogram[upper] - histogram[start - 1];
        let previous =
            histogram[start - 1] - previous_start.checked_sub(1).map_or(0, |i| histogram[i]);
        enough(pile) && pile / 5 >= previous
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
    /// Refined scale in reference units, in source order.
    pub scales: Vec<f64>,
    /// Shutter ratios before refinement, in source order.
    pub nominal_scales: Vec<f64>,
    /// Raw-plane ratio fits, in B, M, T order.
    pub plane_scales: Vec<[Option<f32>; 3]>,
    /// Aligned T ratio diagnostics in the eight reference-level bins.
    pub aligned_ratios: Vec<RatioFit>,
    /// Fitted or overridden shot-noise coefficients, in B, M, T order.
    pub noise_k: [f32; 3],
    /// Accepted noise blocks per plane. Fewer than 50 uses a default or override.
    pub noise_blocks: [usize; 3],
    /// Effective-variance correction per frame.
    pub effective_variance: Vec<f32>,
    /// Effective-variance correction for the shared B/M ghost group.
    pub chroma_variance: Vec<f32>,
    /// Median squared z residuals in flat calibration blocks, in B, M, T order.
    pub median_z_squared: Vec<[Option<f32>; 3]>,
    /// Number of accepted calibration blocks per frame.
    pub calibration_blocks: Vec<usize>,
    /// True when the median reference model term exceeds eight times the frame term.
    pub weak_calibration: Vec<bool>,
    /// Ghost fractions in eight bins from 0.05 to 0.60 and two from 0.60 to 0.85.
    pub ghost_bins: Vec<[Option<f32>; 10]>,
    /// Valid pixel counts for the ghost bins.
    pub ghost_bin_pixels: Vec<[usize; 10]>,
    /// Fraction of valid pixels with ghost weight below 0.1 after the 3x3 minimum.
    pub ghost_fractions: Vec<f32>,
    /// Ghost threshold used per frame. Frame zero has no rejection.
    pub ghost_sigmas: Vec<f32>,
    /// True when alignment ran and its result was weak or skipped.
    pub weak_alignments: Vec<bool>,
    /// Transforms in source order. Frame zero and disabled alignment use identity.
    pub alignments: Vec<Alignment>,
    /// Rejected ratio fits and alignment estimates.
    pub fit_warnings: Vec<String>,
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
    /// Format fitted statistics without classifying them as warnings.
    pub fn statistics(&self) -> Vec<String> {
        let mut lines = vec![format!(
            "X3I noise: k_B={:.9} k_M={:.9} k_T={:.9} blocks_B={} blocks_M={} blocks_T={} (below 50: default or override)",
            self.noise_k[0], self.noise_k[1], self.noise_k[2], self.noise_blocks[0], self.noise_blocks[1], self.noise_blocks[2]
        )];
        for i in 0..self.scales.len() {
            lines.push(format!(
                "X3I frame {}: statistics nominal={:.12} refined={:.12} raw_plane_scales={:?} c={:.6} c_chroma={:.6} median_z2_flat={:?} calibration_blocks={} ghost_fraction={:.9} ghost_threshold=0.1 ghost_status=preliminary black_offset={:?} calibration_weak={} alignment_weak={} ghost_sigma={:.6} ghost_enabled={}",
                self.frame_indices[i], self.nominal_scales[i], self.scales[i], self.plane_scales[i],
                self.effective_variance[i], self.chroma_variance[i], self.median_z_squared[i], self.calibration_blocks[i], self.ghost_fractions[i], self.aligned_ratios[i].black_offset, self.weak_calibration[i], self.weak_alignments[i], self.ghost_sigmas[i], self.ghost_sigmas[i] != 0.0
            ));
            let fit = &self.aligned_ratios[i];
            for bin in 0..8 {
                let lower = 0.05 + 0.55 * bin as f32 / 8.0;
                let upper = 0.05 + 0.55 * (bin + 1) as f32 / 8.0;
                let ratio = fit.bins[bin]
                    .map(|v| format!("{:.9}", f64::from(v) * self.nominal_scales[i]))
                    .unwrap_or_else(|| "-".into());
                lines.push(format!("X3I frame {}: aligned-ratio-bin lower={lower:.5} upper={upper:.5} ratio={ratio} pairs={} accepted={} clean={} frame_mean_read_sigma={:?} level_trend={:?}",
                    self.frame_indices[i], fit.bin_pairs[bin], fit.bin_pairs[bin] >= fit.minimum_pairs && fit.minimum_pairs != 0, fit.clean_bins[bin], fit.bin_read_snr[bin], fit.level_trend));
            }
            for bin in 0..10 {
                let (lower, upper) = ghost_bin_bounds(bin);
                lines.push(format!("X3I frame {}: ghost-bin lower={lower:.5} upper={upper:.5} fraction={:?} pixels={} threshold=0.1", self.frame_indices[i], self.ghost_bins[i][bin], self.ghost_bin_pixels[i][bin]));
            }
            let a = &self.alignments[i];
            lines.push(format!(
                "X3I frame {}: alignment dx={:.6} dy={:.6} peak={:.6} tile_spread={:.6} similarity={:?} retained_tiles={} max_tile_residual={:.6} weak={} skipped={}",
                self.frame_indices[i], a.dx, a.dy, a.peak, a.tile_spread, a.similarity, a.retained_tiles, a.max_tile_residual,
                self.weak_alignments[i], a.skipped
            ));
        }
        lines
    }

    /// Describe fallback and safety clipping counts for a conversion report.
    pub fn warnings(&self) -> Vec<String> {
        let mut warnings = self.fit_warnings.clone();
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

/// Research ghost weight image at the full merge resolution.
/// 255 means weight one and 0 means fully rejected. 128 marks invalid pixels;
/// valid weights that round to 128 use 127 so the marker stays distinct.
#[derive(Clone, Debug, Default)]
pub struct GhostMap {
    /// Source frame index. The reference has no map.
    pub frame_index: usize,
    /// Full merge width in pixels.
    pub width: u32,
    /// Full merge height in pixels.
    pub height: u32,
    /// Final weights after the 3x3 minimum, one byte per pixel in row order.
    pub data: Vec<u8>,
}

fn weak_alignment(alignment: &Alignment) -> bool {
    alignment.skipped || alignment.retained_tiles <= 4 || alignment.max_tile_residual > 1.5
}

fn ghost_map(frame: &FramePlanes, weights: &[f32], frame_index: usize) -> GhostMap {
    GhostMap {
        frame_index,
        width: frame.width,
        height: frame.height,
        data: weights
            .iter()
            .zip(&frame.valid)
            .map(|(&weight, &valid)| {
                if valid == 0 {
                    128
                } else {
                    let code = (weight * 255.0).round() as u8;
                    if code == 128 {
                        127
                    } else {
                        code
                    }
                }
            })
            .collect(),
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
    /// Research ghost maps. Empty unless requested in MergeOptions.
    pub ghost_maps: Vec<GhostMap>,
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

fn median(values: &mut [f32]) -> Option<f32> {
    if values.is_empty() {
        return None;
    }
    let even = values.len().is_multiple_of(2);
    let mid = values.len() / 2;
    let (left, value, _) = values.select_nth_unstable_by(mid, f32::total_cmp);
    Some(if even {
        // The caller uses finite samples only.
        (*value + left.iter().copied().max_by(f32::total_cmp).unwrap()) * 0.5
    } else {
        *value
    })
}

fn raw(frame: &MergeFrame, plane: usize, p: usize) -> f32 {
    let value = if plane == 2 {
        frame.raw_top[p]
    } else {
        frame.raw_half[p * 2 + plane]
    };
    normalise(
        value,
        frame.raw_black[plane],
        1.0 / (frame.raw_white[plane] - frame.raw_black[plane]),
    )
}

// Nearest source sample of plane sample p. B/M use divisor 2 and block centres 2h + 0.5.
fn paired(
    p: usize,
    width: usize,
    height: usize,
    alignment: &Alignment,
    divisor: usize,
) -> Option<usize> {
    let centre = (divisor - 1) as f32 * 0.5;
    let rx = (p % width * divisor) as f32 + centre;
    let ry = (p / width * divisor) as f32 + centre;
    let [sx, sy] = alignment.source_at(rx, ry);
    let x = (p % width) as isize - ((rx - sx) / divisor as f32).round() as isize;
    let y = (p / width) as isize - ((ry - sy) / divisor as f32).round() as isize;
    (sx.is_finite()
        && sy.is_finite()
        && x >= 0
        && y >= 0
        && x < width as isize
        && y < height as isize)
        .then(|| y as usize * width + x as usize)
}

fn ratio_fit(
    reference: &MergeFrame,
    frame: &MergeFrame,
    alignment: &Alignment,
    options: &MergeOptions,
    control: Control<'_>,
) -> Result<[Option<f32>; 3], Error> {
    let mut result = [None; 3];
    let ceiling = options.saturation.unwrap_or([1.0; 3]);
    for c in 0..3 {
        let divisor = if c == 2 { 1 } else { 2 };
        let width = reference.width as usize / divisor;
        let height = reference.height as usize / divisor;
        let mut ratios = Vec::new();
        for y in 0..height {
            control.check()?;
            for x in ((y % 3)..width).step_by(3) {
                let p = y * width + x;
                let Some(q) = paired(p, width, height, alignment, divisor) else {
                    continue;
                };
                let a = raw(reference, c, p);
                let b = raw(frame, c, q);
                if (0.05..=0.60).contains(&(a / ceiling[c]))
                    && b > 0.0
                    && b < ceiling[c] * (1.0 - options.validity_margin)
                {
                    ratios.push(a / b);
                }
            }
        }
        result[c] = median(&mut ratios);
    }
    Ok(result)
}

/// A shared T scale fit and its reference-level bins.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RatioFit {
    /// Scale multiplier, present only when enough pairs qualify.
    pub multiplier: Option<f32>,
    /// Fitted black offset in reference camera units. It does not change pixels.
    pub black_offset: Option<f32>,
    /// Number of sampled pairs used by the scalar fit.
    pub pairs: usize,
    /// Required count for a scalar fit or an accepted bin.
    pub minimum_pairs: usize,
    /// Median multipliers for eight equal bins between 0.05 and 0.60.
    pub bins: [Option<f32>; 8],
    /// Sampled pair counts in each bin after the scalar fit cuts.
    pub bin_pairs: [usize; 8],
    /// Mean raw frame signal divided by its own read noise in each bin.
    pub bin_read_snr: [Option<f32>; 8],
    /// Supported bins above thirty read sigmas, used by the scalar estimate.
    pub clean_bins: [bool; 8],
    /// Largest relative difference between clean bin medians.
    pub level_trend: Option<f32>,
}

fn refined_ratio(
    (reference, frame): (&MergeFrame, &MergeFrame),
    a: &FramePlanes,
    b: &FramePlanes,
    alignment: &Alignment,
    nominal: f64,
    options: &MergeOptions,
    control: Control<'_>,
) -> Result<RatioFit, Error> {
    let ceiling = options.saturation.unwrap_or([1.0; 3])[2];
    let mut samples: [Vec<f32>; 8] = Default::default();
    let mut reference_samples: [Vec<f32>; 8] = Default::default();
    let mut frame_samples: [Vec<f32>; 8] = Default::default();
    let read_noise = options.noise_sigma.unwrap_or(frame.sigma[2]);
    let minimum_frame = 10.0 * read_noise;
    let mut raw_sums = [0.0f64; 8];
    let upper = if nominal < 1.0 { 0.25 } else { 0.60 };
    // Sample one in three pixels to bound scratch. Keep one scale for all channels.
    for y in 0..a.height as usize {
        control.check()?;
        for x in ((y % 3)..a.width as usize).step_by(3) {
            let p = y * a.width as usize + x;
            let level = raw(reference, 2, p) / ceiling;
            let Some(source) = paired(p, b.width as usize, b.height as usize, alignment, 1) else {
                continue;
            };
            if raw(frame, 2, source) > minimum_frame
                && a.valid[p] != 0
                && b.valid[p] != 0
                && (0.05..=upper).contains(&level)
                && level < options.saturation_knee
                && (nominal >= 1.0 || b.d_max[p] < 0.40)
                && a.x[p * 3 + 2] > 0.0
                && b.x[p * 3 + 2] > 0.0
            {
                let ratio = a.x[p * 3 + 2] / b.x[p * 3 + 2];
                if ratio.is_finite() {
                    let bin = (((level - 0.05) / 0.55 * 8.0) as usize).min(7);
                    samples[bin].push(ratio);
                    raw_sums[bin] += f64::from(raw(frame, 2, source));
                    reference_samples[bin].push(a.x[p * 3 + 2]);
                    frame_samples[bin].push(b.x[p * 3 + 2]);
                }
            }
        }
    }
    let mut fit = RatioFit {
        minimum_pairs: 1000.max(a.valid.len().div_ceil(1000)),
        bin_pairs: std::array::from_fn(|i| samples[i].len()),
        bins: std::array::from_fn(|i| median(&mut samples[i])),
        ..Default::default()
    };
    fit.pairs = fit.bin_pairs.iter().sum();
    fit.bin_read_snr = std::array::from_fn(|i| {
        (fit.bin_pairs[i] > 0).then(|| {
            (raw_sums[i] / fit.bin_pairs[i] as f64 / f64::from(read_noise.max(1e-12))) as f32
        })
    });
    fit.clean_bins = std::array::from_fn(|i| {
        fit.bin_pairs[i] >= fit.minimum_pairs && fit.bin_read_snr[i].is_some_and(|v| v > 30.0)
    });
    let mut clean: Vec<_> = (0..8)
        .filter(|&i| fit.clean_bins[i])
        .filter_map(|i| fit.bins[i])
        .collect();
    if clean.len() >= 2 {
        let low = clean.iter().copied().fold(f32::INFINITY, f32::min);
        let high = clean.iter().copied().fold(0.0f32, f32::max);
        fit.level_trend = Some(high / low - 1.0);
        fit.multiplier = median(&mut clean);
    }
    let points: Vec<_> = (0..8)
        .filter(|&i| fit.bin_pairs[i] >= fit.minimum_pairs)
        .map(|i| {
            (
                median(&mut frame_samples[i]).unwrap(),
                median(&mut reference_samples[i]).unwrap(),
            )
        })
        .collect();
    if points.len() >= 2 {
        let mean_x = points.iter().map(|p| f64::from(p.0)).sum::<f64>() / points.len() as f64;
        let mean_y = points.iter().map(|p| f64::from(p.1)).sum::<f64>() / points.len() as f64;
        let xx = points
            .iter()
            .map(|p| (f64::from(p.0) - mean_x).powi(2))
            .sum::<f64>();
        let xy = points
            .iter()
            .map(|p| (f64::from(p.0) - mean_x) * (f64::from(p.1) - mean_y))
            .sum::<f64>();
        if xx > 1e-12 {
            let slope = (xy / xx) as f32;
            let offset = (mean_y - f64::from(slope) * mean_x) as f32;
            if slope.is_finite() && slope > 0.0 && offset.is_finite() {
                fit.black_offset = Some(offset);
            }
        }
    }
    Ok(fit)
}

fn variance(values: &[f32]) -> f32 {
    let mean = values.iter().sum::<f32>() / values.len() as f32;
    values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / (values.len() - 1) as f32
}

fn flat(values: &[f32; 64], noise: f32) -> bool {
    let mut gradient = 0.0;
    for y in 0..8 {
        for x in 0..8 {
            let p = y * 8 + x;
            if x > 0 {
                gradient += (values[p] - values[p - 1]).powi(2);
            }
            if y > 0 {
                gradient += (values[p] - values[p - 8]).powi(2);
            }
        }
    }
    let mean = values.iter().sum::<f32>() / 64.0;
    if gradient / 112.0 > 16.0 * noise.max(1e-12) + (0.02 * mean).powi(2) {
        return false;
    }
    let means: [f32; 4] = std::array::from_fn(|q| {
        let mut sum = 0.0;
        for y in q / 2 * 4..q / 2 * 4 + 4 {
            for x in q % 2 * 4..q % 2 * 4 + 4 {
                sum += values[y * 8 + x];
            }
        }
        sum / 16.0
    });
    let mean = means.iter().sum::<f32>() / 4.0;
    let range = means.iter().copied().fold(f32::NEG_INFINITY, f32::max)
        - means.iter().copied().fold(f32::INFINITY, f32::min);
    range < 0.02 * mean.abs() + noise.max(1e-12).sqrt()
}

fn noise_fit(
    reference: &MergeFrame,
    frame: &MergeFrame,
    alignment: &Alignment,
    scale: f32,
    options: &MergeOptions,
    control: Control<'_>,
) -> Result<([f32; 3], [usize; 3]), Error> {
    let mut k = options.noise_k;
    let mut counts = [0; 3];
    let ceiling = options.saturation.unwrap_or([1.0; 3]);
    for c in 0..3 {
        let divisor = if c == 2 { 1 } else { 2 };
        let width = reference.width as usize / divisor;
        let height = reference.height as usize / divisor;
        let read0 = options.noise_sigma.unwrap_or(reference.sigma[c]).powi(2);
        let read1 = options.noise_sigma.unwrap_or(frame.sigma[c]).powi(2);
        let mut fits = Vec::new();
        for y in (0..height.saturating_sub(7)).step_by(8) {
            control.check()?;
            for x in (0..width.saturating_sub(7)).step_by(8) {
                let mut a = [0.0; 64];
                let mut b = [0.0; 64];
                let mut valid = true;
                for j in 0..64 {
                    let p = (y + j / 8) * width + x + j % 8;
                    let Some(q) = paired(p, width, height, alignment, divisor) else {
                        valid = false;
                        break;
                    };
                    a[j] = raw(reference, c, p);
                    b[j] = raw(frame, c, q);
                    valid &= a[j] < ceiling[c] * (1.0 - options.validity_margin)
                        && b[j] < ceiling[c] * (1.0 - options.validity_margin);
                }
                let mean = a.iter().sum::<f32>() / 64.0;
                if !valid
                    || !(0.05..=0.50).contains(&(mean / ceiling[c]))
                    || !flat(&a, read0 + options.noise_k[c] * mean)
                    || !flat(&b, read1 + options.noise_k[c] * mean / scale)
                {
                    continue;
                }
                let diff: [f32; 64] = std::array::from_fn(|j| a[j] - scale * b[j]);
                let quadratic = options.noise_q[c].powi(2)
                    * (mean.powi(2) + scale.powi(2) * (b.iter().sum::<f32>() / 64.0).powi(2));
                let fit = (variance(&diff) - read0 - scale.powi(2) * read1 - quadratic)
                    / (mean * (1.0 + scale));
                if fit.is_finite() {
                    fits.push(fit);
                }
            }
        }
        counts[c] = fits.len();
        if fits.len() >= 50 {
            k[c] = median(&mut fits).unwrap().max(0.0);
        }
    }
    Ok((k, counts))
}

fn materialise(
    frame: &mut MergeFrame,
    scale: f32,
    options: &MergeOptions,
    control: Control<'_>,
) -> Result<FramePlanes, Error> {
    let factors = validate_frame(frame, control)?;
    let ceiling = options.saturation.unwrap_or([1.0; 3]);
    let d_max = saturation_max(
        frame,
        std::array::from_fn(|c| factors[c] / f64::from(ceiling[c])),
        control,
    )?;
    let mut var = vec![0.0; frame.raw_top.len() * 2];
    let sigma = options.noise_sigma.map_or(frame.sigma, |s| [s; 3]);
    var.par_chunks_mut(frame.width as usize * 2)
        .enumerate()
        .try_for_each(|(y, row)| -> Result<(), Error> {
            control.check()?;
            for (x, pair) in row.chunks_mut(2).enumerate() {
                let p = y * frame.width as usize + x;
                let half = y / 2 * frame.raw_half_width as usize + x / 2;
                let v: [f32; 3] = std::array::from_fn(|c| {
                    let r = raw(frame, c, if c == 2 { p } else { half });
                    let gain = frame.gain.as_ref().map_or(1.0, |g| g[p * 3 + c]);
                    scale.powi(2)
                        * gain.powi(2)
                        * (sigma[c].powi(2)
                            + options.noise_k[c] * r
                            + (options.noise_q[c] * r).powi(2))
                        .max(1e-12)
                });
                pair.copy_from_slice(&[v[0] + v[1], v[2]]);
                if pair.iter().any(|v| !v.is_finite() || *v <= 0.0) {
                    return Err(Error::InvalidData("non-finite X3I variance".into()));
                }
            }
            Ok(())
        })?;
    let mut x = std::mem::take(&mut frame.bmt);
    x.par_chunks_mut(frame.width as usize * 3)
        .try_for_each(|row| -> Result<(), Error> {
            control.check()?;
            for value in row {
                *value *= scale;
                if !value.is_finite() {
                    return Err(Error::InvalidData("non-finite scaled X3I sample".into()));
                }
            }
            Ok(())
        })?;
    let valid = d_max
        .iter()
        .map(|&d| u8::from(d < 1.0 - options.validity_margin))
        .collect();
    Ok(FramePlanes {
        width: frame.width,
        height: frame.height,
        x,
        var,
        d_max,
        valid,
    })
}

#[derive(Clone, Copy)]
struct CalibrationBlock {
    residual: [f32; 3],
    median_squared: [f32; 3],
    reference: [f32; 2],
    frame: [f32; 2],
}

fn retain_top_gain(frame: &mut MergeFrame) {
    if let Some(gain) = &mut frame.gain {
        for p in 0..frame.raw_top.len() {
            gain[p] = gain[p * 3 + 2];
        }
        gain.truncate(frame.raw_top.len());
        gain.shrink_to_fit();
    }
}

fn raw_row_hash(frame: &MergeFrame) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hash = std::collections::hash_map::DefaultHasher::new();
    frame
        .raw_top
        .iter()
        .take(frame.width as usize)
        .for_each(|v| v.hash(&mut hash));
    hash.finish()
}

fn model_terms(frame: &MergeFrame, p: usize, scale: f32, options: &MergeOptions) -> [f32; 2] {
    let r = raw(frame, 2, p);
    let gain = frame.gain.as_ref().map_or(1.0, |g| g[p]);
    let factor = (scale * gain).powi(2);
    [
        factor
            * (options.noise_sigma.unwrap_or(frame.sigma[2]).powi(2)
                + (options.noise_q[2] * r).powi(2)),
        factor * r,
    ]
}

fn calibration_blocks(
    reference: (&MergeFrame, &FramePlanes),
    frame: (&MergeFrame, &FramePlanes),
    alignment: &Alignment,
    scale: f32,
    options: &MergeOptions,
    control: Control<'_>,
) -> Result<Vec<CalibrationBlock>, Error> {
    let (raw_ref, reference) = reference;
    let (raw_frame, frame) = frame;
    let width = reference.width as usize;
    let height = reference.height as usize;
    let mut blocks = Vec::new();
    for y in (0..height.saturating_sub(7)).step_by(8) {
        control.check()?;
        for x in (0..width.saturating_sub(7)).step_by(8) {
            let mut a = [[0.0; 64]; 3];
            let mut b = [[0.0; 64]; 3];
            let mut mr = [0.0; 2];
            let mut mi = [0.0; 2];
            let mut valid = true;
            for j in 0..64 {
                let p = (y + j / 8) * width + x + j % 8;
                let Some(q) = paired(p, width, height, alignment, 1) else {
                    valid = false;
                    break;
                };
                if reference.valid[p] == 0
                    || frame.valid[p] == 0
                    || raw(raw_ref, 2, p) <= 10.0 * options.noise_sigma.unwrap_or(raw_ref.sigma[2])
                    || raw(raw_frame, 2, q)
                        <= 10.0 * options.noise_sigma.unwrap_or(raw_frame.sigma[2])
                {
                    valid = false;
                    break;
                }
                for c in 0..3 {
                    a[c][j] = reference.x[p * 3 + c];
                    b[c][j] = frame.x[p * 3 + c];
                }
                let r = model_terms(raw_ref, p, 1.0, options);
                let f = model_terms(raw_frame, q, scale, options);
                for term in 0..2 {
                    mr[term] += r[term] / 64.0;
                    mi[term] += f[term] / 64.0;
                }
            }
            let vr = (mr[0] + options.noise_k[2] * mr[1]).max(1e-12);
            let vi = (mi[0] + options.noise_k[2] * mi[1]).max(1e-12);
            if !valid || !flat(&a[2], vr) || !flat(&b[2], vi) {
                continue;
            }
            let mut residual = [0.0; 3];
            let mut median_squared = [0.0; 3];
            for c in 0..3 {
                let mut diff: [f32; 64] = std::array::from_fn(|j| b[c][j] - a[c][j]);
                let centre = median(&mut diff).unwrap();
                // Exclude coherent motion and offsets from noise calibration.
                valid &= centre.abs() <= 3.0 * (vr + vi).sqrt()
                    && flat(&a[c], 4.0 * vr)
                    && flat(&b[c], 4.0 * vi);
                let mut squared = diff.map(|v| v * v);
                median_squared[c] = median(&mut squared).unwrap();
                for value in &mut diff {
                    *value = (*value - centre).abs();
                }
                residual[c] = (1.4826 * median(&mut diff).unwrap()).powi(2);
            }
            if valid {
                blocks.push(CalibrationBlock {
                    residual,
                    median_squared,
                    reference: mr,
                    frame: mi,
                });
            }
        }
    }
    Ok(blocks)
}

fn calibrate(
    blocks: &[Vec<CalibrationBlock>],
    k: f32,
    chroma: bool,
    floor: f32,
    ceiling: f32,
) -> Vec<f32> {
    let solve = |blocks: &[CalibrationBlock], reference: f32| {
        if blocks.len() < 50 {
            return 1.0f32.clamp(floor, ceiling);
        }
        let mut values: Vec<_> = blocks
            .iter()
            .map(|b| {
                let residual = if chroma {
                    (b.residual[0] + b.residual[1]) * 0.5
                } else {
                    b.residual[2]
                };
                (residual - reference * (b.reference[0] + k * b.reference[1]).max(1e-12))
                    / (b.frame[0] + k * b.frame[1]).max(1e-12)
            })
            .collect();
        median(&mut values).unwrap().clamp(floor, ceiling)
    };
    std::iter::once(1.0)
        .chain(blocks.iter().skip(1).map(|b| solve(b, 1.0)))
        .collect()
}

fn ghost_weights_masked(
    frame: &FramePlanes,
    estimate: &[f32],
    estimate_var: &[[f32; 2]],
    correction: [f32; 2],
    above_black: &[u8],
    sigma: f32,
    control: Control<'_>,
) -> Result<Vec<f32>, Error> {
    let width = frame.width as usize;
    let height = frame.height as usize;
    let mut weights = vec![1.0f32; frame.valid.len()];
    weights
        .par_chunks_mut(width)
        .enumerate()
        .try_for_each(|(y, row)| -> Result<(), Error> {
            control.check()?;
            for (x, weight) in row.iter_mut().enumerate() {
                let p = y * width + x;
                if frame.valid[p] == 0 || estimate_var[p][1] == 0.0 {
                    continue;
                }
                // Share rejection across channels to keep colour ratios stable.
                let z_squared = (0..3)
                    .map(|c| {
                        let group = usize::from(c == 2);
                        let factor = if above_black.is_empty() || above_black[p] != 0 {
                            correction[group]
                        } else {
                            1.0
                        };
                        (frame.x[p * 3 + c] - estimate[p * 3 + c]).powi(2)
                            / (factor * frame.var[p * 2 + 1] + estimate_var[p][group])
                    })
                    .fold(0.0f32, f32::max);
                *weight = (-z_squared / sigma.powi(2)).exp();
            }
            Ok(())
        })?;
    for row in weights.chunks_mut(width) {
        let mut left = row[0];
        for x in 0..width {
            let current = row[x];
            row[x] = left.min(current).min(row[(x + 1).min(width - 1)]);
            left = current;
        }
    }
    let mut previous = vec![1.0f32; width];
    for y in 0..height {
        control.check()?;
        let (processed, remaining) = weights.split_at_mut((y + 1) * width);
        let row = &mut processed[y * width..];
        for x in 0..width {
            let current = row[x];
            row[x] = previous[x]
                .min(current)
                .min(remaining.get(x).copied().unwrap_or(current));
            previous[x] = current;
        }
    }
    Ok(weights)
}

fn ghost_bin_bounds(bin: usize) -> (f32, f32) {
    if bin < 8 {
        (
            0.05 + 0.55 * bin as f32 / 8.0,
            0.05 + 0.55 * (bin + 1) as f32 / 8.0,
        )
    } else {
        (
            0.60 + 0.125 * (bin - 8) as f32,
            0.60 + 0.125 * (bin - 7) as f32,
        )
    }
}

#[cfg(test)]
fn ghost_weights(
    frame: &FramePlanes,
    estimate: &[f32],
    estimate_var: &[[f32; 2]],
    correction: [f32; 2],
    control: Control<'_>,
) -> Result<Vec<f32>, Error> {
    ghost_weights_masked(frame, estimate, estimate_var, correction, &[], 3.0, control)
}

/// Load each frame once for statistics and once for accumulation (2n calls).
/// Frame zero defines units. The loader must return the same data on both calls.
/// Dimensions, exposure and a hash of the first raw top row detect common changes.
/// The hash is a cheap check, not a check of every sample. Report stages once per index.
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
    let mut reference = load(0)?;
    let mut hashes = vec![raw_row_hash(&reference)];
    let (width, height) = (reference.width, reference.height);
    let camera_to_rgb = reference.camera_to_rgb;
    let reference_us = reference.shutter_us;
    let pixels = reference.raw_top.len();
    let mut fitted = options.clone();
    let mut reference_planes = materialise(&mut reference, 1.0, options, control)?;
    retain_top_gain(&mut reference);
    let reference_luma = if options.align.enabled {
        Some(x3i_align::luma(
            &reference_planes,
            1.0 - options.validity_margin,
            control,
        )?)
    } else {
        None
    };
    // Fits use raw model terms, so these reference planes can be rebuilt later.
    reference_planes.var = Vec::new();
    reference_planes.d_max = Vec::new();
    let mut report = MergeReport {
        saturation: options.saturation.unwrap_or([1.0; 3]),
        nominal_scales: vec![1.0],
        scales: vec![1.0],
        plane_scales: vec![[Some(1.0); 3]],
        aligned_ratios: vec![RatioFit::default()],
        frame_indices: vec![reference.frame_index],
        exposure_us: vec![reference_us],
        alignments: vec![Alignment {
            peak: 1.0,
            ..Default::default()
        }],
        ghost_fractions: vec![0.0; frame_count],
        ghost_sigmas: vec![0.0; frame_count],
        weak_alignments: vec![false; frame_count],
        noise_k: options.noise_k,
        ..Default::default()
    };
    let mut calibration = vec![Vec::new()];
    let mut noise_fitted = false;
    for i in 1..frame_count {
        control.check()?;
        let mut frame = load(i)?;
        hashes.push(raw_row_hash(&frame));
        if (frame.width, frame.height) != (width, height) {
            return Err(Error::InvalidData(
                "X3I frame dimensions differ from the reference".into(),
            ));
        }
        let nominal = reference_us as f64 / frame.shutter_us as f64;
        let planes = materialise(&mut frame, nominal as f32, options, control)?;
        retain_top_gain(&mut frame);
        let alignment = if let Some(reference_luma) = &reference_luma {
            let luma = x3i_align::luma(&planes, 1.0 - options.validity_margin, control)?;
            x3i_align::estimate(reference_luma, &luma, &options.align, control)?
        } else {
            Alignment {
                peak: 1.0,
                ..Default::default()
            }
        };
        let ratios = if alignment.skipped {
            [None; 3]
        } else {
            ratio_fit(&reference, &frame, &alignment, options, control)?
        };
        let mut planes = if options.align.enabled && !alignment.skipped {
            x3i_align::warp(planes, &alignment, control)?
        } else {
            planes
        };
        let fit = refined_ratio(
            (&reference, &frame),
            &reference_planes,
            &planes,
            &alignment,
            nominal,
            options,
            control,
        )?;
        let mut scale = nominal;
        if !options.fixed_ratios && !alignment.skipped {
            if fit
                .level_trend
                .is_some_and(|trend| trend > options.ratio_trend_limit)
            {
                report.fit_warnings.push(format!("X3I ratio warning: frame={i} accepted level bins differ by {:.3} percent (limit {:.3}); keep shutter ratio {nominal:.6}", fit.level_trend.unwrap() * 100.0, options.ratio_trend_limit * 100.0));
            } else if fit.pairs >= fit.minimum_pairs
                && fit.bin_pairs.iter().all(|&n| n < fit.minimum_pairs)
            {
                report.fit_warnings.push(format!("X3I ratio warning: frame={i} no supported level bin ({} total pairs, {} required per bin); keep shutter ratio {nominal:.6}", fit.pairs, fit.minimum_pairs));
            } else if let Some(multiplier) = fit.multiplier {
                let ratio = nominal * f64::from(multiplier);
                if (ratio / nominal).log2().abs() <= 0.3 {
                    scale = ratio;
                } else {
                    report.fit_warnings.push(format!("X3I ratio warning: frame={i} fit {ratio:.6} exceeds 0.3 EV; keep shutter ratio {nominal:.6}"));
                }
            } else {
                report.fit_warnings.push(format!(
                    "X3I ratio warning: frame={i} fewer than two clean supported bins or too few pairs ({}/{}); keep shutter ratio {nominal:.6}", fit.pairs, fit.minimum_pairs
                ));
            }
        }
        if alignment.skipped {
            report.fit_warnings.push(format!(
                "X3I alignment warning: frame={i} skipped; merge with identity, range may be misregistered"
            ));
        }
        if options.fit_noise
            && !noise_fitted
            && !alignment.skipped
            && (nominal.log2() + 1.0).abs() < 0.3
        {
            (report.noise_k, report.noise_blocks) = noise_fit(
                &reference,
                &frame,
                &alignment,
                scale as f32,
                options,
                control,
            )?;
            noise_fitted = true;
        }
        for row in planes.x.chunks_mut(width as usize * 3) {
            control.check()?;
            for v in row {
                *v *= (scale / nominal) as f32;
            }
        }
        calibration.push(if !alignment.skipped && options.ghost_rejection {
            calibration_blocks(
                (&reference, &reference_planes),
                (&frame, &planes),
                &alignment,
                scale as f32,
                options,
                control,
            )?
        } else {
            Vec::new()
        });
        report.nominal_scales.push(nominal);
        report.scales.push(scale);
        report.plane_scales.push(ratios);
        report.aligned_ratios.push(fit);
        report.frame_indices.push(frame.frame_index);
        report.exposure_us.push(frame.shutter_us);
        report.weak_alignments[i] = options.align.enabled && weak_alignment(&alignment);
        report.alignments.push(alignment);
    }
    drop(reference_luma);
    report.calibration_blocks = calibration.iter().map(Vec::len).collect();
    report.weak_calibration = calibration
        .iter()
        .map(|blocks| {
            let mut ratios: Vec<_> = blocks
                .iter()
                .map(|b| {
                    (b.reference[0] + report.noise_k[2] * b.reference[1])
                        / (b.frame[0] + report.noise_k[2] * b.frame[1]).max(1e-12)
                })
                .collect();
            median(&mut ratios).is_some_and(|r| r > 8.0)
        })
        .collect();
    report.ghost_bins = vec![[None; 10]; frame_count];
    report.ghost_bin_pixels = vec![[0; 10]; frame_count];
    report.effective_variance = calibrate(
        &calibration,
        report.noise_k[2],
        false,
        options.variance_floor,
        options.variance_ceiling,
    );
    report.chroma_variance = calibrate(
        &calibration,
        report.noise_k[2],
        true,
        options.variance_floor,
        options.variance_ceiling,
    );
    for i in 1..frame_count {
        if options.ghost_rejection {
            let weak = report.weak_alignments[i];
            report.ghost_sigmas[i] = if weak {
                options
                    .weak_ghost_sigma
                    .unwrap_or(options.ghost_sigma)
                    .min(options.ghost_sigma)
            } else {
                options.ghost_sigma
            };
        }
    }
    report.median_z_squared = calibration
        .iter()
        .enumerate()
        .map(|(i, blocks)| {
            std::array::from_fn(|c| {
                let corrections = if c == 2 {
                    &report.effective_variance
                } else {
                    &report.chroma_variance
                };
                let mut z: Vec<_> = blocks
                    .iter()
                    .map(|b| {
                        let variance = corrections[0]
                            * (b.reference[0] + report.noise_k[2] * b.reference[1]).max(1e-12)
                            + corrections[i]
                                * (b.frame[0] + report.noise_k[2] * b.frame[1]).max(1e-12);
                        b.median_squared[c] / variance
                    })
                    .collect();
                median(&mut z)
            })
        })
        .collect();
    drop(calibration);
    fitted.noise_k = report.noise_k;
    drop(reference);
    drop(reference_planes);
    let mut estimate = Vec::new();
    let mut estimate_var = if options.ghost_rejection {
        vec![[0.0; 2]; pixels]
    } else {
        Vec::new()
    };
    let mut sum = vec![0.0f32; pixels * 3];
    let mut weights = vec![[0.0f32; 2]; pixels];
    // shortcut: five fallback planes increase peak memory, use row-band preparation for large rasters.
    let mut fallback = vec![[0.0f32; 3]; pixels];
    let mut best_snr = vec![[-1.0f32; 2]; pixels];
    let mut shortest_us = u32::MAX;
    // The first valid sample after the reference must come from the shortest exposure.
    let mut order: Vec<_> = (1..frame_count).collect();
    order.sort_by_key(|&i| report.exposure_us[i]);
    order.insert(0, 0);
    let mut reference_bins = Vec::new();
    let mut ghost_maps = Vec::new();
    for i in order {
        control.check()?;
        let mut frame = load(i)?;
        if (frame.width, frame.height, frame.shutter_us) != (width, height, report.exposure_us[i])
            || raw_row_hash(&frame) != hashes[i]
        {
            return Err(Error::InvalidData(
                "X3I frame changed between statistics and accumulation".into(),
            ));
        }
        if i == 0 {
            let ceiling = options.saturation.unwrap_or([1.0; 3])[2];
            reference_bins = (0..pixels)
                .map(|p| {
                    let level = raw(&frame, 2, p) / ceiling;
                    (0..10)
                        .find(|&b| {
                            let (low, high) = ghost_bin_bounds(b);
                            level >= low && level < high
                        })
                        .map_or(255, |b| b as u8)
                })
                .collect::<Vec<_>>();
        }
        let alignment = &report.alignments[i];
        let cutoff = 10.0 * options.noise_sigma.unwrap_or(frame.sigma[2]);
        let above_black: Vec<_> = if options.ghost_rejection {
            (0..pixels)
                .map(|p| {
                    u8::from(
                        paired(p, width as usize, height as usize, alignment, 1)
                            .is_some_and(|q| raw(&frame, 2, q) > cutoff),
                    )
                })
                .collect()
        } else {
            Vec::new()
        };
        let planes = materialise(&mut frame, report.scales[i] as f32, &fitted, control)?;
        drop(frame);
        let planes = if i != 0 && options.align.enabled && !report.alignments[i].skipped {
            x3i_align::warp(planes, &report.alignments[i], control)?
        } else {
            planes
        };
        let correction = [report.chroma_variance[i], report.effective_variance[i]];
        if options.ghost_rejection {
            for p in 0..pixels {
                if p % width as usize == 0 {
                    control.check()?;
                }
                if estimate_var[p][1] == 0.0 && planes.valid[p] != 0 {
                    if i != 0 {
                        estimate[p * 3..p * 3 + 3].copy_from_slice(&planes.x[p * 3..p * 3 + 3]);
                    }
                    estimate_var[p] = correction
                        .map(|c| if above_black[p] != 0 { c } else { 1.0 } * planes.var[p * 2 + 1]);
                }
            }
        }
        let ghost = if options.ghost_rejection && i != 0 {
            ghost_weights_masked(
                &planes,
                &estimate,
                &estimate_var,
                correction,
                &above_black,
                report.ghost_sigmas[i],
                control,
            )?
        } else {
            vec![1.0; pixels]
        };
        if options.ghost_maps && i != 0 {
            ghost_maps.push(ghost_map(&planes, &ghost, report.frame_indices[i]));
        }
        let valid_count = planes.valid.iter().filter(|&&v| v != 0).count();
        report.ghost_fractions[i] = if valid_count == 0 {
            0.0
        } else {
            ghost
                .iter()
                .zip(&planes.valid)
                .filter(|(g, v)| **g < 0.1 && **v != 0)
                .count() as f32
                / valid_count as f32
        };
        let mut rejected = [0usize; 10];
        for p in 0..pixels {
            let bin = reference_bins[p] as usize;
            if bin < 10 && planes.valid[p] != 0 {
                report.ghost_bin_pixels[i][bin] += 1;
                rejected[bin] += usize::from(ghost[p] < 0.1);
            }
        }
        report.ghost_bins[i] = std::array::from_fn(|b| {
            let count = report.ghost_bin_pixels[i][b];
            (count > 0).then(|| rejected[b] as f32 / count as f32)
        });
        let shortest = report.exposure_us[i] < shortest_us;
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
                        let valid = planes.valid[p] != 0;
                        let sat = 1.0
                            / (1.0
                                + ((planes.d_max[p] - options.saturation_knee)
                                    / options.saturation_width)
                                    .exp());
                        for group in 0..2 {
                            let var = planes.var[p * 2 + group];
                            let mut w = if valid { sat * ghost[p] / var } else { 0.0 };
                            if i == 0 && valid {
                                w = w.max(1e-3);
                            }
                            let expected_signal = if group == 0 {
                                (planes.x[p * 3].max(0.0).powi(2)
                                    + planes.x[p * 3 + 1].max(0.0).powi(2))
                                    * 0.5
                            } else {
                                planes.x[p * 3 + 2].max(0.0).powi(2)
                            };
                            let expected = expected_signal / var;
                            let better = valid && expected > best_snr[x][group];
                            for c in if group == 0 { 0..2 } else { 2..3 } {
                                let value = planes.x[p * 3 + c];
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
        if i == 0 && options.ghost_rejection {
            estimate = planes.x;
        }
        shortest_us = shortest_us.min(report.exposure_us[i]);
        report.frames_used += 1;
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
        ghost_maps,
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

    fn gaussian(seed: &mut u64) -> f32 {
        let mut uniform = || {
            *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((*seed >> 11) as f64 + 0.5) / ((1u64 << 53) as f64)
        };
        ((-2.0 * uniform().ln()).sqrt() * (std::f64::consts::TAU * uniform()).cos()) as f32
    }

    fn noisy_frame(
        index: usize,
        scale: f32,
        sigma: [f32; 3],
        k: [f32; 3],
        gain: f32,
        dx: f32,
    ) -> MergeFrame {
        let width = 256;
        let mut seed = 42 + index as u64;
        let mut f = frame((1000.0 / scale) as u32, 0.0);
        f.frame_index = index;
        f.width = width;
        f.height = width;
        f.raw_half_width = width / 2;
        f.raw_half_height = width / 2;
        f.raw_black = [5000.0; 3];
        f.raw_white = [55000.0; 3];
        f.sigma = sigma;
        f.raw_top = vec![0; (width * width) as usize];
        f.raw_half = vec![0; (width * width / 2) as usize];
        f.bmt = vec![0.0; (width * width * 3) as usize];
        f.gain = Some(vec![gain; f.bmt.len()]);
        for c in 0..3 {
            let divisor = if c == 2 { 1 } else { 2 };
            let size = width as usize / divisor;
            for y in 0..size {
                for x in 0..size {
                    let signal = (0.16 + 0.0001 * (x as f32 * divisor as f32 + dx)) / scale;
                    let sample =
                        signal + gaussian(&mut seed) * (sigma[c].powi(2) + k[c] * signal).sqrt();
                    let code = (5000.0 + sample * 50000.0).round() as u16;
                    if c == 2 {
                        f.raw_top[y * size + x] = code;
                    } else {
                        f.raw_half[(y * size + x) * 2 + c] = code;
                    }
                    for yy in y * divisor..(y + 1) * divisor {
                        for xx in x * divisor..(x + 1) * divisor {
                            let g = if gain == 1.0 {
                                gain
                            } else {
                                gain * (1.0 + 0.2 * xx as f32 / width as f32)
                            };
                            f.bmt[(yy * width as usize + xx) * 3 + c] = sample * g;
                            f.gain.as_mut().unwrap()[(yy * width as usize + xx) * 3 + c] = g;
                        }
                    }
                }
            }
        }
        f
    }

    #[test]
    fn pair_noise_fit_handles_read_noise_gain_and_fractional_shift() {
        let expected = [0.00008, 0.00012, 0.00016];
        for (sigma, gain, shift) in [
            ([0.001; 3], 1.0, 0.0),
            ([0.004; 3], 1.0, 0.0),
            ([0.001; 3], 2.5, 0.0),
            ([0.001; 3], 1.7, 2.35),
        ] {
            let reference = noisy_frame(0, 1.0, [0.001; 3], expected, gain, 0.0);
            let frame = noisy_frame(1, 0.5, sigma, expected, gain, shift);
            let alignment = Alignment {
                dx: shift,
                peak: 1.0,
                ..Default::default()
            };
            let (actual, counts) = noise_fit(
                &reference,
                &frame,
                &alignment,
                0.5,
                &MergeOptions::default(),
                Control::none(),
            )
            .unwrap();
            for c in 0..3 {
                assert!(counts[c] >= 50);
                assert!(
                    (actual[c] / expected[c] - 1.0).abs() < 0.2,
                    "plane={c} sigma={sigma:?} gain={gain} shift={shift}: {actual:?}"
                );
            }
        }
        let (actual, counts) = noise_fit(
            &frame(1000, 0.2),
            &frame(2000, 0.2),
            &Alignment::default(),
            0.5,
            &MergeOptions::default(),
            Control::none(),
        )
        .unwrap();
        assert_eq!(actual, MergeOptions::default().noise_k);
        assert!(counts.iter().all(|&n| n < 50));
    }

    #[test]
    fn ratio_refinement_uses_midtones_and_rejects_large_changes() {
        let load = |i: usize, response: f32| {
            let mut f = frame([1000, 2000][i], 0.1);
            f.width = 128;
            f.height = 128;
            f.raw_half_width = 64;
            f.raw_half_height = 64;
            f.raw_top = vec![f.raw_top[0]; 128 * 128];
            f.raw_half = vec![f.raw_half[0]; 128 * 128 / 2];
            f.bmt = vec![f.bmt[0]; 128 * 128 * 3];
            for p in 0..f.raw_top.len() {
                let level = if p < f.raw_top.len() / 2 { 0.07 } else { 0.13 };
                let value = level * if i == 0 { 1.0 } else { 2.0 };
                f.raw_top[p] = (value * 10000.0) as u16;
                f.bmt[p * 3..p * 3 + 3].fill(value);
            }
            if i == 1 {
                for value in &mut f.raw_top {
                    *value = (*value as f32 * response) as u16;
                }
                for value in &mut f.raw_half {
                    *value = (*value as f32 * response) as u16;
                }
                for value in &mut f.bmt {
                    *value *= response;
                }
            }
            // A bright pile outside the fit band must not set the scale.
            f.raw_top[..200].fill(if i == 0 { 8000 } else { 8200 });
            f.raw_half[..100].fill(if i == 0 { 8000 } else { 8200 });
            Ok(f)
        };
        let options = MergeOptions {
            align: AlignOptions {
                enabled: false,
                ..Default::default()
            },
            ghost_rejection: false,
            ..MergeOptions::default()
        };
        let scene = merge(2, |i| load(i, 1.05), &options, Control::none()).unwrap();
        assert!((scene.report.scales[1] - 0.5 / 1.05).abs() < 1e-5);
        let fixed = merge(
            2,
            |i| load(i, 1.05),
            &MergeOptions {
                fixed_ratios: true,
                ..options.clone()
            },
            Control::none(),
        )
        .unwrap();
        assert_eq!(fixed.report.scales[1], 0.5);
        let rejected = merge(2, |i| load(i, 1.5), &options, Control::none()).unwrap();
        assert_eq!(rejected.report.scales[1], 0.5);
        assert!(rejected
            .report
            .warnings()
            .iter()
            .any(|s| s.contains("0.3 EV")));
    }

    #[test]
    fn effective_variance_calibration_recovers_unequal_variances() {
        let options = MergeOptions {
            noise_k: [0.0; 3],
            fit_noise: false,
            ..MergeOptions::default()
        };
        let mut reference = noisy_frame(0, 1.0, [0.002; 3], [0.0; 3], 1.0, 0.0);
        let reference_planes = materialise(&mut reference, 1.0, &options, Control::none()).unwrap();
        retain_top_gain(&mut reference);
        let mut blocks = vec![Vec::new()];
        let expected = [1.0, 0.7, 1.0, 1.7];
        for (i, correction) in expected.iter().enumerate().skip(1) {
            let mut f = noisy_frame(i, 1.0, [0.003; 3], [0.0; 3], 1.0, 0.0);
            for y in 0..256 {
                for x in 0..256 {
                    let signal = 0.16 + 0.0001 * x as f32;
                    f.bmt[(y * 256 + x) * 3 + 2] =
                        signal + (f.bmt[(y * 256 + x) * 3 + 2] - signal) * f32::sqrt(*correction);
                }
            }
            let planes = materialise(&mut f, 1.0, &options, Control::none()).unwrap();
            retain_top_gain(&mut f);
            blocks.push(
                calibration_blocks(
                    (&reference, &reference_planes),
                    (&f, &planes),
                    &Alignment::default(),
                    1.0,
                    &options,
                    Control::none(),
                )
                .unwrap(),
            );
        }
        let actual = calibrate(&blocks, 0.0, false, options.variance_floor, 4.0);
        for i in 0..expected.len() {
            assert!(
                (actual[i] / expected[i] - 1.0).abs() < 0.25,
                "expected={expected:?} actual={actual:?}"
            );
        }
        assert_eq!(
            calibrate(&[vec![], vec![]], 0.0, false, options.variance_floor, 4.0),
            [1.0, 1.0]
        );
        let weights = ghost_weights(
            &reference_planes,
            &reference_planes.x,
            &reference_planes
                .var
                .chunks(2)
                .map(|v| [v[1]; 2])
                .collect::<Vec<_>>(),
            [1.0; 2],
            Control::none(),
        )
        .unwrap();
        assert!(weights.iter().all(|&v| v == 1.0));
    }

    #[test]
    fn calibration_excludes_blocks_near_either_black_floor() {
        let options = MergeOptions::default();
        for dark_reference in [true, false] {
            let mut a = noisy_frame(0, 1.0, [0.002; 3], [0.0; 3], 1.0, 0.0);
            let mut b = noisy_frame(1, 1.0, [0.002; 3], [0.0; 3], 1.0, 0.0);
            if dark_reference {
                a.sigma[2] = 0.03;
            } else {
                b.sigma[2] = 0.03;
            }
            let ap = materialise(&mut a, 1.0, &options, Control::none()).unwrap();
            let bp = materialise(&mut b, 1.0, &options, Control::none()).unwrap();
            retain_top_gain(&mut a);
            retain_top_gain(&mut b);
            assert!(calibration_blocks(
                (&a, &ap),
                (&b, &bp),
                &Alignment::default(),
                1.0,
                &options,
                Control::none()
            )
            .unwrap()
            .is_empty());
        }
    }

    #[test]
    fn calibration_keeps_reference_fixed_and_honours_floor_override() {
        let block = CalibrationBlock {
            residual: [0.4; 3],
            median_squared: [0.182; 3],
            reference: [1.0, 0.0],
            frame: [1.0, 0.0],
        };
        let blocks = [vec![], vec![block; 50]];
        let fit = calibrate(&blocks, 0.0, false, 0.1, 4.0);
        assert_eq!(fit, [1.0, 0.1]);
        assert_eq!(calibrate(&blocks, 0.0, false, 0.5, 4.0), [1.0, 0.5]);
    }

    #[test]
    fn chroma_uses_its_own_effective_variance() {
        let options = MergeOptions {
            // This fixture has independent noise but no registration texture.
            align: AlignOptions {
                enabled: false,
                ..Default::default()
            },
            fixed_ratios: true,
            fit_noise: false,
            noise_k: [0.0; 3],
            ..Default::default()
        };
        let scene = merge(
            3,
            |i| {
                let mut f = noisy_frame(i, 1.0, [0.002; 3], [0.0; 3], 1.0, 0.0);
                for y in 0..256 {
                    for x in 0..256 {
                        for c in 0..2 {
                            let signal = 0.16 + 0.0001 * (x / 2 * 2) as f32;
                            let value = &mut f.bmt[(y * 256 + x) * 3 + c];
                            *value = signal + (*value - signal) * 2.0_f32.sqrt();
                        }
                    }
                }
                Ok(f)
            },
            &options,
            Control::none(),
        )
        .unwrap();
        for i in 1..3 {
            let pair_chroma = scene.report.chroma_variance[0] + scene.report.chroma_variance[i];
            let pair_t = scene.report.effective_variance[0] + scene.report.effective_variance[i];
            assert!((pair_chroma / pair_t - 2.0).abs() < 0.3);
            for c in 0..3 {
                assert!((scene.report.median_z_squared[i][c].unwrap() - 0.455).abs() < 0.12);
            }
        }
    }

    #[test]
    fn moving_square_is_rejected_and_static_background_keeps_noise_benefit() {
        let load = |i| {
            let mut f = noisy_frame(i, 1.0, [0.003; 3], [0.0; 3], 1.0, 0.0);
            if i == 1 {
                for y in 80..120 {
                    for x in 80..120 {
                        f.bmt[(y * 256 + x) * 3..(y * 256 + x) * 3 + 3].fill(0.5);
                    }
                }
            }
            Ok(f)
        };
        let options = MergeOptions {
            align: AlignOptions {
                enabled: false,
                ..Default::default()
            },
            fixed_ratios: true,
            fit_noise: false,
            noise_k: [0.0; 3],
            ..MergeOptions::default()
        };
        let scene = merge(3, load, &options, Control::none()).unwrap();
        let reference = load(0).unwrap();
        let mut merged_noise = Vec::new();
        let mut reference_noise = Vec::new();
        for y in 0..256 {
            for x in 0..256 {
                let signal = 0.16 + 0.0001 * x as f32;
                let p = (y * 256 + x) * 3 + 2;
                if (80..120).contains(&x) && (80..120).contains(&y) {
                    assert!((scene.bmt[p] - signal).abs() < 0.015);
                } else {
                    merged_noise.push(scene.bmt[p] - signal);
                    reference_noise.push(reference.bmt[p] - signal);
                }
            }
        }
        assert!(variance(&merged_noise) < 0.8 * variance(&reference_noise));
        assert_eq!(scene.report.ghost_fractions[0], 0.0);
        assert!(scene.report.ghost_fractions[1] > 0.02);
        assert!(scene.report.ghost_fractions[1] > scene.report.ghost_fractions[2]);
        assert_eq!(scene.report.alignments.len(), 3);
        let unfiltered = merge(
            3,
            load,
            &MergeOptions {
                ghost_rejection: false,
                ..options
            },
            Control::none(),
        )
        .unwrap();
        assert!(unfiltered.bmt[(100 * 256 + 100) * 3 + 2] > 0.25);
    }

    fn crop_planes(f: &FramePlanes, x: usize, y: usize) -> FramePlanes {
        crop_region(f, x, y, 64, 64)
    }

    fn crop_region(
        f: &FramePlanes,
        x: usize,
        y: usize,
        width: usize,
        height: usize,
    ) -> FramePlanes {
        let mut patch = FramePlanes {
            width: width as u32,
            height: height as u32,
            ..Default::default()
        };
        for yy in y..y + height {
            for xx in x..x + width {
                let p = yy * f.width as usize + xx;
                patch.x.extend_from_slice(&f.x[p * 3..p * 3 + 3]);
                patch.var.extend_from_slice(&f.var[p * 2..p * 2 + 2]);
                patch.d_max.push(f.d_max[p]);
                patch.valid.push(f.valid[p]);
            }
        }
        patch
    }

    fn print_building_bins(
        reference: &FramePlanes,
        frame: &FramePlanes,
        levels: &[f32],
        read_snr: &[f32],
        report: &MergeReport,
        index: usize,
    ) {
        let above_black: Vec<_> = read_snr.iter().map(|&v| u8::from(v > 10.0)).collect();
        let reference_var: Vec<_> = reference
            .var
            .as_chunks::<2>()
            .0
            .iter()
            .map(|v| [v[1]; 2])
            .collect();
        let weights = ghost_weights_masked(
            frame,
            &reference.x,
            &reference_var,
            [
                report.chroma_variance[index],
                report.effective_variance[index],
            ],
            &above_black,
            3.0,
            Control::none(),
        )
        .unwrap();
        for bin in 0..10 {
            let (low, high) = ghost_bin_bounds(bin);
            let mut ratios = Vec::new();
            let mut valid = 0;
            let mut rejected = 0;
            let mut snr = 0.0;
            for p in 0..levels.len() {
                if levels[p] < low
                    || levels[p] >= high
                    || reference.valid[p] == 0
                    || frame.valid[p] == 0
                {
                    continue;
                }
                valid += 1;
                rejected += usize::from(weights[p] < 0.1);
                if read_snr[p] > 10.0 && frame.x[p * 3 + 2] > 0.0 {
                    ratios.push(
                        reference.x[p * 3 + 2] / frame.x[p * 3 + 2] * report.scales[index] as f32,
                    );
                    snr += read_snr[p];
                }
            }
            let count = ratios.len();
            let ratio = median(&mut ratios);
            eprintln!("dp0 static building=(200,100,2400,800) frame={index} bin={} lower={low:.5} upper={high:.5} ratio={ratio:?} pairs={count} mean_read_sigma={} valid={valid} ghost_fraction={}", bin + 1, snr / count.max(1) as f32, rejected as f32 / valid.max(1) as f32);
        }
    }

    fn patch_residual_statistics(
        reference: &FramePlanes,
        other: &FramePlanes,
        c_ref: [f32; 2],
        c_other: [f32; 2],
    ) -> ([f32; 3], [f32; 2]) {
        let reference_var: Vec<_> = reference
            .var
            .as_chunks::<2>()
            .0
            .iter()
            .map(|v| c_ref.map(|c| c * v[1]))
            .collect();
        let weights = ghost_weights(
            other,
            &reference.x,
            &reference_var,
            c_other,
            Control::none(),
        )
        .unwrap();
        let valid: Vec<_> = (0..reference.valid.len())
            .filter(|&p| reference.valid[p] != 0 && other.valid[p] != 0)
            .collect();
        assert!(!valid.is_empty());
        let z = std::array::from_fn(|c| {
            let group = usize::from(c == 2);
            let mut squared: Vec<_> = valid
                .iter()
                .map(|&p| {
                    (other.x[p * 3 + c] - reference.x[p * 3 + c]).powi(2)
                        / (c_other[group] * other.var[p * 2 + 1] + reference_var[p][group])
                })
                .collect();
            median(&mut squared).unwrap()
        });
        let fractions = [0.5, 0.1].map(|threshold| {
            valid.iter().filter(|&&p| weights[p] < threshold).count() as f32 / valid.len() as f32
        });
        (z, fractions)
    }

    fn print_patch_tails(
        label: &str,
        reference: &FramePlanes,
        frame: &FramePlanes,
        c_ref: [f32; 2],
        c_frame: [f32; 2],
    ) -> (usize, usize, usize) {
        let mut valid = 0;
        let mut counts = [[0; 2]; 3];
        let mut tail = vec![false; frame.valid.len()];
        let mut coordinates = Vec::new();
        for (p, is_tail) in tail.iter_mut().enumerate() {
            if reference.valid[p] == 0 || frame.valid[p] == 0 {
                continue;
            }
            valid += 1;
            for (c, channel_counts) in counts.iter_mut().enumerate() {
                let group = usize::from(c == 2);
                let z2 = (frame.x[p * 3 + c] - reference.x[p * 3 + c]).powi(2)
                    / (c_frame[group] * frame.var[p * 2 + 1]
                        + c_ref[group] * reference.var[p * 2 + 1]);
                for (i, threshold) in [9.0, -9.0 * 0.1_f32.ln()].into_iter().enumerate() {
                    channel_counts[i] += usize::from(z2 > threshold);
                }
                *is_tail |= z2 > -9.0 * 0.1_f32.ln();
            }
            if *is_tail {
                coordinates.push((p % 64, p / 64));
            }
        }
        let mut components = Vec::new();
        for p in 0..tail.len() {
            if !tail[p] {
                continue;
            }
            let mut pending = vec![p];
            tail[p] = false;
            let mut size = 0;
            while let Some(q) = pending.pop() {
                size += 1;
                let (x, y) = (q % 64, q / 64);
                for yy in y.saturating_sub(1)..=(y + 1).min(63) {
                    for xx in x.saturating_sub(1)..=(x + 1).min(63) {
                        let r = yy * 64 + xx;
                        if tail[r] {
                            tail[r] = false;
                            pending.push(r);
                        }
                    }
                }
            }
            components.push(size);
        }
        components.sort_unstable_by(|a, b| b.cmp(a));
        let clustered = components.iter().filter(|&&n| n >= 2).sum::<usize>();
        let isolated = components.iter().filter(|&&n| n == 1).count();
        assert_eq!(clustered + isolated, coordinates.len());
        let (z, ghost) = patch_residual_statistics(reference, frame, c_ref, c_frame);
        eprintln!("{label}: valid={valid} clustered_pixels={clustered} isolated_pixels={isolated} median_z2={z:?} z2_above_9_BMT={:?} z2_above_20_723_BMT={:?} union_tail={} components8={components:?} first64_coordinates={:?} ghost_below_tenth={}",
            counts.map(|c| c[0]), counts.map(|c| c[1]), coordinates.len(), &coordinates[..coordinates.len().min(64)], ghost[1]);
        (clustered, isolated, valid)
    }

    #[test]
    fn static_tail_components_separate_coherent_and_isolated_pixels() {
        let reference = FramePlanes {
            width: 64,
            height: 64,
            x: vec![0.0; 4096 * 3],
            var: vec![1.0; 4096 * 2],
            d_max: vec![0.0; 4096],
            valid: vec![1; 4096],
        };
        let mut frame = reference.clone();
        for p in [0, 65, 4095] {
            frame.x[p * 3 + 2] = 10.0;
        }
        assert_eq!(
            print_patch_tails(
                "synthetic components",
                &reference,
                &frame,
                [1.0; 2],
                [1.0; 2]
            ),
            (2, 1, 4096)
        );
    }

    #[test]
    fn gaussian_static_rejection_is_small_at_both_calibration_clamps() {
        let pixels = 256 * 256;
        for correction in [0.5_f32, 4.0] {
            let block = CalibrationBlock {
                residual: [1.0 + correction; 3],
                median_squared: [0.455 * (1.0 + correction); 3],
                reference: [1.0, 0.0],
                frame: [1.0, 0.0],
            };
            assert_eq!(
                calibrate(&[vec![], vec![block; 50]], 0.0, false, 0.5, 4.0),
                [1.0, correction]
            );
            let mut seed = 41;
            let reference: Vec<_> = (0..pixels * 3).map(|_| gaussian(&mut seed)).collect();
            let frame = FramePlanes {
                width: 256,
                height: 256,
                x: (0..pixels * 3)
                    .map(|_| gaussian(&mut seed) * correction.sqrt())
                    .collect(),
                var: vec![1.0; pixels * 2],
                d_max: vec![0.1; pixels],
                valid: vec![1; pixels],
            };
            let ghost = ghost_weights(
                &frame,
                &reference,
                &vec![[1.0; 2]; pixels],
                [correction; 2],
                Control::none(),
            )
            .unwrap();
            let fraction = ghost.iter().filter(|&&g| g < 0.1).count() as f32 / pixels as f32;
            assert!(fraction < 0.005, "c={correction} ghost={fraction}");
            for c in 0..3 {
                let mut z: Vec<_> = (0..pixels)
                    .map(|p| {
                        (frame.x[p * 3 + c] - reference[p * 3 + c]).powi(2) / (1.0 + correction)
                    })
                    .collect();
                let z = median(&mut z).unwrap();
                assert!(
                    (z - 0.455).abs() < 0.025,
                    "c={correction} channel={c} z2={z}"
                );
            }
        }
    }

    #[test]
    fn dark_pixels_keep_the_uncorrected_ghost_model() {
        let frame = FramePlanes {
            width: 4,
            height: 4,
            x: vec![3.0; 48],
            var: vec![1.0; 32],
            valid: vec![1; 16],
            d_max: vec![0.0; 16],
        };
        let estimate = vec![0.0; 48];
        let variance = vec![[1.0; 2]; 16];
        let uncorrected =
            ghost_weights(&frame, &estimate, &variance, [1.0; 2], Control::none()).unwrap();
        for correction in [0.5, 4.0] {
            assert_eq!(
                ghost_weights_masked(
                    &frame,
                    &estimate,
                    &variance,
                    [correction; 2],
                    &[0; 16],
                    3.0,
                    Control::none()
                )
                .unwrap(),
                uncorrected
            );
        }
    }

    #[test]
    fn static_dark_noise_has_calibrated_residuals_and_near_zero_rejection() {
        let make = |i| {
            let mut f = noisy_frame(i, 1.0, [0.002; 3], [0.0; 3], 1.0, 0.0);
            for y in 0..256 {
                for x in 0..256 {
                    for c in 0..3 {
                        // Keep only independent noise around a flat dark signal.
                        let xx = if c == 2 { x } else { x / 2 * 2 };
                        f.bmt[(y * 256 + x) * 3 + c] -= 0.14 + xx as f32 * 0.0001;
                    }
                }
            }
            f
        };
        let options = MergeOptions {
            // This fixture has independent noise but no registration texture.
            align: AlignOptions {
                enabled: false,
                ..Default::default()
            },
            fixed_ratios: true,
            fit_noise: false,
            noise_k: [0.0; 3],
            ..Default::default()
        };
        let scene = merge(3, |i| Ok(make(i)), &options, Control::none()).unwrap();
        for z in &scene.report.median_z_squared[1..] {
            for value in z {
                assert!((value.unwrap() - 0.455).abs() < 0.12, "{z:?}");
            }
        }
        assert!(scene.report.ghost_fractions.iter().all(|&v| v < 0.005));
        let mut a = make(0);
        let mut b = make(1);
        let a = materialise(&mut a, 1.0, &options, Control::none()).unwrap();
        let b = materialise(&mut b, 1.0, &options, Control::none()).unwrap();
        let (z, fractions) = patch_residual_statistics(
            &a,
            &b,
            [
                scene.report.chroma_variance[0],
                scene.report.effective_variance[0],
            ],
            [
                scene.report.chroma_variance[1],
                scene.report.effective_variance[1],
            ],
        );
        eprintln!(
            "synthetic static dark: median_z2={z:?} ghost_below_half={} ghost_below_tenth={}",
            fractions[0], fractions[1]
        );
        assert!(fractions[1] < 0.005);
    }

    fn detrended_variance(values: &[f32], width: usize) -> f32 {
        let height = values.len() / width;
        let mean = values.iter().map(|&v| f64::from(v)).sum::<f64>() / values.len() as f64;
        let mut xx = 0.0;
        let mut yy = 0.0;
        let mut xv = 0.0;
        let mut yv = 0.0;
        for (p, &v) in values.iter().enumerate() {
            let x = (p % width) as f64 - (width - 1) as f64 / 2.0;
            let y = (p / width) as f64 - (height - 1) as f64 / 2.0;
            xx += x * x;
            yy += y * y;
            xv += x * (f64::from(v) - mean);
            yv += y * (f64::from(v) - mean);
        }
        let residual: Vec<_> = values
            .iter()
            .enumerate()
            .map(|(p, &v)| {
                (f64::from(v)
                    - mean
                    - xv / xx * ((p % width) as f64 - (width - 1) as f64 / 2.0)
                    - yv / yy * ((p / width) as f64 - (height - 1) as f64 / 2.0))
                    as f32
            })
            .collect();
        variance(&residual)
    }

    #[test]
    fn plane_detrending_preserves_noise_and_removes_gradient() {
        let flat: Vec<_> = (0..4096)
            .map(|p| {
                if (p % 64 + p / 64) % 2 == 0 {
                    0.01
                } else {
                    -0.01
                }
            })
            .collect();
        let tilted: Vec<_> = flat
            .iter()
            .enumerate()
            .map(|(p, v)| v + (p % 64) as f32 * 0.3 + (p / 64) as f32 * 0.7)
            .collect();
        assert!((detrended_variance(&tilted, 64) / variance(&flat) - 1.0).abs() < 0.001);
    }

    #[test]
    fn corpus_phase2_statistics_and_shadow_noise() {
        let corpus = std::env::var_os("X3F_TEST_FILES")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| {
                std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../x3f_test_files")
            });
        let mut failures = Vec::new();
        for name in ["DP0Q0002.X3I", "_DQH0978.X3I"] {
            let path = corpus.join(name);
            if !path.is_file() {
                eprintln!("skip: missing {name}");
                continue;
            }
            let container = crate::X3iContainer::open(path).unwrap();
            let mut options = MergeOptions::default();
            options.process.denoise_intensity = 0;
            options.saturation = Some(options.saturation_for_container(&container).unwrap());
            let mut green = Vec::new();
            let mut dark_patch = (0, 0, Vec::new());
            let start = std::time::Instant::now();
            let scene = merge(
                container.frames().len(),
                |i| {
                    let mut f = container
                        .open_frame(i)?
                        .merge_frame(&options.process, Control::none())?;
                    f.frame_index = i;
                    if i == 0 {
                        green = f
                            .bmt
                            .chunks(3)
                            .enumerate()
                            .filter(|(_, p)| (0.08..=0.11).contains(&p[0]))
                            .step_by(8)
                            .map(|(i, p)| (i, [p[0], p[1], p[2]]))
                            .collect();
                        let width = f.width as usize;
                        let mut best_gradient = f32::INFINITY;
                        for y in
                            (f.height as usize / 10..f.height as usize * 9 / 10 - 64).step_by(64)
                        {
                            for x in (width / 10..width * 9 / 10 - 64).step_by(64) {
                                let patch: Vec<_> = (0..64)
                                    .flat_map(|dy| (0..64).map(move |dx| (y + dy) * width + x + dx))
                                    .map(|p| f.bmt[p * 3 + 2])
                                    .collect();
                                let mean = patch.iter().sum::<f32>() / patch.len() as f32;
                                // Keep the shortest exposure clear of the encoded black floor.
                                if !(0.02..=0.05).contains(&mean) {
                                    continue;
                                }
                                let mut gradient = 0.0;
                                for yy in 0..64 {
                                    for xx in 0..64 {
                                        let p = yy * 64 + xx;
                                        if xx > 0 { gradient += (patch[p] - patch[p - 1]).powi(2); }
                                        if yy > 0 { gradient += (patch[p] - patch[p - 64]).powi(2); }
                                    }
                                }
                                if gradient < best_gradient {
                                    best_gradient = gradient;
                                    dark_patch = (x, y, patch);
                                }
                            }
                        }
                    }
                    let (px, py, _) = &dark_patch;
                    let indices: Vec<_> = (0..64).flat_map(|dy| (0..64).map(move |dx| (py + dy) * f.width as usize + px + dx)).collect();
                    let raw_t: Vec<_> = indices.iter().map(|&p| raw(&f, 2, p)).collect();
                    let gained: Vec<_> = indices.iter().map(|&p| raw(&f, 2, p) * f.gain.as_ref().map_or(1.0, |g| g[p * 3 + 2])).collect();
                    let exported: Vec<_> = indices.iter().map(|&p| f.bmt[p * 3 + 2]).collect();
                    eprintln!("{name}: frame={i} patch raw_T_variance={} gain_T_variance={} exported_T_variance={} sigma_T={} raw_mean={} exported_mean={}", variance(&raw_t), variance(&gained), variance(&exported), f.sigma[2], raw_t.iter().sum::<f32>()/4096.0, exported.iter().sum::<f32>()/4096.0);
                    Ok(f)
                },
                &options,
                Control::none(),
            )
            .unwrap();
            eprintln!(
                "{name}: elapsed={:.3}s report={:?}",
                start.elapsed().as_secs_f64(),
                scene.report
            );
            for line in scene.report.statistics() {
                eprintln!("{name}: {line}");
            }
            assert_eq!(scene.report.frames_used, 7);
            assert_eq!(scene.report.scales.len(), 7);
            assert!(scene.bmt.iter().all(|v| v.is_finite()));
            for (scale, nominal) in scene.report.scales.iter().zip(&scene.report.nominal_scales) {
                assert!((scale / nominal).log2().abs() <= 0.3);
            }
            if name == "_DQH0978.X3I" {
                for i in 1..=3 {
                    let fit = &scene.report.aligned_ratios[i];
                    for bin in 3..=5 {
                        let clean =
                            f64::from(fit.bins[bin].unwrap()) * scene.report.nominal_scales[i];
                        if (scene.report.scales[i] / clean - 1.0).abs() >= 0.003 {
                            failures.push(format!(
                                "H frame {i} bin {} scale={} clean={clean}",
                                bin + 1,
                                scene.report.scales[i]
                            ));
                        }
                    }
                }
            }
            for c in 0..3 {
                let mut merged: Vec<_> = green.iter().map(|(p, _)| scene.bmt[p * 3 + c]).collect();
                let mut reference: Vec<_> = green.iter().map(|(_, v)| v[c]).collect();
                let ratio = median(&mut merged).unwrap() / median(&mut reference).unwrap();
                eprintln!("{name}: green-band plane={c} median_ratio={ratio:.8}");
                if name == "_DQH0978.X3I" {
                    assert!((ratio - 1.0).abs() < 0.01, "green plane {c}: {ratio}");
                }
            }
            let (x, y, reference) = dark_patch;
            assert_eq!(reference.len(), 4096);
            let merged: Vec<_> = (0..64)
                .flat_map(|dy| (0..64).map(move |dx| (y + dy) * scene.width as usize + x + dx))
                .map(|p| scene.bmt[p * 3 + 2])
                .collect();
            let ratio = (variance(&merged) / variance(&reference)).sqrt();
            let detrended =
                (detrended_variance(&merged, 64) / detrended_variance(&reference, 64)).sqrt();
            eprintln!("{name}: dark_patch=({x},{y},64,64) T_stddev_ratio={ratio:.8} detrended_ratio={detrended:.8}");
            // Remove the fitted illumination plane before comparing noise in the selected flat block.
            assert!(
                detrended < 0.8,
                "{name}: selected patch detrended noise ratio {detrended}"
            );

            let fitted = MergeOptions {
                noise_k: scene.report.noise_k,
                ..options
            };
            let old_patch = if name == "DP0Q0002.X3I" {
                (588, 2671)
            } else {
                (5721, 960)
            };
            let mut old_planes = Vec::new();
            let mut building_reference = None;
            let mut building_levels = Vec::new();
            // Warp the full raster before cropping, so the patch uses the merge coordinates.
            let (planes, patch_read_snr): (Vec<_>, Vec<_>) = (0..container.frames().len())
                .map(|i| {
                    let mut frame = container
                        .open_frame(i)
                        .unwrap()
                        .merge_frame(&fitted.process, Control::none())
                        .unwrap();
                    let alignment = &scene.report.alignments[i];
                    let mut raw_sum = 0.0;
                    let mut count = 0;
                    for yy in y..y + 64 {
                        for xx in x..x + 64 {
                            if let Some(p) = paired(
                                yy * scene.width as usize + xx,
                                scene.width as usize,
                                scene.height as usize,
                                alignment,
                                1,
                            ) {
                                raw_sum += raw(&frame, 2, p);
                                count += 1;
                            }
                        }
                    }
                    let read_snr =
                        raw_sum / count as f32 / fitted.noise_sigma.unwrap_or(frame.sigma[2]);
                    let building_snr: Vec<_> = if name == "DP0Q0002.X3I" && i <= 3 {
                        (100..900)
                            .flat_map(|yy| {
                                (200..2600).map(move |xx| yy * scene.width as usize + xx)
                            })
                            .map(|p| {
                                if i == 0 {
                                    building_levels
                                        .push(raw(&frame, 2, p) / fitted.saturation.unwrap()[2]);
                                }
                                paired(p, scene.width as usize, scene.height as usize, alignment, 1)
                                    .map_or(0.0, |q| raw(&frame, 2, q) / frame.sigma[2])
                            })
                            .collect()
                    } else {
                        Vec::new()
                    };
                    let planes = materialise(
                        &mut frame,
                        scene.report.scales[i] as f32,
                        &fitted,
                        Control::none(),
                    )
                    .unwrap();
                    drop(frame);
                    let planes =
                        x3i_align::warp(planes, &scene.report.alignments[i], Control::none())
                            .unwrap();
                    if name == "DP0Q0002.X3I" && i <= 3 {
                        let building = crop_region(&planes, 200, 100, 2400, 800);
                        if i == 0 {
                            building_reference = Some(building);
                        } else {
                            print_building_bins(
                                building_reference.as_ref().unwrap(),
                                &building,
                                &building_levels,
                                &building_snr,
                                &scene.report,
                                i,
                            );
                        }
                    }
                    old_planes.push(crop_planes(&planes, old_patch.0, old_patch.1));
                    (crop_planes(&planes, x, y), read_snr)
                })
                .unzip();
            let old_reference: Vec<_> = old_planes[0]
                .x
                .as_chunks::<3>()
                .0
                .iter()
                .map(|p| p[2])
                .collect();
            let old_merged: Vec<_> = (0..64)
                .flat_map(|dy| {
                    (0..64)
                        .map(move |dx| (old_patch.1 + dy) * scene.width as usize + old_patch.0 + dx)
                })
                .map(|p| scene.bmt[p * 3 + 2])
                .collect();
            let old_ratio = (variance(&old_merged) / variance(&old_reference)).sqrt();
            eprintln!("{name}: original dark_patch={old_patch:?} T_stddev_ratio={old_ratio}");
            if old_ratio >= 0.8 {
                failures.push(format!(
                    "{name}: original shadow patch noise ratio={old_ratio}"
                ));
            }
            for i in 1..planes.len() {
                let (z, ghost) = patch_residual_statistics(
                    &planes[0],
                    &planes[i],
                    [
                        scene.report.chroma_variance[0],
                        scene.report.effective_variance[0],
                    ],
                    [
                        scene.report.chroma_variance[i],
                        scene.report.effective_variance[i],
                    ],
                );
                eprintln!("{name}: static patch frame={i} median_z2={z:?} ghost_below_half={} ghost_below_tenth={}", ghost[0], ghost[1]);
                eprintln!(
                    "{name}: static patch frame={i} mean_over_read_sigma={}",
                    patch_read_snr[i]
                );
            }
            let correction = |i| {
                [
                    scene.report.chroma_variance[i],
                    scene.report.effective_variance[i],
                ]
            };
            for i in 1..old_planes.len() {
                let (z, ghost) = patch_residual_statistics(
                    &old_planes[0],
                    &old_planes[i],
                    correction(0),
                    correction(i),
                );
                eprintln!("{name}: old static patch={old_patch:?} frame={i} median_z2={z:?} ghost_below_half={} ghost_below_tenth={}", ghost[0], ghost[1]);
            }
            if name == "_DQH0978.X3I" {
                for j in [0, 2, 3] {
                    print_patch_tails(
                        &format!("{name}: old patch frame=1 against={j}"),
                        &old_planes[j],
                        &old_planes[1],
                        correction(j),
                        correction(1),
                    );
                }
            }
            for i in 1..planes.len() {
                let (clustered, isolated, valid) = print_patch_tails(
                    &format!("{name}: selected patch frame={i} against=0"),
                    &planes[0],
                    &planes[i],
                    correction(0),
                    correction(i),
                );
                let trusted = !scene.report.weak_alignments[i]
                    && correction(i)
                        .iter()
                        .all(|&c| c >= fitted.variance_floor && c < 4.0);
                let above_black = patch_read_snr[i] > 10.0;
                let coherent_fraction = clustered as f32 / valid as f32;
                let isolated_fraction = isolated as f32 / valid as f32;
                eprintln!("{name}: static assertion frame={i} trusted={trusted} above_black={above_black} coherent_fraction={coherent_fraction} isolated_fraction={isolated_fraction}");
                // Weak alignment and upper-clamped calibration remain diagnostic only.
                // Include the lower clamp because it gives conservative rejection.
                // Below ten read sigmas, the zero clamp makes near-black noise non-Gaussian.
                // Test coherent rejection before the 3x3 minimum: components of two or more.
                // Isolated tails are defect-like; bound them separately at 0.5 percent.
                // Report corpus median z2; the synthetic Gaussian test checks its expected value.
                if name == "_DQH0978.X3I" {
                    assert!(
                        trusted && above_black,
                        "H frame {i} must exercise the static bounds"
                    );
                }
                if trusted && above_black {
                    if coherent_fraction >= 0.01 {
                        failures.push(format!(
                            "{name}: static patch frame={i}: coherent fraction={coherent_fraction}"
                        ));
                    }
                    if isolated_fraction > 0.005 {
                        failures.push(format!(
                            "{name}: static patch frame={i}: isolated fraction={isolated_fraction}"
                        ));
                    }
                }
            }
        }
        assert!(failures.is_empty(), "{failures:?}");
    }

    #[test]
    fn shortest_valid_frame_seeds_ghost_estimate_outside_reference_coverage() {
        let options = MergeOptions {
            align: AlignOptions {
                enabled: false,
                ..Default::default()
            },
            fixed_ratios: true,
            fit_noise: false,
            ..MergeOptions::default()
        };
        let scene = merge(
            3,
            |i| {
                let mut f = frame([1000, 250, 125][i], 2.0);
                if i == 1 {
                    f.bmt.fill(0.7);
                }
                Ok(f)
            },
            &options,
            Control::none(),
        )
        .unwrap();
        assert!(scene.bmt.iter().all(|v| (*v - 2.0).abs() < 0.001));
        assert_eq!(scene.report.ghost_fractions[2], 0.0);
        assert_eq!(scene.report.ghost_fractions[1], 1.0);
        assert_eq!(scene.report.fallback_zero, 0);
        let lines = scene.report.statistics();
        assert_eq!(lines.len(), 61);
        assert!(lines[0].contains("k_B="));
        assert!(lines[1].contains("nominal=1.000000000000 refined=1.000000000000"));
        assert!(lines[20].contains("dx=0.000000"));
        assert!(lines[20].contains("weak=false"));
        assert!(scene.report.weak_alignments.iter().all(|&weak| !weak));
        assert_eq!(scene.report.ghost_sigmas, [0.0, 3.0, 3.0]);
    }

    #[test]
    fn integer_pairing_uses_source_to_reference_shift_and_excludes_borders() {
        assert_eq!(
            paired(
                5 * 16 + 7,
                16,
                16,
                &Alignment {
                    dx: 2.0,
                    dy: -1.0,
                    ..Default::default()
                },
                1
            ),
            Some(6 * 16 + 5)
        );
        assert_eq!(
            paired(
                5 * 16 + 1,
                16,
                16,
                &Alignment {
                    dx: 2.0,
                    ..Default::default()
                },
                1
            ),
            None
        );
        assert_eq!(
            paired(
                15 * 16 + 7,
                16,
                16,
                &Alignment {
                    dy: -1.0,
                    ..Default::default()
                },
                1
            ),
            None
        );
    }

    #[test]
    fn raw_pairs_follow_similarity_and_skipped_alignment() {
        let alignment = Alignment {
            dx: 10.0,
            dy: 8.0,
            similarity: Some([1.0, -0.01, 12.0, 0.01, 1.0, 3.0]),
            ..Default::default()
        };
        for divisor in [1, 2] {
            let width = 2048 / divisor;
            for p in [0, width - 1, width * (width - 1), width * width - 1] {
                assert_eq!(
                    paired(p, width, width, &Alignment::default(), divisor),
                    Some(p)
                );
            }
            for (x, y) in [(100, 100), (900, 100), (100, 900), (900, 900)] {
                let centre = (divisor - 1) as f32 * 0.5;
                let u = (x * divisor) as f32 + centre - 12.0;
                let v = (y * divisor) as f32 + centre - 3.0;
                let sx = ((u + 0.01 * v) / 1.0001 - centre) / divisor as f32;
                let sy = ((v - 0.01 * u) / 1.0001 - centre) / divisor as f32;
                let expected = sy.round() as usize * width + sx.round() as usize;
                assert_eq!(
                    paired(y * width + x, width, width, &alignment, divisor),
                    Some(expected)
                );
            }
        }
        let skipped = Alignment {
            skipped: true,
            ..alignment
        };
        assert_eq!(paired(0, 2048, 2048, &skipped, 1), Some(0));
    }

    #[test]
    fn flat_blocks_exclude_high_frequency_texture() {
        let checker: [f32; 64] =
            std::array::from_fn(|p| if (p / 8 + p % 8) % 2 == 0 { 0.1 } else { 0.3 });
        assert!(!flat(&checker, 1e-5));
        assert!(flat(&[0.2; 64], 1e-5));
    }

    #[test]
    fn pooled_support_without_a_supported_bin_keeps_the_nominal_ratio() {
        let options = MergeOptions {
            align: AlignOptions {
                enabled: false,
                ..Default::default()
            },
            ..Default::default()
        };
        let scene = merge(
            2,
            |i| {
                let mut f = noisy_frame(i, 1.0, [0.0; 3], [0.0; 3], 1.0, 0.0);
                f.height = 16;
                f.raw_half_height = 8;
                f.raw_top.truncate(256 * 16);
                f.raw_half.truncate(128 * 8 * 2);
                f.bmt.truncate(256 * 16 * 3);
                f.gain = None;
                for p in 0..f.raw_top.len() {
                    let level = if p < f.raw_top.len() / 2 { 0.08 } else { 0.16 };
                    let value = level * if i == 0 { 1.0 } else { 1.01 };
                    f.raw_top[p] = (5000.0 + value * 50000.0) as u16;
                    f.bmt[p * 3..p * 3 + 3].fill(value);
                }
                Ok(f)
            },
            &options,
            Control::none(),
        )
        .unwrap();
        let fit = &scene.report.aligned_ratios[1];
        assert!(fit.pairs >= fit.minimum_pairs);
        assert!(fit.bin_pairs.iter().all(|&n| n < fit.minimum_pairs));
        assert_eq!(scene.report.scales[1], 1.0);
        assert!(scene
            .report
            .warnings()
            .iter()
            .any(|w| w.contains("no supported level bin")));
    }

    #[test]
    fn ratio_bins_exclude_frame_samples_near_its_black_floor() {
        let options = MergeOptions::default();
        let mut reference = noisy_frame(0, 1.0, [0.0; 3], [0.0; 3], 1.0, 0.0);
        let mut other = noisy_frame(1, 8.0, [0.0; 3], [0.0; 3], 1.0, 0.0);
        // Keep raw and exported values unchanged; only the frame's noise floor varies.
        other.sigma[2] = 0.003;
        let a = materialise(&mut reference, 1.0, &options, Control::none()).unwrap();
        let b = materialise(&mut other, 8.0, &options, Control::none()).unwrap();
        let fit = refined_ratio(
            (&reference, &other),
            &a,
            &b,
            &Alignment::default(),
            8.0,
            &options,
            Control::none(),
        )
        .unwrap();
        assert_eq!(fit.pairs, 0);
        assert_eq!(fit.bin_pairs, [0; 8]);
        assert_eq!(fit.multiplier, None);
        other.sigma[2] = 0.001;
        let fit = refined_ratio(
            (&reference, &other),
            &a,
            &b,
            &Alignment::default(),
            8.0,
            &options,
            Control::none(),
        )
        .unwrap();
        assert!(fit.pairs >= fit.minimum_pairs);
        assert_eq!(fit.multiplier, None); // One supported bin cannot establish a level trend.
    }

    #[test]
    fn bin_fit_keeps_affine_offset_diagnostic_only() {
        let options = MergeOptions::default();
        let mut reference = noisy_frame(0, 1.0, [0.0; 3], [0.0; 3], 1.0, 0.0);
        let mut frame = reference.clone();
        for p in 0..reference.raw_top.len() {
            let level = match p % 10 {
                0..=6 => 0.08,
                7 => 0.16,
                8 => 0.30,
                _ => 0.45,
            };
            let value = (level - 0.003) / 1.02;
            reference.raw_top[p] = (5000.0 + level * 50000.0) as u16;
            frame.raw_top[p] = (5000.0 + value * 50000.0) as u16;
            reference.bmt[p * 3..p * 3 + 3].fill(level);
            frame.bmt[p * 3..p * 3 + 3].fill(value);
        }
        let a = materialise(&mut reference, 1.0, &options, Control::none()).unwrap();
        let b = materialise(&mut frame, 1.0, &options, Control::none()).unwrap();
        let fit = refined_ratio(
            (&reference, &frame),
            &a,
            &b,
            &Alignment::default(),
            1.0,
            &options,
            Control::none(),
        )
        .unwrap();
        let mut clean: Vec<_> = fit.bins.iter().flatten().copied().collect();
        assert_eq!(fit.multiplier, median(&mut clean));
        assert!((fit.black_offset.unwrap() - 0.003).abs() < 1e-5);
        assert!(fit.level_trend.unwrap() > 0.02);
        frame.sigma[2] = 0.01;
        let dark = refined_ratio(
            (&reference, &frame),
            &a,
            &b,
            &Alignment::default(),
            1.0,
            &options,
            Control::none(),
        )
        .unwrap();
        assert_eq!(dark.clean_bins.iter().filter(|&&v| v).count(), 1);
        assert_eq!(dark.multiplier, None);
    }

    #[test]
    fn ratio_support_and_level_trend_are_reported() {
        let options = MergeOptions::default();
        let mut reference = frame(1000, 0.1);
        let mut other = frame(2000, 0.1);
        let a = materialise(&mut reference, 1.0, &options, Control::none()).unwrap();
        let b = materialise(&mut other, 0.5, &options, Control::none()).unwrap();
        let fit = refined_ratio(
            (&reference, &other),
            &a,
            &b,
            &Alignment::default(),
            0.5,
            &options,
            Control::none(),
        )
        .unwrap();
        assert!(fit.pairs < 1000);
        assert_eq!(fit.multiplier, None);

        let make = |i| {
            let mut f = noisy_frame(
                i,
                if i == 0 { 1.0 } else { 0.5 },
                [0.0; 3],
                [0.0; 3],
                1.0,
                0.0,
            );
            for p in 0..f.raw_top.len() {
                let level = [0.06, 0.13, 0.19][p * 3 / f.raw_top.len()];
                let value = if i == 0 {
                    level
                } else {
                    level * 2.0 * if level == 0.13 { 0.94 } else { 1.0 }
                };
                f.raw_top[p] = (5000.0 + value * 50000.0) as u16;
                f.bmt[p * 3..p * 3 + 3].fill(value);
            }
            f
        };
        let mut reference = make(0);
        let mut other = make(1);
        let a = materialise(&mut reference, 1.0, &options, Control::none()).unwrap();
        let b = materialise(&mut other, 0.5, &options, Control::none()).unwrap();
        let fit = refined_ratio(
            (&reference, &other),
            &a,
            &b,
            &Alignment::default(),
            0.5,
            &options,
            Control::none(),
        )
        .unwrap();
        assert!(fit.bin_pairs[0] >= fit.minimum_pairs && fit.bin_pairs[1] >= fit.minimum_pairs);
        assert!(fit.level_trend.unwrap() > 0.02);
        let rejected = merge(2, |i| Ok(make(i)), &options, Control::none()).unwrap();
        assert_eq!(rejected.report.scales[1], 0.5);
        assert!(rejected
            .report
            .warnings()
            .iter()
            .any(|w| w.contains("level bins differ")));
        let accepted = merge(
            2,
            |i| Ok(make(i)),
            &MergeOptions {
                ratio_trend_limit: 0.20,
                ..options
            },
            Control::none(),
        )
        .unwrap();
        assert!(!accepted
            .report
            .warnings()
            .iter()
            .any(|w| w.contains("level bins differ")));
    }

    fn phase1_options() -> MergeOptions {
        MergeOptions {
            align: AlignOptions {
                enabled: false,
                ..Default::default()
            },
            fixed_ratios: true,
            fit_noise: false,
            ghost_rejection: false,
            ..MergeOptions::default()
        }
    }

    #[test]
    fn ghost_controls_reject_invalid_thresholds_and_clamps() {
        let controls = [
            None,
            None,
            None,
            None,
            Some("0.2"),
            Some("2"),
            Some("2.5"),
            Some("1.5"),
        ];
        let options = MergeOptions::parse_env(None, None, None, None, controls).unwrap();
        assert_eq!(options.variance_ceiling, 2.0);
        assert_eq!(options.ghost_sigma, 2.5);
        assert_eq!(options.weak_ghost_sigma, Some(1.5));
        for index in 5..8 {
            for invalid in ["", "bad", "NaN", "inf", "-1", "0", "1e-30", "1e30"] {
                let mut values = controls;
                values[index] = Some(invalid);
                // Very small and large C_MAX values are valid if they exceed C_MIN.
                if index == 5 && invalid == "1e30" {
                    continue;
                }
                assert!(matches!(
                    MergeOptions::parse_env(None, None, None, None, values),
                    Err(Error::InvalidData(_))
                ));
            }
        }
        let inherited = MergeOptions::parse_env(
            None,
            None,
            None,
            None,
            [None, None, None, None, None, None, Some("2.5"), None],
        )
        .unwrap();
        assert_eq!(inherited.weak_ghost_sigma, None);
        assert_eq!(MergeOptions::default().weak_ghost_sigma, None);
        let mut below_floor = controls;
        below_floor[5] = Some("0.1");
        assert!(matches!(
            MergeOptions::parse_env(None, None, None, None, below_floor),
            Err(Error::InvalidData(_))
        ));
        below_floor[5] = Some("0.2");
        assert!(MergeOptions::parse_env(None, None, None, None, below_floor).is_ok());
    }

    #[test]
    fn weak_alignment_reads_only_consensus_residual_and_skip() {
        let mut alignment = Alignment {
            retained_tiles: 5,
            max_tile_residual: 1.5,
            ..Default::default()
        };
        assert!(!weak_alignment(&alignment));
        alignment.retained_tiles = 4;
        assert!(weak_alignment(&alignment));
        alignment.retained_tiles = 5;
        alignment.max_tile_residual = 1.5001;
        assert!(weak_alignment(&alignment));
        alignment.max_tile_residual = 0.0;
        alignment.skipped = true;
        assert!(weak_alignment(&alignment));
    }

    #[test]
    fn weak_threshold_requires_enabled_alignment_and_rejection() {
        let mut options = MergeOptions {
            ghost_sigma: 2.5,
            weak_ghost_sigma: Some(1.5),
            ..Default::default()
        };
        let load = |_| Ok(frame(1000, 0.1));
        let skipped = merge(2, load, &options, Control::none()).unwrap();
        assert!(skipped.report.alignments[1].skipped);
        assert!(skipped.report.weak_alignments[1]);
        assert_eq!(skipped.report.ghost_sigmas, [0.0, 1.5]);
        options.weak_ghost_sigma = None;
        options.ghost_sigma = 4.0;
        let inherited = merge(2, load, &options, Control::none()).unwrap();
        assert_eq!(inherited.report.ghost_sigmas, [0.0, 4.0]);
        options.ghost_sigma = 2.5;
        options.weak_ghost_sigma = Some(1.5);
        options.align.enabled = false;
        let disabled = merge(2, load, &options, Control::none()).unwrap();
        assert_eq!(disabled.report.weak_alignments, [false, false]);
        assert_eq!(disabled.report.ghost_sigmas, [0.0, 2.5]);
        options.ghost_rejection = false;
        let no_ghost = merge(2, load, &options, Control::none()).unwrap();
        assert_eq!(no_ghost.report.ghost_sigmas, [0.0, 0.0]);
    }

    #[test]
    fn variance_ceiling_bounds_fits_and_low_support_fallbacks() {
        let block = CalibrationBlock {
            residual: [10.0; 3],
            median_squared: [1.0; 3],
            reference: [1.0, 0.0],
            frame: [1.0, 0.0],
        };
        assert_eq!(
            calibrate(&[vec![], vec![block; 50]], 0.0, false, 0.5, 0.75),
            [1.0, 0.75]
        );
        assert_eq!(
            calibrate(&[vec![], vec![]], 0.0, false, 0.5, 0.75),
            [1.0, 0.75]
        );
    }

    #[test]
    fn ghost_map_encodes_final_rejection_static_and_invalid_pixels() {
        let mut frame = FramePlanes {
            width: 16,
            height: 16,
            x: vec![0.1; 16 * 16 * 3],
            var: vec![1e-6; 16 * 16 * 2],
            d_max: vec![0.1; 16 * 16],
            valid: vec![1; 16 * 16],
        };
        frame.x[(8 * 16 + 8) * 3..(8 * 16 + 8) * 3 + 3].fill(0.5);
        frame.valid[0] = 0;
        let weights = ghost_weights_masked(
            &frame,
            &vec![0.1; 16 * 16 * 3],
            &vec![[1e-6; 2]; 16 * 16],
            [1.0; 2],
            &[],
            3.0,
            Control::none(),
        )
        .unwrap();
        let map = ghost_map(&frame, &weights, 5);
        assert_eq!((map.frame_index, map.width, map.height), (5, 16, 16));
        assert_eq!(map.data.len(), 256);
        assert_eq!(map.data[0], 128);
        assert_eq!(map.data[2 * 16 + 2], 255);
        for y in 7..=9 {
            for x in 7..=9 {
                assert_eq!(map.data[y * 16 + x], 0);
            }
        }
        let middle = ghost_map(&frame, &vec![128.0 / 255.0; 256], 5);
        assert_eq!(middle.data.iter().filter(|&&v| v == 128).count(), 1);
    }

    #[test]
    fn ghost_maps_do_not_change_merge_bits_or_reports() {
        let mut options = phase1_options();
        options.ghost_rejection = true;
        let load = |i| {
            let mut f = frame(1000, 0.1);
            f.frame_index = i;
            if i == 1 {
                f.bmt[15] = 0.5;
            }
            Ok(f)
        };
        let off = merge(2, load, &options, Control::none()).unwrap();
        assert!(off.ghost_maps.is_empty());
        options.ghost_maps = true;
        let on = merge(2, load, &options, Control::none()).unwrap();
        assert_eq!(
            off.bmt.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            on.bmt.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        assert_eq!(off.report, on.report);
        assert_eq!(off.s, on.s);
        assert_eq!(on.ghost_maps.len(), 1);
        assert_eq!(on.ghost_maps[0].data.len(), (on.width * on.height) as usize);
    }

    #[test]
    fn phase2_overrides_parse_without_environment_mutation() {
        let options = MergeOptions::parse_env(
            None,
            None,
            None,
            None,
            [
                Some("1"),
                Some("1"),
                Some("1"),
                Some("0.03"),
                Some("0.2"),
                None,
                None,
                None,
            ],
        )
        .unwrap();
        assert!(!options.align.enabled && options.fixed_ratios && !options.ghost_rejection);
        assert_eq!(options.ratio_trend_limit, 0.03);
        assert_eq!(options.variance_floor, 0.2);
        for floor in ["0", "-1", "1.1", "nan", "inf", "bad"] {
            assert!(MergeOptions::parse_env(
                None,
                None,
                None,
                None,
                [None, None, None, None, Some(floor), None, None, None]
            )
            .is_err());
        }
        let defaults = MergeOptions::parse_env(
            None,
            None,
            None,
            None,
            [
                Some("0"),
                Some("0"),
                Some("0"),
                None,
                None,
                None,
                None,
                None,
            ],
        )
        .unwrap();
        assert!(defaults.align.enabled && !defaults.fixed_ratios && defaults.ghost_rejection);
        assert_eq!(MergeOptions::default().ratio_trend_limit, 0.02);
        for limit in ["nan", "inf", "-0.1", "bad"] {
            assert!(MergeOptions::parse_env(
                None,
                None,
                None,
                None,
                [None, None, None, Some(limit), None, None, None, None]
            )
            .is_err());
        }
    }

    #[test]
    fn changing_loader_content_fails_including_reference() {
        for changed in 0..2 {
            let mut calls = [0; 2];
            let result = merge(
                2,
                |i| {
                    calls[i] += 1;
                    let mut f = frame(1000, 0.1);
                    if i == changed && calls[i] == 2 {
                        f.raw_top[0] += 1;
                    }
                    Ok(f)
                },
                &MergeOptions::default(),
                Control::none(),
            );
            assert!(
                matches!(result, Err(Error::InvalidData(message)) if message.contains("changed between"))
            );
        }
    }

    #[test]
    fn default_merge_preserves_invariants_with_all_statistics_enabled() {
        let options = MergeOptions {
            max_s: Some(1.0),
            ..Default::default()
        };
        let scene = merge(
            3,
            |i| {
                let mut f = noisy_frame(i, [1.0, 0.5, 2.0][i], [0.001; 3], [0.0001; 3], 1.0, 0.0);
                // Shared block texture gives alignment features and leaves flat fit blocks.
                let scale = [1.0, 0.5, 2.0][i];
                for y in 0..256 {
                    for x in 0..256 {
                        let delta = ((x / 32 * 7 + y / 32 * 11) % 5) as f32 * 0.025 / scale;
                        for c in 0..3 {
                            f.bmt[(y * 256 + x) * 3 + c] += delta;
                        }
                        f.raw_top[y * 256 + x] += (delta * 50000.0).round() as u16;
                        if x % 2 == 0 && y % 2 == 0 {
                            for c in 0..2 {
                                f.raw_half[((y / 2) * 128 + x / 2) * 2 + c] +=
                                    (delta * 50000.0).round() as u16;
                            }
                        }
                    }
                }
                // All frames invalid here. Record the shortest-exposure fallback.
                f.raw_top[..256 * 8].fill(55000);
                Ok(f)
            },
            &options,
            Control::none(),
        )
        .unwrap();
        assert_eq!(scene.report.frames_used, 3);
        assert!(scene.report.noise_blocks.iter().all(|&n| n >= 50));
        assert!(scene.report.calibration_blocks[1..]
            .iter()
            .all(|&n| n >= 50));
        assert!(scene.bmt.iter().all(|v| v.is_finite() && *v <= 1.0));
        assert!(scene.report.fallback_shortest > 0);
        assert_eq!(scene.report.fallback_best_snr, 0);
        assert_eq!(scene.report.fallback_zero, 0);
        assert!(scene.report.fallback_shortest < scene.bmt.len());
        assert_eq!(scene.report.ghost_fractions[0], 0.0);
        for (scale, nominal) in scene.report.scales.iter().zip(&scene.report.nominal_scales) {
            assert!((scale / nominal).log2().abs() <= 0.3);
        }
    }

    #[test]
    fn fractional_shift_noise_fit_rejects_textured_edges() {
        let expected = [0.00008, 0.00012, 0.00016];
        let make = |index, scale, shift: f32| {
            let mut f = noisy_frame(index, scale, [0.001; 3], expected, 1.0, shift);
            for c in 0..3 {
                let divisor = if c == 2 { 1 } else { 2 };
                let size = 256 / divisor;
                for y in 0..size {
                    for x in 0..size {
                        let xx = x as f32 * divisor as f32 + shift;
                        // Wide flat areas with a fractional boundary and a textured middle band.
                        let edge = (xx - 94.0).clamp(0.0, 1.0) - (xx - 158.0).clamp(0.0, 1.0);
                        let texture = if (96.0..158.0).contains(&xx) {
                            0.015 * (xx * 1.7 + y as f32).sin()
                        } else {
                            0.0
                        };
                        let delta = (0.12 * edge + texture) / scale;
                        let code = if c == 2 {
                            &mut f.raw_top[y * size + x]
                        } else {
                            &mut f.raw_half[(y * size + x) * 2 + c]
                        };
                        *code = (*code as f32 + delta * 50000.0).round() as u16;
                    }
                }
            }
            f
        };
        let reference = make(0, 1.0, 0.0);
        let other = make(1, 0.5, 2.35);
        let (actual, counts) = noise_fit(
            &reference,
            &other,
            &Alignment {
                dx: 2.35,
                ..Default::default()
            },
            0.5,
            &MergeOptions::default(),
            Control::none(),
        )
        .unwrap();
        for c in 0..3 {
            let total = if c == 2 { 1024 } else { 256 };
            assert!(counts[c] >= 50 && counts[c] < total * 9 / 10, "{counts:?}");
            assert!((actual[c] / expected[c] - 1.0).abs() < 0.2, "{actual:?}");
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
    fn ceiling_fallback_warns_for_a_cliff_but_not_a_smooth_histogram() {
        let mut values: Vec<u16> = (0u16..10000)
            .flat_map(|value| {
                std::iter::repeat_n(value, usize::from(value.min(10000 - value)) / 50 + 1)
            })
            .collect();
        assert_eq!(
            estimate_plane_saturation(values.iter().copied(), 0.0, 10000.0, Control::none())
                .unwrap(),
            (1.0, false)
        );
        values.extend(std::iter::repeat_n(4500, 20000));
        assert_eq!(
            estimate_plane_saturation(values.into_iter(), 0.0, 10000.0, Control::none()).unwrap(),
            (1.0, true)
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
            let options = phase1_options();
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
            ..phase1_options()
        };
        let fixed = merge(3, load, &options, Control::none()).unwrap();
        assert!(fixed.bmt.iter().all(|v| (*v - 0.3).abs() < 1e-5));
        let capped = merge(3, load, &phase1_options(), Control::none()).unwrap();
        assert!(capped.bmt.iter().all(|v| *v < 0.15));
        // One is the old ceiling. It must preserve both signal and noise weighting.
        let explicit_default = merge(
            3,
            load,
            &MergeOptions {
                saturation: Some([1.0; 3]),
                ..phase1_options()
            },
            Control::none(),
        )
        .unwrap();
        assert_eq!(capped.bmt, explicit_default.bmt);
    }

    #[test]
    fn soft_shoulder_samples_below_the_ceiling_do_not_pull_the_merge_down() {
        // A sensor that compresses above 0.8 of its ceiling toward 0.96. The
        // ceiling estimate is the top of the pile, so compressed samples sit
        // at 0.9 to 0.96 and carry up to 64 times the reference weight.
        let shoulder = |linear: f32| {
            if linear <= 0.8 {
                linear
            } else {
                0.8 + 0.16 * (1.0 - (-(linear - 0.8) / 0.2).exp())
            }
        };
        let truth = 0.2375_f32;
        let load = |i: usize| {
            let exposure = [1000, 2000, 4000, 8000][i];
            let mut f = frame(exposure, truth);
            let value = shoulder(truth * exposure as f32 / 1000.0);
            f.bmt.fill(value);
            f.raw_top.fill((value * 10000.0).round() as u16);
            f.raw_half.fill((value * 10000.0).round() as u16);
            Ok(f)
        };
        let merged = merge(4, load, &phase1_options(), Control::none()).unwrap();
        assert!(merged.bmt.iter().all(|v| (*v - truth).abs() < 1e-5));
        // The phase 1 defaults let the compressed long exposures pull the
        // result down by more than one tenth.
        let old_defaults = MergeOptions {
            saturation_knee: 0.9,
            saturation_width: 0.03,
            validity_margin: 0.02,
            ..phase1_options()
        };
        let pulled = merge(4, load, &old_defaults, Control::none()).unwrap();
        assert!(pulled.bmt.iter().all(|v| *v < 0.9 * truth));
    }

    #[test]
    fn rolloff_override_parses_and_validates() {
        let options =
            MergeOptions::parse_env(None, None, None, Some("0.75, 0.05, 0.15"), [None; 8]).unwrap();
        assert_eq!(options.saturation_knee, 0.75);
        assert_eq!(options.saturation_width, 0.05);
        assert_eq!(options.validity_margin, 0.15);
        let defaults = phase1_options();
        assert_eq!(options.noise_sigma, defaults.noise_sigma);
        assert_eq!(options.noise_k, defaults.noise_k);
        assert_eq!(options.noise_q, defaults.noise_q);
        assert_eq!(options.saturation, defaults.saturation);
        for value in [
            "",
            "0.8,0.03",
            "0.8,0.03,0.1,0.2",
            "0,0.03,0.1",
            "-0.1,0.03,0.1",
            "1.1,0.03,0.1",
            "0.8,0,0.1",
            "0.8,-0.03,0.1",
            "0.8,0.03,-0.1",
            "0.8,0.03,1",
            "0.8,0.03,0.2",
            "0.9,0.03,0.2",
            "nan,0.03,0.1",
            "0.8,inf,0.1",
            "0.8,0.03,nan",
            "x,0.03,0.1",
        ] {
            assert!(MergeOptions::parse_env(None, None, None, Some(value), [None; 8]).is_err());
        }
    }

    #[test]
    fn masked_border_cannot_set_the_written_encoding_scale() {
        let mut scene = merge(
            1,
            |_| Ok(frame(1000, 2.0)),
            &phase1_options(),
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
                    &phase1_options(),
                    Control::none(),
                )
                .unwrap();
                assert_eq!(
                    &loaded[..exposures.len()],
                    &(0..exposures.len()).collect::<Vec<_>>()
                );
                assert_eq!(loaded.len(), exposures.len() * 2);
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
        let scene = merge(
            exposures.len(),
            |i| {
                let mut seed = 42u64 + i as u64;
                let mut uniform = || {
                    seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                    ((seed >> 11) as f64 + 0.5) / ((1u64 << 53) as f64)
                };
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
            ..phase1_options()
        });
        eprintln!("zero scene: mean {mean}, standard error {standard_error}");
        assert!(
            mean.abs() < 3.0 * standard_error,
            "mean {mean}, standard error {standard_error}"
        );
    }

    #[test]
    fn default_noise_model_zero_scene_has_bounded_negative_bias() {
        let options = phase1_options();
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
            ..phase1_options()
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
            &phase1_options(),
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
                ..phase1_options()
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
            ..phase1_options()
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
            ..phase1_options()
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
            ..phase1_options()
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
        let options = MergeOptions::parse_env(
            Some("0.002, 0.0001, 0.01"),
            Some("8"),
            None,
            None,
            [None; 8],
        )
        .unwrap();
        assert_eq!(options.noise_sigma, Some(0.002));
        assert!(!options.fit_noise);
        assert_eq!(options.noise_k, [0.0001; 3]);
        assert_eq!(options.noise_q, [0.01; 3]);
        let measured =
            MergeOptions::parse_env(Some(",0.0001,0"), None, None, None, [None; 8]).unwrap();
        assert_eq!(measured.noise_sigma, None);
        assert!(!measured.fit_noise);
        assert_eq!(options.max_s, Some(8.0));
        for value in ["1,2", "1,2,3,4", "nan,0,0", "-1,0,0", "0,-1,0", "x,0,0"] {
            assert!(MergeOptions::parse_env(Some(value), None, None, None, [None; 8]).is_err());
        }
        for value in ["nan", "inf", "0.5", "x"] {
            assert!(MergeOptions::parse_env(None, Some(value), None, None, [None; 8]).is_err());
        }
    }

    #[test]
    fn saturation_override_requires_three_positive_finite_fractions() {
        assert_eq!(phase1_options().saturation, None);
        let options =
            MergeOptions::parse_env(None, None, Some("0.44, 0.43, 0.55"), None, [None; 8]).unwrap();
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
                MergeOptions::parse_env(None, None, Some(value), None, [None; 8]).is_err(),
                "{value}"
            );
        }
    }

    #[test]
    fn invalid_frames_options_and_cancellation_fail() {
        let options = phase1_options();
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
            &phase1_options(),
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
            &phase1_options(),
            Control::none(),
        )
        .unwrap();
        assert!(scene
            .bmt
            .chunks(3)
            .all(|p| (p[1] - p[0] * 3.0).abs() < 1e-6));
    }
}
