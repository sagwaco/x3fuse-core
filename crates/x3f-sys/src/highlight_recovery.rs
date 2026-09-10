//! Local, sensor-aware highlight reconstruction for co-sited BMT sensors.
//!
//! The immutable model deliberately owns its source reliability. Neither
//! reconstruction nor fallback reads neighboring image pixels, so callers can
//! evaluate once for headroom and again while encoding the image in place.

const TILE_SIZE: usize = 16;
const PYRAMID_LEVELS: usize = 5;
const MIN_DONORS: f64 = 8.0;
const MAX_DISTANCE: f64 = 128.0;
const MAX_VARIATION: f64 = 0.15;
const MAX_AMPLITUDE_DISAGREEMENT: f64 = 0.10;

/// Source reliability before denoising or gains: 255 is healthy, zero is
/// clipped, and intermediate values describe the sensor's clipping shoulder.
/// Low signal is evaluated separately and must not be marked as clipping.
pub struct SensorReliability {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<[u8; 3]>,
    /// Per-channel black-noise standard deviation in normalized sensor units.
    pub noise: [f64; 3],
}

impl SensorReliability {
    pub fn new(rows: usize, cols: usize, noise: [f64; 3]) -> Option<Self> {
        let count = rows.checked_mul(cols)?;
        if count == 0 {
            return None;
        }
        let mut data = Vec::new();
        data.try_reserve_exact(count).ok()?;
        data.resize(count, [255; 3]);
        Some(Self {
            rows,
            cols,
            data,
            noise: noise.map(|n| if n.is_finite() { n.max(0.0) } else { 0.0 }),
        })
    }
}

/// Quantize sensor shoulder reliability. Noise is accepted alongside the
/// source sample for callers' convenience, but does not indicate saturation.
pub fn channel_reliability(value: f64, _noise: f64, saturation: f64, soft_window: f64) -> u8 {
    if !value.is_finite() || !saturation.is_finite() || saturation <= 0.0 {
        return 0;
    }
    if value >= saturation {
        return 0;
    }
    if !soft_window.is_finite() || soft_window <= 0.0 {
        return 255;
    }
    let t = ((saturation - value) / soft_window).clamp(0.0, 1.0);
    let smooth = t * t * (3.0 - 2.0 * t);
    // Keep the exact clipping identity distinct from quantization in the
    // shoulder: any finite sample below the clipping threshold is nonzero.
    (smooth * 255.0).round().clamp(1.0, 255.0) as u8
}

#[derive(Clone, Copy, Default)]
struct Tile {
    sum: [f64; 3],
    square_sum: [f64; 3],
    count: u64,
    row_sum: f64,
    col_sum: f64,
    min_row: usize,
    max_row: usize,
    min_col: usize,
    max_col: usize,
}

impl Tile {
    fn add(&mut self, row: usize, col: usize, chroma: [f64; 3]) {
        if self.count == 0 {
            self.min_row = row;
            self.max_row = row;
            self.min_col = col;
            self.max_col = col;
        } else {
            self.min_row = self.min_row.min(row);
            self.max_row = self.max_row.max(row);
            self.min_col = self.min_col.min(col);
            self.max_col = self.max_col.max(col);
        }
        self.count += 1;
        self.row_sum += row as f64;
        self.col_sum += col as f64;
        for (c, value) in chroma.into_iter().enumerate() {
            self.sum[c] += value;
            self.square_sum[c] += value * value;
        }
    }

    fn merge(&mut self, other: &Self) {
        if other.count == 0 {
            return;
        }
        if self.count == 0 {
            *self = *other;
            return;
        }
        self.count += other.count;
        self.row_sum += other.row_sum;
        self.col_sum += other.col_sum;
        self.min_row = self.min_row.min(other.min_row);
        self.max_row = self.max_row.max(other.max_row);
        self.min_col = self.min_col.min(other.min_col);
        self.max_col = self.max_col.max(other.max_col);
        for c in 0..3 {
            self.sum[c] += other.sum[c];
            self.square_sum[c] += other.square_sum[c];
        }
    }
}

struct Level {
    rows: usize,
    cols: usize,
    tile_size: usize,
    tiles: Vec<Tile>,
}

impl Level {
    fn new(rows: usize, cols: usize, tile_size: usize) -> Option<Self> {
        let count = rows.checked_mul(cols)?;
        let mut tiles = Vec::new();
        tiles.try_reserve_exact(count).ok()?;
        tiles.resize(count, Tile::default());
        Some(Self {
            rows,
            cols,
            tile_size,
            tiles,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RecoveryResult {
    pub samples: [f64; 3],
    /// Confidence in a local chroma model, not in the neutral fallback.
    pub confidence: f64,
    /// A consistent local donor model was applied. Such pixels must bypass
    /// subsequent neutral reconstruction, RepairPix, and matrix snapping.
    pub recovered: bool,
    /// At least one source channel entered the clipping shoulder. Use
    /// `LocalRecovery::mask` to distinguish exact clipping (zero) if needed.
    pub damaged: bool,
}

pub struct LocalRecovery {
    reliability: SensorReliability,
    levels: Vec<Level>,
    recovery_cap: Option<f64>,
    chroma_enabled: bool,
}

impl LocalRecovery {
    /// Build in deterministic row order from normalized, pre-export samples.
    /// The callback is only used here; it is never retained or called during
    /// pixel reconstruction. Empty donor support still permits fallback.
    pub fn build(
        reliability: SensorReliability,
        sample: impl Fn(usize, usize) -> [f64; 3],
        recovery_cap: Option<f64>,
    ) -> Option<Self> {
        let count = reliability.rows.checked_mul(reliability.cols)?;
        if count == 0 || count != reliability.data.len() {
            return None;
        }
        let tile_rows = reliability.rows.checked_add(TILE_SIZE - 1)? / TILE_SIZE;
        let tile_cols = reliability.cols.checked_add(TILE_SIZE - 1)? / TILE_SIZE;
        let mut first = Level::new(tile_rows, tile_cols, TILE_SIZE)?;
        for row in 0..reliability.rows {
            for col in 0..reliability.cols {
                if reliability.data[row * reliability.cols + col] != [255; 3] {
                    continue;
                }
                let s = sample(row, col);
                if s.iter()
                    .enumerate()
                    .any(|(c, &v)| !v.is_finite() || v <= signal_floor(reliability.noise[c]))
                {
                    continue;
                }
                let sum = s.iter().sum::<f64>();
                if !sum.is_finite() || sum < 0.20 {
                    continue;
                }
                let chroma = s.map(|v| v / sum);
                first.tiles[(row / TILE_SIZE) * tile_cols + col / TILE_SIZE].add(row, col, chroma);
            }
        }
        let mut levels = Vec::new();
        levels.try_reserve_exact(PYRAMID_LEVELS).ok()?;
        levels.push(first);
        while levels.len() < PYRAMID_LEVELS {
            let previous = levels.last()?;
            if previous.rows == 1 && previous.cols == 1 {
                break;
            }
            let mut next = Level::new(
                previous.rows.div_ceil(2),
                previous.cols.div_ceil(2),
                previous.tile_size * 2,
            )?;
            for row in 0..previous.rows {
                for col in 0..previous.cols {
                    next.tiles[(row / 2) * next.cols + col / 2]
                        .merge(&previous.tiles[row * previous.cols + col]);
                }
            }
            levels.push(next);
        }
        Some(Self {
            reliability,
            levels,
            recovery_cap: recovery_cap.filter(|v| v.is_finite() && *v > 0.0),
            chroma_enabled: true,
        })
    }

    /// Preserve X3F_NO_CHROMA_LUT without discarding sensor provenance or
    /// the conservative fallback that uses surviving source channels.
    pub fn set_chroma_enabled(&mut self, enabled: bool) {
        self.chroma_enabled = enabled;
    }

    pub fn mask(&self, row: usize, col: usize) -> [u8; 3] {
        if row >= self.reliability.rows || col >= self.reliability.cols {
            return [0; 3];
        }
        self.reliability.data[row * self.reliability.cols + col]
    }

    pub fn recover(&self, row: usize, col: usize, s: [f64; 3], prior: [f64; 3]) -> RecoveryResult {
        let mut result = RecoveryResult {
            samples: s,
            confidence: 0.0,
            recovered: false,
            damaged: false,
        };
        if row >= self.reliability.rows || col >= self.reliability.cols {
            return result;
        }
        let mask = self.mask(row, col);
        result.damaged = mask != [255; 3];
        if !result.damaged || s.iter().any(|v| !v.is_finite()) {
            return result;
        }
        let anchors = self.anchors(mask, s);
        if !anchors.iter().any(|&v| v) {
            // There is no measured amplitude from which to recover detail.
            return result;
        }
        if self.chroma_enabled {
            if let Some((chroma, confidence)) = self.local_chroma(row, col) {
                if let Some((amplitude, agreement)) = self.amplitude(s, chroma, mask, anchors) {
                    let confidence = confidence * agreement;
                    if confidence > 0.0 {
                        if let Some(samples) =
                            self.reconstruct(s, chroma, amplitude, mask, anchors, confidence)
                        {
                            result.samples = samples;
                            result.confidence = confidence;
                            result.recovered = true;
                            return result;
                        }
                    }
                }
            }
        }
        // A neutral direction is an assumption, so only fill damaged
        // channels. Never replace measured healthy channels with a neutral
        // luminance derived from a clipped maximum.
        if let Some((amplitude, agreement)) = self.amplitude(s, prior, mask, anchors) {
            if let Some(samples) = self.reconstruct(s, prior, amplitude, mask, anchors, agreement) {
                result.samples = samples;
            }
        }
        result
    }

    fn anchors(&self, mask: [u8; 3], s: [f64; 3]) -> [bool; 3] {
        let usable = std::array::from_fn::<_, 3, _>(|c| {
            mask[c] != 0 && s[c] > signal_floor(self.reliability.noise[c])
        });
        let best = (0..3)
            .filter(|&c| usable[c])
            .map(|c| mask[c])
            .max()
            .unwrap_or(0);
        // Fully healthy channels always stay measured. When all channels
        // are in the shoulder, retain the most reliable measured channels
        // as anchors instead of taking the brightest clipped channel.
        std::array::from_fn(|c| usable[c] && mask[c] == best)
    }

    fn amplitude(
        &self,
        s: [f64; 3],
        chroma: [f64; 3],
        mask: [u8; 3],
        anchors: [bool; 3],
    ) -> Option<(f64, f64)> {
        if chroma.iter().any(|v| !v.is_finite() || *v <= 1e-9) {
            return None;
        }
        let mut numerator = 0.0;
        let mut denominator = 0.0;
        let mut smallest = f64::INFINITY;
        let mut largest = 0.0_f64;
        for c in 0..3 {
            if !anchors[c] {
                continue;
            }
            let amplitude = s[c] / chroma[c];
            if !amplitude.is_finite() || amplitude <= 0.0 {
                return None;
            }
            smallest = smallest.min(amplitude);
            largest = largest.max(amplitude);
            let trust = mask[c] as f64 / 255.0;
            let noise = self.reliability.noise[c].max(0.001);
            let weight = trust * trust / (noise * noise);
            numerator += weight * chroma[c] * s[c];
            denominator += weight * chroma[c] * chroma[c];
        }
        if !denominator.is_finite() || denominator <= 0.0 {
            return None;
        }
        let disagreement = (largest - smallest) / (0.5 * (largest + smallest));
        if !disagreement.is_finite() || disagreement > MAX_AMPLITUDE_DISAGREEMENT {
            return None;
        }
        let amplitude = numerator / denominator;
        if !amplitude.is_finite() || amplitude <= 0.0 {
            return None;
        }
        let agreement = (1.0 - disagreement / MAX_AMPLITUDE_DISAGREEMENT).clamp(0.0, 1.0);
        Some((amplitude, agreement))
    }

    fn reconstruct(
        &self,
        s: [f64; 3],
        chroma: [f64; 3],
        amplitude: f64,
        mask: [u8; 3],
        anchors: [bool; 3],
        confidence: f64,
    ) -> Option<[f64; 3]> {
        // Confidence determines whether evidence is usable; it must not
        // turn donor distance into an artificial exposure gradient. All
        // coherent supported neighborhoods have confidence >= 0.5 and
        // therefore reconstruct fully. Ambiguous evidence fades smoothly.
        let confidence = confidence_strength(confidence);
        let anchor_trust = (0..3)
            .filter(|&c| anchors[c])
            .map(|c| mask[c] as f64 / 255.0)
            .fold(0.0_f64, f64::max);
        let mut out = s;
        let mut has_target = false;
        for c in 0..3 {
            if mask[c] == 255 || anchors[c] {
                continue;
            }
            has_target = true;
            let mut predicted = amplitude * chroma[c];
            if let Some(cap) = self.recovery_cap {
                predicted = predicted.min(cap);
            }
            if !predicted.is_finite() || predicted < 0.0 {
                return None;
            }
            // True clipping gives a lower bound, not a noisy color sample
            // that an unrelated donor is free to darken. The tolerance
            // admits small calibration/denoise discrepancies, but even then
            // the actual output is never reduced below the observation.
            let tolerance = (5.0 * self.reliability.noise[c]).max(0.02 * s[c].abs());
            if mask[c] == 0 && predicted + tolerance < s[c] {
                return None;
            }
            let strength = (1.0 - mask[c] as f64 / 255.0) * confidence * anchor_trust;
            out[c] = s[c] + strength * (predicted.max(s[c]) - s[c]);
            if !out[c].is_finite() {
                return None;
            }
        }
        has_target.then_some(out)
    }

    fn local_chroma(&self, row: usize, col: usize) -> Option<([f64; 3], f64)> {
        for level in &self.levels {
            let tile_row = row / level.tile_size;
            let tile_col = col / level.tile_size;
            let mut sum = [0.0; 3];
            let mut square_sum = [0.0; 3];
            let mut weight_sum = 0.0;
            let mut donor_count = 0_u64;
            let mut weighted_distance = 0.0;
            for r in tile_row.saturating_sub(1)..=(tile_row + 1).min(level.rows - 1) {
                for c in tile_col.saturating_sub(1)..=(tile_col + 1).min(level.cols - 1) {
                    let tile = &level.tiles[r * level.cols + c];
                    if tile.count == 0 {
                        continue;
                    }
                    // Bounding-box corners conservatively guarantee that
                    // even a coarse tile cannot import distant donors.
                    let far_row = row.abs_diff(tile.min_row).max(row.abs_diff(tile.max_row));
                    let far_col = col.abs_diff(tile.min_col).max(col.abs_diff(tile.max_col));
                    let far_squared = (far_row as f64).powi(2) + (far_col as f64).powi(2);
                    if far_squared > MAX_DISTANCE * MAX_DISTANCE {
                        continue;
                    }
                    let count = tile.count as f64;
                    let dr = tile.row_sum / count - row as f64;
                    let dc = tile.col_sum / count - col as f64;
                    let distance = dr.hypot(dc);
                    let weight = 1.0 / (1.0 + (distance / level.tile_size as f64).powi(2));
                    donor_count += tile.count;
                    weight_sum += weight * count;
                    weighted_distance += weight * count * distance;
                    for channel in 0..3 {
                        sum[channel] += weight * tile.sum[channel];
                        square_sum[channel] += weight * tile.square_sum[channel];
                    }
                }
            }
            if (donor_count as f64) < MIN_DONORS || weight_sum <= 0.0 {
                continue;
            }
            let mut chroma = [0.0; 3];
            let mut variation = 0.0_f64;
            for c in 0..3 {
                chroma[c] = sum[c] / weight_sum;
                let variance = (square_sum[c] / weight_sum - chroma[c] * chroma[c]).max(0.0);
                variation = variation.max(variance.sqrt() / chroma[c].max(1e-9));
            }
            // Contradictory nearby evidence must not be overridden by a
            // larger neighborhood dominated by some other scene material.
            if !variation.is_finite() || variation >= MAX_VARIATION {
                return None;
            }
            let coherence = (1.0 - variation / MAX_VARIATION).clamp(0.0, 1.0);
            let distance = weighted_distance / weight_sum;
            let proximity = 1.0 - 0.5 * (distance / MAX_DISTANCE).powi(2);
            return Some((chroma, coherence * proximity));
        }
        None
    }
}

fn signal_floor(noise: f64) -> f64 {
    (noise.max(0.0) * 5.0).max(0.002)
}

fn confidence_strength(confidence: f64) -> f64 {
    let t = (2.0 * confidence).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Suppress cancellation-driven false colors in source highlight shoulders.
/// All samples and the neutral direction are after spatial gain; `matrix`
/// maps that domain to linear display RGB. Apply this after componentwise
/// recovery guards, and never restore individual components afterward.
///
/// Source reliability alone does not establish color accuracy for Foveon:
/// small layer-ratio errors can produce large negative RGB components. This
/// guard preserves the measured highlight amplitude while moving the whole
/// vector coherently, including nominally healthy but contaminated layers.
pub fn protect_highlight_color(
    samples: [f64; 3],
    original: [f64; 3],
    mask: [u8; 3],
    neutral: [f64; 3],
    matrix: &[f64; 9],
    gate_thr: f64,
    gate_width: f64,
) -> [f64; 3] {
    if mask == [255; 3]
        || samples
            .iter()
            .chain(original.iter())
            .any(|v| !v.is_finite())
        || neutral.iter().any(|v| !v.is_finite() || *v <= 1e-9)
        || matrix.iter().any(|v| !v.is_finite())
    {
        return samples;
    }
    let rgb = guard_rgb(matrix, samples);
    if rgb.iter().any(|v| !v.is_finite()) {
        return samples;
    }
    let minimum = rgb.into_iter().fold(f64::INFINITY, f64::min);
    let peak = rgb.into_iter().fold(0.0_f64, f64::max);
    // Positive, physically plausible bright colors are not evidence of
    // sensor failure. In particular, a genuine green/yellow subject must
    // not be neutralized just because the old opponent margin is large.
    if minimum >= 0.0 || peak <= 0.0 {
        return samples;
    }
    let smooth = |t: f64| {
        let t = t.clamp(0.0, 1.0);
        t * t * (3.0 - 2.0 * t)
    };
    let threshold = if gate_thr.is_finite() {
        gate_thr.max(0.0)
    } else {
        0.20
    };
    let width = if gate_width.is_finite() {
        gate_width.max(1e-6)
    } else {
        0.30
    };
    let green = rgb[1] - rgb[0].max(rgb[2]);
    let yellow = rgb[0].min(rgb[1]) - rgb[2];
    let magenta = rgb[0].min(rgb[2]) - rgb[1];
    let opponent = smooth((green.max(yellow).max(magenta) - threshold) / width);
    let negative = -minimum;
    // The opponent detector handles green/yellow/magenta pathologies; a
    // substantial negative component catches the remaining cancellation
    // identities. A short continuous ramp rejects numerical overshoots.
    let negative_evidence = smooth(negative / (0.005 * peak).max(1e-9));
    let negative_offset = (threshold - 0.20).max(0.0);
    let negative_width = (0.10 * peak * (width / 0.30)).max(1e-9);
    let negative_strength = smooth((negative - negative_offset) / negative_width);
    let source_strength =
        confidence_strength(1.0 - mask.into_iter().min().unwrap_or(255) as f64 / 255.0);
    let strength = source_strength * negative_evidence * opponent.max(negative_strength);
    if strength <= 0.0 {
        return samples;
    }

    let mut weighted_amplitude = 0.0;
    let mut weight_sum = 0.0;
    let mut fallback_amplitude = 0.0_f64;
    for c in 0..3 {
        fallback_amplitude = fallback_amplitude.max(samples[c].max(original[c]) / neutral[c]);
        let reliability = mask[c] as f64 / 255.0;
        let signal = smooth((original[c] - 0.002) / 0.018);
        let weight = reliability * reliability * signal;
        weighted_amplitude += weight * original[c] / neutral[c];
        weight_sum += weight;
    }
    // Smooth weights avoid anchor identity switches. As the final survivor
    // itself clips, fade continuously to the conservative all-clipped
    // amplitude instead of introducing a luminance/color ring at mask=0.
    let amplitude = if weight_sum > 0.0 {
        let survivor_amplitude = weighted_amplitude / weight_sum;
        let trust = confidence_strength(weight_sum);
        (1.0 - trust) * fallback_amplitude + trust * survivor_amplitude
    } else {
        fallback_amplitude
    };
    if !amplitude.is_finite() || amplitude < 0.0 {
        return samples;
    }
    let target = neutral.map(|v| amplitude * v);
    if target.iter().any(|v| !v.is_finite()) {
        return samples;
    }
    std::array::from_fn(|c| (1.0 - strength) * samples[c] + strength * target[c])
}

fn guard_rgb(matrix: &[f64; 9], samples: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|row| {
        matrix[3 * row] * samples[0]
            + matrix[3 * row + 1] * samples[1]
            + matrix[3 * row + 2] * samples[2]
    })
}

/// Keep legacy highlight hue unless the local estimate agrees in rendered
/// chromaticity, then recover brightness from the measured surviving layers.
/// All vectors are after spatial gain. The returned vector is coherent:
/// componentwise restoration or clipping floors must not follow this step.
pub fn stabilize_highlight_color(
    candidate: [f64; 3],
    stable: [f64; 3],
    original: [f64; 3],
    mask: [u8; 3],
    matrix: &[f64; 9],
) -> [f64; 3] {
    if original.iter().any(|v| !v.is_finite())
        || stable.iter().any(|v| !v.is_finite())
        || matrix.iter().any(|v| !v.is_finite())
    {
        return stable;
    }
    let original_rgb = guard_rgb(matrix, original);
    let healthy = mask == [255; 3];
    if healthy && original_rgb.iter().all(|v| v.is_finite() && *v >= 0.0) {
        return original;
    }
    let stable_rgb = guard_rgb(matrix, stable);
    let Some(stable_chroma) = rgb_chromaticity(stable_rgb) else {
        // The caller can apply the negative-RGB guard with its calibrated
        // neutral direction when the legacy color itself is not usable.
        return stable;
    };
    let stable_sum = stable_rgb.iter().sum::<f64>();
    let smooth = |t: f64| {
        let t = t.clamp(0.0, 1.0);
        t * t * (3.0 - 2.0 * t)
    };
    let original_max = original.into_iter().fold(0.0_f64, f64::max);
    if healthy && original_max <= 0.75 {
        return stable;
    }
    let weights: [f64; 3] = std::array::from_fn(|c| {
        let mut reliability = mask[c] as f64 / 255.0;
        if healthy {
            // A pathological matrix response can precede the metadata
            // clipping threshold. Downweight bright source layers smoothly
            // instead of choosing a discontinuous maximum-channel index.
            reliability *= 1.0 - smooth((original[c] - 0.75) / 0.25);
        }
        let signal = smooth((original[c] - 0.002) / 0.018);
        reliability * reliability * signal
    });
    let source_weight = weights.iter().sum::<f64>();
    if source_weight <= 0.0 {
        return stable;
    }
    let source_support = confidence_strength(source_weight);
    let candidate_rgb = guard_rgb(matrix, candidate);
    let candidate_chroma = rgb_chromaticity(candidate_rgb);
    let chroma_weight = candidate_chroma.map_or(0.0, |chroma| {
        let difference = (0..3)
            .map(|c| (chroma[c] - stable_chroma[c]).abs())
            .fold(0.0_f64, f64::max);
        source_support * (1.0 - smooth((difference - 0.01) / 0.02))
    });
    let stable_direction = stable.map(|v| v / stable_sum);
    let candidate_sum = candidate_rgb.iter().sum::<f64>();
    let direction = if chroma_weight > 0.0 && candidate_sum.is_finite() && candidate_sum > 0.0 {
        std::array::from_fn(|c| {
            (1.0 - chroma_weight) * stable_direction[c]
                + chroma_weight * candidate[c] / candidate_sum
        })
    } else {
        stable_direction
    };
    let mut weighted_amplitude = 0.0;
    let mut amplitude_weight = 0.0;
    for c in 0..3 {
        if direction[c].is_finite() && direction[c] > 1e-9 && weights[c] > 0.0 {
            weighted_amplitude += weights[c] * original[c] / direction[c];
            amplitude_weight += weights[c];
        }
    }
    if amplitude_weight <= 0.0 {
        return stable;
    }
    let measured_amplitude = weighted_amplitude / amplitude_weight;
    let trust = confidence_strength(amplitude_weight);
    let amplitude = (1.0 - trust) * stable_sum + trust * measured_amplitude;
    let result = direction.map(|v| v * amplitude);
    if result.iter().all(|v| v.is_finite()) {
        result
    } else {
        stable
    }
}

fn rgb_chromaticity(rgb: [f64; 3]) -> Option<[f64; 3]> {
    if rgb.iter().any(|v| !v.is_finite() || *v < 0.0) {
        return None;
    }
    let sum = rgb.iter().sum::<f64>();
    if !sum.is_finite() || sum <= 1e-9 {
        return None;
    }
    Some(rgb.map(|v| v / sum))
}

#[cfg(test)]
mod tests {
    use super::*;

    // Each row sums to one, so [1,1,1] is a neutral sensor direction.
    // Small BMT mismatches nevertheless produce large negative RGB values.
    const CANCELLATION_MATRIX: [f64; 9] = [2.0, -2.0, 1.0, -1.0, 3.0, -1.0, 3.0, -3.0, 1.0];

    const IDENTITY_MATRIX: [f64; 9] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

    #[test]
    fn high_gate_threshold_disables_negative_color_guard() {
        let original = [0.29920237, 0.84198446, 0.98995567];
        assert_eq!(
            protect_highlight_color(
                original,
                original,
                [255, 255, 4],
                [1.0; 3],
                &CANCELLATION_MATRIX,
                100.0,
                0.30,
            ),
            original,
        );
    }

    #[test]
    fn stabilization_keeps_healthy_zero_rgb_exact() {
        assert_eq!(
            stabilize_highlight_color([1.0; 3], [0.5; 3], [0.0; 3], [255; 3], &IDENTITY_MATRIX,),
            [0.0; 3],
        );
    }

    #[test]
    fn stabilization_rejects_positive_green_when_legacy_hue_is_blue() {
        let stable = [0.4, 0.6, 1.0];
        let result = stabilize_highlight_color(
            [0.2, 1.4, 0.2],
            stable,
            [0.3, 0.5, 0.7],
            [255, 255, 0],
            &IDENTITY_MATRIX,
        );
        let chroma = rgb_chromaticity(result).unwrap();
        let expected = rgb_chromaticity(stable).unwrap();
        for c in 0..3 {
            assert!((chroma[c] - expected[c]).abs() < 1e-12);
        }
    }

    #[test]
    fn stabilization_chroma_agreement_fades_continuously() {
        let stable = [0.2, 0.3, 0.5];
        let mut previous = stable;
        for step in 0..=80 {
            let difference = step as f64 * 0.0005;
            let candidate = [0.2 + difference, 0.3 - difference, 0.5];
            let result =
                stabilize_highlight_color(candidate, stable, stable, [0, 0, 255], &IDENTITY_MATRIX);
            for c in 0..3 {
                assert!((result[c] - previous[c]).abs() < 0.0015);
            }
            if step == 20 {
                for c in 0..3 {
                    assert!((result[c] - candidate[c]).abs() < 1e-12);
                }
            }
            if step >= 60 {
                for c in 0..3 {
                    assert!((result[c] - stable[c]).abs() < 1e-12);
                }
            }
            previous = result;
        }
    }

    #[test]
    fn stabilization_restores_texture_from_one_survivor_of_flat_neutral_output() {
        for signal in [0.4, 0.45, 0.5, 0.6] {
            let result = stabilize_highlight_color(
                [1.0; 3],
                [1.0; 3],
                [1.0, 1.0, signal],
                [0, 0, 255],
                &IDENTITY_MATRIX,
            );
            for actual in result {
                assert!((actual - signal).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn stabilization_preserves_known_truth_when_recovered_and_stable_hues_agree() {
        let ratio = [2.4, 2.2, 1.0];
        let stable = ratio.map(|v| v * 0.45);
        for signal in [0.55, 0.6, 0.7] {
            let truth = ratio.map(|v| v * signal);
            let result = stabilize_highlight_color(
                truth,
                stable,
                [1.0, 1.0, signal],
                [0, 0, 255],
                &IDENTITY_MATRIX,
            );
            for c in 0..3 {
                assert!((result[c] - truth[c]).abs() / truth[c] < 0.01);
            }
        }
    }

    #[test]
    fn stabilization_keeps_valid_healthy_samples_and_all_clipped_fallback_exact() {
        let original = [0.9, 0.1, 0.2];
        assert_eq!(
            stabilize_highlight_color([1.0; 3], [0.5; 3], original, [255; 3], &IDENTITY_MATRIX,),
            original,
        );
        let stable = [0.7, 0.8, 0.9];
        assert_eq!(
            stabilize_highlight_color([1.0; 3], stable, [1.0; 3], [0; 3], &IDENTITY_MATRIX,),
            stable,
        );
    }

    #[test]
    fn stabilization_uses_stable_hue_for_bright_pathological_healthy_masks() {
        let original = [0.3, 0.84, 0.99];
        let result =
            stabilize_highlight_color(original, [0.6; 3], original, [255; 3], &CANCELLATION_MATRIX);
        let rgb = guard_rgb(&CANCELLATION_MATRIX, result);
        assert!(rgb.iter().all(|v| *v > 0.0));
        assert!((rgb[0] - rgb[1]).abs() < 1e-12);
        assert!((rgb[1] - rgb[2]).abs() < 1e-12);
    }

    #[test]
    fn color_guard_repairs_near_clipped_sky_even_with_unchanged_local_prediction() {
        let original = [0.29920237, 0.84198446, 0.98995567];
        let before = guard_rgb(&CANCELLATION_MATRIX, original);
        assert!(before[0] < 0.0 && before[2] < 0.0 && before[1] > 1.0);
        let repaired = protect_highlight_color(
            original,
            original,
            [255, 255, 4],
            [1.0; 3],
            &CANCELLATION_MATRIX,
            0.20,
            0.30,
        );
        let rgb = guard_rgb(&CANCELLATION_MATRIX, repaired);
        assert!(rgb.iter().all(|v| *v > 0.0));
        assert!((rgb[0] - rgb[1]).abs() < 1e-12);
        assert!((rgb[1] - rgb[2]).abs() < 1e-12);
    }

    #[test]
    fn color_guard_preserves_all_healthy_sources_and_positive_bright_colors() {
        let pathological = [0.3, 0.84, 0.99];
        assert_eq!(
            protect_highlight_color(
                pathological,
                pathological,
                [255; 3],
                [1.0; 3],
                &CANCELLATION_MATRIX,
                0.20,
                0.30,
            ),
            pathological,
        );
        let colored = [1.1, 1.0, 1.0];
        assert!(guard_rgb(&CANCELLATION_MATRIX, colored)
            .iter()
            .all(|v| *v > 0.0));
        assert_eq!(
            protect_highlight_color(
                colored,
                colored,
                [0, 255, 255],
                [1.0; 3],
                &CANCELLATION_MATRIX,
                0.20,
                0.30,
            ),
            colored,
        );
    }

    #[test]
    fn color_guard_retains_texture_when_two_channels_are_flat() {
        for amplitude in [0.30, 0.32, 0.35, 0.40] {
            let original = [amplitude, 0.99, 0.99];
            let repaired = protect_highlight_color(
                original,
                original,
                [255, 0, 0],
                [1.0; 3],
                &CANCELLATION_MATRIX,
                0.20,
                0.30,
            );
            let rgb = guard_rgb(&CANCELLATION_MATRIX, repaired);
            for channel in rgb {
                assert!((channel - amplitude).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn color_guard_removes_magenta_without_anchor_transition_rings() {
        let original = [0.8, 0.3, 1.0];
        let before = guard_rgb(&CANCELLATION_MATRIX, original);
        assert!(before[0] > 0.0 && before[1] < 0.0 && before[2] > 0.0);
        let mut previous: Option<f64> = None;
        for shift in -64_i16..=64 {
            let repaired = protect_highlight_color(
                original,
                original,
                [(128 + shift) as u8, (128 - shift) as u8, 0],
                [1.0; 3],
                &CANCELLATION_MATRIX,
                0.20,
                0.30,
            );
            let rgb = guard_rgb(&CANCELLATION_MATRIX, repaired);
            assert!((rgb[0] - rgb[1]).abs() < 1e-12);
            assert!((rgb[1] - rgb[2]).abs() < 1e-12);
            if let Some(previous) = previous {
                assert!((rgb[0] - previous).abs() < 0.01);
            }
            previous = Some(rgb[0]);
        }
    }

    #[test]
    fn color_guard_is_continuous_when_the_last_survivor_clips() {
        let original = [0.3, 0.99, 0.99];
        let mut previous: Option<f64> = None;
        for reliability in 0..=255_u8 {
            let repaired = protect_highlight_color(
                original,
                original,
                [reliability, 0, 0],
                [1.0; 3],
                &CANCELLATION_MATRIX,
                0.20,
                0.30,
            );
            let rgb = guard_rgb(&CANCELLATION_MATRIX, repaired);
            assert!((rgb[0] - rgb[1]).abs() < 1e-12);
            assert!((rgb[1] - rgb[2]).abs() < 1e-12);
            if let Some(previous) = previous {
                assert!((rgb[0] - previous).abs() < 0.03);
            }
            previous = Some(rgb[0]);
        }
    }

    fn ratios(anchor: usize) -> [f64; 3] {
        let mut ratios = [2.4, 2.4, 2.4];
        ratios[anchor] = 1.0;
        ratios
    }

    fn scene(anchor: usize, cap: Option<f64>) -> LocalRecovery {
        let mut reliability = SensorReliability::new(32, 64, [0.001; 3]).unwrap();
        for row in 0..32 {
            for col in 32..64 {
                reliability.data[row * 64 + col] = [0; 3];
                reliability.data[row * 64 + col][anchor] = 255;
            }
        }
        let ratio = ratios(anchor);
        LocalRecovery::build(reliability, |_, _| ratio.map(|v| v * 0.2), cap).unwrap()
    }

    fn distance_scene(ratio: [f64; 3], clipped_mask: [u8; 3]) -> LocalRecovery {
        let mut reliability = SensorReliability::new(16, 192, [0.001; 3]).unwrap();
        for row in 0..16 {
            for col in 16..192 {
                reliability.data[row * 192 + col] = clipped_mask;
            }
        }
        LocalRecovery::build(reliability, |_, _| ratio.map(|v| v * 0.2), None).unwrap()
    }

    #[test]
    fn coherent_two_clip_recovery_matches_truth_at_multiple_support_distances() {
        let ratio = [2.4, 2.2, 1.0];
        let model = distance_scene(ratio, [0, 0, 255]);
        let truth = ratio.map(|v| v * 0.55);
        for col in [20, 32, 64, 96, 112] {
            let result = model.recover(8, col, [1.0, 1.0, truth[2]], ratio);
            assert!(result.recovered, "donor distance at column {col}");
            for (actual, expected) in result.samples.into_iter().zip(truth) {
                assert!((actual - expected).abs() / expected < 0.01);
            }
            assert_eq!(result.samples[2], truth[2]);
        }
    }

    #[test]
    fn one_clipped_channel_matches_truth_with_two_agreeing_survivors() {
        let ratio = [2.4, 1.2, 1.0];
        let model = distance_scene(ratio, [0, 255, 255]);
        let truth = ratio.map(|v| v * 0.55);
        for col in [20, 32, 64, 96, 112] {
            let result = model.recover(8, col, [1.0, truth[1], truth[2]], ratio);
            assert!(result.recovered);
            assert!((result.samples[0] - truth[0]).abs() / truth[0] < 0.01);
            assert_eq!(result.samples[1], truth[1]);
            assert_eq!(result.samples[2], truth[2]);
        }
        let inconsistent = [1.0, 0.3, 0.7];
        let rejected = model.recover(8, 32, inconsistent, ratio);
        assert!(!rejected.recovered);
        assert_eq!(rejected.samples, inconsistent);
    }

    #[test]
    fn equal_channel_neutral_evidence_survives_unequal_source_clip_thresholds() {
        // Simulate unequal physical clipping limits before the samples
        // enter this common radiometric domain. Chroma itself is neutral.
        let model = distance_scene([1.0; 3], [0, 0, 255]);
        for col in [20, 64, 112] {
            let result = model.recover(8, col, [0.45, 0.50, 0.8], [1.0; 3]);
            assert!(result.recovered);
            for actual in result.samples {
                assert!((actual - 0.8).abs() / 0.8 < 0.01);
            }
            assert_eq!(result.samples[2], 0.8);
        }
    }

    #[test]
    fn recovery_is_continuous_through_shoulder_onset_and_hard_clip() {
        let ratio = [2.4, 1.2, 1.0];
        let mut model = distance_scene(ratio, [0, 255, 255]);
        let truth = ratio.map(|v| v * 0.42);
        let mut previous = 0.94;
        for step in 0..=60 {
            let source = 0.94 + step as f64 * 0.001;
            model.reliability.data[8 * 192 + 32][0] =
                channel_reliability(source, 0.001, 0.99, 0.04);
            let result = model.recover(8, 32, [source.min(0.99), truth[1], truth[2]], ratio);
            assert!(result.samples[0] >= previous - 1e-12);
            assert!(result.samples[0] - previous < 0.004);
            assert_eq!(result.samples[1], truth[1]);
            assert_eq!(result.samples[2], truth[2]);
            previous = result.samples[0];
        }
        assert!((previous - truth[0]).abs() / truth[0] < 0.01);
    }

    #[test]
    fn two_clipped_channels_keep_detail_from_each_possible_survivor() {
        for anchor in 0..3 {
            let model = scene(anchor, None);
            let mut low = [1.0; 3];
            let mut high = low;
            low[anchor] = 0.55;
            high[anchor] = 0.65;
            let a = model.recover(15, 35, low, ratios(anchor));
            let b = model.recover(15, 35, high, ratios(anchor));
            assert!(a.recovered && b.recovered);
            assert_eq!(a.samples[anchor], low[anchor]);
            assert_eq!(b.samples[anchor], high[anchor]);
            for c in 0..3 {
                if c != anchor {
                    assert!(a.samples[c] > 1.2);
                    assert!((b.samples[c] - a.samples[c]) > 0.22);
                }
            }
        }
    }

    #[test]
    fn reliable_neutral_chroma_is_valid_evidence() {
        let model = scene(2, None);
        let result = model.recover(15, 35, [1.0, 1.0, 0.6], ratios(2));
        assert!(result.recovered);
        assert!(result.samples[0] > 1.4);
        assert_eq!(result.samples[0], result.samples[1]);
        assert_eq!(result.samples[2], 0.6);
    }

    #[test]
    fn default_preserves_recovered_values_above_legacy_cap() {
        let model = scene(2, None);
        let result = model.recover(15, 35, [1.0, 1.0, 0.85], ratios(2));
        assert!(result.recovered);
        assert!(result.samples[0] > 2.0);
        let capped = scene(2, Some(1.5)).recover(15, 35, [1.0, 1.0, 0.85], ratios(2));
        assert!(capped.recovered);
        assert!(capped.samples[0] <= 1.5);
    }

    #[test]
    fn gain_does_not_turn_healthy_source_samples_into_clipping() {
        let model = scene(2, None);
        let s = [1.2, 1.3, 0.7];
        let result = model.recover(15, 10, s, ratios(2));
        assert_eq!(result.samples, s);
        assert!(!result.damaged);
        assert!(!result.recovered);
    }

    #[test]
    fn inconsistent_materials_do_not_supply_local_chroma() {
        let mut model = scene(2, None);
        let reliability = std::mem::replace(
            &mut model.reliability,
            SensorReliability::new(1, 1, [0.0; 3]).unwrap(),
        );
        let model = LocalRecovery::build(
            reliability,
            |row, _| {
                if row < 16 {
                    [0.48, 0.48, 0.2]
                } else {
                    [0.70, 0.30, 0.2]
                }
            },
            None,
        )
        .unwrap();
        let result = model.recover(15, 35, [1.0, 1.0, 0.6], [1.0; 3]);
        assert!(!result.recovered);
        assert_eq!(result.samples[2], 0.6);
    }

    #[test]
    fn no_surviving_signal_does_not_invent_detail() {
        let mut model = scene(2, None);
        model.reliability.data[15 * 64 + 35] = [0; 3];
        let s = [1.0; 3];
        assert_eq!(model.recover(15, 35, s, [1.0; 3]).samples, s);
        model.reliability.data[15 * 64 + 35] = [0, 0, 255];
        model.reliability.noise[2] = 0.2;
        let s = [1.0, 1.0, 0.6];
        assert_eq!(model.recover(15, 35, s, ratios(2)).samples, s);
    }

    #[test]
    fn inconsistent_prediction_cannot_lower_a_clipping_bound() {
        let model = scene(2, None);
        let s = [1.0, 1.0, 0.2];
        let result = model.recover(15, 35, s, [1.0; 3]);
        assert!(!result.recovered);
        assert_eq!(result.samples, s);
    }

    #[test]
    fn disabling_chroma_retains_survivor_anchored_fallback() {
        let mut model = scene(2, None);
        model.set_chroma_enabled(false);
        let result = model.recover(15, 35, [1.0, 1.0, 0.6], ratios(2));
        assert!(!result.recovered);
        assert!((result.samples[0] - 1.44).abs() < 1e-12);
        assert_eq!(result.samples[2], 0.6);
    }

    #[test]
    fn empty_donors_still_build_and_repeated_evaluation_is_identical() {
        let mut mask = SensorReliability::new(2, 2, [0.001; 3]).unwrap();
        mask.data.fill([0, 0, 255]);
        let model = LocalRecovery::build(mask, |_, _| [0.0; 3], None).unwrap();
        let first = model.recover(0, 0, [1.0, 1.0, 0.6], ratios(2));
        let second = model.recover(0, 0, [1.0, 1.0, 0.6], ratios(2));
        assert_eq!(first, second);
        assert!(!first.recovered);
        assert!(first.samples[0] > 1.0);
    }

    #[test]
    fn clipping_reliability_has_smooth_onset_and_distinct_hard_clip() {
        assert_eq!(channel_reliability(0.001, 0.1, 0.99, 0.04), 255);
        assert_eq!(channel_reliability(0.94, 0.0, 0.99, 0.04), 255);
        assert_eq!(channel_reliability(0.99, 0.0, 0.99, 0.04), 0);
        assert!(channel_reliability(0.989999, 0.0, 0.99, 0.04) > 0);
        let values: Vec<_> = (0..=40)
            .map(|i| channel_reliability(0.95 + i as f64 * 0.001, 0.0, 0.99, 0.04))
            .collect();
        assert!(values.windows(2).all(|pair| pair[0] >= pair[1]));
        assert!(values.windows(2).all(|pair| pair[0] - pair[1] <= 10));
    }
}
