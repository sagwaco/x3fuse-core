//! Native noise settings for a full-size color export with neutral noise sliders.

use crate::{Error, Result};

// SetUpNoiseParamsF20 @0xbf798 and @0xbfbdc, ordinary HI fallback tables.
const ISO: [[f32; 13]; 7] = [
    [100., 1., 1., 10., 0.025, 0.7, 0.5, 4., 1., 1., 0., 1.5, 1.],
    [400., 1., 0.75, 10., 0.1, 0.7, 0.9, 4., 1., 1., 0., 1.5, 1.],
    [800., 1., 0.5, 10., 0.1, 0.7, 1., 4., 1., 1., 0., 1.5, 1.],
    [
        1600., 1., 0.375, 10., 0.25, 0.7, 1., 4., 1., 1., 0.25, 1.5, 1.,
    ],
    [3200., 1., 0.5, 10., 0.5, 0.7, 1., 4., 1., 1., 0.35, 1.5, 1.],
    [
        6400., 1., 0.375, 10., 0.5, 0.7, 1., 4., 1., 1., 0.6, 1.5, 1.,
    ],
    [12800., 1., 0.25, 10., 0.1, 0.7, 1., 4., 1., 1., 1., 1.5, 1.],
];
const ADDITIONAL: [[f32; 13]; 7] = [
    [
        100., 0.025, 0.025, 0., 0.12, 0.0025, 3., 0., 0., 0., 0., 0., 0.,
    ],
    [
        400., 0.05, 0.05, 0., 0.12, 0.0025, 3., 0., 0., 0., 0., 0., 0.,
    ],
    [
        800., 0.075, 0.075, 0.25, 0.12, 0.0025, 3., 0., 0., 0., 0., 0., 0.,
    ],
    [
        1600., 0.1, 0.1, 0.5, 0.12, 0.0025, 3., 0., 0., 0., 0., 0., 0.,
    ],
    [
        3200., 0., 0., 0.75, 0.12, 0.0025, 3., 0., 0., 0., 0., 0., 0.,
    ],
    [6400., 0., 0., 1., 0.12, 0.0025, 3., 0., 0., 0., 0., 0., 0.],
    [12800., 0., 0., 1., 0.12, 0.0025, 3., 0., 0., 0., 0., 0., 0.],
];

#[derive(Clone, Debug)]
pub(crate) struct NoiseParams {
    pub iso: f32,
    pub iso_gain: f32,
    pub despeckle: f32,
    pub scale: f32,
    pub filter_size: u32,
    pub subsample: bool,
    pub noise_scaling: f32,
    pub alpha: f32,
    pub locality: f32,
    pub edge_bias: f32,
    pub edge_strength: f32,
    pub smooth_edge: bool,
    pub accumulate: bool,
    pub decay: f32,
    pub local_mean_iters: u32,
    pub aniso_iters: u32,
    pub aniso_k: u32,
    pub lambda: f32,
    pub aniso_noise_scale: f32,
    pub color_dq_scale: f32,
    pub epsilon: f32,
    pub mixing: f32,
    pub color_mixing: f32,
    pub residual_strength: f32,
    pub column_noise_scale: f32,
    pub rcps_threshold: f32,
    pub rcps_map_threshold: f32,
    pub rcps_blending: f32,
    pub perm_half: u32,
    pub sharpen_amount: f32,
    pub sharpen_limit: f32,
    pub color_model: [[f64; 3]; 3],
    pub intensity_model: [f32; 3],
    pub fpn_suppression: u32,
}

impl Default for NoiseParams {
    fn default() -> Self {
        Self {
            iso: 100.,
            iso_gain: 1.,
            despeckle: 1.,
            scale: 1.,
            filter_size: 21,
            subsample: true,
            noise_scaling: 0.025,
            alpha: 1.,
            locality: 1.,
            edge_bias: 10.,
            edge_strength: 0.7,
            smooth_edge: true,
            accumulate: false,
            decay: 0.5,
            local_mean_iters: 1,
            aniso_iters: 1,
            aniso_k: 2,
            lambda: 0.25,
            aniso_noise_scale: 1.,
            color_dq_scale: 4.,
            epsilon: 5.,
            mixing: 1.,
            color_mixing: 0.,
            residual_strength: 1.5,
            column_noise_scale: 0.375,
            rcps_threshold: 0.075,
            rcps_map_threshold: 0.1,
            rcps_blending: 1.,
            perm_half: 4,
            sharpen_amount: 1.,
            sharpen_limit: 0.25,
            color_model: [
                [7.428315187914495, 10.099755224070439, 1.735415416250626],
                [
                    0.053895495369036,
                    f64::from_bits(0x3fb03975dca5f74a),
                    0.126304867633964,
                ],
                [
                    3.6466580339e-5,
                    f64::from_bits(0x3f1074024cb62984),
                    f64::from_bits(0x3f0daff27f845a95),
                ],
            ],
            intensity_model: [0.; 3],
            fpn_suppression: 0,
        }
    }
}

impl NoiseParams {
    /// The three UI noise sliders use the middle of their five positions.
    pub fn setup_neutral(
        &mut self,
        estimated_iso: f32,
        capture_iso: f32,
        iso_rows: &[[f32; 13]],
        additional_rows: &[[f32; 13]],
        sliders: &[[f32; 13]],
    ) -> Result<()> {
        let invalid = || Error::InvalidData("unsupported SPP noise calibration");
        let iso_rows = if iso_rows.is_empty() {
            &ISO[..]
        } else {
            iso_rows
        };
        let additional = if additional_rows.is_empty() {
            &ADDITIONAL[..]
        } else {
            additional_rows
        };
        for rows in [iso_rows, additional] {
            if !(2..20).contains(&rows.len())
                || !rows.iter().flatten().all(|v| v.is_finite())
                || rows[0][0] <= 0.
                || !rows.windows(2).all(|r| r[0][0] < r[1][0])
            {
                return Err(invalid());
            }
        }
        if !estimated_iso.is_finite()
            || estimated_iso <= 0.
            || !capture_iso.is_finite()
            || capture_iso <= 0.
            || (!sliders.is_empty()
                && (sliders.len() != 5 || !sliders.iter().flatten().all(|v| v.is_finite())))
            || !self.color_model.iter().flatten().all(|v| v.is_finite())
        {
            return Err(invalid());
        }
        // The native picks this factor before it clamps the table ISO.
        self.column_noise_scale = 0.375
            * if estimated_iso < 800. {
                1.
            } else {
                f32::from_bits(0x3faaa993)
            };
        self.iso = estimated_iso.clamp(iso_rows[0][0], iso_rows.last().unwrap()[0]);
        // The native reads one row past this table when estimated ISO exceeds it.
        if self.iso > additional.last().unwrap()[0] {
            return Err(invalid());
        }
        let upper = bracket(iso_rows, self.iso);
        let lo = iso_rows[upper - 1];
        let hi = iso_rows[upper];
        let t = (self.iso - lo[0]) / (hi[0] - lo[0]);
        let value = |c| f64::from(lo[c]).mul_add(1. - f64::from(t), f64::from(t * hi[c]));
        self.despeckle = value(1) as f32;
        self.scale = value(2) as f32;
        self.filter_size = value(3).mul_add(2., 1.) as u32;
        self.noise_scaling = value(4) as f32;
        self.color_dq_scale = value(7) as f32;
        self.mixing = value(9) as f32;
        self.color_mixing = value(10) as f32;
        self.residual_strength = value(11) as f32;
        self.sharpen_amount = value(12) as f32;

        // The native selects the bracket with estimated ISO and interpolates
        // with capture ISO. The two inputs can select extrapolated values.
        let upper = bracket(additional, self.iso);
        let lo = additional[upper - 1];
        let hi = additional[upper];
        let capture = capture_iso.clamp(additional[0][0], additional.last().unwrap()[0]);
        let t = (capture - lo[0]) / (hi[0] - lo[0]);
        let value = |c| (1. - t).mul_add(lo[c], t * hi[c]);
        self.rcps_threshold = value(1);
        self.rcps_map_threshold = value(2);
        self.rcps_blending = value(3).clamp(0., 1.);
        let slider = if sliders.is_empty() {
            [1.; 13]
        } else {
            sliders[2]
        };
        self.aniso_noise_scale *= slider[0];
        self.scale *= slider[2];
        self.filter_size = (self.filter_size & !1)
            .checked_mul(slider[3] as u32)
            .ok_or_else(invalid)?
            | 1;
        self.color_dq_scale *= slider[7];
        self.mixing = (self.mixing * slider[9]).clamp(0., 1.);
        self.color_mixing = (self.color_mixing * slider[10]).clamp(0., 1.);
        self.residual_strength = (self.residual_strength * slider[11]).max(0.);
        self.sharpen_amount = (self.sharpen_amount * slider[12]).max(0.);

        let log_iso = (f64::from(self.iso.ln()) + f64::from_bits(0xc0126bb1bbb55516))
            / f64::from_bits(0xc010a2b23f3bab72);
        self.alpha = 1.;
        let locality_high = log_iso.mul_add(0.1, 0.2);
        self.locality = (locality_high - 1.).mul_add(0.5, 1.) as f32;
        let strength_high = log_iso.mul_add(-0.25, 0.7);
        let strength_low = log_iso.mul_add(f64::from_bits(0x3fb99999999999a0), 0.8);
        self.edge_strength =
            ((strength_high - strength_low).mul_add(0.5, strength_low) as f32).clamp(0., 0.99);
        let decay_low = log_iso.mul_add(f64::from_bits(0xbf9eb851eb851ec0), 0.95);
        self.decay = ((0.95 - decay_low).mul_add(0.5, decay_low) as f32).clamp(0., 1.);
        let raw_scale = self.scale;
        self.scale = ((f64::from(raw_scale).mul_add(8., 0.5) as i32) as f32 * 0.125).max(0.375);
        if self.scale == 1. && raw_scale < 0.984375 {
            self.scale = 0.875;
        }
        self.intensity_model = self.color_model.map(|r| ((r[0] + r[2] + r[1]) / 9.) as f32);
        Ok(())
    }
}

fn bracket(rows: &[[f32; 13]], iso: f32) -> usize {
    rows.iter()
        .position(|r| r[0] >= iso)
        .unwrap_or(rows.len() - 1)
        .max(1)
}

pub(crate) const ISO_REFERENCE: [[f64; 3]; 15] = [
    [0.01, 0.01, 1.],
    [0.05, 0.02, 1.],
    [0.1, 0.03, 1.],
    [0.2, 0.04, 1.],
    [0.3, 0.07, 1.],
    [0.4, 0.1, 1.],
    [0.5, 0.15, 1.],
    [0.6, 0.2, 1.],
    [0.7, 0.25, 1.],
    [0.8, 0.3, 1.],
    [0.9, 0.37, 1.],
    [0.93, 0.42, 1.],
    [0.96, 0.55, 1.],
    [0.98, 0.6, 1.],
    [0.999, 0.95, 1.],
];

/// CalculateEstimatedISO @0xb1cfc. Histogram channels are contiguous byte planes.
#[allow(clippy::too_many_arguments)]
pub(crate) fn estimated_iso(
    histogram: &[u8],
    percentages: &[f32],
    reference: &[[f64; 3]],
    gains: [f64; 3],
    bit_depth: u32,
    use_wb: bool,
    outliers: Option<usize>,
    snap: bool,
) -> Result<f64> {
    let invalid = || Error::InvalidData("invalid SPP ISO estimate calibration");
    if reference.len() > 15
        || bit_depth > 31
        || percentages.is_empty()
        || percentages.len() > 65536
        || histogram.len() != percentages.len() * 3
        || !percentages.iter().all(|v| v.is_finite())
        || !percentages.windows(2).all(|p| p[0] <= p[1])
        || !reference.iter().flatten().all(|v| v.is_finite())
        || reference.iter().any(|r| r[1] < 0. || r[2] < 0.)
        || !reference.windows(2).all(|r| r[0][0] <= r[1][0])
        || !gains.iter().all(|v| v.is_finite() && *v > 0.)
        || outliers.is_some_and(|n| n > 15)
    {
        return Err(invalid());
    }
    let n = reference.len();
    if n == 0 {
        return Ok(0.);
    }
    let minimum = gains.into_iter().fold(f64::INFINITY, f64::min);
    let gains = if use_wb {
        gains.map(|v| (v / minimum) as f32)
    } else {
        [1.; 3]
    };
    if gains
        .iter()
        .any(|v| !v.is_finite() || *v * 255. >= i32::MAX as f32)
    {
        return Err(invalid());
    }
    let maximum = ((1u32 << bit_depth) as i32) as f32 - 1.;
    let mut ratios = [0.; 15];
    let mut weights = [0.; 15];
    let mut valid = [false; 15];
    let mut j = 0;
    let len = percentages.len();
    for (k, row) in reference.iter().enumerate() {
        while j < len && row[0] > f64::from(percentages[j]) {
            j += 1;
        }
        if j == len {
            return Err(invalid());
        }
        let h = std::array::from_fn::<_, 3, _>(|c| gains[c] * f32::from(histogram[c * len + j]));
        let value = h.into_iter().fold(f32::NEG_INFINITY, f32::max) as i32 as u16;
        valid[k] = maximum > f32::from(value);
        weights[k] = (row[2] * f64::from(u8::from(valid[k]))) as f32;
        ratios[k] = if value == 0 {
            1.
        } else {
            (maximum * row[1] as f32) / f32::from(value)
        };
    }
    let good = valid[..n].iter().filter(|&&v| v).count();
    if good == 0 {
        return Ok(0.);
    }
    let sum = |weights: &[f32; 15]| {
        let (mut numerator, mut denominator) = (0., 0.);
        for k in 0..n {
            // The SIMD prefix has separate multiply/add; only the tail is fused.
            numerator = if k < (n & !3) {
                numerator + ratios[k] * weights[k]
            } else {
                ratios[k].mul_add(weights[k], numerator)
            };
            denominator += weights[k];
        }
        (numerator, denominator)
    };
    if let Some(count) = outliers.filter(|&count| good > count) {
        let (numerator, denominator) = sum(&weights);
        if denominator <= 0. {
            return Err(invalid());
        }
        let mean = numerator / denominator;
        let mut deviation = std::array::from_fn::<_, 15, _>(|k| {
            (ratios[k] - mean).abs() * f32::from(u8::from(valid[k]))
        });
        for _ in 0..count {
            let (mut best, mut index) = (-1000., 0);
            for (k, &value) in deviation[..n].iter().enumerate() {
                if value > best {
                    best = value;
                    index = k;
                }
            }
            deviation[index] = 0.;
            weights[index] = 0.;
        }
    }
    let (numerator, denominator) = sum(&weights);
    let value = (f64::from(numerator) * 100.) / f64::from(denominator);
    if !value.is_finite() || value < 0. {
        return Err(invalid());
    }
    if !snap {
        return Ok(value);
    }
    const SNAPPED: [f64; 21] = [
        100., 200., 400., 800., 1600., 3200., 6400., 133.33, 266.67, 533.33, 1066.7, 2133.3,
        4266.7, 8533.3, 166.67, 333.33, 666.67, 1333.3, 2666.7, 5333.3, 10667.,
    ];
    let first = value - 100.;
    let mut best = if first.abs() < 10000. {
        (first as f32).abs()
    } else {
        10000.
    };
    let mut index = 0;
    for (i, &candidate) in SNAPPED.iter().enumerate().skip(1) {
        let distance = value - candidate;
        if distance.abs() < f64::from(best) {
            best = (distance as f32).abs();
            index = i;
        }
    }
    Ok(SNAPPED[index])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimated_iso_matches_native() {
        let mut reference = Vec::new();
        let mut histogram = Vec::new();
        let mut percentages = Vec::new();
        let (mut gain_values, mut scales) = (Vec::new(), Vec::new());
        let (mut cases, mut total) = (0, 0);
        let float = |s| f32::from_bits(u32::from_str_radix(s, 16).unwrap());
        let double = |s| f64::from_bits(u64::from_str_radix(s, 16).unwrap());
        for line in include_str!("../../tests/fixtures/spp-estimated-iso-native.txt").lines() {
            let p: Vec<_> = line.split_whitespace().collect();
            if p.is_empty() {
                continue;
            }
            match p[0] {
                "#" if p.get(1) == Some(&"G") => {
                    gain_values = p[2..].iter().map(|s| double(s)).collect()
                }
                "#" if p.get(1) == Some(&"S") => {
                    scales = p[2..].iter().map(|s| double(s)).collect()
                }
                "REF" => {
                    reference = p[4..]
                        .chunks_exact(3)
                        .map(|r| std::array::from_fn(|c| f64::from(float(r[c]))))
                        .collect()
                }
                "HIST" => {
                    let bytes = p[3].strip_prefix("u8:").unwrap();
                    histogram = (0..bytes.len())
                        .step_by(2)
                        .map(|i| u8::from_str_radix(&bytes[i..i + 2], 16).unwrap())
                        .collect();
                }
                "PCT" => percentages = p[4..].iter().map(|s| float(s)).collect(),
                "CASE" => {
                    let value = |name: &str| p.iter().find_map(|s| s.strip_prefix(name)).unwrap();
                    let number = |name| value(name).parse::<usize>().unwrap();
                    let rows = value("refRows=");
                    let r = match rows {
                        "camf" => reference.clone(),
                        "absent" => ISO_REFERENCE.to_vec(),
                        n => {
                            let count: usize = n.parse().unwrap();
                            let start = p.iter().position(|&s| s == "ref-f32:").unwrap() + 1;
                            p[start..start + 3 * count]
                                .chunks_exact(3)
                                .map(|r| std::array::from_fn(|c| f64::from(float(r[c]))))
                                .collect()
                        }
                    };
                    let depth = number("HistBitDepth=") as u32;
                    let outliers = (number("outliers=") != 0).then(|| number("outliersNum="));
                    let mut data = histogram[..3 * percentages.len()].to_vec();
                    let mut state = 0x9e37_79b9_u32;
                    let (mut count, mut zeros, mut hash) = (0, 0, 0xcbf2_9ce4_8422_2325_u64);
                    for trial in 0..=number("trials=") {
                        if trial != 0 {
                            for byte in &mut data {
                                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                                let v = (state >> 24) as u8;
                                *byte = match (trial - 1) % 5 {
                                    0 => v,
                                    1 => v & 15,
                                    2 => {
                                        if v < 40 {
                                            255
                                        } else {
                                            v
                                        }
                                    }
                                    3 => {
                                        if v < 128 {
                                            0
                                        } else {
                                            v
                                        }
                                    }
                                    _ => 255 - (v & 7),
                                };
                            }
                        }
                        for &red in &gain_values {
                            for &blue in &gain_values {
                                for &scale in &scales {
                                    let gains = [red * scale, scale, blue * scale];
                                    let result = estimated_iso(
                                        &data,
                                        &percentages,
                                        &r,
                                        gains,
                                        depth,
                                        number("WBGains=") != 0,
                                        outliers,
                                        number("snap=") != 0,
                                    )
                                    .unwrap();
                                    count += 1;
                                    zeros += usize::from(result == 0.);
                                    for byte in result.to_bits().to_le_bytes() {
                                        hash = (hash ^ u64::from(byte))
                                            .wrapping_mul(1_099_511_628_211);
                                    }
                                }
                            }
                        }
                    }
                    assert_eq!(count, number("count="), "{line}");
                    assert_eq!(zeros, number("zeros="), "{line}");
                    assert_eq!(
                        hash,
                        u64::from_str_radix(value("hash="), 16).unwrap(),
                        "{line}"
                    );
                    cases += 1;
                    total += count;
                }
                _ => {}
            }
        }
        assert_eq!((cases, total), (64, 1_712_448));
    }

    #[test]
    fn estimated_iso_rejects_invalid_calibration() {
        let run = |hist: &[u8], percent: &[f32], reference: &[[f64; 3]], gains, depth, outliers| {
            estimated_iso(
                hist, percent, reference, gains, depth, true, outliers, false,
            )
        };
        assert!(run(&[1; 3], &[1.], &ISO_REFERENCE, [1.; 3], 8, None).is_ok());
        assert!(run(&[1; 2], &[1.], &ISO_REFERENCE, [1.; 3], 8, None).is_err());
        assert!(run(&[1; 3], &[0.9], &ISO_REFERENCE, [1.; 3], 8, None).is_err());
        assert!(run(&[1; 3], &[1.], &[[0.; 3]; 16], [1.; 3], 8, None).is_err());
        assert!(run(&[1; 3], &[1.], &ISO_REFERENCE, [0.; 3], 8, None).is_err());
        assert!(run(&[1; 3], &[1.], &ISO_REFERENCE, [1.; 3], 32, None).is_err());
        assert!(run(&[1; 3], &[1.], &ISO_REFERENCE, [1.; 3], 8, Some(16)).is_err());
        assert!(run(&[1; 3], &[1.], &[[0., 1., 0.]], [1.; 3], 8, None).is_err());
    }

    #[test]
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    fn neutral_settings_match_native() {
        let mut tables = [Vec::new(), Vec::new(), Vec::new()];
        let mut base = NoiseParams::default();
        let mut cases = 0;
        let hex = |s| u32::from_str_radix(s, 16).unwrap();
        let float = |s| f32::from_bits(hex(s));
        for line in include_str!("../../tests/fixtures/spp-noise-params-native.txt").lines() {
            let p: Vec<_> = line.split_whitespace().collect();
            match p[0] {
                "TABLE" => {
                    let kind: usize = p[2].parse().unwrap();
                    let values: Vec<_> = p[5..].iter().map(|s| float(s)).collect();
                    if kind < 3 {
                        tables[kind] = values
                            .chunks_exact(13)
                            .map(|r| r.try_into().unwrap())
                            .collect();
                    } else {
                        base.color_model = std::array::from_fn(|r| {
                            std::array::from_fn(|c| f64::from(values[r * 3 + c]))
                        });
                    }
                }
                "SCALARS" => {
                    base.subsample = p[2] != "0";
                    base.edge_bias = float(p[3]);
                    base.smooth_edge = p[4] != "0";
                    base.accumulate = p[5] != "0";
                    base.local_mean_iters = p[6].parse().unwrap();
                    base.aniso_iters = p[7].parse().unwrap();
                    base.aniso_k = p[8].parse().unwrap();
                    base.lambda = float(p[9]);
                    base.aniso_noise_scale = float(p[10]);
                    base.epsilon = float(p[11]);
                    base.perm_half = p[12].parse().unwrap();
                    base.sharpen_limit = float(p[13]);
                    base.fpn_suppression = p[14].parse().unwrap();
                }
                "CASE" => {
                    let mut n = base.clone();
                    n.setup_neutral(float(p[2]), float(p[3]), &tables[0], &tables[1], &tables[2])
                        .unwrap();
                    let actual = [
                        n.iso.to_bits(),
                        n.despeckle.to_bits(),
                        n.scale.to_bits(),
                        n.filter_size,
                        n.subsample as u32,
                        n.noise_scaling.to_bits(),
                        n.alpha.to_bits(),
                        n.locality.to_bits(),
                        n.edge_bias.to_bits(),
                        n.edge_strength.to_bits(),
                        n.smooth_edge as u32,
                        n.accumulate as u32,
                        n.decay.to_bits(),
                        n.local_mean_iters,
                        n.aniso_iters,
                        n.aniso_k,
                        n.lambda.to_bits(),
                        n.aniso_noise_scale.to_bits(),
                        n.color_dq_scale.to_bits(),
                        n.epsilon.to_bits(),
                        n.mixing.to_bits(),
                        n.color_mixing.to_bits(),
                        n.residual_strength.to_bits(),
                        n.column_noise_scale.to_bits(),
                        n.rcps_threshold.to_bits(),
                        n.rcps_map_threshold.to_bits(),
                        n.rcps_blending.to_bits(),
                        n.perm_half,
                        n.sharpen_amount.to_bits(),
                        n.sharpen_limit.to_bits(),
                        n.intensity_model[0].to_bits(),
                        n.intensity_model[1].to_bits(),
                        n.intensity_model[2].to_bits(),
                        n.fpn_suppression,
                    ];
                    let expected: Vec<_> = p[4..].iter().map(|s| hex(s)).collect();
                    for (field, (got, want)) in actual.into_iter().zip(expected).enumerate() {
                        assert_eq!(got, want, "case {cases}, field {field}: {line}");
                    }
                    cases += 1;
                }
                _ => panic!("invalid fixture line"),
            }
        }
        assert_eq!(cases, 176);
    }

    #[test]
    fn rejects_invalid_noise_tables() {
        let mut n = NoiseParams::default();
        assert!(n.setup_neutral(100., 100., &[], &[], &[]).is_ok());
        assert!(n.setup_neutral(f32::NAN, 100., &[], &[], &[]).is_err());
        assert!(n.setup_neutral(100., 100., &ISO[..1], &[], &[]).is_err());
        let mut rows = ISO;
        rows[1][0] = rows[0][0];
        assert!(n.setup_neutral(100., 100., &rows, &[], &[]).is_err());
        assert!(n.setup_neutral(100., 100., &[], &[], &ISO).is_err());
        assert!(n
            .setup_neutral(12800., 100., &[], &ADDITIONAL[..6], &[])
            .is_err());
    }
}
