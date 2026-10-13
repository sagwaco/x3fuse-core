//! Estimate a starting brightness for each affected Merrill pixel.
//! Use camera luminance where possible and layer-based estimates for severe clipping.
//! Nearby pixels supply no color at this stage. Missing brightness is estimated,
//! not measured.

pub(super) struct Result {
    pub samples: [f64; 3],
    pub amplitude: f64,
    pub strength: f64,
    pub available: bool,
}

pub(super) fn recover(measured: [f64; 3], mask: [u8; 3], neutral: [f64; 3]) -> Result {
    let mut result = Result {
        samples: measured,
        amplitude: 0.0,
        strength: 0.0,
        available: false,
    };
    if mask == [255; 3]
        || measured.iter().any(|v| !v.is_finite())
        || neutral.iter().any(|v| !v.is_finite() || *v <= 0.0)
    {
        return result;
    }
    let weights = mask.map(|v| (v as f64 / 255.0).powi(2));
    let support: f64 = weights.iter().sum();
    if support == 0.0 {
        // No reliable layer remains. This estimate cannot supply texture or color.
        return result;
    }
    let amplitude = (0..3)
        .map(|c| weights[c] * (measured[c] / neutral[c]))
        .sum::<f64>()
        / support;
    if !amplitude.is_finite() || amplitude < 0.0 {
        return result;
    }
    // Recovery strength rises smoothly as the lowest layer reliability falls.
    // Nearby color and the rendered RGB values do not set this strength.
    let damage = 1.0 - mask.into_iter().min().unwrap_or(255) as f64 / 255.0;
    let t = (2.0 * damage).clamp(0.0, 1.0);
    let strength = t * t * (3.0 - 2.0 * t);
    result.samples = std::array::from_fn(|c| {
        if strength == 1.0 {
            amplitude * neutral[c]
        } else {
            measured[c] + strength * (amplitude * neutral[c] - measured[c])
        }
    });
    result.amplitude = amplitude;
    result.strength = strength;
    result.available = true;
    result
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a.into_iter().zip(b).map(|(a, b)| a * b).sum()
}

fn recover_luminance(measured: [f64; 3], mask: [u8; 3], neutral: [f64; 3], y: [f64; 3]) -> Result {
    let mut result = recover(measured, mask, neutral);
    if !result.available {
        return result;
    }
    let neutral_y = dot(y, neutral);
    let target_y = dot(y, measured);
    if !neutral_y.is_finite()
        || neutral_y <= 1e-12
        || !target_y.is_finite()
        || target_y <= 0.0
        || !(target_y / neutral_y).is_finite()
    {
        return Result {
            samples: measured,
            amplitude: 0.0,
            strength: 0.0,
            available: false,
        };
    }
    let amplitude = target_y / neutral_y;
    result.samples = std::array::from_fn(|c| {
        if result.strength == 1.0 {
            amplitude * neutral[c]
        } else {
            measured[c] + result.strength * (amplitude * neutral[c] - measured[c])
        }
    });
    result.amplitude = amplitude;
    result
}

// Severe clipping uses the neutral layer ratios to estimate brightness.
// That estimate is not a physical brightness bound for colored material.
pub(super) fn recover_severe(
    measured: [f64; 3],
    mask: [u8; 3],
    neutral: [f64; 3],
    y: [f64; 3],
) -> Result {
    let hold = recover_luminance(measured, mask, neutral, y);
    let mut sorted = mask;
    sorted.sort_unstable();
    if sorted[1] >= 128
        || measured.iter().any(|v| !v.is_finite() || *v < 0.0)
        || neutral.iter().any(|v| !v.is_finite() || *v <= 0.0)
    {
        return hold;
    }
    let neutral_y = dot(y, neutral);
    let measured_y = dot(y, measured);
    if !neutral_y.is_finite() || neutral_y <= 1e-12 || !measured_y.is_finite() {
        return hold;
    }
    let h = |x: f64| {
        let x = x.clamp(0.0, 1.0);
        x * x * (3.0 - 2.0 * x)
    };
    let severity = h(1.0 - 2.0 * sorted[1] as f64 / 255.0);
    let trust = h(2.0 * sorted[2] as f64 / 255.0);
    let amplitudes: [f64; 3] = std::array::from_fn(|c| measured[c] / neutral[c]);
    let envelope = amplitudes.into_iter().fold(0.0_f64, f64::max);
    let weights = mask.map(|v| (v as f64 / 255.0).powi(2));
    let support: f64 = weights.iter().sum();
    let mean = if support > 0.0 {
        dot(weights, amplitudes) / support
    } else {
        envelope
    };
    let guarded = trust * mean + (1.0 - trust) * envelope;
    // Severe clipping can make matrix luminance negative. A zero floor
    // prevents that value from reducing the layer-based brightness estimate
    // and keeps the blend continuous at zero luminance.
    let held = (measured_y / neutral_y).max(0.0);
    let amplitude = held + severity * (guarded - held);
    if !amplitude.is_finite() || amplitude <= 0.0 {
        return hold;
    }
    // With two layers below half reliability, recovery strength is exactly one.
    // Include fully clipped pixels so the estimate remains continuous.
    Result {
        samples: neutral.map(|v| amplitude * v),
        amplitude,
        strength: 1.0,
        available: true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    const P: [f64; 3] = [0.27, 0.59, 1.0];

    fn close(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-12, "{a} != {b}");
    }

    const Y: [f64; 3] = [-2.79537849, 3.41456424, -0.2476];

    #[test]
    fn severe_keeps_two_trusted_layers_exactly_hold() {
        for k in 0..=255 {
            for mask in [[255, 255, k], [128, k, 255], [k, 128, 128]] {
                let s = [0.32, 0.72, 1.0];
                let a = recover_severe(s, mask, P, Y);
                let b = recover_luminance(s, mask, P, Y);
                assert_eq!(a.samples, b.samples);
                assert_eq!(a.amplitude, b.amplitude);
                assert_eq!(a.available, b.available);
                assert_eq!(a.strength, b.strength);
            }
        }
    }

    #[test]
    fn severe_recovers_single_survivor_without_spatial_averaging() {
        for amplitude in [2.0, 2.5, 3.0] {
            let r = recover_severe([P[0] * amplitude, 1.0, 1.0], [255, 0, 0], P, Y);
            close(r.amplitude, amplitude);
            close(r.strength, 1.0);
        }
    }

    #[test]
    fn severe_weak_weights_cannot_take_over_the_estimate() {
        let s = [0.99, 0.98, 0.98];
        let envelope = s[0] / P[0];
        for mask in [[1, 1, 0], [1, 1, 2], [2, 0, 0], [2, 2, 4]] {
            assert!((recover_severe(s, mask, P, Y).amplitude - envelope).abs() < 0.02);
        }
    }

    #[test]
    fn severe_zero_support_has_continuous_limit_not_a_dark_hole() {
        let s = [0.99, 0.98, 0.98];
        let zero = recover_severe(s, [0; 3], P, Y);
        assert!(zero.available);
        close(zero.amplitude, s[0] / P[0]);
        for mask in [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]] {
            assert!((recover_severe(s, mask, P, Y).amplitude - zero.amplitude).abs() < 0.001);
        }
        assert!(!recover_luminance(s, [0; 3], P, Y).available);
    }

    #[test]
    fn severe_gate_has_no_large_entry_step() {
        let s = [0.85, 0.98, 1.0];
        let a = recover_severe(s, [255, 128, 0], P, Y);
        let b = recover_severe(s, [255, 127, 0], P, Y);
        assert!((a.amplitude - b.amplitude).abs() < 0.0002);
    }

    #[test]
    fn severe_scaling_remains_linear_without_a_baked_shoulder() {
        let s = [0.95, 0.98, 1.0];
        for mask in [[255, 1, 0], [2, 2, 4], [0; 3]] {
            let a = recover_severe(s, mask, P, Y);
            let b = recover_severe(s.map(|v| v * 4.0), mask, P, Y);
            close(b.amplitude, 4.0 * a.amplitude);
        }
    }

    #[test]
    fn severe_retains_invalid_input_and_calibration_policy() {
        for (s, p, y) in [
            ([-0.1, 0.8, 1.0], P, Y),
            ([0.8, 0.9, 1.0], P, [f64::INFINITY, 0.0, 0.0]),
            ([0.8, 0.9, 1.0], [0.0, 0.5, 1.0], Y),
            ([0.8, 0.9, 1.0], P, [0.0; 3]),
        ] {
            assert_eq!(
                recover_severe(s, [1, 1, 0], p, y).samples,
                recover_luminance(s, [1, 1, 0], p, y).samples
            );
        }
    }

    #[test]
    fn severe_negative_luminance_uses_layer_estimate() {
        // Clipped DP2 Merrill sensor samples produce negative matrix luminance.
        // Uniform output scaling preserves the recovery decision.
        let p = [0.4 / 1.4262, 0.8658 / 1.4262, 1.0];
        let y = [-0.7552 / p[0], 2.0028 / p[1], -0.2476];
        for s in [
            [17321.0 / 65535.0, 15221.0 / 65535.0, 16137.0 / 65535.0],
            [17250.0 / 65535.0, 15160.0 / 65535.0, 16071.0 / 65535.0],
        ] {
            assert!(dot(y, s) < 0.0);
            for mask in [[0; 3], [255, 0, 0], [1, 1, 0], [255, 127, 0]] {
                let r = recover_severe(s, mask, p, y);
                assert!(r.available && r.amplitude > 0.0);
                close(r.strength, 1.0);
                assert!(dot(y, r.samples) > 0.0);
                for c in 0..3 {
                    close(r.samples[c], r.amplitude * p[c]);
                }
                if mask == [0; 3] || mask == [255, 0, 0] {
                    close(r.amplitude, s[0] / p[0]);
                }
            }
            for mask in [[255; 3], [255, 255, 0], [128; 3]] {
                let r = recover_severe(s, mask, p, y);
                assert!(!r.available);
                assert_eq!(r.samples, s);
            }
        }
    }

    #[test]
    fn severe_is_continuous_across_zero_matrix_luminance() {
        let s = [0.8, -(Y[0] * 0.8 + Y[2]) / Y[1], 1.0];
        for mask in [[0; 3], [255, 0, 0], [255, 64, 0], [255, 127, 0]] {
            let mut low = s;
            let mut high = s;
            low[1] -= 1e-9;
            high[1] += 1e-9;
            assert!(dot(Y, low) < 0.0 && dot(Y, high) > 0.0);
            let a = recover_severe(low, mask, P, Y);
            let b = recover_severe(high, mask, P, Y);
            assert!(a.available && b.available);
            assert!((a.amplitude - b.amplitude).abs() < 1e-7);
        }
    }

    #[test]
    fn severe_negative_luminance_scales_linearly() {
        let s = [0.8, 0.3, 1.0];
        assert!(dot(Y, s) < 0.0);
        for mask in [[0; 3], [255, 0, 0], [2, 2, 4], [255, 127, 0]] {
            let a = recover_severe(s, mask, P, Y);
            let b = recover_severe(s.map(|v| v * 4.0), mask, P, Y);
            assert!(a.available && b.available);
            close(b.amplitude, 4.0 * a.amplitude);
        }
    }

    #[test]
    fn hold_preserves_measured_luminance_through_entire_onset() {
        let s = [0.32, 0.72, 1.0];
        for reliability in 0..=255 {
            let r = recover_luminance(s, [255, 255, reliability], P, Y);
            close(dot(Y, r.samples), dot(Y, s));
        }
    }

    #[test]
    fn hold_preserves_healthy_and_unrecoverable_samples() {
        for mask in [[255; 3], [0; 3]] {
            let s = [0.32, 0.72, 1.0];
            assert_eq!(recover_luminance(s, mask, P, Y).samples, s);
        }
        let s = [1.0, 0.01, 1.0];
        let r = recover_luminance(s, [255, 255, 0], P, Y);
        assert!(!r.available);
        assert_eq!(r.samples, s);
    }

    #[test]
    fn healthy_colors_are_unchanged_even_if_extreme() {
        for s in [[0.5, 0.2, 0.8], [-0.01, 0.4, 0.9], [0.3, 0.8, 1.2]] {
            assert_eq!(recover(s, [255; 3], P).samples, s);
        }
    }

    #[test]
    fn top_clipped_uses_equal_normalized_bottom_and_middle() {
        let r = recover([0.27 * 1.2, 0.59 * 1.4, 1.0], [255, 255, 0], P);
        close(r.amplitude, 1.3);
        assert!(r.available);
        close(r.strength, 1.0);
        for (v, p) in r.samples.into_iter().zip(P) {
            close(v / p, 1.3);
        }
    }

    #[test]
    fn clipped_layer_variation_cannot_drive_tone() {
        let a = recover([0.32, 0.72, 1.0], [255, 255, 0], P).samples;
        let b = recover([0.32, 0.72, 6.0], [255, 255, 0], P).samples;
        assert_eq!(a, b);
    }

    #[test]
    fn survivor_detail_is_linear_and_not_spatially_averaged() {
        for amplitude in [0.8, 1.0, 1.2, 1.6, 2.5] {
            let r = recover([P[0] * amplitude, P[1] * amplitude, 1.0], [255, 255, 0], P);
            close(r.amplitude, amplitude);
        }
    }

    #[test]
    fn any_single_survivor_can_supply_tone() {
        for c in 0..3 {
            let mut mask = [0; 3];
            mask[c] = 255;
            let mut s = [1.0; 3];
            s[c] = P[c] * 1.7;
            close(recover(s, mask, P).amplitude, 1.7);
        }
    }

    #[test]
    fn no_survivor_does_not_invent_detail() {
        let s = [0.5, 1.0, 1.0];
        let r = recover(s, [0; 3], P);
        assert!(!r.available);
        assert_eq!(r.samples, s);
    }

    #[test]
    fn onset_and_channel_dropout_have_no_large_switch() {
        let s = [0.32, 0.72, 1.0];
        let near = recover(s, [255, 255, 254], P);
        for (a, b) in near.samples.into_iter().zip(s) {
            assert!((a - b).abs() < 0.0001);
        }
        let a = recover(s, [255, 255, 1], P);
        let b = recover(s, [255, 255, 0], P);
        assert!((a.amplitude - b.amplitude).abs() < 0.00001);
    }

    #[test]
    fn no_hidden_floor_from_clipped_measurement() {
        let r = recover([P[0] * 0.9, P[1] * 0.9, 1.2], [255, 255, 0], P);
        close(r.samples[2], 0.9);
        // Deliberately a tone-only assumption, not a physically constrained
        // reconstruction of missing spectral measurements.
    }
}
