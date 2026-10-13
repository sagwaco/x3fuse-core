//! Exp-10: noise-aware boundaries for the Merrill highlight-color graph.
//!
//! Smooth reliable log-layer measurements and log-layer ratios, then retain
//! weak edges connected to strong ones. This smooths only the boundary guide;
//! the color solver and its source measurements keep their native resolution.
use super::{layer_trust, LayerSample, MIN_DONOR_SIGNAL};
use crate::{Control, Error::InvalidData};
use rayon::prelude::*;

const SIGMA: f64 = 0.85;
const RADIUS: usize = 3; // Round(4 * SIGMA), as in the tested Gaussian guide.
const MIN_LOG_SIGNAL: f32 = 1e-5;
const MIN_SUPPORT: f32 = 1e-6;
const LAYER_EDGE_FLOOR: f64 = 0.035;
const RATIO_EDGE_FLOOR: f64 = 0.025;
const NOISE_MULTIPLIER: f64 = 3.0;
const WEAK_EDGE: f32 = 0.6;
const NATIVE_EDGE_RATIO: f32 = 1.2;
const CHECK_INTERVAL: usize = 65_536;
const STRONG: u8 = 4;
const NATIVE_SHIFT: u8 = 6;
const CUT_SHIFT: u8 = 4;

pub(super) struct Edges {
    cols: usize,
    bits: Vec<u8>, // Guide footprint in bits 0..1, native cuts in bits 4..5.
}

impl Edges {
    pub fn build(
        rows: usize,
        cols: usize,
        sample: impl Fn(usize, usize) -> LayerSample + Sync,
        control: Control<'_>,
    ) -> crate::Result<Self> {
        control.check()?;
        let size = rows
            .checked_mul(cols)
            .ok_or(InvalidData("edge dimensions overflow"))?;
        if size == 0 {
            return Ok(Self {
                cols,
                bits: Vec::new(),
            });
        }
        // Store threshold membership in one byte and build one feature at a
        // time. Caching all six log/trust planes costs 24 bytes per source pixel.
        let mut bits = vec![0; size];
        let mut noise = [0.0; 3];
        for feature in 0_usize..6 {
            control.check()?;
            let pair = [(0, 1), (0, 2), (1, 2)][feature.saturating_sub(3)];
            let mut values = vec![[0.0_f32; 2]; size];
            values
                .par_chunks_mut(cols)
                .enumerate()
                .try_for_each(|(row, output)| {
                    control.check()?;
                    for (col, out) in output.iter_mut().enumerate() {
                        let LayerSample {
                            measured: m,
                            mask: q,
                            repaired,
                        } = sample(row, col);
                        let layer = |k: usize| {
                            let value = m[k] as f32;
                            if repaired || !value.is_finite() {
                                return [MIN_LOG_SIGNAL.ln(), 0.0];
                            }
                            [
                                value.max(MIN_LOG_SIGNAL).ln(),
                                layer_trust(value as f64, q[k], repaired) as f32,
                            ]
                        };
                        *out = if feature < 3 {
                            layer(feature)
                        } else {
                            let a = layer(pair.0);
                            let b = layer(pair.1);
                            [a[0] - b[0], a[1] * b[1]]
                        };
                    }
                    Ok::<_, crate::Error>(())
                })?;
            let value_weight = |i: usize| (values[i][0], values[i][1]);
            let guide = gaussian_guide(&value_weight, rows, cols, control)?;
            let threshold = if feature < 3 {
                let mut residuals = Vec::new();
                for i in 0..size {
                    if i % CHECK_INTERVAL == 0 {
                        control.check()?;
                    }
                    let (value, weight) = value_weight(i);
                    if weight > 0.99 && value > 0.05_f32.ln() {
                        residuals.push((value - guide[i]).abs());
                    }
                }
                // ponytail: scene residuals also measure texture. Keep the
                // tested scale; use calibrated signal-dependent sensor noise
                // if a broader corpus requires separating the two.
                noise[feature] = median(&mut residuals) as f64 / 0.6745 * 0.35;
                LAYER_EDGE_FLOOR.max(NOISE_MULTIPLIER * noise[feature])
            } else {
                RATIO_EDGE_FLOOR.max(NOISE_MULTIPLIER * noise[pair.0].hypot(noise[pair.1]))
            };
            bits.par_chunks_mut(cols)
                .enumerate()
                .try_for_each(|(row, output)| {
                    control.check()?;
                    for (col, bits) in output.iter_mut().enumerate() {
                        let i = row * cols + col;
                        for (axis, j, valid) in
                            [(0, i + 1, col + 1 < cols), (1, i + cols, row + 1 < rows)]
                        {
                            if valid {
                                // Smoothing erases fine alternating patterns and
                                // their residuals inflate the noise estimate.
                                // Retain strong native steps in intact layers.
                                if feature < 3
                                    && values[i][1].min(values[j][1]) == 1.0
                                    && values[i][0].min(values[j][0])
                                        > (MIN_DONOR_SIGNAL as f32).ln()
                                    && (values[i][0] - values[j][0]).abs() > NATIVE_EDGE_RATIO.ln()
                                {
                                    *bits |= STRONG | (1 << axis) | (1 << (axis + NATIVE_SHIFT));
                                }
                                let step = (guide[i] - guide[j]).abs()
                                    * value_weight(i).1.min(value_weight(j).1).sqrt();
                                let score = (step as f64 / threshold) as f32;
                                if score >= WEAK_EDGE {
                                    *bits |= 1 << axis;
                                    if (values[i][0] - values[j][0]).abs()
                                        * values[i][1].min(values[j][1]).sqrt()
                                        >= threshold as f32
                                    {
                                        *bits |= 1 << (CUT_SHIFT + axis);
                                    }
                                }
                                if score >= 1.0 {
                                    *bits |= STRONG;
                                }
                            }
                        }
                    }
                    Ok::<_, crate::Error>(())
                })?;
        }
        Self::connect_weak_edges(bits, rows, cols, control)
    }

    fn connect_weak_edges(
        bits: Vec<u8>,
        rows: usize,
        cols: usize,
        control: Control<'_>,
    ) -> crate::Result<Self> {
        let mut guide = vec![0; bits.len()];
        let mut stack = Vec::new();
        let mut visited = 0;
        // Hysteresis joins the guide footprint. Native localization below can
        // remove its halo without shrinking the region refined at pixel scale.
        const VISITED: u8 = 4;
        for seed in 0..bits.len() {
            if seed % CHECK_INTERVAL == 0 {
                control.check()?;
            }
            if guide[seed] & VISITED != 0 || bits[seed] & STRONG == 0 {
                continue;
            }
            guide[seed] |= VISITED;
            stack.push(seed);
            while let Some(i) = stack.pop() {
                visited += 1;
                if visited % CHECK_INTERVAL == 0 {
                    control.check()?;
                }
                let (row, col) = (i / cols, i % cols);
                for j in [
                    (row > 0).then(|| i - cols),
                    (row + 1 < rows).then_some(i + cols),
                    (col > 0).then(|| i - 1),
                    (col + 1 < cols).then_some(i + 1),
                ]
                .into_iter()
                .flatten()
                {
                    if bits[j] & 3 != 0 && guide[j] & VISITED == 0 {
                        guide[j] |= VISITED;
                        stack.push(j);
                    }
                }
            }
        }
        for (i, out) in guide.iter_mut().enumerate() {
            if i % CHECK_INTERVAL == 0 {
                control.check()?;
            }
            *out = if *out & VISITED != 0 { bits[i] & 3 } else { 0 };
        }
        guide
            .par_chunks_mut(cols)
            .enumerate()
            .try_for_each(|(row, output)| {
                control.check()?;
                for (col, out) in output.iter_mut().enumerate() {
                    for axis in 0..2 {
                        if *out & (1 << axis) == 0 {
                            continue;
                        }
                        let native = 1 << (NATIVE_SHIFT + axis);
                        // Two strong native transitions within the kernel
                        // footprint identify a fine structure whose guides can
                        // overlap. Localize only there; weak/noisy boundaries
                        // elsewhere retain the tested smoothed guide.
                        let narrow = if axis == 0 {
                            (col.saturating_sub(RADIUS)..=(col + RADIUS).min(cols - 1))
                                .filter(|&c| bits[row * cols + c] & native != 0)
                                .take(2)
                                .count()
                                == 2
                        } else {
                            (row.saturating_sub(RADIUS)..=(row + RADIUS).min(rows - 1))
                                .filter(|&r| bits[r * cols + col] & native != 0)
                                .take(2)
                                .count()
                                == 2
                        };
                        if bits[row * cols + col] & (native | (1 << (CUT_SHIFT + axis))) != 0
                            || !narrow
                        {
                            *out |= 1 << (CUT_SHIFT + axis);
                        }
                    }
                }
                Ok::<_, crate::Error>(())
            })?;
        control.check()?;
        Ok(Self { cols, bits: guide })
    }

    /// The smoothed guide's footprint marks cells whose averages could mix
    /// surfaces, including links beside the actual native transition.
    pub fn near_boundary(&self, a: (usize, usize), b: (usize, usize)) -> bool {
        self.link_bits(a, b) & 1 != 0
    }

    /// Both orders of the same adjacent pair must cut the same solver link.
    pub fn blocked(&self, a: (usize, usize), b: (usize, usize)) -> bool {
        self.link_bits(a, b) & (1 << CUT_SHIFT) != 0
    }

    fn link_bits(&self, a: (usize, usize), b: (usize, usize)) -> u8 {
        debug_assert_eq!(a.0.abs_diff(b.0) + a.1.abs_diff(b.1), 1);
        if a.0 == b.0 {
            self.bits[a.0 * self.cols + a.1.min(b.1)]
        } else {
            self.bits[a.0.min(b.0) * self.cols + a.1] >> 1
        }
    }
}

fn median(values: &mut [f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let mid = values.len() / 2;
    let even = values.len() % 2 == 0;
    let (lower, value, _) = values.select_nth_unstable_by(mid, f32::total_cmp);
    if even {
        (*value + lower.iter().copied().fold(0.0, f32::max)) * 0.5
    } else {
        *value
    }
}

/// Separable, normalized convolution with nearest-pixel border extension.
/// Keep only two intermediate scanlines per worker, not full value, weight,
/// numerator, denominator and temporary planes. Round each pass to f32 exactly
/// as the reference guide does, including the order of symmetric accumulation.
fn gaussian_guide(
    value_weight: &(impl Fn(usize) -> (f32, f32) + Sync),
    rows: usize,
    cols: usize,
    control: Control<'_>,
) -> crate::Result<Vec<f32>> {
    let mut kernel: [f64; 7] =
        std::array::from_fn(|k| (-0.5 * ((k as f64 - RADIUS as f64) / SIGMA).powi(2)).exp());
    let total: f64 = kernel.iter().sum();
    for k in &mut kernel {
        *k /= total;
    }
    let mut guide = vec![0.0; rows * cols];
    guide.par_chunks_mut(cols).enumerate().try_for_each_init(
        || (vec![0.0_f32; cols], vec![0.0_f32; cols]),
        |(numerator, denominator), (row, output)| {
            control.check()?;
            for col in 0..cols {
                let (v, w) = value_weight(row * cols + col);
                let mut n = (v * w) as f64 * kernel[RADIUS];
                let mut d = w as f64 * kernel[RADIUS];
                for distance in (1..=RADIUS).rev() {
                    let (av, aw) = value_weight(row.saturating_sub(distance) * cols + col);
                    let (bv, bw) = value_weight((row + distance).min(rows - 1) * cols + col);
                    n += ((av * aw) as f64 + (bv * bw) as f64) * kernel[RADIUS - distance];
                    d += (aw as f64 + bw as f64) * kernel[RADIUS - distance];
                }
                numerator[col] = n as f32;
                denominator[col] = d as f32;
            }
            for (col, out) in output.iter_mut().enumerate() {
                let mut n = numerator[col] as f64 * kernel[RADIUS];
                let mut d = denominator[col] as f64 * kernel[RADIUS];
                for distance in (1..=RADIUS).rev() {
                    let a = col.saturating_sub(distance);
                    let b = (col + distance).min(cols - 1);
                    n += (numerator[a] as f64 + numerator[b] as f64) * kernel[RADIUS - distance];
                    d +=
                        (denominator[a] as f64 + denominator[b] as f64) * kernel[RADIUS - distance];
                }
                *out = n as f32 / (d as f32).max(MIN_SUPPORT);
            }
            Ok::<_, crate::Error>(())
        },
    )?;
    Ok(guide)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weak_links_need_a_connected_strong_seed_and_are_symmetric() {
        let bits = vec![STRONG | 1, 1, 0, 1, 0];
        let edges = Edges::connect_weak_edges(bits, 1, 5, Control::none()).unwrap();
        assert!(edges.blocked((0, 0), (0, 1)));
        assert!(edges.blocked((0, 2), (0, 1)));
        assert!(!edges.blocked((0, 3), (0, 4)));
    }

    #[test]
    fn only_surviving_unrepaired_layers_can_supply_a_boundary() {
        for transpose in [false, true] {
            for supported in [false, true] {
                let edges = Edges::build(
                    24,
                    24,
                    |row, col| {
                        let step = if (if transpose { row } else { col }) < 12 {
                            0.3
                        } else {
                            0.5
                        };
                        LayerSample {
                            measured: [step, 0.4, 0.8],
                            mask: [if supported { 255 } else { 0 }, 255, 255],
                            repaired: false,
                        }
                    },
                    Control::none(),
                )
                .unwrap();
                let (a, b) = if transpose {
                    ((11, 12), (12, 12))
                } else {
                    ((12, 11), (12, 12))
                };
                assert_eq!(edges.blocked(a, b), supported);
                assert_eq!(edges.blocked(b, a), supported);
            }
        }
        let repaired = Edges::build(
            1,
            2,
            |_, col| LayerSample {
                measured: [0.3 + col as f64, 0.4, 0.8],
                mask: [255; 3],
                repaired: true,
            },
            Control::none(),
        )
        .unwrap();
        assert!(!repaired.blocked((0, 0), (0, 1)));
    }

    #[test]
    fn constant_guides_and_pre_cancelled_work_do_not_create_edges() {
        let edges = Edges::build(
            3,
            7,
            |_, _| LayerSample {
                measured: [0.2, 0.4, 0.8],
                mask: [255; 3],
                repaired: false,
            },
            Control::none(),
        )
        .unwrap();
        assert!(edges.bits.iter().all(|v| *v == 0));
        let cancel = std::sync::atomic::AtomicBool::new(true);
        assert!(matches!(
            Edges::build(1, 1, |_, _| unreachable!(), Control::new(&cancel)),
            Err(crate::Error::Cancelled)
        ));
    }

    #[test]
    fn illumination_ramp_keeps_donors_connected_through_partial_clipping() {
        let edges = Edges::build(
            45,
            32,
            |row, col| {
                let exposure = 0.65 + 0.02 * row as f64;
                let measured = [0.284206, 0.604436, 1.0].map(|v| v * exposure);
                let measured = std::array::from_fn(|k| {
                    (measured[k] * if col >= 16 && k < 2 { 1.25 } else { 1.0 }).min(1.0)
                });
                LayerSample {
                    mask: measured.map(|v| {
                        crate::highlight_recovery::channel_reliability(v, 0.0, 0.99, 0.04)
                    }),
                    measured,
                    repaired: false,
                }
            },
            Control::none(),
        )
        .unwrap();
        for row in 15..42 {
            // Illumination changes alone must not cut the path to intact donors,
            // across the donor-to-clipped transition and inside the highlight.
            for col in [8, 24] {
                assert!(
                    !edges.blocked((row, col), (row + 1, col)),
                    "false exposure boundary at {row},{col}"
                );
            }
            assert!(
                edges.blocked((row, 15), (row, 16)),
                "lost material boundary at {row}"
            );
        }
    }

    #[test]
    fn noisy_guide_keeps_a_supported_step_without_cutting_the_flat_region() {
        let rows = 96;
        let cols = 96;
        let edges = Edges::build(
            rows,
            cols,
            |row, col| {
                // Deterministic zero-mean high-frequency variation, large enough for
                // residual noise to exceed the fixed floor. Individual noise steps
                // stay below the native strong-edge threshold.
                let hash = ((row * cols + col) as u32)
                    .wrapping_mul(747796405)
                    .wrapping_add(2891336453);
                let hash = ((hash >> ((hash >> 28) + 4)) ^ hash).wrapping_mul(277803737);
                let noise = (((hash >> 22) ^ hash) as f64 / u32::MAX as f64 - 0.5) * 0.16;
                LayerSample {
                    measured: [
                        0.4 * (noise + if col >= 48 { 0.17 } else { 0.0 }).exp(),
                        0.5,
                        0.8,
                    ],
                    mask: [255; 3],
                    repaired: false,
                }
            },
            Control::none(),
        )
        .unwrap();
        let boundary = (0..rows)
            .filter(|&r| edges.blocked((r, 47), (r, 48)))
            .count();
        let interior = (0..rows)
            .flat_map(|r| (0..40).map(move |c| (r, c)))
            .filter(|&(r, c)| edges.blocked((r, c), (r, c + 1)))
            .count();
        assert!(
            boundary > rows * 3 / 4,
            "supported boundary coverage {boundary}/{rows}"
        );
        assert!(
            interior < rows,
            "noise cut {interior} of {} flat-region links",
            rows * 40
        );
    }

    #[test]
    fn narrow_images_and_border_steps_are_transpose_invariant() {
        for (rows, cols) in [(1, 1), (1, 17), (2, 17), (5, 19)] {
            let sample = |r, c| LayerSample {
                measured: [
                    if c < cols / 2 { 0.3 } else { 0.5 },
                    0.4 + r as f64 * 0.001,
                    0.8,
                ],
                mask: [255; 3],
                repaired: false,
            };
            let a = Edges::build(rows, cols, sample, Control::none()).unwrap();
            let b = Edges::build(cols, rows, |r, c| sample(c, r), Control::none()).unwrap();
            for r in 0..rows {
                for c in 0..cols {
                    assert_eq!(
                        a.bits[r * cols + c],
                        ((b.bits[c * rows + r] & 0x11) << 1) | ((b.bits[c * rows + r] & 0x22) >> 1)
                    );
                    if c + 1 == cols {
                        assert_eq!(a.bits[r * cols + c] & 1, 0);
                    }
                    if r + 1 == rows {
                        assert_eq!(a.bits[r * cols + c] & 2, 0);
                    }
                }
            }
        }
    }
}
