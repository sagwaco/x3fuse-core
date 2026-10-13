//! Reconstruct Merrill highlight color without changing reconstructed brightness.
//!
//! An 8-pixel graph combines intact boundary color and reliable same-pixel layer
//! ratios. A calibrated-neutral prior supplies missing color constraints, but a
//! balanced pair does not establish white. The missing third-layer relationship
//! can also belong to a colored surface. The graph has no fixed donor radius or
//! scene classes.
//!
//! # Reading order
//!
//! - [`Field::build`] collects measurements and builds the grid once per image.
//! - [`Field::estimate`] reads that grid to choose a color for one pixel.
//! - [`Field::reconstruct`] coordinates the construction work called by `build`:
//!   connect cells, assemble their color equations, solve, and store the results.
//!
//! The entry points sit below the settings. Private construction methods follow,
//! then supporting types and math helpers. Tests are at the end.
use super::{tone_anchor, DngCtx, LocalRecovery};
use crate::{Control, Error::InvalidData};
use std::{cmp::Ordering, collections::BinaryHeap};

// Color behavior settings
// These settings balance measured color, surrounding color, and a neutral
// reference when recovering highlights. A weight controls how strongly one
// source of evidence influences the estimate relative to the others.
//
// A grid cell groups nearby image pixels. Intact, unrepaired pixels with usable
// signals can donate color. Their cell averages provide color references for
// nearby regions needing recovery.

// Reduce the influence of partly clipped layers. Each layer's reliability is
// between zero and one, and this power makes intermediate values smaller.
// Larger powers reduce their influence more strongly.
const RELIABILITY_POWER: i32 = 4;

// Very dark layer measurements gain influence gradually as their signal grows.
// Trust starts at zero at SIGNAL_TRUST_START and reaches full strength after
// a further increase of SIGNAL_TRUST_WIDTH. Raising the start or widening the
// transition reduces the influence of dark measurements. The camera's black
// and white levels set the signal scale, with lens-shading correction applied.
const SIGNAL_TRUST_START: f64 = 0.002;
const SIGNAL_TRUST_WIDTH: f64 = 0.018;

// Minimum signal in every layer of a pixel that supplies boundary color.
// Raising this value requires brighter measurements before borrowing color.
const MIN_DONOR_SIGNAL: f64 = 0.02;

// Brightness matching between neighboring grid cells and when reading the grid
// at each pixel. Larger values allow more borrowing across brightness changes.
// The value is measured in exposure stops.
const LOCAL_TONE_SCALE_EV: f64 = 0.25;

// Color matching uses ratios between sensor layers. Larger values allow more
// borrowing despite differences in those ratios. Comparisons use natural
// logarithms: a difference of 0.035 corresponds to about a 3.6% ratio change.
const PAIR_AGREEMENT_SCALE: f64 = 0.035;

// Check possible boundary colors against the brightest cell in each connected
// region needing recovery. Larger values give less similar boundaries more
// influence. Brightness is measured in stops, and color uses natural-log layer ratios.
const BOUNDARY_TONE_SCALE_EV: f64 = 0.75;
const BOUNDARY_PAIR_SCALE: f64 = 0.12;

// Ignore connections or color proposals whose agreement weight is below this
// value. Raising it discards more weak matches, including when reading the grid
// and comparing its proposed color with an individual pixel's measurements.
const MIN_AFFINITY: f64 = 1e-4;

// Give measured layer ratios more influence in both the grid calculation and
// the final adjustment at each pixel. Increasing this weight strengthens the
// measurements relative to surrounding color and the neutral reference.
const PAIR_DATA_WEIGHT: f64 = 4.0;

// The neutral reference is the camera's white-balanced gray. It helps estimate
// color where measurements leave uncertainty. The most trusted layer pair sets
// how strongly to use that reference. One fully trusted pair leaves only the
// minimum weight, even when the third layer is missing.

// Minimum pull toward neutral in the grid calculation. Keep this positive so
// a cell with no usable color measurements still has a defined color estimate.
const NEUTRAL_WEIGHT_FLOOR: f64 = 0.0001;

// Extra pull toward neutral when no layer pair is trustworthy. This contribution
// decreases as the most trusted pair becomes more reliable.
const NEUTRAL_UNCERTAINTY_WEIGHT: f64 = 0.04;

// Control how quickly that extra pull decreases as trust improves. Larger powers
// reduce the extra neutral pull for partly trusted measurements.
const NEUTRAL_UNCERTAINTY_POWER: i32 = 2;

// Minimum influence of the grid's color estimate when adjusting an individual
// pixel. This weight still applies far from pixels that supply reliable color.
const NATIVE_FIELD_WEIGHT_FLOOR: f64 = 0.25;

// Extra grid-color weight near reliable color donors. At zero distance from a
// donor, the total weight is the floor plus this boost.
const NATIVE_FIELD_WEIGHT_BOOST: f64 = 1.75;

// Control how slowly the extra weight fades with distance from donors. Distance
// counts pixel spacing along connected grid cells, with larger steps across
// cells that disagree. A larger scale carries donor influence farther.
const DONOR_DISTANCE_SCALE_PIXELS: f64 = 256.0;

// Allow this much additional disagreement with trusted measurements when moving
// an estimate toward neutral. The check uses logarithmic layer ratios and gives
// less reliable pairs more room to change. This limit follows the pair-agreement
// setting, so changing that setting also changes the allowed disagreement here.
const PAIR_ERROR_ALLOWANCE: f64 = PAIR_AGREEMENT_SCALE;

// Grid size and required measurements

// Width and height of each color-grid cell, in source pixels. Smaller cells give
// finer spatial sampling and create more cells for the solver to process.
const GRID_STEP_PIXELS: usize = 8;

// Minimum number of usable, intact pixels needed for a cell to supply boundary
// color. Keep this positive, and review it when changing cell size because the
// number of available measurements changes with the area of each cell.
const MIN_BOUNDARY_DONORS: f64 = 8.0;

// Arithmetic safety and solver accuracy
// These limits keep calculations well-defined and control how closely the
// solver satisfies its equations. Changes need numerical checks, since they
// can affect which estimates are accepted, which fallback is used, or whether
// conversion succeeds.

// Smallest layer value used when taking logarithms. Keeps zero and negative
// samples from producing invalid logarithms.
const MIN_LOG_INPUT: f64 = 1e-12;

// Minimum brightness accepted when collecting grid-cell brightness and donor
// measurements. Both calculations need a positive brightness value.
const MIN_CELL_LUMINANCE: f64 = 1e-9;

// Minimum usable response of the camera's color transform to changes in layer
// balance. If one kind of change has too little effect on rendered color,
// rescaling that response can amplify rounding errors. Reject that calibration.
const MIN_METRIC_EIGENVALUE: f64 = 1e-9;

// Reject color estimates with extremely large or small layer ratios. This bound
// applies to the absolute values of their logarithms.
const MAX_ABS_LOG_RATIO: f64 = 30.0;

// Minimum brightness of a color estimate before scaling it to unit brightness.
// That scaling divides by brightness, so a tiny value can magnify errors.
const MIN_DIRECTION_LUMINANCE: f64 = 1e-12;

// Minimum combined weight needed to average nearby grid colors. At or below
// this limit, use the neutral reference as the starting color estimate.
const MIN_INTERPOLATION_WEIGHT: f64 = 1e-12;

// Allowed error when checking the solved equations. Smaller values demand a
// closer solution and can require more iterations. The limit scales with the
// largest absolute value on the equations' right-hand side when it exceeds one.
const SOLVER_TOLERANCE: f64 = 1e-9;

// Maximum number of refinement steps before conversion reports failure.
const MAX_SOLVER_ITERATIONS: i32 = 4000;

// Number of graph items processed between checks for a cancellation request.
// Smaller values improve responsiveness at the cost of more frequent checks.
const CANCEL_CHECK_INTERVAL: usize = 65_536;

// Sensor mask and grid bookkeeping

// The reliability mask uses the largest byte value for a fully usable layer.
const FULL_RELIABILITY: u8 = u8::MAX;

// Mark a missing node or neighbor with the largest index value.
const NONE: usize = usize::MAX;

// Entry points

pub(super) struct Field {
    rows: usize,
    cols: usize,
    nodes: Vec<Node>,
    horizontal: Vec<f64>,
    vertical: Vec<f64>,
    prior: ColorPrior,
}

impl Field {
    /// Build once from repaired and denoised source pixels, before encoding.
    /// Headroom measurement and encoding read the same immutable field.
    pub unsafe fn build(
        ctx: &DngCtx<'_>,
        model: &LocalRecovery,
        data: &[u16],
        stride: usize,
        control: Control<'_>,
    ) -> crate::Result<Self> {
        let rows = (ctx.rows as usize).div_ceil(GRID_STEP_PIXELS);
        let cols = (ctx.cols as usize).div_ceil(GRID_STEP_PIXELS);
        let mut nodes = vec![Node::default(); rows * cols];
        let neutral = unsafe { *(ctx.prior as *const [f64; 3]) };
        let y = ctx
            .camera_y
            .ok_or(InvalidData("color field requires Merrill Y"))?;
        for r in 0..rows {
            control.check()?;
            for c in 0..cols {
                let n = &mut nodes[r * cols + c];
                let mut count = 0.0;
                let mut tones = 0.0;
                let mut donors = 0.0;
                let mut stressed = false;
                for row in r * GRID_STEP_PIXELS..((r + 1) * GRID_STEP_PIXELS).min(ctx.rows as usize)
                {
                    for col in
                        c * GRID_STEP_PIXELS..((c + 1) * GRID_STEP_PIXELS).min(ctx.cols as usize)
                    {
                        let measured = unsafe { ctx.measured(data, stride, row, col) };
                        let mask = model.mask(row, col);
                        let repaired = model.repaired_site(row, col);
                        n.affected |= mask != [FULL_RELIABILITY; 3];
                        let mut tone = tone_anchor::recover_severe(measured, mask, neutral, y);
                        if let Some(g) = ctx.gradient {
                            g.apply(row, col, &mut tone, measured, neutral);
                        }
                        let target = if tone.available {
                            tone.amplitude * dot3(y, neutral)
                        } else {
                            dot3(y, measured)
                        };
                        if target.is_finite() && target > MIN_CELL_LUMINANCE {
                            n.guide.tone += target.ln();
                            tones += 1.0;
                        }
                        let (pair, q) = evidence(measured, mask, repaired);
                        for k in 0..3 {
                            n.guide.pair[k] += q[k] * pair[k];
                            n.guide.q[k] += q[k];
                        }
                        count += 1.0;
                        if !repaired {
                            stressed |= mask != [FULL_RELIABILITY; 3];
                            if mask == [FULL_RELIABILITY; 3]
                                && measured
                                    .iter()
                                    .all(|v| v.is_finite() && *v > MIN_DONOR_SIGNAL)
                                && dot3(y, measured) > MIN_CELL_LUMINANCE
                            {
                                n.color[0] += pair[1];
                                n.color[1] += pair[2];
                                donors += 1.0;
                            }
                        }
                    }
                }
                n.guide.tone = if tones > 0.0 {
                    n.guide.tone / tones
                } else {
                    f64::NAN
                };
                for k in 0..3 {
                    if n.guide.q[k] > 0.0 {
                        n.guide.pair[k] /= n.guide.q[k];
                    }
                    n.guide.q[k] /= count;
                }
                n.fixed = !stressed && donors >= MIN_BOUNDARY_DONORS && n.guide.tone.is_finite();
                if n.fixed {
                    n.affected = false;
                }
                if donors > 0.0 {
                    n.color = n.color.map(|v| v / donors);
                }
            }
        }
        let size = nodes.len();
        let mut result = Self {
            rows,
            cols,
            nodes,
            horizontal: vec![0.0; size],
            vertical: vec![0.0; size],
            prior: ColorPrior::new(neutral, unsafe { *(ctx.conv_matrix as *const [f64; 9]) })?,
        };
        result.reconstruct(control)?;
        Ok(result)
    }

    /// Estimate color without changing brightness. Repaired pixels and single
    /// surviving layers cannot supply independent ratios. Normalize the returned
    /// direction to unit luminance using the camera calibration.
    pub fn estimate(
        &self,
        row: usize,
        col: usize,
        measured: [f64; 3],
        mask: [u8; 3],
        repaired: bool,
        target_y: f64,
        y: [f64; 3],
    ) -> Option<[f64; 3]> {
        if !target_y.is_finite() || target_y <= 0.0 {
            return None;
        }
        let (pair, q) = evidence(measured, mask, repaired);
        let guide = Guide {
            pair,
            q,
            tone: target_y.ln(),
        };
        // Half-pixel offsets align source-pixel centers with grid-cell centers.
        let rr = (row as f64 + 0.5) / GRID_STEP_PIXELS as f64 - 0.5;
        let cc = (col as f64 + 0.5) / GRID_STEP_PIXELS as f64 - 0.5;
        let mut sum = [0.0; 2];
        let mut total = 0.0;
        let mut distance = 0.0;
        for r in rr.floor() as isize..=rr.floor() as isize + 1 {
            for c in cc.floor() as isize..=cc.floor() as isize + 1 {
                if r < 0 || c < 0 || r as usize >= self.rows || c as usize >= self.cols {
                    continue;
                }
                let n = &self.nodes[r as usize * self.cols + c as usize];
                if !n.distance.is_finite() && !n.affected {
                    continue;
                }
                let w = (1.0 - (rr - r as f64).abs())
                    * (1.0 - (cc - c as f64).abs())
                    * affinity(guide, n.guide);
                if w <= 0.0 {
                    continue;
                }
                total += w;
                for (ch, value) in sum.iter_mut().enumerate() {
                    *value += w * n.color[ch];
                }
                distance += w * n.distance;
            }
        }
        let p = self.prior;
        let (prior, distance) = if total > MIN_INTERPOLATION_WEIGHT {
            (sum.map(|v| v / total), distance / total)
        } else {
            (p.neutral, f64::INFINITY)
        };
        let proposed_prior = p.compatible_target(prior, guide);
        let weight = NATIVE_FIELD_WEIGHT_FLOOR
            + NATIVE_FIELD_WEIGHT_BOOST * (-distance / DONOR_DISTANCE_SCALE_PIXELS).exp();
        let proposed = p.fit(guide, proposed_prior, weight);
        let before = p.fit(guide, prior, weight);
        let retained = retained_neutral_move(before, proposed, guide);
        // The fit is affine in its prior, so limit both by the same fraction.
        // Preserve the fallback direction and skip interpolation for an admissible proposal.
        let (fit, prior) = if retained == 1.0 {
            (proposed, proposed_prior)
        } else {
            (
                std::array::from_fn(|k| before[k] + retained * (proposed[k] - before[k])),
                std::array::from_fn(|k| prior[k] + retained * (proposed_prior[k] - prior[k])),
            )
        };
        direction(fit, y)
            .or_else(|| direction(prior, y))
            .or_else(|| direction(p.neutral, y))
    }
}

// Field construction

impl Field {
    fn reconstruct(&mut self, control: Control<'_>) -> crate::Result<()> {
        self.connect(control)?;
        let mut index = vec![NONE; self.nodes.len()];
        let mut active = Vec::new();
        for (i, n) in self.nodes.iter().enumerate() {
            if !n.fixed && n.affected {
                index[i] = active.len();
                active.push(i);
            }
        }
        let mut equations = Vec::with_capacity(active.len());
        let mut rhs = Vec::with_capacity(active.len());
        let mut x = Vec::with_capacity(active.len());
        for &i in &active {
            let n = &self.nodes[i];
            let (mut block, mut target) = data_equation(n.guide);
            self.prior.add(
                &mut block,
                &mut target,
                self.prior.neutral,
                neutral_weight(n.guide),
            );
            let mut neighbors = [NONE; 4];
            let mut weights = [0.0; 4];
            for (k, (j, w)) in self.neighbors(i).into_iter().enumerate() {
                if j == NONE || w == 0.0 {
                    continue;
                }
                // Measure spatial differences in the same calibrated geometry
                // as the neutral prior, including fixed-boundary contributions.
                self.prior.add(
                    &mut block,
                    &mut target,
                    if self.nodes[j].fixed {
                        self.nodes[j].color
                    } else {
                        [0.0; 2]
                    },
                    w,
                );
                if !self.nodes[j].fixed {
                    assert_ne!(index[j], NONE);
                    neighbors[k] = index[j];
                    weights[k] = w;
                }
            }
            equations.push(Equation {
                block,
                neighbors,
                weights,
            });
            rhs.push(target);
            x.push(if n.origin != NONE {
                self.nodes[n.origin].color
            } else {
                self.prior.neutral
            });
        }
        solve(&equations, self.prior.metric, &rhs, &mut x, control)?;
        for (k, &i) in active.iter().enumerate() {
            self.nodes[i].color = x[k];
        }
        Ok(())
    }

    fn connect(&mut self, control: Control<'_>) -> crate::Result<()> {
        for i in 0..self.nodes.len() {
            if i % CANCEL_CHECK_INTERVAL == 0 {
                control.check()?;
            }
            if i % self.cols + 1 < self.cols {
                self.horizontal[i] = affinity(self.nodes[i].guide, self.nodes[i + 1].guide);
            }
            if i / self.cols + 1 < self.rows {
                self.vertical[i] = affinity(self.nodes[i].guide, self.nodes[i + self.cols].guide);
            }
        }
        self.condition_boundaries(control)?;
        let mut queue = BinaryHeap::new();
        for (i, n) in self.nodes.iter().enumerate() {
            if n.fixed {
                queue.push(Visit {
                    distance: 0.0,
                    index: i,
                    origin: i,
                });
            }
        }
        let mut count = 0;
        while let Some(v) = queue.pop() {
            count += 1;
            if count % CANCEL_CHECK_INTERVAL as i32 == 0 {
                control.check()?;
            }
            if self.nodes[v.index].distance <= v.distance {
                continue;
            }
            self.nodes[v.index].distance = v.distance;
            self.nodes[v.index].origin = v.origin;
            for (j, w) in self.neighbors(v.index) {
                if j != NONE && w > 0.0 {
                    let distance = v.distance + GRID_STEP_PIXELS as f64 / w.sqrt();
                    if distance < self.nodes[j].distance {
                        queue.push(Visit {
                            distance,
                            index: j,
                            origin: v.origin,
                        });
                    }
                }
            }
        }
        Ok(())
    }

    fn condition_boundaries(&mut self, control: Control<'_>) -> crate::Result<()> {
        let mut seen = vec![false; self.nodes.len()];
        for i in 0..self.nodes.len() {
            if seen[i] || !self.nodes[i].affected {
                continue;
            }
            control.check()?;
            let mut component = vec![i];
            seen[i] = true;
            let mut peak = i;
            let mut k = 0;
            while k < component.len() {
                if k % CANCEL_CHECK_INTERVAL == 0 {
                    control.check()?;
                }
                let n = component[k];
                k += 1;
                if self.nodes[n].guide.tone > self.nodes[peak].guide.tone {
                    peak = n;
                }
                for (j, w) in self.neighbors(n) {
                    if j != NONE && w > 0.0 && self.nodes[j].affected && !seen[j] {
                        seen[j] = true;
                        component.push(j);
                    }
                }
            }
            for n in component {
                self.nodes[n].region = peak;
            }
        }
        for i in 0..self.nodes.len() {
            if i % CANCEL_CHECK_INTERVAL == 0 {
                control.check()?;
            }
            for horizontal in [true, false] {
                let j = if horizontal {
                    if i % self.cols + 1 >= self.cols {
                        continue;
                    }
                    i + 1
                } else {
                    if i / self.cols + 1 >= self.rows {
                        continue;
                    }
                    i + self.cols
                };
                let a = &self.nodes[i];
                let b = &self.nodes[j];
                let gate = if (!a.fixed && !a.affected) || (!b.fixed && !b.affected) {
                    0.0
                } else if a.affected && b.fixed {
                    surface_agreement(self.nodes[a.region].guide, b.guide)
                } else if b.affected && a.fixed {
                    surface_agreement(self.nodes[b.region].guide, a.guide)
                } else {
                    1.0
                };
                if horizontal {
                    self.horizontal[i] *= gate;
                } else {
                    self.vertical[i] *= gate;
                }
            }
        }
        Ok(())
    }

    fn neighbors(&self, i: usize) -> [(usize, f64); 4] {
        let r = i / self.cols;
        let c = i % self.cols;
        [
            if r > 0 {
                (i - self.cols, self.vertical[i - self.cols])
            } else {
                (NONE, 0.0)
            },
            if c > 0 {
                (i - 1, self.horizontal[i - 1])
            } else {
                (NONE, 0.0)
            },
            if r + 1 < self.rows {
                (i + self.cols, self.vertical[i])
            } else {
                (NONE, 0.0)
            },
            if c + 1 < self.cols {
                (i + 1, self.horizontal[i])
            } else {
                (NONE, 0.0)
            },
        ]
    }
}

// Supporting types and math helpers

fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a.into_iter().zip(b).map(|(x, y)| x * y).sum()
}
fn smooth(x: f64) -> f64 {
    // The fixed cubic coefficients give zero slope at both endpoints.
    let x = x.clamp(0.0, 1.0);
    x * x * (3.0 - 2.0 * x)
}
fn evidence(m: [f64; 3], mask: [u8; 3], repaired: bool) -> ([f64; 3], [f64; 3]) {
    let q: [f64; 3] = std::array::from_fn(|c| {
        if repaired || !m[c].is_finite() {
            0.0
        } else {
            (mask[c] as f64 / FULL_RELIABILITY as f64).powi(RELIABILITY_POWER)
                * smooth((m[c] - SIGNAL_TRUST_START) / SIGNAL_TRUST_WIDTH)
        }
    });
    let l = m.map(|v| v.max(MIN_LOG_INPUT).ln());
    (
        [l[0] - l[1], l[0] - l[2], l[1] - l[2]],
        [q[0] * q[1], q[0] * q[2], q[1] * q[2]],
    )
}
#[derive(Clone, Copy, Default)]
struct Guide {
    pair: [f64; 3],
    q: [f64; 3],
    tone: f64,
}
fn agreement(a: Guide, b: Guide, tone_scale: f64, pair_scale: f64) -> f64 {
    if !a.tone.is_finite() || !b.tone.is_finite() {
        return 0.0;
    }
    let t = (a.tone - b.tone) / (tone_scale * std::f64::consts::LN_2);
    let disagreement = (0..3)
        .map(|k| a.q[k].min(b.q[k]) * ((a.pair[k] - b.pair[k]) / pair_scale).powi(2))
        .fold(0.0_f64, f64::max);
    // Squared distances and the -1/2 factor define the Gaussian form.
    (-0.5 * (t * t + disagreement)).exp()
}
fn affinity(a: Guide, b: Guide) -> f64 {
    let w = agreement(a, b, LOCAL_TONE_SCALE_EV, PAIR_AGREEMENT_SCALE);
    if w < MIN_AFFINITY {
        0.0
    } else {
        w
    }
}
/// Limit the neutralward adjustment without changing the field or evidence weights.
/// Allow one pair-agreement scale of additional weighted ratio error.
/// This heuristic does not define a calibrated noise interval.
fn retained_neutral_move(before: [f64; 2], proposed: [f64; 2], g: Guide) -> f64 {
    let ratios = |v: [f64; 2]| [v[0] - v[1], v[0], v[1]];
    let a = ratios(before);
    let b = ratios(proposed);
    let mut retained: f64 = 1.0;
    for k in 0..3 {
        if g.q[k] <= 0.0 {
            continue;
        }
        let trust = g.q[k].sqrt();
        let old_error = trust * (a[k] - g.pair[k]);
        let new_error = trust * (b[k] - g.pair[k]);
        let bound = old_error.abs() + PAIR_ERROR_ALLOWANCE;
        if new_error.abs() > bound {
            let edge = bound.copysign(new_error);
            retained = retained.min((edge - old_error) / (new_error - old_error));
        }
    }
    retained.clamp(0.0, 1.0)
}
fn neutral_weight(g: Guide) -> f64 {
    let strongest_pair_trust = g.q.into_iter().fold(0.0_f64, f64::max).clamp(0.0, 1.0);
    let uncertainty = 1.0 - strongest_pair_trust;
    NEUTRAL_WEIGHT_FLOOR + NEUTRAL_UNCERTAINTY_WEIGHT * uncertainty.powi(NEUTRAL_UNCERTAINTY_POWER)
}
#[derive(Clone, Copy)]
struct ColorPrior {
    neutral: [f64; 2],
    metric: [f64; 3],
}
impl ColorPrior {
    /// Coarse averages can hide disagreement with the target pixel's measurements.
    /// Reduce the proposal toward calibrated neutral when trusted ratios disagree.
    /// This constraint does not classify the pixel as white.
    fn compatible_target(self, field: [f64; 2], guide: Guide) -> [f64; 2] {
        let proposal = Guide {
            pair: [field[0] - field[1], field[0], field[1]],
            q: [1.0; 3],
            tone: guide.tone,
        };
        let confidence = affinity(guide, proposal);
        std::array::from_fn(|k| self.neutral[k] + confidence * (field[k] - self.neutral[k]))
    }
    fn new(p: [f64; 3], m: [f64; 9]) -> crate::Result<Self> {
        let rgb: [f64; 3] = std::array::from_fn(|r| (0..3).map(|c| m[3 * r + c] * p[c]).sum());
        if p.iter()
            .chain(rgb.iter())
            .any(|v| !v.is_finite() || *v <= 0.0)
        {
            return Err(InvalidData("invalid neutral color metric"));
        }
        // Use the Jacobian of log(R/G) and log(B/G) at calibrated neutral.
        // Equal sensor-layer values do not imply neutral rendered color.
        let j: [[f64; 2]; 2] = [0, 2]
            .map(|r| std::array::from_fn(|c| p[c] * (m[3 * r + c] / rgb[r] - m[3 + c] / rgb[1])));
        let a = j[0][0] * j[0][0] + j[1][0] * j[1][0];
        let b = j[0][1] * j[0][1] + j[1][1] * j[1][1];
        let c = j[0][0] * j[0][1] + j[1][0] * j[1][1];
        // Fixed coefficients of the smaller eigenvalue of a symmetric 2x2 matrix.
        let minimum = 0.5 * (a + b - ((a - b) * (a - b) + 4.0 * c * c).sqrt());
        if !minimum.is_finite() || minimum <= MIN_METRIC_EIGENVALUE {
            return Err(InvalidData("singular neutral color metric"));
        }
        Ok(Self {
            neutral: [(p[0] / p[2]).ln(), (p[1] / p[2]).ln()],
            metric: [a / minimum, b / minimum, c / minimum],
        })
    }
    fn fit(self, guide: Guide, target: [f64; 2], weight: f64) -> [f64; 2] {
        let (mut block, mut rhs) = data_equation(guide);
        self.add(&mut block, &mut rhs, target, weight);
        inverse(block, rhs)
    }
    fn add(self, block: &mut [f64; 3], rhs: &mut [f64; 2], target: [f64; 2], weight: f64) {
        for k in 0..3 {
            block[k] += weight * self.metric[k];
        }
        rhs[0] += weight * (self.metric[0] * target[0] + self.metric[2] * target[1]);
        rhs[1] += weight * (self.metric[2] * target[0] + self.metric[1] * target[1]);
    }
}
fn surface_agreement(a: Guide, b: Guide) -> f64 {
    agreement(a, b, BOUNDARY_TONE_SCALE_EV, BOUNDARY_PAIR_SCALE)
}
#[derive(Clone)]
struct Node {
    guide: Guide,
    fixed: bool,
    color: [f64; 2],
    distance: f64,
    origin: usize,
    affected: bool,
    region: usize,
}
impl Default for Node {
    fn default() -> Self {
        Self {
            guide: Guide::default(),
            fixed: false,
            color: [0.0; 2],
            distance: f64::INFINITY,
            origin: NONE,
            affected: false,
            region: NONE,
        }
    }
}
#[derive(Clone, Copy)]
struct Visit {
    distance: f64,
    index: usize,
    origin: usize,
}
impl PartialEq for Visit {
    fn eq(&self, b: &Self) -> bool {
        self.distance == b.distance && self.index == b.index && self.origin == b.origin
    }
}
impl Eq for Visit {}
impl PartialOrd for Visit {
    fn partial_cmp(&self, b: &Self) -> Option<Ordering> {
        Some(self.cmp(b))
    }
}
impl Ord for Visit {
    fn cmp(&self, b: &Self) -> Ordering {
        b.distance
            .total_cmp(&self.distance)
            .then_with(|| b.index.cmp(&self.index))
            .then_with(|| b.origin.cmp(&self.origin))
    }
}
struct Equation {
    block: [f64; 3],
    neighbors: [usize; 4],
    weights: [f64; 4],
}
fn inverse(a: [f64; 3], x: [f64; 2]) -> [f64; 2] {
    let det = a[0] * a[1] - a[2] * a[2];
    [
        (a[1] * x[0] - a[2] * x[1]) / det,
        (a[0] * x[1] - a[2] * x[0]) / det,
    ]
}
// Use the neutral prior's metric for spatial edges so that both constraints
// measure color differences in the same geometry.
fn product(rows: &[Equation], metric: [f64; 3], x: &[[f64; 2]], out: &mut [[f64; 2]]) {
    for (i, e) in rows.iter().enumerate() {
        let [a, b, c] = e.block;
        out[i] = [a * x[i][0] + c * x[i][1], c * x[i][0] + b * x[i][1]];
        for k in 0..4 {
            if e.neighbors[k] != NONE {
                let z = x[e.neighbors[k]];
                let [a, b, c] = metric;
                out[i][0] -= e.weights[k] * (a * z[0] + c * z[1]);
                out[i][1] -= e.weights[k] * (c * z[0] + b * z[1]);
            }
        }
    }
}
fn norm(x: &[[f64; 2]]) -> f64 {
    x.iter()
        .flatten()
        .map(|v| {
            if v.is_finite() {
                v.abs()
            } else {
                f64::INFINITY
            }
        })
        .fold(0.0_f64, f64::max)
}
fn dot(a: &[[f64; 2]], b: &[[f64; 2]]) -> f64 {
    a.iter()
        .flatten()
        .zip(b.iter().flatten())
        .map(|(a, b)| a * b)
        .sum()
}
fn solve(
    rows: &[Equation],
    metric: [f64; 3],
    rhs: &[[f64; 2]],
    x: &mut [[f64; 2]],
    control: Control<'_>,
) -> crate::Result<()> {
    if rows.is_empty() {
        return Ok(());
    }
    if rows.iter().any(|e| {
        e.block.iter().any(|v| !v.is_finite())
            || e.block[0] * e.block[1] - e.block[2] * e.block[2] <= 0.0
    }) {
        return Err(InvalidData("color field preconditioner not positive"));
    }
    let tolerance = SOLVER_TOLERANCE * norm(rhs).max(1.0);
    if !tolerance.is_finite() {
        return Err(InvalidData("color field RHS not finite"));
    }
    let mut ax = vec![[0.0; 2]; x.len()];
    product(rows, metric, x, &mut ax);
    let mut residual: Vec<[f64; 2]> = rhs
        .iter()
        .zip(&ax)
        .map(|(a, b)| [a[0] - b[0], a[1] - b[1]])
        .collect();
    let mut z: Vec<_> = rows
        .iter()
        .zip(&residual)
        .map(|(e, r)| inverse(e.block, *r))
        .collect();
    let mut direction = z.clone();
    let mut rz = dot(&residual, &z);
    for iteration in 0..=MAX_SOLVER_ITERATIONS {
        control.check()?;
        if norm(&residual) <= tolerance {
            product(rows, metric, x, &mut ax);
            let actual = rhs
                .iter()
                .flatten()
                .zip(ax.iter().flatten())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max);
            if !norm(&ax).is_finite()
                || !actual.is_finite()
                || actual > tolerance
                || !norm(x).is_finite()
            {
                return Err(InvalidData("color field true residual failed"));
            }
            return Ok(());
        }
        if iteration == MAX_SOLVER_ITERATIONS {
            return Err(InvalidData("color field did not converge"));
        }
        product(rows, metric, &direction, &mut ax);
        let denominator = dot(&direction, &ax);
        if denominator <= 0.0 || !denominator.is_finite() || !rz.is_finite() {
            return Err(InvalidData("color field solver breakdown"));
        }
        let alpha = rz / denominator;
        for i in 0..x.len() {
            for ch in 0..2 {
                x[i][ch] += alpha * direction[i][ch];
                residual[i][ch] -= alpha * ax[i][ch];
            }
            z[i] = inverse(rows[i].block, residual[i]);
        }
        let next = dot(&residual, &z);
        let beta = next / rz;
        for i in 0..x.len() {
            for ch in 0..2 {
                direction[i][ch] = z[i][ch] + beta * direction[i][ch];
            }
        }
        rz = next;
    }
    unreachable!()
}
fn data_equation(g: Guide) -> ([f64; 3], [f64; 2]) {
    let [a, b, c] = g.q.map(|v| v * PAIR_DATA_WEIGHT);
    (
        [a + b, a + c, -a],
        [
            a * g.pair[0] + b * g.pair[1],
            -a * g.pair[0] + c * g.pair[2],
        ],
    )
}
fn direction(uv: [f64; 2], y: [f64; 3]) -> Option<[f64; 3]> {
    if uv
        .iter()
        .any(|v| !v.is_finite() || v.abs() > MAX_ABS_LOG_RATIO)
    {
        return None;
    }
    let d = [uv[0].exp(), uv[1].exp(), 1.0];
    let lum = dot3(y, d);
    (lum.is_finite() && lum > MIN_DIRECTION_LUMINANCE).then(|| d.map(|v| v / lum))
}

#[cfg(test)]
mod tests {
    use super::*;
    const P: [f64; 3] = [0.284206, 0.604436, 1.0];
    const Y: [f64; 3] = [-2.6575486148565814, 3.313530567081604, -0.2476];
    const M: [f64; 9] = [
        4.45774, -1.90728, 0.581922, -4.22818, 4.30012, -0.701463, 3.1376, -5.81127, 3.31683,
    ];
    fn line(length: usize) -> Field {
        let prior = ColorPrior::new(P, M).unwrap();
        let g = Guide {
            tone: 0.0,
            ..Guide::default()
        };
        let mut f = Field {
            rows: 1,
            cols: length,
            nodes: vec![
                Node {
                    guide: g,
                    affected: true,
                    ..Node::default()
                };
                length
            ],
            horizontal: vec![0.0; length],
            vertical: vec![0.0; length],
            prior,
        };
        f.nodes[0].fixed = true;
        f.nodes[0].affected = false;
        f.nodes[0].color = prior.neutral;
        f
    }
    #[test]
    fn compatible_non_neutral_color_is_not_rejected_by_a_balanced_pair() {
        let p = ColorPrior::new(P, M).unwrap();
        let field = [p.neutral[0] + 0.3, p.neutral[1] + 0.3];
        let g = Guide {
            pair: [field[0] - field[1], 0.0, 0.0],
            q: [1.0, 0.0, 0.0],
            tone: 0.0,
        };
        assert_eq!(p.compatible_target(field, g), field);
        assert_ne!(field, p.neutral);
    }
    #[test]
    fn only_trusted_conflicting_pairs_can_decline_a_borrowed_color() {
        let p = ColorPrior::new(P, M).unwrap();
        let field = [p.neutral[0] + 0.35, p.neutral[1]];
        let mut g = Guide {
            pair: [p.neutral[0] - p.neutral[1], 0.0, 0.0],
            q: [1.0, 0.0, 0.0],
            tone: 0.0,
        };
        assert_eq!(p.compatible_target(field, g), p.neutral);
        g.q = [0.0; 3];
        assert_eq!(p.compatible_target(field, g), field);
        g.q = [0.001, 0.0, 0.0];
        let weak = p.compatible_target(field, g);
        assert!(weak[0] > p.neutral[0] && weak[0] < field[0]);
    }
    #[test]
    fn neutral_move_retains_compatible_or_unsupported_changes() {
        let mut g = Guide {
            pair: [0.0; 3],
            q: [1.0, 0.0, 0.0],
            tone: 0.0,
        };
        assert_eq!(retained_neutral_move([0.04, 0.0], [0.041, 0.0], g), 1.0);
        assert_eq!(retained_neutral_move([0.04, 0.0], [0.01, 0.0], g), 1.0);
        // The measured B/M ratio does not constrain movement along this direction.
        assert_eq!(retained_neutral_move([0.04, 0.0], [1.04, 1.0], g), 1.0);
        g.q = [0.0; 3];
        assert_eq!(retained_neutral_move([0.04, 0.0], [-0.8, 0.5], g), 1.0);
    }
    #[test]
    fn neutral_move_limits_crossing_past_a_trustworthy_ratio() {
        let g = Guide {
            pair: [0.0; 3],
            q: [1.0, 0.0, 0.0],
            tone: 0.0,
        };
        let retained = retained_neutral_move([0.04, 0.0], [-0.17, 0.0], g);
        assert!((retained - (0.04 + 0.075) / 0.21).abs() < 1e-14);
        assert!((0.04 + retained * (-0.17 - 0.04) + 0.075).abs() < 1e-14);
        let weak = Guide {
            q: [0.01, 0.0, 0.0],
            ..g
        };
        assert_eq!(retained_neutral_move([0.04, 0.0], [-0.17, 0.0], weak), 1.0);
    }
    #[test]
    fn neutral_move_bound_holds_for_each_pair_and_is_maximal() {
        let ratios = |v: [f64; 2]| [v[0] - v[1], v[0], v[1]];
        for a in -4..=4 {
            for b in -4..=4 {
                for c in -4..=4 {
                    for q in [
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                        [1.0, 0.3, 0.1],
                    ] {
                        let before = [a as f64 * 0.09, b as f64 * 0.08];
                        let proposed = [c as f64 * 0.12, -a as f64 * 0.07];
                        let g = Guide {
                            pair: [0.03, 0.1, 0.07],
                            q,
                            tone: 0.0,
                        };
                        let t = retained_neutral_move(before, proposed, g);
                        assert!((0.0..=1.0).contains(&t));
                        let allowed = |t: f64, tolerance: f64| {
                            let fit =
                                std::array::from_fn(|k| before[k] + t * (proposed[k] - before[k]));
                            (0..3).all(|k| {
                                q[k].sqrt() * (ratios(fit)[k] - g.pair[k]).abs()
                                    <= q[k].sqrt() * (ratios(before)[k] - g.pair[k]).abs()
                                        + PAIR_ERROR_ALLOWANCE
                                        + tolerance
                            })
                        };
                        assert!(allowed(t, 1e-12));
                        if t < 1.0 - 1e-7 {
                            assert!(!allowed(t + 1e-7, 0.0));
                        }
                    }
                }
            }
        }
    }
    #[test]
    fn balanced_pair_is_ambiguous_not_extra_evidence_for_white() {
        let p = ColorPrior::new(P, M).unwrap();
        let mut g = Guide {
            pair: [p.neutral[0] - p.neutral[1], p.neutral[0], p.neutral[1]],
            q: [1.0, 0.0, 0.0],
            tone: 0.0,
        };
        assert_eq!(neutral_weight(g), 0.0001);
        g.pair[0] += 0.2;
        assert_eq!(neutral_weight(g), 0.0001);
        g.q = [0.0; 3];
        assert!((neutral_weight(g) - 0.0401).abs() < 1e-15);
    }
    #[test]
    fn one_survivor_repaired_and_nonfinite_sites_cannot_supply_ratios() {
        for (m, mask, repaired) in [
            ([0.3, 0.6, 1.0], [255, 0, 0], false),
            ([0.3, 0.6, 1.0], [255, 255, 0], true),
            ([f64::NAN, 0.6, 1.0], [255, 255, 0], false),
        ] {
            assert_eq!(evidence(m, mask, repaired).1, [0.0; 3]);
        }
    }
    #[test]
    fn donorless_components_are_neutral_without_fabricating_hue() {
        let mut f = line(20);
        f.nodes[0].fixed = false;
        f.nodes[0].affected = true;
        f.reconstruct(Control::none()).unwrap();
        for n in &f.nodes {
            for ch in 0..2 {
                assert!((n.color[ch] - f.prior.neutral[ch]).abs() < 1e-10);
            }
        }
    }
    #[test]
    fn credible_pair_constrains_color_instead_of_forcing_neutral() {
        let mut f = line(20);
        f.nodes[0].fixed = false;
        f.nodes[0].affected = true;
        let p = f.prior.neutral;
        for n in &mut f.nodes {
            n.guide.q = [1.0, 0.0, 0.0];
            n.guide.pair[0] = p[0] - p[1] - 0.2;
        }
        f.reconstruct(Control::none()).unwrap();
        let c = f.nodes[10].color;
        assert!((c[0] - c[1] - (p[0] - p[1] - 0.2)).abs() < 0.001);
        assert!(c[0] < p[0] && c[1] < p[1]);
        assert!((dot3(Y, direction(c, Y).unwrap()) - 1.0).abs() < 1e-12);
    }
    #[test]
    fn uncertain_color_relaxes_smoothly_and_search_has_no_radius_cutoff() {
        let mut f = line(200);
        let p = f.prior;
        let color = [p.neutral[0] - 0.3, p.neutral[1] - 0.12];
        f.nodes[0].color = color;
        f.reconstruct(Control::none()).unwrap();
        let mut previous = f64::INFINITY;
        for n in &f.nodes {
            let a = n.color[0] - p.neutral[0];
            let b = n.color[1] - p.neutral[1];
            let energy = p.metric[0] * a * a + p.metric[1] * b * b + 2.0 * p.metric[2] * a * b;
            assert!(energy < previous + 1e-7);
            previous = energy;
        }
        assert_eq!(f.nodes[0].color, color);
        assert!(previous < 1e-10);
        assert!(f.nodes[199].distance > 1000.0 && f.nodes[199].distance.is_finite());
    }
    #[test]
    fn reference_gate_rejects_a_gentle_walk_to_a_dark_surface() {
        let a = Guide {
            tone: 0.0,
            ..Guide::default()
        };
        let b = Guide {
            tone: -3.0 * std::f64::consts::LN_2,
            ..a
        };
        assert!(surface_agreement(a, b) < 0.001);
        assert!(
            affinity(
                a,
                Guide {
                    tone: -0.2 * std::f64::consts::LN_2,
                    ..a
                }
            ) > 0.7
        );
    }
    #[test]
    fn unaffected_low_signal_cells_do_not_bridge_unrelated_donors() {
        let mut f = line(20);
        f.nodes[0].color[0] -= 0.3;
        for n in &mut f.nodes[1..10] {
            n.affected = false;
        }
        f.reconstruct(Control::none()).unwrap();
        assert!(!f.nodes[19].distance.is_finite());
        for ch in 0..2 {
            assert!((f.nodes[19].color[ch] - f.prior.neutral[ch]).abs() < 1e-10);
        }
    }
    #[test]
    fn native_repaired_targets_do_not_inject_color_and_keep_y() {
        let mut f = line(20);
        f.reconstruct(Control::none()).unwrap();
        let expected = direction(f.prior.neutral, Y).unwrap();
        let a = f
            .estimate(0, 80, [0.9, 0.3, 1.0], [255, 255, 0], true, 1.0, Y)
            .unwrap();
        for ch in 0..3 {
            assert!((a[ch] - expected[ch]).abs() < 1e-10);
        }
        assert!((dot3(Y, a) - 1.0).abs() < 1e-12);
        assert!(f
            .estimate(0, 80, [1.0; 3], [0; 3], false, f64::NAN, Y)
            .is_none());
    }
    #[test]
    fn invalid_metric_and_nonfinite_solve_fail() {
        assert!(ColorPrior::new(P, [0.0; 9]).is_err());
        let rows = [Equation {
            block: [f64::NAN, 1.0, 0.0],
            neighbors: [NONE; 4],
            weights: [0.0; 4],
        }];
        assert!(solve(
            &rows,
            [1.0, 1.0, 0.0],
            &[[1.0; 2]],
            &mut [[0.0; 2]],
            Control::none()
        )
        .is_err());
    }
    #[test]
    fn metric_operator_matches_dense_symmetric_positive_system() {
        let metric = [2.0, 4.0, 0.6];
        let rows = [
            Equation {
                block: [3.0, 5.0, 0.6],
                neighbors: [1, NONE, NONE, NONE],
                weights: [1.0, 0.0, 0.0, 0.0],
            },
            Equation {
                block: [3.0, 5.0, 0.6],
                neighbors: [0, NONE, NONE, NONE],
                weights: [1.0, 0.0, 0.0, 0.0],
            },
        ];
        let dense = [
            [3.0, 0.6, -2.0, -0.6],
            [0.6, 5.0, -0.6, -4.0],
            [-2.0, -0.6, 3.0, 0.6],
            [-0.6, -4.0, 0.6, 5.0],
        ];
        let truth = [[0.3, -0.5], [0.9, 0.1]];
        let flat = [0.3, -0.5, 0.9, 0.1];
        let expected: [f64; 4] =
            std::array::from_fn(|i| (0..4).map(|j| dense[i][j] * flat[j]).sum());
        let mut rhs = [[0.0; 2]; 2];
        product(&rows, metric, &truth, &mut rhs);
        for (actual, expected) in rhs.iter().flatten().zip(expected) {
            assert!((actual - expected).abs() < 1e-14);
        }
        assert!(dot(&truth, &rhs) > 0.0);
        let other = [[-0.1, 0.7], [0.4, -0.8]];
        let mut mapped = [[0.0; 2]; 2];
        product(&rows, metric, &other, &mut mapped);
        assert!((dot(&truth, &mapped) - dot(&other, &rhs)).abs() < 1e-14);
        let mut result = [[0.0; 2]; 2];
        solve(&rows, metric, &rhs, &mut result, Control::none()).unwrap();
        for (a, b) in result.iter().flatten().zip(truth.iter().flatten()) {
            assert!((a - b).abs() < 1e-9);
        }
    }
    #[test]
    fn neutral_boundary_and_ratio_use_consistent_missing_layer_direction() {
        let mut f = line(25);
        let p = f.prior;
        let delta = -0.29;
        for n in &mut f.nodes[1..] {
            n.guide.q = [1.0, 0.0, 0.0];
            n.guide.pair[0] = p.neutral[0] - p.neutral[1] + delta;
        }
        f.reconstruct(Control::none()).unwrap();
        let tangent = inverse(p.metric, [1.0, -1.0]);
        for n in &f.nodes[1..] {
            let z = [n.color[0] - p.neutral[0], n.color[1] - p.neutral[1]];
            // Both anchors and regularization are neutral. The response must lie
            // along C^-1*[1,-1], even next to the intact neutral boundary.
            assert!((z[0] * tangent[1] - z[1] * tangent[0]).abs() < 1e-7);
            assert!(z[0] < 0.0 && z[1] < 0.0);
        }
        assert_eq!(f.nodes[0].color, p.neutral);
    }
    #[test]
    fn cancellation_is_reported_not_silently_replaced() {
        let cancel = std::sync::atomic::AtomicBool::new(true);
        let mut f = line(200);
        assert!(f.reconstruct(Control::new(&cancel)).is_err());
        let rows = [Equation {
            block: [1.0, 1.0, 0.0],
            neighbors: [NONE; 4],
            weights: [0.0; 4],
        }];
        assert!(solve(
            &rows,
            [1.0, 1.0, 0.0],
            &[[1.0; 2]],
            &mut [[0.0; 2]],
            Control::new(&cancel)
        )
        .is_err());
    }
}
