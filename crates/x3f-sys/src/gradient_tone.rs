//! Reconstruct Merrill brightness from differences between neighboring pixels
//! within each sensor layer.
//! Keep each pixel's tone estimate as a weak constraint. Build the field at native
//! resolution before measuring headroom or encoding pixels, so both passes read
//! the same data. Encoding must not read neighbors it may have overwritten.
//! Take logarithms only of positive targets and positive layer measurements.
use super::{tone_anchor, DngCtx, LocalRecovery};
use crate::{Control, Error::InvalidData};

const ANCHOR: f64 = 1.0 / 64.0;
const NONE: u32 = u32::MAX;

pub(super) struct Field {
    cols: usize,
    index: Vec<u32>,
    amplitudes: Vec<f64>,
}

#[derive(Clone, Copy)]
struct Source {
    measured: [f64; 3],
    mask: [u8; 3],
    target: f64,
    amplitude: f64,
    eligible: bool,
}

struct Row {
    diag: f64,
    neighbors: [u32; 4],
    weights: [f64; 4],
}

// Return the weighted layer differences and their total weight.
// Keep small reliability weights small. Zero weight adds no neighbor constraint.
fn edge(a: Source, b: Source) -> (f64, f64) {
    let mut weight = 0.0;
    let mut gradient = 0.0;
    for c in 0..3 {
        let (x, y) = (a.measured[c], b.measured[c]);
        if x.is_finite() && y.is_finite() && x > 0.0 && y > 0.0 {
            let w = (a.mask[c].min(b.mask[c]) as f64 / 255.0).powi(2);
            weight += w;
            gradient += w * (y.ln() - x.ln());
        }
    }
    (weight, gradient)
}

fn multiply(rows: &[Row], x: &[f64], out: &mut [f64]) {
    for (i, row) in rows.iter().enumerate() {
        let mut value = row.diag * x[i];
        for k in 0..4 {
            if row.neighbors[k] != NONE {
                value -= row.weights[k] * x[row.neighbors[k] as usize];
            }
        }
        out[i] = value;
    }
}
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}
fn magnitude(value: f64) -> f64 {
    // f64::max otherwise ignores NaNs, which must not look like convergence.
    if value.is_finite() {
        value.abs()
    } else {
        f64::INFINITY
    }
}
fn norm(a: &[f64]) -> f64 {
    a.iter().map(|&v| magnitude(v)).fold(0.0, f64::max)
}

fn solve(rows: &[Row], rhs: &[f64], x: &mut [f64], control: Control<'_>) -> crate::Result<()> {
    control.check()?;
    let tolerance = 1e-9 * norm(rhs).max(1.0);
    if !tolerance.is_finite() {
        return Err(InvalidData("gradient-tone right-hand side is not finite"));
    }
    let mut ax = vec![0.0; x.len()];
    multiply(rows, x, &mut ax);
    let mut residual: Vec<_> = rhs.iter().zip(&ax).map(|(b, a)| b - a).collect();
    let mut z: Vec<_> = residual
        .iter()
        .zip(rows)
        .map(|(r, row)| r / row.diag)
        .collect();
    let mut direction = z.clone();
    let mut rz = dot(&residual, &z);
    for iteration in 0..=1000 {
        control.check()?;
        if norm(&residual) <= tolerance {
            // Recompute the equation error instead of relying on the running estimate.
            multiply(rows, x, &mut ax);
            let actual = rhs
                .iter()
                .zip(&ax)
                .map(|(b, a)| magnitude(b - a))
                .fold(0.0, f64::max);
            if actual > tolerance {
                return Err(InvalidData("gradient-tone true residual failed"));
            }
            return Ok(());
        }
        if iteration == 1000 {
            return Err(InvalidData("gradient-tone solver did not converge"));
        }
        multiply(rows, &direction, &mut ax);
        let denominator = dot(&direction, &ax);
        if !denominator.is_finite() || denominator <= 0.0 || !rz.is_finite() {
            return Err(InvalidData("gradient-tone solver numerical breakdown"));
        }
        let alpha = rz / denominator;
        for i in 0..x.len() {
            x[i] += alpha * direction[i];
            residual[i] -= alpha * ax[i];
            z[i] = residual[i] / rows[i].diag;
        }
        let next = dot(&residual, &z);
        let beta = next / rz;
        for i in 0..x.len() {
            direction[i] = z[i] + beta * direction[i];
        }
        rz = next;
    }
    unreachable!()
}

impl Field {
    pub unsafe fn build(
        ctx: &DngCtx<'_>,
        model: Option<&LocalRecovery>,
        data: &[u16],
        stride: usize,
        control: Control<'_>,
    ) -> crate::Result<Self> {
        control.check()?;
        let model = model.ok_or(InvalidData(
            "gradient tone requires native source reliability",
        ))?;
        let y = ctx
            .camera_y
            .ok_or(InvalidData("gradient tone requires camera luminance"))?;
        let neutral = unsafe { *(ctx.prior as *const [f64; 3]) };
        let ny = dot(&y, &neutral);
        if !ny.is_finite() || ny <= 0.0 {
            return Err(InvalidData(
                "gradient tone requires positive finite neutral luminance",
            ));
        }
        let cols = ctx.cols as usize;
        let rows = ctx.rows as usize;
        let source = |row: usize, col: usize| {
            let measured = unsafe { ctx.measured(data, stride, row, col) };
            let mask = model.mask(row, col);
            let tone = tone_anchor::recover_severe(measured, mask, neutral, y);
            let eligible = tone.available && tone.strength > 0.0;
            let target = if eligible {
                tone.amplitude * ny
            } else {
                dot(&measured, &y)
            };
            Source {
                measured,
                mask,
                target,
                amplitude: tone.amplitude,
                eligible,
            }
        };
        let mut index = vec![NONE; rows * cols];
        let mut positions = Vec::new();
        let mut sources = Vec::new();
        for row in 0..rows {
            control.check()?;
            for col in 0..cols {
                if model.mask(row, col) == [255; 3] {
                    continue;
                }
                let s = source(row, col);
                if s.eligible {
                    if !s.target.is_finite() || s.target <= 0.0 {
                        return Err(InvalidData("invalid gradient-tone target"));
                    }
                    if sources.len() >= NONE as usize {
                        return Err(InvalidData("gradient-tone field exceeds index capacity"));
                    }
                    index[row * cols + col] = sources.len() as u32;
                    positions.push((row, col));
                    sources.push(s);
                }
            }
        }
        let mut matrix = Vec::with_capacity(sources.len());
        let mut rhs = Vec::with_capacity(sources.len());
        let mut x = Vec::with_capacity(sources.len());
        for (i, &(row, col)) in positions.iter().enumerate() {
            if i % 4096 == 0 {
                control.check()?;
            }
            let a = sources[i];
            let x0 = a.target.ln();
            let mut b = ANCHOR * x0;
            let mut equation = Row {
                diag: ANCHOR,
                neighbors: [NONE; 4],
                weights: [0.0; 4],
            };
            for (k, (dr, dc)) in [(-1, 0), (1, 0), (0, -1), (0, 1)].into_iter().enumerate() {
                let r = row as isize + dr;
                let c = col as isize + dc;
                if r < 0 || c < 0 || r >= rows as isize || c >= cols as isize {
                    continue;
                }
                let (r, c) = (r as usize, c as usize);
                let j = index[r * cols + c];
                let other = if j == NONE {
                    source(r, c)
                } else {
                    sources[j as usize]
                };
                if !other.target.is_finite() || other.target <= 0.0 {
                    continue;
                }
                let (weight, gradient) = edge(a, other);
                equation.diag += weight;
                b -= gradient;
                if j == NONE {
                    b += weight * other.target.ln();
                } else {
                    equation.neighbors[k] = j;
                    equation.weights[k] = weight;
                }
            }
            matrix.push(equation);
            rhs.push(b);
            x.push(x0);
        }
        solve(&matrix, &rhs, &mut x, control)?;
        let mut amplitudes = Vec::with_capacity(x.len());
        for (i, v) in x.iter().enumerate() {
            if i % 4096 == 0 {
                control.check()?;
            }
            if matrix[i].diag == ANCHOR {
                amplitudes.push(sources[i].amplitude);
                continue;
            }
            let amplitude = v.exp() / ny;
            if !amplitude.is_finite() || amplitude <= 0.0 {
                return Err(InvalidData("invalid gradient amplitude"));
            }
            amplitudes.push(amplitude);
        }
        control.check()?;
        Ok(Self {
            cols,
            index,
            amplitudes,
        })
    }

    pub fn apply(
        &self,
        row: usize,
        col: usize,
        result: &mut tone_anchor::Result,
        measured: [f64; 3],
        neutral: [f64; 3],
    ) {
        let i = self.index[row * self.cols + col];
        if i == NONE {
            return;
        }
        assert!(result.available && result.strength > 0.0);
        let amplitude = self.amplitudes[i as usize];
        if amplitude == result.amplitude {
            return;
        }
        result.amplitude = amplitude;
        result.samples = std::array::from_fn(|c| {
            let target = amplitude * neutral[c];
            if result.strength == 1.0 {
                target
            } else {
                measured[c] + result.strength * (target - measured[c])
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn sample(v: f64, mask: [u8; 3]) -> Source {
        Source {
            measured: [v, v * 2.0, v * 3.0],
            mask,
            target: v,
            amplitude: v,
            eligible: true,
        }
    }
    #[test]
    fn gradient_ignores_clipped_layers_and_retains_absolute_support() {
        let a = sample(1.0, [255, 0, 0]);
        let mut b = sample(2.0, [255, 0, 0]);
        b.measured[1] = 1e12;
        let (w, g) = edge(a, b);
        assert_eq!(w, 1.0);
        assert!((g - 2.0_f64.ln()).abs() < 1e-15);
        let (w, _) = edge(sample(1.0, [1, 0, 0]), b);
        assert!(w < 0.000016);
        assert_eq!(edge(sample(1.0, [0; 3]), b), (0.0, 0.0));
        assert_eq!(edge(b, a), (1.0, -g));
    }
    #[test]
    fn solver_matches_analytic_two_pixel_solution() {
        let rows = [
            Row {
                diag: ANCHOR + 1.0,
                neighbors: [1, NONE, NONE, NONE],
                weights: [1.0, 0.0, 0.0, 0.0],
            },
            Row {
                diag: ANCHOR + 1.0,
                neighbors: [0, NONE, NONE, NONE],
                weights: [1.0, 0.0, 0.0, 0.0],
            },
        ];
        let initial = [1.0, 2.0];
        let g = 0.1;
        let rhs = [ANCHOR * initial[0] - g, ANCHOR * initial[1] + g];
        let mut x = initial;
        solve(&rows, &rhs, &mut x, Control::none()).unwrap();
        let difference = (ANCHOR * (initial[1] - initial[0]) + 2.0 * g) / (ANCHOR + 2.0);
        assert!((x[1] - x[0] - difference).abs() < 1e-12);
        assert!((x.iter().sum::<f64>() - 3.0).abs() < 1e-12);
    }
    #[test]
    fn no_evidence_and_empty_system_keep_anchor() {
        let rows = [Row {
            diag: ANCHOR,
            neighbors: [NONE; 4],
            weights: [0.0; 4],
        }];
        let mut x = [1.25];
        solve(&rows, &[ANCHOR * 1.25], &mut x, Control::none()).unwrap();
        assert_eq!(x, [1.25]);
        solve(&[], &[], &mut [], Control::none()).unwrap();
    }
    #[test]
    fn solver_reports_nonfinite_and_breakdown_inputs_without_panicking() {
        let rows = [Row {
            diag: 0.0,
            neighbors: [NONE; 4],
            weights: [0.0; 4],
        }];
        assert!(matches!(
            solve(&rows, &[1.0], &mut [0.0], Control::none()),
            Err(InvalidData("gradient-tone solver numerical breakdown"))
        ));
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(matches!(
                solve(&rows, &[value], &mut [0.0], Control::none()),
                Err(InvalidData("gradient-tone right-hand side is not finite"))
            ));
            assert!(norm(&[0.0, value]).is_infinite());
        }
    }
    #[test]
    fn solver_iteration_limit_returns_an_error_not_an_anchor_fallback() {
        // Deliberately nonsymmetric fixture: positive p.A.p but no CG convergence.
        let rows = [
            Row {
                diag: 1.0,
                neighbors: [NONE; 4],
                weights: [0.0; 4],
            },
            Row {
                diag: 1.0,
                neighbors: [0, NONE, NONE, NONE],
                weights: [1.0, 0.0, 0.0, 0.0],
            },
        ];
        assert!(matches!(
            solve(&rows, &[1.0, 0.0], &mut [0.0; 2], Control::none()),
            Err(InvalidData("gradient-tone solver did not converge"))
        ));
    }
    #[test]
    fn cancelled_solver_does_not_change_the_initial_solution() {
        let cancel = std::sync::atomic::AtomicBool::new(true);
        let mut x = [1.25];
        let rows = [Row {
            diag: ANCHOR,
            neighbors: [NONE; 4],
            weights: [0.0; 4],
        }];
        assert!(matches!(
            solve(&rows, &[ANCHOR * 1.25], &mut x, Control::new(&cancel)),
            Err(crate::Error::Cancelled)
        ));
        assert_eq!(x, [1.25]);
    }

    #[test]
    fn field_preserves_healthy_and_applies_original_onset_once() {
        let field = Field {
            cols: 2,
            index: vec![NONE, 0],
            amplitudes: vec![2.0],
        };
        let measured = [0.3, 0.5, 0.8];
        let neutral = [0.2, 0.6, 1.0];
        let mut r = tone_anchor::Result {
            samples: measured,
            amplitude: 1.0,
            strength: 0.25,
            available: true,
        };
        field.apply(0, 0, &mut r, measured, neutral);
        assert_eq!(r.samples, measured);
        field.apply(0, 1, &mut r, measured, neutral);
        for c in 0..3 {
            assert_eq!(
                r.samples[c],
                measured[c] + 0.25 * (2.0 * neutral[c] - measured[c])
            );
        }
        assert_eq!(r.strength, 0.25);
    }
}
