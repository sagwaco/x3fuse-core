//! Two-stage BM3D: block matching, 2D DCT + group Haar transforms,
//! collaborative hard thresholding / Wiener filtering, and weighted overlap
//! aggregation. This is an independent implementation of the published method:
//! https://www.ipol.im/pub/art/2012/l-bm3d/
//! Fixed fast profile: 8x8 blocks, up to 8 matches, reference stride 4, a
//! 17x17 search sampled every two pixels. No reference-software dependency.

use super::{reflect_101, round_sat_u16, DenoiseBackend, Img3};
use crate::{Control, Result};

const BLOCK: usize = 8;
const AREA: usize = BLOCK * BLOCK;
const GROUP: usize = 8;
const RADIUS: usize = 8;
const STEP: usize = 4;
const BAND: usize = 32;
type Patch = [f32; AREA];

/// Robust noise estimate in the actual sensor YUV domain. NLM's historical
/// `h` is a filter strength, not a calibrated noise standard deviation.
/// Median absolute diagonal Haar coefficients suppress smooth image structure;
/// sample at most ~65K non-overlapping 2x2 blocks per plane.
pub(super) fn estimate_noise(src: &Img3, control: Control<'_>) -> Result<[f64; 3]> {
    if src.rows < 2 || src.cols < 2 {
        return Ok([1.0; 3]);
    }
    let stride = 2
        * (((src.rows * src.cols) as f64 / (4.0 * 65536.0))
            .sqrt()
            .ceil() as usize)
            .max(1);
    let mut samples: [Vec<f32>; 3] = std::array::from_fn(|_| Vec::new());
    for y in (0..src.rows - 1).step_by(stride) {
        control.check()?;
        for x in (0..src.cols - 1).step_by(stride) {
            let p = (y * src.cols + x) * 3;
            for c in 0..3 {
                let detail = (src.data[p + c] as f32
                    - src.data[p + 3 + c] as f32
                    - src.data[p + src.cols * 3 + c] as f32
                    + src.data[p + src.cols * 3 + 3 + c] as f32)
                    * 0.5;
                samples[c].push(detail.abs());
            }
        }
    }
    Ok(samples.map(|mut values| {
        let middle = values.len() / 2;
        let (_, median, _) = values.select_nth_unstable_by(middle, f32::total_cmp);
        (*median as f64 / 0.6744897501960817).max(1.0)
    }))
}

pub(super) fn denoise(src: &Img3, sigma: [f64; 3], control: Control<'_>) -> Result<Img3> {
    control.check()?;
    if src.rows == 0 || src.cols == 0 || sigma == [0.0; 3] {
        return Ok(src.clone());
    }
    if control.denoise_backend != DenoiseBackend::Cpu {
        #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
        let accelerated = super::gpu::bm3d(src, sigma, control);
        #[cfg(not(all(feature = "gpu", not(target_arch = "wasm32"))))]
        let accelerated = Err(crate::Error::Gpu("this build has no GPU denoiser".into()));
        match accelerated {
            Ok(image) => {
                super::log(crate::x3f_verbosity_t_DEBUG, "BM3D: gpu");
                return Ok(image);
            }
            Err(crate::Error::Cancelled) => return Err(crate::Error::Cancelled),
            Err(error) if control.denoise_backend == DenoiseBackend::Gpu => return Err(error),
            Err(error) => super::log(
                crate::x3f_verbosity_t_DEBUG,
                &format!("BM3D: CPU fallback ({error})"),
            ),
        }
    }
    super::log(crate::x3f_verbosity_t_DEBUG, "BM3D: cpu");
    denoise_cpu(src, sigma, control)
}

pub(super) fn denoise_cpu(src: &Img3, sigma: [f64; 3], control: Control<'_>) -> Result<Img3> {
    let mut out = src.clone();
    if src.rows == 0 || src.cols == 0 {
        return Ok(out);
    }
    for (channel, sigma) in sigma.into_iter().enumerate() {
        if sigma == 0.0 {
            continue;
        }
        let noisy: Vec<f32> = src
            .data
            .chunks_exact(3)
            .map(|p| p[channel] as f32)
            .collect();
        let basic = stage(
            &noisy,
            &noisy,
            src.cols,
            src.rows,
            sigma as f32,
            false,
            control,
        )?;
        let final_image = stage(
            &noisy,
            &basic,
            src.cols,
            src.rows,
            sigma as f32,
            true,
            control,
        )?;
        for (pixel, value) in out.data.chunks_exact_mut(3).zip(final_image) {
            pixel[channel] = round_sat_u16(value as f64);
        }
    }
    control.check()?;
    Ok(out)
}

pub(super) fn dct_basis() -> &'static [[f32; BLOCK]; BLOCK] {
    static BASIS: std::sync::OnceLock<[[f32; BLOCK]; BLOCK]> = std::sync::OnceLock::new();
    BASIS.get_or_init(|| {
        std::array::from_fn(|k| {
            std::array::from_fn(|x| {
                let scale = if k == 0 {
                    (1.0 / BLOCK as f32).sqrt()
                } else {
                    (2.0 / BLOCK as f32).sqrt()
                };
                scale
                    * (std::f32::consts::PI * (2 * x + 1) as f32 * k as f32 / (2 * BLOCK) as f32)
                        .cos()
            })
        })
    })
}

fn transform(patch: &mut Patch, inverse: bool) {
    let basis = dct_basis();
    let mut tmp = [0.0; AREA];
    for y in 0..BLOCK {
        for x in 0..BLOCK {
            for k in 0..BLOCK {
                tmp[y * BLOCK + x] +=
                    patch[y * BLOCK + k] * if inverse { basis[k][x] } else { basis[x][k] };
            }
        }
    }
    patch.fill(0.0);
    for y in 0..BLOCK {
        for x in 0..BLOCK {
            for k in 0..BLOCK {
                patch[y * BLOCK + x] +=
                    tmp[k * BLOCK + x] * if inverse { basis[k][y] } else { basis[y][k] };
            }
        }
    }
}

fn haar(values: &mut [f32; GROUP], n: usize, inverse: bool) {
    let mut size = if inverse { 2 } else { n };
    while size >= 2 && size <= n {
        let mut tmp = *values;
        let half = size / 2;
        for i in 0..half {
            if inverse {
                tmp[2 * i] = (values[i] + values[half + i]) * std::f32::consts::FRAC_1_SQRT_2;
                tmp[2 * i + 1] = (values[i] - values[half + i]) * std::f32::consts::FRAC_1_SQRT_2;
            } else {
                tmp[i] = (values[2 * i] + values[2 * i + 1]) * std::f32::consts::FRAC_1_SQRT_2;
                tmp[half + i] =
                    (values[2 * i] - values[2 * i + 1]) * std::f32::consts::FRAC_1_SQRT_2;
            }
        }
        *values = tmp;
        if inverse {
            size *= 2;
        } else {
            size /= 2;
        }
    }
}

struct Plane<'a> {
    data: &'a [f32],
    width: usize,
    height: usize,
}
impl Plane<'_> {
    fn patch(&self, x: usize, y: usize) -> Patch {
        std::array::from_fn(|i| {
            let px = reflect_101((x + i % BLOCK) as i32, self.width as i32) as usize;
            let py = reflect_101((y + i / BLOCK) as i32, self.height as i32) as usize;
            self.data[py * self.width + px]
        })
    }
}

fn matches(guide: &Plane<'_>, x: usize, y: usize, limit: f32) -> ([(usize, usize); GROUP], usize) {
    let reference = guide.patch(x, y);
    let mut selected = [(x, y); GROUP];
    let mut distances = [f32::INFINITY; GROUP];
    distances[0] = 0.0; // Always retain the reference, including on constant fields.
    let mut count = 1;
    for dy in (-(RADIUS as isize)..=RADIUS as isize).step_by(2) {
        for dx in (-(RADIUS as isize)..=RADIUS as isize).step_by(2) {
            let px = x as isize + dx;
            let py = y as isize + dy;
            if (dx == 0 && dy == 0)
                || px < 0
                || py < 0
                || px >= guide.width as isize
                || py >= guide.height as isize
            {
                continue;
            }
            let candidate = guide.patch(px as usize, py as usize);
            let distance = reference
                .iter()
                .zip(candidate)
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f32>()
                / AREA as f32;
            if distance > limit || distance >= distances[GROUP - 1] {
                continue;
            }
            let mut index = GROUP - 1;
            while index > 1 && distance < distances[index - 1] {
                distances[index] = distances[index - 1];
                selected[index] = selected[index - 1];
                index -= 1;
            }
            distances[index] = distance;
            selected[index] = (px as usize, py as usize);
            count = (count + 1).min(GROUP);
        }
    }
    // Haar needs a power-of-two group. The closest matches are first.
    (selected, 1 << count.ilog2())
}

fn filter(
    noisy: &Plane<'_>,
    guide: &Plane<'_>,
    selected: &[(usize, usize); GROUP],
    n: usize,
    sigma: f32,
    wiener: bool,
) -> ([Patch; GROUP], f32) {
    let mut blocks = [[0.0; AREA]; GROUP];
    let mut pilot = [[0.0; AREA]; GROUP];
    for i in 0..n {
        blocks[i] = noisy.patch(selected[i].0, selected[i].1);
        transform(&mut blocks[i], false);
        if wiener {
            pilot[i] = guide.patch(selected[i].0, selected[i].1);
            transform(&mut pilot[i], false);
        }
    }
    let mut norm = 0.0_f32;
    for coefficient in 0..AREA {
        let mut values = std::array::from_fn(|i| blocks[i][coefficient]);
        haar(&mut values, n, false);
        if wiener {
            let mut estimates = std::array::from_fn(|i| pilot[i][coefficient]);
            haar(&mut estimates, n, false);
            for i in 0..n {
                let power = estimates[i] * estimates[i];
                let gain = power / (power + sigma * sigma);
                values[i] *= gain;
                norm += gain * gain;
            }
        } else {
            for value in &mut values[..n] {
                if value.abs() < 2.7 * sigma {
                    *value = 0.0;
                } else {
                    norm += 1.0;
                }
            }
        }
        haar(&mut values, n, true);
        for i in 0..n {
            blocks[i][coefficient] = values[i];
        }
    }
    for block in &mut blocks[..n] {
        transform(block, true);
    }
    // The common sigma^-2 factor cancels during per-plane aggregation.
    (blocks, 1.0 / norm.max(1.0))
}

pub(super) fn window() -> &'static Patch {
    static WINDOW: std::sync::OnceLock<Patch> = std::sync::OnceLock::new();
    WINDOW.get_or_init(|| {
        fn i0(x: f32) -> f32 {
            let mut term = 1.0;
            let mut sum = 1.0;
            for k in 1..12 {
                term *= x * x / (4 * k * k) as f32;
                sum += term;
            }
            sum
        }
        let axis: [f32; BLOCK] = std::array::from_fn(|i| {
            let x = 2.0 * i as f32 / (BLOCK - 1) as f32 - 1.0;
            i0(2.0 * (1.0 - x * x).max(0.0).sqrt()) / i0(2.0)
        });
        std::array::from_fn(|i| axis[i / BLOCK] * axis[i % BLOCK])
    })
}

fn stage(
    noisy: &[f32],
    guide: &[f32],
    width: usize,
    height: usize,
    sigma: f32,
    wiener: bool,
    control: Control<'_>,
) -> Result<Vec<f32>> {
    let noisy = Plane {
        data: noisy,
        width,
        height,
    };
    let guide = Plane {
        data: guide,
        width,
        height,
    };
    let mut out = vec![0.0; width * height];
    let band = |index: usize, output: &mut [f32]| -> Result<()> {
        let top = index * BAND;
        let bottom = top + output.len() / width;
        let mut weights = vec![0.0; output.len()];
        let first = top.saturating_sub(RADIUS + BLOCK - 1).div_ceil(STEP) * STEP;
        let last = (bottom + RADIUS).min(height);
        for y in (first..last).step_by(STEP) {
            for x in (0..width).step_by(STEP) {
                control.check()?;
                let (selected, n) =
                    matches(&guide, x, y, sigma * sigma * if wiener { 2.0 } else { 4.0 });
                let (blocks, weight) = filter(&noisy, &guide, &selected, n, sigma, wiener);
                for i in 0..n {
                    let (px, py) = selected[i];
                    for by in 0..BLOCK {
                        let sy = py + by;
                        if sy < top || sy >= bottom {
                            continue;
                        }
                        for bx in 0..BLOCK.min(width - px) {
                            let dest = (sy - top) * width + px + bx;
                            let w = weight * window()[by * BLOCK + bx];
                            output[dest] += w * blocks[i][by * BLOCK + bx];
                            weights[dest] += w;
                        }
                    }
                }
            }
        }
        for (i, value) in output.iter_mut().enumerate() {
            *value = if weights[i] > 0.0 {
                *value / weights[i]
            } else {
                noisy.data[top * width + i]
            };
        }
        Ok(())
    };
    #[cfg(not(target_arch = "wasm32"))]
    {
        use rayon::prelude::*;
        out.par_chunks_mut(BAND * width)
            .enumerate()
            .try_for_each(|(i, output)| band(i, output))?;
    }
    #[cfg(target_arch = "wasm32")]
    for (i, output) in out.chunks_mut(BAND * width).enumerate() {
        band(i, output)?;
    }
    control.check()?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn transforms_round_trip_and_preserve_energy() {
        let source: Patch = std::array::from_fn(|i| (i as f32 * 0.7).sin() * 100.0);
        let mut transformed = source;
        transform(&mut transformed, false);
        let energy = |values: &[f32]| values.iter().map(|v| v * v).sum::<f32>();
        assert!((energy(&source) - energy(&transformed)).abs() / energy(&source) < 1e-5);
        transform(&mut transformed, true);
        assert!(source
            .iter()
            .zip(transformed)
            .all(|(a, b)| (a - b).abs() < 0.001));
        for n in [1, 2, 4, 8] {
            let original = [1., -2., 3., -4., 5., -6., 7., -8.];
            let mut group = original;
            haar(&mut group, n, false);
            haar(&mut group, n, true);
            assert!(group
                .iter()
                .zip(original)
                .all(|(a, b)| (a - b).abs() < 1e-5));
        }
    }

    #[test]
    fn two_stage_filter_reduces_noise_and_preserves_disabled_planes() {
        let mut src = Img3::new(37, 39);
        let mut seed = 42_u32;
        for (i, pixel) in src.data.chunks_exact_mut(3).enumerate() {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            let signal = if i % 39 < 20 { 20000. } else { 24000. };
            // Centered uniform noise, stddev approximately 300.
            pixel.copy_from_slice(&[
                i as u16,
                (signal + (seed >> 16) as f32 / 65535.0 * 1040.0 - 520.0) as u16,
                30000,
            ]);
        }
        let estimate = estimate_noise(&src, Control::none()).unwrap();
        assert!((240.0..=370.0).contains(&estimate[1]), "{estimate:?}");
        let out = denoise(
            &src,
            [0., 300., 0.],
            Control::none().with_denoise_backend(DenoiseBackend::Cpu),
        )
        .unwrap();
        let mse = |image: &Img3| {
            image
                .data
                .chunks_exact(3)
                .enumerate()
                .map(|(i, p)| {
                    let signal = if i % 39 < 20 { 20000. } else { 24000. };
                    (p[1] as f64 - signal).powi(2)
                })
                .sum::<f64>()
                / (39. * 37.)
        };
        assert!(
            mse(&out) < mse(&src) * 0.3,
            "{} -> {}",
            mse(&src),
            mse(&out)
        );
        assert!(src
            .data
            .chunks_exact(3)
            .zip(out.data.chunks_exact(3))
            .all(|(a, b)| a[0] == b[0] && a[2] == b[2]));
        for (rows, cols) in [(1, 1), (1, 5), (3, 2), (33, 35)] {
            let src = Img3 {
                rows,
                cols,
                data: vec![32000; rows * cols * 3],
            };
            let out = denoise(
                &src,
                [100.; 3],
                Control::none().with_denoise_backend(DenoiseBackend::Cpu),
            )
            .unwrap();
            assert!(out.data.iter().all(|v| v.abs_diff(32000) <= 1));
        }
        let cancel = std::sync::atomic::AtomicBool::new(true);
        assert!(matches!(
            denoise(&src, [100.; 3], Control::new(&cancel)),
            Err(crate::Error::Cancelled)
        ));
    }
}
