//! Tile phase correlation and conservative resampling for the X3I merge.
//! Transforms map frame coordinates onto the reference grid.

use crate::Error;
use x3f_sys::Control;

/// Alignment settings for one conversion.
#[derive(Clone, Debug, PartialEq)]
pub struct AlignOptions {
    /// Estimate and apply a transform for every frame except the reference.
    pub enabled: bool,
    /// Reject tiles with a correlation peak below this value.
    pub min_peak: f32,
    /// Largest shift the estimator searches, in full-resolution pixels.
    pub max_shift_px: u32,
    /// Tiles per axis for the per-tile phase correlation.
    pub tile_grid: u32,
    /// Largest absolute fitted rotation, in degrees.
    pub max_rotation_deg: f32,
    /// Largest absolute difference between fitted scale and one.
    pub max_scale_deviation: f32,
}

impl Default for AlignOptions {
    fn default() -> Self {
        Self {
            enabled: true,
            min_peak: 0.05,
            max_shift_px: 64,
            tile_grid: 3,
            max_rotation_deg: 0.5,
            max_scale_deviation: 0.005,
        }
    }
}

/// Gradient magnitude of exposure-normalised log luminance at half resolution,
/// with a mask that excludes clipped and near-black pixels.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Luma {
    /// Width of the half-resolution map, in pixels.
    pub width: u32,
    /// Height of the half-resolution map, in pixels.
    pub height: u32,
    /// Gradient magnitude per pixel, row-major.
    pub data: Vec<f32>,
    /// One per pixel; nonzero where the pixel takes part in the estimate.
    pub mask: Vec<u8>,
    /// Smoothed block-mean log luminance for refinement. Empty uses `data`.
    pub log: Vec<f32>,
}

/// Transform of one frame relative to the reference, in full-resolution pixels.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Alignment {
    /// Horizontal shift that maps frame coordinates onto the reference.
    pub dx: f32,
    /// Vertical shift that maps frame coordinates onto the reference.
    pub dy: f32,
    /// Normalised correlation peak of the global estimate.
    pub peak: f32,
    /// Largest distance from the median shift among retained tiles, in pixels.
    pub tile_spread: f32,
    /// Largest retained-tile residual of the selected transform, in pixels.
    pub max_tile_residual: f32,
    /// Number of tiles retained for the selected transform.
    pub retained_tiles: u32,
    /// Frame-to-reference transform `[a, b, tx, c, d, ty]` when tiles disagree.
    /// Maps `(x, y)` to `(a*x + b*y + tx, c*x + d*y + ty)`.
    pub similarity: Option<[f32; 6]>,
    /// True when alignment is not trusted. Merge this frame without a transform.
    pub skipped: bool,
}

impl Alignment {
    /// Map a full-resolution reference coordinate to the source frame.
    /// Skipped alignments keep the original coordinate.
    /// A singular transform gives non-finite coordinates; callers must reject them.
    pub fn source_at(&self, x: f32, y: f32) -> [f32; 2] {
        if self.skipped {
            return [x, y];
        }
        let [a, b, tx, c, d, ty] = self
            .similarity
            .unwrap_or([1.0, 0.0, self.dx, 0.0, 1.0, self.dy]);
        let det = a * d - b * c;
        let (u, v) = (x - tx, y - ty);
        [(d * u - b * v) / det, (a * v - c * u) / det]
    }

    /// True when the transform leaves the frame unchanged.
    pub fn is_identity(&self) -> bool {
        match self.similarity {
            Some(matrix) => matrix == [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            None => self.dx == 0.0 && self.dy == 0.0,
        }
    }
}

/// Per-frame planes in reference units, as the merge accumulates them.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct FramePlanes {
    /// Width of the expanded raster, in pixels.
    pub width: u32,
    /// Height of the expanded raster, in pixels.
    pub height: u32,
    /// Three interleaved channels, `scale * bmt`.
    pub x: Vec<f32>,
    /// Two values per pixel: the chroma variance and the luma variance.
    pub var: Vec<f32>,
    /// Saturation input per pixel, relative to the physical ceiling.
    pub d_max: Vec<f32>,
    /// One per pixel; nonzero when the pixel is valid.
    pub valid: Vec<u8>,
}

fn invalid(message: &str) -> Error {
    Error::InvalidData(format!("X3I alignment: {message}"))
}

fn validate_planes(p: &FramePlanes) -> Result<usize, Error> {
    let n = (p.width as usize)
        .checked_mul(p.height as usize)
        .filter(|&n| n > 0 && n <= usize::MAX / 3)
        .ok_or_else(|| invalid("invalid dimensions"))?;
    if p.x.len() != n * 3 || p.var.len() != n * 2 || p.d_max.len() != n || p.valid.len() != n {
        return Err(invalid("plane lengths do not match dimensions"));
    }
    Ok(n)
}

/// Build log luminance from reference-unit BMT, then halve it and take its gradient.
/// Mask luminance at or below 0.001 in reference-white units.
/// `cut` is the clipping limit for `d_max`, relative to the physical ceiling.
pub fn luma(planes: &FramePlanes, cut: f32, control: Control<'_>) -> Result<Luma, Error> {
    control.check()?;
    validate_planes(planes)?;
    if !cut.is_finite() || cut <= 0.0 {
        return Err(invalid("invalid clipping limit"));
    }
    let (w, h) = (planes.width as usize / 2, planes.height as usize / 2);
    let mut log = vec![0.0; w * h];
    let mut mask = vec![0; w * h];
    for y in 0..h {
        control.check()?;
        for x in 0..w {
            let mut sum = 0.0;
            let mut valid = true;
            for j in 0..2 {
                for i in 0..2 {
                    let p = (2 * y + j) * planes.width as usize + 2 * x + i;
                    let rgb = &planes.x[p * 3..p * 3 + 3];
                    let value = (rgb[0] + rgb[1] + rgb[2]) / 3.0;
                    valid &= planes.valid[p] != 0
                        && planes.d_max[p].is_finite()
                        && planes.d_max[p] < cut
                        && value.is_finite()
                        && value > 1e-3;
                    sum += value.max(1e-3).ln();
                }
            }
            log[y * w + x] = if valid { sum * 0.25 } else { 0.0 };
            mask[y * w + x] = u8::from(valid);
        }
    }
    let mut output = Luma {
        width: w as u32,
        height: h as u32,
        data: vec![0.0; w * h],
        mask: vec![0; w * h],
        log: vec![0.0; w * h],
    };
    for y in 1..h.saturating_sub(1) {
        control.check()?;
        for x in 1..w.saturating_sub(1) {
            let p = y * w + x;
            if [
                p,
                p - 1,
                p + 1,
                p - w,
                p + w,
                p - w - 1,
                p - w + 1,
                p + w - 1,
                p + w + 1,
            ]
            .iter()
            .all(|&i| mask[i] != 0)
            {
                output.data[p] = (log[p + 1] - log[p - 1]).hypot(log[p + w] - log[p - w]) * 0.5;
                output.mask[p] = 1;
                // Reduce interpolation bias from sensor noise in spatial refinement.
                output.log[p] = (log[p] * 4.0
                    + (log[p - 1] + log[p + 1] + log[p - w] + log[p + w]) * 2.0
                    + log[p - w - 1]
                    + log[p - w + 1]
                    + log[p + w - 1]
                    + log[p + w + 1])
                    / 16.0;
            }
        }
    }
    Ok(output)
}

// The FFT uses bounded square tiles and needs no plan or external dependency.
fn fft(values: &mut [[f32; 2]], inverse: bool) {
    let n = values.len();
    let mut j = 0;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            values.swap(i, j);
        }
    }
    let mut len = 2;
    while len <= n {
        let angle = if inverse { 1.0 } else { -1.0 } * std::f32::consts::TAU / len as f32;
        let (sin, cos) = angle.sin_cos();
        for block in values.chunks_exact_mut(len) {
            let (mut re, mut im) = (1.0, 0.0);
            for i in 0..len / 2 {
                let a = block[i];
                let b = block[i + len / 2];
                let v = [b[0] * re - b[1] * im, b[0] * im + b[1] * re];
                block[i] = [a[0] + v[0], a[1] + v[1]];
                block[i + len / 2] = [a[0] - v[0], a[1] - v[1]];
                (re, im) = (re * cos - im * sin, re * sin + im * cos);
            }
        }
        len *= 2;
    }
    if inverse {
        for value in values {
            value[0] /= n as f32;
            value[1] /= n as f32;
        }
    }
}

fn fft2(
    values: &mut [[f32; 2]],
    n: usize,
    inverse: bool,
    control: Control<'_>,
) -> Result<(), Error> {
    for row in values.chunks_exact_mut(n) {
        control.check()?;
        fft(row, inverse);
    }
    let mut column = vec![[0.0; 2]; n];
    for x in 0..n {
        control.check()?;
        for y in 0..n {
            column[y] = values[y * n + x];
        }
        fft(&mut column, inverse);
        for y in 0..n {
            values[y * n + x] = column[y];
        }
    }
    Ok(())
}

fn parabola(left: f32, centre: f32, right: f32) -> f32 {
    let curvature = left - 2.0 * centre + right;
    if curvature < -1e-10 {
        (0.5 * (left - right) / curvature).clamp(-0.5, 0.5)
    } else {
        0.0
    }
}

fn cubic(t: f32) -> [f32; 4] {
    let t2 = t * t;
    let t3 = t2 * t;
    [
        -0.5 * t + t2 - 0.5 * t3,
        1.0 - 2.5 * t2 + 1.5 * t3,
        0.5 * t + 2.0 * t2 - 1.5 * t3,
        -0.5 * t2 + 0.5 * t3,
    ]
}

#[derive(Clone, Copy, Debug)]
struct Tile {
    x: f32,
    y: f32,
    dx: f32,
    dy: f32,
    peak: f32,
}

fn tile_shift(
    reference: &Luma,
    frame: &Luma,
    origin: [usize; 2],
    n: usize,
    range: i32,
    control: Control<'_>,
) -> Result<Tile, Error> {
    let w = reference.width as usize;
    let [ox, oy] = origin;
    let mut a = vec![[0.0; 2]; n * n];
    let mut b = a.clone();
    let window: Vec<_> = (0..n)
        .map(|i| 0.5 - 0.5 * (std::f32::consts::TAU * i as f32 / (n - 1) as f32).cos())
        .collect();
    for (map, out) in [(reference, &mut a), (frame, &mut b)] {
        let mut sum = 0.0;
        let mut count = 0;
        for y in 0..n {
            for x in 0..n {
                let p = (oy + y) * w + ox + x;
                if map.mask[p] != 0 {
                    sum += map.data[p];
                    count += 1;
                }
            }
        }
        let mean = sum / count.max(1) as f32;
        for y in 0..n {
            for x in 0..n {
                let p = (oy + y) * w + ox + x;
                if map.mask[p] != 0 {
                    out[y * n + x][0] = (map.data[p] - mean) * window[x] * window[y];
                }
            }
        }
        fft2(out, n, false, control)?;
    }
    for (a, b) in a.iter_mut().zip(b) {
        let re = a[0] * b[0] + a[1] * b[1];
        let im = a[1] * b[0] - a[0] * b[1];
        let norm = re.hypot(im);
        *a = if norm > 1e-12 {
            [re / norm, im / norm]
        } else {
            [0.0; 2]
        };
    }
    fft2(&mut a, n, true, control)?;
    let at = |x: i32, y: i32| {
        a[y.rem_euclid(n as i32) as usize * n + x.rem_euclid(n as i32) as usize][0]
    };
    let (mut dx, mut dy, mut peak) = (0, 0, 0.0);
    for y in -range..=range {
        for x in -range..=range {
            if at(x, y) > peak {
                (dx, dy, peak) = (x, y, at(x, y));
            }
        }
    }
    let mut shift = [
        dx as f32 + parabola(at(dx - 1, dy), peak, at(dx + 1, dy)),
        dy as f32 + parabola(at(dx, dy - 1), peak, at(dx, dy + 1)),
    ];
    let phase_shift = shift;
    // Refine on masked samples to remove the phase peak's fractional-pixel bias.
    for step in [0.5, 0.5, 0.25, 0.125, 0.0625] {
        control.check()?;
        let mut scores = [0.0; 9];
        for (i, score) in scores.iter_mut().enumerate() {
            *score = tile_score(
                reference,
                frame,
                origin,
                n,
                [
                    shift[0] + (i % 3) as f32 * step - step,
                    shift[1] + (i / 3) as f32 * step - step,
                ],
            );
        }
        if scores[4] == f32::NEG_INFINITY {
            shift = phase_shift;
            break;
        }
        let offset = refinement_offset(scores);
        for axis in 0..2 {
            shift[axis] = (shift[axis] + step * offset[axis]).clamp(-range as f32, range as f32);
        }
    }
    Ok(Tile {
        x: (ox as f32 + (n - 1) as f32 * 0.5) * 2.0 + 0.5,
        y: (oy as f32 + (n - 1) as f32 * 0.5) * 2.0 + 0.5,
        dx: shift[0] * 2.0,
        dy: shift[1] * 2.0,
        peak,
    })
}

fn refinement_offset(scores: [f32; 9]) -> [f32; 2] {
    let best = (0..9)
        .max_by(|&i, &j| scores[i].total_cmp(&scores[j]))
        .unwrap();
    // Keep the phase shift when the masked window has no preference.
    let best = if scores[best] > scores[4] { best } else { 4 };
    let mut offset = [(best % 3) as f32 - 1.0, (best / 3) as f32 - 1.0];
    if best == 4 && scores.iter().all(|score| score.is_finite()) {
        let xx = scores[3] - 2.0 * scores[4] + scores[5];
        let yy = scores[1] - 2.0 * scores[4] + scores[7];
        let xy = (scores[8] - scores[6] - scores[2] + scores[0]) * 0.25;
        let gx = (scores[5] - scores[3]) * 0.5;
        let gy = (scores[7] - scores[1]) * 0.5;
        let det = xx * yy - xy * xy;
        if xx < 0.0 && yy < 0.0 && det > 1e-12 {
            offset = [
                ((xy * gy - yy * gx) / det).clamp(-0.5, 0.5),
                ((xy * gx - xx * gy) / det).clamp(-0.5, 0.5),
            ];
        }
    }
    offset
}

fn tile_score(
    reference: &Luma,
    frame: &Luma,
    origin: [usize; 2],
    n: usize,
    shift: [f32; 2],
) -> f32 {
    let w = reference.width as usize;
    let h = reference.height as usize;
    let ref_data = if reference.log.is_empty() {
        &reference.data
    } else {
        &reference.log
    };
    let frame_data = if frame.log.is_empty() {
        &frame.data
    } else {
        &frame.log
    };
    let mut sums = [0.0f64; 6];
    let stride = (n / 96).max(1);
    for y in (origin[1] + n / 4..origin[1] + n * 3 / 4).step_by(stride) {
        for x in (origin[0] + n / 4..origin[0] + n * 3 / 4).step_by(stride) {
            let (sx, sy) = (x as f32 - shift[0], y as f32 - shift[1]);
            let (ix, iy) = (sx.floor() as isize, sy.floor() as isize);
            if ix < 1
                || iy < 1
                || ix + 2 >= w as isize
                || iy + 2 >= h as isize
                || reference.mask[y * w + x] == 0
            {
                continue;
            }
            let wx = cubic(sx - ix as f32);
            let wy = cubic(sy - iy as f32);
            let mut value = 0.0;
            let mut valid = true;
            for (j, &wy) in wy.iter().enumerate() {
                for (i, &wx) in wx.iter().enumerate() {
                    let p = (iy as usize + j - 1) * w + ix as usize + i - 1;
                    valid &= frame.mask[p] != 0;
                    value += frame_data[p] * wx * wy;
                }
            }
            if valid {
                let a = ref_data[y * w + x] as f64;
                let b = value as f64;
                sums[0] += a;
                sums[1] += b;
                sums[2] += a * a;
                sums[3] += b * b;
                sums[4] += a * b;
                sums[5] += 1.0;
            }
        }
    }
    let [a, b, aa, bb, ab, count] = sums;
    let norm = ((aa * count - a * a) * (bb * count - b * b))
        .max(0.0)
        .sqrt();
    if count < 64.0 {
        f32::NEG_INFINITY
    } else if norm < 1e-12 {
        0.0
    } else {
        ((ab * count - a * b) / norm) as f32
    }
}

fn median(mut values: Vec<f32>) -> f32 {
    values.sort_unstable_by(f32::total_cmp);
    let mid = values.len() / 2;
    (values[mid] + values[(values.len() - 1) / 2]) * 0.5
}

// Fit the inverse mapping at reference tile centres, then invert it for the API.
fn similarity(tiles: &[Tile]) -> Option<[f32; 6]> {
    if tiles.len() < 2 {
        return None;
    }
    let count = tiles.len() as f64;
    let mx = tiles.iter().map(|t| t.x as f64).sum::<f64>() / count;
    let my = tiles.iter().map(|t| t.y as f64).sum::<f64>() / count;
    let dx = tiles.iter().map(|t| t.dx as f64).sum::<f64>() / count;
    let dy = tiles.iter().map(|t| t.dy as f64).sum::<f64>() / count;
    let (mut norm, mut a, mut b) = (0.0, 0.0, 0.0);
    for t in tiles {
        let (x, y) = (t.x as f64 - mx, t.y as f64 - my);
        let (u, v) = (t.dx as f64 - dx, t.dy as f64 - dy);
        norm += x * x + y * y;
        a += x * u + y * v;
        b += x * v - y * u;
    }
    if norm < 1.0 {
        return None;
    }
    let a = 1.0 - a / norm;
    let b = -b / norm;
    let tx = mx - dx - a * mx + b * my;
    let ty = my - dy - b * mx - a * my;
    let det = a * a + b * b;
    if det < 1e-6 {
        return None;
    }
    Some([
        (a / det) as f32,
        (b / det) as f32,
        ((-a * tx - b * ty) / det) as f32,
        (-b / det) as f32,
        (a / det) as f32,
        ((b * tx - a * ty) / det) as f32,
    ])
}

/// Estimate the frame-to-reference transform. Untrusted frames are marked skipped.
pub fn estimate(
    reference: &Luma,
    frame: &Luma,
    options: &AlignOptions,
    control: Control<'_>,
) -> Result<Alignment, Error> {
    control.check()?;
    if !options.enabled {
        return Ok(Alignment {
            peak: 1.0,
            ..Default::default()
        });
    }
    let n_pixels = (reference.width as usize)
        .checked_mul(reference.height as usize)
        .ok_or_else(|| invalid("invalid luma dimensions"))?;
    if reference.width != frame.width
        || reference.height != frame.height
        || [reference, frame].iter().any(|m| {
            m.data.len() != n_pixels
                || m.mask.len() != n_pixels
                || (!m.log.is_empty() && m.log.len() != n_pixels)
                || m.data.iter().chain(&m.log).any(|v| !v.is_finite())
        })
        || !options.min_peak.is_finite()
        || !(0.0..=1.0).contains(&options.min_peak)
        || options.tile_grid == 0
        || options.tile_grid > 16
        || options.max_shift_px > 4096
        || !options.max_rotation_deg.is_finite()
        || options.max_rotation_deg < 0.0
        || !options.max_scale_deviation.is_finite()
        || options.max_scale_deviation < 0.0
    {
        return Err(invalid("invalid maps or options"));
    }
    let side = reference.width.min(reference.height) as usize;
    let desired = (side / options.tile_grid as usize)
        .max(options.max_shift_px as usize + 16)
        .next_power_of_two()
        .min(512);
    let side = side.min(desired);
    let n = if side == 0 { 0 } else { 1usize << side.ilog2() };
    let skipped = Alignment {
        skipped: true,
        ..Default::default()
    };
    if n < 32 {
        return Ok(skipped);
    }
    let range = (options.max_shift_px as usize).div_ceil(2);
    if range + 4 >= n / 2 {
        return Ok(skipped);
    }
    let grid = options.tile_grid as usize;
    let mut tiles = Vec::new();
    let mut peaks = Vec::new();
    for y in 0..grid {
        for x in 0..grid {
            control.check()?;
            let origin = [
                (reference.width as usize - n) * (2 * x + 1) / (2 * grid),
                (reference.height as usize - n) * (2 * y + 1) / (2 * grid),
            ];
            let tile = tile_shift(reference, frame, origin, n, range as i32, control)?;
            peaks.push(tile.peak);
            if tile.peak >= options.min_peak && tile.peak > 0.0 {
                tiles.push(tile);
            }
        }
    }
    if tiles.len() < 3 {
        return Ok(Alignment {
            peak: median(peaks),
            ..skipped
        });
    }
    let mut dx = median(tiles.iter().map(|t| t.dx).collect());
    let mut dy = median(tiles.iter().map(|t| t.dy).collect());
    let mut spread = tiles
        .iter()
        .map(|t| (t.dx - dx).hypot(t.dy - dy))
        .fold(0.0f32, f32::max);
    let mut transform = None;
    let residual = |fit: [f32; 6], t: &Tile| {
        let (x, y) = (t.x - t.dx, t.y - t.dy);
        (fit[0] * x + fit[1] * y + fit[2] - t.x).hypot(fit[3] * x + fit[4] * y + fit[5] - t.y)
    };
    if spread > 1.5 {
        // A least-median seed prevents moving tiles from biasing the rejection pass.
        let mut best = None;
        let mut error = f32::INFINITY;
        for i in 0..tiles.len() {
            for j in i + 1..tiles.len() {
                if let Some(fit) = similarity(&[tiles[i], tiles[j]]) {
                    let score = median(tiles.iter().map(|t| residual(fit, t)).collect());
                    if score < error {
                        best = Some(fit);
                        error = score;
                    }
                }
            }
        }
        if let Some(fit) = best {
            let limit = (error * 2.5).max(0.5);
            tiles.retain(|t| residual(fit, t) <= limit);
            if tiles.len() < 3 {
                return Ok(Alignment {
                    peak: median(peaks),
                    ..skipped
                });
            }
            dx = median(tiles.iter().map(|t| t.dx).collect());
            dy = median(tiles.iter().map(|t| t.dy).collect());
            spread = tiles
                .iter()
                .map(|t| (t.dx - dx).hypot(t.dy - dy))
                .fold(0.0f32, f32::max);
            if spread > 1.5 {
                transform = similarity(&tiles).filter(|fit| {
                    let angle = fit[3].atan2(fit[0]).to_degrees();
                    let scale = fit[0].hypot(fit[3]);
                    angle.abs() <= options.max_rotation_deg
                        && (scale - 1.0).abs() <= options.max_scale_deviation
                });
            }
        }
    }
    Ok(Alignment {
        dx,
        dy,
        peak: median(tiles.iter().map(|t| t.peak).collect()),
        tile_spread: spread,
        retained_tiles: tiles.len() as u32,
        max_tile_residual: tiles
            .iter()
            .map(|t| residual(transform.unwrap_or([1.0, 0.0, dx, 0.0, 1.0, dy]), t))
            .fold(0.0f32, f32::max),
        similarity: transform,
        skipped: false,
    })
}

/// Resample with the inverse frame-to-reference transform.
/// Allocate a full output plane set (25 bytes per pixel) while the input is live.
/// The 300-row bands batch cancellation checks; they do not reduce this allocation.
/// All 16 cubic taps must be inside the frame and valid. `d_max` takes their maximum.
/// Identity and skipped alignments return the input allocations unchanged.
pub fn warp(
    planes: FramePlanes,
    alignment: &Alignment,
    control: Control<'_>,
) -> Result<FramePlanes, Error> {
    control.check()?;
    let n = validate_planes(&planes)?;
    if alignment.is_identity() || alignment.skipped {
        return Ok(planes);
    }
    let [a, b, tx, c, d, ty] =
        alignment
            .similarity
            .unwrap_or([1.0, 0.0, alignment.dx, 0.0, 1.0, alignment.dy]);
    let det = a * d - b * c;
    if ![a, b, tx, c, d, ty, det].iter().all(|v| v.is_finite()) || det.abs() < 1e-6 {
        return Err(invalid("invalid transform"));
    }
    let w = planes.width as usize;
    let h = planes.height as usize;
    let mut out = FramePlanes {
        width: planes.width,
        height: planes.height,
        x: vec![0.0; n * 3],
        var: vec![0.0; n * 2],
        d_max: vec![f32::INFINITY; n],
        valid: vec![0; n],
    };
    for band in (0..h).step_by(300) {
        control.check()?;
        for y in band..(band + 300).min(h) {
            control.check()?;
            for x in 0..w {
                let [sx, sy] = alignment.source_at(x as f32, y as f32);
                let ix = sx.floor();
                let iy = sy.floor();
                if !sx.is_finite()
                    || !sy.is_finite()
                    || ix < 1.0
                    || iy < 1.0
                    || ix + 2.0 >= w as f32
                    || iy + 2.0 >= h as f32
                {
                    continue;
                }
                let fx = sx - ix;
                let fy = sy - iy;
                let (ix, iy) = (ix as usize, iy as usize);
                let wx = cubic(fx);
                let wy = cubic(fy);
                let p = y * w + x;
                let mut valid = true;
                let mut maximum = f32::NEG_INFINITY;
                for (j, &wy) in wy.iter().enumerate() {
                    for (i, &wx) in wx.iter().enumerate() {
                        let q = (iy + j - 1) * w + ix + i - 1;
                        valid &= planes.valid[q] != 0 && planes.d_max[q].is_finite();
                        maximum = maximum.max(planes.d_max[q]);
                        for channel in 0..3 {
                            out.x[p * 3 + channel] += planes.x[q * 3 + channel] * wx * wy;
                        }
                    }
                }
                for (j, wy) in [1.0 - fy, fy].into_iter().enumerate() {
                    for (i, wx) in [1.0 - fx, fx].into_iter().enumerate() {
                        let q = (iy + j) * w + ix + i;
                        for channel in 0..2 {
                            out.var[p * 2 + channel] += planes.var[q * 2 + channel] * wx * wy;
                        }
                    }
                }
                out.d_max[p] = maximum;
                out.valid[p] = u8::from(valid);
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};

    fn texture(x: f32, y: f32) -> f32 {
        let mut value = 0.0;
        for i in 1..=16 {
            let f = i as f32;
            value += ((x * (0.013 * f + 0.021) + y * (0.017 * f + 0.013)).sin()
                + (x * (0.027 * f + 0.019) - y * (0.011 * f + 0.037)).cos())
                / (f + 2.0);
        }
        0.2 * (value * 0.35).exp()
    }

    fn synthetic(w: usize, h: usize, transform: [f32; 6]) -> FramePlanes {
        let mut x = Vec::with_capacity(w * h * 3);
        let [a, b, tx, c, d, ty] = transform;
        for y in 0..h {
            for col in 0..w {
                let (u, v) = (
                    a * col as f32 + b * y as f32 + tx,
                    c * col as f32 + d * y as f32 + ty,
                );
                x.extend([texture(u, v); 3]);
            }
        }
        FramePlanes {
            width: w as u32,
            height: h as u32,
            x,
            var: vec![0.01; w * h * 2],
            d_max: vec![0.3; w * h],
            valid: vec![1; w * h],
        }
    }

    #[test]
    fn fractional_shifts_and_search_limit() {
        let reference = luma(
            &synthetic(768, 768, [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
            0.85,
            Control::none(),
        )
        .unwrap();
        for dx in [0.3, 1.7, 12.4, -0.3, -1.7, -12.4, 63.0] {
            let dy = if dx == 63.0 { -61.0 } else { -dx };
            let frame = luma(
                &synthetic(768, 768, [1.0, 0.0, dx, 0.0, 1.0, dy]),
                0.85,
                Control::none(),
            )
            .unwrap();
            let found = estimate(
                &reference,
                &frame,
                &AlignOptions::default(),
                Control::none(),
            )
            .unwrap();
            eprintln!("shift ({dx},{dy}): {found:?}");
            assert!(!found.skipped);
            assert!(
                found.similarity.is_none(),
                "translation must not create a false rotation"
            );
            assert!((found.dx - dx).abs() < 0.1 && (found.dy - dy).abs() < 0.1);
        }
    }

    #[test]
    fn shifted_clipped_edge_and_border_keep_subpixel_accuracy() {
        let make = |dx: f32, dy: f32| {
            let mut p = synthetic(768, 768, [1.0, 0.0, dx, 0.0, 1.0, dy]);
            for y in 0..768 {
                for x in 0..768 {
                    let u = x as f32 + dx;
                    let v = y as f32 + dy;
                    let i = y * 768 + x;
                    if u >= 580.0 {
                        p.d_max[i] = 1.0;
                        p.x[i * 3..i * 3 + 3].fill(1.0);
                    }
                    if !(0.0..768.0).contains(&u) || !(0.0..768.0).contains(&v) {
                        p.valid[i] = 0;
                    }
                }
            }
            p
        };
        let reference = luma(&make(0.0, 0.0), 0.85, Control::none()).unwrap();
        for (dx, dy) in [(12.4, -1.7), (-12.4, 0.3)] {
            let p = make(dx, dy);
            let map = luma(&p, 0.85, Control::none()).unwrap();
            let found =
                estimate(&reference, &map, &AlignOptions::default(), Control::none()).unwrap();
            eprintln!("clipped edge ({dx},{dy}): {found:?}");
            assert!(!found.skipped);
            assert!((found.dx - dx).abs() < 0.1 && (found.dy - dy).abs() < 0.1);
            let warped = warp(p, &found, Control::none()).unwrap();
            for y in 0..768 {
                for x in 0..768 {
                    let i = y * 768 + x;
                    if warped.valid[i] != 0 && warped.d_max[i] < 0.85 {
                        assert!(x < 580);
                    }
                }
            }
        }
    }

    #[test]
    fn rotation_fit_recovers_corners() {
        let size = 1536;
        let angle = 0.2f32.to_radians();
        let (b, a) = angle.sin_cos();
        let centre = (size - 1) as f32 * 0.5;
        let truth = [a, -b, centre * (1.0 - a + b), b, a, centre * (1.0 - a - b)];
        let reference = luma(
            &synthetic(size, size, [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
            0.85,
            Control::none(),
        )
        .unwrap();
        let planes = synthetic(size, size, truth);
        let frame = luma(&planes, 0.85, Control::none()).unwrap();
        let found = estimate(
            &reference,
            &frame,
            &AlignOptions::default(),
            Control::none(),
        )
        .unwrap();
        eprintln!("rotation: {found:?}");
        let fit = found
            .similarity
            .expect("rotation must trigger a similarity fit");
        for x in [0.0, (size - 1) as f32] {
            for y in [0.0, (size - 1) as f32] {
                let dx = (fit[0] - truth[0]) * x + (fit[1] - truth[1]) * y + fit[2] - truth[2];
                let dy = (fit[3] - truth[3]) * x + (fit[4] - truth[4]) * y + fit[5] - truth[5];
                assert!(dx.hypot(dy) < 0.3, "corner ({x},{y}): {dx}, {dy}");
            }
        }
        let warped = warp(planes, &found, Control::none()).unwrap();
        let map = luma(&warped, 0.85, Control::none()).unwrap();
        let residual =
            estimate(&reference, &map, &AlignOptions::default(), Control::none()).unwrap();
        assert!(
            !residual.skipped && residual.dx.hypot(residual.dy) < 0.3,
            "{residual:?}"
        );
        let rejected = estimate(
            &reference,
            &frame,
            &AlignOptions {
                max_rotation_deg: 0.1,
                ..Default::default()
            },
            Control::none(),
        )
        .unwrap();
        assert!(!rejected.skipped && rejected.similarity.is_none());
    }

    #[test]
    fn noisy_fractional_shift_and_warp_remain_accurate() {
        let mut reference = synthetic(1024, 768, [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
        let mut frame = synthetic(1024, 768, [1.0, 0.0, 0.3, 0.0, 1.0, -0.3]);
        for (planes, mut seed) in [(&mut reference, 17u32), (&mut frame, 53u32)] {
            for rgb in planes.x.as_chunks_mut::<3>().0 {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                let noise = ((seed >> 8) as f32 / 16777216.0 - 0.5) * 0.02;
                for value in rgb {
                    *value += noise;
                }
            }
        }
        let reference = luma(&reference, 0.85, Control::none()).unwrap();
        let map = luma(&frame, 0.85, Control::none()).unwrap();
        let found = estimate(&reference, &map, &AlignOptions::default(), Control::none()).unwrap();
        assert!(
            !found.skipped && (found.dx - 0.3).abs() < 0.1 && (found.dy + 0.3).abs() < 0.1,
            "{found:?}"
        );
        let warped = warp(frame, &found, Control::none()).unwrap();
        let map = luma(&warped, 0.85, Control::none()).unwrap();
        let residual =
            estimate(&reference, &map, &AlignOptions::default(), Control::none()).unwrap();
        assert!(
            !residual.skipped && residual.dx.hypot(residual.dy) < 0.3,
            "{residual:?}"
        );
    }

    #[test]
    fn excessive_scale_falls_back_to_translation() {
        let size = 1536;
        let reference = luma(
            &synthetic(size, size, [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
            0.85,
            Control::none(),
        )
        .unwrap();
        let frame = luma(
            &synthetic(size, size, [1.008, 0.0, -6.0, 0.0, 1.008, -6.0]),
            0.85,
            Control::none(),
        )
        .unwrap();
        let found = estimate(
            &reference,
            &frame,
            &AlignOptions::default(),
            Control::none(),
        )
        .unwrap();
        assert!(!found.skipped && found.similarity.is_none(), "{found:?}");
        let allowed = estimate(
            &reference,
            &frame,
            &AlignOptions {
                max_scale_deviation: 0.01,
                ..Default::default()
            },
            Control::none(),
        )
        .unwrap();
        let matrix = allowed
            .similarity
            .expect("relaxed scale bound must allow the fit");
        assert!((matrix[0].hypot(matrix[3]) - 1.008).abs() < 0.001);
    }

    #[test]
    fn similarity_fit_recovers_scale_and_translation() {
        let truth = [1.002, -0.003, 12.4, 0.003, 1.002, -1.7];
        let [a, b, tx, c, d, ty] = truth;
        let tiles: Vec<_> = [(0.0, 0.0), (1000.0, 0.0), (0.0, 1000.0), (1000.0, 1000.0)]
            .into_iter()
            .map(|(x, y)| {
                let rx = a * x + b * y + tx;
                let ry = c * x + d * y + ty;
                Tile {
                    x: rx,
                    y: ry,
                    dx: rx - x,
                    dy: ry - y,
                    peak: 1.0,
                }
            })
            .collect();
        let fit = similarity(&tiles).unwrap();
        for (actual, expected) in fit.into_iter().zip(truth) {
            assert!((actual - expected).abs() < 0.0001);
        }
        assert!(similarity(&[tiles[0]; 3]).is_none());
    }

    #[test]
    fn warp_masks_all_taps_and_interpolates_signal_and_variance() {
        let mut p = synthetic(24, 20, [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
        for y in 0..20 {
            for x in 0..24 {
                let i = y * 24 + x;
                p.x[i * 3..i * 3 + 3].fill(x as f32 + 2.0 * y as f32);
                p.var[i * 2..i * 2 + 2].fill(2.0 * x as f32 + y as f32);
                p.d_max[i] = i as f32 / 480.0;
                p.valid[i] = u8::from(x < 13);
            }
        }
        for (dx, dy) in [(0.3, -1.7), (-12.4, 0.3), (1.0, 0.0)] {
            let result = warp(
                p.clone(),
                &Alignment {
                    dx,
                    dy,
                    ..Default::default()
                },
                Control::none(),
            )
            .unwrap();
            for y in 0..20 {
                for x in 0..24 {
                    let (sx, sy) = (x as f32 - dx, y as f32 - dy);
                    let (ix, iy) = (sx.floor() as i32, sy.floor() as i32);
                    let i = y * 24 + x;
                    let inside = ix >= 1 && iy >= 1 && ix + 2 < 24 && iy + 2 < 20;
                    assert_eq!(result.valid[i] != 0, inside && ix + 2 < 13);
                    if inside {
                        assert!(
                            (result.d_max[i] - p.d_max[(iy as usize + 2) * 24 + ix as usize + 2])
                                .abs()
                                < 1e-7
                        );
                        assert!((result.x[i * 3] - (sx + 2.0 * sy)).abs() < 1e-4);
                        assert!((result.var[i * 2] - (2.0 * sx + sy)).abs() < 1e-4);
                    }
                }
            }
        }
    }

    #[test]
    fn nonfinite_neighbours_keep_the_refinement_centre() {
        assert_eq!(
            refinement_offset([
                0.0,
                f32::NEG_INFINITY,
                0.0,
                f32::NEG_INFINITY,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0
            ]),
            [0.0, 0.0]
        );
    }

    #[test]
    fn masked_refinement_region_preserves_the_phase_estimate() {
        let mut map = Luma {
            width: 128,
            height: 128,
            data: vec![0.0; 128 * 128],
            mask: vec![0; 128 * 128],
            log: Vec::new(),
        };
        for y in 0..128 {
            for x in 0..128 {
                if !(30..100).contains(&x) || !(30..100).contains(&y) {
                    map.mask[y * 128 + x] = 1;
                    map.data[y * 128 + x] = ((37 * x + 71 * y) % 251) as f32 / 251.0;
                }
            }
        }
        assert_eq!(
            tile_score(&map, &map, [0, 0], 128, [0.0, 0.0]),
            f32::NEG_INFINITY
        );
        let found = estimate(&map, &map, &AlignOptions::default(), Control::none()).unwrap();
        assert!(!found.skipped && found.is_identity(), "{found:?}");
        let insufficient = estimate(
            &map,
            &map,
            &AlignOptions {
                tile_grid: 1,
                ..Default::default()
            },
            Control::none(),
        )
        .unwrap();
        assert!(insufficient.skipped && insufficient.peak > 0.05);
    }

    #[test]
    fn masks_black_and_clipped_pixels_and_skips_flat_maps() {
        let mut p = synthetic(192, 192, [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
        p.x.fill(0.2);
        for y in 0..192 {
            for x in 0..192 {
                let i = y * 192 + x;
                if x < 32 {
                    p.x[i * 3..i * 3 + 3].fill(0.0009);
                }
                if x >= 160 {
                    p.d_max[i] = 0.9;
                }
            }
        }
        let map = luma(&p, 0.85, Control::none()).unwrap();
        for y in 0..96 {
            for x in 0..96 {
                if x <= 16 || x >= 79 {
                    assert_eq!(map.mask[y * 96 + x], 0);
                }
            }
        }
        let found = estimate(
            &map,
            &map,
            &AlignOptions {
                max_shift_px: 16,
                ..Default::default()
            },
            Control::none(),
        )
        .unwrap();
        assert!(found.skipped);
        assert_eq!(found.peak, 0.0);
    }

    #[test]
    fn identity_preserves_bits_and_allocations() {
        let p = synthetic(17, 19, [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
        let expected = p.clone();
        let pointers = (
            p.x.as_ptr(),
            p.var.as_ptr(),
            p.d_max.as_ptr(),
            p.valid.as_ptr(),
        );
        let result = warp(p, &Alignment::default(), Control::none()).unwrap();
        assert_eq!(result, expected);
        assert_eq!(
            pointers,
            (
                result.x.as_ptr(),
                result.var.as_ptr(),
                result.d_max.as_ptr(),
                result.valid.as_ptr()
            )
        );
    }

    #[test]
    fn cancellation_discards_a_warp_in_progress() {
        let p = FramePlanes {
            width: 2048,
            height: 2048,
            x: vec![0.2; 2048 * 2048 * 3],
            var: vec![0.01; 2048 * 2048 * 2],
            d_max: vec![0.2; 2048 * 2048],
            valid: vec![1; 2048 * 2048],
        };
        let cancel = AtomicBool::new(false);
        std::thread::scope(|scope| {
            scope.spawn(|| {
                std::thread::sleep(std::time::Duration::from_millis(10));
                cancel.store(true, Ordering::Relaxed);
            });
            assert!(matches!(
                warp(
                    p,
                    &Alignment {
                        dx: 0.3,
                        ..Default::default()
                    },
                    Control::new(&cancel)
                ),
                Err(Error::Cancelled)
            ));
        });
    }

    #[test]
    fn invalid_inputs_and_cancellation() {
        let cancel = AtomicBool::new(true);
        assert!(matches!(
            luma(&FramePlanes::default(), 0.85, Control::new(&cancel)),
            Err(Error::Cancelled)
        ));
        assert!(matches!(
            estimate(
                &Luma::default(),
                &Luma::default(),
                &AlignOptions::default(),
                Control::new(&cancel)
            ),
            Err(Error::Cancelled)
        ));
        assert!(matches!(
            warp(
                FramePlanes::default(),
                &Alignment::default(),
                Control::new(&cancel)
            ),
            Err(Error::Cancelled)
        ));
        cancel.store(false, Ordering::Relaxed);
        assert!(luma(&FramePlanes::default(), 0.85, Control::none()).is_err());
        let p = synthetic(16, 16, [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
        assert!(warp(
            p,
            &Alignment {
                dx: f32::NAN,
                ..Default::default()
            },
            Control::none()
        )
        .is_err());
        let map = Luma {
            width: 64,
            height: 64,
            ..Default::default()
        };
        assert!(estimate(&map, &map, &AlignOptions::default(), Control::none()).is_err());
    }

    fn corpus_planes(
        container: &crate::X3iContainer,
        index: usize,
        ceiling: [f32; 3],
    ) -> FramePlanes {
        let options = crate::ProcessOptions {
            denoise_intensity: 0,
            ..Default::default()
        };
        let frame = container
            .open_frame(index)
            .unwrap()
            .merge_frame(&options, Control::none())
            .unwrap();
        let scale = container.frame_exposure_us(0).unwrap() as f32 / frame.shutter_us as f32;
        let w = frame.width as usize;
        let h = frame.height as usize;
        let mut d_max = vec![0.0f32; w * h];
        for y in 0..h {
            for x in 0..w {
                let half = ((y / 2) * frame.raw_half_width as usize + x / 2) * 2;
                let values = [
                    frame.raw_half[half],
                    frame.raw_half[half + 1],
                    frame.raw_top[y * w + x],
                ];
                for c in 0..3 {
                    let d = (values[c] as f64 - frame.raw_black[c])
                        / (frame.raw_white[c] - frame.raw_black[c])
                        / ceiling[c] as f64;
                    d_max[y * w + x] = d_max[y * w + x].max(d as f32);
                }
            }
        }
        // Cover expansion taps in this fixture without calling the merge's private builder.
        let mut horizontal = vec![0.0f32; w * h];
        for y in 0..h {
            for x in 0..w {
                horizontal[y * w + x] = d_max
                    [y * w + x.saturating_sub(4)..=y * w + (x + 4).min(w - 1)]
                    .iter()
                    .copied()
                    .fold(0.0, f32::max);
            }
        }
        for y in 0..h {
            for x in 0..w {
                d_max[y * w + x] = (y.saturating_sub(4)..=(y + 4).min(h - 1))
                    .map(|sy| horizontal[sy * w + x])
                    .fold(0.0, f32::max);
            }
        }
        let mut p = FramePlanes {
            width: frame.width,
            height: frame.height,
            x: frame.bmt,
            var: vec![1e-4; w * h * 2],
            d_max,
            valid: vec![1; w * h],
        };
        for value in &mut p.x {
            *value *= scale;
        }
        for (valid, d) in p.valid.iter_mut().zip(&p.d_max) {
            *valid = u8::from(*d < 0.98);
        }
        p
    }

    #[test]
    fn corpus_shifts_and_warped_residuals() {
        let root = std::env::var_os("X3F_TEST_FILES")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| {
                std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../x3f_test_files")
            });
        for name in ["DP0Q0002.X3I", "_DQH0978.X3I"] {
            let path = root.join(name);
            if !path.is_file() {
                eprintln!("skip: {name} is absent");
                continue;
            }
            let container = crate::X3iContainer::open(path).unwrap();
            let ceiling = crate::x3i_merge::MergeOptions::default()
                .saturation_for_container(&container)
                .unwrap();
            let reference = luma(
                &corpus_planes(&container, 0, ceiling),
                0.85,
                Control::none(),
            )
            .unwrap();
            let mut previous_drift = 0.0;
            let mut residuals = Vec::new();
            for index in 1..container.frames().len() {
                let p = corpus_planes(&container, index, ceiling);
                let map = luma(&p, 0.85, Control::none()).unwrap();
                let found =
                    estimate(&reference, &map, &AlignOptions::default(), Control::none()).unwrap();
                eprintln!("{name} frame {index}: {found:?}");
                assert!(!found.skipped, "{name} frame {index}");
                assert!(found.dx.abs() < 64.0 && found.dy.abs() < 64.0);
                let drift = found.dx.hypot(found.dy);
                assert!(
                    drift > previous_drift,
                    "{name} frame {index}: drift {drift} <= {previous_drift}"
                );
                previous_drift = drift;
                let matrix = found
                    .similarity
                    .unwrap_or([1.0, 0.0, found.dx, 0.0, 1.0, found.dy]);
                eprintln!(
                    "{name} frame {index}: angle={} deg scale={} max_tile_residual={} px",
                    matrix[3].atan2(matrix[0]).to_degrees(),
                    matrix[0].hypot(matrix[3]),
                    found.max_tile_residual
                );
                {
                    drop(map);
                    let warped = warp(p, &found, Control::none()).unwrap();
                    let map = luma(&warped, 0.85, Control::none()).unwrap();
                    let residual =
                        estimate(&reference, &map, &AlignOptions::default(), Control::none())
                            .unwrap();
                    eprintln!("{name} frame {index} residual: {residual:?}");
                    let uncertainty = found.tile_spread / (found.retained_tiles as f32).sqrt();
                    let residual_uncertainty =
                        residual.tile_spread / (residual.retained_tiles as f32).sqrt();
                    // Noisy tiles limit the check: use retained-tile spread / sqrt(count).
                    // This checks self-consistency; known synthetic transforms check accuracy.
                    let gate = 0.5f32.max(3.0 * residual_uncertainty);
                    let norm = residual.dx.hypot(residual.dy);
                    assert!(norm < 1.5, "{name} frame {index}: residual {norm} px");
                    eprintln!("{name} frame {index}: uncertainty={uncertainty} px residual_uncertainty={residual_uncertainty} px residual_norm={norm} px gate={gate} px");
                    residuals.push((index, residual.skipped, norm, gate));
                }
            }
            assert!(
                residuals
                    .iter()
                    .all(|&(_, skipped, norm, gate)| !skipped && norm < gate),
                "{name}: {residuals:?}"
            );
        }
    }

    #[test]
    #[ignore = "6192 by 4128 memory and timing measurement; run in release mode"]
    fn measure_full_frame() {
        let (w, h) = (6192, 4128);
        let n = w * h;
        let mut p = FramePlanes {
            width: w as u32,
            height: h as u32,
            x: vec![0.2; n * 3],
            var: vec![0.001; n * 2],
            d_max: vec![0.4; n],
            valid: vec![1; n],
        };
        for y in 0..h {
            for x in 0..w {
                let v = 0.2
                    + 0.07 * (x as f32 * 0.17 + y as f32 * 0.21).sin()
                    + 0.06 * (x as f32 * 0.11 - y as f32 * 0.07).cos();
                p.x[(y * w + x) * 3..(y * w + x) * 3 + 3].fill(v);
            }
        }
        let rss = || {
            // SAFETY: getrusage writes into a valid local structure.
            unsafe {
                let mut usage: libc::rusage = std::mem::zeroed();
                assert_eq!(libc::getrusage(libc::RUSAGE_SELF, &mut usage), 0);
                usage.ru_maxrss
            }
        };
        let mode = std::env::var("X3I_ALIGN_MEASURE").unwrap_or_else(|_| "warp".into());
        if mode == "estimate" {
            let reference = luma(&p, 0.85, Control::none()).unwrap();
            let frame = reference.clone();
            let before = rss();
            let start = std::time::Instant::now();
            let found = estimate(
                &reference,
                &frame,
                &AlignOptions::default(),
                Control::none(),
            )
            .unwrap();
            eprintln!("estimate: time={:?}, peak_rss_before={before}, peak_rss_after={}, result={found:?}", start.elapsed(), rss());
            assert!(!found.skipped && found.dx.hypot(found.dy) < 0.1);
            std::hint::black_box(p);
        } else {
            let before = rss();
            let start = std::time::Instant::now();
            let p = warp(
                p,
                &Alignment {
                    dx: 36.3,
                    dy: -16.7,
                    ..Default::default()
                },
                Control::none(),
            )
            .unwrap();
            let invalid = p.valid.iter().filter(|&&v| v == 0).count();
            eprintln!("warp: time={:?}, peak_rss_before={before}, peak_rss_after={}, invalid={invalid}/{n} ({:.6}%)", start.elapsed(), rss(), invalid as f64 * 100.0 / n as f64);
            assert!(invalid > 0 && invalid < n / 50);
            std::hint::black_box(p);
        }
    }
}
