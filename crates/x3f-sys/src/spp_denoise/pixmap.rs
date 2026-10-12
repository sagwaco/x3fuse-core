//! Native SPP pixmap kernels: `FIR` (0x71b48), `LocalMean` (0x72050),
//! `BilateralFilterRGB` (0x72c9c) and `GenerateKernelMap` (0x79928).
//!
//! Sizes are `[columns, rows]`. Strides count f32 samples, or u32 entries.
//! Each image slice must hold `(rows - 1) * stride + columns * channels` samples.
//! Invalid layouts and parameters return an error before any write.

use crate::{Error, Result};

fn plane_len(size: [usize; 2], stride: usize, channels: usize) -> Option<usize> {
    let [cols, rows] = size;
    let row = cols.checked_mul(channels)?;
    if cols == 0 || rows == 0 || stride < row {
        return None;
    }
    (rows - 1).checked_mul(stride)?.checked_add(row)
}

fn plane_ok(len: usize, size: [usize; 2], stride: usize, channels: usize) -> bool {
    plane_len(size, stride, channels).is_some_and(|needed| needed <= len)
}

fn invalid() -> Error {
    Error::InvalidData("invalid SPP pixmap kernel layout")
}

/// Run the native `FIR` recursive filter along each source row.
///
/// For each row `y` in `[border, rows - border)`, a forward pass and a backward
/// pass run over columns `[border, columns - border)`. The weight at column `x`
/// couples `x` to its left neighbor in the forward pass and to its right
/// neighbor in the backward pass. Each pass also accumulates its own gain.
/// The destination is transposed: `size` is `[columns, rows]` for the source and
/// the weights, and `[rows, columns]` for `destination` and `transposed_weights`.
/// Destination pixel `(y, x)` gets the normalized sum of both passes.
/// `transposed_weights`, when present, receives the used weights at `(y, x)`.
///
/// Only rows `[border, columns - border)` and columns `[border, rows - border)`
/// of the transposed outputs change. Columns must be at least `2 * border + 2`
/// and rows at least `2 * border + 1`. Smaller widths make the native read
/// uninitialized scratch.
#[allow(clippy::too_many_arguments)]
pub fn fir(
    source: &[f32],
    source_stride: usize,
    weights: &[f32],
    weights_stride: usize,
    size: [usize; 2],
    border: usize,
    destination: &mut [f32],
    destination_stride: usize,
    transposed_weights: Option<(&mut [f32], usize)>,
) -> Result<()> {
    let [width, height] = size;
    let transposed = [height, width];
    if !border
        .checked_mul(2)
        .is_some_and(|b| width >= b + 2 && height > b)
        || !plane_ok(source.len(), size, source_stride, 3)
        || !plane_ok(weights.len(), size, weights_stride, 1)
        || !plane_ok(destination.len(), transposed, destination_stride, 3)
        || transposed_weights
            .as_ref()
            .is_some_and(|(data, stride)| !plane_ok(data.len(), transposed, *stride, 1))
    {
        return Err(invalid());
    }
    fir_unchecked(
        source,
        source_stride,
        weights,
        weights_stride,
        size,
        border,
        destination,
        destination_stride,
        transposed_weights,
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn fir_unchecked(
    source: &[f32],
    source_stride: usize,
    weights: &[f32],
    weights_stride: usize,
    [width, height]: [usize; 2],
    border: usize,
    destination: &mut [f32],
    destination_stride: usize,
    mut transposed_weights: Option<(&mut [f32], usize)>,
) {
    let (lo, hi) = (border, width - border);
    let mut forward = vec![0.0_f32; 3 * width];
    let mut backward = vec![0.0_f32; 3 * width];
    let mut forward_gain = vec![0.0_f32; width];
    let mut backward_gain = vec![0.0_f32; width];
    for y in border..height - border {
        let src = &source[y * source_stride..];
        let w = &weights[y * weights_stride..];
        forward[3 * lo..3 * lo + 3].copy_from_slice(&src[3 * lo..3 * lo + 3]);
        backward[3 * (hi - 1)..3 * hi].copy_from_slice(&src[3 * (hi - 1)..3 * hi]);
        forward_gain[lo] = 1.0;
        backward_gain[hi - 1] = 1.0;
        if let Some((data, stride)) = transposed_weights.as_mut() {
            for x in lo..hi {
                data[x * *stride + y] = w[x];
            }
        }
        // The native multiplies and adds in separate f32 steps.
        for k in 0..hi - lo - 1 {
            let p = lo + 1 + k;
            for c in 0..3 {
                forward[3 * p + c] = w[p] * forward[3 * (p - 1) + c] + src[3 * p + c];
            }
            forward_gain[p] = w[p] * forward_gain[p - 1] + 1.0;
            let q = hi - 2 - k;
            for c in 0..3 {
                backward[3 * q + c] = w[q] * backward[3 * (q + 1) + c] + src[3 * q + c];
            }
            backward_gain[q] = w[q] * backward_gain[q + 1] + 1.0;
        }
        for x in lo..hi {
            let out = &mut destination[x * destination_stride + 3 * y..][..3];
            for c in 0..3 {
                let i = 3 * x + c;
                out[c] = if x == lo {
                    backward[i] / backward_gain[x]
                } else if x == hi - 1 {
                    forward[i] / forward_gain[x]
                } else {
                    ((forward[i] + backward[i]) - src[i])
                        / ((forward_gain[x] + backward_gain[x]) + -1.0)
                };
            }
        }
    }
}

/// Run the native `LocalMean`: `FIR` along rows, then `FIR` along columns.
///
/// The first pass writes transposed scratch images. The second pass reads them
/// and writes the destination in source orientation. Only rows
/// `[border, rows - border)` and columns `[border, columns - border)` change.
/// Both dimensions must be at least `2 * border + 2`.
///
/// The native caller supplies the two scratch pixmaps and the second pass reads
/// their dimensions. This port assumes transposed scratch dimensions and owns
/// the scratch storage. The result does not depend on the scratch contents.
#[allow(clippy::too_many_arguments)]
pub fn local_mean(
    source: &[f32],
    source_stride: usize,
    weights: &[f32],
    weights_stride: usize,
    size: [usize; 2],
    border: usize,
    destination: &mut [f32],
    destination_stride: usize,
) -> Result<()> {
    let [width, height] = size;
    if !border
        .checked_mul(2)
        .is_some_and(|b| width >= b + 2 && height >= b + 2)
        || !plane_ok(source.len(), size, source_stride, 3)
        || !plane_ok(weights.len(), size, weights_stride, 1)
        || !plane_ok(destination.len(), size, destination_stride, 3)
    {
        return Err(invalid());
    }
    let transposed = [height, width];
    let mut scratch = vec![0.0_f32; 3 * width * height];
    let mut scratch_weights = vec![0.0_f32; width * height];
    fir_unchecked(
        source,
        source_stride,
        weights,
        weights_stride,
        size,
        border,
        &mut scratch,
        3 * height,
        Some((&mut scratch_weights, height)),
    );
    fir_unchecked(
        &scratch,
        3 * height,
        &scratch_weights,
        height,
        transposed,
        border,
        destination,
        destination_stride,
        None,
    );
    Ok(())
}

/// Parameters for the native `BilateralFilterRGB`.
#[derive(Clone, Copy, Debug)]
pub struct BilateralRgbParams<'a> {
    /// Filter size. Half of it, rounded down, joins `border` in the margin.
    pub filter_size: u32,
    /// Kernel map tile size in pixels.
    pub kernel_step: u32,
    /// Noise variance per channel: constant, linear and quadratic terms.
    /// The variance is `(|v| * linear + constant) + (quadratic * v) * v` in f64.
    pub noise_model: [[f64; 3]; 3],
    /// Use every second column in windows with a radius larger than 3.
    pub subsample: bool,
    /// Extra margin at each image edge.
    pub border: u32,
    /// The native global `gExpLUT`: 2048 weights.
    pub exp_lut: &'a [f32; 2048],
    /// The float after `gExpLUT`. It divides each channel variance.
    pub lut_scale: f32,
}

// Walk the processed pixels with the native kernel map counters. The radius
// state is not reset at the start of a row.
fn walk_kernel_map(
    width: u32,
    height: u32,
    map: &[u32],
    params: &BilateralRgbParams,
    mut pixel: impl FnMut(u32, u32, u32) -> Result<()>,
) -> Result<()> {
    let step = params.kernel_step;
    let start = params.filter_size / 2 + params.border;
    let per_row = width.div_ceil(step) as usize;
    let lookup = |i: usize| map.get(i).copied().ok_or_else(invalid);
    let mut radius = lookup(0)?;
    let (mut row_count, mut row_base, mut tile_row) = (step.wrapping_sub(start), 0, 0);
    for y in start..height - start {
        row_count = row_count.wrapping_sub(1);
        if row_count == 0 {
            tile_row += 1;
            row_base = tile_row * per_row;
            radius = lookup(row_base)?;
            row_count = step;
        }
        let (mut col_count, mut tile_col) = (step.wrapping_sub(start), 0);
        for x in start..width - start - 1 {
            col_count = col_count.wrapping_sub(1);
            if col_count == 0 {
                tile_col += 1;
                radius = lookup(row_base + tile_col)?;
                col_count = step;
            }
            pixel(x, y, radius)?;
        }
    }
    Ok(())
}

// The shared per-pixel arithmetic of both native BilateralFilterRGB overloads.
struct Window<'a> {
    source: &'a [f32],
    stride: usize,
    height: u32,
    model: &'a [[f64; 3]; 3],
    scale: f64,
    lut: &'a [f32; 2048],
}

impl Window<'_> {
    // Filter one pixel with radius `r`. `skip` is 1 when every second column
    // is used. Rows run from y - r to min(y + r, height - 1); a window that
    // starts above row 0 keeps the source pixel. The half-width alternates
    // r and r - skip per row. Callers check every read before the first write.
    fn filter(&self, x: u32, y: u32, r: u32, skip: u32) -> [f32; 3] {
        let (source, stride, model) = (self.source, self.stride, self.model);
        let center = y as usize * stride + 3 * x as usize;
        let p: [f32; 3] = source[center..center + 3].try_into().unwrap();
        let first_row = y.wrapping_sub(r);
        let last_row = (y + r).min(self.height - 1);
        if first_row > last_row {
            return p;
        }
        let pd = p.map(f64::from);
        let inv: [f64; 3] = std::array::from_fn(|c| {
            self.scale / ((pd[c].abs() * model[1][c] + model[0][c]) + (model[2][c] * pd[c]) * pd[c])
        });
        // With r = 0 and skip = 1 the span wraps, like the native u32, and
        // every second row is empty.
        let span = (2 * r).wrapping_sub(skip);
        let mut half = r;
        let (mut sum, mut acc) = (0.0_f32, [0.0_f32; 3]);
        for yy in first_row..=last_row {
            let (left, right) = (x.wrapping_sub(half), x.wrapping_add(half));
            let mut xx = left;
            while xx <= right {
                let i = yy as usize * stride + 3 * xx as usize;
                let q: [f32; 3] = source[i..i + 3].try_into().unwrap();
                let d: [f64; 3] = std::array::from_fn(|c| f64::from(q[c]) - pd[c]);
                let e = (d[0] * d[0]).mul_add(inv[0], inv[1] * (d[1] * d[1]));
                let e = (d[2] * d[2]).mul_add(inv[2], e);
                // Round to an integer with the 1.5 * 2^52 bias. Bits 11 to 31
                // must be clear. The native ignores the upper word.
                let bits = ((e + f64::from_bits(0x3e50_1b2b_29a4_692b)) + 6_755_399_441_055_744.0)
                    .to_bits();
                if bits & 0xffff_f800 == 0 {
                    let weight = self.lut[(bits & 0x7ff) as usize];
                    sum += weight;
                    for c in 0..3 {
                        acc[c] = q[c].mul_add(weight, acc[c]);
                    }
                }
                xx += 1 + skip;
            }
            half = span.wrapping_sub(half);
        }
        // A NaN sum also divides.
        if sum != 0.0 {
            acc.map(|a| a / sum)
        } else {
            p
        }
    }
}

/// Run the native `BilateralFilterRGB`.
///
/// The filter writes destination rows `[m, rows - m)` and columns
/// `[m, columns - m - 1)`, where `m = filter_size / 2 + border`. A pixel whose
/// window starts above row 0 keeps its source value. Windows clamp at the last
/// row but not at the right edge, so they can read row padding and later rows.
/// `source` must hold every sample that the windows read.
///
/// `kernel_map` holds radii in tiles of `kernel_step` pixels, with
/// `columns.div_ceil(kernel_step)` tiles in each map row. The native tile
/// counters start `m` pixels into the image, and the column counter does not
/// reset the radius at the start of a row. This port keeps both effects.
///
/// After filtering, the native copies source row `filter_size / 2` over the rows
/// above it and the last unfiltered row over the rows below it. It then does the
/// same with columns in every row. These copies change `source`, not
/// `destination`. A filter size below 2 skips them.
///
/// Columns must be larger than `2 * m + 1` and rows larger than `2 * m`.
/// Each used radius plus the row count must fit in an i32.
pub fn bilateral_filter_rgb(
    source: &mut [f32],
    source_stride: usize,
    destination: &mut [f32],
    destination_stride: usize,
    size: [usize; 2],
    kernel_map: &[u32],
    params: &BilateralRgbParams,
) -> Result<()> {
    let [width, height] = size;
    let start = params.filter_size / 2 + params.border;
    let (Ok(w), Ok(h)) = (u32::try_from(width), u32::try_from(height)) else {
        return Err(invalid());
    };
    if params.kernel_step == 0
        || params.filter_size / 2 > u32::MAX / 4
        || params.border > u32::MAX / 4
        || i32::try_from(height).is_err()
        || w <= 2 * start + 1
        || h <= 2 * start
        || !plane_ok(source.len(), size, source_stride, 3)
        || !plane_ok(destination.len(), size, destination_stride, 3)
    {
        return Err(invalid());
    }
    // Check every source read before the first write.
    walk_kernel_map(w, h, kernel_map, params, |x, y, r| {
        if r > i32::MAX as u32 - h {
            return Err(invalid());
        }
        if y < r {
            return Ok(());
        }
        let last_row = (y + r).min(h - 1) as usize;
        let end = (x as usize + r as usize) * 3 + 3;
        if last_row * source_stride + end > source.len() {
            return Err(invalid());
        }
        Ok(())
    })?;
    let window = Window {
        source,
        stride: source_stride,
        height: h,
        model: &params.noise_model,
        scale: f64::from(params.lut_scale),
        lut: params.exp_lut,
    };
    walk_kernel_map(w, h, kernel_map, params, |x, y, r| {
        let out = window.filter(x, y, r, u32::from(r > 3 && params.subsample));
        let o = y as usize * destination_stride + 3 * x as usize;
        destination[o..o + 3].copy_from_slice(&out);
        Ok(())
    })?;
    let half = params.filter_size as usize / 2;
    if params.filter_size >= 2 {
        let row = 3 * width;
        for y in (0..half).chain(height - half..height) {
            let from = if y < half { half } else { height - half - 1 };
            source.copy_within(
                from * source_stride..from * source_stride + row,
                y * source_stride,
            );
        }
        for y in 0..height {
            let base = y * source_stride;
            for x in (0..half).chain(width - half..width) {
                let from = base + 3 * if x < half { half } else { width - half - 1 };
                source.copy_within(from..from + 3, base + 3 * x);
            }
        }
    }
    Ok(())
}

/// Parameters for the whole-image native `BilateralFilterRGB` (0x645a8).
#[derive(Clone, Copy, Debug)]
pub struct BilateralImageParams<'a> {
    /// Filter size. The radius is `filter_size / 2` for every pixel.
    pub filter_size: u32,
    /// Noise variance per channel: constant, linear and quadratic terms.
    pub noise_model: [[f64; 3]; 3],
    /// Use every second column in every window.
    pub subsample: bool,
    /// The exp table: 2048 weights.
    pub exp_lut: &'a [f32; 2048],
    /// Divides each channel variance.
    pub lut_scale: f32,
    /// First and last requested rows, inclusive. The native clamps them to
    /// `[filter_size / 2, rows - 5]`. Disjoint ranges give the same pixels
    /// as one call over their union.
    pub rows: [u32; 2],
}

/// Run the whole-image native `BilateralFilterRGB` (0x645a8). `TiledDenoise`
/// calls it through `XDoBilateralFilter` (0x7eab0) when it precomputes the
/// full image.
///
/// With `h = filter_size / 2`, the filter writes destination rows
/// `[max(h, rows[0]), min(rows - 5, rows[1])]` and columns
/// `[h, columns - h - 1)`. Other samples stay unchanged. Each pixel uses the
/// tiled filter arithmetic with radius `h`. The source does not change, and
/// windows stay inside the image. There is no kernel map or border copy.
///
/// The image must have at least 5 rows and more than `h` columns, and
/// `h + rows` must fit in an i32.
/// Invalid layouts return an error before any write. Cancellation is checked
/// before each row, like the native cancel flag. After a cancellation, rows
/// that ran keep their output and the destination is incomplete.
/// The function does not allocate.
pub fn bilateral_filter_rgb_image(
    source: &[f32],
    source_stride: usize,
    destination: &mut [f32],
    destination_stride: usize,
    size: [usize; 2],
    params: &BilateralImageParams,
    control: crate::Control<'_>,
) -> Result<()> {
    control.check()?;
    let [width, height] = size;
    let half = params.filter_size / 2;
    let (Ok(w), Ok(h)) = (u32::try_from(width), u32::try_from(height)) else {
        return Err(invalid());
    };
    // A radius wider than the image makes the native column bound wrap.
    if h < 5
        || half >= w
        || i32::try_from(u64::from(h) + u64::from(half)).is_err()
        || i32::try_from(w).is_err()
        || params.rows.iter().any(|&r| i32::try_from(r).is_err())
        || !plane_ok(source.len(), size, source_stride, 3)
        || !plane_ok(destination.len(), size, destination_stride, 3)
    {
        return Err(invalid());
    }
    let first = params.rows[0].max(half);
    let last = params.rows[1].min(h - 5);
    if first > last || half >= w - half - 1 {
        return Ok(());
    }
    let window = Window {
        source,
        stride: source_stride,
        height: h,
        model: &params.noise_model,
        scale: f64::from(params.lut_scale),
        lut: params.exp_lut,
    };
    for y in first..=last {
        control.check()?;
        let row = &mut destination[y as usize * destination_stride..];
        for x in half..w - half - 1 {
            let out = window.filter(x, y, half, u32::from(params.subsample));
            row[3 * x as usize..3 * x as usize + 3].copy_from_slice(&out);
        }
    }
    Ok(())
}

/// Run the native `GenerateKernelMap` and return the number of map entries.
///
/// Each tile of `step` by `step` pixels averages the smallest channel of its
/// pixels. The native tile counters advance one pixel early, so tile 0 gets
/// `step - 1` columns and rows. Each tile divides by its nominal size, not its
/// pixel count. A clamped 3 by 3 minimum filter follows. The radius is 9 below
/// 150, and the other classes split at 300, 500 and 700. A threshold above 400
/// selects radii 7, 6, 5, 3. Other thresholds, including NaN, select 5, 4, 3, 2.
///
/// The map has `columns.div_ceil(step) * rows.div_ceil(step)` entries.
/// A layout whose last pixel maps past the final tile is rejected. The native
/// writes outside its allocation for such a layout. This includes every layout
/// in which `step` divides either dimension.
pub fn generate_kernel_map(
    source: &[f32],
    source_stride: usize,
    size: [usize; 2],
    step: usize,
    threshold: f32,
    kernel_map: &mut [u32],
) -> Result<usize> {
    let [width, height] = size;
    if step == 0
        || u32::try_from(width).is_err()
        || u32::try_from(height).is_err()
        || !plane_ok(source.len(), size, source_stride, 3)
    {
        return Err(invalid());
    }
    let (cols, rows) = (width.div_ceil(step), height.div_ceil(step));
    let count = cols * rows;
    if (height / step) * cols + width / step >= count || kernel_map.len() < count {
        return Err(invalid());
    }
    let mut mean = vec![0.0_f32; count];
    let (mut row_count, mut base) = (step, 0);
    for y in 0..height {
        row_count -= 1;
        if row_count == 0 {
            base += cols;
        }
        let (mut col_count, mut tile) = (step, base);
        for pixel in source[y * source_stride..][..3 * width].chunks_exact(3) {
            col_count -= 1;
            if col_count == 0 {
                col_count = step;
                tile += 1;
            }
            let low = if pixel[1] < pixel[2] {
                pixel[1]
            } else {
                pixel[2]
            };
            mean[tile] += if pixel[0] < low { pixel[0] } else { low };
        }
        if row_count == 0 {
            row_count = step;
        }
    }
    // The native forms these products in u32 before conversion.
    let step32 = step as u32;
    let last_width = (width - (cols - 1) * step) as u32;
    let last_height = (height - (rows - 1) * step) as u32;
    for (i, value) in mean.iter_mut().enumerate() {
        let tile_width = if i % cols == cols - 1 {
            last_width
        } else {
            step32
        };
        let tile_height = if i / cols == rows - 1 {
            last_height
        } else {
            step32
        };
        *value /= tile_width.wrapping_mul(tile_height) as f32;
    }
    let min3 = |a: f32, b: f32, c: f32| {
        let low = if a < b { a } else { b };
        if c < low {
            c
        } else {
            low
        }
    };
    for row in 0..rows {
        let up = row.saturating_sub(1) * cols;
        let mid = row * cols;
        let down = (row + 1).min(rows - 1) * cols;
        for col in 0..cols {
            let (left, right) = (col.saturating_sub(1), (col + 1).min(cols - 1));
            let line = |r: usize| min3(mean[r + col], mean[r + right], mean[r + left]);
            let (above, center, below) = (line(up), line(mid), line(down));
            let low = if center < below { center } else { below };
            let value = if above < low { above } else { low };
            let radii = if threshold > 400.0 {
                [7, 6, 5, 3]
            } else {
                [5, 4, 3, 2]
            };
            kernel_map[mid + col] = if value < 150.0 {
                9
            } else if value < 300.0 {
                radii[0]
            } else if value < 500.0 {
                radii[1]
            } else if value < 700.0 {
                radii[2]
            } else {
                radii[3]
            };
        }
    }
    Ok(count)
}

#[cfg(test)]
mod tests {
    use super::*;

    const GUARD: usize = 8;

    fn fnv<T: Copy>(values: &[T], bits: impl Fn(T) -> u32) -> u64 {
        let mut hash = 0xcbf2_9ce4_8422_2325_u64;
        for &value in values {
            for byte in bits(value).to_le_bytes() {
                hash ^= u64::from(byte);
                hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
        hash
    }

    fn fnv_f32(values: &[f32]) -> u64 {
        fnv(values, f32::to_bits)
    }

    // NaN sign and payload can differ between CPU architectures.
    fn fnv_canonical(values: &[f32]) -> u64 {
        fnv(
            values,
            |v| if v.is_nan() { 0x7fc0_0000 } else { v.to_bits() },
        )
    }

    fn next(state: &mut u32) -> u32 {
        *state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        *state
    }

    fn noise(s: u32) -> f32 {
        ((s >> 16) as i32 - 32768) as f32 / 4096.0
    }

    fn fields(line: &str) -> (Vec<usize>, Vec<u64>) {
        let parts: Vec<_> = line.split_whitespace().skip(1).collect();
        let split = parts.iter().position(|p| p.len() == 16).unwrap();
        (
            parts[..split].iter().map(|p| p.parse().unwrap()).collect(),
            parts[split..]
                .iter()
                .map(|p| u64::from_str_radix(p, 16).unwrap())
                .collect(),
        )
    }

    fn fill_rgb(buf: &mut [f32], stride: usize, [w, h]: [usize; 2], pattern: usize) {
        let mut state = 0x1234_5678;
        for y in 0..h {
            for x in 0..w {
                for c in 0..3 {
                    let s = next(&mut state);
                    buf[GUARD + stride * y + 3 * x + c] = match pattern {
                        0 => noise(s),
                        1 => x as f32 * 0.37 + y as f32 * 1.13 + c as f32 * 0.71,
                        2 if x == w / 2 && y == h / 2 => (16 * (c + 1)) as f32,
                        2 => 0.3,
                        _ if (x + 3 * y + 5 * c) % 7 == 0 => f32::from_bits(0x7fc1_2345),
                        _ => noise(s),
                    };
                }
            }
        }
    }

    fn fill_weights(buf: &mut [f32], stride: usize, [w, h]: [usize; 2], pattern: usize) {
        let cycle = [0.0, 1.0, 1.7, -0.45, 0.61];
        let mut state = 0x9e37_79b9;
        for y in 0..h {
            for x in 0..w {
                let s = next(&mut state);
                buf[GUARD + stride * y + x] = match pattern {
                    0 => 0.05 + 0.9 * ((s >> 8) as f32 / 16_777_216.0),
                    1 => 0.3,
                    _ => cycle[(x + 2 * y) % 5],
                };
            }
        }
    }

    // Returns the source and weight buffers with guards, after the input hash check.
    fn fir_inputs(v: &[usize], hashes: &[u64]) -> (Vec<f32>, Vec<f32>) {
        let [w, h, _, pad, sp, wp] = v[..6] else {
            panic!()
        };
        let mut source = vec![-123.25; 2 * GUARD + (3 * w + pad) * h];
        let mut weights = vec![-7.75; 2 * GUARD + (w + pad) * h];
        fill_rgb(&mut source, 3 * w + pad, [w, h], sp);
        fill_weights(&mut weights, w + pad, [w, h], wp);
        assert_eq!(fnv_f32(&source), hashes[0]);
        assert_eq!(fnv_f32(&weights), hashes[1]);
        (source, weights)
    }

    fn check_output(output: &[f32], raw: u64, canonical: u64, line: &str) {
        #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
        assert_eq!(fnv_f32(output), raw, "{line}");
        let _ = raw;
        assert_eq!(fnv_canonical(output), canonical, "{line}");
    }

    #[test]
    fn fir_matches_native_battery() {
        let fixture = include_str!("../../tests/fixtures/spp-pixmap-fir-native.txt");
        let mut cases = 0;
        for line in fixture.lines().filter(|l| l.starts_with("FIR ")) {
            let (v, hashes) = fields(line);
            let [w, h, b, pad, _, _, t] = v[..] else {
                panic!()
            };
            let (source, weights) = fir_inputs(&v, &hashes);
            let (ds, ts) = (3 * h + pad, h + pad);
            let mut destination = vec![-321.5; 2 * GUARD + ds * w];
            let mut transposed = vec![-654.25; 2 * GUARD + ts * w];
            fir(
                &source[GUARD..],
                3 * w + pad,
                &weights[GUARD..],
                w + pad,
                [w, h],
                b,
                &mut destination[GUARD..GUARD + ds * w],
                ds,
                (t == 1).then_some((&mut transposed[GUARD..GUARD + ts * w], ts)),
            )
            .unwrap();
            check_output(&destination, hashes[2], hashes[4], line);
            assert_eq!(fnv_f32(&transposed), hashes[3], "{line}");
            cases += 1;
        }
        assert_eq!(cases, 1824);
    }

    #[test]
    fn local_mean_matches_native_battery() {
        let fixture = include_str!("../../tests/fixtures/spp-pixmap-local-mean-native.txt");
        let mut cases = 0;
        for line in fixture.lines().filter(|l| l.starts_with("LMEAN ")) {
            let (v, hashes) = fields(line);
            let [w, h, b, pad, _, _] = v[..] else {
                panic!()
            };
            let (source, weights) = fir_inputs(&v, &hashes);
            let stride = 3 * w + pad;
            let mut destination = vec![-321.5; 2 * GUARD + stride * h];
            local_mean(
                &source[GUARD..],
                stride,
                &weights[GUARD..],
                w + pad,
                [w, h],
                b,
                &mut destination[GUARD..GUARD + stride * h],
                stride,
            )
            .unwrap();
            check_output(&destination, hashes[2], hashes[3], line);
            cases += 1;
        }
        assert_eq!(cases, 720);
    }

    fn exp_lut() -> [f32; 2048] {
        std::array::from_fn(|i| 1.0 / (1.0 + i as f32 * 0.0123))
    }

    // The third model gives unit variance. With scale 0.5 and integer pattern 3,
    // odd squared sums give half-integer energies. The bias selects their index.
    const MODELS: [[[f64; 3]; 3]; 3] = [
        [
            [0.8, 1.1, 0.9],
            [0.013, 0.021, 0.017],
            [1.3e-5, 2.1e-5, 0.7e-5],
        ],
        [
            [2.03, -0.5, 5.29],
            [0.12345, 0.0, 0.45678],
            [1e-4, -3e-6, 2.5e-5],
        ],
        [[1.0; 3], [0.0; 3], [0.0; 3]],
    ];

    #[test]
    fn bilateral_filter_rgb_matches_native_battery() {
        let fixture = include_str!("../../tests/fixtures/spp-pixmap-bilateral-native.txt");
        let lut = exp_lut();
        let mut cases = 0;
        for line in fixture.lines().filter(|l| l.starts_with("BILAT ")) {
            let (v, hashes) = fields(line);
            let [w, h, size, border, sub, step, model, pattern] = v[..] else {
                panic!()
            };
            let (ss, ds) = (3 * w + 3, 3 * w + 2);
            let mut source = vec![-123.25_f32; GUARD + ss * h + 48];
            let mut state = 0x3141_5926;
            for y in 0..h {
                for x in 0..w {
                    for c in 0..3 {
                        let s = next(&mut state);
                        source[GUARD + ss * y + 3 * x + c] = match pattern {
                            1 => {
                                300.0
                                    + x as f32 * 7.3
                                    + y as f32 * 3.1
                                    + c as f32 * 11.9
                                    + ((s >> 16) & 15) as f32 * 0.37
                            }
                            2 if (x + 3 * y + 5 * c) % 23 == 0 => f32::from_bits(0x7fc1_2345),
                            3 => 100.0 + ((s >> 16) & 15) as f32,
                            _ => 200.0 + (s >> 16) as f32 * 0.0113,
                        };
                    }
                }
            }
            let per_row = w.div_ceil(step);
            let mut kstate = (0x2545_f491 + w + 7 * h + 31 * step) as u32;
            let map: Vec<u32> = (0..per_row * (h.div_ceil(step) + 2) + 2)
                .map(|_| (next(&mut kstate) >> 16) % 12)
                .collect();
            assert_eq!(fnv_f32(&source), hashes[0], "{line}");
            assert_eq!(fnv(&map, |v| v), hashes[1], "{line}");
            let mut destination = vec![-321.5; 2 * GUARD + ds * h];
            let params = BilateralRgbParams {
                filter_size: size as u32,
                kernel_step: step as u32,
                noise_model: MODELS[model],
                subsample: sub == 1,
                border: border as u32,
                exp_lut: &lut,
                lut_scale: [37.3, 3.19, 0.5][model],
            };
            bilateral_filter_rgb(
                &mut source[GUARD..],
                ss,
                &mut destination[GUARD..GUARD + ds * h],
                ds,
                [w, h],
                &map,
                &params,
            )
            .unwrap();
            // NaN payloads select LUT entries, so NaN cases are checked on arm64 macOS only.
            if pattern != 2 || cfg!(all(target_arch = "aarch64", target_os = "macos")) {
                assert_eq!(fnv_f32(&destination), hashes[2], "{line}");
                assert_eq!(fnv_f32(&source), hashes[3], "{line}");
            }
            cases += 1;
        }
        assert_eq!(cases, 3150);
    }

    fn untiled_params(
        lut: &[f32; 2048],
        model: usize,
        size: u32,
        sub: bool,
    ) -> BilateralImageParams<'_> {
        BilateralImageParams {
            filter_size: size,
            noise_model: MODELS[model],
            subsample: sub,
            exp_lut: lut,
            lut_scale: [37.3, 3.19, 0.5][model],
            rows: [0, i32::MAX as u32],
        }
    }

    #[test]
    fn bilateral_filter_rgb_image_matches_native_battery() {
        let fixture = include_str!("../../tests/fixtures/spp-pixmap-untiled-native.txt");
        let lut = exp_lut();
        let mut cases = 0;
        for line in fixture.lines().filter(|l| l.starts_with("UNTILED ")) {
            let (v, hashes) = fields(line);
            let [w, h, size, sub, _, first, last, pad, model, pattern] = v[..] else {
                panic!()
            };
            let (ss, ds) = (3 * w + pad, 3 * w + pad + 2);
            let mut source = vec![-123.25_f32; 2 * GUARD + ss * h];
            let mut state = (0x3141_5926 + w + 7 * h) as u32;
            for y in 0..h {
                for x in 0..w {
                    for c in 0..3 {
                        let s = next(&mut state);
                        source[GUARD + ss * y + 3 * x + c] = match pattern {
                            1 => {
                                300.0
                                    + x as f32 * 7.3
                                    + y as f32 * 3.1
                                    + c as f32 * 11.9
                                    + ((s >> 16) & 15) as f32 * 0.37
                            }
                            2 if (x + 3 * y + 5 * c) % 23 == 0 => f32::from_bits(0x7fc1_2345),
                            3 => 100.0 + ((s >> 16) & 15) as f32,
                            _ => 200.0 + (s >> 16) as f32 * 0.0113,
                        };
                    }
                }
            }
            assert_eq!(fnv_f32(&source), hashes[0], "{line}");
            let mut destination = vec![-321.5; 2 * GUARD + ds * h];
            let mut params = untiled_params(&lut, model, size as u32, sub == 1);
            params.rows = [first as u32, last as u32];
            bilateral_filter_rgb_image(
                &source[GUARD..GUARD + ss * h],
                ss,
                &mut destination[GUARD..GUARD + ds * h],
                ds,
                [w, h],
                &params,
                crate::Control::none(),
            )
            .unwrap();
            // NaN payloads select LUT entries, so NaN cases are checked on arm64 macOS only.
            if pattern != 2 || cfg!(all(target_arch = "aarch64", target_os = "macos")) {
                assert_eq!(fnv_f32(&destination), hashes[1], "{line}");
            }
            cases += 1;
        }
        assert_eq!(cases, 7840);
    }

    /// The SD1 post-radial raster from the repair corpus dump. Set
    /// X3F_PIXMAP_CROP_SOURCE to its radial.f32 to run the full 2464x1632 crop.
    #[test]
    fn bilateral_filter_rgb_image_matches_native_sd1_crop() {
        let Some(path) = std::env::var_os("X3F_PIXMAP_CROP_SOURCE") else {
            eprintln!("X3F_PIXMAP_CROP_SOURCE unset: SD1 crop skipped");
            return;
        };
        let bytes = std::fs::read(path).unwrap();
        let full: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        assert_eq!(full.len(), 14784 * 3264);
        let fixture = include_str!("../../tests/fixtures/spp-pixmap-untiled-native.txt");
        let lut = exp_lut();
        let mut cases = 0;
        for line in fixture.lines().filter(|l| l.starts_with("CROP ")) {
            let (v, hashes) = fields(line);
            let [x0, y0, w, h, size, sub, _, _] = v[..] else {
                panic!()
            };
            let (ss, ds) = (3 * w, 3 * w + 4);
            let mut crop = Vec::with_capacity(ss * h);
            for y in 0..h {
                crop.extend_from_slice(&full[14784 * (y0 + y) + 3 * x0..][..ss]);
            }
            assert_eq!(fnv_f32(&crop), hashes[0], "{line}");
            let params = BilateralImageParams {
                filter_size: size as u32,
                noise_model: [
                    [4.7, 3.1, 5.9],
                    [0.0213, 0.0117, 0.0291],
                    [2.3e-6, 1.1e-6, 3.7e-6],
                ],
                subsample: sub == 1,
                exp_lut: &lut,
                lut_scale: 255.875,
                rows: [0, i32::MAX as u32],
            };
            let mut destination = vec![-321.5; ds * h];
            bilateral_filter_rgb_image(
                &crop,
                ss,
                &mut destination,
                ds,
                [w, h],
                &params,
                crate::Control::none(),
            )
            .unwrap();
            assert_eq!(fnv_f32(&destination), hashes[1], "{line}");
            cases += 1;
        }
        assert_eq!(cases, 4);
    }

    #[test]
    fn bilateral_filter_rgb_image_rows_split_and_cancel() {
        let lut = exp_lut();
        let (w, h) = (17, 23);
        let mut state = 7_u32;
        let source: Vec<f32> = (0..3 * w * h)
            .map(|_| 200.0 + (next(&mut state) >> 16) as f32 * 0.0113)
            .collect();
        let params = untiled_params(&lut, 0, 5, true);
        let mut whole = vec![-321.5; 3 * w * h];
        bilateral_filter_rgb_image(
            &source,
            3 * w,
            &mut whole,
            3 * w,
            [w, h],
            &params,
            crate::Control::none(),
        )
        .unwrap();
        let mut split = vec![-321.5; 3 * w * h];
        for rows in [[0, 4], [5, 5], [6, 11], [12, u32::MAX >> 1]] {
            let params = BilateralImageParams { rows, ..params };
            bilateral_filter_rgb_image(
                &source,
                3 * w,
                &mut split,
                3 * w,
                [w, h],
                &params,
                crate::Control::none(),
            )
            .unwrap();
        }
        assert_eq!(fnv_f32(&whole), fnv_f32(&split));
        let cancel = std::sync::atomic::AtomicBool::new(true);
        let mut untouched = vec![-321.5; 3 * w * h];
        assert!(matches!(
            bilateral_filter_rgb_image(
                &source,
                3 * w,
                &mut untouched,
                3 * w,
                [w, h],
                &params,
                crate::Control::new(&cancel)
            ),
            Err(Error::Cancelled)
        ));
        assert!(untouched.iter().all(|&v| v == -321.5));
        // Invalid layouts and domains leave the destination unchanged.
        for (len, stride, size, filter_size, rows) in [
            (3 * w * h - 1, 3 * w, [w, h], 5, [0, 9]),
            (3 * w * h, 3 * w - 1, [w, h], 5, [0, 9]),
            (3 * w * h, 3 * w, [w, 4], 5, [0, 9]),
            (3 * w * h, 3 * w, [w, h], 2 * w as u32, [0, 9]),
            (3 * w * h, 3 * w, [w, h], 5, [0, u32::MAX]),
            (3 * w * h, 3 * w, [0, h], 5, [0, 9]),
        ] {
            let params = BilateralImageParams {
                filter_size,
                rows,
                ..params
            };
            let mut destination = vec![-321.5; 3 * w * h];
            assert!(bilateral_filter_rgb_image(
                &source[..len],
                stride,
                &mut destination,
                3 * w,
                size,
                &params,
                crate::Control::none()
            )
            .is_err());
            assert!(destination.iter().all(|&v| v == -321.5));
        }
    }

    #[test]
    fn generate_kernel_map_matches_native_battery() {
        let fixture = include_str!("../../tests/fixtures/spp-pixmap-kernel-map-native.txt");
        let mut cases = 0;
        for line in fixture.lines().filter(|l| l.starts_with("KMAP ")) {
            let parts: Vec<_> = line.split_whitespace().collect();
            let v: Vec<usize> = [1, 2, 3, 4, 6].map(|i| parts[i].parse().unwrap()).to_vec();
            let [w, h, step, pad, pattern] = v[..] else {
                panic!()
            };
            let threshold = f32::from_bits(u32::from_str_radix(parts[5], 16).unwrap());
            let stride = 3 * w + pad;
            let mut source = vec![-123.25_f32; 2 * GUARD + stride * h];
            let mut state = 0x2718_2818;
            for y in 0..h {
                for x in 0..w {
                    for c in 0..3 {
                        let s = next(&mut state);
                        let i = (x + 3 * y + 5 * c) % 143;
                        source[GUARD + stride * y + 3 * x + c] = match pattern {
                            1 => x as f32 * 13.7 + y as f32 * 9.1 + c as f32 * 3.3,
                            2 if i % 11 == 0 => f32::from_bits(0x7fc1_2345),
                            2 if i % 13 == 0 => f32::NEG_INFINITY,
                            3 => c as f32 + 300.0,
                            4 => c as f32 + 700.0,
                            5 => c as f32 + 150.0,
                            _ => (s >> 16) as f32 * 0.0157,
                        };
                    }
                }
            }
            assert_eq!(
                fnv_f32(&source),
                u64::from_str_radix(parts[7], 16).unwrap(),
                "{line}"
            );
            let count = w.div_ceil(step) * h.div_ceil(step);
            let mut map = vec![0xdead_beef_u32; count + 8];
            let n =
                generate_kernel_map(&source[GUARD..], stride, [w, h], step, threshold, &mut map)
                    .unwrap();
            assert_eq!(n, count);
            assert_eq!(
                fnv(&map, |v| v),
                u64::from_str_radix(parts[8], 16).unwrap(),
                "{line}"
            );
            cases += 1;
        }
        assert_eq!(cases, 4416);
    }

    #[test]
    fn invalid_calls_leave_buffers_unchanged() {
        let source = [1.5_f32; 3 * 6 * 5];
        let weights = [0.5_f32; 6 * 5];
        // FIR: short source, short weights, small strides, short outputs,
        // narrow images and border overflow.
        for (src_len, ss, wl, size, b, dl, ds, tl) in [
            (89, 18, 30, [6, 5], 0, 90, 15, 30),
            (90, 17, 30, [6, 5], 0, 90, 15, 30),
            (90, 18, 29, [6, 5], 0, 90, 15, 30),
            (90, 18, 30, [6, 5], 0, 89, 15, 30),
            (90, 18, 30, [6, 5], 0, 90, 14, 30),
            (90, 18, 30, [6, 5], 0, 90, 15, 29),
            (90, 18, 30, [5, 5], 2, 90, 15, 30),
            (90, 18, 30, [6, 5], 3, 90, 15, 30),
            (90, 18, 30, [6, 5], usize::MAX, 90, 15, 30),
            (90, 18, 30, [0, 5], 0, 90, 15, 30),
            (90, 18, 30, [usize::MAX, 5], 0, 90, 15, 30),
        ] {
            let mut destination = vec![-321.5; dl];
            let mut transposed = vec![-654.25; tl];
            assert!(fir(
                &source[..src_len],
                ss,
                &weights[..wl],
                6,
                size,
                b,
                &mut destination,
                ds,
                Some((&mut transposed, 5)),
            )
            .is_err());
            assert!(destination.iter().all(|&v| v == -321.5));
            assert!(transposed.iter().all(|&v| v == -654.25));
        }
        // LocalMean also needs 2 * border + 2 rows.
        for (size, b, dl) in [
            ([6, 5], 2, 90),
            ([6, 5], 0, 89),
            ([6, 1], 0, 90),
            ([1, 5], 0, 90),
        ] {
            let mut destination = vec![-321.5; dl];
            assert!(local_mean(&source, 18, &weights, 6, size, b, &mut destination, 18).is_err());
            assert!(destination.iter().all(|&v| v == -321.5));
        }
        // Kernel map: zero step, a step that divides a dimension, short output.
        for (step, size, len) in [
            (0, [6, 5], 64),
            (3, [6, 5], 64),
            (5, [6, 5], 64),
            (4, [6, 5], 3),
        ] {
            let mut map = vec![77_u32; len];
            assert!(generate_kernel_map(&source, 18, size, step, 300.0, &mut map).is_err());
            assert!(map.iter().all(|&v| v == 77));
        }
        // Bilateral: wide margins, zero step, short map, and a radius that reads
        // past the source slice. The source and the destination stay unchanged.
        let lut = exp_lut();
        let params = BilateralRgbParams {
            filter_size: 3,
            kernel_step: 4,
            noise_model: MODELS[0],
            subsample: false,
            border: 0,
            exp_lut: &lut,
            lut_scale: 37.3,
        };
        let cases: [(u32, u32, u32, &[u32], usize); 6] = [
            (6, 0, 4, &[1; 8], 90),
            (3, 2, 4, &[1; 8], 90),
            (3, 0, 0, &[1; 8], 90),
            (3, 0, 4, &[], 90),
            (3, 0, 4, &[1, 1, 1, 3, 3, 3], 90),
            (3, 0, 4, &[1; 8], 89),
        ];
        for (size, border, step, map, dl) in cases {
            let mut src = source;
            let mut destination = vec![-321.5; dl];
            let params = BilateralRgbParams {
                filter_size: size,
                border,
                kernel_step: step,
                ..params
            };
            assert!(
                bilateral_filter_rgb(&mut src, 18, &mut destination, 18, [6, 5], map, &params)
                    .is_err()
            );
            assert_eq!(src, source);
            assert!(destination.iter().all(|&v| v == -321.5));
        }
        let mut src = source;
        let mut destination = vec![-321.5; 90];
        bilateral_filter_rgb(&mut src, 18, &mut destination, 18, [6, 5], &[1; 8], &params).unwrap();
    }
}
