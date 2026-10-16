//! Bounded residual and diffusion kernels from the Merrill SPP pipeline.

use crate::{Control, Error, Result};

const PIXEL_MAX: f32 = f32::from_bits(0x46ff_fc00);
const SHARP_L_MAX: f32 = f32::from_bits(0x46ff_fe00);

fn lower_bound_zero(value: f64) -> f64 {
    if value < 0.0 {
        0.0
    } else {
        value
    }
}

fn upper_bound_pixel(value: f64) -> f32 {
    (if value > f64::from(PIXEL_MAX) {
        f64::from(PIXEL_MAX)
    } else {
        value
    }) as f32
}

fn required_len(size: [usize; 2], stride: usize, channels: usize) -> Option<usize> {
    let [cols, rows] = size;
    if cols == 0 || rows == 0 || stride < cols.checked_mul(channels)? {
        return None;
    }
    (rows - 1)
        .checked_mul(stride)?
        .checked_add(cols.checked_mul(channels)?)
}

fn check_layout(data_len: usize, size: [usize; 2], stride: usize, channels: usize) -> Result<()> {
    if required_len(size, stride, channels).is_none_or(|needed| data_len < needed) {
        return Err(Error::InvalidData("invalid SPP residual image layout"));
    }
    Ok(())
}

fn tone(value: f32, amount: f64) -> f32 {
    let value = f64::from(value);
    if value < 1.0 {
        (1.0 / amount.mul_add(1.0 / value - 1.0, 1.0)) as f32
    } else {
        amount.mul_add(value - 1.0, 1.0) as f32
    }
}

/// Mix a color residual into a mono base and write RGB scale factors.
#[allow(clippy::too_many_arguments)]
pub fn color_residual_mixing(
    output: &mut [f32],
    output_stride: usize,
    mono: &[f32],
    mono_stride: usize,
    reference: &[f32],
    reference_stride: usize,
    filtered: &[f32],
    filtered_stride: usize,
    size: [usize; 2],
    residual_mixing: f64,
    color_mixing: f64,
    tone_amount: f64,
    channel_weights: [f32; 3],
    epsilon: f32,
) -> Result<()> {
    check_layout(output.len(), size, output_stride, 3)?;
    check_layout(mono.len(), size, mono_stride, 1)?;
    check_layout(reference.len(), size, reference_stride, 3)?;
    check_layout(filtered.len(), size, filtered_stride, 3)?;
    if !residual_mixing.is_finite()
        || !color_mixing.is_finite()
        || !tone_amount.is_finite()
        || !epsilon.is_finite()
        || !channel_weights.iter().all(|value| value.is_finite())
    {
        return Err(Error::InvalidData("invalid SPP color residual parameters"));
    }

    for y in 0..size[1] {
        for x in 0..size[0] {
            let oi = y * output_stride + x * 3;
            let mi = y * mono_stride + x;
            let ri = y * reference_stride + x * 3;
            let fi = y * filtered_stride + x * 3;
            let mut source = [0.0_f32; 3];
            let mut smooth = [0.0_f32; 3];
            for c in 0..3 {
                source[c] = reference[ri + c] * channel_weights[c];
                smooth[c] = (filtered[fi + c] * channel_weights[c]).max(0.0);
            }
            let base = if residual_mixing < 1.0 {
                let numerator = smooth[0] + smooth[1] + smooth[2] + epsilon;
                let denominator = (source[0] + source[1] + source[2]).max(0.0) + epsilon;
                residual_mixing.mul_add(
                    f64::from(mono[mi]),
                    (1.0 - residual_mixing) * f64::from(numerator / denominator),
                ) as f32
            } else {
                mono[mi]
            };
            for c in 0..3 {
                let ratio = (smooth[c] * 3.0 + epsilon) / (source[c] * 3.0 + epsilon);
                let base = ((1.0 - color_mixing) * f64::from(base)) as f32;
                let residual = (color_mixing * f64::from(ratio)) as f32;
                let value = base + residual;
                output[oi + c] = tone(value, tone_amount);
            }
        }
    }
    Ok(())
}

/// Mix an RGB residual into a mono scale map.
#[allow(clippy::too_many_arguments)]
pub fn mono_residual_mixing(
    output: &mut [f32],
    output_stride: usize,
    reference: &[f32],
    reference_stride: usize,
    filtered: &[f32],
    filtered_stride: usize,
    size: [usize; 2],
    residual_mixing: f64,
    tone_amount: f64,
    channel_weights: [f32; 3],
    epsilon: f32,
) -> Result<()> {
    check_layout(output.len(), size, output_stride, 1)?;
    check_layout(reference.len(), size, reference_stride, 3)?;
    check_layout(filtered.len(), size, filtered_stride, 3)?;
    if !residual_mixing.is_finite()
        || !tone_amount.is_finite()
        || !epsilon.is_finite()
        || !channel_weights.iter().all(|value| value.is_finite())
    {
        return Err(Error::InvalidData("invalid SPP mono residual parameters"));
    }

    let rows = size[1] & !1;
    if residual_mixing < 1.0 {
        for y in 0..rows {
            for x in 0..size[0] {
                let oi = y * output_stride + x;
                let ri = y * reference_stride + x * 3;
                let fi = y * filtered_stride + x * 3;
                let mut denominator = 0.0_f64;
                let mut numerator = 0.0_f32;
                for c in 0..3 {
                    denominator += f64::from(reference[ri + c] * channel_weights[c]);
                    numerator += (filtered[fi + c] * channel_weights[c]).max(0.0);
                }
                let ratio = (f64::from(numerator + epsilon)
                    / (denominator.max(0.0) + f64::from(epsilon)))
                    as f32;
                let mixed = residual_mixing.mul_add(
                    f64::from(output[oi]),
                    (1.0 - residual_mixing) * f64::from(ratio),
                ) as f32;
                output[oi] = tone(mixed, tone_amount);
            }
        }
    } else {
        for y in 0..rows {
            for x in 0..size[0] {
                let i = y * output_stride + x;
                output[i] = tone(output[i], tone_amount);
            }
        }
    }
    Ok(())
}

/// Apply per-channel residual scale factors and copy the bounded mono map.
#[allow(clippy::too_many_arguments)]
pub fn apply_color_residual(
    output: &mut [f32],
    output_stride: usize,
    scale: &[f32],
    scale_stride: usize,
    input: &[f32],
    input_stride: usize,
    mono: &[f32],
    mono_stride: usize,
    mono_output: &mut [f32],
    mono_output_stride: usize,
    size: [usize; 2],
    _strength: f32,
) -> Result<()> {
    check_layout(output.len(), size, output_stride, 3)?;
    check_layout(scale.len(), size, scale_stride, 3)?;
    check_layout(input.len(), size, input_stride, 3)?;
    check_layout(mono.len(), size, mono_stride, 1)?;
    check_layout(mono_output.len(), size, mono_output_stride, 1)?;
    if !_strength.is_finite() {
        return Err(Error::InvalidData("invalid SPP residual strength"));
    }
    for y in 0..size[1] {
        for x in 0..size[0] {
            let oi = y * output_stride + x * 3;
            let si = y * scale_stride + x * 3;
            let ii = y * input_stride + x * 3;
            for c in 0..3 {
                let value = lower_bound_zero(f64::from(input[ii + c])) * f64::from(scale[si + c]);
                output[oi + c] = upper_bound_pixel(lower_bound_zero(value));
            }
            let value = mono[y * mono_stride + x];
            mono_output[y * mono_output_stride + x] =
                if value > PIXEL_MAX { PIXEL_MAX } else { value };
        }
    }
    Ok(())
}

/// Apply a mono residual scale to RGB input and copy the bounded mono map.
#[allow(clippy::too_many_arguments)]
pub fn apply_mono_residual(
    output: &mut [f32],
    output_stride: usize,
    scale: &[f32],
    scale_stride: usize,
    input: &[f32],
    input_stride: usize,
    mono: &[f32],
    mono_stride: usize,
    mono_output: &mut [f32],
    mono_output_stride: usize,
    size: [usize; 2],
    _strength: f32,
) -> Result<()> {
    check_layout(output.len(), size, output_stride, 3)?;
    check_layout(scale.len(), size, scale_stride, 1)?;
    check_layout(input.len(), size, input_stride, 3)?;
    check_layout(mono.len(), size, mono_stride, 1)?;
    check_layout(mono_output.len(), size, mono_output_stride, 1)?;
    if !_strength.is_finite() {
        return Err(Error::InvalidData("invalid SPP residual strength"));
    }
    for y in 0..size[1] {
        for x in 0..size[0] {
            let oi = y * output_stride + x * 3;
            let ii = y * input_stride + x * 3;
            let factor = f64::from(scale[y * scale_stride + x]);
            for c in 0..3 {
                let value = lower_bound_zero(f64::from(input[ii + c])) * factor;
                output[oi + c] = upper_bound_pixel(lower_bound_zero(value));
            }
            let value = mono[y * mono_stride + x];
            mono_output[y * mono_output_stride + x] =
                if value > PIXEL_MAX { PIXEL_MAX } else { value };
        }
    }
    Ok(())
}

/// Generate the SPP sharpness luminance map.
#[allow(clippy::too_many_arguments)]
pub fn generate_sharp_l(
    source: &mut [f32],
    source_stride: usize,
    output: &mut [f32],
    output_stride: usize,
    rgb: &[f32],
    rgb_stride: usize,
    size: [usize; 2],
    _amount: f64,
    limit: f32,
) -> Result<()> {
    check_layout(source.len(), size, source_stride, 1)?;
    check_layout(output.len(), size, output_stride, 1)?;
    check_layout(rgb.len(), size, rgb_stride, 3)?;
    if !_amount.is_finite() || !limit.is_finite() {
        return Err(Error::InvalidData("invalid SPP sharpness parameters"));
    }
    let threshold = (limit - 100.0).max(-100.0);
    let curve = if threshold < 1500.0 {
        1.0 - threshold / 1500.0
    } else {
        -100.0
    };
    for y in 0..size[1] {
        for x in 0..size[0] {
            let l = source[y * source_stride + x];
            let weight = {
                let reciprocal = 1.0_f64 / f64::from(l);
                let curved = 1.0 / (reciprocal - 1.0).mul_add(f64::from(curve), 1.0);
                let gate = if l > 1.0 { 1.0 } else { 0.0 };
                let linear = ((l - 1.0).mul_add(curve, 1.0) * gate) as f64;
                (linear + if l <= 1.0 { curved } else { 0.0 }) as f32
            };
            let ri = y * rgb_stride + x * 3;
            let channels = [
                lower_bound_zero(f64::from(rgb[ri])),
                lower_bound_zero(f64::from(rgb[ri + 1])),
                lower_bound_zero(f64::from(rgb[ri + 2])),
            ];
            let weight = f64::from(weight);
            let sum = channels[0].mul_add(weight, channels[1] * weight);
            let sum = channels[2].mul_add(weight, sum);
            output[y * output_stride + x] = (sum as f32).min(SHARP_L_MAX);
        }
    }
    if size[0] >= 5 && size[1] >= 5 {
        let sharp = output.to_vec();
        for y in 2..size[1] - 2 {
            for x in 2..size[0] - 2 {
                let mut sum = 0.0_f32;
                for yy in y - 1..=y + 1 {
                    for xx in x - 1..=x + 1 {
                        sum += sharp[yy * output_stride + xx];
                    }
                }
                source[y * source_stride + x] = sharp[y * output_stride + x] + sum / -9.0;
            }
        }
    }
    Ok(())
}

/// Apply the Stage2B luminance sharpness plane to RGB pixels.
pub fn sharpen(
    rgb: &mut [f32],
    rgb_stride: usize,
    sharp_l: &[i16],
    sharp_l_stride: usize,
    size: [usize; 2],
    amount: f64,
    control: Control<'_>,
) -> Result<()> {
    check_layout(rgb.len(), size, rgb_stride, 3)?;
    check_layout(sharp_l.len(), size, sharp_l_stride, 1)?;
    if !amount.is_finite() {
        return Err(Error::InvalidData("invalid SPP sharpen amount"));
    }

    let len = size[1]
        .checked_mul(rgb_stride)
        .ok_or(Error::InvalidData("invalid SPP sharpen image layout"))?;
    let mut output = Vec::new();
    output
        .try_reserve_exact(len)
        .map_err(|_| Error::Allocation)?;
    output.extend_from_slice(&rgb[..len]);
    control.check()?;

    for y in 0..size[1] {
        for x in 0..size[0] {
            let i = y * rgb_stride + x * 3;
            let channels = [
                lower_bound_zero(f64::from(rgb[i])),
                lower_bound_zero(f64::from(rgb[i + 1])),
                lower_bound_zero(f64::from(rgb[i + 2])),
            ];
            let sum = (channels[0] + channels[1] + channels[2]) as f32;
            let chroma = if sum == 0.0 {
                [0.0; 3]
            } else {
                channels.map(|value| value / f64::from(sum))
            };
            let delta = sharp_l[y * sharp_l_stride + x] as f32 * 0.25;
            let delta = (f64::from(delta) * amount) as f32;
            let delta = if delta > 1000.0 { 1000.0 } else { delta };
            let delta = if delta < -1000.0 { -1000.0 } else { delta };
            let lower = f32::from_bits(0xbeb3_3333) * sum;
            let upper = f32::from_bits(0x3eb3_3333) * sum;
            let delta = if delta > lower { delta } else { lower };
            let delta = if upper < delta { upper } else { delta };
            let luminance = f64::from(sum + delta);
            for channel in 0..3 {
                let value = chroma[channel] * luminance;
                output[i + channel] = (if value > 32767.0 { 32767.0 } else { value }) as f32;
            }
        }
        control.check()?;
    }
    rgb[..len].copy_from_slice(&output);
    Ok(())
}

fn diffuse(
    image: &mut [f32],
    stride: usize,
    size: [usize; 2],
    lambda: f32,
    scale_at: impl Fn(f32) -> f32,
) {
    let [cols, rows] = size;
    if cols < 3 || rows < 3 {
        return;
    }
    let source = image.to_vec();
    for y in 1..rows - 1 {
        for x in 1..cols - 1 {
            let i = y * stride + x;
            let center = source[i];
            let differences = [
                source[i - stride] - center,
                source[i + stride] - center,
                source[i + 1] - center,
                source[i - 1] - center,
            ];
            let scale = scale_at(center);
            let conduction =
                differences.map(|difference| 1.0 / scale.mul_add(difference * difference, 1.0));
            let change = conduction[0].mul_add(differences[0], differences[1] * conduction[1]);
            let change = conduction[2].mul_add(differences[2], change);
            let change = conduction[3].mul_add(differences[3], change);
            image[i] = lambda.mul_add(change, center);
        }
    }
    for y in 0..rows - 2 {
        image[y * stride] = source[(y + 1) * stride];
    }
    image[(rows - 2) * stride + cols - 1] = 0.0;
}

/// Run one native anisotropic diffusion pass with a fixed conduction value.
pub fn anisotropic_diffusion_mono(
    image: &mut [f32],
    stride: usize,
    size: [usize; 2],
    conduction: i32,
    lambda: f32,
) -> Result<()> {
    check_layout(image.len(), size, stride, 1)?;
    let square = conduction.wrapping_mul(conduction);
    if square == 0 || !lambda.is_finite() {
        return Err(Error::InvalidData("invalid SPP diffusion parameters"));
    }
    let scale = 1.0 / square as f32;
    diffuse(image, stride, size, lambda, |_| scale);
    Ok(())
}

/// Run one native anisotropic diffusion pass with a quadratic noise model.
pub fn anisotropic_diffusion_mono_nm(
    image: &mut [f32],
    stride: usize,
    size: [usize; 2],
    noise_model: [f32; 3],
    lambda: f32,
) -> Result<()> {
    check_layout(image.len(), size, stride, 1)?;
    if !lambda.is_finite() || !noise_model.iter().all(|value| value.is_finite()) {
        return Err(Error::InvalidData("invalid SPP diffusion parameters"));
    }
    diffuse(image, stride, size, lambda, |value| {
        let linear = noise_model[1].mul_add(value, noise_model[0]);
        let model = (noise_model[2] * value).mul_add(value, linear);
        let reciprocal = 1.0 / model;
        reciprocal * reciprocal
    });
    Ok(())
}

/// Replace bounded isolated extrema with the nearest 3 by 3 neighbor.
pub fn despeckle3(
    image: &mut [f32],
    stride: usize,
    size: [usize; 2],
    floor: f32,
    level: f32,
) -> Result<()> {
    check_layout(image.len(), size, stride, 1)?;
    if !floor.is_finite() || !level.is_finite() {
        return Err(Error::InvalidData("invalid SPP despeckle parameters"));
    }
    let scaled = (level / 100.0).min(64.0).max(floor);
    let threshold = (64.0 / 1024.0 / scaled).powf(3.0);
    let [cols, rows] = size;
    if cols < 4 || rows < 3 {
        return Ok(());
    }
    for y in 1..rows - 1 {
        for x in 1..cols - 2 {
            let i = y * stride + x;
            let center = image[i];
            let mut low = f32::INFINITY;
            let mut high = f32::NEG_INFINITY;
            for yy in y - 1..=y + 1 {
                for xx in x - 1..=x + 1 {
                    if yy != y || xx != x {
                        let value = image[yy * stride + xx];
                        low = low.min(value);
                        high = high.max(value);
                    }
                }
            }
            if center > high && center - high > threshold {
                image[i] = high;
            } else if center < low && low - center > threshold {
                image[i] = low;
            }
        }
    }
    Ok(())
}

/// Suppress strict 3 by 3 extrema while preserving the outer border.
pub fn osf_filter(image: &mut [f32], stride: usize, size: [usize; 2]) -> Result<()> {
    check_layout(image.len(), size, stride, 1)?;
    let [cols, rows] = size;
    if cols < 3 || rows < 3 {
        return Ok(());
    }
    let source = image.to_vec();
    for y in 1..rows - 1 {
        for x in 1..cols - 1 {
            let i = y * stride + x;
            let center = source[i];
            let mut low = f32::INFINITY;
            let mut high = f32::NEG_INFINITY;
            for yy in y - 1..=y + 1 {
                for xx in x - 1..=x + 1 {
                    if yy != y || xx != x {
                        let value = source[yy * stride + xx];
                        low = low.min(value);
                        high = high.max(value);
                    }
                }
            }
            if center > high {
                image[i] = high;
            } else if center < low {
                image[i] = low;
            }
        }
    }
    for y in 0..rows - 2 {
        image[y * stride] = source[(y + 1) * stride];
    }
    for y in 1..rows - 1 {
        image[y * stride + cols - 1] = source[cols - 1];
    }
    Ok(())
}

/// Build the residual gate and expose the min and max maps used by it.
#[allow(clippy::too_many_arguments)]
pub fn gen_residual_min_max_map(
    map: &mut [u16],
    map_stride: usize,
    source: &[f32],
    source_stride: usize,
    size: [usize; 2],
    filter_size: usize,
    threshold: f32,
    min_map: &mut [f32],
    min_stride: usize,
    max_map: &mut [f32],
    max_stride: usize,
    scratch: &mut [f32],
    scratch_stride: usize,
) -> Result<()> {
    check_layout(map.len(), size, map_stride, 1)?;
    check_layout(source.len(), size, source_stride, 1)?;
    check_layout(min_map.len(), size, min_stride, 1)?;
    check_layout(max_map.len(), size, max_stride, 1)?;
    if filter_size > 4 || !threshold.is_finite() {
        return Err(Error::InvalidData("invalid SPP residual filter size"));
    }
    let [cols, rows] = size;
    check_layout(scratch.len(), [rows, cols], scratch_stride, 1)?;
    extreme_filter_pass(
        scratch,
        scratch_stride,
        source,
        source_stride,
        size,
        filter_size,
        f32::max,
    );
    extreme_filter_pass(
        max_map,
        max_stride,
        scratch,
        scratch_stride,
        [rows, cols],
        filter_size,
        f32::max,
    );
    extreme_filter_pass(
        scratch,
        scratch_stride,
        source,
        source_stride,
        size,
        filter_size,
        f32::min,
    );
    extreme_filter_pass(
        min_map,
        min_stride,
        scratch,
        scratch_stride,
        [rows, cols],
        filter_size,
        f32::min,
    );
    let border = filter_size * 2 + 1;
    if border < rows.saturating_sub(border) && border < cols.saturating_sub(border) {
        for y in border..rows - border {
            for x in border..cols - border {
                map[y * map_stride + x] = u16::from(
                    (max_map[y * max_stride + x] - 1.0).abs() < threshold
                        && (min_map[y * min_stride + x] - 1.0).abs() < threshold,
                );
            }
        }
    }
    Ok(())
}

fn extreme_filter_pass(
    output: &mut [f32],
    output_stride: usize,
    source: &[f32],
    source_stride: usize,
    size: [usize; 2],
    filter_size: usize,
    extreme: fn(f32, f32) -> f32,
) {
    let [cols, rows] = size;
    let border = filter_size * 2 + 1;
    if border >= rows.saturating_sub(border) || filter_size >= cols.saturating_sub(border) {
        return;
    }
    for y in border..rows - border {
        let mut center = filter_size * 2;
        while center < cols - border {
            for x in center - filter_size..=center + filter_size {
                let mut value = source[y * source_stride + x - filter_size];
                for xx in x - filter_size + 1..=x + filter_size {
                    value = extreme(value, source[y * source_stride + xx]);
                }
                output[x * output_stride + y] = value;
            }
            center += border;
        }
    }
}

/// Permute flagged row and column samples, then blend near-neutral residuals.
#[allow(clippy::too_many_arguments)]
pub fn row_column_pattern_suppression(
    source: &mut [f32],
    source_stride: usize,
    output: &mut [f32],
    output_stride: usize,
    map: &[u16],
    map_stride: usize,
    size: [usize; 2],
    filter_size: usize,
    threshold: f32,
    permutation: &[usize],
    blend: f32,
) -> Result<()> {
    check_layout(output.len(), size, output_stride, 1)?;
    check_layout(source.len(), size, source_stride, 1)?;
    check_layout(map.len(), size, map_stride, 1)?;
    if filter_size > 4 {
        return Err(Error::InvalidData(
            "invalid SPP row-column suppression parameters",
        ));
    }
    let radius = filter_size * 2 + 1;
    if permutation.is_empty()
        || permutation.iter().any(|&index| index >= radius)
        || !threshold.is_finite()
        || !blend.is_finite()
    {
        return Err(Error::InvalidData(
            "invalid SPP row-column suppression parameters",
        ));
    }
    let [cols, rows] = size;
    let copy_count = cols * rows;
    output[..copy_count].copy_from_slice(&source[..copy_count]);
    let mut permutation_index = 0;
    if radius < rows.saturating_sub(radius) && radius < cols.saturating_sub(radius) {
        for y in radius..rows - radius {
            for x in radius..cols - radius {
                if map[y * map_stride + x] != 0 {
                    for offset in 0..radius {
                        let a = y * source_stride + x - filter_size + offset;
                        let b =
                            y * source_stride + x - filter_size + permutation[permutation_index];
                        source.swap(a, b);
                        permutation_index = (permutation_index + 1) % permutation.len();
                    }
                }
            }
        }
        permutation_index = 0;
        for x in radius..cols - radius {
            for y in radius..rows - radius {
                if map[y * map_stride + x] != 0 {
                    for offset in 0..radius {
                        let a = (y - filter_size + offset) * source_stride + x;
                        let b =
                            (y - filter_size + permutation[permutation_index]) * source_stride + x;
                        source.swap(a, b);
                        permutation_index = (permutation_index + 1) % permutation.len();
                    }
                }
            }
        }
        let keep = 1.0_f64 - f64::from(blend);
        for y in radius..rows - radius {
            for x in radius..cols - radius {
                let i = y * source_stride + x;
                let oi = y * output_stride + x;
                if (f64::from(output[oi]) - 1.0).abs() < f64::from(threshold) {
                    source[i] = (f64::from(source[i]) * f64::from(blend)
                        + f64::from(output[oi]) * keep) as f32;
                } else {
                    source[i] = output[oi];
                }
            }
        }
    }
    Ok(())
}

/// Remove bounded row-band offsets after horizontal bilateral filtering.
#[allow(clippy::too_many_arguments)]
pub fn match_col_offsets(
    image: &mut [f32],
    image_stride: usize,
    filtered: &mut [f32],
    filtered_stride: usize,
    size: [usize; 2],
    first_row: usize,
    last_row: usize,
    noise_model: &[f64; 9],
    range_lut: &[f32],
    scale: f32,
    scale_divisor: f32,
    sensitivity: f32,
    filter_size: usize,
    enabled: bool,
    channel_scale: [f32; 3],
    phase: usize,
) -> Result<()> {
    check_layout(image.len(), size, image_stride, 3)?;
    check_layout(filtered.len(), size, filtered_stride, 3)?;
    let filtered_clear = size[0]
        .checked_mul(filtered_stride)
        .map(|count| count.min(filtered.len()));
    let base_weight = 2047.0 / (scale / scale_divisor);
    if first_row >= last_row
        || last_row > size[1]
        || filtered_clear.is_none()
        || range_lut.len() < 2048
        || phase >= 5
        || !scale.is_finite()
        || scale == 0.0
        || !scale_divisor.is_finite()
        || scale_divisor == 0.0
        || !base_weight.is_finite()
        || !sensitivity.is_finite()
        || !noise_model.iter().all(|value| value.is_finite())
        || !range_lut[..2048].iter().all(|value| value.is_finite())
        || !channel_scale.iter().all(|value| value.is_finite())
    {
        return Err(Error::InvalidData("invalid SPP column-offset parameters"));
    }
    filtered[..filtered_clear.unwrap()].fill(0.0);
    directional_bilateral_rows(
        filtered,
        filtered_stride,
        image,
        image_stride,
        size,
        filter_size,
        noise_model,
        range_lut,
        base_weight,
        channel_scale,
    );
    let block_count = (last_row - first_row) / 40;
    if block_count == 0 {
        return Ok(());
    }
    let radius = filter_size.saturating_sub(1) / 2;
    let phase_scale = [0.2_f32, 0.4, 0.6, 0.8, 1.0][phase];
    let threshold = if sensitivity <= 400.0 {
        0.05
    } else if sensitivity <= 800.0 {
        0.06
    } else if sensitivity <= 1600.0 {
        0.1
    } else if sensitivity <= 3200.0 {
        0.18
    } else {
        0.3
    } * phase_scale;
    let maximum = f32::from_bits(0x46ff_fc00);
    let mut image_mean = vec![[0.0_f32; 3]; size[0]];
    let mut filtered_mean = vec![[0.0_f32; 3]; size[0]];
    for block in 0..block_count {
        let start = first_row + block * 40;
        let end = if block + 1 == block_count {
            last_row
        } else {
            start + 40
        };
        image_mean.fill([0.0; 3]);
        filtered_mean.fill([0.0; 3]);
        for y in start..end {
            for x in 0..size[0] {
                for c in 0..3 {
                    image_mean[x][c] += image[y * image_stride + x * 3 + c];
                    filtered_mean[x][c] += filtered[y * filtered_stride + x * 3 + c];
                }
            }
        }
        let divisor = (end - start) as f32 + 1.0;
        for x in 0..size[0] {
            for c in 0..3 {
                image_mean[x][c] /= divisor;
                filtered_mean[x][c] /= divisor;
            }
        }
        if radius >= size[0].saturating_sub(radius) {
            continue;
        }
        for y in start..end {
            for x in radius..size[0] - radius {
                let i = y * image_stride + x * 3;
                for c in 0..2 {
                    let difference = image_mean[x][c] - filtered_mean[x][c];
                    let use_full = if enabled {
                        (image[i + c] * threshold).abs() > difference.abs()
                    } else {
                        (f64::from(image[i + c]) * 0.06).abs() > f64::from(difference.abs())
                    };
                    if use_full {
                        image[i + c] -= difference;
                    }
                    image[i + c] = image[i + c].clamp(0.0, maximum);
                }
                if x < size[0] - radius - 1 {
                    let difference = image_mean[x][2] - filtered_mean[x][2];
                    let use_full = if enabled {
                        (image[i + 2] * threshold).abs() > difference.abs()
                    } else {
                        (f64::from(image[i + 2]) * 0.06).abs() > f64::from(difference.abs())
                    };
                    let multiplier: f32 = if use_full { 1.0 } else { 0.0 };
                    image[i + 2] = multiplier
                        .mul_add(-difference, image[i + 2])
                        .clamp(0.0, maximum);
                } else {
                    image[i + 2] = image[i + 2].clamp(0.0, maximum);
                }
            }
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn directional_bilateral_rows(
    output: &mut [f32],
    output_stride: usize,
    source: &[f32],
    source_stride: usize,
    size: [usize; 2],
    filter_size: usize,
    noise_model: &[f64; 9],
    range_lut: &[f32],
    base_weight: f32,
    channel_scale: [f32; 3],
) {
    let radius = filter_size / 2;
    let end = size[0]
        .saturating_add(radius)
        .saturating_sub(filter_size | 1);
    if radius >= end {
        return;
    }
    for y in 0..size[1] {
        for x in radius..end {
            let center_index = y * source_stride + x * 3;
            let center = [
                source[center_index],
                source[center_index + 1],
                source[center_index + 2],
            ];
            let mut inverse = [0.0_f64; 3];
            for c in 0..3 {
                let value = f64::from(center[c]);
                let linear = value.abs() * noise_model[c + 3] + noise_model[c];
                let quadratic = noise_model[c + 6] * value * value;
                let variance = linear + quadratic;
                inverse[c] = f64::from(base_weight) / variance;
            }
            let mut weight_sum = 0.0_f32;
            let mut sum = [0.0_f32; 3];
            let first = x.saturating_sub(radius);
            let last = (x + radius).min(size[0] - 1);
            for xx in first..=last {
                let neighbor_index = y * source_stride + xx * 3;
                let difference0 = (f64::from(source[neighbor_index]) - f64::from(center[0]))
                    * f64::from(channel_scale[0]);
                let difference1 = (f64::from(source[neighbor_index + 1]) - f64::from(center[1]))
                    * f64::from(channel_scale[1]);
                let difference2 = (f64::from(source[neighbor_index + 2]) - f64::from(center[2]))
                    * f64::from(channel_scale[2]);
                let distance = (difference1 * difference1)
                    .mul_add(inverse[1], (difference0 * difference0) * inverse[0]);
                let distance = (difference2 * difference2).mul_add(inverse[2], distance);
                let rounded = distance + f64::from_bits(0x3e50_1b2b_29a4_692b);
                let lut_index =
                    (rounded + f64::from_bits(0x4338_0000_0000_0000)).to_bits() as u32 as usize;
                if lut_index < 2048 {
                    let weight = range_lut[lut_index];
                    weight_sum += weight;
                    for c in 0..3 {
                        sum[c] = source[neighbor_index + c].mul_add(weight, sum[c]);
                    }
                }
            }
            let output_index = y * output_stride + x * 3;
            if weight_sum == 0.0 {
                output[output_index..output_index + 3].copy_from_slice(&center);
            } else {
                for c in 0..3 {
                    output[output_index + c] = sum[c] / weight_sum;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const NATIVE: &str = include_str!("../../tests/fixtures/spp-residual-native.txt");

    fn hash_f32(values: &[f32]) -> u64 {
        values
            .iter()
            .flat_map(|value| value.to_bits().to_le_bytes())
            .fold(1_469_598_103_934_665_603, |hash, byte| {
                (hash ^ u64::from(byte)).wrapping_mul(1_099_511_628_211)
            })
    }

    fn hash_u16(values: &[u16]) -> u64 {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .fold(1_469_598_103_934_665_603, |hash, byte| {
                (hash ^ u64::from(byte)).wrapping_mul(1_099_511_628_211)
            })
    }

    fn expected(name: &str) -> (usize, u64) {
        let fields: Vec<_> = NATIVE
            .lines()
            .find(|line| line.starts_with(name))
            .unwrap()
            .split_whitespace()
            .collect();
        (
            fields[1].parse().unwrap(),
            u64::from_str_radix(fields[2], 16).unwrap(),
        )
    }

    fn assert_f32(name: &str, values: &[f32]) {
        let (count, hash) = expected(name);
        assert_eq!(values.len(), count, "{name} count");
        assert_eq!(hash_f32(values), hash, "{name} hash");
    }

    fn assert_u16(name: &str, values: &[u16]) {
        let (count, hash) = expected(name);
        assert_eq!(values.len(), count, "{name} count");
        assert_eq!(hash_u16(values), hash, "{name} hash");
    }

    fn inputs() -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let mono = (0..54)
            .map(|i| (((i * 17 + 3) % 29) as f32).mul_add(0.071, 0.35))
            .collect();
        let mono2 = (0..54)
            .map(|i| (((i * 11 + 5) % 31) as f32).mul_add(0.063, 0.22))
            .collect();
        let rgb = (0..138)
            .map(|i| (((i * 13 + 7) % 41) as f32).mul_add(0.057, -0.2))
            .collect();
        let rgb2 = (0..138)
            .map(|i| (((i * 19 + 2) % 37) as f32).mul_add(0.049, 0.15))
            .collect();
        (mono, mono2, rgb, rgb2)
    }

    fn battery_pixels(
        cols: usize,
        rows: usize,
        stride: usize,
        channels: usize,
        pattern: usize,
    ) -> Vec<f32> {
        let mut values = vec![f32::from_bits(0xc57a_4000); stride * rows + 11];
        for y in 0..rows {
            for x in 0..cols {
                for c in 0..channels {
                    let i = y * stride + x * channels + c;
                    let mix =
                        ((i as u64 * 2_654_435_761 + pattern as u64 * 2_246_822_519) >> 9) as u32;
                    let mut bits = if pattern == 4 {
                        0x4300_0000
                    } else {
                        0x3f00_0000
                    } | (mix & 0x007f_ffff);
                    if pattern == 0 {
                        bits = 0x3f80_0000;
                    }
                    if pattern == 2 && (x + 2 * y + c) % 3 == 0 {
                        bits |= 0x8000_0000;
                    }
                    if pattern == 3 && x == cols / 2 && y == rows / 2 {
                        bits = if c == 1 { 0x45da_c000 } else { 0xbdcc_cccd };
                    }
                    values[i] = f32::from_bits(bits);
                }
            }
        }
        values
    }

    #[test]
    fn native_focused_residual_battery() {
        let cases = [
            (5, 5, 2, 0, 0.0, 0.713, 0.013),
            (6, 4, 3, 1, 0.371, 0.823, 0.019),
            (7, 7, 5, 2, 1.0, 1.137, 0.007),
            (8, 6, 1, 3, 0.837, 0.619, 0.023),
        ];
        for (k, &(cols, rows, pad, pattern, mix, tone, epsilon)) in cases.iter().enumerate() {
            let size = [cols, rows];
            let mono_stride = cols + pad;
            let rgb_stride = cols * 3 + pad + 1;
            let mut mixed = battery_pixels(cols, rows, mono_stride, 1, pattern);
            let reference = battery_pixels(cols, rows, rgb_stride, 3, (pattern + 1) % 5);
            let filtered = battery_pixels(cols, rows, rgb_stride, 3, (pattern + 3) % 5);
            mono_residual_mixing(
                &mut mixed,
                mono_stride,
                &reference,
                rgb_stride,
                &filtered,
                rgb_stride,
                size,
                mix,
                tone,
                [0.317, 0.493, 0.191],
                epsilon,
            )
            .unwrap();
            assert_f32(&format!("BAT_MONO_MIX_{k}_OUT"), &mixed);

            let scale_map = battery_pixels(cols, rows, mono_stride, 1, (pattern + 1) % 5);
            let mono_in = battery_pixels(cols, rows, mono_stride, 1, (pattern + 2) % 5);
            let rgb_in = battery_pixels(cols, rows, rgb_stride, 3, pattern);
            let mut rgb_out = battery_pixels(cols, rows, rgb_stride, 3, 0);
            let mut mono_out = battery_pixels(cols, rows, mono_stride, 1, 0);
            apply_mono_residual(
                &mut rgb_out,
                rgb_stride,
                &scale_map,
                mono_stride,
                &rgb_in,
                rgb_stride,
                &mono_in,
                mono_stride,
                &mut mono_out,
                mono_stride,
                size,
                0.137_f32 + k as f32 * 0.11,
            )
            .unwrap();
            assert_f32(&format!("BAT_APPLY_MONO_{k}_RGB"), &rgb_out);
            assert_f32(&format!("BAT_APPLY_MONO_{k}_MONO"), &mono_out);
        }
    }

    #[test]
    fn native_diffusion_and_row_column_battery() {
        for k in 0..6 {
            let (cols, rows) = if k & 1 == 0 { (7, 7) } else { (8, 6) };
            let stride = cols + 3;
            let mut fixed = battery_pixels(cols, rows, stride, 1, k % 4);
            let mut modeled = fixed.clone();
            for _ in 0..k % 3 {
                anisotropic_diffusion_mono(
                    &mut fixed,
                    stride,
                    [cols, rows],
                    if k & 1 == 0 { 2 } else { 7 },
                    if k & 1 == 0 { 0.25 } else { 0.137 },
                )
                .unwrap();
                anisotropic_diffusion_mono_nm(
                    &mut modeled,
                    stride,
                    [cols, rows],
                    [0.713, 0.017, 0.0031],
                    if k & 1 == 0 { 0.25 } else { 0.119 },
                )
                .unwrap();
            }
            assert_f32(&format!("BAT_DIFF_{k}_FIXED"), &fixed);
            assert_f32(&format!("BAT_DIFF_{k}_NM"), &modeled);
        }

        let permutations: [&[usize]; 3] = [&[0], &[2, 0, 1], &[8, 0, 7, 1, 6, 2, 5, 3, 4]];
        let dimensions = [(7, 8, 2), (11, 12, 3), (23, 24, 2)];
        for k in 0..4 {
            let filter = if k < 2 { k } else { 4 };
            let dimension = if k < 2 { k } else { 2 };
            let (cols, rows, pad) = dimensions[dimension];
            let stride = cols + pad;
            let mut source = battery_pixels(cols, rows, stride, 1, k + 1);
            let mut output = battery_pixels(cols, rows, stride, 1, 0);
            let mut map = vec![0x7777_u16; stride * rows + 11];
            for y in 0..rows {
                for x in 0..cols {
                    map[y * stride + x] = u16::from((x + 3 * y + k) % 5 == 0);
                }
            }
            row_column_pattern_suppression(
                &mut source,
                stride,
                &mut output,
                stride,
                &map,
                stride,
                [cols, rows],
                filter,
                0.213_f32 + 0.017 * k as f32,
                permutations[dimension],
                if k == 2 { 0.0 } else { 0.25 },
            )
            .unwrap();
            assert_f32(&format!("BAT_RCPS_{k}_SOURCE"), &source);
            assert_f32(&format!("BAT_RCPS_{k}_COPY"), &output);
        }
    }

    #[test]
    fn native_match_col_offsets_battery() {
        let noise_model = [
            17.13, 23.71, 31.19, 0.0713, 0.0831, 0.0973, 0.000213, 0.000317, 0.000419,
        ];
        let range_lut: Vec<f32> = (0..2048)
            .map(|i| f32::from_bits(0x3f80_0000 - i as u32 * 997))
            .collect();
        let cases = [
            (19, 41, 2, 1, 40, 3, true, 2.37, 0.317, 399.75, 0, 1),
            (20, 42, 1, 1, 41, 3, false, 2.63, 0.293, 400.25, 1, 2),
            (23, 48, 4, 2, 46, 3, true, 2.71, 0.311, 800.25, 2, 4),
            (24, 81, 3, 0, 80, 4, true, 2.19, 0.271, 1600.25, 3, 3),
        ];
        for (
            k,
            &(
                cols,
                rows,
                pad,
                first,
                last,
                filter,
                enabled,
                scale,
                divisor,
                sensitivity,
                phase,
                pattern,
            ),
        ) in cases.iter().enumerate()
        {
            let stride = cols * 3 + pad;
            let mut image = battery_pixels(cols, rows, stride, 3, pattern);
            let mut filtered = battery_pixels(cols, rows, stride, 3, 0);
            match_col_offsets(
                &mut image,
                stride,
                &mut filtered,
                stride,
                [cols, rows],
                first,
                last,
                &noise_model,
                &range_lut,
                scale,
                divisor,
                sensitivity,
                filter,
                enabled,
                [1.0; 3],
                phase,
            )
            .unwrap();
            assert_f32(&format!("BAT_MATCH_{k}_IMAGE"), &image);
            assert_f32(&format!("BAT_MATCH_{k}_FILTERED"), &filtered);
        }
    }

    #[test]
    fn native_color_and_local_filter_battery() {
        for k in 0..3 {
            let (cols, rows, pad) = (5 + k, 4 + k, 1 + k);
            let size = [cols, rows];
            let mono_stride = cols + pad;
            let rgb_stride = cols * 3 + pad + 1;
            let mono = battery_pixels(cols, rows, mono_stride, 1, k + 1);
            let reference = battery_pixels(cols, rows, rgb_stride, 3, k + 2);
            let filtered = battery_pixels(cols, rows, rgb_stride, 3, (k + 4) % 5);
            let mut mixed = battery_pixels(cols, rows, rgb_stride, 3, 0);
            color_residual_mixing(
                &mut mixed,
                rgb_stride,
                &mono,
                mono_stride,
                &reference,
                rgb_stride,
                &filtered,
                rgb_stride,
                size,
                if k == 0 {
                    0.0
                } else if k == 1 {
                    0.371
                } else {
                    1.0
                },
                0.613 + 0.071 * k as f64,
                0.817 + 0.113 * k as f64,
                [0.317, 0.493, 0.191],
                0.011 + 0.006 * k as f32,
            )
            .unwrap();
            assert_f32(&format!("BAT_COLOR_MIX_{k}_OUT"), &mixed);

            let mut rgb_out = battery_pixels(cols, rows, rgb_stride, 3, 0);
            let mut mono_out = battery_pixels(cols, rows, mono_stride, 1, 0);
            apply_color_residual(
                &mut rgb_out,
                rgb_stride,
                &filtered,
                rgb_stride,
                &reference,
                rgb_stride,
                &mono,
                mono_stride,
                &mut mono_out,
                mono_stride,
                size,
                0.137 + 0.173 * k as f32,
            )
            .unwrap();
            assert_f32(&format!("BAT_APPLY_COLOR_{k}_RGB"), &rgb_out);
            assert_f32(&format!("BAT_APPLY_COLOR_{k}_MONO"), &mono_out);
        }

        let dimensions = [(4, 4, 2), (5, 5, 1), (8, 7, 3)];
        for (k, &(cols, rows, pad)) in dimensions.iter().enumerate() {
            let size = [cols, rows];
            let mono_stride = cols + pad;
            let rgb_stride = cols * 3 + pad + 1;
            let mut sharp_source = battery_pixels(cols, rows, mono_stride, 1, k + 1);
            let mut sharp_output = battery_pixels(cols, rows, mono_stride, 1, 0);
            let rgb = battery_pixels(cols, rows, rgb_stride, 3, k + 2);
            let mut osf = sharp_source.clone();
            let mut despeckled = sharp_source.clone();
            generate_sharp_l(
                &mut sharp_source,
                mono_stride,
                &mut sharp_output,
                mono_stride,
                &rgb,
                rgb_stride,
                size,
                0.213 + 0.187 * k as f64,
                127.25 + 80.5 * k as f32,
            )
            .unwrap();
            osf_filter(&mut osf, mono_stride, size).unwrap();
            despeckle3(
                &mut despeckled,
                mono_stride,
                size,
                0.011 + 0.007 * k as f32,
                7.3 + 5.125 * k as f32,
            )
            .unwrap();
            assert_f32(&format!("BAT_SHARP_{k}_SOURCE"), &sharp_source);
            assert_f32(&format!("BAT_SHARP_{k}_OUTPUT"), &sharp_output);
            assert_f32(&format!("BAT_OSF_{k}_IMAGE"), &osf);
            assert_f32(&format!("BAT_DESPECKLE_{k}_IMAGE"), &despeckled);
        }
    }

    #[test]
    fn native_rectangular_min_max_and_production_match() {
        let dimensions = [(7, 8, 2, 0), (11, 12, 3, 1), (23, 24, 2, 4)];
        for (k, &(cols, rows, pad, filter)) in dimensions.iter().enumerate() {
            let stride = cols + pad;
            let transposed_stride = rows + 1;
            let source = battery_pixels(cols, rows, stride, 1, k + 1);
            let mut maximum = battery_pixels(cols, rows, stride, 1, 0);
            let mut minimum = battery_pixels(cols, rows, stride, 1, 0);
            let mut scratch = battery_pixels(rows, cols, transposed_stride, 1, 0);
            let mut gate = vec![0x7777; stride * rows + 11];
            gen_residual_min_max_map(
                &mut gate,
                stride,
                &source,
                stride,
                [cols, rows],
                filter,
                0.213 + 17.125 * k as f32,
                &mut minimum,
                stride,
                &mut maximum,
                stride,
                &mut scratch,
                transposed_stride,
            )
            .unwrap();
            assert_u16(&format!("BAT_MINMAX_{k}_GATE"), &gate);
            assert_f32(&format!("BAT_MINMAX_{k}_MAX"), &maximum);
            assert_f32(&format!("BAT_MINMAX_{k}_MIN"), &minimum);
            assert_f32(&format!("BAT_MINMAX_{k}_SCRATCH"), &scratch);
        }

        let (cols, rows) = (696, 488);
        let stride = cols * 3;
        let mut image = battery_pixels(cols, rows, stride, 3, 0);
        let mut filtered = vec![0.0; stride * rows + 11];
        let noise_model = [
            17.13, 23.71, 31.19, 0.0713, 0.0831, 0.0973, 0.000213, 0.000317, 0.000419,
        ];
        let range_lut: Vec<f32> = (0..2048)
            .map(|i| f32::from_bits(0x3f80_0000 - i as u32 * 997))
            .collect();
        match_col_offsets(
            &mut image,
            stride,
            &mut filtered,
            stride,
            [cols, rows],
            7,
            477,
            &noise_model,
            &range_lut,
            8.0,
            0.375,
            100.0,
            27,
            false,
            [1.0; 3],
            2,
        )
        .unwrap();
        assert_f32("BAT_PROD_MATCH_0_IMAGE", &image);
        assert_f32("BAT_PROD_MATCH_0_FILTERED", &filtered);

        let source = battery_pixels(cols, rows, cols, 1, 1);
        let mut maximum = battery_pixels(cols, rows, cols, 1, 0);
        let mut minimum = battery_pixels(cols, rows, cols, 1, 0);
        let mut scratch = battery_pixels(rows, cols, rows, 1, 0);
        let mut gate = vec![0x7777; cols * rows + 11];
        gen_residual_min_max_map(
            &mut gate,
            cols,
            &source,
            cols,
            [cols, rows],
            4,
            0.25,
            &mut minimum,
            cols,
            &mut maximum,
            cols,
            &mut scratch,
            rows,
        )
        .unwrap();
        assert_u16("BAT_PROD_MINMAX_GATE", &gate);
        assert_f32("BAT_PROD_MINMAX_MAX", &maximum);
        assert_f32("BAT_PROD_MINMAX_MIN", &minimum);
        assert_f32("BAT_PROD_MINMAX_SCRATCH", &scratch);
    }

    #[test]
    fn native_residual_and_diffusion_oracle() {
        let size = [7, 6];
        let (mono, mono2, rgb, rgb2) = inputs();
        let mut rgb_out = vec![-99.0; 138];
        let mut mono_out = vec![-77.0; 54];
        apply_color_residual(
            &mut rgb_out,
            23,
            &rgb2,
            23,
            &rgb,
            23,
            &mono,
            9,
            &mut mono_out,
            9,
            size,
            0.37,
        )
        .unwrap();
        assert_f32("APPLY_COLOR_RGB", &rgb_out);
        assert_f32("APPLY_COLOR_MONO", &mono_out);

        rgb_out.fill(-99.0);
        mono_out.fill(-77.0);
        apply_mono_residual(
            &mut rgb_out,
            23,
            &mono2,
            9,
            &rgb,
            23,
            &mono,
            9,
            &mut mono_out,
            9,
            size,
            -0.4,
        )
        .unwrap();
        assert_f32("APPLY_MONO_RGB", &rgb_out);
        assert_f32("APPLY_MONO_MONO", &mono_out);

        let mut diff = mono.clone();
        anisotropic_diffusion_mono(&mut diff, 9, size, 9, 0.137).unwrap();
        assert_f32("DIFF", &diff);
        let mut diff_nm = mono.clone();
        anisotropic_diffusion_mono_nm(&mut diff_nm, 9, size, [0.73, 0.021, 0.004], 0.119).unwrap();
        assert_f32("DIFF_NM", &diff_nm);
        let mut osf = mono.clone();
        osf_filter(&mut osf, 9, size).unwrap();
        assert_f32("OSF", &osf);
        let mut despeckled = mono.clone();
        despeckle3(&mut despeckled, 9, size, 0.018, 17.3).unwrap();
        assert_f32("DESPECKLE", &despeckled);

        let mut sharp_source = mono.clone();
        let mut sharp_output = vec![-77.0; 54];
        generate_sharp_l(
            &mut sharp_source,
            9,
            &mut sharp_output,
            9,
            &rgb,
            23,
            size,
            0.43,
            287.5,
        )
        .unwrap();
        assert_f32("SHARP_OUTPUT", &sharp_output);
        assert_f32("SHARP_SOURCE", &sharp_source);

        let mut color_mix = vec![-99.0; 138];
        color_residual_mixing(
            &mut color_mix,
            23,
            &mono,
            9,
            &rgb,
            23,
            &rgb2,
            23,
            size,
            0.37,
            0.61,
            0.82,
            [0.3, 0.5, 0.2],
            0.017,
        )
        .unwrap();
        assert_f32("COLOR_MIX", &color_mix);
        let mut mono_mix = mono.clone();
        mono_residual_mixing(
            &mut mono_mix,
            9,
            &rgb,
            23,
            &rgb2,
            23,
            size,
            0.37,
            0.82,
            [0.3, 0.5, 0.2],
            0.017,
        )
        .unwrap();
        assert_f32("MONO_MIX", &mono_mix);
    }

    #[test]
    fn native_min_max_and_row_column_oracle() {
        let size = [23, 23];
        let source: Vec<f32> = (0..529).map(|i| ((i / 23) * 100 + i % 23) as f32).collect();
        let mut max_pass = vec![-31.0; 529];
        let mut min_pass = vec![-41.0; 529];
        extreme_filter_pass(&mut max_pass, 23, &source, 23, size, 4, f32::max);
        extreme_filter_pass(&mut min_pass, 23, &source, 23, size, 4, f32::min);
        assert_f32("MAX_PASS", &max_pass);
        assert_f32("MIN_PASS", &min_pass);
        let mut gate = vec![0x7777; 529];
        let mut max = vec![-31.0; 529];
        let mut min = vec![-41.0; 529];
        let mut scratch = vec![-51.0; 529];
        gen_residual_min_max_map(
            &mut gate,
            23,
            &source,
            23,
            size,
            4,
            5000.0,
            &mut min,
            23,
            &mut max,
            23,
            &mut scratch,
            23,
        )
        .unwrap();
        assert_u16("HELPER_GATE", &gate);
        assert_f32("HELPER_MAX", &max);
        assert_f32("HELPER_MIN", &min);
        assert_f32("HELPER_SCRATCH", &scratch);

        let mut residual: Vec<f32> = (0..529)
            .map(|i| (((i * 17 + 3) % 21) as f32).mul_add(0.025, 0.75))
            .collect();
        let flags: Vec<u16> = (0..529).map(|i| u16::from((i * 7 + 1) % 6 == 0)).collect();
        let mut copy = vec![-61.0; 529];
        row_column_pattern_suppression(
            &mut residual,
            23,
            &mut copy,
            23,
            &flags,
            23,
            size,
            4,
            0.27,
            &[8, 0, 7, 1, 6, 2, 5, 3, 4],
            0.25,
        )
        .unwrap();
        assert_f32("HELPER_RC_SOURCE", &residual);
        assert_f32("HELPER_RC_COPY", &copy);
    }

    #[test]
    fn native_match_col_offsets_oracle() {
        let size = [23, 48];
        let stride = 69;
        let mut image: Vec<f32> = (0..stride * size[1])
            .map(|i| (((i * 23 + 11) % 67) as f32).mul_add(3.25, 120.0))
            .collect();
        let mut filtered = vec![-55.0; image.len()];
        let noise_model = [20.0, 25.0, 30.0, 0.07, 0.08, 0.09, 0.0002, 0.0003, 0.0004];
        let range_lut: Vec<f32> = (0..2048)
            .map(|i| f32::from_bits(0x3f80_0000 - i as u32 * 1000))
            .collect();
        assert_f32("MATCH_LUT", &range_lut);
        match_col_offsets(
            &mut image,
            stride,
            &mut filtered,
            stride,
            size,
            2,
            46,
            &noise_model,
            &range_lut,
            2.5,
            0.31,
            1.0,
            3,
            true,
            [0.7, 1.1, 0.4],
            2,
        )
        .unwrap();
        assert_f32("MATCH_IMAGE", &image);
        assert_f32("MATCH_FILTERED", &filtered);
    }

    #[test]
    fn invalid_layout_does_not_mutate() {
        let mut image = vec![1.0; 69 * 48];
        let before = image.clone();
        let mut filtered = vec![2.0; 69 * 48];
        let filtered_before = filtered.clone();
        let result = match_col_offsets(
            &mut image,
            69,
            &mut filtered,
            69,
            [23, 48],
            2,
            46,
            &[1.0; 9],
            &[1.0; 16],
            2.5,
            0.31,
            1.0,
            3,
            true,
            [1.0; 3],
            2,
        );
        assert!(result.is_err());
        assert_eq!(image, before);
        assert_eq!(filtered, filtered_before);
    }

    #[test]
    fn invalid_inputs_do_not_mutate_kernel_outputs() {
        let mono = vec![1.0_f32];
        let rgb = vec![1.0_f32; 3];
        let mut mono_out = vec![7.0_f32];
        let mut rgb_out = vec![7.0_f32; 3];
        let mut auxiliary = vec![8.0_f32];
        let mono_before = mono_out.clone();
        let rgb_before = rgb_out.clone();
        let auxiliary_before = auxiliary.clone();

        assert!(color_residual_mixing(
            &mut rgb_out,
            3,
            &mono,
            1,
            &rgb,
            3,
            &rgb,
            3,
            [1, 1],
            f64::NAN,
            1.0,
            1.0,
            [1.0; 3],
            0.0,
        )
        .is_err());
        assert!(mono_residual_mixing(
            &mut mono_out,
            1,
            &rgb,
            3,
            &rgb,
            3,
            [1, 1],
            f64::NAN,
            1.0,
            [1.0; 3],
            0.0,
        )
        .is_err());
        assert!(apply_color_residual(
            &mut rgb_out,
            3,
            &rgb,
            3,
            &rgb,
            3,
            &mono,
            1,
            &mut mono_out,
            1,
            [1, 1],
            f32::NAN,
        )
        .is_err());
        assert!(apply_mono_residual(
            &mut rgb_out,
            3,
            &mono,
            1,
            &rgb,
            3,
            &mono,
            1,
            &mut mono_out,
            1,
            [1, 1],
            f32::NAN,
        )
        .is_err());
        assert!(generate_sharp_l(
            &mut mono_out,
            1,
            &mut auxiliary,
            1,
            &rgb,
            3,
            [1, 1],
            f64::NAN,
            1.0,
        )
        .is_err());
        assert!(anisotropic_diffusion_mono(&mut mono_out, 1, [1, 1], 0, 0.25).is_err());
        assert!(
            anisotropic_diffusion_mono_nm(&mut mono_out, 1, [1, 1], [f32::NAN; 3], 0.25,).is_err()
        );
        assert!(despeckle3(&mut mono_out, 1, [1, 1], f32::NAN, 1.0).is_err());
        assert!(osf_filter(&mut mono_out, 1, [2, 2]).is_err());

        let mut gate = vec![0x7777_u16];
        let mut min = vec![8.0_f32];
        let mut max = vec![9.0_f32];
        let mut scratch = vec![10.0_f32];
        let maps_before = (gate.clone(), min.clone(), max.clone(), scratch.clone());
        assert!(gen_residual_min_max_map(
            &mut gate,
            1,
            &mono,
            1,
            [1, 1],
            0,
            f32::NAN,
            &mut min,
            1,
            &mut max,
            1,
            &mut scratch,
            1,
        )
        .is_err());
        assert!(row_column_pattern_suppression(
            &mut mono_out,
            1,
            &mut auxiliary,
            1,
            &[0],
            1,
            [1, 1],
            0,
            f32::NAN,
            &[0],
            0.25,
        )
        .is_err());

        assert_eq!(mono_out, mono_before);
        assert_eq!(rgb_out, rgb_before);
        assert_eq!(auxiliary, auxiliary_before);
        assert_eq!((gate, min, max, scratch), maps_before);
    }

    #[test]
    fn sharpen_matches_native_shapes_padding_and_strengths() {
        let fixture = include_str!("../../tests/fixtures/spp-pipeline-sharpen.txt");
        for line in fixture.lines().filter(|line| line.starts_with("SHARP ")) {
            let fields: Vec<_> = line.split_whitespace().collect();
            let test: u32 = fields[1].parse().unwrap();
            let width: usize = fields[2].parse().unwrap();
            let rows: usize = fields[3].parse().unwrap();
            let rgb_stride: usize = fields[4].parse().unwrap();
            let amount: f64 = fields[5].parse().unwrap();
            let expected = u64::from_str_radix(fields[6], 16).unwrap();
            let sharp_stride = width + 3;
            let mut rgb = vec![-12345.25_f32; rgb_stride * rows];
            let mut sharp_l = vec![0x5a5a_i16; sharp_stride * rows];
            let mut state = 0x8f3a_21d7_u32.wrapping_add(test);
            for y in 0..rows {
                for x in 0..width {
                    for channel in 0..3 {
                        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                        let value = ((state >> 8) % 70_000) as i32 - 12_000;
                        let mut value = value as f32 / 7.0;
                        if x == 0 && y == 0 {
                            value = [-17.25, 0.0, 33.5][channel];
                        }
                        if x + 1 == width && y + 1 == rows {
                            value = if channel == 2 { 40_000.0 } else { 1.25 };
                        }
                        rgb[y * rgb_stride + x * 3 + channel] = value;
                    }
                    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    sharp_l[y * sharp_stride + x] = (state >> 16) as i16;
                }
            }
            sharp_l[0] = i16::MIN;
            sharp_l[(rows - 1) * sharp_stride + width - 1] = i16::MAX;
            sharpen(
                &mut rgb,
                rgb_stride,
                &sharp_l,
                sharp_stride,
                [width, rows],
                amount,
                Control::none(),
            )
            .unwrap();
            assert_eq!(crate::float_front::fnv1a_f32(&rgb), expected, "case {test}");
            for y in 0..rows {
                assert!(rgb[y * rgb_stride + width * 3..(y + 1) * rgb_stride]
                    .iter()
                    .all(|value| value.to_bits() == (-12345.25_f32).to_bits()));
            }
        }

        let mut rgb = vec![7.25_f32; 24];
        let original = rgb.clone();
        let sharp_l = vec![11_i16; 8];
        assert!(sharpen(&mut rgb, 6, &sharp_l, 2, [3, 4], 1.0, Control::none()).is_err());
        assert_eq!(rgb, original);

        let cancelled = std::sync::atomic::AtomicBool::new(true);
        assert!(matches!(
            sharpen(
                &mut rgb,
                6,
                &sharp_l,
                2,
                [2, 4],
                1.0,
                Control::new(&cancelled),
            ),
            Err(Error::Cancelled)
        ));
        assert_eq!(rgb, original);
    }
}
