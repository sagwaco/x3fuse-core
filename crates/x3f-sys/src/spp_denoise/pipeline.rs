//! Neutral color-export tile orchestration from `ProcessTiles` at 0x7c1b4.

use std::ops::Range;

use crate::merrill::ColorDq;
use crate::{Control, Error, Result};

use super::params::NoiseParams;
use super::pixmap::{bilateral_filter_rgb, generate_kernel_map, local_mean, BilateralRgbParams};
use super::residual::{
    anisotropic_diffusion_mono, anisotropic_diffusion_mono_nm, apply_color_residual,
    apply_mono_residual, color_residual_mixing, gen_residual_min_max_map, generate_sharp_l,
    match_col_offsets, mono_residual_mixing, osf_filter, row_column_pattern_suppression,
};
use super::{
    compute_edge_map, compute_resample_params, resample_image, EdgeMapParams, ResampleParams,
};

const LUT_LEN: usize = 2048;
const KERNEL_STEP: usize = 32;
// The middle banding slider selects table index 2 and filter size 27.
const MATCH_FILTER_SIZE: usize = 27;
const MATCH_PHASE: usize = 2;

/// The RGB input used before the common residual chain.
pub(crate) enum TileInput<'a> {
    /// Run the native per-tile bilateral and local-mean stages.
    Ordinary,
    /// Upsample one tile extracted from the whole-image prefilter result.
    Prefiltered { rgb: &'a [f32], stride: usize },
}

/// Per-export immutable state for neutral color tile processing.
pub(crate) struct Pipeline {
    params: NoiseParams,
    matrix: [[f64; 3]; 3],
    color_dq: ColorDq,
    sensor_scale: f32,
    channel_weights: [f32; 3],
    edge_model: [[f64; 3]; 2],
    exp_lut: Vec<f32>,
    edge_lut: Vec<f32>,
    edge_max: f32,
    edge_factor: f32,
    permutation: Vec<usize>,
    resample: Option<(ResampleParams, ResampleParams)>,
}

impl Pipeline {
    /// Build the LUTs and permutation stream that native SPP builds once per export.
    pub(crate) fn new(
        mut params: NoiseParams,
        matrix: [[f64; 3]; 3],
        color_dq_strengths: [f64; 3],
        sensor_scale: f32,
        adjusted_wb: [f32; 3],
    ) -> Result<Self> {
        if !sensor_scale.is_finite()
            || sensor_scale == 0.0
            || !adjusted_wb
                .iter()
                .all(|value| value.is_finite() && *value != 0.0)
            || !matrix.iter().flatten().all(|value| value.is_finite())
            || !valid_params(&params)
        {
            return Err(Error::InvalidData("invalid SPP tile pipeline parameters"));
        }
        if params.scale > 0.54 && params.scale < 1.0 {
            params.decay = params.decay.min(0.92);
        }

        let channel_weights = adjusted_wb.map(|gain| 1.0 / ((gain * sensor_scale) * 3.0));
        if !channel_weights.iter().all(|value| value.is_finite()) {
            return Err(Error::InvalidData("invalid SPP tile luminance weights"));
        }
        let sensor_squared = sensor_scale * sensor_scale;
        let edge_model = [
            params.color_model[0].map(|value| value / f64::from(sensor_squared)),
            params.color_model[1].map(|value| value / f64::from(sensor_scale)),
        ];

        let mut exp_lut = allocated_vec(LUT_LEN, 0.0_f32)?;
        for (index, value) in exp_lut.iter_mut().enumerate() {
            *value = (index as f32 * f32::from_bits(0xbb80_1002)).exp();
        }
        let edge_max = 4.0 / params.locality;
        let edge_factor = 2047.0 / edge_max;
        let normalization = 1.0_f64 / (1.0 - f64::from(params.edge_strength));
        if !edge_max.is_finite() || edge_max <= 0.0 || !edge_factor.is_finite() {
            return Err(Error::InvalidData("invalid SPP edge LUT parameters"));
        }
        let mut edge_lut = allocated_vec(LUT_LEN, 0.0_f32)?;
        for (index, value) in edge_lut.iter_mut().enumerate() {
            let raw =
                (index as f32 * f32::from_bits(0xbb00_1002)).tanh() + 1.0 - params.edge_strength;
            let normalized = (normalization * f64::from(raw)) as f32;
            *value = params.decay * if normalized < 0.0 { 0.0 } else { normalized };
        }

        let radius = params
            .perm_half
            .checked_mul(2)
            .and_then(|value| value.checked_add(1))
            .ok_or(Error::InvalidData("invalid SPP permutation size"))?;
        let count = usize::try_from(radius)
            .ok()
            .and_then(|value| value.checked_mul(512))
            .ok_or(Error::InvalidData("invalid SPP permutation size"))?;
        let mut permutation = allocated_vec(count, 0_usize)?;
        let (mut a, mut b, mut c, mut d): (u32, u32, u32, u32) =
            (0x0549_1333, 0x1f12_3bb5, 0x159a_55e5, 0x075b_cd15);
        for value in &mut permutation {
            let t = d ^ d.wrapping_shl(11);
            let next = (a >> 19) ^ (t >> 8) ^ a ^ t;
            (d, c, b, a) = (c, b, a, next);
            *value = (next % radius) as usize;
        }
        let resample = (params.scale != 1.0).then(|| {
            (
                compute_resample_params(params.scale),
                compute_resample_params(1.0 / params.scale),
            )
        });

        Ok(Self {
            params,
            matrix,
            color_dq: ColorDq::new(color_dq_strengths)?,
            sensor_scale,
            channel_weights,
            edge_model,
            exp_lut,
            edge_lut,
            edge_max,
            edge_factor,
            permutation,
            resample,
        })
    }

    /// Parameters shared with the whole-image prefilter edge-map stage.
    pub(crate) fn edge_params(&self) -> EdgeMapParams {
        EdgeMapParams {
            bias: self.params.edge_bias,
            scale: self.params.iso_gain,
            noise_model: Some(self.edge_model),
            accumulate: self.params.accumulate,
            smooth: self.params.smooth_edge,
            border: 0,
        }
    }

    /// The native per-export range LUT used by both bilateral overloads.
    pub(crate) fn exp_lut(&self) -> &[f32; LUT_LEN] {
        self.exp_lut.as_slice().try_into().unwrap()
    }

    /// The native `gfArgFactor` stored immediately after the range LUT.
    pub(crate) fn exp_lut_scale(&self) -> f32 {
        2047.0 / (8.0 / self.params.noise_scaling)
    }

    /// Apply the per-export edge LUT with the native half-open bounds.
    pub(crate) fn remap_edge_map(
        &self,
        edge: &mut [f32],
        stride: usize,
        size: [usize; 2],
        border: usize,
    ) -> Result<()> {
        let needed = required_len(size, stride, 1)
            .ok_or(Error::InvalidData("invalid SPP edge map layout"))?;
        if edge.len() < needed || border > size[0] / 2 || border > size[1] / 2 {
            return Err(Error::InvalidData("invalid SPP edge map layout"));
        }
        remap_edges(
            edge,
            stride,
            size,
            border,
            &self.edge_lut,
            self.edge_max,
            self.edge_factor,
        );
        Ok(())
    }

    /// Process one tile and commit only a complete, non-cancelled result.
    pub(crate) fn process_tile(
        &self,
        tile: &mut [f32],
        stride: usize,
        size: [usize; 2],
        match_rows: Range<usize>,
        input: TileInput<'_>,
        control: Control<'_>,
    ) -> Result<Vec<f32>> {
        let [cols, rows] = size;
        let row_samples = cols
            .checked_mul(3)
            .ok_or(Error::InvalidData("invalid SPP tile layout"))?;
        let needed =
            required_len(size, stride, 3).ok_or(Error::InvalidData("invalid SPP tile layout"))?;
        if tile.len() < needed
            || match_rows.start >= match_rows.end
            || match_rows.end > rows
            || cols <= self.params.filter_size as usize + 1
            || rows <= self.params.filter_size as usize
        {
            return Err(Error::InvalidData("invalid SPP tile layout"));
        }
        if let TileInput::Prefiltered { rgb, stride, .. } = &input {
            if *stride < 3 || rgb.len() < 3 {
                return Err(Error::InvalidData("invalid SPP prefiltered tile layout"));
            }
        }

        let pixels = cols
            .checked_mul(rows)
            .ok_or(Error::InvalidData("invalid SPP tile extent"))?;
        let rgb_len = pixels
            .checked_mul(3)
            .ok_or(Error::InvalidData("invalid SPP tile extent"))?;
        let mut work = allocated_vec(
            rows.checked_mul(stride)
                .ok_or(Error::InvalidData("invalid SPP tile extent"))?,
            0.0_f32,
        )?;
        for y in 0..rows {
            work[y * stride..y * stride + row_samples]
                .copy_from_slice(&tile[y * stride..y * stride + row_samples]);
        }
        let mut smooth = allocated_vec(rgb_len, 0.0_f32)?;
        let mut match_scratch = allocated_vec(rgb_len, 0.0_f32)?;
        let mut mono = allocated_vec(pixels, 0.0_f32)?;
        let mut aux = allocated_vec(pixels, 0.0_f32)?;
        let mut aux2 = allocated_vec(pixels, 0.0_f32)?;
        let mut map = allocated_vec(pixels, 0_u16)?;
        let mut transposed = allocated_vec(pixels, 0.0_f32)?;
        control.check()?;

        match input {
            TileInput::Ordinary => {
                let filtered_size = if self.params.scale == 1.0 {
                    size
                } else {
                    [
                        scaled_dimension(cols, self.params.scale)?,
                        scaled_dimension(rows, self.params.scale)?,
                    ]
                };
                let filtered_stride = filtered_size[0] * 3;
                let (filter_source, filter_source_stride, filter_destination): (
                    &mut [f32],
                    usize,
                    &mut [f32],
                ) = if let Some((downsample, _)) = &self.resample {
                    let mut down = allocated_vec(rgb_len, 0.0_f32)?;
                    resample_image(
                        &work,
                        stride,
                        &mut down,
                        filtered_size,
                        filtered_stride,
                        downsample,
                    )?;
                    smooth.fill(0.0);
                    // Native B buffers retain full-tile capacity behind the smaller pixmap wrapper.
                    match_scratch.fill(0.0);
                    smooth[..down.len()].copy_from_slice(&down);
                    (&mut smooth, filtered_stride, &mut match_scratch)
                } else {
                    (&mut work, stride, &mut smooth)
                };
                let map_len = filtered_size[0]
                    .div_ceil(KERNEL_STEP)
                    .checked_mul(filtered_size[1].div_ceil(KERNEL_STEP))
                    .ok_or(Error::InvalidData("invalid SPP kernel map extent"))?;
                let mut kernel_map = allocated_vec(map_len, 0_u32)?;
                generate_kernel_map(
                    filter_source,
                    filter_source_stride,
                    filtered_size,
                    KERNEL_STEP,
                    self.params.iso,
                    &mut kernel_map,
                )?;
                control.check()?;
                bilateral_filter_rgb(
                    filter_source,
                    filter_source_stride,
                    filter_destination,
                    filtered_stride,
                    filtered_size,
                    &kernel_map,
                    &BilateralRgbParams {
                        filter_size: self.params.filter_size,
                        kernel_step: KERNEL_STEP as u32,
                        noise_model: self.params.color_model,
                        subsample: self.params.subsample,
                        border: 0,
                        exp_lut: self.exp_lut(),
                        lut_scale: self.exp_lut_scale(),
                    },
                )?;
                control.check()?;
                let edge_pixels = filtered_size[0] * filtered_size[1];
                compute_edge_map(
                    filter_destination,
                    filtered_stride,
                    &mut mono[..edge_pixels],
                    filtered_size,
                    filtered_size[0],
                    &self.edge_params(),
                )?;
                self.remap_edge_map(&mut mono[..edge_pixels], filtered_size[0], filtered_size, 8)?;
                control.check()?;
                let source = filter_destination.to_vec();
                local_mean(
                    &source,
                    filtered_stride,
                    &mono[..edge_pixels],
                    filtered_size[0],
                    filtered_size,
                    8,
                    filter_destination,
                    filtered_stride,
                )?;
                if let Some((_, upsample)) = &self.resample {
                    let filtered = filter_destination.to_vec();
                    smooth.fill(0.0);
                    resample_image(
                        &filtered,
                        filtered_stride,
                        &mut smooth,
                        size,
                        row_samples,
                        upsample,
                    )?;
                }
            }
            TileInput::Prefiltered {
                rgb,
                stride: source_stride,
            } => {
                let (_, upsample) = self.resample.as_ref().ok_or(Error::InvalidData(
                    "SPP prefiltered tile requires resampling",
                ))?;
                resample_image(rgb, source_stride, &mut smooth, size, row_samples, upsample)?;
            }
        }
        control.check()?;

        let noise_model = [
            self.params.color_model[0][0],
            self.params.color_model[0][1],
            self.params.color_model[0][2],
            self.params.color_model[1][0],
            self.params.color_model[1][1],
            self.params.color_model[1][2],
            self.params.color_model[2][0],
            self.params.color_model[2][1],
            self.params.color_model[2][2],
        ];
        match_col_offsets(
            &mut work,
            stride,
            &mut match_scratch,
            row_samples,
            size,
            match_rows.start,
            match_rows.end,
            &noise_model,
            &self.exp_lut,
            8.0,
            self.params.column_noise_scale,
            self.params.iso,
            MATCH_FILTER_SIZE,
            false,
            [1.0; 3],
            MATCH_PHASE,
        )?;
        control.check()?;

        make_luminance(&work, stride, &mut mono, cols, size, self.channel_weights);
        let intensity_model = self
            .params
            .intensity_model
            .map(|value| value * self.params.aniso_noise_scale);
        anisotropic_diffusion_mono_nm(&mut mono, cols, size, intensity_model, self.params.lambda)?;
        control.check()?;
        anisotropic_diffusion_mono(
            &mut mono,
            cols,
            size,
            self.params.aniso_k as i32,
            self.params.lambda,
        )?;
        osf_filter(&mut mono, cols, size)?;
        make_ratio(
            &mut mono,
            cols,
            &smooth,
            row_samples,
            size,
            self.channel_weights,
            self.params.epsilon,
        );
        control.check()?;

        gen_residual_min_max_map(
            &mut map,
            cols,
            &mono,
            cols,
            size,
            self.params.perm_half as usize,
            self.params.rcps_map_threshold,
            &mut aux,
            cols,
            &mut aux2,
            cols,
            &mut transposed,
            rows,
        )?;
        row_column_pattern_suppression(
            &mut mono,
            cols,
            &mut aux,
            cols,
            &map,
            cols,
            size,
            self.params.perm_half as usize,
            self.params.rcps_threshold,
            &self.permutation,
            self.params.rcps_blending,
        )?;
        control.check()?;
        let mut sharp_source = mono.clone();
        let color_scale = self.params.color_mixing > 0.0;
        if color_scale {
            color_residual_mixing(
                &mut match_scratch,
                row_samples,
                &mono,
                cols,
                &smooth,
                row_samples,
                &work,
                stride,
                size,
                f64::from(self.params.mixing),
                f64::from(self.params.color_mixing),
                f64::from(self.params.residual_strength),
                self.channel_weights,
                self.params.epsilon,
            )?;
        } else {
            mono_residual_mixing(
                &mut mono,
                cols,
                &smooth,
                row_samples,
                &work,
                stride,
                size,
                f64::from(self.params.mixing),
                f64::from(self.params.residual_strength),
                self.channel_weights,
                self.params.epsilon,
            )?;
        }
        apply_matrix_color_dq(&mut smooth, row_samples, size, self.matrix, &self.color_dq);
        copy_rgb(&smooth, row_samples, &mut work, stride, size);
        generate_sharp_l(
            &mut sharp_source,
            cols,
            &mut aux,
            cols,
            &work,
            stride,
            size,
            f64::from(self.params.sharpen_amount),
            self.params.iso,
        )?;
        if color_scale {
            apply_color_residual(
                &mut work,
                stride,
                &match_scratch,
                row_samples,
                &smooth,
                row_samples,
                &sharp_source,
                cols,
                &mut aux2,
                cols,
                size,
                self.params.residual_strength,
            )?;
        } else {
            apply_mono_residual(
                &mut work,
                stride,
                &mono,
                cols,
                &smooth,
                row_samples,
                &sharp_source,
                cols,
                &mut aux2,
                cols,
                size,
                self.params.residual_strength,
            )?;
        }
        control.check()?;
        for y in 0..rows {
            tile[y * stride..y * stride + row_samples]
                .copy_from_slice(&work[y * stride..y * stride + row_samples]);
        }
        Ok(aux2)
    }
}

fn valid_params(params: &NoiseParams) -> bool {
    params.iso.is_finite()
        && params.iso > 0.0
        && params.iso_gain.is_finite()
        && [0.375, 0.5, 0.625, 0.75, 0.875, 1.0].contains(&params.scale)
        && params.filter_size >= 2
        && params.noise_scaling.is_finite()
        && params.noise_scaling != 0.0
        && params.alpha == 1.0
        && params.locality.is_finite()
        && params.locality > 0.0
        && params.edge_bias.is_finite()
        && params.edge_strength.is_finite()
        && params.edge_strength != 1.0
        && params.decay.is_finite()
        && params.lambda.is_finite()
        && params.aniso_k != 0
        && params.aniso_noise_scale.is_finite()
        && params.epsilon.is_finite()
        && params.column_noise_scale.is_finite()
        && params.column_noise_scale != 0.0
        && params.rcps_threshold.is_finite()
        && params.rcps_map_threshold.is_finite()
        && params.rcps_blending.is_finite()
        && params.perm_half <= 4
        && params.sharpen_amount.is_finite()
        && params
            .color_model
            .iter()
            .flatten()
            .all(|value| value.is_finite())
        && params.intensity_model.iter().all(|value| value.is_finite())
}

fn scaled_dimension(dimension: usize, scale: f32) -> Result<usize> {
    let dimension = u32::try_from(dimension)
        .map_err(|_| Error::InvalidData("invalid SPP scaled tile extent"))?;
    let scaled = (f64::from(dimension as f32) * f64::from(scale)).ceil() as u64;
    usize::try_from((scaled + 1) & !1)
        .map_err(|_| Error::InvalidData("invalid SPP scaled tile extent"))
}

fn allocated_vec<T: Clone>(len: usize, value: T) -> Result<Vec<T>> {
    let mut output = Vec::new();
    output
        .try_reserve_exact(len)
        .map_err(|_| Error::Allocation)?;
    output.resize(len, value);
    Ok(output)
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

fn remap_edges(
    edge: &mut [f32],
    stride: usize,
    size: [usize; 2],
    border: usize,
    lut: &[f32],
    maximum: f32,
    factor: f32,
) {
    let [cols, rows] = size;
    if rows <= border * 2 || cols <= border * 2 + 2 {
        return;
    }
    for y in border..rows - border {
        for x in border + 1..cols - border - 1 {
            let value = edge[y * stride + x];
            let value = if value < maximum { value } else { maximum };
            edge[y * stride + x] = lut[(value * factor) as usize];
        }
    }
}

fn make_luminance(
    rgb: &[f32],
    rgb_stride: usize,
    mono: &mut [f32],
    mono_stride: usize,
    size: [usize; 2],
    weights: [f32; 3],
) {
    for y in 0..size[1] {
        for x in 0..size[0] {
            let i = y * rgb_stride + x * 3;
            let value = rgb[i + 1] * weights[1];
            let value = rgb[i].mul_add(weights[0], value);
            mono[y * mono_stride + x] = rgb[i + 2].mul_add(weights[2], value);
        }
    }
}

fn make_ratio(
    mono: &mut [f32],
    mono_stride: usize,
    rgb: &[f32],
    rgb_stride: usize,
    size: [usize; 2],
    weights: [f32; 3],
    epsilon: f32,
) {
    for y in 0..size[1] {
        for x in 0..size[0] {
            let i = y * rgb_stride + x * 3;
            let luma = rgb[i] * weights[0] + rgb[i + 1] * weights[1] + rgb[i + 2] * weights[2];
            let j = y * mono_stride + x;
            mono[j] = (mono[j] + epsilon) / (luma.max(0.0) + epsilon);
        }
    }
}

fn apply_matrix_color_dq(
    rgb: &mut [f32],
    stride: usize,
    size: [usize; 2],
    matrix: [[f64; 3]; 3],
    color_dq: &ColorDq,
) {
    for y in 0..size[1] {
        for x in 0..size[0] {
            let i = y * stride + x * 3;
            let p = [
                f64::from(rgb[i]),
                f64::from(rgb[i + 1]),
                f64::from(rgb[i + 2]),
            ];
            let q = matrix.map(|row| {
                let value = row[1] * p[1];
                let value = row[0].mul_add(p[0], value);
                row[2].mul_add(p[2], value)
            });
            rgb[i..i + 3].copy_from_slice(&color_dq.apply(q));
        }
    }
}

fn copy_rgb(
    source: &[f32],
    source_stride: usize,
    destination: &mut [f32],
    destination_stride: usize,
    size: [usize; 2],
) {
    let count = size[0] * 3;
    for y in 0..size[1] {
        destination[y * destination_stride..y * destination_stride + count]
            .copy_from_slice(&source[y * source_stride..y * source_stride + count]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicBool;

    fn pipeline_at(scale: f32) -> Pipeline {
        let mut params = NoiseParams::default();
        params.scale = scale;
        params.intensity_model = params
            .color_model
            .map(|row| ((row[0] + row[2] + row[1]) / 9.0) as f32);
        Pipeline::new(
            params,
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [4.0; 3],
            1.0,
            [1.0; 3],
        )
        .unwrap()
    }

    fn pipeline() -> Pipeline {
        pipeline_at(1.0)
    }

    fn tile(pattern: usize, cols: usize, rows: usize) -> Vec<f32> {
        let mut tile = vec![0.0_f32; cols * rows * 3];
        let mut state = 0x9e37_79b9_u32.wrapping_add(pattern as u32);
        for y in 0..rows {
            for x in 0..cols {
                for channel in 0..3 {
                    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    let value = match pattern {
                        0 => 100.125 + ((state >> 12) % 9000) as f32 / 17.0,
                        1 => 31.375 + x as f32 * 0.713 + y as f32 * 1.137 + channel as f32 * 3.19,
                        2 => 512.3125,
                        _ if x == cols / 2 && y == rows / 2 => 4095.625 + channel as f32 * 127.25,
                        _ => 73.1875,
                    };
                    tile[(y * cols + x) * 3 + channel] = value;
                }
            }
        }
        tile
    }

    #[test]
    fn construction_matches_native_tables_and_permutation_prefix() {
        let pipeline = pipeline();
        let fixture = include_str!("../../tests/fixtures/spp-pipeline-tables.txt");
        let expected = fixture
            .lines()
            .find(|line| line.starts_with("TABLE "))
            .unwrap();
        let actual = format!(
            "TABLE exp={:08x},{:08x},{:08x} edge={:08x},{:08x},{:08x} perm={}",
            pipeline.exp_lut[0].to_bits(),
            pipeline.exp_lut[1].to_bits(),
            pipeline.exp_lut[2047].to_bits(),
            pipeline.edge_lut[0].to_bits(),
            pipeline.edge_lut[1].to_bits(),
            pipeline.edge_lut[2047].to_bits(),
            pipeline.permutation[..16]
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(","),
        );
        assert_eq!(actual.split("perm=").last(), expected.split("perm=").last());
        // Native LUT hashes depend on the arm64 macOS expf and tanhf functions.
        if !cfg!(all(target_os = "macos", target_arch = "aarch64")) {
            return;
        }
        assert_eq!(actual, expected);
        let expected_edge = fixture
            .lines()
            .find_map(|line| line.strip_prefix("EDGEHASH "))
            .unwrap();
        assert_eq!(
            crate::float_front::fnv1a_f32(&pipeline.edge_lut),
            u64::from_str_radix(expected_edge, 16).unwrap()
        );
    }

    #[test]
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    fn ordinary_rgb_chain_matches_native_composition() {
        let fixture = include_str!("../../tests/fixtures/spp-pipeline-tables.txt");
        let expected: Vec<(u64, u64)> = fixture
            .lines()
            .filter(|line| line.starts_with("PIPE "))
            .map(|line| {
                let fields: Vec<_> = line.split_whitespace().collect();
                (
                    u64::from_str_radix(fields[2], 16).unwrap(),
                    u64::from_str_radix(fields[3], 16).unwrap(),
                )
            })
            .collect();
        assert_eq!(expected.len(), 4);
        for (pattern, (expected_rgb, expected_sharp)) in expected.into_iter().enumerate() {
            let (cols, rows) = (97, 89);
            let mut tile = tile(pattern, cols, rows);
            let sharp = pipeline()
                .process_tile(
                    &mut tile,
                    cols * 3,
                    [cols, rows],
                    0..80,
                    TileInput::Ordinary,
                    Control::none(),
                )
                .unwrap();
            assert_eq!(crate::float_front::fnv1a_f32(&tile), expected_rgb);
            assert_eq!(crate::float_front::fnv1a_f32(&sharp), expected_sharp);
        }
    }

    #[test]
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    fn scaled_ordinary_chains_match_native_composition() {
        let fixture = include_str!("../../tests/fixtures/spp-pipeline-tables.txt");
        for line in fixture.lines().filter(|line| line.starts_with("SCALED ")) {
            let fields: Vec<_> = line.split_whitespace().collect();
            let pattern: usize = fields[1].parse().unwrap();
            let scale: f32 = fields[2].parse().unwrap();
            let expected_rgb = u64::from_str_radix(fields[3], 16).unwrap();
            let expected_sharp = u64::from_str_radix(fields[4], 16).unwrap();
            let (cols, rows) = (97, 89);
            let mut tile = tile(pattern, cols, rows);
            let sharp = pipeline_at(scale)
                .process_tile(
                    &mut tile,
                    cols * 3,
                    [cols, rows],
                    0..80,
                    TileInput::Ordinary,
                    Control::none(),
                )
                .unwrap();
            assert_eq!(crate::float_front::fnv1a_f32(&tile), expected_rgb);
            assert_eq!(crate::float_front::fnv1a_f32(&sharp), expected_sharp);
        }
    }

    #[test]
    fn portable_tile_core_produces_finite_color_and_sharpness() {
        let mut rgb = tile(0, 97, 89);
        let sharp = pipeline()
            .process_tile(
                &mut rgb,
                291,
                [97, 89],
                0..80,
                TileInput::Ordinary,
                Control::none(),
            )
            .unwrap();
        assert_eq!(sharp.len(), 97 * 89);
        assert!(sharp.iter().all(|v| v.is_finite()));
        // Native tile borders can contain NaNs; the 40-pixel halo is discarded.
        for y in 40..89 - 40 {
            for x in 40..97 - 40 {
                assert!(rgb[(y * 97 + x) * 3..(y * 97 + x + 1) * 3]
                    .iter()
                    .all(|v| v.is_finite() && (0.0..=32766.0).contains(v)));
            }
        }
        assert!(rgb.windows(2).any(|pair| pair[0] != pair[1]));
    }

    #[test]
    fn invalid_and_cancelled_calls_do_not_mutate_the_tile() {
        assert_eq!(scaled_dimension(696, 0.625).unwrap(), 436);
        assert_eq!(scaled_dimension(488, 0.625).unwrap(), 306);
        assert_eq!(scaled_dimension(696, 0.75).unwrap(), 522);
        assert_eq!(scaled_dimension(488, 0.75).unwrap(), 366);
        assert_eq!(scaled_dimension(696, 0.875).unwrap(), 610);
        assert_eq!(scaled_dimension(488, 0.875).unwrap(), 428);

        for invalid_params in [
            NoiseParams {
                scale: 0.8,
                ..NoiseParams::default()
            },
            NoiseParams {
                alpha: 0.5,
                ..NoiseParams::default()
            },
        ] {
            assert!(Pipeline::new(
                invalid_params,
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                [4.0; 3],
                1.0,
                [1.0; 3],
            )
            .is_err());
        }

        let pipeline = pipeline();
        let mut tile = vec![7.0_f32; 64 * 48 * 3];
        let original = tile.clone();
        assert!(pipeline
            .process_tile(
                &mut tile,
                64 * 3,
                [64, 48],
                0..49,
                TileInput::Ordinary,
                Control::none()
            )
            .is_err());
        assert_eq!(tile, original);

        let cancelled = AtomicBool::new(true);
        assert!(matches!(
            pipeline.process_tile(
                &mut tile,
                64 * 3,
                [64, 48],
                0..48,
                TileInput::Ordinary,
                Control::new(&cancelled),
            ),
            Err(Error::Cancelled)
        ));
        assert_eq!(tile, original);
    }
}
