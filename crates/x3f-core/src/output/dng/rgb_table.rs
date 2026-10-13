//! Native color rendering and Adobe working-space coordinates.

use x3f_sys::{ColorModeCalibration, ColorModeParameters, ColorModeSensor, Control, Result};

use super::metadata::{mat3_inverse, mat3_mul};

// Adobe PCStoXYZ and dng_color_space::SetMatrixToPCS use this xy white.
pub(super) const PCS_WHITE: [f64; 3] = [0.3457 / 0.3585, 1.0, (1.0 - 0.3457 - 0.3585) / 0.3585];
const PROPHOTO_TO_XYZ_D50: [f64; 9] = [
    0.7977, 0.1352, 0.0313, 0.2880, 0.7119, 0.0001, 0.0, 0.0, 0.8249,
];
const SRGB_TO_XYZ_D50: [f64; 9] = [
    0.4361, 0.3851, 0.1431, 0.2225, 0.7169, 0.0606, 0.0139, 0.0971, 0.7141,
];

pub(super) fn xyz_d50_to_prophoto() -> [f64; 9] {
    mat3_inverse(&normalize_space(PROPHOTO_TO_XYZ_D50))
}

pub(super) fn srgb_to_xyz_d50() -> [f64; 9] {
    normalize_space(SRGB_TO_XYZ_D50)
}

fn normalize_space(mut matrix: [f64; 9]) -> [f64; 9] {
    for (row, white) in matrix.as_chunks_mut::<3>().0.iter_mut().zip(PCS_WHITE) {
        let sum: f64 = row.iter().sum();
        for value in row {
            *value *= white / sum;
        }
    }
    matrix
}

pub(super) fn multiply(matrix: &[f64; 9], rgb: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|r| (0..3).fold(0.0, |value, c| matrix[r * 3 + c].mul_add(rgb[c], value)))
}

pub(super) fn quantized_output(matrix: [f64; 9]) -> [f64; 9] {
    matrix.map(|v| (v.mul_add(4096.0, 0.5) as i32) as f64 / 4096.0)
}

pub(super) struct NativeRendering {
    pub(super) matrix: [f64; 9],
    pub(super) offset: [f64; 3],
    output: [f64; 9],
    sensor: ColorModeSensor,
    color_dq: Option<x3f_sys::ColorModeColorDq>,
    tone: Vec<f64>,
    quantized: x3f_sys::ColorModeRenderLuts,
    srgb_to_prophoto: [f64; 9],
}

impl NativeRendering {
    pub(super) fn new(
        calibration: &ColorModeCalibration,
        params: &ColorModeParameters,
        control: Control<'_>,
    ) -> Result<Self> {
        let (matrix, offset) = calibration.render_matrix(params);
        // Native RGB48 stores the output matrix with 12 fractional bits.
        let output = quantized_output(calibration.output_to_srgb);
        Ok(Self {
            matrix,
            offset,
            output,
            sensor: params.sensor,
            color_dq: calibration.color_dq.clone(),
            tone: x3f_sys::color_mode_linear_tone_lut(params, 0.0, control)?,
            quantized: x3f_sys::ColorModeRenderLuts::from_color_mode(
                params,
                0.0,
                calibration.output_to_srgb,
                2,
                control,
            )?,
            srgb_to_prophoto: mat3_mul(&xyz_d50_to_prophoto(), &srgb_to_xyz_d50()),
        })
    }

    pub(super) fn sensor_to_srgb16(&self, sensor: [f64; 3]) -> [u16; 3] {
        let indices = self
            .counts(sensor)
            .map(|count| x3f_sys::color_mode_tone_index(self.sensor, count));
        self.quantized.render_indices(indices)
    }

    pub(super) fn sensor_to_srgb8(&self, sensor: [f64; 3]) -> [u8; 3] {
        self.sensor_to_srgb16(sensor)
            .map(|v| ((u32::from(v) + 128) / 257) as u8)
    }

    pub(super) fn counts(&self, sensor: [f64; 3]) -> [f64; 3] {
        let counts = multiply(&self.matrix, sensor);
        let counts = std::array::from_fn(|c| counts[c] + self.offset[c]);
        self.color_dq
            .as_ref()
            .map_or(counts, |dq| dq.apply(counts).map(f64::from))
    }

    fn render(&self, sensor: [f64; 3], table: bool) -> [f64; 3] {
        let counts = self.counts(sensor);
        if !table {
            let indices = counts.map(|count| x3f_sys::color_mode_tone_index(self.sensor, count));
            return self
                .quantized
                .render_linear_indices(indices)
                .map(|v| f64::from(v) / 65535.0);
        }
        let tone = std::array::from_fn(|c| {
            let count = counts[c];
            let count = if self.sensor == ColorModeSensor::Merrill {
                count.clamp(0.0, 32766.0)
            } else {
                count
            };
            // The full ProPhoto cube includes colors outside the sensor gamut.
            let count = count.clamp(0.0, 65535.0);
            let index = count as usize;
            let fraction = count - index as f64;
            self.tone[index] * (1.0 - fraction) + self.tone[(index + 1).min(65535)] * fraction
        });
        multiply(&self.output, tone).map(|v| v.clamp(0.0, 1.0))
    }

    pub(super) fn sensor_to_prophoto(&self, sensor: [f64; 3]) -> [f64; 3] {
        multiply(&self.srgb_to_prophoto, self.render(sensor, true)).map(|v| v.clamp(0.0, 1.0))
    }

    #[cfg(test)]
    pub(super) fn sensor_to_prophoto_exact(&self, sensor: [f64; 3]) -> [f64; 3] {
        multiply(&self.srgb_to_prophoto, self.render(sensor, false)).map(|v| v.clamp(0.0, 1.0))
    }

    #[cfg(test)]
    pub(super) fn prophoto_to_srgb(&self) -> [f64; 9] {
        mat3_inverse(&self.srgb_to_prophoto)
    }
}

pub(super) fn srgb_encode(v: f64) -> f64 {
    if v <= 0.0031308 {
        12.92 * v
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    }
}

pub(super) fn srgb_decode(v: f64) -> f64 {
    if v <= 0.04045 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smooth_tone_matches_native_index_at_integer_counts() {
        let identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        for sensor in [ColorModeSensor::Merrill, ColorModeSensor::Quattro] {
            let params = ColorModeParameters {
                mode: x3f_sys::ColorMode::Standard,
                sensor,
                as_shot_fallback: false,
                creative_matrix: None,
                contrast: 0.0,
                tone: x3f_sys::ColorModeToneParameters {
                    start: -1.17,
                    end: 1.65,
                    steepness: 3.0,
                    steepness2: 1.7,
                    breakpoint: 0.1,
                    scale: 1.0,
                },
                inverse_gamma: 1.0 / 2.2,
                pixel_saturation: 4095.0,
            };
            let calibration = ColorModeCalibration {
                sensor_to_srgb: identity,
                wb_gain: [1.0; 3],
                normalized_to_counts: [1.0; 3],
                count_offset: [0.0; 3],
                iso_gain: 1.0,
                output_to_srgb: identity,
                color_dq: None,
            };
            let rendering = NativeRendering::new(&calibration, &params, Control::none()).unwrap();
            for count in -1..=65536 {
                let input = [f64::from(count); 3];
                let index = x3f_sys::color_mode_tone_index(sensor, input[0]);
                let expected = rendering.tone[index];
                assert_eq!(rendering.render(input, true), [expected; 3]);
                let linear = (expected * 65535.0 + 0.5) as u16;
                assert_eq!(
                    rendering.render(input, false),
                    [f64::from(linear) / 65535.0; 3]
                );
                let encoded = rendering.quantized.render_indices([index; 3]);
                assert_eq!(
                    rendering.sensor_to_srgb8(input),
                    encoded.map(|v| ((u32::from(v) + 128) / 257) as u8)
                );
            }
            for count in [-1e9, 1e9] {
                let expected = rendering.tone[x3f_sys::color_mode_tone_index(sensor, count)];
                assert_eq!(rendering.render([count; 3], true), [expected; 3]);
                let linear = (expected * 65535.0 + 0.5) as u16;
                assert_eq!(
                    rendering.render([count; 3], false),
                    [f64::from(linear) / 65535.0; 3]
                );
            }
        }
    }
}
