//! Classic profile coordinates for the native count-domain renderer.

use x3f_sys::{ColorModeCalibration, ColorModeParameters, Error, Result};

use super::metadata::{mat3_diag, mat3_inverse, mat3_mul};
use super::rgb_table::{
    multiply, quantized_output, srgb_to_xyz_d50, xyz_d50_to_prophoto, PCS_WHITE,
};
use super::rgb_table::{srgb_decode, srgb_encode, NativeRendering};

pub(super) fn serialized(matrix: [f64; 9]) -> [f64; 9] {
    matrix.map(|v| (f64::from(v as f32) * 10_000.0).round() / 10_000.0)
}

pub(super) fn sdk_color(mut matrix: [f64; 9]) -> [f64; 9] {
    let maximum = multiply(&matrix, PCS_WHITE).into_iter().fold(0.0, f64::max);
    if maximum > 0.0 && !(0.99..=1.01).contains(&maximum) {
        matrix = matrix.map(|v| v / maximum);
    }
    matrix.map(|v| (v * 10_000.0).round() / 10_000.0)
}

pub(super) fn sdk_forward(mut matrix: [f64; 9]) -> Result<[f64; 9]> {
    for (row, white) in matrix.as_chunks_mut::<3>().0.iter_mut().zip(PCS_WHITE) {
        let sum: f64 = row.iter().sum();
        if !sum.is_finite() || sum <= 0.0 {
            return Err(Error::InvalidData("invalid classic profile white point"));
        }
        for value in row {
            *value *= white / sum;
        }
    }
    Ok(matrix)
}

pub(super) fn white_xy(color: [f64; 9], neutral: [f64; 3]) -> Result<[f64; 2]> {
    let xyz = multiply(&mat3_inverse(&sdk_color(serialized(color))), neutral);
    let sum: f64 = xyz.iter().sum();
    let xy = [xyz[0] / sum, xyz[1] / sum].map(|v| (v * 1_000_000_000.0).round() / 1_000_000_000.0);
    if xy.iter().any(|v| !v.is_finite() || *v <= 0.0) || xy[0] + xy[1] >= 1.0 {
        return Err(Error::InvalidData("invalid classic profile as-shot white"));
    }
    Ok(xy)
}

pub(super) fn xy_xyz(xy: [f64; 2]) -> [f64; 3] {
    [xy[0] / xy[1], 1.0, (1.0 - xy[0] - xy[1]) / xy[1]]
}

pub(super) fn legacy_color(color: [f64; 9], white: [f64; 3], xy: [f64; 2]) -> Result<[f64; 9]> {
    let color = sdk_color(serialized(color));
    let xy = xy.map(|v| (v * 1_000_000.0).round() / 1_000_000.0);
    let current = multiply(&color, xy_xyz(xy));
    let maximum = current.into_iter().fold(0.0, f64::max);
    if current.iter().any(|v| !v.is_finite() || *v <= 0.0) {
        return Err(Error::InvalidData(
            "legacy profile has no positive as-shot white",
        ));
    }
    let diagonal = std::array::from_fn(|c| white[c] / (current[c] / maximum));
    Ok(serialized(sdk_color(mat3_mul(
        &mat3_diag(&diagonal),
        &color,
    ))))
}

pub(super) struct Matrices {
    pub(super) color: [f64; 9],
    pub(super) forward: [f64; 9],
    #[cfg(test)]
    pub(super) diagonal: [f64; 3],
    #[cfg(test)]
    pub(super) white: [f64; 3],
    pub(super) camera_to_prophoto: [f64; 9],
}

impl Matrices {
    #[cfg(test)]
    pub(super) fn input(&self, sensor: [f64; 3], exposure: f64) -> [f64; 3] {
        let sensor = std::array::from_fn(|c| sensor[c].min(self.white[c]));
        multiply(&self.camera_to_prophoto, sensor)
            .map(|v| (v.clamp(0.0, 1.0) * exposure.exp2()).clamp(0.0, 1.0))
    }

    pub(super) fn new(
        native: &ColorModeCalibration,
        params: &ColorModeParameters,
        color0: [f64; 9],
        neutral: [f64; 3],
        xy: [f64; 2],
        count_scale: f64,
    ) -> Result<Self> {
        let (counts, _) = native.render_matrix(params);
        let basis = mat3_mul(&srgb_to_xyz_d50(), &quantized_output(native.output_to_srgb));
        let absolute =
            mat3_mul(&basis, &mat3_mul(&counts, &mat3_diag(&neutral))).map(|v| v / count_scale);
        Self::from_absolute(absolute, color0, xy)
    }

    pub(super) fn from_absolute(
        absolute: [f64; 9],
        color0: [f64; 9],
        xy: [f64; 2],
    ) -> Result<Self> {
        let diagonal = multiply(&super::profiles::checked_inverse(&absolute)?, PCS_WHITE);
        if diagonal.iter().any(|v| !v.is_finite() || *v <= 0.0) {
            return Err(Error::InvalidData(
                "classic Color Mode has no positive camera white",
            ));
        }
        let color = serialized(sdk_color(mat3_mul(&mat3_diag(&diagonal), &color0)));
        let forward = serialized(sdk_forward(mat3_mul(&absolute, &mat3_diag(&diagonal)))?);
        // Adobe rounds AsShotWhiteXY to six decimal places on read.
        let xy = xy.map(|v| (v * 1_000_000.0).round() / 1_000_000.0);
        let unpinned = multiply(&sdk_color(color), xy_xyz(xy));
        let maximum = unpinned.into_iter().fold(0.0, f64::max);
        let white = unpinned.map(|v| v / maximum);
        if white
            .iter()
            .any(|v| !v.is_finite() || *v < 0.001 || *v > 1.0)
        {
            return Err(Error::InvalidData(
                "classic Color Mode camera white reaches the Adobe pin",
            ));
        }
        if color
            .iter()
            .chain(&forward)
            .any(|v| !v.is_finite() || v.abs() > i32::MAX as f64 / 10_000.0)
        {
            return Err(Error::InvalidData(
                "classic profile matrix cannot fit metadata",
            ));
        }
        let camera_to_prophoto = mat3_mul(
            &xyz_d50_to_prophoto(),
            &mat3_mul(&sdk_forward(forward)?, &mat3_diag(&white.map(|v| 1.0 / v))),
        );
        Ok(Self {
            color,
            forward,
            #[cfg(test)]
            diagonal,
            #[cfg(test)]
            white,
            camera_to_prophoto,
        })
    }
}

pub(super) struct ToneCurve {
    pub(super) knots: Vec<f32>,
    table: Vec<f32>,
}

impl ToneCurve {
    pub(super) fn fit(mut render: impl FnMut(f64) -> f64) -> Result<Self> {
        // Align knots with Adobe's 4096-interval tone table to retain plateaus.
        let mut knots = Vec::with_capacity(8194);
        let mut previous = 0.0;
        for i in 0..=4096 {
            let x = i as f32 / 4096.0;
            let y = render(f64::from(x)) as f32;
            let y = match i {
                0 => 0.0,
                4096 => 1.0,
                _ => y,
            };
            if !y.is_finite() {
                return Err(Error::InvalidData("non-finite classic tone sample"));
            }
            // Native FP32 tone samples can fall slightly near white.
            let y = y.clamp(previous, 1.0);
            knots.extend([x, y]);
            previous = y;
        }
        Self::from_knots(knots)
    }

    pub(super) fn from_knots(knots: Vec<f32>) -> Result<Self> {
        if knots.len() != 8194 {
            return Err(Error::InvalidData(
                "classic tone curve needs 4097 fixed-grid points",
            ));
        }
        let points = knots.as_chunks::<2>().0;
        if points.iter().enumerate().any(|(i, p)| {
            p[0] != i as f32 / 4096.0 || !p[1].is_finite() || !(0.0..=1.0).contains(&p[1])
        }) || points[0][1] != 0.0
            || points[4096][1] != 1.0
            || points.windows(2).any(|p| p[1][1] < p[0][1])
        {
            return Err(Error::InvalidData("invalid classic fixed-grid tone curve"));
        }
        // SDK Initialize samples these exact knots with subSample=false.
        let table = points.iter().map(|p| p[1]).collect();
        Ok(Self { knots, table })
    }

    #[cfg(test)]
    fn evaluate(&self, value: f64) -> f64 {
        let position = value.clamp(0.0, 1.0) as f32 * 4096.0;
        let index = (position as usize).min(4095);
        let fraction = position - index as f32;
        f64::from(self.table[index] * (1.0 - fraction) + self.table[index + 1] * fraction)
    }

    fn inverse(&self, value: f64) -> f64 {
        let value = value as f32;
        let j = self.table.partition_point(|&v| v < value).clamp(1, 4096);
        let lower = self.table[j - 1];
        let upper = self.table[j];
        let fraction = if upper > lower {
            ((value - lower) / (upper - lower)).clamp(0.0, 1.0)
        } else {
            0.0
        };
        (j as f64 - 1.0 + f64::from(fraction)) / 4096.0
    }

    #[cfg(test)]
    pub(super) fn apply(&self, rgb: [f64; 3]) -> [f64; 3] {
        let rgb = rgb.map(|v| v.clamp(0.0, 1.0) as f32);
        let low = rgb.into_iter().fold(1.0, f32::min);
        let high = rgb.into_iter().fold(0.0, f32::max);
        let a = self.evaluate(f64::from(low)) as f32;
        let b = self.evaluate(f64::from(high)) as f32;
        rgb.map(|v| {
            f64::from(if high == low {
                a
            } else {
                a + (b - a) * (v - low) / (high - low)
            })
        })
    }

    fn unapply(&self, rgb: [f64; 3]) -> [f64; 3] {
        let low = rgb.into_iter().fold(1.0, f64::min);
        let high = rgb.into_iter().fold(0.0, f64::max);
        let a = self.inverse(low);
        let b = self.inverse(high);
        rgb.map(|v| {
            if high == low {
                a
            } else {
                a + (b - a) * (v - low) / (high - low)
            }
        })
    }
}

pub(super) fn to_hsv(rgb: [f64; 3]) -> [f64; 3] {
    let max = rgb.into_iter().fold(0.0, f64::max);
    let min = rgb.into_iter().fold(1.0, f64::min);
    let delta = max - min;
    let hue = if delta == 0.0 {
        0.0
    } else if max == rgb[0] {
        ((rgb[1] - rgb[2]) / delta).rem_euclid(6.0)
    } else if max == rgb[1] {
        (rgb[2] - rgb[0]) / delta + 2.0
    } else {
        (rgb[0] - rgb[1]) / delta + 4.0
    };
    [hue, if max > 0.0 { delta / max } else { 0.0 }, max]
}

pub(super) fn from_hsv([h, s, v]: [f64; 3]) -> [f64; 3] {
    let h = h.rem_euclid(6.0);
    let fraction = h - h.floor();
    let a = v * (1.0 - s);
    let b = v * (1.0 - s * fraction);
    let c = v * (1.0 - s * (1.0 - fraction));
    match h as usize {
        0 => [v, c, a],
        1 => [b, v, a],
        2 => [a, v, c],
        3 => [a, b, v],
        4 => [c, a, v],
        _ => [v, a, b],
    }
}

pub(super) const LOOK_DIMS: [usize; 3] = [72, 17, 65];

pub(super) struct ClassicFit {
    pub(super) curve: ToneCurve,
    pub(super) map: Vec<f32>,
    #[cfg(test)]
    pub(super) capped: usize,
    #[cfg(test)]
    pub(super) gray_endpoint: f64,
}

impl ClassicFit {
    pub(super) fn new(
        matrices: &Matrices,
        native: &NativeRendering,
        highlight_scale: f64,
        exposure: f64,
        control: x3f_sys::Control<'_>,
    ) -> Result<Self> {
        // LinearRaw has Stage3Gain=1. Negative total EV needs Adobe's exposure tone.
        if !exposure.is_finite()
            || exposure < 0.0
            || !highlight_scale.is_finite()
            || highlight_scale <= 0.0
        {
            return Err(Error::InvalidData(
                "classic profile requires finite nonnegative total exposure",
            ));
        }
        let inverse = super::profiles::checked_inverse(&matrices.camera_to_prophoto)?
            .map(|v| v * highlight_scale / exposure.exp2());
        if inverse.iter().any(|v| !v.is_finite()) {
            return Err(Error::InvalidData("invalid classic profile inverse"));
        }
        let render = |rgb| native.sensor_to_prophoto(multiply(&inverse, rgb));
        let gray = |value| {
            let rgb = render([value; 3]);
            0.2880 * rgb[0] + 0.7119 * rgb[1] + 0.0001 * rgb[2]
        };
        #[cfg(test)]
        let gray_endpoint = gray(1.0);
        let curve = ToneCurve::fit(gray)?;
        let [h_count, s_count, v_count] = LOOK_DIMS;
        let mut map = Vec::with_capacity(h_count * s_count * v_count * 3);
        #[cfg(test)]
        let mut capped = 0;
        for v in 0..v_count {
            control.check()?;
            let encoded = v as f64 / (v_count - 1) as f64;
            let value = srgb_decode(encoded);
            for h in 0..h_count {
                let hue = h as f64 * 6.0 / h_count as f64;
                for s in 0..s_count {
                    let saturation = s as f64 / (s_count - 1) as f64;
                    if s == 0 || v == 0 {
                        map.extend_from_slice(&[0.0, 1.0, 1.0]);
                        continue;
                    }
                    let input = from_hsv([hue, saturation, value]);
                    let target = to_hsv(curve.unapply(render(input)));
                    let hue_shift = ((target[0] - hue) * 60.0 + 180.0).rem_euclid(360.0) - 180.0;
                    let sat_scale = target[1] / saturation;
                    #[cfg(test)]
                    if sat_scale > 8.0 {
                        capped += 1;
                    }
                    let value_scale = srgb_encode(target[2]) / encoded;
                    let entry = [hue_shift, sat_scale.min(8.0), value_scale];
                    if entry.iter().any(|v| !v.is_finite()) {
                        return Err(Error::InvalidData("non-finite classic profile look table"));
                    }
                    map.extend(entry.map(|v| v as f32));
                }
            }
        }
        Ok(Self {
            curve,
            map,
            #[cfg(test)]
            capped,
            #[cfg(test)]
            gray_endpoint,
        })
    }

    #[cfg(test)]
    pub(super) fn apply(&self, input: [f64; 3]) -> [f64; 3] {
        let [h, s, v] = to_hsv(input.map(|v| v.clamp(0.0, 1.0)));
        let [hn, sn, vn] = LOOK_DIMS;
        let position = [
            h * hn as f64 / 6.0,
            s * (sn - 1) as f64,
            srgb_encode(v) * (vn - 1) as f64,
        ];
        let index = [
            position[0] as usize % hn,
            (position[1] as usize).min(sn - 2),
            (position[2] as usize).min(vn - 2),
        ];
        let fraction: [f64; 3] = std::array::from_fn(|c| position[c] - index[c] as f64);
        let mut entry = [0.0; 3];
        for dh in 0..2 {
            for ds in 0..2 {
                for dv in 0..2 {
                    let weight = [dh, ds, dv]
                        .into_iter()
                        .enumerate()
                        .map(|(i, j)| {
                            if j == 0 {
                                1.0 - fraction[i]
                            } else {
                                fraction[i]
                            }
                        })
                        .product::<f64>();
                    let offset =
                        (((index[2] + dv) * hn + (index[0] + dh) % hn) * sn + index[1] + ds) * 3;
                    for (c, value) in entry.iter_mut().enumerate() {
                        *value += weight * f64::from(self.map[offset + c]);
                    }
                }
            }
        }
        let rgb = from_hsv([
            h + entry[0] / 60.0,
            (s * entry[1]).clamp(0.0, 1.0),
            srgb_decode((srgb_encode(v) * entry[2]).clamp(0.0, 1.0)),
        ]);
        self.curve.apply(rgb)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sdk_tone_preserves_channel_order_and_has_a_matching_inverse() {
        let curve = ToneCurve::fit(|x| x * x).unwrap();
        for rgb in [
            [0.2, 0.5, 0.9],
            [0.9, 0.5, 0.2],
            [0.5, 0.2, 0.9],
            [0.5; 3],
            [0.0, 0.0, 1.0],
        ] {
            let output = curve.apply(rgb);
            let low = rgb.into_iter().fold(1.0, f64::min);
            let high = rgb.into_iter().fold(0.0, f64::max);
            for (actual, value) in output.into_iter().zip(rgb) {
                let expected = if high == low {
                    low * low
                } else {
                    low * low + (high * high - low * low) * (value - low) / (high - low)
                };
                assert!((actual - expected).abs() < 2e-7);
            }
            for (actual, expected) in curve.unapply(output).into_iter().zip(rgb) {
                assert!((actual - expected).abs() < 2e-7);
            }
        }
        let curve = ToneCurve::fit(|x| (2.0 * x).min(1.0)).unwrap();
        assert_eq!(curve.knots.len(), 8194);
        for i in 0..=4096 {
            assert_eq!(curve.table[i], ((2.0 * i as f64 / 4096.0).min(1.0)) as f32);
        }
        assert!(ToneCurve::from_knots(vec![0.0, 0.0, 0.0, 1.0]).is_err());
        assert!(ToneCurve::from_knots(vec![0.0, 0.5, 1.0, 0.0]).is_err());
    }

    #[test]
    fn fitted_tone_bounds_samples_and_flattens_reversals() {
        let sample = |x: f64| match (x * 4096.0) as usize {
            1 => -0.1,
            2048 => 0.49,
            4095 => 1.01,
            _ => x,
        };
        let raw: Vec<_> = (0..=4096)
            .flat_map(|i| {
                let x = i as f32 / 4096.0;
                [x, sample(f64::from(x)) as f32]
            })
            .collect();
        assert!(ToneCurve::from_knots(raw).is_err());
        let curve = ToneCurve::fit(sample).unwrap();
        for i in 0..=4096 {
            let expected = match i {
                1 => 0.0,
                2048 => 2047.0 / 4096.0,
                4095 => 1.0,
                _ => i as f32 / 4096.0,
            };
            assert_eq!(curve.table[i], expected);
        }
        assert!((curve.inverse(0.5) - 0.5).abs() < 1.0 / 4096.0);
    }

    #[test]
    fn fitted_tone_rejects_non_finite_samples() {
        for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(ToneCurve::fit(|x| if x == 0.5 { invalid } else { x }).is_err());
        }
    }

    #[test]
    fn camera_white_clip_precedes_matrix_and_exposure() {
        let matrix = Matrices {
            color: [0.0; 9],
            forward: [0.0; 9],
            diagonal: [1.0; 3],
            white: [0.2, 0.6, 1.0],
            camera_to_prophoto: [2.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        };
        assert_eq!(matrix.input([0.4, 0.5, 0.9], 0.0), [0.0, 0.5, 0.9]);
        assert_eq!(matrix.input([0.4, 0.5, 0.9], 1.0), [0.0, 1.0, 1.0]);
        let identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        assert!(Matrices::from_absolute(identity.map(|v| -v), identity, [1.0 / 3.0; 2]).is_err());
        assert!(
            Matrices::from_absolute(identity, mat3_diag(&[0.00001, 1.0, 1.0]), [1.0 / 3.0; 2])
                .is_err()
        );
    }
}
