//! Camera coordinates shared by DNG white balance and all embedded profiles.
//!
//! Digital ISO gain has a scalar exposure component and a relative channel
//! component. The latter belongs in the color metadata: baking it into the
//! raster would clip otherwise usable samples at the published white level.

use super::metadata::{mat3_diag, mat3_mul};
use crate::Reader;

const WB_CALIBRATION: &str = "Overcast";

pub(super) struct ColorCalibration {
    pub(super) neutral: [f64; 3],
    pub(super) digital_gain_ev: f64,
    calibration_diagonal: [f64; 3],
}

impl ColorCalibration {
    pub(super) fn new(reader: &Reader, wb: &str) -> Option<Self> {
        Self::from_gains(
            reader.dng_gain(Some(wb))?,
            reader.dng_gain(Some(WB_CALIBRATION))?,
            reader.dng_digital_iso_gain(),
        )
    }

    fn from_gains(
        shot_gain: [f64; 3],
        calibration_gain: [f64; 3],
        digital_gain: [f64; 3],
    ) -> Option<Self> {
        if shot_gain
            .iter()
            .chain(&calibration_gain)
            .chain(&digital_gain)
            .any(|&gain| !gain.is_finite() || gain <= 0.0)
        {
            return None;
        }

        // Keep uniform gains exactly neutral, including the absent-gain
        // and Merrill cases supplied by dng_digital_iso_gain.
        let mean = if digital_gain[0] == digital_gain[1] && digital_gain[1] == digital_gain[2] {
            digital_gain[0]
        } else {
            (digital_gain[0] * digital_gain[1] * digital_gain[2]).cbrt()
        };
        if !mean.is_finite() || mean <= 0.0 {
            return None;
        }

        let relative_gain = digital_gain.map(|gain| gain / mean);
        let neutral = std::array::from_fn(|i| 1.0 / (shot_gain[i] * relative_gain[i]));
        let calibration_diagonal =
            std::array::from_fn(|i| 1.0 / (calibration_gain[i] * relative_gain[i]));
        if neutral
            .iter()
            .chain(&calibration_diagonal)
            .any(|&value| !value.is_finite() || value <= 0.0)
        {
            return None;
        }

        Some(Self {
            neutral,
            digital_gain_ev: mean.log2(),
            calibration_diagonal,
        })
    }

    /// Absorb CameraCalibration into ColorMatrix, so readers need only the
    /// mandatory matrix tag. This is a row scale because ColorMatrix maps
    /// XYZ into camera coordinates. A diagonal calibration cancels from the
    /// DNG ForwardMatrix transform; ForwardMatrix must remain unchanged.
    pub(super) fn fold_color_matrix(&self, matrix: &[f64; 9]) -> [f64; 9] {
        mat3_mul(&mat3_diag(&self.calibration_diagonal), matrix)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::output::dng::metadata::{bradford_d65_to_d50, mat3_diag, mat3_inverse, mat3_mul};

    const D65: [f64; 3] = [0.95047, 1.0, 1.08883];
    const D50: [f64; 3] = [0.96422, 1.0, 0.82521];
    const SHOT_GAIN: [f64; 3] = [1.12, 1.05, 0.99];
    const CALIBRATION_GAIN: [f64; 3] = [1.09, 1.05, 1.02];

    fn bmt_to_xyz() -> [f64; 9] {
        // Invertible Foveon-like matrix with substantial channel mixing.
        [
            1.02,
            -0.19,
            D65[0] - 0.83,
            -0.12,
            1.29,
            -0.17,
            0.45,
            -2.3,
            D65[2] + 1.85,
        ]
    }

    fn mul_vector(matrix: &[f64; 9], vector: &[f64; 3]) -> [f64; 3] {
        std::array::from_fn(|r| (0..3).map(|c| matrix[3 * r + c] * vector[c]).sum())
    }

    fn assert_close<const N: usize>(actual: [f64; N], expected: [f64; N], tolerance: f64) {
        for i in 0..N {
            assert!(
                (actual[i] - expected[i]).abs() < tolerance,
                "entry {i}: {} != {}",
                actual[i],
                expected[i]
            );
        }
    }

    #[test]
    fn uniform_digital_gain_changes_only_exposure() {
        let unit = ColorCalibration::from_gains(SHOT_GAIN, CALIBRATION_GAIN, [1.0; 3]).unwrap();
        let doubled = ColorCalibration::from_gains(SHOT_GAIN, CALIBRATION_GAIN, [2.0; 3]).unwrap();
        assert_eq!(unit.neutral, SHOT_GAIN.map(|gain| 1.0 / gain));
        assert_eq!(unit.neutral, doubled.neutral);
        assert_eq!(unit.calibration_diagonal, doubled.calibration_diagonal);
        assert_eq!(unit.digital_gain_ev, 0.0);
        assert_eq!(doubled.digital_gain_ev, 1.0);
    }

    #[test]
    fn quattro_gain_preserves_relative_channel_correction() {
        let color =
            ColorCalibration::from_gains(SHOT_GAIN, CALIBRATION_GAIN, [4.0, 4.0, 2.0]).unwrap();
        let relative: [f64; 3] = std::array::from_fn(|i| 1.0 / (color.neutral[i] * SHOT_GAIN[i]));
        assert_close(
            relative,
            [2.0_f64.cbrt(), 2.0_f64.cbrt(), 0.5_f64.powf(2.0 / 3.0)],
            1e-12,
        );
        assert!((relative[0] * relative[1] * relative[2] - 1.0).abs() < 1e-12);
        assert!((color.digital_gain_ev - 5.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn corrected_physical_neutral_maps_to_d50() {
        let digital_gain = [4.0, 4.0, 2.0];
        let color =
            ColorCalibration::from_gains(SHOT_GAIN, CALIBRATION_GAIN, digital_gain).unwrap();
        let forward = mat3_mul(&bradford_d65_to_d50(), &bmt_to_xyz());
        let physical_neutral = std::array::from_fn(|i| 1.0 / (SHOT_GAIN[i] * digital_gain[i]));
        let white_balance = mat3_diag(&color.neutral.map(|value| 1.0 / value));
        let output = mul_vector(&mat3_mul(&forward, &white_balance), &physical_neutral);
        assert_close(output.map(|value| value / output[1]), D50, 1e-3);

        // Dropping the relative gain produces a large cast, even when its
        // common exposure component is restored by BaselineExposure.
        let legacy = mul_vector(
            &mat3_mul(&forward, &mat3_diag(&SHOT_GAIN)),
            &physical_neutral,
        );
        assert!((legacy[2] / legacy[1] - D50[2]).abs() > 0.5);
    }

    #[test]
    fn folded_calibration_preserves_white_balance_and_both_transforms() {
        let color =
            ColorCalibration::from_gains(SHOT_GAIN, CALIBRATION_GAIN, [4.0, 4.0, 2.0]).unwrap();
        let matrix = mat3_inverse(&bmt_to_xyz());
        let calibration = mat3_diag(&color.calibration_diagonal);
        let xyz_to_camera = mat3_mul(&calibration, &matrix);
        let folded = color.fold_color_matrix(&matrix);
        assert_close(folded, xyz_to_camera, 1e-12);

        let forward = mat3_mul(&bradford_d65_to_d50(), &bmt_to_xyz());
        let individual_to_reference = mat3_inverse(&calibration);
        // Capture WB and two user-selected white points. The inverse CM
        // must infer the same xy; the ForwardMatrix transform must also
        // survive removing CameraCalibration at each selection.
        for neutral in [
            color.neutral,
            mul_vector(&folded, &[1.1, 1.0, 0.6]),
            mul_vector(&folded, &[0.9, 1.0, 1.2]),
        ] {
            assert_close(
                mul_vector(&mat3_inverse(&folded), &neutral),
                mul_vector(&mat3_inverse(&xyz_to_camera), &neutral),
                1e-12,
            );
            let reference_neutral = mul_vector(&individual_to_reference, &neutral);
            let explicit_transform = mat3_mul(
                &mat3_mul(
                    &forward,
                    &mat3_diag(&reference_neutral.map(|value| 1.0 / value)),
                ),
                &individual_to_reference,
            );
            let folded_transform =
                mat3_mul(&forward, &mat3_diag(&neutral.map(|value| 1.0 / value)));
            assert_close(folded_transform, explicit_transform, 1e-12);
        }
    }

    #[test]
    fn invalid_gains_do_not_produce_invalid_dng_color_tags() {
        for invalid in [0.0, -1.0, f64::INFINITY, f64::NAN] {
            assert!(
                ColorCalibration::from_gains(SHOT_GAIN, CALIBRATION_GAIN, [invalid, 1.0, 1.0])
                    .is_none()
            );
        }
    }
}
