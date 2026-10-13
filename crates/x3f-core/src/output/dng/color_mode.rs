//! Build a DNG camera profile from the color-mode settings stored in the X3F.
//!
//! DP1, DP2 and DP3 Merrill use the same calculation with each file's calibration.
//! The profile contains a color lookup table and a tone curve, leaving the raw
//! samples untouched. The lookup table accounts for the color and exposure changes
//! that Adobe applies before using the profile. It also includes ColorDQ, Sigma's
//! per-pixel color correction, which does not use neighboring pixels.
//!
//! The file's sRGB or Adobe RGB setting chooses the color conversion used to build
//! the profile. It does not choose the DNG reader's output space. Denoising, repair,
//! color shading and recovery still run before rendering. This profile does not
//! reproduce the full Sigma Photo Pro pipeline.

mod color_dq;
mod look;
mod mode;
mod tone;

use x3f_sys::{self as sys, Control};

use super::{
    color::ColorCalibration,
    metadata::{mat3_diag, mat3_inverse, mat3_mul},
    profiles, tags,
    tiff_writer::{DirectoryWriter, Value},
};
use crate::{ColorMode, Error, Reader};
use color_dq::ColorDq;
use mode::ModeParameters;
use tone::{curve_at, exposure_tone, profile_curve, SATURATION};

// Rendering setup

// The Sigma Photo Pro setup used by this recipe doubles brightness before its
// color correction and tone curve. Store that one-stop gain in the profile so
// the raw samples keep their original brightness and highlight headroom.
const SETUP_GAIN: f64 = 2.0;

// ForwardMatrix describes camera-to-color conversion, and BaselineExposure
// sets the DNG's default brightness. Both tags store rounded numbers. Build the
// profile with those same numbers, since they are what the reader actually uses.
const TAG_DENOMINATOR: i32 = 10000;

// Adobe evaluates the profile in a working space with ProPhoto RGB primaries,
// also called ROMM RGB. D50 is its reference white, roughly 5000 K daylight.
// Scale the matrix rows so white agrees between the camera and that working space.
const PCS_WHITE: [f64; 3] = [0.3457 / 0.3585, 1.0, (1.0 - 0.3457 - 0.3585) / 0.3585];
const ROMM: [f64; 9] = [
    0.7977, 0.1352, 0.0313, 0.2880, 0.7119, 0.0001, 0.0, 0.0, 0.8249,
];

// Sigma Photo Pro uses these coefficients to convert Adobe RGB to ROMM RGB
// with no saturation adjustment. Its source and destination IDs are 3 and 11.
// The coefficients are stored as integers divided by 4096. Keep that rounding,
// including the slightly uneven row sums, to preserve its color conversion.
// This matrix does not replace the camera's own calibration.
const MATRIX_SCALE: f64 = 4096.0;
const ADOBE_TO_ROMM: [f64; 9] = [
    3032.0 / MATRIX_SCALE,
    464.0 / MATRIX_SCALE,
    601.0 / MATRIX_SCALE,
    564.0 / MATRIX_SCALE,
    3412.0 / MATRIX_SCALE,
    120.0 / MATRIX_SCALE,
    97.0 / MATRIX_SCALE,
    302.0 / MATRIX_SCALE,
    3697.0 / MATRIX_SCALE,
];

// Source ID 2 is Sigma Photo Pro's sRGB conversion. Choose it when the X3F says
// sRGB, even if the DNG reader will render to a different output color space.
const SRGB_TO_ROMM: [f64; 9] = [
    2168.0 / MATRIX_SCALE,
    1352.0 / MATRIX_SCALE,
    576.0 / MATRIX_SCALE,
    403.0 / MATRIX_SCALE,
    3578.0 / MATRIX_SCALE,
    115.0 / MATRIX_SCALE,
    69.0 / MATRIX_SCALE,
    482.0 / MATRIX_SCALE,
    3546.0 / MATRIX_SCALE,
];

// The sRGB transfer curve relates linear brightness to display-encoded values.
// Use it to decode Sigma's tone response and to space the DNG table's brightness
// samples. This does not apply an sRGB curve to the stored camera samples.
const SRGB_ENCODED_JOIN: f64 = 0.04045;
const SRGB_LINEAR_JOIN: f64 = 0.0031308;
const SRGB_SLOPE: f64 = 12.92;
const SRGB_OFFSET: f64 = 0.055;
const SRGB_SCALE: f64 = 1.055;
const SRGB_POWER: f64 = 2.4;

pub(super) struct CameraProfile {
    mode: ColorMode,
    camera: [f64; 9],
    output_matrix: [f64; 9],
    gain: [f64; 3],
    black: [f64; 3],
    iso: f64,
    tone: Vec<f64>,
    dq: ColorDq,
}

impl CameraProfile {
    /// Read calibration while the decoded sensor samples are still unprocessed.
    /// Use the requested white-balance preset from CAMF, the X3F's camera metadata.
    /// Auto uses the file's saved calibration rather than estimating white balance.
    pub(super) fn prepare(
        reader: &Reader,
        wb: &str,
        mode: ColorMode,
        control: Control<'_>,
    ) -> Result<Self, Error> {
        control.check()?;
        if !matches!(
            reader.dng_prop("CAMMODEL").as_deref(),
            Some("SIGMA DP1 Merrill" | "SIGMA DP2 Merrill" | "SIGMA DP3 Merrill")
        ) {
            return Err(invalid("requires DP Merrill calibration"));
        }
        let mut camera = reader
            .dng_camf_wb_matrix_3x3("WhiteBalanceColorCorrections", wb)
            .ok_or_else(|| invalid("missing selected white-balance matrix"))?;
        let parameters = ModeParameters::read(reader, mode)?;
        if mode != ColorMode::Standard {
            // Sigma Photo Pro combines white-balance and color-mode calibration
            // before ColorDQ: combined = camera matrix * mode matrix. Keep that
            // multiplication order. Unlike Sigma's processing, the embedded profile
            // applies the result after the converter's denoising and recovery.
            camera = mat3_mul(&camera, &parameters.matrix);
        }
        let output_matrix = source_to_romm(reader.dng_prop("COLORSPACE").as_deref())?;
        let gain = reader
            .dng_gain(Some(wb))
            .ok_or_else(|| invalid("missing gain"))?;
        let capture = reader
            .dng_camf_float("CaptureISO")
            .ok_or_else(|| invalid("missing capture ISO"))?;
        let sensor = reader
            .dng_camf_float("SensorISO")
            .ok_or_else(|| invalid("missing sensor ISO"))?;
        let iso = capture / sensor;
        let mut area: sys::x3f_area16_t = unsafe { std::mem::zeroed() };
        let mut black = [0.0; 3];
        let mut deviation = [0.0; 3];
        let mut maximum = [0; 3];
        let mut amplitudes = [0.0; 3];
        // SAFETY: all reads use the caller's loaded reader, with bounded output arrays.
        let valid = unsafe {
            sys::x3f_image_area(reader.x3f.as_ptr(), &mut area) != 0
                && sys::get_black_level(
                    reader.x3f.as_ptr(),
                    &mut area,
                    1, // Match how preprocessing maps the covered black-reference pixels.
                    3, // Merrill has three measured layers.
                    black.as_mut_ptr(),
                    deviation.as_mut_ptr(),
                ) != 0
                && sys::x3f_get_max_raw(reader.x3f.as_ptr(), maximum.as_mut_ptr()) != 0
                && sys::x3f_get_camf_matrix(
                    reader.x3f.as_ptr(),
                    c"ColorDQCamRGB".as_ptr() as *mut _,
                    3,
                    0,
                    0,
                    sys::matrix_type_t_M_FLOAT,
                    amplitudes.as_mut_ptr().cast(),
                ) != 0
        };
        if !valid || maximum != [SATURATION as u32; 3] {
            return Err(invalid("missing calibration or unverified raw scale"));
        }
        if camera
            .iter()
            .chain(&gain)
            .chain(&black)
            .any(|v| !v.is_finite())
            || gain.iter().any(|v| *v <= 0.0)
            || black.iter().any(|v| !(0.0..SATURATION as f64).contains(v))
            || !iso.is_finite()
            || iso <= 0.0
        {
            return Err(invalid("invalid calibration"));
        }
        let tone = parameters.tone;
        // Sigma computes the ISO ratio with 32-bit floats to set ColorDQ strength.
        // Keep that rounding. The extra one-stop gain brightens the values fed into
        // ColorDQ, but it must not also increase the correction's strength.
        let dq_iso = (capture as f32 / sensor as f32).min(color_dq::MAX_ISO_GAIN) as f64;
        let dq = ColorDq::new(amplitudes.map(|v| v * dq_iso))?;
        control.check()?;
        Ok(Self {
            mode,
            camera,
            output_matrix,
            gain,
            black,
            iso,
            tone,
            dq,
        })
    }

    /// Add the camera profile using the calibration and exposure stored in the DNG.
    /// Do not change raw samples, camera calibration, exposure tags or previews.
    pub(super) fn embed(
        &self,
        reader: &Reader,
        wb: &str,
        calibration: &ColorCalibration,
        scale: f64,
        ifd: &mut DirectoryWriter,
        control: Control<'_>,
    ) -> Result<(), Error> {
        let forward = profiles::default_forward_matrix(reader, wb, calibration)
            .ok_or_else(|| invalid("missing forward matrix"))?;
        let forward = forward.map(|v| {
            let (n, d) = profiles::srational_pair(v as f64, TAG_DENOMINATOR);
            n as f64 / d as f64
        });
        let forward = normalize_white(forward);
        let neutral = calibration.neutral_tag().map(|(n, d)| n as f64 / d as f64);
        let maximum = neutral.into_iter().fold(0.0, f64::max);
        let white = neutral.map(|v| v / maximum);
        let camera_to_romm = mat3_mul(
            &mat3_mul(&mat3_inverse(&normalize_white(ROMM)), &forward),
            &mat3_diag(&white.map(|v| 1.0 / v)),
        );
        let exposure =
            super::baseline_exposure(Some(self.iso.log2()), calibration.digital_gain_ev, scale)
                .ok_or_else(|| invalid("invalid exposure"))?;
        let (n, d) = profiles::srational_pair(exposure as f32 as f64, TAG_DENOMINATOR);
        let published_exposure = n as f64 / d as f64;
        // Adobe applies positive BaselineExposure before the color lookup table.
        // Account for that existing gain rather than applying it twice. The table
        // must also account for stored highlight headroom and Sigma's one-stop gain.
        let ramp_gain = 2.0_f64.powf(published_exposure.max(0.0));
        let ratio = SETUP_GAIN * 2.0_f64.powf(published_exposure) / ramp_gain;
        let sensor_scale = std::array::from_fn(|i| {
            self.gain[i] * (SATURATION as f64 - self.black[i]) / SATURATION as f64 * ratio
        });
        let adapter = mat3_mul(
            &mat3_mul(&self.camera, &mat3_diag(&sensor_scale)),
            &mat3_inverse(&camera_to_romm),
        );
        if adapter.iter().any(|v| !v.is_finite()) {
            return Err(invalid("singular coordinate mapping"));
        }
        let neutral_scale = mul(&adapter, [1.0; 3]).iter().sum::<f64>() / 3.0;
        if !neutral_scale.is_finite() || neutral_scale <= 0.0 {
            return Err(invalid("invalid neutral coordinate scale"));
        }
        // Adobe requires the color table to leave gray brightness unchanged.
        // Put overall brightness in the tone curve instead. When BaselineExposure
        // is negative, Adobe darkens the shadows but keeps white at white. Build
        // the curve to account for that adjustment.
        let curve = profile_curve(&self.tone, neutral_scale, published_exposure);
        let effective_tone: Vec<f64> = (0..=SATURATION)
            .map(|i| {
                curve_at(
                    &curve,
                    exposure_tone(i as f64 / SATURATION as f64, published_exposure),
                )
            })
            .collect();
        let table = look::bake(
            &adapter,
            &self.output_matrix,
            &self.tone,
            &effective_tone,
            &self.dq,
            control,
        )?;
        let name = std::ffi::CString::new(self.mode.as_str()).expect("mode names contain no NUL");
        ifd.add(tags::PROFILE_NAME, Value::Ascii(name.clone()));
        ifd.add(tags::AS_SHOT_PROFILE_NAME, Value::Ascii(name));
        ifd.add(
            tags::PROFILE_LOOK_TABLE_DIMS,
            Value::Long(look::DIMS.to_vec()),
        );
        ifd.add(tags::PROFILE_LOOK_TABLE_DATA, Value::Float(table));
        ifd.add(
            tags::PROFILE_LOOK_TABLE_ENCODING,
            Value::Long(vec![look::ENCODING_SRGB]),
        );
        ifd.add(tags::PROFILE_TONE_CURVE, Value::Float(curve));
        ifd.add(
            tags::DNG_BACKWARD_VERSION,
            Value::Byte(tags::DNG_VERSION_1_4_0_0.to_vec()),
        );
        Ok(())
    }
}

fn source_to_romm(space: Option<&str>) -> Result<[f64; 9], Error> {
    match space {
        Some("sRGB") => Ok(SRGB_TO_ROMM),
        Some("AdobeRGB") => Ok(ADOBE_TO_ROMM),
        _ => Err(invalid("missing or unsupported source color space")),
    }
}

fn invalid(message: &str) -> Error {
    Error::InvalidData(format!("Color-mode rendering: {message}"))
}

fn normalize_white(m: [f64; 9]) -> [f64; 9] {
    std::array::from_fn(|i| {
        m[i] * PCS_WHITE[i / 3] / m[(i / 3) * 3..(i / 3) * 3 + 3].iter().sum::<f64>()
    })
}

fn mul(m: &[f64; 9], v: [f64; 3]) -> [f64; 3] {
    // Keep the same multiply-and-add order as the reference profile so rounding
    // does not change the generated table.
    std::array::from_fn(|r| m[r * 3].mul_add(v[0], m[r * 3 + 1].mul_add(v[1], m[r * 3 + 2] * v[2])))
}

fn srgb_decode(x: f64) -> f64 {
    if x <= SRGB_ENCODED_JOIN {
        x / SRGB_SLOPE
    } else {
        ((x + SRGB_OFFSET) / SRGB_SCALE).powf(SRGB_POWER)
    }
}

fn srgb_encode(x: f64) -> f64 {
    if x <= SRGB_LINEAR_JOIN {
        x * SRGB_SLOPE
    } else {
        SRGB_SCALE * x.powf(1.0 / SRGB_POWER) - SRGB_OFFSET
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_color_property_selects_native_routing() {
        assert_eq!(source_to_romm(Some("sRGB")).unwrap(), SRGB_TO_ROMM);
        assert_eq!(source_to_romm(Some("AdobeRGB")).unwrap(), ADOBE_TO_ROMM);
        assert_ne!(SRGB_TO_ROMM, ADOBE_TO_ROMM);
        for space in [None, Some(""), Some("ProPhoto"), Some("unknown")] {
            assert!(source_to_romm(space).is_err());
        }
    }

    #[test]
    fn camera_profiles_use_every_white_balance_in_the_file() {
        let directory = std::env::var_os("X3F_TEST_FILES")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| {
                std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../x3f_test_files")
            });
        let Ok(entries) = std::fs::read_dir(directory) else {
            eprintln!("skip: set X3F_TEST_FILES for file-provided white-balance checks");
            return;
        };
        let mut paths: Vec<_> = entries
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .filter(|path| {
                path.extension()
                    .and_then(|value| value.to_str())
                    .is_some_and(|value| value.eq_ignore_ascii_case("x3f"))
            })
            .collect();
        paths.sort();
        let mut models = std::collections::HashSet::new();
        for path in paths {
            let mut reader = Reader::open(&path).unwrap();
            reader.load_property_list().unwrap();
            let Some(model) = reader.dng_prop("CAMMODEL") else {
                continue;
            };
            if !matches!(
                model.as_str(),
                "SIGMA DP1 Merrill" | "SIGMA DP2 Merrill" | "SIGMA DP3 Merrill"
            ) || !models.insert(model.clone())
            {
                continue;
            }
            reader.load_camf().unwrap();
            reader.load_raw().unwrap();
            let mut names = std::ptr::null_mut();
            let mut values = std::ptr::null_mut();
            let mut count = 0;
            // SAFETY: the loaded reader owns the list and returned strings.
            let found = unsafe {
                sys::x3f_get_camf_property_list(
                    reader.x3f.as_ptr(),
                    c"WhiteBalanceColorCorrections".as_ptr() as *mut _,
                    &mut names,
                    &mut values,
                    &mut count,
                )
            };
            assert_ne!(found, 0);
            assert!(count > 0 && !names.is_null());
            // SAFETY: a successful accessor returns count reader-owned names.
            let presets: Vec<_> = unsafe { std::slice::from_raw_parts(names, count as usize) }
                .iter()
                .map(|&name| {
                    // SAFETY: CAMF list names are NUL-terminated strings.
                    unsafe { std::ffi::CStr::from_ptr(name) }
                        .to_str()
                        .unwrap()
                        .to_owned()
                })
                .collect();
            let mut checked = 0;
            for wb in &presets {
                let matrix_name = reader
                    .dng_camf_property("WhiteBalanceColorCorrections", wb)
                    .unwrap();
                let wb_matrix = reader.dng_camf_matrix_3x3(&matrix_name).unwrap();
                let gain = reader.dng_gain(Some(wb)).unwrap();
                for mode in [
                    ColorMode::Standard,
                    ColorMode::Neutral,
                    ColorMode::Vivid,
                    ColorMode::Portrait,
                    ColorMode::Landscape,
                    ColorMode::FcBlue,
                ] {
                    if reader
                        .dng_camf_property("ColorModeCompensations", mode.as_str())
                        .is_none()
                    {
                        continue;
                    }
                    let parameters = ModeParameters::read(&reader, mode).unwrap();
                    let profile = CameraProfile::prepare(&reader, wb, mode, Control::none())
                        .unwrap_or_else(|error| panic!("{model} {wb} {mode}: {error}"));
                    let expected = if mode == ColorMode::Standard {
                        wb_matrix
                    } else {
                        mat3_mul(&wb_matrix, &parameters.matrix)
                    };
                    assert_eq!(profile.camera, expected);
                    assert_eq!(profile.gain, gain);
                    checked += 1;
                }
            }
            if presets.iter().any(|wb| wb == "Sunlight") {
                let daylight = CameraProfile::prepare(
                    &reader,
                    "Daylight",
                    ColorMode::Standard,
                    Control::none(),
                )
                .unwrap();
                let sunlight = CameraProfile::prepare(
                    &reader,
                    "Sunlight",
                    ColorMode::Standard,
                    Control::none(),
                )
                .unwrap();
                assert_eq!(daylight.camera, sunlight.camera);
                assert_eq!(daylight.gain, sunlight.gain);
            }
            assert!(CameraProfile::prepare(
                &reader,
                "MissingCalibration",
                ColorMode::Standard,
                Control::none(),
            )
            .is_err());
            eprintln!("{model}: {presets:?}, {checked} color-mode calibrations checked");
        }
        if models.is_empty() {
            eprintln!("skip: no supported DP Merrill files in the corpus");
        }
    }

    #[test]
    fn normalized_romm_matches_adobe_white() {
        let white = mul(&normalize_white(ROMM), [1.0; 3]);
        for i in 0..3 {
            assert!((white[i] - PCS_WHITE[i]).abs() < 1e-12);
        }
    }
}
