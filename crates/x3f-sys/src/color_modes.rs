//! Camera Color Modes resolved from the loaded CAMF and property sections.

use std::ffi::CStr;
use std::ptr;
use std::str::FromStr;

use crate::{Control, Error, Result, *};

/// A camera look selected for one conversion.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ColorMode {
    AsShot,
    Standard,
    Vivid,
    Neutral,
    Portrait,
    Landscape,
    FCBlue,
    SunsetRed,
    ForestGreen,
    FCYellow,
    Cinema,
    TealAndOrange,
}

impl ColorMode {
    pub const NAMED: [Self; 11] = [
        Self::Standard,
        Self::Vivid,
        Self::Neutral,
        Self::Portrait,
        Self::Landscape,
        Self::FCBlue,
        Self::SunsetRed,
        Self::ForestGreen,
        Self::FCYellow,
        Self::Cinema,
        Self::TealAndOrange,
    ];

    /// The in-camera label used for display and exported profiles.
    pub const fn name(self) -> &'static str {
        match self {
            Self::AsShot => "as-shot",
            Self::Standard => "Standard",
            Self::Vivid => "Vivid",
            Self::Neutral => "Neutral",
            Self::Portrait => "Portrait",
            Self::Landscape => "Landscape",
            Self::FCBlue => "FOV Classic Blue",
            Self::SunsetRed => "Sunset Red",
            Self::ForestGreen => "Forest Green",
            Self::FCYellow => "FOV Classic Yellow",
            Self::Cinema => "Cinema",
            Self::TealAndOrange => "Teal and Orange",
        }
    }

    pub(crate) const fn key(self) -> &'static CStr {
        match self {
            Self::AsShot => c"as-shot",
            Self::Standard => c"Standard",
            Self::Vivid => c"Vivid",
            Self::Neutral => c"Neutral",
            Self::Portrait => c"Portrait",
            Self::Landscape => c"Landscape",
            Self::FCBlue => c"FCBlue",
            Self::SunsetRed => c"SunsetRed",
            Self::ForestGreen => c"ForestGreen",
            Self::FCYellow => c"FCYellow",
            Self::Cinema => c"Cinema",
            Self::TealAndOrange => c"TealAndOrange",
        }
    }

    pub(crate) fn enum_id(self) -> usize {
        Self::NAMED
            .iter()
            .position(|&m| m == self)
            .map_or(0, |i| i + 3)
    }
}

impl std::fmt::Display for ColorMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

impl FromStr for ColorMode {
    type Err = Error;

    fn from_str(value: &str) -> Result<Self> {
        std::iter::once(Self::AsShot)
            .chain(Self::NAMED)
            .find(|mode| {
                value.eq_ignore_ascii_case(mode.name())
                    || value.as_bytes().eq_ignore_ascii_case(mode.key().to_bytes())
            })
            .ok_or(Error::InvalidData(
                "unknown color mode; use as-shot or a named camera color mode",
            ))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ColorModeSensor {
    Merrill,
    Quattro,
}

/// Get the integer tone index after native RGB count bounds.
/// Quattro ordinary exports round in XDoCreateRestoredImg before Stage4.
/// Merrill bounds the ColorDQ result before its Stage4 double +0.5 conversion.
pub fn color_mode_tone_index(sensor: ColorModeSensor, count: f64) -> usize {
    let count = count as f32;
    match sensor {
        ColorModeSensor::Merrill => (f64::from(count.clamp(0.0, 32766.0)) + 0.5) as usize,
        ColorModeSensor::Quattro => ((count.clamp(0.0, 65535.0) + 0.5) as usize).min(65535),
    }
}

pub use crate::merrill::{
    ColorDq as ColorModeColorDq, RenderLuts as ColorModeRenderLuts,
    ToneParams as ColorModeToneParameters,
};

/// Validated native parameters. Normal strength is effect 0 and alpha 1.
/// The Color Modes baseline keeps Sigma HP correction off.
/// Existing x3fuse highlight recovery is a separate processing operation.
#[derive(Clone, Copy, Debug)]
pub struct ColorModeParameters {
    pub mode: ColorMode,
    pub sensor: ColorModeSensor,
    pub as_shot_fallback: bool,
    /// Row-major native C: ISO * CC * C * diag(WB).
    /// None records the native Standard absent-list or zero-sentinel branch.
    pub creative_matrix: Option<[f64; 9]>,
    pub contrast: f64,
    pub tone: ColorModeToneParameters,
    /// Reciprocal native domain gamma. Output transfer is separate.
    pub inverse_gamma: f64,
    /// Sensor ImageDepth saturation in native count units.
    pub pixel_saturation: f64,
}

/// Read-only calibration from normalized DNG sensor samples to native counts.
#[derive(Clone, Debug)]
pub struct ColorModeCalibration {
    /// Native WB correction before the creative matrix and scalar ISO gain.
    pub sensor_to_srgb: [f64; 9],
    pub wb_gain: [f64; 3],
    /// For x in [0,1], gained input is (x * this + count_offset) * wb_gain.
    pub normalized_to_counts: [f64; 3],
    /// Native black offset in count units, before WB gain.
    pub count_offset: [f64; 3],
    pub iso_gain: f64,
    /// Apply after linear tone, before output transfer.
    pub output_to_srgb: [f64; 9],
    /// Apply to native RGB counts before bounds and tone. Merrill only.
    pub color_dq: Option<ColorModeColorDq>,
}

impl ColorModeCalibration {
    /// Matrix and offset before tone, in native linear RGB count units.
    pub fn render_matrix(&self, mode: &ColorModeParameters) -> ([f64; 9], [f64; 3]) {
        let mut cc = self.sensor_to_srgb.map(|v| v * self.iso_gain);
        if let Some(c) = mode.creative_matrix {
            cc = std::array::from_fn(|i| {
                let (r, column) = (i / 3, i % 3);
                (0..3).fold(0.0, |v, k| cc[r * 3 + k].mul_add(c[k * 3 + column], v))
            });
        }
        let offset = std::array::from_fn(|r| {
            (0..3).fold(0.0, |v, c| {
                cc[r * 3 + c].mul_add(self.wb_gain[c] * self.count_offset[c], v)
            })
        });
        let matrix =
            std::array::from_fn(|i| cc[i] * self.wb_gain[i % 3] * self.normalized_to_counts[i % 3]);
        (matrix, offset)
    }
}

fn as_shot(bytes: &[u8]) -> Result<(ColorMode, bool)> {
    let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    let bytes = &bytes[..end];
    if matches!(bytes, b"Monochrome" | b"Sepia") {
        return Err(Error::InvalidData(
            "as-shot monochrome and sepia color modes are unsupported",
        ));
    }
    if bytes == b"Fov Classic Blue" {
        return Ok((ColorMode::FCBlue, false));
    }
    Ok(ColorMode::NAMED
        .iter()
        .copied()
        .find(|mode| bytes == mode.key().to_bytes())
        .map_or((ColorMode::Standard, true), |mode| (mode, false)))
}

pub(crate) unsafe fn sensor(x3f: *mut x3f_t) -> Result<ColorModeSensor> {
    if x3f.is_null() {
        return Err(Error::InvalidData("missing color mode image"));
    }
    if unsafe { crate::process::merrill_count_scale(x3f) }.is_some() {
        return Ok(ColorModeSensor::Merrill);
    }
    // The directory image type identifies Quattro before RAW decode. PROP is absent.
    // sd Quattro and H share the dp Quattro tone and creative stages.
    unsafe { crate::process::sd_quattro_camera(x3f) }?;
    let raw = unsafe { x3f_get_raw(x3f) };
    if !raw.is_null()
        && matches!(
            unsafe { (*raw).header.data_subsection.image_data.type_format },
            0x0001_0023 | 0x0001_0025 | 0x0001_0027
        )
        && unsafe { (*x3f).header.version } >= 0x0003_0000
    {
        return Ok(ColorModeSensor::Quattro);
    }
    Err(Error::InvalidData(
        "color modes require a supported Merrill or Quattro process version",
    ))
}

unsafe fn properties<'a>(x3f: *mut x3f_t, list: &CStr) -> Result<Vec<(&'a CStr, &'a CStr)>> {
    let (mut names, mut values, mut count) = (ptr::null_mut(), ptr::null_mut(), 0);
    if unsafe {
        x3f_get_camf_property_list(
            x3f,
            list.as_ptr().cast_mut(),
            &mut names,
            &mut values,
            &mut count,
        )
    } == 0
    {
        return Ok(Vec::new());
    }
    if count > 256 || (count != 0 && (names.is_null() || values.is_null())) {
        return Err(Error::InvalidData("invalid color mode CAMF property list"));
    }
    (0..count as usize)
        .map(|i| {
            let (name, value) = unsafe { (*names.add(i), *values.add(i)) };
            if name.is_null() || value.is_null() {
                return Err(Error::InvalidData("invalid color mode CAMF property entry"));
            }
            Ok(unsafe { (CStr::from_ptr(name), CStr::from_ptr(value)) })
        })
        .collect()
}

pub(crate) unsafe fn float_matrix(
    x3f: *mut x3f_t,
    key: &CStr,
    shape: &[usize],
) -> Result<Vec<f64>> {
    let (mut a, mut b, mut data) = (0, 0, ptr::null_mut());
    if unsafe {
        x3f_get_camf_matrix_var(
            x3f,
            key.as_ptr().cast_mut(),
            &mut a,
            if shape.len() == 2 {
                &mut b
            } else {
                ptr::null_mut()
            },
            ptr::null_mut(),
            matrix_type_t_M_FLOAT,
            &mut data,
        )
    } == 0
        || a as usize != shape[0]
        || (shape.len() == 2 && b as usize != shape[1])
    {
        return Err(Error::InvalidData(
            "missing or invalid color mode CAMF matrix",
        ));
    }
    let values = unsafe { std::slice::from_raw_parts(data.cast::<f64>(), shape.iter().product()) };
    if values.iter().any(|v| !v.is_finite()) {
        return Err(Error::InvalidData("nonfinite color mode CAMF matrix"));
    }
    Ok(values.to_vec())
}

unsafe fn contrast_scalar(x3f: *mut x3f_t, key: &CStr) -> Result<f64> {
    let (mut size, mut data) = (0, ptr::null_mut());
    if unsafe {
        x3f_get_camf_matrix_var(
            x3f,
            key.as_ptr().cast_mut(),
            &mut size,
            ptr::null_mut(),
            ptr::null_mut(),
            matrix_type_t_M_FLOAT,
            &mut data,
        )
    } == 0
        || size != 1
    {
        // Native leaves contrast at zero when the scalar read fails.
        return Ok(0.0);
    }
    let value = unsafe { *data.cast::<f64>() };
    if !value.is_finite() {
        return Err(Error::InvalidData("nonfinite color mode contrast"));
    }
    Ok(value)
}

pub(crate) fn valid_matrix(m: &[f64; 9]) -> bool {
    let determinant = m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6])
        + m[2] * (m[3] * m[7] - m[4] * m[6]);
    let norm = m.iter().copied().map(f64::abs).fold(0.0, f64::max);
    m.iter().all(|v| v.is_finite())
        && norm > 0.0
        && (determinant / norm / norm / norm).abs() > 1e-10
}

pub(crate) fn fcblue_matrix(dp: bool) -> [f64; 9] {
    let bits = if dp {
        [
            0x3ff8d42c3c9eecc0,
            0x3fd46edaa92e6213,
            0xbfebdfce3150dae4,
            0x3fd69d2bf5515055,
            0x3ff6160956c0d6f5,
            0xbfe77aa8a82a5615,
            0x3fc08ad688377186,
            0x3fb73b13d244dbd2,
            0x3fe8f5e84f09528f,
        ]
    } else {
        [
            0x3ff9b97785729b28,
            0x3fd2a44417870542,
            0xbfecc5197a24894c,
            0x3fd8915bdd76683c,
            0x3ff5d78811b1d92b,
            0xbfe7f7b9e060fe48,
            0x3fc2fbe33acd5b68,
            0x3fb4ca0c282c6ef4,
            0x3fe8a7c7c5260f5e,
        ]
    };
    bits.map(f64::from_bits)
}

fn quattro_tone(rows: &[[f32; 7]], mode: ColorMode) -> ColorModeToneParameters {
    let id = if !rows.is_empty() && rows.len() <= mode.enum_id() {
        0
    } else {
        mode.enum_id()
    };
    let [start, end, steepness, breakpoint, steepness2] = match id {
        3 => [-1.19, 2.85, 3.0, 0.078, 1.55],
        4 => [-1.20, 2.85, 3.55, 0.070, 1.60],
        5 => [-1.19, 2.85, 2.80, 0.080, 1.56],
        6 => [-1.05, 2.65, 3.10, 0.030, 1.97],
        12 => [-1.20, 2.85, 3.50, 0.070, 1.90],
        _ => [-1.17, 2.85, 3.0, 0.1, 1.4],
    };
    // The ordinary built-in switch replaces all row fields. Only bounds survive.
    ColorModeToneParameters {
        start,
        end,
        steepness,
        steepness2,
        breakpoint,
        scale: 1.0,
    }
}

/// Resolve one camera mode without changing headers, samples or global state.
/// # Safety
/// `x3f` must own loaded CAMF and optional PROP sections for this call.
pub unsafe fn resolve_color_mode(
    x3f: *mut x3f_t,
    selection: ColorMode,
) -> Result<ColorModeParameters> {
    let sensor = unsafe { sensor(x3f) }?;
    let (mode, as_shot_fallback) = if selection == ColorMode::AsShot {
        if sensor == ColorModeSensor::Quattro && unsafe { (*x3f).header.version } >= 0x0004_0000 {
            let recorded = match unsafe { crate::process::merrill_exif(x3f, Control::none()) } {
                Ok(exif) => exif.picture_mode,
                Err(Error::InvalidData(_)) => None,
                Err(error) => return Err(error),
            };
            if recorded.is_none() {
                unsafe {
                    x3f_printf(
                        x3f_verbosity_t_WARN,
                        c"Missing as-shot PictureMode; using Standard\n".as_ptr(),
                    );
                }
            }
            let result = as_shot(recorded.as_deref().unwrap_or("").as_bytes())?;
            if recorded.is_some() && result.1 {
                unsafe {
                    x3f_printf(
                        x3f_verbosity_t_WARN,
                        c"Unknown as-shot PictureMode; using Standard\n".as_ptr(),
                    );
                }
            }
            result
        } else {
            as_shot(&unsafe { (*x3f).header.color_mode.map(|b| b as u8) })?
        }
    } else {
        (selection, false)
    };
    if sensor == ColorModeSensor::Merrill && mode.enum_id() > 8 {
        return Err(Error::InvalidData(
            "selected color mode is not supported by Merrill sensors",
        ));
    }
    let list = unsafe { properties(x3f, c"ColorModeCompensations") }?;
    let mut creative_matrix: Option<[f64; 9]> = match list
        .iter()
        .find(|(name, _)| *name == mode.key())
    {
        Some(_) if mode == ColorMode::TealAndOrange && sensor == ColorModeSensor::Quattro => None,
        Some((_, key))
            if unsafe { crate::meta::camf_find_by_name(x3f, key.as_ptr()) }.is_none()
                && matches!(mode, ColorMode::Standard | ColorMode::FCBlue) =>
        {
            None
        }
        Some((_, key)) => Some(
            unsafe { float_matrix(x3f, key, &[3, 3]) }?
                .try_into()
                .unwrap(),
        ),
        None => None,
    };
    if !list.is_empty() {
        if mode == ColorMode::FCBlue
            && sensor == ColorModeSensor::Merrill
            && creative_matrix.is_none_or(|m| m[0] == 0.0)
        {
            creative_matrix = Some(fcblue_matrix(
                unsafe { crate::process::merrill_count_scale(x3f) } == Some(2.0),
            ));
        }
        if mode == ColorMode::TealAndOrange && sensor == ColorModeSensor::Quattro {
            creative_matrix = Some([
                1.06673, 0.05757, -0.12415, 0.25086, 0.38293, 0.36634, -0.01669, -0.04191, 1.05873,
            ]);
        }
    }
    if creative_matrix.is_none_or(|m| m[0] == 0.0) {
        if mode != ColorMode::Standard {
            return Err(Error::InvalidData(
                "selected color mode has no usable CAMF compensation matrix",
            ));
        }
        creative_matrix = None;
    } else if !valid_matrix(creative_matrix.as_ref().unwrap()) {
        return Err(Error::InvalidData(
            "singular or ill-conditioned color mode compensation matrix",
        ));
    }
    let contrast_list = unsafe { properties(x3f, c"ColorModeContrastCompensations") }?;
    let ordinal = contrast_list
        .iter()
        .position(|(name, _)| *name == mode.key())
        .unwrap_or(contrast_list.len());
    let mut contrast = match contrast_list.get(ordinal) {
        Some((_, key)) => unsafe { contrast_scalar(x3f, key) }?,
        None => 0.0,
    };
    if sensor == ColorModeSensor::Merrill && mode == ColorMode::FCBlue && contrast == 0.0 {
        contrast = 0.3;
    }
    let mut tone_rows = Vec::new();
    if unsafe { crate::meta::camf_find_by_name(x3f, c"TCColorModeSettings".as_ptr()) }.is_some() {
        let (mut rows, mut cols, mut data) = (0, 0, ptr::null_mut());
        if unsafe {
            x3f_get_camf_matrix_var(
                x3f,
                c"TCColorModeSettings".as_ptr().cast_mut(),
                &mut rows,
                &mut cols,
                ptr::null_mut(),
                matrix_type_t_M_FLOAT,
                &mut data,
            )
        } == 0
            || cols != 7
            || !(1..=256).contains(&rows)
        {
            return Err(Error::InvalidData(
                "invalid color mode tone table; expected rows of seven floats",
            ));
        }
        for row in unsafe { std::slice::from_raw_parts(data.cast::<f64>(), rows as usize * 7) }
            .chunks_exact(7)
        {
            let row = std::array::from_fn(|i| row[i] as f32);
            if sensor == ColorModeSensor::Merrill && row.iter().any(|v| !v.is_finite()) {
                return Err(Error::InvalidData("nonfinite color mode tone table"));
            }
            tone_rows.push(row);
        }
    }
    let tone = match sensor {
        ColorModeSensor::Merrill => {
            ColorModeToneParameters::from_color_mode_rows(&tone_rows, ordinal)
        }
        ColorModeSensor::Quattro => quattro_tone(&tone_rows, mode),
    };
    if tone.start >= tone.end
        || tone.steepness <= 0.0
        || tone.steepness2 <= 0.0
        || tone.scale <= 0.0
    {
        return Err(Error::InvalidData("degenerate color mode tone parameters"));
    }
    let mut depth = if sensor == ColorModeSensor::Merrill {
        12
    } else {
        14
    };
    unsafe {
        x3f_get_camf_unsigned(x3f, c"ImageDepth".as_ptr().cast_mut(), &mut depth);
    }
    if !(1..=14).contains(&depth) {
        return Err(Error::InvalidData(
            "color mode ImageDepth exceeds the supported tone extent",
        ));
    }
    unsafe { crate::process::color_mode_base_calibration(x3f, ptr::null_mut(), sensor) }?;
    // Export rejects an sd Quattro gained range past the native count clamp.
    let mut maximum = [0; 3];
    if unsafe { x3f_get_max_raw(x3f, maximum.as_mut_ptr()) } != 0 {
        unsafe { crate::process::sd_quattro_preprocess(x3f, maximum) }?;
    }
    if sensor == ColorModeSensor::Merrill {
        unsafe { crate::process::merrill_color_dq(x3f) }?;
    }
    if as_shot_fallback && sensor == ColorModeSensor::Merrill {
        unsafe {
            x3f_printf(
                x3f_verbosity_t_WARN,
                c"Unknown or empty as-shot color mode; using Standard\n".as_ptr(),
            );
        }
    }
    Ok(ColorModeParameters {
        mode,
        sensor,
        as_shot_fallback,
        creative_matrix,
        contrast,
        tone,
        inverse_gamma: if sensor == ColorModeSensor::Merrill {
            1.0 / 2.2
        } else {
            1.0 / 2.2222222222222223
        },
        pixel_saturation: f64::from((1u32 << depth) - 1),
    })
}

/// Return named choices accepted by the same resolver used for rendering.
/// # Safety
/// `x3f` must own loaded CAMF and optional PROP sections for this call.
pub unsafe fn available_color_modes(x3f: *mut x3f_t) -> Result<Vec<ColorMode>> {
    unsafe { sensor(x3f) }?;
    Ok(ColorMode::NAMED
        .into_iter()
        .filter(|&mode| unsafe { resolve_color_mode(x3f, mode) }.is_ok())
        .collect())
}

/// Build a linear LUT before output gamma and integer quantization.
/// Entry i maps native count i to linear RGB in [0,1].
pub fn color_mode_linear_tone_lut(
    params: &ColorModeParameters,
    exposure: f64,
    control: Control<'_>,
) -> Result<Vec<f64>> {
    crate::merrill::generate_color_mode_linear_tone_lut(params, exposure, control)
}

/// Build the native integer tone LUT in sensor-count coordinates.
pub fn color_mode_tone_lut(
    params: &ColorModeParameters,
    exposure: f64,
    control: Control<'_>,
) -> Result<Vec<u16>> {
    crate::merrill::generate_color_mode_tone_lut(params, exposure, control)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_and_as_shot_keep_selection_and_fallback_distinct() {
        assert_eq!(
            ColorMode::NAMED.map(ColorMode::name),
            [
                "Standard",
                "Vivid",
                "Neutral",
                "Portrait",
                "Landscape",
                "FOV Classic Blue",
                "Sunset Red",
                "Forest Green",
                "FOV Classic Yellow",
                "Cinema",
                "Teal and Orange",
            ]
        );
        for mode in ColorMode::NAMED {
            assert_eq!(mode.name().parse::<ColorMode>().unwrap(), mode);
            assert_eq!(mode.to_string(), mode.name());
            assert_eq!(
                mode.name().to_lowercase().parse::<ColorMode>().unwrap(),
                mode
            );
            let key = mode.key().to_str().unwrap();
            assert_eq!(key.parse::<ColorMode>().unwrap(), mode);
            assert_eq!(key.to_lowercase().parse::<ColorMode>().unwrap(), mode);
            assert_eq!(as_shot(key.as_bytes()).unwrap(), (mode, false));
        }
        assert_eq!("as-shot".parse::<ColorMode>().unwrap(), ColorMode::AsShot);
        assert_ne!(ColorMode::AsShot, ColorMode::Standard);
        assert!("Monochrome".parse::<ColorMode>().is_err());
        assert!("unknown".parse::<ColorMode>().is_err());
        assert_eq!(
            as_shot(b"Neutral\0padding").unwrap(),
            (ColorMode::Neutral, false)
        );
        assert_eq!(
            as_shot(b"Fov Classic Blue").unwrap(),
            (ColorMode::FCBlue, false)
        );
        for bytes in [b"".as_slice(), b"unknown", &[b'x'; 32]] {
            assert_eq!(as_shot(bytes).unwrap(), (ColorMode::Standard, true));
        }
        for bytes in [b"Monochrome".as_slice(), b"Sepia"] {
            assert!(as_shot(bytes).is_err());
        }
        assert_eq!(ProcessingOptions::default().color_mode, None);
    }

    #[test]
    fn ordinary_tone_indices_bound_and_round_native_counts() {
        let cases = [
            (-100.0, 0),
            (-1.75, 0),
            (-0.25, 0),
            (300.25, 300),
            (500.75, 501),
            (1000.5, 1001),
            (65534.25, 65534),
            (65534.5, 65535),
            (65535.0, 65535),
            (70000.0, 65535),
        ];
        // XDoCreateRestoredImg oracle: native_bound_oracle.cpp and REA 0xe26f0.
        for (count, expected) in cases {
            assert_eq!(
                color_mode_tone_index(ColorModeSensor::Quattro, count),
                expected
            );
        }
        assert_eq!(color_mode_tone_index(ColorModeSensor::Merrill, -100.0), 0);
        assert_eq!(color_mode_tone_index(ColorModeSensor::Merrill, 300.25), 300);
        assert_eq!(color_mode_tone_index(ColorModeSensor::Merrill, 500.75), 501);
        assert_eq!(
            color_mode_tone_index(ColorModeSensor::Merrill, 70000.0),
            32766
        );
        // The native intermediate uses float32, including before rounding.
        let count = 300.5 - 1e-9;
        assert_eq!(color_mode_tone_index(ColorModeSensor::Quattro, count), 301);
        assert_eq!(color_mode_tone_index(ColorModeSensor::Merrill, count), 301);
    }

    #[test]
    fn quattro_bounds_reset_before_the_builtin_switch() {
        let rows = [[f32::NAN; 7]; 7];
        let cinema = quattro_tone(&[], ColorMode::Cinema);
        assert_eq!(cinema.steepness, 3.5);
        let reset = quattro_tone(&rows, ColorMode::Cinema);
        assert_eq!(reset.steepness, 3.0);
        assert_eq!(reset.end, 2.85);
        assert_eq!(reset.steepness2, 1.4);
        assert_eq!(reset.scale, 1.0);
        let vivid = quattro_tone(&rows, ColorMode::Vivid);
        assert_eq!(vivid.steepness, 3.55);
        let portrait_reset = quattro_tone(&rows[..6], ColorMode::Portrait);
        assert_eq!(
            (
                portrait_reset.steepness,
                portrait_reset.breakpoint,
                portrait_reset.steepness2
            ),
            (3.0, 0.1, 1.4)
        );
        let portrait = quattro_tone(&rows, ColorMode::Portrait);
        assert_eq!(
            (portrait.steepness, portrait.breakpoint, portrait.steepness2),
            (3.1, 0.03, 1.97)
        );
        let mut sentinel = [[8.0; 7]; 13];
        assert_eq!(quattro_tone(&sentinel, ColorMode::Cinema).steepness, 3.5);
        sentinel[12][0] = 0.0;
        assert_eq!(quattro_tone(&sentinel, ColorMode::Cinema).scale, 1.0);
        assert_eq!(
            quattro_tone(&sentinel, ColorMode::TealAndOrange).steepness2,
            1.4
        );
    }

    #[test]
    fn calibration_composes_right_before_per_channel_count_gain() {
        let params = ColorModeParameters {
            mode: ColorMode::Vivid,
            sensor: ColorModeSensor::Merrill,
            as_shot_fallback: false,
            creative_matrix: Some([1.0, 2.0, 0.0, 0.0, 1.0, 3.0, 4.0, 0.0, 1.0]),
            contrast: 0.3,
            tone: ColorModeToneParameters::from_color_mode_rows(&[], 0),
            inverse_gamma: 1.0 / 2.2,
            pixel_saturation: 4095.0,
        };
        let calibration = ColorModeCalibration {
            sensor_to_srgb: [2.0, 0.0, 1.0, 1.0, 3.0, 0.0, 0.0, 1.0, 4.0],
            wb_gain: [2.0, 3.0, 4.0],
            normalized_to_counts: [40.0, 50.0, 80.0],
            count_offset: [2.0, -5.0, 10.0],
            iso_gain: 2.0,
            output_to_srgb: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            color_dq: None,
        };
        let (matrix, offset) = calibration.render_matrix(&params);
        let input = [0.25, 0.5, 0.75];
        let output = std::array::from_fn::<_, 3, _>(|r| {
            (0..3).fold(offset[r], |v, c| matrix[r * 3 + c].mul_add(input[c], v))
        });
        assert_eq!(output, [1328.0, 5688.0, 4808.0]);
    }

    #[test]
    fn invalid_matrix_conditioning_is_rejected() {
        assert!(valid_matrix(&[
            1.0, 0.2, 0.3, -0.1, 1.0, 0.4, 0.2, 0.3, 0.8
        ]));
        assert!(!valid_matrix(&[0.0; 9]));
        assert!(!valid_matrix(&[
            1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1e-12
        ]));
        assert!(!valid_matrix(&[f64::NAN; 9]));
        assert_ne!(fcblue_matrix(true), fcblue_matrix(false));
        assert!(valid_matrix(&fcblue_matrix(true)));
        assert!(valid_matrix(&fcblue_matrix(false)));
    }

    #[test]
    fn corpus_modes_resolve_before_raw_decode_and_calibration_preserves_samples() {
        let root = std::env::var_os("X3F_TEST_FILES")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| {
                std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../x3f_test_files")
            });
        let mut checked = 0;
        for name in ["sigma_sd1_merrill_15.x3f", "_SDI8284.X3F", "SDQH5087.X3F"] {
            let path = root.join(name);
            if !path.exists() {
                continue;
            }
            let path = std::ffi::CString::new(path.to_str().unwrap()).unwrap();
            unsafe {
                let file = libc::fopen(path.as_ptr(), c"rb".as_ptr());
                assert!(!file.is_null());
                let x3f = x3f_new_from_file(file.cast());
                assert!(!x3f.is_null());
                for section in [x3f_get_camf(x3f), x3f_get_prop(x3f)] {
                    if !section.is_null() {
                        assert_eq!(x3f_load_data(x3f, section), x3f_return_e_X3F_OK);
                    }
                }
                let header_mode = (*x3f).header.color_mode;
                let raw = x3f_get_raw(x3f);
                assert!(!raw.is_null());
                assert!((*raw).header.data_subsection.image_data.data.is_null());
                let modes = available_color_modes(x3f).unwrap();
                assert!(modes.contains(&ColorMode::Standard), "{name}: {modes:?}");
                assert!(modes.contains(&ColorMode::Vivid), "{name}: {modes:?}");
                for mode in modes {
                    assert_eq!(resolve_color_mode(x3f, mode).unwrap().mode, mode);
                }
                assert!((*raw).header.data_subsection.image_data.data.is_null());
                assert_eq!((*x3f).header.color_mode, header_mode);
                assert_eq!(x3f_load_data(x3f, raw), x3f_return_e_X3F_OK);
                let mut area = x3f_area16_t::default();
                assert_ne!(x3f_image_area(x3f, &mut area), 0);
                let samples = std::slice::from_raw_parts(
                    area.data,
                    area.row_stride as usize * area.rows as usize,
                );
                let hash = |values: &[u16]| {
                    values.iter().fold(0xcbf29ce484222325_u64, |h, &v| {
                        (h ^ u64::from(v)).wrapping_mul(0x100000001b3)
                    })
                };
                let before = hash(samples);
                let calibration =
                    color_mode_calibration(x3f, ptr::null_mut(), Control::none()).unwrap();
                assert!(calibration
                    .normalized_to_counts
                    .iter()
                    .all(|v| v.is_finite() && *v > 0.0));
                assert_eq!(hash(samples), before);
                assert_eq!((*x3f).header.color_mode, header_mode);
                let recorded = resolve_color_mode(x3f, ColorMode::AsShot).unwrap();
                assert_ne!(recorded.mode, ColorMode::AsShot);
                if name == "_SDI8284.X3F" {
                    assert_eq!(recorded.sensor, ColorModeSensor::Quattro);
                    assert_eq!(recorded.mode, ColorMode::Standard);
                    assert!(!recorded.as_shot_fallback);
                    let named = resolve_color_mode(x3f, ColorMode::Standard).unwrap();
                    assert_eq!(
                        calibration.render_matrix(&recorded),
                        calibration.render_matrix(&named)
                    );
                    assert_eq!(
                        color_mode_tone_lut(&recorded, 0.0, Control::none()).unwrap(),
                        color_mode_tone_lut(&named, 0.0, Control::none()).unwrap()
                    );
                }
                assert_eq!((*x3f).header.color_mode, header_mode);
                assert_eq!(x3f_delete(x3f), x3f_return_e_X3F_OK);
                libc::fclose(file);
                checked += 1;
            }
        }
        if checked == 0 {
            eprintln!("Color mode corpus check skipped: no Merrill/Quattro fixtures");
        }
    }
}
