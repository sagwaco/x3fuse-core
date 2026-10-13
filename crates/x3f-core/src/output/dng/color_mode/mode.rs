//! Read the selected color mode's color and tone settings from the X3F metadata.

use super::{
    invalid,
    tone::{native_tone, ToneShape},
};
use crate::{ColorMode, Error, Reader};
use x3f_sys as sys;

// The file stores eight rows of seven tone settings. Standard uses the first
// row. Other modes can reuse it only when every row agrees. If the rows differ,
// reject the file rather than guessing which settings belong to the chosen mode.
const TONE_MODE_COUNT: usize = 8;
const PARAMETERS_PER_MODE: usize = 7;
const SETTINGS_COUNT: usize = TONE_MODE_COUNT * PARAMETERS_PER_MODE;

// Sigma changes contrast by multiplying both tone-curve slopes by
// 2 raised to (contrast / 2). Accept offsets within its slider's -2..2 range.
const CONTRAST_LIMIT: f64 = 2.0;
const CONTRAST_DIVISOR: f64 = 2.0;
const IDENTITY: [f64; 9] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

pub(super) struct ModeParameters {
    pub matrix: [f64; 9],
    pub tone: Vec<f64>,
}

impl ModeParameters {
    pub fn read(reader: &Reader, mode: ColorMode) -> Result<Self, Error> {
        let name = mode.as_str();
        let matrix_name = reader
            .dng_camf_property("ColorModeCompensations", name)
            .ok_or_else(|| invalid(&format!("missing {name} matrix entry")))?;
        let contrast_name = reader
            .dng_camf_property("ColorModeContrastCompensations", name)
            .ok_or_else(|| invalid(&format!("missing {name} contrast entry")))?;
        let matrix = reader
            .dng_camf_matrix_3x3(&matrix_name)
            .ok_or_else(|| invalid(&format!("missing {name} matrix")))?;
        let contrast = reader
            .dng_camf_float(&contrast_name)
            .ok_or_else(|| invalid(&format!("missing {name} contrast")))?;
        if matrix.iter().any(|v| !v.is_finite()) {
            return Err(invalid("invalid color-mode matrix"));
        }
        if mode == ColorMode::Standard && (matrix != IDENTITY || contrast != 0.0) {
            return Err(invalid("unsupported Standard matrix or contrast"));
        }
        // MultiAxisTable entries describe additional hue and saturation changes,
        // not just a color matrix and contrast. Do not silently leave them out.
        if reader
            .dng_camf_multi_axis_table(&format!("MultiAxisTable_{name}"))
            .is_some()
        {
            return Err(invalid("multi-axis color modes are not supported"));
        }
        let mut settings = [0.0; SETTINGS_COUNT];
        // SAFETY: the reader owns the loaded metadata. The accessor checks the
        // requested dimensions, and settings has room for all eight rows.
        let valid = unsafe {
            sys::x3f_get_camf_matrix(
                reader.x3f.as_ptr(),
                c"TCColorModeSettings".as_ptr() as *mut _,
                TONE_MODE_COUNT as i32,
                PARAMETERS_PER_MODE as i32,
                0,
                sys::matrix_type_t_M_FLOAT,
                settings.as_mut_ptr().cast(),
            )
        };
        if valid == 0 {
            return Err(invalid("missing color-mode tone settings"));
        }
        Ok(Self {
            matrix,
            tone: native_tone(tone_parameters(&settings, mode, contrast)?),
        })
    }
}

fn tone_parameters(
    settings: &[f64; SETTINGS_COUNT],
    mode: ColorMode,
    contrast: f64,
) -> Result<ToneShape, Error> {
    if settings.iter().any(|v| !v.is_finite())
        || !contrast.is_finite()
        || !(-CONTRAST_LIMIT..=CONTRAST_LIMIT).contains(&contrast)
    {
        return Err(invalid("invalid mode tone parameters"));
    }
    let rows = settings.as_chunks::<PARAMETERS_PER_MODE>().0;
    if mode != ColorMode::Standard && rows.iter().any(|row| row != &rows[0]) {
        return Err(invalid("mode-specific tone-shape rows are not supported"));
    }
    let [scale, start, end, steep1, _gamma, breakpoint, steep2] = rows[0];
    if scale != 1.0
        || start >= end
        || steep1 <= 0.0
        || steep2 <= 0.0
        || breakpoint <= 0.0
        || breakpoint >= 1.0
    {
        return Err(invalid("unsupported tone shape"));
    }
    let factor = 2.0_f64.powf(contrast / CONTRAST_DIVISOR);
    Ok(ToneShape {
        start,
        end,
        lower_steepness: steep1 * factor,
        breakpoint,
        upper_steepness: steep2 * factor,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn settings() -> [f64; SETTINGS_COUNT] {
        let row = [
            1.0,
            -1.17_f32 as f64,
            1.65_f32 as f64,
            3.0,
            0.45_f32 as f64,
            0.1_f32 as f64,
            1.7_f32 as f64,
        ];
        std::array::from_fn(|i| row[i % PARAMETERS_PER_MODE])
    }

    #[test]
    fn mode_contrast_changes_only_the_two_tone_slopes() {
        let values = settings();
        let original = tone_parameters(&values, ColorMode::Standard, 0.0).unwrap();
        for (mode, contrast) in [
            (ColorMode::Neutral, -0.3_f32 as f64),
            (ColorMode::Vivid, 0.3_f32 as f64),
            (ColorMode::Portrait, -0.25),
            (ColorMode::Landscape, 0.25),
            (ColorMode::FcBlue, 0.3_f32 as f64),
        ] {
            let p = tone_parameters(&values, mode, contrast).unwrap();
            let factor = 2.0_f64.powf(contrast / 2.0);
            assert_eq!(
                p,
                ToneShape {
                    lower_steepness: original.lower_steepness * factor,
                    upper_steepness: original.upper_steepness * factor,
                    ..original
                }
            );
            let curve = native_tone(p);
            assert!(curve.windows(2).all(|v| v[1] >= v[0]));
        }
    }

    #[test]
    fn unsupported_or_nonfinite_tone_parameters_fail() {
        let values = settings();
        for contrast in [f64::NAN, f64::INFINITY, -2.1, 2.1] {
            assert!(tone_parameters(&values, ColorMode::Neutral, contrast).is_err());
        }
        let mut changed = values;
        changed[PARAMETERS_PER_MODE + 1] += 0.1;
        assert!(tone_parameters(&changed, ColorMode::Neutral, -0.3).is_err());
        assert!(tone_parameters(&changed, ColorMode::Standard, 0.0).is_ok());
        for (index, value) in [
            (0, 2.0),
            (1, 2.0),
            (3, 0.0),
            (5, 0.0),
            (5, 1.0),
            (6, -1.0),
            (7, f64::NAN),
        ] {
            changed = values;
            changed[index] = value;
            assert!(tone_parameters(&changed, ColorMode::Standard, 0.0).is_err());
        }
    }
}
