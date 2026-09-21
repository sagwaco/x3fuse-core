//! A fresh sensor decode feeding an immutable, unclipped editor source.

use std::{path::Path, sync::atomic::AtomicBool};
use x3f_sys::{self as sys, Control};

use crate::output::dng::{color::ColorCalibration, exif::CaptureMetadata, metadata::mat3_inverse};
use crate::{ConversionStage, Error, Reader};

/// Sensor preparation choices. Creative adjustments belong to the renderer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SceneLinearOptions {
    /// Interpolate known bad sensor pixels (default true).
    pub fix_bad: bool,
    /// Lens shading; None uses the existing camera-specific default.
    pub apply_sgain: Option<bool>,
    /// Existing sensor NLM intensity, 0 through 10 (default 10).
    pub denoise_intensity: u8,
    /// Reconstruct clipped sensor layers (default false).
    pub highlight_recovery: bool,
}

impl Default for SceneLinearOptions {
    fn default() -> Self {
        Self {
            fix_bad: true,
            apply_sgain: None,
            denoise_intensity: 10,
            highlight_recovery: false,
        }
    }
}

/// Matrices are row-major and operate on column vectors in normalized BMT.
/// For a desired XYZ white W, let n' = xyz_to_camera * W. A camera-space
/// WB adjustment over the as-shot image is M * diag(n/n') * inverse(M),
/// with n = as_shot_neutral and M = camera_to_rgb. Normalize the relative
/// gains by green to separate balance from exposure. As Shot is identity.
#[derive(Debug, Clone, Copy)]
pub struct WhiteBalanceCalibration {
    /// Camera BMT to extended-linear sRGB, including as-shot WB and ISO.
    pub camera_to_rgb: [f64; 9],
    /// D65 XYZ to calibrated camera coordinates, matching DNG calibration.
    pub xyz_to_camera: [f64; 9],
    /// As-shot neutral including relative per-channel digital ISO gain.
    pub as_shot_neutral: [f64; 3],
}

/// Immutable editor source, active-area cropped but not orientation-rotated.
/// Samples are tightly packed RGB in extended-linear sRGB/D65. Negative
/// channels and values above one are retained. As-shot WB and capture/digital
/// ISO gain are already applied; gamma, creative EV and display tone are not.
/// The existing sensor front end still uses its integer intermediate buffers.
#[derive(Debug)]
pub struct SceneLinearImage {
    /// Three f32 samples per pixel, in row-major RGB order.
    pub data: Vec<f32>,
    /// Active-area width.
    pub width: u32,
    /// Active-area height.
    pub height: u32,
    /// EXIF orientation, 1 through 8. The renderer applies this exactly once.
    pub orientation: u16,
    /// Calibration for camera-space white-balance edits.
    pub white_balance: WhiteBalanceCalibration,
}

/// Camera framing inferred from the embedded JPEG, matching DNG DefaultUserCrop.
/// Returns normalized `[x, y, width, height]` within the EXIF-oriented active
/// area. This reads metadata only; it neither decodes nor crops RAW samples.
/// Missing framing metadata returns None (use the full active area).
pub fn read_as_shot_crop(input: &Path) -> Result<Option<[f32; 4]>, Error> {
    let mut reader = Reader::open(input)?;
    reader.load_camf()?;
    reader.load_property_list()?;
    // SAFETY: the fresh reader owns the directory and metadata throughout.
    let active = unsafe {
        let raw = sys::x3f_get_raw(reader.x3f.as_ptr());
        if raw.is_null() {
            return Ok(None);
        }
        let header = (*raw).header.data_subsection.image_data;
        if header.columns == 0 || header.rows == 0 {
            return Ok(None);
        }
        // The metadata rectangle transform uses dimensions, never sample data.
        let mut area = sys::x3f_area16_t {
            data: std::ptr::null_mut(),
            buf: std::ptr::null_mut(),
            rows: header.rows,
            columns: header.columns,
            channels: 0,
            row_stride: 0,
        };
        let mut rect = [0; 4];
        if sys::x3f_get_camf_rect(
            reader.x3f.as_ptr(),
            c"ActiveImageArea".as_ptr() as *mut _,
            &mut area,
            1,
            rect.as_mut_ptr(),
        ) == 0
            || rect[0] > rect[2]
            || rect[1] > rect[3]
            || rect[2] >= header.columns
            || rect[3] >= header.rows
        {
            return Ok(None);
        }
        [rect[1], rect[0], rect[3] + 1, rect[2] + 1]
    };
    let Some(crop) = crate::output::dng::compute_default_user_crop(&reader, &active) else {
        return Ok(None);
    };
    let orientation = CaptureMetadata::from_reader(&reader)
        .orientation
        .unwrap_or(1);
    Ok(Some(oriented_as_shot_crop(crop, orientation)))
}

fn oriented_as_shot_crop([top, left, bottom, right]: [f32; 4], orientation: u16) -> [f32; 4] {
    // Camera framing is centered, so mirrors and 180-degree rotation preserve it.
    if (5..=8).contains(&orientation) {
        [top, left, bottom - top, right - left]
    } else {
        [left, top, right - left, bottom - top]
    }
}

/// Decode and prepare a new reader for each call. Never mutates the source file.
/// Emits Read, Decode and Process; no output is written. Cancellation and
/// processing options are per-call and safe across independent preparations.
pub fn prepare_scene_linear(
    input: &Path,
    options: &SceneLinearOptions,
    cancel: &AtomicBool,
    mut on_stage: impl FnMut(ConversionStage),
) -> Result<SceneLinearImage, Error> {
    let control = Control::new(cancel);
    control.check()?;
    if options.denoise_intensity > 10 {
        return Err(Error::InvalidData(
            "denoise intensity must be 0 through 10".into(),
        ));
    }
    on_stage(ConversionStage::Read);
    let reader = Reader::open_with_control(input, control)?;
    on_stage(ConversionStage::Decode);
    control.check()?;
    // SAFETY: this call uniquely owns the fresh reader and all loaded sections.
    unsafe {
        let x3f = reader.x3f.as_ptr();
        for entry in [sys::x3f_get_prop(x3f), sys::x3f_get_thumb_jpeg(x3f)] {
            if !entry.is_null() {
                sys::load_data(x3f, entry, control)?;
            }
        }
        sys::load_data(x3f, sys::x3f_get_camf(x3f), control)?;
        sys::load_data(x3f, sys::x3f_get_raw(x3f), control)?;
    }
    let wb = reader.dng_default_wb();
    let color = ColorCalibration::new(&reader, &wb)
        .ok_or_else(|| Error::InvalidData("missing white-balance calibration".into()))?;
    let bmt_to_xyz = reader
        .dng_bmt_to_xyz(Some(&wb))
        .ok_or_else(|| Error::InvalidData("missing camera color matrix".into()))?;
    validate_matrix(&bmt_to_xyz)?;
    let xyz_to_camera = color.fold_color_matrix(&mat3_inverse(&bmt_to_xyz));
    validate_matrix(&xyz_to_camera)?;
    let orientation = CaptureMetadata::from_reader(&reader)
        .orientation
        .filter(|v| (1..=8).contains(v))
        .unwrap_or(1);
    on_stage(ConversionStage::Process);
    control.check()?;
    // SAFETY: preparation consumes only this fresh reader's decoded buffers.
    let linear = unsafe {
        sys::get_scene_linear_controlled(
            reader.x3f.as_ptr(),
            options.fix_bad,
            options.denoise_intensity,
            reader.resolve_sgain(options.apply_sgain) != 0,
            options.highlight_recovery,
            control,
        )
    }?;
    validate_matrix(&linear.camera_to_rgb)?;
    control.check()?;
    Ok(SceneLinearImage {
        data: linear.data,
        width: linear.width,
        height: linear.height,
        orientation,
        white_balance: WhiteBalanceCalibration {
            camera_to_rgb: linear.camera_to_rgb,
            xyz_to_camera,
            as_shot_neutral: color.neutral,
        },
    })
}

fn validate_matrix(m: &[f64; 9]) -> Result<(), Error> {
    let determinant = m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6])
        + m[2] * (m[3] * m[7] - m[4] * m[6]);
    if m.iter().all(|v| v.is_finite()) && determinant.is_finite() && determinant.abs() > 1e-12 {
        Ok(())
    } else {
        Err(Error::InvalidData(
            "nonfinite or singular camera calibration".into(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn as_shot_crop_is_normalized_in_display_orientation() {
        let bounds = [0.125, 0.0, 0.875, 1.0];
        for orientation in 1..=8 {
            let expected = if orientation <= 4 {
                [0.0, 0.125, 1.0, 0.75]
            } else {
                [0.125, 0.0, 0.75, 1.0]
            };
            assert_eq!(oriented_as_shot_crop(bounds, orientation), expected);
            assert_eq!(
                oriented_as_shot_crop([0.0, 0.0, 1.0, 1.0], orientation),
                [0.0, 0.0, 1.0, 1.0]
            );
        }
    }

    #[test]
    fn preparation_rejects_invalid_options_and_cancels_before_io() {
        let path = Path::new("nonexistent.X3F");
        let options = SceneLinearOptions::default();
        assert!(matches!(
            prepare_scene_linear(path, &options, &AtomicBool::new(true), |_| {}),
            Err(Error::Cancelled)
        ));
        let invalid = SceneLinearOptions {
            denoise_intensity: 11,
            ..options
        };
        assert!(matches!(
            prepare_scene_linear(path, &invalid, &AtomicBool::new(false), |_| {}),
            Err(Error::InvalidData(_))
        ));
        assert!(validate_matrix(&[0.0; 9]).is_err());
        assert!(validate_matrix(&[f64::NAN; 9]).is_err());
        assert!(validate_matrix(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]).is_ok());
    }
}
