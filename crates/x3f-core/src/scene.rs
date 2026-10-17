//! A fresh sensor decode feeding an immutable, unclipped editor source.

use std::{path::Path, sync::atomic::AtomicBool};
use x3f_sys::{self as sys, Control};

use crate::output::dng::{color::ColorCalibration, exif::CaptureMetadata, metadata::mat3_inverse};
use crate::{ConversionStage, Error, Reader};

/// Sensor preparation choices. Creative adjustments belong to the renderer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SceneLinearOptions {
    /// Interpolate known bad sensor pixels (default true).
    pub fix_bad: bool,
    /// Lens shading; None uses the existing camera-specific default.
    pub apply_sgain: Option<bool>,
    /// Sensor denoising intensity, 0 through 10 (default 10).
    /// X3I merge denoise also requires a positive `X3F_X3I_DENOISE_SCALE`.
    pub denoise_intensity: u8,
    /// Select GPU acceleration or portable CPU execution (default: Auto).
    pub denoise_backend: crate::DenoiseBackend,
    /// Algorithm and advanced sensor-space denoising controls.
    pub denoise_options: crate::DenoiseOptions,
    /// Reconstruct clipped sensor layers (default false).
    /// An X3I merge ignores this option and keeps extended highlight values.
    pub highlight_recovery: bool,
    /// X3I input: merge all frames (default) or use only the middle exposure.
    /// The middle exposure honours `highlight_recovery` and denoise as for X3F.
    /// No effect on X3F input.
    pub x3i_source: crate::X3iSource,
}

impl Default for SceneLinearOptions {
    fn default() -> Self {
        Self {
            fix_bad: true,
            apply_sgain: None,
            denoise_intensity: 10,
            denoise_backend: crate::DenoiseBackend::Auto,
            denoise_options: crate::DenoiseOptions::default(),
            highlight_recovery: false,
            x3i_source: crate::X3iSource::default(),
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
    /// Native Color Mode calibration at the as-shot white balance.
    /// Input is `inverse(white_balance.camera_to_rgb) * scene_rgb`, with
    /// the selected spatial gain and sensor processing already applied.
    /// None means an unsupported sensor or missing or invalid native calibration.
    pub color_mode_calibration: Option<sys::ColorModeCalibration>,
}

/// Camera framing inferred from the embedded JPEG, matching DNG DefaultUserCrop.
/// Returns normalized `[x, y, width, height]` within the EXIF-oriented active
/// area. This reads metadata only; it neither decodes nor crops RAW samples.
/// Missing framing metadata returns None (use the full active area).
pub fn read_as_shot_crop(input: &Path) -> Result<Option<[f32; 4]>, Error> {
    let mut reader = if crate::is_x3i(input)? {
        crate::X3iContainer::open(input)?.open_frame(0)?
    } else {
        Reader::open(input)?
    };
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
    let control = Control::new(cancel)
        .with_denoise_backend(options.denoise_backend)
        .with_denoise_options(options.denoise_options)?;
    control.check()?;
    if options.denoise_intensity > 10 {
        return Err(Error::InvalidData(
            "denoise intensity must be 0 through 10".into(),
        ));
    }
    on_stage(ConversionStage::Read);
    let container = if crate::is_x3i(input)? {
        Some(crate::X3iContainer::open(input)?)
    } else {
        None
    };
    let middle = match &container {
        Some(container) if options.x3i_source == crate::X3iSource::MiddleExposure => {
            Some(container.middle_exposure_frame()?)
        }
        _ => None,
    };
    let denoise_scale = if container.is_some() && middle.is_none() {
        crate::x3i_output::read_denoise_scale(options.denoise_intensity, options.denoise_options)?
    } else {
        0.0
    };
    let reader = match (&container, middle) {
        (Some(container), Some(index)) => container.open_frame_with_control(index, control)?,
        (Some(container), None) => container.open_frame(0)?,
        _ => Reader::open_with_control(input, control)?,
    };
    // A middle exposure takes the single-frame path below, as an X3F does.
    let container = container.filter(|_| middle.is_none());
    let control = reader.with_control(control);
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
    if container.is_none() {
        on_stage(ConversionStage::Process);
        control.check()?;
    }
    let cwb = crate::wb_cstring(Some(&wb))?;
    // SAFETY: RAW is still unprocessed, and the reader and WB name outlive the call.
    let color_mode_calibration = match unsafe {
        sys::color_mode_calibration(reader.x3f.as_ptr(), crate::cwb_ptr(&cwb), control)
    } {
        Ok(calibration) => Some(calibration),
        Err(sys::Error::InvalidData(_)) => None,
        Err(error) => return Err(error.into()),
    };
    if let Some(container) = container {
        // Each frame must carry its own input window through preparation.
        let merge_control = Control::new(cancel);
        let process = crate::ProcessOptions {
            fix_bad: options.fix_bad,
            apply_sgain: options.apply_sgain,
            denoise_intensity: options.denoise_intensity,
            denoise_backend: options.denoise_backend,
            denoise_options: options.denoise_options,
            ..Default::default()
        };
        let mut merged = crate::x3i_output::merge_frames(
            &container,
            &reader,
            &process,
            merge_control,
            &mut on_stage,
            &mut Vec::new(),
        )?;
        if denoise_scale > 0.0 {
            let mut image = crate::x3i_merge::to_image(&merged)?;
            if crate::x3i_output::denoise(&mut image, &process, denoise_scale, merge_control)? {
                crate::x3i_output::restore_denoised(&mut merged, &image, merge_control)?;
            }
        }
        validate_matrix(&merged.camera_to_rgb)?;
        let [top, left, bottom, right] = crate::x3i_output::bounds(&merged, &reader, true);
        let mut data = Vec::with_capacity((bottom - top) as usize * (right - left) as usize * 3);
        for row in top..bottom {
            merge_control.check()?;
            for col in left..right {
                let offset = (row as usize * merged.width as usize + col as usize) * 3;
                for value in crate::x3i_output::multiply(
                    &merged.camera_to_rgb,
                    &merged.bmt[offset..offset + 3],
                ) {
                    let value = value as f32;
                    if !value.is_finite() {
                        return Err(Error::InvalidData("nonfinite X3I scene RGB".into()));
                    }
                    data.push(value);
                }
            }
        }
        merge_control.check()?;
        return Ok(SceneLinearImage {
            data,
            width: right - left,
            height: bottom - top,
            orientation,
            white_balance: WhiteBalanceCalibration {
                camera_to_rgb: merged.camera_to_rgb,
                xyz_to_camera,
                as_shot_neutral: color.neutral,
            },
            color_mode_calibration,
        });
    }
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
        color_mode_calibration,
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
    fn scene_calibration_matches_fresh_raw_without_changing_pixels() {
        let corpus = std::env::var_os("X3F_TEST_FILES")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("../../x3f_test_files"));
        let mut checked = 0;
        for name in ["_SDI8040.X3F", "_SDI8284.X3F"] {
            let path = corpus.join(name);
            if !path.is_file() {
                continue;
            }
            let load = || {
                let mut reader = Reader::open(&path).unwrap();
                reader.load_camf().unwrap();
                reader.load_property_list().unwrap();
                reader.load_thumbnail_jpeg().unwrap();
                reader.load_raw().unwrap();
                reader
            };
            let expected_calibration = {
                let reader = load();
                let cwb = crate::wb_cstring(Some(&reader.dng_default_wb())).unwrap();
                // SAFETY: this reader owns unprocessed RAW and loaded metadata.
                unsafe {
                    sys::color_mode_calibration(
                        reader.x3f.as_ptr(),
                        crate::cwb_ptr(&cwb),
                        Control::none(),
                    )
                }
                .unwrap()
            };
            for apply_sgain in [false, true] {
                let expected_scene = {
                    let reader = load();
                    // SAFETY: this fresh reader owns all buffers consumed by preparation.
                    unsafe {
                        sys::get_scene_linear_controlled(
                            reader.x3f.as_ptr(),
                            true,
                            0,
                            apply_sgain,
                            false,
                            Control::none(),
                        )
                    }
                    .unwrap()
                };
                let scene = prepare_scene_linear(
                    &path,
                    &SceneLinearOptions {
                        denoise_intensity: 0,
                        apply_sgain: Some(apply_sgain),
                        ..Default::default()
                    },
                    &AtomicBool::new(false),
                    |_| {},
                )
                .unwrap();
                assert!(
                    scene.data == expected_scene.data,
                    "{name}, sgain={apply_sgain}"
                );
                assert_eq!(
                    (scene.width, scene.height),
                    (expected_scene.width, expected_scene.height)
                );
                assert_eq!(
                    scene.white_balance.camera_to_rgb,
                    expected_scene.camera_to_rgb
                );
                let actual = scene.color_mode_calibration.unwrap();
                assert_eq!(actual.sensor_to_srgb, expected_calibration.sensor_to_srgb);
                assert_eq!(actual.wb_gain, expected_calibration.wb_gain);
                assert_eq!(
                    actual.normalized_to_counts,
                    expected_calibration.normalized_to_counts
                );
                assert_eq!(actual.count_offset, expected_calibration.count_offset);
                assert_eq!(actual.iso_gain, expected_calibration.iso_gain);
                assert_eq!(actual.output_to_srgb, expected_calibration.output_to_srgb);
                assert_eq!(
                    actual.color_dq.is_some(),
                    expected_calibration.color_dq.is_some()
                );
                if let (Some(actual), Some(expected)) =
                    (&actual.color_dq, &expected_calibration.color_dq)
                {
                    assert_eq!(
                        actual.apply([100.0, 500.0, 900.0]),
                        expected.apply([100.0, 500.0, 900.0])
                    );
                }
            }
            checked += 1;
        }
        if checked == 0 {
            eprintln!("skip: set X3F_TEST_FILES for scene Color Mode calibration coverage");
        }
    }

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
