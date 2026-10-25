//! Shared merge preparation and output conversion for X3I.

use x3f_sys::{self as sys, Control};

use crate::output::dng::metadata::mat3_mul;
use crate::x3i_merge::{self, MergedScene};
use crate::{
    ColorEncoding, ConversionStage, Error, Image, ImageLevels, ProcessOptions, Reader, X3iContainer,
};

pub(crate) fn merge_frames(
    container: &X3iContainer,
    frame0: &Reader,
    options: &ProcessOptions,
    control: Control<'_>,
    on_stage: &mut impl FnMut(ConversionStage),
    warnings: &mut Vec<String>,
) -> Result<MergedScene, Error> {
    let mut merge_options = x3i_merge::MergeOptions::from_env()?;
    merge_options.process = options.clone();
    merge_options.ghost_maps = options.x3i_ghost_maps;
    merge_options.process.denoise_intensity = 0;
    merge_options.saturation =
        Some(merge_options.saturation_for_with_warnings(container, control, warnings)?);
    let mut reported = vec![false; container.frames().len()];
    let mut scene = x3i_merge::merge(
        container.frames().len(),
        |i| {
            if !reported[i] {
                if i != 0 {
                    on_stage(ConversionStage::Decode);
                }
                reported[i] = true;
            } else {
                on_stage(ConversionStage::Process);
            }
            control.check()?;
            // Preparation mutates RAW. Each pass needs a fresh reader, including frame 0.
            let mut frame = container
                .open_frame(i)?
                .merge_frame(&merge_options.process, control)?;
            frame.frame_index = i;
            Ok(frame)
        },
        &merge_options,
        control,
    )?;
    if let Some(active) = frame0.dng_active_area_for_size(scene.height, scene.width) {
        x3i_merge::mask_to_active_area(&mut scene, active, control)?;
    }
    Ok(scene)
}

fn denoise_scale(value: Option<&str>, intensity: u8) -> Result<f32, Error> {
    let scale: f32 = value
        .unwrap_or("0")
        .parse()
        .map_err(|_| Error::InvalidData("invalid X3F_X3I_DENOISE_SCALE".into()))?;
    if !scale.is_finite() || scale < 0.0 {
        return Err(Error::InvalidData(
            "X3F_X3I_DENOISE_SCALE must be finite and nonnegative".into(),
        ));
    }
    Ok((f64::from(scale) * f64::from(intensity.min(10)) / 10.0) as f32)
}

pub(crate) fn read_denoise_scale(
    intensity: u8,
    options: sys::DenoiseOptions,
) -> Result<f32, Error> {
    if intensity == 0 || (!options.luminance && !options.chroma) {
        return Ok(0.0);
    }
    denoise_scale(
        std::env::var("X3F_X3I_DENOISE_SCALE").ok().as_deref(),
        intensity,
    )
}

pub(crate) fn denoise(
    image: &mut Image,
    options: &ProcessOptions,
    scale: f32,
    control: Control<'_>,
) -> Result<bool, Error> {
    if options.denoise_intensity == 0
        || (!options.denoise_options.luminance && !options.denoise_options.chroma)
    {
        return Ok(false);
    }
    let scale = (f64::from(scale) / image.dng_highlight_scale) as f32;
    if scale == 0.0 {
        return Ok(false);
    }
    let control = control
        .with_denoise_backend(options.denoise_backend)
        .with_denoise_options(options.denoise_options)?;
    sys::denoise_u16(
        &mut image.data,
        image.columns,
        image.rows,
        image.channels,
        image.row_stride,
        scale,
        control,
    )?;
    Ok(true)
}

pub(crate) fn restore_denoised(
    scene: &mut MergedScene,
    image: &Image,
    control: Control<'_>,
) -> Result<(), Error> {
    for (row, codes) in scene
        .bmt
        .chunks_exact_mut(scene.width as usize * 3)
        .zip(image.data.chunks_exact(image.row_stride as usize))
    {
        control.check()?;
        for (value, code) in row.iter_mut().zip(codes) {
            *value = (f64::from(*code) * scene.s / 65535.0) as f32;
        }
    }
    Ok(())
}

pub(crate) fn bounds(scene: &MergedScene, reader: &Reader, crop: bool) -> [u32; 4] {
    if crop {
        if let Some(active) = reader.dng_active_area_for_size(scene.height, scene.width) {
            return active;
        }
    }
    [0, 0, scene.height, scene.width]
}

pub(crate) fn multiply(matrix: &[f64; 9], pixel: &[f32]) -> [f64; 3] {
    std::array::from_fn(|r| {
        (0..3)
            .map(|c| matrix[r * 3 + c] * f64::from(pixel[c]))
            .sum()
    })
}

pub(crate) struct Rendering {
    matrix: [f64; 9],
    lut: Vec<f64>,
    mode: Option<(
        sys::ColorModeSensor,
        [f64; 9],
        [f64; 3],
        sys::ColorModeRenderLuts,
    )>,
    native: bool,
}

impl Rendering {
    pub(crate) fn resolve(
        reader: &Reader,
        options: &ProcessOptions,
        control: Control<'_>,
        warnings: &mut Vec<String>,
    ) -> Result<Self, Error> {
        if options.spp_denoise {
            return Err(Error::InvalidData(
                "Merrill denoise does not support X3I input".into(),
            ));
        }
        if matches!(
            options.color_encoding,
            ColorEncoding::Unprocessed | ColorEncoding::Qtop
        ) {
            return Err(Error::InvalidData(
                "X3I raw output requires a selected frame".into(),
            ));
        }
        if options.color_mode.is_some()
            && (options.cineon || options.color_encoding == ColorEncoding::None)
        {
            return Err(Error::InvalidData(
                "color modes require a rendered color encoding without Cineon".into(),
            ));
        }
        let mode = if let Some(mode) = options.color_mode {
            // SAFETY: frame 0 still owns unprocessed RAW and loaded metadata.
            let params = unsafe { sys::resolve_color_mode(reader.x3f.as_ptr(), mode) }?;
            if params.as_shot_fallback {
                warnings.push("Unknown or empty as-shot Color Mode resolved to Standard".into());
            }
            let wb = crate::wb_cstring(Some(
                options.wb.as_deref().unwrap_or(&reader.dng_default_wb()),
            ))?;
            // SAFETY: the reader and WB string outlive calibration.
            let calibration = unsafe {
                sys::color_mode_calibration(reader.x3f.as_ptr(), crate::cwb_ptr(&wb), control)
            }?;
            Some((params, calibration))
        } else {
            None
        };
        let mut matrix = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let mut lut = vec![0.0; 65536];
        // SAFETY: these functions write fixed-size matrices or the allocated LUT.
        unsafe {
            match options.color_encoding {
                ColorEncoding::AdobeRgb => sys::x3f_XYZ_to_AdobeRGB(matrix.as_mut_ptr()),
                ColorEncoding::ProPhotoRgb => {
                    let mut adapt = [0.0; 9];
                    sys::x3f_Bradford_D65_to_D50(adapt.as_mut_ptr());
                    sys::x3f_XYZ_to_ProPhotoRGB(matrix.as_mut_ptr());
                    matrix = mat3_mul(&matrix, &adapt);
                }
                _ => {}
            }
            if matches!(
                options.color_encoding,
                ColorEncoding::AdobeRgb | ColorEncoding::ProPhotoRgb
            ) {
                let mut srgb_to_xyz = [0.0; 9];
                sys::x3f_sRGB_to_XYZ(srgb_to_xyz.as_mut_ptr());
                matrix = mat3_mul(&matrix, &srgb_to_xyz);
            }
            if options.cineon {
                let scale = std::env::var("X3F_CINEON_SCALE")
                    .ok()
                    .and_then(|v| v.parse::<f64>().ok())
                    .filter(|v| v.is_finite() && *v > 0.0)
                    .unwrap_or(100.0);
                sys::x3f_cineon_log_LUT(lut.as_mut_ptr(), lut.len() as i32, 65535, scale);
            } else {
                match options.color_encoding {
                    ColorEncoding::Srgb => {
                        sys::x3f_sRGB_LUT(lut.as_mut_ptr(), lut.len() as i32, 65535)
                    }
                    ColorEncoding::AdobeRgb => {
                        sys::x3f_gamma_LUT(lut.as_mut_ptr(), lut.len() as i32, 65535, 2.2)
                    }
                    ColorEncoding::ProPhotoRgb => {
                        sys::x3f_gamma_LUT(lut.as_mut_ptr(), lut.len() as i32, 65535, 1.8)
                    }
                    _ => sys::x3f_gamma_LUT(lut.as_mut_ptr(), lut.len() as i32, 65535, 1.0),
                }
            }
        }
        let mode = mode
            .map(|(params, calibration)| {
                let luts = sys::ColorModeRenderLuts::from_color_mode(
                    &params,
                    0.0,
                    mat3_mul(&matrix, &calibration.output_to_srgb),
                    2,
                    control,
                )?;
                let (matrix, offset) = calibration.render_matrix(&params);
                Ok::<_, Error>((params.sensor, matrix, offset, luts))
            })
            .transpose()?;
        Ok(Self {
            matrix,
            lut,
            mode,
            native: options.color_encoding == ColorEncoding::None,
        })
    }

    pub(crate) fn render(
        &self,
        scene: &MergedScene,
        reader: &Reader,
        crop: bool,
        control: Control<'_>,
    ) -> Result<Image, Error> {
        let [top, left, bottom, right] = bounds(scene, reader, crop);
        let mut data = Vec::with_capacity((bottom - top) as usize * (right - left) as usize * 3);
        let matrix = if self.native {
            self.matrix
        } else {
            mat3_mul(&self.matrix, &scene.camera_to_rgb)
        };
        for row in top..bottom {
            control.check()?;
            for col in left..right {
                let offset = (row as usize * scene.width as usize + col as usize) * 3;
                let bmt = &scene.bmt[offset..offset + 3];
                let rgb = if let Some((sensor, mode_matrix, offset, luts)) = &self.mode {
                    let counts = multiply(mode_matrix, bmt);
                    let indices = std::array::from_fn(|c| {
                        sys::color_mode_tone_index(*sensor, counts[c] + offset[c])
                    });
                    luts.render_linear_indices(indices)
                        .map(|v| f64::from(v) / 65535.0)
                } else {
                    multiply(&matrix, bmt)
                };
                for value in rgb {
                    // SAFETY: the LUT contains all 65536 entries for this call.
                    data.push(unsafe {
                        sys::x3f_LUT_lookup(
                            self.lut.as_ptr().cast_mut(),
                            self.lut.len() as i32,
                            value,
                        )
                    });
                }
            }
        }
        Ok(Image {
            data,
            rows: bottom - top,
            columns: right - left,
            channels: 3,
            row_stride: (right - left) * 3,
            levels: ImageLevels {
                black: [0.0; 3],
                white: [65535; 3],
            },
            dng_highlight_scale: 1.0,
            dng_shoulder_ceiling: 1.0,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rendered_samples_use_the_camera_matrix_transfer_and_clip() {
        let mut bytes = b"FOVb".to_vec();
        bytes.extend(0x40000u32.to_le_bytes());
        bytes.extend([0; 16]);
        bytes.extend(b"SECp");
        for word in [0x20000u32, 0, 0, 0, 0] {
            bytes.extend(word.to_le_bytes());
        }
        bytes.extend(b"SECd");
        for word in [0x20000u32, 1, 24, 24, u32::from_le_bytes(*b"PROP"), 48] {
            bytes.extend(word.to_le_bytes());
        }
        let input = std::env::temp_dir().join(format!("x3i-render-{}.x3f", std::process::id()));
        std::fs::write(&input, &bytes).unwrap();
        let reader = Reader::open(&input).unwrap();
        let scene = MergedScene {
            bmt: vec![0.25, 0.25, 1.0, -1.0, 2.0, 3.0],
            width: 2,
            height: 1,
            s: 3.0,
            camera_to_rgb: [2.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.5],
            report: Default::default(),
            ghost_maps: Vec::new(),
        };
        let control = Control::none();
        let rendering = Rendering::resolve(
            &reader,
            &ProcessOptions::default(),
            control,
            &mut Vec::new(),
        )
        .unwrap();
        let mut image = rendering.render(&scene, &reader, false, control).unwrap();
        let encode =
            |value: f64| ((1.055 * value.powf(1.0 / 2.4) - 0.055) * 65535.0).round() as u16;
        assert_eq!(
            image.data,
            [encode(0.5), encode(0.25), encode(0.5), 0, 65535, 65535]
        );
        let disabled = ProcessOptions {
            denoise_options: sys::DenoiseOptions {
                luminance: false,
                chroma: false,
                ..Default::default()
            },
            ..Default::default()
        };
        let unchanged = image.data.clone();
        assert!(!denoise(&mut image, &disabled, 0.3, control).unwrap());
        assert_eq!(image.data, unchanged);
        let options = ProcessOptions {
            color_encoding: ColorEncoding::None,
            ..Default::default()
        };
        let rendering = Rendering::resolve(&reader, &options, control, &mut Vec::new()).unwrap();
        assert_eq!(
            rendering
                .render(&scene, &reader, false, control)
                .unwrap()
                .data,
            [16384, 16384, 65535, 0, 65535, 65535]
        );
        let cancelled = std::sync::atomic::AtomicBool::new(true);
        assert!(matches!(
            rendering.render(&scene, &reader, false, Control::new(&cancelled)),
            Err(Error::Cancelled)
        ));
        drop(reader);
        std::fs::remove_file(input).unwrap();
    }

    #[test]
    fn enabled_denoise_restores_reference_codes_and_preserves_black_margins() {
        let mut scene = MergedScene {
            bmt: vec![0.0; 8 * 8 * 3],
            width: 8,
            height: 8,
            s: 3.0,
            camera_to_rgb: [0.0; 9],
            report: Default::default(),
            ghost_maps: Vec::new(),
        };
        for row in 1..7 {
            for col in 0..8 {
                let offset = (row * 8 + col) * 3;
                scene.bmt[offset..offset + 3].copy_from_slice(&[
                    0.3 + (col % 3) as f32 * 0.001,
                    0.3 - (col % 2) as f32 * 0.001,
                    0.3,
                ]);
            }
        }
        let mut image = x3i_merge::to_image(&scene).unwrap();
        let options = ProcessOptions {
            denoise_backend: sys::DenoiseBackend::Cpu,
            ..Default::default()
        };
        assert!(denoise(
            &mut image,
            &options,
            denoise_scale(Some("0.3"), 10).unwrap(),
            Control::none()
        )
        .unwrap());
        restore_denoised(&mut scene, &image, Control::none()).unwrap();
        for (value, code) in scene.bmt.iter().zip(&image.data) {
            assert!((f64::from(*value) * 65535.0 / scene.s - f64::from(*code)).abs() <= 1.0);
        }
        assert!(scene.bmt[..8 * 3].iter().all(|v| *v == 0.0));
        assert!(scene.bmt[7 * 8 * 3..].iter().all(|v| *v == 0.0));
    }

    #[test]
    fn reduced_denoise_strength_tracks_intensity() {
        assert_eq!(denoise_scale(None, 10).unwrap(), 0.0);
        assert_eq!(denoise_scale(Some("0.3"), 5).unwrap(), 0.3 * 0.5);
        assert_eq!(denoise_scale(Some("0"), 10).unwrap(), 0.0);
        for invalid in ["-1", "NaN", "inf", "bad"] {
            assert!(denoise_scale(Some(invalid), 10).is_err());
        }
    }
}
