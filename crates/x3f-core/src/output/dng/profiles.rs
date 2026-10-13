//! Legacy and native DNG camera-profile payloads.
//!
//! The primary profile uses IFD0. Extra profiles use an IFD after an
//! eight-byte profile header, with offsets relative to that header.
//! Legacy MMCR profiles keep their original bytes. Explicit Color Modes
//! use IIRC with classic matrices, a tone curve and a LookTable.

use std::ffi::CString;

use crate::{Error, Image, Preview, Reader};

use super::classic_profile as classic;
use super::color::ColorCalibration;
use super::metadata::{
    adobe_rgb_to_xyz, bradford_d65_to_d50, mat3_diag, mat3_inverse, mat3_mul, mat3_ones,
    mat3_to_f32,
};
#[cfg(test)]
use super::rgb_table::xyz_d50_to_prophoto;
use super::rgb_table::{multiply, NativeRendering};
use super::tags as t;
use super::tiff_writer::{DirectoryWriter, Value};

pub(super) struct PreparedColorModes {
    modes: Vec<x3f_sys::ColorModeParameters>,
    selected: usize,
    native: x3f_sys::ColorModeCalibration,
}

pub(super) struct ColorModeProfiles {
    name: String,
    color: [f32; 9],
    forward: [f32; 9],
    fit: classic::ClassicFit,
    pub(super) white_xy: [f64; 2],
    exposure_offset: f32,
    pub(super) extra_blob: Vec<u8>,
    pub(super) extra_offsets: Vec<u32>,
    rendering: NativeRendering,
}

impl PreparedColorModes {
    pub(super) fn resolve(
        reader: &Reader,
        wb: &str,
        selection: x3f_sys::ColorMode,
        warnings: &mut Vec<String>,
        control: x3f_sys::Control<'_>,
    ) -> Result<Self, Error> {
        control.check()?;
        // Resolve before raw processing consumes the decoded sensor buffer.
        let selected = unsafe { x3f_sys::resolve_color_mode(reader.x3f.as_ptr(), selection) }?;
        if selected.as_shot_fallback {
            warnings.push("Unknown or empty as-shot Color Mode resolved to Standard".to_owned());
        }
        let available = unsafe { x3f_sys::available_color_modes(reader.x3f.as_ptr()) }?;
        let index = available
            .iter()
            .position(|&mode| mode == selected.mode)
            .ok_or(x3f_sys::Error::InvalidData(
                "selected DNG Color Mode is unavailable for this file",
            ))?;
        let mut modes = Vec::with_capacity(available.len());
        for mode in available {
            control.check()?;
            modes.push(if mode == selected.mode {
                selected
            } else {
                unsafe { x3f_sys::resolve_color_mode(reader.x3f.as_ptr(), mode) }?
            });
        }
        let wb = CString::new(wb)
            .map_err(|_| x3f_sys::Error::InvalidData("white balance contains NUL"))?;
        let native = unsafe {
            x3f_sys::color_mode_calibration(reader.x3f.as_ptr(), wb.as_ptr().cast_mut(), control)
        }?;
        Ok(Self {
            modes,
            selected: index,
            native,
        })
    }

    pub(super) fn finish(
        self,
        reader: &Reader,
        wb: &str,
        calibration: &ColorCalibration,
        highlight_scale: f64,
        _warnings: &mut Vec<String>,
        control: x3f_sys::Control<'_>,
    ) -> Result<ColorModeProfiles, Error> {
        let bmt = reader
            .dng_bmt_to_xyz(Some(wb))
            .ok_or(x3f_sys::Error::InvalidData(
                "missing DNG physical color calibration",
            ))?;
        let color0 = classic::sdk_color(classic::serialized(
            calibration.fold_color_matrix(&checked_inverse(&bmt)?),
        ));
        let neutral = calibration
            .neutral
            .map(|v| (v * 10_000.0).round().max(1.0) / 10_000.0);
        let maximum = neutral.into_iter().fold(0.0, f64::max);
        let neutral = neutral.map(|v| (v / maximum).clamp(0.001, 1.0));
        let white_xy = classic::white_xy(color0, neutral)?;
        let standard = self
            .modes
            .iter()
            .find(|p| p.mode == x3f_sys::ColorMode::Standard)
            .ok_or(x3f_sys::Error::InvalidData("missing Standard calibration"))?;
        let (counts, _) = self.native.render_matrix(standard);
        let basis = mat3_mul(
            &super::rgb_table::srgb_to_xyz_d50(),
            &super::rgb_table::quantized_output(self.native.output_to_srgb),
        );
        let count_scale = multiply(&mat3_mul(&basis, &counts), neutral)[1];
        if !count_scale.is_finite() || count_scale <= 0.0 {
            return Err(x3f_sys::Error::InvalidData("invalid native neutral count scale").into());
        }
        let baseline =
            super::camera_baseline_exposure(reader, calibration, highlight_scale).unwrap_or(0.0);
        let (exposure_offset, exposure) = profile_exposure(baseline as f32, 0.0)?;
        let mut extra_blob = Vec::new();
        let mut extra_offsets = Vec::new();
        let mut primary = None;
        for (index, mode) in self.modes.iter().enumerate() {
            control.check()?;
            let matrices =
                classic::Matrices::new(&self.native, mode, color0, neutral, white_xy, count_scale)?;
            let rendering = NativeRendering::new(&self.native, mode, control)?;
            let fit = classic::ClassicFit::new(
                &matrices,
                &rendering,
                highlight_scale,
                exposure,
                control,
            )?;
            let name = format!("x3fuse {}", mode.mode.name());
            let color = mat3_to_f32(&matrices.color);
            let forward = mat3_to_f32(&matrices.forward);
            if index == self.selected {
                primary = Some((name, color, forward, fit, rendering));
            } else {
                append_profile(
                    &mut extra_blob,
                    &mut extra_offsets,
                    &build_profile_blob(
                        &name,
                        color,
                        forward,
                        Some(&fit.curve.knots),
                        None,
                        Some(ProfileLook::Classic(&fit.map)),
                        exposure_offset,
                    ),
                )?;
            }
        }
        // WhiteXY needs a row correction to retain each legacy as-shot neutral.
        let physical_white = multiply(
            &color0,
            classic::xy_xyz(white_xy.map(|v| (v * 1_000_000.0).round() / 1_000_000.0)),
        );
        let maximum = physical_white.into_iter().fold(0.0, f64::max);
        let physical_white = physical_white.map(|v| v / maximum);
        for profile in &PROFILES[1..] {
            let (color, forward) = compute_matrices(reader, wb, calibration, profile).ok_or(
                x3f_sys::Error::InvalidData("missing legacy synthetic profile calibration"),
            )?;
            let color = classic::legacy_color(color.map(f64::from), physical_white, white_xy)?;
            append_profile(
                &mut extra_blob,
                &mut extra_offsets,
                &build_mmcr_profile(profile.name, mat3_to_f32(&color), forward, None, None),
            )?;
        }
        let (name, color, forward, fit, rendering) = primary.ok_or(x3f_sys::Error::InvalidData(
            "missing selected DNG Color Mode profile",
        ))?;
        Ok(ColorModeProfiles {
            name,
            color,
            forward,
            fit,
            white_xy,
            exposure_offset,
            extra_blob,
            extra_offsets,
            rendering,
        })
    }
}

pub(super) fn checked_inverse(matrix: &[f64; 9]) -> x3f_sys::Result<[f64; 9]> {
    let inverse = mat3_inverse(matrix);
    let norm = |m: &[f64; 9]| {
        m.as_chunks::<3>()
            .0
            .iter()
            .map(|r| r.iter().map(|v| v.abs()).sum::<f64>())
            .fold(0.0, f64::max)
    };
    if matrix.iter().chain(&inverse).any(|v| !v.is_finite()) || norm(matrix) * norm(&inverse) > 1e8
    {
        return Err(x3f_sys::Error::InvalidData(
            "singular or ill-conditioned DNG color calibration",
        ));
    }
    Ok(inverse)
}

fn profile_exposure(baseline: f32, target: f64) -> Result<(f32, f64), Error> {
    if !baseline.is_finite()
        || baseline.abs() > i32::MAX as f32 / 10000.0
        || !target.is_finite()
        || target < 0.0
        || target > i32::MAX as f64 / 10000.0
    {
        return Err(x3f_sys::Error::InvalidData("invalid DNG profile exposure").into());
    }
    let (n, d) = srational_pair(f64::from(baseline), 10_000);
    let baseline = f64::from(n) / f64::from(d);
    let offset = (target - baseline) as f32;
    let (n, d) = srational_pair(f64::from(offset), 10_000);
    let total = baseline + f64::from(n) / f64::from(d);
    if !offset.is_finite() || offset.abs() > i32::MAX as f32 / 10000.0 || total < 0.0 {
        return Err(x3f_sys::Error::InvalidData("DNG profile exposure cannot fit metadata").into());
    }
    Ok((offset, total))
}

fn append_profile(blob: &mut Vec<u8>, offsets: &mut Vec<u32>, profile: &[u8]) -> Result<(), Error> {
    if !blob.len().is_multiple_of(2) {
        blob.push(0);
    }
    let offset = u32::try_from(blob.len())
        .map_err(|_| x3f_sys::Error::InvalidData("DNG profile offset exceeds 32 bits"))?;
    let total = blob
        .len()
        .checked_add(profile.len())
        .filter(|&n| n <= u32::MAX as usize)
        .ok_or(x3f_sys::Error::InvalidData(
            "DNG profile data exceeds 32 bits",
        ))?;
    blob.reserve(total - blob.len());
    blob.extend_from_slice(profile);
    offsets.push(offset);
    Ok(())
}

impl ColorModeProfiles {
    pub(super) fn write_default(&self, ifd: &mut DirectoryWriter) {
        add_profile_tags(ifd, &self.name, self.color, self.forward);
        ifd.add(
            t::CALIBRATION_ILLUMINANT1,
            Value::Short(vec![t::CALIB_ILLUMINANT_D65]),
        );
        ifd.add(
            t::PROFILE_TONE_CURVE,
            Value::Float(self.fit.curve.knots.clone()),
        );
        ifd.add(
            t::PROFILE_LOOK_TABLE_DIMS,
            Value::Long(classic::LOOK_DIMS.map(|v| v as u32).to_vec()),
        );
        ifd.add(
            t::PROFILE_LOOK_TABLE_DATA,
            Value::Float(self.fit.map.clone()),
        );
        ifd.add(t::PROFILE_LOOK_TABLE_ENCODING, Value::Long(vec![1]));
        ifd.add(
            t::BASELINE_EXPOSURE_OFFSET,
            srational_from_floats(&[self.exposure_offset], 10_000),
        );
        ifd.add(
            t::AS_SHOT_PROFILE_NAME,
            Value::Ascii(CString::new(self.name.as_str()).expect("profile name without NUL")),
        );
        ifd.add(t::DNG_VERSION, Value::Byte(t::DNG_VERSION_1_4_0_0.to_vec()));
        ifd.add(
            t::DNG_BACKWARD_VERSION,
            Value::Byte(t::DNG_VERSION_1_4_0_0.to_vec()),
        );
    }

    pub(super) fn preview(
        &self,
        image: &Image,
        crop: Option<super::strip::CropWindow>,
        max_width: u32,
        control: x3f_sys::Control<'_>,
    ) -> Result<Preview, Error> {
        let (top, left, rows, columns) = crop.unwrap_or((0, 0, image.rows, image.columns));
        if rows == 0
            || columns == 0
            || max_width == 0
            || (0..3).any(|c| {
                !image.levels.black[c].is_finite()
                    || image.levels.white[c] as f64 <= image.levels.black[c]
            })
        {
            return Err(
                x3f_sys::Error::InvalidData("invalid DNG preview dimensions or levels").into(),
            );
        }
        let reduction = columns.div_ceil(max_width);
        let width = columns.div_ceil(reduction);
        let height = rows.div_ceil(reduction);
        let mut data = Vec::with_capacity((width * height * 3) as usize);
        for row in 0..height {
            control.check()?;
            for col in 0..width {
                let mut input = [0.0; 3];
                let mut count = 0u32;
                for y in row * reduction..((row + 1) * reduction).min(rows) {
                    for x in col * reduction..((col + 1) * reduction).min(columns) {
                        for (c, sum) in input.iter_mut().enumerate() {
                            *sum += (image.sample(top + y, left + x, c as u32) as f64
                                - image.levels.black[c])
                                / (image.levels.white[c] as f64 - image.levels.black[c]);
                        }
                        count += 1;
                    }
                }
                let input = input.map(|v| v / f64::from(count) * image.dng_highlight_scale);
                data.extend(self.rendering.sensor_to_srgb8(input));
            }
        }
        Ok(Preview {
            data,
            rows: height,
            columns: width,
            channels: 3,
            row_stride: width * 3,
        })
    }
}

/// Matrix source for a legacy profile.
#[derive(Clone, Copy)]
enum BmtSource {
    /// Use the body's legacy calibration. Explicit modes use native calibration.
    Default,
    /// Use Adobe RGB for legacy grayscale and Unconverted profiles.
    AdobeRgb,
}

#[derive(Clone, Copy)]
pub(crate) struct CameraProfile {
    pub name: &'static str,
    bmt_source: BmtSource,
    grayscale_mix: Option<[f64; 3]>,
    /// CAMF entry whose presence gates this profile. None = always emit;
    /// `Some(name)` = emit only when the file's CAMF carries that key.
    /// Used to keep Merrill firmware (which only ships 6 of the 11
    /// in-camera color modes) from advertising Quattro-only modes
    /// like FCYellow / Cinema / ForestGreen / SunsetRed / Monochrome.
    requires_camf: Option<&'static str>,
    /// `CMCM_<mode>` 3×3 CCM CAMF entry to fold into `bmt_to_xyz`.
    /// `None` = no CCM (used for grayscale / unconverted, which have
    /// their own bmt_source). When the named matrix is absent (older
    /// firmware), we silently fall back to identity rather than
    /// dropping the profile, so the profile's name stays visible
    /// even on bodies that don't carry the per-mode CCM.
    cmcm_camf: Option<&'static str>,
    /// CMCC contrast-parameter CAMF entry that drives the
    /// `ProfileToneCurve` for this profile. `None` = no tone curve
    /// (used for grayscale / unconverted, where no contrast bend
    /// applies). The CAMF value is a single float in `[-0.3, +0.3]`
    /// that we feed through `tone_curves::build_curve`.
    cmcc_camf: Option<&'static str>,
    /// `MultiAxisTable_<mode>` CAMF entry that drives the
    /// `ProfileHueSatMapData1` for this profile. `None` = no hue/sat
    /// map (grayscale / unconverted). The CAMF value is a `float[2][5][21]`
    /// table that `hue_sat_map::build_hue_sat_map` collapses to a
    /// 21×2×1 DNG hue/sat map.
    multi_axis_camf: Option<&'static str>,
}

/// All known camera profiles. The 11 in-camera "color modes" Sigma writes
/// into Quattro-era CAMF (`CMCM_*`) come first in in-camera menu order;
/// Merrill files have a subset (Standard/Vivid/Neutral/Portrait/Landscape/
/// FCBlue) and the rest are silently skipped at emit time via
/// `requires_camf`. The Grayscale + Unconverted entries are synthetic
/// profiles the legacy C writer also emitted, with their own forward
/// matrices.
///
/// Profile names match Sigma's CAMF short-form (`FCBlue`, `FCYellow`,
/// `ForestGreen`, `SunsetRed`) rather than the verbose
/// `FOV Classic Blue` etc., to align with the reference Sigma-aware
/// DNG output that Lightroom recognises by name.
const PROFILES: &[CameraProfile] = &[
    CameraProfile {
        name: "Standard",
        bmt_source: BmtSource::Default,
        grayscale_mix: None,
        requires_camf: None,
        cmcm_camf: Some("CMCM_Standard"),
        cmcc_camf: Some("CMCC_Standard"),
        multi_axis_camf: Some("MultiAxisTable_Standard"),
    },
    // Retain the historical disabled entries. Explicit modes use the
    // shared native calibration and resolver instead of these profiles.
    /*
    CameraProfile {
        name: "Vivid",
        bmt_source: BmtSource::Default,
        grayscale_mix: None,
        requires_camf: Some("CMCM_Vivid"),
        cmcm_camf: Some("CMCM_Vivid"),
        cmcc_camf: Some("CMCC_Vivid"),
        multi_axis_camf: Some("MultiAxisTable_Vivid"),
    },
    CameraProfile {
        name: "Neutral",
        bmt_source: BmtSource::Default,
        grayscale_mix: None,
        requires_camf: Some("CMCM_Neutral"),
        cmcm_camf: Some("CMCM_Neutral"),
        cmcc_camf: Some("CMCC_Neutral"),
        multi_axis_camf: Some("MultiAxisTable_Neutral"),
    },
    CameraProfile {
        name: "Portrait",
        bmt_source: BmtSource::Default,
        grayscale_mix: None,
        requires_camf: Some("CMCM_Portrait"),
        cmcm_camf: Some("CMCM_Portrait"),
        cmcc_camf: Some("CMCC_Portrait"),
        multi_axis_camf: Some("MultiAxisTable_Portrait"),
    },
    CameraProfile {
        name: "Landscape",
        bmt_source: BmtSource::Default,
        grayscale_mix: None,
        requires_camf: Some("CMCM_Landscape"),
        cmcm_camf: Some("CMCM_Landscape"),
        cmcc_camf: Some("CMCC_Landscape"),
        multi_axis_camf: Some("MultiAxisTable_Landscape"),
    },
    CameraProfile {
        name: "FCBlue",
        bmt_source: BmtSource::Default,
        grayscale_mix: None,
        // FCBlue is special-cased as the *default* profile when
        // `header.color_mode == "FCBlue"`; it's also always emitted (no
        // gating CAMF) so older firmware that lacks CMCM_FCBlue still
        // gets the named profile.
        requires_camf: None,
        cmcm_camf: Some("CMCM_FCBlue"),
        cmcc_camf: Some("CMCC_FCBlue"),
        multi_axis_camf: Some("MultiAxisTable_FCBlue"),
    },
    CameraProfile {
        name: "FCYellow",
        bmt_source: BmtSource::Default,
        grayscale_mix: None,
        requires_camf: Some("CMCM_FCYellow"),
        cmcm_camf: Some("CMCM_FCYellow"),
        cmcc_camf: Some("CMCC_FCYellow"),
        multi_axis_camf: Some("MultiAxisTable_FCYellow"),
    },
    CameraProfile {
        name: "Cinema",
        bmt_source: BmtSource::Default,
        grayscale_mix: None,
        requires_camf: Some("CMCM_Cinema"),
        cmcm_camf: Some("CMCM_Cinema"),
        cmcc_camf: Some("CMCC_Cinema"),
        multi_axis_camf: Some("MultiAxisTable_Cinema"),
    },
    CameraProfile {
        name: "ForestGreen",
        bmt_source: BmtSource::Default,
        grayscale_mix: None,
        requires_camf: Some("CMCM_ForestGreen"),
        cmcm_camf: Some("CMCM_ForestGreen"),
        cmcc_camf: Some("CMCC_ForestGreen"),
        multi_axis_camf: Some("MultiAxisTable_ForestGreen"),
    },
    CameraProfile {
        name: "SunsetRed",
        bmt_source: BmtSource::Default,
        grayscale_mix: None,
        requires_camf: Some("CMCM_SunsetRed"),
        cmcm_camf: Some("CMCM_SunsetRed"),
        cmcc_camf: Some("CMCC_SunsetRed"),
        multi_axis_camf: Some("MultiAxisTable_SunsetRed"),
    },
    CameraProfile {
        name: "Monochrome",
        bmt_source: BmtSource::Default,
        grayscale_mix: None,
        requires_camf: Some("CMCM_Monochrome"),
        cmcm_camf: Some("CMCM_Monochrome"),
        cmcc_camf: Some("CMCC_Monochrome"),
        multi_axis_camf: Some("MultiAxisTable_Monochrome"),
    },
    */
    CameraProfile {
        name: "Monochrome",
        bmt_source: BmtSource::AdobeRgb,
        grayscale_mix: Some([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]),
        requires_camf: None,
        cmcm_camf: None,
        cmcc_camf: None,
        multi_axis_camf: None,
    },
    CameraProfile {
        name: "Monochrome (red filter)",
        bmt_source: BmtSource::AdobeRgb,
        grayscale_mix: Some([2.0, -1.0, 0.0]),
        requires_camf: None,
        cmcm_camf: None,
        cmcc_camf: None,
        multi_axis_camf: None,
    },
    CameraProfile {
        name: "Monochrome (yellow filter)",
        bmt_source: BmtSource::AdobeRgb,
        grayscale_mix: Some([1.5, -0.5, 0.0]),
        requires_camf: None,
        cmcm_camf: None,
        cmcc_camf: None,
        multi_axis_camf: None,
    },
    CameraProfile {
        name: "Monochrome (green filter)",
        bmt_source: BmtSource::AdobeRgb,
        grayscale_mix: Some([-0.5, 2.0, -0.5]),
        requires_camf: None,
        cmcm_camf: None,
        cmcc_camf: None,
        multi_axis_camf: None,
    },
    CameraProfile {
        name: "Monochrome (blue filter)",
        bmt_source: BmtSource::AdobeRgb,
        grayscale_mix: Some([0.0, -1.0, 2.0]),
        requires_camf: None,
        cmcm_camf: None,
        cmcc_camf: None,
        multi_axis_camf: None,
    },
    CameraProfile {
        name: "Unconverted",
        bmt_source: BmtSource::AdobeRgb,
        grayscale_mix: None,
        requires_camf: None,
        cmcm_camf: None,
        cmcc_camf: None,
        multi_axis_camf: None,
    },
];

/// CIE XYZ values of the D50 reference illuminant. Used when computing
/// ForwardMatrix1 for grayscale profiles.
const D50_XYZ: [f64; 3] = [0.96422, 1.00000, 0.82521];

/// (color_matrix1, forward_matrix1) pair as DNG-tag-ready f32 row-major.
fn compute_matrices(
    reader: &Reader,
    wb: &str,
    calibration: &ColorCalibration,
    profile: &CameraProfile,
) -> Option<([f32; 9], [f32; 9])> {
    // Body-cal-gating: skip profiles whose required CAMF entry isn't in
    // the file (e.g. Merrill firmware doesn't ship FCYellow / Cinema /
    // etc.).
    if let Some(camf) = profile.requires_camf {
        reader.dng_camf_matrix_3x3(camf)?;
    }

    let base_bmt_to_xyz = match profile.bmt_source {
        BmtSource::Default => reader.dng_bmt_to_xyz(Some(wb))?,
        BmtSource::AdobeRgb => adobe_rgb_to_xyz(),
    };

    // Fold the per-mode CMCM CCM into bmt_to_xyz so each picture profile
    // gets its own ColorMatrix1 + ForwardMatrix1. Mirrors the legacy
    // `get_bmt_to_xyz_fcblue` path: out_xyz = base_bmt_to_xyz @ CMCM @
    // bmt_input, i.e. CMCM acts on camera-native RGB before the base
    // body calibration. CMCM_Standard is identity; CMCM_Vivid /
    // CMCM_FCBlue / etc. carry the per-mode chroma shifts.
    //
    // If the profile doesn't name a CMCM (grayscale / unconverted) or
    // the CAMF entry is missing on this body, we leave bmt_to_xyz at
    // the base — same effect as identity CMCM.
    let bmt_to_xyz = if let Some(cmcm_name) = profile.cmcm_camf {
        if let Some(cmcm) = reader.dng_camf_matrix_3x3(cmcm_name) {
            mat3_mul(&base_bmt_to_xyz, &cmcm)
        } else {
            base_bmt_to_xyz
        }
    } else {
        base_bmt_to_xyz
    };

    // Fold the individual-camera and relative digital-gain calibration
    // into every profile's mandatory ColorMatrix. A separate diagonal
    // CameraCalibration is ignored by some readers; it cancels from the
    // ForwardMatrix transform, so that matrix remains unchanged.
    let color_matrix1 = mat3_to_f32(&calibration.fold_color_matrix(&mat3_inverse(&bmt_to_xyz)));

    let bmt_to_d50 = if let Some(mix) = profile.grayscale_mix {
        // (D50 × ones) × diag(mix) — see x3f_output_dng.c:206-213
        let mix_diag = mat3_diag(&mix);
        let weighted = mat3_mul(&mat3_ones(), &mix_diag);
        let d50_diag = mat3_diag(&D50_XYZ);
        mat3_mul(&d50_diag, &weighted)
    } else {
        mat3_mul(&bradford_d65_to_d50(), &bmt_to_xyz)
    };
    let forward_matrix1 = mat3_to_f32(&bmt_to_d50);

    Some((color_matrix1, forward_matrix1))
}

/// Pick the index of the profile to use as the file's default. Mirrors
/// `write_camera_profiles`: pre-Quattro-2.3 files always use "Default";
/// 2.3+ files with `color_mode == "FCBlue"` use the `FCBlue` profile.
pub(crate) fn default_profile_index(reader: &Reader) -> usize {
    const X3F_VERSION_2_3: u32 = 0x0002_0003;
    if reader.header_version() >= X3F_VERSION_2_3 && reader.dng_color_mode() == "FCBlue" {
        if let Some(idx) = PROFILES.iter().position(|p| p.name == "FCBlue") {
            return idx;
        }
    }
    0
}

/// Add ColorMatrix1 + ForwardMatrix1 + ProfileName + DefaultBlackRender
/// for a single profile to an existing IFD writer.
fn add_profile_tags(ifd: &mut DirectoryWriter, name: &str, color: [f32; 9], forward: [f32; 9]) {
    // libtiff's stock definition uses SRATIONAL for ColorMatrix; the
    // dngtags extender adds FORWARDMATRIX as SRATIONAL too. Matrix values
    // are well within ±i32::MAX/10000 so the conversion is lossless to
    // 4 decimal places.
    ifd.add(t::COLOR_MATRIX1, srational_from_floats(&color, 10_000));
    ifd.add(t::FORWARD_MATRIX1, srational_from_floats(&forward, 10_000));
    ifd.add(
        t::PROFILE_NAME,
        Value::Ascii(CString::new(name).expect("profile name without NUL")),
    );
    ifd.add(t::DEFAULT_BLACK_RENDER, Value::Long(vec![1]));
}

/// Write the default profile's tags onto the parent DNG IFD0 and append
/// the AsShotProfileName tag. Returns the index of the default profile.
pub(super) fn write_default_profile(
    reader: &Reader,
    wb: &str,
    calibration: &ColorCalibration,
    ifd0: &mut DirectoryWriter,
) -> Option<usize> {
    let idx = default_profile_index(reader);
    let profile = &PROFILES[idx];
    let (color, forward) = compute_matrices(reader, wb, calibration, profile)?;
    add_profile_tags(ifd0, profile.name, color, forward);

    // Tone curve + hue/sat map for the default profile, mirroring what
    // we emit for the MMCR profiles in `build_extra_profiles_blob` so
    // the in-camera "as shot" mode renders the same in Lightroom whether
    // it's selected via Profile dropdown or as the file default.
    if let Some(curve) = profile
        .cmcc_camf
        .and_then(|name| reader.dng_camf_float(name))
        .and_then(super::tone_curves::build_curve)
    {
        ifd0.add(t::PROFILE_TONE_CURVE, Value::Float(curve));
    }
    if let Some(hsm) = profile
        .multi_axis_camf
        .and_then(|name| super::hue_sat_map::build_hue_sat_map(reader, name))
    {
        ifd0.add(
            t::PROFILE_HUE_SAT_MAP_DIMS,
            Value::Long(super::hue_sat_map::HUE_SAT_MAP_DIMS.to_vec()),
        );
        ifd0.add(t::PROFILE_HUE_SAT_MAP_DATA1, Value::Float(hsm));
        ifd0.add(
            t::PROFILE_HUE_SAT_MAP_ENCODING,
            Value::Long(vec![super::hue_sat_map::HUE_SAT_MAP_ENCODING]),
        );
    }

    ifd0.add(
        t::AS_SHOT_PROFILE_NAME,
        Value::Ascii(CString::new(profile.name).expect("profile name without NUL")),
    );
    Some(idx)
}

/// Build the `ExtraCameraProfiles` blob: every non-default profile
/// rendered as a big-endian MMCR mini-TIFF, concatenated with one offset
/// per profile (offsets are relative to the start of the parent file).
///
/// Returns `(blob, per_profile_relative_offsets)`. The parent writer
/// appends `blob` at some absolute file position `base`, then writes the
/// LONG[] tag value as `[base + off for off in offsets]`.
pub(super) fn build_extra_profiles_blob(
    reader: &Reader,
    wb: &str,
    calibration: &ColorCalibration,
    default_idx: usize,
) -> Option<(Vec<u8>, Vec<u32>)> {
    let mut blob: Vec<u8> = Vec::new();
    let mut offsets: Vec<u32> = Vec::new();

    for (i, profile) in PROFILES.iter().enumerate() {
        if i == default_idx {
            continue;
        }
        // CMCM-derived profiles silently drop out when the CAMF entry
        // isn't in the file (e.g. Merrill cameras don't ship FCYellow /
        // Cinema / etc.). Only the *default* profile is allowed to fail
        // the conversion — that lookup happens in `write_default_profile`.
        let Some((color, forward)) = compute_matrices(reader, wb, calibration, profile) else {
            continue;
        };

        // Synthesise a ProfileToneCurve from the profile's CMCC_<mode>
        // contrast parameter. Identity curves (CMCC ≈ 0) are skipped; the
        // CPP reference also omits the tag in that case.
        let tone_curve = profile
            .cmcc_camf
            .and_then(|name| reader.dng_camf_float(name))
            .and_then(super::tone_curves::build_curve);

        // Synthesise a ProfileHueSatMapData1 from the profile's
        // MultiAxisTable_<mode> CAMF entry. Identity tables (no hue
        // shift, sat == 1 everywhere) are skipped — the DNG reader
        // would do nothing with them and emitting them just bloats
        // the blob.
        let hue_sat_map = profile
            .multi_axis_camf
            .and_then(|name| super::hue_sat_map::build_hue_sat_map(reader, name));

        // Two-byte alignment between concatenated profile blobs.
        if !blob.len().is_multiple_of(2) {
            blob.push(0);
        }
        let blob_off = blob.len() as u32;

        let mmcr = build_mmcr_profile(
            profile.name,
            color,
            forward,
            tone_curve.as_deref(),
            hue_sat_map.as_deref(),
        );
        blob.extend_from_slice(&mmcr);
        offsets.push(blob_off);
    }

    Some((blob, offsets))
}

/// Keep the original big-endian MMCR layout for legacy profiles.
fn build_mmcr_profile(
    name: &str,
    color: [f32; 9],
    forward: [f32; 9],
    tone_curve: Option<&[f32]>,
    hue_sat_map: Option<&[f32]>,
) -> Vec<u8> {
    build_profile_blob(name, color, forward, tone_curve, hue_sat_map, None, 0.0)
}

#[derive(Clone, Copy)]
enum ProfileLook<'a> {
    Classic(&'a [f32]),
}

impl<'a> ProfileLook<'a> {
    fn classic(self) -> Option<&'a [f32]> {
        match self {
            Self::Classic(table) => Some(table),
        }
    }
}

/// Native profiles use IIRC. Legacy profiles keep the original MMCR layout.
fn build_profile_blob(
    name: &str,
    color: [f32; 9],
    forward: [f32; 9],
    tone_curve: Option<&[f32]>,
    hue_sat_map: Option<&[f32]>,
    look: Option<ProfileLook<'_>>,
    exposure_offset: f32,
) -> Vec<u8> {
    let look_table = look.and_then(ProfileLook::classic);
    let little = look.is_some();
    let short = |v: u16| {
        if little {
            v.to_le_bytes()
        } else {
            v.to_be_bytes()
        }
    };
    let long = |v: u32| {
        if little {
            v.to_le_bytes()
        } else {
            v.to_be_bytes()
        }
    };
    let name_c = CString::new(name).expect("profile name without NUL");
    let name_bytes = name_c.as_bytes_with_nul();

    // Place the IFD at offset 8, then its external values. Adobe rejects
    // the older layout that put external values before the profile IFD.
    // Legacy profiles omit illuminant, LookTable and exposure-offset tags.
    // We have to know each entry's value-or-offset slot before we can
    // emit the IFD body. So lay out the externals first (in a side
    // buffer), then write the IFD body referencing them, then append the
    // externals onto the blob at the tail.
    let header_bytes: usize = 8;
    let mut entry_count: u16 = 5; // base set: Compression, CM1, Name, FM1, DefaultBlackRender
    if hue_sat_map.is_some() {
        entry_count += 3; // Dims + Data1 + Encoding
    }
    if tone_curve.is_some() {
        entry_count += 1;
    }
    if look.is_some() {
        entry_count += 2; // Illuminant + exposure offset
    }
    if look_table.is_some() {
        entry_count += 3; // Dims + data + encoding
    }
    let ifd_body_len: usize = 2 + (entry_count as usize) * 12 + 4;
    let externals_start: usize = header_bytes + ifd_body_len;

    // Build the externals section.
    let mut ext: Vec<u8> = Vec::new();
    fn pad2(v: &mut Vec<u8>) {
        if !v.len().is_multiple_of(2) {
            v.push(0);
        }
    }
    pad2(&mut ext);
    let color_off = (externals_start + ext.len()) as u32;
    for v in color {
        let (n, d) = srational_pair(v as f64, 10_000);
        ext.extend_from_slice(&long(n as u32));
        ext.extend_from_slice(&long(d as u32));
    }
    pad2(&mut ext);
    let forward_off = (externals_start + ext.len()) as u32;
    for v in forward {
        let (n, d) = srational_pair(v as f64, 10_000);
        ext.extend_from_slice(&long(n as u32));
        ext.extend_from_slice(&long(d as u32));
    }
    let tone_curve_meta = tone_curve.map(|curve| {
        // FLOAT is 4 bytes / 4-byte aligned; ext is currently 8-aligned
        // (matrix payloads are 8 byte multiples) so no extra pad needed.
        let off = (externals_start + ext.len()) as u32;
        for &v in curve {
            ext.extend_from_slice(&long(v.to_bits()));
        }
        (off, curve.len() as u32)
    });
    let hue_sat_map_meta = hue_sat_map.map(|hsm| {
        // Both Dims (LONG[3], 12 bytes) and Data2 (FLOAT[N]) live as
        // externals. Pack them in tag-id order — Dims (50937), then
        // Data2 (50939). Both are 4-byte aligned.
        let dims_off = (externals_start + ext.len()) as u32;
        for &d in &super::hue_sat_map::HUE_SAT_MAP_DIMS {
            ext.extend_from_slice(&long(d));
        }
        let data_off = (externals_start + ext.len()) as u32;
        for &v in hsm {
            ext.extend_from_slice(&long(v.to_bits()));
        }
        (dims_off, data_off, hsm.len() as u32)
    });
    let look_table_meta = look_table.map(|table| {
        let dims_off = (externals_start + ext.len()) as u32;
        for dimension in classic::LOOK_DIMS {
            ext.extend_from_slice(&long(dimension as u32));
        }
        let data_off = (externals_start + ext.len()) as u32;
        for value in table {
            ext.extend_from_slice(&long(value.to_bits()));
        }
        (dims_off, data_off, table.len() as u32)
    });
    let exposure_meta = look.map(|_| {
        let exposure_off = (externals_start + ext.len()) as u32;
        let (n, d) = srational_pair(f64::from(exposure_offset), 10_000);
        ext.extend_from_slice(&long(n as u32));
        ext.extend_from_slice(&long(d as u32));
        exposure_off
    });
    let name_inline = name_bytes.len() <= 4;
    let name_off = if name_inline {
        0
    } else {
        pad2(&mut ext);
        let off = (externals_start + ext.len()) as u32;
        ext.extend_from_slice(name_bytes);
        off
    };

    // Build the IFD body.
    let compression_inline = {
        // SHORT inline: value goes in the first 2 bytes of the 4-byte slot
        // (BE), rest zero — TIFF inline-value convention.
        let mut b = [0u8; 4];
        b[..2].copy_from_slice(&short(1));
        b
    };
    let mut entries: Vec<(u16, u16, u32, [u8; 4])> = Vec::with_capacity(entry_count as usize);
    entries.push((
        259, // TIFF Compression
        super::tiff_writer::TIFF_TYPE_SHORT,
        1,
        compression_inline,
    ));
    entries.push((
        t::COLOR_MATRIX1,
        super::tiff_writer::TIFF_TYPE_SRATIONAL,
        9,
        long(color_off),
    ));
    if look.is_some() {
        let mut illuminant = [0u8; 4];
        illuminant[..2].copy_from_slice(&short(t::CALIB_ILLUMINANT_D65));
        entries.push((
            t::CALIBRATION_ILLUMINANT1,
            super::tiff_writer::TIFF_TYPE_SHORT,
            1,
            illuminant,
        ));
    }
    entries.push((
        t::PROFILE_NAME,
        super::tiff_writer::TIFF_TYPE_ASCII,
        name_bytes.len() as u32,
        if name_inline {
            let mut b = [0u8; 4];
            b[..name_bytes.len()].copy_from_slice(name_bytes);
            b
        } else {
            long(name_off)
        },
    ));
    if let Some((dims_off, data_off, data_count)) = hue_sat_map_meta {
        entries.push((
            t::PROFILE_HUE_SAT_MAP_DIMS,
            super::tiff_writer::TIFF_TYPE_LONG,
            3,
            long(dims_off),
        ));
        entries.push((
            t::PROFILE_HUE_SAT_MAP_DATA1,
            super::tiff_writer::TIFF_TYPE_FLOAT,
            data_count,
            long(data_off),
        ));
    }
    if let Some((tone_off, tone_count)) = tone_curve_meta {
        entries.push((
            t::PROFILE_TONE_CURVE,
            super::tiff_writer::TIFF_TYPE_FLOAT,
            tone_count,
            long(tone_off),
        ));
    }
    entries.push((
        t::FORWARD_MATRIX1,
        super::tiff_writer::TIFF_TYPE_SRATIONAL,
        9,
        long(forward_off),
    ));
    if let Some((dims_off, data_off, data_count)) = look_table_meta {
        entries.push((
            t::PROFILE_LOOK_TABLE_DIMS,
            super::tiff_writer::TIFF_TYPE_LONG,
            3,
            long(dims_off),
        ));
        entries.push((
            t::PROFILE_LOOK_TABLE_DATA,
            super::tiff_writer::TIFF_TYPE_FLOAT,
            data_count,
            long(data_off),
        ));
    }
    if hue_sat_map_meta.is_some() {
        // LONG inline: 4-byte value goes directly in the slot.
        entries.push((
            t::PROFILE_HUE_SAT_MAP_ENCODING,
            super::tiff_writer::TIFF_TYPE_LONG,
            1,
            long(super::hue_sat_map::HUE_SAT_MAP_ENCODING),
        ));
    }
    if look_table_meta.is_some() {
        entries.push((
            t::PROFILE_LOOK_TABLE_ENCODING,
            super::tiff_writer::TIFF_TYPE_LONG,
            1,
            long(1),
        ));
    }
    if let Some(exposure_off) = exposure_meta {
        entries.push((
            t::BASELINE_EXPOSURE_OFFSET,
            super::tiff_writer::TIFF_TYPE_SRATIONAL,
            1,
            long(exposure_off),
        ));
    }
    entries.push((
        t::DEFAULT_BLACK_RENDER,
        super::tiff_writer::TIFF_TYPE_LONG,
        1,
        long(1),
    ));
    debug_assert_eq!(entries.len(), entry_count as usize);

    let mut buf: Vec<u8> = Vec::with_capacity(externals_start + ext.len());
    buf.extend_from_slice(if little { b"IIRC" } else { b"MMCR" });
    buf.extend_from_slice(&long(header_bytes as u32)); // IFD0 offset = 8
    debug_assert_eq!(buf.len(), header_bytes);

    buf.extend_from_slice(&short(entry_count));
    for (tag, ty, count, value4) in entries {
        buf.extend_from_slice(&short(tag));
        buf.extend_from_slice(&short(ty));
        buf.extend_from_slice(&long(count));
        buf.extend_from_slice(&value4);
    }
    buf.extend_from_slice(&long(0)); // next IFD = none
    debug_assert_eq!(buf.len(), externals_start);

    buf.extend_from_slice(&ext);

    buf
}

fn srational_pair(v: f64, denom: i32) -> (i32, i32) {
    // Saturating cast — matrix values are bounded by [-2, 2] in practice;
    // anything outside i32 here would mean the source matrix is broken.
    let num = (v * denom as f64).round();
    let num = num.clamp(i32::MIN as f64, i32::MAX as f64) as i32;
    (num, denom)
}

pub(crate) fn srational_from_floats(values: &[f32], denom: i32) -> Value {
    Value::SRational(
        values
            .iter()
            .map(|&v| srational_pair(v as f64, denom))
            .collect(),
    )
}

#[cfg(test)]
#[path = "profile_tests.rs"]
mod profile_tests;

#[cfg(test)]
mod tests {
    use super::*;

    /// CIE D65 white point in XYZ (Y=1). Sigma's `bmt_to_xyz @ (1,1,1)`
    /// equals this by construction (`raw_to_xyz @ raw_neutral = D65`).
    const D65_XYZ: [f64; 3] = [0.95047, 1.00000, 1.08883];

    /// CIE D50 white point in XYZ (Y=1). DNG PCS reference white.
    const D50_XYZ_REF: [f64; 3] = [0.96422, 1.00000, 0.82521];

    /// Verifies the spec invariant `FM @ (1,1,1) = D50_XYZ` for the writer's
    /// FM derivation, on the assumption that `bmt_to_xyz @ (1,1,1) = D65_XYZ`
    /// (which is how the C-side `x3f_get_bmt_to_xyz` is constructed —
    /// `raw_to_xyz @ raw_neutral = D65` and `bmt_to_xyz = raw_to_xyz @
    /// diag(raw_neutral)`, so `bmt_to_xyz @ (1,1,1) = raw_to_xyz @
    /// raw_neutral = D65`).
    ///
    /// Failing this test means non-Adobe DNG readers (Apple RAW Engine,
    /// Capture One) — which consume FM verbatim without auto-normalizing —
    /// will render with chroma casts.
    #[test]
    fn forward_matrix_satisfies_d50_invariant_at_unit_input() {
        // Construct a synthetic bmt_to_xyz where each column = D65/3 so
        // that bmt @ (1,1,1) = D65 exactly.
        let bmt: [f64; 9] = [
            D65_XYZ[0] / 3.0,
            D65_XYZ[0] / 3.0,
            D65_XYZ[0] / 3.0,
            D65_XYZ[1] / 3.0,
            D65_XYZ[1] / 3.0,
            D65_XYZ[1] / 3.0,
            D65_XYZ[2] / 3.0,
            D65_XYZ[2] / 3.0,
            D65_XYZ[2] / 3.0,
        ];

        let fm = mat3_mul(&bradford_d65_to_d50(), &bmt);
        let fm_at_unit = [
            fm[0] + fm[1] + fm[2],
            fm[3] + fm[4] + fm[5],
            fm[6] + fm[7] + fm[8],
        ];

        for i in 0..3 {
            assert!(
                (fm_at_unit[i] - D50_XYZ_REF[i]).abs() < 1e-3,
                "FM @ (1,1,1)[{i}] = {} but expected D50_XYZ[{i}] = {}",
                fm_at_unit[i],
                D50_XYZ_REF[i],
            );
        }
    }

    #[test]
    fn srational_round_trips_through_denom() {
        let (n, d) = srational_pair(1.5, 10_000);
        assert_eq!(d, 10_000);
        assert_eq!(n, 15_000);
    }

    #[test]
    fn negative_values_keep_sign() {
        let (n, _d) = srational_pair(-0.6789, 10_000);
        assert_eq!(n, -6789);
    }

    #[test]
    fn mmcr_blob_starts_with_magic() {
        let color = [1.0_f32; 9];
        let fwd = [0.5_f32; 9];
        let blob = build_mmcr_profile("Test Profile", color, fwd, None, None);
        assert_eq!(&blob[..4], b"MMCR");
    }

    #[test]
    fn mmcr_blob_ifd0_offset_points_inside_blob() {
        let blob = build_mmcr_profile("Test", [0.0_f32; 9], [0.0_f32; 9], None, None);
        let off = u32::from_be_bytes([blob[4], blob[5], blob[6], blob[7]]) as usize;
        assert!(
            off < blob.len(),
            "IFD0 offset {off} >= blob len {}",
            blob.len()
        );
        let entry_count = u16::from_be_bytes([blob[off], blob[off + 1]]);
        assert_eq!(entry_count, 5);
    }

    #[test]
    fn mmcr_entries_are_in_tag_order() {
        let blob = build_mmcr_profile("X", [0.0_f32; 9], [0.0_f32; 9], None, None);
        let off = u32::from_be_bytes([blob[4], blob[5], blob[6], blob[7]]) as usize;
        let read_tag = |i: usize| -> u16 {
            u16::from_be_bytes([blob[off + 2 + i * 12], blob[off + 2 + i * 12 + 1]])
        };
        let tags: Vec<u16> = (0..5).map(read_tag).collect();
        let mut sorted = tags.clone();
        sorted.sort();
        assert_eq!(tags, sorted);
    }

    #[test]
    fn mmcr_ifd_body_at_offset_8() {
        // Lightroom's DNG SDK rejects MMCR blobs whose IFD body isn't
        // immediately after the 8-byte header. Match the libtiff layout
        // exactly: header(8) + IFD body + externals.
        let blob = build_mmcr_profile("Vivid", [0.0_f32; 9], [0.0_f32; 9], None, None);
        let off = u32::from_be_bytes([blob[4], blob[5], blob[6], blob[7]]) as usize;
        assert_eq!(off, 8, "IFD body must start at offset 8");
    }

    #[test]
    fn mmcr_externals_after_ifd_body() {
        // ColorMatrix1 is at index 1 (after Compression). Its value-or-
        // offset slot must be > the IFD body's tail offset.
        let blob = build_mmcr_profile("Vivid", [0.0_f32; 9], [0.0_f32; 9], None, None);
        let ifd_off = u32::from_be_bytes([blob[4], blob[5], blob[6], blob[7]]) as usize;
        let ifd_end = ifd_off + 2 + 5 * 12 + 4;
        let cm_p = ifd_off + 2 + 12; // entry 1 = ColorMatrix1
        let cm_value = u32::from_be_bytes([
            blob[cm_p + 8],
            blob[cm_p + 9],
            blob[cm_p + 10],
            blob[cm_p + 11],
        ]) as usize;
        assert!(
            cm_value >= ifd_end,
            "ColorMatrix1 external value at {cm_value} should be after IFD body end {ifd_end}",
        );
    }

    #[test]
    fn mmcr_with_hue_sat_map_has_three_extra_tags_in_order() {
        // Saturation is the fastest varying dimension: the neutral and
        // saturated correction records for each hue are adjacent.
        let mut hsm: Vec<f32> = Vec::with_capacity(21 * 2 * 3);
        for h in 0..21 {
            for _ in 0..2 {
                hsm.push(h as f32 * 0.5); // hue shift
                hsm.push(1.0 + h as f32 * 0.01); // sat scale
                hsm.push(1.0); // value scale
            }
        }
        let blob = build_mmcr_profile(
            "Vivid",
            [0.0_f32; 9],
            [0.0_f32; 9],
            None,
            Some(hsm.as_slice()),
        );
        let off = u32::from_be_bytes([blob[4], blob[5], blob[6], blob[7]]) as usize;
        let entry_count = u16::from_be_bytes([blob[off], blob[off + 1]]);
        // Base 5 + Dims/Data1/Encoding = 8.
        assert_eq!(entry_count, 8);
        let read_tag = |i: usize| -> u16 {
            u16::from_be_bytes([blob[off + 2 + i * 12], blob[off + 2 + i * 12 + 1]])
        };
        let tags: Vec<u16> = (0..entry_count as usize).map(read_tag).collect();
        // Tag-id-ascending invariant must hold.
        let mut sorted = tags.clone();
        sorted.sort();
        assert_eq!(tags, sorted);
        // The three new tags must be present.
        for &expect in &[
            t::PROFILE_HUE_SAT_MAP_DIMS,
            t::PROFILE_HUE_SAT_MAP_DATA1,
            t::PROFILE_HUE_SAT_MAP_ENCODING,
        ] {
            assert!(tags.contains(&expect), "missing tag {expect}");
        }
        let index = tags
            .iter()
            .position(|&tag| tag == t::PROFILE_HUE_SAT_MAP_DATA1)
            .unwrap();
        let entry = off + 2 + index * 12;
        let count = u32::from_be_bytes(blob[entry + 4..entry + 8].try_into().unwrap());
        let data_offset =
            u32::from_be_bytes(blob[entry + 8..entry + 12].try_into().unwrap()) as usize;
        assert_eq!(count, 126);
        let decoded: Vec<f32> = blob[data_offset..data_offset + count as usize * 4]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|bytes| f32::from_be_bytes(*bytes))
            .collect();
        assert_eq!(decoded, hsm, "MMCR must preserve saturation-fastest order");
    }

    #[test]
    fn mmcr_hue_sat_map_encoding_is_inline_linear() {
        let hsm: Vec<f32> = [0.0, 1.0, 1.0].repeat(21 * 2);
        let blob = build_mmcr_profile("X", [0.0_f32; 9], [0.0_f32; 9], None, Some(hsm.as_slice()));
        let off = u32::from_be_bytes([blob[4], blob[5], blob[6], blob[7]]) as usize;
        let entry_count = u16::from_be_bytes([blob[off], blob[off + 1]]);
        let mut found = false;
        for i in 0..entry_count as usize {
            let p = off + 2 + i * 12;
            let tag = u16::from_be_bytes([blob[p], blob[p + 1]]);
            if tag == t::PROFILE_HUE_SAT_MAP_ENCODING {
                assert_eq!(u16::from_be_bytes([blob[p + 2], blob[p + 3]]), 4);
                assert_eq!(
                    u32::from_be_bytes(blob[p + 4..p + 8].try_into().unwrap()),
                    1
                );
                let val =
                    u32::from_be_bytes([blob[p + 8], blob[p + 9], blob[p + 10], blob[p + 11]]);
                // V=1 must not acquire a gamma transform: RawTherapee
                // 5.12 darkens even this identity table with encoding 1.
                assert_eq!(val, 0, "V=1 maps require explicit linear encoding");
                found = true;
            }
        }
        assert!(found, "Encoding tag missing");
    }

    #[test]
    fn mmcr_hue_sat_map_dims_payload_is_21_2_1() {
        let hsm: Vec<f32> = [0.0, 1.0, 1.0].repeat(21 * 2);
        let blob = build_mmcr_profile("X", [0.0_f32; 9], [0.0_f32; 9], None, Some(hsm.as_slice()));
        let off = u32::from_be_bytes([blob[4], blob[5], blob[6], blob[7]]) as usize;
        let entry_count = u16::from_be_bytes([blob[off], blob[off + 1]]);
        let mut dims_off = None;
        for i in 0..entry_count as usize {
            let p = off + 2 + i * 12;
            let tag = u16::from_be_bytes([blob[p], blob[p + 1]]);
            if tag == t::PROFILE_HUE_SAT_MAP_DIMS {
                dims_off =
                    Some(
                        u32::from_be_bytes([blob[p + 8], blob[p + 9], blob[p + 10], blob[p + 11]])
                            as usize,
                    );
            }
        }
        let dims_off = dims_off.expect("Dims tag missing");
        let h = u32::from_be_bytes([
            blob[dims_off],
            blob[dims_off + 1],
            blob[dims_off + 2],
            blob[dims_off + 3],
        ]);
        let s = u32::from_be_bytes([
            blob[dims_off + 4],
            blob[dims_off + 5],
            blob[dims_off + 6],
            blob[dims_off + 7],
        ]);
        let v = u32::from_be_bytes([
            blob[dims_off + 8],
            blob[dims_off + 9],
            blob[dims_off + 10],
            blob[dims_off + 11],
        ]);
        assert_eq!([h, s, v], [21, 2, 1]);
    }

    #[test]
    fn mmcr_has_compression_tag() {
        // libtiff writes baseline TIFF tags (Compression) into every
        // directory it produces. Lightroom's DNG SDK appears to require
        // their presence to recognise MMCR profiles. Specifically, the
        // first entry must be tag 259 (Compression), type SHORT, value 1.
        let blob = build_mmcr_profile("X", [0.0_f32; 9], [0.0_f32; 9], None, None);
        let off = u32::from_be_bytes([blob[4], blob[5], blob[6], blob[7]]) as usize;
        let mut found = false;
        for i in 0..5 {
            let p = off + 2 + i * 12;
            let tag = u16::from_be_bytes([blob[p], blob[p + 1]]);
            if tag == 259 {
                let ty = u16::from_be_bytes([blob[p + 2], blob[p + 3]]);
                let count =
                    u32::from_be_bytes([blob[p + 4], blob[p + 5], blob[p + 6], blob[p + 7]]);
                let val = u16::from_be_bytes([blob[p + 8], blob[p + 9]]);
                assert_eq!(ty, super::super::tiff_writer::TIFF_TYPE_SHORT);
                assert_eq!(count, 1);
                assert_eq!(val, 1, "Compression must be Uncompressed (=1)");
                found = true;
                break;
            }
        }
        assert!(found, "Compression tag missing from MMCR profile");
    }
}
