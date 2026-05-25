//! Camera-profile generation and the `ExtraCameraProfiles` (MMCR) blob.
//!
//! Mirrors the structure of `src/x3f_output_dng.c::write_camera_profile`
//! and `write_camera_profiles`. The legacy code wrote the default profile's
//! tags directly into the parent DNG IFD0, then for each additional
//! profile spun up a tiny libtiff "TIFFFdOpen(..., "wb")" big-endian file
//! to a `tmpfile`, stripped its 4-byte TIFF magic, prepended `"MMCR"`, and
//! concatenated all such blobs into the parent file with one offset per
//! profile in the `ExtraCameraProfiles` tag.
//!
//! We replicate that byte-for-byte: the default profile is written via the
//! parent's `DirectoryWriter`, and the `MmcrBuilder` here produces a single
//! big-endian IFD with `"MMCR"` magic in place of `"MM\x00\x2A"`.

use std::ffi::CString;

use crate::Reader;

use super::metadata::{
    adobe_rgb_to_xyz, bradford_d65_to_d50, mat3_diag, mat3_inverse, mat3_mul, mat3_ones,
    mat3_to_f32,
};
use super::tags as t;
use super::tiff_writer::{DirectoryWriter, Value};

/// Source of the camera-native → CIE XYZ matrix for a profile.
#[derive(Clone, Copy)]
enum BmtSource {
    /// Body's standard calibration matrix (`x3f_get_bmt_to_xyz(wb)` chain).
    /// Picture-mode profiles fold their `CMCM_<mode>` 3×3 CCM into this
    /// matrix to differentiate chroma per mode — Standard's CCM is
    /// identity, while Vivid / FCBlue / etc. carry per-mode chroma
    /// shifts. This matches the legacy `get_bmt_to_xyz_fcblue` path
    /// in `src/x3f_output_dng.c:149` and is the *primary* chromatic
    /// differentiator between picture profiles. (`ProfileToneCurve` is
    /// the secondary tonal differentiator; `ProfileHueSatMapData1` is
    /// the tertiary fine-tune.)
    ///
    /// An earlier iteration of this writer skipped the CCM
    /// pre-multiplication on the theory that it caused Lightroom to
    /// filter the profiles out of the Profile dropdown. That diagnosis
    /// was wrong — the actual cause was the MMCR layout being
    /// non-standard. With the layout fixed (see `build_mmcr_profile`),
    /// CCM-pre-multiplied matrices validate fine.
    Default,
    /// "Don't convert" — assume the working space is Adobe RGB. Used by
    /// the grayscale profiles and the `Unconverted` debug profile.
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
    /// 21×1×1 DNG hue/sat map.
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
    // Disabled: these CMCM-derived picture profiles don't render as
    // expected in Lightroom. Kept here (commented) rather than deleted so
    // they can be re-enabled once the CMCM folding is sorted out.
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

    let color_matrix1 = mat3_to_f32(&mat3_inverse(&bmt_to_xyz));

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
pub(crate) fn write_default_profile(
    reader: &Reader,
    wb: &str,
    ifd0: &mut DirectoryWriter,
) -> Option<usize> {
    let idx = default_profile_index(reader);
    let profile = &PROFILES[idx];
    let (color, forward) = compute_matrices(reader, wb, profile)?;
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
pub(crate) fn build_extra_profiles_blob(
    reader: &Reader,
    wb: &str,
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
        let Some((color, forward)) = compute_matrices(reader, wb, profile) else {
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

/// Build one `MMCR`-prefixed big-endian TIFF holding a single profile's
/// tags. Done by hand (no `TiffWriter`) because the existing C path
/// produces big-endian profiles regardless of host byte order, and the
/// surface is only four tags.
fn build_mmcr_profile(
    name: &str,
    color: [f32; 9],
    forward: [f32; 9],
    tone_curve: Option<&[f32]>,
    hue_sat_map: Option<&[f32]>,
) -> Vec<u8> {
    let name_c = CString::new(name).expect("profile name without NUL");
    let name_bytes = name_c.as_bytes_with_nul();

    // CPP reference layout (which Lightroom successfully parses):
    //   offset 0  : "MMCR" magic
    //   offset 4  : 4-byte BE IFD0 offset = 8
    //   offset 8  : IFD body — count(2) + N×12 entries + next-IFD(4)
    //   offset 8+: external values for tags whose size > 4
    //
    // Earlier versions of this writer placed externals BEFORE the IFD body
    // (with a non-8 IFD0 offset). Lightroom's DNG SDK rejected those MMCR
    // blobs entirely — the profiles never showed up in the Profile dropdown
    // or even in the "Browse all profiles" view. Match the libtiff layout
    // exactly: IFD body immediately after the 8-byte header, externals
    // appended at the tail.
    //
    // Tag set (in tag-ID order):
    //   259    Compression                (SHORT inline, value 1 = uncompressed)
    //   50721  ColorMatrix1               (SRATIONAL[9], external)
    //   50936  ProfileName                (ASCII, external if > 4 bytes)
    //   50937  ProfileHueSatMapDims       (LONG[3], external) — optional
    //   50938  ProfileHueSatMapData1      (FLOAT[H*S*V*3], external) — optional
    //   50940  ProfileToneCurve           (FLOAT[N], external) — optional, only
    //                                      when the profile's CMCC_<mode> ≠ 0
    //   50964  ForwardMatrix1             (SRATIONAL[9], external)
    //   51107  ProfileHueSatMapEncoding   (LONG inline, value 1 = sRGB) — optional
    //   51110  DefaultBlackRender         (LONG inline, value 1)
    //
    // CalibrationIlluminant1 is intentionally omitted: CPP's MMCR blobs
    // don't carry it (Adobe inherits the parent file's IFD0 illuminant)
    // and including it had no observable effect, so we drop it to stay
    // byte-for-byte compatible with the working reference.
    //
    // The hue/sat map is *not* present in CPP MMCR blobs — Sigma's
    // reference build leaves the per-mode hue/sat correction to its own
    // post-render pass and only ships ColorMatrix + ForwardMatrix +
    // ToneCurve. We add it so DNG-aware processors (Lightroom, Capture
    // One, RawTherapee) that don't run Sigma's pipeline can apply the
    // same hue/sat correction the camera would have.

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
        ext.extend_from_slice(&n.to_be_bytes());
        ext.extend_from_slice(&d.to_be_bytes());
    }
    pad2(&mut ext);
    let forward_off = (externals_start + ext.len()) as u32;
    for v in forward {
        let (n, d) = srational_pair(v as f64, 10_000);
        ext.extend_from_slice(&n.to_be_bytes());
        ext.extend_from_slice(&d.to_be_bytes());
    }
    let tone_curve_meta = tone_curve.map(|curve| {
        // FLOAT is 4 bytes / 4-byte aligned; ext is currently 8-aligned
        // (matrix payloads are 8 byte multiples) so no extra pad needed.
        let off = (externals_start + ext.len()) as u32;
        for &v in curve {
            ext.extend_from_slice(&v.to_be_bytes());
        }
        (off, curve.len() as u32)
    });
    let hue_sat_map_meta = hue_sat_map.map(|hsm| {
        // Both Dims (LONG[3], 12 bytes) and Data2 (FLOAT[N]) live as
        // externals. Pack them in tag-id order — Dims (50937), then
        // Data2 (50939). Both are 4-byte aligned.
        let dims_off = (externals_start + ext.len()) as u32;
        for &d in &super::hue_sat_map::HUE_SAT_MAP_DIMS {
            ext.extend_from_slice(&d.to_be_bytes());
        }
        let data_off = (externals_start + ext.len()) as u32;
        for &v in hsm {
            ext.extend_from_slice(&v.to_be_bytes());
        }
        (dims_off, data_off, hsm.len() as u32)
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
        b[..2].copy_from_slice(&1u16.to_be_bytes());
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
        color_off.to_be_bytes(),
    ));
    entries.push((
        t::PROFILE_NAME,
        super::tiff_writer::TIFF_TYPE_ASCII,
        name_bytes.len() as u32,
        if name_inline {
            let mut b = [0u8; 4];
            b[..name_bytes.len()].copy_from_slice(name_bytes);
            b
        } else {
            name_off.to_be_bytes()
        },
    ));
    if let Some((dims_off, data_off, data_count)) = hue_sat_map_meta {
        entries.push((
            t::PROFILE_HUE_SAT_MAP_DIMS,
            super::tiff_writer::TIFF_TYPE_LONG,
            3,
            dims_off.to_be_bytes(),
        ));
        entries.push((
            t::PROFILE_HUE_SAT_MAP_DATA1,
            super::tiff_writer::TIFF_TYPE_FLOAT,
            data_count,
            data_off.to_be_bytes(),
        ));
    }
    if let Some((tone_off, tone_count)) = tone_curve_meta {
        entries.push((
            t::PROFILE_TONE_CURVE,
            super::tiff_writer::TIFF_TYPE_FLOAT,
            tone_count,
            tone_off.to_be_bytes(),
        ));
    }
    entries.push((
        t::FORWARD_MATRIX1,
        super::tiff_writer::TIFF_TYPE_SRATIONAL,
        9,
        forward_off.to_be_bytes(),
    ));
    if hue_sat_map_meta.is_some() {
        // LONG inline: 4-byte value goes directly in the slot.
        entries.push((
            t::PROFILE_HUE_SAT_MAP_ENCODING,
            super::tiff_writer::TIFF_TYPE_LONG,
            1,
            super::hue_sat_map::HUE_SAT_MAP_ENCODING.to_be_bytes(),
        ));
    }
    entries.push((
        t::DEFAULT_BLACK_RENDER,
        super::tiff_writer::TIFF_TYPE_LONG,
        1,
        1u32.to_be_bytes(),
    ));
    debug_assert_eq!(entries.len(), entry_count as usize);

    let mut buf: Vec<u8> = Vec::with_capacity(externals_start + ext.len());
    buf.extend_from_slice(b"MMCR");
    buf.extend_from_slice(&(header_bytes as u32).to_be_bytes()); // IFD0 offset = 8
    debug_assert_eq!(buf.len(), header_bytes);

    buf.extend_from_slice(&entry_count.to_be_bytes());
    for (tag, ty, count, value4) in entries {
        buf.extend_from_slice(&tag.to_be_bytes());
        buf.extend_from_slice(&ty.to_be_bytes());
        buf.extend_from_slice(&count.to_be_bytes());
        buf.extend_from_slice(&value4);
    }
    buf.extend_from_slice(&0u32.to_be_bytes()); // next IFD = none
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
        let cm_p = ifd_off + 2 + 1 * 12; // entry 1 = ColorMatrix1
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
        // Synth a 21×1×1 hue/sat map.
        let mut hsm: Vec<f32> = Vec::with_capacity(21 * 3);
        for h in 0..21 {
            hsm.push(h as f32 * 0.5); // hue shift
            hsm.push(1.0 + h as f32 * 0.01); // sat scale
            hsm.push(1.0); // value scale
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
    }

    #[test]
    fn mmcr_hue_sat_map_encoding_is_inline_srgb() {
        let hsm = vec![0.0_f32; 21 * 3];
        let blob = build_mmcr_profile("X", [0.0_f32; 9], [0.0_f32; 9], None, Some(hsm.as_slice()));
        let off = u32::from_be_bytes([blob[4], blob[5], blob[6], blob[7]]) as usize;
        let entry_count = u16::from_be_bytes([blob[off], blob[off + 1]]);
        let mut found = false;
        for i in 0..entry_count as usize {
            let p = off + 2 + i * 12;
            let tag = u16::from_be_bytes([blob[p], blob[p + 1]]);
            if tag == t::PROFILE_HUE_SAT_MAP_ENCODING {
                let val =
                    u32::from_be_bytes([blob[p + 8], blob[p + 9], blob[p + 10], blob[p + 11]]);
                assert_eq!(val, 1, "Encoding must be sRGB (=1)");
                found = true;
            }
        }
        assert!(found, "Encoding tag missing");
    }

    #[test]
    fn mmcr_hue_sat_map_dims_payload_is_21_1_1() {
        let hsm = vec![0.0_f32; 21 * 3];
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
        assert_eq!([h, s, v], [21, 1, 1]);
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
