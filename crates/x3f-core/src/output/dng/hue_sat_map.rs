//! `ProfileHueSatMapData1` synthesis from Sigma's `MultiAxisTable_<mode>`.
//!
//! Each in-camera color mode has a `MultiAxisTable_<mode>` CAMF entry — a
//! `float[2][5][21]` table that Sigma's JPEG renderer applies as a hue
//! and saturation correction in HSV space. The axes are:
//!
//! - **D2 (x)**: 21 hue bins covering `[0, 360)` at `360/21 ≈ 17.14°`
//!   intervals. Bin `h` is centered at `h * 360/21` degrees.
//! - **D1 (y)**: 5 bins. Reserved by Sigma for a future
//!   value/saturation axis but every shipping camera writes identical
//!   rows — we collapse this to a 1-bin DNG axis.
//! - **D0 (group)**: 2 channels — group 0 holds the **hue shift in
//!   degrees**, group 1 holds the **saturation multiplier**.
//!
//! DNG `ProfileHueSatMapData1` (tag 50938) is a flat float array of
//! `H × S × V × 3` triplets in `(value, saturation, hue)`-major order
//! (hue varies fastest). Each triplet is
//! `(hue_shift_deg, sat_scale, value_scale)`. We emit
//! `dims = [21, 1, 1]` so the DNG table reduces to 21 triplets. The DNG
//! reader interpolates in HSV space.
//!
//! Why *Data1* and not *Data2* (tag 50939): Adobe's DNG SDK pairs Data1
//! with `CalibrationIlluminant1` / `ColorMatrix1` and Data2 with the
//! second illuminant. When a profile has only one illuminant — which
//! is the case for every Sigma camera profile we emit — `Data2` is
//! silently ignored by the SDK because there's no second illuminant
//! to anchor it. An earlier version of this writer emitted `Data2` and
//! Lightroom showed the profiles with no chroma differences as a
//! result.
//!
//! Encoding (tag 51107) is set to **1 = sRGB**, matching how Sigma applies
//! the table — after gamma, in display HSV — rather than 0 = linear.
//! (We tried omitting this tag — DNG 1.6 says it's N/A when V=1 — but Apple
//! RAW Engine then fails to decode the IFD0 preview strip and produces
//! black thumbnails. Apple's preview pipeline appears to walk the camera
//! profile, and the missing encoding tag short-circuits its renderer.)

use crate::Reader;

/// 21-bin hue + 1-bin sat + 1-bin value DNG hue/sat map. Output is a flat
/// `Vec<f32>` of 21 × 3 = 63 floats, each triplet
/// `(hue_shift_deg, sat_scale, value_scale)`.
///
/// Returns `None` if:
/// - the named CAMF entry is missing or the wrong shape, or
/// - every entry is identity (hue shift = 0, sat = 1) — in which case the
///   tag is omitted, matching how `tone_curves::build_curve` handles
///   identity tone curves.
pub(crate) fn build_hue_sat_map(reader: &Reader, name: &str) -> Option<Vec<f32>> {
    let raw = reader.dng_camf_multi_axis_table(name)?;

    // Layout: raw[group * 5 * 21 + y * 21 + x]. Group 0 = hue shift,
    // group 1 = sat scale. Use y = 0 (rows are uniform across y in
    // practice, see module doc).
    const N_HUE: usize = 21;
    let hue_row = &raw[0..N_HUE];
    let sat_row = &raw[5 * N_HUE..5 * N_HUE + N_HUE];

    // Identity check: hue shifts all zero and sat scales all one.
    let is_identity =
        hue_row.iter().all(|&v| v.abs() < 1e-6) && sat_row.iter().all(|&v| (v - 1.0).abs() < 1e-6);
    if is_identity {
        return None;
    }

    let mut out: Vec<f32> = Vec::with_capacity(N_HUE * 3);
    for h in 0..N_HUE {
        out.push(hue_row[h] as f32);
        out.push(sat_row[h] as f32);
        out.push(1.0);
    }
    Some(out)
}

/// `ProfileHueSatMapDims` (tag 50937) value for `build_hue_sat_map`'s
/// output: `[hue_divisions, sat_divisions, value_divisions]`. Every map
/// we emit uses the same dims, so this is a constant.
pub(crate) const HUE_SAT_MAP_DIMS: [u32; 3] = [21, 1, 1];

/// `ProfileHueSatMapEncoding` (tag 51107). 1 = sRGB (operates after the
/// sRGB gamma); 0 = linear. Sigma applies these in display HSV, so sRGB.
pub(crate) const HUE_SAT_MAP_ENCODING: u32 = 1;

#[cfg(test)]
mod tests {
    // Numerical / shape correctness is exercised end-to-end via the MMCR
    // builder tests in `profiles.rs`. This module's logic is too thin
    // (slice indexing + identity check) to need its own unit tests.
}
