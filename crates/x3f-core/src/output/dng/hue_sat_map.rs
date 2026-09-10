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
//!   rows — we use the first row for each hue correction.
//! - **D0 (group)**: 2 channels — group 0 holds the **hue shift in
//!   degrees**, group 1 holds the **saturation multiplier**.
//!
//! DNG `ProfileHueSatMapData1` (tag 50938) is a flat float array of
//! `H × S × V` triplets, with value in the outer loop, hue in the
//! middle loop, and saturation varying fastest. Each triplet is
//! `(hue_shift_deg, sat_scale, value_scale)`. DNG requires at least two
//! saturation samples, so we emit `dims = [21, 2, 1]` and duplicate
//! each hue correction at both saturation endpoints. This keeps the
//! correction independent of saturation when the reader interpolates.
//! Every value scale remains 1, as required at zero saturation.
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
//! DNG hue/saturation maps operate on HSV derived from linear ProPhoto
//! RGB. Encoding (tag 51107) affects only the value coordinate of 3D
//! maps; the DNG specification says it is inapplicable when V=1. We
//! explicitly write **0 = linear** for compatibility: RawTherapee 5.12
//! and Adobe DNG SDK 1.7.1 apply an inverse sRGB curve to V for encoding
//! 1 even in their V=1 paths, where they skipped the forward curve.
//! This darkens the image even with an identity map. Encoding 0 avoids
//! that extra transform.

use crate::Reader;

/// 21-bin hue + 2-bin sat + 1-bin value DNG hue/sat map. Output is a flat
/// `Vec<f32>` of 21 × 2 × 3 = 126 floats, each triplet
/// `(hue_shift_deg, sat_scale, value_scale)`.
///
/// Returns `None` if:
/// - the named CAMF entry is missing or the wrong shape, or
/// - every entry is identity (hue shift = 0, sat = 1) — in which case the
///   tag is omitted, matching how `tone_curves::build_curve` handles
///   identity tone curves.
pub(crate) fn build_hue_sat_map(reader: &Reader, name: &str) -> Option<Vec<f32>> {
    let raw = reader.dng_camf_multi_axis_table(name)?;
    synthesize_hue_sat_map(&raw)
}

fn synthesize_hue_sat_map(raw: &[f64; 210]) -> Option<Vec<f32>> {
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

    let mut out: Vec<f32> = Vec::with_capacity(N_HUE * 2 * 3);
    for h in 0..N_HUE {
        let correction = [hue_row[h] as f32, sat_row[h] as f32, 1.0];
        // Saturation varies fastest; identical endpoints retain the
        // same correction at every input saturation.
        out.extend_from_slice(&correction);
        out.extend_from_slice(&correction);
    }
    Some(out)
}

/// `ProfileHueSatMapDims` (tag 50937) value for `build_hue_sat_map`'s
/// output: `[hue_divisions, sat_divisions, value_divisions]`. Every map
/// we emit uses the same dims, so this is a constant.
pub(crate) const HUE_SAT_MAP_DIMS: [u32; 3] = [21, 2, 1];

/// `ProfileHueSatMapEncoding` (tag 51107). Encoding is inapplicable to
/// our V=1 map; explicit linear encoding avoids the unpaired inverse
/// sRGB transform in RawTherapee 5.12 and Adobe DNG SDK 1.7.1.
pub(crate) const HUE_SAT_MAP_ENCODING: u32 = 0;

#[cfg(test)]
mod tests {
    use super::*;

    fn identity_table() -> [f64; 210] {
        let mut raw = [0.0; 210];
        raw[5 * 21..].fill(1.0);
        raw
    }

    #[test]
    fn identity_map_is_omitted() {
        assert!(synthesize_hue_sat_map(&identity_table()).is_none());
    }

    #[test]
    fn hue_and_saturation_corrections_are_retained() {
        let mut hue_only = identity_table();
        hue_only[0] = 1.0;
        assert!(synthesize_hue_sat_map(&hue_only).is_some());

        let mut saturation_only = identity_table();
        saturation_only[5 * 21] = 0.5;
        assert!(synthesize_hue_sat_map(&saturation_only).is_some());
    }

    #[test]
    fn map_has_two_saturation_samples_per_hue_in_dng_order() {
        let mut raw = identity_table();
        for h in 0..21 {
            raw[h] = h as f64 - 10.0;
            raw[5 * 21 + h] = 0.5 + h as f64 / 32.0;
        }

        let map = synthesize_hue_sat_map(&raw).unwrap();
        assert_eq!(HUE_SAT_MAP_DIMS, [21, 2, 1]);
        assert_eq!(map.len(), 126);
        for (h, endpoints) in map.chunks_exact(6).enumerate() {
            let correction = [h as f32 - 10.0, 0.5 + h as f32 / 32.0, 1.0];
            assert_eq!(&endpoints[..3], &correction);
            assert_eq!(&endpoints[3..], &correction);
        }
    }
}
