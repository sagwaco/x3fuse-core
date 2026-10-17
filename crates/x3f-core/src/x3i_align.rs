//! Frame alignment for the X3I merge. See `_local/X3I_SFD_PLAN.md`, section 4.4.
//!
//! The bodies in this file are placeholders: a zero shift and an identity
//! warp. The phase 2 alignment task replaces them and keeps the signatures,
//! so the merge can call them before that work lands.

use crate::Error;
use x3f_sys::Control;

/// Alignment settings for one conversion.
#[derive(Clone, Debug, PartialEq)]
pub struct AlignOptions {
    /// Estimate and apply a transform for every frame except the reference.
    pub enabled: bool,
    /// Frames with a correlation peak below this value are skipped.
    pub min_peak: f32,
    /// Largest shift the estimator searches, in full-resolution pixels.
    pub max_shift_px: u32,
    /// Tiles per axis for the per-tile phase correlation.
    pub tile_grid: u32,
}

impl Default for AlignOptions {
    fn default() -> Self {
        Self {
            enabled: true,
            min_peak: 0.05,
            max_shift_px: 64,
            tile_grid: 3,
        }
    }
}

/// Gradient magnitude of exposure-normalised log luminance at half resolution,
/// with a mask that excludes clipped and near-black pixels.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Luma {
    /// Width of the half-resolution map, in pixels.
    pub width: u32,
    /// Height of the half-resolution map, in pixels.
    pub height: u32,
    /// Gradient magnitude per pixel, row-major.
    pub data: Vec<f32>,
    /// One per pixel; nonzero where the pixel takes part in the estimate.
    pub mask: Vec<u8>,
}

/// Transform of one frame relative to the reference, in full-resolution pixels.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Alignment {
    /// Horizontal shift that maps frame coordinates onto the reference.
    pub dx: f32,
    /// Vertical shift that maps frame coordinates onto the reference.
    pub dy: f32,
    /// Normalised correlation peak of the global estimate.
    pub peak: f32,
    /// Largest disagreement between tile shifts, in pixels.
    pub tile_spread: f32,
    /// Similarity transform `[a, b, tx, c, d, ty]` when the tiles disagree.
    pub similarity: Option<[f32; 6]>,
    /// True when the frame is merged without a transform because the
    /// estimate was not trusted.
    pub skipped: bool,
}

impl Alignment {
    /// True when the transform leaves the frame unchanged.
    pub fn is_identity(&self) -> bool {
        self.dx == 0.0 && self.dy == 0.0 && self.similarity.is_none()
    }
}

/// Per-frame planes in reference units, as the merge accumulates them.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct FramePlanes {
    /// Width of the expanded raster, in pixels.
    pub width: u32,
    /// Height of the expanded raster, in pixels.
    pub height: u32,
    /// Three interleaved channels, `scale * bmt`.
    pub x: Vec<f32>,
    /// Two values per pixel: the chroma variance and the luma variance.
    pub var: Vec<f32>,
    /// Saturation input per pixel, relative to the physical ceiling.
    pub d_max: Vec<f32>,
    /// One per pixel; nonzero when the pixel is valid.
    pub valid: Vec<u8>,
}

/// Build the alignment image of one frame. `cut` is the validity cut
/// relative to the ceiling. Placeholder: returns an empty image.
pub fn luma(planes: &FramePlanes, cut: f32, control: Control<'_>) -> Result<Luma, Error> {
    control.check()?;
    let _ = (planes, cut);
    Ok(Luma::default())
}

/// Estimate the transform of `frame` relative to `reference`.
/// Placeholder: returns a zero shift with a full peak.
pub fn estimate(
    reference: &Luma,
    frame: &Luma,
    options: &AlignOptions,
    control: Control<'_>,
) -> Result<Alignment, Error> {
    control.check()?;
    let _ = (reference, frame, options);
    Ok(Alignment {
        peak: 1.0,
        ..Alignment::default()
    })
}

/// Resample the planes of one frame onto the reference grid. Masks are never
/// interpolated: an output pixel is valid only when all source taps are valid
/// and inside the frame, and its `d_max` is the maximum over those taps.
/// Placeholder: an identity transform returns the planes unchanged.
pub fn warp(
    planes: FramePlanes,
    alignment: &Alignment,
    control: Control<'_>,
) -> Result<FramePlanes, Error> {
    control.check()?;
    if alignment.is_identity() || alignment.skipped {
        return Ok(planes);
    }
    Err(Error::InvalidData(
        "X3I alignment warp is not implemented yet".into(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn placeholder_alignment_is_identity_and_warp_returns_input() {
        let planes = FramePlanes {
            width: 2,
            height: 1,
            x: vec![1.0; 6],
            var: vec![1.0; 4],
            d_max: vec![0.5; 2],
            valid: vec![1; 2],
        };
        let luma_ref = luma(&planes, 0.85, Control::none()).unwrap();
        let alignment = estimate(
            &luma_ref,
            &luma_ref,
            &AlignOptions::default(),
            Control::none(),
        )
        .unwrap();
        assert!(alignment.is_identity());
        assert!(!alignment.skipped);
        let warped = warp(planes.clone(), &alignment, Control::none()).unwrap();
        assert_eq!(warped, planes);
    }
}
