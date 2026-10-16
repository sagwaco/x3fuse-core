//! Per-frame export that the X3I merge consumes.
//!
//! The merge in `x3f-core` reads one `MergeFrame` for each exposure of an
//! X3I container. The producer lives in `process.rs`. The design is in
//! `_local/X3I_SFD_PLAN.md`, sections 4.2 and 4.5.

/// One prepared Quattro exposure. Values are in the units of this frame:
/// 1.0 is the white level of the frame.
///
/// Invariants: `width == 2 * raw_half_width`, `height == 2 * raw_half_height`,
/// and `raw_top` covers exactly the expanded raster.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MergeFrame {
    /// Position of the frame in the X3I directory. Frame 0 is the 0 EV reference.
    pub frame_index: usize,
    /// Exposure time of the frame in microseconds, from CAMF `CaptureShutter`.
    pub shutter_us: u32,
    /// Width of the expanded full-resolution raster, in pixels.
    pub width: u32,
    /// Height of the expanded full-resolution raster, in pixels.
    pub height: u32,
    /// Expanded camera samples, three `f32` per pixel, row-major, interleaved,
    /// `width * height * 3` long. The channel order is the order of the
    /// processed area that `get_image_impl` returns for the `NONE` encoding,
    /// which is the order `camera_to_rgb` consumes. Channel 2 is the top layer.
    /// Each sample is `(value - black) / (white - black)` of the processed
    /// image levels: white is 1.0 and the intermediate bias is removed.
    /// These values come from the expanded u16 area. They include quantization
    /// and clamps: Y = min(4T, 65535), and the intermediate bias floor is 0.
    /// Expansion uses repaired intermediate planes when `fix_bad` is enabled.
    /// Spatial gain is multiplied in when `gain` is `Some`. No color matrix
    /// is applied.
    pub bmt: Vec<f32>,
    /// Spatial gain per pixel and channel that is already multiplied into
    /// `bmt`, `width * height * 3` long. `None` means a gain of 1.0 everywhere.
    pub gain: Option<Vec<f32>>,
    /// Top-layer intermediate counts after normalization and bad-pixel repair
    /// when `fix_bad` is enabled. These are the samples that expansion reads.
    /// One `u16` per expanded pixel, `width * height` long.
    /// The intermediate encoding clamps at 0, which truncates negative noise.
    pub raw_top: Vec<u16>,
    /// Intermediate counts of the two half-resolution layers after normalization
    /// and bad-pixel repair when `fix_bad` is enabled. These are the samples that
    /// expansion reads, interleaved in channel order 0, 1,
    /// `raw_half_width * raw_half_height * 2` long. The snapshot excludes the
    /// top-layer sum in channel 2. The encoding clamps at 0 and truncates negative noise.
    pub raw_half: Vec<u16>,
    /// Width of the half-resolution raw planes, in samples.
    pub raw_half_width: u32,
    /// Height of the half-resolution raw planes, in samples.
    pub raw_half_height: u32,
    /// Intermediate black level per channel (0, 1, top), in snapshot counts.
    pub raw_black: [f64; 3],
    /// Intermediate white level per channel (0, 1, top), in snapshot counts.
    /// A normalized snapshot sample is `(value - raw_black) / (raw_white - raw_black)`.
    pub raw_white: [f64; 3],
    /// Read noise per channel (0, 1, top) as a fraction of
    /// `raw_white - raw_black`, measured on the dark shields of this frame.
    pub sigma: [f32; 3],
    /// Camera-to-RGB matrix of this frame at the export white balance, row-major.
    pub camera_to_rgb: [f64; 9],
}
