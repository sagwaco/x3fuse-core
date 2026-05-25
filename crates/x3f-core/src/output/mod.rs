//! Output writers (PPM, TIFF, DNG, …) that consume a processed [`Image`].
//!
//! The legacy C library has one `x3f_dump_raw_data_as_<fmt>` per format,
//! each calling `x3f_get_image` internally. The Rust port factors that out:
//! [`Reader::get_image`] returns an `Image`; format-specific writers in
//! submodules of this module take the `Image` and emit bytes.
//!
//! [`Image`]: crate::Image
//! [`Reader::get_image`]: crate::Reader::get_image

pub mod dng;
pub mod ppm;
pub mod tiff;
