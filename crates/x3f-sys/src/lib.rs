//! Low-level layer for the x3f Foveon raw converter.
//!
//! The decode + processing pipeline is native Rust in this crate's modules;
//! the only C/C++ left (an optional OpenCV denoise pass plus log/version
//! shims) lives in `csrc/`. Denoise also has a portable, pure-Rust
//! Non-Local Means implementation in [`denoise`] that owns the
//! `x3f_denoise` / `x3f_denoise_active` symbols on every target without an
//! opencv-mobile prebuilt (wasm, offline, …). This crate exposes the C ABI
//! verbatim — function names, struct layouts, and ownership semantics match
//! the C headers — so it is **unsafe** to use directly. Almost all callers
//! should consume the `x3f-core` crate instead, which wraps these bindings in
//! safe RAII types.
//!
//! `#[repr(C)]` struct/enum layouts are generated at build time by `bindgen`
//! from `wrapper.h` (see `csrc/`).

#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]
#![allow(clippy::all)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

// libc-compat shim. On every target except wasm32-unknown-unknown this
// is a transparent `pub use libc::*;`. On wasm32 it provides Rust-native
// equivalents (allocator, types, file I/O stubs); see the module header.
// Submodules opt in via `use crate::sysabi as libc;` so the existing
// `libc::*` call sites resolve through the shim on wasm32 without
// touching their bodies.
pub mod sysabi;

// `#[no_mangle]` Rust shim for the variadic `x3f_printf` symbol still
// referenced from the bindgen-generated bindings on wasm32. On non-wasm
// targets it comes from compiled C (`csrc/x3f_printf.c`); on wasm32 the C
// path is unavailable (no wasm-libc) so we provide the symbol directly. See
// the module header for why a 3-arg `x3f_printf` satisfies all variadic call
// sites at the wasm import boundary. (`x3f_denoise` / `x3f_denoise_active` /
// `x3f_set_use_opencl` used to live here too; they now come from the portable
// Rust denoise in `denoise.rs` — see below.)
#[cfg(target_arch = "wasm32")]
mod wasm_c_shims;
// `#[used]` anchor so cross-crate dead-code elimination doesn't strip
// `x3f_printf` before the bindgen-generated extern decl finds it at link time.
#[cfg(target_arch = "wasm32")]
mod _wasm_c_shim_anchors {
    use core::ffi::{c_char, c_int};

    #[used]
    static A1: unsafe extern "C" fn(c_int, *const c_char, c_int) = super::wasm_c_shims::x3f_printf;
}

// Portable, pure-Rust Non-Local Means denoise. Always compiled (so host CI
// type-checks and unit-tests it), but its `#[no_mangle]` C-ABI entry points
// (`x3f_denoise`, `x3f_denoise_active`, `x3f_set_use_opencl`) are only emitted
// when opencv-mobile was NOT linked (`cfg(not(x3f_opencv))`, set by build.rs):
// wasm, offline / docs.rs, and any triple without a prebuilt. On an OpenCV
// build the C++ in `csrc/` owns those symbols and these are gated out; set
// `X3F_PORTABLE_DENOISE=1` to route the Rust path anyway (A/B testing).
mod denoise;
// `#[used]` anchors so the rlib keeps the Rust denoise symbols for the
// bindgen-generated extern decls the rest of the crate calls through (same
// rationale as the wasm `x3f_printf` anchor above).
#[cfg(not(x3f_opencv))]
mod _denoise_anchors {
    use core::ffi::{c_int, c_void};

    #[used]
    static D1: unsafe extern "C" fn(*mut c_void, c_int, f32) = super::denoise::x3f_denoise;
    #[used]
    static D2: unsafe extern "C" fn(*mut c_void, c_int, c_int, f32) =
        super::denoise::x3f_denoise_active;
    #[used]
    static D3: unsafe extern "C" fn(c_int) = super::denoise::x3f_set_use_opencl;
}

// Native Rust replacement for `x3f_expand_quattro`. The legacy C symbol of
// the same name in `src/x3f_process.c` resolves to this `#[no_mangle] extern
// "C"` definition at link time. No competing C stub defines it (the old
// `csrc/denoise_stub.c` was deleted once denoise moved to `src/denoise.rs`);
// a second definition would be a duplicate-symbol link error. The `#[used]`
// static below anchors the symbol so cross-crate dead-code elimination cannot
// strip it before the C call site sees it.
mod quattro;

// M5b: native Rust TRUE entropy decoder. Opt-in via the `X3F_RUST_DECODE`
// env var, which is checked in src/x3f_io.c's `true_decode`. The Rust
// definition lives here and is exported under `x3f_rust_true_decode`.
mod entropy;

// M4a: native Rust port of src/x3f_meta.c — read-only accessors over an
// already-parsed x3f_t. Each x3f_get_* function is `#[no_mangle] extern
// "C"`; the corresponding C source was removed from the cc-rs build,
// and the bindgen forward declarations are blocklisted to avoid the
// "name defined multiple times" Rust error. We re-export the Rust
// definitions in the same `x3f_sys::x3f_get_*` paths so x3f-core's
// FFI call sites do not need to change.
mod meta;
pub use meta::{
    x3f_get_camf_float, x3f_get_camf_float_vector, x3f_get_camf_matrix, x3f_get_camf_matrix_for_wb,
    x3f_get_camf_matrix_var, x3f_get_camf_property, x3f_get_camf_property_list,
    x3f_get_camf_signed, x3f_get_camf_signed_vector, x3f_get_camf_text, x3f_get_camf_unsigned,
    x3f_get_max_raw, x3f_get_prop_entry, x3f_get_wb,
};

// M4b: native Rust port of src/x3f_print_meta.c — text dump of the
// parsed x3f_t. Same pattern as meta.rs: the C source is removed
// from cc-rs, bindgen forward decls blocklisted, Rust definitions
// re-exported.
//
// Excluded from `wasm32-unknown-unknown` (M8d-α): the byte-identical
// %g / %9f format-string output goes through `libc::printf` /
// `fprintf`, which Rust can't variadically shim on stable. The
// metadata-dump entry point isn't reachable from any wasm consumer
// surface anyway (it writes a file path); on wasm32 we provide a
// stub `x3f_dump_meta_data` that returns `X3F_ARGUMENT_ERROR`.
#[cfg(not(target_arch = "wasm32"))]
mod print_meta;
#[cfg(not(target_arch = "wasm32"))]
pub use print_meta::{max_printed_matrix_elements, x3f_dump_meta_data, x3f_print_meta};

#[cfg(target_arch = "wasm32")]
mod print_meta_wasm_stub;
#[cfg(target_arch = "wasm32")]
pub use print_meta_wasm_stub::{max_printed_matrix_elements, x3f_dump_meta_data, x3f_print_meta};

// M4c: native Rust port of x3f_new_from_file (the X3F header + directory
// walker) and the file-IO helpers from src/x3f_io.c.
//
// M5e: the cleanup machinery (cleanup_huffman_tree, cleanup_true,
// cleanup_quattro, cleanup_huffman, free_camf_entry), the delete
// orchestrator (x3f_delete), the directory-entry searchers
// (x3f_get_raw / _thumb_{plain,huffman,jpeg} / _camf / _prop), and
// the `legacy_offset` / `auto_legacy_offset` globals all moved to
// Rust. With this, src/x3f_io.c is dropped from the cc-rs source
// list — its content is now entirely in src/io.rs + src/load.rs +
// src/entropy.rs + src/quattro.rs.
mod io;
pub use io::{
    auto_legacy_offset, legacy_offset, x3f_delete, x3f_get_camf, x3f_get_prop, x3f_get_raw,
    x3f_get_thumb_huffman, x3f_get_thumb_jpeg, x3f_get_thumb_plain, x3f_new_from_file,
};

// M4d: native Rust port of x3f_load_data and the section data loaders
// (property list, image RAW/thumb/JPEG, CAMF type 2/4/5 + entry walker).
// What remains in src/x3f_io.c after this milestone is just the cleanup
// machinery (`x3f_delete`, `cleanup_*`, `free_camf_entry`) and the
// directory-entry searchers (`x3f_get_*`).
mod load;
pub use load::{x3f_err, x3f_load_data, x3f_load_image_block};

// M6a: native Rust port of src/x3f_matrix.c — 3×3 / 3×1 matrix math,
// color-space conversion matrices, and gamma/sRGB LUT helpers. Same
// pattern as the M4 ports: C source removed from cc-rs, bindgen forward
// declarations blocklisted, Rust definitions re-exported.
mod matrix;
pub use matrix::{
    x3f_3x1_comp_mul, x3f_3x1_invert, x3f_3x1_print, x3f_3x3_3x1_mul, x3f_3x3_3x3_mul,
    x3f_3x3_diag, x3f_3x3_identity, x3f_3x3_inverse, x3f_3x3_ones, x3f_3x3_print,
    x3f_AdobeRGB_to_XYZ, x3f_Bradford_D50_to_D65, x3f_Bradford_D65_to_D50, x3f_CIERGB_to_XYZ,
    x3f_LUT_lookup, x3f_ProPhotoRGB_to_XYZ, x3f_XYZ_to_AdobeRGB, x3f_XYZ_to_ProPhotoRGB,
    x3f_XYZ_to_sRGB, x3f_cineon_log_LUT, x3f_gamma_LUT, x3f_sRGB_LUT, x3f_sRGB_to_XYZ,
    x3f_scalar_3x1_mul, x3f_scalar_3x3_mul,
};

// M6b: native Rust port of src/x3f_histogram.c — debug histogram CSV
// dump driven by the `-histogram` / `-loghist` CLI flags. The processed
// image still comes from `x3f_get_image` (still C in `x3f_process.c`).
//
// Excluded on wasm32-unknown-unknown (M8d-α): writes to a `FILE*` via
// `libc::fprintf` (variadic, unstable to shim). The histogram debug
// surface isn't reachable from any wasm consumer entrypoint; we
// substitute a stub returning `X3F_ARGUMENT_ERROR`.
#[cfg(not(target_arch = "wasm32"))]
mod histogram;
#[cfg(not(target_arch = "wasm32"))]
pub use histogram::x3f_dump_raw_data_as_histogram;

#[cfg(target_arch = "wasm32")]
mod histogram_wasm_stub;
#[cfg(target_arch = "wasm32")]
pub use histogram_wasm_stub::x3f_dump_raw_data_as_histogram;

// M6c: native Rust port of src/x3f_image.c — image-area accessors
// (raw + Quattro top), coordinate-based cropping, and CAMF-driven
// cropping (KeepImageArea, ActiveImageArea, DarkShieldColRange).
// Crops produce *views* into the parent area; no allocations.
mod image;
pub use image::{
    x3f_crop_area, x3f_crop_area8, x3f_crop_area8_camf, x3f_crop_area_camf, x3f_crop_area_column,
    x3f_get_camf_rect, x3f_image_area, x3f_image_area_qtop,
};

// M6d: native Rust port of src/x3f_spatial_gain.c — per-pixel lens
// shading correction tables. Reads CAMF, picks four nearest neighbours
// in (1/aperture, lens-position) space, builds bilinear weights, and
// either keeps the raw tables for on-the-fly combination via
// `x3f_calc_spatial_gain` or pre-interpolates a dense table per channel.
mod spatial_gain;
pub use spatial_gain::{
    x3f_calc_spatial_gain, x3f_cleanup_spatial_gain, x3f_get_classic_spatial_gain,
    x3f_get_interp_merrill_type_spatial_gain, x3f_get_merrill_type_spatial_gain,
    x3f_get_spatial_gain,
};

// M6e: incremental native Rust port of src/x3f_process.c. M6e1 lifts
// the color math accessors (x3f_get_gain, x3f_get_bmt_to_xyz,
// x3f_get_raw_to_xyz) — the pure CAMF-and-matrix stages — out of C.
// M6e2 adds the black-level / intermediate-range stats; M6e3 the
// bad-pixel interpolation; M6e4 (in highlight.rs) the highlight
// recovery family. preprocess_data / convert_data / convert entry
// points remain C until later phases.
mod process;
pub use process::{
    x3f_get_bmt_to_xyz, x3f_get_dng_highlight_scale, x3f_get_dng_shoulder_knee, x3f_get_gain,
    x3f_get_image, x3f_get_preview, x3f_get_raw_to_xyz, x3f_set_cineon,
    x3f_set_dng_highlight_recovery,
};

// M6e4: highlight-recovery family (highlight_params, chroma LUT, sat
// map, RepairPix, reconstruct_highlights). Active research code —
// ported verbatim from src/x3f_process.c with mirror #[repr(C)] structs
// so the still-C preprocess_data / convert_data can keep allocating
// these on the stack and passing them by pointer.
mod highlight;
pub use highlight::{
    build_sat_map, chroma_lut_apply_pixel, chroma_lut_apply_pixel_bmt,
    chroma_lut_apply_stats_print, chroma_lut_apply_stats_t, chroma_lut_build_from_image,
    chroma_lut_init_defaults, chroma_lut_t, compute_chroma_prior, get_highlight_params,
    highlight_params_t, reconstruct_highlights, repair_pix_apply_pixel, repair_pix_init_defaults,
    repair_pix_t,
};
