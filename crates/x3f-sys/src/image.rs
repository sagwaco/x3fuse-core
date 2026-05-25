//! M6c — native Rust port of `src/x3f_image.c`.
//!
//! Image-area helpers used by `x3f_process.c`: pulling the 16-bit raster
//! out of a parsed `x3f_t`, cropping to coordinate ranges or to CAMF-named
//! regions (`KeepImageArea`, `ActiveImageArea`, `DarkShieldColRange`),
//! and the Quattro top-layer accessor.
//!
//! Memory ownership: every "crop" is a *view* — the returned area shares
//! the parent's `buf` pointer (or `NULL` when the parent owns the buffer
//! through `cleanup_huffman` / `cleanup_true` / `cleanup_quattro`). No
//! allocations happen here. Same pattern as the C source.
//!
//! `x3f_crop_area8_camf` deliberately casts an `x3f_area8_t *` to
//! `x3f_area16_t *` when calling `x3f_get_camf_rect` — the two structs
//! share field offsets for the metadata that function reads (`rows`,
//! `columns`), so the cast is safe and matches the C source.
#![allow(clippy::missing_safety_doc)]

use std::ptr;

use crate::*;
// libc compat — see `sysabi.rs`.
use crate::sysabi as libc;

// ----------------------------------------------------------------------
// Image-area accessors
// ----------------------------------------------------------------------

#[no_mangle]
pub unsafe extern "C" fn x3f_image_area(x3f: *mut x3f_t, image: *mut x3f_area16_t) -> libc::c_int {
    let de = unsafe { x3f_get_raw(x3f) };
    if de.is_null() {
        return 0;
    }
    let id = unsafe { &(*de).header.data_subsection.image_data };
    let huf = id.huffman;
    let tru = id.tru;

    let mut area: *const x3f_area16_t = ptr::null();
    if !huf.is_null() {
        area = unsafe { &(*huf).x3rgb16 };
    }
    if !tru.is_null() {
        area = unsafe { &(*tru).x3rgb16 };
    }

    if area.is_null() || unsafe { (*area).data.is_null() } {
        return 0;
    }

    unsafe {
        *image = *area;
        // cleanup_true / cleanup_huffman owns the underlying buf;
        // null this so the consumer doesn't double-free.
        (*image).buf = ptr::null_mut();
    }
    1
}

#[no_mangle]
pub unsafe extern "C" fn x3f_image_area_qtop(
    x3f: *mut x3f_t,
    image: *mut x3f_area16_t,
) -> libc::c_int {
    let de = unsafe { x3f_get_raw(x3f) };
    if de.is_null() {
        return 0;
    }
    let id = unsafe { &(*de).header.data_subsection.image_data };
    let q = id.quattro;
    if q.is_null() || unsafe { (*q).top16.data.is_null() } {
        return 0;
    }

    unsafe {
        *image = (*q).top16;
        // cleanup_quattro owns the underlying buf.
        (*image).buf = ptr::null_mut();
    }
    1
}

// ----------------------------------------------------------------------
// Coordinate-based cropping
// ----------------------------------------------------------------------

#[no_mangle]
pub unsafe extern "C" fn x3f_crop_area(
    coord: *mut u32,
    image: *mut x3f_area16_t,
    crop: *mut x3f_area16_t,
) -> libc::c_int {
    let c = unsafe { std::slice::from_raw_parts(coord, 4) };
    if c[0] > c[2] || c[1] > c[3] {
        return 0;
    }
    let img = unsafe { &*image };
    if c[2] >= img.columns || c[3] >= img.rows {
        return 0;
    }

    unsafe {
        let off = (img.channels * c[0] + img.row_stride * c[1]) as isize;
        (*crop).data = img.data.offset(off);
        (*crop).columns = c[2] - c[0] + 1;
        (*crop).rows = c[3] - c[1] + 1;
        (*crop).channels = img.channels;
        (*crop).row_stride = img.row_stride;
        (*crop).buf = img.buf;
    }
    1
}

#[no_mangle]
pub unsafe extern "C" fn x3f_crop_area8(
    coord: *mut u32,
    image: *mut x3f_area8_t,
    crop: *mut x3f_area8_t,
) -> libc::c_int {
    let c = unsafe { std::slice::from_raw_parts(coord, 4) };
    if c[0] > c[2] || c[1] > c[3] {
        return 0;
    }
    let img = unsafe { &*image };
    if c[2] >= img.columns || c[3] >= img.rows {
        return 0;
    }

    unsafe {
        let off = (img.channels * c[0] + img.row_stride * c[1]) as isize;
        (*crop).data = img.data.offset(off);
        (*crop).columns = c[2] - c[0] + 1;
        (*crop).rows = c[3] - c[1] + 1;
        (*crop).channels = img.channels;
        (*crop).row_stride = img.row_stride;
        (*crop).buf = img.buf;
    }
    1
}

// ----------------------------------------------------------------------
// CAMF-driven cropping
// ----------------------------------------------------------------------

/// Translate a coordinate rect into the resolution and origin of `image`,
/// using `KeepImageArea` as the reference frame.
///
/// For `rescale = 0`, image MUST be at the resolution of KeepImageArea;
/// it can be larger but not smaller.
///
/// For `rescale = 1`, image's bounds MUST exactly match KeepImageArea's,
/// but their resolutions can differ (we scale the rect to image's grid).
unsafe fn x3f_transform_rect_to_keep_image(
    x3f: *mut x3f_t,
    image: *mut x3f_area16_t,
    rescale: libc::c_int,
    rect: *mut u32,
) -> libc::c_int {
    let mut keep: [u32; 4] = [0; 4];
    let ok = unsafe {
        x3f_get_camf_matrix(
            x3f,
            c"KeepImageArea".as_ptr() as *mut libc::c_char,
            4,
            0,
            0,
            matrix_type_t_M_UINT,
            keep.as_mut_ptr() as *mut libc::c_void,
        )
    };
    if ok == 0 {
        return 0;
    }

    let keep_cols = keep[2] - keep[0] + 1;
    let keep_rows = keep[3] - keep[1] + 1;

    let r = unsafe { std::slice::from_raw_parts_mut(rect, 4) };

    // Make sure that at least some part of rect is within the bounds of
    // KeepImageArea.
    if r[0] > keep[2] || r[1] > keep[3] || r[2] < keep[0] || r[3] < keep[1] {
        unsafe {
            x3f_printf(
                x3f_verbosity_t_WARN,
                c"CAMF rect (%u,%u,%u,%u) completely out of bounds : KeepImageArea (%u,%u,%u,%u)\n"
                    .as_ptr(),
                r[0],
                r[1],
                r[2],
                r[3],
                keep[0],
                keep[1],
                keep[2],
                keep[3],
            );
        }
        return 0;
    }

    // Clip rect to the bounds of KeepImageArea.
    if r[0] < keep[0] {
        r[0] = keep[0];
    }
    if r[1] < keep[1] {
        r[1] = keep[1];
    }
    if r[2] > keep[2] {
        r[2] = keep[2];
    }
    if r[3] > keep[3] {
        r[3] = keep[3];
    }

    // Translate so coordinates are relative to the origin of KeepImageArea.
    r[0] -= keep[0];
    r[1] -= keep[1];
    r[2] -= keep[0];
    r[3] -= keep[1];

    let img = unsafe { &*image };

    if rescale != 0 {
        // Rescale rect from KeepImageArea resolution to image resolution.
        r[0] = r[0] * img.columns / keep_cols;
        r[1] = r[1] * img.rows / keep_rows;
        r[2] = r[2] * img.columns / keep_cols;
        r[3] = r[3] * img.rows / keep_rows;
    } else if keep_cols > img.columns || keep_rows > img.rows {
        unsafe {
            x3f_printf(
                x3f_verbosity_t_WARN,
                c"KeepImageArea (%u,%u,%u,%u) out of bounds : image size (%u,%u)\n".as_ptr(),
                keep[0],
                keep[1],
                keep[2],
                keep[3],
                img.columns,
                img.rows,
            );
        }
        return 0;
    }

    1
}

#[no_mangle]
pub unsafe extern "C" fn x3f_get_camf_rect(
    x3f: *mut x3f_t,
    name: *mut libc::c_char,
    image: *mut x3f_area16_t,
    rescale: libc::c_int,
    rect: *mut u32,
) -> libc::c_int {
    let ok = unsafe {
        x3f_get_camf_matrix(
            x3f,
            name,
            4,
            0,
            0,
            matrix_type_t_M_UINT,
            rect as *mut libc::c_void,
        )
    };
    if ok == 0 {
        return 0;
    }
    unsafe { x3f_transform_rect_to_keep_image(x3f, image, rescale, rect) }
}

#[no_mangle]
pub unsafe extern "C" fn x3f_crop_area_column(
    x3f: *mut x3f_t,
    which_side: col_side_t,
    image: *mut x3f_area16_t,
    rescale: libc::c_int,
    crop: *mut x3f_area16_t,
) -> libc::c_int {
    let mut column: [u32; 4] = [0; 4];
    let ok = unsafe {
        x3f_get_camf_matrix(
            x3f,
            c"DarkShieldColRange".as_ptr() as *mut libc::c_char,
            2,
            2,
            0,
            matrix_type_t_M_UINT,
            column.as_mut_ptr() as *mut libc::c_void,
        )
    };
    if ok == 0 {
        return 0;
    }

    let mut rect: [u32; 4] = [0; 4];
    rect[1] = 0;
    rect[3] = u32::MAX;

    if which_side == col_side_t_COL_SIDE_LEFT {
        rect[0] = column[0];
        rect[2] = column[1];
    } else if which_side == col_side_t_COL_SIDE_RIGHT {
        rect[0] = column[2];
        rect[2] = column[3];
    } else {
        return 0;
    }

    if unsafe { x3f_transform_rect_to_keep_image(x3f, image, rescale, rect.as_mut_ptr()) } != 0 {
        let r = unsafe { x3f_crop_area(rect.as_mut_ptr(), image, crop) };
        debug_assert!(r != 0);
        return 1;
    }
    0
}

#[no_mangle]
pub unsafe extern "C" fn x3f_crop_area_camf(
    x3f: *mut x3f_t,
    name: *mut libc::c_char,
    image: *mut x3f_area16_t,
    rescale: libc::c_int,
    crop: *mut x3f_area16_t,
) -> libc::c_int {
    let mut rect: [u32; 4] = [0; 4];
    if unsafe { x3f_get_camf_rect(x3f, name, image, rescale, rect.as_mut_ptr()) } == 0 {
        return 0;
    }
    let r = unsafe { x3f_crop_area(rect.as_mut_ptr(), image, crop) };
    debug_assert!(r != 0);
    1
}

#[no_mangle]
pub unsafe extern "C" fn x3f_crop_area8_camf(
    x3f: *mut x3f_t,
    name: *mut libc::c_char,
    image: *mut x3f_area8_t,
    rescale: libc::c_int,
    crop: *mut x3f_area8_t,
) -> libc::c_int {
    let mut rect: [u32; 4] = [0; 4];
    // Cast area8 → area16 to call x3f_get_camf_rect; the two structs
    // share field offsets for `rows` and `columns`, which is all that
    // function reads. Identical to the C source.
    let img_as_16 = image as *mut x3f_area16_t;
    if unsafe { x3f_get_camf_rect(x3f, name, img_as_16, rescale, rect.as_mut_ptr()) } == 0 {
        return 0;
    }
    let r = unsafe { x3f_crop_area8(rect.as_mut_ptr(), image, crop) };
    debug_assert!(r != 0);
    1
}

// ----------------------------------------------------------------------
// Symbol anchors
// ----------------------------------------------------------------------

#[used]
static _ANCHOR_IMG_AREA: unsafe extern "C" fn(*mut x3f_t, *mut x3f_area16_t) -> libc::c_int =
    x3f_image_area;
#[used]
static _ANCHOR_IMG_AREA_QTOP: unsafe extern "C" fn(*mut x3f_t, *mut x3f_area16_t) -> libc::c_int =
    x3f_image_area_qtop;
#[used]
static _ANCHOR_CROP_AREA: unsafe extern "C" fn(
    *mut u32,
    *mut x3f_area16_t,
    *mut x3f_area16_t,
) -> libc::c_int = x3f_crop_area;
#[used]
static _ANCHOR_CROP_AREA8: unsafe extern "C" fn(
    *mut u32,
    *mut x3f_area8_t,
    *mut x3f_area8_t,
) -> libc::c_int = x3f_crop_area8;
#[used]
static _ANCHOR_GET_CAMF_RECT: unsafe extern "C" fn(
    *mut x3f_t,
    *mut libc::c_char,
    *mut x3f_area16_t,
    libc::c_int,
    *mut u32,
) -> libc::c_int = x3f_get_camf_rect;
#[used]
static _ANCHOR_CROP_AREA_COL: unsafe extern "C" fn(
    *mut x3f_t,
    col_side_t,
    *mut x3f_area16_t,
    libc::c_int,
    *mut x3f_area16_t,
) -> libc::c_int = x3f_crop_area_column;
#[used]
static _ANCHOR_CROP_AREA_CAMF: unsafe extern "C" fn(
    *mut x3f_t,
    *mut libc::c_char,
    *mut x3f_area16_t,
    libc::c_int,
    *mut x3f_area16_t,
) -> libc::c_int = x3f_crop_area_camf;
#[used]
static _ANCHOR_CROP_AREA8_CAMF: unsafe extern "C" fn(
    *mut x3f_t,
    *mut libc::c_char,
    *mut x3f_area8_t,
    libc::c_int,
    *mut x3f_area8_t,
) -> libc::c_int = x3f_crop_area8_camf;
