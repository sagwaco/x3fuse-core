//! M4-slice-1 — native Rust port of `src/x3f_meta.c`.
//!
//! Pure read-only accessors over an already-parsed `x3f_t` (CAMF + PROP
//! sections). Walks `x3f->directory_section.directory_entry[]` looking for
//! the right section, then iterates its entry table by name.
//!
//! Unlike the M5 entropy decoders, these functions use the bindgen-
//! generated FFI types directly — `x3f_t`, `x3f_camf_t`, `camf_entry_t`,
//! etc. Layouts come from the same `wrapper.h` the C code sees, so there
//! is no risk of mirror-struct drift. The bindgen union for
//! `data_subsection` is accessed via the `*_bindgen_ty_1.camf` /
//! `.property_list` field which the C code's `union { ... } data_subsection`
//! becomes.
//!
//! Symbol export: each `x3f_get_*` function is `#[no_mangle] extern "C"`,
//! blocklisted in bindgen, and anchored via a `#[used]` static. The C
//! code in `src/x3f_process.c` and elsewhere resolves to these definitions
//! at link time. The corresponding C source `x3f_meta.c` is removed from
//! the `x3f-sys` `cc-rs` build.

use std::ffi::CStr;
use std::os::raw::{c_char, c_int, c_uint, c_void};
use std::ptr;

use crate::*;

/// Identifier-tag values for CAMF entries. The C definitions in
/// `src/x3f_io.h` are `#define X3F_CMbT (uint32_t)(0x54624d43)` etc., a
/// form that bindgen's macro translator skips, so they're inlined here.
/// Byte order: ASCII 'C','M','b',{T|M|P} stored little-endian.
const ID_TEXT: u32 = 0x5462_4d43; // CMbT
const ID_MATRIX: u32 = 0x4d62_4d43; // CMbM
const ID_PROPERTY: u32 = 0x5062_4d43; // CMbP

#[inline]
unsafe fn cstr_eq(a: *const c_char, b: *const c_char) -> bool {
    if a.is_null() || b.is_null() {
        return false;
    }
    // Both pointers come from the C library and reference NUL-terminated
    // strings inside the parsed x3f_t. No allocations, no encoding.
    unsafe { CStr::from_ptr(a) == CStr::from_ptr(b) }
}

/// Find the CAMF section in the directory and return its decoded entry
/// table. Returns null if the file has no CAMF section (extremely unusual)
/// or the section was never loaded with `x3f_load_data`.
unsafe fn camf_table<'a>(x3f: *mut x3f_t) -> Option<&'a [camf_entry_t]> {
    let de = unsafe { x3f_get_camf(x3f) };
    if de.is_null() {
        return None;
    }
    let camf: *mut x3f_camf_t = unsafe { &mut (*de).header.data_subsection.camf };
    let table = unsafe { (*camf).entry_table };
    if table.element.is_null() || table.size == 0 {
        return None;
    }
    Some(unsafe { std::slice::from_raw_parts(table.element, table.size as usize) })
}

unsafe fn camf_find_by_name<'a>(x3f: *mut x3f_t, name: *const c_char) -> Option<&'a camf_entry_t> {
    let table = unsafe { camf_table(x3f) }?;
    for entry in table {
        if unsafe { cstr_eq(name, entry.name_address) } {
            return Some(entry);
        }
    }
    None
}

#[no_mangle]
pub unsafe extern "C" fn x3f_get_camf_text(
    x3f: *mut x3f_t,
    name: *mut c_char,
    text: *mut *mut c_char,
) -> c_int {
    let Some(entry) = (unsafe { camf_find_by_name(x3f, name) }) else {
        return 0;
    };
    if entry.id != ID_TEXT {
        return 0;
    }
    unsafe { *text = entry.text };
    1
}

#[no_mangle]
pub unsafe extern "C" fn x3f_get_camf_matrix_var(
    x3f: *mut x3f_t,
    name: *mut c_char,
    dim0: *mut c_int,
    dim1: *mut c_int,
    dim2: *mut c_int,
    typ: matrix_type_t,
    matrix: *mut *mut c_void,
) -> c_int {
    let Some(entry) = (unsafe { camf_find_by_name(x3f, name) }) else {
        return 0;
    };
    if entry.id != ID_MATRIX {
        return 0;
    }
    if entry.matrix_decoded_type != typ {
        return 0;
    }

    let dims = entry.matrix_dim_entry;
    match entry.matrix_dim {
        3 => {
            if dim2.is_null() || dim1.is_null() || dim0.is_null() {
                return 0;
            }
            unsafe {
                *dim2 = (*dims.add(2)).size as c_int;
                *dim1 = (*dims.add(1)).size as c_int;
                *dim0 = (*dims.add(0)).size as c_int;
            }
        }
        2 => {
            if !dim2.is_null() || dim1.is_null() || dim0.is_null() {
                return 0;
            }
            unsafe {
                *dim1 = (*dims.add(1)).size as c_int;
                *dim0 = (*dims.add(0)).size as c_int;
            }
        }
        1 => {
            if !dim2.is_null() || !dim1.is_null() || dim0.is_null() {
                return 0;
            }
            unsafe { *dim0 = (*dims.add(0)).size as c_int };
        }
        _ => return 0,
    }

    unsafe { *matrix = entry.matrix_decoded };
    1
}

#[no_mangle]
pub unsafe extern "C" fn x3f_get_camf_matrix(
    x3f: *mut x3f_t,
    name: *mut c_char,
    dim0: c_int,
    dim1: c_int,
    dim2: c_int,
    typ: matrix_type_t,
    matrix: *mut c_void,
) -> c_int {
    let Some(entry) = (unsafe { camf_find_by_name(x3f, name) }) else {
        return 0;
    };
    if entry.id != ID_MATRIX {
        return 0;
    }
    if entry.matrix_decoded_type != typ {
        return 0;
    }

    let dims = entry.matrix_dim_entry;
    let ok = match entry.matrix_dim {
        3 => unsafe {
            dim2 == (*dims.add(2)).size as c_int
                && dim1 == (*dims.add(1)).size as c_int
                && dim0 == (*dims.add(0)).size as c_int
        },
        2 => unsafe {
            dim2 == 0
                && dim1 == (*dims.add(1)).size as c_int
                && dim0 == (*dims.add(0)).size as c_int
        },
        1 => unsafe { dim2 == 0 && dim1 == 0 && dim0 == (*dims.add(0)).size as c_int },
        _ => false,
    };
    if !ok {
        return 0;
    }

    let elem_size = if entry.matrix_decoded_type == matrix_type_t_M_FLOAT {
        std::mem::size_of::<f64>()
    } else {
        std::mem::size_of::<u32>()
    };
    let bytes = elem_size * entry.matrix_elements as usize;
    unsafe {
        std::ptr::copy_nonoverlapping(entry.matrix_decoded as *const u8, matrix as *mut u8, bytes);
    }
    1
}

#[no_mangle]
pub unsafe extern "C" fn x3f_get_camf_float(
    x3f: *mut x3f_t,
    name: *mut c_char,
    val: *mut f64,
) -> c_int {
    unsafe {
        x3f_get_camf_matrix(
            x3f,
            name,
            1,
            0,
            0,
            matrix_type_t_M_FLOAT,
            val as *mut c_void,
        )
    }
}

#[no_mangle]
pub unsafe extern "C" fn x3f_get_camf_float_vector(
    x3f: *mut x3f_t,
    name: *mut c_char,
    val: *mut f64,
) -> c_int {
    unsafe {
        x3f_get_camf_matrix(
            x3f,
            name,
            3,
            0,
            0,
            matrix_type_t_M_FLOAT,
            val as *mut c_void,
        )
    }
}

#[no_mangle]
pub unsafe extern "C" fn x3f_get_camf_unsigned(
    x3f: *mut x3f_t,
    name: *mut c_char,
    val: *mut u32,
) -> c_int {
    unsafe { x3f_get_camf_matrix(x3f, name, 1, 0, 0, matrix_type_t_M_UINT, val as *mut c_void) }
}

#[no_mangle]
pub unsafe extern "C" fn x3f_get_camf_signed(
    x3f: *mut x3f_t,
    name: *mut c_char,
    val: *mut i32,
) -> c_int {
    unsafe { x3f_get_camf_matrix(x3f, name, 1, 0, 0, matrix_type_t_M_INT, val as *mut c_void) }
}

#[no_mangle]
pub unsafe extern "C" fn x3f_get_camf_signed_vector(
    x3f: *mut x3f_t,
    name: *mut c_char,
    val: *mut i32,
) -> c_int {
    unsafe { x3f_get_camf_matrix(x3f, name, 3, 0, 0, matrix_type_t_M_INT, val as *mut c_void) }
}

#[no_mangle]
pub unsafe extern "C" fn x3f_get_camf_property_list(
    x3f: *mut x3f_t,
    list: *mut c_char,
    names: *mut *mut *mut c_char,
    values: *mut *mut *mut c_char,
    num: *mut c_uint,
) -> c_int {
    let Some(entry) = (unsafe { camf_find_by_name(x3f, list) }) else {
        return 0;
    };
    if entry.id != ID_PROPERTY {
        return 0;
    }
    unsafe {
        *names = entry.property_name;
        *values = entry.property_value as *mut *mut c_char;
        *num = entry.property_num;
    }
    1
}

#[no_mangle]
pub unsafe extern "C" fn x3f_get_camf_property(
    x3f: *mut x3f_t,
    list: *mut c_char,
    name: *mut c_char,
    value: *mut *mut c_char,
) -> c_int {
    let mut names: *mut *mut c_char = ptr::null_mut();
    let mut values: *mut *mut c_char = ptr::null_mut();
    let mut num: c_uint = 0;
    if unsafe { x3f_get_camf_property_list(x3f, list, &mut names, &mut values, &mut num) } == 0 {
        return 0;
    }
    for i in 0..num as usize {
        let n = unsafe { *names.add(i) };
        if unsafe { cstr_eq(n, name) } {
            unsafe { *value = *values.add(i) };
            return 1;
        }
    }
    0
}

#[no_mangle]
pub unsafe extern "C" fn x3f_get_prop_entry(
    x3f: *mut x3f_t,
    name: *mut c_char,
    value: *mut *mut c_char,
) -> c_int {
    let de = unsafe { x3f_get_prop(x3f) };
    if de.is_null() {
        return 0;
    }
    let pl: *mut x3f_property_list_t = unsafe { &mut (*de).header.data_subsection.property_list };
    let table = unsafe { (*pl).property_table };
    if table.element.is_null() {
        return 0;
    }
    let entries = unsafe { std::slice::from_raw_parts(table.element, table.size as usize) };
    for entry in entries {
        if unsafe { cstr_eq(name, entry.name_utf8) } {
            unsafe { *value = entry.value_utf8 };
            return 1;
        }
    }
    0
}

#[no_mangle]
pub unsafe extern "C" fn x3f_get_wb(x3f: *mut x3f_t) -> *mut c_char {
    let mut wb_code: u32 = 0;
    let key = c"WhiteBalance";
    if unsafe { x3f_get_camf_unsigned(x3f, key.as_ptr() as *mut c_char, &mut wb_code) } != 0 {
        // Quattro: WhiteBalance is a numeric code mapping to a string.
        // Strings live in static storage; cast the byte literal to *mut c_char
        // because the C signature returns *mut. Callers do not free.
        let s: &'static [u8] = match wb_code {
            1 => b"Auto\0",
            2 => b"Sunlight\0",
            3 => b"Shadow\0",
            4 => b"Overcast\0",
            5 => b"Incandescent\0",
            6 => b"Florescent\0",
            7 => b"Flash\0",
            8 => b"Custom\0",
            11 => b"ColorTemp\0",
            12 => b"AutoLSP\0",
            _ => b"Auto\0",
        };
        return s.as_ptr() as *mut c_char;
    }

    // Fall back to the header's pre-Quattro WB string. This is a fixed-size
    // array inside x3f_t.header, returned as a non-owning *mut c_char.
    let hdr_wb: *mut c_char = unsafe { (*x3f).header.white_balance.as_mut_ptr() };
    hdr_wb
}

#[no_mangle]
pub unsafe extern "C" fn x3f_get_camf_matrix_for_wb(
    x3f: *mut x3f_t,
    list: *mut c_char,
    wb: *mut c_char,
    dim0: c_int,
    dim1: c_int,
    matrix: *mut f64,
) -> c_int {
    let mut matrix_name: *mut c_char = ptr::null_mut();
    if unsafe { x3f_get_camf_property(x3f, list, wb, &mut matrix_name) } == 0 {
        // SD1 workaround: legacy "Daylight" preset is named "Sunlight" in
        // some firmware. Recurse with the alternate name.
        let daylight = c"Daylight";
        if unsafe { cstr_eq(wb, daylight.as_ptr()) } {
            let sunlight = c"Sunlight";
            return unsafe {
                x3f_get_camf_matrix_for_wb(
                    x3f,
                    list,
                    sunlight.as_ptr() as *mut c_char,
                    dim0,
                    dim1,
                    matrix,
                )
            };
        }
        return 0;
    }
    unsafe {
        x3f_get_camf_matrix(
            x3f,
            matrix_name,
            dim0,
            dim1,
            0,
            matrix_type_t_M_FLOAT,
            matrix as *mut c_void,
        )
    }
}

unsafe fn is_true_engine(x3f: *mut x3f_t) -> bool {
    let mut names: *mut *mut c_char = ptr::null_mut();
    let mut values: *mut *mut c_char = ptr::null_mut();
    let mut num: c_uint = 0;
    let cc = c"WhiteBalanceColorCorrections";
    let dp1cc = c"DP1_WhiteBalanceColorCorrections";
    let gains = c"WhiteBalanceGains";
    let dp1gains = c"DP1_WhiteBalanceGains";
    let has_cc = unsafe {
        x3f_get_camf_property_list(
            x3f,
            cc.as_ptr() as *mut c_char,
            &mut names,
            &mut values,
            &mut num,
        ) != 0
            || x3f_get_camf_property_list(
                x3f,
                dp1cc.as_ptr() as *mut c_char,
                &mut names,
                &mut values,
                &mut num,
            ) != 0
    };
    let has_gains = unsafe {
        x3f_get_camf_property_list(
            x3f,
            gains.as_ptr() as *mut c_char,
            &mut names,
            &mut values,
            &mut num,
        ) != 0
            || x3f_get_camf_property_list(
                x3f,
                dp1gains.as_ptr() as *mut c_char,
                &mut names,
                &mut values,
                &mut num,
            ) != 0
    };
    has_cc && has_gains
}

#[no_mangle]
pub unsafe extern "C" fn x3f_get_max_raw(x3f: *mut x3f_t, max_raw: *mut u32) -> c_int {
    let mut image_depth: u32 = 0;
    let depth_key = c"ImageDepth";
    if unsafe { x3f_get_camf_unsigned(x3f, depth_key.as_ptr() as *mut c_char, &mut image_depth) }
        != 0
    {
        let max = (1u32 << image_depth) - 1;
        unsafe {
            *max_raw.add(0) = max;
            *max_raw.add(1) = max;
            *max_raw.add(2) = max;
        }
        return 1;
    }

    // RawSaturationLevel for TRUE engine, SaturationLevel for pre-TRUE.
    let key = if unsafe { is_true_engine(x3f) } {
        c"RawSaturationLevel"
    } else {
        c"SaturationLevel"
    };
    unsafe { x3f_get_camf_signed_vector(x3f, key.as_ptr() as *mut c_char, max_raw as *mut i32) }
}

// Anchor every #[no_mangle] symbol so cross-crate dead-code elimination
// cannot strip them before the legacy C callers (process.c, image.c, etc.)
// are linked. Each fn pointer is its own static — same pattern as
// quattro.rs and entropy.rs.
#[used]
static _A1: unsafe extern "C" fn(*mut x3f_t, *mut c_char, *mut *mut c_char) -> c_int =
    x3f_get_camf_text;
#[used]
static _A2: unsafe extern "C" fn(
    *mut x3f_t,
    *mut c_char,
    *mut c_int,
    *mut c_int,
    *mut c_int,
    matrix_type_t,
    *mut *mut c_void,
) -> c_int = x3f_get_camf_matrix_var;
#[used]
static _A3: unsafe extern "C" fn(
    *mut x3f_t,
    *mut c_char,
    c_int,
    c_int,
    c_int,
    matrix_type_t,
    *mut c_void,
) -> c_int = x3f_get_camf_matrix;
#[used]
static _A4: unsafe extern "C" fn(*mut x3f_t, *mut c_char, *mut f64) -> c_int = x3f_get_camf_float;
#[used]
static _A5: unsafe extern "C" fn(*mut x3f_t, *mut c_char, *mut f64) -> c_int =
    x3f_get_camf_float_vector;
#[used]
static _A6: unsafe extern "C" fn(*mut x3f_t, *mut c_char, *mut u32) -> c_int =
    x3f_get_camf_unsigned;
#[used]
static _A7: unsafe extern "C" fn(*mut x3f_t, *mut c_char, *mut i32) -> c_int = x3f_get_camf_signed;
#[used]
static _A8: unsafe extern "C" fn(*mut x3f_t, *mut c_char, *mut i32) -> c_int =
    x3f_get_camf_signed_vector;
#[used]
static _A9: unsafe extern "C" fn(
    *mut x3f_t,
    *mut c_char,
    *mut *mut *mut c_char,
    *mut *mut *mut c_char,
    *mut c_uint,
) -> c_int = x3f_get_camf_property_list;
#[used]
static _A10: unsafe extern "C" fn(*mut x3f_t, *mut c_char, *mut c_char, *mut *mut c_char) -> c_int =
    x3f_get_camf_property;
#[used]
static _A11: unsafe extern "C" fn(*mut x3f_t, *mut c_char, *mut *mut c_char) -> c_int =
    x3f_get_prop_entry;
#[used]
static _A12: unsafe extern "C" fn(*mut x3f_t) -> *mut c_char = x3f_get_wb;
#[used]
static _A13: unsafe extern "C" fn(
    *mut x3f_t,
    *mut c_char,
    *mut c_char,
    c_int,
    c_int,
    *mut f64,
) -> c_int = x3f_get_camf_matrix_for_wb;
#[used]
static _A14: unsafe extern "C" fn(*mut x3f_t, *mut u32) -> c_int = x3f_get_max_raw;
