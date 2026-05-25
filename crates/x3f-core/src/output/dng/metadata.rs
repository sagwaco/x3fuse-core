//! Thin wrappers over the C-side metadata accessors used by the DNG writer.
//!
//! These all return `Option`: `None` means the entry isn't present in the
//! file (which the legacy C code handled by either silently skipping the
//! corresponding tag or, in a few cases, hard-failing the conversion). The
//! caller decides which behaviour to apply.
//!
//! The 3×3 matrix helpers are exposed alongside because they're used in the
//! same call-chains; a future native port of `x3f_matrix.c` can replace
//! these without touching the writer.

use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::ptr;

use x3f_sys as sys;

use crate::Reader;

fn cstr(s: &str) -> CString {
    // All call-sites pass static strings; a NUL byte would be a programmer
    // bug, not user input.
    CString::new(s).expect("camf entry name contains NUL")
}

impl Reader {
    pub(crate) fn dng_camf_text(&self, name: &str) -> Option<String> {
        let cname = cstr(name);
        let mut out: *mut c_char = ptr::null_mut();
        // SAFETY: x3f is valid for self's lifetime; cname outlives the call.
        let ok = unsafe {
            sys::x3f_get_camf_text(self.x3f.as_ptr(), cname.as_ptr() as *mut _, &mut out)
        };
        if ok == 0 || out.is_null() {
            return None;
        }
        // SAFETY: out points to a NUL-terminated string owned by the
        // x3f_t — we copy it into Rust ownership and leave the original
        // intact (no free).
        let s = unsafe { CStr::from_ptr(out) };
        Some(s.to_string_lossy().into_owned())
    }

    pub(crate) fn dng_camf_float(&self, name: &str) -> Option<f64> {
        let cname = cstr(name);
        let mut val = 0.0_f64;
        let ok = unsafe {
            sys::x3f_get_camf_float(self.x3f.as_ptr(), cname.as_ptr() as *mut _, &mut val)
        };
        (ok != 0).then_some(val)
    }

    /// Read a 3×3 `M_FLOAT` CAMF matrix into a flat row-major array.
    pub(crate) fn dng_camf_matrix_3x3(&self, name: &str) -> Option<[f64; 9]> {
        let cname = cstr(name);
        let mut buf = [0.0_f64; 9];
        // x3f_get_camf_matrix uses M_FLOAT==3 per x3f_io.h; we just want the
        // 9 doubles back. The C signature takes void* + a type tag; we hard-
        // code M_FLOAT here.
        let ok = unsafe {
            sys::x3f_get_camf_matrix(
                self.x3f.as_ptr(),
                cname.as_ptr() as *mut _,
                3,
                3,
                0,
                sys::matrix_type_t_M_FLOAT,
                buf.as_mut_ptr() as *mut _,
            )
        };
        (ok != 0).then_some(buf)
    }

    /// Read a `MultiAxisTable_<mode>` CAMF entry: a `float[2][5][21]` table
    /// where group 0 = hue shift in degrees, group 1 = saturation
    /// multiplier, x = 21 hue bins, y = 5 (currently uniform — Sigma
    /// reserves the axis but every shipping camera writes identical rows).
    /// Returned slice is row-major in the declared order
    /// `[group][y][x]`, so e.g. group-0 row-0 starts at index 0,
    /// group-1 row-0 starts at 5*21 = 105.
    pub(crate) fn dng_camf_multi_axis_table(&self, name: &str) -> Option<[f64; 210]> {
        let cname = cstr(name);
        let mut buf = [0.0_f64; 210];
        let ok = unsafe {
            sys::x3f_get_camf_matrix(
                self.x3f.as_ptr(),
                cname.as_ptr() as *mut _,
                2,
                5,
                21,
                sys::matrix_type_t_M_FLOAT,
                buf.as_mut_ptr() as *mut _,
            )
        };
        (ok != 0).then_some(buf)
    }

    /// `(active_area, image_dims_used_for_rescale)` style from the C version
    /// — i.e. the rect translated into Adobe DNG conventions: `[top, left,
    /// bottom, right]` with bottom/right being one past the last
    /// row/column. The C code's `get_camf_rect_as_dngrect` does the same
    /// transform; we replicate it here.
    pub(crate) fn dng_active_area(&self, image: &crate::Image) -> Option<[u32; 4]> {
        let cname = cstr("ActiveImageArea");
        let mut sigma_rect = [0u32; 4];
        let mut area = sys::x3f_area16_t {
            data: image.data.as_ptr() as *mut u16,
            buf: ptr::null_mut(),
            rows: image.rows,
            columns: image.columns,
            channels: image.channels,
            row_stride: image.row_stride,
        };
        let ok = unsafe {
            sys::x3f_get_camf_rect(
                self.x3f.as_ptr(),
                cname.as_ptr() as *mut _,
                &mut area,
                1,
                sigma_rect.as_mut_ptr(),
            )
        };
        if ok == 0 {
            return None;
        }
        // Sigma's [top, left, bottom, right] (inclusive) → Adobe's [top,
        // left, bottom+1, right+1] with order swap to the native
        // [top, left, bottom, right]. Source: src/x3f_output_dng.c:36.
        Some([
            sigma_rect[1],
            sigma_rect[0],
            sigma_rect[3] + 1,
            sigma_rect[2] + 1,
        ])
    }

    /// White-balance gain triplet for the given preset (or the file's
    /// default WB if `wb` is None).
    pub(crate) fn dng_gain(&self, wb: Option<&str>) -> Option<[f64; 3]> {
        let cwb = wb.map(cstr);
        let wb_ptr = cwb
            .as_ref()
            .map(|s| s.as_ptr() as *mut _)
            .unwrap_or(ptr::null_mut());
        let mut g = [0.0_f64; 3];
        let ok = unsafe { sys::x3f_get_gain(self.x3f.as_ptr(), wb_ptr, g.as_mut_ptr()) };
        (ok != 0).then_some(g)
    }

    /// Camera-native (BMT) → CIE XYZ matrix for a given white-balance.
    pub(crate) fn dng_bmt_to_xyz(&self, wb: Option<&str>) -> Option<[f64; 9]> {
        let cwb = wb.map(cstr);
        let wb_ptr = cwb
            .as_ref()
            .map(|s| s.as_ptr() as *mut _)
            .unwrap_or(ptr::null_mut());
        let mut m = [0.0_f64; 9];
        let ok = unsafe { sys::x3f_get_bmt_to_xyz(self.x3f.as_ptr(), wb_ptr, m.as_mut_ptr()) };
        (ok != 0).then_some(m)
    }

    /// File's recorded white-balance preset name (ASCII, owned by C — we
    /// copy).
    pub(crate) fn dng_default_wb(&self) -> String {
        // SAFETY: x3f is valid; x3f_get_wb returns a pointer into the
        // parsed metadata, alive for the Reader's lifetime.
        let p = unsafe { sys::x3f_get_wb(self.x3f.as_ptr()) };
        if p.is_null() {
            return String::new();
        }
        unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
    }

    /// Read a `PROP` table entry by ASCII key. Returns `None` if the file
    /// has no PROP section (Quattro and later) or the key isn't present.
    pub(crate) fn dng_prop(&self, name: &str) -> Option<String> {
        let cname = cstr(name);
        let mut out: *mut c_char = ptr::null_mut();
        // SAFETY: x3f valid for self's lifetime; cname outlives the call.
        let ok = unsafe {
            sys::x3f_get_prop_entry(self.x3f.as_ptr(), cname.as_ptr() as *mut _, &mut out)
        };
        if ok == 0 || out.is_null() {
            return None;
        }
        // SAFETY: out points into the parsed PROP table; we copy.
        Some(
            unsafe { CStr::from_ptr(out) }
                .to_string_lossy()
                .into_owned(),
        )
    }

    /// TIFF/DNG `Orientation` tag value derived from `PROP[ROTATION]`,
    /// or `None` if the file has no PROP table (Quattro). Sigma stores
    /// the same "clockwise degrees needed to display" semantic the JPEG
    /// EXIF `Orientation` uses, so the mapping is direct: 0→1, 90→6,
    /// 180→3, 270→8. Callers wanting a JPEG-EXIF fallback should go
    /// through `CaptureMetadata` (see `output::dng::exif`).
    pub(crate) fn prop_orientation(&self) -> Option<u16> {
        let s = self.dng_prop("ROTATION")?;
        let rot = s.trim().parse::<i32>().ok()?;
        Some(match rot.rem_euclid(360) {
            90 => 6,
            180 => 3,
            270 => 8,
            _ => 1,
        })
    }

    /// `header.color_mode` (e.g. `"FCBlue"`). Used to pick the default
    /// camera profile. May be empty for older files.
    pub(crate) fn dng_color_mode(&self) -> String {
        // The header field is a fixed-size char array; treat it as an
        // ASCII C-string and copy.
        // SAFETY: x3f is valid; the field is owned by the parsed header.
        let buf = unsafe { (*self.x3f.as_ptr()).header.color_mode.as_ptr() };
        if buf.is_null() {
            return String::new();
        }
        unsafe { CStr::from_ptr(buf) }
            .to_string_lossy()
            .into_owned()
    }
}

// --- 3×3 matrix helpers -------------------------------------------------
// All are thin wrappers over the C library so the Rust DNG writer is
// numerically identical to the legacy output. A native port can land in M6
// without touching this code.

pub(crate) fn mat3_inverse(a: &[f64; 9]) -> [f64; 9] {
    let mut out = [0.0_f64; 9];
    // SAFETY: stack pointers, fixed-size arrays.
    unsafe { sys::x3f_3x3_inverse(a.as_ptr() as *mut _, out.as_mut_ptr()) };
    out
}

pub(crate) fn mat3_mul(a: &[f64; 9], b: &[f64; 9]) -> [f64; 9] {
    let mut out = [0.0_f64; 9];
    unsafe { sys::x3f_3x3_3x3_mul(a.as_ptr() as *mut _, b.as_ptr() as *mut _, out.as_mut_ptr()) };
    out
}

pub(crate) fn vec3_invert(a: &[f64; 3]) -> [f64; 3] {
    let mut out = [0.0_f64; 3];
    unsafe { sys::x3f_3x1_invert(a.as_ptr() as *mut _, out.as_mut_ptr()) };
    out
}

pub(crate) fn mat3_diag(a: &[f64; 3]) -> [f64; 9] {
    let mut out = [0.0_f64; 9];
    unsafe { sys::x3f_3x3_diag(a.as_ptr() as *mut _, out.as_mut_ptr()) };
    out
}

pub(crate) fn mat3_ones() -> [f64; 9] {
    let mut out = [0.0_f64; 9];
    unsafe { sys::x3f_3x3_ones(out.as_mut_ptr()) };
    out
}

pub(crate) fn bradford_d65_to_d50() -> [f64; 9] {
    let mut out = [0.0_f64; 9];
    unsafe { sys::x3f_Bradford_D65_to_D50(out.as_mut_ptr()) };
    out
}

pub(crate) fn adobe_rgb_to_xyz() -> [f64; 9] {
    let mut out = [0.0_f64; 9];
    unsafe { sys::x3f_AdobeRGB_to_XYZ(out.as_mut_ptr()) };
    out
}

/// 9 doubles → 9 floats (DNG matrix tags are SRATIONAL or FLOAT; we use
/// FLOAT to match the C path's `TIFFSetField(..., 9, color_matrix1)`).
pub(crate) fn mat3_to_f32(a: &[f64; 9]) -> [f32; 9] {
    [
        a[0] as f32,
        a[1] as f32,
        a[2] as f32,
        a[3] as f32,
        a[4] as f32,
        a[5] as f32,
        a[6] as f32,
        a[7] as f32,
        a[8] as f32,
    ]
}

pub(crate) fn vec3_to_f32(a: &[f64; 3]) -> [f32; 3] {
    [a[0] as f32, a[1] as f32, a[2] as f32]
}
