//! M6a — native Rust port of `src/x3f_matrix.c`.
//!
//! Pure 3×3 / 3×1 matrix math, color-space conversion matrices, and
//! gamma/sRGB LUT helpers. No state, no allocation. Input/output buffers
//! are caller-owned `*mut f64`/`*mut u16` per the C ABI.
//!
//! Each function is exported via `#[no_mangle] extern "C"` under the
//! existing C name. The C source is removed from `build.rs`'s cc-rs
//! source list, the bindgen forward declarations are blocklisted, and
//! `lib.rs` re-exports the Rust definitions in the same `x3f_sys::*`
//! paths so call sites in `x3f_process.c`, `x3f-core/src/output/dng/`,
//! and the rest of the workspace do not change.
//!
//! Print routines (`x3f_3x1_print`, `x3f_3x3_print`) keep the original
//! `%10g` format string and forward through C's `x3f_printf` so the
//! verbosity gate, level prefix, and `x3f_printf_callback` redirection
//! all remain in effect.
#![allow(clippy::missing_safety_doc)]

use crate::{x3f_printf, x3f_verbosity_t};
// libc compat — see `sysabi.rs`. matrix.rs uses `libc::printf` / `fprintf`
// for byte-identical %10g format output; on wasm32 those are unsupported
// (variadic), so we cfg-gate the print routines below rather than route
// through this alias. The alias still helps for any non-print libc use.
use crate::sysabi as libc;

#[inline]
unsafe fn r3<'a>(a: *const f64) -> &'a [f64; 3] {
    unsafe { &*(a as *const [f64; 3]) }
}

#[inline]
unsafe fn r3_mut<'a>(a: *mut f64) -> &'a mut [f64; 3] {
    unsafe { &mut *(a as *mut [f64; 3]) }
}

#[inline]
unsafe fn r9<'a>(a: *const f64) -> &'a [f64; 9] {
    unsafe { &*(a as *const [f64; 9]) }
}

#[inline]
unsafe fn r9_mut<'a>(a: *mut f64) -> &'a mut [f64; 9] {
    unsafe { &mut *(a as *mut [f64; 9]) }
}

#[no_mangle]
pub unsafe extern "C" fn x3f_3x1_invert(a: *mut f64, ainv: *mut f64) {
    let a = unsafe { r3(a) };
    let o = unsafe { r3_mut(ainv) };
    o[0] = 1.0 / a[0];
    o[1] = 1.0 / a[1];
    o[2] = 1.0 / a[2];
}

#[no_mangle]
pub unsafe extern "C" fn x3f_3x1_comp_mul(a: *mut f64, b: *mut f64, c: *mut f64) {
    let a = unsafe { r3(a) };
    let b = unsafe { r3(b) };
    let c = unsafe { r3_mut(c) };
    c[0] = a[0] * b[0];
    c[1] = a[1] * b[1];
    c[2] = a[2] * b[2];
}

#[no_mangle]
pub unsafe extern "C" fn x3f_scalar_3x1_mul(a: f64, b: *mut f64, c: *mut f64) {
    let b = unsafe { r3(b) };
    let c = unsafe { r3_mut(c) };
    c[0] = a * b[0];
    c[1] = a * b[1];
    c[2] = a * b[2];
}

#[no_mangle]
pub unsafe extern "C" fn x3f_scalar_3x3_mul(a: f64, b: *mut f64, c: *mut f64) {
    let b = unsafe { r9(b) };
    let c = unsafe { r9_mut(c) };
    for i in 0..9 {
        c[i] = a * b[i];
    }
}

#[no_mangle]
pub unsafe extern "C" fn x3f_3x3_3x1_mul(a: *mut f64, b: *mut f64, c: *mut f64) {
    let m = unsafe { r9(a) };
    let v = unsafe { r3(b) };
    let o = unsafe { r3_mut(c) };
    o[0] = m[0] * v[0] + m[1] * v[1] + m[2] * v[2];
    o[1] = m[3] * v[0] + m[4] * v[1] + m[5] * v[2];
    o[2] = m[6] * v[0] + m[7] * v[1] + m[8] * v[2];
}

#[no_mangle]
pub unsafe extern "C" fn x3f_3x3_3x3_mul(a: *mut f64, b: *mut f64, c: *mut f64) {
    let a = unsafe { r9(a) };
    let b = unsafe { r9(b) };
    let c = unsafe { r9_mut(c) };
    c[0] = a[0] * b[0] + a[1] * b[3] + a[2] * b[6];
    c[1] = a[0] * b[1] + a[1] * b[4] + a[2] * b[7];
    c[2] = a[0] * b[2] + a[1] * b[5] + a[2] * b[8];

    c[3] = a[3] * b[0] + a[4] * b[3] + a[5] * b[6];
    c[4] = a[3] * b[1] + a[4] * b[4] + a[5] * b[7];
    c[5] = a[3] * b[2] + a[4] * b[5] + a[5] * b[8];

    c[6] = a[6] * b[0] + a[7] * b[3] + a[8] * b[6];
    c[7] = a[6] * b[1] + a[7] * b[4] + a[8] * b[7];
    c[8] = a[6] * b[2] + a[7] * b[5] + a[8] * b[8];
}

#[no_mangle]
pub unsafe extern "C" fn x3f_3x3_inverse(a: *mut f64, ainv: *mut f64) {
    let m = unsafe { r9(a) };
    let cap_a = m[4] * m[8] - m[5] * m[7];
    let cap_b = -(m[3] * m[8] - m[5] * m[6]);
    let cap_c = m[3] * m[7] - m[4] * m[6];

    let cap_d = -(m[1] * m[8] - m[2] * m[7]);
    let cap_e = m[0] * m[8] - m[2] * m[6];
    let cap_f = -(m[0] * m[7] - m[1] * m[6]);

    let cap_g = m[1] * m[5] - m[2] * m[4];
    let cap_h = -(m[0] * m[5] - m[2] * m[3]);
    let cap_i = m[0] * m[4] - m[1] * m[3];

    let det = m[0] * cap_a + m[1] * cap_b + m[2] * cap_c;

    let o = unsafe { r9_mut(ainv) };
    o[0] = cap_a / det;
    o[1] = cap_d / det;
    o[2] = cap_g / det;
    o[3] = cap_b / det;
    o[4] = cap_e / det;
    o[5] = cap_h / det;
    o[6] = cap_c / det;
    o[7] = cap_f / det;
    o[8] = cap_i / det;
}

#[no_mangle]
pub unsafe extern "C" fn x3f_3x3_diag(a: *mut f64, b: *mut f64) {
    let a = unsafe { r3(a) };
    let b = unsafe { r9_mut(b) };
    b[0] = a[0];
    b[1] = 0.0;
    b[2] = 0.0;
    b[3] = 0.0;
    b[4] = a[1];
    b[5] = 0.0;
    b[6] = 0.0;
    b[7] = 0.0;
    b[8] = a[2];
}

#[no_mangle]
pub unsafe extern "C" fn x3f_3x3_identity(a: *mut f64) {
    let a = unsafe { r9_mut(a) };
    a[0] = 1.0;
    a[1] = 0.0;
    a[2] = 0.0;
    a[3] = 0.0;
    a[4] = 1.0;
    a[5] = 0.0;
    a[6] = 0.0;
    a[7] = 0.0;
    a[8] = 1.0;
}

#[no_mangle]
pub unsafe extern "C" fn x3f_3x3_ones(a: *mut f64) {
    let a = unsafe { r9_mut(a) };
    for v in a.iter_mut() {
        *v = 1.0;
    }
}

#[no_mangle]
pub unsafe extern "C" fn x3f_3x1_print(level: x3f_verbosity_t, a: *mut f64) {
    let a = unsafe { r3(a) };
    unsafe {
        x3f_printf(level, c"%10g\n".as_ptr(), a[0]);
        x3f_printf(level, c"%10g\n".as_ptr(), a[1]);
        x3f_printf(level, c"%10g\n".as_ptr(), a[2]);
    }
}

#[no_mangle]
pub unsafe extern "C" fn x3f_3x3_print(level: x3f_verbosity_t, a: *mut f64) {
    let a = unsafe { r9(a) };
    unsafe {
        x3f_printf(level, c"%10g %10g %10g\n".as_ptr(), a[0], a[1], a[2]);
        x3f_printf(level, c"%10g %10g %10g\n".as_ptr(), a[3], a[4], a[5]);
        x3f_printf(level, c"%10g %10g %10g\n".as_ptr(), a[6], a[7], a[8]);
    }
}

#[no_mangle]
pub unsafe extern "C" fn x3f_XYZ_to_ProPhotoRGB(a: *mut f64) {
    let a = unsafe { r9_mut(a) };
    a[0] = 1.3460;
    a[1] = -0.2556;
    a[2] = -0.0511;
    a[3] = -0.5446;
    a[4] = 1.5082;
    a[5] = 0.0205;
    a[6] = 0.0000;
    a[7] = 0.0000;
    a[8] = 1.2123;
}

#[no_mangle]
pub unsafe extern "C" fn x3f_ProPhotoRGB_to_XYZ(a: *mut f64) {
    let a = unsafe { r9_mut(a) };
    a[0] = 0.7977;
    a[1] = 0.1352;
    a[2] = 0.0313;
    a[3] = 0.2880;
    a[4] = 0.7119;
    a[5] = 0.0001;
    a[6] = 0.0000;
    a[7] = 0.0000;
    a[8] = 0.8249;
}

#[no_mangle]
pub unsafe extern "C" fn x3f_XYZ_to_AdobeRGB(a: *mut f64) {
    let a = unsafe { r9_mut(a) };
    a[0] = 2.04159;
    a[1] = -0.56501;
    a[2] = -0.34473;
    a[3] = -0.96924;
    a[4] = 1.87597;
    a[5] = 0.04156;
    a[6] = 0.01344;
    a[7] = -0.11836;
    a[8] = 1.01517;
}

#[no_mangle]
pub unsafe extern "C" fn x3f_AdobeRGB_to_XYZ(a: *mut f64) {
    let a = unsafe { r9_mut(a) };
    a[0] = 0.57667;
    a[1] = 0.18556;
    a[2] = 0.18823;
    a[3] = 0.29737;
    a[4] = 0.62736;
    a[5] = 0.07529;
    a[6] = 0.02703;
    a[7] = 0.07069;
    a[8] = 0.99134;
}

#[no_mangle]
pub unsafe extern "C" fn x3f_XYZ_to_sRGB(a: *mut f64) {
    let a = unsafe { r9_mut(a) };
    a[0] = 3.2406;
    a[1] = -1.5372;
    a[2] = -0.4986;
    a[3] = -0.9689;
    a[4] = 1.8758;
    a[5] = 0.0415;
    a[6] = 0.0557;
    a[7] = -0.2040;
    a[8] = 1.0570;
}

#[no_mangle]
pub unsafe extern "C" fn x3f_sRGB_to_XYZ(a: *mut f64) {
    let a = unsafe { r9_mut(a) };
    a[0] = 0.4124;
    a[1] = 0.3576;
    a[2] = 0.1805;
    a[3] = 0.2126;
    a[4] = 0.7152;
    a[5] = 0.0722;
    a[6] = 0.0193;
    a[7] = 0.1192;
    a[8] = 0.9505;
}

#[no_mangle]
pub unsafe extern "C" fn x3f_CIERGB_to_XYZ(a: *mut f64) {
    let a = unsafe { r9_mut(a) };
    a[0] = 0.49;
    a[1] = 0.31;
    a[2] = 0.20;
    a[3] = 0.17697;
    a[4] = 0.81240;
    a[5] = 0.01063;
    a[6] = 0.00;
    a[7] = 0.01;
    a[8] = 0.99;
}

#[no_mangle]
pub unsafe extern "C" fn x3f_Bradford_D50_to_D65(a: *mut f64) {
    let a = unsafe { r9_mut(a) };
    a[0] = 0.9555766;
    a[1] = -0.0230393;
    a[2] = 0.0631636;
    a[3] = -0.0282895;
    a[4] = 1.0099416;
    a[5] = 0.0210077;
    a[6] = 0.0122982;
    a[7] = -0.0204830;
    a[8] = 1.3299098;
}

#[no_mangle]
pub unsafe extern "C" fn x3f_Bradford_D65_to_D50(a: *mut f64) {
    let a = unsafe { r9_mut(a) };
    a[0] = 1.0478112;
    a[1] = 0.0228866;
    a[2] = -0.0501270;
    a[3] = 0.0295424;
    a[4] = 0.9904844;
    a[5] = -0.0170491;
    a[6] = -0.0092345;
    a[7] = 0.0150436;
    a[8] = 0.7521316;
}

#[no_mangle]
pub unsafe extern "C" fn x3f_sRGB_LUT(lut: *mut f64, size: libc::c_int, max: u16) {
    let n = size as usize;
    let lut = unsafe { std::slice::from_raw_parts_mut(lut, n) };
    let a = 0.055_f64;
    let thres = 0.0031308_f64;
    let max_f = max as f64;
    for i in 0..n {
        let lin = (i as f64) / ((size - 1) as f64);
        let mut srgb = if lin <= thres {
            12.92 * lin
        } else {
            (1.0 + a) * lin.powf(1.0 / 2.4) - a
        };
        srgb *= max_f;

        lut[i] = if srgb < 0.0 {
            0.0
        } else if srgb > max_f {
            max_f
        } else {
            srgb
        };
    }
}

#[no_mangle]
pub unsafe extern "C" fn x3f_gamma_LUT(lut: *mut f64, size: libc::c_int, max: u16, gamma: f64) {
    let n = size as usize;
    let lut = unsafe { std::slice::from_raw_parts_mut(lut, n) };
    let max_f = max as f64;
    for i in 0..n {
        let lin = (i as f64) / ((size - 1) as f64);
        let mut gam = lin.powf(1.0 / gamma);
        gam *= max_f;

        lut[i] = if gam < 0.0 {
            0.0
        } else if gam > max_f {
            max_f
        } else {
            gam
        };
    }
}

/// Cineon-style log tone curve, written into `lut[0..size]` over an input
/// range of `[0, 1]`. The forward (encoding) curve is
///
/// ```text
///     y = log(scale·x + 1) / log(scale + 1)
/// ```
///
/// which has `y(0) = 0`, `y(1) = 1`, a steep slope at zero (shadows lifted)
/// and a gentle slope at one (highlights softly compressed). `scale` is the
/// curve's "flatness" knob — larger values produce a flatter midtone, with
/// `scale = 12` chosen as the default because it sits close to the Cineon
/// film-print curve. `scale ≤ 0` is treated as a programmer error and
/// produces a linear identity LUT (so a divide-by-`log(1)` cannot happen).
#[no_mangle]
pub unsafe extern "C" fn x3f_cineon_log_LUT(
    lut: *mut f64,
    size: libc::c_int,
    max: u16,
    scale: f64,
) {
    let n = size as usize;
    let lut = unsafe { std::slice::from_raw_parts_mut(lut, n) };
    let max_f = max as f64;

    if !(scale > 0.0 && scale.is_finite()) {
        // Defensive fallback: degenerate scale → linear identity. Keeps
        // the writer output predictable if the env-var override produces
        // a nonsense value.
        for i in 0..n {
            let lin = (i as f64) / ((size - 1) as f64);
            lut[i] = (lin * max_f).clamp(0.0, max_f);
        }
        return;
    }

    let denom = (scale + 1.0).ln();
    for i in 0..n {
        let lin = (i as f64) / ((size - 1) as f64);
        let y = (scale * lin + 1.0).ln() / denom;
        let scaled = y * max_f;
        lut[i] = scaled.clamp(0.0, max_f);
    }
}

#[no_mangle]
pub unsafe extern "C" fn x3f_LUT_lookup(lut: *mut f64, size: libc::c_int, val: f64) -> u16 {
    let n = size as usize;
    let lut = unsafe { std::slice::from_raw_parts(lut, n) };
    let index = val * ((size - 1) as f64);
    let i = index.floor() as i64;
    let frac = index - (i as f64);

    if i < 0 {
        lut[0].round() as u16
    } else if i >= (size as i64 - 1) {
        lut[n - 1].round() as u16
    } else {
        let i = i as usize;
        (lut[i] + frac * (lut[i + 1] - lut[i])).round() as u16
    }
}

// Anchors so cross-crate dead-code elimination cannot strip the symbols
// before the C call sites in `x3f_process.c` (still C through M6a) see
// them. Without these, link-time pruning of an unused public Rust fn can
// hide the `#[no_mangle]` definition entirely.
#[used]
static _A_3X1_INVERT: unsafe extern "C" fn(*mut f64, *mut f64) = x3f_3x1_invert;
#[used]
static _A_3X1_COMP_MUL: unsafe extern "C" fn(*mut f64, *mut f64, *mut f64) = x3f_3x1_comp_mul;
#[used]
static _A_SCALAR_3X1_MUL: unsafe extern "C" fn(f64, *mut f64, *mut f64) = x3f_scalar_3x1_mul;
#[used]
static _A_SCALAR_3X3_MUL: unsafe extern "C" fn(f64, *mut f64, *mut f64) = x3f_scalar_3x3_mul;
#[used]
static _A_3X3_3X1_MUL: unsafe extern "C" fn(*mut f64, *mut f64, *mut f64) = x3f_3x3_3x1_mul;
#[used]
static _A_3X3_3X3_MUL: unsafe extern "C" fn(*mut f64, *mut f64, *mut f64) = x3f_3x3_3x3_mul;
#[used]
static _A_3X3_INVERSE: unsafe extern "C" fn(*mut f64, *mut f64) = x3f_3x3_inverse;
#[used]
static _A_3X3_DIAG: unsafe extern "C" fn(*mut f64, *mut f64) = x3f_3x3_diag;
#[used]
static _A_3X3_IDENTITY: unsafe extern "C" fn(*mut f64) = x3f_3x3_identity;
#[used]
static _A_3X3_ONES: unsafe extern "C" fn(*mut f64) = x3f_3x3_ones;
#[used]
static _A_3X1_PRINT: unsafe extern "C" fn(x3f_verbosity_t, *mut f64) = x3f_3x1_print;
#[used]
static _A_3X3_PRINT: unsafe extern "C" fn(x3f_verbosity_t, *mut f64) = x3f_3x3_print;
#[used]
static _A_XYZ_TO_PPRGB: unsafe extern "C" fn(*mut f64) = x3f_XYZ_to_ProPhotoRGB;
#[used]
static _A_PPRGB_TO_XYZ: unsafe extern "C" fn(*mut f64) = x3f_ProPhotoRGB_to_XYZ;
#[used]
static _A_XYZ_TO_ARGB: unsafe extern "C" fn(*mut f64) = x3f_XYZ_to_AdobeRGB;
#[used]
static _A_ARGB_TO_XYZ: unsafe extern "C" fn(*mut f64) = x3f_AdobeRGB_to_XYZ;
#[used]
static _A_XYZ_TO_SRGB: unsafe extern "C" fn(*mut f64) = x3f_XYZ_to_sRGB;
#[used]
static _A_SRGB_TO_XYZ: unsafe extern "C" fn(*mut f64) = x3f_sRGB_to_XYZ;
#[used]
static _A_CIERGB_TO_XYZ: unsafe extern "C" fn(*mut f64) = x3f_CIERGB_to_XYZ;
#[used]
static _A_BRADFORD_D50_TO_D65: unsafe extern "C" fn(*mut f64) = x3f_Bradford_D50_to_D65;
#[used]
static _A_BRADFORD_D65_TO_D50: unsafe extern "C" fn(*mut f64) = x3f_Bradford_D65_to_D50;
#[used]
static _A_SRGB_LUT: unsafe extern "C" fn(*mut f64, libc::c_int, u16) = x3f_sRGB_LUT;
#[used]
static _A_GAMMA_LUT: unsafe extern "C" fn(*mut f64, libc::c_int, u16, f64) = x3f_gamma_LUT;
#[used]
static _A_CINEON_LOG_LUT: unsafe extern "C" fn(*mut f64, libc::c_int, u16, f64) =
    x3f_cineon_log_LUT;
#[used]
static _A_LUT_LOOKUP: unsafe extern "C" fn(*mut f64, libc::c_int, f64) -> u16 = x3f_LUT_lookup;

#[cfg(test)]
mod tests {
    use super::*;

    // Mirrors the fixture in src/x3f_matrix_test.c.
    const A: [f64; 9] = [11.0, 12.0, 13.0, 21.0, 22.0, 23.0, 31.0, 32.0, 33.0];
    const B: [f64; 9] = [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0];
    const X: [f64; 3] = [1.0, 2.0, 3.0];

    fn assert_close_3x3(a: &[f64; 9], b: &[f64; 9], eps: f64) {
        for i in 0..9 {
            assert!(
                (a[i] - b[i]).abs() < eps,
                "idx {i}: {} vs {} (eps {eps})",
                a[i],
                b[i]
            );
        }
    }

    #[test]
    fn mul_3x3_3x3_with_swap_b_reverses_columns() {
        // B reverses column order, so A*B == reverse-columns(A).
        let mut a = A;
        let mut b = B;
        let mut c = [0.0f64; 9];
        unsafe { x3f_3x3_3x3_mul(a.as_mut_ptr(), b.as_mut_ptr(), c.as_mut_ptr()) };
        let expected = [13.0, 12.0, 11.0, 23.0, 22.0, 21.0, 33.0, 32.0, 31.0];
        assert_eq!(c, expected);
    }

    #[test]
    fn mul_3x3_3x1_matches_hand_calc() {
        let mut a = A;
        let mut x = X;
        let mut y = [0.0f64; 3];
        unsafe { x3f_3x3_3x1_mul(a.as_mut_ptr(), x.as_mut_ptr(), y.as_mut_ptr()) };
        // A·x = [11+24+39, 21+44+69, 31+64+99]
        assert_eq!(y, [74.0, 134.0, 194.0]);
    }

    #[test]
    fn diag_lifts_3x1_to_diagonal_3x3() {
        let mut x = X;
        let mut d = [9.9f64; 9];
        unsafe { x3f_3x3_diag(x.as_mut_ptr(), d.as_mut_ptr()) };
        assert_eq!(d, [1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0,]);
    }

    #[test]
    fn identity_3x3_is_identity() {
        let mut i = [9.9f64; 9];
        unsafe { x3f_3x3_identity(i.as_mut_ptr()) };
        assert_eq!(i, [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,]);
    }

    #[test]
    fn ones_3x3_is_all_ones() {
        let mut o = [0.0f64; 9];
        unsafe { x3f_3x3_ones(o.as_mut_ptr()) };
        assert_eq!(o, [1.0; 9]);
    }

    #[test]
    fn argb_inverse_round_trips_to_identity() {
        let mut argb = [0.0f64; 9];
        let mut inv = [0.0f64; 9];
        let mut i = [0.0f64; 9];
        unsafe {
            x3f_XYZ_to_AdobeRGB(argb.as_mut_ptr());
            x3f_3x3_inverse(argb.as_mut_ptr(), inv.as_mut_ptr());
            x3f_3x3_3x3_mul(inv.as_mut_ptr(), argb.as_mut_ptr(), i.as_mut_ptr());
        }
        let identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        assert_close_3x3(&i, &identity, 1e-3);
    }

    #[test]
    fn invert_3x1_is_componentwise_reciprocal() {
        let mut a = [2.0, 4.0, 8.0];
        let mut o = [0.0; 3];
        unsafe { x3f_3x1_invert(a.as_mut_ptr(), o.as_mut_ptr()) };
        assert_eq!(o, [0.5, 0.25, 0.125]);
    }

    #[test]
    fn comp_mul_3x1() {
        let mut a = [1.0, 2.0, 3.0];
        let mut b = [4.0, 5.0, 6.0];
        let mut c = [0.0; 3];
        unsafe { x3f_3x1_comp_mul(a.as_mut_ptr(), b.as_mut_ptr(), c.as_mut_ptr()) };
        assert_eq!(c, [4.0, 10.0, 18.0]);
    }

    #[test]
    fn scalar_3x3_mul() {
        let mut b = [1.0; 9];
        let mut c = [0.0; 9];
        unsafe { x3f_scalar_3x3_mul(2.5, b.as_mut_ptr(), c.as_mut_ptr()) };
        assert_eq!(c, [2.5; 9]);
    }

    #[test]
    fn srgb_lut_endpoints() {
        let mut lut = [0.0f64; 1024];
        unsafe { x3f_sRGB_LUT(lut.as_mut_ptr(), 1024, 0xFFFF) };
        // First sample: lin=0 → srgb=0.
        assert_eq!(lut[0], 0.0);
        // Last sample: lin=1 → srgb=1*max.
        assert!((lut[1023] - 65535.0).abs() < 1e-6);
    }

    #[test]
    fn gamma_lut_2_2_endpoints() {
        let mut lut = [0.0f64; 256];
        unsafe { x3f_gamma_LUT(lut.as_mut_ptr(), 256, 0xFFFF, 2.2) };
        assert_eq!(lut[0], 0.0);
        assert!((lut[255] - 65535.0).abs() < 1e-6);
    }

    #[test]
    fn cineon_log_lut_lifts_shadows_and_pulls_highlights() {
        // Forward curve: y = log(s·x + 1) / log(s + 1). With s=12,
        // y(0.05) ≈ 0.241, y(0.95) ≈ 0.974 — i.e. shadow values rise
        // far above the linear x=y reference, and highlight values
        // sit just below it. This is the "lifted shadows / pulled
        // highlights / flat midtones" property the cineon mode promises.
        let n = 1024;
        let mut lut = vec![0.0f64; n];
        unsafe { x3f_cineon_log_LUT(lut.as_mut_ptr(), n as libc::c_int, 0xFFFF, 12.0) };
        assert_eq!(lut[0], 0.0);
        assert!((lut[n - 1] - 65535.0).abs() < 1e-6);
        // Compare lut sample at 5% input vs. linear. With scale=12 the
        // analytical gain is log(1.6)/log(13)/0.05 ≈ 3.66×, so checking
        // for >= 3× confirms the shadow lift while staying robust to
        // small index-quantization drift.
        let i_low = (0.05 * (n - 1) as f64) as usize;
        let linear_low: f64 = 0.05 * 65535.0;
        assert!(
            lut[i_low] > linear_low * 3.0,
            "shadows not lifted enough: lut[5%] = {} vs linear {linear_low}",
            lut[i_low],
        );
        // And at 95%, log output sits *above* the linear reference but
        // close to the max — that's the "compression" shape (gentle
        // shoulder, not hard rolloff).
        let i_hi = (0.95 * (n - 1) as f64) as usize;
        let linear_hi: f64 = 0.95 * 65535.0;
        assert!(
            lut[i_hi] > linear_hi && lut[i_hi] < 65535.0,
            "highlights not pulled correctly: lut[95%] = {} vs linear {linear_hi}",
            lut[i_hi]
        );
        // Strict monotonicity preserves grading precision.
        for w in lut.windows(2) {
            assert!(w[1] >= w[0], "LUT non-monotonic at {} -> {}", w[0], w[1]);
        }
    }

    #[test]
    fn cineon_log_lut_falls_back_to_linear_for_bad_scale() {
        let n = 64;
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let mut lut = vec![0.0f64; n];
            unsafe { x3f_cineon_log_LUT(lut.as_mut_ptr(), n as libc::c_int, 0xFFFF, bad) };
            // Linear identity: lut[i] ≈ i/(n-1) * max
            for (i, v) in lut.iter().enumerate() {
                let expected = (i as f64) / ((n - 1) as f64) * 65535.0;
                assert!(
                    (v - expected).abs() < 1.0,
                    "bad-scale fallback drifted for scale={bad}: lut[{i}] = {v}"
                );
            }
        }
    }

    #[test]
    fn lut_lookup_endpoints_and_midpoint() {
        let mut lut = [0.0f64; 5];
        for (i, v) in lut.iter_mut().enumerate() {
            *v = (i as f64) * 1000.0;
        }
        // val=0 → lut[0]=0
        assert_eq!(unsafe { x3f_LUT_lookup(lut.as_mut_ptr(), 5, 0.0) }, 0);
        // val=1 → lut[4]=4000
        assert_eq!(unsafe { x3f_LUT_lookup(lut.as_mut_ptr(), 5, 1.0) }, 4000);
        // val=0.25 → index=1.0 → exactly lut[1]=1000
        assert_eq!(unsafe { x3f_LUT_lookup(lut.as_mut_ptr(), 5, 0.25) }, 1000);
        // val=0.125 → index=0.5 → lut[0] + 0.5*(lut[1]-lut[0])=500
        assert_eq!(unsafe { x3f_LUT_lookup(lut.as_mut_ptr(), 5, 0.125) }, 500);
        // val outside [0,1] clamps.
        assert_eq!(unsafe { x3f_LUT_lookup(lut.as_mut_ptr(), 5, -0.5) }, 0);
        assert_eq!(unsafe { x3f_LUT_lookup(lut.as_mut_ptr(), 5, 2.0) }, 4000);
    }
}
