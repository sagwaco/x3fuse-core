//! M6e — incremental native Rust port of `src/x3f_process.c`.
//!
//! `x3f_process.c` is the largest module in the legacy converter
//! (~2300 LOC) and houses the entire processing pipeline: black/white
//! levels, color matrix and gamma LUT, bad-pixel interpolation, the
//! highlight-recovery code (chroma LUT, sat map, RepairPix, matrix-
//! pathology gate) the user is actively researching, and the
//! `convert_data` hot loop. We port it in small phases so each step is
//! independently verifiable against the tier-2 / tier-3 corpus.
//!
//! ## Phase status
//! - **M6e1 — color math accessors** (this file). `x3f_get_gain`,
//!   `x3f_get_bmt_to_xyz`, `x3f_get_raw_to_xyz`, plus the internal
//!   `get_raw_neutral` helper. Read white-balance / color-correction
//!   matrices from CAMF and combine them with the matrix helpers. No
//!   pixel data is touched.
//! - **M6e2 — black-level + intermediate-range stats** (this file).
//!   `sum_area`, `sum_area_sqdev`, `get_black_level`,
//!   `get_max_intermediate`, `get_intermediate_bias`. Statistics over
//!   the DarkShield CAMF rects and the per-aperture-clamped
//!   intermediate-buffer range. The three "external" helpers
//!   (`get_black_level`, `get_max_intermediate`, `get_intermediate_bias`)
//!   are exported under their original names so the still-C
//!   `preprocess_data` keeps calling them via `extern` declarations
//!   in `src/x3f_process.c`.
//! - Later phases will pull in bad-pixel interpolation, highlight
//!   recovery, and `convert_data`.
#![allow(clippy::missing_safety_doc)]

use std::ffi::CStr;
use std::ptr;

use crate::*;
// libc compat — see `sysabi.rs`.
use crate::sysabi as libc;

// ----------------------------------------------------------------------
// CAMERAID constants — bindgen drops them because `src/x3f_io.h` uses
// `(uint32_t)<n>` cast syntax for the values, which bindgen treats as
// expressions rather than integer constants. Hardcode here.
// ----------------------------------------------------------------------
const X3F_CAMERAID_SDQ: u32 = 40;
const X3F_CAMERAID_SDQH: u32 = 41;

// M7c — Send+Sync wrapper for read-only `*const u16` slices that need
// to be captured by parallel rayon closures (rayon requires the closure
// to be `Sync`, raw pointers aren't auto-Sync). The wrapped pointer is
// always shared read-only across threads in our use sites.
//
// The pointer is exposed via an `as_ptr()` method rather than a public
// field so Rust 2021's disjoint-capture rule captures the whole
// (Sync) struct rather than the bare `*const u16` field.
#[derive(Copy, Clone)]
struct SyncConstU16Ptr(*const u16);
unsafe impl Send for SyncConstU16Ptr {}
unsafe impl Sync for SyncConstU16Ptr {}
impl SyncConstU16Ptr {
    #[inline(always)]
    fn as_ptr(self) -> *const u16 {
        self.0
    }
}

// ----------------------------------------------------------------------
// INTERMEDIATE_* constants from the C source. The denoise pipeline
// rescales by 4× internally so the intermediate buffer is 14-bit.
// ----------------------------------------------------------------------
const INTERMEDIATE_DEPTH: u32 = 14;
const INTERMEDIATE_UNIT: u32 = (1u32 << INTERMEDIATE_DEPTH) - 1;
const INTERMEDIATE_BIAS_FACTOR: f64 = 4.0;

const D65_XYZ: [f64; 3] = [0.95047, 1.00000, 1.08883];

/// Compute the raw-space neutral that corresponds to D65 white through
/// the given `raw_to_xyz` matrix.
unsafe fn get_raw_neutral(raw_to_xyz: *mut f64, raw_neutral: *mut f64) {
    let mut xyz_to_raw = [0.0_f64; 9];
    let mut d65 = D65_XYZ;
    unsafe {
        x3f_3x3_inverse(raw_to_xyz, xyz_to_raw.as_mut_ptr());
        x3f_3x3_3x1_mul(xyz_to_raw.as_mut_ptr(), d65.as_mut_ptr(), raw_neutral);
    }
}

#[no_mangle]
pub unsafe extern "C" fn x3f_get_gain(
    x3f: *mut x3f_t,
    wb: *mut libc::c_char,
    gain: *mut f64,
) -> libc::c_int {
    let mut cam_to_xyz = [0.0_f64; 9];
    let mut wb_correction = [0.0_f64; 9];
    let mut gain_fact = [0.0_f64; 3];

    // Try the direct WhiteBalanceGains path first; fall back to
    // computing gain from the illuminant + correction matrices via
    // the raw-neutral inverse.
    //
    // The C source has a deliberately unusual structure here:
    //
    //   if (A() || B());
    //   else if (C() && D()) { ... }
    //   else return 0;
    //
    // The empty body on the first branch means: try A, if A fails try
    // B; either succeeds → fall through with gain populated. We mirror
    // that as a short-circuit OR.
    let direct = unsafe {
        x3f_get_camf_matrix_for_wb(x3f, c"WhiteBalanceGains".as_ptr() as *mut _, wb, 3, 0, gain)
            != 0
            || x3f_get_camf_matrix_for_wb(
                x3f,
                c"DP1_WhiteBalanceGains".as_ptr() as *mut _,
                wb,
                3,
                0,
                gain,
            ) != 0
    };

    if !direct {
        let illum = unsafe {
            x3f_get_camf_matrix_for_wb(
                x3f,
                c"WhiteBalanceIlluminants".as_ptr() as *mut _,
                wb,
                3,
                3,
                cam_to_xyz.as_mut_ptr(),
            ) != 0
        };
        let corr = unsafe {
            x3f_get_camf_matrix_for_wb(
                x3f,
                c"WhiteBalanceCorrections".as_ptr() as *mut _,
                wb,
                3,
                3,
                wb_correction.as_mut_ptr(),
            ) != 0
        };
        if !(illum && corr) {
            return 0;
        }

        let mut raw_to_xyz = [0.0_f64; 9];
        let mut raw_neutral = [0.0_f64; 3];
        unsafe {
            x3f_3x3_3x3_mul(
                wb_correction.as_mut_ptr(),
                cam_to_xyz.as_mut_ptr(),
                raw_to_xyz.as_mut_ptr(),
            );
            get_raw_neutral(raw_to_xyz.as_mut_ptr(), raw_neutral.as_mut_ptr());
            x3f_3x1_invert(raw_neutral.as_mut_ptr(), gain);
        }
    }

    // Optional adjustment factors — each multiplied into gain in turn,
    // matching the C call order.
    if unsafe {
        x3f_get_camf_float_vector(
            x3f,
            c"SensorAdjustmentGainFact".as_ptr() as *mut _,
            gain_fact.as_mut_ptr(),
        )
    } != 0
    {
        unsafe { x3f_3x1_comp_mul(gain_fact.as_mut_ptr(), gain, gain) };
    }
    if unsafe {
        x3f_get_camf_float_vector(
            x3f,
            c"TempGainFact".as_ptr() as *mut _,
            gain_fact.as_mut_ptr(),
        )
    } != 0
    {
        unsafe { x3f_3x1_comp_mul(gain_fact.as_mut_ptr(), gain, gain) };
    }
    if unsafe {
        x3f_get_camf_float_vector(
            x3f,
            c"FNumberGainFact".as_ptr() as *mut _,
            gain_fact.as_mut_ptr(),
        )
    } != 0
    {
        unsafe { x3f_3x1_comp_mul(gain_fact.as_mut_ptr(), gain, gain) };
    }

    unsafe {
        x3f_printf(x3f_verbosity_t_DEBUG, c"gain\n".as_ptr());
        x3f_3x1_print(x3f_verbosity_t_DEBUG, gain);
    }

    1
}

#[no_mangle]
pub unsafe extern "C" fn x3f_get_bmt_to_xyz(
    x3f: *mut x3f_t,
    wb: *mut libc::c_char,
    bmt_to_xyz: *mut f64,
) -> libc::c_int {
    let mut cc_matrix = [0.0_f64; 9];
    let mut cam_to_xyz = [0.0_f64; 9];
    let mut wb_correction = [0.0_f64; 9];

    let cc_path = unsafe {
        x3f_get_camf_matrix_for_wb(
            x3f,
            c"WhiteBalanceColorCorrections".as_ptr() as *mut _,
            wb,
            3,
            3,
            cc_matrix.as_mut_ptr(),
        ) != 0
            || x3f_get_camf_matrix_for_wb(
                x3f,
                c"DP1_WhiteBalanceColorCorrections".as_ptr() as *mut _,
                wb,
                3,
                3,
                cc_matrix.as_mut_ptr(),
            ) != 0
    };

    if cc_path {
        let mut srgb_to_xyz = [0.0_f64; 9];
        unsafe {
            x3f_sRGB_to_XYZ(srgb_to_xyz.as_mut_ptr());
            x3f_3x3_3x3_mul(srgb_to_xyz.as_mut_ptr(), cc_matrix.as_mut_ptr(), bmt_to_xyz);
        }
    } else {
        let illum = unsafe {
            x3f_get_camf_matrix_for_wb(
                x3f,
                c"WhiteBalanceIlluminants".as_ptr() as *mut _,
                wb,
                3,
                3,
                cam_to_xyz.as_mut_ptr(),
            ) != 0
        };
        let corr = unsafe {
            x3f_get_camf_matrix_for_wb(
                x3f,
                c"WhiteBalanceCorrections".as_ptr() as *mut _,
                wb,
                3,
                3,
                wb_correction.as_mut_ptr(),
            ) != 0
        };
        if !(illum && corr) {
            return 0;
        }

        let mut raw_to_xyz = [0.0_f64; 9];
        let mut raw_neutral = [0.0_f64; 3];
        let mut raw_neutral_mat = [0.0_f64; 9];
        unsafe {
            x3f_3x3_3x3_mul(
                wb_correction.as_mut_ptr(),
                cam_to_xyz.as_mut_ptr(),
                raw_to_xyz.as_mut_ptr(),
            );
            get_raw_neutral(raw_to_xyz.as_mut_ptr(), raw_neutral.as_mut_ptr());
            x3f_3x3_diag(raw_neutral.as_mut_ptr(), raw_neutral_mat.as_mut_ptr());
            x3f_3x3_3x3_mul(
                raw_to_xyz.as_mut_ptr(),
                raw_neutral_mat.as_mut_ptr(),
                bmt_to_xyz,
            );
        }
    }

    unsafe {
        x3f_printf(x3f_verbosity_t_DEBUG, c"bmt_to_xyz\n".as_ptr());
        x3f_3x3_print(x3f_verbosity_t_DEBUG, bmt_to_xyz);
    }
    1
}

#[no_mangle]
pub unsafe extern "C" fn x3f_get_raw_to_xyz(
    x3f: *mut x3f_t,
    wb: *mut libc::c_char,
    raw_to_xyz: *mut f64,
) -> libc::c_int {
    let mut bmt_to_xyz = [0.0_f64; 9];
    // C declares `gain[9]` even though it only uses the first 3 slots
    // before passing through x3f_3x3_diag. Mirror the size to keep
    // any out-of-bounds reads identical to the C version. (None
    // exist; this is just being explicit.)
    let mut gain = [0.0_f64; 9];
    let mut gain_mat = [0.0_f64; 9];

    if unsafe { x3f_get_gain(x3f, wb, gain.as_mut_ptr()) } == 0 {
        return 0;
    }
    if unsafe { x3f_get_bmt_to_xyz(x3f, wb, bmt_to_xyz.as_mut_ptr()) } == 0 {
        return 0;
    }

    unsafe {
        x3f_3x3_diag(gain.as_mut_ptr(), gain_mat.as_mut_ptr());
        x3f_3x3_3x3_mul(bmt_to_xyz.as_mut_ptr(), gain_mat.as_mut_ptr(), raw_to_xyz);

        x3f_printf(x3f_verbosity_t_DEBUG, c"raw_to_xyz\n".as_ptr());
        x3f_3x3_print(x3f_verbosity_t_DEBUG, raw_to_xyz);
    }
    1
}

// ----------------------------------------------------------------------
// M6e2 — black-level + intermediate-range statistics
// ----------------------------------------------------------------------

/// Per-channel sum of all pixels in `area`. Returns the pixel count
/// (rows × cols).
fn sum_area(area: &x3f_area16_t, colors: usize, sum: &mut [u64]) -> u32 {
    for s in sum.iter_mut().take(colors) {
        *s = 0;
    }
    for row in 0..area.rows as usize {
        for col in 0..area.columns as usize {
            for color in 0..colors {
                let idx = area.row_stride as usize * row + area.channels as usize * col + color;
                let val = unsafe { *area.data.add(idx) };
                sum[color] += val as u64;
            }
        }
    }
    area.columns * area.rows
}

/// Per-channel sum of squared deviation from `mean`. Returns the pixel
/// count.
fn sum_area_sqdev(area: &x3f_area16_t, colors: usize, mean: &[f64], sum: &mut [f64]) -> u32 {
    for s in sum.iter_mut().take(colors) {
        *s = 0.0;
    }
    for row in 0..area.rows as usize {
        for col in 0..area.columns as usize {
            for color in 0..colors {
                let idx = area.row_stride as usize * row + area.channels as usize * col + color;
                let val = unsafe { *area.data.add(idx) } as f64;
                let dev = val - mean[color];
                sum[color] += dev * dev;
            }
        }
    }
    area.columns * area.rows
}

#[no_mangle]
pub unsafe extern "C" fn get_black_level(
    x3f: *mut x3f_t,
    image: *mut x3f_area16_t,
    rescale: libc::c_int,
    colors: libc::c_int,
    black_level: *mut f64,
    black_dev: *mut f64,
) -> libc::c_int {
    const BOTTOM: usize = 1;
    const RIGHT: usize = 3;

    let img = unsafe { &mut *image };
    if (img.channels as i32) < colors {
        return 0;
    }
    let colors = colors as usize;

    let side: [col_side_t; 4] = [
        col_side_t_COL_SIDE_WRONG,
        col_side_t_COL_SIDE_WRONG,
        col_side_t_COL_SIDE_LEFT,
        col_side_t_COL_SIDE_RIGHT,
    ];
    let name: [&CStr; 4] = [
        c"DarkShieldTop",
        c"DarkShieldBottom",
        c"Left",  // only used in printout
        c"Right", // only used in printout
    ];
    let mut use_: [bool; 4] = [true; 4];

    // Workaround: DP2 firmware bug (DarkShieldBottom incorrectly
    // specified) and Merrill family workaround (skip RIGHT column —
    // bright "shielded" region; see github issue #117).
    let mut cammodel: *mut libc::c_char = ptr::null_mut();
    if unsafe { x3f_get_prop_entry(x3f, c"CAMMODEL".as_ptr() as *mut _, &mut cammodel) } != 0 {
        let model_str = unsafe { CStr::from_ptr(cammodel) };
        let model = model_str.to_bytes();
        if model == b"SIGMA DP2" {
            use_[BOTTOM] = false;
        }
        if model == b"SIGMA DP1 Merrill"
            || model == b"SIGMA DP2 Merrill"
            || model == b"SIGMA DP3 Merrill"
            || model == b"SIGMA SD1 Merrill"
        {
            use_[RIGHT] = false;
        }
    }

    // Workaround: sd Quattro H firmware bug — DarkShieldBottom
    // incorrectly specified.
    let mut cameraid: u32 = 0;
    if unsafe { x3f_get_camf_unsigned(x3f, c"CAMERAID".as_ptr() as *mut _, &mut cameraid) } != 0
        && cameraid == X3F_CAMERAID_SDQH
    {
        use_[BOTTOM] = false;
    }

    let mut area: [x3f_area16_t; 4] = unsafe { std::mem::zeroed() };

    // Real CAMF rects — DarkShieldTop / DarkShieldBottom.
    for i in 0..2 {
        if use_[i] {
            use_[i] = unsafe {
                x3f_crop_area_camf(
                    x3f,
                    name[i].as_ptr() as *mut _,
                    image,
                    rescale,
                    &mut area[i],
                )
            } != 0;
        }
    }

    // Column-based rects — left and right shielded columns.
    for i in 2..4 {
        if use_[i] {
            use_[i] =
                unsafe { x3f_crop_area_column(x3f, side[i], image, rescale, &mut area[i]) } != 0;
        }
    }

    for i in 0..4 {
        unsafe {
            if use_[i] {
                x3f_printf(
                    x3f_verbosity_t_DEBUG,
                    c"Calculate black level for %s\n".as_ptr(),
                    name[i].as_ptr(),
                );
            } else {
                x3f_printf(
                    x3f_verbosity_t_DEBUG,
                    c"Do not calculate black level for %s\n".as_ptr(),
                    name[i].as_ptr(),
                );
            }
        }
    }

    let mut pixels_sum: u64 = 0;
    let mut black: Vec<u64> = vec![0; colors];
    let mut black_sum: Vec<u64> = vec![0; colors];

    unsafe { x3f_printf(x3f_verbosity_t_DEBUG, c"Dark level\n".as_ptr()) };

    for i in 0..4 {
        if use_[i] {
            let pixels = sum_area(&area[i], colors, &mut black) as u64;
            pixels_sum += pixels;
            unsafe {
                x3f_printf(
                    x3f_verbosity_t_DEBUG,
                    c"  %s (%d)\n".as_ptr(),
                    name[i].as_ptr(),
                    pixels as libc::c_int,
                );
            }
            for color in 0..colors {
                unsafe {
                    x3f_printf(
                        x3f_verbosity_t_DEBUG,
                        c"    mean[%d] = %f\n".as_ptr(),
                        color as libc::c_int,
                        black[color] as f64 / pixels as f64,
                    );
                }
                black_sum[color] += black[color];
            }
        }
    }

    if pixels_sum == 0 {
        return 0;
    }

    let bl_slice = unsafe { std::slice::from_raw_parts_mut(black_level, colors) };
    for i in 0..colors {
        bl_slice[i] = black_sum[i] as f64 / pixels_sum as f64;
    }

    let mut pixels_sum: u64 = 0;
    let mut black_sqdev: Vec<f64> = vec![0.0; colors];
    let mut black_sqdev_sum: Vec<f64> = vec![0.0; colors];

    for i in 0..4 {
        if use_[i] {
            let pixels = sum_area_sqdev(&area[i], colors, bl_slice, &mut black_sqdev) as u64;
            pixels_sum += pixels;
            for color in 0..colors {
                black_sqdev_sum[color] += black_sqdev[color];
            }
        }
    }

    if pixels_sum == 0 {
        return 0;
    }

    unsafe { x3f_printf(x3f_verbosity_t_DEBUG, c"  SUM\n".as_ptr()) };
    let bd_slice = unsafe { std::slice::from_raw_parts_mut(black_dev, colors) };
    for i in 0..colors {
        bd_slice[i] = (black_sqdev_sum[i] / pixels_sum as f64).sqrt();
        unsafe {
            x3f_printf(
                x3f_verbosity_t_DEBUG,
                c"    level[%d] = %f\n".as_ptr(),
                i as libc::c_int,
                bl_slice[i],
            );
            x3f_printf(
                x3f_verbosity_t_DEBUG,
                c"    dev[%d] = %g\n".as_ptr(),
                i as libc::c_int,
                bd_slice[i],
            );
        }
    }

    1
}

#[no_mangle]
pub unsafe extern "C" fn get_max_intermediate(
    x3f: *mut x3f_t,
    wb: *mut libc::c_char,
    intermediate_bias: f64,
    max_intermediate: *mut u32,
) -> libc::c_int {
    let mut gain = [0.0_f64; 3];
    if unsafe { x3f_get_gain(x3f, wb, gain.as_mut_ptr()) } == 0 {
        return 0;
    }

    // Cap the gains to 1.0 to avoid clipping (i.e. divide by max).
    let mut maxgain = 0.0_f64;
    for &g in &gain {
        if g > maxgain {
            maxgain = g;
        }
    }
    let max_slice = unsafe { std::slice::from_raw_parts_mut(max_intermediate, 3) };
    for i in 0..3 {
        let v =
            gain[i] * (INTERMEDIATE_UNIT as f64 - intermediate_bias) / maxgain + intermediate_bias;
        max_slice[i] = v.round() as i32 as u32;
    }
    1
}

#[no_mangle]
pub unsafe extern "C" fn get_intermediate_bias(
    x3f: *mut x3f_t,
    wb: *mut libc::c_char,
    black_level: *mut f64,
    black_dev: *mut f64,
    intermediate_bias: *mut f64,
) -> libc::c_int {
    let mut max_raw = [0u32; 3];
    let mut max_intermediate = [0u32; 3];

    if unsafe { x3f_get_max_raw(x3f, max_raw.as_mut_ptr()) } == 0 {
        return 0;
    }
    if unsafe { get_max_intermediate(x3f, wb, 0.0, max_intermediate.as_mut_ptr()) } == 0 {
        return 0;
    }

    let bl = unsafe { std::slice::from_raw_parts(black_level, 3) };
    let bd = unsafe { std::slice::from_raw_parts(black_dev, 3) };
    let mut bias_max = 0.0_f64;
    for i in 0..3 {
        let bias = INTERMEDIATE_BIAS_FACTOR * bd[i] * max_intermediate[i] as f64
            / (max_raw[i] as f64 - bl[i]);
        if bias > bias_max {
            bias_max = bias;
        }
    }
    unsafe {
        *intermediate_bias = bias_max;
    }
    1
}

// ----------------------------------------------------------------------
// M6e3 — bad-pixel interpolation
// ----------------------------------------------------------------------

#[derive(Clone, Copy)]
struct BadPixel {
    c: i32,
    r: i32,
}

/// Coordinate grid for marking the autofocus pixel pattern on
/// sd Quattro / sd Quattro H sensors. Mirrors the C `grid_t`:
/// initial / final / pitch / size for both column and row axes.
struct Grid {
    ci: i32,
    cf: i32,
    cp: i32,
    cs: i32,
    ri: i32,
    rf: i32,
    rp: i32,
    rs: i32,
}

#[inline]
fn pn(c: i32, r: i32, cs: i32) -> i32 {
    r * cs + c
}

#[inline]
fn inb(c: i32, r: i32, cs: i32, rs: i32) -> bool {
    c >= 0 && c < cs && r >= 0 && r < rs
}

/// Mirrors C's `TEST_PIX`: 1 if the pixel is marked bad OR out of
/// bounds, 0 otherwise. Out-of-bounds is treated as "bad" so neighbor
/// interpolation at image edges falls back to the in-bounds neighbors.
#[inline]
fn test_pix(vec: &[u32], c: i32, r: i32, cs: i32, rs: i32) -> bool {
    if !inb(c, r, cs, rs) {
        return true;
    }
    let n = pn(c, r, cs) as usize;
    (vec[n >> 5] & (1u32 << (n & 0x1f))) != 0
}

/// Mirrors C's `MARK_PIX`. Adds a bad pixel to the list and bitvec
/// only if it's not already marked. Out-of-bounds + already-marked
/// triggers a warning.
fn mark_pix(list: &mut Vec<BadPixel>, vec: &mut [u32], c: i32, r: i32, cs: i32, rs: i32) {
    if !test_pix(vec, c, r, cs, rs) {
        list.push(BadPixel { c, r });
        let n = pn(c, r, cs) as usize;
        vec[n >> 5] |= 1u32 << (n & 0x1f);
    } else if !inb(c, r, cs, rs) {
        unsafe {
            x3f_printf(
                x3f_verbosity_t_WARN,
                c"Bad pixel (%u,%u) out of bounds : (%u,%u)\n".as_ptr(),
                c as u32,
                r as u32,
                cs as u32,
                rs as u32,
            );
        }
    }
}

/// Mirrors C's `CLEAR_PIX`. Caller is responsible for ensuring the
/// pixel is in bounds (the C source uses `assert`).
#[inline]
fn clear_pix(vec: &mut [u32], c: i32, r: i32, cs: i32) {
    let n = pn(c, r, cs) as usize;
    vec[n >> 5] &= !(1u32 << (n & 0x1f));
}

#[no_mangle]
pub unsafe extern "C" fn interpolate_bad_pixels(
    x3f: *mut x3f_t,
    image: *mut x3f_area16_t,
    colors: libc::c_int,
) {
    let img = unsafe { &*image };
    let cs = img.columns as i32;
    let rs = img.rows as i32;
    let row_stride = img.row_stride as usize;
    let channels = img.channels as usize;
    let colors_us = colors as usize;
    let data = img.data;

    let bitvec_len = ((rs as usize) * (cs as usize) + 31) / 32;
    let mut bad_pixel_vec: Vec<u32> = vec![0; bitvec_len];
    let mut bad_pixels: Vec<BadPixel> = Vec::new();

    // ---- BEGIN — collect bad pixels ----

    if colors == 3 {
        let mut keep = [0u32; 4];
        let keep_ok = unsafe {
            x3f_get_camf_matrix(
                x3f,
                c"KeepImageArea".as_ptr() as *mut _,
                4,
                0,
                0,
                matrix_type_t_M_UINT,
                keep.as_mut_ptr() as *mut libc::c_void,
            ) != 0
        };

        if keep_ok {
            let mut bp_num: libc::c_int = 0;
            let mut bp: *mut u32 = ptr::null_mut();
            let bp_ok = unsafe {
                x3f_get_camf_matrix_var(
                    x3f,
                    c"BadPixels".as_ptr() as *mut _,
                    &mut bp_num,
                    ptr::null_mut(),
                    ptr::null_mut(),
                    matrix_type_t_M_UINT,
                    &mut bp as *mut *mut u32 as *mut *mut libc::c_void,
                ) != 0
            };
            if bp_ok {
                for i in 0..bp_num as usize {
                    let v = unsafe { *bp.add(i) };
                    let c = (((v & 0x000fff00) >> 8) as i32) - keep[0] as i32;
                    let r = (((v & 0xfff00000) >> 20) as i32) - keep[1] as i32;
                    mark_pix(&mut bad_pixels, &mut bad_pixel_vec, c, r, cs, rs);
                }
            }
        }

        // BadPixelsF20: rows/cols swapped due to firmware bug.
        let mut bpf20_cols: libc::c_int = 0;
        let mut bpf20_rows: libc::c_int = 0;
        let mut bpf20: *mut u32 = ptr::null_mut();
        let f20_ok = unsafe {
            x3f_get_camf_matrix_var(
                x3f,
                c"BadPixelsF20".as_ptr() as *mut _,
                &mut bpf20_cols,
                &mut bpf20_rows,
                ptr::null_mut(),
                matrix_type_t_M_UINT,
                &mut bpf20 as *mut *mut u32 as *mut *mut libc::c_void,
            ) != 0
                && bpf20_cols == 3
        };
        if f20_ok {
            for row in 0..bpf20_rows as usize {
                let c = unsafe { *bpf20.add(3 * row + 1) } as i32;
                let r = unsafe { *bpf20.add(3 * row) } as i32;
                mark_pix(&mut bad_pixels, &mut bad_pixel_vec, c, r, cs, rs);
            }
        }

        // Jpeg_BadClusters: same firmware quirk.
        let mut jbc_cols: libc::c_int = 0;
        let mut jbc_rows: libc::c_int = 0;
        let mut jbc: *mut u32 = ptr::null_mut();
        let jbc_ok = unsafe {
            x3f_get_camf_matrix_var(
                x3f,
                c"Jpeg_BadClusters".as_ptr() as *mut _,
                &mut jbc_cols,
                &mut jbc_rows,
                ptr::null_mut(),
                matrix_type_t_M_UINT,
                &mut jbc as *mut *mut u32 as *mut *mut libc::c_void,
            ) != 0
                && jbc_cols == 3
        };
        if jbc_ok {
            for row in 0..jbc_rows as usize {
                let c = unsafe { *jbc.add(3 * row + 1) } as i32;
                let r = unsafe { *jbc.add(3 * row) } as i32;
                mark_pix(&mut bad_pixels, &mut bad_pixel_vec, c, r, cs, rs);
            }
        }

        // HighlightPixelsInfo: stride-marked pattern of bright pixels.
        let mut hpinfo = [0u32; 4];
        let hp_ok = unsafe {
            x3f_get_camf_matrix(
                x3f,
                c"HighlightPixelsInfo".as_ptr() as *mut _,
                2,
                2,
                0,
                matrix_type_t_M_UINT,
                hpinfo.as_mut_ptr() as *mut libc::c_void,
            ) != 0
        };
        if hp_ok {
            let mut row = hpinfo[1] as i32;
            while row < rs {
                let mut col = hpinfo[0] as i32;
                while col < cs {
                    mark_pix(&mut bad_pixels, &mut bad_pixel_vec, col, row, cs, rs);
                    col += hpinfo[2] as i32;
                }
                row += hpinfo[3] as i32;
            }
        }
    } // colors == 3

    // BadPixelsLumaF23 (colors == 1) / BadPixelsChromaF23 (colors == 3):
    // packed runs of (row, col0, col1, ..., 0)*. The C iterator uses a
    // somewhat unusual `i++` inside the else branch to skip the
    // terminator — we mirror that logic explicitly.
    let f23_name: &CStr = if colors == 1 {
        c"BadPixelsLumaF23"
    } else if colors == 3 {
        c"BadPixelsChromaF23"
    } else {
        c""
    };
    if !f23_name.to_bytes().is_empty() {
        let mut bpf23_len: libc::c_int = 0;
        let mut bpf23: *mut u32 = ptr::null_mut();
        let f23_ok = unsafe {
            x3f_get_camf_matrix_var(
                x3f,
                f23_name.as_ptr() as *mut _,
                &mut bpf23_len,
                ptr::null_mut(),
                ptr::null_mut(),
                matrix_type_t_M_UINT,
                &mut bpf23 as *mut *mut u32 as *mut *mut libc::c_void,
            ) != 0
        };
        if f23_ok {
            let mut row: i32 = -1;
            let mut i: usize = 0;
            while i < bpf23_len as usize {
                let v = unsafe { *bpf23.add(i) } as i32;
                if row == -1 {
                    row = v;
                } else if v == 0 {
                    row = -1;
                } else {
                    mark_pix(&mut bad_pixels, &mut bad_pixel_vec, v, row, cs, rs);
                    i += 1; // skip the next entry, matching the C `i++` quirk
                }
                i += 1;
            }
        }
    }

    // sd Quattro / Quattro H: hardcoded autofocus pixel grid.
    let mut cameraid: u32 = 0;
    if unsafe { x3f_get_camf_unsigned(x3f, c"CAMERAID".as_ptr() as *mut _, &mut cameraid) } != 0 {
        let g: Option<&Grid> = if cameraid == X3F_CAMERAID_SDQ {
            const SDQ_AF_LUMA: Grid = Grid {
                ci: 217,
                cf: 5641,
                cp: 16,
                cs: 1,
                ri: 464,
                rf: 3312,
                rp: 32,
                rs: 2,
            };
            const SDQ_AF_CHROMA: Grid = Grid {
                ci: 108,
                cf: 2820,
                cp: 8,
                cs: 1,
                ri: 232,
                rf: 1656,
                rp: 16,
                rs: 1,
            };
            if colors == 1 {
                Some(&SDQ_AF_LUMA)
            } else {
                Some(&SDQ_AF_CHROMA)
            }
        } else if cameraid == X3F_CAMERAID_SDQH {
            const SDQH_AF_LUMA: Grid = Grid {
                ci: 233,
                cf: 6425,
                cp: 16,
                cs: 1,
                ri: 592,
                rf: 3888,
                rp: 32,
                rs: 2,
            };
            const SDQH_AF_CHROMA: Grid = Grid {
                ci: 116,
                cf: 2820,
                cp: 8,
                cs: 1,
                ri: 296,
                rf: 1944,
                rp: 16,
                rs: 1,
            };
            if colors == 1 {
                Some(&SDQH_AF_LUMA)
            } else {
                Some(&SDQH_AF_CHROMA)
            }
        } else {
            None
        };
        if let Some(g) = g {
            unsafe {
                x3f_printf(
                    x3f_verbosity_t_DEBUG,
                    c"Create AF grid for removing bad pixels\n".as_ptr(),
                );
            }
            let mut row = g.ri;
            while row <= g.rf {
                let mut col = g.ci;
                while col <= g.cf {
                    for r in 0..g.rs {
                        for c in 0..g.cs {
                            mark_pix(
                                &mut bad_pixels,
                                &mut bad_pixel_vec,
                                col + c,
                                row + r,
                                cs,
                                rs,
                            );
                        }
                    }
                    col += g.cp;
                }
                row += g.rp;
            }
        }
    }

    // ---- END — collecting; BEGIN — fixing ----

    if !bad_pixels.is_empty() {
        unsafe {
            x3f_printf(
                x3f_verbosity_t_DEBUG,
                c"There are bad pixels to fix\n".as_ptr(),
            );
        }
    }

    let mut fix_corner = false;
    let mut stat_pass: i32 = 0;

    while !bad_pixels.is_empty() {
        let mut all_four = 0;
        let mut two_linear = 0;
        let mut two_corner = 0;
        let mut left = 0;

        let mut still_bad: Vec<BadPixel> = Vec::with_capacity(bad_pixels.len());
        let mut fixed: Vec<BadPixel> = Vec::with_capacity(bad_pixels.len());

        for p in bad_pixels.iter() {
            let pc = p.c;
            let pr = p.r;
            let outp_idx = pr as usize * row_stride + pc as usize * channels;

            let mut inp: [Option<usize>; 4] = [None; 4];
            let mut num = 0;

            // Collect status of neighbour pixels (left, right, up, down).
            if !test_pix(&bad_pixel_vec, pc - 1, pr, cs, rs) {
                num += 1;
                inp[0] = Some(pr as usize * row_stride + (pc - 1) as usize * channels);
            }
            if !test_pix(&bad_pixel_vec, pc + 1, pr, cs, rs) {
                num += 1;
                inp[1] = Some(pr as usize * row_stride + (pc + 1) as usize * channels);
            }
            if !test_pix(&bad_pixel_vec, pc, pr - 1, cs, rs) {
                num += 1;
                inp[2] = Some((pr - 1) as usize * row_stride + pc as usize * channels);
            }
            if !test_pix(&bad_pixel_vec, pc, pr + 1, cs, rs) {
                num += 1;
                inp[3] = Some((pr + 1) as usize * row_stride + pc as usize * channels);
            }

            // Test if interpolation is possible.
            if inp[0].is_some() && inp[1].is_some() && inp[2].is_some() && inp[3].is_some() {
                all_four += 1;
            } else if inp[0].is_some() && inp[1].is_some() {
                inp[2] = None;
                inp[3] = None;
                num = 2;
                two_linear += 1;
            } else if inp[2].is_some() && inp[3].is_some() {
                inp[0] = None;
                inp[1] = None;
                num = 2;
                two_linear += 1;
            } else if fix_corner && num == 2 {
                two_corner += 1;
            } else {
                left += 1;
                still_bad.push(*p);
                continue;
            }

            // Interpolate the actual pixel.
            for color in 0..colors_us {
                let mut sum: u32 = 0;
                for slot in &inp {
                    if let Some(off) = slot {
                        sum += unsafe { *data.add(off + color) } as u32;
                    }
                }
                unsafe {
                    *data.add(outp_idx + color) = ((sum + (num as u32) / 2) / num as u32) as u16;
                }
            }
            fixed.push(*p);
        }

        unsafe {
            x3f_printf(
                x3f_verbosity_t_DEBUG,
                c"Bad pixels pass %d: %d fixed (%d all_four, %d linear, %d corner), %d left\n"
                    .as_ptr(),
                stat_pass as libc::c_int,
                (all_four + two_linear + two_corner) as libc::c_int,
                all_four as libc::c_int,
                two_linear as libc::c_int,
                two_corner as libc::c_int,
                left as libc::c_int,
            );
        }

        if fixed.is_empty() {
            // Nothing fixed this pass.
            if !fix_corner {
                fix_corner = true;
            } else {
                unsafe {
                    x3f_printf(
                        x3f_verbosity_t_WARN,
                        c"Failed to interpolate %d bad pixels\n".as_ptr(),
                        left as libc::c_int,
                    );
                }
                // C dumps remaining list onto fixed and clears bad
                // list to force termination — we mirror that by
                // dropping still_bad and leaving bad_pixels empty.
                fixed.append(&mut still_bad);
                bad_pixels.clear();
                // Clear bitvec for everything fixed so we exit cleanly.
                for p in &fixed {
                    if inb(p.c, p.r, cs, rs) {
                        clear_pix(&mut bad_pixel_vec, p.c, p.r, cs);
                    }
                }
                break;
            }
        } else {
            // Clear bitvec bits for all pixels fixed in this pass —
            // *after* the inner loop, so neighbor-lookups within this
            // pass still see the pre-pass bitvec (matches C).
            for p in &fixed {
                if inb(p.c, p.r, cs, rs) {
                    clear_pix(&mut bad_pixel_vec, p.c, p.r, cs);
                }
            }
        }

        bad_pixels = still_bad;
        stat_pass += 1;
    }
}

// ----------------------------------------------------------------------
// M6e5 — pre-matrix WB-conditional radial color shading
// ----------------------------------------------------------------------
//
// Sigma's pre-matrix radial chroma correction. Decoded from
// `priProcessF20PreprocessStage` @ 0xada9c-0xae010 in the Merrill DLL.
// Per pixel at (row, col):
//
//   rr  = sqrt((col-cx)^2 + (row-cy)^2) / denom
//   f_B = 1 + a*rr^4 + b*rr^2     (channel 0 = Bottom, red-receptive)
//   f_T = 1 + c*rr^4 + d*rr^2     (channel 2 = Top, blue-receptive)
//   f_M = 1                       (channel 1 = Middle, untouched)
//
// Coefficients come from a per-WB 2x2 CAMF matrix
// (`WhiteBalanceColorShadingFactor`). Merrill-only; Quattro uses a
// different correction (BData/C8x_y in the Quattro DLL).

struct WbColorShading {
    a: f64,
    b: f64,
    c: f64,
    d: f64,
    cx: f64,
    cy: f64,
    denom: f64,
}

unsafe fn is_merrill_model(x3f: *mut x3f_t) -> bool {
    let mut cammodel: *mut libc::c_char = ptr::null_mut();
    if unsafe { x3f_get_prop_entry(x3f, c"CAMMODEL".as_ptr() as *mut _, &mut cammodel) } == 0 {
        return false;
    }
    let model = unsafe { CStr::from_ptr(cammodel) }.to_bytes();
    matches!(
        model,
        b"SIGMA DP1 Merrill" | b"SIGMA DP2 Merrill" | b"SIGMA DP3 Merrill" | b"SIGMA SD1 Merrill"
    )
}

unsafe fn get_wb_color_shading(
    x3f: *mut x3f_t,
    wb: *mut libc::c_char,
    fallback_rows: i32,
    fallback_cols: i32,
) -> Option<WbColorShading> {
    if !unsafe { is_merrill_model(x3f) } {
        return None;
    }

    // The C source first calls x3f_get_camf_property_list as a
    // presence check (it returns the list of WB names within the
    // CAMF block), then x3f_get_camf_matrix_for_wb to actually read
    // the 2x2 matrix for the requested WB. Preserve both calls so
    // the path-through-CAMF is identical.
    let mut prop_names: *mut *mut libc::c_char = ptr::null_mut();
    let mut prop_values: *mut *mut libc::c_char = ptr::null_mut();
    let mut prop_num: u32 = 0;
    if unsafe {
        x3f_get_camf_property_list(
            x3f,
            c"WhiteBalanceColorShadingFactor".as_ptr() as *mut _,
            &mut prop_names,
            &mut prop_values,
            &mut prop_num,
        )
    } == 0
    {
        return None;
    }

    let mut mat = [0.0_f64; 4];
    if unsafe {
        x3f_get_camf_matrix_for_wb(
            x3f,
            c"WhiteBalanceColorShadingFactor".as_ptr() as *mut _,
            wb,
            2,
            2,
            mat.as_mut_ptr(),
        )
    } == 0
    {
        return None;
    }

    let (cx, cy, half_w, half_h);
    let mut active_rect = [0u32; 4];
    if unsafe {
        x3f_get_camf_matrix(
            x3f,
            c"ActiveImageArea".as_ptr() as *mut _,
            4,
            0,
            0,
            matrix_type_t_M_UINT,
            active_rect.as_mut_ptr() as *mut libc::c_void,
        )
    } != 0
    {
        half_w = (active_rect[2] - active_rect[0]) as f64 / 2.0;
        half_h = (active_rect[3] - active_rect[1]) as f64 / 2.0;
        cx = active_rect[0] as f64 + half_w;
        cy = active_rect[1] as f64 + half_h;
    } else {
        half_w = fallback_cols as f64 / 2.0;
        half_h = fallback_rows as f64 / 2.0;
        cx = half_w;
        cy = half_h;
    }
    let denom = (half_w * half_w + half_h * half_h).sqrt();
    if denom <= 0.0 {
        return None;
    }

    Some(WbColorShading {
        a: mat[0],
        b: mat[1],
        c: mat[2],
        d: mat[3],
        cx,
        cy,
        denom,
    })
}

#[inline]
fn wb_color_shading_factors(s: &WbColorShading, row: i32, col: i32) -> (f64, f64) {
    let dx = col as f64 - s.cx;
    let dy = row as f64 - s.cy;
    let rr2 = (dx * dx + dy * dy) / (s.denom * s.denom);
    let rr4 = rr2 * rr2;
    let f_b = 1.0 + s.a * rr4 + s.b * rr2;
    let f_t = 1.0 + s.c * rr4 + s.d * rr2;
    (f_b, f_t)
}

#[no_mangle]
pub unsafe extern "C" fn apply_wb_color_shading(
    x3f: *mut x3f_t,
    wb: *mut libc::c_char,
    image: *mut x3f_area16_t,
) -> libc::c_int {
    let img = unsafe { &mut *image };
    let s = match unsafe { get_wb_color_shading(x3f, wb, img.rows as i32, img.columns as i32) } {
        Some(s) => s,
        None => return 0,
    };

    unsafe {
        x3f_printf(
            x3f_verbosity_t_DEBUG,
            c"WBColorShadingFactor[%s]: a=%g b=%g c=%g d=%g\n".as_ptr(),
            wb,
            s.a,
            s.b,
            s.c,
            s.d,
        );
        x3f_printf(
            x3f_verbosity_t_DEBUG,
            c"WBColorShadingFactor: cx=%g cy=%g denom=%g\n".as_ptr(),
            s.cx,
            s.cy,
            s.denom,
        );
    }

    let row_stride = img.row_stride as usize;
    let channels = img.channels as usize;
    let cols = img.columns as i32;
    let total_u16 = img.rows as usize * row_stride;
    let data = unsafe { std::slice::from_raw_parts_mut(img.data, total_u16) };
    use rayon::prelude::*;
    data.par_chunks_mut(row_stride)
        .enumerate()
        .for_each(|(row, row_data)| {
            let row = row as i32;
            for col in 0..cols {
                let (f_b, f_t) = wb_color_shading_factors(&s, row, col);
                let off = col as usize * channels;
                let p0 = row_data[off] as f64;
                let p2 = row_data[off + 2] as f64;
                let v0 = (p0 * f_b).round() as i32;
                let v2 = (p2 * f_t).round() as i32;
                row_data[off] = v0.clamp(0, 65535) as u16;
                row_data[off + 2] = v2.clamp(0, 65535) as u16;
            }
        });
    1
}

// ----------------------------------------------------------------------
// M6e6 — preprocess_data orchestrator
// ----------------------------------------------------------------------
//
// Run sequence:
//
//   1. Pull the RAW area (and Quattro top16 if present)
//   2. Compute per-channel black level + black noise stddev from
//      DarkShield rects and column ranges (M6e2)
//   3. Read max_raw and compute intermediate_bias / max_intermediate
//      (M6e2 helpers); populate the caller's `ilevels` struct
//   4. Read DigitalISOGain from CAMF (skipped on Merrill — see notes)
//   5. Per-pixel: out = scale * (raw - black) + intermediate_bias,
//      clamped to [0, 65535] uint16
//   6. For Quattro: downsample top16 4-pixel sums into image[2], then
//      preprocess top16 full resolution with the same rule
//   7. WB-conditional radial color shading (M6e5; Merrill-only)
//   8. Bad-pixel interpolation (M6e3) — twice on Quattro: top first,
//      then image

#[no_mangle]
pub unsafe extern "C" fn preprocess_data(
    x3f: *mut x3f_t,
    fix_bad: libc::c_int,
    wb: *mut libc::c_char,
    ilevels: *mut x3f_image_levels_t,
) -> libc::c_int {
    let mut image: x3f_area16_t = unsafe { std::mem::zeroed() };
    let mut qtop: x3f_area16_t = unsafe { std::mem::zeroed() };

    let quattro = unsafe { x3f_image_area_qtop(x3f, &mut qtop) } != 0;
    let colors_in = if quattro { 2 } else { 3 };

    if unsafe { x3f_image_area(x3f, &mut image) } == 0 || image.channels < 3 {
        return 0;
    }
    if quattro
        && (qtop.channels < 1 || qtop.rows < 2 * image.rows || qtop.columns < 2 * image.columns)
    {
        return 0;
    }

    let mut black_level = [0.0_f64; 3];
    let mut black_dev = [0.0_f64; 3];

    let bl_image_ok = unsafe {
        get_black_level(
            x3f,
            &mut image,
            1,
            colors_in,
            black_level.as_mut_ptr(),
            black_dev.as_mut_ptr(),
        )
    };
    let bl_qtop_ok = if quattro {
        unsafe {
            get_black_level(
                x3f,
                &mut qtop,
                0,
                1,
                black_level.as_mut_ptr().add(2),
                black_dev.as_mut_ptr().add(2),
            )
        }
    } else {
        1
    };
    if bl_image_ok == 0 || bl_qtop_ok == 0 {
        unsafe {
            x3f_printf(x3f_verbosity_t_ERR, c"Could not get black level\n".as_ptr());
        }
        return 0;
    }
    unsafe {
        x3f_printf(
            x3f_verbosity_t_DEBUG,
            c"black_level = {%g,%g,%g}, black_dev = {%g,%g,%g}\n".as_ptr(),
            black_level[0],
            black_level[1],
            black_level[2],
            black_dev[0],
            black_dev[1],
            black_dev[2],
        );
    }

    let mut max_raw = [0u32; 3];
    if unsafe { x3f_get_max_raw(x3f, max_raw.as_mut_ptr()) } == 0 {
        unsafe {
            x3f_printf(
                x3f_verbosity_t_ERR,
                c"Could not get maximum RAW level\n".as_ptr(),
            );
        }
        return 0;
    }
    unsafe {
        x3f_printf(
            x3f_verbosity_t_DEBUG,
            c"max_raw = {%u,%u,%u}\n".as_ptr(),
            max_raw[0],
            max_raw[1],
            max_raw[2],
        );
    }

    let il = unsafe { &mut *ilevels };

    let mut intermediate_bias: f64 = 0.0;
    if unsafe {
        get_intermediate_bias(
            x3f,
            wb,
            black_level.as_mut_ptr(),
            black_dev.as_mut_ptr(),
            &mut intermediate_bias,
        )
    } == 0
    {
        unsafe {
            x3f_printf(
                x3f_verbosity_t_ERR,
                c"Could not get intermediate bias\n".as_ptr(),
            );
        }
        return 0;
    }
    unsafe {
        x3f_printf(
            x3f_verbosity_t_DEBUG,
            c"intermediate_bias = %g\n".as_ptr(),
            intermediate_bias,
        );
    }
    il.black[0] = intermediate_bias;
    il.black[1] = intermediate_bias;
    il.black[2] = intermediate_bias;

    if unsafe { get_max_intermediate(x3f, wb, intermediate_bias, il.white.as_mut_ptr()) } == 0 {
        unsafe {
            x3f_printf(
                x3f_verbosity_t_ERR,
                c"Could not get maximum intermediate level\n".as_ptr(),
            );
        }
        return 0;
    }
    unsafe {
        x3f_printf(
            x3f_verbosity_t_DEBUG,
            c"max_intermediate = {%u,%u,%u}\n".as_ptr(),
            il.white[0],
            il.white[1],
            il.white[2],
        );
    }

    // DigitalISOGain skipped for Merrill (the legacy comment cites
    // "color cast issues"). For all other models, read from CAMF; if
    // missing, default to (1, 1, 1).
    let mut digital_iso_gain = [1.0_f64; 3];
    let merrill = {
        let mut cammodel: *mut libc::c_char = ptr::null_mut();
        if unsafe { x3f_get_prop_entry(x3f, c"CAMMODEL".as_ptr() as *mut _, &mut cammodel) } != 0 {
            let model = unsafe { CStr::from_ptr(cammodel) }.to_bytes();
            matches!(
                model,
                b"SIGMA DP1 Merrill"
                    | b"SIGMA DP2 Merrill"
                    | b"SIGMA DP3 Merrill"
                    | b"SIGMA SD1 Merrill"
            )
        } else {
            false
        }
    };
    if merrill {
        unsafe {
            x3f_printf(
                x3f_verbosity_t_DEBUG,
                c"Merrill model detected, skipping DigitalISOGain\n".as_ptr(),
            );
            x3f_printf(
                x3f_verbosity_t_DEBUG,
                c"Using digital_ISO_Gain = {1.0,1.0,1.0} for Merrill model\n".as_ptr(),
            );
        }
    } else if unsafe {
        x3f_get_camf_float_vector(
            x3f,
            c"DigitalISOGain".as_ptr() as *mut _,
            digital_iso_gain.as_mut_ptr(),
        )
    } != 0
    {
        unsafe {
            x3f_printf(
                x3f_verbosity_t_DEBUG,
                c"digital_ISO_Gain = {%f,%f,%f}\n".as_ptr(),
                digital_iso_gain[0],
                digital_iso_gain[1],
                digital_iso_gain[2],
            );
        }
    } else {
        digital_iso_gain = [1.0; 3];
    }

    let mut scale = [0.0_f64; 3];
    for color in 0..3 {
        scale[color] = ((il.white[color] as f64 - il.black[color])
            / (max_raw[color] as f64 - black_level[color]))
            * digital_iso_gain[color];
    }

    // Preprocess image data (HUF/TRU -> x3rgb16). Per pixel:
    //   out = round(scale * (raw - black_level) + ilevels.black)
    let img_row_stride = image.row_stride as usize;
    let img_channels = image.channels as usize;
    let img_cols = image.columns as usize;
    let il_black = il.black;
    {
        use rayon::prelude::*;
        let total = image.rows as usize * img_row_stride;
        let img_data = unsafe { std::slice::from_raw_parts_mut(image.data, total) };
        let colors = colors_in as usize;
        img_data
            .par_chunks_mut(img_row_stride)
            .for_each(|row_data| {
                for col in 0..img_cols {
                    let off = img_channels * col;
                    for color in 0..colors {
                        let v = row_data[off + color] as f64;
                        let out = (scale[color] * (v - black_level[color]) + il_black[color])
                            .round() as i32;
                        row_data[off + color] = if out < 0 {
                            0
                        } else if out > 65535 {
                            65535
                        } else {
                            out as u16
                        };
                    }
                }
            });
    }

    if quattro {
        // Downsample top16 4-pixel sums into image[2]. The qtop read
        // is shared / read-only; only image.data is written. Wrap the
        // qtop pointer in a Send+Sync newtype so rayon can capture it.
        let q_row_stride = qtop.row_stride as usize;
        let q_channels = qtop.channels as usize;
        let qptr = SyncConstU16Ptr(qtop.data as *const u16);
        let total_image = image.rows as usize * img_row_stride;
        let img_data = unsafe { std::slice::from_raw_parts_mut(image.data, total_image) };
        use rayon::prelude::*;
        img_data
            .par_chunks_mut(img_row_stride)
            .enumerate()
            .for_each(|(row, row_data)| {
                for col in 0..img_cols {
                    let r1_off = q_row_stride * (2 * row) + q_channels * (2 * col);
                    let r2_off = q_row_stride * (2 * row + 1) + q_channels * (2 * col);
                    let qp = qptr.as_ptr();
                    let sum = unsafe {
                        *qp.add(r1_off) as u32
                            + *qp.add(r1_off + q_channels) as u32
                            + *qp.add(r2_off) as u32
                            + *qp.add(r2_off + q_channels) as u32
                    };
                    let v = sum as f64 / 4.0;
                    let out = (scale[2] * (v - black_level[2]) + il_black[2]).round() as i32;
                    let off = img_channels * col + 2;
                    row_data[off] = if out < 0 {
                        0
                    } else if out > 65535 {
                        65535
                    } else {
                        out as u16
                    };
                }
            });

        // Preprocess top16 at full resolution. Pure per-pixel scalar
        // — row-parallel.
        let q_total = qtop.rows as usize * q_row_stride;
        let q_data = unsafe { std::slice::from_raw_parts_mut(qtop.data, q_total) };
        let q_cols = qtop.columns as usize;
        q_data.par_chunks_mut(q_row_stride).for_each(|row_data| {
            for col in 0..q_cols {
                let idx = q_channels * col;
                let v = row_data[idx] as f64;
                let out = (scale[2] * (v - black_level[2]) + il_black[2]).round() as i32;
                row_data[idx] = if out < 0 {
                    0
                } else if out > 65535 {
                    65535
                } else {
                    out as u16
                };
            }
        });
        if fix_bad != 0 {
            unsafe { interpolate_bad_pixels(x3f, &mut qtop, 1) };
        }
    }

    // WB-conditional radial color shading (Merrill-only; the function
    // returns 0 for non-Merrill bodies, and the caller ignores that).
    unsafe { apply_wb_color_shading(x3f, wb, &mut image) };

    if fix_bad != 0 {
        unsafe { interpolate_bad_pixels(x3f, &mut image, 3) };
    }

    1
}

// ----------------------------------------------------------------------
// M6e7 — get_conv + convert_data hot loop
// ----------------------------------------------------------------------

const LUTSIZE: libc::c_int = 1024;
const MAX_CORR: usize = 6; // x3f_spatial_gain.h MAXCORR

#[no_mangle]
pub unsafe extern "C" fn get_conv(
    x3f: *mut x3f_t,
    encoding: x3f_color_encoding_t,
    wb: *mut libc::c_char,
    lutsize: libc::c_int,
    max_out: u16,
    lut: *mut f64,
    conv_matrix: *mut f64,
) -> libc::c_int {
    let mut raw_to_xyz = [0.0_f64; 9];
    let mut xyz_to_rgb = [0.0_f64; 9];
    let mut raw_to_rgb = [0.0_f64; 9];

    let mut sensor_iso = 0.0_f64;
    let mut capture_iso = 0.0_f64;
    let iso_scaling = if unsafe {
        x3f_get_camf_float(x3f, c"SensorISO".as_ptr() as *mut _, &mut sensor_iso) != 0
            && x3f_get_camf_float(x3f, c"CaptureISO".as_ptr() as *mut _, &mut capture_iso) != 0
    } {
        unsafe {
            x3f_printf(
                x3f_verbosity_t_DEBUG,
                c"SensorISO = %g\n".as_ptr(),
                sensor_iso,
            );
            x3f_printf(
                x3f_verbosity_t_DEBUG,
                c"CaptureISO = %g\n".as_ptr(),
                capture_iso,
            );
        }
        capture_iso / sensor_iso
    } else {
        unsafe {
            x3f_printf(
                x3f_verbosity_t_WARN,
                c"Could not calculate ISO scaling, assuming %g\n".as_ptr(),
                1.0_f64,
            );
        }
        1.0
    };

    if unsafe { x3f_get_raw_to_xyz(x3f, wb, raw_to_xyz.as_mut_ptr()) } == 0 {
        unsafe {
            x3f_printf(
                x3f_verbosity_t_ERR,
                c"Could not get raw_to_xyz for white balance: %s\n".as_ptr(),
                wb,
            );
        }
        return 0;
    }

    // Cineon-log TIFF mode replaces every encoding's gamma LUT with a
    // shared log curve `y = log(scale·x + 1) / log(scale + 1)`. The
    // matrix is still picked per `encoding` so the user gets sRGB /
    // Adobe RGB / ProPhoto RGB *primaries* with the log curve baked in,
    // which is what colour-grading apps expect when paired with the
    // ICC profile whose TRC samples the inverse curve.
    let cineon = CINEON.with(|c| c.get());
    let cineon_scale = if cineon { cineon_scale_from_env() } else { 0.0 };
    match encoding {
        x3f_color_encoding_e_SRGB => unsafe {
            if cineon {
                x3f_cineon_log_LUT(lut, lutsize, max_out, cineon_scale);
            } else {
                x3f_sRGB_LUT(lut, lutsize, max_out);
            }
            x3f_XYZ_to_sRGB(xyz_to_rgb.as_mut_ptr());
        },
        x3f_color_encoding_e_ARGB => unsafe {
            if cineon {
                x3f_cineon_log_LUT(lut, lutsize, max_out, cineon_scale);
            } else {
                x3f_gamma_LUT(lut, lutsize, max_out, 2.2);
            }
            x3f_XYZ_to_AdobeRGB(xyz_to_rgb.as_mut_ptr());
        },
        x3f_color_encoding_e_PPRGB => unsafe {
            let mut xyz_to_prophotorgb = [0.0_f64; 9];
            let mut d65_to_d50 = [0.0_f64; 9];
            if cineon {
                x3f_cineon_log_LUT(lut, lutsize, max_out, cineon_scale);
            } else {
                x3f_gamma_LUT(lut, lutsize, max_out, 1.8);
            }
            x3f_XYZ_to_ProPhotoRGB(xyz_to_prophotorgb.as_mut_ptr());
            // ProPhoto RGB's standard white point is D50.
            x3f_Bradford_D65_to_D50(d65_to_d50.as_mut_ptr());
            x3f_3x3_3x3_mul(
                xyz_to_prophotorgb.as_mut_ptr(),
                d65_to_d50.as_mut_ptr(),
                xyz_to_rgb.as_mut_ptr(),
            );
        },
        _ => {
            unsafe {
                x3f_printf(
                    x3f_verbosity_t_ERR,
                    c"Unknown color space %d\n".as_ptr(),
                    encoding as libc::c_int,
                );
            }
            return 0;
        }
    }

    unsafe {
        x3f_3x3_3x3_mul(
            xyz_to_rgb.as_mut_ptr(),
            raw_to_xyz.as_mut_ptr(),
            raw_to_rgb.as_mut_ptr(),
        );
        x3f_scalar_3x3_mul(iso_scaling, raw_to_rgb.as_mut_ptr(), conv_matrix);

        x3f_printf(x3f_verbosity_t_DEBUG, c"raw_to_rgb\n".as_ptr());
        x3f_3x3_print(x3f_verbosity_t_DEBUG, raw_to_rgb.as_mut_ptr());
        x3f_printf(x3f_verbosity_t_DEBUG, c"conv_matrix\n".as_ptr());
        x3f_3x3_print(x3f_verbosity_t_DEBUG, conv_matrix);
    }
    1
}

// M7a — per-row context for the parallel `convert_data` body. All raw
// pointers are read-shared (CLUT, sgain, prior, etc.) or per-thread
// disjoint (`row_data` is the only writable slice and rayon hands each
// thread a non-overlapping row chunk). The `unsafe Send + Sync` impls
// reflect that — the underlying C contracts of the FFI helpers are
// read-only despite their `*mut` ABI signatures (verified by
// inspection of x3f_calc_spatial_gain, chroma_lut_apply_pixel,
// reconstruct_highlights, repair_pix_apply_pixel, x3f_3x3_3x1_mul,
// x3f_LUT_lookup).
#[derive(Copy, Clone)]
struct ConvCtx {
    sgain: *mut x3f_spatial_gain_corr_t,
    sgain_num: libc::c_int,
    conv_matrix: *mut f64,
    lut: *mut f64,
    prior: *const f64,
    hp: *const highlight_params_t,
    clut: *const chroma_lut_t,
    use_clut: bool,
    repair: *const repair_pix_t,
    use_repair: bool,
    sat_map: *const u8,
    ev_scale: f64,
    gate_thr: f64,
    gate_width: f64,
    black: [f64; 3],
    white: [u32; 3],
    rows: i32,
    cols: i32,
    channels: usize,
}

// SAFETY: see ConvCtx doc-comment above. The pointers are either
// read-only over the lifetime of convert_data or alias `row_data` only
// in shape (the `img.data` whole-slice pointer); rayon's row chunks
// guarantee disjoint writes.
unsafe impl Send for ConvCtx {}
unsafe impl Sync for ConvCtx {}

// M7d — inlinable per-pixel matrix multiply and LUT lookup. The
// `extern "C"` versions in `matrix.rs` are ABI-locked and cross-crate-
// boundary opaque, so the per-pixel hot loop pays a function-call cost
// for every pixel × every channel even with LTO. These mirrors do
// exactly the same arithmetic in the same order (no precision change,
// no FP reordering), and `#[inline(always)]` lets LLVM register-
// allocate the conv_matrix into vector regs and auto-vectorize the
// FMA chain inside `convert_row` / `dng_clip_row`.
#[inline(always)]
fn mat3x1_mul_native(m: &[f64; 9], v: [f64; 3]) -> [f64; 3] {
    [
        m[0] * v[0] + m[1] * v[1] + m[2] * v[2],
        m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
        m[6] * v[0] + m[7] * v[1] + m[8] * v[2],
    ]
}

#[inline(always)]
fn lut_lookup_native(lut: &[f64], val: f64) -> u16 {
    let n = lut.len();
    let index = val * ((n - 1) as f64);
    let i = index.floor() as i64;
    let frac = index - (i as f64);

    if i < 0 {
        lut[0].round() as u16
    } else if i >= (n as i64 - 1) {
        lut[n - 1].round() as u16
    } else {
        let i = i as usize;
        (lut[i] + frac * (lut[i + 1] - lut[i])).round() as u16
    }
}

#[inline(always)]
unsafe fn convert_row(
    ctx: &ConvCtx,
    row: i32,
    row_data: &mut [u16],
    stats: *mut chroma_lut_apply_stats_t,
) {
    // Hoist the conv_matrix and LUT into typed slices/array refs once
    // per row so the inner loop sees an inlinable native helper instead
    // of an extern "C" indirect call per pixel × per matrix mul.
    let m: &[f64; 9] = unsafe { &*(ctx.conv_matrix as *const [f64; 9]) };
    let lut: &[f64] = unsafe { std::slice::from_raw_parts(ctx.lut, LUTSIZE as usize) };

    for col in 0..ctx.cols {
        let off = col as usize * ctx.channels;

        let mut sat_ratio = [0.0_f64; 3];
        let mut sg = [0.0_f64; 3];
        for color in 0..3 {
            let v = row_data[off + color] as f64;
            sat_ratio[color] =
                (v - ctx.black[color]) / (ctx.white[color] as f64 - ctx.black[color]);
            sg[color] = unsafe {
                x3f_calc_spatial_gain(
                    ctx.sgain,
                    ctx.sgain_num,
                    row,
                    col,
                    color as i32,
                    ctx.rows,
                    ctx.cols,
                )
            };
        }

        // CLUT recovers scene-derived chroma where the bin's donor
        // evidence supports it; defers to L*p's smooth ramp on
        // near-neutral bins.
        let clut_applied = if ctx.use_clut {
            unsafe { chroma_lut_apply_pixel(sat_ratio.as_mut_ptr(), ctx.clut, stats) }
        } else {
            0
        };
        if !ctx.use_clut || clut_applied == 0 {
            unsafe {
                reconstruct_highlights(sat_ratio.as_mut_ptr(), ctx.prior, ctx.hp);
            }
        }

        if ctx.use_repair {
            unsafe {
                repair_pix_apply_pixel(
                    sat_ratio.as_mut_ptr(),
                    ctx.prior,
                    ctx.repair,
                    ctx.sat_map,
                    row,
                    col,
                    ctx.rows,
                    ctx.cols,
                );
            }
        }

        // Matrix-pathology gate. Compute the conv_matrix output
        // direction; if it's chromatically pathological (pure-
        // green dominant OR B-suppressed yellow), snap toward
        // `u_max * prior[c]` with a strength that ramps 0..1
        // over the gate_thr..gate_thr+gate_width margin band.
        {
            let mp_in = [
                sg[0] * sat_ratio[0],
                sg[1] * sat_ratio[1],
                sg[2] * sat_ratio[2],
            ];
            let mp = mat3x1_mul_native(m, mp_in);
            let margin_g = mp[1] - if mp[0] > mp[2] { mp[0] } else { mp[2] };
            let margin_y = if mp[0] < mp[1] { mp[0] } else { mp[1] } - mp[2];
            let margin = if margin_g > margin_y {
                margin_g
            } else {
                margin_y
            };
            let mut strength = (margin - ctx.gate_thr) / ctx.gate_width;
            if strength > 0.0 {
                if strength > 1.0 {
                    strength = 1.0;
                }
                let prior = unsafe { std::slice::from_raw_parts(ctx.prior, 3) };
                let mut u_max = 0.0_f64;
                for color in 0..3 {
                    let pc = if prior[color] > 1e-12 {
                        prior[color]
                    } else {
                        1e-12
                    };
                    let u = sat_ratio[color] / pc;
                    if u > u_max {
                        u_max = u;
                    }
                }
                for color in 0..3 {
                    sat_ratio[color] =
                        (1.0 - strength) * sat_ratio[color] + strength * (u_max * prior[color]);
                }
            }
        }

        // Final matrix multiply + EV scale + gamma LUT.
        let input = [
            sg[0] * sat_ratio[0],
            sg[1] * sat_ratio[1],
            sg[2] * sat_ratio[2],
        ];
        let output = mat3x1_mul_native(m, input);
        for color in 0..3 {
            let scaled = output[color] * ctx.ev_scale;
            row_data[off + color] = lut_lookup_native(lut, scaled);
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn convert_data(
    x3f: *mut x3f_t,
    image: *mut x3f_area16_t,
    ilevels: *mut x3f_image_levels_t,
    encoding: x3f_color_encoding_t,
    apply_sgain: libc::c_int,
    wb: *mut libc::c_char,
) -> libc::c_int {
    let max_out: u16 = 65535; // TODO: should be possible to adjust

    let img = unsafe { &mut *image };
    if img.channels < 3 {
        return 0;
    }

    let mut conv_matrix = [0.0_f64; 9];
    let mut lut = [0.0_f64; LUTSIZE as usize];

    if unsafe {
        get_conv(
            x3f,
            encoding,
            wb,
            LUTSIZE,
            max_out,
            lut.as_mut_ptr(),
            conv_matrix.as_mut_ptr(),
        )
    } == 0
    {
        return 0;
    }

    // Spatial gain (M6d) — corr table on the stack, MAXCORR=6 slots.
    let mut sgain: [x3f_spatial_gain_corr_t; MAX_CORR] = unsafe { std::mem::zeroed() };
    let sgain_num = if apply_sgain != 0 {
        let n = unsafe { x3f_get_spatial_gain(x3f, wb, sgain.as_mut_ptr()) };
        if n == 0 {
            unsafe {
                x3f_printf(
                    x3f_verbosity_t_WARN,
                    c"Could not get spatial gain\n".as_ptr(),
                );
            }
        }
        n
    } else {
        0
    };

    let mut hp: highlight_params_t = unsafe { std::mem::zeroed() };
    let mut prior = [0.0_f64; 3];
    unsafe {
        get_highlight_params(x3f, &mut hp);
        compute_chroma_prior(conv_matrix.as_ptr(), prior.as_mut_ptr());
    }

    // Cineon-log TIFF mode: snapshot the thread-local once so every
    // downstream decision (CLUT, RepairPix, ConvCtx field population)
    // reads the same value. Defaults to false for external C callers
    // that never set the hook, preserving the unchanged pre-cineon path.
    let cineon = CINEON.with(|c| c.get());

    // Sigma-style chromaticity LUT (production highlight path). Forced
    // off in cineon mode — it bakes scene-derived chroma recovery into
    // the pixels, which fights the colour-grader's input transform.
    let mut clut: chroma_lut_t = unsafe { std::mem::zeroed() };
    let mut use_clut = !cineon && !env_present("X3F_NO_CHROMA_LUT");
    if use_clut {
        unsafe { chroma_lut_init_defaults(&mut clut) };
        if unsafe { chroma_lut_build_from_image(&mut clut, img, ilevels, prior.as_ptr()) } == 0 {
            use_clut = false;
        }
    }
    let mut clut_stats: chroma_lut_apply_stats_t = unsafe { std::mem::zeroed() };
    let trace_clut = env_present("X3F_CHROMA_LUT_TRACE");

    // RepairPix (opt-in via X3F_REPAIR_PIX). Forced off in cineon mode
    // for the same reason as the CLUT.
    let mut repair: repair_pix_t = unsafe { std::mem::zeroed() };
    let mut sat_map: *mut u8 = ptr::null_mut();
    let mut use_repair = !cineon && env_present("X3F_REPAIR_PIX");
    if use_repair {
        unsafe { repair_pix_init_defaults(&mut repair) };
        sat_map = unsafe { build_sat_map(img, ilevels, repair.sat_threshold) };
        if sat_map.is_null() {
            use_repair = false;
        } else {
            repair.valid = 1;
        }
    }

    // Optional EV adjustment: scales the linear output before the
    // gamma LUT, simulating an under/over-exposed render.
    let ev_scale = match env_atof("X3F_EV") {
        Some(v) => 2.0_f64.powf(v),
        None => 1.0,
    };

    // Matrix-pathology gate tunables.
    let mut gate_thr = 0.20_f64;
    let mut gate_width = 0.30_f64;
    if let Some(v) = env_atof("X3F_GATE_THR") {
        gate_thr = v;
    }
    if let Some(v) = env_atof("X3F_GATE_WIDTH") {
        gate_width = v;
    }
    if gate_width < 1e-6 {
        gate_width = 1e-6;
    }

    let row_stride = img.row_stride as usize;
    let channels = img.channels as usize;
    let il = unsafe { &mut *ilevels };

    let ctx = ConvCtx {
        sgain: sgain.as_mut_ptr(),
        sgain_num,
        conv_matrix: conv_matrix.as_mut_ptr(),
        lut: lut.as_mut_ptr(),
        prior: prior.as_ptr(),
        hp: &hp,
        clut: &clut,
        use_clut,
        repair: &repair,
        use_repair,
        sat_map,
        ev_scale,
        gate_thr,
        gate_width,
        black: il.black,
        white: il.white,
        rows: img.rows as i32,
        cols: img.columns as i32,
        channels,
    };

    let total_u16 = (img.rows as usize) * row_stride;
    let data = unsafe { std::slice::from_raw_parts_mut(img.data, total_u16) };

    if trace_clut {
        // Serial path — `clut_stats` accumulator is not Sync.
        let stats_ptr = &mut clut_stats as *mut _;
        for (row, row_data) in data.chunks_mut(row_stride).enumerate() {
            unsafe { convert_row(&ctx, row as i32, row_data, stats_ptr) };
        }
    } else {
        use rayon::prelude::*;
        data.par_chunks_mut(row_stride)
            .enumerate()
            .for_each(|(row, row_data)| {
                unsafe { convert_row(&ctx, row as i32, row_data, ptr::null_mut()) };
            });
    }

    if !sat_map.is_null() {
        unsafe { libc::free(sat_map as *mut libc::c_void) };
    }
    unsafe { x3f_cleanup_spatial_gain(sgain.as_mut_ptr(), sgain_num) };
    if trace_clut {
        unsafe { chroma_lut_apply_stats_print(&clut_stats, c"tiff".as_ptr()) };
    }

    il.black[0] = 0.0;
    il.black[1] = 0.0;
    il.black[2] = 0.0;
    il.white[0] = max_out as u32;
    il.white[1] = max_out as u32;
    il.white[2] = max_out as u32;

    1
}

// `env_atof` / `env_present` aliases — these helpers live in
// highlight.rs but we re-declare locally so process.rs builds without
// tripping the orphan rule on private helpers.
fn env_atof(name: &str) -> Option<f64> {
    let v = std::env::var(name).ok()?;
    let mut buf: Vec<u8> = Vec::with_capacity(v.len() + 1);
    buf.extend_from_slice(v.as_bytes());
    buf.push(0);
    Some(unsafe { libc::atof(buf.as_ptr() as *const libc::c_char) })
}

fn env_present(name: &str) -> bool {
    std::env::var_os(name).is_some()
}

/// Resolve the Cineon log-curve scale from the `X3F_CINEON_SCALE` env var,
/// falling back to `100.0` — chosen empirically as the sweet spot
/// between "visibly flat for grading" and "shadow noise dominating".
/// Deep shadows lift from 5 % linear to ~39 % encoded; midtones (50 %)
/// land around 85 %. Smaller values produce a gentler curve (12 ≈
/// Cineon film-print density, 50 ≈ moderate log); larger values are
/// flatter still at the cost of shadow noise.
fn cineon_scale_from_env() -> f64 {
    const DEFAULT: f64 = 100.0;
    match env_atof("X3F_CINEON_SCALE") {
        Some(s) if s.is_finite() && s > 0.0 => s,
        _ => DEFAULT,
    }
}

// ----------------------------------------------------------------------
// M6e8 — run_denoising + expand_quattro
// ----------------------------------------------------------------------
//
// Both are thin wrappers. `run_denoising` crops to ActiveImageArea
// and dispatches to the still-C `x3f_denoise` (M0 stub by default; the
// real OpenCV-backed bilateral/NLM was retired). `expand_quattro`
// arranges crop rectangles, allocates the expanded RGB buffer, and
// hands off to the Rust `x3f_expand_quattro` (M5a).

#[no_mangle]
pub unsafe extern "C" fn run_denoising(x3f: *mut x3f_t) -> libc::c_int {
    let mut original_image: x3f_area16_t = unsafe { std::mem::zeroed() };
    let mut image: x3f_area16_t = unsafe { std::mem::zeroed() };

    if unsafe { x3f_image_area(x3f, &mut original_image) } == 0 {
        return 0;
    }
    if unsafe {
        x3f_crop_area_camf(
            x3f,
            c"ActiveImageArea".as_ptr() as *mut _,
            &mut original_image,
            1,
            &mut image,
        )
    } == 0
    {
        image = original_image;
        unsafe {
            x3f_printf(
                x3f_verbosity_t_WARN,
                c"Could not get active area, denoising entire image\n".as_ptr(),
            );
        }
    }

    let mut sensorid: *mut libc::c_char = ptr::null_mut();
    let mut t = x3f_denoise_type_t_X3F_DENOISE_STD;
    if unsafe { x3f_get_prop_entry(x3f, c"SENSORID".as_ptr() as *mut _, &mut sensorid) } != 0 {
        let s = unsafe { CStr::from_ptr(sensorid) }.to_bytes();
        if s == b"F20" {
            t = x3f_denoise_type_t_X3F_DENOISE_F20;
        }
    }

    unsafe { x3f_denoise(&mut image, t) };
    1
}

#[no_mangle]
pub unsafe extern "C" fn expand_quattro(
    x3f: *mut x3f_t,
    denoise: libc::c_int,
    expanded: *mut x3f_area16_t,
) -> libc::c_int {
    let mut image: x3f_area16_t = unsafe { std::mem::zeroed() };
    let mut active: x3f_area16_t = unsafe { std::mem::zeroed() };
    let mut qtop: x3f_area16_t = unsafe { std::mem::zeroed() };
    let mut qtop_crop: x3f_area16_t = unsafe { std::mem::zeroed() };
    let mut active_exp: x3f_area16_t = unsafe { std::mem::zeroed() };

    if unsafe { x3f_image_area_qtop(x3f, &mut qtop) } == 0 {
        return 0;
    }
    if unsafe { x3f_image_area(x3f, &mut image) } == 0 {
        return 0;
    }

    if denoise != 0
        && unsafe {
            x3f_crop_area_camf(
                x3f,
                c"ActiveImageArea".as_ptr() as *mut _,
                &mut image,
                1,
                &mut active,
            )
        } == 0
    {
        active = image;
        unsafe {
            x3f_printf(
                x3f_verbosity_t_WARN,
                c"Could not get active area, denoising entire image\n".as_ptr(),
            );
        }
    }

    let mut rect = [0u32; 4];
    rect[0] = 0;
    rect[1] = 0;
    rect[2] = 2 * image.columns - 1;
    rect[3] = 2 * image.rows - 1;
    if unsafe { x3f_crop_area(rect.as_mut_ptr(), &mut qtop, &mut qtop_crop) } == 0 {
        return 0;
    }

    let exp = unsafe { &mut *expanded };
    exp.columns = qtop_crop.columns;
    exp.rows = qtop_crop.rows;
    exp.channels = 3;
    exp.row_stride = exp.columns * exp.channels;
    let bytes = exp.rows as usize * exp.row_stride as usize * std::mem::size_of::<u16>();
    exp.buf = unsafe { libc::malloc(bytes) };
    exp.data = exp.buf as *mut u16;

    if denoise != 0
        && unsafe {
            x3f_crop_area_camf(
                x3f,
                c"ActiveImageArea".as_ptr() as *mut _,
                expanded,
                0,
                &mut active_exp,
            )
        } == 0
    {
        active_exp = *exp;
        unsafe {
            x3f_printf(
                x3f_verbosity_t_WARN,
                c"Could not get active area, denoising entire image\n".as_ptr(),
            );
        }
    }

    let active_ptr = if denoise != 0 {
        &mut active as *mut _
    } else {
        ptr::null_mut()
    };
    let active_exp_ptr = if denoise != 0 {
        &mut active_exp as *mut _
    } else {
        ptr::null_mut()
    };
    unsafe {
        // x3f_expand_quattro is exported by quattro.rs as #[no_mangle].
        // Re-declare with x3f_area16_t (bindgen) signature; layout
        // matches quattro.rs's #[repr(C)] Area16 struct exactly.
        extern "C" {
            fn x3f_expand_quattro(
                image: *mut x3f_area16_t,
                active: *mut x3f_area16_t,
                qtop: *mut x3f_area16_t,
                expanded: *mut x3f_area16_t,
                active_exp: *mut x3f_area16_t,
            );
        }
        x3f_expand_quattro(
            &mut image,
            active_ptr,
            &mut qtop_crop,
            expanded,
            active_exp_ptr,
        );
    }
    1
}

// ----------------------------------------------------------------------
// M6e9 — apply_highlight_clip_dng + g_dng_highlight_scale
// ----------------------------------------------------------------------
//
// DNG path replacement for convert_data's highlight reconstruction.
// Runs AFTER denoise on the post-preprocess raw plane: per-pixel
// CLUT/RepairPix/L*p/matrix-pathology gate → bake-sg-into-raw →
// scan for the global sat_ratio max → uniformly scale every raw
// value by 1/global_max so channels fit within WhiteLevel; publish
// the scale via `g_dng_highlight_scale` so the DNG writer can add
// log2(scale) to BaselineExposure (Lightroom restores brightness
// on import).

// `dng_highlight_scale` is per-image but flowed across an FFI boundary
// (Rust `apply_highlight_clip_dng` writes; x3f-core's DNG writer reads
// in `output::dng::tags`). We use a thread-local Cell so the M7d batch
// CLI parallel-iter can process multiple files concurrently without a
// race — each top-level rayon task runs to completion on a single
// worker thread, so the apply-then-read pair is consistent per-thread.
//
// The inner per-file rayon (M7a's `par_chunks_mut`) runs nested inside
// the outer task and rejoins before the post-loop scalar write below,
// so it never touches this cell.
thread_local! {
    static DNG_HIGHLIGHT_SCALE: std::cell::Cell<f64> = const { std::cell::Cell::new(1.0) };
    /// Controls the DNG-path highlight-recovery pipeline. When `false`
    /// (default), `apply_highlight_clip_dng` skips the chroma LUT, L*p
    /// reconstruction, repair_pix, and matrix-pathology gate, ships the
    /// raster within sensor-native `WhiteLevel` via a per-pixel uniform
    /// cap, and publishes `DNG_HIGHLIGHT_SCALE = 1.0` so the writer
    /// emits BaselineExposure ≈ 0. Renderers that don't honour the
    /// log2(scale) BE nudge (Capture One, Apple RAW Engine) get a
    /// self-consistent DNG that matches the pre-Rust C writer's output.
    /// When `true`, the original recovery + global_max scale-down + BE
    /// compensation runs — Adobe Camera Raw, Lightroom, and
    /// RawTherapee/LibRaw honour the BE nudge and benefit from the
    /// recovered chroma, but other renderers cast green/blue.
    static DNG_HIGHLIGHT_RECOVERY: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };

    /// Per-conversion Cineon-log TIFF toggle. Set by callers immediately
    /// before `x3f_get_image` and read by `convert_data` / `get_conv` /
    /// `x3f_get_image` to:
    ///   - replace the encoding-specific gamma LUT with a Cineon-style
    ///     log curve (`y = log(scale·x + 1) / log(scale + 1)`) so the
    ///     TIFF carries lifted shadows / pulled highlights / a flat
    ///     midtone slope, ready for grading;
    ///   - force the chroma-LUT and RepairPix highlight-recovery
    ///     passes off (they're creative interpretation, not science);
    ///   - skip `apply_highlight_clip_dng` when encoding is `NONE`
    ///     (that path is a DNG tone curve, not relevant for cineon).
    /// The matrix-pathology gate stays on regardless — it's a sanity
    /// rail against truly broken Foveon highlights, not a creative pass.
    /// Defaults to false so external C callers (which never set the
    /// hook) get the unchanged pre-cineon behaviour.
    static CINEON: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[no_mangle]
pub unsafe extern "C" fn x3f_get_dng_highlight_scale() -> f64 {
    DNG_HIGHLIGHT_SCALE.with(|c| c.get())
}

/// Toggle the DNG-path highlight-recovery pipeline. See
/// `DNG_HIGHLIGHT_RECOVERY` for the on/off behaviour.
#[no_mangle]
pub unsafe extern "C" fn x3f_set_dng_highlight_recovery(enabled: libc::c_int) {
    DNG_HIGHLIGHT_RECOVERY.with(|c| c.set(enabled != 0));
}

/// Toggle the Cineon-log TIFF processing mode. See `CINEON` for the on/off
/// behaviour.
#[no_mangle]
pub unsafe extern "C" fn x3f_set_cineon(enabled: libc::c_int) {
    CINEON.with(|c| c.set(enabled != 0));
}

const DNG_LUTSIZE: libc::c_int = LUTSIZE; // mirrors the C #define LUTSIZE 1024

// M7a — per-row context for the parallel DNG-path body. Same
// Send + Sync contract as ConvCtx; differs in that the DNG path
// always applies sgain (no `apply_sgain` toggle) and bakes sg
// into raw rather than going through the gamma LUT, so it has no
// `lut`/`ev_scale` fields.
#[derive(Copy, Clone)]
struct DngCtx {
    sgain: *mut x3f_spatial_gain_corr_t,
    sgain_num: libc::c_int,
    conv_matrix: *mut f64,
    prior: *const f64,
    hp: *const highlight_params_t,
    clut: *const chroma_lut_t,
    use_clut: bool,
    repair: *const repair_pix_t,
    use_repair: bool,
    sat_map: *const u8,
    gate_thr: f64,
    gate_width: f64,
    black: [f64; 3],
    white: [u32; 3],
    rows: i32,
    cols: i32,
    channels: usize,
    /// Per-conversion DNG highlight-recovery toggle (snapshot of the
    /// `DNG_HIGHLIGHT_RECOVERY` thread-local taken at the entry to
    /// `apply_highlight_clip_dng`).
    recovery: bool,
}

unsafe impl Send for DngCtx {}
unsafe impl Sync for DngCtx {}

#[inline(always)]
unsafe fn dng_clip_row(
    ctx: &DngCtx,
    row: i32,
    row_data: &mut [u16],
    stats: *mut chroma_lut_apply_stats_t,
) {
    // M7d — see convert_row above. The conv_matrix view is hoisted out
    // of the per-pixel loop so the matrix-pathology preview becomes an
    // inlinable native FMA chain instead of an extern "C" call.
    let m: &[f64; 9] = unsafe { &*(ctx.conv_matrix as *const [f64; 9]) };

    for col in 0..ctx.cols {
        let off = col as usize * ctx.channels;

        let mut sat_ratio = [0.0_f64; 3];
        let mut sg = [0.0_f64; 3];
        for color in 0..3 {
            let v = row_data[off + color] as f64;
            sat_ratio[color] =
                (v - ctx.black[color]) / (ctx.white[color] as f64 - ctx.black[color]);
            sg[color] = unsafe {
                x3f_calc_spatial_gain(
                    ctx.sgain,
                    ctx.sgain_num,
                    row,
                    col,
                    color as i32,
                    ctx.rows,
                    ctx.cols,
                )
            };
        }

        // CLUT or L*p fallback. (DNG path uses bare sat_ratio in
        // the chroma gate — no per-pixel sg — because sg gets
        // baked into the raw output further down. The matrix-
        // pathology preview below DOES include sg.)
        // Recovery is per-pixel gated on `ctx.recovery` so the cost
        // of dispatching is paid once per row when off, and the
        // matrix-pathology preview block below matches the same gate.
        let clut_applied = if ctx.recovery && ctx.use_clut {
            unsafe { chroma_lut_apply_pixel(sat_ratio.as_mut_ptr(), ctx.clut, stats) }
        } else {
            0
        };
        if ctx.recovery && (!ctx.use_clut || clut_applied == 0) {
            unsafe {
                reconstruct_highlights(sat_ratio.as_mut_ptr(), ctx.prior, ctx.hp);
            }
        }

        if ctx.recovery && ctx.use_repair {
            unsafe {
                repair_pix_apply_pixel(
                    sat_ratio.as_mut_ptr(),
                    ctx.prior,
                    ctx.repair,
                    ctx.sat_map,
                    row,
                    col,
                    ctx.rows,
                    ctx.cols,
                );
            }
        }

        // Matrix-pathology preview (with sg) — same as convert_data.
        // Gated together with the rest of the recovery pipeline.
        if ctx.recovery {
            let mp_in = [
                sg[0] * sat_ratio[0],
                sg[1] * sat_ratio[1],
                sg[2] * sat_ratio[2],
            ];
            let mp = mat3x1_mul_native(m, mp_in);
            let margin_g = mp[1] - if mp[0] > mp[2] { mp[0] } else { mp[2] };
            let margin_y = if mp[0] < mp[1] { mp[0] } else { mp[1] } - mp[2];
            let margin = if margin_g > margin_y {
                margin_g
            } else {
                margin_y
            };
            let mut strength = (margin - ctx.gate_thr) / ctx.gate_width;
            if strength > 0.0 {
                if strength > 1.0 {
                    strength = 1.0;
                }
                let prior = unsafe { std::slice::from_raw_parts(ctx.prior, 3) };
                let mut u_max = 0.0_f64;
                for color in 0..3 {
                    let pc = if prior[color] > 1e-12 {
                        prior[color]
                    } else {
                        1e-12
                    };
                    let u = sat_ratio[color] / pc;
                    if u > u_max {
                        u_max = u;
                    }
                }
                for color in 0..3 {
                    sat_ratio[color] =
                        (1.0 - strength) * sat_ratio[color] + strength * (u_max * prior[color]);
                }
            }
        }

        // Bake sg into the raw value written to DNG. Lightroom
        // does not honour our GainMap opcode in practice, so
        // baking sg into raw makes LR render the DNG identically
        // to our TIFF (with the GainMap opcode then suppressed
        // in the DNG writer to avoid double-application).
        //
        // When recovery is OFF we additionally apply a per-pixel
        // uniform cap at `sat_ratio_after_sg = 1.0`: if any channel
        // of `sg * sat_ratio` exceeds 1, scale all three by `1/max`
        // so the pixel sits exactly on the sensor-native WhiteLevel
        // cap. Chromaticity is preserved per-pixel; the raster is
        // strictly within WhiteLevel; downstream renderers see a
        // self-consistent DNG with no overshoot. When recovery is
        // ON we let values overshoot here — the global_max scan +
        // scale-down loop below pulls them back uniformly so the
        // BE-compensated render preserves recovered highlights.
        let v0 = sg[0] * sat_ratio[0];
        let v1 = sg[1] * sat_ratio[1];
        let v2 = sg[2] * sat_ratio[2];
        let cap_scale = if !ctx.recovery {
            let pixel_max = v0.max(v1).max(v2);
            if pixel_max > 1.0 {
                1.0 / pixel_max
            } else {
                1.0
            }
        } else {
            1.0
        };
        for (color, vc) in [v0, v1, v2].into_iter().enumerate() {
            let range = ctx.white[color] as f64 - ctx.black[color];
            let v = vc * cap_scale * range + ctx.black[color];
            let out = v.round() as i32;
            row_data[off + color] = if out < 0 {
                0
            } else if out > 65535 {
                65535
            } else {
                out as u16
            };
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn apply_highlight_clip_dng(
    x3f: *mut x3f_t,
    image: *mut x3f_area16_t,
    ilevels: *mut x3f_image_levels_t,
    wb: *mut libc::c_char,
) {
    let img = unsafe { &mut *image };
    if img.channels < 3 {
        return;
    }

    // Matrix-derived prior + matrix-pathology preview both need
    // conv_matrix; build it via the sRGB path. The prior direction
    // is independent of the output xyz_to_rgb as long as the white
    // point matches (sRGB and raw_to_xyz are both D65).
    let mut conv_matrix = [0.0_f64; 9];
    let mut lut_dummy = [0.0_f64; DNG_LUTSIZE as usize];
    if unsafe {
        get_conv(
            x3f,
            x3f_color_encoding_e_SRGB,
            wb,
            DNG_LUTSIZE,
            65535,
            lut_dummy.as_mut_ptr(),
            conv_matrix.as_mut_ptr(),
        )
    } == 0
    {
        return;
    }

    // Spatial gain — same as convert_data uses. We don't bake sg
    // into DNG raw values (Adobe applies via OpcodeList2 GainMap;
    // disabled in our writer per the comment block below). But sg
    // IS applied to the matrix-pathology preview so the gate
    // predicts the same chromaticity Adobe will see post-shading.
    let mut sgain: [x3f_spatial_gain_corr_t; MAX_CORR] = unsafe { std::mem::zeroed() };
    let sgain_num = unsafe { x3f_get_spatial_gain(x3f, wb, sgain.as_mut_ptr()) };

    let mut hp: highlight_params_t = unsafe { std::mem::zeroed() };
    let mut prior = [0.0_f64; 3];
    unsafe {
        get_highlight_params(x3f, &mut hp);
        compute_chroma_prior(conv_matrix.as_ptr(), prior.as_mut_ptr());
    }

    let mut clut: chroma_lut_t = unsafe { std::mem::zeroed() };
    let mut use_clut = !env_present("X3F_NO_CHROMA_LUT");
    if use_clut {
        unsafe { chroma_lut_init_defaults(&mut clut) };
        if unsafe { chroma_lut_build_from_image(&mut clut, img, ilevels, prior.as_ptr()) } == 0 {
            use_clut = false;
        }
    }
    let mut clut_stats: chroma_lut_apply_stats_t = unsafe { std::mem::zeroed() };
    let trace_clut = env_present("X3F_CHROMA_LUT_TRACE");

    let mut repair: repair_pix_t = unsafe { std::mem::zeroed() };
    let mut sat_map: *mut u8 = ptr::null_mut();
    let mut use_repair = env_present("X3F_REPAIR_PIX");
    if use_repair {
        unsafe { repair_pix_init_defaults(&mut repair) };
        sat_map = unsafe { build_sat_map(img, ilevels, repair.sat_threshold) };
        if sat_map.is_null() {
            use_repair = false;
        } else {
            repair.valid = 1;
        }
    }

    let mut gate_thr = 0.20_f64;
    let mut gate_width = 0.30_f64;
    if let Some(v) = env_atof("X3F_GATE_THR") {
        gate_thr = v;
    }
    if let Some(v) = env_atof("X3F_GATE_WIDTH") {
        gate_width = v;
    }
    if gate_width < 1e-6 {
        gate_width = 1e-6;
    }

    let row_stride = img.row_stride as usize;
    let channels = img.channels as usize;
    let il = unsafe { &mut *ilevels };

    let recovery = DNG_HIGHLIGHT_RECOVERY.with(|c| c.get());
    let dctx = DngCtx {
        sgain: sgain.as_mut_ptr(),
        sgain_num,
        conv_matrix: conv_matrix.as_mut_ptr(),
        prior: prior.as_ptr(),
        hp: &hp,
        clut: &clut,
        use_clut,
        repair: &repair,
        use_repair,
        sat_map,
        gate_thr,
        gate_width,
        black: il.black,
        white: il.white,
        rows: img.rows as i32,
        cols: img.columns as i32,
        channels,
        recovery,
    };

    let total_u16 = (img.rows as usize) * row_stride;
    let data = unsafe { std::slice::from_raw_parts_mut(img.data, total_u16) };

    // Pass 1: per-pixel highlight recovery + bake sg into raw.
    if trace_clut {
        let stats_ptr = &mut clut_stats as *mut _;
        for (row, row_data) in data.chunks_mut(row_stride).enumerate() {
            unsafe { dng_clip_row(&dctx, row as i32, row_data, stats_ptr) };
        }
    } else {
        use rayon::prelude::*;
        data.par_chunks_mut(row_stride)
            .enumerate()
            .for_each(|(row, row_data)| {
                unsafe { dng_clip_row(&dctx, row as i32, row_data, ptr::null_mut()) };
            });
    }

    // Pass 2 + Pass 3 only run when recovery is ON. With recovery OFF,
    // Pass 1 already capped each pixel at sat_ratio ≤ 1 via the
    // per-pixel uniform cap, so the raster is strictly within
    // sensor-native WhiteLevel and we publish DNG_HIGHLIGHT_SCALE = 1
    // (the writer omits the BaselineExposure log2 nudge).
    //
    // With recovery ON, recovered pixels overshoot WhiteLevel; Pass 2
    // scans for the global max sat_ratio so Pass 3 can divide-down
    // uniformly, and the writer adds `log2(global_max)` to
    // BaselineExposure so Lightroom / ACR / RawTherapee restore the
    // brightness on import — the recovered highlight detail unfolds
    // back to its captured luminance via the renderer's tone curve.
    let mut global_max = 1.0_f64;
    if recovery {
        global_max = {
            use rayon::prelude::*;
            let black = il.black;
            let white = il.white;
            let cols = img.columns as usize;
            data.par_chunks(row_stride)
                .map(|row_data| {
                    let mut m = 1.0_f64;
                    for col in 0..cols {
                        let off = col * channels;
                        for color in 0..3 {
                            let v = row_data[off + color] as f64;
                            let range = white[color] as f64 - black[color];
                            let sr = (v - black[color]) / range;
                            if sr > m {
                                m = sr;
                            }
                        }
                    }
                    m
                })
                .reduce(|| 1.0_f64, f64::max)
        };
        if global_max > 1.0 {
            use rayon::prelude::*;
            let inv = 1.0 / global_max;
            let black = il.black;
            let white = il.white;
            let cols = img.columns as usize;
            data.par_chunks_mut(row_stride).for_each(|row_data| {
                for col in 0..cols {
                    let off = col * channels;
                    for color in 0..3 {
                        let v = row_data[off + color] as f64;
                        let range = white[color] as f64 - black[color];
                        let sr = (v - black[color]) / range * inv;
                        let out = (sr * range + black[color]).round() as i32;
                        row_data[off + color] = if out < 0 {
                            0
                        } else if out > 65535 {
                            65535
                        } else {
                            out as u16
                        };
                    }
                }
            });
        }
    }

    unsafe { x3f_cleanup_spatial_gain(sgain.as_mut_ptr(), sgain_num) };
    if !sat_map.is_null() {
        unsafe { libc::free(sat_map as *mut libc::c_void) };
    }
    if trace_clut {
        unsafe { chroma_lut_apply_stats_print(&clut_stats, c"dng".as_ptr()) };
    }

    // Publish the highlight scale on this thread *last*. Any nested
    // rayon (the par_chunks_mut above) could have work-stolen another
    // file's apply_highlight_clip_dng onto this thread, clobbering the
    // cell mid-stream — so we set after Pass 3 (and after every other
    // op that could re-enter rayon) to guarantee that the last write
    // on T's cell from this stack frame is *this* file's value. See
    // x3f-core::image::Reader::get_image which snapshots the cell
    // immediately after this returns.
    //
    // With recovery ON, this is `global_max` so the writer adds
    // log2(global_max) to BaselineExposure (Lightroom/ACR/RawTherapee
    // pull recovered highlights back via the BE nudge). With recovery
    // OFF the raster is already within WhiteLevel, so we publish 1.0
    // and the writer emits BE = log2(captureISO/sensorISO) only.
    DNG_HIGHLIGHT_SCALE.with(|c| c.set(if recovery { global_max } else { 1.0 }));
}

// ----------------------------------------------------------------------
// M6e10 — public entry points (x3f_get_image, x3f_get_preview)
// ----------------------------------------------------------------------

#[no_mangle]
pub unsafe extern "C" fn x3f_get_image(
    x3f: *mut x3f_t,
    image: *mut x3f_area16_t,
    ilevels: *mut x3f_image_levels_t,
    encoding: x3f_color_encoding_t,
    crop: libc::c_int,
    fix_bad: libc::c_int,
    denoise: libc::c_int,
    apply_sgain: libc::c_int,
    mut wb: *mut libc::c_char,
) -> libc::c_int {
    if wb.is_null() {
        wb = unsafe { x3f_get_wb(x3f) };
    }

    if encoding == x3f_color_encoding_e_QTOP {
        let mut qtop: x3f_area16_t = unsafe { std::mem::zeroed() };
        if unsafe { x3f_image_area_qtop(x3f, &mut qtop) } == 0 {
            return 0;
        }
        if crop == 0
            || unsafe {
                x3f_crop_area_camf(
                    x3f,
                    c"ActiveImageArea".as_ptr() as *mut _,
                    &mut qtop,
                    0,
                    image,
                )
            } == 0
        {
            unsafe { *image = qtop };
        }
        return (ilevels.is_null()) as libc::c_int;
    }

    let mut original_image: x3f_area16_t = unsafe { std::mem::zeroed() };
    if unsafe { x3f_image_area(x3f, &mut original_image) } == 0 {
        return 0;
    }
    if crop == 0
        || unsafe {
            x3f_crop_area_camf(
                x3f,
                c"ActiveImageArea".as_ptr() as *mut _,
                &mut original_image,
                1,
                image,
            )
        } == 0
    {
        unsafe { *image = original_image };
    }

    if encoding == x3f_color_encoding_e_UNPROCESSED {
        return (ilevels.is_null()) as libc::c_int;
    }

    let mut il: x3f_image_levels_t = unsafe { std::mem::zeroed() };
    if unsafe { preprocess_data(x3f, fix_bad, wb, &mut il) } == 0 {
        return 0;
    }

    let mut is_quattro = false;
    let mut expanded: x3f_area16_t = unsafe { std::mem::zeroed() };
    if unsafe { expand_quattro(x3f, denoise, &mut expanded) } != 0 {
        // NOTE: expand_quattro destroys the data of original_image
        if crop == 0
            || unsafe {
                x3f_crop_area_camf(
                    x3f,
                    c"ActiveImageArea".as_ptr() as *mut _,
                    &mut expanded,
                    0,
                    image,
                )
            } == 0
        {
            unsafe { *image = expanded };
        }
        original_image = expanded;
        is_quattro = true;
    } else if denoise != 0 && unsafe { run_denoising(x3f) } == 0 {
        return 0;
    }

    // `apply_highlight_clip_dng` is the DNG path's clip-curve renderer.
    // Cineon-log callers using `-color none` (camera-native log) want
    // the log curve baked into raw BMT samples without the DNG tone
    // curve fighting it — skip it for them, just like the Quattro
    // branch already does.
    let cineon = CINEON.with(|c| c.get());
    if encoding == x3f_color_encoding_e_NONE && !is_quattro && !cineon {
        unsafe { apply_highlight_clip_dng(x3f, image, &mut il, wb) };
    }

    if encoding != x3f_color_encoding_e_NONE
        && unsafe { convert_data(x3f, &mut original_image, &mut il, encoding, apply_sgain, wb) }
            == 0
    {
        unsafe { libc::free((*image).buf) };
        return 0;
    }

    if !ilevels.is_null() {
        unsafe { *ilevels = il };
    }
    1
}

#[no_mangle]
pub unsafe extern "C" fn x3f_get_preview(
    x3f: *mut x3f_t,
    image: *mut x3f_area16_t,
    ilevels: *mut x3f_image_levels_t,
    encoding: x3f_color_encoding_t,
    apply_sgain: libc::c_int,
    wb: *mut libc::c_char,
    max_width: u32,
    preview: *mut x3f_area8_t,
) -> libc::c_int {
    let max_out: u16 = 255;
    let img = unsafe { &mut *image };
    if img.channels < 3 {
        return 0;
    }

    let mut conv_matrix = [0.0_f64; 9];
    let mut lut = [0.0_f64; LUTSIZE as usize];
    if unsafe {
        get_conv(
            x3f,
            encoding,
            wb,
            LUTSIZE,
            max_out,
            lut.as_mut_ptr(),
            conv_matrix.as_mut_ptr(),
        )
    } == 0
    {
        return 0;
    }

    let mut sgain: [x3f_spatial_gain_corr_t; MAX_CORR] = unsafe { std::mem::zeroed() };
    let sgain_num = if apply_sgain != 0 {
        let n = unsafe { x3f_get_spatial_gain(x3f, wb, sgain.as_mut_ptr()) };
        if n == 0 {
            unsafe {
                x3f_printf(
                    x3f_verbosity_t_WARN,
                    c"Could not get spatial gain\n".as_ptr(),
                );
            }
        }
        n
    } else {
        0
    };

    let reduction = ((img.columns + max_width - 1) / max_width) as usize;
    let reduction2 = (reduction * reduction) as f64;
    let pv = unsafe { &mut *preview };
    pv.columns = img.columns / reduction as u32;
    pv.rows = img.rows / reduction as u32;
    pv.channels = 3;
    pv.row_stride = pv.columns * pv.channels;

    let bytes = pv.rows as usize * pv.row_stride as usize;
    let alloc = unsafe { libc::malloc(bytes) };
    pv.buf = alloc;
    pv.data = alloc as *mut u8;

    let img_row_stride = img.row_stride as usize;
    let img_channels = img.channels as usize;
    let pv_row_stride = pv.row_stride as usize;
    let pv_channels = pv.channels as usize;
    let il = unsafe { &*ilevels };

    for row in 0..pv.rows as i32 {
        for col in 0..pv.columns as i32 {
            let mut input = [0.0_f64; 3];
            for color in 0..3 {
                let mut acc: u32 = 0;
                for r in 0..reduction {
                    for c in 0..reduction {
                        let idx = img_row_stride * (row as usize * reduction + r)
                            + img_channels * (col as usize * reduction + c)
                            + color;
                        acc += unsafe { *img.data.add(idx) } as u32;
                    }
                }
                let sg = unsafe {
                    x3f_calc_spatial_gain(
                        sgain.as_mut_ptr(),
                        sgain_num,
                        row,
                        col,
                        color as i32,
                        pv.rows as i32,
                        pv.columns as i32,
                    )
                };
                input[color] = sg * ((acc as f64) / reduction2 - il.black[color])
                    / (il.white[color] as f64 - il.black[color]);
            }

            let mut output = [0.0_f64; 3];
            unsafe {
                x3f_3x3_3x1_mul(
                    conv_matrix.as_mut_ptr(),
                    input.as_mut_ptr(),
                    output.as_mut_ptr(),
                );
            }
            for color in 0..3 {
                let v = unsafe { x3f_LUT_lookup(lut.as_mut_ptr(), LUTSIZE, output[color]) };
                let idx = pv_row_stride * row as usize + pv_channels * col as usize + color;
                unsafe { *pv.data.add(idx) = v as u8 };
            }
        }
    }

    unsafe { x3f_cleanup_spatial_gain(sgain.as_mut_ptr(), sgain_num) };

    unsafe {
        x3f_crop_area8_camf(
            x3f,
            c"ActiveImageArea".as_ptr() as *mut _,
            preview,
            1,
            preview,
        );
    }

    1
}

// ----------------------------------------------------------------------
// Symbol anchors so cross-crate dead-code elimination can't strip the
// Rust definitions before the still-C call sites in x3f_process.c link.
// ----------------------------------------------------------------------

#[used]
static _A_GET_GAIN: unsafe extern "C" fn(*mut x3f_t, *mut libc::c_char, *mut f64) -> libc::c_int =
    x3f_get_gain;
#[used]
static _A_GET_BMT_TO_XYZ: unsafe extern "C" fn(
    *mut x3f_t,
    *mut libc::c_char,
    *mut f64,
) -> libc::c_int = x3f_get_bmt_to_xyz;
#[used]
static _A_GET_RAW_TO_XYZ: unsafe extern "C" fn(
    *mut x3f_t,
    *mut libc::c_char,
    *mut f64,
) -> libc::c_int = x3f_get_raw_to_xyz;
#[used]
static _A_GET_BLACK_LEVEL: unsafe extern "C" fn(
    *mut x3f_t,
    *mut x3f_area16_t,
    libc::c_int,
    libc::c_int,
    *mut f64,
    *mut f64,
) -> libc::c_int = get_black_level;
#[used]
static _A_GET_MAX_INTERMEDIATE: unsafe extern "C" fn(
    *mut x3f_t,
    *mut libc::c_char,
    f64,
    *mut u32,
) -> libc::c_int = get_max_intermediate;
#[used]
static _A_GET_INTERMEDIATE_BIAS: unsafe extern "C" fn(
    *mut x3f_t,
    *mut libc::c_char,
    *mut f64,
    *mut f64,
    *mut f64,
) -> libc::c_int = get_intermediate_bias;
#[used]
static _A_INTERPOLATE_BAD_PIXELS: unsafe extern "C" fn(*mut x3f_t, *mut x3f_area16_t, libc::c_int) =
    interpolate_bad_pixels;
#[used]
static _A_APPLY_WB_COLOR_SHADING: unsafe extern "C" fn(
    *mut x3f_t,
    *mut libc::c_char,
    *mut x3f_area16_t,
) -> libc::c_int = apply_wb_color_shading;
#[used]
static _A_PREPROCESS_DATA: unsafe extern "C" fn(
    *mut x3f_t,
    libc::c_int,
    *mut libc::c_char,
    *mut x3f_image_levels_t,
) -> libc::c_int = preprocess_data;
#[used]
static _A_GET_CONV: unsafe extern "C" fn(
    *mut x3f_t,
    x3f_color_encoding_t,
    *mut libc::c_char,
    libc::c_int,
    u16,
    *mut f64,
    *mut f64,
) -> libc::c_int = get_conv;
#[used]
static _A_CONVERT_DATA: unsafe extern "C" fn(
    *mut x3f_t,
    *mut x3f_area16_t,
    *mut x3f_image_levels_t,
    x3f_color_encoding_t,
    libc::c_int,
    *mut libc::c_char,
) -> libc::c_int = convert_data;
#[used]
static _A_RUN_DENOISING: unsafe extern "C" fn(*mut x3f_t) -> libc::c_int = run_denoising;
#[used]
static _A_EXPAND_QUATTRO: unsafe extern "C" fn(
    *mut x3f_t,
    libc::c_int,
    *mut x3f_area16_t,
) -> libc::c_int = expand_quattro;
#[used]
static _A_APPLY_HIGHLIGHT_CLIP_DNG: unsafe extern "C" fn(
    *mut x3f_t,
    *mut x3f_area16_t,
    *mut x3f_image_levels_t,
    *mut libc::c_char,
) = apply_highlight_clip_dng;
#[used]
static _A_GET_DNG_HIGHLIGHT_SCALE: unsafe extern "C" fn() -> f64 = x3f_get_dng_highlight_scale;
#[used]
static _A_SET_DNG_HIGHLIGHT_RECOVERY: unsafe extern "C" fn(libc::c_int) =
    x3f_set_dng_highlight_recovery;
#[used]
static _A_SET_CINEON: unsafe extern "C" fn(libc::c_int) = x3f_set_cineon;
#[used]
static _A_X3F_GET_IMAGE: unsafe extern "C" fn(
    *mut x3f_t,
    *mut x3f_area16_t,
    *mut x3f_image_levels_t,
    x3f_color_encoding_t,
    libc::c_int,
    libc::c_int,
    libc::c_int,
    libc::c_int,
    *mut libc::c_char,
) -> libc::c_int = x3f_get_image;
#[used]
static _A_X3F_GET_PREVIEW: unsafe extern "C" fn(
    *mut x3f_t,
    *mut x3f_area16_t,
    *mut x3f_image_levels_t,
    x3f_color_encoding_t,
    libc::c_int,
    *mut libc::c_char,
    u32,
    *mut x3f_area8_t,
) -> libc::c_int = x3f_get_preview;
