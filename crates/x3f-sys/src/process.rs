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

use crate::Control;

use std::ffi::CStr;
use std::ptr;

use crate::*;
// libc compat — see `sysabi.rs`.
use crate::highlight_recovery::{channel_reliability, LocalRecovery, SensorReliability};
use crate::sysabi as libc;

mod spp;

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

unsafe fn merrill_trimmed_black(
    area: &x3f_area16_t,
    control: Control<'_>,
) -> crate::Result<[f64; 3]> {
    let count = area.rows as usize * area.columns as usize;
    if count == 0 || area.channels < 3 {
        return Err(crate::Error::InvalidData("empty Merrill dark shield"));
    }
    let mut samples = Vec::with_capacity(count);
    let mut black = [0.0; 3];
    let trim = count / 100;
    for (channel, level) in black.iter_mut().enumerate() {
        samples.clear();
        for row in 0..area.rows as usize {
            control.check()?;
            for col in 0..area.columns as usize {
                let index = row * area.row_stride as usize + col * area.channels as usize + channel;
                samples.push(unsafe { *area.data.add(index) } as i16);
            }
        }
        samples.sort_unstable();
        let kept = &samples[trim..count - trim];
        *level = kept.iter().map(|&v| f64::from(v)).sum::<f64>() / kept.len() as f64;
    }
    Ok(black)
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

    let mut digital_gain = [1.0; 3];
    unsafe { x3f_get_digital_iso_gain(x3f, digital_gain.as_mut_ptr()) };
    let levels = intermediate_levels(gain, digital_gain, intermediate_bias);
    if levels == [0; 3] {
        unsafe {
            x3f_printf(
                x3f_verbosity_t_ERR,
                c"Invalid DigitalISOGain value(s); expected finite positive channel gains\n"
                    .as_ptr(),
            );
        }
        return 0;
    }

    unsafe { std::slice::from_raw_parts_mut(max_intermediate, 3) }.copy_from_slice(&levels);
    1
}

fn intermediate_levels(
    mut gain: [f64; 3],
    digital_gain: [f64; 3],
    intermediate_bias: f64,
) -> [u32; 3] {
    if !digital_gain
        .iter()
        .all(|&gain| gain.is_finite() && gain > 0.0)
    {
        return [0; 3];
    }

    // Quattro's Yis4T upsampler adds the same high-frequency T detail to
    // all three intermediate channels. Those counts must therefore have
    // a neutral white-balanced basis, including relative DigitalISOGain.
    // This changes the encoding ranges, not normalized raw values: the
    // same range is used by preprocessing and its later inverse.
    // Uniform digital gain cancels from the normalization; skip it to
    // preserve the existing arithmetic exactly for those cameras.
    if digital_gain[0] != digital_gain[1] || digital_gain[1] != digital_gain[2] {
        for color in 0..3 {
            gain[color] *= digital_gain[color];
        }
    }

    // Cap the gains to 1.0 to avoid clipping (i.e. divide by max).
    let mut maxgain = 0.0_f64;
    for &g in &gain {
        if g > maxgain {
            maxgain = g;
        }
    }
    std::array::from_fn(|i| {
        let v =
            gain[i] * (INTERMEDIATE_UNIT as f64 - intermediate_bias) / maxgain + intermediate_bias;
        v.round() as i32 as u32
    })
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
    unsafe { interpolate_bad_pixels_controlled(x3f, image, colors, Control::none()) }
}

unsafe fn interpolate_bad_pixels_controlled(
    x3f: *mut x3f_t,
    image: *mut x3f_area16_t,
    colors: libc::c_int,
    control: Control<'_>,
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
                    if control.check().is_err() {
                        return;
                    }
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
                if control.check().is_err() {
                    return;
                }
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
                if control.check().is_err() {
                    return;
                }
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
        if hp_ok
            && hpinfo[..2].iter().all(|&v| v <= i32::MAX as u32)
            && hpinfo[2..]
                .iter()
                .all(|&v| (1..=i32::MAX as u32).contains(&v))
        {
            // Camera metadata must not create a zero/negative stride or an
            // overflowing final increment. Range stepping bounds both axes.
            for row in (hpinfo[1] as i32..rs).step_by(hpinfo[3] as usize) {
                if control.check().is_err() {
                    return;
                }
                for col in (hpinfo[0] as i32..cs).step_by(hpinfo[2] as usize) {
                    if control.check().is_err() {
                        return;
                    }
                    mark_pix(&mut bad_pixels, &mut bad_pixel_vec, col, row, cs, rs);
                }
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
                if control.check().is_err() {
                    return;
                }
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
                if control.check().is_err() {
                    return;
                }
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
        if control.check().is_err() {
            return;
        }
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

pub(crate) unsafe fn merrill_count_scale(x3f: *mut x3f_t) -> Option<f64> {
    // X3F 4+ stores a process version; model alone cannot select its gain.
    if !matches!(unsafe { (*x3f).header.version }, 0x0002_0003 | 0x0003_0000) {
        return None;
    }
    let mut model = ptr::null_mut();
    if unsafe { x3f_get_prop_entry(x3f, c"CAMMODEL".as_ptr() as *mut _, &mut model) } == 0
        || model.is_null()
    {
        return None;
    }
    // GatherParams @0xb312c: SD1 process 2.0.x divides WB by 0.55; DP doubles it.
    match unsafe { CStr::from_ptr(model) }.to_bytes() {
        b"SIGMA SD1 Merrill" => Some(1.0 / 0.55),
        b"SIGMA DP1 Merrill" | b"SIGMA DP2 Merrill" | b"SIGMA DP3 Merrill" => Some(2.0),
        _ => None,
    }
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
    unsafe { apply_wb_color_shading_controlled(x3f, wb, image, Control::none()) }
}

unsafe fn apply_wb_color_shading_controlled(
    x3f: *mut x3f_t,
    wb: *mut libc::c_char,
    image: *mut x3f_area16_t,
    control: Control<'_>,
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
            if control.check().is_err() {
                return;
            }
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
//   4. Per-pixel: out = scale * (raw - black) + intermediate_bias,
//      clamped to [0, 65535] uint16. DigitalISOGain is deliberately
//      NOT part of `scale` — it lives in get_conv's matrix and the
//      DNG BaselineExposure so sensor saturation maps to `il.white`
//   5. For Quattro: downsample top16 4-pixel sums into image[2], then
//      preprocess top16 full resolution with the same rule
//   6. WB-conditional radial color shading (M6e5; Merrill-only)
//   7. Bad-pixel interpolation (M6e3) — twice on Quattro: top first,
//      then image

#[no_mangle]
pub unsafe extern "C" fn preprocess_data(
    x3f: *mut x3f_t,
    fix_bad: libc::c_int,
    wb: *mut libc::c_char,
    ilevels: *mut x3f_image_levels_t,
) -> libc::c_int {
    unsafe {
        preprocess_data_impl(x3f, fix_bad, wb, ilevels, None, None, None, Control::none())
            .unwrap_or(0)
    }
}

unsafe fn snapshot_merge_plane(
    area: &x3f_area16_t,
    width: u32,
    height: u32,
    channels: usize,
    control: Control<'_>,
) -> crate::Result<Vec<u16>> {
    if area.data.is_null()
        || width == 0
        || height == 0
        || width > area.columns
        || height > area.rows
        || channels > area.channels as usize
        || width as usize * area.channels as usize > area.row_stride as usize
    {
        return Err(crate::Error::InvalidData("invalid merge raw plane"));
    }
    let length = (width as usize)
        .checked_mul(height as usize)
        .and_then(|n| n.checked_mul(channels))
        .ok_or(crate::Error::Allocation)?;
    let mut output = Vec::new();
    output
        .try_reserve_exact(length)
        .map_err(|_| crate::Error::Allocation)?;
    for row in 0..height as usize {
        control.check()?;
        for col in 0..width as usize {
            let off = row * area.row_stride as usize + col * area.channels as usize;
            let pixel = unsafe { std::slice::from_raw_parts(area.data.add(off), channels) };
            output.extend_from_slice(pixel);
        }
    }
    Ok(output)
}

// Provenance belongs to one conversion, never to the C image layout or a
// thread-local slot that nested Rayon work could overwrite.
unsafe fn preprocess_data_impl(
    x3f: *mut x3f_t,
    fix_bad: libc::c_int,
    wb: *mut libc::c_char,
    ilevels: *mut x3f_image_levels_t,
    provenance: Option<&mut Option<SensorReliability>>,
    merrill_black: Option<[f64; 3]>,
    merge: Option<&mut MergeFrame>,
    control: Control<'_>,
) -> crate::Result<libc::c_int> {
    if control.check().is_err() {
        return Ok(0);
    }
    let mut image: x3f_area16_t = unsafe { std::mem::zeroed() };
    let mut qtop: x3f_area16_t = unsafe { std::mem::zeroed() };

    let quattro = unsafe { x3f_image_area_qtop(x3f, &mut qtop) } != 0;
    let colors_in = if quattro { 2 } else { 3 };

    if unsafe { x3f_image_area(x3f, &mut image) } == 0 || image.channels < 3 {
        return Ok(0);
    }
    if quattro
        && (qtop.channels < 1 || qtop.rows < 2 * image.rows || qtop.columns < 2 * image.columns)
    {
        return Ok(0);
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
        return Ok(0);
    }
    if let Some(level) = merrill_black {
        black_level = level;
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
        return Ok(0);
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

    let render_gain = if merrill_black.is_some() {
        match unsafe { merrill_wb_gain(x3f, wb) } {
            Some(gain) => Some(gain),
            None => return Ok(0),
        }
    } else {
        None
    };
    let mut intermediate_bias: f64 = 0.0;
    let bias_ok = if let Some(gain) = render_gain {
        let levels = intermediate_levels(gain, [1.0; 3], 0.0);
        if levels == [0; 3] {
            return Ok(0);
        }
        for c in 0..3 {
            intermediate_bias = intermediate_bias.max(
                INTERMEDIATE_BIAS_FACTOR * black_dev[c] * f64::from(levels[c])
                    / (f64::from(max_raw[c]) - black_level[c]),
            );
        }
        1
    } else {
        unsafe {
            get_intermediate_bias(
                x3f,
                wb,
                black_level.as_mut_ptr(),
                black_dev.as_mut_ptr(),
                &mut intermediate_bias,
            )
        }
    };
    if bias_ok == 0 {
        unsafe {
            x3f_printf(
                x3f_verbosity_t_ERR,
                c"Could not get intermediate bias\n".as_ptr(),
            );
        }
        return Ok(0);
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

    let levels_ok = if let Some(gain) = render_gain {
        il.white = intermediate_levels(gain, [1.0; 3], intermediate_bias);
        i32::from(il.white != [0; 3])
    } else {
        unsafe { get_max_intermediate(x3f, wb, intermediate_bias, il.white.as_mut_ptr()) }
    };
    if levels_ok == 0 {
        unsafe {
            x3f_printf(
                x3f_verbosity_t_ERR,
                c"Could not get maximum intermediate level\n".as_ptr(),
            );
        }
        return Ok(0);
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

    // Intermediate ranges are proportional to WhiteBalanceGains × DigitalISOGain,
    // with the same relative gain used by the inverse normalization in
    // preprocess_data so the final normalized sample domain stays unchanged.
    // The geometric-mean component goes to BaselineExposure; only the relative
    // gain is visible in per-channel pre-scaling and therefore in both
    // recovered highlights and raw sample ratios.
    let mut scale = [0.0_f64; 3];
    for color in 0..3 {
        scale[color] = (il.white[color] as f64 - il.black[color])
            / (max_raw[color] as f64 - black_level[color]);
    }

    // Preprocess image data (HUF/TRU -> x3rgb16). Per pixel:
    //   out = round(scale * (raw - black_level) + ilevels.black)
    let img_row_stride = image.row_stride as usize;
    let img_channels = image.channels as usize;
    let img_cols = image.columns as usize;
    let il_black = il.black;
    if image.rows == 0 || img_cols == 0 || img_row_stride == 0 {
        return Ok(0);
    }
    let mut reliability = if provenance.is_some() && !quattro {
        let noise = std::array::from_fn(|c| {
            black_dev[c].abs() / (max_raw[c] as f64 - black_level[c]).max(1.0)
        });
        // Provenance is optional: allocation failure falls back to the
        // established recovery instead of aborting an otherwise valid image.
        SensorReliability::new(image.rows as usize, img_cols, noise)
    } else {
        None
    };
    {
        use rayon::prelude::*;
        let total = image.rows as usize * img_row_stride;
        let img_data = unsafe { std::slice::from_raw_parts_mut(image.data, total) };
        let colors = colors_in as usize;
        let normalize_row = |row_data: &mut [u16], mut mask: Option<&mut [[u8; 3]]>| {
            if control.check().is_err() {
                return;
            }
            for col in 0..img_cols {
                let off = img_channels * col;
                for color in 0..colors {
                    let v = row_data[off + color] as f64;
                    if let Some(mask) = mask.as_deref_mut() {
                        let normalized =
                            (v - black_level[color]) / (max_raw[color] as f64 - black_level[color]);
                        mask[col][color] = channel_reliability(normalized, 0.0, 0.99, 0.04);
                    }
                    let out =
                        (scale[color] * (v - black_level[color]) + il_black[color]).round() as i32;
                    row_data[off + color] = if out < 0 {
                        0
                    } else if out > 65535 {
                        65535
                    } else {
                        out as u16
                    };
                }
            }
        };
        if let Some(mask) = reliability.as_mut() {
            img_data
                .par_chunks_mut(img_row_stride)
                .zip(mask.data.par_chunks_mut(img_cols))
                .for_each(|(row_data, row_mask)| normalize_row(row_data, Some(row_mask)));
        } else {
            img_data
                .par_chunks_mut(img_row_stride)
                .for_each(|row_data| normalize_row(row_data, None));
        }
    }

    if let Some(mask) = reliability.as_mut() {
        // Decode against full stored-raster coordinates while the original
        // clipping evidence still precedes WBCS, denoise, and crop. This
        // changes only optional DNG recovery provenance, never the pixels.
        unsafe { mask.merge_merrill_camera_clipping(x3f) };
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
                if control.check().is_err() {
                    return;
                }
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
            if control.check().is_err() {
                return;
            }
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
            unsafe { interpolate_bad_pixels_controlled(x3f, &mut qtop, 1, control) };
        }
    }

    // WB-conditional radial color shading (Merrill-only; the function
    // returns 0 for non-Merrill bodies, and the caller ignores that).
    unsafe { apply_wb_color_shading_controlled(x3f, wb, &mut image, control) };

    // Shading can reach the storage rail even when the sensor did not clip.
    // Denoise or bad-pixel interpolation must not turn that lost measurement
    // back into a supposedly reliable source sample.
    if let Some(mask) = reliability.as_mut() {
        use rayon::prelude::*;
        let data =
            unsafe { std::slice::from_raw_parts(image.data, image.rows as usize * img_row_stride) };
        mask.data
            .par_chunks_mut(img_cols)
            .zip(data.par_chunks(img_row_stride))
            .for_each(|(mask_row, row)| {
                if control.check().is_err() {
                    return;
                }
                for col in 0..img_cols {
                    for c in 0..3 {
                        if row[col * img_channels + c] == u16::MAX {
                            mask_row[col][c] = 0;
                        }
                    }
                }
            });
    }

    if fix_bad != 0 {
        unsafe { interpolate_bad_pixels_controlled(x3f, &mut image, 3, control) };
    }

    if let Some(merge) = merge {
        merge.raw_half_width = image.columns;
        merge.raw_half_height = image.rows;
        merge.raw_black = il.black;
        merge.raw_white = il.white.map(f64::from);
        merge.sigma = std::array::from_fn(|c| {
            (black_dev[c].abs() / (max_raw[c] as f64 - black_level[c])) as f32
        });
        // Capture the normalized and repaired planes that expansion will read.
        merge.raw_half =
            unsafe { snapshot_merge_plane(&image, image.columns, image.rows, 2, control) }?;
        merge.raw_top =
            unsafe { snapshot_merge_plane(&qtop, 2 * image.columns, 2 * image.rows, 1, control) }?;
    }

    if let Some(output) = provenance {
        *output = reliability;
    }

    Ok(control.check().is_ok() as libc::c_int)
}

// ----------------------------------------------------------------------
// M6e7 — get_conv + convert_data hot loop
// ----------------------------------------------------------------------

const LUTSIZE: libc::c_int = 1024;
const MAX_CORR: usize = 6; // x3f_spatial_gain.h MAXCORR

/// Per-channel CAMF `DigitalISOGain`. Always fills `gain[0..3]`:
/// `(1, 1, 1)` for Merrill bodies (the legacy comment cites "color
/// cast issues") and when the CAMF entry is absent. Returns 1 iff a
/// real CAMF entry was used. Single source of truth — get_conv folds
/// the gain into the conversion matrix and the DNG writer carries it
/// in BaselineExposure; it must never be baked into raw samples (see
/// preprocess_data).
#[no_mangle]
pub unsafe extern "C" fn x3f_get_digital_iso_gain(x3f: *mut x3f_t, gain: *mut f64) -> libc::c_int {
    let mut digital_iso_gain = [1.0_f64; 3];
    let mut used_camf = 0;
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
        used_camf = 1;
    } else {
        digital_iso_gain = [1.0; 3];
    }
    unsafe { std::slice::from_raw_parts_mut(gain, 3) }.copy_from_slice(&digital_iso_gain);
    used_camf
}

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
    unsafe {
        get_conv_controlled(
            x3f,
            encoding,
            wb,
            lutsize,
            max_out,
            lut,
            conv_matrix,
            CINEON.with(|c| c.get()),
        )
    }
}

// Merrill GatherParams @0xb2d14..0xb2de8 uses float32 ISO arithmetic.
fn merrill_render_iso_gain(
    capture: f64,
    sensor: f64,
    fallback: u32,
    iso_range: u32,
) -> Option<f64> {
    let capture = capture as f32;
    let sensor = sensor as f32;
    let mut capture = if capture == 0.0 {
        fallback as f32
    } else {
        capture
    };
    let sensor = if sensor == 0.0 { 100.0 } else { sensor };
    if !capture.is_finite() || capture <= 0.0 || !sensor.is_finite() || sensor <= 0.0 {
        return None;
    }
    if iso_range != 0 {
        capture = match capture {
            800.0 => 666.0,
            1600.0 => 1200.0,
            3200.0 => 2133.0,
            6400.0 => 4266.0,
            _ => capture,
        };
    }
    let gain = capture / sensor;
    (gain.is_finite() && gain > 0.0).then_some(gain as f64)
}

unsafe fn merrill_iso_gain(x3f: *mut x3f_t) -> Option<f64> {
    let (mut capture, mut sensor, mut iso_range) = (0.0, 0.0, 0);
    unsafe {
        x3f_get_camf_float(x3f, c"CaptureISO".as_ptr() as *mut _, &mut capture);
        x3f_get_camf_float(x3f, c"SensorISO".as_ptr() as *mut _, &mut sensor);
        x3f_get_camf_unsigned(x3f, c"ISOrange".as_ptr() as *mut _, &mut iso_range);
    }
    let mut fallback = 0;
    if capture as f32 == 0.0 {
        // The CLI has no SPP adjustment struct; PROP ISO is its as-shot fallback.
        let mut value = ptr::null_mut();
        if unsafe { x3f_get_prop_entry(x3f, c"ISO".as_ptr() as *mut _, &mut value) } != 0
            && !value.is_null()
        {
            fallback = unsafe { CStr::from_ptr(value) }
                .to_str()
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(0);
        }
    }
    merrill_render_iso_gain(capture, sensor, fallback, iso_range)
}

fn merrill_render_color_mode(mode: &[u8]) -> Option<&'static CStr> {
    // As-shot CLI policy: SPP receives an editable adjustment enum, while this
    // renderer uses the header's mode. Its separate monochrome path is unported.
    let end = mode
        .iter()
        .position(|&byte| byte == 0)
        .unwrap_or(mode.len());
    match &mode[..end] {
        b"Monochrome" | b"Sepia" => None,
        b"Vivid" => Some(c"Vivid"),
        b"Neutral" => Some(c"Neutral"),
        b"Portrait" => Some(c"Portrait"),
        b"Landscape" => Some(c"Landscape"),
        b"FCBlue" => Some(c"FCBlue"),
        _ => Some(c"Standard"),
    }
}

fn merrill_temperature_gain(gain: [f64; 3]) -> [f64; 3] {
    if gain
        .iter()
        .all(|v| v.is_finite() && (0.5..=1.5).contains(v))
    {
        gain
    } else {
        [1.0; 3]
    }
}

/// GatherParams @0xb3464..0xb3498 request gain, in native order:
/// `baseGain = process 2.0.0.0/2.0.1.0/2.0.2.0 ? wb / 0.55 : wb + wb`
/// (per channel; the division is one fdiv, not a multiply by the
/// reciprocal), then
/// `gain = ((baseGain * FNumberGainFact) * TempGainFact) * SensorAdjustmentGainFact`
/// with the shared missing/wrong-shape default 1 and the native
/// temperature validity guard.
unsafe fn merrill_request_gain(
    x3f: *mut x3f_t,
    wb: *mut libc::c_char,
    control: Control<'_>,
) -> Option<[f64; 3]> {
    let mut wb_gain = [0.0; 3];
    if unsafe {
        x3f_get_camf_matrix_for_wb(
            x3f,
            c"WhiteBalanceGains".as_ptr() as *mut _,
            wb,
            3,
            0,
            wb_gain.as_mut_ptr(),
        )
    } == 0
    {
        if !wb.is_null() && unsafe { CStr::from_ptr(wb) }.to_bytes() == b"Sunlight" {
            return unsafe { merrill_request_gain(x3f, c"Daylight".as_ptr() as *mut _, control) };
        }
        return None;
    }
    // Same model-based process selection as merrill_count_scale: the
    // corpus derivation maps SD1 Merrill to process 2.0.x (divide) and
    // DP1/2/3 Merrill to 2.1.x (double).
    if !matches!(unsafe { (*x3f).header.version }, 0x0002_0003 | 0x0003_0000) {
        return None;
    }
    let mut model = ptr::null_mut();
    if unsafe { x3f_get_prop_entry(x3f, c"CAMMODEL".as_ptr().cast_mut(), &mut model) } == 0
        || model.is_null()
    {
        return None;
    }
    let divide = match unsafe { CStr::from_ptr(model) }.to_bytes() {
        b"SIGMA SD1 Merrill" => true,
        b"SIGMA DP1 Merrill" | b"SIGMA DP2 Merrill" | b"SIGMA DP3 Merrill" => false,
        _ => return None,
    };
    let base: [f64; 3] = if divide {
        std::array::from_fn(|c| wb_gain[c] / 0.55)
    } else {
        std::array::from_fn(|c| wb_gain[c] + wb_gain[c])
    };
    let mut fnumber = [1.0; 3];
    let mut temperature = [1.0; 3];
    let mut sensor = [1.0; 3];
    unsafe {
        if x3f_get_camf_float_vector(
            x3f,
            c"FNumberGainFact".as_ptr() as *mut _,
            fnumber.as_mut_ptr(),
        ) == 0
        {
            x3f_get_camf_float_vector(
                x3f,
                c"FnumberGainFact".as_ptr() as *mut _,
                fnumber.as_mut_ptr(),
            );
        }
        x3f_get_camf_float_vector(
            x3f,
            c"TempGainFact".as_ptr() as *mut _,
            temperature.as_mut_ptr(),
        );
        x3f_get_camf_float_vector(
            x3f,
            c"SensorAdjustmentGainFact".as_ptr() as *mut _,
            sensor.as_mut_ptr(),
        );
    }
    let temperature = merrill_temperature_gain(temperature);
    Some(std::array::from_fn(|c| {
        ((base[c] * fnumber[c]) * temperature[c]) * sensor[c]
    }))
}

pub(crate) unsafe fn merrill_float_gain_front(
    x3f: *mut x3f_t,
    wb: *mut libc::c_char,
    apply_sgain: bool,
    control: Control<'_>,
) -> crate::Result<crate::float_front::FloatFront> {
    let mut raw: x3f_area16_t = unsafe { std::mem::zeroed() };
    let mut shield: x3f_area16_t = unsafe { std::mem::zeroed() };
    if unsafe {
        x3f_image_area(x3f, &mut raw) == 0
            || x3f_crop_area_camf(
                x3f,
                c"DarkShieldTop".as_ptr() as *mut _,
                &mut raw,
                1,
                &mut shield,
            ) == 0
    } {
        return Err(crate::Error::InvalidData(
            "missing Merrill float black calibration",
        ));
    }
    let black = unsafe { merrill_trimmed_black(&shield, control) }?;
    let exif = unsafe { merrill_exif(x3f, control) }?;
    let spatial = unsafe { merrill_spatial_tables(x3f, apply_sgain, exif.f_number, control) }?;
    let gain = unsafe { merrill_request_gain(x3f, wb, control) }.ok_or(
        crate::Error::InvalidData("missing Merrill float request gain"),
    )?;
    let tables = spatial.map(|tables| {
        std::array::from_fn(|c| {
            crate::float_front::GainTable::new(
                tables[c].values.clone(),
                tables[c].cols,
                tables[c].rows,
            )
        })
    });
    crate::float_front::FloatFront::new(black, gain, tables)
}

/// Native resolution string. The request builds it from
/// adjustment+0x190 (1 = HI, 3 = MED, 4 = LOW, else empty); the
/// corpus-bound choice reads PROP RESOLUTION. Both retained
/// references are HI.
unsafe fn merrill_resolution_string(x3f: *mut x3f_t) -> String {
    let mut property = ptr::null_mut();
    if unsafe { x3f_get_prop_entry(x3f, c"RESOLUTION".as_ptr().cast_mut(), &mut property) } == 0
        || property.is_null()
    {
        return String::new();
    }
    unsafe { CStr::from_ptr(property) }
        .to_string_lossy()
        .into_owned()
}

/// Half-open GetRectFromCamfile rects and the DarkShieldColRange
/// matrix shared by the radial and highlight stages.
pub(crate) unsafe fn merrill_rects(
    x3f: *mut x3f_t,
) -> crate::Result<([i32; 4], [i32; 4], [i32; 4])> {
    let read_rect = |name: &std::ffi::CStr| -> Option<[i32; 4]> {
        let mut values = [0u32; 4];
        if unsafe {
            x3f_get_camf_matrix(
                x3f,
                name.as_ptr() as *mut _,
                4,
                0,
                0,
                matrix_type_t_M_UINT,
                values.as_mut_ptr().cast(),
            )
        } == 0
        {
            return None;
        }
        // GetRectFromCamfile adds one to the stored inclusive right and
        // bottom endpoints, giving half-open bounds.
        Some([
            values[0] as i32,
            values[1] as i32,
            values[2] as i32 + 1,
            values[3] as i32 + 1,
        ])
    };
    let mut col_range = [0u32; 4];
    let (Some(active), Some(keep)) = (read_rect(c"ActiveImageArea"), read_rect(c"KeepImageArea"))
    else {
        return Err(crate::Error::InvalidData(
            "missing Merrill radial rectangles",
        ));
    };
    if unsafe {
        x3f_get_camf_matrix(
            x3f,
            c"DarkShieldColRange".as_ptr() as *mut _,
            2,
            2,
            0,
            matrix_type_t_M_UINT,
            col_range.as_mut_ptr().cast(),
        )
    } == 0
    {
        return Err(crate::Error::InvalidData(
            "missing Merrill radial column range",
        ));
    }
    Ok((active, keep, col_range.map(|v| v as i32)))
}

/// Stage 2 of the ordered float frontend: native WB radial shading.
/// Returns None when the CAMF property list or the WB matrix is absent;
/// the native preprocess skips the whole radial block in that case
/// (0xa8198, 0xa81f0).
unsafe fn merrill_radial_shading(
    x3f: *mut x3f_t,
    wb: *mut libc::c_char,
    control: Control<'_>,
) -> crate::Result<Option<crate::float_front::RadialShading>> {
    control.check()?;
    let mut coefficients = [0.0_f64; 4];
    if unsafe {
        x3f_get_camf_matrix_for_wb(
            x3f,
            c"WhiteBalanceColorShadingFactor".as_ptr() as *mut _,
            wb,
            2,
            2,
            coefficients.as_mut_ptr(),
        )
    } == 0
    {
        return Ok(None);
    }
    let (active, keep, col_range) = unsafe { merrill_rects(x3f)? };
    let resolution = unsafe { merrill_resolution_string(x3f) };
    Ok(Some(crate::float_front::RadialShading::new(
        active,
        keep,
        col_range,
        coefficients,
        &resolution,
    )?))
}

/// Native badness selection (inline at 0xa6b2c..0xa6c9c), validated
/// against all 53 corpus CAMFs and the archived instruction replay
/// (followup-repair/badness). Returns the selected badness label.
///
/// `thresholds` is the BadnessTable as four (temperature, exposure,
/// label) triples, already validated for shape and finiteness.
/// `exposure` is `CaptureShutter / 1000 + (request+0x198 > 1 ?
/// BurstExposureAdjust : 0)`. All arithmetic rounds to f32 separately;
/// the exponential is `expf`.
fn merrill_badness_label(
    thresholds: &[(f32, f32, f32); 4],
    sensor_temperature: f32,
    exposure: f32,
) -> u32 {
    const K: f32 = f32::from_bits(0x3db1_7218); // ln(2)/8 native literal.
    let mut values = [0.0_f32; 4];
    for (i, &(temperature, exposure_threshold, _)) in thresholds.iter().enumerate() {
        values[i] = exposure_threshold * (temperature * K).exp();
    }
    let severity = (sensor_temperature * K).exp() * exposure;
    // Native min/max reductions start at threshold[0] clamped against
    // +-1e6, then compare the remaining three values.
    let mut max = values[0].max(-1.0e6_f32);
    let mut min = values[0].min(1.0e6_f32);
    for &value in &values[1..] {
        if value > max {
            max = value;
        }
        if value < min {
            min = value;
        }
    }
    let label = if severity <= min {
        thresholds[3].2
    } else if severity >= max {
        thresholds[0].2
    } else {
        // Start at row 3 and move backwards while severity >
        // threshold[row]; equality selects that row's label.
        let mut row = 3;
        while row > 0 && severity > values[row] {
            row -= 1;
        }
        thresholds[row].2
    };
    // ARM64 FCVTZU: negative labels convert to zero.
    (label.max(0.0) as u32).min(u32::from(u16::MAX))
}

/// The native BurstExposureAdjust gate for the badness exposure
/// (followup-repair/usburst-lineage). The managed X3F filler reads
/// PROP BURST and, when empty, retries PROP NUMOFBURST; an empty
/// string leaves the managed default 0xFFFF (the absent sentinel,
/// which adds the adjustment), a failed parse leaves 0, and a
/// parsed value stays. The native exposure adds
/// BurstExposureAdjust only when `uint32(request+0x198) > 1`, so
/// ordinary as-shot values 0 and 1 exclude it.
pub(crate) unsafe fn merrill_burst_exposure_adjust_included(x3f: *mut x3f_t) -> bool {
    let mut burst: u16 = u16::MAX;
    for key in [c"BURST", c"NUMOFBURST"] {
        let mut value: *mut libc::c_char = ptr::null_mut();
        if unsafe { crate::x3f_get_prop_entry(x3f, key.as_ptr() as *mut _, &mut value) } != 0 {
            let text = unsafe { std::ffi::CStr::from_ptr(value) }
                .to_string_lossy()
                .to_string();
            // Only an empty string retries the next tag; whitespace is
            // malformed and parses to 0, as the managed decode does.
            if !text.is_empty() {
                burst = text.trim().parse().unwrap_or(0);
                break;
            }
        }
    }
    u32::from(burst) > 1
}

/// The selected badness label and BPBadnessOffset (default 1). Reads
/// BadnessTable and the exposure inputs and selects the label natively
/// (0xa6b..0xa6c90). Returns `None` when BadnessTable is absent.
pub(crate) unsafe fn merrill_badness_selection(
    x3f: *mut x3f_t,
    burst_exposure_adjust_included: bool,
    control: Control<'_>,
) -> crate::Result<Option<(u32, u32)>> {
    control.check()?;
    let mut thresholds_raw = [0.0_f64; 12];
    if unsafe {
        x3f_get_camf_matrix(
            x3f,
            c"BadnessTable".as_ptr() as *mut _,
            4,
            3,
            0,
            matrix_type_t_M_FLOAT,
            thresholds_raw.as_mut_ptr().cast(),
        )
    } == 0
    {
        return Ok(None);
    }
    let mut thresholds = [(0.0_f32, 0.0_f32, 0.0_f32); 4];
    for (i, triple) in thresholds.iter_mut().enumerate() {
        let values = [
            thresholds_raw[3 * i] as f32,
            thresholds_raw[3 * i + 1] as f32,
            thresholds_raw[3 * i + 2] as f32,
        ];
        if !values.iter().all(|v| v.is_finite()) {
            return Err(crate::Error::InvalidData(
                "invalid Merrill badness calibration",
            ));
        }
        *triple = (values[0], values[1], values[2]);
    }
    let shutter = unsafe {
        let mut value = 0.0_f64;
        x3f_get_camf_float(x3f, c"CaptureShutter".as_ptr().cast_mut(), &mut value);
        value
    };
    let temperature = unsafe {
        let mut value = 0.0_f64;
        x3f_get_camf_float(x3f, c"SensorTemperature".as_ptr().cast_mut(), &mut value);
        value
    };
    let burst_adjust = if burst_exposure_adjust_included {
        let mut value = 0.0_f64;
        unsafe {
            x3f_get_camf_float(x3f, c"BurstExposureAdjust".as_ptr().cast_mut(), &mut value);
        }
        value as f32
    } else {
        0.0_f32
    };
    // Native rounds each step in f32: CaptureShutter / 1000.f, then
    // the burst add (followup-repair/badness/FINDINGS.txt).
    let exposure = shutter as f32 / 1000.0_f32 + burst_adjust;
    let mut offset: u32 = 1;
    unsafe {
        x3f_get_camf_unsigned(x3f, c"BPBadnessOffset".as_ptr().cast_mut(), &mut offset);
    }
    let label = merrill_badness_label(&thresholds, temperature as f32, exposure);
    Ok(Some((label, offset)))
}

/// True when a CAMF matrix entry with this name exists in any numeric type
/// or shape. Native lookups are untyped; the typed getters report a wrong
/// type as absent.
unsafe fn merrill_camf_matrix_present(x3f: *mut x3f_t, name: &CStr) -> bool {
    for typ in [
        matrix_type_t_M_FLOAT,
        matrix_type_t_M_UINT,
        matrix_type_t_M_INT,
    ] {
        let (mut a, mut b, mut c) = (0, 0, 0);
        let mut data: *mut libc::c_void = ptr::null_mut();
        let name = name.as_ptr() as *mut _;
        if unsafe {
            x3f_get_camf_matrix_var(
                x3f,
                name,
                &mut a,
                ptr::null_mut(),
                ptr::null_mut(),
                typ,
                &mut data,
            ) != 0
                || x3f_get_camf_matrix_var(
                    x3f,
                    name,
                    &mut a,
                    &mut b,
                    ptr::null_mut(),
                    typ,
                    &mut data,
                ) != 0
                || x3f_get_camf_matrix_var(x3f, name, &mut a, &mut b, &mut c, typ, &mut data) != 0
        } {
            return true;
        }
    }
    false
}

/// One camera bad-pixel list (BadPixelsF20 or Jpeg_BadClusters) in the
/// native positional layout: (row, column, flags) per record. Returns
/// `None` when the entry is absent. A present entry must be a 3 x N
/// unsigned matrix of u16 values; the native getter reads it untyped.
pub(crate) unsafe fn merrill_bad_pixel_list(
    x3f: *mut x3f_t,
    name: &CStr,
    control: Control<'_>,
) -> crate::Result<Option<Vec<crate::float_front::BadPixelRecord>>> {
    control.check()?;
    let invalid = crate::Error::InvalidData("invalid Merrill bad pixel list");
    let mut cols: libc::c_int = 0;
    let mut rows: libc::c_int = 0;
    let mut list: *mut u32 = ptr::null_mut();
    if unsafe {
        x3f_get_camf_matrix_var(
            x3f,
            name.as_ptr() as *mut _,
            &mut cols,
            &mut rows,
            ptr::null_mut(),
            matrix_type_t_M_UINT,
            &mut list as *mut *mut u32 as *mut *mut libc::c_void,
        )
    } == 0
    {
        return if unsafe { merrill_camf_matrix_present(x3f, name) } {
            Err(invalid)
        } else {
            Ok(None)
        };
    }
    if cols != 3 || rows < 0 {
        return Err(invalid);
    }
    let words = unsafe { std::slice::from_raw_parts(list, 3 * rows as usize) };
    if words.iter().any(|&v| v > u32::from(u16::MAX)) {
        return Err(invalid);
    }
    Ok(Some(
        words
            .chunks_exact(3)
            .map(|w| crate::float_front::BadPixelRecord {
                row: w[0] as u16,
                column: w[1] as u16,
                flags: w[2] as u16,
            })
            .collect(),
    ))
}

/// ColorNoiseModel as nine doubles: constant, linear and quadratic rows of
/// C0, C1, C2. The native reads the first nine stored floats and leaves
/// zeros when the entry is absent (0xa8020..0xa8080). A present entry of
/// another shape or type is rejected.
pub(crate) unsafe fn merrill_color_noise_model(x3f: *mut x3f_t) -> crate::Result<[f64; 9]> {
    let mut values = [0.0_f64; 9];
    if unsafe {
        x3f_get_camf_matrix(
            x3f,
            c"ColorNoiseModel".as_ptr() as *mut _,
            3,
            3,
            0,
            matrix_type_t_M_FLOAT,
            values.as_mut_ptr().cast(),
        )
    } != 0
    {
        if !values.iter().all(|v| v.is_finite()) {
            return Err(crate::Error::InvalidData("invalid Merrill noise model"));
        }
        return Ok(values);
    }
    if unsafe { merrill_camf_matrix_present(x3f, c"ColorNoiseModel") } {
        return Err(crate::Error::InvalidData("invalid Merrill noise model"));
    }
    Ok(values)
}

/// BadClustersF20 packed words, read as one or two dimensions of u16
/// values like the native type-6 reader (0x45468). Returns `None` when
/// the entry is absent.
pub(crate) unsafe fn merrill_packed_cluster_words(
    x3f: *mut x3f_t,
) -> crate::Result<Option<Vec<u16>>> {
    let name = c"BadClustersF20";
    let invalid = crate::Error::InvalidData("invalid Merrill packed cluster list");
    let (mut a, mut b): (libc::c_int, libc::c_int) = (0, 0);
    let mut data: *mut u32 = ptr::null_mut();
    let out = &mut data as *mut *mut u32 as *mut *mut libc::c_void;
    let count = if unsafe {
        x3f_get_camf_matrix_var(
            x3f,
            name.as_ptr() as *mut _,
            &mut a,
            ptr::null_mut(),
            ptr::null_mut(),
            matrix_type_t_M_UINT,
            out,
        )
    } != 0
    {
        a as usize
    } else if unsafe {
        x3f_get_camf_matrix_var(
            x3f,
            name.as_ptr() as *mut _,
            &mut a,
            &mut b,
            ptr::null_mut(),
            matrix_type_t_M_UINT,
            out,
        )
    } != 0
    {
        a as usize * b as usize
    } else if unsafe { merrill_camf_matrix_present(x3f, name) } {
        return Err(invalid);
    } else {
        return Ok(None);
    };
    let words = unsafe { std::slice::from_raw_parts(data, count) };
    if words.iter().any(|&v| v > u32::from(u16::MAX)) {
        return Err(invalid);
    }
    Ok(Some(words.iter().map(|&v| v as u16).collect()))
}

/// Stage 4 of the ordered float frontend: native highlight restoration
/// from the camera saturation maps. Reads SatMapR/G/B (SatMapFullR/G/B
/// fallback), decodes the RLE or Full payload, derives the native
/// threshold from ExpEstimate/ExpNet, and runs the stage. Skips when
/// SkipStepFlags bit 11 is set (a missing channel map makes preprocessing
/// OR 0x3800 into skip, which includes bit 11).
pub(crate) struct MerrillHighlight {
    pub restore: crate::float_front::HighlightRestoration,
    pub sat: Vec<u8>,
    pub valid: [i32; 4],
}

pub(crate) unsafe fn merrill_highlight_restoration(
    x3f: *mut x3f_t,
    active: [i32; 4],
    keep: [i32; 4],
    col_range: [i32; 4],
    control: Control<'_>,
) -> crate::Result<Option<MerrillHighlight>> {
    control.check()?;
    let mut skip = 0_u32;
    unsafe {
        x3f_get_camf_unsigned(x3f, c"SkipStepFlags".as_ptr().cast_mut(), &mut skip);
    }
    if skip & (1 << 11) != 0 {
        return Ok(None);
    }
    // The saturation maps are packed over the full decoded image; the
    // valid rect uses the translated active bounds like the radial stage.
    let mut area: x3f_area16_t = unsafe { std::mem::zeroed() };
    if unsafe { x3f_image_area(x3f, &mut area) } == 0 {
        return Err(crate::Error::InvalidData("missing Merrill image area"));
    }
    let width = area.columns as usize;
    let height = area.rows as usize;
    let resolution = unsafe { merrill_resolution_string(x3f) };
    let valid =
        crate::float_front::RadialShading::valid_pix_rect(active, keep, col_range, &resolution);
    // Decode the three channel maps. Native presence is untyped and the
    // whole archived corpus carries type-2 entries; a present entry of
    // another type is native-undefined, so this typed port treats it
    // as absent (Full fallback or whole-stage skip) instead of reading
    // an untyped payload.
    let names = [
        (c"SatMapR", c"SatMapFullR", 0_usize),
        (c"SatMapG", c"SatMapFullG", 1),
        (c"SatMapB", c"SatMapFullB", 2),
    ];
    let pixels = width.checked_mul(height).ok_or(crate::Error::InvalidData(
        "invalid Merrill saturation map size",
    ))?;
    let mut sat = vec![0_u8; pixels * 3];
    let mut decoded_channels = 0_u32;
    for (rle_name, full_name, channel) in names {
        // The camera maps are one-dimensional type-2 entries; dim0
        // carries the word count and the other dimension pointers stay
        // null (the accessor rejects a present dim1 for a 1-D matrix).
        let mut count: libc::c_int = 0;
        let mut words: *mut u32 = ptr::null_mut();
        let found = unsafe {
            x3f_get_camf_matrix_var(
                x3f,
                rle_name.as_ptr() as *mut _,
                &mut count,
                ptr::null_mut(),
                ptr::null_mut(),
                matrix_type_t_M_UINT,
                &mut words as *mut *mut u32 as *mut *mut libc::c_void,
            ) != 0
        };
        if found {
            let slice = unsafe { std::slice::from_raw_parts(words, count.max(0) as usize) };
            // Native adds u32 positions without bounds; validate the runs
            // against the map before decoding (malformed runs reject).
            let words64: Vec<u64> = slice.iter().map(|&v| u64::from(v)).collect();
            let mut position = 0_u64;
            let mut index = 0;
            while index + 1 < words64.len() {
                let start = position + words64[index];
                let Some(end) = start.checked_add(words64[index + 1]) else {
                    return Err(crate::Error::InvalidData("invalid Merrill saturation run"));
                };
                if end > pixels as u64 {
                    return Err(crate::Error::InvalidData(
                        "invalid Merrill saturation run extent",
                    ));
                }
                position = end;
                index += 2;
            }
            crate::float_front::decode_sat_map_rle(slice, &mut sat, channel);
            decoded_channels |= 1 << channel;
            continue;
        }
        let mut count: libc::c_int = 0;
        let mut words: *mut u32 = ptr::null_mut();
        let found_full = unsafe {
            x3f_get_camf_matrix_var(
                x3f,
                full_name.as_ptr() as *mut _,
                &mut count,
                ptr::null_mut(),
                ptr::null_mut(),
                matrix_type_t_M_UINT,
                &mut words as *mut *mut u32 as *mut *mut libc::c_void,
            ) != 0
        };
        if found_full {
            let slice = unsafe { std::slice::from_raw_parts(words, count.max(0) as usize) };
            let coverage = 32_u64 * slice.len().saturating_sub(1) as u64;
            if coverage > pixels as u64 {
                return Err(crate::Error::InvalidData(
                    "invalid Merrill saturation map extent",
                ));
            }
            crate::float_front::decode_sat_map_full(slice, &mut sat, channel);
            decoded_channels |= 1 << channel;
        }
        // A channel missing both forms: skip |= 0x3800 (bit 11) natively,
        // which the decoded_channels check mirrors.
    }
    if decoded_channels != 0b111 {
        return Ok(None);
    }
    let exp_estimate = unsafe {
        let mut value = 0.0_f64;
        x3f_get_camf_float(x3f, c"ExpEstimate".as_ptr().cast_mut(), &mut value);
        value as f32
    };
    let exp_net = unsafe {
        let mut value = 0.0_f64;
        x3f_get_camf_float(x3f, c"ExpNet".as_ptr().cast_mut(), &mut value);
        value as f32
    };
    let restore = crate::float_front::HighlightRestoration::new(exp_estimate, exp_net);
    Ok(Some(MerrillHighlight {
        restore,
        sat,
        valid,
    }))
}

/// The despeckle request strength (request+0x100). The manager copies
/// GatherParams field+0x14 to request+0x100; SetUpNoiseParams blends
/// the CAMF ISONoiseSettings column 1 when the camfile version is 3,
/// else the static resolution table. The archived static proof pins
/// column 1 to exactly 1.0 in every row of all 53 corpus CAMFs, which
/// makes the blend exactly 1.0f for every interpolation fraction, and
/// the static HI table is all 1.0. Non-constant columns and the
/// MED/LOW static tables need the un-reconstructed ISO lookup and
/// stay outside the verified contract. The supported corpus models
/// are all camfile version 3 (the same model mapping the request
/// gain uses).
unsafe fn merrill_despeckle_strength(x3f: *mut x3f_t) -> crate::Result<f32> {
    let mut dim0: libc::c_int = 0;
    let mut dim1: libc::c_int = 0;
    let mut table: *mut f64 = ptr::null_mut();
    let found = unsafe {
        x3f_get_camf_matrix_var(
            x3f,
            c"ISONoiseSettings".as_ptr() as *mut _,
            &mut dim0,
            &mut dim1,
            ptr::null_mut(),
            matrix_type_t_M_FLOAT,
            &mut table as *mut *mut f64 as *mut *mut libc::c_void,
        ) != 0
    };
    if found {
        let rows = dim0.max(0) as usize;
        let columns = dim1.max(0) as usize;
        let Some(elements) = rows.checked_mul(columns) else {
            return Err(crate::Error::InvalidData("invalid Merrill noise table"));
        };
        if rows == 0 || columns < 2 {
            return Err(crate::Error::InvalidData("invalid Merrill noise table"));
        }
        let values = unsafe { std::slice::from_raw_parts(table, elements) };
        // Constant column 1: the blend is exactly 1.0f for every t.
        if (0..rows).any(|row| values[row * columns + 1] != 1.0) {
            return Err(crate::Error::InvalidData(
                "unverified Merrill despeckle strength",
            ));
        }
        Ok(1.0)
    } else {
        let resolution = unsafe { merrill_resolution_string(x3f) };
        if resolution.contains("HI") {
            Ok(1.0)
        } else {
            Err(crate::Error::InvalidData(
                "unverified Merrill despeckle strength",
            ))
        }
    }
}

/// Stage 5 of the ordered float frontend: raster despeckle. Reads the
/// ApplyGains request gains, the typed float DespAdjust vector (the
/// strict getter ignores the corpus's int16 entry, so the default
/// 15,15,5 applies on every corpus file), the request strength and
/// the translated active rect. Skips when SkipStepFlags bit 14 is
/// set (the native bit also skips the following CROffsetCorrection
/// state, which is stage 6).
pub(crate) unsafe fn merrill_despeckle(
    x3f: *mut x3f_t,
    wb: *mut libc::c_char,
    control: Control<'_>,
) -> crate::Result<Option<crate::float_front::Despeckle>> {
    match unsafe { merrill_despeckle_inputs(x3f, wb, None, control)? } {
        Some((gain, adjust, rect)) => Ok(Some(crate::float_front::Despeckle::new(
            gain, adjust, rect,
        )?)),
        None => Ok(None),
    }
}

/// The native despeckle inputs: request gain, strength-scaled DespAdjust
/// and the translated active rect, or `None` when the stage skips.
/// `strength` is the request strength (NoiseParams+0x14). `None` derives
/// it with `merrill_despeckle_strength`, which supports only the verified
/// constant ISONoiseSettings column and the HI table.
pub(crate) unsafe fn merrill_despeckle_inputs(
    x3f: *mut x3f_t,
    wb: *mut libc::c_char,
    strength: Option<f32>,
    control: Control<'_>,
) -> crate::Result<Option<([f64; 3], [f64; 3], [u32; 4])>> {
    control.check()?;
    let mut skip = 0_u32;
    unsafe {
        x3f_get_camf_unsigned(x3f, c"SkipStepFlags".as_ptr().cast_mut(), &mut skip);
    }
    if skip & (1 << 14) != 0 {
        return Ok(None);
    }
    // The gain triple is the same ApplyGains request gain; an
    // unsupported model path cannot derive it, so the stage skips.
    let Some(gain) = (unsafe { merrill_request_gain(x3f, wb, control) }) else {
        return Ok(None);
    };
    // GetFColorFloatFromCamfile("DespAdjust", {15,15,5}): typed float
    // 1-D [3] lookup; the corpus stores int16, so the default applies.
    let mut adjust = [15.0_f64, 15.0, 5.0];
    unsafe {
        x3f_get_camf_matrix(
            x3f,
            c"DespAdjust".as_ptr() as *mut _,
            3,
            0,
            0,
            matrix_type_t_M_FLOAT,
            adjust.as_mut_ptr().cast(),
        );
    }
    let strength = match strength {
        Some(strength) if strength.is_finite() => strength,
        Some(_) => {
            return Err(crate::Error::InvalidData(
                "invalid Merrill despeckle strength",
            ))
        }
        None => unsafe { merrill_despeckle_strength(x3f)? },
    };
    let adjust = std::array::from_fn(|channel| adjust[channel] * f64::from(strength));
    // ActiveImageArea translated by KeepImageArea left/top (0xa5e44):
    // not the ValidPixRect the radial and highlight stages use.
    let (active, keep, _col_range) = unsafe { merrill_rects(x3f)? };
    let translated = [
        active[0] - keep[0],
        active[1] - keep[1],
        active[2] - keep[0],
        active[3] - keep[1],
    ];
    if translated.iter().any(|&v| v < 0) {
        return Err(crate::Error::InvalidData(
            "invalid Merrill despeckle rectangle",
        ));
    }
    let rect = [
        translated[0] as u32,
        translated[1] as u32,
        translated[2] as u32,
        translated[3] as u32,
    ];
    Ok(Some((gain, adjust, rect)))
}

/// Stage 6 inputs: the typed CaptureISO scalar, the ApplyGains
/// request gains and the translated active rect (Keep-local, the same
/// rect the despeckle stage uses).
pub(crate) struct MerrillCrOffset {
    pub iso: f32,
    pub gains: [f64; 3],
    pub rect: [i32; 4],
}

/// Stage 6 of the ordered float frontend: row/column chroma offset
/// correction. Reads the typed CaptureISO scalar (default 0), the
/// ApplyGains request gain triple and the translated active rect.
/// Skips when SkipStepFlags bit 14 is set (the native bit gates both
/// this state and despeckle) or the gain triple cannot be derived.
pub(crate) unsafe fn merrill_cr_offset(
    x3f: *mut x3f_t,
    wb: *mut libc::c_char,
    control: Control<'_>,
) -> crate::Result<Option<MerrillCrOffset>> {
    control.check()?;
    let mut skip = 0_u32;
    unsafe {
        x3f_get_camf_unsigned(x3f, c"SkipStepFlags".as_ptr().cast_mut(), &mut skip);
    }
    if skip & (1 << 14) != 0 {
        return Ok(None);
    }
    let Some(gains) = (unsafe { merrill_request_gain(x3f, wb, control) }) else {
        return Ok(None);
    };
    let iso = unsafe {
        let mut value = 0.0_f64;
        x3f_get_camf_float(x3f, c"CaptureISO".as_ptr().cast_mut(), &mut value);
        value as f32
    };
    // ActiveImageArea translated by KeepImageArea left/top, the same
    // Keep-local rect the despeckle stage uses (0xa5e44).
    let (active, keep, _col_range) = unsafe { merrill_rects(x3f)? };
    let rect = [
        active[0] - keep[0],
        active[1] - keep[1],
        active[2] - keep[0],
        active[3] - keep[1],
    ];
    if rect.iter().any(|&v| v < 0) {
        return Err(crate::Error::InvalidData(
            "invalid Merrill offset rectangle",
        ));
    }
    Ok(Some(MerrillCrOffset { iso, gains, rect }))
}

/// The owned Merrill float raster after the ordered native preprocess
/// stages, in gained sensor-count units. Interleaved RGB, `row_stride`
/// floats per row; row padding keeps the decoded layout.
pub(crate) struct MerrillPreparedFloat {
    pub data: Vec<f32>,
    pub width: usize,
    pub height: usize,
    pub row_stride: usize,
}

/// Run the ordered Merrill float preprocess stages on the decoded image:
///
/// 1. DetectBadReds over BadPixelsF20 (with primary overrides), then over
///    Jpeg_BadClusters, on the raw float raster before any gain.
/// 2. ApplyGains (black, request gain and spatial tables).
/// 3. WB radial shading, when present.
/// 4. Unless SkipStepFlags bit 9: listed repair of the primary list with
///    the overrides, the JPEG list, then the detected list, all at the
///    BPBadnessOffset minimum; then packed clusters at the label minus
///    BCBadnessOffset with BadClusterMargin.
/// 5. Highlight restoration, raster despeckle and row/column offsets.
///
/// With `fix_bad` false, steps 1 and 4 do not run; the other stages do.
/// `despeckle_strength` is the calibrated request strength (NoiseParams+0x14);
/// `None` uses the verified constant-column derivation, which returns an
/// error for unverified ISO tables.
/// `observe` receives each stage name, the raster after the stage and a
/// short detail. Every error and cancellation propagates. With `fix_bad`,
/// a missing primary list or BadnessTable is an error: the native caller
/// reads an uninitialized count without the list.
pub(crate) unsafe fn merrill_prepare_float(
    x3f: *mut x3f_t,
    wb: *mut libc::c_char,
    apply_sgain: bool,
    fix_bad: bool,
    despeckle_strength: Option<f32>,
    control: Control<'_>,
    observe: &mut dyn FnMut(&'static str, &[f32], String),
) -> crate::Result<MerrillPreparedFloat> {
    use crate::float_front::repair;
    control.check()?;
    let mut area: x3f_area16_t = unsafe { std::mem::zeroed() };
    if unsafe { x3f_image_area(x3f, &mut area) } == 0 || area.data.is_null() {
        return Err(crate::Error::InvalidData("missing Merrill image area"));
    }
    let (width, height, row_stride) = (
        area.columns as usize,
        area.rows as usize,
        area.row_stride as usize,
    );
    let front = unsafe { merrill_float_gain_front(x3f, wb, apply_sgain, control)? };
    let decoded = unsafe { std::slice::from_raw_parts(area.data, height * row_stride) };
    let mut data = crate::float_front::raw_float(decoded)?;
    observe("raw", &data, String::new());

    let (mut primary, mut jpeg, mut detected, mut overrides) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    if fix_bad {
        primary = unsafe { merrill_bad_pixel_list(x3f, c"BadPixelsF20", control)? }.ok_or(
            crate::Error::InvalidData("missing Merrill primary bad pixel list"),
        )?;
        jpeg = unsafe { merrill_bad_pixel_list(x3f, c"Jpeg_BadClusters", control)? }
            .unwrap_or_default();
        let noise = unsafe { merrill_color_noise_model(x3f)? };
        repair::detect_bad_reds(
            &data,
            width,
            height,
            row_stride,
            &primary,
            &noise,
            &mut detected,
            Some(&mut overrides),
            control,
        )?;
        let from_primary = detected.len();
        if !jpeg.is_empty() {
            repair::detect_bad_reds(
                &data,
                width,
                height,
                row_stride,
                &jpeg,
                &noise,
                &mut detected,
                None,
                control,
            )?;
        }
        let changed = overrides
            .iter()
            .zip(&primary)
            .filter(|(o, r)| **o != u32::from(r.flags & 0xff))
            .count();
        observe(
            "detect",
            &data,
            format!(
                "primary={} jpeg={} detected={} from_primary={} overrides_changed={}",
                primary.len(),
                jpeg.len(),
                detected.len(),
                from_primary,
                changed
            ),
        );
    } else {
        observe("detect", &data, "skipped fix_bad=false".to_string());
    }

    front.apply_in_place(&mut data, width, height, row_stride, control)?;
    let bits = |v: [f64; 3]| v.map(f64::to_bits);
    observe(
        "gains",
        &data,
        format!(
            "black={:x?} gain={:x?} spatial={}",
            bits(front.black()),
            bits(front.gain()),
            front.spatial().is_some()
        ),
    );
    let radial = unsafe { merrill_radial_shading(x3f, wb, control)? };
    if let Some(radial) = &radial {
        radial.apply(&mut data, width, height, row_stride, control)?;
    }
    observe("radial", &data, format!("applied={}", radial.is_some()));

    let mut skip = 0_u32;
    unsafe {
        x3f_get_camf_unsigned(x3f, c"SkipStepFlags".as_ptr().cast_mut(), &mut skip);
    }
    if fix_bad && skip & (1 << 9) == 0 {
        let burst = unsafe { merrill_burst_exposure_adjust_included(x3f) };
        let (label, offset) = unsafe { merrill_badness_selection(x3f, burst, control)? }.ok_or(
            crate::Error::InvalidData("missing Merrill badness calibration"),
        )?;
        let listed = crate::float_front::ListedRepair::new(label, offset);
        listed.apply(
            &mut data, width, height, row_stride, &primary, &overrides, control,
        )?;
        observe(
            "listed",
            &data,
            format!("label={label} minimum={}", listed.minimum_badness()),
        );
        listed.apply(&mut data, width, height, row_stride, &jpeg, &[], control)?;
        observe("jpeg", &data, format!("records={}", jpeg.len()));
        listed.apply(
            &mut data,
            width,
            height,
            row_stride,
            &detected,
            &[],
            control,
        )?;
        observe("detected", &data, format!("records={}", detected.len()));
        let words = unsafe { merrill_packed_cluster_words(x3f)? };
        let detail = if let Some(words) = words {
            let mut cluster_offset = 1_u32;
            let mut margin = 25_u32;
            unsafe {
                x3f_get_camf_unsigned(
                    x3f,
                    c"BCBadnessOffset".as_ptr().cast_mut(),
                    &mut cluster_offset,
                );
                x3f_get_camf_unsigned(x3f, c"BadClusterMargin".as_ptr().cast_mut(), &mut margin);
            }
            let badness = if label == 0 {
                0
            } else {
                label.wrapping_sub(cluster_offset)
            };
            let stats = repair::repair_bad_pixel_clusters(
                &mut data, width, height, row_stride, &words, badness, margin, control,
            )?;
            format!(
                "badness={badness} margin={margin} records={} points={} zero={}",
                stats.records, stats.points, stats.zero_fallback
            )
        } else {
            "absent".to_string()
        };
        observe("clusters", &data, detail);
    } else {
        observe("listed", &data, format!("skipped fix_bad={fix_bad}"));
    }

    let (active, keep, col_range) = unsafe { merrill_rects(x3f)? };
    let highlight =
        unsafe { merrill_highlight_restoration(x3f, active, keep, col_range, control)? };
    let detail = if let Some(highlight) = highlight {
        let mut dense = vec![0_u8; width * height];
        let mut sat = highlight.sat;
        let stats = highlight.restore.restore(
            &mut data,
            width,
            height,
            row_stride,
            &mut sat,
            &mut dense,
            highlight.valid,
            control,
        )?;
        format!(
            "donors={} removed={} occupied={} dense={} lut={}",
            stats.donors, stats.removed, stats.occupied, stats.dense_pixels, stats.have_lut
        )
    } else {
        "none".to_string()
    };
    observe("highlight", &data, detail);
    let despeckle = unsafe { merrill_despeckle_inputs(x3f, wb, despeckle_strength, control)? };
    let detail = match despeckle {
        Some((gain, adjust, rect)) => format!(
            "records={}",
            crate::float_front::Despeckle::new(gain, adjust, rect)?
                .apply(&mut data, width, height, row_stride, control)?
        ),
        None => "none".to_string(),
    };
    observe("despeckle", &data, detail);
    let offset = unsafe { merrill_cr_offset(x3f, wb, control)? };
    let detail = match offset {
        Some(offset) => {
            let stats = crate::cr_offset::correct(
                &mut data,
                width,
                height,
                row_stride,
                offset.iso,
                offset.gains,
                offset.rect,
                control,
            )?;
            format!(
                "patches={} accepted={} rows={}",
                stats.patches, stats.accepted, stats.rows
            )
        }
        None => "none".to_string(),
    };
    observe("offset", &data, detail);
    Ok(MerrillPreparedFloat {
        data,
        width,
        height,
        row_stride,
    })
}

unsafe fn merrill_wb_gain(x3f: *mut x3f_t, wb: *mut libc::c_char) -> Option<[f64; 3]> {
    let mut gain = [0.0; 3];
    if unsafe {
        x3f_get_camf_matrix_for_wb(
            x3f,
            c"WhiteBalanceGains".as_ptr() as *mut _,
            wb,
            3,
            0,
            gain.as_mut_ptr(),
        )
    } == 0
    {
        if !wb.is_null() && unsafe { CStr::from_ptr(wb) }.to_bytes() == b"Sunlight" {
            return unsafe { merrill_wb_gain(x3f, c"Daylight".as_ptr() as *mut _) };
        }
        return None;
    }
    let mut aperture = [1.0; 3];
    let mut temperature = [1.0; 3];
    let mut sensor = [1.0; 3];
    unsafe {
        if x3f_get_camf_float_vector(
            x3f,
            c"FNumberGainFact".as_ptr() as *mut _,
            aperture.as_mut_ptr(),
        ) == 0
        {
            x3f_get_camf_float_vector(
                x3f,
                c"FnumberGainFact".as_ptr() as *mut _,
                aperture.as_mut_ptr(),
            );
        }
        x3f_get_camf_float_vector(
            x3f,
            c"TempGainFact".as_ptr() as *mut _,
            temperature.as_mut_ptr(),
        );
        x3f_get_camf_float_vector(
            x3f,
            c"SensorAdjustmentGainFact".as_ptr() as *mut _,
            sensor.as_mut_ptr(),
        );
    }
    let temperature = merrill_temperature_gain(temperature);
    Some(std::array::from_fn(|c| {
        ((gain[c] * aperture[c]) * temperature[c]) * sensor[c]
    }))
}

pub(crate) unsafe fn color_mode_base_calibration(
    x3f: *mut x3f_t,
    mut wb: *mut libc::c_char,
    sensor: crate::ColorModeSensor,
) -> crate::Result<([f64; 9], [f64; 3], f64)> {
    let invalid =
        || crate::Error::InvalidData("missing or invalid native color mode WB/ISO calibration");
    if wb.is_null() {
        wb = unsafe { x3f_get_wb(x3f) };
    }
    let mut cc = [0.0; 9];
    let mut found = unsafe {
        x3f_get_camf_matrix_for_wb(
            x3f,
            c"WhiteBalanceColorCorrections".as_ptr().cast_mut(),
            wb,
            3,
            3,
            cc.as_mut_ptr(),
        )
    };
    if found == 0 && !wb.is_null() && unsafe { CStr::from_ptr(wb) } == c"Sunlight" {
        wb = c"Daylight".as_ptr().cast_mut();
        found = unsafe {
            x3f_get_camf_matrix_for_wb(
                x3f,
                c"WhiteBalanceColorCorrections".as_ptr().cast_mut(),
                wb,
                3,
                3,
                cc.as_mut_ptr(),
            )
        };
    }
    if found == 0 || !crate::color_modes::valid_matrix(&cc) {
        return Err(invalid());
    }
    let (gain, iso) = match sensor {
        crate::ColorModeSensor::Merrill => (
            unsafe { merrill_wb_gain(x3f, wb) }.ok_or_else(invalid)?,
            unsafe { merrill_iso_gain(x3f) }.ok_or_else(invalid)?,
        ),
        crate::ColorModeSensor::Quattro => {
            let mut gain = [0.0; 3];
            if unsafe {
                x3f_get_camf_matrix_for_wb(
                    x3f,
                    c"WhiteBalanceGains".as_ptr().cast_mut(),
                    wb,
                    3,
                    0,
                    gain.as_mut_ptr(),
                )
            } == 0
                || gain.iter().any(|v| !v.is_finite() || *v <= 0.0)
            {
                return Err(invalid());
            }
            // GatherParams normalizes the unadjusted WB gains by their minimum.
            let minimum = gain.into_iter().fold(f64::INFINITY, f64::min);
            let mut temperature = [1.0; 3];
            unsafe {
                x3f_get_camf_float_vector(
                    x3f,
                    c"TempGainFact".as_ptr().cast_mut(),
                    temperature.as_mut_ptr(),
                );
            }
            let temperature = merrill_temperature_gain(temperature);
            let gain = if unsafe { sd_quattro_camera(x3f) }?.is_some() {
                // Color uses GatherParams WB_B (params+0x208): temperature-only
                // and green-normalized, with no HI/LOW sensor adjustment
                // (lib95 0xa2308, lib92 0x9f308, Stage2 0x89ad4).
                let fluorescent = if !wb.is_null()
                    && unsafe { CStr::from_ptr(wb) }
                        .to_bytes()
                        .eq_ignore_ascii_case(b"Fluorescent")
                {
                    unsafe { sd_quattro_fluorescent(x3f) }?
                } else {
                    [1.0; 3]
                };
                let b: [f64; 3] =
                    std::array::from_fn(|c| temperature[c] * gain[c] * fluorescent[c]);
                b.map(|v| v / b[1])
            } else {
                let mut adjustment = [1.0; 3];
                let mut version = [0_u32; 3];
                if unsafe {
                    x3f_get_camf_matrix(
                        x3f,
                        c"CamsysVersion".as_ptr().cast_mut(),
                        3,
                        0,
                        0,
                        matrix_type_t_M_UINT,
                        version.as_mut_ptr().cast(),
                    )
                } == 0
                {
                    return Err(crate::Error::InvalidData(
                        "missing Quattro CamsysVersion calibration",
                    ));
                }
                let version = u64::from(version[0]) * 1_000_000
                    + u64::from(version[1]) * 1_000
                    + u64::from(version[2]);
                if version <= 1999 {
                    unsafe {
                        x3f_get_camf_float_vector(
                            x3f,
                            c"SensorAdjustmentGainFact".as_ptr().cast_mut(),
                            adjustment.as_mut_ptr(),
                        );
                    }
                }
                std::array::from_fn(|c| (temperature[c] * adjustment[c]) * (gain[c] / minimum))
            };
            let (mut capture, mut range) = (0.0, 0);
            unsafe {
                x3f_get_camf_float(x3f, c"CaptureISO".as_ptr().cast_mut(), &mut capture);
                x3f_get_camf_unsigned(x3f, c"ISOrange".as_ptr().cast_mut(), &mut range);
            }
            let iso = merrill_render_iso_gain(capture, 100.0, 0, range).ok_or_else(invalid)?;
            (gain, iso)
        }
    };
    if gain.iter().any(|v| !v.is_finite() || *v <= 0.0) || !iso.is_finite() || iso <= 0.0 {
        return Err(invalid());
    }
    Ok((cc, gain, iso))
}

/// Capture calibration before image processing consumes or changes RAW samples.
/// # Safety
/// `x3f` must own loaded CAMF and unprocessed RAW, with optional PROP.
/// `wb` must be null or point to a NUL-terminated name. JPEG is loaded as needed.
pub unsafe fn color_mode_calibration(
    x3f: *mut x3f_t,
    wb: *mut libc::c_char,
    control: Control<'_>,
) -> crate::Result<crate::ColorModeCalibration> {
    control.check()?;
    let sensor = unsafe { crate::color_modes::sensor(x3f) }?;
    let (sensor_to_srgb, wb_gain, iso_gain) =
        unsafe { color_mode_base_calibration(x3f, wb, sensor) }?;
    let mut raw = x3f_area16_t::default();
    let mut top = x3f_area16_t::default();
    if unsafe { x3f_image_area(x3f, &mut raw) } == 0 || raw.data.is_null() {
        return Err(crate::Error::InvalidData(
            "color mode calibration requires unprocessed RAW samples",
        ));
    }
    let mut black = [0.0; 3];
    let mut deviation = [0.0; 3];
    let quattro = sensor == crate::ColorModeSensor::Quattro;
    if unsafe {
        get_black_level(
            x3f,
            &mut raw,
            1,
            if quattro { 2 } else { 3 },
            black.as_mut_ptr(),
            deviation.as_mut_ptr(),
        )
    } == 0
        || (quattro
            && (unsafe { x3f_image_area_qtop(x3f, &mut top) } == 0
                || unsafe {
                    get_black_level(
                        x3f,
                        &mut top,
                        0,
                        1,
                        black.as_mut_ptr().add(2),
                        deviation.as_mut_ptr().add(2),
                    )
                } == 0))
    {
        return Err(crate::Error::InvalidData(
            "missing color mode black calibration",
        ));
    }
    let mut maximum = [0; 3];
    if unsafe { x3f_get_max_raw(x3f, maximum.as_mut_ptr()) } == 0 {
        return Err(crate::Error::InvalidData("missing color mode sensor range"));
    }
    let (count_scale, native_black) = if sensor == crate::ColorModeSensor::Merrill {
        let mut shield = x3f_area16_t::default();
        if unsafe {
            x3f_crop_area_camf(
                x3f,
                c"DarkShieldTop".as_ptr().cast_mut(),
                &mut raw,
                1,
                &mut shield,
            )
        } == 0
        {
            return Err(crate::Error::InvalidData(
                "missing Merrill trimmed-black calibration",
            ));
        }
        ([unsafe { merrill_count_scale(x3f) }.unwrap(); 3], unsafe {
            merrill_trimmed_black(&shield, control)
        }?)
    } else if let Some((gain, offset)) = unsafe { sd_quattro_preprocess(x3f, maximum) }? {
        (gain, [offset; 3])
    } else {
        ([1.0; 3], black)
    };
    let (normalized_to_counts, count_offset) =
        native_count_span(maximum, black, native_black, count_scale);
    if normalized_to_counts
        .iter()
        .any(|v| !v.is_finite() || *v <= 0.0)
        || count_offset.iter().any(|v| !v.is_finite())
    {
        return Err(crate::Error::InvalidData(
            "invalid color mode sensor count range",
        ));
    }
    let exif = unsafe { merrill_exif(x3f, control) }?;
    let source = exif.color_space.ok_or(crate::Error::InvalidData(
        "missing color mode EXIF source color space",
    ))?;
    let output_to_srgb = color_mode_output_matrix(sensor, x3f_color_encoding_e_SRGB, source)
        .ok_or(crate::Error::InvalidData(
            "unsupported color mode EXIF source color space",
        ))?;
    let color_dq = if sensor == crate::ColorModeSensor::Merrill {
        Some(unsafe { merrill_color_dq(x3f) }?)
    } else {
        None
    };
    control.check()?;
    Ok(crate::ColorModeCalibration {
        sensor_to_srgb,
        wb_gain,
        normalized_to_counts,
        count_offset,
        iso_gain,
        output_to_srgb,
        color_dq,
    })
}

/// Map x in [0,1] to native counts: (x * span + offset) is
/// scale * (raw - native_black) for raw = black + x * (maximum - black).
/// dp Quattro keeps its pinned measured-black, unit-scale convention, although
/// lib95 also subtracts RawOffset and divides by SensorISO / 100 there.
fn native_count_span(
    maximum: [u32; 3],
    black: [f64; 3],
    native_black: [f64; 3],
    scale: [f64; 3],
) -> ([f64; 3], [f64; 3]) {
    (
        std::array::from_fn(|c| scale[c] * (f64::from(maximum[c]) - black[c])),
        std::array::from_fn(|c| scale[c] * (black[c] - native_black[c])),
    )
}

/// sd Quattro and H preprocess gain and black for
/// min(DigitalISOGain * (raw - RawOffset) * 100 / SensorISO, 65535)
/// (lib95 priProcessF23PreprocessStage 0x897e8, lib92 0x88324).
/// Only lib92 (sd Quattro H) reads DigitalISOGain. Returns None for other cameras.
pub(crate) unsafe fn sd_quattro_preprocess(
    x3f: *mut x3f_t,
    maximum: [u32; 3],
) -> crate::Result<Option<([f64; 3], f64)>> {
    let Some(camera) = unsafe { sd_quattro_camera(x3f) }? else {
        return Ok(None);
    };
    // Missing RawOffset uses the engine default: lib95 0x893b4, lib92 0x880ac.
    let mut raw_offset = if camera == X3F_CAMERAID_SDQ {
        2048
    } else {
        256
    };
    let (mut digital, mut sensor_iso) = ([1.0; 3], 0.0);
    unsafe {
        if camera == X3F_CAMERAID_SDQH {
            x3f_get_camf_float_vector(
                x3f,
                c"DigitalISOGain".as_ptr().cast_mut(),
                digital.as_mut_ptr(),
            );
        }
        x3f_get_camf_float(x3f, c"SensorISO".as_ptr().cast_mut(), &mut sensor_iso);
        x3f_get_camf_unsigned(x3f, c"RawOffset".as_ptr().cast_mut(), &mut raw_offset);
    }
    // GatherParams uses unit sensor ISO gain for zero (lib95 0xa18c0).
    if sensor_iso == 0.0 {
        sensor_iso = 100.0;
    }
    let raw_offset = f64::from(raw_offset);
    sd_quattro_gain(digital, sensor_iso, raw_offset, maximum)
        .map(|gain| Some((gain, raw_offset)))
        .ok_or(crate::Error::InvalidData(
            "unsupported sd Quattro digital gain, SensorISO or gained count range",
        ))
}

/// Native sd Quattro Fluorescent table (lib95 0x1a3da0): F-number, R, G, B rows.
const SDQ_FLUORESCENT: [u32; 24] = [
    0x3fb33333, 0x40000000, 0x40333333, 0x40800000, 0x40b33333, 0x41000000, //
    0x3f7c25af, 0x3f7df84d, 0x3f7f16b1, 0x3f7fc3b5, 0x3f800000, 0x3f800000, //
    0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000, //
    0x3f81a416, 0x3f80a5ce, 0x3f803223, 0x3f800cc8, 0x3f800000, 0x3f800000,
];

/// Fluorescent WB gain at the EXIF F-number. sd Quattro uses the built-in
/// table; sd Quattro H reads CAMF FnumberGainTable, and its absence is unity.
unsafe fn sd_quattro_fluorescent(x3f: *mut x3f_t) -> crate::Result<[f64; 3]> {
    let invalid = |m| Err(crate::Error::InvalidData(m));
    let f = unsafe { merrill_exif(x3f, Control::none()) }?
        .f_number
        .filter(|&(n, d)| n != 0 && d != 0)
        .map(|(n, d)| (f64::from(n) / f64::from(d)) as f32);
    let Some(f) = f else {
        return invalid("sd Quattro Fluorescent color modes require an EXIF FNumber");
    };
    let table: [f32; 24] = if unsafe { sd_quattro_camera(x3f) }? == Some(X3F_CAMERAID_SDQ) {
        SDQ_FLUORESCENT.map(f32::from_bits)
    } else if unsafe { crate::meta::camf_find_by_name(x3f, c"FnumberGainTable".as_ptr()) }.is_none()
    {
        return Ok([1.0; 3]);
    } else {
        let values =
            unsafe { crate::color_modes::float_matrix(x3f, c"FnumberGainTable", &[4, 6]) }?;
        std::array::from_fn(|i| values[i] as f32)
    };
    fluorescent_gain(&table, f).ok_or(crate::Error::InvalidData(
        "invalid sd Quattro FnumberGainTable",
    ))
}

/// FluorescentColorBaranceAdjustment (lib92 0x9fbfc, lib95 0xa218c) in float32.
fn fluorescent_gain(t: &[f32; 24], f: f32) -> Option<[f64; 3]> {
    if !t.iter().all(|v| v.is_finite()) || !t[..6].windows(2).all(|w| w[0] < w[1]) {
        return None;
    }
    let node = |i: usize| [t[6 + i], t[12 + i], t[18 + i]].map(f64::from);
    if f <= t[0] {
        return Some(node(0));
    }
    if f > t[5] {
        return Some(node(5));
    }
    let i = (1..6).find(|&i| t[i] >= f)?;
    // Native fmadd: (f - t0) * t1 + ((t1 - f) * t0) rounds once.
    Some([6, 12, 18].map(|b| {
        let near = (t[i] - f) * t[b + i - 1];
        f64::from((f - t[i - 1]).mul_add(t[b + i], near) / (t[i] - t[i - 1]))
    }))
}

/// Reject gains whose sensor range would exceed the native 65535 count clamp,
/// because the linear render matrix cannot express that clamp.
fn sd_quattro_gain(
    digital: [f64; 3],
    sensor_iso: f64,
    raw_offset: f64,
    maximum: [u32; 3],
) -> Option<[f64; 3]> {
    let positive = |v: f64| v.is_finite() && v > 0.0;
    if !positive(sensor_iso) || !digital.into_iter().all(positive) {
        return None;
    }
    let gain = digital.map(|g| g * (100.0 / sensor_iso));
    (0..3)
        .all(|c| {
            let top = gain[c] * (f64::from(maximum[c]) - raw_offset);
            positive(gain[c]) && top > 0.0 && top <= 65535.0
        })
        .then_some(gain)
}

/// Return the sd Quattro or H CAMERAID when it matches the RAW image type.
/// Reject a mismatch, so a new format never uses dp Quattro calibration.
pub(crate) unsafe fn sd_quattro_camera(x3f: *mut x3f_t) -> crate::Result<Option<u32>> {
    let raw = unsafe { x3f_get_raw(x3f) };
    let format = if raw.is_null() {
        0
    } else {
        unsafe { (*raw).header.data_subsection.image_data.type_format }
    };
    let mut camera = 0;
    unsafe {
        x3f_get_camf_unsigned(x3f, c"CAMERAID".as_ptr().cast_mut(), &mut camera);
    }
    match (format, camera) {
        (0x0001_0025, X3F_CAMERAID_SDQ) | (0x0001_0027, X3F_CAMERAID_SDQH) => Ok(Some(camera)),
        (0x0001_0025 | 0x0001_0027, _) | (_, X3F_CAMERAID_SDQ | X3F_CAMERAID_SDQH) => Err(
            crate::Error::InvalidData("sd Quattro CAMERAID does not match the RAW image type"),
        ),
        _ => Ok(None),
    }
}

fn merrill_render_matrix(
    encoding: x3f_color_encoding_t,
    mut cc: [f64; 9],
    mode: Option<[f64; 9]>,
    gain: [f64; 3],
    iso: f64,
) -> Option<[f64; 9]> {
    // Native Stage4 source-sRGB matrices, @0xac024/0xac0ac/0xac470.
    let mut target = match encoding {
        x3f_color_encoding_e_SRGB => [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        x3f_color_encoding_e_ARGB => [
            0.715162633,
            0.28483729,
            0.0000294,
            -0.0000408,
            0.999979,
            -0.0000327,
            -0.0000616,
            0.041159,
            0.95904,
        ],
        x3f_color_encoding_e_PPRGB => [
            0.52936, 0.330103, 0.140661, 0.098388, 0.873463, 0.028098, 0.016851, 0.117711, 0.865681,
        ],
        _ => return None,
    };
    if !iso.is_finite() || iso <= 0.0 || gain.iter().any(|v| !v.is_finite() || *v <= 0.0) {
        return None;
    }
    for value in &mut cc {
        *value *= iso;
    }
    if let Some(mut mode) = mode.filter(|m| m[0] != 0.0) {
        let mut composed = [0.0; 9];
        unsafe { x3f_3x3_3x3_mul(cc.as_mut_ptr(), mode.as_mut_ptr(), composed.as_mut_ptr()) };
        cc = composed;
    }
    let mut output = [0.0; 9];
    unsafe { x3f_3x3_3x3_mul(target.as_mut_ptr(), cc.as_mut_ptr(), output.as_mut_ptr()) };
    for row in output.chunks_exact_mut(3) {
        for (value, gain) in row.iter_mut().zip(gain) {
            *value *= gain;
        }
    }
    output.iter().all(|v| v.is_finite()).then_some(output)
}

unsafe fn get_render_conv_controlled(
    x3f: *mut x3f_t,
    encoding: x3f_color_encoding_t,
    wb: *mut libc::c_char,
    lutsize: libc::c_int,
    max_out: u16,
    lut: *mut f64,
    conv_matrix: *mut f64,
    cineon: bool,
    resolved: Option<&crate::ColorModeParameters>,
) -> libc::c_int {
    let mut selected_wb = wb;
    let mut native_matrix = None;
    if !cineon && unsafe { is_merrill_model(x3f) } {
        let mode = unsafe { (*x3f).header.color_mode.map(|byte| byte as u8) };
        if let Some(mode) = resolved
            .map(|p| p.mode.key())
            .or_else(|| merrill_render_color_mode(&mode))
        {
            let mut cc = [0.0; 9];
            let mut found = unsafe {
                x3f_get_camf_matrix_for_wb(
                    x3f,
                    c"WhiteBalanceColorCorrections".as_ptr() as *mut _,
                    selected_wb,
                    3,
                    3,
                    cc.as_mut_ptr(),
                )
            };
            // The shared accessor already retries Daylight -> Sunlight.
            if found == 0
                && !wb.is_null()
                && unsafe { CStr::from_ptr(wb) }.to_bytes() == b"Sunlight"
            {
                selected_wb = c"Daylight".as_ptr() as *mut _;
                found = unsafe {
                    x3f_get_camf_matrix_for_wb(
                        x3f,
                        c"WhiteBalanceColorCorrections".as_ptr() as *mut _,
                        selected_wb,
                        3,
                        3,
                        cc.as_mut_ptr(),
                    )
                };
            }
            if found != 0 {
                let Some(iso) = (unsafe { merrill_iso_gain(x3f) }) else {
                    return 0;
                };
                let Some(gain) = (unsafe { merrill_wb_gain(x3f, selected_wb) }) else {
                    return 0;
                };
                let mut compensation = [0.0; 9];
                let has_compensation = unsafe {
                    x3f_get_camf_matrix_for_wb(
                        x3f,
                        c"ColorModeCompensations".as_ptr() as *mut _,
                        mode.as_ptr() as *mut _,
                        3,
                        3,
                        compensation.as_mut_ptr(),
                    )
                } != 0;
                native_matrix = merrill_render_matrix(
                    encoding,
                    cc,
                    resolved.map_or(has_compensation.then_some(compensation), |p| {
                        p.creative_matrix
                    }),
                    gain,
                    iso,
                );
                if native_matrix.is_none() {
                    return 0;
                }
            } else {
                return 0;
            }
        }
    }
    let result = unsafe {
        get_conv_controlled(
            x3f,
            encoding,
            selected_wb,
            lutsize,
            max_out,
            lut,
            conv_matrix,
            cineon,
        )
    };
    if result != 0 {
        if let Some(matrix) = native_matrix {
            // The caller separately restores count units and applies tone/output.
            unsafe { std::slice::from_raw_parts_mut(conv_matrix, 9) }.copy_from_slice(&matrix);
        }
    }
    result
}

unsafe fn get_conv_controlled(
    x3f: *mut x3f_t,
    encoding: x3f_color_encoding_t,
    wb: *mut libc::c_char,
    lutsize: libc::c_int,
    max_out: u16,
    lut: *mut f64,
    conv_matrix: *mut f64,
    cineon: bool,
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

        // Fold the per-channel CAMF DigitalISOGain into the matrix as
        // conv_matrix · diag(g): the matrix input is per-channel
        // sat_ratio, so the gain is a column scale. Applying it here —
        // after highlight reconstruction has run on honest sat_ratio —
        // rather than baking it into the raw samples keeps sensor
        // saturation at `il.white`. Guarded so the no-gain path stays
        // bit-identical.
        let mut dig_gain = [1.0_f64; 3];
        x3f_get_digital_iso_gain(x3f, dig_gain.as_mut_ptr());
        if dig_gain != [1.0; 3] {
            let cm = std::slice::from_raw_parts_mut(conv_matrix, 9);
            for row in 0..3 {
                for col in 0..3 {
                    cm[3 * row + col] *= dig_gain[col];
                }
            }
        }

        x3f_printf(x3f_verbosity_t_DEBUG, c"raw_to_rgb\n".as_ptr());
        x3f_3x3_print(x3f_verbosity_t_DEBUG, raw_to_rgb.as_mut_ptr());
        x3f_printf(x3f_verbosity_t_DEBUG, c"conv_matrix\n".as_ptr());
        x3f_3x3_print(x3f_verbosity_t_DEBUG, conv_matrix);
    }
    1
}

struct MerrillRender {
    black: [f64; 3],
    spatial: Option<[crate::spatial_gain::MerrillSpatialTable; 3]>,
    matrix: [f64; 9],
    color_dq: crate::merrill::ColorDq,
    luts: crate::merrill::RenderLuts,
}

struct QuattroColorModeRender {
    matrix: [f64; 9],
    offset: [f64; 3],
    recovery_matrix: [f64; 9],
    luts: crate::merrill::RenderLuts,
}

impl QuattroColorModeRender {
    fn new(
        calibration: &crate::ColorModeCalibration,
        params: &crate::ColorModeParameters,
        luts: crate::merrill::RenderLuts,
    ) -> crate::Result<Self> {
        let (matrix, offset) = calibration.render_matrix(params);
        let mut physical = *params;
        physical.creative_matrix = None;
        let (physical_matrix, _) = calibration.render_matrix(&physical);
        // Quattro recovery uses physical calibration before the selected look.
        // It classifies normalized sensor input, so it omits the native pedestal.
        let recovery_matrix = physical_matrix.map(|v| v / params.pixel_saturation);
        if matrix
            .iter()
            .chain(&recovery_matrix)
            .any(|v| !v.is_finite())
        {
            return Err(crate::Error::InvalidData(
                "invalid Quattro count calibration",
            ));
        }
        Ok(Self {
            matrix,
            offset,
            recovery_matrix,
            luts,
        })
    }

    fn render(&self, input: [f64; 3]) -> [u16; 3] {
        let counts = mat3x1_mul_native(&self.matrix, input);
        let indices = std::array::from_fn(|c| {
            crate::color_mode_tone_index(
                crate::ColorModeSensor::Quattro,
                counts[c] + self.offset[c],
            )
        });
        // shortcut: use the native ordinary restore bounds with neutral residual
        // gain until the native spatial filters replace the existing NLM path.
        self.luts.render_indices(indices)
    }
}

unsafe fn make_quattro_color_mode_render(
    x3f: *mut x3f_t,
    wb: *mut libc::c_char,
    encoding: x3f_color_encoding_t,
    params: &crate::ColorModeParameters,
    control: Control<'_>,
) -> crate::Result<QuattroColorModeRender> {
    let calibration = unsafe { color_mode_calibration(x3f, wb, control) }?;
    let exif = unsafe { merrill_exif(x3f, control) }?;
    let luts =
        unsafe { make_color_mode_render_luts(x3f, encoding, &exif, false, Some(params), control) }?;
    QuattroColorModeRender::new(&calibration, params, luts)
}

fn recovery_conv_matrix(legacy: [f64; 9], quattro: Option<&QuattroColorModeRender>) -> [f64; 9] {
    quattro.map_or(legacy, |render| render.recovery_matrix)
}

impl MerrillRender {
    fn render(&self, input: [f64; 3]) -> [u16; 3] {
        let output = std::array::from_fn(|c| {
            let row = &self.matrix[c * 3..c * 3 + 3];
            row[2].mul_add(input[2], row[0].mul_add(input[0], row[1] * input[1]))
        });
        let output = self.color_dq.apply(output);
        // shortcut: neutral residual gain with ApplyColorResidual's
        // [0,32766] bounds (0x7936c..0x793dc) until native tile filters replace NLM.
        self.luts.render(output.map(|v| v.clamp(0.0, 32766.0)))
    }
}

pub(crate) unsafe fn merrill_color_dq(x3f: *mut x3f_t) -> crate::Result<crate::merrill::ColorDq> {
    use crate::merrill::{color_dq_noise_scale, ColorDq};
    let mut skip = 0;
    unsafe {
        x3f_get_camf_unsigned(x3f, c"SkipStepFlags".as_ptr() as *mut _, &mut skip);
    }
    if skip & (1 << 23) != 0 {
        return Err(crate::Error::InvalidData(
            "Merrill denoise bypass is not reconstructed",
        ));
    }
    let invalid = || crate::Error::InvalidData("unsupported Merrill ColorDQ calibration");
    let mut tables = [Vec::<[f32; 13]>::new(), Vec::new()];
    for (table, name) in tables
        .iter_mut()
        .zip([c"ISONoiseSettings", c"DenoisingSlidersStep"])
    {
        let (mut cols, mut rows, mut data) = (0, 0, ptr::null_mut());
        // CAMF dimensions are outermost-first; the last dimension is the
        // contiguous 13-value settings row (native SetUpNoiseParams @0xad49c).
        if unsafe {
            x3f_get_camf_matrix_var(
                x3f,
                name.as_ptr() as *mut _,
                &mut rows,
                &mut cols,
                ptr::null_mut(),
                matrix_type_t_M_FLOAT,
                &mut data,
            )
        } != 0
        {
            if cols != 13 || !(1..20).contains(&rows) {
                return Err(invalid());
            }
            let values =
                unsafe { std::slice::from_raw_parts(data.cast::<f64>(), (cols * rows) as usize) };
            table.extend(
                values
                    .chunks_exact(13)
                    .map(|row| std::array::from_fn(|c| row[c] as f32)),
            );
        }
    }
    let sliders = (!tables[1].is_empty()).then_some(tables[1].as_slice());
    // shortcut: accept slider-independent ColorDQ until SPP's adjustment inputs
    // and EstimatedISO are reconstructed; never substitute CaptureISO for it.
    if sliders.is_some_and(|rows| rows.iter().any(|r| r[7] != rows[0][7])) {
        return Err(invalid());
    }
    let noise_scale = color_dq_noise_scale(&tables[0], sliders, None, 0.0)?;
    let iso = unsafe { merrill_iso_gain(x3f) }.ok_or_else(invalid)?;
    let mut strengths = [0.0; 3];
    if unsafe {
        x3f_get_camf_float_vector(
            x3f,
            c"ColorDQCamRGB".as_ptr() as *mut _,
            strengths.as_mut_ptr(),
        )
    } == 0
    {
        return Err(invalid());
    }
    let scale = f64::from(noise_scale).min(iso);
    ColorDq::new(strengths.map(|v| v * scale))
}

unsafe fn make_merrill_render(
    x3f: *mut x3f_t,
    wb: *mut libc::c_char,
    encoding: x3f_color_encoding_t,
    apply_sgain: bool,
    control: Control<'_>,
    resolved: Option<&crate::ColorModeParameters>,
) -> crate::Result<MerrillRender> {
    control.check()?;
    let mut raw: x3f_area16_t = unsafe { std::mem::zeroed() };
    let mut shield: x3f_area16_t = unsafe { std::mem::zeroed() };
    let mut max_raw = [0; 3];
    // Native GetRectFromCamfile adds one to stored right/bottom endpoints;
    // crop_area_camf already treats those CAMF coordinates as inclusive.
    if unsafe {
        x3f_image_area(x3f, &mut raw) == 0
            || x3f_crop_area_camf(
                x3f,
                c"DarkShieldTop".as_ptr() as *mut _,
                &mut raw,
                1,
                &mut shield,
            ) == 0
            || x3f_get_max_raw(x3f, max_raw.as_mut_ptr()) == 0
    } {
        return Err(crate::Error::InvalidData(
            "missing Merrill black calibration",
        ));
    }
    let black = unsafe { merrill_trimmed_black(&shield, control) }?;
    let exif = unsafe { merrill_exif(x3f, control) }?;
    let spatial = unsafe { merrill_spatial_tables(x3f, apply_sgain, exif.f_number, control) }?;
    if spatial.is_some() && (raw.rows < 2 || raw.columns == 0) {
        return Err(crate::Error::InvalidData(
            "invalid Merrill spatial image size",
        ));
    }
    let span: [f64; 3] = std::array::from_fn(|c| f64::from(max_raw[c]) - black[c]);
    if span.iter().any(|v| !v.is_finite() || *v <= 0.0) {
        return Err(crate::Error::InvalidData("invalid Merrill sensor range"));
    }
    let count_scale = unsafe { merrill_count_scale(x3f) }.ok_or(crate::Error::InvalidData(
        "unsupported Merrill process version",
    ))?;
    let mut matrix = [0.0; 9];
    let mut unused_gamma = [0.0; 2];
    if unsafe {
        get_render_conv_controlled(
            x3f,
            x3f_color_encoding_e_SRGB,
            wb,
            2,
            65535,
            unused_gamma.as_mut_ptr(),
            matrix.as_mut_ptr(),
            false,
            resolved,
        )
    } == 0
    {
        return Err(crate::Error::InvalidData(
            "missing Merrill render calibration",
        ));
    }
    let color_dq = unsafe { merrill_color_dq(x3f) }?;
    // shortcut: fold count gain here until float preprocessing preserves native rounding.
    for row in matrix.chunks_exact_mut(3) {
        for c in 0..3 {
            row[c] *= count_scale * span[c];
        }
    }
    if matrix.iter().any(|v| !v.is_finite()) {
        return Err(crate::Error::InvalidData(
            "invalid Merrill count calibration",
        ));
    }
    let luts =
        unsafe { make_color_mode_render_luts(x3f, encoding, &exif, false, resolved, control) }?;
    Ok(MerrillRender {
        black,
        spatial,
        matrix,
        color_dq,
        luts,
    })
}

unsafe fn make_color_mode_render_luts(
    x3f: *mut x3f_t,
    encoding: x3f_color_encoding_t,
    exif: &crate::jpeg_exif::JpegExif,
    spp: bool,
    resolved: Option<&crate::ColorModeParameters>,
    control: Control<'_>,
) -> crate::Result<crate::merrill::RenderLuts> {
    let Some(params) = resolved else {
        return unsafe { make_merrill_render_luts(x3f, encoding, exif, spp, control) };
    };
    let source = exif.color_space.ok_or(crate::Error::InvalidData(
        "missing color mode EXIF source color space",
    ))?;
    let target = color_mode_output_matrix(params.sensor, encoding, source).ok_or(
        crate::Error::InvalidData("unsupported color mode output encoding"),
    )?;
    let native_target = match encoding {
        x3f_color_encoding_e_SRGB => 2,
        x3f_color_encoding_e_ARGB => 3,
        x3f_color_encoding_e_PPRGB => 11,
        _ => {
            return Err(crate::Error::InvalidData(
                "color modes require a rendered color encoding",
            ))
        }
    };
    crate::merrill::RenderLuts::from_color_mode(
        params,
        env_atof("X3F_EV").unwrap_or(0.0),
        target,
        native_target,
        control,
    )
}

unsafe fn make_merrill_render_luts(
    x3f: *mut x3f_t,
    encoding: x3f_color_encoding_t,
    exif: &crate::jpeg_exif::JpegExif,
    spp: bool,
    control: Control<'_>,
) -> crate::Result<crate::merrill::RenderLuts> {
    use crate::merrill::{RenderLuts, ToneParams, ToneSettings};
    let mode_bytes = unsafe { (*x3f).header.color_mode.map(|b| b as u8) };
    let mode = merrill_render_color_mode(&mode_bytes).unwrap_or(c"Standard");
    let (mut names, mut values, mut count) = (ptr::null_mut(), ptr::null_mut(), 0);
    let (mut contrast, mut ordinal) = (0.0, 0);
    if unsafe {
        x3f_get_camf_property_list(
            x3f,
            c"ColorModeContrastCompensations".as_ptr() as *mut _,
            &mut names,
            &mut values,
            &mut count,
        )
    } != 0
    {
        ordinal = count as usize;
        for i in 0..count as usize {
            let name = unsafe { *names.add(i) };
            if !name.is_null() && unsafe { CStr::from_ptr(name) } == mode {
                ordinal = i;
                unsafe {
                    x3f_get_camf_float(x3f, *values.add(i), &mut contrast);
                }
                break;
            }
        }
    }
    if spp && mode == c"FCBlue" && contrast == 0. {
        contrast = 0.3;
    }
    let (mut cols, mut rows, mut data) = (0, 0, ptr::null_mut());
    let mut settings_rows = Vec::new();
    if unsafe {
        x3f_get_camf_matrix_var(
            x3f,
            c"TCColorModeSettings".as_ptr() as *mut _,
            &mut rows,
            &mut cols,
            ptr::null_mut(),
            matrix_type_t_M_FLOAT,
            &mut data,
        )
    } != 0
        && cols == 7
        && (1..=256).contains(&rows)
    {
        let values =
            unsafe { std::slice::from_raw_parts(data.cast::<f64>(), (rows * cols) as usize) };
        settings_rows.extend(
            values
                .chunks_exact(7)
                .map(|r| std::array::from_fn(|i| r[i] as f32)),
        );
    }
    let mut depth = 12;
    unsafe {
        x3f_get_camf_unsigned(x3f, c"ImageDepth".as_ptr() as *mut _, &mut depth);
    }
    if !(1..=16).contains(&depth) {
        return Err(crate::Error::InvalidData("invalid Merrill image depth"));
    }
    let source_color = exif.color_space.ok_or(crate::Error::InvalidData(
        "missing Merrill EXIF source color space",
    ))?;
    let target = merrill_output_matrix(encoding, source_color).ok_or(crate::Error::InvalidData(
        "unsupported Merrill RGB encoding",
    ))?;
    let native_target = match encoding {
        x3f_color_encoding_e_SRGB => 2,
        x3f_color_encoding_e_ARGB => 3,
        x3f_color_encoding_e_PPRGB => 11,
        _ => unreachable!(),
    };
    let luts = RenderLuts::new(
        ToneParams::from_color_mode_rows(&settings_rows, ordinal),
        ToneSettings {
            pixel_saturation: f64::from((1u32 << depth) - 1),
            exposure: env_atof("X3F_EV").unwrap_or(0.0),
            shadows: 0.0,
            contrast,
            highlights: 0.0,
            alternate: false,
        },
        target,
        native_target,
        control,
    )?;
    Ok(luts)
}

unsafe fn merrill_spatial_tables(
    x3f: *mut x3f_t,
    enabled: bool,
    f_number: Option<(u32, u32)>,
    control: Control<'_>,
) -> crate::Result<Option<[crate::spatial_gain::MerrillSpatialTable; 3]>> {
    let mut skip = 0;
    unsafe {
        x3f_get_camf_unsigned(x3f, c"SkipStepFlags".as_ptr() as *mut _, &mut skip);
    }
    if !enabled || skip & (1 << 4) != 0 {
        return Ok(None);
    }
    // Native preprocess @0xa5b74 divides the request rational in f64, then casts at 0xa6618.
    let (numerator, denominator) = match f_number.filter(|(_, d)| *d != 0) {
        Some(value) => value,
        None => {
            let mut property = ptr::null_mut();
            if unsafe { x3f_get_prop_entry(x3f, c"APERTURE".as_ptr().cast_mut(), &mut property) }
                == 0
                || property.is_null()
            {
                return Err(crate::Error::InvalidData(
                    "missing Merrill spatial aperture",
                ));
            }
            unsafe { CStr::from_ptr(property) }
                .to_str()
                .ok()
                .and_then(merrill_prop_aperture)
                .ok_or(crate::Error::InvalidData(
                    "invalid Merrill spatial aperture",
                ))?
        }
    };
    let aperture = (f64::from(numerator) / f64::from(denominator)) as f32;
    unsafe { crate::spatial_gain::merrill_spatial_gain(x3f, aperture, control) }
}

fn merrill_prop_aperture(value: &str) -> Option<(u32, u32)> {
    let value = value.trim();
    if value.contains('.') {
        // Managed fillAdjustParamFromFileInfoX3F @0x1014df3c8..448.
        let numerator = f64::from(value.parse::<f32>().ok()? * 100000.0).round_ties_even();
        (numerator.is_finite() && numerator > 0.0 && numerator <= u32::MAX as f64)
            .then_some((numerator as u32, 100000))
    } else if let Some((n, d)) = value.split_once('/') {
        let (n, d) = (n.trim().parse::<u32>().ok()?, d.trim().parse::<u32>().ok()?);
        (n > 0 && d > 0).then_some((n, d))
    } else {
        let numerator = value.parse::<u32>().ok()?.checked_mul(100000)?;
        (numerator > 0).then_some((numerator, 100000))
    }
}

pub(crate) unsafe fn merrill_exif(
    x3f: *mut x3f_t,
    control: Control<'_>,
) -> crate::Result<crate::jpeg_exif::JpegExif> {
    control.check()?;
    let entry = unsafe { x3f_get_thumb_jpeg(x3f) };
    if entry.is_null() {
        return Err(crate::Error::InvalidData("missing Merrill JPEG EXIF"));
    }
    if unsafe { (*entry).header.data_subsection.image_data.data.is_null() } {
        unsafe { crate::load_data(x3f, entry, control) }?;
    }
    let image = unsafe { &(*entry).header.data_subsection.image_data };
    if image.data.is_null() || image.data_size == 0 {
        return Err(crate::Error::InvalidData("empty Merrill JPEG EXIF"));
    }
    let jpeg =
        unsafe { std::slice::from_raw_parts(image.data.cast::<u8>(), image.data_size as usize) };
    crate::jpeg_exif::parse_jpeg_exif(jpeg)
        .ok_or(crate::Error::InvalidData("invalid Merrill JPEG EXIF"))
}

fn merrill_output_matrix(
    encoding: x3f_color_encoding_t,
    exif_color_space: u16,
) -> Option<[f64; 9]> {
    let identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let target = merrill_render_matrix(encoding, identity, None, [1.0; 3], 1.0)?;
    if exif_color_space == 1 {
        return Some(target);
    }
    if encoding == x3f_color_encoding_e_ARGB {
        return Some(identity);
    }
    // Native CalculateInverse(C3) @0xac604, verified by direct library call.
    let inverse = [
        1.3982606199511893,
        -0.39828280643279224,
        -5.6444684264386543e-5,
        5.7053088178776595e-5,
        1.0000033459087134,
        3.4094961680870946e-5,
        8.7362994382552355e-5,
        -0.04294260086663016,
        1.0427079091680009,
    ];
    Some(std::array::from_fn(|i| {
        let (r, c) = (i / 3, i % 3);
        let mut value = 0.0;
        for k in 0..3 {
            value = target[3 * r + k].mul_add(inverse[3 * k + c], value);
        }
        value
    }))
}

fn color_mode_output_matrix(
    sensor: crate::ColorModeSensor,
    encoding: x3f_color_encoding_t,
    exif_color_space: u16,
) -> Option<[f64; 9]> {
    if sensor == crate::ColorModeSensor::Merrill {
        return merrill_output_matrix(encoding, exif_color_space);
    }
    let identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    // Quattro Stage4 skips matching spaces and otherwise uses its fixed target matrix.
    // It does not apply the Merrill Adobe source inverse.
    if encoding == x3f_color_encoding_e_ARGB && exif_color_space != 1 {
        return Some(identity);
    }
    merrill_render_matrix(encoding, identity, None, [1.0; 3], 1.0)
}

fn merrill_setup_or_legacy(
    setup: crate::Result<MerrillRender>,
) -> crate::Result<Option<MerrillRender>> {
    match setup {
        Ok(render) => Ok(Some(render)),
        Err(error @ crate::Error::InvalidData(_)) => {
            if let Ok(message) = std::ffi::CString::new(error.to_string()) {
                unsafe {
                    x3f_printf(
                        x3f_verbosity_t_WARN,
                        c"Merrill RGB setup unsupported; using legacy rendering: %s\n".as_ptr(),
                        message.as_ptr(),
                    );
                }
            }
            Ok(None)
        }
        Err(error) => Err(error),
    }
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
struct ConvCtx<'a> {
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
    merrill: Option<&'a MerrillRender>,
    quattro_mode: Option<&'a QuattroColorModeRender>,
    #[cfg(test)]
    recovered_inputs: Option<&'a std::sync::Mutex<Vec<[u64; 3]>>>,
}

// SAFETY: see ConvCtx doc-comment above. The pointers are either
// read-only over the lifetime of convert_data or alias `row_data` only
// in shape (the `img.data` whole-slice pointer); rayon's row chunks
// guarantee disjoint writes.
unsafe impl Send for ConvCtx<'_> {}
unsafe impl Sync for ConvCtx<'_> {}

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
) -> crate::Result<()> {
    // Hoist the conv_matrix and LUT into typed slices/array refs once
    // per row so the inner loop sees an inlinable native helper instead
    // of an extern "C" indirect call per pixel × per matrix mul.
    let m: &[f64; 9] = unsafe { &*(ctx.conv_matrix as *const [f64; 9]) };
    let lut: &[f64] = unsafe { std::slice::from_raw_parts(ctx.lut, LUTSIZE as usize) };
    let mut spatial = ctx
        .merrill
        .and_then(|render| render.spatial.as_ref())
        .map(|tables| {
            crate::spatial_gain::MerrillSpatialRow::new(
                tables,
                row as usize,
                ctx.rows as usize,
                ctx.cols as usize,
            )
        })
        .transpose()?;

    for col in 0..ctx.cols {
        let off = col as usize * ctx.channels;

        let mut sat_ratio = [0.0_f64; 3];
        // shortcut: native interpolation still runs after legacy preprocessing;
        // move it to ApplyGains when the float front end replaces normalized u16.
        let mut sg = spatial.as_mut().map_or([1.0; 3], |row| row.next().unwrap());
        for color in 0..3 {
            let v = row_data[off + color] as f64;
            sat_ratio[color] =
                (v - ctx.black[color]) / (ctx.white[color] as f64 - ctx.black[color]);
            if ctx.merrill.is_none() {
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
        #[cfg(test)]
        if let Some(inputs) = ctx.recovered_inputs {
            inputs.lock().unwrap().push(input.map(f64::to_bits));
        }
        if let Some(native) = ctx.merrill {
            let encoded = native.render(input);
            row_data[off..off + 3].copy_from_slice(&encoded);
            continue;
        }
        if let Some(native) = ctx.quattro_mode {
            row_data[off..off + 3].copy_from_slice(&native.render(input));
            continue;
        }
        let output = mat3x1_mul_native(m, input);
        for color in 0..3 {
            let scaled = output[color] * ctx.ev_scale;
            row_data[off + color] = lut_lookup_native(lut, scaled);
        }
    }
    Ok(())
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
    unsafe {
        convert_data_controlled(
            x3f,
            image,
            ilevels,
            encoding,
            apply_sgain,
            wb,
            Control::none(),
            CINEON.with(|c| c.get()),
            None,
            None,
            None,
            None,
        )
        .is_ok() as libc::c_int
    }
}

unsafe fn convert_data_controlled(
    x3f: *mut x3f_t,
    image: *mut x3f_area16_t,
    ilevels: *mut x3f_image_levels_t,
    encoding: x3f_color_encoding_t,
    apply_sgain: libc::c_int,
    wb: *mut libc::c_char,
    control: Control<'_>,
    cineon: bool,
    rgb_chroma_lut: Option<bool>,
    merrill: Option<MerrillRender>,
    resolved: Option<&crate::ColorModeParameters>,
    quattro_mode: Option<QuattroColorModeRender>,
) -> crate::Result<()> {
    let max_out: u16 = 65535; // TODO: should be possible to adjust

    let img = unsafe { &mut *image };
    if img.channels < 3 {
        return Err(crate::Error::InvalidData(
            "RGB conversion needs three channels",
        ));
    }

    let mut conv_matrix = [0.0_f64; 9];
    let mut lut = [0.0_f64; LUTSIZE as usize];

    let matrix_ok = if merrill.is_some() {
        unsafe {
            get_render_conv_controlled(
                x3f,
                encoding,
                wb,
                LUTSIZE,
                max_out,
                lut.as_mut_ptr(),
                conv_matrix.as_mut_ptr(),
                cineon,
                resolved,
            )
        }
    } else {
        unsafe {
            get_conv_controlled(
                x3f,
                encoding,
                wb,
                LUTSIZE,
                max_out,
                lut.as_mut_ptr(),
                conv_matrix.as_mut_ptr(),
                cineon,
            )
        }
    };
    if matrix_ok == 0 {
        return Err(crate::Error::InvalidData(
            "could not prepare RGB calibration",
        ));
    }
    conv_matrix = recovery_conv_matrix(conv_matrix, quattro_mode.as_ref());

    // Spatial gain (M6d) — corr table on the stack, MAXCORR=6 slots.
    let mut sgain: [x3f_spatial_gain_corr_t; MAX_CORR] = unsafe { std::mem::zeroed() };
    let sgain_num = if apply_sgain != 0 && merrill.is_none() {
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

    // The explicit per-call Cineon option is shared by all downstream
    // decisions (CLUT, RepairPix, ConvCtx field population).

    // Sigma-style chromaticity LUT (production highlight path). Forced
    // off in cineon mode — it bakes scene-derived chroma recovery into
    // the pixels, which fights the colour-grader's input transform.
    let mut clut: chroma_lut_t = unsafe { std::mem::zeroed() };
    let mut use_clut =
        !cineon && rgb_chroma_lut.unwrap_or_else(|| !env_present("X3F_NO_CHROMA_LUT"));
    if use_clut {
        unsafe { chroma_lut_init_defaults(&mut clut) };
        if unsafe {
            crate::highlight::chroma_lut_build_from_image_masked(
                &mut clut,
                img,
                ilevels,
                prior.as_ptr(),
                None,
                control,
                true,
            )
        } == 0
        {
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
        sat_map = unsafe {
            crate::highlight::build_sat_map_controlled(img, ilevels, repair.sat_threshold, control)
        };
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
        merrill: merrill.as_ref(),
        quattro_mode: quattro_mode.as_ref(),
        #[cfg(test)]
        recovered_inputs: None,
    };

    let total_u16 = (img.rows as usize) * row_stride;
    let data = unsafe { std::slice::from_raw_parts_mut(img.data, total_u16) };

    let result = if trace_clut {
        // Serial path — `clut_stats` accumulator is not Sync.
        let stats_ptr = &mut clut_stats as *mut _;
        data.chunks_mut(row_stride)
            .enumerate()
            .try_for_each(|(row, row_data)| {
                control.check()?;
                unsafe { convert_row(&ctx, row as i32, row_data, stats_ptr) }
            })
    } else {
        use rayon::prelude::*;
        data.par_chunks_mut(row_stride)
            .enumerate()
            .try_for_each(|(row, row_data)| {
                control.check()?;
                unsafe { convert_row(&ctx, row as i32, row_data, ptr::null_mut()) }
            })
    };

    if !sat_map.is_null() {
        unsafe { libc::free(sat_map as *mut libc::c_void) };
    }
    unsafe { x3f_cleanup_spatial_gain(sgain.as_mut_ptr(), sgain_num) };
    if trace_clut {
        unsafe { chroma_lut_apply_stats_print(&clut_stats, c"tiff".as_ptr()) };
    }
    control.check().and(result)?;

    il.black[0] = 0.0;
    il.black[1] = 0.0;
    il.black[2] = 0.0;
    il.white[0] = max_out as u32;
    il.white[1] = max_out as u32;
    il.white[2] = max_out as u32;

    Ok(())
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
// and hands off to the pure-Rust NLM in `src/denoise.rs`.
// `expand_quattro` arranges crop rectangles, allocates the expanded RGB
// buffer, and hands off to the Rust `x3f_expand_quattro` (M5a).

/// Map the 0..=10 denoise-intensity knob to the NLM sigma scale (0.0..=1.0)
/// the denoise kernels multiply onto each sensor's base `h`. 10 → 1.0 is
/// full-strength denoise; 0 → 0.0 is a no-op (callers gate it out before
/// reaching the kernel anyway).
#[inline]
fn denoise_scale(intensity: libc::c_int) -> f32 {
    (intensity.clamp(0, 10) as f32) / 10.0
}

#[no_mangle]
pub unsafe extern "C" fn run_denoising(x3f: *mut x3f_t, intensity: libc::c_int) -> libc::c_int {
    unsafe {
        run_denoising_controlled(
            x3f,
            intensity,
            Control::none().with_denoise_backend(crate::DenoiseBackend::Cpu),
        )
    }
    .is_ok() as libc::c_int
}

unsafe fn run_denoising_controlled(
    x3f: *mut x3f_t,
    intensity: libc::c_int,
    control: Control<'_>,
) -> crate::Result<()> {
    let mut original_image: x3f_area16_t = unsafe { std::mem::zeroed() };
    let mut image: x3f_area16_t = unsafe { std::mem::zeroed() };

    if unsafe { x3f_image_area(x3f, &mut original_image) } == 0 {
        return Err(crate::Error::InvalidData("missing denoising image"));
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
    let mut t = crate::denoise::DENOISE_STD;
    if unsafe { x3f_get_prop_entry(x3f, c"SENSORID".as_ptr() as *mut _, &mut sensorid) } != 0 {
        let s = unsafe { CStr::from_ptr(sensorid) }.to_bytes();
        if s == b"F20" {
            t = crate::denoise::DENOISE_F20;
        }
    }

    let scale = denoise_scale(intensity);
    unsafe {
        crate::denoise::denoise_area_controlled(
            &mut image as *mut x3f_area16_t as *mut crate::quattro::Area16,
            t,
            scale,
            control,
        )
    }
}

#[no_mangle]
pub unsafe extern "C" fn expand_quattro(
    x3f: *mut x3f_t,
    intensity: libc::c_int,
    expanded: *mut x3f_area16_t,
) -> libc::c_int {
    unsafe {
        expand_quattro_controlled(
            x3f,
            intensity,
            expanded,
            Control::none().with_denoise_backend(crate::DenoiseBackend::Cpu),
        )
    }
    .is_ok() as libc::c_int
}

unsafe fn expand_quattro_controlled(
    x3f: *mut x3f_t,
    intensity: libc::c_int,
    expanded: *mut x3f_area16_t,
    control: Control<'_>,
) -> crate::Result<()> {
    control.check()?;
    let mut image: x3f_area16_t = unsafe { std::mem::zeroed() };
    let mut active: x3f_area16_t = unsafe { std::mem::zeroed() };
    let mut qtop: x3f_area16_t = unsafe { std::mem::zeroed() };
    let mut qtop_crop: x3f_area16_t = unsafe { std::mem::zeroed() };
    let mut active_exp: x3f_area16_t = unsafe { std::mem::zeroed() };

    if unsafe { x3f_image_area_qtop(x3f, &mut qtop) } == 0 {
        return Err(crate::Error::InvalidData("missing Quattro top layer"));
    }
    if unsafe { x3f_image_area(x3f, &mut image) } == 0 {
        return Err(crate::Error::InvalidData("missing Quattro image"));
    }
    if image.columns == 0 || image.rows == 0 {
        return Err(crate::Error::InvalidData("empty Quattro image"));
    }

    if intensity != 0
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
        return Err(crate::Error::InvalidData("invalid Quattro crop"));
    }

    let exp = unsafe { &mut *expanded };
    exp.columns = qtop_crop.columns;
    exp.rows = qtop_crop.rows;
    exp.channels = 3;
    exp.row_stride = exp.columns * exp.channels;
    let bytes = exp.rows as usize * exp.row_stride as usize * std::mem::size_of::<u16>();
    exp.buf = unsafe { libc::malloc(bytes) };
    exp.data = exp.buf as *mut u16;
    if exp.buf.is_null() {
        return Err(crate::Error::Allocation);
    }

    if intensity != 0
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

    let active_ptr: *mut x3f_area16_t = if intensity != 0 {
        &mut active as *mut _
    } else {
        ptr::null_mut()
    };
    let active_exp_ptr: *mut x3f_area16_t = if intensity != 0 {
        &mut active_exp as *mut _
    } else {
        ptr::null_mut()
    };
    let result = unsafe {
        crate::quattro::expand_quattro_controlled(
            (&mut image as *mut x3f_area16_t).cast(),
            active_ptr.cast(),
            (&mut qtop_crop as *mut x3f_area16_t).cast(),
            expanded.cast(),
            active_exp_ptr.cast(),
            denoise_scale(intensity),
            control,
        )
    };
    if result.is_err() {
        unsafe {
            libc::free((*expanded).buf);
            *expanded = std::mem::zeroed();
        }
    }
    result
}

// ----------------------------------------------------------------------
// DNG recovery: immutable source evidence, floating-point evaluation, and
// one final integer encoding. The legacy exported entry points stay stable.
// ----------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DngMapping {
    Linear,
    Shoulder,
}

// Bound outlier-driven exposure and quantization loss without reinstating the
// legacy 1.75 reconstruction cap. Four stops retain at least 12 bits across
// the nominal white range; brighter outliers clip locally during encoding.
const DNG_MAX_HEADROOM: f64 = 16.0;

thread_local! {
    static DNG_HIGHLIGHT_SCALE: std::cell::Cell<f64> = const { std::cell::Cell::new(1.0) };
    static DNG_SHOULDER_CEILING: std::cell::Cell<f64> = const { std::cell::Cell::new(1.0) };
    static DNG_HIGHLIGHT_RECOVERY: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static DNG_HIGHLIGHT_MAPPING: std::cell::Cell<DngMapping> = const { std::cell::Cell::new(DngMapping::Linear) };
    static CINEON: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[no_mangle]
pub unsafe extern "C" fn x3f_get_dng_highlight_scale() -> f64 {
    DNG_HIGHLIGHT_SCALE.with(|c| c.get())
}

#[no_mangle]
pub unsafe extern "C" fn x3f_get_dng_shoulder_ceiling() -> f64 {
    DNG_SHOULDER_CEILING.with(|c| c.get())
}

fn dng_shoulder_knee() -> f64 {
    env_atof("X3F_DNG_SHOULDER_KNEE")
        .filter(|v| v.is_finite())
        .unwrap_or(0.85)
        .clamp(0.5, 0.99)
}

#[no_mangle]
pub unsafe extern "C" fn x3f_get_dng_shoulder_knee() -> f64 {
    dng_shoulder_knee()
}

/// C1 soft compression used only by the explicitly selected shoulder mode.
#[inline]
fn shoulder_compress(lum: f64, knee: f64, shoulder_s: f64) -> f64 {
    let t = (lum - knee) / (1.0 - knee);
    let compressed = 1.0 - (1.0 - (t / shoulder_s).min(1.0)).powf(shoulder_s);
    knee + (1.0 - knee) * compressed
}

#[no_mangle]
pub unsafe extern "C" fn x3f_set_dng_highlight_recovery(enabled: libc::c_int) {
    DNG_HIGHLIGHT_RECOVERY.with(|c| c.set(enabled != 0));
}

/// Additive ABI: 0 selects linear headroom, 1 selects a baked shoulder.
#[no_mangle]
pub unsafe extern "C" fn x3f_set_dng_highlight_mapping(mapping: libc::c_int) {
    DNG_HIGHLIGHT_MAPPING.with(|c| {
        c.set(if mapping == 1 {
            DngMapping::Shoulder
        } else {
            DngMapping::Linear
        })
    });
}

#[no_mangle]
pub unsafe extern "C" fn x3f_set_cineon(enabled: libc::c_int) {
    CINEON.with(|c| c.set(enabled != 0));
}

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
    recovery: bool,
}

// All pointed-to tables are immutable for the lifetime of both row passes.
unsafe impl Send for DngCtx {}
unsafe impl Sync for DngCtx {}

/// Both encoding passes evaluate the same original pixel against the same
/// frozen model. No neighboring raster access, environment reads, or model
/// updates are allowed here, so the second pass can write rows in place.
#[inline]
unsafe fn dng_evaluate_pixel(
    ctx: &DngCtx,
    local: Option<&LocalRecovery>,
    row: usize,
    col: usize,
    pixel: &[u16],
    stats: *mut chroma_lut_apply_stats_t,
) -> [f64; 3] {
    let m = unsafe { &*(ctx.conv_matrix as *const [f64; 9]) };
    let prior = unsafe { &*(ctx.prior as *const [f64; 3]) };
    let original: [f64; 3] = std::array::from_fn(|c| {
        (pixel[c] as f64 - ctx.black[c]) / (ctx.white[c] as f64 - ctx.black[c])
    });
    let sg: [f64; 3] = std::array::from_fn(|c| unsafe {
        x3f_calc_spatial_gain(
            ctx.sgain,
            ctx.sgain_num,
            row as i32,
            col as i32,
            c as i32,
            ctx.rows,
            ctx.cols,
        )
    });
    let mut samples = original;
    let mut confident = false;
    let mut damaged = true;
    let mut mask = [0; 3];
    let local_prior = std::array::from_fn(|c| {
        if sg[c].is_finite() && sg[c] > 0.0 {
            prior[c] / sg[c]
        } else {
            prior[c]
        }
    });
    if ctx.recovery {
        if let Some(model) = local {
            let result = model.recover(row, col, samples, local_prior);
            samples = result.samples;
            confident = result.recovered;
            damaged = result.damaged;
            mask = model.mask(row, col);
        } else {
            // Quattro retains its existing reconstruction. Expanded B/M
            // samples are not independent sensor measurements.
            let applied = if ctx.use_clut {
                unsafe { chroma_lut_apply_pixel_bmt(samples.as_mut_ptr(), ctx.clut, stats) }
            } else {
                0
            };
            if applied == 0 {
                unsafe { reconstruct_highlights(samples.as_mut_ptr(), ctx.prior, ctx.hp) };
            }
        }
        if !confident && damaged && (local.is_none() || mask.contains(&0)) {
            let p = if local.is_some() { &local_prior } else { prior };
            if ctx.use_repair {
                unsafe {
                    repair_pix_apply_pixel(
                        samples.as_mut_ptr(),
                        p.as_ptr(),
                        ctx.repair,
                        ctx.sat_map,
                        row as i32,
                        col as i32,
                        ctx.rows,
                        ctx.cols,
                    );
                }
            }
            let mp = mat3x1_mul_native(m, std::array::from_fn(|c| sg[c] * samples[c]));
            let margin_g = mp[1] - mp[0].max(mp[2]);
            let margin_y = mp[0].min(mp[1]) - mp[2];
            let strength =
                ((margin_g.max(margin_y) - ctx.gate_thr) / ctx.gate_width).clamp(0.0, 1.0);
            if strength > 0.0 {
                // Native recovery cannot let a clipped maximum replace the
                // amplitude of a measured surviving layer.
                let best = mask.into_iter().max().unwrap_or(0);
                let has_survivor = local.is_some() && best > 0;
                let mut amplitude = 0.0_f64;
                for c in 0..3 {
                    if !has_survivor || mask[c] == best {
                        let measured = if has_survivor {
                            original[c]
                        } else {
                            samples[c]
                        };
                        amplitude = amplitude.max(measured / p[c].max(1e-12));
                    }
                }
                for c in 0..3 {
                    samples[c] = (1.0 - strength) * samples[c] + strength * amplitude * p[c];
                }
            }
        }
        if local.is_some() {
            let best = mask.into_iter().max().unwrap_or(0);
            for c in 0..3 {
                if mask[c] == 255 || (best > 0 && mask[c] == best) {
                    samples[c] = original[c];
                } else {
                    // A missing channel cannot be reconstructed below the
                    // observed sample that established its lower bound.
                    samples[c] = samples[c].max(original[c]);
                }
            }
        }
    }
    let output = std::array::from_fn(|c| {
        let value = sg[c] * samples[c];
        if value.is_finite() {
            value
        } else {
            let measured = sg[c] * original[c];
            if measured.is_finite() {
                measured
            } else {
                original[c]
            }
        }
    });
    if ctx.recovery && local.is_some() {
        let measured = std::array::from_fn(|c| sg[c] * original[c]);
        // Match the stabilizer's healthy-color passthrough without evaluating
        // the legacy reference (and spatial gain) a second time. A healthy
        // mask alone is insufficient: matrix-pathological colors still need
        // the safeguard below, including apparently unclipped highlights.
        if mask == [255; 3]
            && mat3x1_mul_native(m, measured)
                .into_iter()
                .all(|value| value.is_finite() && value >= 0.0)
        {
            return measured;
        }
        // A small BMT ratio error can become a large visible hue error. Use
        // the established recovery as a conservative color reference, while
        // the original surviving layers supply intensity and fine detail.
        let mut stable_ctx = *ctx;
        stable_ctx.prior = local_prior.as_ptr();
        let camera_model = local.filter(|model| model.has_camera_maps() && mask.contains(&0));
        let camera_reference = camera_model.and_then(|model| {
            let lut = if ctx.use_clut {
                Some(unsafe { &*ctx.clut })
            } else {
                None
            };
            model.camera_reference(row, col, original, local_prior, lut)
        });
        // If there is no trustworthy survivor, retain the established
        // fallback, but never let its numeric-only LUT treat a camera-flagged
        // layer below the numeric rail as a healthy amplitude donor.
        if camera_model.is_some() {
            stable_ctx.use_clut = false;
        }
        let stable = if let Some(reference) = camera_reference {
            std::array::from_fn(|c| sg[c] * reference[c])
        } else {
            unsafe { dng_evaluate_pixel(&stable_ctx, None, row, col, pixel, ptr::null_mut()) }
        };
        let color_mask = local.map_or(mask, |model| model.color_mask(row, col));
        let stable = crate::highlight_recovery::protect_highlight_color(
            stable,
            measured,
            color_mask,
            *prior,
            m,
            ctx.gate_thr,
            ctx.gate_width,
        );
        crate::highlight_recovery::stabilize_highlight_color(
            output, stable, measured, color_mask, m,
        )
    } else {
        output
    }
}

#[inline]
fn encode_dng_recovered(
    samples: [f64; 3],
    maximum: f64,
    mapping: DngMapping,
    knee: f64,
) -> [u16; 3] {
    let scale = match mapping {
        DngMapping::Linear => 1.0 / maximum,
        DngMapping::Shoulder => {
            let lum = samples.into_iter().fold(0.0_f64, f64::max);
            if maximum > 1.0 && lum > knee {
                shoulder_compress(lum, knee, (maximum - knee) / (1.0 - knee)) / lum
            } else {
                1.0
            }
        }
    };
    samples.map(|v| (v * scale * 65535.0).round().clamp(0.0, 65535.0) as u16)
}

/// Recovery-off encoding deliberately retains the original arithmetic and
/// per-channel intermediate quantization for the byte-parity contract.
unsafe fn dng_encode_off_row(ctx: &DngCtx, row: usize, data: &mut [u16]) {
    for col in 0..ctx.cols as usize {
        let off = col * ctx.channels;
        let values = unsafe {
            dng_evaluate_pixel(ctx, None, row, col, &data[off..off + 3], ptr::null_mut())
        };
        let pixel_max = values[0].max(values[1]).max(values[2]);
        let scale = if pixel_max > 1.0 {
            1.0 / pixel_max
        } else {
            1.0
        };
        for c in 0..3 {
            let range = ctx.white[c] as f64 - ctx.black[c];
            let v = values[c] * scale * range + ctx.black[c];
            data[off + c] = (v.round() as i32).clamp(0, 65535) as u16;
        }
    }
}

unsafe fn dng_active_bounds(x3f: *mut x3f_t, image: &x3f_area16_t) -> [usize; 4] {
    let all = [0, 0, image.rows as usize, image.columns as usize];
    let stride = image.row_stride as usize;
    let channels = image.channels as usize;
    if image.data.is_null() || stride == 0 || channels == 0 {
        return all;
    }
    let mut top: x3f_area16_t = unsafe { std::mem::zeroed() };
    let quattro = unsafe { x3f_image_area_qtop(x3f, &mut top) } != 0;
    let mut source = *image;
    let mut active: x3f_area16_t = unsafe { std::mem::zeroed() };
    if unsafe {
        x3f_crop_area_camf(
            x3f,
            c"ActiveImageArea".as_ptr() as *mut _,
            &mut source,
            (!quattro) as i32,
            &mut active,
        )
    } == 0
    {
        return all;
    }
    // The CAMF wrapper can report success after an unsuccessful inner crop.
    // Validate the returned view before any pointer or stride arithmetic.
    if active.data.is_null()
        || active.row_stride != image.row_stride
        || active.channels != image.channels
    {
        return all;
    }
    let Some(byte_offset) = (active.data as usize).checked_sub(image.data as usize) else {
        return all;
    };
    let sample_size = std::mem::size_of::<u16>();
    if byte_offset % sample_size != 0 {
        return all;
    }
    let offset = byte_offset / sample_size;
    let row = offset / stride;
    let col_offset = offset % stride;
    if col_offset % channels != 0 {
        return all;
    }
    let col = col_offset / channels;
    let Some(bottom) = row.checked_add(active.rows as usize) else {
        return all;
    };
    let Some(right) = col.checked_add(active.columns as usize) else {
        return all;
    };
    if bottom > all[2] || right > all[3] || bottom <= row || right <= col {
        all
    } else {
        [row, col, bottom, right]
    }
}

#[no_mangle]
pub unsafe extern "C" fn apply_highlight_clip_dng(
    x3f: *mut x3f_t,
    image: *mut x3f_area16_t,
    ilevels: *mut x3f_image_levels_t,
    wb: *mut libc::c_char,
) {
    let recovery = DNG_HIGHLIGHT_RECOVERY.with(|c| c.get());
    let mapping = DNG_HIGHLIGHT_MAPPING.with(|c| c.get());
    let mut info = ProcessingInfo {
        highlight_scale: 1.0,
        shoulder_ceiling: 1.0,
    };
    let _ = unsafe {
        apply_highlight_clip_dng_impl(
            x3f,
            image,
            ilevels,
            wb,
            None,
            recovery,
            mapping,
            false,
            Control::none(),
            &mut info,
            None,
            None,
            true,
        )
    };
    DNG_HIGHLIGHT_SCALE.with(|c| c.set(info.highlight_scale));
    DNG_SHOULDER_CEILING.with(|c| c.set(info.shoulder_ceiling));
}

unsafe fn merge_pixel_gain(ctx: &DngCtx, row: usize, col: usize) -> [f64; 3] {
    std::array::from_fn(|c| {
        let gain = unsafe {
            x3f_calc_spatial_gain(
                ctx.sgain,
                ctx.sgain_num,
                row as i32,
                col as i32,
                c as i32,
                ctx.rows,
                ctx.cols,
            )
        };
        if gain.is_finite() && gain > 0.0 && (gain as f32).is_finite() && gain as f32 > 0.0 {
            gain
        } else {
            1.0
        }
    })
}

unsafe fn apply_highlight_clip_dng_impl(
    x3f: *mut x3f_t,
    image: *mut x3f_area16_t,
    ilevels: *mut x3f_image_levels_t,
    wb: *mut libc::c_char,
    provenance: Option<SensorReliability>,
    recovery: bool,
    mapping: DngMapping,
    already_cropped: bool,
    control: Control<'_>,
    info: &mut ProcessingInfo,
    linear: Option<&mut SceneLinearData>,
    mut merge: Option<&mut MergeFrame>,
    apply_sgain: bool,
) -> crate::Result<()> {
    if control.check().is_err() {
        return Ok(());
    }
    if image.is_null() || ilevels.is_null() {
        return Ok(());
    }
    let img = unsafe { &mut *image };
    let il = unsafe { &mut *ilevels };
    if img.data.is_null()
        || img.channels < 3
        || img.rows == 0
        || img.columns == 0
        || (0..3).any(|c| !il.black[c].is_finite() || il.white[c] as f64 <= il.black[c])
    {
        return Ok(());
    }
    let stride = img.row_stride as usize;
    let channels = img.channels as usize;
    let cols = img.columns as usize;
    let rows = img.rows as usize;
    let Some(row_len) = cols.checked_mul(channels) else {
        return Ok(());
    };
    if stride < row_len {
        return Ok(());
    }
    // Cropped views need not own padding after their final visible row.
    let Some(total) = (rows - 1)
        .checked_mul(stride)
        .and_then(|length| length.checked_add(row_len))
        .filter(|length| *length <= isize::MAX as usize / std::mem::size_of::<u16>())
    else {
        return Ok(());
    };
    let mut conv_matrix = [0.0_f64; 9];
    let mut lut_dummy = [0.0_f64; LUTSIZE as usize];
    if unsafe {
        get_conv_controlled(
            x3f,
            x3f_color_encoding_e_SRGB,
            wb,
            LUTSIZE,
            65535,
            lut_dummy.as_mut_ptr(),
            conv_matrix.as_mut_ptr(),
            false,
        )
    } == 0
    {
        return Ok(());
    }

    if let Some(merge) = merge.as_deref_mut() {
        let length = rows
            .checked_mul(cols)
            .and_then(|n| n.checked_mul(3))
            .ok_or(crate::Error::Allocation)?;
        merge
            .bmt
            .try_reserve_exact(length)
            .map_err(|_| crate::Error::Allocation)?;
        merge.bmt.resize(length, 0.0);
    }
    let mut sgain: [x3f_spatial_gain_corr_t; MAX_CORR] = unsafe { std::mem::zeroed() };
    let sgain_num = if apply_sgain {
        unsafe { x3f_get_spatial_gain(x3f, wb, sgain.as_mut_ptr()) }
    } else {
        0
    };
    if let Some(merge) = merge.as_deref_mut() {
        if sgain_num > 0 {
            let mut gain = Vec::new();
            if gain.try_reserve_exact(merge.bmt.len()).is_err() {
                unsafe { x3f_cleanup_spatial_gain(sgain.as_mut_ptr(), sgain_num) };
                return Err(crate::Error::Allocation);
            }
            gain.resize(merge.bmt.len(), 0.0);
            merge.gain = Some(gain);
        }
    }
    let mut hp: highlight_params_t = unsafe { std::mem::zeroed() };
    let mut prior = [0.0; 3];
    unsafe {
        get_highlight_params(x3f, &mut hp);
        compute_chroma_prior(conv_matrix.as_ptr(), prior.as_mut_ptr());
    }
    // Editor preparation is controlled by its own options, not process-wide
    // research overrides. Keep the legacy environment contract unchanged.
    let legacy = linear.is_none() && merge.is_none();
    let no_chroma = legacy && env_present("X3F_NO_CHROMA_LUT");
    let cap = if legacy {
        env_atof("X3F_CHROMA_LUT_CAP").filter(|v| v.is_finite() && *v > 0.0)
    } else {
        None
    };
    let mut local = provenance.and_then(|mask| {
        if mask.rows != rows || mask.cols != cols {
            return None;
        }
        let source = unsafe { std::slice::from_raw_parts(img.data, total) };
        LocalRecovery::build_controlled(
            mask,
            |row, col| {
                let off = row * stride + col * channels;
                std::array::from_fn(|c| {
                    (source[off + c] as f64 - il.black[c]) / (il.white[c] as f64 - il.black[c])
                })
            },
            cap,
            control,
        )
    });
    if let Some(model) = local.as_mut() {
        model.set_chroma_enabled(!no_chroma);
    }

    let mut clut: chroma_lut_t = unsafe { std::mem::zeroed() };
    let mut use_clut = recovery && !no_chroma;
    if use_clut {
        crate::highlight::chroma_lut_defaults(&mut clut, legacy);
        let source_mask = local.as_ref().filter(|model| model.has_camera_maps());
        use_clut = unsafe {
            crate::highlight::chroma_lut_build_from_image_masked(
                &mut clut,
                img,
                il,
                prior.as_ptr(),
                source_mask,
                control,
                legacy,
            )
        } != 0;
    }
    let mut stats: chroma_lut_apply_stats_t = unsafe { std::mem::zeroed() };
    let trace = legacy && env_present("X3F_CHROMA_LUT_TRACE");
    let mut repair: repair_pix_t = unsafe { std::mem::zeroed() };
    let mut sat_map = ptr::null_mut();
    let mut use_repair = legacy && recovery && env_present("X3F_REPAIR_PIX");
    if use_repair {
        unsafe { repair_pix_init_defaults(&mut repair) };
        sat_map = unsafe {
            crate::highlight::build_sat_map_controlled(img, il, repair.sat_threshold, control)
        };
        use_repair = !sat_map.is_null();
        repair.valid = use_repair as i32;
    }
    let gate_thr = legacy
        .then(|| env_atof("X3F_GATE_THR"))
        .flatten()
        .filter(|v| v.is_finite())
        .unwrap_or(0.20);
    let gate_width = legacy
        .then(|| env_atof("X3F_GATE_WIDTH"))
        .flatten()
        .filter(|v| v.is_finite())
        .unwrap_or(0.30)
        .max(1e-6);
    let knee = dng_shoulder_knee();
    let ctx = DngCtx {
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
        rows: rows as i32,
        cols: cols as i32,
        channels,
        recovery,
    };
    let bounds = if already_cropped {
        [0, 0, rows, cols]
    } else {
        unsafe { dng_active_bounds(x3f, img) }
    };
    let data = unsafe { std::slice::from_raw_parts_mut(img.data, total) };
    if trace && recovery {
        for (row, col) in [
            (rows / 20, cols / 2),
            (rows / 3, cols / 2),
            (rows / 2, cols * 9 / 10),
        ] {
            let off = row * stride + col * channels;
            let source: [f64; 3] = std::array::from_fn(|c| {
                (data[off + c] as f64 - ctx.black[c]) / (ctx.white[c] as f64 - ctx.black[c])
            });
            let gain: [f64; 3] = std::array::from_fn(|c| unsafe {
                x3f_calc_spatial_gain(
                    ctx.sgain,
                    ctx.sgain_num,
                    row as i32,
                    col as i32,
                    c as i32,
                    ctx.rows,
                    ctx.cols,
                )
            });
            let native = local.as_ref().map(|model| {
                model.recover(
                    row,
                    col,
                    source,
                    std::array::from_fn(|c| prior[c] / gain[c]),
                )
            });
            let output = unsafe {
                dng_evaluate_pixel(
                    &ctx,
                    local.as_ref(),
                    row,
                    col,
                    &data[off..off + 3],
                    ptr::null_mut(),
                )
            };
            eprintln!("DNG sample ({row},{col}): source={source:?} mask={:?} gain={gain:?} native={native:?} rgb_before={:?} rgb_after={:?}",
                local.as_ref().map(|m| m.mask(row, col)),
                mat3x1_mul_native(&conv_matrix, std::array::from_fn(|c| source[c] * gain[c])),
                mat3x1_mul_native(&conv_matrix, output));
        }
    }
    let mut maximum = 1.0_f64;
    use rayon::prelude::*;
    if let Some(linear) = linear {
        // Reuse sensor reconstruction, but never pass editor data through the
        // DNG's pixel normalization, image-maximum scaling or u16 encoding.
        let length = rows.checked_mul(cols).and_then(|n| n.checked_mul(3));
        if let Some(length) = length.filter(|&n| linear.data.try_reserve_exact(n).is_ok()) {
            linear.data.resize(length, 0.0);
            linear
                .data
                .par_chunks_mut(cols * 3)
                .enumerate()
                .for_each(|(row, out)| {
                    if control.check().is_err() {
                        return;
                    }
                    for (col, pixel) in out.chunks_exact_mut(3).enumerate() {
                        let off = row * stride + col * channels;
                        let samples = unsafe {
                            dng_evaluate_pixel(
                                &ctx,
                                local.as_ref(),
                                row,
                                col,
                                &data[off..off + 3],
                                ptr::null_mut(),
                            )
                        };
                        pixel.copy_from_slice(&linear_rgb(&conv_matrix, samples));
                    }
                });
            linear.width = cols as u32;
            linear.height = rows as u32;
            linear.camera_to_rgb = conv_matrix;
        }
    } else if let Some(merge) = merge {
        merge
            .bmt
            .par_chunks_mut(cols * 3)
            .enumerate()
            .for_each(|(row, out)| {
                if control.check().is_err() {
                    return;
                }
                for (col, pixel) in out.chunks_exact_mut(3).enumerate() {
                    let off = row * stride + col * channels;
                    let gain = unsafe { merge_pixel_gain(&ctx, row, col) };
                    for c in 0..3 {
                        pixel[c] = (((data[off + c] as f64 - ctx.black[c])
                            / (ctx.white[c] as f64 - ctx.black[c]))
                            * gain[c]) as f32;
                    }
                }
            });
        if let Some(gain) = merge.gain.as_mut() {
            gain.par_chunks_mut(cols * 3)
                .enumerate()
                .for_each(|(row, out)| {
                    if control.check().is_err() {
                        return;
                    }
                    for (col, pixel) in out.chunks_exact_mut(3).enumerate() {
                        pixel.copy_from_slice(
                            &unsafe { merge_pixel_gain(&ctx, row, col) }.map(|v| v as f32),
                        );
                    }
                });
        }
        merge.width = cols as u32;
        merge.height = rows as u32;
        merge.camera_to_rgb = conv_matrix;
    } else if recovery {
        let evaluate_row = |row: usize, data: &[u16], stats: *mut chroma_lut_apply_stats_t| {
            let mut maximum = 1.0_f64;
            if control.check().is_err() || row < bounds[0] || row >= bounds[2] {
                return maximum;
            }
            for col in bounds[1]..bounds[3] {
                let off = col * channels;
                let values = unsafe {
                    dng_evaluate_pixel(&ctx, local.as_ref(), row, col, &data[off..off + 3], stats)
                };
                for value in values {
                    if value.is_finite() {
                        maximum = maximum.max(value);
                    }
                }
            }
            maximum
        };
        maximum = if trace {
            data.chunks(stride)
                .enumerate()
                .fold(1.0_f64, |m, (row, data)| {
                    m.max(evaluate_row(row, data, &mut stats))
                })
        } else {
            data.par_chunks(stride)
                .enumerate()
                .map(|(row, data)| evaluate_row(row, data, ptr::null_mut()))
                .reduce(|| 1.0, f64::max)
        };
        maximum = maximum.clamp(1.0, DNG_MAX_HEADROOM);
        data.par_chunks_mut(stride)
            .enumerate()
            .for_each(|(row, data)| {
                if control.check().is_err() {
                    return;
                }
                for col in 0..cols {
                    let off = col * channels;
                    if row < bounds[0] || row >= bounds[2] || col < bounds[1] || col >= bounds[3] {
                        // Masked borders do not participate in recovery or
                        // exposure selection. Encode them explicitly as black
                        // rather than silently clipping unscanned values.
                        data[off..off + channels].fill(0);
                        continue;
                    }
                    let values = unsafe {
                        dng_evaluate_pixel(
                            &ctx,
                            local.as_ref(),
                            row,
                            col,
                            &data[off..off + 3],
                            ptr::null_mut(),
                        )
                    };
                    let encoded = encode_dng_recovered(values, maximum, mapping, knee);
                    data[off..off + 3].copy_from_slice(&encoded);
                }
            });
        il.black = [0.0; 3];
        il.white = [65535; 3];
    } else {
        data.par_chunks_mut(stride)
            .enumerate()
            .for_each(|(row, data)| {
                if control.check().is_err() {
                    return;
                }
                unsafe { dng_encode_off_row(&ctx, row, data) };
            });
    }
    unsafe { x3f_cleanup_spatial_gain(sgain.as_mut_ptr(), sgain_num) };
    if !sat_map.is_null() {
        unsafe { libc::free(sat_map as *mut libc::c_void) };
    }
    if trace {
        if local.is_none() {
            unsafe { chroma_lut_apply_stats_print(&stats, c"dng".as_ptr()) };
        }
        eprintln!(
            "DNG recovery: local={}, mapping={mapping:?}, headroom={maximum:.6}",
            local.is_some()
        );
    }
    info.highlight_scale = if recovery && mapping == DngMapping::Linear {
        maximum
    } else {
        1.0
    };
    info.shoulder_ceiling = if recovery && mapping == DngMapping::Shoulder {
        maximum
    } else {
        1.0
    };
    Ok(())
}

fn crop_reliability(
    mask: SensorReliability,
    source: &x3f_area16_t,
    image: &x3f_area16_t,
) -> Option<SensorReliability> {
    let camera_map_channels = mask.camera_map_channels;
    let mut cropped = crop_reliability_view(mask, source, image)?;
    cropped.camera_map_channels = camera_map_channels;
    Some(cropped)
}

fn crop_reliability_view(
    mask: SensorReliability,
    source: &x3f_area16_t,
    image: &x3f_area16_t,
) -> Option<SensorReliability> {
    if source.data == image.data
        && mask.rows == image.rows as usize
        && mask.cols == image.columns as usize
    {
        return Some(mask);
    }
    let offset = unsafe { image.data.offset_from(source.data) };
    if offset < 0 || source.row_stride == 0 || source.channels == 0 {
        return None;
    }
    let row = offset as usize / source.row_stride as usize;
    let col = (offset as usize % source.row_stride as usize) / source.channels as usize;
    let rows = image.rows as usize;
    let cols = image.columns as usize;
    if row + rows > mask.rows || col + cols > mask.cols {
        return None;
    }
    let mut cropped = SensorReliability::new(rows, cols, mask.noise)?;
    for r in 0..rows {
        cropped.data[r * cols..(r + 1) * cols].copy_from_slice(
            &mask.data[(row + r) * mask.cols + col..(row + r) * mask.cols + col + cols],
        );
    }
    Some(cropped)
}

#[cfg(test)]
mod dng_recovery_encoding_tests {
    use super::{encode_dng_recovered, DngMapping};

    #[test]
    fn linear_encoding_retains_detail_above_legacy_caps() {
        let levels = [1.5, 1.75, 2.0, 4.0, 8.0, 12.0];
        let mut previous = 0;
        for value in levels {
            let pixel = encode_dng_recovered(
                [value, value * 0.5, value * 0.25],
                12.0,
                DngMapping::Linear,
                0.85,
            );
            assert!(pixel[0] > previous);
            for (encoded, expected) in pixel.into_iter().zip([value, value * 0.5, value * 0.25]) {
                assert!(
                    (encoded as f64 * 12.0 / 65535.0 - expected).abs() <= 6.0 / 65535.0 + 1e-12
                );
            }
            previous = pixel[0];
        }
    }

    #[test]
    fn shoulder_preserves_uncompressed_midtones_and_ratios() {
        let low = encode_dng_recovered([0.4, 0.2, 0.1], 4.0, DngMapping::Shoulder, 0.85);
        assert_eq!(low, [26214, 13107, 6554]);
        let high = encode_dng_recovered([4.0, 2.0, 1.0], 4.0, DngMapping::Shoulder, 0.85);
        assert_eq!(high, [65535, 32768, 16384]);
    }
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
    denoise: libc::c_int, // 0..=10 denoise intensity (0 = off); see denoise_scale
    apply_sgain: libc::c_int,
    wb: *mut libc::c_char,
) -> libc::c_int {
    let options = ProcessingOptions {
        color_mode: None,
        spp_denoise: false,
        dng_highlight_recovery: DNG_HIGHLIGHT_RECOVERY.with(|c| c.get()),
        dng_shoulder: DNG_HIGHLIGHT_MAPPING.with(|c| c.get()) == DngMapping::Shoulder,
        cineon: CINEON.with(|c| c.get()),
        rgb_chroma_lut: None,
    };
    match unsafe {
        get_image_controlled(
            x3f,
            image,
            ilevels,
            encoding,
            crop,
            fix_bad,
            denoise,
            apply_sgain,
            wb,
            options,
            Control::none().with_denoise_backend(crate::DenoiseBackend::Cpu),
        )
    } {
        Ok(info) => {
            DNG_HIGHLIGHT_SCALE.with(|c| c.set(info.highlight_scale));
            DNG_SHOULDER_CEILING.with(|c| c.set(info.shoulder_ceiling));
            1
        }
        Err(_) => 0,
    }
}

/// Options passed directly to one image processing operation.
#[derive(Clone, Copy, Default)]
pub struct ProcessingOptions {
    /// Select a camera look for this color conversion. None keeps the legacy path.
    pub color_mode: Option<crate::ColorMode>,
    /// Select the Merrill SPP float pipeline for this color conversion.
    pub spp_denoise: bool,
    pub dng_highlight_recovery: bool,
    pub dng_shoulder: bool,
    pub cineon: bool,
    /// Per-conversion RGB override; None preserves X3F_NO_CHROMA_LUT behavior.
    pub rgb_chroma_lut: Option<bool>,
}

/// Encoding metadata belonging to the returned image, never to a worker thread.
pub struct ProcessingInfo {
    pub highlight_scale: f64,
    pub shoulder_ceiling: f64,
}

/// Owned, active-area, unrotated extended-linear sRGB with as-shot WB and ISO.
#[derive(Default)]
pub struct SceneLinearData {
    pub data: Vec<f32>,
    pub width: u32,
    pub height: u32,
    pub camera_to_rgb: [f64; 9],
}

#[inline]
fn linear_rgb(matrix: &[f64; 9], samples: [f64; 3]) -> [f32; 3] {
    mat3x1_mul_native(matrix, samples).map(|value| value as f32)
}

/// Prepare one freshly decoded reader; the sensor pipeline consumes its data.
/// # Safety
/// `x3f` must uniquely own loaded CAMF, property and decoded RAW sections.
pub unsafe fn get_scene_linear_controlled(
    x3f: *mut x3f_t,
    fix_bad: bool,
    denoise: u8,
    apply_sgain: bool,
    recovery: bool,
    control: Control<'_>,
) -> crate::Result<SceneLinearData> {
    control.check()?;
    if x3f.is_null() {
        return Err(crate::Error::InvalidData("missing processing image"));
    }
    let mut area: x3f_area16_t = unsafe { std::mem::zeroed() };
    let mut levels: x3f_image_levels_t = unsafe { std::mem::zeroed() };
    let mut info = ProcessingInfo {
        highlight_scale: 1.0,
        shoulder_ceiling: 1.0,
    };
    let mut output = SceneLinearData::default();
    let result = unsafe {
        get_image_impl(
            x3f,
            &mut area,
            &mut levels,
            x3f_color_encoding_e_NONE,
            1,
            fix_bad as i32,
            denoise.min(10) as i32,
            apply_sgain as i32,
            ptr::null_mut(),
            ProcessingOptions {
                dng_highlight_recovery: recovery,
                ..ProcessingOptions::default()
            },
            control,
            &mut info,
            Some(&mut output),
            None,
        )
    };
    // As in get_image_controlled, the area owns the processed allocation;
    // the reader no longer owns this buffer after preprocessing.
    unsafe {
        libc::free(area.buf);
    }
    control.check()?;
    result?;
    if output.data.is_empty() || output.data.iter().any(|v| !v.is_finite()) {
        return Err(crate::Error::InvalidData(
            "could not prepare finite scene-linear image",
        ));
    }
    Ok(output)
}

/// Prepare one Quattro exposure without crop, denoise, or highlight recovery.
/// # Safety
/// `x3f` must uniquely own loaded CAMF, property and freshly decoded RAW sections.
pub unsafe fn get_merge_frame_controlled(
    x3f: *mut x3f_t,
    fix_bad: bool,
    apply_sgain: bool,
    control: Control<'_>,
) -> crate::Result<MergeFrame> {
    control.check()?;
    if x3f.is_null() {
        return Err(crate::Error::InvalidData("missing processing image"));
    }
    let mut qtop: x3f_area16_t = unsafe { std::mem::zeroed() };
    if unsafe { x3f_image_area_qtop(x3f, &mut qtop) } == 0 {
        return Err(crate::Error::InvalidData(
            "merge export requires a Quattro sensor",
        ));
    }
    let mut shutter = 0.0;
    if unsafe { x3f_get_camf_float(x3f, c"CaptureShutter".as_ptr() as *mut _, &mut shutter) } == 0
        || !shutter.is_finite()
        || shutter < 1.0
        || shutter > u32::MAX as f64
    {
        return Err(crate::Error::InvalidData("invalid merge exposure time"));
    }
    let mut output = MergeFrame {
        shutter_us: shutter.round() as u32,
        ..MergeFrame::default()
    };
    let mut area: x3f_area16_t = unsafe { std::mem::zeroed() };
    let mut levels: x3f_image_levels_t = unsafe { std::mem::zeroed() };
    let mut info = ProcessingInfo {
        highlight_scale: 1.0,
        shoulder_ceiling: 1.0,
    };
    let result = unsafe {
        get_image_impl(
            x3f,
            &mut area,
            &mut levels,
            x3f_color_encoding_e_NONE,
            0,
            fix_bad as i32,
            0,
            apply_sgain as i32,
            ptr::null_mut(),
            ProcessingOptions::default(),
            control,
            &mut info,
            None,
            Some(&mut output),
        )
    };
    unsafe { libc::free(area.buf) };
    control.check()?;
    result?;
    if output.bmt.is_empty()
        || output.bmt.iter().any(|v| !v.is_finite())
        || output.camera_to_rgb.iter().any(|v| !v.is_finite())
        || (0..3).any(|c| {
            !output.sigma[c].is_finite()
                || output.sigma[c] < 0.0
                || output.raw_black[c] >= output.raw_white[c]
        })
    {
        return Err(crate::Error::InvalidData(
            "could not prepare finite merge frame",
        ));
    }
    Ok(output)
}

/// Process a distinct reader with explicit per-call options and cancellation.
///
/// # Safety
/// Inputs obey the same ownership and valid image contracts as `x3f_get_image`.
pub unsafe fn get_image_controlled(
    x3f: *mut x3f_t,
    image: *mut x3f_area16_t,
    ilevels: *mut x3f_image_levels_t,
    encoding: x3f_color_encoding_t,
    crop: libc::c_int,
    fix_bad: libc::c_int,
    denoise: libc::c_int,
    apply_sgain: libc::c_int,
    wb: *mut libc::c_char,
    options: ProcessingOptions,
    control: Control<'_>,
) -> crate::Result<ProcessingInfo> {
    control.check()?;
    if x3f.is_null() || image.is_null() {
        return Err(crate::Error::InvalidData("missing processing image"));
    }
    let mut info = ProcessingInfo {
        highlight_scale: 1.0,
        shoulder_ceiling: 1.0,
    };
    let result = unsafe {
        get_image_impl(
            x3f,
            image,
            ilevels,
            encoding,
            crop,
            fix_bad,
            denoise,
            apply_sgain,
            wb,
            options,
            control,
            &mut info,
            None,
            None,
        )
    };
    let result = control.check().and(result);
    if result.is_err() {
        unsafe {
            libc::free((*image).buf);
            (*image).buf = ptr::null_mut();
            (*image).data = ptr::null_mut();
        }
    }
    result?;
    Ok(info)
}

unsafe fn get_image_impl(
    x3f: *mut x3f_t,
    image: *mut x3f_area16_t,
    ilevels: *mut x3f_image_levels_t,
    encoding: x3f_color_encoding_t,
    crop: libc::c_int,
    fix_bad: libc::c_int,
    denoise: libc::c_int,
    apply_sgain: libc::c_int,
    mut wb: *mut libc::c_char,
    options: ProcessingOptions,
    control: Control<'_>,
    info: &mut ProcessingInfo,
    linear: Option<&mut SceneLinearData>,
    mut merge: Option<&mut MergeFrame>,
) -> crate::Result<()> {
    let dng_recovery = options.dng_highlight_recovery;
    if options.spp_denoise {
        if control.denoise_options != crate::DenoiseOptions::default() {
            return Err(crate::Error::InvalidData(
                "SPP denoising requires default denoise options",
            ));
        }
        if options.rgb_chroma_lut == Some(true) {
            return Err(crate::Error::InvalidData(
                "SPP processing cannot combine with RGB chroma LUT recovery; select one pipeline",
            ));
        }
        spp::validate_mode(
            encoding,
            options.cineon,
            denoise,
            control.denoise_backend,
            unsafe { is_merrill_model(x3f) },
            &unsafe { merrill_resolution_string(x3f) },
        )?;
    }
    let dng_mapping = if options.dng_shoulder {
        DngMapping::Shoulder
    } else {
        DngMapping::Linear
    };
    let cineon = options.cineon;
    if wb.is_null() {
        wb = unsafe { x3f_get_wb(x3f) };
    }

    let resolved = if let Some(mode) = options.color_mode {
        if cineon
            || !matches!(
                encoding,
                x3f_color_encoding_e_SRGB | x3f_color_encoding_e_ARGB | x3f_color_encoding_e_PPRGB
            )
        {
            return Err(crate::Error::InvalidData(
                "color modes require a rendered color encoding without Cineon",
            ));
        }
        control.check()?;
        Some(unsafe { crate::resolve_color_mode(x3f, mode) }?)
    } else {
        None
    };

    if options.spp_denoise {
        return unsafe {
            spp::render(
                x3f,
                image,
                ilevels,
                encoding,
                crop != 0,
                fix_bad != 0,
                apply_sgain != 0,
                wb,
                control,
                resolved.as_ref(),
            )
        };
    }

    if encoding == x3f_color_encoding_e_QTOP {
        let mut qtop: x3f_area16_t = unsafe { std::mem::zeroed() };
        if unsafe { x3f_image_area_qtop(x3f, &mut qtop) } == 0 {
            return Err(crate::Error::InvalidData("missing Quattro top layer"));
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
        return if ilevels.is_null() {
            Ok(())
        } else {
            Err(crate::Error::InvalidData("unprocessed image has no levels"))
        };
    }

    let mut original_image: x3f_area16_t = unsafe { std::mem::zeroed() };
    if unsafe { x3f_image_area(x3f, &mut original_image) } == 0 {
        return Err(crate::Error::InvalidData("missing processing image"));
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
        return if ilevels.is_null() {
            Ok(())
        } else {
            Err(crate::Error::InvalidData("unprocessed image has no levels"))
        };
    }

    let mut il: x3f_image_levels_t = unsafe { std::mem::zeroed() };
    let mut provenance = None;
    let merrill_rgb = !cineon
        && matches!(
            encoding,
            x3f_color_encoding_e_SRGB | x3f_color_encoding_e_ARGB | x3f_color_encoding_e_PPRGB
        )
        && unsafe { is_merrill_model(x3f) }
        && (resolved.is_some()
            || merrill_render_color_mode(&unsafe { (*x3f).header.color_mode.map(|b| b as u8) })
                .is_some());
    // Validate before render-specific black/gain preprocessing mutates samples.
    let merrill = if merrill_rgb {
        let setup = unsafe {
            make_merrill_render(
                x3f,
                wb,
                encoding,
                apply_sgain != 0,
                control,
                resolved.as_ref(),
            )
        };
        if resolved.is_some() {
            Some(setup?)
        } else {
            merrill_setup_or_legacy(setup)?
        }
    } else {
        None
    };
    let quattro_mode = if let Some(params) = resolved
        .as_ref()
        .filter(|p| p.sensor == crate::ColorModeSensor::Quattro)
    {
        Some(unsafe { make_quattro_color_mode_render(x3f, wb, encoding, params, control) }?)
    } else {
        None
    };
    // Research trace: X3F_FLOAT_FRONT_TRACE prints each ordered float
    // stage of merrill_prepare_float before normalization mutates the
    // raster. Default conversions never run this path, and the trace
    // never changes output pixels.
    if merrill_rgb && std::env::var_os("X3F_FLOAT_FRONT_TRACE").is_some() {
        let mut observe = |stage: &'static str, data: &[f32], detail: String| {
            eprintln!(
                "X3F float front: stage={stage} fnv1a={:016x} {detail}",
                crate::float_front::fnv1a_f32(data)
            );
        };
        match unsafe {
            merrill_prepare_float(x3f, wb, apply_sgain != 0, true, None, control, &mut observe)
        } {
            Ok(prepared) => eprintln!(
                "X3F float front: prepared {}x{}",
                prepared.width, prepared.height
            ),
            Err(error) => eprintln!("X3F float front: error {error:?}"),
        }
    }
    let capture = dng_recovery && encoding == x3f_color_encoding_e_NONE && !cineon;
    if unsafe {
        preprocess_data_impl(
            x3f,
            fix_bad,
            wb,
            &mut il,
            if capture { Some(&mut provenance) } else { None },
            merrill.as_ref().map(|render| render.black),
            merge.as_deref_mut(),
            control,
        )
    }? == 0
    {
        control.check()?;
        return Err(crate::Error::InvalidData("could not preprocess image"));
    }

    let mut qtop: x3f_area16_t = unsafe { std::mem::zeroed() };
    let is_quattro = unsafe { x3f_image_area_qtop(x3f, &mut qtop) } != 0;
    let mut expanded: x3f_area16_t = unsafe { std::mem::zeroed() };
    if is_quattro {
        unsafe { expand_quattro_controlled(x3f, denoise, &mut expanded, control) }?;
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
    } else if denoise != 0 {
        unsafe { run_denoising_controlled(x3f, denoise, control) }?;
    }

    control.check()?;

    // `apply_highlight_clip_dng` is the DNG path's clip-curve renderer.
    // Quattro must also run this pass when recovery is requested; its
    // expanded BMT raster uses the same normalized camera coordinates.
    // Keep its recovery-off path unchanged, including spatial gain.
    // Cineon-log callers using `-color none` need the log curve without
    // the DNG highlight pass.
    let applied_dng = encoding == x3f_color_encoding_e_NONE
        && !cineon
        && (!is_quattro || dng_recovery || linear.is_some() || merge.is_some());
    if applied_dng {
        let current = unsafe { &*image };
        let already_cropped = current.data != original_image.data
            || current.rows != original_image.rows
            || current.columns != original_image.columns;
        let provenance =
            provenance.and_then(|mask| crop_reliability(mask, &original_image, unsafe { &*image }));
        let spatial_gain = (linear.is_none() && merge.is_none()) || apply_sgain != 0;
        unsafe {
            apply_highlight_clip_dng_impl(
                x3f,
                image,
                &mut il,
                wb,
                provenance,
                dng_recovery,
                dng_mapping,
                already_cropped,
                control,
                info,
                linear,
                merge,
                spatial_gain,
            )
        }?;
    }

    if encoding != x3f_color_encoding_e_NONE {
        unsafe {
            convert_data_controlled(
                x3f,
                &mut original_image,
                &mut il,
                encoding,
                apply_sgain,
                wb,
                control,
                cineon,
                options.rgb_chroma_lut,
                merrill,
                resolved.as_ref(),
                quattro_mode,
            )
        }?;
    }

    if !ilevels.is_null() {
        unsafe { *ilevels = il };
    }
    control.check()
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
    unsafe {
        x3f_get_preview_with_scale(
            x3f,
            image,
            ilevels,
            encoding,
            apply_sgain,
            wb,
            max_width,
            preview,
            1.0,
        )
    }
}

/// Rust-only preview extension. Restore the image's linear encoding scale in
/// floating point, before the display transform, without changing the C ABI.
pub unsafe fn x3f_get_preview_with_scale(
    x3f: *mut x3f_t,
    image: *mut x3f_area16_t,
    ilevels: *mut x3f_image_levels_t,
    encoding: x3f_color_encoding_t,
    apply_sgain: libc::c_int,
    wb: *mut libc::c_char,
    max_width: u32,
    preview: *mut x3f_area8_t,
    exposure_scale: f64,
) -> libc::c_int {
    unsafe {
        get_preview_controlled(
            x3f,
            image,
            ilevels,
            encoding,
            apply_sgain,
            wb,
            max_width,
            preview,
            exposure_scale,
            Control::none(),
        )
    }
}

/// Preview rendering with a cancellation token local to its conversion.
/// # Safety
/// Pointer and ownership requirements match `x3f_get_preview_with_scale`.
pub unsafe fn get_preview_controlled(
    x3f: *mut x3f_t,
    image: *mut x3f_area16_t,
    ilevels: *mut x3f_image_levels_t,
    encoding: x3f_color_encoding_t,
    apply_sgain: libc::c_int,
    wb: *mut libc::c_char,
    max_width: u32,
    preview: *mut x3f_area8_t,
    exposure_scale: f64,
    control: Control<'_>,
) -> libc::c_int {
    let max_out: u16 = 255;
    let img = unsafe { &mut *image };
    if control.check().is_err()
        || img.channels < 3
        || max_width == 0
        || img.columns == 0
        || img.rows == 0
    {
        return 0;
    }

    let mut conv_matrix = [0.0_f64; 9];
    let mut lut = [0.0_f64; LUTSIZE as usize];
    if unsafe {
        get_conv_controlled(
            x3f,
            encoding,
            wb,
            LUTSIZE,
            max_out,
            lut.as_mut_ptr(),
            conv_matrix.as_mut_ptr(),
            false,
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
    if alloc.is_null() {
        unsafe {
            x3f_cleanup_spatial_gain(sgain.as_mut_ptr(), sgain_num);
        }
        return 0;
    }
    pv.buf = alloc;
    pv.data = alloc as *mut u8;

    let img_row_stride = img.row_stride as usize;
    let img_channels = img.channels as usize;
    let pv_row_stride = pv.row_stride as usize;
    let pv_channels = pv.channels as usize;
    let il = unsafe { &*ilevels };

    for row in 0..pv.rows as i32 {
        if control.check().is_err() {
            unsafe {
                x3f_cleanup_spatial_gain(sgain.as_mut_ptr(), sgain_num);
                libc::free(pv.buf);
            }
            pv.buf = ptr::null_mut();
            pv.data = ptr::null_mut();
            return 0;
        }
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
                    / (il.white[color] as f64 - il.black[color])
                    * exposure_scale;
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
static _A_RUN_DENOISING: unsafe extern "C" fn(*mut x3f_t, libc::c_int) -> libc::c_int =
    run_denoising;
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

#[cfg(test)]
mod tests {
    #[test]
    fn merge_gain_records_channel_and_pixel_coordinates() {
        use super::*;
        let mut values = [1.0, 2.0, 3.0, 3.0, 4.0, 5.0, 5.0, 6.0, 7.0, 7.0, 8.0, 9.0];
        let mut correction = x3f_spatial_gain_corr_t {
            gain: values.as_mut_ptr(),
            rows: 2,
            cols: 2,
            channels: 3,
            rowpitch: 1,
            colpitch: 1,
            ..unsafe { std::mem::zeroed() }
        };
        let mut ctx: DngCtx = unsafe { std::mem::zeroed() };
        ctx.rows = 2;
        ctx.cols = 2;
        assert_eq!(unsafe { merge_pixel_gain(&ctx, 0, 0) }, [1.0; 3]);
        ctx.sgain = &mut correction;
        ctx.sgain_num = 1;
        assert_eq!(unsafe { merge_pixel_gain(&ctx, 0, 0) }, [1.0, 2.0, 3.0]);
        assert_eq!(unsafe { merge_pixel_gain(&ctx, 1, 1) }, [4.0, 5.0, 6.0]);
        values[..3].copy_from_slice(&[f64::NAN, -1.0, 0.0]);
        assert_eq!(unsafe { merge_pixel_gain(&ctx, 0, 0) }, [1.0; 3]);
        values[..3].copy_from_slice(&[f64::INFINITY, f64::MAX, f64::MIN_POSITIVE]);
        assert_eq!(unsafe { merge_pixel_gain(&ctx, 0, 0) }, [1.0; 3]);
    }

    // This comparison decodes four full-resolution rasters. Run it in release mode.
    #[test]
    #[ignore = "run with --release --include-ignored for the full-raster comparison"]
    fn merge_camera_matrix_matches_uncropped_linear_export() {
        use super::*;
        let corpus = std::env::var_os("X3F_TEST_FILES")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| {
                std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../x3f_test_files")
            });
        let path = corpus.join("_SDI8284.X3F");
        if !path.is_file() {
            eprintln!("skip: _SDI8284.X3F is required for merge export coverage");
            return;
        }
        let path = std::ffi::CString::new(path.as_os_str().as_encoded_bytes()).unwrap();
        let prepare = |apply_sgain, linear: bool| unsafe {
            let file = libc::fopen(path.as_ptr(), c"rb".as_ptr());
            assert!(!file.is_null());
            let x3f = x3f_new_from_file(file.cast());
            assert!(!x3f.is_null());
            for entry in [x3f_get_prop(x3f), x3f_get_camf(x3f), x3f_get_raw(x3f)] {
                if !entry.is_null() {
                    crate::load_data(x3f, entry, Control::none()).unwrap();
                }
            }
            let mut output = SceneLinearData::default();
            let mut merge = MergeFrame::default();
            if linear {
                let mut area = std::mem::zeroed();
                let mut levels = std::mem::zeroed();
                let mut info = ProcessingInfo {
                    highlight_scale: 1.0,
                    shoulder_ceiling: 1.0,
                };
                let result = get_image_impl(
                    x3f,
                    &mut area,
                    &mut levels,
                    x3f_color_encoding_e_NONE,
                    0,
                    1,
                    0,
                    apply_sgain as i32,
                    ptr::null_mut(),
                    ProcessingOptions::default(),
                    Control::none(),
                    &mut info,
                    Some(&mut output),
                    None,
                );
                libc::free(area.buf);
                result.unwrap();
            } else {
                merge =
                    get_merge_frame_controlled(x3f, true, apply_sgain, Control::none()).unwrap();
            }
            x3f_delete(x3f);
            libc::fclose(file);
            (output, merge)
        };
        for apply_sgain in [false, true] {
            let (_, merge) = prepare(apply_sgain, false);
            let (linear, _) = prepare(apply_sgain, true);
            assert_eq!((merge.width, merge.height), (linear.width, linear.height));
            assert_eq!(merge.camera_to_rgb, linear.camera_to_rgb);
            assert_eq!(merge.bmt.len(), linear.data.len());
            let mut max_relative = 0.0_f64;
            let mut max_absolute = 0.0_f64;
            for (pixel, expected) in merge.bmt.chunks_exact(3).zip(linear.data.chunks_exact(3)) {
                let rgb = mat3x1_mul_native(
                    &merge.camera_to_rgb,
                    [pixel[0] as f64, pixel[1] as f64, pixel[2] as f64],
                );
                for c in 0..3 {
                    // The f32 camera export adds rounding before matrix cancellation.
                    let scale = (0..3)
                        .map(|k| (merge.camera_to_rgb[3 * c + k] * pixel[k] as f64).abs())
                        .sum::<f64>()
                        .max(1e-6);
                    let error = (rgb[c] - expected[c] as f64).abs();
                    max_absolute = max_absolute.max(error);
                    max_relative = max_relative.max(error / (expected[c] as f64).abs().max(1e-30));
                    assert!(
                        error
                            <= 1e-4 * (expected[c] as f64).abs()
                                + 4.0 * f32::EPSILON as f64 * scale
                    );
                }
            }
            eprintln!("merge matrix gain={apply_sgain}: max absolute={max_absolute:e}, max relative={max_relative:e}");
            if apply_sgain {
                let gain = merge
                    .gain
                    .as_ref()
                    .expect("fixture has spatial gain tables");
                assert_eq!(gain.len(), merge.bmt.len());
                assert!(gain.iter().all(|v| v.is_finite() && *v > 0.0));
                assert!(gain.iter().any(|v| *v != 1.0));
            } else {
                assert!(merge.gain.is_none());
            }
        }
    }

    #[test]
    fn merge_raw_snapshot_crops_padding_and_unused_channels() {
        use super::*;
        let mut data = vec![1, 2, 99, 3, 4, 99, 88, 88, 5, 6, 99, 7, 8, 99];
        let area = x3f_area16_t {
            data: data.as_mut_ptr(),
            columns: 2,
            rows: 2,
            channels: 3,
            row_stride: 8,
            ..unsafe { std::mem::zeroed() }
        };
        let half = unsafe { snapshot_merge_plane(&area, 2, 2, 2, Control::none()) }.unwrap();
        assert_eq!(half, [1, 2, 3, 4, 5, 6, 7, 8]);
        let top = unsafe { snapshot_merge_plane(&area, 1, 2, 1, Control::none()) }.unwrap();
        assert_eq!(top, [1, 5]);
        assert!(matches!(
            unsafe { snapshot_merge_plane(&area, 3, 2, 1, Control::none()) },
            Err(crate::Error::InvalidData(_))
        ));
        let cancel = std::sync::atomic::AtomicBool::new(true);
        assert!(matches!(
            unsafe { snapshot_merge_plane(&area, 2, 2, 2, Control::new(&cancel)) },
            Err(crate::Error::Cancelled)
        ));
        assert_eq!(data, [1, 2, 99, 3, 4, 99, 88, 88, 5, 6, 99, 7, 8, 99]);
    }

    use super::{intermediate_levels, shoulder_compress, INTERMEDIATE_UNIT};

    #[test]
    fn spp_rejects_custom_denoise_options_before_processing() {
        use super::*;
        let defaults = crate::DenoiseOptions::default();
        for denoise_options in [
            crate::DenoiseOptions {
                algorithm: crate::DenoiseAlgorithm::Bm3d,
                ..defaults
            },
            crate::DenoiseOptions {
                luminance: true,
                ..defaults
            },
            crate::DenoiseOptions {
                chroma: false,
                ..defaults
            },
            crate::DenoiseOptions {
                strength: 2.0,
                ..defaults
            },
            crate::DenoiseOptions {
                nlm: crate::NlmOptions {
                    luma_patch: 5,
                    ..defaults.nlm
                },
                ..defaults
            },
            crate::DenoiseOptions {
                nlm: crate::NlmOptions {
                    chroma_patch: 5,
                    ..defaults.nlm
                },
                ..defaults
            },
            crate::DenoiseOptions {
                nlm: crate::NlmOptions {
                    luma_search: 13,
                    ..defaults.nlm
                },
                ..defaults
            },
            crate::DenoiseOptions {
                nlm: crate::NlmOptions {
                    chroma_search: 13,
                    ..defaults.nlm
                },
                ..defaults
            },
        ] {
            let mut x3f: x3f_t = unsafe { std::mem::zeroed() };
            let mut image: x3f_area16_t = unsafe { std::mem::zeroed() };
            let result = unsafe {
                get_image_controlled(
                    &mut x3f,
                    &mut image,
                    ptr::null_mut(),
                    x3f_color_encoding_e_SRGB,
                    1,
                    1,
                    10,
                    1,
                    ptr::null_mut(),
                    ProcessingOptions {
                        spp_denoise: true,
                        ..ProcessingOptions::default()
                    },
                    Control::none()
                        .with_denoise_options(denoise_options)
                        .unwrap(),
                )
            };
            assert!(matches!(
                result,
                Err(crate::Error::InvalidData(
                    "SPP denoising requires default denoise options"
                ))
            ));
            assert!(image.buf.is_null());
            assert!(image.data.is_null());
        }
    }

    #[test]
    fn merrill_colordq_camf_boundaries_use_native_skip_key_and_table_shapes() {
        use super::*;
        unsafe {
            let mut iso_rows = [1.0_f64; 26];
            iso_rows[0] = 100.0;
            iso_rows[13] = 200.0;
            iso_rows[7] = 4.0;
            iso_rows[20] = 4.0;
            let mut slider_rows = [1.0_f64; 65];
            let mut strengths = [8.0_f64; 3];
            let mut capture = 500.0_f64;
            let mut sensor = 100.0_f64;
            let mut skip = 0_u32;
            let mut entries: [camf_entry_t; 6] = std::mem::zeroed();
            let mut dims: [[camf_dim_entry_t; 2]; 6] = std::mem::zeroed();
            for (i, (name, data, width, height, typ)) in [
                (
                    c"ISONoiseSettings",
                    iso_rows.as_mut_ptr().cast(),
                    13,
                    2,
                    matrix_type_t_M_FLOAT,
                ),
                (
                    c"DenoisingSlidersStep",
                    slider_rows.as_mut_ptr().cast(),
                    13,
                    5,
                    matrix_type_t_M_FLOAT,
                ),
                (
                    c"ColorDQCamRGB",
                    strengths.as_mut_ptr().cast(),
                    3,
                    0,
                    matrix_type_t_M_FLOAT,
                ),
                (
                    c"CaptureISO",
                    (&mut capture as *mut f64).cast(),
                    1,
                    0,
                    matrix_type_t_M_FLOAT,
                ),
                (
                    c"SensorISO",
                    (&mut sensor as *mut f64).cast(),
                    1,
                    0,
                    matrix_type_t_M_FLOAT,
                ),
                (
                    c"SkipStepFlags",
                    (&mut skip as *mut u32).cast(),
                    1,
                    0,
                    matrix_type_t_M_UINT,
                ),
            ]
            .into_iter()
            .enumerate()
            {
                // CAMF stores dimensions outermost-first: [rows, columns].
                dims[i][0].size = if height == 0 { width } else { height };
                dims[i][1].size = if height == 0 { 0 } else { width };
                entries[i].id = u32::from_le_bytes(*b"CMbM");
                entries[i].name_address = name.as_ptr() as *mut _;
                entries[i].matrix_dim = if height == 0 { 1 } else { 2 };
                entries[i].matrix_dim_entry = dims[i].as_mut_ptr();
                entries[i].matrix_elements = width * height.max(1);
                entries[i].matrix_decoded_type = typ;
                entries[i].matrix_decoded = data;
            }
            let mut section: x3f_directory_entry_t = std::mem::zeroed();
            section.header.identifier = u32::from_le_bytes(*b"SECc");
            section.header.data_subsection.camf.entry_table = camf_entry_table_t {
                size: entries.len() as u32,
                element: entries.as_mut_ptr(),
            };
            let mut x3f: x3f_t = std::mem::zeroed();
            x3f.directory_section.num_directory_entries = 1;
            x3f.directory_section.directory_entry = &mut section;
            let input = [-10.0, 100.0, 80.0];
            let expected = crate::merrill::ColorDq::new([32.0; 3])
                .unwrap()
                .apply(input);
            assert_eq!(merrill_color_dq(&mut x3f).unwrap().apply(input), expected);

            *entries[5].matrix_decoded.cast::<u32>() = 1 << 23;
            assert!(matches!(
                merrill_color_dq(&mut x3f),
                Err(crate::Error::InvalidData(
                    "Merrill denoise bypass is not reconstructed"
                ))
            ));
            entries[5].name_address = c"CAMFSkipStepFlags".as_ptr() as *mut _;
            assert_eq!(merrill_color_dq(&mut x3f).unwrap().apply(input), expected);
            entries[5].name_address = c"SkipStepFlags".as_ptr() as *mut _;
            *entries[5].matrix_decoded.cast::<u32>() = 1 << 22;
            assert_eq!(merrill_color_dq(&mut x3f).unwrap().apply(input), expected);

            // Neither disabled spatial gain nor native bit 4 needs aperture
            // or legacy gain metadata; this fixture deliberately has neither.
            assert!(
                merrill_spatial_tables(&mut x3f, true, Some((8, 1)), Control::none())
                    .unwrap()
                    .is_none()
            );
            assert!(matches!(
                merrill_spatial_tables(&mut x3f, true, Some((8, 0)), Control::none()),
                Err(crate::Error::InvalidData(
                    "missing Merrill spatial aperture"
                ))
            ));
            assert!(
                merrill_spatial_tables(&mut x3f, false, None, Control::none())
                    .unwrap()
                    .is_none()
            );
            assert!(matches!(
                merrill_spatial_tables(&mut x3f, true, None, Control::none()),
                Err(crate::Error::InvalidData(
                    "missing Merrill spatial aperture"
                ))
            ));
            *entries[5].matrix_decoded.cast::<u32>() = 1 << 4;
            assert!(
                merrill_spatial_tables(&mut x3f, true, None, Control::none())
                    .unwrap()
                    .is_none()
            );
            assert_eq!(merrill_color_dq(&mut x3f).unwrap().apply(input), expected);
            entries[5].name_address = c"CAMFSkipStepFlags".as_ptr() as *mut _;
            assert!(matches!(
                merrill_spatial_tables(&mut x3f, true, None, Control::none()),
                Err(crate::Error::InvalidData(
                    "missing Merrill spatial aperture"
                ))
            ));
            entries[5].name_address = c"SkipStepFlags".as_ptr() as *mut _;
            *entries[5].matrix_decoded.cast::<u32>() = 1 << 22;

            // A readable table with the wrong row width is unsupported.
            for i in 0..2 {
                dims[i][1].size = 12;
                entries[i].matrix_elements = 12 * dims[i][0].size;
                assert!(merrill_color_dq(&mut x3f).is_err());
                dims[i][1].size = 13;
                entries[i].matrix_elements = 13 * dims[i][0].size;
            }
            (*entries[1].matrix_dim_entry).size = 4;
            entries[1].matrix_elements = 52;
            assert!(merrill_color_dq(&mut x3f).is_err());
            (*entries[1].matrix_dim_entry).size = 5;
            entries[1].matrix_elements = 65;

            // Missing or unreadable ISO matrices use the ordinary native
            // fallback column (4 at every ISO in HI/MED/LOW).
            entries[0].name_address = c"MissingISO".as_ptr() as *mut _;
            assert_eq!(merrill_color_dq(&mut x3f).unwrap().apply(input), expected);
            entries[0].name_address = c"ISONoiseSettings".as_ptr() as *mut _;
            entries[0].matrix_dim = 1;
            entries[0].matrix_elements = 2;
            assert_eq!(merrill_color_dq(&mut x3f).unwrap().apply(input), expected);
            entries[0].matrix_dim = 2;
            entries[0].matrix_elements = 26;

            // Missing or unreadable slider matrices use the native all-one
            // column. This differs from a readable matrix of the wrong shape.
            entries[1].name_address = c"MissingSliders".as_ptr() as *mut _;
            assert_eq!(merrill_color_dq(&mut x3f).unwrap().apply(input), expected);
            entries[1].name_address = c"DenoisingSlidersStep".as_ptr() as *mut _;
            entries[1].matrix_dim = 1;
            entries[1].matrix_elements = 5;
            assert_eq!(merrill_color_dq(&mut x3f).unwrap().apply(input), expected);
            entries[1].matrix_dim = 2;
            (*section
                .header
                .data_subsection
                .camf
                .entry_table
                .element
                .add(1))
            .matrix_elements = 65;

            // Unknown EstimatedISO and slider positions cannot be guessed.
            *entries[0].matrix_decoded.cast::<f64>().add(20) = 8.0;
            assert!(merrill_color_dq(&mut x3f).is_err());
            *entries[0].matrix_decoded.cast::<f64>().add(20) = 4.0;
            *entries[1].matrix_decoded.cast::<f64>().add(20) = 2.0;
            assert!(merrill_color_dq(&mut x3f).is_err());
        }
    }

    #[test]
    fn merrill_prop_aperture_preserves_managed_rounding() {
        use super::merrill_prop_aperture as parse;
        assert_eq!(parse("2.82843"), Some((282843, 100000)));
        assert_eq!(parse("2.800005"), Some((280000, 100000)));
        assert_eq!(parse("2.800015"), Some((280002, 100000)));
        assert_eq!(parse("8"), Some((800000, 100000)));
        assert_eq!(parse(" 28 / 10 "), Some((28, 10)));
        for value in [
            "",
            "NaN",
            "inf",
            "0",
            "-2.8",
            "8/0",
            "2/3/4",
            "4294967295",
            "1.0e30",
        ] {
            assert_eq!(parse(value), None, "{value}");
        }
    }

    #[test]
    fn merrill_output_profiles_use_native_source_branch() {
        use super::*;
        let q12 = |encoding, source| {
            merrill_output_matrix(encoding, source)
                .unwrap()
                .map(|v| v.mul_add(4096.0, 0.5) as i32)
        };
        // Native C3 inverse call and Stage4 right-multiply @0xac608..0xac6b0.
        assert_eq!(
            q12(x3f_color_encoding_e_SRGB, 65535),
            [5727, -1630, 0, 0, 4096, 0, 0, -175, 4271]
        );
        assert_eq!(
            q12(x3f_color_encoding_e_PPRGB, 65535),
            [3032, 464, 601, 564, 3412, 120, 97, 302, 3697]
        );
        assert_eq!(
            q12(x3f_color_encoding_e_ARGB, 65535),
            [4096, 0, 0, 0, 4096, 0, 0, 0, 4096]
        );
        assert_eq!(
            q12(x3f_color_encoding_e_SRGB, 1),
            [4096, 0, 0, 0, 4096, 0, 0, 0, 4096]
        );
        assert_eq!(
            q12(x3f_color_encoding_e_ARGB, 1),
            [2929, 1167, 0, 0, 4096, 0, 0, 169, 3928]
        );
        assert_eq!(
            q12(x3f_color_encoding_e_PPRGB, 1),
            [2168, 1352, 576, 403, 3578, 115, 69, 482, 3546]
        );
        assert_eq!(
            q12(x3f_color_encoding_e_PPRGB, 0),
            q12(x3f_color_encoding_e_PPRGB, 65535)
        );
        assert!(merrill_output_matrix(x3f_color_encoding_e_NONE, 1).is_none());
    }

    #[test]
    fn merrill_render_conditions_counts_before_signed_output_conversion() {
        use crate::merrill::{RenderLuts, ToneParams, ToneSettings};
        let identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let luts = RenderLuts::new(
            ToneParams::from_color_mode_rows(&[], 0),
            ToneSettings {
                pixel_saturation: 4095.0,
                exposure: 0.0,
                shadows: 0.0,
                contrast: 0.0,
                highlights: 0.0,
                alternate: false,
            },
            identity,
            2,
            crate::Control::none(),
        )
        .unwrap();
        let expected = luts.render([0.0, 32766.0, 1000.0]);
        let mut render = super::MerrillRender {
            black: [0.0; 3],
            spatial: None,
            matrix: identity,
            color_dq: crate::merrill::ColorDq::new([0.1; 3]).unwrap(),
            luts,
        };
        assert_eq!(render.render([-20.0, 65536.0, 1000.0]), expected);
        assert_eq!(expected[0], 0);
        assert!(expected[1] > 65500);
        let color_dq = crate::merrill::ColorDq::new([8.0; 3]).unwrap();
        let input = [-20.0, 100.0, 50.0];
        let corrected = color_dq.apply(input).map(|v| v.clamp(0.0, 32766.0));
        let expected = render.luts.render(corrected);
        assert_ne!(render.render(input), expected);
        render.color_dq = color_dq;
        assert_eq!(render.render(input), expected);
    }

    #[test]
    fn quattro_render_bounds_native_counts_before_tone_lookup() {
        use crate::merrill::{RenderLuts, ToneParams};
        let identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let params = crate::ColorModeParameters {
            mode: crate::ColorMode::Standard,
            sensor: crate::ColorModeSensor::Quattro,
            as_shot_fallback: false,
            creative_matrix: None,
            contrast: 0.0,
            tone: ToneParams {
                start: -1.19,
                end: 2.85,
                steepness: 3.0,
                steepness2: 1.55,
                breakpoint: 0.078,
                scale: 1.0,
            },
            inverse_gamma: 1.0 / 2.2222222222222223,
            pixel_saturation: 16383.0,
        };
        let luts =
            RenderLuts::from_color_mode(&params, 0.0, identity, 2, crate::Control::none()).unwrap();
        assert!(
            RenderLuts::from_color_mode(&params, 0.0, identity, 0, crate::Control::none()).is_err()
        );
        let expected = luts.render([0.0, 65535.0, 501.0]);
        assert_ne!(luts.render([-100.0, 70000.0, 500.75]), expected);
        let mut render = super::QuattroColorModeRender {
            matrix: identity,
            offset: [0.0; 3],
            recovery_matrix: identity,
            luts,
        };
        assert_eq!(render.render([-100.0, 70000.0, 500.75]), expected);
        assert_eq!(expected[0], 0);
        assert!(expected[1] > 65500);
        // The sd Quattro pedestal shifts counts before rounding and bounds.
        render.offset = [100.0, -2000.0, 0.5];
        assert_eq!(render.render([-200.0, 70000.0, 500.25]), expected);
    }

    #[test]
    fn sd_quattro_fluorescent_interpolates_native_table() {
        let t = super::SDQ_FLUORESCENT.map(f32::from_bits);
        let wide = [0.98495, 1.0, 1.01282];
        let at = |f: f32| super::fluorescent_gain(&t, f).unwrap();
        // Bounds return the end nodes; lib95 0xa2224 holds the f/1.4 node.
        assert_eq!(at(1.0), [0.98495_f32, 1.0, 1.01282].map(f64::from));
        assert_eq!(at(1.4), at(1.0));
        assert_eq!(at(16.0), [1.0; 3]);
        // Native float32 oracle (sdq-native-evidence oracle/fl.c).
        let oracle = [
            (1.8, [0.989696622, 1.0, 1.00764668]),
            (2.4, [0.994255006, 1.0, 1.00329506]),
            (3.5, [0.997979999, 1.0, 1.0008651]),
            (5.0, [0.999655008, 1.0, 1.00014627]),
            (5.6, [1.0; 3]),
            (7.0, [1.0; 3]),
        ];
        for (f, expected) in oracle {
            assert_eq!(at(f), expected.map(|v: f64| f64::from(v as f32)), "f/{f}");
        }
        assert!(at(2.4)[0] > wide[0] && at(2.4)[2] < wide[2]);
        assert!(super::fluorescent_gain(&t, f32::NAN).is_none());
        let mut unsorted = t;
        unsorted.swap(1, 2);
        assert!(super::fluorescent_gain(&unsorted, 2.4).is_none());
        let mut nonfinite = t;
        nonfinite[20] = f32::INFINITY;
        assert!(super::fluorescent_gain(&nonfinite, 2.4).is_none());
    }

    #[test]
    fn sd_quattro_preprocess_maps_digital_gain_and_raw_offset() {
        let maximum = [16383; 3];
        // _DQH1429: DigitalISOGain [4,4,2], SensorISO 100, RawOffset 256.
        let gain = super::sd_quattro_gain([4.0, 4.0, 2.0], 100.0, 256.0, maximum).unwrap();
        assert_eq!(gain, [4.0, 4.0, 2.0]);
        let black = [256.5, 256.25, 257.0];
        let (span, offset) = super::native_count_span(maximum, black, [256.0; 3], gain);
        assert_eq!(offset, [2.0, 1.0, 2.0]);
        // x = 0 is the measured black; x = 1 is the gained raw maximum.
        let top: [f64; 3] = std::array::from_fn(|c| span[c] + offset[c]);
        assert_eq!(top, [64508.0, 64508.0, 32254.0]);
        // SensorISO 200 halves the counts; SensorISO 50 doubles them past the clamp.
        assert_eq!(
            super::sd_quattro_gain([1.0; 3], 200.0, 2048.0, maximum),
            Some([0.5; 3])
        );
        assert!(super::sd_quattro_gain([4.0, 4.0, 2.0], 50.0, 256.0, maximum).is_none());
        // Exact clamp boundary: 65535 = g * (16383 - 256) is accepted.
        let limit = 65535.0 / 16127.0;
        assert!(super::sd_quattro_gain([limit; 3], 100.0, 256.0, maximum).is_some());
        assert!(
            super::sd_quattro_gain([limit * (1.0 + 1e-12); 3], 100.0, 256.0, maximum).is_none()
        );
        // Two invalid signs must not cancel into a positive gain.
        assert!(super::sd_quattro_gain([-1.0; 3], -100.0, 256.0, maximum).is_none());
        for bad in [0.0, -1.0, f64::NAN] {
            assert!(super::sd_quattro_gain([1.0, bad, 1.0], 100.0, 256.0, maximum).is_none());
            assert!(super::sd_quattro_gain([1.0; 3], bad, 256.0, maximum).is_none());
        }
        assert!(super::sd_quattro_gain([1.0; 3], 100.0, 16383.0, maximum).is_none());
    }

    #[test]
    fn sd_quattro_preprocess_reads_camf_defaults_and_camera_dispatch() {
        use super::*;
        unsafe {
            let mut camera = X3F_CAMERAID_SDQ;
            let mut digital = [2.0_f64, 3.0, 4.0];
            let mut sensor_iso = 0.0_f64;
            let mut entries: [camf_entry_t; 3] = std::mem::zeroed();
            let mut dims: [camf_dim_entry_t; 3] = std::mem::zeroed();
            for (i, (name, data, size, typ)) in [
                (
                    c"CAMERAID",
                    (&mut camera as *mut u32).cast(),
                    1,
                    matrix_type_t_M_UINT,
                ),
                (
                    c"DigitalISOGain",
                    digital.as_mut_ptr().cast(),
                    3,
                    matrix_type_t_M_FLOAT,
                ),
                (
                    c"SensorISO",
                    (&mut sensor_iso as *mut f64).cast(),
                    1,
                    matrix_type_t_M_FLOAT,
                ),
            ]
            .into_iter()
            .enumerate()
            {
                dims[i].size = size;
                entries[i].id = u32::from_le_bytes(*b"CMbM");
                entries[i].name_address = name.as_ptr().cast_mut();
                entries[i].matrix_dim = 1;
                entries[i].matrix_dim_entry = &mut dims[i];
                entries[i].matrix_elements = size;
                entries[i].matrix_decoded_type = typ;
                entries[i].matrix_decoded = data;
            }
            let mut sections: [x3f_directory_entry_t; 2] = std::mem::zeroed();
            sections[0].header.identifier = u32::from_le_bytes(*b"SECc");
            sections[0].header.data_subsection.camf.entry_table = camf_entry_table_t {
                size: 2,
                element: entries.as_mut_ptr(),
            };
            sections[1].header.identifier = u32::from_le_bytes(*b"SECi");
            let mut x3f: x3f_t = std::mem::zeroed();
            x3f.directory_section.num_directory_entries = 2;
            x3f.directory_section.directory_entry = sections.as_mut_ptr();
            for (id, format, offset, gain) in [
                (X3F_CAMERAID_SDQ, 0x0001_0025, 2048.0, [1.0; 3]),
                (X3F_CAMERAID_SDQH, 0x0001_0027, 256.0, digital),
            ] {
                *entries[0].matrix_decoded.cast::<u32>() = id;
                sections[1].header.data_subsection.image_data.type_format = format;
                // RawOffset is absent. SensorISO is absent, then explicitly zero.
                *entries[2].matrix_decoded.cast::<f64>() = 0.0;
                for count in [2, 3] {
                    sections[0].header.data_subsection.camf.entry_table.size = count;
                    assert_eq!(
                        sd_quattro_preprocess(&mut x3f, [16383; 3]).unwrap(),
                        Some((gain, offset)),
                        "CAMERAID {id}, CAMF entries {count}"
                    );
                }
                *entries[2].matrix_decoded.cast::<f64>() = 200.0;
                assert_eq!(
                    sd_quattro_preprocess(&mut x3f, [16383; 3]).unwrap(),
                    Some((gain.map(|g| g / 2.0), offset))
                );
                sections[1].header.data_subsection.image_data.type_format =
                    if id == X3F_CAMERAID_SDQ {
                        0x0001_0027
                    } else {
                        0x0001_0025
                    };
                assert!(sd_quattro_preprocess(&mut x3f, [16383; 3]).is_err());
            }
        }
    }

    #[test]
    fn quattro_highlight_recovery_keeps_sensor_input_independent_of_mode() {
        use super::*;
        let identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let calibration = crate::ColorModeCalibration {
            sensor_to_srgb: identity,
            wb_gain: [1.0; 3],
            normalized_to_counts: [16383.0; 3],
            count_offset: [0.0; 3],
            iso_gain: 1.0,
            output_to_srgb: identity,
            color_dq: None,
        };
        let standard = crate::ColorModeParameters {
            mode: crate::ColorMode::Standard,
            sensor: crate::ColorModeSensor::Quattro,
            as_shot_fallback: false,
            creative_matrix: Some(identity),
            contrast: 0.0,
            tone: crate::merrill::ToneParams {
                start: -1.19,
                end: 2.85,
                steepness: 3.0,
                steepness2: 1.55,
                breakpoint: 0.078,
                scale: 1.0,
            },
            inverse_gamma: 1.0 / 2.2222222222222223,
            pixel_saturation: 16383.0,
        };
        let mut vivid = standard;
        vivid.mode = crate::ColorMode::Vivid;
        vivid.creative_matrix = Some([0.45, 0.0, 0.0, 0.0, 1.3, 0.0, 0.0, 0.0, 0.65]);
        let make = |params: &crate::ColorModeParameters| {
            let luts = crate::ColorModeRenderLuts::from_color_mode(
                params,
                0.0,
                identity,
                2,
                Control::none(),
            )
            .unwrap();
            QuattroColorModeRender::new(&calibration, params, luts).unwrap()
        };
        let renders = [make(&standard), make(&vivid)];
        assert_eq!(
            renders[0].matrix.map(|v| v / standard.pixel_saturation),
            renders[0].recovery_matrix
        );
        assert_eq!(renders[0].recovery_matrix, renders[1].recovery_matrix);
        assert_ne!(renders[0].matrix, renders[1].matrix);
        let legacy = identity.map(|v| v * 2.0);
        assert_eq!(recovery_conv_matrix(legacy, None), legacy);
        let hp = highlight_params_t {
            blending_low: 0.75,
            blending_high: 1.5,
            restore_thresh: 1.75,
            sat_factor: 1.0,
            chan_thresh: 0.5,
        };
        let clut: chroma_lut_t = unsafe { std::mem::zeroed() };
        let repair: repair_pix_t = unsafe { std::mem::zeroed() };
        let lut = [0.0; LUTSIZE as usize];
        let input = [20000_u16, 40000, 65535];
        let mut recovered = Vec::new();
        let mut outputs = Vec::new();
        for render in &renders {
            let recovery_matrix = recovery_conv_matrix(legacy, Some(render));
            assert_eq!(recovery_matrix, renders[0].recovery_matrix);
            let mut prior = [0.0; 3];
            unsafe { compute_chroma_prior(recovery_matrix.as_ptr(), prior.as_mut_ptr()) };
            let inputs = std::sync::Mutex::new(Vec::new());
            let ctx = ConvCtx {
                sgain: ptr::null_mut(),
                sgain_num: 0,
                conv_matrix: recovery_matrix.as_ptr().cast_mut(),
                lut: lut.as_ptr().cast_mut(),
                prior: prior.as_ptr(),
                hp: &hp,
                clut: &clut,
                use_clut: false,
                repair: &repair,
                use_repair: false,
                sat_map: ptr::null(),
                ev_scale: 1.0,
                gate_thr: 0.20,
                gate_width: 0.30,
                black: [0.0; 3],
                white: [65535; 3],
                rows: 1,
                cols: 1,
                channels: 3,
                merrill: None,
                quattro_mode: Some(render),
                recovered_inputs: Some(&inputs),
            };
            let mut row = input;
            unsafe { convert_row(&ctx, 0, &mut row, ptr::null_mut()) }.unwrap();
            recovered.push(inputs.into_inner().unwrap());
            outputs.push(row);
        }
        assert_eq!(recovered[0], recovered[1]);
        assert_eq!(recovered[0], vec![[1.0_f64.to_bits(); 3]]);
        assert_ne!(outputs[0], outputs[1]);
    }

    #[test]
    fn merrill_unsupported_setup_falls_back_but_execution_errors_propagate() {
        assert!(
            super::merrill_setup_or_legacy(Err(crate::Error::InvalidData(
                "unsupported tone extent"
            )))
            .unwrap()
            .is_none()
        );
        assert!(matches!(
            super::merrill_setup_or_legacy(Err(crate::Error::Cancelled)),
            Err(crate::Error::Cancelled)
        ));
        assert!(matches!(
            super::merrill_setup_or_legacy(Err(crate::Error::Gpu("device lost".into()))),
            Err(crate::Error::Gpu(message)) if message == "device lost"
        ));
    }

    #[test]
    fn merrill_missing_or_invalid_exif_selects_legacy_before_rendering() {
        use super::*;
        let mut x3f: x3f_t = unsafe { std::mem::zeroed() };
        let check = |x3f: &mut x3f_t, expected| {
            let result = unsafe { merrill_exif(x3f, Control::none()) };
            assert!(
                matches!(&result, Err(crate::Error::InvalidData(reason)) if *reason == expected)
            );
            let setup = result.map(|_| unreachable!("invalid EXIF must reject native setup"));
            assert!(merrill_setup_or_legacy(setup).unwrap().is_none());
        };
        check(&mut x3f, "missing Merrill JPEG EXIF");

        let mut jpeg = [0xff_u8, 0xd8, 0xff, 0xd9];
        let mut section: x3f_directory_entry_t = unsafe { std::mem::zeroed() };
        section.header.identifier = u32::from_le_bytes(*b"SECi");
        section.header.data_subsection.image_data.type_format = 0x0002_0012;
        section.header.data_subsection.image_data.data = jpeg.as_mut_ptr().cast();
        section.header.data_subsection.image_data.data_size = jpeg.len() as u32;
        x3f.directory_section.num_directory_entries = 1;
        x3f.directory_section.directory_entry = &mut section;
        check(&mut x3f, "invalid Merrill JPEG EXIF");
        unsafe {
            (*x3f.directory_section.directory_entry)
                .header
                .data_subsection
                .image_data
                .data_size = 0;
        }
        check(&mut x3f, "empty Merrill JPEG EXIF");
        assert_eq!(jpeg, [0xff, 0xd8, 0xff, 0xd9]);
    }

    #[test]
    fn merrill_missing_black_calibration_falls_back_before_touching_samples() {
        use super::*;
        let mut pixels = [100, 200, 300];
        let mut raw: x3f_huffman_t = unsafe { std::mem::zeroed() };
        raw.x3rgb16 = x3f_area16_t {
            data: pixels.as_mut_ptr(),
            buf: ptr::null_mut(),
            columns: 1,
            rows: 1,
            channels: 3,
            row_stride: 3,
        };
        let mut section: x3f_directory_entry_t = unsafe { std::mem::zeroed() };
        section.header.identifier = u32::from_le_bytes(*b"SECi");
        section.header.data_subsection.image_data.type_format = 0x0001_001e;
        section.header.data_subsection.image_data.huffman = &mut raw;
        let mut x3f: x3f_t = unsafe { std::mem::zeroed() };
        x3f.directory_section.num_directory_entries = 1;
        x3f.directory_section.directory_entry = &mut section;
        let setup = unsafe {
            make_merrill_render(
                &mut x3f,
                c"Sunlight".as_ptr() as *mut _,
                x3f_color_encoding_e_SRGB,
                true,
                Control::none(),
                None,
            )
        };
        assert!(matches!(
            &setup,
            Err(crate::Error::InvalidData(
                "missing Merrill black calibration"
            ))
        ));
        assert!(merrill_setup_or_legacy(setup).unwrap().is_none());
        assert_eq!(pixels, [100, 200, 300]);
    }

    #[test]
    fn merrill_temperature_validation_resets_the_whole_vector() {
        use super::merrill_temperature_gain as validate;
        assert_eq!(validate([0.5, 1.0, 1.5]), [0.5, 1.0, 1.5]);
        for invalid in [0.0, 0.49, 1.51, f64::NAN, f64::INFINITY] {
            for c in 0..3 {
                let mut gains = [0.8, 1.2, 1.4];
                gains[c] = invalid;
                assert_eq!(validate(gains), [1.0; 3]);
            }
        }
    }

    #[test]
    fn merrill_black_trims_signed_tails_and_includes_camf_last_pixel() {
        let mut pixels = vec![0u16; 34 * 10];
        for row in 0..10 {
            for col in 0..10 {
                pixels[row * 34 + col * 3..row * 34 + col * 3 + 3].copy_from_slice(&[10, 20, 30]);
            }
        }
        pixels[..3].fill(i16::MIN as u16);
        pixels[9 * 34 + 9 * 3..9 * 34 + 9 * 3 + 3].fill(i16::MAX as u16);
        let mut image = crate::x3f_area16_t {
            data: pixels.as_mut_ptr(),
            buf: std::ptr::null_mut(),
            rows: 10,
            columns: 10,
            channels: 3,
            row_stride: 34,
        };
        let mut cropped = unsafe { std::mem::zeroed() };
        assert_ne!(
            unsafe { crate::x3f_crop_area([0, 0, 9, 9].as_mut_ptr(), &mut image, &mut cropped) },
            0
        );
        assert_eq!(
            unsafe { super::merrill_trimmed_black(&cropped, crate::Control::none()) }.unwrap(),
            [10.0, 20.0, 30.0]
        );
        let cancel = std::sync::atomic::AtomicBool::new(true);
        assert!(matches!(
            unsafe { super::merrill_trimmed_black(&cropped, crate::Control::new(&cancel)) },
            Err(crate::Error::Cancelled)
        ));
    }

    #[test]
    fn merrill_render_iso_uses_native_float_remaps_and_zero_fallbacks() {
        use super::merrill_render_iso_gain as iso;
        for (capture, effective) in [
            (800.0, 666.0),
            (1600.0, 1200.0),
            (3200.0, 2133.0),
            (6400.0, 4266.0),
        ] {
            assert_eq!(
                iso(capture, 100.0, 0, 1),
                Some((effective / 100.0_f32) as f64)
            );
            assert_eq!(
                iso(capture, 100.0, 0, 0),
                Some((capture as f32 / 100.0) as f64)
            );
        }
        assert_eq!(iso(800.5, 100.0, 0, 1), Some((800.5_f32 / 100.0) as f64));
        assert_eq!(iso(0.0, 0.0, 800, 1), Some((666.0_f32 / 100.0) as f64));
        assert_eq!(
            iso(f64::MIN_POSITIVE, f64::MIN_POSITIVE, 800, 1),
            Some((666.0_f32 / 100.0) as f64)
        );
        for invalid in [f64::NAN, f64::INFINITY, -1.0, f64::MAX] {
            assert_eq!(iso(invalid, 100.0, 100, 1), None);
            assert_eq!(iso(100.0, invalid, 100, 1), None);
        }
        assert_eq!(iso(0.0, 100.0, 0, 0), None);
    }

    #[test]
    fn merrill_render_matrix_preserves_native_mode_and_wb_order() {
        use super::*;
        let cc = [1.0, 2.0, 0.0, 0.0, 1.0, 3.0, 4.0, 0.0, 1.0];
        let mode = [1.0, 0.0, 2.0, 3.0, 1.0, 0.0, 0.0, 4.0, 1.0];
        let result = merrill_render_matrix(
            x3f_color_encoding_e_SRGB,
            cc,
            Some(mode),
            [2.0, 3.0, 5.0],
            2.0,
        );
        // CC*mode is [[7,2,2],[3,13,3],[4,4,9]], followed by column WB gain.
        assert_eq!(
            result,
            Some([28.0, 12.0, 20.0, 12.0, 78.0, 30.0, 16.0, 24.0, 90.0])
        );
        assert_eq!(
            merrill_render_matrix(x3f_color_encoding_e_SRGB, cc, Some([0.0; 9]), [1.0; 3], 1.0),
            Some(cc)
        );
    }

    #[test]
    fn merrill_render_profile_constants_and_as_shot_mode_policy() {
        use super::*;
        let identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        assert_eq!(
            merrill_render_matrix(x3f_color_encoding_e_SRGB, identity, None, [1.0; 3], 1.0),
            Some(identity)
        );
        let adobe = merrill_render_matrix(x3f_color_encoding_e_ARGB, identity, None, [1.0; 3], 1.0)
            .unwrap();
        let prophoto =
            merrill_render_matrix(x3f_color_encoding_e_PPRGB, identity, None, [1.0; 3], 1.0)
                .unwrap();
        // Native matrix columns: response to each input primary, including small negative terms.
        assert_eq!(
            [adobe[0], adobe[3], adobe[6]],
            [0.715162633, -0.0000408, -0.0000616]
        );
        assert_eq!(
            [prophoto[2], prophoto[5], prophoto[8]],
            [0.140661, 0.028098, 0.865681]
        );
        assert_eq!(merrill_render_color_mode(b"Neutral"), Some(c"Neutral"));
        assert_eq!(
            merrill_render_color_mode(b"Neutral\0padding"),
            Some(c"Neutral")
        );
        assert_eq!(merrill_render_color_mode(b""), Some(c"Standard"));
        assert_eq!(merrill_render_color_mode(b"unknown"), Some(c"Standard"));
        assert_eq!(merrill_render_color_mode(&[b'x'; 32]), Some(c"Standard"));
        assert_eq!(merrill_render_color_mode(b"Monochrome"), None);
        assert_eq!(merrill_render_color_mode(b"Sepia"), None);
    }

    #[test]
    fn merrill_render_reads_camf_mode_and_overwrites_legacy_zero_iso_matrix() {
        use super::*;
        unsafe {
            let mut cc = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
            let mut gain = [2.0, 3.0, 5.0];
            let mut mode = [1.0, 0.0, 2.0, 3.0, 1.0, 0.0, 0.0, 4.0, 1.0];
            let mut capture = 800.0_f64;
            let mut sensor = 0.0_f64;
            let mut iso_range = 1_u32;
            let mut entries: [camf_entry_t; 14] = std::mem::zeroed();
            let mut dims: [[camf_dim_entry_t; 2]; 10] = std::mem::zeroed();
            let mut keys = [
                c"Sunlight".as_ptr() as *mut _,
                c"Sunlight".as_ptr() as *mut _,
                c"Vivid".as_ptr() as *mut _,
            ];
            let mut values = [
                c"CC".as_ptr() as *mut u8,
                c"WB".as_ptr() as *mut u8,
                c"Mode".as_ptr() as *mut u8,
            ];
            for (i, name) in [
                c"WhiteBalanceColorCorrections",
                c"WhiteBalanceGains",
                c"ColorModeCompensations",
            ]
            .into_iter()
            .enumerate()
            {
                entries[i].id = u32::from_le_bytes(*b"CMbP");
                entries[i].name_address = name.as_ptr() as *mut _;
                entries[i].property_num = 1;
                entries[i].property_name = &mut keys[i];
                entries[i].property_value = &mut values[i];
            }
            for (i, (name, data, width, height, typ)) in [
                (c"CC", cc.as_mut_ptr().cast(), 3, 3, matrix_type_t_M_FLOAT),
                (c"WB", gain.as_mut_ptr().cast(), 3, 0, matrix_type_t_M_FLOAT),
                (
                    c"Mode",
                    mode.as_mut_ptr().cast(),
                    3,
                    3,
                    matrix_type_t_M_FLOAT,
                ),
                (
                    c"CaptureISO",
                    (&mut capture as *mut f64).cast(),
                    1,
                    0,
                    matrix_type_t_M_FLOAT,
                ),
                (
                    c"SensorISO",
                    (&mut sensor as *mut f64).cast(),
                    1,
                    0,
                    matrix_type_t_M_FLOAT,
                ),
                (
                    c"ISOrange",
                    (&mut iso_range as *mut u32).cast(),
                    1,
                    0,
                    matrix_type_t_M_UINT,
                ),
            ]
            .into_iter()
            .enumerate()
            {
                dims[i][0].size = width;
                dims[i][1].size = height;
                entries[i + 3].id = u32::from_le_bytes(*b"CMbM");
                entries[i + 3].name_address = name.as_ptr() as *mut _;
                entries[i + 3].matrix_dim = if height == 0 { 1 } else { 2 };
                entries[i + 3].matrix_dim_entry = dims[i].as_mut_ptr();
                entries[i + 3].matrix_elements = width * height.max(1);
                entries[i + 3].matrix_decoded_type = typ;
                entries[i + 3].matrix_decoded = data;
            }
            let mut contrasts = [
                c"Vivid".as_ptr().cast_mut(),
                c"Standard".as_ptr().cast_mut(),
            ];
            let mut contrast_values = [
                c"CMCC_Vivid".as_ptr() as *mut u8,
                c"Missing".as_ptr() as *mut u8,
            ];
            entries[9].id = u32::from_le_bytes(*b"CMbP");
            entries[9].name_address = c"ColorModeContrastCompensations".as_ptr().cast_mut();
            entries[9].property_num = 2;
            entries[9].property_name = contrasts.as_mut_ptr();
            entries[9].property_value = contrast_values.as_mut_ptr();
            let mut tone_rows = [
                [1.0, -1.1, 2.5, 3.0, 0.45, 0.07, 1.4],
                [1.0, -0.8, 2.9, 2.7, 0.45, 0.10, 2.0],
            ];
            let mut contrast = 0.25_f64;
            let mut color_dq = [8.0_f64; 3];
            let mut camsys = [2_u32, 0, 0];
            for (i, (name, data, width, height)) in [
                (c"TCColorModeSettings", tone_rows.as_mut_ptr().cast(), 2, 7),
                (c"CMCC_Vivid", (&mut contrast as *mut f64).cast(), 1, 0),
                (c"ColorDQCamRGB", color_dq.as_mut_ptr().cast(), 3, 0),
                (c"CamsysVersion", camsys.as_mut_ptr().cast(), 3, 0),
            ]
            .into_iter()
            .enumerate()
            {
                dims[i + 6][0].size = width;
                dims[i + 6][1].size = height;
                entries[i + 10].id = u32::from_le_bytes(*b"CMbM");
                entries[i + 10].name_address = name.as_ptr().cast_mut();
                entries[i + 10].matrix_dim = if height == 0 { 1 } else { 2 };
                entries[i + 10].matrix_dim_entry = dims[i + 6].as_mut_ptr();
                entries[i + 10].matrix_elements = width * height.max(1);
                entries[i + 10].matrix_decoded_type = matrix_type_t_M_FLOAT;
                entries[i + 10].matrix_decoded = data;
            }
            entries[13].matrix_decoded_type = matrix_type_t_M_UINT;
            let mut model: x3f_property_t = std::mem::zeroed();
            model.name_utf8 = c"CAMMODEL".as_ptr() as *mut _;
            model.value_utf8 = c"SIGMA DP2 Merrill".as_ptr() as *mut _;
            let mut sections: [x3f_directory_entry_t; 3] = std::mem::zeroed();
            sections[0].header.identifier = u32::from_le_bytes(*b"SECc");
            sections[0].header.data_subsection.camf.entry_table = camf_entry_table_t {
                size: entries.len() as u32,
                element: entries.as_mut_ptr(),
            };
            sections[1].header.identifier = u32::from_le_bytes(*b"SECp");
            sections[1]
                .header
                .data_subsection
                .property_list
                .property_table = x3f_property_table_t {
                size: 1,
                element: &mut model,
            };
            let mut x3f: x3f_t = std::mem::zeroed();
            x3f.directory_section.num_directory_entries = 2;
            x3f.directory_section.directory_entry = sections.as_mut_ptr();
            let model = sections[1]
                .header
                .data_subsection
                .property_list
                .property_table
                .element;
            for version in [0x0002_0003, 0x0003_0000] {
                x3f.header.version = version;
                for name in [
                    c"SIGMA DP1 Merrill",
                    c"SIGMA DP2 Merrill",
                    c"SIGMA DP3 Merrill",
                ] {
                    (*model).value_utf8 = name.as_ptr() as *mut _;
                    assert_eq!(merrill_count_scale(&mut x3f), Some(2.0));
                }
                (*model).value_utf8 = c"SIGMA SD1 Merrill".as_ptr() as *mut _;
                assert_eq!(merrill_count_scale(&mut x3f), Some(1.0 / 0.55));
                (*model).value_utf8 = c"Unknown".as_ptr() as *mut _;
                assert_eq!(merrill_count_scale(&mut x3f), None);
            }
            (*model).value_utf8 = c"SIGMA SD1 Merrill".as_ptr() as *mut _;
            for unsupported in [0x0002_0002, 0x0003_0001, 0x0004_0000] {
                x3f.header.version = unsupported;
                assert_eq!(merrill_count_scale(&mut x3f), None);
            }
            x3f.header.version = 0x0003_0000;
            (*model).value_utf8 = c"SIGMA DP2 Merrill".as_ptr() as *mut _;
            for (dst, src) in x3f.header.color_mode.iter_mut().zip(b"Vivid\0") {
                *dst = *src as libc::c_char;
            }
            for (dst, src) in x3f.header.white_balance.iter_mut().zip(b"Sunlight\0") {
                *dst = *src as libc::c_char;
            }
            let recorded_header = x3f.header.color_mode;
            let selected = crate::resolve_color_mode(&mut x3f, crate::ColorMode::Vivid).unwrap();
            assert_eq!(selected.creative_matrix, Some(mode));
            assert_eq!(selected.contrast, 0.25);
            assert_eq!(selected.tone.start, f64::from(tone_rows[0][1] as f32));
            assert_eq!(
                crate::resolve_color_mode(&mut x3f, crate::ColorMode::AsShot)
                    .unwrap()
                    .mode,
                crate::ColorMode::Vivid
            );
            x3f.header.color_mode.fill(0);
            for (dst, src) in x3f.header.color_mode.iter_mut().zip(b"Monochrome") {
                *dst = *src as libc::c_char;
            }
            assert!(crate::resolve_color_mode(&mut x3f, crate::ColorMode::AsShot).is_err());
            assert_eq!(
                crate::resolve_color_mode(&mut x3f, crate::ColorMode::Vivid)
                    .unwrap()
                    .mode,
                crate::ColorMode::Vivid
            );
            x3f.header.color_mode = recorded_header;
            contrasts.swap(0, 1);
            contrast_values.swap(0, 1);
            let reordered = crate::resolve_color_mode(&mut x3f, crate::ColorMode::Vivid).unwrap();
            assert_eq!(reordered.creative_matrix, selected.creative_matrix);
            assert_eq!(reordered.contrast, selected.contrast);
            assert_eq!(reordered.tone.start, f64::from(tone_rows[1][1] as f32));
            assert_eq!(x3f.header.color_mode, recorded_header);
            let standard = crate::resolve_color_mode(&mut x3f, crate::ColorMode::Standard).unwrap();
            assert_eq!(standard.creative_matrix, None);
            assert_eq!(standard.contrast, 0.0);
            let fcblue = crate::resolve_color_mode(&mut x3f, crate::ColorMode::FCBlue).unwrap();
            assert_eq!(
                fcblue.creative_matrix.unwrap().map(f64::to_bits),
                [
                    0x3ff8d42c3c9eecc0,
                    0x3fd46edaa92e6213,
                    0xbfebdfce3150dae4,
                    0x3fd69d2bf5515055,
                    0x3ff6160956c0d6f5,
                    0xbfe77aa8a82a5615,
                    0x3fc08ad688377186,
                    0x3fb73b13d244dbd2,
                    0x3fe8f5e84f09528f,
                ]
            );
            assert_eq!(fcblue.contrast, 0.3);
            (*model).value_utf8 = c"SIGMA SD1 Merrill".as_ptr().cast_mut();
            assert_eq!(
                crate::resolve_color_mode(&mut x3f, crate::ColorMode::FCBlue)
                    .unwrap()
                    .creative_matrix
                    .unwrap()
                    .map(f64::to_bits),
                [
                    0x3ff9b97785729b28,
                    0x3fd2a44417870542,
                    0xbfecc5197a24894c,
                    0x3fd8915bdd76683c,
                    0x3ff5d78811b1d92b,
                    0xbfe7f7b9e060fe48,
                    0x3fc2fbe33acd5b68,
                    0x3fb4ca0c282c6ef4,
                    0x3fe8a7c7c5260f5e,
                ]
            );
            (*model).value_utf8 = c"SIGMA DP2 Merrill".as_ptr().cast_mut();
            dims[7][0].size = 2;
            assert_eq!(
                crate::resolve_color_mode(&mut x3f, crate::ColorMode::Vivid)
                    .unwrap()
                    .contrast,
                0.0
            );
            dims[7][0].size = 1;
            assert_eq!(dims[7][0].size, 1);
            contrast = f64::NAN;
            assert!(contrast.is_nan());
            assert!(crate::resolve_color_mode(&mut x3f, crate::ColorMode::Vivid).is_err());
            contrast = 0.25;
            assert_eq!(contrast, selected.contrast);
            mode[8] = 0.0;
            mode[2] = 0.0;
            assert!(crate::resolve_color_mode(&mut x3f, crate::ColorMode::Vivid).is_err());
            assert!(!crate::available_color_modes(&mut x3f)
                .unwrap()
                .contains(&crate::ColorMode::Vivid));
            mode[8] = 1.0;
            mode[2] = 2.0;
            assert_eq!(
                crate::resolve_color_mode(&mut x3f, crate::ColorMode::Vivid)
                    .unwrap()
                    .creative_matrix,
                Some(mode)
            );
            color_dq[0] = f64::NAN;
            assert!(color_dq[0].is_nan());
            assert!(crate::resolve_color_mode(&mut x3f, crate::ColorMode::Vivid).is_err());
            color_dq[0] = 8.0;
            assert_eq!(color_dq[0], 8.0);
            // Quattro uses the compiled Teal matrix only with a non-empty list.
            sections[2].header.identifier = u32::from_le_bytes(*b"SECi");
            sections[2].header.data_subsection.image_data.type_format = 0x0001_0023;
            assert_eq!(
                sections[2].header.data_subsection.image_data.type_format,
                0x0001_0023
            );
            x3f.directory_section.num_directory_entries = 3;
            x3f.header.version = 0x0004_0000;
            let teal =
                crate::resolve_color_mode(&mut x3f, crate::ColorMode::TealAndOrange).unwrap();
            let expected_teal = [
                1.06673, 0.05757, -0.12415, 0.25086, 0.38293, 0.36634, -0.01669, -0.04191, 1.05873,
            ];
            assert_eq!(teal.creative_matrix, Some(expected_teal));
            *entries[2].property_name = c"TealAndOrange".as_ptr().cast_mut();
            assert_eq!(
                crate::resolve_color_mode(&mut x3f, crate::ColorMode::TealAndOrange)
                    .unwrap()
                    .creative_matrix,
                Some(expected_teal)
            );
            assert_ne!(mode, expected_teal);
            entries[2].property_num = 0;
            assert!(crate::resolve_color_mode(&mut x3f, crate::ColorMode::TealAndOrange).is_err());
            entries[2].name_address = c"AbsentCompensations".as_ptr().cast_mut();
            assert!(crate::resolve_color_mode(&mut x3f, crate::ColorMode::TealAndOrange).is_err());
            entries[2].name_address = c"ColorModeCompensations".as_ptr().cast_mut();
            entries[2].property_num = 1;
            *entries[2].property_name = c"Vivid".as_ptr().cast_mut();
            x3f.directory_section.num_directory_entries = 2;
            x3f.header.version = 0x0003_0000;
            let mut lut = [0.0; 16];
            let mut matrix = [0.0; 9];
            assert_eq!(
                get_render_conv_controlled(
                    &mut x3f,
                    x3f_color_encoding_e_SRGB,
                    c"Sunlight".as_ptr() as *mut _,
                    16,
                    65535,
                    lut.as_mut_ptr(),
                    matrix.as_mut_ptr(),
                    false,
                    None,
                ),
                1
            );
            let expected = merrill_render_matrix(
                x3f_color_encoding_e_SRGB,
                cc,
                Some(mode),
                gain,
                (666.0_f32 / 100.0) as f64,
            )
            .unwrap();
            assert_eq!(matrix, expected);
            assert!(matrix.iter().all(|v| v.is_finite()));
            assert_eq!(lut[0], 0.0);
            assert!((lut[15] - 65535.0).abs() < 1e-6);

            // The header/API may say Sunlight while both CAMF lists use Daylight.
            *entries[0].property_name = c"Daylight".as_ptr() as *mut _;
            *entries[1].property_name = c"Daylight".as_ptr() as *mut _;
            assert_eq!(
                merrill_wb_gain(&mut x3f, c"Sunlight".as_ptr() as *mut _),
                Some(gain)
            );
            assert_eq!(
                get_render_conv_controlled(
                    &mut x3f,
                    x3f_color_encoding_e_SRGB,
                    c"Sunlight".as_ptr() as *mut _,
                    16,
                    65535,
                    lut.as_mut_ptr(),
                    matrix.as_mut_ptr(),
                    false,
                    None,
                ),
                1
            );
            assert_eq!(matrix, expected);

            // Invalid native calibration must fail instead of retaining a NaN/Inf fallback.
            sensor = f64::NAN;
            assert!(sensor.is_nan());
            assert_eq!(
                get_render_conv_controlled(
                    &mut x3f,
                    x3f_color_encoding_e_SRGB,
                    c"Sunlight".as_ptr() as *mut _,
                    16,
                    65535,
                    lut.as_mut_ptr(),
                    matrix.as_mut_ptr(),
                    false,
                    None,
                ),
                0
            );
        }
    }

    #[test]
    fn scene_linear_handoff_retains_negative_channels_and_headroom() {
        use super::*;
        let mut matrix = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let hp = unsafe { std::mem::zeroed::<highlight_params_t>() };
        let clut = unsafe { std::mem::zeroed::<chroma_lut_t>() };
        let repair = unsafe { std::mem::zeroed::<repair_pix_t>() };
        let prior = [1.0; 3];
        let ctx = DngCtx {
            sgain: ptr::null_mut(),
            sgain_num: 0,
            conv_matrix: matrix.as_mut_ptr(),
            prior: prior.as_ptr(),
            hp: &hp,
            clut: &clut,
            use_clut: false,
            repair: &repair,
            use_repair: false,
            sat_map: ptr::null(),
            gate_thr: 0.2,
            gate_width: 0.3,
            black: [100.0; 3],
            white: [1100; 3],
            rows: 1,
            cols: 1,
            channels: 3,
            recovery: false,
        };
        let source = [2100, 600, 0];
        let sensor = unsafe { dng_evaluate_pixel(&ctx, None, 0, 0, &source, ptr::null_mut()) };
        assert_eq!(linear_rgb(&matrix, sensor), [2.0, 0.5, -0.1]);
        let exposure_down = linear_rgb(&matrix, sensor).map(|v| v * 0.25);
        assert_eq!(exposure_down, [0.5, 0.125, -0.025]);
        // The existing DNG path deliberately has a different bounded encoding.
        let mut encoded = source;
        unsafe { dng_encode_off_row(&ctx, 0, &mut encoded) };
        assert_eq!(encoded, [1100, 350, 50]);
    }

    #[test]
    fn highlight_pixel_patterns_reject_invalid_coordinates_and_strides() {
        use crate::*;

        let render = |mut pattern: [u32; 4]| {
            let mut pixels = [10u16; 3 * 3 * 3];
            pixels[12..15].fill(1000);
            unsafe {
                let mut dimensions: [camf_dim_entry_t; 2] = std::mem::zeroed();
                dimensions[0].size = 2;
                dimensions[1].size = 2;
                let mut entry: camf_entry_t = std::mem::zeroed();
                entry.id = u32::from_le_bytes(*b"CMbM");
                entry.name_address = c"HighlightPixelsInfo".as_ptr() as *mut _;
                entry.matrix_dim = 2;
                entry.matrix_dim_entry = dimensions.as_mut_ptr();
                entry.matrix_elements = 4;
                entry.matrix_decoded_type = matrix_type_t_M_UINT;
                entry.matrix_decoded = pattern.as_mut_ptr().cast();
                let mut section: x3f_directory_entry_t = std::mem::zeroed();
                section.header.identifier = u32::from_le_bytes(*b"SECc");
                section.header.data_subsection.camf.entry_table.element = &mut entry;
                section.header.data_subsection.camf.entry_table.size = 1;
                let mut x3f: x3f_t = std::mem::zeroed();
                x3f.directory_section.directory_entry = &mut section;
                x3f.directory_section.num_directory_entries = 1;
                let mut image = x3f_area16_t {
                    data: pixels.as_mut_ptr(),
                    buf: std::ptr::null_mut(),
                    rows: 3,
                    columns: 3,
                    channels: 3,
                    row_stride: 9,
                };
                super::interpolate_bad_pixels(&mut x3f, &mut image, 3);
            }
            pixels
        };

        for axis in 0..2 {
            for invalid in [0, i32::MAX as u32 + 1, u32::MAX] {
                let mut pattern = [1, 1, 2, 2];
                pattern[axis + 2] = invalid;
                assert_eq!(&render(pattern)[12..15], &[1000; 3]);
            }
            let mut pattern = [1, 1, 2, 2];
            pattern[axis] = u32::MAX;
            assert_eq!(&render(pattern)[12..15], &[1000; 3]);
        }
        // A valid large step must stop at the image edge without signed overflow.
        assert_eq!(render([1, 1, i32::MAX as u32, i32::MAX as u32]), [10; 27]);
        assert_eq!(render([1, 1, 2, 2]), [10; 27]);
    }

    #[test]
    fn uniform_digital_gain_preserves_intermediate_levels() {
        let gain = [1.13, 1.05, 0.98];
        let unit = intermediate_levels(gain, [1.0; 3], 13.137854439984503);
        assert_eq!(
            unit,
            intermediate_levels(gain, [4.0; 3], 13.137854439984503)
        );
        assert_eq!(unit[0], INTERMEDIATE_UNIT);
    }

    #[test]
    fn quattro_top_detail_preserves_neutral_color_with_unequal_digital_gain() {
        use crate::quattro::{x3f_expand_quattro, Area16};
        use std::ptr::null_mut;

        let gain = [1.1, 1.05, 0.99];
        let digital_gain = [4.0, 4.0, 2.0];
        let black = 100.0;
        let white = intermediate_levels(gain, digital_gain, black);
        let ranges: [f64; 3] = std::array::from_fn(|c| white[c] as f64 - black);
        let total_gain: [f64; 3] = std::array::from_fn(|c| gain[c] * digital_gain[c]);
        let encode = |luminance: f64, c: usize| {
            (black + ranges[c] * luminance / total_gain[c]).round() as u16
        };

        // A neutral high-frequency pattern whose 2x2 average is constant.
        // B/M record only that average; the full-resolution T plane records
        // alternating brighter and darker samples. Neither is clipped.
        let mut low = Vec::new();
        for _ in 0..16 {
            low.extend((0..3).map(|c| encode(0.1, c)));
        }
        let luminance = |row: usize, col: usize| {
            if (row + col).is_multiple_of(2) {
                0.14
            } else {
                0.06
            }
        };
        let mut top: Vec<u16> = (0..8)
            .flat_map(|row| (0..8).map(move |col| encode(luminance(row, col), 2)))
            .collect();
        let mut output = vec![0u16; 8 * 8 * 3];
        let mut image = Area16 {
            data: low.as_mut_ptr(),
            buf: null_mut(),
            rows: 4,
            columns: 4,
            channels: 3,
            row_stride: 12,
        };
        let mut qtop = Area16 {
            data: top.as_mut_ptr(),
            buf: null_mut(),
            rows: 8,
            columns: 8,
            channels: 1,
            row_stride: 8,
        };
        let mut expanded = Area16 {
            data: output.as_mut_ptr(),
            buf: null_mut(),
            rows: 8,
            columns: 8,
            channels: 3,
            row_stride: 24,
        };
        unsafe {
            x3f_expand_quattro(
                &mut image,
                null_mut(),
                &mut qtop,
                &mut expanded,
                null_mut(),
                0.0,
            );
        }
        for row in 0..8 {
            for col in 0..8 {
                for c in 0..3 {
                    let count = output[(row * 8 + col) * 3 + c] as f64;
                    let balanced = (count - black) / ranges[c] * total_gain[c];
                    assert!(
                        (balanced - luminance(row, col)).abs() < 1e-3,
                        "pixel ({row},{col}) channel {c}: {balanced} is not neutral detail {}",
                        luminance(row, col),
                    );
                }
            }
        }
    }

    fn s_for(global_max: f64, knee: f64) -> f64 {
        (global_max - knee) / (1.0 - knee)
    }

    #[test]
    fn shoulder_is_identity_when_no_overshoot() {
        // global_max == 1.0 ⇒ s == 1 ⇒ g(t) = t.
        let s = s_for(1.0, 0.85);
        for lum in [0.86, 0.9, 0.95, 1.0] {
            assert!((shoulder_compress(lum, 0.85, s) - lum).abs() < 1e-12);
        }
    }

    #[test]
    fn shoulder_maps_global_max_to_white() {
        let knee = 0.85;
        for global_max in [1.1, 1.75, 3.59] {
            let s = s_for(global_max, knee);
            let top = shoulder_compress(global_max, knee, s);
            assert!((top - 1.0).abs() < 1e-12, "g(s) = {top}");
        }
    }

    #[test]
    fn shoulder_is_continuous_and_unit_slope_at_knee() {
        let knee = 0.85;
        let s = s_for(2.0, knee);
        // g(knee) == knee (continuous against the linear segment).
        assert!((shoulder_compress(knee, knee, s) - knee).abs() < 1e-12);
        // Forward difference at the knee ≈ 1 (C1 join).
        let eps = 1e-7;
        let slope = (shoulder_compress(knee + eps, knee, s) - knee) / eps;
        assert!((slope - 1.0).abs() < 1e-4, "slope at knee = {slope}");
    }

    #[test]
    fn shoulder_is_monotone() {
        let knee = 0.85;
        let s = s_for(3.0, knee);
        let mut prev = knee;
        let mut lum = knee;
        while lum < 3.0 {
            lum += 0.01;
            let v = shoulder_compress(lum, knee, s);
            assert!(v >= prev - 1e-12, "non-monotone at lum={lum}");
            assert!(v <= 1.0 + 1e-12);
            prev = v;
        }
    }
}
