//! M6e4 — native Rust port of the highlight-recovery family from
//! `src/x3f_process.c`. This is the user's active research area: the
//! matrix-pathology gate, scene-derived chroma LUT, RepairPix
//! desaturation blend, and the `reconstruct_highlights` L*p snap.
//!
//! **Verbatim port.** The C source had been iterated on heavily before
//! the port; defaults, env-var override names, gate ordering, soft
//! window widths, etc. are load-bearing. The intent here is byte-for-
//! byte parity with the pre-port C output, so this file mirrors the
//! C structure 1:1 — including the function names, the field names,
//! the comment blocks (where they document the *why*), and the env
//! var spellings. Algorithmic improvements belong in a separate PR.
//!
//! ## Structs and the C/Rust boundary
//!
//! `highlight_params_t`, `chroma_lut_t`, `chroma_lut_apply_stats_t`,
//! and `repair_pix_t` are still allocated on the C-side stack inside
//! `preprocess_data` / `convert_data` / `apply_highlight_clip_dng`.
//! C calls into Rust through `#[no_mangle] extern "C"` entry points,
//! passing pointers to those structs. The Rust mirror types here use
//! `#[repr(C)]` and assert size/alignment in the test module so any
//! struct-layout drift between the C declarations in `x3f_process.c`
//! and these mirrors is caught at test time.
//!
//! ## Env-var parity
//!
//! The C source uses `getenv(...)` + `atof(...)` for tunable overrides.
//! We mirror that with `std::env::var(...)` + `libc::atof(...)` on a
//! NUL-terminated copy, so badly-formatted env values (`"0.5xyz"`)
//! parse identically (both stop at the first non-numeric character).
#![allow(clippy::missing_safety_doc)]
#![allow(non_camel_case_types)]

use std::ptr;

use crate::*;
// libc compat — see `sysabi.rs`.
use crate::sysabi as libc;

// ----------------------------------------------------------------------
// highlight_params_t  (matches the C typedef in src/x3f_process.c)
// ----------------------------------------------------------------------

#[repr(C)]
pub struct highlight_params_t {
    pub blending_low: f64,   // CAMF HighlightBlendingLow,   default 0.75
    pub blending_high: f64,  // CAMF HighlightBlendingHigh,  default 1.5
    pub restore_thresh: f64, // CAMF HighlightRestoreThresh, default 1.75
    pub sat_factor: f64,     // CAMF HighlightSatFactor,     default 1.0
    pub chan_thresh: f64,    // CAMF HighlightChanThresh1,   default 0.5
}

#[no_mangle]
pub unsafe extern "C" fn get_highlight_params(x3f: *mut x3f_t, p: *mut highlight_params_t) {
    let pp = unsafe { &mut *p };
    pp.blending_low = 0.75;
    pp.blending_high = 1.5;
    pp.restore_thresh = 1.75;
    pp.sat_factor = 1.0;
    pp.chan_thresh = 0.5;
    unsafe {
        x3f_get_camf_float(
            x3f,
            c"HighlightBlendingLow".as_ptr() as *mut _,
            &mut pp.blending_low,
        );
        x3f_get_camf_float(
            x3f,
            c"HighlightBlendingHigh".as_ptr() as *mut _,
            &mut pp.blending_high,
        );
        x3f_get_camf_float(
            x3f,
            c"HighlightRestoreThresh".as_ptr() as *mut _,
            &mut pp.restore_thresh,
        );
        x3f_get_camf_float(
            x3f,
            c"HighlightSatFactor".as_ptr() as *mut _,
            &mut pp.sat_factor,
        );
        x3f_get_camf_float(
            x3f,
            c"HighlightChanThresh1".as_ptr() as *mut _,
            &mut pp.chan_thresh,
        );
    }
}

// ----------------------------------------------------------------------
// compute_chroma_prior — sat_ratio direction that the conv_matrix maps
// to neutral display white. Used as the L*p snap target for
// reconstruct_highlights *and* as the neutral_tm baseline for the CLUT
// bin-evidence gate.
// ----------------------------------------------------------------------

#[no_mangle]
pub unsafe extern "C" fn compute_chroma_prior(conv_matrix: *const f64, p_out: *mut f64) {
    let mut inv = [0.0_f64; 9];
    let mut one = [1.0_f64; 3];
    let mut dir = [0.0_f64; 3];
    unsafe {
        x3f_3x3_inverse(conv_matrix as *mut f64, inv.as_mut_ptr());
        x3f_3x3_3x1_mul(inv.as_mut_ptr(), one.as_mut_ptr(), dir.as_mut_ptr());
    }

    let mut mx = 0.0_f64;
    for v in &dir {
        if *v > mx {
            mx = *v;
        }
    }
    let out = unsafe { std::slice::from_raw_parts_mut(p_out, 3) };
    if mx <= 1e-12 {
        out[0] = 1.0;
        out[1] = 1.0;
        out[2] = 1.0;
        return;
    }
    for c in 0..3 {
        out[c] = dir[c] / mx;
    }
}

// ----------------------------------------------------------------------
// reconstruct_highlights — per-pixel matrix-pathology gate. Lifts
// asymmetrically-clipped BMT towards L*p so the matrix sees proportional
// channels and produces neutral-white instead of a multi-colour cast.
// ----------------------------------------------------------------------

#[no_mangle]
pub unsafe extern "C" fn reconstruct_highlights(
    s: *mut f64,
    p: *const f64,
    hp: *const highlight_params_t,
) {
    let hp = unsafe { &*hp };
    let s = unsafe { std::slice::from_raw_parts_mut(s, 3) };
    let p = unsafe { std::slice::from_raw_parts(p, 3) };

    let mut s_max = 0.0_f64;
    let mut u_max = 0.0_f64;
    for c in 0..3 {
        let pc = if p[c] > 1e-12 { p[c] } else { 1e-12 };
        let u_c = s[c] / pc;
        if s[c] > s_max {
            s_max = s[c];
        }
        if u_c > u_max {
            u_max = u_c;
        }
    }
    if s_max <= hp.blending_low {
        return;
    }

    let big_l = u_max * hp.sat_factor;

    if u_max >= hp.restore_thresh {
        for c in 0..3 {
            s[c] = big_l * p[c];
        }
        return;
    }

    let mut tt = (s_max - hp.blending_low) / (1.0 - hp.blending_low);
    if tt < 0.0 {
        tt = 0.0;
    }
    if tt > 1.0 {
        tt = 1.0;
    }
    for c in 0..3 {
        let recon = big_l * p[c];
        s[c] = (1.0 - tt) * s[c] + tt * recon;
    }
}

// ----------------------------------------------------------------------
// chroma_lut_t  (matches C declaration in src/x3f_process.c)
// ----------------------------------------------------------------------

pub const CHROMA_LUT_BINS: usize = 256;
const CHROMA_LUT_EPS: f64 = 1e-12;

#[repr(C)]
pub struct chroma_lut_t {
    pub lut: [f32; CHROMA_LUT_BINS],
    pub valid: libc::c_int,
    pub sat_threshold: f64,
    pub donor_min_brightness: f64,
    pub recovery_cap: f64,
    pub asymmetric_max: f64,
    pub soft_window: f64,
    pub neutral_tm: f64,
    pub blend_threshold: f64,
    pub blend_divisor: f64,
    // --- M8 DNG highlight-recovery extension: per-channel tables -----
    // The original `lut` reconstructs a clipped T from unclipped B,M
    // (bin = B/(B+M), value = T/M). These two extend the same scheme to
    // the other clip identities so a single-clipped-channel pixel keeps
    // its scene chroma instead of falling through to the neutral L*p
    // snap in `reconstruct_highlights`:
    //   lut_b: bin = M/(M+T), value = B/M  (B clipped; M,T donors)
    //   lut_m: bin = B/(B+T), value = M/T  (M clipped; B,T donors)
    // Only consumed by `chroma_lut_apply_pixel_bmt` (the DNG recovery
    // path); `chroma_lut_apply_pixel` (TIFF/PPM path, MD5-pinned)
    // ignores them.
    pub lut_b: [f32; CHROMA_LUT_BINS],
    pub lut_m: [f32; CHROMA_LUT_BINS],
    pub valid_b: libc::c_int,
    pub valid_m: libc::c_int,
    pub neutral_bm: f64,
    pub neutral_mt: f64,
}

/// Read an env var as a NUL-terminated bytestring and run it through
/// `libc::atof` so badly-formatted values (`"0.5xyz"`) parse identically
/// to the C version.
fn env_atof(name: &str) -> Option<f64> {
    let v = std::env::var(name).ok()?;
    let mut buf: Vec<u8> = Vec::with_capacity(v.len() + 1);
    buf.extend_from_slice(v.as_bytes());
    buf.push(0);
    Some(unsafe { libc::atof(buf.as_ptr() as *const libc::c_char) })
}

fn env_atoi(name: &str) -> Option<libc::c_int> {
    let v = std::env::var(name).ok()?;
    let mut buf: Vec<u8> = Vec::with_capacity(v.len() + 1);
    buf.extend_from_slice(v.as_bytes());
    buf.push(0);
    Some(unsafe { libc::atoi(buf.as_ptr() as *const libc::c_char) })
}

fn env_present(name: &str) -> bool {
    std::env::var_os(name).is_some()
}

#[no_mangle]
pub unsafe extern "C" fn chroma_lut_init_defaults(lut: *mut chroma_lut_t) {
    let lut = unsafe { &mut *lut };
    for v in lut.lut.iter_mut() {
        *v = 0.0;
    }
    for v in lut.lut_b.iter_mut() {
        *v = 0.0;
    }
    for v in lut.lut_m.iter_mut() {
        *v = 0.0;
    }
    lut.valid = 0;
    lut.valid_b = 0;
    lut.valid_m = 0;
    lut.neutral_tm = 1.0;
    lut.neutral_bm = 1.0;
    lut.neutral_mt = 1.0;
    lut.sat_threshold = 0.99;
    lut.donor_min_brightness = 0.20;
    lut.recovery_cap = 1.75; // matches HighlightRestoreThresh
                             // 0.95: production. The earlier 0.50 default was tuned against the
                             // SPP oracle before the matrix-pathology gate, brightness-blend,
                             // and bin-evidence soft window were added. With those downstream
                             // safety nets in place the wider ASYM window is net-positive.
    lut.asymmetric_max = 0.95;
    // 0.20: tuned visually on CLIPPED_IMAGE_MERRILL hand/bottle scene.
    lut.soft_window = 0.20;
    // Brightness-blend defaults — see C source for tuning notes.
    lut.blend_threshold = 0.75;
    lut.blend_divisor = 0.10;

    if let Some(v) = env_atof("X3F_CHROMA_LUT_SAT") {
        lut.sat_threshold = v;
    }
    if let Some(v) = env_atof("X3F_CHROMA_LUT_DONOR") {
        lut.donor_min_brightness = v;
    }
    if let Some(v) = env_atof("X3F_CHROMA_LUT_CAP") {
        lut.recovery_cap = v;
    }
    if let Some(v) = env_atof("X3F_CHROMA_LUT_ASYM") {
        lut.asymmetric_max = v;
    }
    if let Some(v) = env_atof("X3F_CHROMA_LUT_SOFT") {
        lut.soft_window = v;
    }
    if let Some(v) = env_atof("X3F_CHROMA_LUT_BLEND_THRESH") {
        lut.blend_threshold = v;
    }
    if let Some(v) = env_atof("X3F_CHROMA_LUT_BLEND_DIV") {
        lut.blend_divisor = v;
    }
    if lut.blend_divisor < 1e-6 {
        lut.blend_divisor = 1e-6;
    }
}

#[no_mangle]
pub unsafe extern "C" fn chroma_lut_build_from_image(
    lut: *mut chroma_lut_t,
    image: *const x3f_area16_t,
    ilevels: *const x3f_image_levels_t,
    prior: *const f64,
) -> libc::c_int {
    let lut = unsafe { &mut *lut };
    let image = unsafe { &*image };
    let ilevels = unsafe { &*ilevels };
    let prior = unsafe { std::slice::from_raw_parts(prior, 3) };

    if image.channels < 3 {
        return 0;
    }

    let mut corr_sum = vec![0.0_f64; CHROMA_LUT_BINS];
    let mut corr_count = vec![0u32; CHROMA_LUT_BINS];
    // M8 — accumulators for the B- and M-clip reconstruction tables.
    let mut corr_sum_b = vec![0.0_f64; CHROMA_LUT_BINS];
    let mut corr_count_b = vec![0u32; CHROMA_LUT_BINS];
    let mut corr_sum_m = vec![0.0_f64; CHROMA_LUT_BINS];
    let mut corr_count_m = vec![0u32; CHROMA_LUT_BINS];

    #[inline]
    fn ratio_bin(num: f64, den: f64) -> usize {
        let mut bin = (num * (CHROMA_LUT_BINS as f64 - 1.0) / den) as i32;
        if bin < 0 {
            bin = 0;
        } else if bin > CHROMA_LUT_BINS as i32 - 1 {
            bin = CHROMA_LUT_BINS as i32 - 1;
        }
        bin as usize
    }

    let row_stride = image.row_stride as usize;
    let channels = image.channels as usize;
    for row in 0..image.rows as usize {
        for col in 0..image.columns as usize {
            let mut s = [0.0_f64; 3];
            for c in 0..3 {
                let idx = row_stride * row + channels * col + c;
                let v = unsafe { *image.data.add(idx) } as f64;
                s[c] = (v - ilevels.black[c]) / (ilevels.white[c] as f64 - ilevels.black[c]);
            }

            let sb = if s[0] > CHROMA_LUT_EPS {
                s[0]
            } else {
                CHROMA_LUT_EPS
            };
            let sm = if s[1] > CHROMA_LUT_EPS {
                s[1]
            } else {
                CHROMA_LUT_EPS
            };
            let st = if s[2] > CHROMA_LUT_EPS {
                s[2]
            } else {
                CHROMA_LUT_EPS
            };

            // T table: donor needs T unclipped and B+M bright enough.
            if s[2] < lut.sat_threshold && s[0] + s[1] > lut.donor_min_brightness {
                let bin = ratio_bin(sb, sb + sm);
                corr_sum[bin] += st / sm;
                corr_count[bin] += 1;
            }
            // B table: donor needs B unclipped and M+T bright enough.
            if s[0] < lut.sat_threshold && s[1] + s[2] > lut.donor_min_brightness {
                let bin = ratio_bin(sm, sm + st);
                corr_sum_b[bin] += sb / sm;
                corr_count_b[bin] += 1;
            }
            // M table: donor needs M unclipped and B+T bright enough.
            if s[1] < lut.sat_threshold && s[0] + s[2] > lut.donor_min_brightness {
                let bin = ratio_bin(sb, sb + st);
                corr_sum_m[bin] += sm / st;
                corr_count_m[bin] += 1;
            }
        }
    }

    let neutral_tm = if prior[1] > CHROMA_LUT_EPS {
        prior[2] / prior[1]
    } else {
        1.0
    };
    let neutral_bm = if prior[1] > CHROMA_LUT_EPS {
        prior[0] / prior[1]
    } else {
        1.0
    };
    let neutral_mt = if prior[2] > CHROMA_LUT_EPS {
        prior[1] / prior[2]
    } else {
        1.0
    };

    // Empty-bin nearest-populated fill (default radius 16; capped so a
    // genuinely-absent chromaticity falls through to the neutral ratio).
    let mut fill_dist = 16_i32;
    if let Some(v) = env_atoi("X3F_CHROMA_LUT_FILL_DIST") {
        fill_dist = v;
    }
    if fill_dist < 0 {
        fill_dist = 0;
    }

    /// Average per-bin sums, fill empty bins from the nearest populated
    /// neighbour (falling back to `neutral`). Returns the populated-bin
    /// count (0 = table unusable).
    fn finish_table(
        sum: &mut [f64],
        count: &[u32],
        neutral: f64,
        fill_dist: i32,
        out: &mut [f32; CHROMA_LUT_BINS],
    ) -> i32 {
        let mut populated = 0;
        for c in 0..CHROMA_LUT_BINS {
            if count[c] > 0 {
                sum[c] /= count[c] as f64;
                populated += 1;
            }
        }
        if populated == 0 {
            return 0;
        }
        for c in 0..CHROMA_LUT_BINS {
            if count[c] > 0 {
                continue;
            }
            let mut found: i32 = -1;
            for d in 1..=fill_dist {
                let lo = c as i32 - d;
                let hi = c as i32 + d;
                if lo >= 0 && count[lo as usize] > 0 {
                    found = lo;
                    break;
                }
                if hi < CHROMA_LUT_BINS as i32 && count[hi as usize] > 0 {
                    found = hi;
                    break;
                }
            }
            if found >= 0 {
                sum[c] = sum[found as usize];
            } else {
                sum[c] = neutral;
            }
        }
        for c in 0..CHROMA_LUT_BINS {
            out[c] = sum[c] as f32;
        }
        populated
    }

    let populated = finish_table(
        &mut corr_sum,
        &corr_count,
        neutral_tm,
        fill_dist,
        &mut lut.lut,
    );
    if populated == 0 {
        lut.valid = 0;
        return 0;
    }
    lut.valid = 1;
    lut.neutral_tm = neutral_tm;

    lut.valid_b = (finish_table(
        &mut corr_sum_b,
        &corr_count_b,
        neutral_bm,
        fill_dist,
        &mut lut.lut_b,
    ) > 0) as libc::c_int;
    lut.neutral_bm = neutral_bm;
    lut.valid_m = (finish_table(
        &mut corr_sum_m,
        &corr_count_m,
        neutral_mt,
        fill_dist,
        &mut lut.lut_m,
    ) > 0) as libc::c_int;
    lut.neutral_mt = neutral_mt;

    if env_present("X3F_CHROMA_LUT_TRACE") {
        unsafe {
            x3f_printf(
                x3f_verbosity_t_INFO,
                c"chroma_lut: %d/%d bins populated, lut[0]=%.3f lut[64]=%.3f lut[128]=%.3f lut[192]=%.3f lut[255]=%.3f (neutral_tm=%.3f), sat=%.3f donor=%.3f cap=%.3f\n".as_ptr(),
                populated as libc::c_int,
                CHROMA_LUT_BINS as libc::c_int,
                lut.lut[0] as f64,
                lut.lut[64] as f64,
                lut.lut[128] as f64,
                lut.lut[192] as f64,
                lut.lut[CHROMA_LUT_BINS - 1] as f64,
                neutral_tm,
                lut.sat_threshold,
                lut.donor_min_brightness,
                lut.recovery_cap,
            );
        }
    }
    // X3F_CHROMA_LUT_DUMP debug trace — uses variadic libc::fprintf to
    // write per-bin lines to stderr. Gated out on wasm32-unknown-unknown
    // because Rust can't shim variadic `fprintf` on stable; the trace
    // surface isn't reachable from any wasm consumer entrypoint anyway.
    #[cfg(not(target_arch = "wasm32"))]
    if env_present("X3F_CHROMA_LUT_DUMP") {
        for i in 0..CHROMA_LUT_BINS {
            unsafe {
                libc::fprintf(
                    libc_stderr(),
                    c"BIN %d %u %.4f\n".as_ptr(),
                    i as libc::c_int,
                    corr_count[i],
                    lut.lut[i] as f64,
                );
            }
        }
    }
    1
}

// ----------------------------------------------------------------------
// chroma_lut_apply_stats_t  (matches the C typedef)
// ----------------------------------------------------------------------

#[repr(C)]
pub struct chroma_lut_apply_stats_t {
    pub total_eval: u64,
    pub t_strength_kill: u64,
    pub asym_kill: u64,
    pub bm_clip_kill: u64,
    pub bin_evidence_kill: u64,
    pub applied: u64,
    pub bin_kill_hist: [u64; CHROMA_LUT_BINS],
    pub bin_applied_hist: [u64; CHROMA_LUT_BINS],
}

#[no_mangle]
pub unsafe extern "C" fn chroma_lut_apply_pixel(
    s: *mut f64,
    lut: *const chroma_lut_t,
    stats: *mut chroma_lut_apply_stats_t,
) -> libc::c_int {
    let lut = unsafe { &*lut };
    if lut.valid == 0 {
        return 0;
    }
    let s = unsafe { std::slice::from_raw_parts_mut(s, 3) };
    if !stats.is_null() {
        unsafe { (*stats).total_eval += 1 };
    }

    // Soft t-clip ramp: full strength at s[T] >= sat_threshold,
    // fading to 0 at s[T] = sat_threshold - soft_window.
    let mut t_strength = (s[2] - (lut.sat_threshold - lut.soft_window)) / lut.soft_window;
    if t_strength <= 0.0 {
        if !stats.is_null() {
            unsafe { (*stats).t_strength_kill += 1 };
        }
        return 0;
    }
    if t_strength > 1.0 {
        t_strength = 1.0;
    }

    // Asymmetry guard.
    let s_bm_max = if s[0] > s[1] { s[0] } else { s[1] };
    let mut asym_strength = (lut.asymmetric_max - s_bm_max) / lut.soft_window;
    if asym_strength <= 0.0 {
        if !stats.is_null() {
            unsafe { (*stats).asym_kill += 1 };
        }
        return 0;
    }
    if asym_strength > 1.0 {
        asym_strength = 1.0;
    }

    // Either of B/M actually clipped is a hard kill.
    if s[0] >= lut.sat_threshold || s[1] >= lut.sat_threshold {
        if !stats.is_null() {
            unsafe { (*stats).bm_clip_kill += 1 };
        }
        return 0;
    }

    // Sigma's brightness blend — weighted_lum from sorted s[] drives a
    // parallel strength signal that overrides asym damping in bright
    // regions. max() with t_strength*asym_strength.
    let strength = {
        let (x0, x1, x2) = (s[0], s[1], s[2]);
        let (mx, md);
        if x0 >= x1 && x0 >= x2 {
            mx = x0;
            md = if x1 >= x2 { x1 } else { x2 };
        } else if x1 >= x2 {
            mx = x1;
            md = if x0 >= x2 { x0 } else { x2 };
        } else {
            mx = x2;
            md = if x0 >= x1 { x0 } else { x1 };
        }
        let weighted_lum = mx * (2.0 / 3.0) + md * (1.0 / 3.0);
        let mut blend_t = (weighted_lum - lut.blend_threshold) / lut.blend_divisor;
        if blend_t < 0.0 {
            blend_t = 0.0;
        } else if blend_t > 1.0 {
            blend_t = 1.0;
        }
        let base = t_strength * asym_strength;
        if blend_t > base {
            blend_t
        } else {
            base
        }
    };

    let sb = if s[0] > CHROMA_LUT_EPS {
        s[0]
    } else {
        CHROMA_LUT_EPS
    };
    let sm = if s[1] > CHROMA_LUT_EPS {
        s[1]
    } else {
        CHROMA_LUT_EPS
    };

    let mut bin = (sb * (CHROMA_LUT_BINS as f64 - 1.0) / (sb + sm)) as i32;
    if bin < 0 {
        bin = 0;
    } else if bin > CHROMA_LUT_BINS as i32 - 1 {
        bin = CHROMA_LUT_BINS as i32 - 1;
    }
    let bin = bin as usize;

    let bin_v = lut.lut[bin] as f64;
    let rel_diff = (bin_v - lut.neutral_tm) / lut.neutral_tm;
    if rel_diff < 0.05 && rel_diff > -0.05 {
        if !stats.is_null() {
            unsafe {
                (*stats).bin_evidence_kill += 1;
                (*stats).bin_kill_hist[bin] += 1;
            }
        }
        return 0;
    }

    let mut recovered = sm * bin_v;
    if recovered > lut.recovery_cap {
        recovered = lut.recovery_cap;
    }

    s[2] = (1.0 - strength) * s[2] + strength * recovered;

    if !stats.is_null() {
        unsafe {
            (*stats).applied += 1;
            (*stats).bin_applied_hist[bin] += 1;
        }
    }
    1
}

// ----------------------------------------------------------------------
// chroma_lut_apply_pixel_bmt — M8 generalized per-channel reconstruction
// (DNG recovery path only).
//
// `chroma_lut_apply_pixel` above only ever repairs a clipped T from
// unclipped B,M; pixels where B or M clipped fall through to the
// neutral L*p snap in `reconstruct_highlights`, which destroys scene
// chroma. Sigma Photo Pro's pipeline ("HPRestoration" + the HR/HN
// blend in F23RestoreHighlights) instead reconstructs *each* clipped
// channel from the unclipped ones and only neutralizes as a last
// resort — that per-channel restoration is why SPP keeps more color
// in recovered highlights. This function mirrors that: it tries the
// original T repair first (identical math — it delegates), then the
// B-clipped and M-clipped identities using the `lut_b` / `lut_m`
// tables, with the same gate structure (soft clip ramp on the clipped
// channel, donor-asymmetry damping, donor hard-clip kill, bin-evidence
// kill, recovery cap) transposed to each case.
//
// Returns 1 when any channel was repaired; 0 means the caller should
// fall back to `reconstruct_highlights`.
// ----------------------------------------------------------------------

#[no_mangle]
pub unsafe extern "C" fn chroma_lut_apply_pixel_bmt(
    s: *mut f64,
    lut: *const chroma_lut_t,
    stats: *mut chroma_lut_apply_stats_t,
) -> libc::c_int {
    // T-clipped identity first: byte-identical to the original pass.
    if unsafe { chroma_lut_apply_pixel(s, lut, stats) } != 0 {
        return 1;
    }
    let lut = unsafe { &*lut };
    if lut.valid == 0 {
        return 0;
    }
    let s = unsafe { std::slice::from_raw_parts_mut(s, 3) };

    // (clipped channel, donor a, donor b, amplitude-from-db, table,
    //  neutral, valid). The bin is always da/(da+db); the table value is
    //  a ratio against the amplitude donor:
    //   B identity: B' = M * lut_b[bin(M/(M+T))]  (amplitude = da = M)
    //   M identity: M' = T * lut_m[bin(B/(B+T))]  (amplitude = db = T)
    for &(x, da, db, amp_from_db, table, neutral, valid) in &[
        (
            0usize,
            1usize,
            2usize,
            false,
            &lut.lut_b,
            lut.neutral_bm,
            lut.valid_b,
        ),
        (
            1usize,
            0usize,
            2usize,
            true,
            &lut.lut_m,
            lut.neutral_mt,
            lut.valid_m,
        ),
    ] {
        if valid == 0 {
            continue;
        }

        // Soft clip ramp on the channel being repaired (mirrors
        // t_strength in the T pass).
        let mut x_strength = (s[x] - (lut.sat_threshold - lut.soft_window)) / lut.soft_window;
        if x_strength <= 0.0 {
            continue;
        }
        if x_strength > 1.0 {
            x_strength = 1.0;
        }

        // Donors actually clipped is a hard kill (mirrors bm_clip_kill).
        if s[da] >= lut.sat_threshold || s[db] >= lut.sat_threshold {
            continue;
        }

        // Donor asymmetry damping (mirrors the asym guard).
        let d_max = if s[da] > s[db] { s[da] } else { s[db] };
        let mut asym_strength = (lut.asymmetric_max - d_max) / lut.soft_window;
        if asym_strength <= 0.0 {
            continue;
        }
        if asym_strength > 1.0 {
            asym_strength = 1.0;
        }

        // Brightness-blend override — same weighted_lum signal as the
        // T pass, so very bright regions keep full strength even when
        // the asym guard damps.
        let strength = {
            let (x0, x1, x2) = (s[0], s[1], s[2]);
            let (mx, md);
            if x0 >= x1 && x0 >= x2 {
                mx = x0;
                md = if x1 >= x2 { x1 } else { x2 };
            } else if x1 >= x2 {
                mx = x1;
                md = if x0 >= x2 { x0 } else { x2 };
            } else {
                mx = x2;
                md = if x0 >= x1 { x0 } else { x1 };
            }
            let weighted_lum = mx * (2.0 / 3.0) + md * (1.0 / 3.0);
            let mut blend_t = (weighted_lum - lut.blend_threshold) / lut.blend_divisor;
            if blend_t < 0.0 {
                blend_t = 0.0;
            } else if blend_t > 1.0 {
                blend_t = 1.0;
            }
            let base = x_strength * asym_strength;
            if blend_t > base {
                blend_t
            } else {
                base
            }
        };

        let va = if s[da] > CHROMA_LUT_EPS {
            s[da]
        } else {
            CHROMA_LUT_EPS
        };
        let vb = if s[db] > CHROMA_LUT_EPS {
            s[db]
        } else {
            CHROMA_LUT_EPS
        };

        let mut bin = (va * (CHROMA_LUT_BINS as f64 - 1.0) / (va + vb)) as i32;
        if bin < 0 {
            bin = 0;
        } else if bin > CHROMA_LUT_BINS as i32 - 1 {
            bin = CHROMA_LUT_BINS as i32 - 1;
        }
        let bin = bin as usize;

        // Bin-evidence kill: a near-neutral table entry carries no
        // chroma information beyond what the neutral fallback gives.
        let bin_v = table[bin] as f64;
        let rel_diff = (bin_v - neutral) / neutral;
        if rel_diff < 0.05 && rel_diff > -0.05 {
            if !stats.is_null() {
                unsafe {
                    (*stats).bin_evidence_kill += 1;
                    (*stats).bin_kill_hist[bin] += 1;
                }
            }
            continue;
        }

        // Reconstruct from the amplitude donor (M for the B identity,
        // T for the M identity).
        let amp = if amp_from_db { vb } else { va };
        let mut recovered = amp * bin_v;
        if recovered > lut.recovery_cap {
            recovered = lut.recovery_cap;
        }

        s[x] = (1.0 - strength) * s[x] + strength * recovered;

        if !stats.is_null() {
            unsafe {
                (*stats).applied += 1;
                (*stats).bin_applied_hist[bin] += 1;
            }
        }
        return 1;
    }
    0
}

/// Print one-line trace stats for the chroma LUT to stderr — exercised
/// when `X3F_CHROMA_LUT_TRACE` / `X3F_CHROMA_LUT_TRACE_HIST` env vars
/// are set. No-op on wasm32-unknown-unknown (variadic `fprintf` can't
/// be shimmed on stable Rust; the trace surface isn't reachable from
/// any wasm consumer anyway).
#[cfg(not(target_arch = "wasm32"))]
#[no_mangle]
pub unsafe extern "C" fn chroma_lut_apply_stats_print(
    s: *const chroma_lut_apply_stats_t,
    path: *const libc::c_char,
) {
    let s = unsafe { &*s };
    if s.total_eval == 0 {
        return;
    }
    let pct = 100.0 * (s.applied as f64) / (s.total_eval as f64);
    unsafe {
        libc::fprintf(
            libc_stderr(),
            c"chroma_lut[%s] eval=%llu t_kill=%llu asym_kill=%llu bm_kill=%llu bin_evidence_kill=%llu applied=%llu (%.1f%% applied)\n".as_ptr(),
            path,
            s.total_eval,
            s.t_strength_kill,
            s.asym_kill,
            s.bm_clip_kill,
            s.bin_evidence_kill,
            s.applied,
            pct,
        );
    }
    if env_present("X3F_CHROMA_LUT_TRACE_HIST") {
        for b in 0..CHROMA_LUT_BINS {
            if s.bin_kill_hist[b] != 0 || s.bin_applied_hist[b] != 0 {
                unsafe {
                    libc::fprintf(
                        libc_stderr(),
                        c"chroma_lut_hist[%s] bin=%d kill=%llu applied=%llu\n".as_ptr(),
                        path,
                        b as libc::c_int,
                        s.bin_kill_hist[b],
                        s.bin_applied_hist[b],
                    );
                }
            }
        }
    }
}

/// wasm32 stub for `chroma_lut_apply_stats_print` — a no-op. See the
/// non-wasm version above for what it does on host.
#[cfg(target_arch = "wasm32")]
#[no_mangle]
pub unsafe extern "C" fn chroma_lut_apply_stats_print(
    _s: *const chroma_lut_apply_stats_t,
    _path: *const libc::c_char,
) {
}

// ----------------------------------------------------------------------
// repair_pix_t  (matches the C typedef)
// ----------------------------------------------------------------------

#[repr(C)]
pub struct repair_pix_t {
    pub valid: libc::c_int,
    pub sat_threshold: f64,
    pub blend_threshold: f64,
    pub blend_divisor: f64,
    pub anchor_clamp_lo: f64,
}

#[no_mangle]
pub unsafe extern "C" fn repair_pix_init_defaults(rp: *mut repair_pix_t) {
    let rp = unsafe { &mut *rp };
    rp.valid = 0;
    rp.sat_threshold = 0.97;
    rp.blend_threshold = 0.85;
    rp.blend_divisor = 0.10;
    rp.anchor_clamp_lo = 0.50;

    if let Some(v) = env_atof("X3F_REPAIR_PIX_SAT") {
        rp.sat_threshold = v;
    }
    if let Some(v) = env_atof("X3F_REPAIR_PIX_THRESH") {
        rp.blend_threshold = v;
    }
    if let Some(v) = env_atof("X3F_REPAIR_PIX_DIVISOR") {
        rp.blend_divisor = v;
    }
    if let Some(v) = env_atof("X3F_REPAIR_PIX_ANCHOR_LO") {
        rp.anchor_clamp_lo = v;
    }
}

#[no_mangle]
pub unsafe extern "C" fn build_sat_map(
    image: *const x3f_area16_t,
    ilevels: *const x3f_image_levels_t,
    sat_threshold: f64,
) -> *mut u8 {
    let image = unsafe { &*image };
    let ilevels = unsafe { &*ilevels };

    if image.channels < 3 {
        return ptr::null_mut();
    }
    let n = image.rows as usize * image.columns as usize;
    let map = unsafe { libc::calloc(n, 1) as *mut u8 };
    if map.is_null() {
        return ptr::null_mut();
    }

    let row_stride = image.row_stride as usize;
    let channels = image.channels as usize;
    let cols = image.columns as usize;
    for row in 0..image.rows as usize {
        for col in 0..cols {
            let mut flag: u8 = 0;
            for c in 0..3 {
                let idx = row_stride * row + channels * col + c;
                let v = unsafe { *image.data.add(idx) } as f64;
                let s = (v - ilevels.black[c]) / (ilevels.white[c] as f64 - ilevels.black[c]);
                if s >= sat_threshold {
                    flag = 1;
                    break;
                }
            }
            unsafe {
                *map.add(row * cols + col) = flag;
            }
        }
    }
    map
}

#[no_mangle]
pub unsafe extern "C" fn repair_pix_apply_pixel(
    s: *mut f64,
    p: *const f64,
    rp: *const repair_pix_t,
    sat_map: *const u8,
    row: libc::c_int,
    col: libc::c_int,
    rows: libc::c_int,
    cols: libc::c_int,
) {
    let rp = unsafe { &*rp };
    let s = unsafe { std::slice::from_raw_parts_mut(s, 3) };
    let p = unsafe { std::slice::from_raw_parts(p, 3) };

    let mut big_l = [0.0_f64; 3];
    let mut pp = [0.0_f64; 3];
    for c in 0..3 {
        pp[c] = if p[c] > 1e-12 { p[c] } else { 1e-12 };
        big_l[c] = s[c] / pp[c];
    }

    // Sort big_l into (max, mid, min). Mirrors the C 6-way branch.
    let (max_v, mid_v, _min_v);
    if big_l[0] >= big_l[1] {
        if big_l[1] >= big_l[2] {
            max_v = big_l[0];
            mid_v = big_l[1];
            _min_v = big_l[2];
        } else if big_l[0] >= big_l[2] {
            max_v = big_l[0];
            mid_v = big_l[2];
            _min_v = big_l[1];
        } else {
            max_v = big_l[2];
            mid_v = big_l[0];
            _min_v = big_l[1];
        }
    } else if big_l[0] >= big_l[2] {
        max_v = big_l[1];
        mid_v = big_l[0];
        _min_v = big_l[2];
    } else if big_l[1] >= big_l[2] {
        max_v = big_l[1];
        mid_v = big_l[2];
        _min_v = big_l[0];
    } else {
        max_v = big_l[2];
        mid_v = big_l[1];
        _min_v = big_l[0];
    }

    let weighted_lum = max_v * (2.0 / 3.0) + mid_v * (1.0 / 3.0);
    let mut blend_t = (weighted_lum - rp.blend_threshold) / rp.blend_divisor;
    if blend_t < 0.0 {
        blend_t = 0.0;
    } else if blend_t > 1.0 {
        blend_t = 1.0;
    }

    // 5x5 saturation-neighbourhood sum (interior only).
    let mut sum5 = 0.0_f64;
    if !sat_map.is_null() && row >= 2 && col >= 2 && row < rows - 2 && col < cols - 2 {
        let cols_us = cols as usize;
        for r in (row - 2)..=(row + 2) {
            for ccol in (col - 2)..=(col + 2) {
                let idx = r as usize * cols_us + ccol as usize;
                sum5 += unsafe { *sat_map.add(idx) } as f64;
            }
        }
    }
    let mut neigh_t = sum5 * (1.0 / 25.0);
    if neigh_t > 1.0 {
        neigh_t = 1.0;
    }

    let mut big_s = (0.5 * (blend_t * blend_t + neigh_t * neigh_t)).sqrt();
    if big_s <= 0.0 {
        return;
    }
    if big_s > 1.0 {
        big_s = 1.0;
    }

    // Trace path — mirrors the C "static int traced" gate so only the
    // first matching pixel emits a line per process invocation.
    if env_present("X3F_REPAIR_PIX_TRACE") {
        let trace_r = env_atoi("X3F_REPAIR_PIX_TRACE_R").unwrap_or(-1);
        let trace_c = env_atoi("X3F_REPAIR_PIX_TRACE_C").unwrap_or(-1);
        if row == trace_r && col == trace_c {
            // Atomic-ish "fire once" gate. A static AtomicBool would
            // match the C `static int traced=0` exactly; in practice
            // this trace path is opt-in via env var so the slight
            // simplification doesn't matter for parity.
            use std::sync::atomic::{AtomicBool, Ordering};
            static TRACED: AtomicBool = AtomicBool::new(false);
            if !TRACED.swap(true, Ordering::Relaxed) {
                unsafe {
                    x3f_printf(
                        x3f_verbosity_t_INFO,
                        c"repair_pix trace (r=%d c=%d): s_in=(%.4f,%.4f,%.4f) prior=(%.4f,%.4f,%.4f) L=(%.4f,%.4f,%.4f) sorted=(%.4f,%.4f) wlum=%.4f blend_t=%.4f neigh_sum=%.1f neigh_t=%.4f S=%.4f anchor_L=%.4f\n".as_ptr(),
                        row, col,
                        s[0], s[1], s[2], p[0], p[1], p[2],
                        big_l[0], big_l[1], big_l[2],
                        max_v, mid_v,
                        weighted_lum, blend_t, sum5, neigh_t, big_s, big_l[0],
                    );
                }
            }
        }
    }

    // Anchor: max(L) with a low clamp so dark pixels can't pull the
    // blend below a sane luminance.
    let mut anchor_l = max_v;
    if anchor_l < rp.anchor_clamp_lo {
        anchor_l = rp.anchor_clamp_lo;
    }

    // Channel index of max L is the anchor; other channels blend toward
    // anchor_L.
    let mut p_min_idx = 0usize;
    if big_l[1] > big_l[p_min_idx] {
        p_min_idx = 1;
    }
    if big_l[2] > big_l[p_min_idx] {
        p_min_idx = 2;
    }

    for c in 0..3 {
        if c == p_min_idx {
            continue;
        }
        let l_new = big_s * anchor_l + (1.0 - big_s) * big_l[c];
        s[c] = l_new * pp[c];
    }
}

// ----------------------------------------------------------------------
// libc stderr accessor — `libc` doesn't expose `stderr` directly on all
// platforms; on macOS+linux it's `__stderrp` / `stderr`.
// ----------------------------------------------------------------------
extern "C" {
    #[cfg(target_vendor = "apple")]
    static __stderrp: *mut libc::FILE;
    #[cfg(not(target_vendor = "apple"))]
    static stderr: *mut libc::FILE;
}

#[inline]
fn libc_stderr() -> *mut libc::FILE {
    #[cfg(target_vendor = "apple")]
    unsafe {
        __stderrp
    }
    #[cfg(not(target_vendor = "apple"))]
    unsafe {
        stderr
    }
}

// ----------------------------------------------------------------------
// Symbol anchors so cross-crate dead-code elimination can't strip the
// Rust definitions before the still-C call sites see them.
// ----------------------------------------------------------------------

#[used]
static _A_GET_HIGHLIGHT_PARAMS: unsafe extern "C" fn(*mut x3f_t, *mut highlight_params_t) =
    get_highlight_params;
#[used]
static _A_COMPUTE_CHROMA_PRIOR: unsafe extern "C" fn(*const f64, *mut f64) = compute_chroma_prior;
#[used]
static _A_RECONSTRUCT_HIGHLIGHTS: unsafe extern "C" fn(
    *mut f64,
    *const f64,
    *const highlight_params_t,
) = reconstruct_highlights;
#[used]
static _A_CHROMA_LUT_INIT: unsafe extern "C" fn(*mut chroma_lut_t) = chroma_lut_init_defaults;
#[used]
static _A_CHROMA_LUT_BUILD: unsafe extern "C" fn(
    *mut chroma_lut_t,
    *const x3f_area16_t,
    *const x3f_image_levels_t,
    *const f64,
) -> libc::c_int = chroma_lut_build_from_image;
#[used]
static _A_CHROMA_LUT_APPLY: unsafe extern "C" fn(
    *mut f64,
    *const chroma_lut_t,
    *mut chroma_lut_apply_stats_t,
) -> libc::c_int = chroma_lut_apply_pixel;
#[used]
static _A_CHROMA_LUT_APPLY_BMT: unsafe extern "C" fn(
    *mut f64,
    *const chroma_lut_t,
    *mut chroma_lut_apply_stats_t,
) -> libc::c_int = chroma_lut_apply_pixel_bmt;
#[used]
static _A_CHROMA_LUT_STATS: unsafe extern "C" fn(
    *const chroma_lut_apply_stats_t,
    *const libc::c_char,
) = chroma_lut_apply_stats_print;
#[used]
static _A_REPAIR_PIX_INIT: unsafe extern "C" fn(*mut repair_pix_t) = repair_pix_init_defaults;
#[used]
static _A_BUILD_SAT_MAP: unsafe extern "C" fn(
    *const x3f_area16_t,
    *const x3f_image_levels_t,
    f64,
) -> *mut u8 = build_sat_map;
#[used]
static _A_REPAIR_PIX_APPLY: unsafe extern "C" fn(
    *mut f64,
    *const f64,
    *const repair_pix_t,
    *const u8,
    libc::c_int,
    libc::c_int,
    libc::c_int,
    libc::c_int,
) = repair_pix_apply_pixel;

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{align_of, size_of};

    // Layout asserts: if the C source's typedef changes, the C-side
    // struct allocations passed to these Rust impls would silently
    // misalign. Pin the sizes here so any drift fails at test time.
    #[test]
    fn highlight_params_t_layout() {
        // 5 doubles, no padding.
        assert_eq!(size_of::<highlight_params_t>(), 5 * 8);
        assert_eq!(align_of::<highlight_params_t>(), 8);
    }

    #[test]
    fn chroma_lut_t_layout() {
        // Original layout: float[256] = 1024 bytes, then int (with 4
        // bytes pad to 8-align doubles), then 8 doubles = 1096. The M8
        // extension appends 2×float[256] (2048), 2×int (8), and 2
        // doubles (16) with no extra padding (1096 and 2048+8 are both
        // 8-multiples... 2048 + 8 = 2056, 8-aligned).
        assert_eq!(
            size_of::<chroma_lut_t>(),
            (1024 + 4 + 4 + 8 * 8) + 2 * 1024 + 2 * 4 + 2 * 8
        );
        assert_eq!(align_of::<chroma_lut_t>(), 8);
    }

    #[test]
    fn chroma_lut_apply_stats_t_layout() {
        // 6 u64 + 2x[u64; 256] = (6 + 512) * 8
        assert_eq!(size_of::<chroma_lut_apply_stats_t>(), (6 + 512) * 8);
        assert_eq!(align_of::<chroma_lut_apply_stats_t>(), 8);
    }

    #[test]
    fn repair_pix_t_layout() {
        // int (with 4 bytes pad) + 4 doubles
        assert_eq!(size_of::<repair_pix_t>(), 8 + 4 * 8);
        assert_eq!(align_of::<repair_pix_t>(), 8);
    }

    #[test]
    fn compute_chroma_prior_neutral_matrix_returns_ones() {
        // Identity-ish: matrix that maps (1,1,1) onto (1,1,1) — i.e.
        // any row-stochastic matrix where each row sums to 1. The
        // simplest such is the identity. inv(I) = I, dot with (1,1,1)
        // = (1,1,1), normalized → (1,1,1).
        let m = [1.0_f64, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let mut out = [0.0_f64; 3];
        unsafe { compute_chroma_prior(m.as_ptr(), out.as_mut_ptr()) };
        assert!((out[0] - 1.0).abs() < 1e-12);
        assert!((out[1] - 1.0).abs() < 1e-12);
        assert!((out[2] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn reconstruct_highlights_below_threshold_passthrough() {
        let hp = highlight_params_t {
            blending_low: 0.75,
            blending_high: 1.5,
            restore_thresh: 1.75,
            sat_factor: 1.0,
            chan_thresh: 0.5,
        };
        let mut s = [0.5_f64, 0.5, 0.5];
        let p = [1.0_f64, 1.0, 1.0];
        unsafe { reconstruct_highlights(s.as_mut_ptr(), p.as_ptr(), &hp) };
        assert_eq!(s, [0.5, 0.5, 0.5]);
    }

    /// Build a `chroma_lut_t` by hand with all three tables valid and a
    /// constant table value, so the per-identity apply paths can be
    /// exercised without an image.
    fn synth_lut(table_value: f32) -> chroma_lut_t {
        let mut lut: chroma_lut_t = unsafe { std::mem::zeroed() };
        unsafe { chroma_lut_init_defaults(&mut lut) };
        lut.lut = [table_value; CHROMA_LUT_BINS];
        lut.lut_b = [table_value; CHROMA_LUT_BINS];
        lut.lut_m = [table_value; CHROMA_LUT_BINS];
        lut.valid = 1;
        lut.valid_b = 1;
        lut.valid_m = 1;
        lut.neutral_tm = 1.0;
        lut.neutral_bm = 1.0;
        lut.neutral_mt = 1.0;
        lut
    }

    #[test]
    fn bmt_apply_repairs_clipped_t_like_original() {
        let lut = synth_lut(1.5);
        // T clipped (1.0), B/M healthy donors well below asymmetric_max.
        let mut s = [0.5_f64, 0.5, 1.0];
        let mut s2 = s;
        let r1 = unsafe { chroma_lut_apply_pixel(s.as_mut_ptr(), &lut, ptr::null_mut()) };
        let r2 = unsafe { chroma_lut_apply_pixel_bmt(s2.as_mut_ptr(), &lut, ptr::null_mut()) };
        assert_eq!(r1, 1);
        assert_eq!(r2, 1);
        assert_eq!(s, s2, "T identity must match the original pass exactly");
        // T' = M * lut = 0.5 * 1.5 at full strength.
        assert!((s[2] - 0.75).abs() < 1e-9, "T' = {}", s[2]);
    }

    #[test]
    fn bmt_apply_repairs_clipped_b_from_m_and_t() {
        let lut = synth_lut(1.5);
        // B clipped, M/T healthy donors. The original pass returns 0
        // (T not in clip zone); the generalized pass must repair B.
        let mut s_orig = [1.0_f64, 0.5, 0.5];
        let mut s = s_orig;
        let r_old = unsafe { chroma_lut_apply_pixel(s_orig.as_mut_ptr(), &lut, ptr::null_mut()) };
        assert_eq!(r_old, 0);
        let r = unsafe { chroma_lut_apply_pixel_bmt(s.as_mut_ptr(), &lut, ptr::null_mut()) };
        assert_eq!(r, 1);
        // B' = M * 1.5 at full strength (B at sat ⇒ x_strength = 1;
        // donors at 0.5 ⇒ asym undamped) = 0.75 — chroma kept instead
        // of a neutral snap.
        assert!((s[0] - 0.75).abs() < 1e-9, "B' = {}", s[0]);
        assert_eq!(s[1], 0.5);
        assert_eq!(s[2], 0.5);
    }

    #[test]
    fn bmt_apply_repairs_clipped_m_from_t_amplitude() {
        let lut = synth_lut(1.4);
        // M clipped, B/T healthy. M' = T * lut_m = 0.5 * 1.4 = 0.7.
        let mut s = [0.5_f64, 1.0, 0.5];
        let r = unsafe { chroma_lut_apply_pixel_bmt(s.as_mut_ptr(), &lut, ptr::null_mut()) };
        assert_eq!(r, 1);
        // Tolerance covers the f32 table storage (1.4 is not exact).
        assert!((s[1] - 0.7).abs() < 1e-6, "M' = {}", s[1]);
        assert_eq!(s[0], 0.5);
        assert_eq!(s[2], 0.5);
    }

    #[test]
    fn bmt_apply_kills_when_donors_clipped() {
        let lut = synth_lut(1.5);
        // B clipped AND T clipped: B identity has a clipped donor (T),
        // M identity has a clipped donor (B) — no repair, caller falls
        // back to reconstruct_highlights.
        let mut s = [1.0_f64, 0.5, 1.0];
        // T identity also dies (donor B clipped → bm_clip kill).
        let r = unsafe { chroma_lut_apply_pixel_bmt(s.as_mut_ptr(), &lut, ptr::null_mut()) };
        assert_eq!(r, 0);
        assert_eq!(s, [1.0, 0.5, 1.0]);
    }

    #[test]
    fn bmt_apply_respects_bin_evidence_kill() {
        // Table ≈ neutral ⇒ no chroma evidence ⇒ no repair.
        let lut = synth_lut(1.01);
        let mut s = [1.0_f64, 0.5, 0.5];
        let r = unsafe { chroma_lut_apply_pixel_bmt(s.as_mut_ptr(), &lut, ptr::null_mut()) };
        assert_eq!(r, 0);
    }

    #[test]
    fn reconstruct_highlights_blowout_snaps_to_l_p() {
        let hp = highlight_params_t {
            blending_low: 0.75,
            blending_high: 1.5,
            restore_thresh: 1.75,
            sat_factor: 1.0,
            chan_thresh: 0.5,
        };
        let mut s = [2.0_f64, 2.0, 2.0]; // u_max = 2.0 > restore_thresh
        let p = [1.0_f64, 1.0, 1.0];
        unsafe { reconstruct_highlights(s.as_mut_ptr(), p.as_ptr(), &hp) };
        // L = u_max * sat_factor = 2.0; output = L * p = (2.0, 2.0, 2.0).
        assert_eq!(s, [2.0, 2.0, 2.0]);
    }
}
