//! M6d — native Rust port of `src/x3f_spatial_gain.c`.
//!
//! Per-pixel "spatial gain" (lens shading) correction tables. The Sigma
//! cameras encode a 4D family of correction tables in CAMF: per-aperture,
//! per-focus-distance entries. We pick the four nearest neighbours in
//! (1/aperture, lens-position) space, build bilinear interpolation
//! weights, and either stash the raw tables for `x3f_calc_spatial_gain`
//! to combine on-the-fly, or pre-interpolate them into one dense table
//! per channel.
//!
//! Quattro HP: six channels (R, G, B0..B3) where B0..B3 are 2×2
//! sub-sampled with row/col offsets.
//!
//! Memory ownership: when `corr->malloc != 0`, `corr->gain` is owned by
//! `libc::malloc` (paired with `x3f_cleanup_spatial_gain` →
//! `libc::free`). When `malloc == 0`, `corr->gain` aliases the CAMF
//! payload and the cleanup is a no-op.
//!
//! The `alloca`-backed linked list of candidate gains in the C source is
//! replaced by a `Vec<MerrillSpatialGain>`; iteration order is preserved
//! by walking the Vec in reverse so any `<`-tiebreaks resolve identically
//! to the C version.
#![allow(clippy::missing_safety_doc)]

use std::ffi::CStr;
use std::mem;
use std::ptr;

use crate::*;
// libc compat — see `sysabi.rs`.
use crate::sysabi as libc;

#[derive(Clone)]
struct MerrillSpatialGain {
    name: *mut libc::c_char,
    x: f64,
    y: f64,
}

#[inline]
fn lens_position(focal_length: f64, object_distance: f64) -> f64 {
    1.0 / (1.0 / focal_length - 1.0 / object_distance)
}

unsafe fn get_focal_length(x3f: *mut x3f_t) -> f64 {
    let mut flength: *mut libc::c_char = ptr::null_mut();
    if unsafe { x3f_get_prop_entry(x3f, c"FLENGTH".as_ptr() as *mut _, &mut flength) } != 0 {
        unsafe { libc::atof(flength) }
    } else {
        let focal_length = 30.0_f64;
        unsafe {
            x3f_printf(
                x3f_verbosity_t_WARN,
                c"Could not get focal length, assuming %g mm\n".as_ptr(),
                focal_length,
            );
        }
        focal_length
    }
}

unsafe fn get_object_distance(x3f: *mut x3f_t) -> f64 {
    let mut object_distance: f64 = 0.0;
    if unsafe {
        x3f_get_camf_float(
            x3f,
            c"ObjectDistance".as_ptr() as *mut _,
            &mut object_distance,
        )
    } != 0
    {
        // Convert cm to mm.
        object_distance * 10.0
    } else {
        let object_distance = f64::INFINITY;
        unsafe {
            x3f_printf(
                x3f_verbosity_t_WARN,
                c"Could not get object distance, assuming %g mm\n".as_ptr(),
                object_distance,
            );
        }
        object_distance
    }
}

unsafe fn get_mod(x3f: *mut x3f_t) -> f64 {
    let mut lens_information: i32 = 0;
    if unsafe {
        x3f_get_camf_signed(
            x3f,
            c"LensInformation".as_ptr() as *mut _,
            &mut lens_information,
        )
    } == 0
    {
        lens_information = -1;
    }
    match lens_information {
        1003 => 200.0, // DP1 Merrill
        1004 => 280.0, // DP2 Merrill
        1005 => 226.0, // DP3 Merrill
        _ => {
            let mod_ = 280.0_f64;
            unsafe {
                x3f_printf(
                    x3f_verbosity_t_WARN,
                    c"Could not get MOD, assuming %g mm\n".as_ptr(),
                    mod_,
                );
            }
            mod_
        }
    }
}

/// Read `GainsTable<chan>` / `MinGains<chan>` / `Delta<chan>` for a
/// named CAMF block. On success, populates `*mgain`, asserts row/col
/// dimensions match across calls (or fills them on first call), and
/// fills `mingain` / `delta`. Returns 1 on success, 0 on failure.
unsafe fn get_merrill_type_gains_table(
    x3f: *mut x3f_t,
    name: *mut libc::c_char,
    chan: &CStr,
    mgain: *mut *mut u32,
    rows: *mut libc::c_int,
    cols: *mut libc::c_int,
    mingain: *mut f64,
    delta: *mut f64,
) -> libc::c_int {
    let mut buf = [0u8; 32];
    let mut val: *mut libc::c_char = ptr::null_mut();

    // GainsTable<chan>
    let n = format_into(&mut buf, "GainsTable", chan.to_bytes());
    let key = unsafe { CStr::from_bytes_with_nul_unchecked(&buf[..n + 1]) };
    let mut rows_tmp: libc::c_int = 0;
    let mut cols_tmp: libc::c_int = 0;
    let ok = unsafe {
        x3f_get_camf_property(x3f, name, key.as_ptr() as *mut _, &mut val) != 0
            && x3f_get_camf_matrix_var(
                x3f,
                val,
                &mut rows_tmp,
                &mut cols_tmp,
                ptr::null_mut(),
                matrix_type_t_M_UINT,
                mgain as *mut *mut libc::c_void,
            ) != 0
    };
    if !ok {
        return 0;
    }
    unsafe {
        if (*rows != -1 && *rows != rows_tmp) || (*cols != -1 && *cols != cols_tmp) {
            return 0;
        }
        *rows = rows_tmp;
        *cols = cols_tmp;
    }

    // MinGains<chan>
    let n = format_into(&mut buf, "MinGains", chan.to_bytes());
    let key = unsafe { CStr::from_bytes_with_nul_unchecked(&buf[..n + 1]) };
    if unsafe { x3f_get_camf_property(x3f, name, key.as_ptr() as *mut _, &mut val) } == 0 {
        return 0;
    }
    unsafe {
        *mingain = libc::atof(val);
    }

    // Delta<chan>
    let n = format_into(&mut buf, "Delta", chan.to_bytes());
    let key = unsafe { CStr::from_bytes_with_nul_unchecked(&buf[..n + 1]) };
    if unsafe { x3f_get_camf_property(x3f, name, key.as_ptr() as *mut _, &mut val) } == 0 {
        return 0;
    }
    unsafe {
        *delta = libc::atof(val);
    }

    1
}

/// Concatenate `prefix` + `suffix` into `buf` and NUL-terminate.
/// Panics if the result would overflow `buf`.
fn format_into(buf: &mut [u8; 32], prefix: &str, suffix: &[u8]) -> usize {
    let pb = prefix.as_bytes();
    let n = pb.len() + suffix.len();
    assert!(n + 1 <= buf.len());
    buf[..pb.len()].copy_from_slice(pb);
    buf[pb.len()..n].copy_from_slice(suffix);
    buf[n] = 0;
    n
}

/// Parse `"SpatialGainHPProps_<int><EOF>"`. Returns `Some(int)` if the
/// pattern matches with no trailing characters; `None` otherwise.
fn parse_hp_props_index(s: &CStr) -> Option<u32> {
    let bytes = s.to_bytes();
    let prefix = b"SpatialGainHPProps_";
    let rest = bytes.strip_prefix(prefix.as_slice())?;
    if rest.is_empty() {
        return None;
    }
    let s = std::str::from_utf8(rest).ok()?;
    s.parse::<u32>().ok()
}

/// Parse `"SpatialGainsProps_<int>_<3char><EOF>"`. The C version uses
/// `sscanf("SpatialGainsProps_%d_%3s%c", &idx, focus, &dummy) == 2` —
/// 2 means: int + 3-char string matched, dummy did NOT match (i.e. no
/// trailing chars). Returns `Some((idx, [u8; 3]))` on match.
fn parse_focus_props(s: &CStr) -> Option<(u32, [u8; 3])> {
    let bytes = s.to_bytes();
    let prefix = b"SpatialGainsProps_";
    let rest = bytes.strip_prefix(prefix.as_slice())?;
    let underscore = rest.iter().position(|&b| b == b'_')?;
    let int_part = std::str::from_utf8(&rest[..underscore]).ok()?;
    let idx: u32 = int_part.parse().ok()?;
    let after = &rest[underscore + 1..];
    if after.len() != 3 {
        return None;
    }
    let mut focus = [0u8; 3];
    focus.copy_from_slice(after);
    Some((idx, focus))
}

/// Parse `"SpatialGainsProps_<dbl>_<dbl><EOF>"`. The C version uses
/// `sscanf("SpatialGainsProps_%lf_%lf%c", &a, &b, &dummy) == 2`.
/// Returns `Some((aperture, lenspos))`.
fn parse_aperture_lenspos_props(s: &CStr) -> Option<(f64, f64)> {
    let bytes = s.to_bytes();
    let prefix = b"SpatialGainsProps_";
    let rest = bytes.strip_prefix(prefix.as_slice())?;
    // sscanf "%lf" in C will greedily consume digits, optional sign,
    // decimal point, and exponent. We delegate to f64::from_str on the
    // "longest match" boundary: locate the underscore between the two
    // numbers.
    let underscore = rest.iter().position(|&b| b == b'_')?;
    let a_str = std::str::from_utf8(&rest[..underscore]).ok()?;
    let b_str = std::str::from_utf8(&rest[underscore + 1..]).ok()?;
    let a: f64 = a_str.parse().ok()?;
    let b: f64 = b_str.parse().ok()?;
    Some((a, b))
}

#[no_mangle]
pub unsafe extern "C" fn x3f_get_merrill_type_spatial_gain(
    x3f: *mut x3f_t,
    hp_flag: libc::c_int,
    corr: *mut x3f_spatial_gain_corr_t,
) -> libc::c_int {
    let mut capture_aperture: f64 = 0.0;
    if unsafe {
        x3f_get_camf_float(
            x3f,
            c"CaptureAperture".as_ptr() as *mut _,
            &mut capture_aperture,
        )
    } == 0
    {
        return 0;
    }

    let mut include_blocks: *mut *mut libc::c_char = ptr::null_mut();
    let mut include_blocks_val: *mut *mut libc::c_char = ptr::null_mut();
    let mut include_blocks_num: u32 = 0;
    if unsafe {
        x3f_get_camf_property_list(
            x3f,
            c"IncludeBlocks".as_ptr() as *mut _,
            &mut include_blocks,
            &mut include_blocks_val,
            &mut include_blocks_num,
        )
    } == 0
    {
        return 0;
    }

    let mut spatial_gain_fstop: *mut f64 = ptr::null_mut();
    let mut num_fstop: libc::c_int = 0;
    let mut corr_num: libc::c_int = 3;
    let mut gains: Vec<MerrillSpatialGain> = Vec::new();
    let x;
    let y;

    if hp_flag != 0 {
        // Quattro HP — 6 channels (R, G, B0, B1, B2, B3).
        let ok = unsafe {
            x3f_get_camf_matrix_var(
                x3f,
                c"SpatialGainHP_Fstop".as_ptr() as *mut _,
                &mut num_fstop,
                ptr::null_mut(),
                ptr::null_mut(),
                matrix_type_t_M_FLOAT,
                &mut spatial_gain_fstop as *mut *mut f64 as *mut *mut libc::c_void,
            )
        };
        if ok == 0 {
            return 0;
        }
        corr_num = 6;

        for i in 0..include_blocks_num {
            let block = unsafe { *include_blocks.offset(i as isize) };
            let block_cstr = unsafe { CStr::from_ptr(block) };
            let Some(aperture_index) = parse_hp_props_index(block_cstr) else {
                continue;
            };
            let mut names: *mut *mut libc::c_char = ptr::null_mut();
            let mut values: *mut *mut libc::c_char = ptr::null_mut();
            let mut num: u32 = 0;
            let pl_ok = unsafe {
                x3f_get_camf_property_list(x3f, block, &mut names, &mut values, &mut num)
            };
            if pl_ok != 0 && (aperture_index as libc::c_int) < num_fstop {
                let fstop_value = unsafe { *spatial_gain_fstop.offset(aperture_index as isize) };
                gains.push(MerrillSpatialGain {
                    name: block,
                    x: 1.0 / fstop_value,
                    y: 0.0,
                });
            }
        }

        x = 1.0 / capture_aperture;
        y = 0.0;
    } else {
        // Merrill or non-HP Quattro.
        let ok = unsafe {
            x3f_get_camf_matrix_var(
                x3f,
                c"SpatialGain_Fstop".as_ptr() as *mut _,
                &mut num_fstop,
                ptr::null_mut(),
                ptr::null_mut(),
                matrix_type_t_M_FLOAT,
                &mut spatial_gain_fstop as *mut *mut f64 as *mut *mut libc::c_void,
            )
        };
        if ok != 0 {
            for i in 0..include_blocks_num {
                let block = unsafe { *include_blocks.offset(i as isize) };
                let block_cstr = unsafe { CStr::from_ptr(block) };
                let Some((aperture_index, focus)) = parse_focus_props(block_cstr) else {
                    continue;
                };
                let mut names: *mut *mut libc::c_char = ptr::null_mut();
                let mut values: *mut *mut libc::c_char = ptr::null_mut();
                let mut num: u32 = 0;
                let pl_ok = unsafe {
                    x3f_get_camf_property_list(x3f, block, &mut names, &mut values, &mut num)
                };
                if pl_ok == 0 || (aperture_index as libc::c_int) >= num_fstop {
                    continue;
                }
                let lenspos = if &focus == b"INF" {
                    lens_position(unsafe { get_focal_length(x3f) }, f64::INFINITY)
                } else if &focus == b"MOD" {
                    lens_position(unsafe { get_focal_length(x3f) }, unsafe { get_mod(x3f) })
                } else {
                    continue;
                };
                let fstop_value = unsafe { *spatial_gain_fstop.offset(aperture_index as isize) };
                gains.push(MerrillSpatialGain {
                    name: block,
                    x: 1.0 / fstop_value,
                    y: lenspos,
                });
            }
        } else {
            for i in 0..include_blocks_num {
                let block = unsafe { *include_blocks.offset(i as isize) };
                let block_cstr = unsafe { CStr::from_ptr(block) };
                let Some((aperture, lenspos)) = parse_aperture_lenspos_props(block_cstr) else {
                    continue;
                };
                let mut names: *mut *mut libc::c_char = ptr::null_mut();
                let mut values: *mut *mut libc::c_char = ptr::null_mut();
                let mut num: u32 = 0;
                let pl_ok = unsafe {
                    x3f_get_camf_property_list(x3f, block, &mut names, &mut values, &mut num)
                };
                if pl_ok == 0 {
                    continue;
                }
                gains.push(MerrillSpatialGain {
                    name: block,
                    x: 1.0 / aperture,
                    y: lenspos,
                });
            }
        }

        x = 1.0 / capture_aperture;
        y = lens_position(unsafe { get_focal_length(x3f) }, unsafe {
            get_object_distance(x3f)
        });
    }

    // Find the closest neighbour in each of the 4 quadrants. C
    // traverses a linked list built by prepending, so iteration is
    // last-pushed-first. Mirror that with `.iter().rev()`.
    let mut q_closest: [Option<&MerrillSpatialGain>; 4] = [None; 4];
    let mut q_closest_dx = [
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
        f64::INFINITY,
    ];
    let mut q_closest_dy = [
        f64::INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
    ];
    let mut q_closest_d2 = [f64::INFINITY; 4];

    for g in gains.iter().rev() {
        let dx = g.x - x;
        let dy = g.y - y;
        let d2 = dx * dx + dy * dy;
        let q = if dx > 0.0 && dy > 0.0 {
            0
        } else if dx > 0.0 {
            3
        } else if dy > 0.0 {
            1
        } else {
            2
        };
        if d2 < q_closest_d2[q] {
            q_closest[q] = Some(g);
            q_closest_dx[q] = dx;
            q_closest_dy[q] = dy;
            q_closest_d2[q] = d2;
        }
    }

    let mut q_weight_x = [0.0_f64; 4];
    let mut q_weight_y = [0.0_f64; 4];
    let mut q_weight = [0.0_f64; 4];

    q_weight_x[0] = q_closest_dx[1] / (q_closest_dx[1] - q_closest_dx[0]);
    q_weight_x[1] = q_closest_dx[0] / (q_closest_dx[0] - q_closest_dx[1]);
    q_weight_x[2] = q_closest_dx[3] / (q_closest_dx[3] - q_closest_dx[2]);
    q_weight_x[3] = q_closest_dx[2] / (q_closest_dx[2] - q_closest_dx[3]);

    q_weight_y[0] = q_closest_dy[3] / (q_closest_dy[3] - q_closest_dy[0]);
    q_weight_y[1] = q_closest_dy[2] / (q_closest_dy[2] - q_closest_dy[1]);
    q_weight_y[2] = q_closest_dy[1] / (q_closest_dy[1] - q_closest_dy[2]);
    q_weight_y[3] = q_closest_dy[0] / (q_closest_dy[0] - q_closest_dy[3]);

    unsafe {
        x3f_printf(x3f_verbosity_t_DEBUG, c"x = %f y = %f\n".as_ptr(), x, y);
        for i in 0..4 {
            if let Some(g) = q_closest[i] {
                x3f_printf(
                    x3f_verbosity_t_DEBUG,
                    c"q = %d name = %s x = %f y = %f\n".as_ptr(),
                    i as libc::c_int,
                    g.name,
                    g.x,
                    g.y,
                );
            } else {
                x3f_printf(
                    x3f_verbosity_t_DEBUG,
                    c"q = %d name = NULL\n".as_ptr(),
                    i as libc::c_int,
                );
            }
            x3f_printf(
                x3f_verbosity_t_DEBUG,
                c"q = %d dx = %f dy = %f d2 = %f wx = %f wy = %f\n".as_ptr(),
                i as libc::c_int,
                q_closest_dx[i],
                q_closest_dy[i],
                q_closest_d2[i],
                q_weight_x[i],
                q_weight_y[i],
            );
        }
    }

    for i in 0..4 {
        if q_weight_x[i].is_nan() {
            q_weight_x[i] = 1.0;
        }
        if q_weight_y[i].is_nan() {
            q_weight_y[i] = 1.0;
        }
        q_weight[i] = q_weight_x[i] * q_weight_y[i];
    }

    unsafe {
        for i in 0..4 {
            let name = match q_closest[i] {
                Some(g) => g.name,
                None => c"NULL".as_ptr() as *mut _,
            };
            x3f_printf(
                x3f_verbosity_t_DEBUG,
                c"q = %d name = %s w = %f\n".as_ptr(),
                i as libc::c_int,
                name,
                q_weight[i],
            );
        }
    }

    unsafe {
        for i in 0..corr_num as usize {
            let c = corr.add(i);
            (*c).gain = ptr::null_mut();
            (*c).malloc = 0;
            (*c).rows = -1;
            (*c).cols = -1;
            (*c).rowoff = 0;
            (*c).coloff = 0;
            (*c).rowpitch = 1;
            (*c).colpitch = 1;
            (*c).chan = i as libc::c_int;
            (*c).channels = 1;
            (*c).mgain_num = 0;
        }
    }

    if hp_flag != 0 {
        unsafe {
            (*corr.add(2)).rowoff = 0;
            (*corr.add(2)).coloff = 0;
            (*corr.add(2)).rowpitch = 2;
            (*corr.add(2)).colpitch = 2;
            (*corr.add(2)).chan = 2;

            (*corr.add(3)).rowoff = 0;
            (*corr.add(3)).coloff = 1;
            (*corr.add(3)).rowpitch = 2;
            (*corr.add(3)).colpitch = 2;
            (*corr.add(3)).chan = 2;

            (*corr.add(4)).rowoff = 1;
            (*corr.add(4)).coloff = 0;
            (*corr.add(4)).rowpitch = 2;
            (*corr.add(4)).colpitch = 2;
            (*corr.add(4)).chan = 2;

            (*corr.add(5)).rowoff = 1;
            (*corr.add(5)).coloff = 1;
            (*corr.add(5)).rowpitch = 2;
            (*corr.add(5)).colpitch = 2;
            (*corr.add(5)).chan = 2;
        }
    }

    let channels_normal: [&CStr; 6] = [c"R", c"G", c"B", c"", c"", c""];
    let channels_hp: [&CStr; 6] = [c"R", c"G", c"B0", c"B1", c"B2", c"B3"];
    let channels: &[&CStr; 6] = if hp_flag != 0 {
        &channels_hp
    } else {
        &channels_normal
    };

    for i in 0..4 {
        let Some(g) = q_closest[i] else {
            continue;
        };
        for j in 0..corr_num as usize {
            unsafe {
                let c = corr.add(j);
                let m_idx = (*c).mgain_num as usize;
                (*c).mgain_num += 1;
                let m = (*c).mgain.as_mut_ptr().add(m_idx);
                (*m).weight = q_weight[i];
                let ok = get_merrill_type_gains_table(
                    x3f,
                    g.name,
                    channels[j],
                    &mut (*m).gain,
                    &mut (*c).rows,
                    &mut (*c).cols,
                    &mut (*m).mingain,
                    &mut (*m).delta,
                );
                if ok == 0 {
                    return 0;
                }
            }
        }
    }

    unsafe {
        for i in 0..corr_num as usize {
            if (*corr.add(i)).mgain_num == 0 {
                return 0;
            }
        }
    }

    corr_num
}

#[no_mangle]
pub unsafe extern "C" fn x3f_get_interp_merrill_type_spatial_gain(
    x3f: *mut x3f_t,
    hp_flag: libc::c_int,
    corr: *mut x3f_spatial_gain_corr_t,
) -> libc::c_int {
    let corr_num = unsafe { x3f_get_merrill_type_spatial_gain(x3f, hp_flag, corr) };

    for i in 0..corr_num as usize {
        unsafe {
            let c = corr.add(i);
            let num = ((*c).rows * (*c).cols * (*c).channels) as usize;
            (*c).gain = libc::malloc(num * mem::size_of::<f64>()) as *mut f64;
            (*c).malloc = 1;

            for j in 0..num {
                let mut sum = 0.0_f64;
                for g in 0..(*c).mgain_num as usize {
                    let m = (*c).mgain.as_ptr().add(g);
                    sum += (*m).weight * ((*m).mingain + (*m).delta * (*(*m).gain.add(j)) as f64);
                }
                *(*c).gain.add(j) = sum;
            }
        }
    }

    corr_num
}

#[no_mangle]
pub unsafe extern "C" fn x3f_get_classic_spatial_gain(
    x3f: *mut x3f_t,
    wb: *mut libc::c_char,
    corr: *mut x3f_spatial_gain_corr_t,
) -> libc::c_int {
    let mut gain_name: *mut libc::c_char = ptr::null_mut();
    let prop_ok = unsafe {
        x3f_get_camf_property(
            x3f,
            c"SpatialGainTables".as_ptr() as *mut _,
            wb,
            &mut gain_name,
        ) != 0
            && x3f_get_camf_matrix_var(
                x3f,
                gain_name,
                &mut (*corr).rows,
                &mut (*corr).cols,
                &mut (*corr).channels,
                matrix_type_t_M_FLOAT,
                &mut (*corr).gain as *mut *mut f64 as *mut *mut libc::c_void,
            ) != 0
    };

    let fallback_ok = if prop_ok {
        true
    } else {
        unsafe {
            x3f_get_camf_matrix_var(
                x3f,
                c"SpatialGain".as_ptr() as *mut _,
                &mut (*corr).rows,
                &mut (*corr).cols,
                &mut (*corr).channels,
                matrix_type_t_M_FLOAT,
                &mut (*corr).gain as *mut *mut f64 as *mut *mut libc::c_void,
            ) != 0
        }
    };

    if !prop_ok && !fallback_ok {
        return 0;
    }

    unsafe {
        (*corr).malloc = 0;
        (*corr).rowoff = 0;
        (*corr).coloff = 0;
        (*corr).rowpitch = 1;
        (*corr).colpitch = 1;
        (*corr).chan = 0;
        (*corr).mgain_num = 0;
    }
    1
}

#[no_mangle]
pub unsafe extern "C" fn x3f_get_spatial_gain(
    x3f: *mut x3f_t,
    wb: *mut libc::c_char,
    corr: *mut x3f_spatial_gain_corr_t,
) -> libc::c_int {
    let mut corr_num = 0;
    corr_num +=
        unsafe { x3f_get_interp_merrill_type_spatial_gain(x3f, 0, corr.add(corr_num as usize)) };
    if corr_num == 0 {
        corr_num += unsafe { x3f_get_classic_spatial_gain(x3f, wb, corr.add(corr_num as usize)) };
    }
    corr_num
}

#[no_mangle]
pub unsafe extern "C" fn x3f_cleanup_spatial_gain(
    corr: *mut x3f_spatial_gain_corr_t,
    corr_num: libc::c_int,
) {
    for i in 0..corr_num as usize {
        unsafe {
            let c = corr.add(i);
            if (*c).malloc != 0 {
                libc::free((*c).gain as *mut libc::c_void);
            }
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn x3f_calc_spatial_gain(
    corr: *mut x3f_spatial_gain_corr_t,
    corr_num: libc::c_int,
    row: libc::c_int,
    col: libc::c_int,
    chan: libc::c_int,
    rows: libc::c_int,
    cols: libc::c_int,
) -> f64 {
    let mut gain = 1.0_f64;
    let rrel = row as f64 / rows as f64;
    let crel = col as f64 / cols as f64;

    for i in 0..corr_num as usize {
        let c = unsafe { &mut *corr.add(i) };
        let ch = chan - c.chan;
        if ch < 0 || ch >= c.channels {
            continue;
        }
        if row % c.rowpitch != c.rowoff {
            continue;
        }
        if col % c.colpitch != c.coloff {
            continue;
        }

        let rc = rrel * (c.rows - 1) as f64;
        let ri = rc.floor() as libc::c_int;
        let rf = rc - ri as f64;

        let cc = crel * (c.cols - 1) as f64;
        let ci = cc.floor() as libc::c_int;
        let cf = cc - ci as f64;

        let r1: *const f64;
        let r2: *const f64;
        unsafe {
            if ri < 0 {
                r1 = c.gain;
                r2 = c.gain;
            } else if ri >= c.rows - 1 {
                let off = ((c.rows - 1) * c.cols * c.channels) as isize;
                r1 = c.gain.offset(off);
                r2 = c.gain.offset(off);
            } else {
                r1 = c.gain.offset((ri * c.cols * c.channels) as isize);
                r2 = c.gain.offset(((ri + 1) * c.cols * c.channels) as isize);
            }
        }

        // NB: the C source has a known bug — the first branch of the
        // ci<0 check uses an `if` (not `else if`) for `ci>=c->cols-1`,
        // so when ci<0 we fall through and overwrite co1/co2 from the
        // (c->cols-1) branch. We preserve the bug verbatim for byte-
        // identity with the legacy converter. See src/x3f_spatial_gain.c
        // line 457-460 in the original.
        let co1: libc::c_int;
        let co2: libc::c_int;
        if ci < 0 {
            // C wrote `co1 = co2 = ch;` here, but then the next `if`
            // (not `else if`) immediately reassigns. Mimic that.
            let _co1_intermediate = ch;
            let _co2_intermediate = ch;
            if ci >= c.cols - 1 {
                // Unreachable when ci<0 and c.cols>=1, but emit the
                // assignment for fidelity.
                co1 = (c.cols - 1) * c.channels + ch;
                co2 = (c.cols - 1) * c.channels + ch;
            } else {
                co1 = ci * c.channels + ch;
                co2 = (ci + 1) * c.channels + ch;
            }
        } else if ci >= c.cols - 1 {
            co1 = (c.cols - 1) * c.channels + ch;
            co2 = (c.cols - 1) * c.channels + ch;
        } else {
            co1 = ci * c.channels + ch;
            co2 = (ci + 1) * c.channels + ch;
        }

        let r1c1 = unsafe { *r1.offset(co1 as isize) };
        let r1c2 = unsafe { *r1.offset(co2 as isize) };
        let r2c1 = unsafe { *r2.offset(co1 as isize) };
        let r2c2 = unsafe { *r2.offset(co2 as isize) };

        let gr1 = r1c1 + cf * (r1c2 - r1c1);
        let gr2 = r2c1 + cf * (r2c2 - r2c1);

        gain *= gr1 + rf * (gr2 - gr1);
    }
    gain
}

// Symbol anchors so cross-crate dead-code elimination can't strip the
// Rust definitions before the still-C call sites in x3f_process.c link.
#[used]
static _A_GET_MERRILL: unsafe extern "C" fn(
    *mut x3f_t,
    libc::c_int,
    *mut x3f_spatial_gain_corr_t,
) -> libc::c_int = x3f_get_merrill_type_spatial_gain;
#[used]
static _A_GET_INTERP_MERRILL: unsafe extern "C" fn(
    *mut x3f_t,
    libc::c_int,
    *mut x3f_spatial_gain_corr_t,
) -> libc::c_int = x3f_get_interp_merrill_type_spatial_gain;
#[used]
static _A_GET_CLASSIC: unsafe extern "C" fn(
    *mut x3f_t,
    *mut libc::c_char,
    *mut x3f_spatial_gain_corr_t,
) -> libc::c_int = x3f_get_classic_spatial_gain;
#[used]
static _A_GET_SGAIN: unsafe extern "C" fn(
    *mut x3f_t,
    *mut libc::c_char,
    *mut x3f_spatial_gain_corr_t,
) -> libc::c_int = x3f_get_spatial_gain;
#[used]
static _A_CLEANUP_SGAIN: unsafe extern "C" fn(*mut x3f_spatial_gain_corr_t, libc::c_int) =
    x3f_cleanup_spatial_gain;
#[used]
static _A_CALC_SGAIN: unsafe extern "C" fn(
    *mut x3f_spatial_gain_corr_t,
    libc::c_int,
    libc::c_int,
    libc::c_int,
    libc::c_int,
    libc::c_int,
    libc::c_int,
) -> f64 = x3f_calc_spatial_gain;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lens_position_at_infinity_returns_focal_length() {
        // 1/(1/30 - 1/inf) = 30
        assert!((lens_position(30.0, f64::INFINITY) - 30.0).abs() < 1e-9);
    }

    #[test]
    fn parse_hp_props_index_works() {
        assert_eq!(parse_hp_props_index(c"SpatialGainHPProps_3"), Some(3));
        assert_eq!(parse_hp_props_index(c"SpatialGainHPProps_12"), Some(12));
        // Trailing garbage rejected by parse::<u32>.
        assert_eq!(parse_hp_props_index(c"SpatialGainHPProps_3x"), None);
        assert_eq!(parse_hp_props_index(c"SomethingElse_3"), None);
    }

    #[test]
    fn parse_focus_props_works() {
        assert_eq!(
            parse_focus_props(c"SpatialGainsProps_4_INF"),
            Some((4, *b"INF"))
        );
        assert_eq!(
            parse_focus_props(c"SpatialGainsProps_2_MOD"),
            Some((2, *b"MOD"))
        );
        // Trailing chars after 3-char focus → reject.
        assert_eq!(parse_focus_props(c"SpatialGainsProps_4_INFX"), None);
        // Wrong shape.
        assert_eq!(parse_focus_props(c"SpatialGainsProps_INF"), None);
    }

    #[test]
    fn parse_aperture_lenspos_props_works() {
        assert_eq!(
            parse_aperture_lenspos_props(c"SpatialGainsProps_4.0_30.5"),
            Some((4.0, 30.5))
        );
        // C99 sscanf %lf delegates to strtod, which parses "INF" as
        // f64::INFINITY. f64::from_str matches that behaviour, so
        // this case does match — even though in practice the older-
        // CAMF code path that consumes lenspos values from the block
        // name never sees "INF" in the second slot (the INF/MOD
        // tokens only appear in the newer SpatialGain_Fstop path,
        // which uses parse_focus_props instead).
        let parsed = parse_aperture_lenspos_props(c"SpatialGainsProps_2.8_INF").unwrap();
        assert!((parsed.0 - 2.8).abs() < 1e-12);
        assert!(parsed.1.is_infinite() && parsed.1 > 0.0);
        // Garbled input rejected.
        assert!(parse_aperture_lenspos_props(c"SpatialGainsProps_2.8x_INF").is_none());
    }
}
