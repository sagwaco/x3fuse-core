//! M6b — native Rust port of `src/x3f_histogram.c`.
//!
//! `x3f_dump_raw_data_as_histogram` walks the processed RAW image and
//! emits a per-channel histogram CSV. Pure debug surface — driven by the
//! `-histogram` / `-loghist` CLI flags. The processed image still comes
//! from `x3f_get_image` (still C in `x3f_process.c` until M6e), so this
//! port is just the pixel walk + I/O.
//!
//! Output is byte-for-byte identical to the C version: we keep the
//! `%5d , %6d , %6d , %6d\n` and `%5d, %5d , %6d , %6d , %6d\n` format
//! strings and write through `libc::fprintf` so f64 / Display drift
//! doesn't matter.
//!
//! Memory ownership: `x3f_get_image` allocates `image.buf` via
//! `libc::malloc` (in `src/x3f_image.c::cleanup_one_channel` → through
//! the C output_writers' allocators). We release it with `libc::free`
//! exactly like the C source did.
#![allow(clippy::missing_safety_doc)]

use std::mem;
use std::ptr;

use crate::*;
// libc compat — see `sysabi.rs`.
use crate::sysabi as libc;

const BASE: f64 = 2.0;
const STEPS: i32 = 10;

#[inline]
fn ilog(i: i32) -> i32 {
    if i <= 0 {
        0
    } else {
        let log = (i as f64).log10() / BASE.log10();
        (STEPS as f64 * log) as i32
    }
}

#[inline]
fn ilog_inv(i: i32) -> i32 {
    BASE.powf(i as f64 / STEPS as f64).round() as i32
}

#[no_mangle]
pub unsafe extern "C" fn x3f_dump_raw_data_as_histogram(
    x3f: *mut x3f_t,
    outfilename: *mut libc::c_char,
    encoding: x3f_color_encoding_t,
    crop: libc::c_int,
    fix_bad: libc::c_int,
    denoise: libc::c_int,
    apply_sgain: libc::c_int,
    wb: *mut libc::c_char,
    log_hist: libc::c_int,
) -> x3f_return_t {
    let f_out = unsafe { libc::fopen(outfilename, c"wb".as_ptr()) };
    if f_out.is_null() {
        return x3f_return_e_X3F_OUTFILE_ERROR;
    }

    let mut image: x3f_area16_t = unsafe { mem::zeroed() };
    let ok = unsafe {
        x3f_get_image(
            x3f,
            &mut image,
            ptr::null_mut(),
            encoding,
            crop,
            fix_bad,
            denoise,
            apply_sgain,
            wb,
        )
    };
    if ok == 0 || image.channels < 3 {
        unsafe { libc::fclose(f_out) };
        return x3f_return_e_X3F_ARGUMENT_ERROR;
    }

    // 3 × 65536 u32 bins.
    let bin_count = 1usize << 16;
    let mut histograms: [*mut u32; 3] = [ptr::null_mut(); 3];
    for slot in &mut histograms {
        *slot = unsafe { libc::calloc(bin_count, mem::size_of::<u32>()) as *mut u32 };
    }

    let mut max: u16 = 0;
    let row_stride = image.row_stride as usize;
    let channels = image.channels as usize;

    for row in 0..image.rows as usize {
        for col in 0..image.columns as usize {
            for color in 0..3usize {
                let mut val = unsafe { *image.data.add(row_stride * row + channels * col + color) };
                if log_hist != 0 {
                    val = ilog(val as i32) as u16;
                }
                unsafe {
                    *histograms[color].add(val as usize) += 1;
                }
                if val > max {
                    max = val;
                }
            }
        }
    }

    for i in 0..=max as usize {
        let v0 = unsafe { *histograms[0].add(i) };
        let v1 = unsafe { *histograms[1].add(i) };
        let v2 = unsafe { *histograms[2].add(i) };

        if v0 != 0 || v1 != 0 || v2 != 0 {
            if log_hist != 0 {
                unsafe {
                    libc::fprintf(
                        f_out,
                        c"%5d, %5d , %6d , %6d , %6d\n".as_ptr(),
                        i as libc::c_int,
                        ilog_inv(i as i32),
                        v0,
                        v1,
                        v2,
                    );
                }
            } else {
                unsafe {
                    libc::fprintf(
                        f_out,
                        c"%5d , %6d , %6d , %6d\n".as_ptr(),
                        i as libc::c_int,
                        v0,
                        v1,
                        v2,
                    );
                }
            }
        }
    }

    for slot in &histograms {
        unsafe { libc::free(*slot as *mut libc::c_void) };
    }

    unsafe { libc::fclose(f_out) };
    unsafe { libc::free(image.buf) };

    x3f_return_e_X3F_OK
}

#[used]
static _ANCHOR_DUMP_HIST: unsafe extern "C" fn(
    *mut x3f_t,
    *mut libc::c_char,
    x3f_color_encoding_t,
    libc::c_int,
    libc::c_int,
    libc::c_int,
    libc::c_int,
    *mut libc::c_char,
    libc::c_int,
) -> x3f_return_t = x3f_dump_raw_data_as_histogram;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ilog_zero_and_negative_clamp_to_zero() {
        assert_eq!(ilog(0), 0);
        assert_eq!(ilog(-5), 0);
    }

    #[test]
    fn ilog_matches_c_impl_at_powers_of_two() {
        // log_2(1)*10 = 0; log_2(2)*10 = 10; log_2(4)*10 = 20; log_2(1024)*10=100
        assert_eq!(ilog(1), 0);
        assert_eq!(ilog(2), 10);
        assert_eq!(ilog(4), 20);
        assert_eq!(ilog(1024), 100);
    }

    #[test]
    fn ilog_inv_round_trips_to_value() {
        // ilog_inv(10) = 2^1 = 2; ilog_inv(20) = 4; ilog_inv(100) = 1024.
        assert_eq!(ilog_inv(0), 1);
        assert_eq!(ilog_inv(10), 2);
        assert_eq!(ilog_inv(20), 4);
        assert_eq!(ilog_inv(100), 1024);
    }
}
