//! `wasm32-unknown-unknown` stub for `x3f_dump_raw_data_as_histogram`.
//!
//! The real implementation in `histogram.rs` writes a CSV histogram to
//! a `FILE*` opened via `libc::fopen`, then pumps rows through
//! `libc::fprintf`. Both are out of reach on wasm32 — the buffer-based
//! reader API replaces filesystem-driven debug dumps with structured
//! data the JS host can format itself.

use crate::sysabi as libc;
use crate::*;

/// # Safety
/// All pointers may be NULL on wasm32; we always return `X3F_ARGUMENT_ERROR`.
#[no_mangle]
pub unsafe extern "C" fn x3f_dump_raw_data_as_histogram(
    _x3f: *mut x3f_t,
    _outfilename: *mut libc::c_char,
    _encoding: x3f_color_encoding_t,
    _crop: libc::c_int,
    _fix_bad: libc::c_int,
    _denoise: libc::c_int,
    _apply_sgain: libc::c_int,
    _wb: *mut libc::c_char,
    _log_hist: libc::c_int,
) -> x3f_return_t {
    x3f_return_e_X3F_ARGUMENT_ERROR
}
