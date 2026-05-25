//! `wasm32-unknown-unknown` stub for `x3f_print_meta` / `x3f_dump_meta_data`.
//!
//! The real implementation in `print_meta.rs` uses `libc::printf` /
//! `libc::fprintf` for byte-identical `%9f` / `%12g` format-string
//! output. Rust can't variadically shim those on stable, so on wasm32
//! we substitute trivial stubs:
//!
//! * `x3f_dump_meta_data` returns `X3F_ARGUMENT_ERROR` — there's no
//!   filesystem to write to anyway, and the public x3f-core
//!   `Reader::dump_meta(&Path)` API isn't reachable from a browser
//!   surface (paths require a host filesystem).
//! * `x3f_print_meta` is a no-op.
//! * `max_printed_matrix_elements` is preserved as a `static mut` —
//!   it's part of the C ABI and a future structured-metadata wasm
//!   API may consume it.
//!
//! The bindgen forward declarations remain blocklisted in `build.rs`
//! and the symbols below take their place.

use std::os::raw::c_char;

use crate::*;

#[no_mangle]
pub static mut max_printed_matrix_elements: u32 = 100;

/// # Safety
/// `x3f` may be NULL; we don't dereference it.
#[no_mangle]
pub unsafe extern "C" fn x3f_print_meta(_x3f: *mut x3f_t) {}

/// # Safety
/// All pointers may be NULL; we always return an error.
#[no_mangle]
pub unsafe extern "C" fn x3f_dump_meta_data(_x3f: *mut x3f_t, _path: *mut c_char) -> x3f_return_t {
    x3f_return_e_X3F_ARGUMENT_ERROR
}
