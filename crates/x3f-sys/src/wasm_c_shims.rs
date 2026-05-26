//! `#[no_mangle]` Rust shims for the four C symbols still referenced
//! by the bindgen-generated bindings on wasm32 targets:
//! `x3f_printf` (variadic), `x3f_denoise`, `x3f_denoise_active`,
//! `x3f_set_use_opencl`.
//!
//! On every other target these symbols come from compiled C/C++
//! (`src/x3f_printf.c`, `src/x3f_denoise.cpp`, the `denoise_stub.c`
//! fallback). On wasm32 we can't run cc-rs (Apple's bundled clang
//! has no wasm-libc sysroot, and the actual sources `#include <stdio.h>`
//! anyway), so the linker would otherwise leave these as unresolved
//! `(import "env" …)` entries in the cdylib — making the wasm
//! impossible to instantiate without a JS / wasmtime host that
//! provides them.
//!
//! The shims are no-ops:
//!
//! * **`x3f_printf`** — Rust's stable `extern "C"` does not let us
//!   define a variadic function. We work around that by providing a
//!   3-arg signature `(level, fmt, _arg)`. Rust on wasm32 lowers
//!   every variadic call site to import-type `(i32, i32, i32)`
//!   regardless of how many extra args the C source declared, so a
//!   3-arg impl satisfies the import. (You can verify the import
//!   type via `wasm-tools print` — it's always `type 4` aka
//!   `(func (param i32 i32 i32))`.) The extra arg is ignored
//!   either way; verbose-mode debug printing is silently dropped on
//!   wasm32. A future Rust port of `x3f_printf` could route through
//!   `tracing` or a JS console hook.
//!
//! * **`x3f_denoise`** / **`x3f_denoise_active`** — already no-ops on
//!   wasm32 in the legacy build (denoise_stub.c). Same here.
//!
//! * **`x3f_set_use_opencl`** — kept for ABI parity with the legacy
//!   `-ocl` CLI flag, which is a silent no-op everywhere now.
//!
//! Note: bindgen's forward declarations for these functions remain
//! in `bindings.rs`. Rust accepts an `extern "C" {}` foreign-fn decl
//! AND a separate `#[no_mangle]` definition with the same symbol
//! name, as long as they live in different module paths. The decl
//! sits at crate root (`crate::x3f_printf`); the definition sits at
//! `crate::wasm_c_shims::x3f_printf`. They don't shadow at the Rust
//! level; at the wasm linker level both reference the same symbol
//! and resolve cleanly.
#![cfg(target_arch = "wasm32")]

use crate::sysabi::{c_char, c_int, c_void};

/// Variadic-call-compatible no-op. See module docs for the 3-arg
/// signature rationale.
///
/// # Safety
/// Caller-provided pointers may be NULL; we don't dereference them.
#[no_mangle]
pub unsafe extern "C" fn x3f_printf(_level: c_int, _fmt: *const c_char, _arg: c_int) {}

/// `x3f_set_use_opencl(flag)` — no-op stub.
///
/// # Safety
/// `_flag` is unused.
#[no_mangle]
pub unsafe extern "C" fn x3f_set_use_opencl(_flag: c_int) {}

/// `x3f_denoise(image, type, scale)` — no-op stub.
///
/// # Safety
/// `_image` may be NULL; we don't dereference it.
#[no_mangle]
pub unsafe extern "C" fn x3f_denoise(_image: *mut c_void, _type: c_int, _scale: f32) {}

/// `x3f_denoise_active(area, type, stage, scale)` — no-op stub.
///
/// # Safety
/// `_area` may be NULL; we don't dereference it.
#[no_mangle]
pub unsafe extern "C" fn x3f_denoise_active(
    _area: *mut c_void,
    _type: c_int,
    _stage: c_int,
    _scale: f32,
) {
}
