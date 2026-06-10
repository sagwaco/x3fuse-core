//! `#[no_mangle]` Rust shim for the variadic `x3f_printf` symbol still
//! referenced by the bindgen-generated bindings on wasm32 targets.
//!
//! On every other target this symbol comes from compiled C
//! (`csrc/x3f_printf.c`). On wasm32 we can't run cc-rs (Apple's bundled
//! clang has no wasm-libc sysroot, and the source `#include`s `<stdio.h>`
//! anyway), so the linker would otherwise leave it as an unresolved
//! `(import "env" …)` entry in the cdylib — making the wasm impossible to
//! instantiate without a JS / wasmtime host that provides it.
//!
//! Rust's stable `extern "C"` does not let us define a variadic function.
//! We work around that by providing a 3-arg signature `(level, fmt, _arg)`.
//! Rust on wasm32 lowers every variadic call site to import-type
//! `(i32, i32, i32)` regardless of how many extra args the C source
//! declared, so a 3-arg impl satisfies the import. (You can verify the
//! import type via `wasm-tools print` — it's always `type 4` aka
//! `(func (param i32 i32 i32))`.) The extra arg is ignored either way;
//! verbose-mode debug printing is silently dropped on wasm32. A future Rust
//! port of `x3f_printf` could route through `tracing` or a JS console hook.
//!
//! (Denoise used to be a no-op shim here too; it is now the pure-Rust NLM in
//! `denoise.rs`, called directly by the pipeline on every target including
//! wasm.)
//!
//! Note: bindgen's forward declaration for `x3f_printf` remains in
//! `bindings.rs`. Rust accepts an `extern "C" {}` foreign-fn decl AND a
//! separate `#[no_mangle]` definition with the same symbol name, as long as
//! they live in different module paths. The decl sits at crate root
//! (`crate::x3f_printf`); the definition sits at
//! `crate::wasm_c_shims::x3f_printf`. They don't shadow at the Rust level; at
//! the wasm linker level both reference the same symbol and resolve cleanly.
#![cfg(target_arch = "wasm32")]

use crate::sysabi::{c_char, c_int};

/// Variadic-call-compatible no-op. See module docs for the 3-arg
/// signature rationale.
///
/// # Safety
/// Caller-provided pointers may be NULL; we don't dereference them.
#[no_mangle]
pub unsafe extern "C" fn x3f_printf(_level: c_int, _fmt: *const c_char, _arg: c_int) {}
