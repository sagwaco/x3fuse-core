//! Process-wide knobs that the legacy C library exposes as mutable globals.
//!
//! These are wrapped in safe, ergonomic setters so callers do not have to
//! touch `unsafe` directly. They are still globals — calling them from
//! multiple threads concurrently is unsound. The CLI sets them once at
//! startup before any conversion runs.

use x3f_sys as sys;

/// Function-pointer signature passed to [`set_log_callback`].
///
/// `level` is the verbosity, `message` is a UTF-8 (assumed; the C library
/// only ever produces ASCII) message with no level prefix and no enforced
/// trailing newline. Implementations should be infallible — a panic across
/// FFI is undefined behaviour.
pub type LogCallback = fn(level: Verbosity, message: &str);

unsafe extern "C" fn log_trampoline(level: sys::x3f_verbosity_t, msg: *const std::os::raw::c_char) {
    let cb = unsafe { LOG_CALLBACK };
    let Some(cb) = cb else { return };
    let msg = if msg.is_null() {
        ""
    } else {
        // SAFETY: x3f_printf vsnprintf-produces a NUL-terminated buffer.
        match unsafe { std::ffi::CStr::from_ptr(msg) }.to_str() {
            Ok(s) => s,
            Err(_) => return,
        }
    };
    let v = match level {
        sys::x3f_verbosity_t_ERR => Verbosity::Error,
        sys::x3f_verbosity_t_WARN => Verbosity::Warn,
        sys::x3f_verbosity_t_INFO => Verbosity::Info,
        _ => Verbosity::Debug,
    };
    cb(v, msg);
}

static mut LOG_CALLBACK: Option<LogCallback> = None;

/// Route C-side log messages to a Rust callback instead of stdout/stderr.
///
/// Pass `None` to restore the default (writes to stdout/stderr with level
/// prefixes). Setting a callback is intended for library embedders (mobile,
/// WASM) that need to capture diagnostics. The CLI uses the default.
pub fn set_log_callback(cb: Option<LogCallback>) {
    // SAFETY: globals are written before any worker threads exist.
    unsafe {
        LOG_CALLBACK = cb;
        sys::x3f_printf_callback = match cb {
            Some(_) => Some(log_trampoline),
            None => None,
        };
    }
}

/// Verbosity level for the C-side `x3f_printf` log macro.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verbosity {
    /// Errors only.
    Error,
    /// Warnings and errors.
    Warn,
    /// Info, warnings, errors (the default).
    Info,
    /// Everything, including per-pixel debug spam.
    Debug,
}

impl Verbosity {
    fn to_raw(self) -> sys::x3f_verbosity_t {
        // The C enum is: ERR=0, WARN=1, INFO=2, DEBUG=3. We match it
        // verbatim by name in `sys` (x3f_verbosity_t is a c_uint alias and
        // the values are exposed as constants by bindgen).
        match self {
            Verbosity::Error => sys::x3f_verbosity_t_ERR,
            Verbosity::Warn => sys::x3f_verbosity_t_WARN,
            Verbosity::Info => sys::x3f_verbosity_t_INFO,
            Verbosity::Debug => sys::x3f_verbosity_t_DEBUG,
        }
    }
}

/// Set the global C-side verbosity. Default is `Info`.
pub fn set_verbosity(v: Verbosity) {
    // SAFETY: write to a global int. Not racy in practice because the CLI
    // calls this only at startup before any worker threads exist.
    unsafe {
        sys::x3f_printf_level = v.to_raw();
    }
}

/// Override the legacy SD14-and-older offset detection. `Some(off)` forces
/// the offset; `None` returns to automatic detection.
pub fn set_offset_legacy(offset: Option<i32>) {
    // SAFETY: globals are written before any decoding starts.
    unsafe {
        match offset {
            Some(o) => {
                sys::legacy_offset = o;
                sys::auto_legacy_offset = 0;
            }
            None => {
                sys::auto_legacy_offset = 1;
            }
        }
    }
}

/// Cap the number of matrix elements printed in metadata dumps. Default 100.
pub fn set_max_printed_matrix_elements(n: u32) {
    // SAFETY: see set_offset_legacy.
    unsafe {
        sys::max_printed_matrix_elements = n;
    }
}
