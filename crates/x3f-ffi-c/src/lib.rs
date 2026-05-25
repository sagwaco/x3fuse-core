//! C ABI for the Rust x3f library.
//!
//! Surface is intentionally small in M8a: enough to satisfy the per-target
//! smoke test the PORT-PLAN calls for ("load file → metadata dump →
//! thumbnail extract"). Process-to-DNG/TIFF and other heavier surfaces will
//! follow once the iOS/Android/WASM build wiring is in place.
//!
//! Conventions
//! -----------
//! * **Handles** (`X3FReader *`) are opaque pointers owned by this
//!   library. Allocate with `x3f_reader_open`; free with
//!   `x3f_reader_close`. Mixing this library's allocator with the
//!   caller's malloc/free across handles is undefined.
//! * **Return codes**: 0 = success, non-zero = error. On error, callers
//!   may pull a human-readable message from `x3f_last_error()` (thread-
//!   local; cleared on the next successful call).
//! * **Strings**: every `*const c_char` is expected to be NUL-terminated
//!   UTF-8. Output strings owned by the library (`x3f_last_error`) live
//!   until the next call on the same thread.
//! * **Threading**: each `X3FReader` is single-threaded; callers must not
//!   share a handle across threads without external synchronization.
//!   Distinct handles are independent.

#![deny(unsafe_op_in_unsafe_fn)]

use std::cell::RefCell;
use std::ffi::{c_char, c_int, CStr, CString};
use std::path::Path;
use std::ptr;

use x3f_core::Reader;

/// Opaque handle for an open X3F reader. Created by `x3f_reader_open`,
/// freed by `x3f_reader_close`. The empty-enum idiom guarantees the
/// type cannot be instantiated from C and forces cbindgen to emit a
/// forward declaration rather than a struct body.
pub enum X3FReader {}

thread_local! {
    static LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
}

fn set_last_error(msg: impl Into<String>) {
    let cstr = CString::new(msg.into()).unwrap_or_else(|_| {
        // Caller's message contained NULs; replace them with '?'.
        CString::new("error message contained NUL bytes").unwrap()
    });
    LAST_ERROR.with(|cell| *cell.borrow_mut() = Some(cstr));
}

fn clear_last_error() {
    LAST_ERROR.with(|cell| *cell.borrow_mut() = None);
}

/// Returns the last error message produced on this thread, or NULL if no
/// error has occurred since the last successful call. The pointer is owned
/// by the library and remains valid until the next call on the same thread
/// that produces or clears an error.
///
/// # Safety
///
/// The returned pointer must not be freed. Copy the string before making
/// further library calls if you need to retain it.
#[no_mangle]
pub extern "C" fn x3f_last_error() -> *const c_char {
    LAST_ERROR.with(|cell| match &*cell.borrow() {
        Some(s) => s.as_ptr(),
        None => ptr::null(),
    })
}

/// Decode `path` as a UTF-8 NUL-terminated C string and turn it into a
/// `Path`. Sets last-error and returns `None` on failure.
///
/// # Safety
///
/// `path` must be a valid NUL-terminated string or NULL.
unsafe fn cstr_to_path<'a>(path: *const c_char) -> Option<&'a Path> {
    if path.is_null() {
        set_last_error("path argument was NULL");
        return None;
    }
    // SAFETY: caller guarantees path is NUL-terminated.
    let cstr = unsafe { CStr::from_ptr(path) };
    match cstr.to_str() {
        Ok(s) => Some(Path::new(s)),
        Err(_) => {
            set_last_error("path was not valid UTF-8");
            None
        }
    }
}

/// Opens and parses an X3F file's directory. Returns NULL on error; call
/// `x3f_last_error()` to retrieve the failure reason.
///
/// # Safety
///
/// `path` must be a NUL-terminated UTF-8 string or NULL.
#[no_mangle]
pub unsafe extern "C" fn x3f_reader_open(path: *const c_char) -> *mut X3FReader {
    // SAFETY: caller contract.
    let Some(p) = (unsafe { cstr_to_path(path) }) else {
        return ptr::null_mut();
    };
    match Reader::open(p) {
        Ok(r) => {
            clear_last_error();
            Box::into_raw(Box::new(r)).cast()
        }
        Err(e) => {
            set_last_error(e.to_string());
            ptr::null_mut()
        }
    }
}

/// Parses an X3F file from an in-memory byte buffer. Use this from
/// environments without a host filesystem (browser WASM, JNI, Swift
/// `Data`-bridged buffers). The bytes are *copied* internally; the
/// caller may free `data` immediately after this returns.
///
/// Returns NULL on parse failure or on an unsupported host (currently
/// Windows, where libc has no `fmemopen` equivalent — `x3f_reader_open`
/// is the right path there). Call `x3f_last_error()` for details.
///
/// # Safety
///
/// `data` must be valid for `len` bytes (or `len` may be 0 with a
/// NULL `data`, which always returns NULL).
#[no_mangle]
pub unsafe extern "C" fn x3f_reader_open_from_bytes(data: *const u8, len: usize) -> *mut X3FReader {
    #[cfg(any(unix, target_arch = "wasm32"))]
    {
        if data.is_null() || len == 0 {
            set_last_error("x3f_reader_open_from_bytes: data was NULL or len was 0");
            return ptr::null_mut();
        }
        // SAFETY: caller contract — data valid for `len` bytes.
        let bytes = unsafe { core::slice::from_raw_parts(data, len) };
        match Reader::from_bytes(bytes) {
            Ok(r) => {
                clear_last_error();
                Box::into_raw(Box::new(r)).cast()
            }
            Err(e) => {
                set_last_error(e.to_string());
                ptr::null_mut()
            }
        }
    }
    #[cfg(not(any(unix, target_arch = "wasm32")))]
    {
        let _ = (data, len);
        set_last_error(
            "x3f_reader_open_from_bytes: unsupported on this host (no fmemopen); use x3f_reader_open",
        );
        ptr::null_mut()
    }
}

/// Frees an `X3FReader` previously returned by `x3f_reader_open`. NULL is
/// a no-op (matches `free`).
///
/// # Safety
///
/// `handle` must be either NULL or a pointer previously returned by
/// `x3f_reader_open` and not yet closed.
#[no_mangle]
pub unsafe extern "C" fn x3f_reader_close(handle: *mut X3FReader) {
    if handle.is_null() {
        return;
    }
    // SAFETY: caller contract — handle was a `Box::into_raw` from
    // `x3f_reader_open` and has not been freed.
    let _ = unsafe { Box::from_raw(handle.cast::<Reader>()) };
}

/// Reads the X3F header version recorded at file open. Returns 0 if
/// `handle` is NULL.
///
/// # Safety
///
/// `handle` must be NULL or a live pointer from `x3f_reader_open`.
#[no_mangle]
pub unsafe extern "C" fn x3f_reader_header_version(handle: *const X3FReader) -> u32 {
    if handle.is_null() {
        return 0;
    }
    // SAFETY: caller contract.
    let r = unsafe { &*handle.cast::<Reader>() };
    r.header_version()
}

/// Loads CAMF + property list and writes a textual metadata dump to
/// `out_path`. Output format matches the legacy `x3f_extract -meta`.
/// Returns 0 on success, non-zero on error (call `x3f_last_error`).
///
/// # Safety
///
/// `handle` must be a live pointer from `x3f_reader_open`. `out_path`
/// must be a NUL-terminated UTF-8 string.
#[no_mangle]
pub unsafe extern "C" fn x3f_reader_dump_meta(
    handle: *mut X3FReader,
    out_path: *const c_char,
) -> c_int {
    if handle.is_null() {
        set_last_error("reader handle was NULL");
        return -1;
    }
    // SAFETY: caller contract.
    let r = unsafe { &mut *handle.cast::<Reader>() };
    // SAFETY: caller contract.
    let Some(p) = (unsafe { cstr_to_path(out_path) }) else {
        return -1;
    };

    if let Err(e) = r.load_camf() {
        set_last_error(format!("load_camf: {e}"));
        return -1;
    }
    if let Err(e) = r.load_property_list() {
        set_last_error(format!("load_property_list: {e}"));
        return -1;
    }
    if let Err(e) = r.dump_meta(p) {
        set_last_error(format!("dump_meta: {e}"));
        return -1;
    }
    clear_last_error();
    0
}

/// Loads and writes the embedded JPEG thumbnail to `out_path`. Returns 0
/// on success, non-zero on error.
///
/// # Safety
///
/// `handle` must be a live pointer from `x3f_reader_open`. `out_path`
/// must be a NUL-terminated UTF-8 string.
#[no_mangle]
pub unsafe extern "C" fn x3f_reader_dump_jpeg_thumbnail(
    handle: *mut X3FReader,
    out_path: *const c_char,
) -> c_int {
    if handle.is_null() {
        set_last_error("reader handle was NULL");
        return -1;
    }
    // SAFETY: caller contract.
    let r = unsafe { &mut *handle.cast::<Reader>() };
    // SAFETY: caller contract.
    let Some(p) = (unsafe { cstr_to_path(out_path) }) else {
        return -1;
    };

    if let Err(e) = r.load_thumbnail_jpeg() {
        set_last_error(format!("load_thumbnail_jpeg: {e}"));
        return -1;
    }
    if let Err(e) = r.dump_jpeg(p) {
        set_last_error(format!("dump_jpeg: {e}"));
        return -1;
    }
    clear_last_error();
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn last_error_is_null_initially() {
        assert!(x3f_last_error().is_null());
    }

    #[test]
    fn open_with_null_path_returns_null_and_sets_error() {
        let h = unsafe { x3f_reader_open(ptr::null()) };
        assert!(h.is_null());
        assert!(!x3f_last_error().is_null());
    }

    #[test]
    fn open_with_nonexistent_path_returns_null() {
        let cpath = CString::new("/tmp/definitely-does-not-exist.X3F").unwrap();
        let h = unsafe { x3f_reader_open(cpath.as_ptr()) };
        assert!(h.is_null());
        let err = x3f_last_error();
        assert!(!err.is_null());
        let msg = unsafe { CStr::from_ptr(err) }.to_str().unwrap();
        assert!(msg.contains("could not open input file"), "got: {msg}");
    }

    #[test]
    fn close_null_handle_is_safe() {
        // Mirrors the `free(NULL)` contract.
        unsafe { x3f_reader_close(ptr::null_mut()) };
    }

    #[test]
    fn header_version_with_null_handle_is_zero() {
        assert_eq!(unsafe { x3f_reader_header_version(ptr::null()) }, 0);
    }
}
