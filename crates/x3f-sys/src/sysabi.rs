//! libc compatibility shim for `wasm32-unknown-unknown`.
//!
//! `libc 0.2` exports almost nothing on `wasm32-unknown-unknown` (no
//! sysroot, no allocator FFI). On every other supported target this
//! module is just `pub use libc::*;` — a transparent re-export. On
//! `wasm32-unknown-unknown` it provides Rust-native equivalents:
//!
//! * **Types** (`c_int` / `c_char` / `c_void` / `c_long`, plus the
//!   opaque `FILE`) come from `core::ffi`. Layouts are
//!   ABI-equivalent to libc's on every platform we care about.
//!
//! * **Allocator** (`malloc` / `calloc` / `realloc` / `free`) goes
//!   through Rust's global allocator. Allocations are sized with a
//!   `usize` length prefix so `realloc` / `free` know the original
//!   layout — `Box::from_raw` needs a layout, and the C ABI doesn't
//!   carry one. The implementation is the standard "store the layout
//!   8 bytes before the user pointer" trick.
//!
//! * **Memory** (`memcpy` / `memset`) calls into the corresponding
//!   `core::ptr` intrinsics. On wasm32 these compile to a single
//!   `memory.copy` / `memory.fill` opcode under the hood.
//!
//! * **Conversion** (`atof` / `atoi`) parses the NUL-terminated input
//!   via `str::parse` and silently returns 0 on failure (matching C's
//!   error-as-zero behaviour).
//!
//! * **File I/O** dispatches against an in-memory buffer (M8d-α-2).
//!   `fmemopen(buf, size, "rb")` allocates a heap-side `MemFile`
//!   cursor and returns a `*mut FILE` aliasing it; `fread` / `fseek`
//!   / `ftell` / `fgetc` / `fclose` walk that cursor. `fopen(path)`
//!   still returns NULL — there's no host filesystem on
//!   `wasm32-unknown-unknown` — but the buffer-based reader path
//!   (`x3f-core::Reader::from_bytes`) goes through `fmemopen` and
//!   works fully at runtime. The MemFile owns a copy of the input
//!   bytes so the FILE* stays valid past the caller's borrow.
//!
//! * **Process exit** (`abort` / `exit`) maps to `core::arch::wasm32::
//!   unreachable()` (an unrecoverable trap).
//!
//! * **`printf` / `fprintf`** are *not* shimmed here — they're variadic
//!   and Rust's stable variadic-fn support is incomplete. The two
//!   files that depend on them (`print_meta.rs` and the matrix print
//!   routines in `matrix.rs`) are cfg-gated out on wasm32 entirely.
//!   See those files' module headers for details.

#![allow(dead_code)]

#[cfg(not(target_arch = "wasm32"))]
pub use libc::*;

#[cfg(target_arch = "wasm32")]
pub use wasm_shim::*;

#[cfg(target_arch = "wasm32")]
mod wasm_shim {
    use core::ffi::CStr;
    pub use core::ffi::{c_char, c_int, c_long, c_void};

    /// Opaque-to-C `FILE` handle. Internally, every non-NULL `*mut FILE`
    /// returned by `fmemopen` aliases a heap-allocated `MemFile`; the
    /// other I/O calls cast back to `*mut MemFile` and walk its cursor.
    /// Defining FILE this way (zero-sized struct) lets bindgen's existing
    /// `*mut FILE` parameter types compile unchanged.
    #[repr(C)]
    pub struct FILE {
        _opaque: [u8; 0],
    }

    /// Heap-side cursor backing the wasm32 `FILE*`. Owns its byte buffer
    /// so the `FILE*` stays valid past the original `fmemopen` call's
    /// borrow.
    pub(super) struct MemFile {
        data: Vec<u8>,
        pos: usize,
        eof: bool,
    }

    pub const SEEK_SET: c_int = 0;
    pub const SEEK_CUR: c_int = 1;
    pub const SEEK_END: c_int = 2;

    // ----- Allocator: layout-aware malloc/calloc/realloc/free ----

    // We prepend each allocation with a usize storing the user size, so
    // `free` / `realloc` can rebuild the `Layout` for `dealloc`.
    const PREFIX: usize = core::mem::size_of::<usize>();
    // Align to 16 bytes so any C struct alignment requirement is met.
    const ALIGN: usize = 16;

    fn layout(size: usize) -> core::alloc::Layout {
        // Caller asked for `size` bytes; we need `size + PREFIX`.
        // alignment is fixed at ALIGN.
        let total = size.checked_add(PREFIX).expect("malloc size overflow");
        core::alloc::Layout::from_size_align(total, ALIGN).expect("invalid layout")
    }

    /// # Safety
    /// `size` may be 0, in which case behaviour matches C: returns a
    /// valid pointer that can be `free`'d but mustn't be dereferenced.
    pub unsafe fn malloc(size: usize) -> *mut c_void {
        let l = layout(size.max(1));
        // SAFETY: layout has non-zero size (we max'd to 1) and 16-byte
        // alignment which is valid.
        let raw = unsafe { std::alloc::alloc(l) };
        if raw.is_null() {
            return core::ptr::null_mut();
        }
        // SAFETY: we just allocated PREFIX+size bytes; the first PREFIX
        // bytes are ours.
        unsafe { core::ptr::write(raw as *mut usize, size) };
        // SAFETY: stays within the allocation we own.
        unsafe { raw.add(PREFIX).cast() }
    }

    pub unsafe fn calloc(n: usize, size: usize) -> *mut c_void {
        let total = n.checked_mul(size).expect("calloc size overflow");
        let l = layout(total.max(1));
        // SAFETY: layout is valid; alloc_zeroed zero-initializes.
        let raw = unsafe { std::alloc::alloc_zeroed(l) };
        if raw.is_null() {
            return core::ptr::null_mut();
        }
        // SAFETY: same reasoning as malloc.
        unsafe { core::ptr::write(raw as *mut usize, total) };
        unsafe { raw.add(PREFIX).cast() }
    }

    /// # Safety
    /// `ptr` must be NULL or a pointer returned by `malloc`/`calloc`/
    /// `realloc` from this module.
    pub unsafe fn free(ptr: *mut c_void) {
        if ptr.is_null() {
            return;
        }
        // SAFETY: pointer was returned by our allocator, so PREFIX bytes
        // before it hold the size we recorded.
        let raw = unsafe { (ptr as *mut u8).sub(PREFIX) };
        let size = unsafe { core::ptr::read(raw as *mut usize) };
        let l = layout(size.max(1));
        // SAFETY: same layout we passed to alloc.
        unsafe { std::alloc::dealloc(raw, l) };
    }

    /// # Safety
    /// `ptr` must be NULL or returned by this module's allocator.
    pub unsafe fn realloc(ptr: *mut c_void, new_size: usize) -> *mut c_void {
        if ptr.is_null() {
            return unsafe { malloc(new_size) };
        }
        if new_size == 0 {
            unsafe { free(ptr) };
            return core::ptr::null_mut();
        }
        let raw = unsafe { (ptr as *mut u8).sub(PREFIX) };
        let old_size = unsafe { core::ptr::read(raw as *mut usize) };
        let old_layout = layout(old_size.max(1));
        let new_layout = layout(new_size);
        // SAFETY: see alloc::Allocator::grow / shrink contract.
        let new_raw = unsafe { std::alloc::realloc(raw, old_layout, new_layout.size()) };
        if new_raw.is_null() {
            return core::ptr::null_mut();
        }
        unsafe { core::ptr::write(new_raw as *mut usize, new_size) };
        unsafe { new_raw.add(PREFIX).cast() }
    }

    // ----- Memory ops -------------------------------------------------

    /// # Safety
    /// `dst` and `src` must be valid for `n` bytes; ranges may not overlap.
    pub unsafe fn memcpy(dst: *mut c_void, src: *const c_void, n: usize) -> *mut c_void {
        unsafe { core::ptr::copy_nonoverlapping(src as *const u8, dst as *mut u8, n) };
        dst
    }

    /// # Safety
    /// `s` must be valid for `n` bytes.
    pub unsafe fn memset(s: *mut c_void, c: c_int, n: usize) -> *mut c_void {
        unsafe { core::ptr::write_bytes(s as *mut u8, c as u8, n) };
        s
    }

    // ----- atof / atoi -----------------------------------------------

    /// C `atof` semantics — leading whitespace skipped, parse fails
    /// silently to 0.0.
    ///
    /// # Safety
    /// `s` must be a NUL-terminated string.
    pub unsafe fn atof(s: *const c_char) -> f64 {
        if s.is_null() {
            return 0.0;
        }
        unsafe { CStr::from_ptr(s) }
            .to_str()
            .unwrap_or("")
            .trim_start()
            .parse::<f64>()
            .unwrap_or(0.0)
    }

    /// # Safety
    /// `s` must be a NUL-terminated string.
    pub unsafe fn atoi(s: *const c_char) -> c_int {
        if s.is_null() {
            return 0;
        }
        unsafe { CStr::from_ptr(s) }
            .to_str()
            .unwrap_or("")
            .trim_start()
            .parse::<c_int>()
            .unwrap_or(0)
    }

    // ----- File I/O: MemFile cursor over an in-memory byte buffer ----

    /// `fopen(path, mode)` — always NULL on wasm32-unknown-unknown.
    /// The browser/runtime has no host filesystem; consumers must use
    /// `fmemopen` (or higher up, `Reader::from_bytes`) to feed data in.
    ///
    /// # Safety
    /// Caller-provided pointers must be NUL-terminated strings or NULL.
    pub unsafe extern "C" fn fopen(_path: *const c_char, _mode: *const c_char) -> *mut FILE {
        core::ptr::null_mut()
    }

    /// `fdopen(fd, mode)` — always NULL. There are no fds on wasm32.
    ///
    /// # Safety
    /// `_fd` is unused.
    pub unsafe extern "C" fn fdopen(_fd: c_int, _mode: *const c_char) -> *mut FILE {
        core::ptr::null_mut()
    }

    /// `fmemopen(buf, size, mode)` — wrap `size` bytes at `buf` in a
    /// heap-allocated `MemFile` and return a `*mut FILE` cursor over
    /// the *copy*. (POSIX `fmemopen` borrows the buffer; we copy
    /// instead so the FILE* outlives the caller's borrow, which makes
    /// `Reader::from_bytes(&[u8])` safe to call from JS without
    /// keeping the input pinned.)
    ///
    /// `mode` is parsed permissively: any value containing `'w'` or
    /// `'a'` returns NULL (we're read-only); anything else is treated
    /// as read-mode. The legacy x3f code path always passes `"rb"` so
    /// this matches.
    ///
    /// # Safety
    /// `buf` must be valid for `size` bytes (or `size` may be 0, in
    /// which case `buf` is unread). `mode` must be NUL-terminated or
    /// NULL.
    #[no_mangle]
    pub unsafe extern "C" fn fmemopen(
        buf: *const c_void,
        size: usize,
        mode: *const c_char,
    ) -> *mut FILE {
        // Reject write/append modes; we only implement read.
        if !mode.is_null() {
            let m = unsafe { CStr::from_ptr(mode) }.to_bytes();
            if m.iter().any(|&c| c == b'w' || c == b'a') {
                return core::ptr::null_mut();
            }
        }
        let data: Vec<u8> = if size == 0 || buf.is_null() {
            Vec::new()
        } else {
            // SAFETY: caller contract — `buf` valid for `size` bytes.
            let slice = unsafe { core::slice::from_raw_parts(buf as *const u8, size) };
            slice.to_vec()
        };
        let mf = Box::new(MemFile {
            data,
            pos: 0,
            eof: false,
        });
        Box::into_raw(mf) as *mut FILE
    }

    /// `fclose(stream)` — frees the heap-side `MemFile`. Returns 0.
    /// NULL is a no-op (matches POSIX in spirit; real `fclose(NULL)`
    /// is undefined, but every non-NULL FILE* we hand out is from
    /// `fmemopen` and thus owned).
    ///
    /// # Safety
    /// `stream` must be NULL or a `FILE*` returned by `fmemopen`.
    #[no_mangle]
    pub unsafe extern "C" fn fclose(stream: *mut FILE) -> c_int {
        if stream.is_null() {
            return 0;
        }
        // SAFETY: caller contract — the FILE* came from `fmemopen`
        // and thus is a `Box::into_raw`-derived pointer.
        let _ = unsafe { Box::from_raw(stream as *mut MemFile) };
        0
    }

    /// `fseek(stream, offset, whence)` — sets the cursor. Returns 0
    /// on success, -1 on out-of-range or NULL stream.
    ///
    /// # Safety
    /// `stream` must be NULL or a live `FILE*` from `fmemopen`.
    #[no_mangle]
    pub unsafe extern "C" fn fseek(stream: *mut FILE, offset: c_long, whence: c_int) -> c_int {
        if stream.is_null() {
            return -1;
        }
        // SAFETY: caller contract.
        let mf = unsafe { &mut *(stream as *mut MemFile) };
        let len = mf.data.len() as i64;
        let off = offset as i64;
        let new_pos = match whence {
            x if x == SEEK_SET => off,
            x if x == SEEK_CUR => mf.pos as i64 + off,
            x if x == SEEK_END => len + off,
            _ => return -1,
        };
        if new_pos < 0 || new_pos > len {
            return -1;
        }
        mf.pos = new_pos as usize;
        mf.eof = false;
        0
    }

    /// `ftell(stream)` — current byte offset of the cursor. Returns
    /// -1 if `stream` is NULL.
    ///
    /// # Safety
    /// `stream` must be NULL or a live `FILE*` from `fmemopen`.
    #[no_mangle]
    pub unsafe extern "C" fn ftell(stream: *mut FILE) -> c_long {
        if stream.is_null() {
            return -1;
        }
        // SAFETY: caller contract.
        let mf = unsafe { &*(stream as *mut MemFile) };
        mf.pos as c_long
    }

    /// `fread(buf, size, n, stream)` — read up to `size * n` bytes
    /// from the MemFile cursor into `buf`. Returns the count of
    /// fully-read items (NOT bytes), matching POSIX. Sets the EOF
    /// flag on a short read.
    ///
    /// # Safety
    /// `buf` must be valid for `size * n` bytes; `stream` must be
    /// NULL or a live `FILE*` from `fmemopen`.
    #[no_mangle]
    pub unsafe extern "C" fn fread(
        buf: *mut c_void,
        size: usize,
        n: usize,
        stream: *mut FILE,
    ) -> usize {
        if stream.is_null() || size == 0 || n == 0 {
            return 0;
        }
        // SAFETY: caller contract.
        let mf = unsafe { &mut *(stream as *mut MemFile) };
        let total = match size.checked_mul(n) {
            Some(t) => t,
            None => return 0,
        };
        let avail = mf.data.len().saturating_sub(mf.pos);
        let to_read = total.min(avail);
        if to_read > 0 {
            // SAFETY: `buf` valid per caller; `mf.data[mf.pos..]` has
            // at least `to_read` bytes.
            unsafe {
                core::ptr::copy_nonoverlapping(
                    mf.data.as_ptr().add(mf.pos),
                    buf as *mut u8,
                    to_read,
                );
            }
            mf.pos += to_read;
        }
        if to_read < total {
            mf.eof = true;
        }
        to_read / size
    }

    /// `fgetc(stream)` — read one byte and advance, or return -1 (EOF)
    /// at end-of-buffer.
    ///
    /// # Safety
    /// `stream` must be NULL or a live `FILE*` from `fmemopen`.
    #[no_mangle]
    pub unsafe extern "C" fn fgetc(stream: *mut FILE) -> c_int {
        if stream.is_null() {
            return -1;
        }
        // SAFETY: caller contract.
        let mf = unsafe { &mut *(stream as *mut MemFile) };
        if mf.pos >= mf.data.len() {
            mf.eof = true;
            return -1;
        }
        let b = mf.data[mf.pos];
        mf.pos += 1;
        b as c_int
    }

    // ----- abort / exit ---------------------------------------------

    /// Equivalent to a wasm trap. The C ABI promises this is `-> !`; on
    /// wasm32 we use `core::arch::wasm32::unreachable` which compiles
    /// to a `unreachable` opcode. Marked `unsafe` to match libc's
    /// signature on host so call sites' `unsafe { libc::abort() }`
    /// don't trigger an unused_unsafe warning on wasm32.
    pub unsafe fn abort() -> ! {
        core::arch::wasm32::unreachable()
    }

    /// Same as `abort` for our purposes; the exit code is discarded.
    /// Marked `unsafe` to match libc's signature.
    pub unsafe fn exit(_code: c_int) -> ! {
        core::arch::wasm32::unreachable()
    }
}
