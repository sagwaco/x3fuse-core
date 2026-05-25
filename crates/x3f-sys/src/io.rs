//! M4c — native Rust port of `x3f_new_from_file` and the file-IO helpers
//! `x3f_get1` / `x3f_get2` / `x3f_get4` / `x3f_get4f` / `GETN` from
//! `src/x3f_io.c`.
//!
//! Allocates the `x3f_t` and walks the file header + directory section,
//! populating the parsed structure. Section payload (property strings,
//! image bytes, CAMF body) is **not** loaded here — that's still done by
//! the C `x3f_load_data` and friends in `x3f_io.c`.
//!
//! Memory ownership: the `x3f_t` and the `directory_entry[]` array are
//! allocated with `libc::calloc` so the still-C `x3f_delete` can release
//! them with `libc::free` in the usual way.
//!
//! Symbol export: `x3f_new_from_file` is `#[no_mangle] extern "C"`,
//! blocklisted in bindgen, anchored via a `#[used]` static, and re-exported
//! through `lib.rs` so existing `sys::x3f_new_from_file` callers keep
//! working unchanged.
//!
//! Endianness: X3F files are little-endian. We read each multi-byte value
//! one byte at a time and assemble in LE, matching `src/x3f_io.c` byte-for-
//! byte (independent of host endianness).
#![allow(clippy::missing_safety_doc)]

use std::mem;
use std::ptr;

use crate::*;
// Shadow the external `libc` crate name with our compat shim. On
// every target except wasm32-unknown-unknown this is a transparent
// `pub use libc::*;`; on wasm32 it provides Rust-native equivalents.
// See `sysabi.rs`.
use crate::sysabi as libc;

const X3F_FOVB: u32 = 0x6256_4f46; // FOVb
const X3F_VERSION_2_1: u32 = (2 << 16) + 1;
const X3F_VERSION_2_3: u32 = (2 << 16) + 3;
const X3F_VERSION_3_0: u32 = 3 << 16;
const X3F_VERSION_4_0: u32 = 4 << 16;
const X3F_SECP: u32 = 0x7043_4553; // SECp
const X3F_SECI: u32 = 0x6943_4553; // SECi
const X3F_SECC: u32 = 0x6343_4553; // SECc

const SIZE_UNIQUE_IDENTIFIER: usize = 16;
const SIZE_WHITE_BALANCE: usize = 32;
const SIZE_COLOR_MODE: usize = 32;
const NUM_EXT_DATA_2_1: usize = 32;
const NUM_EXT_DATA_3_0: usize = 64;

#[inline]
unsafe fn get1(f: *mut libc::FILE) -> u32 {
    unsafe { (libc::fgetc(f) as u32) & 0xFF }
}

#[inline]
unsafe fn get4(f: *mut libc::FILE) -> u32 {
    let b0 = unsafe { get1(f) };
    let b1 = unsafe { get1(f) };
    let b2 = unsafe { get1(f) };
    let b3 = unsafe { get1(f) };
    b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)
}

#[inline]
unsafe fn get4f(f: *mut libc::FILE) -> f32 {
    let bits = unsafe { get4(f) };
    f32::from_bits(bits)
}

/// Loop on partial reads (mirrors PUT_GET_N). On a 0-byte short read the
/// C source prints "Failure to access file" and exits — preserve that.
unsafe fn getn(f: *mut libc::FILE, buf: *mut u8, size: usize) {
    let mut left = size;
    let mut p = buf;
    while left != 0 {
        let cur = unsafe { libc::fread(p as *mut libc::c_void, 1, left, f) };
        if cur == 0 {
            unsafe {
                x3f_printf(x3f_verbosity_t_ERR, c"Failure to access file\n".as_ptr());
                libc::exit(1);
            }
        }
        left -= cur;
        p = unsafe { p.add(cur) };
    }
}

#[no_mangle]
pub unsafe extern "C" fn x3f_new_from_file(infile: *mut FILE) -> *mut x3f_t {
    let x3f = unsafe { libc::calloc(1, mem::size_of::<x3f_t>()) as *mut x3f_t };
    if x3f.is_null() {
        return ptr::null_mut();
    }

    unsafe {
        (*x3f).info.error = ptr::null_mut();
        (*x3f).info.input.file = infile;
        (*x3f).info.output.file = ptr::null_mut();
    }

    if infile.is_null() {
        // Match the C: stash a static "No infile" string and return the
        // partially-initialised struct so the caller can read .info.error.
        unsafe {
            (*x3f).info.error = c"No infile".as_ptr() as *mut _;
        }
        return x3f;
    }

    let f = infile as *mut libc::FILE;

    // Read file header.
    unsafe {
        libc::fseek(f, 0, libc::SEEK_SET);
        (*x3f).header.identifier = get4(f);

        if (*x3f).header.identifier != X3F_FOVB {
            x3f_printf(x3f_verbosity_t_ERR, c"Faulty file type\n".as_ptr());
            x3f_delete(x3f);
            return ptr::null_mut();
        }

        (*x3f).header.version = get4(f);
        getn(
            f,
            (*x3f).header.unique_identifier.as_mut_ptr(),
            SIZE_UNIQUE_IDENTIFIER,
        );

        // For version >= 4.0 (Quattro-and-newer), the rest of the header
        // fields are unknown and left zero.
        if (*x3f).header.version < X3F_VERSION_4_0 {
            (*x3f).header.mark_bits = get4(f);
            (*x3f).header.columns = get4(f);
            (*x3f).header.rows = get4(f);
            (*x3f).header.rotation = get4(f);
            if (*x3f).header.version >= X3F_VERSION_2_1 {
                let num_ext_data: usize = if (*x3f).header.version >= X3F_VERSION_3_0 {
                    NUM_EXT_DATA_3_0
                } else {
                    NUM_EXT_DATA_2_1
                };
                getn(
                    f,
                    (*x3f).header.white_balance.as_mut_ptr() as *mut u8,
                    SIZE_WHITE_BALANCE,
                );
                if (*x3f).header.version >= X3F_VERSION_2_3 {
                    getn(
                        f,
                        (*x3f).header.color_mode.as_mut_ptr() as *mut u8,
                        SIZE_COLOR_MODE,
                    );
                }
                getn(f, (*x3f).header.extended_types.as_mut_ptr(), num_ext_data);
                for i in 0..num_ext_data {
                    (*x3f).header.extended_data[i] = get4f(f);
                }
            }
        }

        // Jump to the directory section: last 4 bytes of the file are the
        // directory offset.
        libc::fseek(f, -4, libc::SEEK_END);
        let dir_offset = get4(f) as libc::c_long;
        libc::fseek(f, dir_offset, libc::SEEK_SET);

        (*x3f).directory_section.identifier = get4(f);
        (*x3f).directory_section.version = get4(f);
        (*x3f).directory_section.num_directory_entries = get4(f);

        let n = (*x3f).directory_section.num_directory_entries as usize;
        if n > 0 {
            let size = n * mem::size_of::<x3f_directory_entry_t>();
            (*x3f).directory_section.directory_entry =
                libc::calloc(1, size) as *mut x3f_directory_entry_t;
        }

        // Walk each directory entry. Read its header by seeking into the
        // entry, then return to the directory pos for the next iteration.
        for d in 0..n {
            let de = (*x3f).directory_section.directory_entry.add(d);
            (*de).input.offset = get4(f);
            (*de).input.size = get4(f);
            (*de).output.offset = 0;
            (*de).output.size = 0;
            (*de).type_ = get4(f);

            let save_dir_pos = libc::ftell(f);
            libc::fseek(f, (*de).input.offset as libc::c_long, libc::SEEK_SET);

            (*de).header.identifier = get4(f);
            (*de).header.version = get4(f);

            match (*de).header.identifier {
                X3F_SECP => {
                    let pl = &mut (*de).header.data_subsection.property_list;
                    pl.num_properties = get4(f);
                    pl.character_format = get4(f);
                    pl.reserved = get4(f);
                    pl.total_length = get4(f);
                    pl.data = ptr::null_mut();
                    pl.data_size = 0;
                }
                X3F_SECI => {
                    let id = &mut (*de).header.data_subsection.image_data;
                    id.type_ = get4(f);
                    id.format = get4(f);
                    id.type_format = (id.type_ << 16) + id.format;
                    id.columns = get4(f);
                    id.rows = get4(f);
                    id.row_stride = get4(f);
                    id.huffman = ptr::null_mut();
                    id.data = ptr::null_mut();
                    id.data_size = 0;
                }
                X3F_SECC => {
                    let camf = &mut (*de).header.data_subsection.camf;
                    camf.type_ = get4(f);
                    camf.__bindgen_anon_1.tN.val0 = get4(f);
                    camf.__bindgen_anon_1.tN.val1 = get4(f);
                    camf.__bindgen_anon_1.tN.val2 = get4(f);
                    camf.__bindgen_anon_1.tN.val3 = get4(f);
                    camf.data = ptr::null_mut();
                    camf.data_size = 0;
                    camf.table.element = ptr::null_mut();
                    camf.table.size = 0;
                    camf.tree.nodes = ptr::null_mut();
                    camf.decoded_data = ptr::null_mut();
                    camf.decoded_data_size = 0;
                    camf.entry_table.element = ptr::null_mut();
                    camf.entry_table.size = 0;
                }
                _ => {}
            }

            libc::fseek(f, save_dir_pos, libc::SEEK_SET);
        }
    }

    x3f
}

// Cross-crate dead-code-elimination guard: anchor the no-mangle symbol so
// that callers in x3f-core (which see only the bindgen `extern { fn ... }`
// declaration) link against this Rust definition.
#[used]
static _ANCHOR_NEW_FROM_FILE: unsafe extern "C" fn(*mut FILE) -> *mut x3f_t = x3f_new_from_file;

// =============================================================================
// M5e — port of the remaining `src/x3f_io.c` content (cleanup + getters).
//
// What lands here:
//   - the legacy_offset / auto_legacy_offset globals (consumed by the Rust
//     huffman decoder in entropy.rs)
//   - cleanup helpers (cleanup_huffman_tree / _true / _quattro / _huffman /
//     free_camf_entry)
//   - x3f_delete (the delete orchestrator)
//   - x3f_get_raw + x3f_get_thumb_{plain,huffman,jpeg} + x3f_get_camf +
//     x3f_get_prop (directory-entry searchers)
//
// All buffers freed here were `libc::malloc`/`calloc`/`realloc`-allocated
// in the Rust loader (`load.rs`, `io.rs`, `entropy.rs`), so the matching
// deallocator is `libc::free`. Once this lands, `src/x3f_io.c` holds no
// function bodies and is dropped from the cc-rs source list — the file
// stays on disk only so the comment header documents the port history.
//
// This is the prerequisite step for the deeper M5e refactor (replacing
// `x3f_area16_t.{buf,data}` with a `Vec<u16>` + `Plane<'a, T>` view): with
// allocation AND cleanup both in Rust, ownership can move into a `Box<[u16]>`
// or `Vec<u16>` without needing FFI symbol changes.
// =============================================================================

// `legacy_offset` / `auto_legacy_offset` — global tunables for the
// older Huffman decoder. Originally `int legacy_offset = 0;` and
// `bool_t auto_legacy_offset = 1;` in src/x3f_io.c. The bindgen-
// generated extern declarations in entropy.rs and globals.rs resolve
// to these definitions at link time.
#[no_mangle]
pub static mut legacy_offset: libc::c_int = 0;

#[no_mangle]
pub static mut auto_legacy_offset: libc::c_int = 1;

unsafe fn cleanup_huffman_tree(htp: *mut x3f_hufftree_t) {
    unsafe {
        if !(*htp).nodes.is_null() {
            libc::free((*htp).nodes as *mut libc::c_void);
            (*htp).nodes = ptr::null_mut();
        }
    }
}

unsafe fn cleanup_true(trup: *mut *mut x3f_true_t) {
    unsafe {
        let tru = *trup;
        if tru.is_null() {
            return;
        }
        x3f_printf(x3f_verbosity_t_DEBUG, c"Cleanup TRUE data\n".as_ptr());

        if !(*tru).table.element.is_null() {
            libc::free((*tru).table.element as *mut libc::c_void);
            (*tru).table.element = ptr::null_mut();
        }
        if !(*tru).plane_size.element.is_null() {
            libc::free((*tru).plane_size.element as *mut libc::c_void);
            (*tru).plane_size.element = ptr::null_mut();
        }
        cleanup_huffman_tree(&mut (*tru).tree);
        if !(*tru).x3rgb16.buf.is_null() {
            libc::free((*tru).x3rgb16.buf as *mut libc::c_void);
            (*tru).x3rgb16.buf = ptr::null_mut();
        }

        libc::free(tru as *mut libc::c_void);
        *trup = ptr::null_mut();
    }
}

unsafe fn cleanup_quattro(qp: *mut *mut x3f_quattro_t) {
    unsafe {
        let q = *qp;
        if q.is_null() {
            return;
        }
        x3f_printf(x3f_verbosity_t_DEBUG, c"Cleanup Quattro\n".as_ptr());

        if !(*q).top16.buf.is_null() {
            libc::free((*q).top16.buf as *mut libc::c_void);
            (*q).top16.buf = ptr::null_mut();
        }
        libc::free(q as *mut libc::c_void);
        *qp = ptr::null_mut();
    }
}

unsafe fn cleanup_huffman(hufp: *mut *mut x3f_huffman_t) {
    unsafe {
        let huf = *hufp;
        if huf.is_null() {
            return;
        }
        x3f_printf(x3f_verbosity_t_DEBUG, c"Cleanup Huffman\n".as_ptr());

        if !(*huf).mapping.element.is_null() {
            libc::free((*huf).mapping.element as *mut libc::c_void);
            (*huf).mapping.element = ptr::null_mut();
        }
        if !(*huf).table.element.is_null() {
            libc::free((*huf).table.element as *mut libc::c_void);
            (*huf).table.element = ptr::null_mut();
        }
        cleanup_huffman_tree(&mut (*huf).tree);
        if !(*huf).row_offsets.element.is_null() {
            libc::free((*huf).row_offsets.element as *mut libc::c_void);
            (*huf).row_offsets.element = ptr::null_mut();
        }
        if !(*huf).rgb8.buf.is_null() {
            libc::free((*huf).rgb8.buf as *mut libc::c_void);
            (*huf).rgb8.buf = ptr::null_mut();
        }
        if !(*huf).x3rgb16.buf.is_null() {
            libc::free((*huf).x3rgb16.buf as *mut libc::c_void);
            (*huf).x3rgb16.buf = ptr::null_mut();
        }
        libc::free(huf as *mut libc::c_void);
        *hufp = ptr::null_mut();
    }
}

unsafe fn free_camf_entry(entry: *mut camf_entry_t) {
    unsafe {
        if !(*entry).property_name.is_null() {
            libc::free((*entry).property_name as *mut libc::c_void);
            (*entry).property_name = ptr::null_mut();
        }
        if !(*entry).property_value.is_null() {
            libc::free((*entry).property_value as *mut libc::c_void);
            (*entry).property_value = ptr::null_mut();
        }
        if !(*entry).matrix_decoded.is_null() {
            libc::free((*entry).matrix_decoded as *mut libc::c_void);
            (*entry).matrix_decoded = ptr::null_mut();
        }
        if !(*entry).matrix_dim_entry.is_null() {
            libc::free((*entry).matrix_dim_entry as *mut libc::c_void);
            (*entry).matrix_dim_entry = ptr::null_mut();
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn x3f_delete(x3f: *mut x3f_t) -> x3f_return_t {
    if x3f.is_null() {
        return x3f_return_e_X3F_ARGUMENT_ERROR;
    }
    unsafe {
        x3f_printf(x3f_verbosity_t_DEBUG, c"X3F Delete\n".as_ptr());

        let ds = &mut (*x3f).directory_section;
        for d in 0..ds.num_directory_entries as usize {
            let de = ds.directory_entry.add(d);
            let deh = &mut (*de).header;

            match deh.identifier {
                X3F_SECP => {
                    let pl = &mut deh.data_subsection.property_list;
                    for i in 0..pl.property_table.size as usize {
                        let p = pl.property_table.element.add(i);
                        if !(*p).name_utf8.is_null() {
                            libc::free((*p).name_utf8 as *mut libc::c_void);
                            (*p).name_utf8 = ptr::null_mut();
                        }
                        if !(*p).value_utf8.is_null() {
                            libc::free((*p).value_utf8 as *mut libc::c_void);
                            (*p).value_utf8 = ptr::null_mut();
                        }
                    }
                    if !pl.property_table.element.is_null() {
                        libc::free(pl.property_table.element as *mut libc::c_void);
                        pl.property_table.element = ptr::null_mut();
                    }
                    if !pl.data.is_null() {
                        libc::free(pl.data as *mut libc::c_void);
                        pl.data = ptr::null_mut();
                    }
                }
                X3F_SECI => {
                    let id = &mut deh.data_subsection.image_data;
                    cleanup_huffman(&mut id.huffman);
                    cleanup_true(&mut id.tru);
                    cleanup_quattro(&mut id.quattro);
                    if !id.data.is_null() {
                        libc::free(id.data as *mut libc::c_void);
                        id.data = ptr::null_mut();
                    }
                }
                X3F_SECC => {
                    let camf = &mut deh.data_subsection.camf;
                    if !camf.data.is_null() {
                        libc::free(camf.data as *mut libc::c_void);
                        camf.data = ptr::null_mut();
                    }
                    if !camf.table.element.is_null() {
                        libc::free(camf.table.element as *mut libc::c_void);
                        camf.table.element = ptr::null_mut();
                    }
                    cleanup_huffman_tree(&mut camf.tree);
                    if !camf.decoded_data.is_null() {
                        libc::free(camf.decoded_data as *mut libc::c_void);
                        camf.decoded_data = ptr::null_mut();
                    }
                    for i in 0..camf.entry_table.size as usize {
                        free_camf_entry(camf.entry_table.element.add(i));
                    }
                    if !camf.entry_table.element.is_null() {
                        libc::free(camf.entry_table.element as *mut libc::c_void);
                        camf.entry_table.element = ptr::null_mut();
                    }
                }
                _ => {}
            }
        }

        if !ds.directory_entry.is_null() {
            libc::free(ds.directory_entry as *mut libc::c_void);
            ds.directory_entry = ptr::null_mut();
        }
        libc::free(x3f as *mut libc::c_void);
    }
    x3f_return_e_X3F_OK
}

// Section/image-type identifier constants (mirrors of the X3F_* macros
// in src/x3f_io.h). Bindgen exposes these as `u32` constants but the
// names collide with the locals already used in `x3f_new_from_file`,
// so we keep ones we share:
const X3F_IMAGE_RAW_HUFFMAN_X530: u32 = 0x0003_0005;
const X3F_IMAGE_RAW_HUFFMAN_10BIT: u32 = 0x0003_0006;
const X3F_IMAGE_RAW_TRUE: u32 = 0x0003_001e;
const X3F_IMAGE_RAW_MERRILL: u32 = 0x0001_001e;
const X3F_IMAGE_RAW_QUATTRO: u32 = 0x0001_0023;
const X3F_IMAGE_RAW_SDQ: u32 = 0x0001_0025;
const X3F_IMAGE_RAW_SDQH: u32 = 0x0001_0027;
const X3F_IMAGE_THUMB_PLAIN: u32 = 0x0002_0003;
const X3F_IMAGE_THUMB_HUFFMAN: u32 = 0x0002_000b;
const X3F_IMAGE_THUMB_JPEG: u32 = 0x0002_0012;

/// First directory entry of `identifier`-type, optionally restricted to
/// matching `image_type` for SECi entries. Mirrors `x3f_get` in
/// src/x3f_io.c. Returns NULL when nothing matches.
unsafe fn x3f_get_de(
    x3f: *mut x3f_t,
    identifier: u32,
    image_type: u32,
) -> *mut x3f_directory_entry_t {
    if x3f.is_null() {
        return ptr::null_mut();
    }
    unsafe {
        let ds = &(*x3f).directory_section;
        for d in 0..ds.num_directory_entries as usize {
            let de = ds.directory_entry.add(d);
            let deh = &(*de).header;
            if deh.identifier == identifier {
                if identifier == X3F_SECI {
                    let id = &deh.data_subsection.image_data;
                    if id.type_format == image_type {
                        return de;
                    }
                } else {
                    return de;
                }
            }
        }
    }
    ptr::null_mut()
}

#[no_mangle]
pub unsafe extern "C" fn x3f_get_raw(x3f: *mut x3f_t) -> *mut x3f_directory_entry_t {
    unsafe {
        for &t in &[
            X3F_IMAGE_RAW_HUFFMAN_X530,
            X3F_IMAGE_RAW_HUFFMAN_10BIT,
            X3F_IMAGE_RAW_TRUE,
            X3F_IMAGE_RAW_MERRILL,
            X3F_IMAGE_RAW_QUATTRO,
            X3F_IMAGE_RAW_SDQ,
            X3F_IMAGE_RAW_SDQH,
        ] {
            let de = x3f_get_de(x3f, X3F_SECI, t);
            if !de.is_null() {
                return de;
            }
        }
    }
    ptr::null_mut()
}

#[no_mangle]
pub unsafe extern "C" fn x3f_get_thumb_plain(x3f: *mut x3f_t) -> *mut x3f_directory_entry_t {
    unsafe { x3f_get_de(x3f, X3F_SECI, X3F_IMAGE_THUMB_PLAIN) }
}

#[no_mangle]
pub unsafe extern "C" fn x3f_get_thumb_huffman(x3f: *mut x3f_t) -> *mut x3f_directory_entry_t {
    unsafe { x3f_get_de(x3f, X3F_SECI, X3F_IMAGE_THUMB_HUFFMAN) }
}

#[no_mangle]
pub unsafe extern "C" fn x3f_get_thumb_jpeg(x3f: *mut x3f_t) -> *mut x3f_directory_entry_t {
    unsafe { x3f_get_de(x3f, X3F_SECI, X3F_IMAGE_THUMB_JPEG) }
}

#[no_mangle]
pub unsafe extern "C" fn x3f_get_camf(x3f: *mut x3f_t) -> *mut x3f_directory_entry_t {
    unsafe { x3f_get_de(x3f, X3F_SECC, 0) }
}

#[no_mangle]
pub unsafe extern "C" fn x3f_get_prop(x3f: *mut x3f_t) -> *mut x3f_directory_entry_t {
    unsafe { x3f_get_de(x3f, X3F_SECP, 0) }
}

// Cross-crate DCE anchors so the `#[no_mangle]` symbols survive LTO.
#[used]
static _A_DELETE: unsafe extern "C" fn(*mut x3f_t) -> x3f_return_t = x3f_delete;
#[used]
static _A_GET_RAW: unsafe extern "C" fn(*mut x3f_t) -> *mut x3f_directory_entry_t = x3f_get_raw;
#[used]
static _A_GET_THUMB_PLAIN: unsafe extern "C" fn(*mut x3f_t) -> *mut x3f_directory_entry_t =
    x3f_get_thumb_plain;
#[used]
static _A_GET_THUMB_HUFFMAN: unsafe extern "C" fn(*mut x3f_t) -> *mut x3f_directory_entry_t =
    x3f_get_thumb_huffman;
#[used]
static _A_GET_THUMB_JPEG: unsafe extern "C" fn(*mut x3f_t) -> *mut x3f_directory_entry_t =
    x3f_get_thumb_jpeg;
#[used]
static _A_GET_CAMF: unsafe extern "C" fn(*mut x3f_t) -> *mut x3f_directory_entry_t = x3f_get_camf;
#[used]
static _A_GET_PROP: unsafe extern "C" fn(*mut x3f_t) -> *mut x3f_directory_entry_t = x3f_get_prop;
