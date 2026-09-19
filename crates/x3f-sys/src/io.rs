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

/// Parse a container without terminating the embedding process on invalid data.
/// The caller retains ownership of `infile` and owns the returned X3F handle.
pub unsafe fn new_from_file(
    infile: *mut FILE,
    control: crate::control::Control<'_>,
) -> crate::control::Result<*mut x3f_t> {
    use crate::control::Error;
    use crate::parse::{alloc, Input};
    let mut input = unsafe { Input::new(infile.cast(), control) }?;
    let x3f = unsafe { alloc::<x3f_t>(1) }?;
    unsafe { (*x3f).info.input.file = infile };
    let parsed = (|| -> crate::control::Result<()> {
        unsafe {
            let h = &mut (*x3f).header;
            h.identifier = input.u32()?;
            if h.identifier != X3F_FOVB {
                return Err(Error::InvalidData("invalid X3F signature"));
            }
            h.version = input.u32()?;
            input.read(&mut h.unique_identifier)?;
            if h.version < X3F_VERSION_4_0 {
                h.mark_bits = input.u32()?;
                h.columns = input.u32()?;
                h.rows = input.u32()?;
                h.rotation = input.u32()?;
                if h.version >= X3F_VERSION_2_1 {
                    let n = if h.version >= X3F_VERSION_3_0 {
                        NUM_EXT_DATA_3_0
                    } else {
                        NUM_EXT_DATA_2_1
                    };
                    input.read(std::slice::from_raw_parts_mut(
                        h.white_balance.as_mut_ptr().cast(),
                        SIZE_WHITE_BALANCE,
                    ))?;
                    if !h.white_balance.contains(&0) {
                        return Err(Error::InvalidData("unterminated white balance"));
                    }
                    if h.version >= X3F_VERSION_2_3 {
                        input.read(std::slice::from_raw_parts_mut(
                            h.color_mode.as_mut_ptr().cast(),
                            SIZE_COLOR_MODE,
                        ))?;
                        if !h.color_mode.contains(&0) {
                            return Err(Error::InvalidData("unterminated color mode"));
                        }
                    }
                    input.read(&mut h.extended_types[..n])?;
                    for value in &mut h.extended_data[..n] {
                        *value = f32::from_bits(input.u32()?);
                    }
                }
            }
            let header_end = input.position();
            let directory_tail = input
                .end()
                .checked_sub(4)
                .ok_or(Error::InvalidData("missing directory"))?;
            input.seek(directory_tail)?;
            let directory_offset = input.u32()? as u64;
            if directory_offset < header_end {
                return Err(Error::InvalidData("directory overlaps header"));
            }
            input.seek(directory_offset)?;
            let ds = &mut (*x3f).directory_section;
            ds.identifier = input.u32()?;
            if ds.identifier != 0x6443_4553 {
                return Err(Error::InvalidData("invalid directory signature"));
            }
            ds.version = input.u32()?;
            let count = input.u32()? as usize;
            if count > input.remaining().saturating_sub(4) / 12 {
                return Err(Error::InvalidData("directory count exceeds input"));
            }
            ds.directory_entry = alloc::<x3f_directory_entry_t>(count)?;
            ds.num_directory_entries = count as u32;
            for i in 0..count {
                control.check()?;
                let de = &mut *ds.directory_entry.add(i);
                de.input.offset = input.u32()?;
                de.input.size = input.u32()?;
                de.type_ = input.u32()?;
                let next = input.position();
                let start = de.input.offset as u64;
                let end = start + de.input.size as u64;
                if start < header_end || end > directory_offset || de.input.size < 8 {
                    return Err(Error::InvalidData("section outside container data"));
                }
                input.seek(start)?;
                de.header.identifier = input.u32()?;
                de.header.version = input.u32()?;
                match de.header.identifier {
                    X3F_SECP => {
                        if de.input.size < 24 {
                            return Err(Error::InvalidData("truncated property header"));
                        }
                        let pl = &mut de.header.data_subsection.property_list;
                        pl.num_properties = input.u32()?;
                        pl.character_format = input.u32()?;
                        pl.reserved = input.u32()?;
                        pl.total_length = input.u32()?;
                    }
                    X3F_SECI => {
                        if de.input.size < 28 {
                            return Err(Error::InvalidData("truncated image header"));
                        }
                        let id = &mut de.header.data_subsection.image_data;
                        id.type_ = input.u32()?;
                        id.format = input.u32()?;
                        if id.type_ > 0xffff || id.format > 0xffff {
                            return Err(Error::InvalidData("invalid image format"));
                        }
                        id.type_format = (id.type_ << 16) | id.format;
                        id.columns = input.u32()?;
                        id.rows = input.u32()?;
                        id.row_stride = input.u32()?;
                        if id.columns == 0 || id.rows == 0 {
                            return Err(Error::InvalidData("empty image dimensions"));
                        }
                        crate::parse::checked_size(id.columns as usize, id.rows as usize)
                            .and_then(|pixels| crate::parse::checked_size(pixels, 6))?;
                    }
                    X3F_SECC => {
                        if de.input.size < 28 {
                            return Err(Error::InvalidData("truncated CAMF header"));
                        }
                        let camf = &mut de.header.data_subsection.camf;
                        camf.type_ = input.u32()?;
                        camf.__bindgen_anon_1.tN.val0 = input.u32()?;
                        camf.__bindgen_anon_1.tN.val1 = input.u32()?;
                        camf.__bindgen_anon_1.tN.val2 = input.u32()?;
                        camf.__bindgen_anon_1.tN.val3 = input.u32()?;
                    }
                    _ => {}
                }
                input.seek(next)?;
            }
        }
        Ok(())
    })();
    if let Err(error) = parsed {
        unsafe { x3f_delete(x3f) };
        return Err(error);
    }
    Ok(x3f)
}

#[no_mangle]
pub unsafe extern "C" fn x3f_new_from_file(infile: *mut FILE) -> *mut x3f_t {
    unsafe { new_from_file(infile, crate::control::Control::none()) }.unwrap_or(ptr::null_mut())
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

pub(crate) unsafe fn cleanup_entry(de: *mut x3f_directory_entry_t) {
    unsafe {
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
                pl.property_table.size = 0;
                if !pl.data.is_null() {
                    libc::free(pl.data as *mut libc::c_void);
                    pl.data = ptr::null_mut();
                }
                pl.data_size = 0;
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
                id.data_size = 0;
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
                camf.data_size = 0;
                camf.table.size = 0;
                camf.tree.free_node_index = 0;
                camf.decoded_data_size = 0;
                camf.entry_table.size = 0;
                camf.decoding_start = ptr::null_mut();
                camf.decoding_size = 0;
            }
            _ => {}
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
            cleanup_entry(de);
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

#[cfg(all(test, not(target_arch = "wasm32")))]
mod parser_tests {
    use super::*;
    use crate::control::{Control, Error};
    use std::sync::atomic::AtomicBool;

    fn with_file(bytes: &[u8], f: impl FnOnce(*mut FILE)) {
        unsafe {
            let file = libc::tmpfile();
            assert!(!file.is_null());
            assert_eq!(
                libc::fwrite(bytes.as_ptr().cast(), 1, bytes.len(), file),
                bytes.len()
            );
            f(file.cast());
            libc::fclose(file);
        }
    }

    fn container(section: &[u8]) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"FOVb");
        bytes.extend_from_slice(&X3F_VERSION_4_0.to_le_bytes());
        bytes.extend_from_slice(&[0; 16]);
        bytes.extend_from_slice(section);
        let directory = bytes.len() as u32;
        bytes.extend_from_slice(b"SECd");
        bytes.extend_from_slice(&0x20000u32.to_le_bytes());
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&24u32.to_le_bytes());
        bytes.extend_from_slice(&(section.len() as u32).to_le_bytes());
        bytes.extend_from_slice(b"PROP");
        bytes.extend_from_slice(&directory.to_le_bytes());
        bytes
    }

    #[test]
    fn malformed_containers_return_errors_and_can_be_retried() {
        let mut property = Vec::new();
        for word in [X3F_SECP, 0x20000, 1, 0, 0, 0, u32::MAX, 0, 0] {
            property.extend_from_slice(&word.to_le_bytes());
        }
        let good_header = container(&property);
        for bytes in [
            &b"FOVb"[..],
            &good_header[..20],
            &good_header[..good_header.len() - 2],
        ] {
            with_file(bytes, |file| unsafe {
                assert!(new_from_file(file, Control::none()).is_err());
                assert!(x3f_new_from_file(file).is_null());
            });
        }
        with_file(&good_header, |file| unsafe {
            let reader = new_from_file(file, Control::none()).unwrap();
            let section = x3f_get_prop(reader);
            for _ in 0..3 {
                assert!(crate::load_data(reader, section, Control::none()).is_err());
                let pl = (*section).header.data_subsection.property_list;
                assert!(pl.data.is_null());
                assert!(pl.property_table.element.is_null());
            }
            assert!(matches!(
                crate::load_data(reader, section, Control::new(&AtomicBool::new(true))),
                Err(Error::Cancelled)
            ));
            x3f_delete(reader);
        });
        with_file(&good_header, |file| unsafe {
            assert!(matches!(
                new_from_file(file, Control::new(&AtomicBool::new(true))),
                Err(Error::Cancelled)
            ));
        });
    }

    #[test]
    fn directory_count_and_section_ranges_are_bounded() {
        let mut section = Vec::new();
        for word in [X3F_SECP, 0x20000, 0, 0, 0, 0] {
            section.extend_from_slice(&word.to_le_bytes());
        }
        let bytes = container(&section);
        let directory = 24 + section.len();
        for (offset, value) in [
            (directory + 8, u32::MAX),
            (directory + 12, 0),
            (directory + 16, u32::MAX),
        ] {
            let mut corrupt = bytes.clone();
            corrupt[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
            with_file(&corrupt, |file| unsafe {
                assert!(new_from_file(file, Control::none()).is_err());
            });
        }
        with_file(&bytes, |file| unsafe {
            let reader = new_from_file(file, Control::none()).unwrap();
            crate::load_data(reader, x3f_get_prop(reader), Control::none()).unwrap();
            x3f_delete(reader);
        });
    }

    #[test]
    fn raw_block_can_be_decoded_and_malformed_true_data_is_recoverable() {
        let mut section = Vec::new();
        for word in [X3F_SECI, 0x20000, 1, 30, 2, 2, 0] {
            section.extend_from_slice(&word.to_le_bytes());
        }
        for word in [123u16, 123, 123, 0] {
            section.extend_from_slice(&word.to_le_bytes());
        }
        section.extend_from_slice(&[1, 0, 1, 128, 0, 0]);
        for _ in 0..3 {
            section.extend_from_slice(&1u32.to_le_bytes());
        }
        section.extend_from_slice(&[0; 33]);
        with_file(&container(&section), |file| unsafe {
            let reader = new_from_file(file, Control::none()).unwrap();
            let raw = x3f_get_raw(reader);
            crate::load_image_block(reader, raw, Control::none()).unwrap();
            crate::load_data(reader, raw, Control::none()).unwrap();
            let tru = &*(*raw).header.data_subsection.image_data.tru;
            assert!(std::slice::from_raw_parts(tru.x3rgb16.data, 12)
                .iter()
                .all(|&v| v == 123));
            x3f_delete(reader);
        });
        for offset in [36, 42] {
            let mut malformed = section.clone();
            malformed[offset] = 255;
            with_file(&container(&malformed), |file| unsafe {
                let reader = new_from_file(file, Control::none()).unwrap();
                let raw = x3f_get_raw(reader);
                assert!(crate::load_data(reader, raw, Control::none()).is_err());
                let id = (*raw).header.data_subsection.image_data;
                assert!(id.data.is_null() && id.tru.is_null());
                x3f_delete(reader);
            });
        }
    }
}
