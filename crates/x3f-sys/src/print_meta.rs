//! M4b — native Rust port of `src/x3f_print_meta.c`.
//!
//! Pretty-prints the parsed `x3f_t` to a file for the `-meta` CLI flag,
//! and to stdout for the (unused-in-main-flow) `x3f_print_meta` debug
//! function.
//!
//! Strict byte-for-byte port: the tier-2 `merrill_meta_md5` and
//! `quattro_meta_md5` hashes pin the legacy text format. We use
//! `libc::fopen` / `libc::fprintf` / `libc::fclose` directly so the C
//! `printf`-style format strings (`%08x`, `%9f`, `%12g`, …) yield the
//! exact same bytes — Rust's f64 Display uses Grisu/Ryu shortest-round-
//! trip and would diverge on `%g` and `%9f` formatting.
//!
//! The bindgen forward declarations for `x3f_dump_meta_data`,
//! `x3f_print_meta`, and `max_printed_matrix_elements` are blocklisted
//! in build.rs; lib.rs re-exports the Rust impls under the same path.

use std::os::raw::{c_char, c_int};

use crate::*;

// Version constants. The C `#define X3F_VERSION_2_1 X3F_VERSION(2,1)` form
// expands to a function-style macro that bindgen does not translate, so
// they're inlined here.
const X3F_VERSION_2_1: u32 = (2 << 16) + 1;
const X3F_VERSION_2_3: u32 = (2 << 16) + 3;
const X3F_VERSION_3_0: u32 = 3 << 16;
const X3F_VERSION_4_0: u32 = 4 << 16;

// Section identifiers — same FOURCC trick as in meta.rs.
const X3F_SECP: u32 = 0x7043_4553; // SECp
const X3F_SECI: u32 = 0x6943_4553; // SECi
const X3F_SECC: u32 = 0x6343_4553; // SECc

/// Caller-cap on how many matrix elements to print before truncating with
/// `... (N skipped) ...`. Mutable global; the CLI lets users override via
/// `-matrixmax`. Defined in Rust now — the bindgen forward declaration is
/// blocklisted, the Rust impl is re-exported from `lib.rs`.
#[no_mangle]
pub static mut max_printed_matrix_elements: u32 = 100;

/// Convert a 32-bit FOURCC-style identifier into a NUL-terminated 4-char
/// C string. Caller-supplied buffer; the C version used a static buffer
/// shared across calls, but we plumb a per-call buffer for thread safety.
#[inline]
unsafe fn id_to_cstr(id: u32, buf: &mut [u8; 5]) -> *const c_char {
    buf[0] = (id & 0xff) as u8;
    buf[1] = ((id >> 8) & 0xff) as u8;
    buf[2] = ((id >> 16) & 0xff) as u8;
    buf[3] = ((id >> 24) & 0xff) as u8;
    buf[4] = 0;
    buf.as_ptr() as *const c_char
}

unsafe fn print_matrix_element(f_out: *mut libc::FILE, entry: &camf_entry_t, i: u32) {
    match entry.matrix_decoded_type {
        matrix_type_t_M_FLOAT => {
            let v = unsafe { *(entry.matrix_decoded as *const f64).add(i as usize) };
            unsafe { libc::fprintf(f_out, c"%12g ".as_ptr(), v) };
        }
        matrix_type_t_M_INT => {
            let v = unsafe { *(entry.matrix_decoded as *const i32).add(i as usize) };
            unsafe { libc::fprintf(f_out, c"%12d ".as_ptr(), v) };
        }
        matrix_type_t_M_UINT => {
            let v = unsafe { *(entry.matrix_decoded as *const u32).add(i as usize) };
            // C uses %12d to print the uint as signed; we mirror.
            unsafe { libc::fprintf(f_out, c"%12d ".as_ptr(), v) };
        }
        _ => {}
    }
}

unsafe fn print_matrix(f_out: *mut libc::FILE, entry: &camf_entry_t) {
    let dim = entry.matrix_dim;
    let dims = entry.matrix_dim_entry;
    let linesize = unsafe { (*dims.add((dim - 1) as usize)).size };
    let mut blocksize: u32 = u32::MAX;
    let totalsize = entry.matrix_elements;

    match entry.matrix_decoded_type {
        matrix_type_t_M_FLOAT => unsafe { libc::fprintf(f_out, c"float ".as_ptr()) },
        matrix_type_t_M_INT => unsafe { libc::fprintf(f_out, c"integer ".as_ptr()) },
        matrix_type_t_M_UINT => unsafe { libc::fprintf(f_out, c"unsigned integer ".as_ptr()) },
        _ => 0,
    };

    match dim {
        1 => unsafe {
            let d0 = (*dims.add(0)).size;
            libc::fprintf(f_out, c"[%d]\n".as_ptr(), d0);
            libc::fprintf(f_out, c"x: %s\n".as_ptr(), (*dims.add(0)).name);
        },
        2 => unsafe {
            let d0 = (*dims.add(0)).size;
            let d1 = (*dims.add(1)).size;
            libc::fprintf(f_out, c"[%d][%d]\n".as_ptr(), d0, d1);
            libc::fprintf(f_out, c"x: %s\n".as_ptr(), (*dims.add(1)).name);
            libc::fprintf(f_out, c"y: %s\n".as_ptr(), (*dims.add(0)).name);
        },
        3 => unsafe {
            let d0 = (*dims.add(0)).size;
            let d1 = (*dims.add(1)).size;
            let d2 = (*dims.add(2)).size;
            libc::fprintf(f_out, c"[%d][%d][%d]\n".as_ptr(), d0, d1, d2);
            libc::fprintf(f_out, c"x: %s\n".as_ptr(), (*dims.add(2)).name);
            libc::fprintf(f_out, c"y: %s\n".as_ptr(), (*dims.add(1)).name);
            libc::fprintf(f_out, c"z: %s (i.e. group)\n".as_ptr(), (*dims.add(0)).name);
            blocksize = linesize * (*dims.add((dim - 2) as usize)).size;
        },
        _ => unsafe {
            libc::fprintf(
                f_out,
                c"\nNot support for higher than 3D in printout\n".as_ptr(),
            );
            libc::fprintf(
                libc_stderr(),
                c"Not support for higher than 3D in printout\n".as_ptr(),
            );
        },
    }

    let cap = unsafe { max_printed_matrix_elements };
    for i in 0..totalsize {
        unsafe { print_matrix_element(f_out, entry, i) };
        if (i + 1) % linesize == 0 {
            unsafe { libc::fprintf(f_out, c"\n".as_ptr()) };
        }
        if (i + 1) % blocksize == 0 {
            unsafe { libc::fprintf(f_out, c"\n".as_ptr()) };
        }
        if i >= cap.saturating_sub(1) {
            unsafe {
                libc::fprintf(
                    f_out,
                    c"\n... (%d skipped) ...\n".as_ptr(),
                    totalsize - i - 1,
                )
            };
            break;
        }
    }
}

#[inline]
fn libc_stderr() -> *mut libc::FILE {
    // We avoid relying on the platform-specific `stderr` FILE* extern (macOS:
    // `__stderrp`, Linux: `stderr`) by reopening fd 2. Only used by the >3D
    // matrix diagnostic, which the corpus never triggers — fd 2 is always
    // available and the resulting FILE* is leaked, which is fine for a
    // never-actually-printed message.
    unsafe { libc::fdopen(2, c"w".as_ptr()) }
}

unsafe fn print_file_header_meta_data(f_out: *mut libc::FILE, x3f: *mut x3f_t) {
    let h = unsafe { &(*x3f).header };
    unsafe {
        libc::fprintf(f_out, c"BEGIN: file header meta data\n\n".as_ptr());
        libc::fprintf(f_out, c"header.\n".as_ptr());
    }
    let mut idbuf = [0u8; 5];
    let id_s = unsafe { id_to_cstr(h.identifier, &mut idbuf) };
    unsafe {
        libc::fprintf(
            f_out,
            c"  identifier        = %08x (%s)\n".as_ptr(),
            h.identifier,
            id_s,
        );
        libc::fprintf(f_out, c"  version           = %08x\n".as_ptr(), h.version);
    }
    if h.version < X3F_VERSION_4_0 {
        unsafe {
            libc::fprintf(
                f_out,
                c"  unique_identifier = %02x...\n".as_ptr(),
                h.unique_identifier[0] as u32,
            );
            libc::fprintf(f_out, c"  mark_bits         = %08x\n".as_ptr(), h.mark_bits);
            libc::fprintf(
                f_out,
                c"  columns           = %08x (%d)\n".as_ptr(),
                h.columns,
                h.columns,
            );
            libc::fprintf(
                f_out,
                c"  rows              = %08x (%d)\n".as_ptr(),
                h.rows,
                h.rows,
            );
            libc::fprintf(
                f_out,
                c"  rotation          = %08x (%d)\n".as_ptr(),
                h.rotation,
                h.rotation,
            );
        }
        if h.version >= X3F_VERSION_2_1 {
            let num_ext = if h.version >= X3F_VERSION_3_0 {
                NUM_EXT_DATA_3_0
            } else {
                NUM_EXT_DATA_2_1
            };
            unsafe {
                libc::fprintf(
                    f_out,
                    c"  white_balance     = %s\n".as_ptr(),
                    h.white_balance.as_ptr(),
                );
            }
            if h.version >= X3F_VERSION_2_3 {
                unsafe {
                    libc::fprintf(
                        f_out,
                        c"  color_mode        = %s\n".as_ptr(),
                        h.color_mode.as_ptr(),
                    );
                }
            }
            unsafe { libc::fprintf(f_out, c"  extended_types\n".as_ptr()) };
            for i in 0..num_ext as usize {
                let typ = h.extended_types[i] as u32;
                let data = h.extended_data[i] as f64;
                unsafe {
                    libc::fprintf(
                        f_out,
                        c"    %2d: %3d = %9f\n".as_ptr(),
                        i as c_int,
                        typ,
                        data,
                    );
                }
            }
        }
    }
    unsafe { libc::fprintf(f_out, c"END: file header meta data\n\n".as_ptr()) };
}

unsafe fn print_camf_meta_data2(f_out: *mut libc::FILE, camf: *const x3f_camf_t) {
    unsafe { libc::fprintf(f_out, c"BEGIN: CAMF meta data\n\n".as_ptr()) };

    let table = unsafe { (*camf).entry_table };
    if table.size != 0 {
        let entries = unsafe { std::slice::from_raw_parts(table.element, table.size as usize) };
        let stdout = libc_stdout();
        for (i, entry) in entries.iter().enumerate() {
            if f_out == stdout {
                let mut idbuf = [0u8; 5];
                let id_s = unsafe { id_to_cstr(entry.id, &mut idbuf) };
                unsafe {
                    libc::fprintf(
                        f_out,
                        c"          element[%d].name = \"%s\"\n".as_ptr(),
                        i as c_int,
                        entry.name_address,
                    );
                    libc::fprintf(
                        f_out,
                        c"            id = %x (%s)\n".as_ptr(),
                        entry.id,
                        id_s,
                    );
                    libc::fprintf(
                        f_out,
                        c"            entry_size = %d\n".as_ptr(),
                        entry.entry_size,
                    );
                    libc::fprintf(
                        f_out,
                        c"            name_size = %d\n".as_ptr(),
                        entry.name_size,
                    );
                    libc::fprintf(
                        f_out,
                        c"            value_size = %d\n".as_ptr(),
                        entry.value_size,
                    );
                }
            }

            if entry.text_size != 0 {
                unsafe {
                    libc::fprintf(
                        f_out,
                        c"BEGIN: CAMF text meta data (%s)\n".as_ptr(),
                        entry.name_address,
                    );
                    libc::fprintf(f_out, c"\"%s\"\n".as_ptr(), entry.text);
                    libc::fprintf(f_out, c"END: CAMF text meta data\n\n".as_ptr());
                }
            }

            if entry.property_num != 0 {
                unsafe {
                    libc::fprintf(
                        f_out,
                        c"BEGIN: CAMF property meta data (%s)\n".as_ptr(),
                        entry.name_address,
                    );
                }
                for j in 0..entry.property_num as usize {
                    let name = unsafe { *entry.property_name.add(j) };
                    let value = unsafe { *entry.property_value.add(j) };
                    unsafe {
                        libc::fprintf(
                            f_out,
                            c"              \"%s\" = \"%s\"\n".as_ptr(),
                            name,
                            value,
                        );
                    }
                }
                unsafe { libc::fprintf(f_out, c"END: CAMF property meta data\n\n".as_ptr()) };
            }

            if entry.matrix_dim != 0 {
                unsafe {
                    libc::fprintf(
                        f_out,
                        c"BEGIN: CAMF matrix meta data (%s)\n".as_ptr(),
                        entry.name_address,
                    );
                }
                if f_out == stdout {
                    let dentry = entry.matrix_dim_entry;
                    unsafe {
                        libc::fprintf(
                            f_out,
                            c"            matrix_type = %d\n".as_ptr(),
                            entry.matrix_type,
                        );
                        libc::fprintf(
                            f_out,
                            c"            matrix_dim = %d\n".as_ptr(),
                            entry.matrix_dim,
                        );
                        libc::fprintf(
                            f_out,
                            c"            matrix_data_off = %d\n".as_ptr(),
                            entry.matrix_data_off,
                        );
                    }
                    for j in 0..entry.matrix_dim as usize {
                        let de = unsafe { &*dentry.add(j) };
                        let oo: *const c_char = if (j as u32) == de.n {
                            c"".as_ptr()
                        } else {
                            c" (out of order)".as_ptr()
                        };
                        unsafe {
                            libc::fprintf(f_out, c"            %d\n".as_ptr(), j as c_int);
                            libc::fprintf(f_out, c"              size = %d\n".as_ptr(), de.size);
                            libc::fprintf(
                                f_out,
                                c"              name_offset = %d\n".as_ptr(),
                                de.name_offset,
                            );
                            libc::fprintf(f_out, c"              n = %d%s\n".as_ptr(), de.n, oo);
                            libc::fprintf(
                                f_out,
                                c"              name = \"%s\"\n".as_ptr(),
                                de.name,
                            );
                        }
                    }
                    unsafe {
                        libc::fprintf(
                            f_out,
                            c"            matrix_element_size = %d\n".as_ptr(),
                            entry.matrix_element_size,
                        );
                        libc::fprintf(
                            f_out,
                            c"            matrix_elements = %d\n".as_ptr(),
                            entry.matrix_elements,
                        );
                        libc::fprintf(
                            f_out,
                            c"            matrix_estimated_element_size = %g\n".as_ptr(),
                            entry.matrix_estimated_element_size,
                        );
                    }
                }
                unsafe { print_matrix(f_out, entry) };
                unsafe { libc::fprintf(f_out, c"END: CAMF matrix meta data\n\n".as_ptr()) };
            }
        }
    }
    unsafe { libc::fprintf(f_out, c"END: CAMF meta data\n\n".as_ptr()) };
}

unsafe fn print_camf_meta_data(f_out: *mut libc::FILE, x3f: *mut x3f_t) {
    let de = unsafe { x3f_get_camf(x3f) };
    if de.is_null() {
        unsafe { libc::fprintf(f_out, c"INFO: No CAMF meta data found\n\n".as_ptr()) };
        return;
    }
    let camf: *const x3f_camf_t = unsafe { &(*de).header.data_subsection.camf };
    unsafe { print_camf_meta_data2(f_out, camf) };
}

unsafe fn print_prop_meta_data2(f_out: *mut libc::FILE, pl: *const x3f_property_list_t) {
    unsafe { libc::fprintf(f_out, c"BEGIN: PROP meta data\n\n".as_ptr()) };
    let table = unsafe { (*pl).property_table };
    if table.size != 0 {
        let n = unsafe { (*pl).num_properties } as usize;
        let entries = unsafe { std::slice::from_raw_parts(table.element, n) };
        for (i, p) in entries.iter().enumerate() {
            unsafe {
                libc::fprintf(
                    f_out,
                    c"          [%d] \"%s\" = \"%s\"\n".as_ptr(),
                    i as c_int,
                    p.name_utf8,
                    p.value_utf8,
                );
            }
        }
    }
    unsafe { libc::fprintf(f_out, c"END: PROP meta data\n\n".as_ptr()) };
}

unsafe fn print_prop_meta_data(f_out: *mut libc::FILE, x3f: *mut x3f_t) {
    let de = unsafe { x3f_get_prop(x3f) };
    if de.is_null() {
        unsafe { libc::fprintf(f_out, c"INFO: No PROP meta data found\n\n".as_ptr()) };
        return;
    }
    let pl: *const x3f_property_list_t = unsafe { &(*de).header.data_subsection.property_list };
    unsafe { print_prop_meta_data2(f_out, pl) };
}

#[inline]
fn libc_stdout() -> *mut libc::FILE {
    unsafe { libc::fdopen(1, c"w".as_ptr()) }
}

/// Debug entry point — dumps to stdout. Not invoked by the main `-meta`
/// CLI flow; only `src/x3f_io_test.c` (a unit-test main not built by
/// cc-rs) called this. We port for FFI-symbol completeness.
#[no_mangle]
pub unsafe extern "C" fn x3f_print_meta(x3f: *mut x3f_t) {
    if x3f.is_null() {
        unsafe { libc::printf(c"Null x3f\n".as_ptr()) };
        return;
    }
    let info = unsafe { &(*x3f).info };
    unsafe {
        libc::printf(c"info.\n".as_ptr());
        libc::printf(c"  error = %s\n".as_ptr(), info.error);
        libc::printf(c"  input.\n".as_ptr());
        libc::printf(c"    file = %p\n".as_ptr(), info.input.file);
        libc::printf(c"  output.\n".as_ptr());
        libc::printf(c"    file = %p\n".as_ptr(), info.output.file);
    }

    let stdout = libc_stdout();
    unsafe { print_file_header_meta_data(stdout, x3f) };

    let ds = unsafe { &(*x3f).directory_section };
    let mut idbuf = [0u8; 5];
    let id_s = unsafe { id_to_cstr(ds.identifier, &mut idbuf) };
    unsafe {
        libc::printf(c"directory_section.\n".as_ptr());
        libc::printf(
            c"  identifier            = %08x (%s)\n".as_ptr(),
            ds.identifier,
            id_s,
        );
        libc::printf(c"  version               = %08x\n".as_ptr(), ds.version);
        libc::printf(
            c"  num_directory_entries = %08x\n".as_ptr(),
            ds.num_directory_entries,
        );
        libc::printf(
            c"  directory_entry       = %p\n".as_ptr(),
            ds.directory_entry,
        );
    }

    for d in 0..ds.num_directory_entries as isize {
        let de = unsafe { ds.directory_entry.offset(d) };
        let deh = unsafe { &(*de).header };
        let mut tbuf = [0u8; 5];
        let mut ibuf = [0u8; 5];
        let type_s = unsafe { id_to_cstr((*de).type_, &mut tbuf) };
        let ident_s = unsafe { id_to_cstr(deh.identifier, &mut ibuf) };
        unsafe {
            libc::printf(c"  directory_entry.\n".as_ptr());
            libc::printf(c"    input.\n".as_ptr());
            libc::printf(c"      offset = %08x\n".as_ptr(), (*de).input.offset);
            libc::printf(c"      size   = %08x\n".as_ptr(), (*de).input.size);
            libc::printf(c"    output.\n".as_ptr());
            libc::printf(c"      offset = %08x\n".as_ptr(), (*de).output.offset);
            libc::printf(c"      size   = %08x\n".as_ptr(), (*de).output.size);
            libc::printf(c"    type     = %08x (%s)\n".as_ptr(), (*de).type_, type_s);
            libc::printf(c"    header.\n".as_ptr());
            libc::printf(
                c"      identifier = %08x (%s)\n".as_ptr(),
                deh.identifier,
                ident_s,
            );
            libc::printf(c"      version    = %08x\n".as_ptr(), deh.version);
        }

        if deh.identifier == X3F_SECP {
            let pl = unsafe { &deh.data_subsection.property_list };
            unsafe {
                libc::printf(c"      data_subsection.property_list.\n".as_ptr());
                libc::printf(
                    c"        num_properties   = %08x\n".as_ptr(),
                    pl.num_properties,
                );
                libc::printf(
                    c"        character_format = %08x\n".as_ptr(),
                    pl.character_format,
                );
                libc::printf(c"        reserved         = %08x\n".as_ptr(), pl.reserved);
                libc::printf(
                    c"        total_length     = %08x\n".as_ptr(),
                    pl.total_length,
                );
                libc::printf(
                    c"    property_table       = %x %p\n".as_ptr(),
                    pl.property_table.size,
                    pl.property_table.element,
                );
                libc::printf(c"        data             = %p\n".as_ptr(), pl.data);
                libc::printf(c"        data_size        = %x\n".as_ptr(), pl.data_size);
                print_prop_meta_data2(stdout, pl);
            }
        }

        if deh.identifier == X3F_SECI {
            let id = unsafe { &deh.data_subsection.image_data };
            let huf = id.huffman;
            let tru = id.tru;
            let q = id.quattro;
            unsafe {
                libc::printf(c"      data_subsection.image_data.\n".as_ptr());
                libc::printf(c"        type        = %08x\n".as_ptr(), id.type_);
                libc::printf(c"        format      = %08x\n".as_ptr(), id.format);
                libc::printf(c"        type_format = %08x\n".as_ptr(), id.type_format);
                libc::printf(
                    c"        columns     = %08x (%d)\n".as_ptr(),
                    id.columns,
                    id.columns,
                );
                libc::printf(
                    c"        rows        = %08x (%d)\n".as_ptr(),
                    id.rows,
                    id.rows,
                );
                libc::printf(
                    c"        row_stride  = %08x (%d)\n".as_ptr(),
                    id.row_stride,
                    id.row_stride,
                );

                if huf.is_null() {
                    libc::printf(c"        huffman     = %p\n".as_ptr(), huf);
                } else {
                    let h = &*huf;
                    libc::printf(c"        huffman->\n".as_ptr());
                    libc::printf(
                        c"          mapping     = %x %p\n".as_ptr(),
                        h.mapping.size,
                        h.mapping.element,
                    );
                    libc::printf(
                        c"          table       = %x %p\n".as_ptr(),
                        h.table.size,
                        h.table.element,
                    );
                    libc::printf(
                        c"          tree        = %d %p\n".as_ptr(),
                        h.tree.free_node_index,
                        h.tree.nodes,
                    );
                    libc::printf(
                        c"          row_offsets = %x %p\n".as_ptr(),
                        h.row_offsets.size,
                        h.row_offsets.element,
                    );
                    libc::printf(
                        c"          rgb8        = %x %x %p\n".as_ptr(),
                        h.rgb8.columns,
                        h.rgb8.rows,
                        h.rgb8.data,
                    );
                    libc::printf(
                        c"          x3rgb16     = %x %x %p\n".as_ptr(),
                        h.x3rgb16.columns,
                        h.x3rgb16.rows,
                        h.x3rgb16.data,
                    );
                }

                if tru.is_null() {
                    libc::printf(c"        tru         = %p\n".as_ptr(), tru);
                } else {
                    let t = &*tru;
                    libc::printf(c"        tru->\n".as_ptr());
                    libc::printf(c"          seed[0]     = %x\n".as_ptr(), t.seed[0] as u32);
                    libc::printf(c"          seed[1]     = %x\n".as_ptr(), t.seed[1] as u32);
                    libc::printf(c"          seed[2]     = %x\n".as_ptr(), t.seed[2] as u32);
                    libc::printf(c"          unknown     = %x\n".as_ptr(), t.unknown as u32);
                    libc::printf(
                        c"          table       = %x %p\n".as_ptr(),
                        t.table.size,
                        t.table.element,
                    );
                    libc::printf(
                        c"          plane_size  = %x %p (".as_ptr(),
                        t.plane_size.size,
                        t.plane_size.element,
                    );
                    for i in 0..t.plane_size.size as isize {
                        let v = *t.plane_size.element.offset(i);
                        libc::printf(c" %d".as_ptr(), v);
                    }
                    libc::printf(c" )\n".as_ptr());
                    libc::printf(c"          plane_address (".as_ptr());
                    for i in 0..TRUE_PLANES as usize {
                        libc::printf(c" %p".as_ptr(), t.plane_address[i]);
                    }
                    libc::printf(c" )\n".as_ptr());
                    libc::printf(
                        c"          tree        = %d %p\n".as_ptr(),
                        t.tree.free_node_index,
                        t.tree.nodes,
                    );
                    libc::printf(
                        c"          x3rgb16     = %x %x %p\n".as_ptr(),
                        t.x3rgb16.columns,
                        t.x3rgb16.rows,
                        t.x3rgb16.data,
                    );
                }

                if q.is_null() {
                    libc::printf(c"        quattro     = %p\n".as_ptr(), q);
                } else {
                    let qq = &*q;
                    libc::printf(c"        quattro->\n".as_ptr());
                    libc::printf(c"          planes (".as_ptr());
                    for i in 0..TRUE_PLANES as usize {
                        libc::printf(c" %d".as_ptr(), qq.plane[i].columns as u32);
                        libc::printf(c"x%d".as_ptr(), qq.plane[i].rows as u32);
                    }
                    libc::printf(c" )\n".as_ptr());
                    libc::printf(
                        c"          unknown     = %x %d\n".as_ptr(),
                        qq.unknown,
                        qq.unknown,
                    );
                }

                libc::printf(c"        data        = %p\n".as_ptr(), id.data);
            }
        }

        if deh.identifier == X3F_SECC {
            let camf = unsafe { &deh.data_subsection.camf };
            unsafe {
                libc::printf(c"      data_subsection.camf.\n".as_ptr());
                libc::printf(c"        type             = %x\n".as_ptr(), camf.type_);
                match camf.type_ {
                    2 => {
                        let mut buf = [0u8; 5];
                        let it = id_to_cstr(camf.__bindgen_anon_1.t2.infotype, &mut buf);
                        libc::printf(c"        type2\n".as_ptr());
                        libc::printf(
                            c"          reserved         = %x\n".as_ptr(),
                            camf.__bindgen_anon_1.t2.reserved,
                        );
                        libc::printf(
                            c"          infotype         = %08x (%s)\n".as_ptr(),
                            camf.__bindgen_anon_1.t2.infotype,
                            it,
                        );
                        libc::printf(
                            c"          infotype_version = %08x\n".as_ptr(),
                            camf.__bindgen_anon_1.t2.infotype_version,
                        );
                        libc::printf(
                            c"          crypt_key        = %08x\n".as_ptr(),
                            camf.__bindgen_anon_1.t2.crypt_key,
                        );
                    }
                    4 => {
                        libc::printf(c"        type4\n".as_ptr());
                        libc::printf(
                            c"          decoded_data_size= %x\n".as_ptr(),
                            camf.__bindgen_anon_1.t4.decoded_data_size,
                        );
                        libc::printf(
                            c"          decode_bias      = %x\n".as_ptr(),
                            camf.__bindgen_anon_1.t4.decode_bias,
                        );
                        libc::printf(
                            c"          block_size       = %x\n".as_ptr(),
                            camf.__bindgen_anon_1.t4.block_size,
                        );
                        libc::printf(
                            c"          block_count      = %x\n".as_ptr(),
                            camf.__bindgen_anon_1.t4.block_count,
                        );
                    }
                    5 => {
                        libc::printf(c"        type5\n".as_ptr());
                        libc::printf(
                            c"          decoded_data_size= %x\n".as_ptr(),
                            camf.__bindgen_anon_1.t5.decoded_data_size,
                        );
                        libc::printf(
                            c"          decode_bias      = %x\n".as_ptr(),
                            camf.__bindgen_anon_1.t5.decode_bias,
                        );
                        libc::printf(
                            c"          unknown2         = %x\n".as_ptr(),
                            camf.__bindgen_anon_1.t5.unknown2,
                        );
                        libc::printf(
                            c"          unknown3         = %x\n".as_ptr(),
                            camf.__bindgen_anon_1.t5.unknown3,
                        );
                    }
                    _ => {
                        libc::printf(c"       (Unknown CAMF type)\n".as_ptr());
                        libc::printf(
                            c"          val0             = %x\n".as_ptr(),
                            camf.__bindgen_anon_1.tN.val0,
                        );
                        libc::printf(
                            c"          val1             = %x\n".as_ptr(),
                            camf.__bindgen_anon_1.tN.val1,
                        );
                        libc::printf(
                            c"          val2             = %x\n".as_ptr(),
                            camf.__bindgen_anon_1.tN.val2,
                        );
                        libc::printf(
                            c"          val3             = %x\n".as_ptr(),
                            camf.__bindgen_anon_1.tN.val3,
                        );
                    }
                }
                libc::printf(c"        data             = %p\n".as_ptr(), camf.data);
                libc::printf(c"        data_size        = %x\n".as_ptr(), camf.data_size);
                libc::printf(
                    c"        decoding_start   = %p\n".as_ptr(),
                    camf.decoding_start,
                );
                libc::printf(
                    c"        decoding_size    = %x\n".as_ptr(),
                    camf.decoding_size,
                );
                libc::printf(
                    c"        table            = %x %p\n".as_ptr(),
                    camf.table.size,
                    camf.table.element,
                );
                libc::printf(
                    c"          tree           = %d %p\n".as_ptr(),
                    camf.tree.free_node_index,
                    camf.tree.nodes,
                );
                libc::printf(
                    c"        decoded_data     = %p\n".as_ptr(),
                    camf.decoded_data,
                );
                libc::printf(
                    c"        decoded_data_size= %x\n".as_ptr(),
                    camf.decoded_data_size,
                );
                libc::printf(
                    c"        entry_table      = %x %p\n".as_ptr(),
                    camf.entry_table.size,
                    camf.entry_table.element,
                );
                print_camf_meta_data2(stdout, camf);
            }
        }
    }
}

/// Public C entry — write a textual metadata dump to `outfilename`.
#[no_mangle]
pub unsafe extern "C" fn x3f_dump_meta_data(
    x3f: *mut x3f_t,
    outfilename: *mut c_char,
) -> x3f_return_t {
    let f = unsafe { libc::fopen(outfilename, c"wb".as_ptr()) };
    if f.is_null() {
        return x3f_return_e_X3F_OUTFILE_ERROR;
    }
    unsafe {
        print_file_header_meta_data(f, x3f);
        print_camf_meta_data(f, x3f);
        print_prop_meta_data(f, x3f);
        libc::fclose(f);
    }
    x3f_return_e_X3F_OK
}

#[used]
static _A_DUMP: unsafe extern "C" fn(*mut x3f_t, *mut c_char) -> x3f_return_t = x3f_dump_meta_data;
#[used]
static _A_PRINT: unsafe extern "C" fn(*mut x3f_t) = x3f_print_meta;
