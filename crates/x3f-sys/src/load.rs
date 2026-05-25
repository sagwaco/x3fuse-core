//! M4d — native Rust port of `x3f_load_data` and friends from
//! `src/x3f_io.c`.
//!
//! Covers:
//!   - the public dispatch entry points: `x3f_load_data`,
//!     `x3f_load_image_block`, `x3f_err`
//!   - section loaders: `x3f_load_property_list`, `x3f_load_image`
//!     (verbatim, TRUE, Huffman compressed/uncompressed, JPEG, pixmap),
//!     `x3f_load_camf`
//!   - CAMF decryption: type-2 (older SD9/SD14), type-4 (TRUE/Merrill),
//!     type-5 (Quattro)
//!   - CAMF entry table walker: `x3f_setup_camf_entries` and the per-
//!     kind setup helpers (`text`, `property`, `matrix`)
//!   - Huffman tree builders (`new_huffman_tree`, `add_code_to_tree`,
//!     `populate_*_huffman_tree`) shared between image and CAMF
//!   - TRU / Quattro / Huffman allocators (`new_*` only — the C-side
//!     `cleanup_*` helpers in `x3f_io.c` still own teardown via
//!     `x3f_delete`)
//!   - file byte readers and `read_data_block`/`read_data_set_offset`
//!
//! Memory ownership: every heap allocation here is via `libc::malloc`,
//! `libc::calloc`, or `libc::realloc` so the still-C `x3f_delete` (which
//! uses `free()` exclusively) can release it correctly.
//!
//! Symbol export: `x3f_load_data`, `x3f_load_image_block`, and `x3f_err`
//! are `#[no_mangle] extern "C"`, blocklisted in bindgen, anchored via
//! `#[used]`, and re-exported through `lib.rs`.
//!
//! Endianness: the input file is little-endian. We read each multi-byte
//! value byte-by-byte and assemble in LE — independent of host endianness.
#![allow(clippy::missing_safety_doc)]
#![allow(clippy::too_many_arguments)]

use std::mem;
use std::os::raw::{c_char, c_int, c_void};
use std::ptr;

use crate::*;
// libc compat — see `sysabi.rs`. Shadows the external libc crate so
// `libc::*` resolves through our wasm32-unknown-unknown shim there.
use crate::sysabi as libc;

// ---------------------------------------------------------------------------
// Format constants (mirroring x3f_io.h).
// ---------------------------------------------------------------------------

const X3F_PROPERTY_LIST_HEADER_SIZE: u32 = 24;
const X3F_IMAGE_HEADER_SIZE: u32 = 28;
const X3F_CAMF_HEADER_SIZE: u32 = 28;

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

const X3F_SECP: u32 = 0x7043_4553;
const X3F_SECI: u32 = 0x6943_4553;
const X3F_SECC: u32 = 0x6343_4553;

const X3F_CMBP: u32 = 0x5062_4d43;
const X3F_CMBT: u32 = 0x5462_4d43;
const X3F_CMBM: u32 = 0x4d62_4d43;

const TRUE_PLANES: usize = 3;

// Huffman code packing: bit 27..31 = length, bits 0..26 = code.
const fn huf_tree_max_nodes(leaves: usize) -> usize {
    (HUF_TREE_MAX_LENGTH + 1) * leaves
}
const HUF_TREE_MAX_LENGTH: usize = 27;
#[inline]
fn huf_tree_get_length(v: u32) -> u32 {
    (v >> 27) & 0x1f
}
#[inline]
fn huf_tree_get_code(v: u32) -> u32 {
    v & 0x07ff_ffff
}

const UNDEFINED_LEAF: u32 = 0xffff_ffff;

// extern decls for the entropy decoder entry points still in entropy.rs.
extern "C" {
    fn x3f_rust_true_decode(id: *mut x3f_image_data_t);
    fn x3f_rust_huffman_decode(id: *mut x3f_image_data_t, bits: c_int);
    fn x3f_rust_simple_decode(id: *mut x3f_image_data_t, bits: c_int, row_stride: c_int);
}

// ---------------------------------------------------------------------------
// Byte readers (mirror x3f_get1/get2/get4/GETN/GET4F).
// ---------------------------------------------------------------------------

#[inline]
unsafe fn get1(f: *mut libc::FILE) -> u32 {
    unsafe { (libc::fgetc(f) as u32) & 0xFF }
}

#[inline]
unsafe fn get2(f: *mut libc::FILE) -> u32 {
    let b0 = unsafe { get1(f) };
    let b1 = unsafe { get1(f) };
    b0 | (b1 << 8)
}

#[inline]
unsafe fn get4(f: *mut libc::FILE) -> u32 {
    let b0 = unsafe { get1(f) };
    let b1 = unsafe { get1(f) };
    let b2 = unsafe { get1(f) };
    let b3 = unsafe { get1(f) };
    b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)
}

unsafe fn getn(f: *mut libc::FILE, buf: *mut u8, size: usize) {
    let mut left = size;
    let mut p = buf;
    while left != 0 {
        let cur = unsafe { libc::fread(p as *mut c_void, 1, left, f) };
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

// ---------------------------------------------------------------------------
// Huffman tree builders.
// ---------------------------------------------------------------------------

unsafe fn new_huffman_tree(htp: *mut x3f_hufftree_t, bits: i32) {
    let leaves = 1usize << bits;
    let bytes = huf_tree_max_nodes(leaves) * mem::size_of::<x3f_huffnode_t>();
    unsafe {
        (*htp).free_node_index = 0;
        (*htp).nodes = libc::calloc(1, bytes) as *mut x3f_huffnode_t;
    }
}

unsafe fn new_node(tree: *mut x3f_hufftree_t) -> *mut x3f_huffnode_t {
    unsafe {
        let idx = (*tree).free_node_index as usize;
        let n = (*tree).nodes.add(idx);
        (*n).branch[0] = ptr::null_mut();
        (*n).branch[1] = ptr::null_mut();
        (*n).leaf = UNDEFINED_LEAF;
        (*tree).free_node_index += 1;
        n
    }
}

// PATTERN_BIT_POS(_len, _bit) = (_len) - (_bit) - 1
#[inline]
fn pattern_bit_pos(length: u32, bit: u32) -> u32 {
    length - bit - 1
}

unsafe fn add_code_to_tree(tree: *mut x3f_hufftree_t, length: u32, code: u32, value: u32) {
    unsafe {
        let mut t: *mut x3f_huffnode_t = (*tree).nodes;
        for i in 0..length {
            let pos = pattern_bit_pos(length, i);
            let bit = ((code >> pos) & 1) as usize;
            let mut t_next = (*t).branch[bit];
            if t_next.is_null() {
                t_next = new_node(tree);
                (*t).branch[bit] = t_next;
            }
            t = t_next;
        }
        (*t).leaf = value;
    }
}

unsafe fn populate_true_huffman_tree(tree: *mut x3f_hufftree_t, table: *const x3f_true_huffman_t) {
    unsafe {
        new_node(tree); // root
        let n = (*table).size as usize;
        for i in 0..n {
            let element = (*table).element.add(i);
            let length = (*element).code_size as u32;
            if length != 0 {
                // add_code_to_tree wants the code right-adjusted.
                let code = (((*element).code as u32) >> (8 - length)) & 0xff;
                let value = i as u32;
                add_code_to_tree(tree, length, code, value);
            }
        }
    }
}

unsafe fn populate_huffman_tree(
    tree: *mut x3f_hufftree_t,
    table: *const x3f_table32_t,
    mapping: *const x3f_table16_t,
) {
    unsafe {
        new_node(tree); // root
        let n = (*table).size as usize;
        for i in 0..n {
            let element = *(*table).element.add(i);
            if element != 0 {
                let length = huf_tree_get_length(element);
                let code = huf_tree_get_code(element);
                let value = if (*table).size == (*mapping).size {
                    *(*mapping).element.add(i) as u32
                } else {
                    i as u32
                };
                add_code_to_tree(tree, length, code, value);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// TRU / Quattro / Huffman allocators. Use libc::calloc so x3f_delete's
// libc::free path works unchanged.
// ---------------------------------------------------------------------------

unsafe fn new_true(trup: *mut *mut x3f_true_t) -> *mut x3f_true_t {
    unsafe {
        // Caller's responsibility to ensure *trup is null before calling
        // (matches the C: cleanup_true(TRUP); calloc; assign).
        let tru = libc::calloc(1, mem::size_of::<x3f_true_t>()) as *mut x3f_true_t;
        // calloc already zeroes everything; we just need to install the pointer.
        *trup = tru;
        tru
    }
}

unsafe fn new_quattro(qp: *mut *mut x3f_quattro_t) -> *mut x3f_quattro_t {
    unsafe {
        let q = libc::calloc(1, mem::size_of::<x3f_quattro_t>()) as *mut x3f_quattro_t;
        *qp = q;
        q
    }
}

unsafe fn new_huffman(hufp: *mut *mut x3f_huffman_t) -> *mut x3f_huffman_t {
    unsafe {
        let h = libc::calloc(1, mem::size_of::<x3f_huffman_t>()) as *mut x3f_huffman_t;
        *hufp = h;
        h
    }
}

// ---------------------------------------------------------------------------
// Bit reader for CAMF type-4 / type-5 decoders. Same shape as
// `bit_state_t` in x3f_io.c: pre-explode each byte into 8 bits, MSB first.
// ---------------------------------------------------------------------------

struct BitState {
    next: *const u8,
    bit_offset: u8,
    bits: [u8; 8],
}

impl BitState {
    unsafe fn new(addr: *const u8) -> Self {
        BitState {
            next: addr,
            bit_offset: 8,
            bits: [0; 8],
        }
    }

    #[inline]
    unsafe fn get_bit(&mut self) -> u8 {
        if self.bit_offset == 8 {
            let mut byte = unsafe { *self.next };
            for i in (0..8).rev() {
                self.bits[i] = byte & 1;
                byte >>= 1;
            }
            self.next = unsafe { self.next.add(1) };
            self.bit_offset = 0;
        }
        let b = self.bits[self.bit_offset as usize];
        self.bit_offset += 1;
        b
    }
}

#[inline]
unsafe fn get_true_diff(bs: &mut BitState, htp: *const x3f_hufftree_t) -> i32 {
    unsafe {
        let nodes = (*htp).nodes;
        let mut node: *const x3f_huffnode_t = nodes;
        while !(*node).branch[0].is_null() || !(*node).branch[1].is_null() {
            let bit = bs.get_bit();
            let next = (*node).branch[bit as usize];
            if next.is_null() {
                x3f_printf(
                    x3f_verbosity_t_ERR,
                    c"Huffman coding got unexpected bit\n".as_ptr(),
                );
                return 0;
            }
            node = next;
        }
        let bits = (*node).leaf as u32;
        if bits == 0 {
            return 0;
        }
        let first_bit = bs.get_bit();
        let mut diff = first_bit as i32;
        for _ in 1..bits {
            diff = (diff << 1) + bs.get_bit() as i32;
        }
        if first_bit == 0 {
            diff -= (1 << bits) - 1;
        }
        diff
    }
}

// ---------------------------------------------------------------------------
// File data block helpers.
// ---------------------------------------------------------------------------

unsafe fn read_data_set_offset(
    info: *mut x3f_info_t,
    de: *mut x3f_directory_entry_t,
    header_size: u32,
) {
    unsafe {
        let i_off = (*de).input.offset + header_size;
        libc::fseek(
            (*info).input.file as *mut libc::FILE,
            i_off as libc::c_long,
            libc::SEEK_SET,
        );
    }
}

unsafe fn read_data_block(
    data: *mut *mut c_void,
    info: *mut x3f_info_t,
    de: *mut x3f_directory_entry_t,
    footer: u32,
) -> u32 {
    unsafe {
        let f = (*info).input.file as *mut libc::FILE;
        let here = libc::ftell(f) as i64;
        let end = ((*de).input.offset + (*de).input.size) as i64;
        let size = (end - here - footer as i64) as u32;
        let buf = libc::malloc(size as usize);
        *data = buf;
        getn(f, buf as *mut u8, size as usize);
        size
    }
}

// ---------------------------------------------------------------------------
// utf16le_to_utf8 — convert a NUL-terminated UTF-16 LE string into a libc-
// malloc'd UTF-8 NUL-terminated C string. The C-side x3f_delete frees these
// with `free()`, so we MUST allocate via libc::malloc.
// ---------------------------------------------------------------------------

unsafe fn utf16le_to_utf8(input: *const u16) -> *mut c_char {
    unsafe {
        // Find length (in u16 units, not counting NUL).
        let mut n = 0usize;
        while *input.add(n) != 0 {
            n += 1;
        }
        let units: Vec<u16> = (0..n).map(|i| *input.add(i)).collect();
        // Decode UTF-16, encode UTF-8. Replace ill-formed sequences with U+FFFD
        // (defensive — the C iconv would assert on these).
        let s: String = char::decode_utf16(units.into_iter())
            .map(|r| r.unwrap_or(char::REPLACEMENT_CHARACTER))
            .collect();
        let bytes = s.as_bytes();
        let buf = libc::malloc(bytes.len() + 1) as *mut u8;
        ptr::copy_nonoverlapping(bytes.as_ptr(), buf, bytes.len());
        *buf.add(bytes.len()) = 0;
        buf as *mut c_char
    }
}

// ---------------------------------------------------------------------------
// Section loaders.
// ---------------------------------------------------------------------------

unsafe fn x3f_load_image_verbatim(info: *mut x3f_info_t, de: *mut x3f_directory_entry_t) {
    unsafe {
        x3f_printf(x3f_verbosity_t_DEBUG, c"Load image verbatim\n".as_ptr());
        let id = &mut (*de).header.data_subsection.image_data;
        id.data_size = read_data_block(&mut id.data, info, de, 0);
    }
}

unsafe fn x3f_load_property_list(info: *mut x3f_info_t, de: *mut x3f_directory_entry_t) {
    unsafe {
        let f = (*info).input.file as *mut libc::FILE;
        read_data_set_offset(info, de, X3F_PROPERTY_LIST_HEADER_SIZE);

        let pl = &mut (*de).header.data_subsection.property_list;
        let num = pl.num_properties as usize;

        // Resize the property table (initial element is NULL — see x3f_new_from_file).
        pl.property_table.size = num as u32;
        pl.property_table.element = libc::realloc(
            pl.property_table.element as *mut c_void,
            num * mem::size_of::<x3f_property_t>(),
        ) as *mut x3f_property_t;

        for i in 0..num {
            let p = pl.property_table.element.add(i);
            (*p).name_offset = get4(f);
            (*p).value_offset = get4(f);
        }

        pl.data_size = read_data_block(&mut pl.data, info, de, 0);

        for i in 0..num {
            let p = pl.property_table.element.add(i);
            // PL->data is a void* of u16 stride. ((utf16_t *)PL->data + offset)
            let base = pl.data as *mut u16;
            (*p).name = base.add((*p).name_offset as usize);
            (*p).value = base.add((*p).value_offset as usize);
            (*p).name_utf8 = utf16le_to_utf8((*p).name);
            (*p).value_utf8 = utf16le_to_utf8((*p).value);
        }
    }
}

unsafe fn x3f_load_true(info: *mut x3f_info_t, de: *mut x3f_directory_entry_t) {
    unsafe {
        let f = (*info).input.file as *mut libc::FILE;
        let id = &mut (*de).header.data_subsection.image_data;
        let tru = new_true(&mut id.tru);

        let is_quattro = matches!(
            id.type_format,
            X3F_IMAGE_RAW_QUATTRO | X3F_IMAGE_RAW_SDQ | X3F_IMAGE_RAW_SDQH
        );

        let q: *mut x3f_quattro_t = if is_quattro {
            x3f_printf(x3f_verbosity_t_DEBUG, c"Load Quattro extra info\n".as_ptr());
            let q = new_quattro(&mut id.quattro);
            for i in 0..TRUE_PLANES {
                (*q).plane[i].columns = get2(f) as u16;
                (*q).plane[i].rows = get2(f) as u16;
            }
            if (*q).plane[0].rows as u32 == id.rows / 2 {
                x3f_printf(x3f_verbosity_t_DEBUG, c"Quattro layout\n".as_ptr());
                (*q).quattro_layout = 1;
            } else if (*q).plane[0].rows as u32 == id.rows {
                x3f_printf(x3f_verbosity_t_DEBUG, c"Binned Quattro\n".as_ptr());
                (*q).quattro_layout = 0;
            } else {
                x3f_printf(
                    x3f_verbosity_t_ERR,
                    c"Quattro file with unknown layer size\n".as_ptr(),
                );
                libc::abort();
            }
            q
        } else {
            ptr::null_mut()
        };

        x3f_printf(x3f_verbosity_t_DEBUG, c"Load TRUE\n".as_ptr());

        // Read TRUE header.
        (*tru).seed[0] = get2(f) as u16;
        (*tru).seed[1] = get2(f) as u16;
        (*tru).seed[2] = get2(f) as u16;
        (*tru).unknown = get2(f) as u16;

        // GET_TRUE_HUFF_TABLE: variable-length, NUL-terminated by code_size==0.
        (*tru).table.element = ptr::null_mut();
        let mut i = 0usize;
        loop {
            (*tru).table.size = (i + 1) as u32;
            (*tru).table.element = libc::realloc(
                (*tru).table.element as *mut c_void,
                (i + 1) * mem::size_of::<x3f_true_huffman_element_t>(),
            ) as *mut x3f_true_huffman_element_t;
            let elem = (*tru).table.element.add(i);
            (*elem).code_size = get1(f) as u8;
            (*elem).code = get1(f) as u8;
            if (*elem).code_size == 0 {
                break;
            }
            i += 1;
        }

        if is_quattro {
            x3f_printf(
                x3f_verbosity_t_DEBUG,
                c"Load Quattro extra info 2\n".as_ptr(),
            );
            (*q).unknown = get4(f);
        }

        // GET_TABLE(TRU->plane_size, GET4, TRUE_PLANES)
        (*tru).plane_size.size = TRUE_PLANES as u32;
        (*tru).plane_size.element = libc::realloc(
            (*tru).plane_size.element as *mut c_void,
            TRUE_PLANES * mem::size_of::<u32>(),
        ) as *mut u32;
        for i in 0..TRUE_PLANES {
            *(*tru).plane_size.element.add(i) = get4(f);
        }

        // Read image data.
        id.data_size = read_data_block(&mut id.data, info, de, 0);

        // Build huffman tree.
        new_huffman_tree(&mut (*tru).tree, 8);
        populate_true_huffman_tree(&mut (*tru).tree, &(*tru).table);

        (*tru).plane_address[0] = id.data as *mut u8;
        for i in 1..TRUE_PLANES {
            let prev_size = *(*tru).plane_size.element.add(i - 1);
            let aligned = (prev_size + 15) / 16 * 16;
            (*tru).plane_address[i] = (*tru).plane_address[i - 1].add(aligned as usize);
        }

        if is_quattro && (*q).quattro_layout != 0 {
            // Quattro layout: x3rgb16 sized for plane[0], top16 sized for plane[2].
            let columns = (*q).plane[0].columns as u32;
            let rows = (*q).plane[0].rows as u32;
            let channels: u32 = 3;
            let size = columns * rows * channels;

            (*tru).x3rgb16.columns = columns;
            (*tru).x3rgb16.rows = rows;
            (*tru).x3rgb16.channels = channels;
            (*tru).x3rgb16.row_stride = columns * channels;
            let buf = libc::malloc((size as usize) * mem::size_of::<u16>()) as *mut u16;
            (*tru).x3rgb16.buf = buf as *mut c_void;
            (*tru).x3rgb16.data = buf;

            let columns = (*q).plane[2].columns as u32;
            let rows = (*q).plane[2].rows as u32;
            let channels: u32 = 1;
            let size = columns * rows * channels;
            (*q).top16.columns = columns;
            (*q).top16.rows = rows;
            (*q).top16.channels = channels;
            (*q).top16.row_stride = columns * channels;
            let buf = libc::malloc((size as usize) * mem::size_of::<u16>()) as *mut u16;
            (*q).top16.buf = buf as *mut c_void;
            (*q).top16.data = buf;
        } else {
            let size = id.columns * id.rows * 3;
            (*tru).x3rgb16.columns = id.columns;
            (*tru).x3rgb16.rows = id.rows;
            (*tru).x3rgb16.channels = 3;
            (*tru).x3rgb16.row_stride = id.columns * 3;
            let buf = libc::malloc((size as usize) * mem::size_of::<u16>()) as *mut u16;
            (*tru).x3rgb16.buf = buf as *mut c_void;
            (*tru).x3rgb16.data = buf;
        }

        x3f_rust_true_decode(id as *mut _);
    }
}

unsafe fn x3f_load_huffman_compressed(
    info: *mut x3f_info_t,
    de: *mut x3f_directory_entry_t,
    bits: i32,
    _use_map_table: i32,
) {
    unsafe {
        let f = (*info).input.file as *mut libc::FILE;
        let id = &mut (*de).header.data_subsection.image_data;
        let huf = id.huffman;
        let table_size = 1usize << bits;
        let row_offsets_size = id.rows as usize * mem::size_of::<u32>();

        x3f_printf(x3f_verbosity_t_DEBUG, c"Load huffman compressed\n".as_ptr());

        // GET_TABLE(HUF->table, GET4, table_size)
        (*huf).table.size = table_size as u32;
        (*huf).table.element = libc::realloc(
            (*huf).table.element as *mut c_void,
            table_size * mem::size_of::<u32>(),
        ) as *mut u32;
        for i in 0..table_size {
            *(*huf).table.element.add(i) = get4(f);
        }

        id.data_size = read_data_block(&mut id.data, info, de, row_offsets_size as u32);

        // GET_TABLE(HUF->row_offsets, GET4, ID->rows)
        (*huf).row_offsets.size = id.rows;
        (*huf).row_offsets.element = libc::realloc(
            (*huf).row_offsets.element as *mut c_void,
            id.rows as usize * mem::size_of::<u32>(),
        ) as *mut u32;
        for i in 0..(id.rows as usize) {
            *(*huf).row_offsets.element.add(i) = get4(f);
        }

        x3f_printf(x3f_verbosity_t_DEBUG, c"Make huffman tree ...\n".as_ptr());
        new_huffman_tree(&mut (*huf).tree, bits);
        populate_huffman_tree(&mut (*huf).tree, &(*huf).table, &(*huf).mapping);
        x3f_printf(x3f_verbosity_t_DEBUG, c"... DONE\n".as_ptr());

        x3f_rust_huffman_decode(id as *mut _, bits);
    }
}

unsafe fn x3f_load_huffman_not_compressed(
    info: *mut x3f_info_t,
    de: *mut x3f_directory_entry_t,
    bits: i32,
    _use_map_table: i32,
    row_stride: i32,
) {
    unsafe {
        let id = &mut (*de).header.data_subsection.image_data;
        x3f_printf(
            x3f_verbosity_t_DEBUG,
            c"Load huffman not compressed\n".as_ptr(),
        );
        id.data_size = read_data_block(&mut id.data, info, de, 0);
        x3f_rust_simple_decode(id as *mut _, bits, row_stride);
    }
}

unsafe fn x3f_load_huffman(
    info: *mut x3f_info_t,
    de: *mut x3f_directory_entry_t,
    bits: i32,
    use_map_table: i32,
    row_stride: i32,
) {
    unsafe {
        let f = (*info).input.file as *mut libc::FILE;
        let id = &mut (*de).header.data_subsection.image_data;
        let huf = new_huffman(&mut id.huffman);

        if use_map_table != 0 {
            let table_size = 1usize << bits;
            (*huf).mapping.size = table_size as u32;
            (*huf).mapping.element = libc::realloc(
                (*huf).mapping.element as *mut c_void,
                table_size * mem::size_of::<u16>(),
            ) as *mut u16;
            for i in 0..table_size {
                *(*huf).mapping.element.add(i) = get2(f) as u16;
            }
        }

        match id.type_format {
            X3F_IMAGE_RAW_HUFFMAN_X530 | X3F_IMAGE_RAW_HUFFMAN_10BIT => {
                let size = id.columns * id.rows * 3;
                (*huf).x3rgb16.columns = id.columns;
                (*huf).x3rgb16.rows = id.rows;
                (*huf).x3rgb16.channels = 3;
                (*huf).x3rgb16.row_stride = id.columns * 3;
                let buf = libc::malloc((size as usize) * mem::size_of::<u16>()) as *mut u16;
                (*huf).x3rgb16.buf = buf as *mut c_void;
                (*huf).x3rgb16.data = buf;
            }
            X3F_IMAGE_THUMB_HUFFMAN => {
                let size = id.columns * id.rows * 3;
                (*huf).rgb8.columns = id.columns;
                // NOTE: preserves a long-standing bug in the C source: the
                // next line should set rgb8.rows but writes to rgb8.columns
                // again. Leaves rgb8.rows zero. Kept verbatim for byte-for-
                // byte parity with the legacy decoder.
                (*huf).rgb8.columns = id.rows;
                (*huf).rgb8.channels = 3;
                (*huf).rgb8.row_stride = id.columns * 3;
                let buf = libc::malloc((size as usize) * mem::size_of::<u8>()) as *mut u8;
                (*huf).rgb8.buf = buf as *mut c_void;
                (*huf).rgb8.data = buf;
            }
            _ => {
                x3f_printf(
                    x3f_verbosity_t_ERR,
                    c"Unknown huffman image type\n".as_ptr(),
                );
            }
        }

        if row_stride == 0 {
            x3f_load_huffman_compressed(info, de, bits, use_map_table);
        } else {
            x3f_load_huffman_not_compressed(info, de, bits, use_map_table, row_stride);
        }
    }
}

unsafe fn x3f_load_pixmap(info: *mut x3f_info_t, de: *mut x3f_directory_entry_t) {
    unsafe {
        x3f_printf(x3f_verbosity_t_DEBUG, c"Load pixmap\n".as_ptr());
        x3f_load_image_verbatim(info, de);
    }
}

unsafe fn x3f_load_jpeg(info: *mut x3f_info_t, de: *mut x3f_directory_entry_t) {
    unsafe {
        x3f_printf(x3f_verbosity_t_DEBUG, c"Load JPEG\n".as_ptr());
        x3f_load_image_verbatim(info, de);
    }
}

unsafe fn x3f_load_image(info: *mut x3f_info_t, de: *mut x3f_directory_entry_t) {
    unsafe {
        let id = &mut (*de).header.data_subsection.image_data;
        read_data_set_offset(info, de, X3F_IMAGE_HEADER_SIZE);
        match id.type_format {
            X3F_IMAGE_RAW_TRUE
            | X3F_IMAGE_RAW_MERRILL
            | X3F_IMAGE_RAW_QUATTRO
            | X3F_IMAGE_RAW_SDQ
            | X3F_IMAGE_RAW_SDQH => x3f_load_true(info, de),
            X3F_IMAGE_RAW_HUFFMAN_X530 | X3F_IMAGE_RAW_HUFFMAN_10BIT => {
                x3f_load_huffman(info, de, 10, 1, id.row_stride as i32)
            }
            X3F_IMAGE_THUMB_PLAIN => x3f_load_pixmap(info, de),
            X3F_IMAGE_THUMB_HUFFMAN => x3f_load_huffman(info, de, 8, 0, id.row_stride as i32),
            X3F_IMAGE_THUMB_JPEG => x3f_load_jpeg(info, de),
            _ => {
                x3f_printf(x3f_verbosity_t_ERR, c"Unknown image type\n".as_ptr());
            }
        }
    }
}

// ---------------------------------------------------------------------------
// CAMF decoders. The encrypted CAMF blob comes in three flavors keyed by
// CAMF->type:
//   2  — older SD9-SD14: simple stream cipher seeded by `crypt_key`
//   4  — TRUE/Merrill: zigzag-style 2D predictor + Huffman
//   5  — Quattro: 1D running-sum + Huffman
// ---------------------------------------------------------------------------

unsafe fn x3f_load_camf_decode_type2(camf: *mut x3f_camf_t) {
    unsafe {
        let mut key = (*camf).__bindgen_anon_1.t2.crypt_key as u64;
        (*camf).decoded_data_size = (*camf).data_size;
        (*camf).decoded_data = libc::malloc((*camf).decoded_data_size as usize);
        let src = (*camf).data as *const u8;
        let dst = (*camf).decoded_data as *mut u8;
        let n = (*camf).data_size as usize;
        for i in 0..n {
            let old = *src.add(i);
            key = (key * 1597 + 51749) % 244944;
            // tmp = (uint32_t)(key * (int64_t)301593171 >> 24)
            // Cast to i64 for the multiply (C uses int64_t).
            let tmp = ((key as i64) * 301_593_171_i64 >> 24) as u32;
            // new = old ^ ((((key << 8) - tmp) >> 1) + tmp) >> 17
            let a = ((key as u32) << 8).wrapping_sub(tmp);
            let mix = ((a >> 1).wrapping_add(tmp)) >> 17;
            let new_byte = old ^ (mix as u8);
            *dst.add(i) = new_byte;
        }
    }
}

unsafe fn camf_decode_type4(camf: *mut x3f_camf_t) {
    unsafe {
        let seed = (*camf).__bindgen_anon_1.t4.decode_bias as i32;
        let dst_size = (*camf).__bindgen_anon_1.t4.decoded_data_size;
        let rows = (*camf).__bindgen_anon_1.t4.block_count;
        let cols = (*camf).__bindgen_anon_1.t4.block_size;

        (*camf).decoded_data_size = dst_size;
        let dst_base = libc::malloc(dst_size as usize) as *mut u8;
        libc::memset(dst_base as *mut c_void, 0, dst_size as usize);
        (*camf).decoded_data = dst_base as *mut c_void;
        let dst_end = dst_base.add(dst_size as usize);

        let mut bs = BitState::new((*camf).decoding_start);

        let mut row_start_acc = [[seed; 2]; 2];
        let mut odd_dst: bool = false;

        let mut dst = dst_base;

        'outer: for row in 0..rows {
            let odd_row = (row & 1) as usize;
            let mut acc = [0i32; 2];
            for col in 0..cols {
                let odd_col = (col & 1) as usize;
                let diff = get_true_diff(&mut bs, &(*camf).tree);
                let prev = if col < 2 {
                    row_start_acc[odd_row][odd_col]
                } else {
                    acc[odd_col]
                };
                let value = prev + diff;
                acc[odd_col] = value;
                if col < 2 {
                    row_start_acc[odd_row][odd_col] = value;
                }
                if !odd_dst {
                    *dst = ((value >> 4) & 0xff) as u8;
                    dst = dst.add(1);
                    if dst >= dst_end {
                        break 'outer;
                    }
                    *dst = ((value << 4) & 0xf0) as u8;
                } else {
                    *dst |= ((value >> 8) & 0x0f) as u8;
                    dst = dst.add(1);
                    if dst >= dst_end {
                        break 'outer;
                    }
                    *dst = (value & 0xff) as u8;
                    dst = dst.add(1);
                    if dst >= dst_end {
                        break 'outer;
                    }
                }
                odd_dst = !odd_dst;
            }
        }
    }
}

const CAMF_T4_DATA_SIZE_OFFSET: usize = 28;
const CAMF_T4_DATA_OFFSET: usize = 32;

unsafe fn x3f_load_camf_decode_type4(camf: *mut x3f_camf_t) {
    unsafe {
        // Build the variable-length true-huffman element table from the
        // leading bytes of CAMF->data, terminated by a zero code_size byte.
        let mut p = (*camf).data as *mut u8;
        let mut element: *mut x3f_true_huffman_element_t = ptr::null_mut();
        let mut i = 0usize;
        while *p != 0 {
            element = libc::realloc(
                element as *mut c_void,
                (i + 1) * mem::size_of::<x3f_true_huffman_element_t>(),
            ) as *mut x3f_true_huffman_element_t;
            (*element.add(i)).code_size = *p;
            p = p.add(1);
            (*element.add(i)).code = *p;
            p = p.add(1);
            i += 1;
        }
        (*camf).table.size = i as u32;
        (*camf).table.element = element;

        let data_base = (*camf).data as *mut u8;
        let size_word = data_base.add(CAMF_T4_DATA_SIZE_OFFSET) as *const u32;
        (*camf).decoding_size = ptr::read_unaligned(size_word);
        (*camf).decoding_start = data_base.add(CAMF_T4_DATA_OFFSET);

        new_huffman_tree(&mut (*camf).tree, 8);
        populate_true_huffman_tree(&mut (*camf).tree, &(*camf).table);

        camf_decode_type4(camf);
    }
}

unsafe fn camf_decode_type5(camf: *mut x3f_camf_t) {
    unsafe {
        let mut acc = (*camf).__bindgen_anon_1.t5.decode_bias as i32;
        let dst_size = (*camf).__bindgen_anon_1.t5.decoded_data_size;
        (*camf).decoded_data_size = dst_size;
        let dst_base = libc::malloc(dst_size as usize) as *mut u8;
        (*camf).decoded_data = dst_base as *mut c_void;
        let mut bs = BitState::new((*camf).decoding_start);
        let mut dst = dst_base;
        for _ in 0..dst_size {
            let diff = get_true_diff(&mut bs, &(*camf).tree);
            acc = acc.wrapping_add(diff);
            *dst = (acc & 0xff) as u8;
            dst = dst.add(1);
        }
    }
}

const CAMF_T5_DATA_SIZE_OFFSET: usize = 28;
const CAMF_T5_DATA_OFFSET: usize = 32;

unsafe fn x3f_load_camf_decode_type5(camf: *mut x3f_camf_t) {
    unsafe {
        let mut p = (*camf).data as *mut u8;
        let mut element: *mut x3f_true_huffman_element_t = ptr::null_mut();
        let mut i = 0usize;
        while *p != 0 {
            element = libc::realloc(
                element as *mut c_void,
                (i + 1) * mem::size_of::<x3f_true_huffman_element_t>(),
            ) as *mut x3f_true_huffman_element_t;
            (*element.add(i)).code_size = *p;
            p = p.add(1);
            (*element.add(i)).code = *p;
            p = p.add(1);
            i += 1;
        }
        (*camf).table.size = i as u32;
        (*camf).table.element = element;

        let data_base = (*camf).data as *mut u8;
        let size_word = data_base.add(CAMF_T5_DATA_SIZE_OFFSET) as *const u32;
        (*camf).decoding_size = ptr::read_unaligned(size_word);
        (*camf).decoding_start = data_base.add(CAMF_T5_DATA_OFFSET);

        new_huffman_tree(&mut (*camf).tree, 8);
        populate_true_huffman_tree(&mut (*camf).tree, &(*camf).table);

        camf_decode_type5(camf);
    }
}

// ---------------------------------------------------------------------------
// CAMF entry walker — decoded blob is a sequence of self-describing entries
// keyed by a 4-byte tag (CMbT/CMbP/CMbM = Text / Property / Matrix).
// ---------------------------------------------------------------------------

unsafe fn x3f_setup_camf_text_entry(entry: *mut camf_entry_t) {
    unsafe {
        let v = (*entry).value_address as *const u8;
        (*entry).text_size = ptr::read_unaligned(v as *const u32);
        (*entry).text = v.add(4) as *mut c_char;
    }
}

unsafe fn x3f_setup_camf_property_entry(entry: *mut camf_entry_t) {
    unsafe {
        let e = (*entry).entry as *mut u8;
        let v = (*entry).value_address as *mut u8;
        let num = ptr::read_unaligned(v as *const u32);
        (*entry).property_num = num;
        let off = ptr::read_unaligned(v.add(4) as *const u32);

        (*entry).property_name =
            libc::malloc((num as usize) * mem::size_of::<*mut c_char>()) as *mut *mut c_char;
        (*entry).property_value =
            libc::malloc((num as usize) * mem::size_of::<*mut u8>()) as *mut *mut u8;

        for i in 0..(num as usize) {
            let name_off = off + ptr::read_unaligned(v.add(8 + 8 * i) as *const u32);
            let value_off = off + ptr::read_unaligned(v.add(8 + 8 * i + 4) as *const u32);
            *(*entry).property_name.add(i) = e.add(name_off as usize) as *mut c_char;
            *(*entry).property_value.add(i) = e.add(value_off as usize);
        }
    }
}

fn set_matrix_element_info(t: u32) -> (u32, matrix_type_t) {
    match t {
        0 => (2, matrix_type_t_M_INT),
        1 => (4, matrix_type_t_M_UINT),
        2 => (4, matrix_type_t_M_UINT),
        3 => (4, matrix_type_t_M_FLOAT),
        5 => (1, matrix_type_t_M_UINT),
        6 => (2, matrix_type_t_M_UINT),
        _ => {
            unsafe {
                x3f_printf(
                    x3f_verbosity_t_ERR,
                    c"Unknown matrix type (%ud)\n".as_ptr(),
                    t,
                )
            };
            unsafe { libc::abort() };
        }
    }
}

unsafe fn get_matrix_copy(entry: *mut camf_entry_t) {
    unsafe {
        let element_size = (*entry).matrix_element_size;
        let elements = (*entry).matrix_elements as usize;
        let dec_type = (*entry).matrix_decoded_type;
        let unit = if dec_type == matrix_type_t_M_FLOAT {
            mem::size_of::<f64>()
        } else {
            mem::size_of::<u32>()
        };
        let size = unit * elements;
        let dst = libc::malloc(size);
        (*entry).matrix_decoded = dst;
        let src = (*entry).matrix_data;

        match element_size {
            4 => match dec_type {
                t if t == matrix_type_t_M_INT || t == matrix_type_t_M_UINT => {
                    libc::memcpy(dst, src, size);
                }
                t if t == matrix_type_t_M_FLOAT => {
                    let s = src as *const f32;
                    let d = dst as *mut f64;
                    for i in 0..elements {
                        *d.add(i) = ptr::read_unaligned(s.add(i)) as f64;
                    }
                }
                _ => {
                    x3f_printf(
                        x3f_verbosity_t_ERR,
                        c"Invalid matrix element type of size 4\n".as_ptr(),
                    );
                    libc::abort();
                }
            },
            2 => match dec_type {
                t if t == matrix_type_t_M_INT => {
                    let s = src as *const i16;
                    let d = dst as *mut i32;
                    for i in 0..elements {
                        *d.add(i) = ptr::read_unaligned(s.add(i)) as i32;
                    }
                }
                t if t == matrix_type_t_M_UINT => {
                    let s = src as *const u16;
                    let d = dst as *mut u32;
                    for i in 0..elements {
                        *d.add(i) = ptr::read_unaligned(s.add(i)) as u32;
                    }
                }
                _ => {
                    x3f_printf(
                        x3f_verbosity_t_ERR,
                        c"Invalid matrix element type of size 2\n".as_ptr(),
                    );
                    libc::abort();
                }
            },
            1 => match dec_type {
                t if t == matrix_type_t_M_INT => {
                    let s = src as *const i8;
                    let d = dst as *mut i32;
                    for i in 0..elements {
                        *d.add(i) = ptr::read_unaligned(s.add(i)) as i32;
                    }
                }
                t if t == matrix_type_t_M_UINT => {
                    let s = src as *const u8;
                    let d = dst as *mut u32;
                    for i in 0..elements {
                        *d.add(i) = ptr::read_unaligned(s.add(i)) as u32;
                    }
                }
                _ => {
                    x3f_printf(
                        x3f_verbosity_t_ERR,
                        c"Invalid matrix element type of size 1\n".as_ptr(),
                    );
                    libc::abort();
                }
            },
            _ => {
                x3f_printf(
                    x3f_verbosity_t_ERR,
                    c"Unknown size %d\n".as_ptr(),
                    element_size,
                );
                libc::abort();
            }
        }
    }
}

unsafe fn x3f_setup_camf_matrix_entry(entry: *mut camf_entry_t) {
    unsafe {
        let e = (*entry).entry as *mut u8;
        let v = (*entry).value_address as *mut u8;

        let t = ptr::read_unaligned(v as *const u32);
        (*entry).matrix_type = t;
        let dim = ptr::read_unaligned(v.add(4) as *const u32);
        (*entry).matrix_dim = dim;
        let off = ptr::read_unaligned(v.add(8) as *const u32);
        (*entry).matrix_data_off = off;

        let dentry = libc::malloc((dim as usize) * mem::size_of::<camf_dim_entry_t>())
            as *mut camf_dim_entry_t;
        (*entry).matrix_dim_entry = dentry;

        let mut totalsize: u32 = 1;
        for i in 0..(dim as usize) {
            let sz = ptr::read_unaligned(v.add(12 + 12 * i) as *const u32);
            (*dentry.add(i)).size = sz;
            let no = ptr::read_unaligned(v.add(12 + 12 * i + 4) as *const u32);
            (*dentry.add(i)).name_offset = no;
            let n = ptr::read_unaligned(v.add(12 + 12 * i + 8) as *const u32);
            (*dentry.add(i)).n = n;
            (*dentry.add(i)).name = e.add(no as usize) as *mut c_char;
            if (*dentry.add(i)).n != i as u32 {
                x3f_printf(
                    x3f_verbosity_t_DEBUG,
                    c"Matrix entry for %s/%s is out of order (index/%d != order/%d)\n".as_ptr(),
                    (*entry).name_address,
                    (*dentry.add(i)).name,
                    (*dentry.add(i)).n,
                    i as i32,
                );
            }
            totalsize = totalsize.wrapping_mul(sz);
        }

        let (elem_size, dec_type) = set_matrix_element_info(t);
        (*entry).matrix_element_size = elem_size;
        (*entry).matrix_decoded_type = dec_type;
        (*entry).matrix_data = e.add(off as usize) as *mut c_void;

        (*entry).matrix_elements = totalsize;
        (*entry).matrix_used_space = (*entry).entry_size - off;
        (*entry).matrix_estimated_element_size =
            (*entry).matrix_used_space as f64 / totalsize as f64;

        get_matrix_copy(entry);
    }
}

unsafe fn x3f_setup_camf_entries(camf: *mut x3f_camf_t) {
    unsafe {
        let p_start = (*camf).decoded_data as *mut u8;
        let end = p_start.add((*camf).decoded_data_size as usize);
        let mut p = p_start;
        let mut entry: *mut camf_entry_t = ptr::null_mut();
        let mut i: u32 = 0;

        x3f_printf(x3f_verbosity_t_DEBUG, c"SETUP CAMF ENTRIES\n".as_ptr());

        while p < end {
            let p4 = p as *const u32;
            let id = ptr::read_unaligned(p4);
            match id {
                X3F_CMBP | X3F_CMBT | X3F_CMBM => {}
                _ => {
                    x3f_printf(
                        x3f_verbosity_t_ERR,
                        c"Unknown CAMF entry %x @ %p\n".as_ptr(),
                        id,
                        p4,
                    );
                    x3f_printf(
                        x3f_verbosity_t_ERR,
                        c"  start = %p end = %p\n".as_ptr(),
                        (*camf).decoded_data,
                        end,
                    );
                    x3f_printf(
                        x3f_verbosity_t_ERR,
                        c"  left = %ld\n".as_ptr(),
                        (end as isize - p as isize) as libc::c_long,
                    );
                    x3f_printf(x3f_verbosity_t_ERR, c"Stop parsing CAMF\n".as_ptr());
                    break;
                }
            }

            entry = libc::realloc(
                entry as *mut c_void,
                ((i + 1) as usize) * mem::size_of::<camf_entry_t>(),
            ) as *mut camf_entry_t;
            let cur = entry.add(i as usize);

            (*cur).entry = p as *mut c_void;
            (*cur).id = id;
            (*cur).version = ptr::read_unaligned(p4.add(1));
            (*cur).entry_size = ptr::read_unaligned(p4.add(2));
            (*cur).name_offset = ptr::read_unaligned(p4.add(3));
            (*cur).value_offset = ptr::read_unaligned(p4.add(4));

            (*cur).name_address = p.add((*cur).name_offset as usize) as *mut c_char;
            (*cur).value_address = p.add((*cur).value_offset as usize) as *mut c_void;
            (*cur).name_size = (*cur).value_offset - (*cur).name_offset;
            (*cur).value_size = (*cur).entry_size - (*cur).value_offset;

            (*cur).text_size = 0;
            (*cur).text = ptr::null_mut();
            (*cur).property_num = 0;
            (*cur).property_name = ptr::null_mut();
            (*cur).property_value = ptr::null_mut();
            (*cur).matrix_type = 0;
            (*cur).matrix_dim = 0;
            (*cur).matrix_data_off = 0;
            (*cur).matrix_data = ptr::null_mut();
            (*cur).matrix_dim_entry = ptr::null_mut();
            (*cur).matrix_decoded = ptr::null_mut();

            match (*cur).id {
                X3F_CMBP => x3f_setup_camf_property_entry(cur),
                X3F_CMBT => x3f_setup_camf_text_entry(cur),
                X3F_CMBM => x3f_setup_camf_matrix_entry(cur),
                _ => {}
            }

            p = p.add((*cur).entry_size as usize);
            i += 1;
        }

        (*camf).entry_table.size = i;
        (*camf).entry_table.element = entry;

        x3f_printf(
            x3f_verbosity_t_DEBUG,
            c"SETUP CAMF ENTRIES (READY) Found %d entries\n".as_ptr(),
            i as i32,
        );
    }
}

unsafe fn x3f_load_camf(info: *mut x3f_info_t, de: *mut x3f_directory_entry_t) {
    unsafe {
        let camf = &mut (*de).header.data_subsection.camf as *mut x3f_camf_t;
        x3f_printf(
            x3f_verbosity_t_DEBUG,
            c"Loading CAMF of type %d\n".as_ptr(),
            (*camf).type_ as i32,
        );
        read_data_set_offset(info, de, X3F_CAMF_HEADER_SIZE);
        (*camf).data_size = read_data_block(&mut (*camf).data, info, de, 0);

        match (*camf).type_ {
            2 => x3f_load_camf_decode_type2(camf),
            4 => x3f_load_camf_decode_type4(camf),
            5 => x3f_load_camf_decode_type5(camf),
            _ => {
                x3f_printf(x3f_verbosity_t_ERR, c"Unknown CAMF type\n".as_ptr());
            }
        }

        if !(*camf).decoded_data.is_null() {
            x3f_setup_camf_entries(camf);
        } else {
            x3f_printf(x3f_verbosity_t_ERR, c"No decoded CAMF data\n".as_ptr());
        }
    }
}

// ---------------------------------------------------------------------------
// Public extern "C" entry points: x3f_load_data, x3f_load_image_block, x3f_err.
// ---------------------------------------------------------------------------

#[no_mangle]
pub unsafe extern "C" fn x3f_load_data(
    x3f: *mut x3f_t,
    de: *mut x3f_directory_entry_t,
) -> x3f_return_t {
    unsafe {
        let info = &mut (*x3f).info as *mut x3f_info_t;
        if de.is_null() {
            return x3f_return_e_X3F_ARGUMENT_ERROR;
        }
        match (*de).header.identifier {
            X3F_SECP => x3f_load_property_list(info, de),
            X3F_SECI => x3f_load_image(info, de),
            X3F_SECC => x3f_load_camf(info, de),
            _ => {
                x3f_printf(
                    x3f_verbosity_t_ERR,
                    c"Unknown directory entry type\n".as_ptr(),
                );
                return x3f_return_e_X3F_INTERNAL_ERROR;
            }
        }
        x3f_return_e_X3F_OK
    }
}

#[no_mangle]
pub unsafe extern "C" fn x3f_load_image_block(
    x3f: *mut x3f_t,
    de: *mut x3f_directory_entry_t,
) -> x3f_return_t {
    unsafe {
        let info = &mut (*x3f).info as *mut x3f_info_t;
        if de.is_null() {
            return x3f_return_e_X3F_ARGUMENT_ERROR;
        }
        x3f_printf(x3f_verbosity_t_DEBUG, c"Load image block\n".as_ptr());
        match (*de).header.identifier {
            X3F_SECI => {
                read_data_set_offset(info, de, X3F_IMAGE_HEADER_SIZE);
                x3f_load_image_verbatim(info, de);
            }
            _ => {
                x3f_printf(
                    x3f_verbosity_t_ERR,
                    c"Unknown image directory entry type\n".as_ptr(),
                );
                return x3f_return_e_X3F_INTERNAL_ERROR;
            }
        }
        x3f_return_e_X3F_OK
    }
}

#[no_mangle]
pub unsafe extern "C" fn x3f_err(err: x3f_return_t) -> *mut c_char {
    let s: &core::ffi::CStr = match err {
        x3f_return_e_X3F_OK => c"ok",
        x3f_return_e_X3F_ARGUMENT_ERROR => c"argument error",
        x3f_return_e_X3F_INFILE_ERROR => c"infile error",
        x3f_return_e_X3F_OUTFILE_ERROR => c"outfile error",
        x3f_return_e_X3F_INTERNAL_ERROR => c"internal error",
        _ => c"undefined error",
    };
    s.as_ptr() as *mut c_char
}

// Cross-crate DCE anchors.
#[used]
static _ANCHOR_LOAD_DATA: unsafe extern "C" fn(
    *mut x3f_t,
    *mut x3f_directory_entry_t,
) -> x3f_return_t = x3f_load_data;
#[used]
static _ANCHOR_LOAD_IMAGE_BLOCK: unsafe extern "C" fn(
    *mut x3f_t,
    *mut x3f_directory_entry_t,
) -> x3f_return_t = x3f_load_image_block;
#[used]
static _ANCHOR_X3F_ERR: unsafe extern "C" fn(x3f_return_t) -> *mut c_char = x3f_err;
