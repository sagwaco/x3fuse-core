//! M5b: native Rust TRUE entropy decoder.
//!
//! Opt-in via `X3F_RUST_DECODE=1`. The C dispatch in `src/x3f_io.c`'s
//! `true_decode` checks the env var and calls [`x3f_rust_true_decode`]
//! here instead of the per-color C loop. The C side still owns:
//!   - reading the file bytes into `ID->data`
//!   - reading the Huffman table header
//!   - building the Huffman tree (`populate_true_huffman_tree`)
//!   - allocating the `TRU->x3rgb16` and `Q->top16` output buffers
//!   - computing `TRU->plane_address[]` and the per-plane geometry
//!
//! All this Rust code does is the per-color decode loop: walk the prebuilt
//! Huffman tree to read a symbol-length, read that many magnitude bits,
//! sign-extend, accumulate horizontally, and write into the output buffer.
//! It is exact integer math — Tier-3 must show byte-equality between the
//! C and Rust paths.
//!
//! Symbol export: `#[no_mangle] extern "C"` like quattro.rs. A `#[used]`
//! static anchors the symbol so cross-crate DCE does not strip it before
//! the legacy C library's call site is linked.

use std::os::raw::c_void;

// ---------------------------------------------------------------------------
// Mirrored C structs from x3f_io.h. Layout-asserted by the test below; if
// the headers change we want a hard build break, not silent mis-decoding.
// ---------------------------------------------------------------------------

#[repr(C)]
struct Area16 {
    data: *mut u16,
    buf: *mut c_void,
    rows: u32,
    columns: u32,
    channels: u32,
    row_stride: u32,
}

#[repr(C)]
struct HuffNode {
    branch: [*mut HuffNode; 2],
    leaf: u32,
}

#[repr(C)]
struct HuffTree {
    free_node_index: u32,
    nodes: *mut HuffNode,
}

#[repr(C)]
struct TrueHuffmanElement {
    code_size: u8,
    code: u8,
}

#[repr(C)]
struct TrueHuffmanTable {
    size: u32,
    element: *mut TrueHuffmanElement,
}

#[repr(C)]
struct Table32 {
    size: u32,
    element: *mut u32,
}

#[repr(C)]
struct Table16 {
    size: u32,
    element: *mut u16,
}

#[repr(C)]
struct Area8 {
    data: *mut u8,
    buf: *mut c_void,
    rows: u32,
    columns: u32,
    channels: u32,
    row_stride: u32,
}

#[repr(C)]
struct Huffman {
    mapping: Table16,
    table: Table32,
    tree: HuffTree,
    row_offsets: Table32,
    rgb8: Area8,
    x3rgb16: Area16,
}

const TRUE_PLANES: usize = 3;

#[repr(C)]
struct True {
    seed: [u16; TRUE_PLANES],
    unknown: u16,
    table: TrueHuffmanTable,
    plane_size: Table32,
    plane_address: [*mut u8; TRUE_PLANES],
    tree: HuffTree,
    x3rgb16: Area16,
}

#[repr(C)]
struct QuattroPlane {
    columns: u16,
    rows: u16,
}

#[repr(C)]
struct Quattro {
    plane: [QuattroPlane; TRUE_PLANES],
    unknown: u32,
    quattro_layout: i32, // C `bool_t` is `int`
    top16: Area16,
}

#[repr(C)]
pub(crate) struct ImageData {
    type_: u32,
    format: u32,
    type_format: u32,
    columns: u32,
    rows: u32,
    row_stride: u32,
    huffman: *mut Huffman,
    tru: *mut True,
    quattro: *mut Quattro,
    data: *mut c_void,
    data_size: u32,
}

// Format constants from x3f_io.h.
const X3F_IMAGE_RAW_QUATTRO: u32 = 0x0001_0023;
const X3F_IMAGE_RAW_SDQ: u32 = 0x0001_0025;
const X3F_IMAGE_RAW_SDQH: u32 = 0x0001_0027;

#[inline]
fn is_quattro_format(t: u32) -> bool {
    matches!(
        t,
        X3F_IMAGE_RAW_QUATTRO | X3F_IMAGE_RAW_SDQ | X3F_IMAGE_RAW_SDQH
    )
}

// ---------------------------------------------------------------------------
// Bit reader. Mirrors the C `bit_state_t` semantics: each byte is split into
// eight individual bits at consume time, MSB first. The C code pre-allocates
// an 8-byte staging array for each byte; we keep the same shape so that any
// off-by-one in either implementation surfaces identically.
// ---------------------------------------------------------------------------

struct BitReader {
    next: *const u8,
    bit_offset: u8,
    bits: [u8; 8],
}

impl BitReader {
    unsafe fn new(addr: *const u8) -> Self {
        BitReader {
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

/// One TRUE-coded difference: walk the Huffman tree to a leaf to get a bit
/// length, then read that many magnitude bits and sign-extend per the
/// classic JPEG-style scheme.
#[inline]
unsafe fn get_true_diff(br: &mut BitReader, tree: *const HuffTree) -> i32 {
    let nodes = unsafe { (*tree).nodes };
    let mut node: *const HuffNode = nodes;

    while !unsafe { (*node).branch[0] }.is_null() || !unsafe { (*node).branch[1] }.is_null() {
        let bit = unsafe { br.get_bit() };
        let next = unsafe { (*node).branch[bit as usize] };
        if next.is_null() {
            // Mirror the C error path: log-and-zero rather than panic.
            return 0;
        }
        node = next;
    }

    let bits = unsafe { (*node).leaf } as u32;
    if bits == 0 {
        return 0;
    }

    let first_bit = unsafe { br.get_bit() };
    let mut diff = first_bit as i32;
    for _ in 1..bits {
        diff = (diff << 1) + unsafe { br.get_bit() } as i32;
    }
    if first_bit == 0 {
        diff -= (1 << bits) - 1;
    }
    diff
}

/// Decode one of the three color planes of a TRUE-coded RAW image. Mirrors
/// `true_decode_one_color` in src/x3f_io.c.
unsafe fn true_decode_one_color(id_ptr: *const ImageData, color: usize) {
    let id = unsafe { &*id_ptr };
    let tru = unsafe { &*id.tru };

    let seed = tru.seed[color] as i32;
    let mut rows = id.rows;
    let mut cols = id.columns;

    let (area_data_base, area_cols, area_channels) = if is_quattro_format(id.type_format) {
        let q = unsafe { &*id.quattro };
        rows = q.plane[color].rows as u32;
        cols = q.plane[color].columns as u32;
        if q.quattro_layout != 0 && color == 2 {
            // Quattro top plane is single-channel and lives in q.top16.
            (q.top16.data, q.top16.columns, q.top16.channels as usize)
        } else {
            let cs = tru.x3rgb16.channels as usize;
            (
                unsafe { tru.x3rgb16.data.add(color) },
                tru.x3rgb16.columns,
                cs,
            )
        }
    } else {
        let cs = tru.x3rgb16.channels as usize;
        (
            unsafe { tru.x3rgb16.data.add(color) },
            tru.x3rgb16.columns,
            cs,
        )
    };

    let mut br = unsafe { BitReader::new(tru.plane_address[color]) };
    let mut row_start_acc = [[seed; 2]; 2];
    let mut dst = area_data_base;

    for row in 0..rows {
        let odd_row = (row & 1) as usize;
        let mut acc = [0i32; 2];

        for col in 0..cols {
            let odd_col = (col & 1) as usize;
            let diff = unsafe { get_true_diff(&mut br, &tru.tree) };
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

            // Discard padding columns at the right for binned Quattro plane 2.
            if col >= area_cols {
                continue;
            }

            unsafe { *dst = value as u16 };
            dst = unsafe { dst.add(area_channels) };
        }
    }
}

/// Native Rust replacement for `true_decode` in src/x3f_io.c. Decodes all
/// three color planes of a TRUE-coded RAW image into the C-side preallocated
/// `TRU->x3rgb16` (and `Q->top16` for quattro_layout color 2) buffers.
///
/// M7d — the three color planes have *independent* Huffman bitstreams
/// (`tru.plane_address[color]`) and write to *disjoint* output regions
/// (color 0/1/2 stagger into `tru.x3rgb16.data` at offsets {0,1,2} with
/// stride `channels`=3, or color 2 in Quattro layout writes to its own
/// `q.top16.data` buffer entirely). Shared reads are immutable
/// (`tru.tree`, `tru.seed[]`, dimensions). Decoding the three planes on
/// parallel rayon workers cuts entropy-decode wall-time by ~3× on
/// Merrill TRUE inputs, which dominate single-image cost.
///
/// # Safety
///
/// `id` must be a non-null `*mut x3f_image_data_t` whose `tru` field has been
/// fully populated by `x3f_load_true`: Huffman tree built, plane addresses
/// computed, output buffers allocated.
#[no_mangle]
pub(crate) unsafe extern "C" fn x3f_rust_true_decode(id: *mut ImageData) {
    assert!(!id.is_null(), "x3f_rust_true_decode: NULL ImageData");

    // Sync-wrap the pointer so rayon's `Sync` closure bound is satisfied.
    // SAFETY: see fn-level comment — disjoint writes per color, immutable
    // shared reads through the `*const ImageData`.
    //
    // The pointer is exposed via a method (`as_ptr()`) rather than a tuple
    // field so Rust 2021's disjoint-capture rule captures the whole
    // (Sync) struct rather than the bare `*const ImageData` field.
    #[derive(Copy, Clone)]
    struct SyncId(*const ImageData);
    unsafe impl Send for SyncId {}
    unsafe impl Sync for SyncId {}
    impl SyncId {
        #[inline(always)]
        fn as_ptr(self) -> *const ImageData {
            self.0
        }
    }
    let sid = SyncId(id as *const ImageData);

    use rayon::prelude::*;
    (0..3usize).into_par_iter().for_each(|color| {
        unsafe { true_decode_one_color(sid.as_ptr(), color) };
    });
}

#[used]
static _ANCHOR_X3F_RUST_TRUE_DECODE: unsafe extern "C" fn(*mut ImageData) = x3f_rust_true_decode;

// ---------------------------------------------------------------------------
// M5c — Huffman + simple decoders.
//
// `huffman_decode` is the legacy path for X3F_IMAGE_RAW_HUFFMAN_X530 and
// X3F_IMAGE_RAW_HUFFMAN_10BIT (older SD9/SD10-class raws), and for
// X3F_IMAGE_THUMB_HUFFMAN (older thumbnails). Each row starts at an offset
// listed in HUF->row_offsets; per pixel we walk the Huffman tree to a leaf,
// add its value to the running per-channel accumulator. Output is 8-bit
// for thumbs, 16-bit for raw.
//
// `simple_decode` is the uncompressed variant: each row is laid out as
// packed `bits`-bit RGB triples in u32 words; we mask out each component
// and (optionally) map it through HUF->mapping for X3F-lossy-compression
// reverse-mapping.
//
// **Validation status:** the tier-3 differential test only exercises the
// THUMB_HUFFMAN path on corpus that has Huffman thumbnails. We currently
// have no SD9/SD10 raw files (see docs/PORT-PLAN.md M5 corpus question)
// so RAW_HUFFMAN_* and `simple_decode` are sensor-validated at most via
// the same code path on thumbnails. Layout asserts + structural unit
// tests catch the obvious bugs; full sensor parity comes when the corpus
// is augmented.
//
// Format constants
const X3F_IMAGE_RAW_HUFFMAN_X530: u32 = 0x0003_0005;
const X3F_IMAGE_RAW_HUFFMAN_10BIT: u32 = 0x0003_0006;
const X3F_IMAGE_THUMB_HUFFMAN: u32 = 0x0002_000b;

extern "C" {
    static mut legacy_offset: std::os::raw::c_int;
    static mut auto_legacy_offset: std::os::raw::c_int;
}

#[inline]
unsafe fn get_huffman_diff(br: &mut BitReader, tree: *const HuffTree) -> i32 {
    let nodes = unsafe { (*tree).nodes };
    let mut node: *const HuffNode = nodes;
    while !unsafe { (*node).branch[0] }.is_null() || !unsafe { (*node).branch[1] }.is_null() {
        let bit = unsafe { br.get_bit() };
        let next = unsafe { (*node).branch[bit as usize] };
        if next.is_null() {
            return 0;
        }
        node = next;
    }
    unsafe { (*node).leaf as i32 }
}

#[inline]
unsafe fn get_simple_diff(huf: &Huffman, index: u16) -> i32 {
    if huf.mapping.size == 0 {
        index as i32
    } else {
        let elem = unsafe { *huf.mapping.element.add(index as usize) };
        elem as i32
    }
}

/// Mirror of `huffman_decode_row`. Decodes one row of either a 16-bit raw
/// (X3F_IMAGE_RAW_HUFFMAN_*) or an 8-bit thumb (X3F_IMAGE_THUMB_HUFFMAN).
unsafe fn huffman_decode_row(
    id: &ImageData,
    huf: &Huffman,
    row: u32,
    offset: i32,
    minimum: &mut i32,
) {
    let row_off = unsafe { *huf.row_offsets.element.add(row as usize) };
    let stream_base = unsafe { (id.data as *const u8).add(row_off as usize) };
    let mut br = unsafe { BitReader::new(stream_base) };

    let mut c: [i16; 3] = [offset as i16, offset as i16, offset as i16];
    let cols = id.columns as usize;
    let row_us = row as usize;

    for col in 0..cols {
        for color in 0..3usize {
            let diff = unsafe { get_huffman_diff(&mut br, &huf.tree) };
            c[color] = c[color].wrapping_add(diff as i16);
            let c_val = c[color] as i32;
            let c_fix = if c_val < 0 {
                if c_val < *minimum {
                    *minimum = c_val;
                }
                0u32
            } else {
                c_val as u32
            };

            match id.type_format {
                X3F_IMAGE_RAW_HUFFMAN_X530 | X3F_IMAGE_RAW_HUFFMAN_10BIT => unsafe {
                    *huf.x3rgb16.data.add(3 * (row_us * cols + col) + color) = c_fix as u16;
                },
                X3F_IMAGE_THUMB_HUFFMAN => unsafe {
                    *huf.rgb8.data.add(3 * (row_us * cols + col) + color) = c_fix as u8;
                },
                // The C falls through to a printf and silently corrupts; we
                // do the same (no write) — should be unreachable since the
                // dispatcher only calls us for these three type_formats.
                _ => {}
            }
        }
    }
}

/// Mirror of `huffman_decode`: per-row decode driver with the auto-
/// legacy-offset retry. If `auto_legacy_offset` is set and any row dipped
/// below zero on the first pass, we re-decode all rows with the offset
/// shifted up so the negative excursion lands at zero.
unsafe fn huffman_decode_impl(id: *mut ImageData, _bits: i32) {
    let id_ref = unsafe { &*id };
    let huf = unsafe { &*id_ref.huffman };
    let mut minimum: i32 = 0;
    let mut offset: i32 = unsafe { legacy_offset };

    for row in 0..id_ref.rows {
        unsafe { huffman_decode_row(id_ref, huf, row, offset, &mut minimum) };
    }

    let auto = unsafe { auto_legacy_offset };
    if auto != 0 && minimum < 0 {
        offset = -minimum;
        for row in 0..id_ref.rows {
            unsafe { huffman_decode_row(id_ref, huf, row, offset, &mut minimum) };
        }
    }
}

/// Mirror of `simple_decode_row`. Each row is `row_stride` bytes of
/// packed `bits`-per-component RGB triples in 32-bit words.
unsafe fn simple_decode_row(id: &ImageData, huf: &Huffman, bits: i32, row: u32, row_stride: u32) {
    let mask: u32 = match bits {
        8 => 0xff,
        9 => 0x1ff,
        10 => 0x3ff,
        11 => 0x7ff,
        12 => 0xfff,
        _ => return, // C logs and zeros; unreachable from the dispatcher
    };

    let row_off = (row as usize) * (row_stride as usize);
    let row_words = unsafe { (id.data as *const u32).add(row_off / 4) };

    let mut c: [u16; 3] = [0; 3];
    let cols = id.columns as usize;
    let row_us = row as usize;

    for col in 0..cols {
        let val = unsafe { *row_words.add(col) };
        for color in 0..3usize {
            let idx = ((val >> (color * bits as usize)) & mask) as u16;
            let d = unsafe { get_simple_diff(huf, idx) };
            c[color] = c[color].wrapping_add(d as u16);

            match id.type_format {
                X3F_IMAGE_RAW_HUFFMAN_X530 | X3F_IMAGE_RAW_HUFFMAN_10BIT => {
                    // C: `c_fix = (int16_t)c[color] > 0 ? c[color] : 0;`
                    let signed = c[color] as i16;
                    let c_fix = if signed > 0 { c[color] } else { 0 };
                    unsafe {
                        *huf.x3rgb16.data.add(3 * (row_us * cols + col) + color) = c_fix;
                    }
                }
                X3F_IMAGE_THUMB_HUFFMAN => {
                    let signed = (c[color] as u8) as i8;
                    let c_fix = if signed > 0 { c[color] as u8 } else { 0 };
                    unsafe {
                        *huf.rgb8.data.add(3 * (row_us * cols + col) + color) = c_fix;
                    }
                }
                _ => {}
            }
        }
    }
}

unsafe fn simple_decode_impl(id: *mut ImageData, bits: i32, row_stride: i32) {
    let id_ref = unsafe { &*id };
    let huf = unsafe { &*id_ref.huffman };
    for row in 0..id_ref.rows {
        unsafe { simple_decode_row(id_ref, huf, bits, row, row_stride as u32) };
    }
}

/// C entry point matching the `huffman_decode(I, DE, bits)` signature.
/// Called from the X3F_RUST_DECODE dispatch in src/x3f_io.c.
#[no_mangle]
pub(crate) unsafe extern "C" fn x3f_rust_huffman_decode(
    id: *mut ImageData,
    bits: std::os::raw::c_int,
) {
    assert!(!id.is_null(), "x3f_rust_huffman_decode: NULL ImageData");
    unsafe { huffman_decode_impl(id, bits) };
}

/// C entry point matching the `simple_decode(I, DE, bits, row_stride)`
/// signature.
#[no_mangle]
pub(crate) unsafe extern "C" fn x3f_rust_simple_decode(
    id: *mut ImageData,
    bits: std::os::raw::c_int,
    row_stride: std::os::raw::c_int,
) {
    assert!(!id.is_null(), "x3f_rust_simple_decode: NULL ImageData");
    unsafe { simple_decode_impl(id, bits, row_stride) };
}

#[used]
static _ANCHOR_X3F_RUST_HUFFMAN_DECODE: unsafe extern "C" fn(*mut ImageData, std::os::raw::c_int) =
    x3f_rust_huffman_decode;

#[used]
static _ANCHOR_X3F_RUST_SIMPLE_DECODE: unsafe extern "C" fn(
    *mut ImageData,
    std::os::raw::c_int,
    std::os::raw::c_int,
) = x3f_rust_simple_decode;

#[cfg(test)]
mod tests {
    use super::*;

    /// Layout assertions: the mirrored structs must match the bindgen-
    /// generated FFI structs byte-for-byte. If `x3f_io.h` changes shape,
    /// catch it at test time rather than at first decode of a real file.
    #[test]
    fn struct_layouts_match_bindgen() {
        use std::mem::{align_of, size_of};

        assert_eq!(
            size_of::<Area16>(),
            size_of::<crate::x3f_area16_t>(),
            "Area16"
        );
        assert_eq!(align_of::<Area16>(), align_of::<crate::x3f_area16_t>());

        assert_eq!(size_of::<HuffNode>(), size_of::<crate::x3f_huffnode_s>());
        assert_eq!(align_of::<HuffNode>(), align_of::<crate::x3f_huffnode_s>());

        assert_eq!(size_of::<HuffTree>(), size_of::<crate::x3f_hufftree_t>());

        assert_eq!(
            size_of::<TrueHuffmanElement>(),
            size_of::<crate::x3f_true_huffman_element_t>()
        );
        assert_eq!(
            size_of::<TrueHuffmanTable>(),
            size_of::<crate::x3f_true_huffman_t>()
        );
        assert_eq!(size_of::<Table32>(), size_of::<crate::x3f_table32_t>());

        assert_eq!(size_of::<True>(), size_of::<crate::x3f_true_t>());
        assert_eq!(align_of::<True>(), align_of::<crate::x3f_true_t>());

        assert_eq!(size_of::<Quattro>(), size_of::<crate::x3f_quattro_t>());

        assert_eq!(size_of::<Table16>(), size_of::<crate::x3f_table16_t>());
        assert_eq!(size_of::<Area8>(), size_of::<crate::x3f_area8_t>());
        assert_eq!(size_of::<Huffman>(), size_of::<crate::x3f_huffman_t>());
        assert_eq!(align_of::<Huffman>(), align_of::<crate::x3f_huffman_t>());

        assert_eq!(size_of::<ImageData>(), size_of::<crate::x3f_image_data_t>());
        assert_eq!(
            align_of::<ImageData>(),
            align_of::<crate::x3f_image_data_t>()
        );
    }

    /// Round-trip the bit reader on a known byte sequence. The C reads MSB
    /// first within each byte; this test pins that ordering.
    #[test]
    fn bit_reader_msb_first_within_byte() {
        let bytes: Vec<u8> = vec![0b1010_0110, 0b1100_0011];
        let mut br = unsafe { BitReader::new(bytes.as_ptr()) };
        let mut got = 0u32;
        for _ in 0..16 {
            got = (got << 1) | unsafe { br.get_bit() } as u32;
        }
        assert_eq!(got, 0b1010_0110_1100_0011);
    }

    /// Synthesise a tiny TRUE-coded plane and decode it. Tree has one symbol
    /// of length 1: code 0 -> magnitude bits 0 (no bits read, diff = 0).
    /// All bits 0 -> all diffs = 0 -> all values = seed.
    #[test]
    fn decode_constant_seed_plane() {
        // Tree: single root node with two branches both leading to the
        // length-0 leaf (so both bits 0 and 1 yield 0-magnitude diffs).
        let mut leaf = HuffNode {
            branch: [std::ptr::null_mut(); 2],
            leaf: 0,
        };
        let leaf_ptr = &mut leaf as *mut HuffNode;
        let mut root = HuffNode {
            branch: [leaf_ptr, leaf_ptr],
            leaf: 0,
        };
        let tree = HuffTree {
            free_node_index: 2,
            nodes: &mut root as *mut HuffNode,
        };

        // 1 byte of bit stream is enough to satisfy a 4x4 plane: 16 reads,
        // 16 bits in 2 bytes (we provide both as 0).
        let stream = vec![0u8; 4];

        let mut buf = vec![0u16; 4 * 4 * 3];
        let area = Area16 {
            data: buf.as_mut_ptr(),
            buf: std::ptr::null_mut(),
            rows: 4,
            columns: 4,
            channels: 3,
            row_stride: 4 * 3,
        };

        let mut tru = True {
            seed: [123, 0, 0],
            unknown: 0,
            table: TrueHuffmanTable {
                size: 0,
                element: std::ptr::null_mut(),
            },
            plane_size: Table32 {
                size: 0,
                element: std::ptr::null_mut(),
            },
            plane_address: [
                stream.as_ptr() as *mut u8,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            ],
            tree,
            x3rgb16: area,
        };

        let id = ImageData {
            type_: 3,
            format: 0x1e,
            type_format: 0x0003_001e, // TRUE
            columns: 4,
            rows: 4,
            row_stride: 12,
            huffman: std::ptr::null_mut(),
            tru: &mut tru as *mut True,
            quattro: std::ptr::null_mut(),
            data: std::ptr::null_mut(),
            data_size: 0,
        };

        unsafe { true_decode_one_color(&id as *const ImageData, 0) };

        for row in 0..4 {
            for col in 0..4 {
                let v = buf[(row * 4 + col) * 3]; // color 0 of pixel (row, col)
                assert_eq!(v, 123, "row {row} col {col}");
            }
        }
    }
}
