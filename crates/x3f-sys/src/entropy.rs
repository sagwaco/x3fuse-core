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
pub(crate) struct HuffTree {
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

use crate::control::{Control, Error, Result};

pub(crate) struct BitReader<'a> {
    bytes: &'a [u8],
    bit: usize,
}

impl<'a> BitReader<'a> {
    pub(crate) fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, bit: 0 }
    }
    #[inline]
    pub(crate) fn get_bit(&mut self) -> Result<u8> {
        let byte = *self
            .bytes
            .get(self.bit / 8)
            .ok_or(Error::InvalidData("truncated entropy stream"))?;
        let result = (byte >> (7 - self.bit % 8)) & 1;
        self.bit += 1;
        Ok(result)
    }
}

#[inline]
unsafe fn read_leaf(br: &mut BitReader<'_>, tree: *const HuffTree) -> Result<u32> {
    if tree.is_null() {
        return Err(Error::InvalidData("missing Huffman tree"));
    }
    let tree = unsafe { &*tree };
    if tree.nodes.is_null() || tree.free_node_index == 0 {
        return Err(Error::InvalidData("empty Huffman tree"));
    }
    let start = tree.nodes as usize;
    let bytes = tree.free_node_index as usize * std::mem::size_of::<HuffNode>();
    let mut node = tree.nodes;
    for _ in 0..=27 {
        let offset = (node as usize)
            .checked_sub(start)
            .ok_or(Error::InvalidData("Huffman node outside tree"))?;
        if offset >= bytes || offset % std::mem::size_of::<HuffNode>() != 0 {
            return Err(Error::InvalidData("Huffman node outside tree"));
        }
        let value = unsafe { &*node };
        if value.branch[0].is_null() && value.branch[1].is_null() {
            if value.leaf == u32::MAX {
                return Err(Error::InvalidData("undefined Huffman code"));
            }
            return Ok(value.leaf);
        }
        node = value.branch[br.get_bit()? as usize];
        if node.is_null() {
            return Err(Error::InvalidData("invalid Huffman code"));
        }
    }
    Err(Error::InvalidData("Huffman tree exceeds maximum depth"))
}

#[inline]
pub(crate) unsafe fn true_diff(br: &mut BitReader<'_>, tree: *const HuffTree) -> Result<i32> {
    let bits = unsafe { read_leaf(br, tree) }?;
    if bits > 31 {
        return Err(Error::InvalidData("TRUE difference exceeds 31 bits"));
    }
    if bits == 0 {
        return Ok(0);
    }
    let first = br.get_bit()?;
    let mut difference = first as i64;
    for _ in 1..bits {
        difference = (difference << 1) | br.get_bit()? as i64;
    }
    if first == 0 {
        difference -= (1i64 << bits) - 1;
    }
    Ok(difference as i32)
}

unsafe fn true_decode_one_color(
    id_ptr: *const ImageData,
    color: usize,
    control: Control<'_>,
) -> Result<()> {
    let id = unsafe { &*id_ptr };
    if id.tru.is_null() || id.data.is_null() {
        return Err(Error::InvalidData("missing TRUE data"));
    }
    let tru = unsafe { &*id.tru };
    if tru.plane_size.size != 3 || tru.plane_size.element.is_null() {
        return Err(Error::InvalidData("missing TRUE plane lengths"));
    }
    let mut rows = id.rows;
    let mut cols = id.columns;
    let mut output = &tru.x3rgb16;
    let mut channel = color;
    if is_quattro_format(id.type_format) {
        if id.quattro.is_null() {
            return Err(Error::InvalidData("missing Quattro planes"));
        }
        let q = unsafe { &*id.quattro };
        rows = q.plane[color].rows as u32;
        cols = q.plane[color].columns as u32;
        if q.quattro_layout != 0 && color == 2 {
            output = &q.top16;
            channel = 0;
        }
    }
    if output.data.is_null()
        || rows != output.rows
        || cols < output.columns
        || channel >= output.channels as usize
        || output.row_stride
            < output
                .columns
                .checked_mul(output.channels)
                .ok_or(Error::Allocation)?
    {
        return Err(Error::InvalidData("invalid TRUE output geometry"));
    }
    let start = (tru.plane_address[color] as usize)
        .checked_sub(id.data as usize)
        .ok_or(Error::InvalidData("TRUE plane outside image"))?;
    let length = unsafe { *tru.plane_size.element.add(color) } as usize;
    if start
        .checked_add(length)
        .map_or(true, |end| end > id.data_size as usize)
    {
        return Err(Error::InvalidData("TRUE plane outside image"));
    }
    let bytes = unsafe { std::slice::from_raw_parts(tru.plane_address[color], length) };
    let mut br = BitReader::new(bytes);
    let mut starts = [[tru.seed[color] as i32; 2]; 2];
    for row in 0..rows {
        control.check()?;
        let parity = (row & 1) as usize;
        let mut acc = [0i32; 2];
        for col in 0..cols {
            if col % 4096 == 0 {
                control.check()?;
            }
            let odd = (col & 1) as usize;
            let previous = if col < 2 {
                starts[parity][odd]
            } else {
                acc[odd]
            };
            let diff = unsafe { true_diff(&mut br, &tru.tree) }?;
            let value = previous
                .checked_add(diff)
                .ok_or(Error::InvalidData("TRUE predictor overflow"))?;
            acc[odd] = value;
            if col < 2 {
                starts[parity][odd] = value;
            }
            if col < output.columns {
                let index = row as usize * output.row_stride as usize
                    + col as usize * output.channels as usize
                    + channel;
                unsafe { *output.data.add(index) = value as u16 };
            }
        }
    }
    Ok(())
}

pub(crate) unsafe fn true_decode(id: *mut ImageData, control: Control<'_>) -> Result<()> {
    control.check()?;
    if id.is_null() {
        return Err(Error::InvalidData("missing TRUE image"));
    }
    #[derive(Clone, Copy)]
    struct SyncId(*const ImageData);
    unsafe impl Send for SyncId {}
    unsafe impl Sync for SyncId {}
    impl SyncId {
        fn get(self) -> *const ImageData {
            self.0
        }
    }
    let shared = SyncId(id);
    use rayon::prelude::*;
    (0..3usize)
        .into_par_iter()
        .try_for_each(|color| unsafe { true_decode_one_color(shared.get(), color, control) })
}

#[no_mangle]
pub(crate) unsafe extern "C" fn x3f_rust_true_decode(id: *mut ImageData) {
    let _ = unsafe { true_decode(id, Control::none()) };
}

const X3F_IMAGE_RAW_HUFFMAN_X530: u32 = 0x0003_0005;
const X3F_IMAGE_RAW_HUFFMAN_10BIT: u32 = 0x0003_0006;
const X3F_IMAGE_THUMB_HUFFMAN: u32 = 0x0002_000b;

unsafe fn write_huffman_pixel(
    id: &ImageData,
    huf: &Huffman,
    row: u32,
    col: usize,
    color: usize,
    value: u32,
) -> Result<()> {
    let index = 3 * (row as usize * id.columns as usize + col) + color;
    match id.type_format {
        X3F_IMAGE_RAW_HUFFMAN_X530 | X3F_IMAGE_RAW_HUFFMAN_10BIT => {
            if huf.x3rgb16.data.is_null() {
                return Err(Error::InvalidData("missing Huffman output"));
            }
            unsafe { *huf.x3rgb16.data.add(index) = value as u16 };
        }
        X3F_IMAGE_THUMB_HUFFMAN => {
            if huf.rgb8.data.is_null() {
                return Err(Error::InvalidData("missing Huffman thumbnail output"));
            }
            unsafe { *huf.rgb8.data.add(index) = value as u8 };
        }
        _ => return Err(Error::InvalidData("unsupported Huffman format")),
    }
    Ok(())
}

unsafe fn huffman_decode_row(
    id: &ImageData,
    huf: &Huffman,
    row: u32,
    offset: i32,
    minimum: &mut i32,
    control: Control<'_>,
) -> Result<()> {
    let start = unsafe { *huf.row_offsets.element.add(row as usize) } as usize;
    let end = if row + 1 < id.rows {
        (unsafe { *huf.row_offsets.element.add(row as usize + 1) }) as usize
    } else {
        id.data_size as usize
    };
    if start >= end || end > id.data_size as usize {
        return Err(Error::InvalidData("Huffman row outside image"));
    }
    let bytes = unsafe { std::slice::from_raw_parts(id.data.cast::<u8>().add(start), end - start) };
    let mut br = BitReader::new(bytes);
    let mut acc = [offset as i16; 3];
    for col in 0..id.columns as usize {
        if col % 4096 == 0 {
            control.check()?;
        }
        for (color, value) in acc.iter_mut().enumerate() {
            let diff = unsafe { read_leaf(&mut br, &huf.tree) }? as i32;
            *value = value.wrapping_add(diff as i16);
            *minimum = (*minimum).min(*value as i32);
            unsafe {
                write_huffman_pixel(id, huf, row, col, color, (*value as i32).max(0) as u32)
            }?;
        }
    }
    Ok(())
}

pub(crate) unsafe fn huffman_decode(
    id: *mut ImageData,
    _bits: i32,
    control: Control<'_>,
) -> Result<()> {
    control.check()?;
    if id.is_null() {
        return Err(Error::InvalidData("missing Huffman image"));
    }
    let id = unsafe { &*id };
    if id.huffman.is_null() || id.data.is_null() {
        return Err(Error::InvalidData("missing Huffman data"));
    }
    let huf = unsafe { &*id.huffman };
    if huf.row_offsets.size != id.rows || huf.row_offsets.element.is_null() {
        return Err(Error::InvalidData("missing Huffman row offsets"));
    }
    let mut minimum = 0i32;
    let mut offset = unsafe { crate::legacy_offset };
    for row in 0..id.rows {
        control.check()?;
        unsafe { huffman_decode_row(id, huf, row, offset, &mut minimum, control) }?;
    }
    if unsafe { crate::auto_legacy_offset } != 0 && minimum < 0 {
        offset = -minimum;
        for row in 0..id.rows {
            control.check()?;
            unsafe { huffman_decode_row(id, huf, row, offset, &mut minimum, control) }?;
        }
    }
    Ok(())
}

pub(crate) unsafe fn simple_decode(
    id: *mut ImageData,
    bits: i32,
    row_stride: i32,
    control: Control<'_>,
) -> Result<()> {
    control.check()?;
    if id.is_null() || row_stride <= 0 || !(8..=10).contains(&bits) {
        return Err(Error::InvalidData("invalid simple decoder dimensions"));
    }
    let id = unsafe { &*id };
    if id.huffman.is_null() || id.data.is_null() {
        return Err(Error::InvalidData("missing simple image data"));
    }
    let huf = unsafe { &*id.huffman };
    let stride = row_stride as usize;
    if id.columns as usize * 4 > stride
        || stride
            .checked_mul(id.rows as usize)
            .map_or(true, |n| n > id.data_size as usize)
    {
        return Err(Error::InvalidData("simple image raster exceeds input"));
    }
    let mask = (1u32 << bits) - 1;
    for row in 0..id.rows {
        control.check()?;
        let mut acc = [0u16; 3];
        for col in 0..id.columns as usize {
            if col % 4096 == 0 {
                control.check()?;
            }
            let address = unsafe { id.data.cast::<u8>().add(row as usize * stride + col * 4) };
            let packed = unsafe { u32::from_le(ptr_read_u32(address)) };
            for (color, value) in acc.iter_mut().enumerate() {
                let index = ((packed >> (color * bits as usize)) & mask) as usize;
                let diff = if huf.mapping.size == 0 {
                    index as u16
                } else {
                    if index >= huf.mapping.size as usize || huf.mapping.element.is_null() {
                        return Err(Error::InvalidData("simple mapping index outside table"));
                    }
                    unsafe { *huf.mapping.element.add(index) }
                };
                *value = value.wrapping_add(diff);
                let positive = if id.type_format == X3F_IMAGE_THUMB_HUFFMAN {
                    (*value as u8 as i8) > 0
                } else {
                    (*value as i16) > 0
                };
                unsafe {
                    write_huffman_pixel(
                        id,
                        huf,
                        row,
                        col,
                        color,
                        if positive { *value as u32 } else { 0 },
                    )
                }?;
            }
        }
    }
    Ok(())
}

unsafe fn ptr_read_u32(pointer: *const u8) -> u32 {
    unsafe { std::ptr::read_unaligned(pointer.cast::<u32>()) }
}

#[no_mangle]
pub(crate) unsafe extern "C" fn x3f_rust_huffman_decode(
    id: *mut ImageData,
    bits: std::os::raw::c_int,
) {
    let _ = unsafe { huffman_decode(id, bits, Control::none()) };
}

#[no_mangle]
pub(crate) unsafe extern "C" fn x3f_rust_simple_decode(
    id: *mut ImageData,
    bits: std::os::raw::c_int,
    row_stride: std::os::raw::c_int,
) {
    let _ = unsafe { simple_decode(id, bits, row_stride, Control::none()) };
}

#[used]
static _ANCHOR_X3F_RUST_TRUE_DECODE: unsafe extern "C" fn(*mut ImageData) = x3f_rust_true_decode;
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
        let mut br = BitReader::new(&bytes);
        let mut got = 0u32;
        for _ in 0..16 {
            got = (got << 1) | br.get_bit().unwrap() as u32;
        }
        assert_eq!(got, 0b1010_0110_1100_0011);
        assert!(matches!(br.get_bit(), Err(Error::InvalidData(_))));
    }

    #[test]
    fn malformed_codes_and_cancelled_decode_return_errors() {
        for leaf in [32, u32::MAX] {
            let mut nodes = [HuffNode {
                branch: [std::ptr::null_mut(); 2],
                leaf,
            }];
            let tree = HuffTree {
                free_node_index: 1,
                nodes: nodes.as_mut_ptr(),
            };
            assert!(unsafe { true_diff(&mut BitReader::new(&[0]), &tree) }.is_err());
        }
        let token = std::sync::atomic::AtomicBool::new(true);
        assert!(matches!(
            unsafe { true_decode(std::ptr::null_mut(), Control::new(&token)) },
            Err(Error::Cancelled)
        ));
    }

    /// Synthesise a tiny TRUE-coded plane and decode it. Tree has one symbol
    /// of length 1: code 0 -> magnitude bits 0 (no bits read, diff = 0).
    /// All bits 0 -> all diffs = 0 -> all values = seed.
    #[test]
    fn decode_constant_seed_plane() {
        // Tree: single root node with two branches both leading to the
        // length-0 leaf (so both bits 0 and 1 yield 0-magnitude diffs).
        let mut nodes = [
            HuffNode {
                branch: [std::ptr::null_mut(); 2],
                leaf: u32::MAX,
            },
            HuffNode {
                branch: [std::ptr::null_mut(); 2],
                leaf: 0,
            },
        ];
        nodes[0].branch = [&mut nodes[1] as *mut HuffNode; 2];
        let tree = HuffTree {
            free_node_index: 2,
            nodes: nodes.as_mut_ptr(),
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

        let mut sizes = [4u32; 3];
        let mut tru = True {
            seed: [123, 0, 0],
            unknown: 0,
            table: TrueHuffmanTable {
                size: 0,
                element: std::ptr::null_mut(),
            },
            plane_size: Table32 {
                size: 3,
                element: sizes.as_mut_ptr(),
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
            data: stream.as_ptr() as *mut c_void,
            data_size: stream.len() as u32,
        };

        unsafe { true_decode_one_color(&id as *const ImageData, 0, Control::none()) }.unwrap();

        for row in 0..4 {
            for col in 0..4 {
                let v = buf[(row * 4 + col) * 3]; // color 0 of pixel (row, col)
                assert_eq!(v, 123, "row {row} col {col}");
            }
        }
    }
}
