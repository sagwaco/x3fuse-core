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

use std::os::raw::c_char;
use std::ptr;

use crate::*;
// libc compat — see `sysabi.rs`. Shadows the external libc crate so
// `libc::*` resolves through our wasm32-unknown-unknown shim there.

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

use crate::control::{Control, Error, Result};
use crate::parse::{alloc, bytes_at, checked_size, copy_alloc, cstr_at, u32_at, Input};

/// Build a bounded tree. Its allocation is owned by the parsed section.
unsafe fn build_tree(tree: &mut x3f_hufftree_t, codes: &[(u32, u32, u32)]) -> Result<()> {
    let capacity = 1 + codes.iter().map(|c| c.0 as usize).sum::<usize>();
    tree.nodes = unsafe { alloc::<x3f_huffnode_t>(capacity) }?;
    tree.free_node_index = 1;
    unsafe { (*tree.nodes).leaf = u32::MAX };
    for &(length, code, value) in codes {
        if length == 0 || length > 27 || code >> length != 0 {
            return Err(Error::InvalidData("invalid Huffman code length"));
        }
        let mut node = tree.nodes;
        for bit_index in (0..length).rev() {
            unsafe {
                if (*node).leaf != u32::MAX {
                    return Err(Error::InvalidData("overlapping Huffman codes"));
                }
                let bit = ((code >> bit_index) & 1) as usize;
                if (*node).branch[bit].is_null() {
                    let index = tree.free_node_index as usize;
                    if index >= capacity {
                        return Err(Error::InvalidData("Huffman tree overflow"));
                    }
                    let next = tree.nodes.add(index);
                    (*next).leaf = u32::MAX;
                    (*node).branch[bit] = next;
                    tree.free_node_index += 1;
                }
                node = (*node).branch[bit];
            }
        }
        unsafe {
            if !(*node).branch[0].is_null()
                || !(*node).branch[1].is_null()
                || (*node).leaf != u32::MAX
            {
                return Err(Error::InvalidData("overlapping Huffman codes"));
            }
            (*node).leaf = value;
        }
    }
    if codes.is_empty() {
        return Err(Error::InvalidData("empty Huffman table"));
    }
    Ok(())
}

unsafe fn true_tree(tree: &mut x3f_hufftree_t, table: &x3f_true_huffman_t) -> Result<()> {
    if table.size > 32 {
        return Err(Error::InvalidData("TRUE magnitude exceeds 31 bits"));
    }
    let mut codes = Vec::new();
    for i in 0..table.size as usize {
        let element = unsafe { *table.element.add(i) };
        let length = element.code_size as u32;
        if length == 0 {
            continue;
        }
        if length > 8 {
            return Err(Error::InvalidData("TRUE code exceeds one byte"));
        }
        codes.push((length, (element.code as u32) >> (8 - length), i as u32));
    }
    unsafe { build_tree(tree, &codes) }
}

unsafe fn section_input<'a>(
    x3f: *mut x3f_t,
    de: *mut x3f_directory_entry_t,
    header_size: u32,
    control: Control<'a>,
) -> Result<Input<'a>> {
    if x3f.is_null() || de.is_null() {
        return Err(Error::InvalidData("missing section"));
    }
    let section = unsafe { &*de };
    if section.input.size < header_size {
        return Err(Error::InvalidData("truncated section header"));
    }
    let mut input = unsafe { Input::new((*x3f).info.input.file.cast(), control) }?;
    let start = section.input.offset as u64 + header_size as u64;
    let end = section.input.offset as u64 + section.input.size as u64;
    input.seek(start)?;
    input.limit(end)?;
    Ok(input)
}

unsafe fn load_properties(
    pl: &mut x3f_property_list_t,
    input: &mut Input<'_>,
    control: Control<'_>,
) -> Result<()> {
    let n = pl.num_properties as usize;
    if n > input.remaining() / 8 {
        return Err(Error::InvalidData("property count exceeds section"));
    }
    pl.property_table.element = unsafe { alloc::<x3f_property_t>(n) }?;
    pl.property_table.size = n as u32;
    for i in 0..n {
        let property = unsafe { &mut *pl.property_table.element.add(i) };
        property.name_offset = input.u32()?;
        property.value_offset = input.u32()?;
    }
    pl.data_size = input.remaining() as u32;
    pl.data = unsafe { input.read_alloc(input.remaining()) }?.cast();
    if n == 0 {
        return Ok(());
    }
    if pl.data_size == 0 {
        return Err(Error::InvalidData("missing property strings"));
    }
    let data = unsafe { std::slice::from_raw_parts(pl.data.cast::<u8>(), pl.data_size as usize) };
    for i in 0..n {
        control.check()?;
        let property = unsafe { &mut *pl.property_table.element.add(i) };
        for (offset, wide, utf8) in [
            (
                property.name_offset,
                &mut property.name,
                &mut property.name_utf8,
            ),
            (
                property.value_offset,
                &mut property.value,
                &mut property.value_utf8,
            ),
        ] {
            let byte_offset = checked_size(offset as usize, 2)?;
            let bytes = data
                .get(byte_offset..)
                .ok_or(Error::InvalidData("property string outside section"))?;
            let count = bytes
                .chunks_exact(2)
                .position(|pair| pair == [0, 0])
                .ok_or(Error::InvalidData("unterminated property string"))?;
            let units = bytes[..count * 2]
                .chunks_exact(2)
                .map(|pair| u16::from_le_bytes([pair[0], pair[1]]));
            let mut string = String::new();
            string
                .try_reserve_exact(checked_size(count, 3)?)
                .map_err(|_| Error::Allocation)?;
            for decoded in char::decode_utf16(units) {
                string.push(decoded.unwrap_or(char::REPLACEMENT_CHARACTER));
            }
            *wide = unsafe { pl.data.cast::<u8>().add(byte_offset).cast() };
            let encoded = unsafe { alloc::<u8>(string.len() + 1) }?;
            unsafe { ptr::copy_nonoverlapping(string.as_ptr(), encoded, string.len()) };
            *utf8 = encoded.cast();
        }
    }
    Ok(())
}

unsafe fn allocate_image(
    area: &mut x3f_area16_t,
    rows: u32,
    columns: u32,
    channels: u32,
) -> Result<()> {
    if rows == 0 || columns == 0 {
        return Err(Error::InvalidData("empty image plane"));
    }
    let stride = columns.checked_mul(channels).ok_or(Error::Allocation)?;
    let count = checked_size(rows as usize, stride as usize)?;
    let data = unsafe { alloc::<u16>(count) }?;
    *area = x3f_area16_t {
        data,
        buf: data.cast(),
        rows,
        columns,
        channels,
        row_stride: stride,
    };
    Ok(())
}

unsafe fn load_true(
    id: &mut x3f_image_data_t,
    input: &mut Input<'_>,
    control: Control<'_>,
) -> Result<()> {
    id.tru = unsafe { alloc::<x3f_true_t>(1) }?;
    let tru = unsafe { &mut *id.tru };
    let is_quattro = matches!(
        id.type_format,
        X3F_IMAGE_RAW_QUATTRO | X3F_IMAGE_RAW_SDQ | X3F_IMAGE_RAW_SDQH
    );
    if is_quattro {
        id.quattro = unsafe { alloc::<x3f_quattro_t>(1) }?;
        let q = unsafe { &mut *id.quattro };
        for plane in &mut q.plane {
            plane.columns = input.u16()?;
            plane.rows = input.u16()?;
        }
        q.quattro_layout = if q.plane[0].rows as u32 == id.rows / 2 {
            1
        } else if q.plane[0].rows as u32 == id.rows {
            0
        } else {
            return Err(Error::InvalidData("unknown Quattro layer size"));
        };
        if q.plane[0].columns != q.plane[1].columns || q.plane[0].rows != q.plane[1].rows {
            return Err(Error::InvalidData("inconsistent Quattro lower planes"));
        }
        if q.quattro_layout != 0 {
            if q.plane[0].columns as u32 * 2 > q.plane[2].columns as u32
                || q.plane[0].rows as u32 * 2 > q.plane[2].rows as u32
            {
                return Err(Error::InvalidData("inconsistent Quattro top plane"));
            }
        } else if q
            .plane
            .iter()
            .any(|p| p.rows as u32 != id.rows || (p.columns as u32) < id.columns)
        {
            return Err(Error::InvalidData("inconsistent binned Quattro planes"));
        }
    }
    for seed in &mut tru.seed {
        *seed = input.u16()?;
    }
    tru.unknown = input.u16()?;
    let mut table = Vec::new();
    loop {
        let element = x3f_true_huffman_element_t {
            code_size: input.u8()?,
            code: input.u8()?,
        };
        let last = element.code_size == 0;
        table.push(element);
        if last {
            break;
        }
        if table.len() >= 32 {
            return Err(Error::InvalidData("unterminated TRUE table"));
        }
    }
    tru.table.element = unsafe { copy_alloc(&table) }?;
    tru.table.size = table.len() as u32;
    if is_quattro {
        unsafe { (*id.quattro).unknown = input.u32()? };
    }
    tru.plane_size.element = unsafe { alloc::<u32>(TRUE_PLANES) }?;
    tru.plane_size.size = TRUE_PLANES as u32;
    for i in 0..TRUE_PLANES {
        unsafe { *tru.plane_size.element.add(i) = input.u32()? };
    }
    id.data_size = input.remaining() as u32;
    id.data = unsafe { input.read_alloc(input.remaining()) }?.cast();
    unsafe { true_tree(&mut tru.tree, &tru.table) }?;
    let mut offset = 0usize;
    for i in 0..TRUE_PLANES {
        let size = unsafe { *tru.plane_size.element.add(i) } as usize;
        if size == 0
            || offset
                .checked_add(size)
                .map_or(true, |end| end > id.data_size as usize)
        {
            return Err(Error::InvalidData("TRUE plane outside image data"));
        }
        tru.plane_address[i] = unsafe { id.data.cast::<u8>().add(offset) };
        offset = offset
            .checked_add((size + 15) & !15)
            .ok_or(Error::Allocation)?;
    }
    if is_quattro && unsafe { (*id.quattro).quattro_layout } != 0 {
        let q = unsafe { &mut *id.quattro };
        unsafe {
            allocate_image(
                &mut tru.x3rgb16,
                q.plane[0].rows as u32,
                q.plane[0].columns as u32,
                3,
            )
        }?;
        unsafe {
            allocate_image(
                &mut q.top16,
                q.plane[2].rows as u32,
                q.plane[2].columns as u32,
                1,
            )
        }?;
    } else {
        unsafe { allocate_image(&mut tru.x3rgb16, id.rows, id.columns, 3) }?;
    }
    unsafe { crate::entropy::true_decode(id as *mut _ as *mut crate::entropy::ImageData, control) }
}

unsafe fn load_huffman(
    id: &mut x3f_image_data_t,
    input: &mut Input<'_>,
    bits: u32,
    mapped: bool,
    control: Control<'_>,
) -> Result<()> {
    id.huffman = unsafe { alloc::<x3f_huffman_t>(1) }?;
    let h = unsafe { &mut *id.huffman };
    let n = 1usize << bits;
    if mapped {
        h.mapping.element = unsafe { alloc::<u16>(n) }?;
        h.mapping.size = n as u32;
        for i in 0..n {
            unsafe { *h.mapping.element.add(i) = input.u16()? };
        }
    }
    if id.type_format == X3F_IMAGE_THUMB_HUFFMAN {
        let count = checked_size(checked_size(id.columns as usize, id.rows as usize)?, 3)?;
        let data = unsafe { alloc::<u8>(count) }?;
        // Retain the legacy thumbnail structure fields used by metadata dumps.
        h.rgb8 = x3f_area8_t {
            data,
            buf: data.cast(),
            columns: id.rows,
            rows: 0,
            channels: 3,
            row_stride: id.columns * 3,
        };
    } else {
        unsafe { allocate_image(&mut h.x3rgb16, id.rows, id.columns, 3) }?;
    }
    if id.row_stride == 0 {
        h.table.element = unsafe { alloc::<u32>(n) }?;
        h.table.size = n as u32;
        for i in 0..n {
            unsafe { *h.table.element.add(i) = input.u32()? };
        }
        let footer = checked_size(id.rows as usize, 4)?;
        let size = input
            .remaining()
            .checked_sub(footer)
            .ok_or(Error::InvalidData("missing Huffman row offsets"))?;
        id.data = unsafe { input.read_alloc(size) }?.cast();
        id.data_size = size as u32;
        h.row_offsets.element = unsafe { alloc::<u32>(id.rows as usize) }?;
        h.row_offsets.size = id.rows;
        for i in 0..id.rows as usize {
            unsafe { *h.row_offsets.element.add(i) = input.u32()? };
        }
        let mut codes = Vec::new();
        for i in 0..n {
            let packed = unsafe { *h.table.element.add(i) };
            if packed == 0 {
                continue;
            }
            let value = if h.mapping.size == h.table.size {
                (unsafe { *h.mapping.element.add(i) }) as u32
            } else {
                i as u32
            };
            codes.push((packed >> 27, packed & 0x07ff_ffff, value));
        }
        unsafe { build_tree(&mut h.tree, &codes) }?;
        unsafe {
            crate::entropy::huffman_decode(
                id as *mut _ as *mut crate::entropy::ImageData,
                bits as i32,
                control,
            )
        }
    } else {
        id.data_size = input.remaining() as u32;
        id.data = unsafe { input.read_alloc(input.remaining()) }?.cast();
        unsafe {
            crate::entropy::simple_decode(
                id as *mut _ as *mut crate::entropy::ImageData,
                bits as i32,
                id.row_stride as i32,
                control,
            )
        }
    }
}

unsafe fn load_image(
    id: &mut x3f_image_data_t,
    input: &mut Input<'_>,
    control: Control<'_>,
) -> Result<()> {
    match id.type_format {
        X3F_IMAGE_RAW_TRUE
        | X3F_IMAGE_RAW_MERRILL
        | X3F_IMAGE_RAW_QUATTRO
        | X3F_IMAGE_RAW_SDQ
        | X3F_IMAGE_RAW_SDQH => unsafe { load_true(id, input, control) },
        X3F_IMAGE_RAW_HUFFMAN_X530 | X3F_IMAGE_RAW_HUFFMAN_10BIT => unsafe {
            load_huffman(id, input, 10, true, control)
        },
        X3F_IMAGE_THUMB_HUFFMAN => unsafe { load_huffman(id, input, 8, false, control) },
        X3F_IMAGE_THUMB_PLAIN | X3F_IMAGE_THUMB_JPEG => {
            id.data_size = input.remaining() as u32;
            id.data = unsafe { input.read_alloc(input.remaining()) }?.cast();
            if id.type_format == X3F_IMAGE_THUMB_PLAIN
                && checked_size(id.rows as usize, id.row_stride as usize)? > id.data_size as usize
            {
                return Err(Error::InvalidData("truncated thumbnail raster"));
            }
            Ok(())
        }
        _ => Err(Error::InvalidData("unsupported image encoding")),
    }
}

unsafe fn decode_camf(camf: &mut x3f_camf_t, control: Control<'_>) -> Result<()> {
    let data =
        unsafe { std::slice::from_raw_parts(camf.data.cast::<u8>(), camf.data_size as usize) };
    if camf.type_ == 2 {
        camf.decoded_data_size = camf.data_size;
        camf.decoded_data = unsafe { alloc::<u8>(data.len()) }?.cast();
        let mut key = unsafe { camf.__bindgen_anon_1.t2.crypt_key } as u64;
        for (i, &old) in data.iter().enumerate() {
            if i % 4096 == 0 {
                control.check()?;
            }
            key = (key * 1597 + 51749) % 244944;
            let tmp = ((key as i64) * 301_593_171_i64 >> 24) as u32;
            let a = ((key as u32) << 8).wrapping_sub(tmp);
            let mix = ((a >> 1).wrapping_add(tmp)) >> 17;
            unsafe { *camf.decoded_data.cast::<u8>().add(i) = old ^ mix as u8 };
        }
        return Ok(());
    }
    if !matches!(camf.type_, 4 | 5) {
        return Err(Error::InvalidData("unsupported CAMF encoding"));
    }
    bytes_at(data, 0, 32)?;
    let mut table = Vec::new();
    for pair in data[..28].chunks_exact(2) {
        if pair[0] == 0 {
            break;
        }
        table.push(x3f_true_huffman_element_t {
            code_size: pair[0],
            code: pair[1],
        });
    }
    if table.len() == 14 {
        return Err(Error::InvalidData("unterminated CAMF Huffman table"));
    }
    camf.table.element = unsafe { copy_alloc(&table) }?;
    camf.table.size = table.len() as u32;
    camf.decoding_size = u32_at(data, 28)?;
    if camf.decoding_size as usize > data.len() - 32 {
        return Err(Error::InvalidData("CAMF bitstream outside section"));
    }
    camf.decoding_start = unsafe { camf.data.cast::<u8>().add(32) };
    unsafe { true_tree(&mut camf.tree, &camf.table) }?;
    let size = unsafe { camf.__bindgen_anon_1.tN.val0 } as usize;
    if size == 0 {
        return Err(Error::InvalidData("empty decoded CAMF"));
    }
    camf.decoded_data = unsafe { alloc::<u8>(size) }?.cast();
    camf.decoded_data_size = size as u32;
    let mut bits = crate::entropy::BitReader::new(bytes_at(data, 32, camf.decoding_size as usize)?);
    let seed = unsafe { camf.__bindgen_anon_1.tN.val1 } as i32;
    if camf.type_ == 5 {
        let mut acc = seed;
        for i in 0..size {
            if i % 4096 == 0 {
                control.check()?;
            }
            let diff = unsafe {
                crate::entropy::true_diff(
                    &mut bits,
                    &camf.tree as *const _ as *const crate::entropy::HuffTree,
                )
            }?;
            acc = acc.wrapping_add(diff);
            unsafe { *camf.decoded_data.cast::<u8>().add(i) = acc as u8 };
        }
    } else {
        let rows = unsafe { camf.__bindgen_anon_1.t4.block_count };
        let cols = unsafe { camf.__bindgen_anon_1.t4.block_size };
        if rows == 0 || cols == 0 {
            return Err(Error::InvalidData("empty CAMF block grid"));
        }
        checked_size(rows as usize, cols as usize)?;
        let mut starts = [[seed; 2]; 2];
        let mut at = 0usize;
        let mut odd = false;
        'rows: for row in 0..rows {
            control.check()?;
            let mut acc = [0i32; 2];
            for col in 0..cols {
                if col % 4096 == 0 {
                    control.check()?;
                }
                let channel = (col & 1) as usize;
                let previous = if col < 2 {
                    starts[(row & 1) as usize][channel]
                } else {
                    acc[channel]
                };
                let diff = unsafe {
                    crate::entropy::true_diff(
                        &mut bits,
                        &camf.tree as *const _ as *const crate::entropy::HuffTree,
                    )
                }?;
                let value = previous
                    .checked_add(diff)
                    .ok_or(Error::InvalidData("CAMF predictor overflow"))?;
                acc[channel] = value;
                if col < 2 {
                    starts[(row & 1) as usize][channel] = value;
                }
                let dst = camf.decoded_data.cast::<u8>();
                unsafe {
                    if !odd {
                        *dst.add(at) = ((value >> 4) & 0xff) as u8;
                        at += 1;
                        if at >= size {
                            break 'rows;
                        }
                        *dst.add(at) = ((value << 4) & 0xf0) as u8;
                    } else {
                        *dst.add(at) |= ((value >> 8) & 0x0f) as u8;
                        at += 1;
                        if at >= size {
                            break 'rows;
                        }
                        *dst.add(at) = (value & 0xff) as u8;
                        at += 1;
                        if at >= size {
                            break 'rows;
                        }
                    }
                }
                odd = !odd;
            }
        }
        // Some cameras pad the decoded CAMF allocation with zeros; retain it.
    }
    Ok(())
}

unsafe fn setup_camf_entry(
    entry: &mut camf_entry_t,
    data: &[u8],
    control: Control<'_>,
) -> Result<()> {
    let base = data.as_ptr() as *mut u8;
    let value_offset = entry.value_offset as usize;
    let value = data
        .get(value_offset..)
        .ok_or(Error::InvalidData("CAMF value outside entry"))?;
    match entry.id {
        X3F_CMBT => {
            entry.text_size = u32_at(value, 0)?;
            bytes_at(value, 4, entry.text_size as usize)?;
            cstr_at(value, 4)?;
            entry.text = unsafe { base.add(value_offset + 4).cast() };
        }
        X3F_CMBP => {
            let n = u32_at(value, 0)? as usize;
            let string_offset = u32_at(value, 4)? as usize;
            bytes_at(value, 8, checked_size(n, 8)?)?;
            entry.property_name = unsafe { alloc::<*mut c_char>(n) }?;
            entry.property_value = unsafe { alloc::<*mut u8>(n) }?;
            entry.property_num = n as u32;
            for i in 0..n {
                control.check()?;
                let name = string_offset
                    .checked_add(u32_at(value, 8 + 8 * i)? as usize)
                    .ok_or(Error::Allocation)?;
                let val = string_offset
                    .checked_add(u32_at(value, 12 + 8 * i)? as usize)
                    .ok_or(Error::Allocation)?;
                cstr_at(data, name)?;
                cstr_at(data, val)?;
                unsafe {
                    *entry.property_name.add(i) = base.add(name).cast();
                    *entry.property_value.add(i) = base.add(val);
                }
            }
        }
        X3F_CMBM => {
            entry.matrix_type = u32_at(value, 0)?;
            entry.matrix_dim = u32_at(value, 4)?;
            entry.matrix_data_off = u32_at(value, 8)?;
            let dims = entry.matrix_dim as usize;
            bytes_at(value, 12, checked_size(dims, 12)?)?;
            let (element_size, decoded_type) = match entry.matrix_type {
                0 => (2, matrix_type_t_M_INT),
                1 | 2 => (4, matrix_type_t_M_UINT),
                3 => (4, matrix_type_t_M_FLOAT),
                5 => (1, matrix_type_t_M_UINT),
                6 => (2, matrix_type_t_M_UINT),
                _ => return Err(Error::InvalidData("unknown CAMF matrix type")),
            };
            entry.matrix_element_size = element_size;
            entry.matrix_decoded_type = decoded_type;
            entry.matrix_dim_entry = unsafe { alloc::<camf_dim_entry_t>(dims) }?;
            let mut elements = 1usize;
            for i in 0..dims {
                let dim = unsafe { &mut *entry.matrix_dim_entry.add(i) };
                dim.size = u32_at(value, 12 + 12 * i)?;
                dim.name_offset = u32_at(value, 16 + 12 * i)?;
                dim.n = u32_at(value, 20 + 12 * i)?;
                cstr_at(data, dim.name_offset as usize)?;
                dim.name = unsafe { base.add(dim.name_offset as usize).cast() };
                elements = checked_size(elements, dim.size as usize)?;
            }
            entry.matrix_elements = u32::try_from(elements).map_err(|_| Error::Allocation)?;
            let offset = entry.matrix_data_off as usize;
            let raw = bytes_at(data, offset, checked_size(elements, element_size as usize)?)?;
            entry.matrix_data = unsafe { base.add(offset).cast() };
            entry.matrix_used_space = (data.len() - offset) as u32;
            entry.matrix_estimated_element_size = if elements == 0 {
                0.0
            } else {
                entry.matrix_used_space as f64 / elements as f64
            };
            let unit = if decoded_type == matrix_type_t_M_FLOAT {
                8
            } else {
                4
            };
            entry.matrix_decoded = unsafe { alloc::<u8>(checked_size(elements, unit)?) }?.cast();
            for (i, bytes) in raw.chunks_exact(element_size as usize).enumerate() {
                if i % 4096 == 0 {
                    control.check()?;
                }
                unsafe {
                    match entry.matrix_type {
                        0 => {
                            *entry.matrix_decoded.cast::<i32>().add(i) =
                                i16::from_le_bytes([bytes[0], bytes[1]]) as i32
                        }
                        1 | 2 => {
                            *entry.matrix_decoded.cast::<u32>().add(i) =
                                u32::from_le_bytes(bytes.try_into().unwrap())
                        }
                        3 => {
                            *entry.matrix_decoded.cast::<f64>().add(i) =
                                f32::from_le_bytes(bytes.try_into().unwrap()) as f64
                        }
                        5 => *entry.matrix_decoded.cast::<u32>().add(i) = bytes[0] as u32,
                        6 => {
                            *entry.matrix_decoded.cast::<u32>().add(i) =
                                u16::from_le_bytes([bytes[0], bytes[1]]) as u32
                        }
                        _ => unreachable!(),
                    }
                }
            }
        }
        _ => return Err(Error::InvalidData("unknown CAMF entry type")),
    }
    Ok(())
}

unsafe fn setup_camf(camf: &mut x3f_camf_t, control: Control<'_>) -> Result<()> {
    if camf.decoded_data_size == 0 {
        return Err(Error::InvalidData("empty CAMF"));
    }
    let data = unsafe {
        std::slice::from_raw_parts(
            camf.decoded_data.cast::<u8>(),
            camf.decoded_data_size as usize,
        )
    };
    // Count/validate entry ranges before allocating; trailing zero padding is normal.
    let mut offset = 0usize;
    let mut count = 0usize;
    while offset < data.len() {
        control.check()?;
        if data[offset..].iter().all(|&v| v == 0) {
            break;
        }
        let id = u32_at(data, offset)?;
        if !matches!(id, X3F_CMBP | X3F_CMBT | X3F_CMBM) {
            return Err(Error::InvalidData("unknown CAMF entry signature"));
        }
        let size = u32_at(data, offset + 8)? as usize;
        if size < 20 {
            return Err(Error::InvalidData("invalid CAMF entry length"));
        }
        bytes_at(data, offset, size)?;
        offset += size;
        count += 1;
    }
    camf.entry_table.element = unsafe { alloc::<camf_entry_t>(count) }?;
    camf.entry_table.size = count as u32;
    offset = 0;
    for i in 0..count {
        control.check()?;
        let entry = unsafe { &mut *camf.entry_table.element.add(i) };
        entry.id = u32_at(data, offset)?;
        entry.version = u32_at(data, offset + 4)?;
        entry.entry_size = u32_at(data, offset + 8)?;
        entry.name_offset = u32_at(data, offset + 12)?;
        entry.value_offset = u32_at(data, offset + 16)?;
        let bytes = bytes_at(data, offset, entry.entry_size as usize)?;
        let name = entry.name_offset as usize;
        let val = entry.value_offset as usize;
        if name < 20 || val < name || val > bytes.len() {
            return Err(Error::InvalidData("invalid CAMF name/value offsets"));
        }
        cstr_at(&bytes[..val], name)?;
        let base = bytes.as_ptr() as *mut u8;
        entry.entry = base.cast();
        entry.name_address = unsafe { base.add(name).cast() };
        entry.value_address = unsafe { base.add(val).cast() };
        entry.name_size = (val - name) as u32;
        entry.value_size = (bytes.len() - val) as u32;
        unsafe { setup_camf_entry(entry, bytes, control) }?;
        offset += bytes.len();
    }
    Ok(())
}

/// Load a section using explicit cancellation and recoverable Rust errors.
/// `de` must belong to `x3f`; failed loads release their partial allocations.
pub unsafe fn load_data(
    x3f: *mut x3f_t,
    de: *mut x3f_directory_entry_t,
    control: Control<'_>,
) -> Result<()> {
    control.check()?;
    if x3f.is_null() || de.is_null() {
        return Err(Error::InvalidData("missing section"));
    }
    let result = (|| -> Result<()> {
        unsafe {
            match (*de).header.identifier {
                X3F_SECP => {
                    let mut input = section_input(x3f, de, X3F_PROPERTY_LIST_HEADER_SIZE, control)?;
                    let pl = &mut (*de).header.data_subsection.property_list;
                    if !pl.data.is_null() {
                        return Ok(());
                    }
                    load_properties(pl, &mut input, control)
                }
                X3F_SECI => {
                    let mut input = section_input(x3f, de, X3F_IMAGE_HEADER_SIZE, control)?;
                    let id = &mut (*de).header.data_subsection.image_data;
                    if !id.data.is_null() {
                        if !id.tru.is_null()
                            || !id.huffman.is_null()
                            || matches!(
                                id.type_format,
                                X3F_IMAGE_THUMB_JPEG | X3F_IMAGE_THUMB_PLAIN
                            )
                        {
                            return Ok(());
                        }
                        // A prior load_image_block owns encoded bytes, but has
                        // not decoded pixels. Reload after releasing that copy.
                        crate::io::cleanup_entry(de);
                    }
                    let id = &mut (*de).header.data_subsection.image_data;
                    load_image(id, &mut input, control)
                }
                X3F_SECC => {
                    let mut input = section_input(x3f, de, X3F_CAMF_HEADER_SIZE, control)?;
                    let camf = &mut (*de).header.data_subsection.camf;
                    if !camf.data.is_null() {
                        return Ok(());
                    }
                    camf.data_size = input.remaining() as u32;
                    if camf.data_size == 0 {
                        return Err(Error::InvalidData("empty CAMF section"));
                    }
                    camf.data = input.read_alloc(input.remaining())?.cast();
                    decode_camf(camf, control)?;
                    setup_camf(camf, control)
                }
                _ => Err(Error::InvalidData("unsupported section type")),
            }
        }
    })();
    if result.is_err() {
        unsafe { crate::io::cleanup_entry(de) };
    }
    result
}

pub unsafe fn load_image_block(
    x3f: *mut x3f_t,
    de: *mut x3f_directory_entry_t,
    control: Control<'_>,
) -> Result<()> {
    let mut input = unsafe { section_input(x3f, de, X3F_IMAGE_HEADER_SIZE, control) }?;
    if unsafe { (*de).header.identifier } != X3F_SECI {
        return Err(Error::InvalidData("not an image section"));
    }
    let id = unsafe { &mut (*de).header.data_subsection.image_data };
    if !id.data.is_null() {
        return Ok(());
    }
    id.data_size = input.remaining() as u32;
    id.data = unsafe { input.read_alloc(input.remaining()) }?.cast();
    Ok(())
}

#[no_mangle]
pub unsafe extern "C" fn x3f_load_data(
    x3f: *mut x3f_t,
    de: *mut x3f_directory_entry_t,
) -> x3f_return_t {
    if x3f.is_null() || de.is_null() {
        return x3f_return_e_X3F_ARGUMENT_ERROR;
    }
    if !matches!(
        unsafe { (*de).header.identifier },
        X3F_SECP | X3F_SECI | X3F_SECC
    ) {
        return x3f_return_e_X3F_INTERNAL_ERROR;
    }
    match unsafe { load_data(x3f, de, Control::none()) } {
        Ok(()) => x3f_return_e_X3F_OK,
        Err(_) => x3f_return_e_X3F_INFILE_ERROR,
    }
}

#[no_mangle]
pub unsafe extern "C" fn x3f_load_image_block(
    x3f: *mut x3f_t,
    de: *mut x3f_directory_entry_t,
) -> x3f_return_t {
    if x3f.is_null() || de.is_null() {
        return x3f_return_e_X3F_ARGUMENT_ERROR;
    }
    if unsafe { (*de).header.identifier } != X3F_SECI {
        return x3f_return_e_X3F_INTERNAL_ERROR;
    }
    match unsafe { load_image_block(x3f, de, Control::none()) } {
        Ok(()) => x3f_return_e_X3F_OK,
        Err(_) => x3f_return_e_X3F_INFILE_ERROR,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn c_wrappers_preserve_legacy_argument_and_section_statuses() {
        unsafe {
            let mut reader: x3f_t = std::mem::zeroed();
            let mut section: x3f_directory_entry_t = std::mem::zeroed();
            for load in [x3f_load_data, x3f_load_image_block] {
                assert_eq!(
                    load(ptr::null_mut(), &mut section),
                    x3f_return_e_X3F_ARGUMENT_ERROR
                );
                assert_eq!(
                    load(&mut reader, ptr::null_mut()),
                    x3f_return_e_X3F_ARGUMENT_ERROR
                );
                assert_eq!(
                    load(&mut reader, &mut section),
                    x3f_return_e_X3F_INTERNAL_ERROR
                );
            }
            section.header.identifier = X3F_SECP;
            assert_eq!(
                x3f_load_image_block(&mut reader, &mut section),
                x3f_return_e_X3F_INTERNAL_ERROR
            );
        }
    }

    fn parse_camf(bytes: &[u8]) -> Result<()> {
        unsafe {
            let mut entry: x3f_directory_entry_t = std::mem::zeroed();
            entry.header.identifier = X3F_SECC;
            let camf = &mut entry.header.data_subsection.camf;
            camf.decoded_data = copy_alloc(bytes)?.cast();
            camf.decoded_data_size = bytes.len() as u32;
            let result = setup_camf(camf, Control::none());
            crate::io::cleanup_entry(&mut entry);
            result
        }
    }

    #[test]
    fn malformed_camf_entries_are_recoverable() {
        let mut valid = Vec::new();
        for value in [X3F_CMBT, 0x10000, 34, 20, 25] {
            valid.extend_from_slice(&value.to_le_bytes());
        }
        valid.extend_from_slice(b"name\0");
        valid.extend_from_slice(&5u32.to_le_bytes());
        valid.extend_from_slice(b"text\0");
        parse_camf(&valid).unwrap();
        for (offset, value) in [
            (8, 0),
            (8, u32::MAX),
            (12, 35),
            (16, 10),
            (25, u32::MAX),
            (0, X3F_CMBM),
        ] {
            let mut corrupt = valid.clone();
            corrupt[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
            assert!(parse_camf(&corrupt).is_err());
        }
        for length in 1..valid.len() {
            assert!(parse_camf(&valid[..length]).is_err());
        }
        valid[24] = b'x';
        assert!(parse_camf(&valid).is_err());
    }

    #[test]
    fn overlapping_and_oversized_huffman_tables_are_rejected() {
        for codes in [
            vec![(9, 0, 0), (1, 0, 1)],
            vec![(1, 0, 0), (1, 0, 1)],
            vec![(32, 0, 0)],
            vec![],
        ] {
            let mut tree: x3f_hufftree_t = unsafe { std::mem::zeroed() };
            assert!(unsafe { build_tree(&mut tree, &codes) }.is_err());
            unsafe { crate::sysabi::free(tree.nodes.cast()) };
        }
    }
}
