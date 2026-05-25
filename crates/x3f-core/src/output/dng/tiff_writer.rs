//! Minimal little-endian TIFF writer used by the DNG output module.
//!
//! TIFF/DNG is well-suited to a one-shot writer: an IFD is a sorted list of
//! 12-byte entries, each carrying either a 4-byte inline value or a 4-byte
//! offset into the file. We don't need random-access editing of an existing
//! file — only forward writing from a `Write + Seek` sink, with one
//! back-patch at the end to fill in the IFD0 offset in the header.
//!
//! Build pattern:
//!
//! ```ignore
//! let mut tiff = TiffWriter::new(out)?;          // writes 8-byte header
//! let mut ifd0 = DirectoryWriter::new();
//! ifd0.add(tag::IMAGE_WIDTH, Value::Long(vec![w]));
//! // … add tags / write strips ahead of building the IFD …
//! let raw_strip_offset = tiff.write_data(&strip_bytes)?;
//! ifd0.add(tag::STRIP_OFFSETS, Value::Long(vec![raw_strip_offset]));
//! let ifd0_offset = ifd0.build(&mut tiff)?;
//! tiff.finalize(ifd0_offset)?;
//! ```
//!
//! SubIFDs are not modelled with a dedicated abstraction: the caller just
//! builds the child IFD first, then writes the resulting offset into the
//! parent's `SubIFDs` tag (DNG tag 330) before building the parent.

use std::collections::BTreeMap;
use std::ffi::CString;
use std::io::{self, Seek, SeekFrom, Write};

pub const TIFF_TYPE_BYTE: u16 = 1;
pub const TIFF_TYPE_ASCII: u16 = 2;
pub const TIFF_TYPE_SHORT: u16 = 3;
pub const TIFF_TYPE_LONG: u16 = 4;
pub const TIFF_TYPE_RATIONAL: u16 = 5;
pub const TIFF_TYPE_SBYTE: u16 = 6;
pub const TIFF_TYPE_UNDEFINED: u16 = 7;
pub const TIFF_TYPE_SSHORT: u16 = 8;
pub const TIFF_TYPE_SLONG: u16 = 9;
pub const TIFF_TYPE_SRATIONAL: u16 = 10;
pub const TIFF_TYPE_FLOAT: u16 = 11;
pub const TIFF_TYPE_DOUBLE: u16 = 12;

/// One IFD entry value. Variants map 1:1 to the TIFF type codes.
// Signed/double variants round out the TIFF type set but aren't emitted by
// the DNG writer today; kept for completeness of this minimal writer.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub enum Value {
    Byte(Vec<u8>),
    /// Caller supplies a NUL-terminated `CString`; we write its full bytes
    /// (including the NUL) and the count is `bytes.len()`.
    Ascii(CString),
    Short(Vec<u16>),
    Long(Vec<u32>),
    Rational(Vec<(u32, u32)>),
    SByte(Vec<i8>),
    Undefined(Vec<u8>),
    SShort(Vec<i16>),
    SLong(Vec<i32>),
    SRational(Vec<(i32, i32)>),
    Float(Vec<f32>),
    Double(Vec<f64>),
}

impl Value {
    pub fn tiff_type(&self) -> u16 {
        match self {
            Value::Byte(_) => TIFF_TYPE_BYTE,
            Value::Ascii(_) => TIFF_TYPE_ASCII,
            Value::Short(_) => TIFF_TYPE_SHORT,
            Value::Long(_) => TIFF_TYPE_LONG,
            Value::Rational(_) => TIFF_TYPE_RATIONAL,
            Value::SByte(_) => TIFF_TYPE_SBYTE,
            Value::Undefined(_) => TIFF_TYPE_UNDEFINED,
            Value::SShort(_) => TIFF_TYPE_SSHORT,
            Value::SLong(_) => TIFF_TYPE_SLONG,
            Value::SRational(_) => TIFF_TYPE_SRATIONAL,
            Value::Float(_) => TIFF_TYPE_FLOAT,
            Value::Double(_) => TIFF_TYPE_DOUBLE,
        }
    }

    pub fn count(&self) -> u32 {
        match self {
            Value::Byte(v) => v.len() as u32,
            Value::Ascii(s) => s.as_bytes_with_nul().len() as u32,
            Value::Short(v) => v.len() as u32,
            Value::Long(v) => v.len() as u32,
            Value::Rational(v) => v.len() as u32,
            Value::SByte(v) => v.len() as u32,
            Value::Undefined(v) => v.len() as u32,
            Value::SShort(v) => v.len() as u32,
            Value::SLong(v) => v.len() as u32,
            Value::SRational(v) => v.len() as u32,
            Value::Float(v) => v.len() as u32,
            Value::Double(v) => v.len() as u32,
        }
    }

    fn element_size(&self) -> u32 {
        match self {
            Value::Byte(_) | Value::Ascii(_) | Value::SByte(_) | Value::Undefined(_) => 1,
            Value::Short(_) | Value::SShort(_) => 2,
            Value::Long(_) | Value::SLong(_) | Value::Float(_) => 4,
            Value::Rational(_) | Value::SRational(_) | Value::Double(_) => 8,
        }
    }

    pub fn byte_size(&self) -> u32 {
        self.count() * self.element_size()
    }

    fn write_payload<W: Write>(&self, w: &mut W) -> io::Result<()> {
        match self {
            Value::Byte(v) => w.write_all(v),
            Value::Ascii(s) => w.write_all(s.as_bytes_with_nul()),
            Value::Short(v) => {
                for &x in v {
                    w.write_all(&x.to_le_bytes())?;
                }
                Ok(())
            }
            Value::Long(v) => {
                for &x in v {
                    w.write_all(&x.to_le_bytes())?;
                }
                Ok(())
            }
            Value::Rational(v) => {
                for &(n, d) in v {
                    w.write_all(&n.to_le_bytes())?;
                    w.write_all(&d.to_le_bytes())?;
                }
                Ok(())
            }
            Value::SByte(v) => {
                for &x in v {
                    w.write_all(&[x as u8])?;
                }
                Ok(())
            }
            Value::Undefined(v) => w.write_all(v),
            Value::SShort(v) => {
                for &x in v {
                    w.write_all(&x.to_le_bytes())?;
                }
                Ok(())
            }
            Value::SLong(v) => {
                for &x in v {
                    w.write_all(&x.to_le_bytes())?;
                }
                Ok(())
            }
            Value::SRational(v) => {
                for &(n, d) in v {
                    w.write_all(&n.to_le_bytes())?;
                    w.write_all(&d.to_le_bytes())?;
                }
                Ok(())
            }
            Value::Float(v) => {
                for &x in v {
                    w.write_all(&x.to_le_bytes())?;
                }
                Ok(())
            }
            Value::Double(v) => {
                for &x in v {
                    w.write_all(&x.to_le_bytes())?;
                }
                Ok(())
            }
        }
    }

    /// Render the value into the 4-byte inline slot of an IFD entry.
    /// Caller must verify `byte_size() <= 4` first.
    fn write_inline(&self, dst: &mut [u8; 4]) -> io::Result<()> {
        let mut cursor = io::Cursor::new(&mut dst[..]);
        self.write_payload(&mut cursor)?;
        Ok(())
    }
}

/// Convenience builders so callsites read close to the legacy C
/// `TIFFSetField` lines they replace.
impl From<u16> for Value {
    fn from(v: u16) -> Self {
        Value::Short(vec![v])
    }
}

impl From<u32> for Value {
    fn from(v: u32) -> Self {
        Value::Long(vec![v])
    }
}

impl From<&[u32]> for Value {
    fn from(v: &[u32]) -> Self {
        Value::Long(v.to_vec())
    }
}

impl<const N: usize> From<[u32; N]> for Value {
    fn from(v: [u32; N]) -> Self {
        Value::Long(v.to_vec())
    }
}

impl From<&[u16]> for Value {
    fn from(v: &[u16]) -> Self {
        Value::Short(v.to_vec())
    }
}

impl From<&[f32]> for Value {
    fn from(v: &[f32]) -> Self {
        Value::Float(v.to_vec())
    }
}

impl<const N: usize> From<[f32; N]> for Value {
    fn from(v: [f32; N]) -> Self {
        Value::Float(v.to_vec())
    }
}

pub struct TiffWriter<W: Write + Seek> {
    out: W,
    /// File position of the 4-byte slot in the header that holds the IFD0
    /// offset. We back-patch it in [`finalize`] once IFD0 is written.
    ifd0_slot: u64,
}

impl<W: Write + Seek> TiffWriter<W> {
    /// Write the 8-byte header (`II*\0` magic + IFD0-offset placeholder).
    pub fn new(mut out: W) -> io::Result<Self> {
        out.write_all(b"II")?;
        out.write_all(&42u16.to_le_bytes())?;
        let ifd0_slot = out.stream_position()?;
        out.write_all(&0u32.to_le_bytes())?;
        Ok(Self { out, ifd0_slot })
    }

    pub fn position(&mut self) -> io::Result<u64> {
        self.out.stream_position()
    }

    /// Pad to the next 2-byte word boundary so subsequent values (and IFDs)
    /// land on TIFF-spec-aligned offsets.
    pub fn align_word(&mut self) -> io::Result<()> {
        let pos = self.position()?;
        if pos % 2 != 0 {
            self.out.write_all(&[0])?;
        }
        Ok(())
    }

    /// Append a raw byte blob to the file. Returns the (post-alignment)
    /// offset at which the blob starts.
    pub fn write_data(&mut self, data: &[u8]) -> io::Result<u32> {
        self.align_word()?;
        let offset = u32_offset(self.position()?)?;
        self.out.write_all(data)?;
        Ok(offset)
    }

    /// Same but for 16-bit samples (e.g. uncompressed RGB16 strips). Each
    /// sample is written little-endian.
    #[allow(dead_code)] // writer primitive; no current caller
    pub fn write_data_u16_le(&mut self, data: &[u16]) -> io::Result<u32> {
        self.align_word()?;
        let offset = u32_offset(self.position()?)?;
        for &x in data {
            self.out.write_all(&x.to_le_bytes())?;
        }
        Ok(offset)
    }

    /// Patch the header IFD0-offset slot. Must be called exactly once, after
    /// the root IFD has been built.
    pub fn finalize(mut self, ifd0_offset: u32) -> io::Result<W> {
        self.out.seek(SeekFrom::Start(self.ifd0_slot))?;
        self.out.write_all(&ifd0_offset.to_le_bytes())?;
        // Restore append position so callers who keep using the writer
        // after this (we don't, but be polite) don't get surprised.
        self.out.seek(SeekFrom::End(0))?;
        Ok(self.out)
    }

    /// Mutable access to the underlying writer for the rare callers that
    /// need to seek (e.g. an external streaming compressor that wraps it).
    pub fn writer_mut(&mut self) -> &mut W {
        &mut self.out
    }
}

/// One IFD's worth of tags. Tag IDs are kept sorted via `BTreeMap` because
/// the TIFF spec requires it and many DNG readers (Lightroom included)
/// reject non-monotonic IFDs without a clear error message.
#[derive(Default)]
pub struct DirectoryWriter {
    entries: BTreeMap<u16, Value>,
    next_ifd: u32,
}

impl DirectoryWriter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add(&mut self, tag: u16, value: impl Into<Value>) {
        self.entries.insert(tag, value.into());
    }

    /// Set the offset to the next chained IFD. Default is 0 (no chain).
    /// Note: this is the IFD0 → IFD1 *chain* link, not the SubIFD tag.
    #[allow(dead_code)] // IFD-chaining helper; no current caller
    pub fn set_next_ifd(&mut self, offset: u32) {
        self.next_ifd = offset;
    }

    #[allow(dead_code)] // no current caller
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Two-pass write: first stash any > 4-byte values in the file body
    /// (capturing their offsets), then emit the 12-byte-per-entry IFD body
    /// that references them.
    ///
    /// Returns the offset at which the IFD body starts — the value to plug
    /// into the parent (header IFD0 slot, SubIFDs tag, or chain link).
    pub fn build<W: Write + Seek>(self, tiff: &mut TiffWriter<W>) -> io::Result<u32> {
        if self.entries.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "TIFF IFD must contain at least one entry",
            ));
        }

        let mut external_offsets: BTreeMap<u16, u32> = BTreeMap::new();
        for (tag, value) in &self.entries {
            if value.byte_size() > 4 {
                tiff.align_word()?;
                let off = u32_offset(tiff.position()?)?;
                value.write_payload(tiff.writer_mut())?;
                external_offsets.insert(*tag, off);
            }
        }

        tiff.align_word()?;
        let ifd_offset = u32_offset(tiff.position()?)?;
        let count = u16::try_from(self.entries.len()).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "IFD has more than 65535 entries",
            )
        })?;
        tiff.writer_mut().write_all(&count.to_le_bytes())?;

        for (tag, value) in &self.entries {
            tiff.writer_mut().write_all(&tag.to_le_bytes())?;
            tiff.writer_mut()
                .write_all(&value.tiff_type().to_le_bytes())?;
            tiff.writer_mut().write_all(&value.count().to_le_bytes())?;
            if let Some(off) = external_offsets.get(tag) {
                tiff.writer_mut().write_all(&off.to_le_bytes())?;
            } else {
                let mut inline = [0u8; 4];
                value.write_inline(&mut inline)?;
                tiff.writer_mut().write_all(&inline)?;
            }
        }

        tiff.writer_mut().write_all(&self.next_ifd.to_le_bytes())?;
        Ok(ifd_offset)
    }
}

fn u32_offset(pos: u64) -> io::Result<u32> {
    u32::try_from(pos).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "TIFF offset exceeds 4 GiB (need BigTIFF)",
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn build_one(tag: u16, value: Value) -> Vec<u8> {
        let mut buf = Cursor::new(Vec::new());
        let mut tiff = TiffWriter::new(&mut buf).unwrap();
        let mut ifd = DirectoryWriter::new();
        ifd.add(tag, value);
        let off = ifd.build(&mut tiff).unwrap();
        tiff.finalize(off).unwrap();
        buf.into_inner()
    }

    fn read_u32_le(b: &[u8], off: usize) -> u32 {
        u32::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]])
    }
    fn read_u16_le(b: &[u8], off: usize) -> u16 {
        u16::from_le_bytes([b[off], b[off + 1]])
    }

    #[test]
    fn header_is_le_with_magic_42() {
        let bytes = build_one(0x0100, Value::Long(vec![42]));
        assert_eq!(&bytes[..2], b"II");
        assert_eq!(read_u16_le(&bytes, 2), 42);
    }

    #[test]
    fn ifd0_offset_in_header_is_back_patched() {
        let bytes = build_one(0x0100, Value::Long(vec![123]));
        let ifd0 = read_u32_le(&bytes, 4) as usize;
        assert!(ifd0 >= 8 && ifd0 < bytes.len());
        let entry_count = read_u16_le(&bytes, ifd0);
        assert_eq!(entry_count, 1);
    }

    #[test]
    fn small_value_inlined_in_entry() {
        let bytes = build_one(0x0100, Value::Long(vec![0xCAFEBABE]));
        let ifd0 = read_u32_le(&bytes, 4) as usize;
        // Entry layout: tag(2), type(2), count(4), value(4)
        assert_eq!(read_u16_le(&bytes, ifd0 + 2), 0x0100); // tag
        assert_eq!(read_u16_le(&bytes, ifd0 + 4), TIFF_TYPE_LONG);
        assert_eq!(read_u32_le(&bytes, ifd0 + 6), 1);
        assert_eq!(read_u32_le(&bytes, ifd0 + 10), 0xCAFEBABE);
    }

    #[test]
    fn large_value_externalised_with_correct_offset() {
        // 9 floats = 36 bytes, won't fit inline.
        let v: Vec<f32> = (0..9).map(|i| i as f32 * 1.5).collect();
        let bytes = build_one(0x0100, Value::Float(v.clone()));
        let ifd0 = read_u32_le(&bytes, 4) as usize;
        assert_eq!(read_u16_le(&bytes, ifd0 + 4), TIFF_TYPE_FLOAT);
        assert_eq!(read_u32_le(&bytes, ifd0 + 6), 9);
        let payload_off = read_u32_le(&bytes, ifd0 + 10) as usize;
        // Payload precedes the IFD body in the file (we write externals
        // first), so the offset must be < ifd0.
        assert!(payload_off < ifd0);
        for (i, &expected) in v.iter().enumerate() {
            let got = f32::from_le_bytes([
                bytes[payload_off + i * 4],
                bytes[payload_off + i * 4 + 1],
                bytes[payload_off + i * 4 + 2],
                bytes[payload_off + i * 4 + 3],
            ]);
            assert_eq!(got, expected);
        }
    }

    #[test]
    fn ascii_value_includes_nul_in_count() {
        let bytes = build_one(0x010E, Value::Ascii(CString::new("Hi").unwrap()));
        let ifd0 = read_u32_le(&bytes, 4) as usize;
        assert_eq!(read_u16_le(&bytes, ifd0 + 4), TIFF_TYPE_ASCII);
        assert_eq!(read_u32_le(&bytes, ifd0 + 6), 3); // "Hi\0"
                                                      // 3 bytes fits inline.
        assert_eq!(&bytes[ifd0 + 10..ifd0 + 13], b"Hi\0");
    }

    #[test]
    fn entries_sorted_by_tag_id() {
        let mut buf = Cursor::new(Vec::new());
        let mut tiff = TiffWriter::new(&mut buf).unwrap();
        let mut ifd = DirectoryWriter::new();
        // Add out of order; build() must sort.
        ifd.add(0x0200, Value::Long(vec![2]));
        ifd.add(0x0100, Value::Long(vec![1]));
        ifd.add(0x0150, Value::Long(vec![3]));
        let off = ifd.build(&mut tiff).unwrap();
        tiff.finalize(off).unwrap();
        let bytes = buf.into_inner();
        let ifd0 = read_u32_le(&bytes, 4) as usize;
        assert_eq!(read_u16_le(&bytes, ifd0 + 2), 0x0100);
        assert_eq!(read_u16_le(&bytes, ifd0 + 2 + 12), 0x0150);
        assert_eq!(read_u16_le(&bytes, ifd0 + 2 + 24), 0x0200);
    }

    #[test]
    fn next_ifd_link_written_at_tail() {
        let mut buf = Cursor::new(Vec::new());
        let mut tiff = TiffWriter::new(&mut buf).unwrap();
        let mut ifd = DirectoryWriter::new();
        ifd.add(0x0100, Value::Long(vec![1]));
        ifd.set_next_ifd(0xDEAD_BEEF);
        let off = ifd.build(&mut tiff).unwrap();
        tiff.finalize(off).unwrap();
        let bytes = buf.into_inner();
        // Tail = ifd_off + 2 (count) + 12 (one entry).
        let tail = off as usize + 2 + 12;
        assert_eq!(read_u32_le(&bytes, tail), 0xDEAD_BEEF);
    }

    #[test]
    fn empty_directory_errors() {
        let mut buf = Cursor::new(Vec::new());
        let mut tiff = TiffWriter::new(&mut buf).unwrap();
        let ifd = DirectoryWriter::new();
        let err = ifd.build(&mut tiff).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
    }

    #[test]
    fn write_data_pads_to_word_boundary() {
        let mut buf = Cursor::new(Vec::new());
        let mut tiff = TiffWriter::new(&mut buf).unwrap();
        // Header is 8 bytes (already even). Write one byte to make position
        // odd, then write_data should pad.
        tiff.writer_mut().write_all(&[0xAA]).unwrap();
        assert_eq!(tiff.position().unwrap() % 2, 1);
        let off = tiff.write_data(&[0xBB, 0xCC]).unwrap();
        assert_eq!(off % 2, 0);
        // We need to actually finalise so the header gets a valid IFD0.
        let mut ifd = DirectoryWriter::new();
        ifd.add(0x0100, Value::Long(vec![1]));
        let ifd_off = ifd.build(&mut tiff).unwrap();
        tiff.finalize(ifd_off).unwrap();
    }
}
