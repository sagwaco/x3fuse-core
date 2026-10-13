//! Read a DCP look table and tone curve without importing camera calibration.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;

use crate::output::dng::tags;
use crate::output::dng::tiff_writer::{
    DirectoryWriter, Value, TIFF_TYPE_ASCII, TIFF_TYPE_FLOAT, TIFF_TYPE_LONG,
};
use crate::Error;

const MAX_BYTES: usize = 16 * 1024 * 1024;
const PROFILE_MAGIC: u16 = 0x4352;

/// A DCP's look table and tone curve, without camera calibration.
#[derive(Debug)]
pub struct DcpLook {
    name: String,
    dims: [u32; 3],
    encoding: u32,
    table: Vec<f32>,
    tone: Vec<f32>,
}

impl DcpLook {
    /// Read a DCP file up to 16 MiB in size.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, Error> {
        let path = path.as_ref();
        let io_error = |source| Error::Io {
            path: path.display().to_string(),
            source,
        };
        if !std::fs::metadata(path).map_err(io_error)?.is_file() {
            return Err(invalid("expected a regular profile file"));
        }
        let mut bytes = Vec::new();
        File::open(path)
            .map_err(io_error)?
            .take((MAX_BYTES + 1) as u64)
            .read_to_end(&mut bytes)
            .map_err(io_error)?;
        Self::from_bytes(&bytes)
    }

    /// Read a little-endian or big-endian extended-profile DCP container.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, Error> {
        let directory = Directory::parse(bytes)?;
        let name = directory.text(tags::PROFILE_NAME)?;
        let policy = directory.optional_long(tags::PROFILE_EMBED_POLICY, 0)?;
        if !matches!(policy, 0 | 1 | 3) {
            return Err(invalid("profile forbids embedding or has unknown policy"));
        }

        let dimensions = directory.field(tags::PROFILE_LOOK_TABLE_DIMS, TIFF_TYPE_LONG, Some(3))?;
        let dims = [
            directory.uint(dimensions.bytes),
            directory.uint(&dimensions.bytes[4..]),
            directory.uint(&dimensions.bytes[8..]),
        ];
        if dims[0] < 1 || dims[1] < 2 || dims[2] < 1 {
            return Err(invalid("invalid look dimensions"));
        }
        let count = dims
            .iter()
            .try_fold(3usize, |count, &dimension| {
                count.checked_mul(dimension as usize)
            })
            .filter(|&count| count <= MAX_BYTES / 4)
            .ok_or_else(|| invalid("look dimensions exceed bounds"))?;
        let encoding = directory.optional_long(tags::PROFILE_LOOK_TABLE_ENCODING, 0)?;
        if encoding > 1 {
            return Err(invalid("unsupported look encoding"));
        }
        let table = directory.floats(tags::PROFILE_LOOK_TABLE_DATA, Some(count))?;
        for (index, entry) in table.as_chunks::<3>().0.iter().enumerate() {
            if !entry.iter().all(|value| value.is_finite()) || entry[1] < 0.0 || entry[2] < 0.0 {
                return Err(invalid(
                    "look values must be finite with nonnegative scales",
                ));
            }
            if index % dims[1] as usize == 0 && entry[2] != 1.0 {
                return Err(invalid("neutral ValueScale must equal one"));
            }
        }
        let tone = directory.floats(tags::PROFILE_TONE_CURVE, None)?;
        if tone.len() < 4
            || tone.len() % 2 != 0
            || tone[..2] != [0.0, 0.0]
            || tone[tone.len() - 2..] != [1.0, 1.0]
            || !tone.iter().all(|value| value.is_finite())
            || tone
                .windows(4)
                .step_by(2)
                .any(|pair| pair[2] <= pair[0] || pair[3] <= pair[1])
        {
            return Err(invalid(
                "tone must be strictly increasing from (0,0) to (1,1)",
            ));
        }
        Ok(Self {
            name,
            dims,
            encoding,
            table,
            tone,
        })
    }

    /// Source profile name. Embedding leaves the DNG's profile name unchanged.
    pub fn name(&self) -> &str {
        &self.name
    }

    pub(crate) fn embed(self, ifd: &mut DirectoryWriter) {
        ifd.add(
            tags::PROFILE_LOOK_TABLE_DIMS,
            Value::Long(self.dims.to_vec()),
        );
        ifd.add(tags::PROFILE_LOOK_TABLE_DATA, Value::Float(self.table));
        ifd.add(
            tags::PROFILE_LOOK_TABLE_ENCODING,
            Value::Long(vec![self.encoding]),
        );
        ifd.add(tags::PROFILE_TONE_CURVE, Value::Float(self.tone));
        ifd.add(
            tags::DNG_BACKWARD_VERSION,
            Value::Byte(tags::DNG_VERSION_1_4_0_0.to_vec()),
        );
    }
}

struct Tag<'a> {
    kind: u16,
    count: usize,
    bytes: &'a [u8],
}

struct Directory<'a> {
    big_endian: bool,
    tags: BTreeMap<u16, Tag<'a>>,
}

impl<'a> Directory<'a> {
    fn parse(bytes: &'a [u8]) -> Result<Self, Error> {
        if !(8..=MAX_BYTES).contains(&bytes.len()) {
            return Err(invalid("file size outside supported bounds"));
        }
        let big_endian = match &bytes[..2] {
            b"MM" => true,
            b"II" => false,
            _ => return Err(invalid("invalid byte order")),
        };
        let short = |data: &[u8]| {
            let word = [data[0], data[1]];
            if big_endian {
                u16::from_be_bytes(word)
            } else {
                u16::from_le_bytes(word)
            }
        };
        if short(&bytes[2..]) != PROFILE_MAGIC {
            return Err(invalid("expected extended-profile DCP header"));
        }
        let mut directory = Self {
            big_endian,
            tags: BTreeMap::new(),
        };
        let offset = directory.uint(&bytes[4..]) as usize;
        if offset < 8 || offset > bytes.len() - 2 {
            return Err(invalid("invalid directory offset"));
        }
        let count = short(&bytes[offset..]) as usize;
        let end = offset + 2 + count * 12;
        if count == 0 || end > bytes.len() {
            return Err(invalid("invalid directory size"));
        }
        for entry in bytes[offset + 2..end].as_chunks::<12>().0 {
            let id = short(entry);
            let kind = short(&entry[2..]);
            let count = directory.uint(&entry[4..]) as usize;
            let size = match kind {
                1 | 2 | 6 | 7 => 1,
                3 | 8 => 2,
                4 | 9 | 11 | 13 => 4,
                5 | 10 | 12 => 8,
                _ => return Err(invalid("unknown TIFF field type")),
            };
            let length = count
                .checked_mul(size)
                .filter(|&length| length > 0 && length <= bytes.len())
                .ok_or_else(|| invalid("invalid tag size"))?;
            let data = if length <= 4 {
                &entry[8..8 + length]
            } else {
                let start = directory.uint(&entry[8..]) as usize;
                let stop = start
                    .checked_add(length)
                    .filter(|&stop| stop <= bytes.len())
                    .ok_or_else(|| invalid("tag outside file"))?;
                if start < 8 || (stop > offset && start < end) {
                    return Err(invalid("tag overlaps header or directory"));
                }
                &bytes[start..stop]
            };
            if directory
                .tags
                .insert(
                    id,
                    Tag {
                        kind,
                        count,
                        bytes: data,
                    },
                )
                .is_some()
            {
                return Err(invalid("duplicate tag"));
            }
        }
        Ok(directory)
    }

    fn uint(&self, bytes: &[u8]) -> u32 {
        let word = [bytes[0], bytes[1], bytes[2], bytes[3]];
        if self.big_endian {
            u32::from_be_bytes(word)
        } else {
            u32::from_le_bytes(word)
        }
    }

    fn field(&self, id: u16, kind: u16, count: Option<usize>) -> Result<&Tag<'a>, Error> {
        let tag = self
            .tags
            .get(&id)
            .ok_or_else(|| invalid(&format!("missing tag {id}")))?;
        if tag.kind != kind || count.is_some_and(|count| count != tag.count) {
            return Err(invalid(&format!("invalid tag {id}")));
        }
        Ok(tag)
    }

    fn optional_long(&self, id: u16, default: u32) -> Result<u32, Error> {
        if self.tags.contains_key(&id) {
            Ok(self.uint(self.field(id, TIFF_TYPE_LONG, Some(1))?.bytes))
        } else {
            Ok(default)
        }
    }

    fn text(&self, id: u16) -> Result<String, Error> {
        let bytes = self.field(id, TIFF_TYPE_ASCII, None)?.bytes;
        if bytes.len() < 2 || bytes.last() != Some(&0) || bytes[..bytes.len() - 1].contains(&0) {
            return Err(invalid("invalid profile string"));
        }
        std::str::from_utf8(&bytes[..bytes.len() - 1])
            .map(str::to_owned)
            .map_err(|_| invalid("profile text is not UTF-8"))
    }

    fn floats(&self, id: u16, count: Option<usize>) -> Result<Vec<f32>, Error> {
        let tag = self.field(id, TIFF_TYPE_FLOAT, count)?;
        Ok(tag
            .bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|bytes| f32::from_bits(self.uint(bytes)))
            .collect())
    }
}

fn invalid(message: &str) -> Error {
    Error::InvalidData(format!("DCP look: {message}"))
}

#[cfg(test)]
#[path = "../tests/dcp_fixture.rs"]
mod fixture;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::output::dng::tiff_writer::TiffWriter;
    use std::ffi::CString;
    use std::io::Cursor;

    fn profile() -> Vec<u8> {
        fixture::profile("Test camera", false)
    }

    fn entry(bytes: &[u8], tag: u16) -> usize {
        (10..10 + u16::from_le_bytes(bytes[8..10].try_into().unwrap()) as usize * 12)
            .step_by(12)
            .find(|&offset| {
                u16::from_le_bytes(bytes[offset..offset + 2].try_into().unwrap()) == tag
            })
            .unwrap()
    }

    fn set_word(bytes: &mut [u8], tag: u16, field_offset: usize, value: u32) {
        let offset = entry(bytes, tag) + field_offset;
        bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }

    fn set_float(bytes: &mut [u8], tag: u16, component: usize, value: f32) {
        let field = entry(bytes, tag) + 8;
        let offset = u32::from_le_bytes(bytes[field..field + 4].try_into().unwrap()) as usize
            + component * 4;
        bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }

    fn omit(bytes: &mut [u8], tag: u16) {
        let offset = entry(bytes, tag);
        bytes[offset..offset + 2].copy_from_slice(&65000u16.to_le_bytes());
    }

    #[test]
    fn both_byte_orders_preserve_nonidentity_values() {
        for big_endian in [false, true] {
            let look = DcpLook::from_bytes(&fixture::profile("Test camera", big_endian)).unwrap();
            assert_eq!(look.name(), "Test look");
            assert_eq!(look.dims, [2, 2, 2]);
            assert_eq!(look.encoding, 1);
            assert_eq!(&look.table[..6], &[0.0, 1.0, 1.0, 12.0, 0.8, 1.1]);
            assert_eq!(&look.table[18..], &[0.0, 1.0, 1.0, -4.0, 1.1, 0.8]);
            assert_eq!(look.tone, [0.0, 0.0, 0.4, 0.6, 1.0, 1.0]);
        }
    }

    #[test]
    fn source_camera_is_ignored_and_optional() {
        for camera in ["Camera A", "Camera B"] {
            let look = DcpLook::from_bytes(&fixture::profile(camera, false)).unwrap();
            assert_eq!(look.table, fixture::TABLE);
            assert_eq!(look.tone, fixture::TONE);
        }
        let mut bytes = profile();
        omit(&mut bytes, tags::UNIQUE_CAMERA_MODEL);
        let look = DcpLook::from_bytes(&bytes).unwrap();
        assert_eq!(look.table, fixture::TABLE);
        assert_eq!(look.tone, fixture::TONE);
    }

    #[test]
    fn missing_encoding_defaults_to_linear() {
        let mut bytes = profile();
        omit(&mut bytes, tags::PROFILE_LOOK_TABLE_ENCODING);
        assert_eq!(DcpLook::from_bytes(&bytes).unwrap().encoding, 0);
    }

    #[test]
    fn accepts_supported_embedding_policies_and_missing_policy() {
        for policy in [0, 1, 3] {
            let mut bytes = profile();
            set_word(&mut bytes, tags::PROFILE_EMBED_POLICY, 8, policy);
            DcpLook::from_bytes(&bytes).unwrap();
        }
        let mut bytes = profile();
        omit(&mut bytes, tags::PROFILE_EMBED_POLICY);
        DcpLook::from_bytes(&bytes).unwrap();
    }

    #[test]
    fn rejects_missing_required_fields() {
        for tag in [
            tags::PROFILE_NAME,
            tags::PROFILE_TONE_CURVE,
            tags::PROFILE_LOOK_TABLE_DIMS,
            tags::PROFILE_LOOK_TABLE_DATA,
        ] {
            let mut bytes = profile();
            omit(&mut bytes, tag);
            assert!(DcpLook::from_bytes(&bytes).is_err(), "missing tag {tag}");
        }
    }

    #[test]
    fn rejects_bad_headers_offsets_counts_types_and_duplicates() {
        for (offset, data) in [
            (0, vec![0, 0]),
            (2, vec![42, 0]),
            (4, u32::MAX.to_le_bytes().to_vec()),
            (8, vec![255, 255]),
            (12, vec![255, 255]),
            (14, vec![0; 4]),
        ] {
            let mut bytes = profile();
            bytes[offset..offset + data.len()].copy_from_slice(&data);
            assert!(DcpLook::from_bytes(&bytes).is_err());
        }
        let mut bytes = profile();
        let duplicate = entry(&bytes, tags::PROFILE_NAME);
        bytes[duplicate..duplicate + 2].copy_from_slice(&tags::UNIQUE_CAMERA_MODEL.to_le_bytes());
        assert!(DcpLook::from_bytes(&bytes).is_err());
        for offset in [0, 4, 8, 9, u32::MAX] {
            let mut bytes = profile();
            set_word(&mut bytes, tags::PROFILE_LOOK_TABLE_DATA, 8, offset);
            assert!(DcpLook::from_bytes(&bytes).is_err());
        }
        for count in [0, 23, 25, u32::MAX] {
            let mut bytes = profile();
            set_word(&mut bytes, tags::PROFILE_LOOK_TABLE_DATA, 4, count);
            assert!(DcpLook::from_bytes(&bytes).is_err());
        }
    }

    #[test]
    fn rejects_invalid_dimensions() {
        for dims in [[0, 2, 2], [2, 1, 2], [2, 2, 0], [u32::MAX; 3], [2, 2, 3]] {
            let mut bytes = profile();
            let field = entry(&bytes, tags::PROFILE_LOOK_TABLE_DIMS) + 8;
            let offset = u32::from_le_bytes(bytes[field..field + 4].try_into().unwrap()) as usize;
            for (index, dimension) in dims.iter().enumerate() {
                bytes[offset + index * 4..offset + index * 4 + 4]
                    .copy_from_slice(&dimension.to_le_bytes());
            }
            assert!(DcpLook::from_bytes(&bytes).is_err(), "{dims:?}");
        }
    }

    #[test]
    fn rejects_embedding_policy_encoding_and_wrong_scalar_types() {
        for (tag, value) in [
            (tags::PROFILE_EMBED_POLICY, 2),
            (tags::PROFILE_EMBED_POLICY, 99),
            (tags::PROFILE_LOOK_TABLE_ENCODING, 2),
        ] {
            let mut bytes = profile();
            set_word(&mut bytes, tag, 8, value);
            assert!(DcpLook::from_bytes(&bytes).is_err());
        }
        for tag in [
            tags::PROFILE_EMBED_POLICY,
            tags::PROFILE_LOOK_TABLE_ENCODING,
            tags::PROFILE_LOOK_TABLE_DIMS,
        ] {
            let mut bytes = profile();
            let offset = entry(&bytes, tag) + 2;
            bytes[offset..offset + 2].copy_from_slice(&TIFF_TYPE_FLOAT.to_le_bytes());
            assert!(DcpLook::from_bytes(&bytes).is_err());
        }
    }

    #[test]
    fn rejects_invalid_table_and_curve() {
        for (tag, component, value) in [
            (tags::PROFILE_LOOK_TABLE_DATA, 0, f32::NAN),
            (tags::PROFILE_LOOK_TABLE_DATA, 3, f32::INFINITY),
            (tags::PROFILE_LOOK_TABLE_DATA, 1, -1.0),
            (tags::PROFILE_LOOK_TABLE_DATA, 5, -1.0),
            (tags::PROFILE_LOOK_TABLE_DATA, 2, 0.5),
            (tags::PROFILE_TONE_CURVE, 0, 0.1),
            (tags::PROFILE_TONE_CURVE, 2, 0.0),
            (tags::PROFILE_TONE_CURVE, 3, 0.0),
            (tags::PROFILE_TONE_CURVE, 3, f32::INFINITY),
            (tags::PROFILE_TONE_CURVE, 5, 0.9),
        ] {
            let mut bytes = profile();
            set_float(&mut bytes, tag, component, value);
            assert!(
                DcpLook::from_bytes(&bytes).is_err(),
                "{tag}, {component}, {value}"
            );
        }
        for count in [1, 3, 5] {
            let mut bytes = profile();
            set_word(&mut bytes, tags::PROFILE_TONE_CURVE, 4, count);
            assert!(DcpLook::from_bytes(&bytes).is_err());
        }
    }

    #[test]
    fn rejects_invalid_names_and_accepts_unicode() {
        let mut bytes = profile();
        let field = entry(&bytes, tags::PROFILE_NAME) + 8;
        let offset = u32::from_le_bytes(bytes[field..field + 4].try_into().unwrap()) as usize;
        bytes[offset..offset + 10].copy_from_slice("カラー\0".as_bytes());
        assert_eq!(DcpLook::from_bytes(&bytes).unwrap().name(), "カラー");
        for replacement in [b"\0est look\0", b"Test lookX", b"\xffest look\0"] {
            let mut bytes = profile();
            bytes[offset..offset + replacement.len()].copy_from_slice(replacement);
            assert!(DcpLook::from_bytes(&bytes).is_err());
        }
    }

    #[test]
    fn unused_calibration_does_not_affect_imported_look() {
        let mut bytes = profile();
        let original = DcpLook::from_bytes(&bytes).unwrap();
        let field = entry(&bytes, tags::COLOR_MATRIX1) + 8;
        let offset = u32::from_le_bytes(bytes[field..field + 4].try_into().unwrap()) as usize;
        bytes[offset..offset + 4].copy_from_slice(&99999i32.to_le_bytes());
        let changed = DcpLook::from_bytes(&bytes).unwrap();
        assert_eq!(original.table, changed.table);
        assert_eq!(original.tone, changed.tone);
    }

    fn serialized_directory(ifd: DirectoryWriter) -> Vec<u8> {
        let mut writer = TiffWriter::new(Cursor::new(Vec::new())).unwrap();
        let root = ifd.build(&mut writer).unwrap();
        let mut bytes = writer.finalize(root).unwrap().into_inner();
        bytes[2..4].copy_from_slice(&PROFILE_MAGIC.to_le_bytes());
        bytes
    }

    #[test]
    fn embedding_changes_only_look_fields_and_backward_version() {
        let mut ifd = DirectoryWriter::new();
        let preserved = [
            (
                tags::PROFILE_NAME,
                Value::Ascii(CString::new("Camera default").unwrap()),
            ),
            (
                tags::UNIQUE_CAMERA_MODEL,
                Value::Ascii(CString::new("Test camera").unwrap()),
            ),
            (tags::COLOR_MATRIX1, Value::SRational(vec![(7, 9); 9])),
            (tags::FORWARD_MATRIX1, Value::SRational(vec![(3, 5); 9])),
            (
                tags::AS_SHOT_NEUTRAL,
                Value::Rational(vec![(1, 2), (1, 1), (3, 4)]),
            ),
            (tags::BASELINE_EXPOSURE, Value::SRational(vec![(2, 1)])),
            (
                tags::PROFILE_HUE_SAT_MAP_DATA1,
                Value::Float(vec![0.0, 1.0, 1.0]),
            ),
            (tags::EXTRA_CAMERA_PROFILES, Value::Long(vec![100, 200])),
        ];
        let mut expected = DirectoryWriter::new();
        for (tag, value) in preserved {
            expected.add(tag, value.clone());
            ifd.add(tag, value);
        }
        ifd.add(
            tags::PROFILE_TONE_CURVE,
            Value::Float(vec![0.0, 0.0, 1.0, 1.0]),
        );
        ifd.add(tags::DNG_BACKWARD_VERSION, Value::Byte(vec![1, 3, 0, 0]));
        DcpLook::from_bytes(&profile()).unwrap().embed(&mut ifd);
        let actual_bytes = serialized_directory(ifd);
        let expected_bytes = serialized_directory(expected);
        let actual = Directory::parse(&actual_bytes).unwrap();
        let expected = Directory::parse(&expected_bytes).unwrap();
        assert_eq!(actual.tags.len(), expected.tags.len() + 5);
        for (id, original) in &expected.tags {
            let written = &actual.tags[id];
            assert_eq!(
                (written.kind, written.count, written.bytes),
                (original.kind, original.count, original.bytes)
            );
        }
        let look = DcpLook::from_bytes(&actual_bytes).unwrap();
        let source = DcpLook::from_bytes(&profile()).unwrap();
        assert_eq!(look.name(), "Camera default");
        assert_eq!(look.table, source.table);
        assert_eq!(look.tone, source.tone);
        assert_eq!(
            actual.tags[&tags::DNG_BACKWARD_VERSION].bytes,
            &[1, 4, 0, 0]
        );
    }

    #[test]
    fn truncated_and_mutated_containers_never_panic() {
        for big_endian in [false, true] {
            let bytes = fixture::profile("Test camera", big_endian);
            for end in 0..bytes.len() {
                assert!(std::panic::catch_unwind(|| DcpLook::from_bytes(&bytes[..end])).is_ok());
            }
            for index in 0..bytes.len() {
                let mut modified = bytes.clone();
                modified[index] ^= 0xff;
                assert!(std::panic::catch_unwind(|| DcpLook::from_bytes(&modified)).is_ok());
            }
        }
        assert!(DcpLook::from_bytes(&vec![0; MAX_BYTES + 1]).is_err());
    }
}
