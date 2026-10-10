//! Shared JPEG APP1/EXIF reader for rendering calibration and DNG metadata.
//! Reads IFD0 and the EXIF sub-IFD; ignores MakerNotes, GPS and IFD1.

#[derive(Default, Debug, Clone)]
pub struct JpegExif {
    pub make: Option<String>,
    pub model: Option<String>,
    pub software: Option<String>,
    pub datetime: Option<String>,
    pub orientation: Option<u16>,

    pub exposure_time: Option<(u32, u32)>,
    pub f_number: Option<(u32, u32)>,
    pub exposure_program: Option<u16>,
    pub color_space: Option<u16>,
    pub iso: Option<u16>,
    pub date_time_original: Option<String>,
    pub exposure_bias: Option<(i32, i32)>,
    pub flash: Option<u16>,
    pub focal_length: Option<(u32, u32)>,
    pub focal_length_in_35mm: Option<u16>,
    pub body_serial: Option<String>,
}

/// Find the APP1/Exif TIFF block in `jpeg` and parse the fields above.
/// Returns `None` if the JPEG has no APP1 segment or the TIFF header is
/// malformed; *partial* parses still return `Some` with whatever was
/// readable.
pub fn parse_jpeg_exif(jpeg: &[u8]) -> Option<JpegExif> {
    let tiff = find_app1_tiff(jpeg)?;
    let header = tiff.get(..8)?;
    let endian = if &header[..2] == b"II" {
        Endian::Le
    } else if &header[..2] == b"MM" {
        Endian::Be
    } else {
        return None;
    };
    if endian.read_u16(&tiff[2..4])? != 42 {
        return None;
    }
    let ifd0_off = endian.read_u32(&tiff[4..8])? as usize;

    let mut out = JpegExif::default();
    let mut exif_subifd_off: Option<usize> = None;

    walk_ifd(tiff, ifd0_off, endian, |tag, ty, count, val_or_off| {
        match tag {
            0x010F => out.make = read_ascii(tiff, ty, count, val_or_off, endian),
            0x0110 => out.model = read_ascii(tiff, ty, count, val_or_off, endian),
            0x0112 => out.orientation = read_u16(ty, val_or_off, endian),
            0x0131 => out.software = read_ascii(tiff, ty, count, val_or_off, endian),
            0x0132 => out.datetime = read_ascii(tiff, ty, count, val_or_off, endian),
            0x8769 => {
                // EXIF sub-IFD pointer (LONG, count=1, value-as-offset).
                exif_subifd_off = endian.read_u32(&val_or_off).map(|v| v as usize);
            }
            _ => {}
        }
    });

    if let Some(off) = exif_subifd_off {
        walk_ifd(tiff, off, endian, |tag, ty, count, val_or_off| match tag {
            0x829A => out.exposure_time = read_rational(tiff, ty, count, val_or_off, endian),
            0x829D => out.f_number = read_rational(tiff, ty, count, val_or_off, endian),
            // shortcut: accept the corpus SHORT scalar; trace other native forms before using them for rendering.
            0xA001 if count == 1 && ty == 3 => out.color_space = read_u16(ty, val_or_off, endian),
            0x8822 => out.exposure_program = read_u16(ty, val_or_off, endian),
            0x8827 => out.iso = read_u16(ty, val_or_off, endian),
            0x9003 => out.date_time_original = read_ascii(tiff, ty, count, val_or_off, endian),
            0x9204 => out.exposure_bias = read_srational(tiff, ty, count, val_or_off, endian),
            0x9209 => out.flash = read_u16(ty, val_or_off, endian),
            0x920A => out.focal_length = read_rational(tiff, ty, count, val_or_off, endian),
            0xA405 => out.focal_length_in_35mm = read_u16(ty, val_or_off, endian),
            0xA431 => out.body_serial = read_ascii(tiff, ty, count, val_or_off, endian),
            _ => {}
        });
    }

    Some(out)
}

#[derive(Copy, Clone)]
enum Endian {
    Le,
    Be,
}

impl Endian {
    fn read_u16(self, b: &[u8]) -> Option<u16> {
        let bytes = <[u8; 2]>::try_from(b.get(..2)?).ok()?;
        Some(match self {
            Endian::Le => u16::from_le_bytes(bytes),
            Endian::Be => u16::from_be_bytes(bytes),
        })
    }

    fn read_u32(self, b: &[u8]) -> Option<u32> {
        let bytes = <[u8; 4]>::try_from(b.get(..4)?).ok()?;
        Some(match self {
            Endian::Le => u32::from_le_bytes(bytes),
            Endian::Be => u32::from_be_bytes(bytes),
        })
    }

    fn read_i32(self, b: &[u8]) -> Option<i32> {
        let bytes = <[u8; 4]>::try_from(b.get(..4)?).ok()?;
        Some(match self {
            Endian::Le => i32::from_le_bytes(bytes),
            Endian::Be => i32::from_be_bytes(bytes),
        })
    }
}

/// Locate the embedded TIFF block ("II*\0" or "MM\0*") inside a JPEG's
/// APP1/Exif marker. Returns a slice starting at the TIFF header.
fn find_app1_tiff(jpeg: &[u8]) -> Option<&[u8]> {
    if jpeg.get(..2)? != b"\xFF\xD8" {
        return None;
    }
    let mut i = 2;
    while i + 4 <= jpeg.len() {
        if jpeg[i] != 0xFF {
            return None;
        }
        let marker = jpeg[i + 1];
        // SOS (FFDA) starts the entropy-coded scan; everything after is
        // image data, no more APP segments.
        if marker == 0xDA {
            return None;
        }
        let seg_len = u16::from_be_bytes([jpeg[i + 2], jpeg[i + 3]]) as usize;
        if seg_len < 2 || i + 2 + seg_len > jpeg.len() {
            return None;
        }
        let body = &jpeg[i + 4..i + 2 + seg_len];
        if marker == 0xE1 && body.get(..6) == Some(b"Exif\0\0") {
            return Some(&body[6..]);
        }
        i += 2 + seg_len;
    }
    None
}

/// Walk a TIFF IFD at `off` bytes into `tiff`, calling `f` once per
/// entry. Each callback receives the 2-byte tag, 2-byte type, 4-byte
/// count, and the 4-byte raw value-or-offset slot exactly as it appears
/// in the entry — the caller decides whether to dereference it.
fn walk_ifd(tiff: &[u8], off: usize, endian: Endian, mut f: impl FnMut(u16, u16, u32, [u8; 4])) {
    let Some(n) = endian.read_u16(tiff.get(off..).unwrap_or(&[])) else {
        return;
    };
    for i in 0..n as usize {
        let Some(p) = off.checked_add(2 + i * 12) else {
            return;
        };
        let Some(entry) = tiff.get(p..).and_then(|v| v.get(..12)) else {
            return;
        };
        let tag = endian.read_u16(&entry[..2]).unwrap_or(0);
        let ty = endian.read_u16(&entry[2..4]).unwrap_or(0);
        let count = endian.read_u32(&entry[4..8]).unwrap_or(0);
        let val_or_off: [u8; 4] = entry[8..12].try_into().unwrap();
        f(tag, ty, count, val_or_off);
    }
}

fn read_ascii(
    tiff: &[u8],
    ty: u16,
    count: u32,
    val_or_off: [u8; 4],
    endian: Endian,
) -> Option<String> {
    if ty != 2 || count == 0 {
        return None;
    }
    let total = count as usize;
    let bytes = if total <= 4 {
        let n = total.min(4);
        val_or_off[..n].to_vec()
    } else {
        let off = endian.read_u32(&val_or_off)? as usize;
        tiff.get(off..)?.get(..total)?.to_vec()
    };
    let s: String = bytes
        .iter()
        .take_while(|&&c| c != 0)
        .map(|&c| c as char)
        .collect();
    let trimmed = s.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

fn read_u16(ty: u16, val_or_off: [u8; 4], endian: Endian) -> Option<u16> {
    match ty {
        3 /* SHORT */ => endian.read_u16(&val_or_off[..2]),
        4 /* LONG */ => endian.read_u32(&val_or_off).map(|v| v.min(u16::MAX as u32) as u16),
        _ => None,
    }
}

fn read_rational(
    tiff: &[u8],
    ty: u16,
    count: u32,
    val_or_off: [u8; 4],
    endian: Endian,
) -> Option<(u32, u32)> {
    if ty != 5 || count == 0 {
        return None;
    }
    // RATIONAL is always 8 bytes per entry → out-of-line.
    let off = endian.read_u32(&val_or_off)? as usize;
    let value = tiff.get(off..)?.get(..8)?;
    let n = endian.read_u32(&value[..4])?;
    let d = endian.read_u32(&value[4..])?;
    if d == 0 {
        return None;
    }
    Some((n, d))
}

fn read_srational(
    tiff: &[u8],
    ty: u16,
    count: u32,
    val_or_off: [u8; 4],
    endian: Endian,
) -> Option<(i32, i32)> {
    if ty != 10 || count == 0 {
        return None;
    }
    let off = endian.read_u32(&val_or_off)? as usize;
    let value = tiff.get(off..)?.get(..8)?;
    let n = endian.read_i32(&value[..4])?;
    let d = endian.read_i32(&value[4..])?;
    if d == 0 {
        return None;
    }
    Some((n, d))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn calibration_jpeg(big: bool) -> Vec<u8> {
        let u16bytes = |v: u16| {
            if big {
                v.to_be_bytes()
            } else {
                v.to_le_bytes()
            }
        };
        let u32bytes = |v: u32| {
            if big {
                v.to_be_bytes()
            } else {
                v.to_le_bytes()
            }
        };
        let mut tiff = if big { b"MM".to_vec() } else { b"II".to_vec() };
        tiff.extend(u16bytes(42));
        tiff.extend(u32bytes(8));
        tiff.extend(u16bytes(1));
        tiff.extend(u16bytes(0x8769));
        tiff.extend(u16bytes(4));
        tiff.extend(u32bytes(1));
        tiff.extend(u32bytes(26));
        tiff.extend(u32bytes(0));
        tiff.extend(u16bytes(2));
        tiff.extend(u16bytes(0x829d));
        tiff.extend(u16bytes(5));
        tiff.extend(u32bytes(1));
        tiff.extend(u32bytes(56));
        tiff.extend(u16bytes(0xa001));
        tiff.extend(u16bytes(3));
        tiff.extend(u32bytes(1));
        tiff.extend(u16bytes(65535));
        tiff.extend(u16bytes(0));
        tiff.extend(u32bytes(0));
        tiff.extend(u32bytes(28));
        tiff.extend(u32bytes(10));
        let mut jpeg = b"\xff\xd8\xff\xe1".to_vec();
        jpeg.extend(((tiff.len() + 8) as u16).to_be_bytes());
        jpeg.extend(b"Exif\0\0");
        jpeg.extend(tiff);
        jpeg
    }

    #[test]
    fn calibration_tags_preserve_endian_and_rational_values() {
        for big in [false, true] {
            let jpeg = calibration_jpeg(big);
            let tags = parse_jpeg_exif(&jpeg).unwrap();
            assert_eq!(tags.f_number, Some((28, 10)));
            assert_eq!(tags.color_space, Some(65535));
            for end in 0..jpeg.len() {
                assert!(parse_jpeg_exif(&jpeg[..end]).is_none());
            }
        }
    }

    #[test]
    fn malformed_calibration_tags_do_not_panic_or_supply_values() {
        for length in 0..8 {
            let mut jpeg = b"\xff\xd8\xff\xe1".to_vec();
            jpeg.extend(((length + 8) as u16).to_be_bytes());
            jpeg.extend(b"Exif\0\0");
            jpeg.extend(vec![0; length]);
            assert!(parse_jpeg_exif(&jpeg).is_none());
        }
        let mut jpeg = calibration_jpeg(false);
        // TIFF starts at byte 12; rational denominator is at TIFF byte 60.
        jpeg[72..76].fill(0);
        assert_eq!(parse_jpeg_exif(&jpeg).unwrap().f_number, None);
        // Out-of-range EXIF sub-IFD pointer and out-of-range rational offset.
        for offset in [30, 48] {
            let mut jpeg = calibration_jpeg(false);
            jpeg[offset..offset + 4].fill(255);
            assert_eq!(parse_jpeg_exif(&jpeg).unwrap().f_number, None);
        }
    }
}
