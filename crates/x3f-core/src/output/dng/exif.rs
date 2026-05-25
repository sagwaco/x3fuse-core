//! Capture-metadata extraction + DNG EXIF wiring.
//!
//! Two output surfaces:
//!
//! - A handful of standard TIFF tags written into IFD0 directly
//!   (`Make`, `Model`, `Software`, `DateTime`).
//! - A linked EXIF sub-IFD (TIFF tag 34665 in IFD0) that holds the
//!   photographic metadata (`ExposureTime`, `FNumber`, `ISO`, etc).
//!
//! Two input shapes:
//!
//! - **Merrill / DP\* / SD1**: data lives in the file's `PROP` table
//!   (`APERTURE`, `SHUTTER`, `ISO`, `TIME`, `CAMMANUF`, …). Every value is
//!   an ASCII string we parse out.
//! - **Quattro / SDQH**: no `PROP` table; we fall back to the `Capture*`
//!   CAMF matrices (`CaptureAperture`, `CaptureExpTime` in µs, etc) and
//!   the `CameraSerialNumber` CAMF text. Make / Model / DateTime aren't
//!   recoverable from CAMF on the files we've seen, so those slots stay
//!   empty for Quattro until we add JPEG-EXIF parsing as a fallback.

use std::ffi::CString;

use crate::Reader;

use super::tags as t;
use super::tiff_writer::{DirectoryWriter, Value};

/// Capture metadata parsed out of the X3F. Every field is `Option`
/// because most values are either Merrill-only or Quattro-only; the
/// emitter writes a tag iff the corresponding field is `Some`.
#[derive(Default, Debug, Clone)]
pub(crate) struct CaptureMetadata {
    // IFD0 / standard TIFF top-level
    pub make: Option<String>,
    pub model: Option<String>,
    pub software: Option<String>,
    pub datetime: Option<String>, // "YYYY:MM:DD HH:MM:SS"

    /// TIFF/DNG `Orientation` tag value (1, 3, 6, or 8). PROP[ROTATION]
    /// on Merrill, JPEG IFD0 Orientation on Quattro.
    pub orientation: Option<u16>,

    // EXIF sub-IFD
    pub exposure_time: Option<(u32, u32)>,
    pub f_number: Option<(u32, u32)>,
    pub exposure_program: Option<u16>,
    pub iso: Option<u16>,
    pub date_time_original: Option<String>,
    pub exposure_bias: Option<(i32, i32)>,
    pub flash: Option<u16>,
    pub focal_length: Option<(u32, u32)>,
    pub focal_length_in_35mm: Option<u16>,
    pub body_serial: Option<String>,
    /// Raw PROP[LENSMODEL] value (Sigma's internal lens identifier code,
    /// e.g. `"1004"` for the DP2 Merrill's built-in 30mm or `"32776"`
    /// for the SD1M 30mm prime). Used only by the opcode-lookup path;
    /// not written to the DNG (the EXIF LensModel tag wants a friendly
    /// string, which we'd take from JPEG-EXIF if we wired that up).
    pub lens_model: Option<String>,
}

impl CaptureMetadata {
    pub(crate) fn from_reader(reader: &Reader) -> Self {
        let mut m = Self::default();
        m.orientation = reader.prop_orientation();

        // ---- Merrill: PROP table ---------------------------------------
        if let Some(s) = reader.dng_prop("CAMMANUF") {
            m.make = Some(trim_owned(s));
        }
        if let Some(s) = reader.dng_prop("CAMMODEL") {
            m.model = Some(trim_owned(s));
        }
        if let Some(s) = reader.dng_prop("FIRMVERS") {
            m.software = Some(format!("Sigma firmware {}", trim_owned(s)));
        }
        if let Some(s) = reader.dng_prop("TIME") {
            if let Ok(unix) = s.trim().parse::<i64>() {
                let f = format_unix_time(unix);
                m.datetime = Some(f.clone());
                m.date_time_original = Some(f);
            }
        }
        // Prefer the SH_DESC / AP_DESC strings ("1/25", "2.8") because they
        // give a clean RATIONAL; SHUTTER / APERTURE are the precise math
        // values (1/sqrt(2)·N etc) that round awkwardly.
        if let Some(et) = reader
            .dng_prop("SH_DESC")
            .and_then(|s| parse_shutter_desc(&s))
        {
            m.exposure_time = Some(et);
        } else if let Some(s) = reader
            .dng_prop("SHUTTER")
            .and_then(|s| s.trim().parse::<f64>().ok())
        {
            m.exposure_time = float_to_rational(s);
        }
        if let Some(s) = reader
            .dng_prop("AP_DESC")
            .and_then(|s| s.trim().parse::<f64>().ok())
        {
            m.f_number = float_to_rational(s);
        } else if let Some(s) = reader
            .dng_prop("APERTURE")
            .and_then(|s| s.trim().parse::<f64>().ok())
        {
            m.f_number = float_to_rational(s);
        }
        if let Some(s) = reader.dng_prop("PMODE") {
            m.exposure_program = Some(prop_pmode_to_exif(s.trim()));
        }
        if let Some(iso) = reader
            .dng_prop("ISO")
            .and_then(|s| s.trim().parse::<u32>().ok())
        {
            m.iso = Some(u16_clamp(iso));
        }
        if let Some(s) = reader
            .dng_prop("EXPCOMP")
            .and_then(|s| s.trim().parse::<f64>().ok())
        {
            m.exposure_bias = float_to_srational(s);
        }
        if let Some(s) = reader.dng_prop("FLASH") {
            m.flash = Some(if s.trim().eq_ignore_ascii_case("on") {
                0x01
            } else {
                0x00
            });
        }
        if let Some(fl) = reader
            .dng_prop("FLENGTH")
            .and_then(|s| s.trim().parse::<f64>().ok())
        {
            m.focal_length = float_to_rational(fl);
        }
        if let Some(fl35) = reader
            .dng_prop("FLEQ35MM")
            .and_then(|s| s.trim().parse::<u32>().ok())
        {
            m.focal_length_in_35mm = Some(u16_clamp(fl35));
        }
        if let Some(s) = reader.dng_prop("CAMSERIAL") {
            m.body_serial = Some(trim_owned(s));
        }
        if let Some(s) = reader.dng_prop("LENSMODEL") {
            let v = trim_owned(s);
            if !v.is_empty() {
                m.lens_model = Some(v);
            }
        }

        // ---- Quattro: CAMF Capture* + CameraSerialNumber ---------------
        if m.f_number.is_none() {
            if let Some(v) = reader.dng_camf_float("CaptureAperture") {
                m.f_number = float_to_rational(v);
            }
        }
        if m.exposure_time.is_none() {
            // CaptureExpTime is in microseconds; CaptureShutter is the
            // (1/√2)^N "shutter speed value" representation, harder to use.
            if let Some(us) = reader.dng_camf_float("CaptureExpTime") {
                let secs = us / 1_000_000.0;
                m.exposure_time = float_to_rational(secs);
            }
        }
        if m.iso.is_none() {
            if let Some(iso) = reader.dng_camf_float("CaptureISO") {
                m.iso = Some(u16_clamp(iso.round().max(0.0) as u32));
            }
        }
        if m.exposure_bias.is_none() {
            if let Some(ev) = reader.dng_camf_float("CaptureExpComp") {
                m.exposure_bias = float_to_srational(ev);
            }
        }
        if m.body_serial.is_none() {
            if let Some(s) = reader.dng_camf_text("CameraSerialNumber") {
                m.body_serial = Some(trim_owned(s));
            }
        }

        // ---- Final fallback: parse the embedded JPEG's EXIF ----------
        //   Sigma writes a normal APP1/EXIF block into the thumbnail with
        //   the camera's view of capture metadata (Make / Model / Software
        //   / DateTime / Exposure / Lens info / serial). Quattro files
        //   don't have a PROP table, so without this fallback Make / Model
        //   / DateTime stay blank for them. We only fill in fields the
        //   PROP/CAMF passes left empty.
        if let Some(jpeg) = reader.dng_thumb_jpeg_bytes() {
            if let Some(j) = parse_jpeg_exif(&jpeg) {
                m.fill_from_jpeg(&j);
            }
        }
        m
    }

    fn fill_from_jpeg(&mut self, j: &JpegExif) {
        if self.orientation.is_none() {
            self.orientation = j.orientation;
        }
        if self.make.is_none() {
            self.make = j.make.clone();
        }
        if self.model.is_none() {
            self.model = j.model.clone();
        }
        if self.software.is_none() {
            self.software = j.software.clone();
        }
        if self.datetime.is_none() {
            self.datetime = j.datetime.clone();
        }
        if self.date_time_original.is_none() {
            self.date_time_original = j.date_time_original.clone().or_else(|| j.datetime.clone());
        }
        if self.exposure_time.is_none() {
            self.exposure_time = j.exposure_time;
        }
        if self.f_number.is_none() {
            self.f_number = j.f_number;
        }
        if self.exposure_program.is_none() {
            self.exposure_program = j.exposure_program;
        }
        if self.iso.is_none() {
            self.iso = j.iso;
        }
        if self.exposure_bias.is_none() {
            self.exposure_bias = j.exposure_bias;
        }
        if self.flash.is_none() {
            self.flash = j.flash;
        }
        if self.focal_length.is_none() {
            self.focal_length = j.focal_length;
        }
        if self.focal_length_in_35mm.is_none() {
            self.focal_length_in_35mm = j.focal_length_in_35mm;
        }
        if self.body_serial.is_none() {
            self.body_serial = j.body_serial.clone();
        }
    }
}

/// Add `Make` / `Model` / `Software` / `DateTime` to an IFD writer (no-op
/// for fields the source didn't carry).
pub(crate) fn add_top_level_tags(meta: &CaptureMetadata, ifd: &mut DirectoryWriter) {
    if let Some(s) = ascii_value(&meta.make) {
        ifd.add(t::MAKE, s);
    }
    if let Some(s) = ascii_value(&meta.model) {
        ifd.add(t::MODEL, s);
    }
    if let Some(s) = ascii_value(&meta.software) {
        ifd.add(t::SOFTWARE, s);
    }
    if let Some(s) = ascii_value(&meta.datetime) {
        ifd.add(t::DATETIME, s);
    }
}

/// Build the EXIF sub-IFD from the parsed metadata. Returns `None` if
/// none of the EXIF fields are populated (so we don't emit an empty
/// sub-IFD or an ExifIFDPointer that points at junk).
pub(crate) fn build_subifd(meta: &CaptureMetadata) -> Option<DirectoryWriter> {
    let mut ifd = DirectoryWriter::new();
    let mut any = false;
    if let Some((n, d)) = meta.exposure_time {
        ifd.add(t::EXIF_EXPOSURE_TIME, Value::Rational(vec![(n, d)]));
        any = true;
    }
    if let Some((n, d)) = meta.f_number {
        ifd.add(t::EXIF_F_NUMBER, Value::Rational(vec![(n, d)]));
        any = true;
    }
    if let Some(p) = meta.exposure_program {
        ifd.add(t::EXIF_EXPOSURE_PROGRAM, Value::Short(vec![p]));
        any = true;
    }
    if let Some(iso) = meta.iso {
        ifd.add(t::EXIF_ISO_SPEED_RATINGS, Value::Short(vec![iso]));
        any = true;
    }
    if let Some(s) = ascii_value(&meta.date_time_original) {
        ifd.add(t::EXIF_DATE_TIME_ORIGINAL, s.clone());
        ifd.add(t::EXIF_DATE_TIME_DIGITIZED, s);
        any = true;
    }
    if let Some((n, d)) = meta.exposure_bias {
        ifd.add(t::EXIF_EXPOSURE_BIAS_VALUE, Value::SRational(vec![(n, d)]));
        any = true;
    }
    if let Some(f) = meta.flash {
        ifd.add(t::EXIF_FLASH, Value::Short(vec![f]));
        any = true;
    }
    if let Some((n, d)) = meta.focal_length {
        ifd.add(t::EXIF_FOCAL_LENGTH, Value::Rational(vec![(n, d)]));
        any = true;
    }
    if let Some(fl35) = meta.focal_length_in_35mm {
        ifd.add(t::EXIF_FOCAL_LENGTH_IN_35MM, Value::Short(vec![fl35]));
        any = true;
    }
    if let Some(s) = ascii_value(&meta.body_serial) {
        ifd.add(t::EXIF_BODY_SERIAL_NUMBER, s);
        any = true;
    }
    any.then_some(ifd)
}

// --- helpers -----------------------------------------------------------

fn trim_owned(s: String) -> String {
    s.trim().to_string()
}

fn ascii_value(s: &Option<String>) -> Option<Value> {
    let s = s.as_deref()?.trim();
    if s.is_empty() {
        return None;
    }
    // Drop interior NULs defensively — Sigma's PROP entries have been
    // observed to contain a stray space-only value (" ") which is fine,
    // but a NUL would corrupt the tag.
    let cleaned: String = s.chars().filter(|&c| c != '\0').collect();
    let c = CString::new(cleaned).ok()?;
    Some(Value::Ascii(c))
}

/// Parse a Sigma `SH_DESC` shutter description like `"1/25"` or `"2.5"`
/// into an EXIF RATIONAL. Anything we can't parse cleanly returns
/// `None` so the caller falls back to the float-based path.
fn parse_shutter_desc(s: &str) -> Option<(u32, u32)> {
    let s = s.trim();
    if let Some((num, den)) = s.split_once('/') {
        let n: u32 = num.trim().parse().ok()?;
        let d: u32 = den.trim().parse().ok()?;
        if d == 0 {
            return None;
        }
        return Some((n, d));
    }
    let v: f64 = s.parse().ok()?;
    float_to_rational(v)
}

/// Convert a non-negative float to `(num, denom)` with denom 1_000_000
/// (microsecond / six-decimal precision). Returns `None` on NaN or inf.
fn float_to_rational(v: f64) -> Option<(u32, u32)> {
    if !v.is_finite() || v < 0.0 {
        return None;
    }
    const DEN: u32 = 1_000_000;
    let num = (v * DEN as f64).round();
    if num > u32::MAX as f64 {
        // value too large — drop precision until it fits
        if v > u32::MAX as f64 {
            return None;
        }
        return Some((v.round() as u32, 1));
    }
    Some((num as u32, DEN))
}

/// Same idea but signed (for ExposureBiasValue, which is an SRATIONAL).
fn float_to_srational(v: f64) -> Option<(i32, i32)> {
    if !v.is_finite() {
        return None;
    }
    const DEN: i32 = 100_000;
    let num = (v * DEN as f64).round();
    let num = num.clamp(i32::MIN as f64, i32::MAX as f64) as i32;
    Some((num, DEN))
}

fn u16_clamp(v: u32) -> u16 {
    v.min(u16::MAX as u32) as u16
}

/// Map Sigma's PROP `PMODE` ("A" / "M" / "P" / "S") to the EXIF
/// `ExposureProgram` enum. Unknown / blank → 0 ("Not defined").
fn prop_pmode_to_exif(s: &str) -> u16 {
    match s {
        "M" => 1, // Manual
        "P" => 2, // Normal program
        "A" => 3, // Aperture priority
        "S" => 4, // Shutter priority
        _ => 0,
    }
}

/// Format a Unix timestamp (seconds since 1970-01-01 UTC) as
/// `"YYYY:MM:DD HH:MM:SS"` for EXIF DateTime/DateTimeOriginal. Hand-
/// rolled because we don't want a `chrono` dep just for this.
fn format_unix_time(unix: i64) -> String {
    let (y, mo, d, h, mi, s) = unix_to_ymd_hms(unix);
    format!("{y:04}:{mo:02}:{d:02} {h:02}:{mi:02}:{s:02}")
}

// --- JPEG APP1/EXIF parser --------------------------------------------
//
// Just enough to pull standard tags out of IFD0 + the EXIF sub-IFD. We
// don't bother with MakerNotes, GPS, IFD1 or thumbnails — none of those
// are needed for our DNG output.

#[derive(Default, Debug, Clone)]
pub(crate) struct JpegExif {
    pub make: Option<String>,
    pub model: Option<String>,
    pub software: Option<String>,
    pub datetime: Option<String>,
    pub orientation: Option<u16>,

    pub exposure_time: Option<(u32, u32)>,
    pub f_number: Option<(u32, u32)>,
    pub exposure_program: Option<u16>,
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
pub(crate) fn parse_jpeg_exif(jpeg: &[u8]) -> Option<JpegExif> {
    let tiff = find_app1_tiff(jpeg)?;
    let endian = if &tiff[..2] == b"II" {
        Endian::Le
    } else if &tiff[..2] == b"MM" {
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
    let Some(n) = endian.read_u16(&tiff.get(off..).unwrap_or(&[])[..]) else {
        return;
    };
    for i in 0..n as usize {
        let p = off + 2 + i * 12;
        let Some(entry) = tiff.get(p..p + 12) else {
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
        tiff.get(off..off + total)?.to_vec()
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
    let n = endian.read_u32(tiff.get(off..off + 4)?)?;
    let d = endian.read_u32(tiff.get(off + 4..off + 8)?)?;
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
    let n = endian.read_i32(tiff.get(off..off + 4)?)?;
    let d = endian.read_i32(tiff.get(off + 4..off + 8)?)?;
    if d == 0 {
        return None;
    }
    Some((n, d))
}

fn unix_to_ymd_hms(unix: i64) -> (i32, u32, u32, u32, u32, u32) {
    // Howard Hinnant's day-number algorithm, adapted for i64. Treats the
    // input as UTC. Sigma's PROP[TIME] is the camera clock in local time
    // serialised as a unix-epoch-style integer; we just present it back
    // with the same offset, which is what every other Sigma tool does.
    let secs_per_day = 86_400_i64;
    let mut z = unix.div_euclid(secs_per_day);
    let mut day_secs = unix.rem_euclid(secs_per_day);
    if day_secs < 0 {
        day_secs += secs_per_day;
        z -= 1;
    }
    let h = (day_secs / 3600) as u32;
    let mi = ((day_secs % 3600) / 60) as u32;
    let s = (day_secs % 60) as u32;
    z += 719_468; // shift epoch to 0000-03-01
    let era = if z >= 0 {
        z / 146_097
    } else {
        (z - 146_096) / 146_097
    };
    let doe = (z - era * 146_097) as u32;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y = (yoe as i64 + era * 400) as i32;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let mo = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if mo <= 2 { y + 1 } else { y };
    (y, mo, d, h, mi, s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_shutter_fraction() {
        assert_eq!(parse_shutter_desc("1/25"), Some((1, 25)));
        assert_eq!(parse_shutter_desc(" 1 / 1000 "), Some((1, 1000)));
    }

    #[test]
    fn parse_shutter_decimal_falls_back() {
        let (n, d) = parse_shutter_desc("0.5").unwrap();
        // 0.5 == 500_000 / 1_000_000
        assert_eq!(n as f64 / d as f64, 0.5);
    }

    #[test]
    fn parse_shutter_zero_denom_rejected() {
        assert_eq!(parse_shutter_desc("1/0"), None);
    }

    #[test]
    fn float_to_rational_lossless_decimals() {
        let (n, d) = float_to_rational(2.8).unwrap();
        assert_eq!(d, 1_000_000);
        assert_eq!(n, 2_800_000);
    }

    #[test]
    fn unix_epoch_matches_known_date() {
        // 0 unix → 1970-01-01 00:00:00 UTC.
        assert_eq!(format_unix_time(0), "1970:01:01 00:00:00");
        // 1_756_598_400 = exactly 20331 * 86400 → start of 2025-08-31 UTC.
        assert_eq!(format_unix_time(1_756_598_400), "2025:08:31 00:00:00");
        // PROP[TIME] from Sigma DP2M0643.X3F. The clock is camera-local
        // serialised as a naive unix-epoch integer, so the formatted
        // string is the camera's local wall-clock at capture.
        assert_eq!(format_unix_time(1_756_630_646), "2025:08:31 08:57:26");
    }

    #[test]
    fn pmode_mapping() {
        assert_eq!(prop_pmode_to_exif("A"), 3);
        assert_eq!(prop_pmode_to_exif("M"), 1);
        assert_eq!(prop_pmode_to_exif(""), 0);
        assert_eq!(prop_pmode_to_exif("?"), 0);
    }

    /// Build a minimal JPEG with an APP1/Exif segment containing a few
    /// known IFD0 tags + an EXIF sub-IFD; exercise the parser end-to-end.
    fn synth_jpeg_with_exif() -> Vec<u8> {
        // Layout, all little-endian:
        //   TIFF header (8 B): "II" + 0x002A + IFD0 offset (8)
        //   IFD0 at offset 8: 4 entries, then next-IFD = 0
        //   Tags (in numeric order):
        //     0x010F Make    ASCII "SIGMA\0"          → external @ off A
        //     0x0110 Model   ASCII "Quattro test\0"   → external @ off B
        //     0x0132 DateTime ASCII "2025:01:02 03:04:05\0" → external @ off C
        //     0x8769 ExifIFD LONG  → value-as-offset @ off D
        //   EXIF sub-IFD at off D: 2 entries
        //     0x829A ExposureTime  RATIONAL (1, 100)  → external @ off E
        //     0x8827 ISO           SHORT  100 (inline)
        let make = b"SIGMA\0";
        let model = b"Quattro test\0";
        let datetime = b"2025:01:02 03:04:05\0";

        let mut tiff: Vec<u8> = Vec::new();
        tiff.extend_from_slice(b"II");
        tiff.extend_from_slice(&42u16.to_le_bytes());
        tiff.extend_from_slice(&8u32.to_le_bytes()); // IFD0 starts at 8

        // We'll fill IFD0 with placeholders, then patch external offsets
        // once we know where each payload landed.
        let ifd0_at = tiff.len();
        assert_eq!(ifd0_at, 8);
        tiff.extend_from_slice(&4u16.to_le_bytes()); // 4 entries
                                                     // Reserve 4 * 12 entry bytes + 4 next-ifd bytes
        let entries_at = tiff.len();
        tiff.resize(entries_at + 4 * 12 + 4, 0);

        // Append external strings
        let make_off = tiff.len() as u32;
        tiff.extend_from_slice(make);
        if tiff.len() % 2 == 1 {
            tiff.push(0);
        }
        let model_off = tiff.len() as u32;
        tiff.extend_from_slice(model);
        if tiff.len() % 2 == 1 {
            tiff.push(0);
        }
        let datetime_off = tiff.len() as u32;
        tiff.extend_from_slice(datetime);
        if tiff.len() % 2 == 1 {
            tiff.push(0);
        }

        // EXIF sub-IFD
        let exif_off = tiff.len() as u32;
        tiff.extend_from_slice(&2u16.to_le_bytes()); // 2 entries
        let exif_entries_at = tiff.len();
        tiff.resize(exif_entries_at + 2 * 12 + 4, 0);
        // ExposureTime payload (8 bytes)
        let et_off = tiff.len() as u32;
        tiff.extend_from_slice(&1u32.to_le_bytes());
        tiff.extend_from_slice(&100u32.to_le_bytes());

        // Patch IFD0 entries
        let write_entry =
            |buf: &mut Vec<u8>, idx: usize, tag: u16, ty: u16, count: u32, val: [u8; 4]| {
                let p = entries_at + idx * 12;
                buf[p..p + 2].copy_from_slice(&tag.to_le_bytes());
                buf[p + 2..p + 4].copy_from_slice(&ty.to_le_bytes());
                buf[p + 4..p + 8].copy_from_slice(&count.to_le_bytes());
                buf[p + 8..p + 12].copy_from_slice(&val);
            };
        write_entry(
            &mut tiff,
            0,
            0x010F,
            2,
            make.len() as u32,
            make_off.to_le_bytes(),
        );
        write_entry(
            &mut tiff,
            1,
            0x0110,
            2,
            model.len() as u32,
            model_off.to_le_bytes(),
        );
        write_entry(
            &mut tiff,
            2,
            0x0132,
            2,
            datetime.len() as u32,
            datetime_off.to_le_bytes(),
        );
        write_entry(&mut tiff, 3, 0x8769, 4, 1, exif_off.to_le_bytes());

        // EXIF entries
        let p = exif_entries_at;
        tiff[p..p + 2].copy_from_slice(&0x829Au16.to_le_bytes());
        tiff[p + 2..p + 4].copy_from_slice(&5u16.to_le_bytes()); // RATIONAL
        tiff[p + 4..p + 8].copy_from_slice(&1u32.to_le_bytes());
        tiff[p + 8..p + 12].copy_from_slice(&et_off.to_le_bytes());
        let p = exif_entries_at + 12;
        tiff[p..p + 2].copy_from_slice(&0x8827u16.to_le_bytes());
        tiff[p + 2..p + 4].copy_from_slice(&3u16.to_le_bytes()); // SHORT
        tiff[p + 4..p + 8].copy_from_slice(&1u32.to_le_bytes());
        tiff[p + 8..p + 10].copy_from_slice(&100u16.to_le_bytes());

        // Wrap in JPEG: SOI + APP1(Exif\0\0 + tiff) + SOS-shaped sentinel
        // (parser stops at SOS, never reads further).
        let mut jpeg = Vec::new();
        jpeg.extend_from_slice(&[0xFF, 0xD8]); // SOI
        jpeg.extend_from_slice(&[0xFF, 0xE1]);
        let app1_body_len = (6 + tiff.len() + 2) as u16; // "Exif\0\0" + tiff + 2-byte length-includes-itself
        jpeg.extend_from_slice(&app1_body_len.to_be_bytes());
        jpeg.extend_from_slice(b"Exif\0\0");
        jpeg.extend_from_slice(&tiff);
        // SOS marker so the loop terminates cleanly
        jpeg.extend_from_slice(&[0xFF, 0xDA, 0x00, 0x02]);
        jpeg
    }

    #[test]
    fn jpeg_parser_extracts_ifd0_and_exif_subifd() {
        let jpeg = synth_jpeg_with_exif();
        let exif = parse_jpeg_exif(&jpeg).expect("parser returned None");
        assert_eq!(exif.make.as_deref(), Some("SIGMA"));
        assert_eq!(exif.model.as_deref(), Some("Quattro test"));
        assert_eq!(exif.datetime.as_deref(), Some("2025:01:02 03:04:05"));
        assert_eq!(exif.exposure_time, Some((1, 100)));
        assert_eq!(exif.iso, Some(100));
    }

    #[test]
    fn jpeg_parser_returns_none_without_app1() {
        // Bare SOI + SOS, no APP1.
        let jpeg = vec![0xFF, 0xD8, 0xFF, 0xDA, 0x00, 0x02];
        assert!(parse_jpeg_exif(&jpeg).is_none());
    }

    #[test]
    fn jpeg_parser_rejects_non_jpeg_input() {
        assert!(parse_jpeg_exif(b"not a jpeg").is_none());
        assert!(parse_jpeg_exif(&[]).is_none());
    }
}
