//! Pure-Rust DNG output writer. Replaces `x3f_output_dng.c`.
//!
//! Output structure (matches the legacy C):
//!
//! ```text
//! TIFF header (II + magic + IFD0-offset)
//! Preview strip bytes               (8-bit RGB, downsampled to 300 px wide)
//! Raw strip bytes                    (16-bit RGB, full resolution; optional zlib)
//! ExtraCameraProfiles blob           (concatenated MMCR mini-TIFFs)
//! IFD1 (raw) — external values + body
//! IFD0 (preview) — external values + body
//! ```
//!
//! The header IFD0-offset slot is patched at the very end. IFD0 has a
//! `SubIFDs` tag pointing at IFD1 (the raw IFD); the standard `next_ifd`
//! chain is left at 0 (DNG readers prefer the SubIFD layout).
//!
//! What we deliberately do NOT emit (matches legacy DNG output behaviour):
//! - `OpcodeList2` (spatial-gain GainMap). The legacy `write_spatial_gain`
//!   function exists but is unreferenced; the comment cites a "double-
//!   application of sg" risk in readers that honour the opcode.
//! - `LinearizationTable`. The image is already linear.
//! - DNG raw compression formats (LJPEG-92, JXL). We support uncompressed
//!   and Adobe Deflate, matching the legacy C.

mod exif;
mod hue_sat_map;
mod metadata;
mod opcodes;
mod profiles;
mod strip;
pub(crate) mod tags;
pub(crate) mod tiff_writer;
mod tone_curves;

use std::ffi::CString;
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;

use crate::{Error, Image, ProcessOptions, Reader};

use metadata::{vec3_invert, vec3_to_f32};
use profiles::{build_extra_profiles_blob, srational_from_floats, write_default_profile};
use tiff_writer::{DirectoryWriter, TiffWriter, Value};

/// White-balance preset used to compute `CameraCalibration1`. Matches the
/// `WB_D65` define in `x3f_output_dng.c`.
const WB_CALIBRATION: &str = "Overcast";

const ROWS_PER_STRIP: u32 = 32;
const PREVIEW_MAX_WIDTH: u32 = 300;

/// Write `reader`'s processed image to `path` as a DNG file.
///
/// `opts` is honoured the same way the legacy CLI honoured its DNG flags:
/// `compress` enables Adobe Deflate + horizontal predictor on the raw
/// plane; `wb` overrides the file's recorded white balance; `apply_sgain`,
/// `fix_bad`, `denoise` flow through to image processing.
pub fn write(reader: &Reader, path: impl AsRef<Path>, opts: &ProcessOptions) -> Result<(), Error> {
    let path = path.as_ref();

    // Resolve white balance up front — used both for image processing and
    // for the matrix tags.
    let mut opts = opts.clone();
    if opts.wb.is_none() {
        opts.wb = Some(reader.dng_default_wb());
    }
    // x3f_get_image is called with NONE encoding and crop=0 for DNG. We
    // always emit the full uncropped sensor frame and let Lightroom/etc.
    // honour the embedded ActiveArea + DefaultUserCrop tags. This matches
    // the legacy C path's hardcoded `x3f_get_image(..., NONE, 0, ...)`.
    opts.color_encoding = crate::ColorEncoding::None;
    opts.crop = false;
    // `cineon` is a TIFF-only modifier. The CLI rejects `-cineon -dng`
    // up front, but library callers can pass the same `ProcessOptions`
    // to `dump_dng` and `dump_tiff`; force it off here so the DNG path
    // never accidentally skips `apply_highlight_clip_dng` or applies a
    // log curve to the DNG raw plane.
    opts.cineon = false;
    let wb = opts.wb.clone().unwrap_or_default();

    let image = reader.get_image(&opts)?;
    if image.channels != 3 {
        return Err(Error::Library(crate::LibraryError::Argument));
    }

    // The legacy CPP and earlier Rust ports wrote the full raw frame
    // (including the masked-pixel border) and marked the usable region
    // with ActiveArea. RawTherapee 5.12 has a Sigma-DNG hot path
    // (rtengine/rawimage.cc:1276) that indexes into its decoded `image`
    // buffer with stride `raw_width` while the buffer was actually filled
    // at stride `iwidth = ImageWidth-left_margin`; when raw_width >
    // iwidth (i.e. ActiveArea trims columns) the read runs off the end of
    // the heap allocation and crashes. Cropping the raster to the active
    // area before writing — and emitting `ActiveArea = (0,0,h,w)` — keeps
    // raw_width == iwidth in RT and removes the crash.
    let crop = active_area_crop(reader, &image);
    let (out_rows, out_cols) = crop
        .map(|(_, _, r, c)| (r, c))
        .unwrap_or((image.rows, image.columns));

    let preview = reader.get_preview(&image, &opts, PREVIEW_MAX_WIDTH)?;
    let preview_bytes = strip::encode_preview_strip(&preview);
    let raw_strips =
        strip::encode_strips(&image, ROWS_PER_STRIP, opts.compress, crop).map_err(|source| {
            Error::Io {
                path: path.display().to_string(),
                source,
            }
        })?;

    // The DNG highlight-recovery scale is captured on `image` itself
    // (snapshotted on the rendering thread immediately after
    // `apply_highlight_clip_dng` set it). Reading the FFI side-channel
    // here would race with rayon work-stealing in batch mode — see
    // `Image::dng_highlight_scale`.
    let highlight_scale = image.dng_highlight_scale;

    let f = BufWriter::new(File::create(path).map_err(|source| Error::Io {
        path: path.display().to_string(),
        source,
    })?);
    let mut tiff = TiffWriter::new(f).map_err(io_err(path))?;

    // -- Preview strip data ------------------------------------------
    let preview_strip_offset = tiff.write_data(&preview_bytes).map_err(io_err(path))?;
    let preview_strip_bytes = preview_bytes.len() as u32;

    // -- Raw strip data (one or many, depending on compression) ------
    let mut raw_strip_offsets: Vec<u32> = Vec::with_capacity(raw_strips.len());
    let mut raw_strip_byte_counts: Vec<u32> = Vec::with_capacity(raw_strips.len());
    for s in &raw_strips {
        let off = tiff.write_data(&s.bytes).map_err(io_err(path))?;
        raw_strip_offsets.push(off);
        raw_strip_byte_counts.push(s.bytes.len() as u32);
    }

    // -- ExtraCameraProfiles blob (default profile is written into IFD0
    //    directly; everything else goes here as MMCR-prefixed mini-TIFFs).
    //    We write the blob ahead of IFD1 so its offsets are known when
    //    we build IFD0.
    let default_idx = profiles::default_profile_index(reader);
    let (extra_blob, extra_rel_offsets) = build_extra_profiles_blob(reader, &wb, default_idx)
        .ok_or(Error::Library(crate::LibraryError::Argument))?;
    let extra_offsets_abs: Vec<u32> = if extra_blob.is_empty() {
        Vec::new()
    } else {
        let base = tiff.write_data(&extra_blob).map_err(io_err(path))?;
        extra_rel_offsets.iter().map(|o| base + o).collect()
    };

    // Capture metadata up front — its `orientation` flows into both the
    // raw IFD and the preview IFD, and the rest goes into the EXIF sub-
    // IFD. Reading is read-only against the parsed file, so this is
    // cheap to compute once and use everywhere.
    let capture_meta = exif::CaptureMetadata::from_reader(reader);
    let orientation = capture_meta.orientation.unwrap_or(1);

    // -- OpcodeList3 lookup (optional, --opcodes-dir) -----------------
    let opcode_blob = opts
        .opcodes_dir
        .as_deref()
        .and_then(|dir| opcodes::load_for(&capture_meta, dir));

    // -- Build IFD1 (raw) first so we know its offset for IFD0's SubIFDs.
    let mut ifd1 = DirectoryWriter::new();
    populate_raw_ifd(
        reader,
        &image,
        (out_rows, out_cols),
        &mut ifd1,
        &raw_strip_offsets,
        &raw_strip_byte_counts,
        opts.compress,
    )?;
    if let Some(blob) = opcode_blob {
        ifd1.add(tags::OPCODE_LIST3, Value::Undefined(blob));
    }
    let ifd1_offset = ifd1.build(&mut tiff).map_err(io_err(path))?;

    // -- EXIF sub-IFD (linked from IFD0 via EXIF_IFD_POINTER) ---------
    //    Built before IFD0 so its offset is known when we populate IFD0.
    let exif_subifd_offset = if let Some(sub) = exif::build_subifd(&capture_meta) {
        Some(sub.build(&mut tiff).map_err(io_err(path))?)
    } else {
        None
    };

    // -- Build IFD0 (preview + DNG metadata + camera profiles).
    let mut ifd0 = DirectoryWriter::new();
    populate_preview_ifd(
        &preview,
        preview_strip_offset,
        preview_strip_bytes,
        &mut ifd0,
        opts.compress,
        orientation,
    );
    exif::add_top_level_tags(&capture_meta, &mut ifd0);
    ifd0.add(tags::SUB_IFDS, Value::Long(vec![ifd1_offset]));
    if let Some(off) = exif_subifd_offset {
        ifd0.add(tags::EXIF_IFD_POINTER, Value::Long(vec![off]));
    }
    add_dng_top_level_tags(reader, &capture_meta, &wb, highlight_scale, &mut ifd0)?;
    if write_default_profile(reader, &wb, &mut ifd0).is_none() {
        return Err(Error::Library(crate::LibraryError::Argument));
    }
    if !extra_offsets_abs.is_empty() {
        ifd0.add(tags::EXTRA_CAMERA_PROFILES, Value::Long(extra_offsets_abs));
    }
    let ifd0_offset = ifd0.build(&mut tiff).map_err(io_err(path))?;

    let mut inner = tiff.finalize(ifd0_offset).map_err(io_err(path))?;
    inner.flush().map_err(io_err(path))?;
    Ok(())
}

fn io_err(path: &Path) -> impl Fn(io::Error) -> Error + '_ {
    move |source| Error::Io {
        path: path.display().to_string(),
        source,
    }
}

fn populate_preview_ifd(
    preview: &crate::Preview,
    strip_offset: u32,
    strip_bytes: u32,
    ifd: &mut DirectoryWriter,
    compress: bool,
    orientation: u16,
) {
    ifd.add(
        tags::NEW_SUBFILE_TYPE,
        Value::Long(vec![tags::SUBFILETYPE_REDUCED_IMAGE]),
    );
    ifd.add(tags::IMAGE_WIDTH, Value::Long(vec![preview.columns]));
    ifd.add(tags::IMAGE_LENGTH, Value::Long(vec![preview.rows]));
    ifd.add(
        tags::BITS_PER_SAMPLE,
        Value::Short(vec![8; preview.channels as usize]),
    );
    ifd.add(
        tags::COMPRESSION,
        Value::Short(vec![tags::COMPRESSION_NONE]),
    );
    ifd.add(
        tags::PHOTOMETRIC_INTERPRETATION,
        Value::Short(vec![tags::PHOTOMETRIC_RGB]),
    );
    ifd.add(tags::STRIP_OFFSETS, Value::Long(vec![strip_offset]));
    ifd.add(tags::ORIENTATION, Value::Short(vec![orientation]));
    ifd.add(
        tags::SAMPLES_PER_PIXEL,
        Value::Short(vec![preview.channels as u16]),
    );
    ifd.add(tags::ROWS_PER_STRIP, Value::Long(vec![preview.rows]));
    ifd.add(tags::STRIP_BYTE_COUNTS, Value::Long(vec![strip_bytes]));
    ifd.add(
        tags::PLANAR_CONFIGURATION,
        Value::Short(vec![tags::PLANAR_CONFIG_CONTIG]),
    );
    let backward = if compress {
        tags::DNG_VERSION_1_4_0_0
    } else {
        tags::DNG_VERSION_1_3_0_0
    };
    ifd.add(
        tags::DNG_VERSION,
        Value::Byte(tags::DNG_VERSION_1_4_0_0.to_vec()),
    );
    ifd.add(tags::DNG_BACKWARD_VERSION, Value::Byte(backward.to_vec()));
}

fn populate_raw_ifd(
    reader: &Reader,
    image: &Image,
    out_dims: (u32, u32),
    ifd: &mut DirectoryWriter,
    strip_offsets: &[u32],
    strip_byte_counts: &[u32],
    compress: bool,
) -> Result<(), Error> {
    let (out_rows, out_cols) = out_dims;
    // Orientation lives only in IFD0 (the preview IFD); the legacy C code
    // does not duplicate it onto the raw SubIFD, and Capture One renders
    // the raw plane completely black when it sees Orientation here even
    // with the default top-left value.
    ifd.add(tags::NEW_SUBFILE_TYPE, Value::Long(vec![0]));
    ifd.add(tags::IMAGE_WIDTH, Value::Long(vec![out_cols]));
    ifd.add(tags::IMAGE_LENGTH, Value::Long(vec![out_rows]));
    ifd.add(tags::BITS_PER_SAMPLE, Value::Short(vec![16, 16, 16]));
    let comp = if compress {
        tags::COMPRESSION_ADOBE_DEFLATE
    } else {
        tags::COMPRESSION_NONE
    };
    ifd.add(tags::COMPRESSION, Value::Short(vec![comp]));
    ifd.add(
        tags::PHOTOMETRIC_INTERPRETATION,
        Value::Short(vec![tags::PHOTOMETRIC_LINEAR_RAW]),
    );
    ifd.add(tags::STRIP_OFFSETS, Value::Long(strip_offsets.to_vec()));
    ifd.add(tags::SAMPLES_PER_PIXEL, Value::Short(vec![3]));
    ifd.add(tags::ROWS_PER_STRIP, Value::Long(vec![ROWS_PER_STRIP]));
    ifd.add(
        tags::STRIP_BYTE_COUNTS,
        Value::Long(strip_byte_counts.to_vec()),
    );
    ifd.add(
        tags::PLANAR_CONFIGURATION,
        Value::Short(vec![tags::PLANAR_CONFIG_CONTIG]),
    );
    if compress {
        ifd.add(tags::PREDICTOR, Value::Short(vec![2]));
    }

    // Per-channel black/white levels.
    // BlackLevel must be SHORT, LONG, or (unsigned) RATIONAL per the DNG
    // 1.6 spec — Capture One rejects SRATIONAL here and renders the raw
    // plane black when it can't read the level.
    ifd.add(tags::BLACK_LEVEL_REPEAT_DIM, Value::Short(vec![1, 1]));
    let denom: u32 = 10_000;
    ifd.add(
        tags::BLACK_LEVEL,
        Value::Rational(
            image
                .levels
                .black
                .iter()
                .map(|&b| (((b as f64) * denom as f64).round().max(0.0) as u32, denom))
                .collect(),
        ),
    );
    ifd.add(tags::WHITE_LEVEL, Value::Long(image.levels.white.to_vec()));

    // Adobe DNG hints. ChromaBlurRadius=0 prevents downstream chroma
    // denoise that would mush our highlight-recovery work; the rest are
    // there because removing them caused subtle Lightroom misbehaviour
    // in earlier C-side experiments (per the legacy code's choice list).
    ifd.add(tags::CHROMA_BLUR_RADIUS, Value::Rational(vec![(0, 1)]));
    ifd.add(tags::CFA_PLANE_COLOR, Value::Byte(vec![0, 1, 2]));
    ifd.add(tags::DEFAULT_SCALE, Value::Rational(vec![(1, 1), (1, 1)]));
    ifd.add(tags::LINEAR_RESPONSE_LIMIT, Value::Rational(vec![(1, 1)]));
    ifd.add(tags::ANTI_ALIAS_STRENGTH, Value::Rational(vec![(0, 1)]));

    if let (Some(make), Some(model)) = (reader.dng_camf_text("Make"), reader.dng_camf_text("Model"))
    {
        let unique = format!("{make} {model}");
        if let Ok(c) = CString::new(unique) {
            ifd.add(tags::UNIQUE_CAMERA_MODEL, Value::Ascii(c));
        }
    }

    if let Some(active) = reader.dng_active_area(image) {
        // The raster we just wrote IS the active area (see the call site
        // in `write` for the why), so ActiveArea is the whole frame.
        // DefaultUserCrop is in normalised active-area coordinates, so it
        // doesn't depend on whether we cropped or not — it still uses the
        // original (top, left, bottom, right) rectangle.
        let normalised = [0, 0, out_rows, out_cols];
        ifd.add(tags::ACTIVE_AREA, Value::Long(normalised.to_vec()));
        if let Some(crop) = compute_default_user_crop(reader, &active) {
            ifd.add(tags::DEFAULT_USER_CROP, Value::Float(crop.to_vec()));
        }
    }

    Ok(())
}

/// Resolve the active-area sub-rectangle (`(top, left, rows, cols)`) we
/// want to crop to before writing the raw IFD. Returns `None` when the
/// file has no `ActiveImageArea` CAMF entry, when the rectangle is
/// degenerate (zero size), or when it doesn't fit inside `image` —
/// callers fall back to writing the full frame.
fn active_area_crop(reader: &Reader, image: &Image) -> Option<strip::CropWindow> {
    let active = reader.dng_active_area(image)?;
    let [top, left, bottom, right] = active;
    if bottom <= top || right <= left || bottom > image.rows || right > image.columns {
        return None;
    }
    Some((top, left, bottom - top, right - left))
}

fn add_dng_top_level_tags(
    reader: &Reader,
    capture_meta: &exif::CaptureMetadata,
    wb: &str,
    highlight_scale: f64,
    ifd: &mut DirectoryWriter,
) -> Result<(), Error> {
    // BaselineExposure = log2(capture_iso/sensor_iso) + log2(highlight_scale).
    if let (Some(sensor), Some(capture)) = (
        reader.dng_camf_float("SensorISO"),
        reader.dng_camf_float("CaptureISO"),
    ) {
        let mut be = (capture / sensor).log2();
        if highlight_scale > 1.0 {
            be += highlight_scale.log2();
        }
        ifd.add(
            tags::BASELINE_EXPOSURE,
            srational_from_floats(&[be as f32], 10_000),
        );
    }

    // AsShotNeutral = 1 / gain at the file's WB.
    let gain = reader
        .dng_gain(Some(wb))
        .ok_or(Error::Library(crate::LibraryError::Argument))?;
    let neutral = vec3_to_f32(&vec3_invert(&gain));
    ifd.add(
        tags::AS_SHOT_NEUTRAL,
        Value::Rational(
            neutral
                .iter()
                .map(|&v| (((v as f64) * 10_000.0).round().max(0.0) as u32, 10_000_u32))
                .collect(),
        ),
    );

    // CameraCalibration1: diag(1 / gain at D65). Matches C exactly.
    let gain_d65 = reader
        .dng_gain(Some(WB_CALIBRATION))
        .ok_or(Error::Library(crate::LibraryError::Argument))?;
    let inv_d65 = vec3_invert(&gain_d65);
    let diag = metadata::mat3_diag(&inv_d65);
    ifd.add(
        tags::CAMERA_CALIBRATION1,
        srational_from_floats(&metadata::mat3_to_f32(&diag), 10_000),
    );
    // CalibrationIlluminant1 is per-camera-line. For Merrill and
    // SD-series bodies we omit it entirely (matches the pre-Rust C
    // writer; Capture One renders with green cast when CI1 is set on
    // these matrices because they aren't actually D65-calibrated).
    // For the Quattro line, we explicitly label as WhiteFluorescent
    // (15) — Sigma's own native Quattro DNGs ship a ForwardMatrix2
    // under CalibrationIlluminant2=15 that's byte-equivalent to what
    // we emit, and Capture One renders Quattro DNGs correctly when
    // we adopt that label. SD Quattro H groups with the Quattro line
    // for consistency; renderers with built-in profiles for newer
    // bodies (Apple, C1's SDQH path) override our label anyway.
    if let Some(model) = capture_meta.model.as_deref() {
        if model.to_ascii_lowercase().contains("quattro") {
            ifd.add(
                tags::CALIBRATION_ILLUMINANT1,
                Value::Short(vec![tags::CALIB_ILLUMINANT_WHITE_FLUORESCENT]),
            );
        }
    }

    Ok(())
}

/// Replicates the JPEG-thumbnail-aspect-driven `DefaultUserCrop` logic
/// from `src/x3f_output_dng.c:516`. Returns
/// `[top, left, bottom, right]` as fractions of the active area.
///
/// Returns `None` when there's no embedded JPEG to derive the aspect from
/// — in that case the C path silently omits the tag and so do we.
fn compute_default_user_crop(reader: &Reader, active_area: &[u32; 4]) -> Option<[f32; 4]> {
    let (jpeg_w, jpeg_h) = reader.dng_thumb_jpeg_dims()?;
    if jpeg_w == 0 || jpeg_h == 0 {
        return None;
    }
    let active_w = (active_area[3] - active_area[1]) as f32;
    let active_h = (active_area[2] - active_area[0]) as f32;
    let jpeg_aspect = jpeg_w as f32 / jpeg_h as f32;

    let (crop_w, crop_h) = if jpeg_aspect > active_w / active_h {
        let w = active_w;
        let h = w / jpeg_aspect;
        (w, h)
    } else {
        let h = active_h;
        let w = h * jpeg_aspect;
        (w, h)
    };
    let crop_x = (active_w - crop_w) / 2.0;
    let crop_y = (active_h - crop_h) / 2.0;
    Some([
        crop_y / active_h,
        crop_x / active_w,
        (crop_y + crop_h) / active_h,
        (crop_x + crop_w) / active_w,
    ])
}

impl Reader {
    /// Width and height of the embedded JPEG thumbnail. Used by DNG to
    /// compute `DefaultUserCrop`.
    pub(crate) fn dng_thumb_jpeg_dims(&self) -> Option<(u32, u32)> {
        // SAFETY: x3f is valid for self's lifetime.
        let de = unsafe { x3f_sys::x3f_get_thumb_jpeg(self.x3f.as_ptr()) };
        if de.is_null() {
            return None;
        }
        // Loading is required for the dimensions to be populated.
        // SAFETY: de is non-null.
        let r = unsafe { x3f_sys::x3f_load_data(self.x3f.as_ptr(), de) };
        if r != x3f_sys::x3f_return_e_X3F_OK {
            return None;
        }
        // SAFETY: de is non-null and now loaded.
        let img = unsafe { (*de).header.data_subsection.image_data };
        if img.columns == 0 || img.rows == 0 {
            return None;
        }
        Some((img.columns, img.rows))
    }

    /// Raw bytes of the embedded JPEG thumbnail. Returns `None` if the
    /// file has no JPEG section (rare) or the load failed. The bytes are
    /// copied into Rust ownership; the underlying buffer is small (~150
    /// KB for Merrill, ~7 MB for Quattro) so this isn't worth borrowing.
    pub(crate) fn dng_thumb_jpeg_bytes(&self) -> Option<Vec<u8>> {
        // SAFETY: x3f is valid for self's lifetime.
        let de = unsafe { x3f_sys::x3f_get_thumb_jpeg(self.x3f.as_ptr()) };
        if de.is_null() {
            return None;
        }
        // SAFETY: de is non-null.
        let r = unsafe { x3f_sys::x3f_load_data(self.x3f.as_ptr(), de) };
        if r != x3f_sys::x3f_return_e_X3F_OK {
            return None;
        }
        // SAFETY: de is non-null and loaded; .image_data.data points at
        // the JPEG payload of length (*de).input.size.
        let (data, size) = unsafe {
            (
                (*de).header.data_subsection.image_data.data,
                (*de).input.size as usize,
            )
        };
        if data.is_null() || size == 0 {
            return None;
        }
        // SAFETY: data + size are owned by the parsed file and stay live
        // for self's lifetime; we copy out to an owned Vec.
        Some(unsafe { std::slice::from_raw_parts(data as *const u8, size) }.to_vec())
    }
}
