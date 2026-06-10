//! TIFF + DNG tag-ID constants used by the DNG writer.
//!
//! The TIFF baseline tags come from libtiff's `tiff.h`; the DNG-specific
//! tags come from Adobe's DNG 1.6 spec and were transcribed in
//! `src/x3f_dngtags.h`. Tag values that aren't referenced by the writer are
//! omitted; add as needed.

#![allow(dead_code)]

// --- TIFF baseline -----------------------------------------------------

pub const NEW_SUBFILE_TYPE: u16 = 254;
pub const IMAGE_WIDTH: u16 = 256;
pub const IMAGE_LENGTH: u16 = 257;
pub const BITS_PER_SAMPLE: u16 = 258;
pub const COMPRESSION: u16 = 259;
pub const PHOTOMETRIC_INTERPRETATION: u16 = 262;
pub const MAKE: u16 = 271;
pub const MODEL: u16 = 272;
pub const STRIP_OFFSETS: u16 = 273;
pub const ORIENTATION: u16 = 274;
pub const SAMPLES_PER_PIXEL: u16 = 277;
pub const ROWS_PER_STRIP: u16 = 278;
pub const STRIP_BYTE_COUNTS: u16 = 279;
pub const PLANAR_CONFIGURATION: u16 = 284;
pub const SOFTWARE: u16 = 305;
pub const DATETIME: u16 = 306;
pub const SUB_IFDS: u16 = 330;

// --- EXIF sub-IFD pointer (TIFF private tag) ---------------------------

pub const EXIF_IFD_POINTER: u16 = 34665;

// --- EXIF tags (sub-IFD only) ------------------------------------------

pub const EXIF_EXPOSURE_TIME: u16 = 33434;
pub const EXIF_F_NUMBER: u16 = 33437;
pub const EXIF_EXPOSURE_PROGRAM: u16 = 34850;
pub const EXIF_ISO_SPEED_RATINGS: u16 = 34855;
pub const EXIF_DATE_TIME_ORIGINAL: u16 = 36867;
pub const EXIF_DATE_TIME_DIGITIZED: u16 = 36868;
pub const EXIF_EXPOSURE_BIAS_VALUE: u16 = 37380;
pub const EXIF_FLASH: u16 = 37385;
pub const EXIF_FOCAL_LENGTH: u16 = 37386;
pub const EXIF_FOCAL_LENGTH_IN_35MM: u16 = 41989;
pub const EXIF_BODY_SERIAL_NUMBER: u16 = 42033;

// --- DNG-specific (from x3f_dngtags.h) ---------------------------------

pub const CFA_PLANE_COLOR: u16 = 50710;
pub const CFA_LAYOUT: u16 = 50711;
pub const BLACK_LEVEL_REPEAT_DIM: u16 = 50713;
pub const BLACK_LEVEL: u16 = 50714;
pub const WHITE_LEVEL: u16 = 50717;
pub const DEFAULT_SCALE: u16 = 50718;
pub const DEFAULT_CROP_ORIGIN: u16 = 50719;
pub const DEFAULT_CROP_SIZE: u16 = 50720;
pub const COLOR_MATRIX1: u16 = 50721;
pub const COLOR_MATRIX2: u16 = 50722;
pub const CAMERA_CALIBRATION1: u16 = 50723;
pub const CAMERA_CALIBRATION2: u16 = 50724;
pub const ANALOG_BALANCE: u16 = 50727;
pub const AS_SHOT_NEUTRAL: u16 = 50728;
pub const BASELINE_EXPOSURE: u16 = 50730;
pub const LINEAR_RESPONSE_LIMIT: u16 = 50734;
pub const ANTI_ALIAS_STRENGTH: u16 = 50738;
pub const DNG_PRIVATE_DATA: u16 = 50740;
pub const CALIBRATION_ILLUMINANT1: u16 = 50778;
pub const CALIBRATION_ILLUMINANT2: u16 = 50779;
pub const BEST_QUALITY_SCALE: u16 = 50780;
pub const ACTIVE_AREA: u16 = 50829;
pub const CHROMA_BLUR_RADIUS: u16 = 50737;
pub const UNIQUE_CAMERA_MODEL: u16 = 50708;
pub const DNG_VERSION: u16 = 50706;
pub const DNG_BACKWARD_VERSION: u16 = 50707;
pub const EXTRA_CAMERA_PROFILES: u16 = 50933;
pub const AS_SHOT_PROFILE_NAME: u16 = 50934;
pub const PROFILE_NAME: u16 = 50936;
pub const PROFILE_HUE_SAT_MAP_DIMS: u16 = 50937;
/// First-illuminant hue/sat map. Adobe's DNG SDK pairs this with
/// `CalibrationIlluminant1` / `ColorMatrix1`. Profiles with only one
/// calibration illuminant must use *Data1*; `Data2` (50939) is reserved
/// for the second illuminant and is silently ignored when no
/// `ColorMatrix2` is present.
pub const PROFILE_HUE_SAT_MAP_DATA1: u16 = 50938;
pub const PROFILE_TONE_CURVE: u16 = 50940;
pub const FORWARD_MATRIX1: u16 = 50964;
pub const PROFILE_HUE_SAT_MAP_ENCODING: u16 = 51107;
pub const FORWARD_MATRIX2: u16 = 50965;
pub const OPCODE_LIST1: u16 = 51008;
pub const OPCODE_LIST2: u16 = 51009;
pub const OPCODE_LIST3: u16 = 51022;
pub const DEFAULT_BLACK_RENDER: u16 = 51110;
pub const DEFAULT_USER_CROP: u16 = 51125;

// --- TIFF Photometric values -------------------------------------------

pub const PHOTOMETRIC_RGB: u16 = 2;
pub const PHOTOMETRIC_LINEAR_RAW: u16 = 34892;

// --- Compression values ------------------------------------------------

pub const COMPRESSION_NONE: u16 = 1;
/// "New-style" JPEG (TIFF 6.0 §22); in DNG this means lossless JPEG
/// (ITU-T T.81 process 14) — the only compression the spec allows for
/// 16-bit integer raw data.
pub const COMPRESSION_LOSSLESS_JPEG: u16 = 7;

// --- Orientation values ------------------------------------------------

pub const ORIENTATION_TOP_LEFT: u16 = 1;

// --- Planar configuration values ---------------------------------------

pub const PLANAR_CONFIG_CONTIG: u16 = 1;

// --- New subfile type bits ---------------------------------------------

pub const SUBFILETYPE_REDUCED_IMAGE: u32 = 1;

// --- DNG version bytes -------------------------------------------------

pub const DNG_VERSION_1_4_0_0: [u8; 4] = [1, 4, 0, 0];
pub const DNG_VERSION_1_3_0_0: [u8; 4] = [1, 3, 0, 0];

// --- CalibrationIlluminant values --------------------------------------

pub const CALIB_ILLUMINANT_D65: u16 = 21;
pub const CALIB_ILLUMINANT_D50: u16 = 23;
/// TIFF/EXIF code 15 — "white fluorescent (WW 3250–3800 K)".
/// Sigma's native Quattro DNGs use this as `CalibrationIlluminant2` and
/// the `ForwardMatrix2` shipped under that label is byte-equivalent (to
/// 4 decimal places) to what we emit for our Quattro picture profiles.
pub const CALIB_ILLUMINANT_WHITE_FLUORESCENT: u16 = 15;
/// TIFF/EXIF code 17 — Standard Light A (≈2856 K). Sigma's native Quattro
/// DNGs use this as `CalibrationIlluminant1`.
pub const CALIB_ILLUMINANT_STANDARD_A: u16 = 17;
