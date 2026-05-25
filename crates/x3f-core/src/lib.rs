//! Safe Rust API for reading and converting Sigma Foveon X3F raw images.
//!
//! The decoding and processing pipeline is native Rust over the low-level
//! [`x3f_sys`] layer; the only non-Rust code is an optional OpenCV-backed
//! denoise pass.
//!
//! The headline type is [`Reader`], which owns the C `x3f_t` and the input
//! `FILE*`. Open a file, optionally load extra sections (CAMF/property
//! list/JPEG thumbnail/RAW), and then dispatch to one of the `dump_*` methods
//! to produce an output file.
//!
//! # Example
//!
//! ```no_run
//! use x3f_core::{Reader, ProcessOptions};
//! let mut r = Reader::open("input.X3F").unwrap();
//! r.load_camf().unwrap();
//! r.load_raw().unwrap();
//! r.dump_dng("out.dng", &ProcessOptions::default()).unwrap();
//! ```

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

use std::ffi::CString;
// `OsStrExt::as_bytes` lives in different platform modules — `std::os::unix`
// on Unix-likes, `std::os::wasi` on WASI. There is no Windows form, but
// the Windows port is gated separately upstream (legacy build). We only
// need it for `path_to_cstring`.
#[cfg(unix)]
use std::os::unix::ffi::OsStrExt;
#[cfg(target_os = "wasi")]
use std::os::wasi::ffi::OsStrExt;
use std::path::Path;
use std::ptr::NonNull;

use thiserror::Error;
use x3f_sys as sys;
// Shadow the external libc crate with x3f-sys's compat shim. On every
// target except wasm32-unknown-unknown this is `pub use libc::*;`; on
// wasm32 it provides Rust-native equivalents for the few libc symbols
// we touch here (FILE / fopen / fclose / free).
use x3f_sys::sysabi as libc;

mod globals;
mod icc;
mod image;
pub mod output;

pub use globals::{
    set_log_callback, set_max_printed_matrix_elements, set_offset_legacy, set_verbosity,
    LogCallback, Verbosity,
};
pub use image::{Image, ImageLevels, Preview};

/// Color space the processed RGB output is converted to.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum ColorEncoding {
    /// Pass-through; no scaling, no gamma, no color-space conversion.
    None,
    /// sRGB (default for processed output).
    #[default]
    Srgb,
    /// Adobe RGB (1998).
    AdobeRgb,
    /// ProPhoto RGB.
    ProPhotoRgb,
    /// Raw/unprocessed (for `-unprocessed`).
    Unprocessed,
    /// Quattro top-layer dump (for `-qtop`).
    Qtop,
}

impl ColorEncoding {
    fn to_raw(self) -> sys::x3f_color_encoding_t {
        match self {
            ColorEncoding::None => sys::x3f_color_encoding_e_NONE,
            ColorEncoding::Srgb => sys::x3f_color_encoding_e_SRGB,
            ColorEncoding::AdobeRgb => sys::x3f_color_encoding_e_ARGB,
            ColorEncoding::ProPhotoRgb => sys::x3f_color_encoding_e_PPRGB,
            ColorEncoding::Unprocessed => sys::x3f_color_encoding_e_UNPROCESSED,
            ColorEncoding::Qtop => sys::x3f_color_encoding_e_QTOP,
        }
    }
}

/// Per-conversion processing options. Mirrors the parameters of the legacy
/// `x3f_dump_*` C entry points.
#[derive(Debug, Clone)]
pub struct ProcessOptions {
    /// Target color space / processing mode for the output.
    pub color_encoding: ColorEncoding,
    /// Crop to the active sensor area (default: true).
    pub crop: bool,
    /// Run bad-pixel interpolation (default: true).
    pub fix_bad: bool,
    /// Apply spatial-gain (lens shading) correction. `None` defers to the
    /// per-camera default (off for Quattro, on for older sensors).
    pub apply_sgain: Option<bool>,
    /// White-balance preset name (e.g. "Daylight"). `None` uses the file's
    /// recorded WB.
    pub wb: Option<String>,
    /// ZIP-compress DNG/TIFF output.
    pub compress: bool,
    /// Run the OpenCV NLM denoise pass before color conversion. Backed by
    /// opencv-mobile's `fastNlMeansDenoising` on every target except
    /// `wasm32-unknown-unknown` (where the build falls back to a no-op stub
    /// because the WASM ecosystem here cannot link C++ stdlib).
    pub denoise: bool,
    /// Directory of pre-rendered DNG `OpcodeList3` blobs (Sigma's per-
    /// model / per-aperture flat-fielding gain maps). When set, the DNG
    /// writer looks up the blob matching the camera's model and aperture
    /// and embeds it into the raw IFD's `OpcodeList3` tag. Files are
    /// expected in the x3fuse layout: `<MODEL>[_<LENSID>]_FF_DNG_Opcodelist3_<APERTURE>`.
    /// `None` (default) skips opcode embedding.
    pub opcodes_dir: Option<std::path::PathBuf>,
    /// Enable the DNG-path Foveon highlight-recovery pipeline (chroma
    /// LUT + L*p reconstruction + matrix-pathology gate, with the
    /// recovered raster scaled to fit within `u16` and a matching
    /// `BaselineExposure` nudge so renderers can pull recovered
    /// highlights back via negative exposure compensation).
    ///
    /// **Renderer compatibility:** Adobe Camera Raw / Lightroom and
    /// RawTherapee/LibRaw honour the `BaselineExposure` log2 nudge and
    /// render these DNGs correctly with the recovered highlight chroma
    /// in place. Capture One and Apple RAW Engine do not — they cast
    /// green/blue on Merrill files when this is enabled. Default is
    /// `false` (matches the pre-Rust C writer's output, renders
    /// correctly across all four).
    pub dng_highlight_recovery: bool,
    /// Cineon-style log TIFF mode. When `true`, the conversion pipeline:
    ///
    ///   - replaces the encoding-specific gamma LUT with a Cineon-style log
    ///     curve `y = log(scale·x + 1) / log(scale + 1)` so shadows are
    ///     lifted, highlights are softly compressed, and midtones sit on a
    ///     gentle slope — the "flat for grading" look;
    ///   - bypasses the chroma-LUT and RepairPix highlight-recovery passes
    ///     (which bake creative interpretation into the pixels);
    ///   - keeps the matrix-pathology gate (it's a sanity rail against
    ///     truly broken Foveon highlights, not a creative pass);
    ///   - keeps WB, demosaic, denoise, spatial gain, EV scaling, and the
    ///     RGB matrix multiply.
    ///
    /// Only honoured by the TIFF writer; the writer embeds an ICC profile
    /// declaring the chosen primaries with a *linear* TRC so colour-
    /// managed readers (Lightroom, Capture One, DaVinci, Preview) leave
    /// the log encoding intact — the file looks flat in viewers, and the
    /// colourist applies their own log→linear input transform in the
    /// grading app. `scale` defaults to 100 (the empirical sweet spot
    /// between "flat for grading" and "shadow noise dominating"); set
    /// the `X3F_CINEON_SCALE` environment variable to override (12 ≈
    /// Cineon film-print, 50 ≈ moderate log; larger is flatter still at
    /// the cost of shadow noise).
    pub cineon: bool,
}

impl Default for ProcessOptions {
    fn default() -> Self {
        Self {
            color_encoding: ColorEncoding::default(),
            crop: true,
            fix_bad: true,
            apply_sgain: None,
            wb: None,
            compress: false,
            denoise: true,
            opcodes_dir: None,
            dng_highlight_recovery: false,
            cineon: false,
        }
    }
}

/// Errors returned by this crate.
#[derive(Debug, Error)]
pub enum Error {
    /// The input file could not be opened.
    #[error("could not open input file `{path}`: {source}")]
    OpenInput {
        /// Path that failed to open.
        path: String,
        /// Underlying OS error.
        source: std::io::Error,
    },
    /// A path contains an interior NUL byte and cannot be passed to C.
    #[error("path contains a NUL byte and cannot be passed to C")]
    PathNul,
    /// The white-balance preset name contains an interior NUL byte.
    #[error("white-balance preset contains a NUL byte")]
    WbNul,
    /// The file could not be parsed as an X3F container.
    #[error("could not parse X3F file (x3f_new_from_file returned NULL)")]
    Parse,
    /// The underlying library returned a non-OK status.
    #[error("x3f returned {0:?}")]
    Library(LibraryError),
    /// An I/O error occurred while writing the output file.
    #[error("IO error writing `{path}`: {source}")]
    Io {
        /// Output path being written.
        path: String,
        /// Underlying OS error.
        source: std::io::Error,
    },
}

/// Subset of `x3f_return_t` that maps to errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LibraryError {
    /// Invalid argument (`X3F_ARGUMENT_ERROR`).
    Argument,
    /// Input-file error (`X3F_INFILE_ERROR`).
    Infile,
    /// Output-file error (`X3F_OUTFILE_ERROR`).
    Outfile,
    /// Internal library error (`X3F_INTERNAL_ERROR`).
    Internal,
    /// An unrecognized non-OK status code.
    Unknown(u32),
}

impl LibraryError {
    fn from_raw(r: sys::x3f_return_t) -> Result<(), Self> {
        if r == sys::x3f_return_e_X3F_OK {
            return Ok(());
        }
        Err(match r {
            sys::x3f_return_e_X3F_ARGUMENT_ERROR => LibraryError::Argument,
            sys::x3f_return_e_X3F_INFILE_ERROR => LibraryError::Infile,
            sys::x3f_return_e_X3F_OUTFILE_ERROR => LibraryError::Outfile,
            sys::x3f_return_e_X3F_INTERNAL_ERROR => LibraryError::Internal,
            other => LibraryError::Unknown(other),
        })
    }
}

fn check(r: sys::x3f_return_t) -> Result<(), Error> {
    LibraryError::from_raw(r).map_err(Error::Library)
}

fn path_to_cstring(path: &Path) -> Result<CString, Error> {
    // On Unix and WASI, `OsStrExt::as_bytes()` returns the platform's
    // raw bytes. On `wasm32-unknown-unknown` neither cfg fires; we fall
    // back to `as_encoded_bytes()`, which is stable since 1.74 and
    // returns a UTF-8-ish byte representation suitable for `CString`.
    // The wasm32-unknown-unknown path can't actually open files at
    // runtime (the `fopen` shim returns NULL — see `sysabi.rs`), so
    // this is essentially documentation that the API surface compiles.
    #[cfg(any(unix, target_os = "wasi"))]
    let bytes = path.as_os_str().as_bytes();
    #[cfg(not(any(unix, target_os = "wasi")))]
    let bytes = path.as_os_str().as_encoded_bytes();
    CString::new(bytes).map_err(|_| Error::PathNul)
}

/// Owns a parsed X3F file and the underlying C `FILE*`.
pub struct Reader {
    file: NonNull<libc::FILE>,
    x3f: NonNull<sys::x3f_t>,
    /// When constructed via [`Reader::from_bytes`], holds the heap
    /// copy of the input buffer that the FILE* reads through. The
    /// field is dropped after `x3f_delete` + `fclose` in [`Drop`],
    /// so freeing order is FILE* first, then backing bytes. None
    /// for the path-based [`Reader::open`].
    _backing: Option<Vec<u8>>,
}

// The C library does not document thread-safety guarantees, but a Reader is
// effectively a unique handle: there is no shared interior state once a file
// is opened. We allow Send but not Sync.
unsafe impl Send for Reader {}

impl Reader {
    /// Parse an X3F file from an in-memory byte buffer. Available on
    /// every Unix-like host (via libc's `fmemopen`) and on `wasm32-*`
    /// (via x3f-sys's MemFile shim). Not available on Windows because
    /// MSVC's CRT has no `fmemopen` equivalent — `Reader::open` is
    /// the right path there.
    ///
    /// The buffer is *copied* into a heap-side cursor that the
    /// underlying parser reads through, so the input slice does not
    /// need to outlive the returned `Reader`. Use this entry point
    /// from environments without a host filesystem (browser WASM,
    /// JNI, Swift bindings receiving a `Data`).
    #[cfg(any(unix, target_arch = "wasm32"))]
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, Error> {
        let mode = c"rb";
        // SAFETY: `bytes` is valid for `bytes.len()` bytes; mode is a
        // valid NUL-terminated string. On host, `fmemopen` is libc's
        // (it borrows `bytes`); on wasm32 our sysabi shim copies the
        // buffer into a heap-owned MemFile so the FILE* outlives this
        // call. We mirror that behaviour on host by stashing a pinned
        // copy in `Reader._backing` below — that way both paths have
        // the same lifetime semantics, and callers don't have to
        // think about which target they're on.
        let owned: Vec<u8> = bytes.to_vec();
        // libc's `fmemopen` and our wasm shim's both take `size_t` (=
        // `usize` on every target we support); pass `owned.len()`
        // directly without a cast so the call typechecks under both.
        let file_ptr = unsafe {
            libc::fmemopen(
                owned.as_ptr() as *mut libc::c_void,
                owned.len(),
                mode.as_ptr(),
            )
        };
        let Some(file) = NonNull::new(file_ptr) else {
            return Err(Error::OpenInput {
                path: "<bytes>".to_string(),
                source: std::io::Error::other("fmemopen returned NULL"),
            });
        };
        // SAFETY: file is a valid FILE*; sys::FILE is layout-equivalent
        // to libc::FILE.
        let x3f_ptr = unsafe { sys::x3f_new_from_file(file.as_ptr() as *mut sys::FILE) };
        let Some(x3f) = NonNull::new(x3f_ptr) else {
            // SAFETY: file is the FILE* we just opened.
            unsafe { libc::fclose(file.as_ptr()) };
            return Err(Error::Parse);
        };
        Ok(Reader {
            file,
            x3f,
            _backing: Some(owned),
        })
    }

    /// Open and parse an X3F file's directory.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, Error> {
        let path = path.as_ref();
        let cpath = path_to_cstring(path)?;
        let mode = c"rb";
        // SAFETY: cpath and mode are valid C strings owned by this stack
        // frame. fopen returns NULL on failure, otherwise a fresh FILE* we
        // own.
        let file_ptr = unsafe { libc::fopen(cpath.as_ptr(), mode.as_ptr()) };
        let Some(file) = NonNull::new(file_ptr) else {
            return Err(Error::OpenInput {
                path: path.display().to_string(),
                source: std::io::Error::last_os_error(),
            });
        };
        // SAFETY: file is a valid open FILE* that x3f_new_from_file may read
        // from. The library does not take ownership; we keep the FILE* alive
        // for the lifetime of Reader. libc::FILE and bindgen's FILE are both
        // opaque aliases of the same platform stdio struct; the cast is a
        // pointer reinterpretation only.
        let x3f_ptr = unsafe { sys::x3f_new_from_file(file.as_ptr() as *mut sys::FILE) };
        let Some(x3f) = NonNull::new(x3f_ptr) else {
            // SAFETY: file is the FILE* we just opened.
            unsafe { libc::fclose(file.as_ptr()) };
            return Err(Error::Parse);
        };
        Ok(Reader {
            file,
            x3f,
            _backing: None,
        })
    }

    /// X3F format version recorded in the file header.
    pub fn header_version(&self) -> u32 {
        // SAFETY: x3f points to a valid x3f_t for the lifetime of self.
        unsafe { (*self.x3f.as_ptr()).header.version }
    }

    fn load_directory_entry(&mut self, de: *mut sys::x3f_directory_entry_t) -> Result<(), Error> {
        if de.is_null() {
            return Err(Error::Library(LibraryError::Argument));
        }
        // SAFETY: x3f and de are non-null and valid.
        let r = unsafe { sys::x3f_load_data(self.x3f.as_ptr(), de) };
        check(r)
    }

    /// Decode the embedded JPEG thumbnail into memory.
    pub fn load_thumbnail_jpeg(&mut self) -> Result<(), Error> {
        // SAFETY: self.x3f is valid for the lifetime of self.
        let de = unsafe { sys::x3f_get_thumb_jpeg(self.x3f.as_ptr()) };
        self.load_directory_entry(de)
    }

    /// Load the CAMF (camera firmware) metadata block.
    pub fn load_camf(&mut self) -> Result<(), Error> {
        // SAFETY: self.x3f is valid.
        let de = unsafe { sys::x3f_get_camf(self.x3f.as_ptr()) };
        self.load_directory_entry(de)
    }

    /// Load the property list. Some files (Quattro) do not have one; this
    /// returns Ok(()) silently in that case.
    pub fn load_property_list(&mut self) -> Result<(), Error> {
        // SAFETY: self.x3f is valid.
        let de = unsafe { sys::x3f_get_prop(self.x3f.as_ptr()) };
        if de.is_null() {
            return Ok(());
        }
        self.load_directory_entry(de)
    }

    /// Load and decode the RAW image data.
    pub fn load_raw(&mut self) -> Result<(), Error> {
        // SAFETY: self.x3f is valid.
        let de = unsafe { sys::x3f_get_raw(self.x3f.as_ptr()) };
        if de.is_null() {
            return Err(Error::Library(LibraryError::Argument));
        }
        // SAFETY: de is non-null.
        let r = unsafe { sys::x3f_load_data(self.x3f.as_ptr(), de) };
        check(r)
    }

    /// Load the RAW image block without decoding (used for `-raw` dump).
    pub fn load_unconverted_raw(&mut self) -> Result<(), Error> {
        // SAFETY: self.x3f is valid.
        let de = unsafe { sys::x3f_get_raw(self.x3f.as_ptr()) };
        if de.is_null() {
            return Err(Error::Library(LibraryError::Argument));
        }
        // SAFETY: de is non-null.
        let r = unsafe { sys::x3f_load_image_block(self.x3f.as_ptr(), de) };
        check(r)
    }

    /// Dump the metadata to a file (textual format matching the legacy CLI).
    pub fn dump_meta(&self, out: impl AsRef<Path>) -> Result<(), Error> {
        let cpath = path_to_cstring(out.as_ref())?;
        // SAFETY: x3f is valid and cpath outlives the call.
        let r = unsafe { sys::x3f_dump_meta_data(self.x3f.as_ptr(), cpath.as_ptr() as *mut _) };
        check(r)
    }

    /// Dump the embedded JPEG thumbnail (a byte-blob copy of the embedded
    /// JFIF stream — the original `x3f_dump.c` did the same).
    pub fn dump_jpeg(&self, out: impl AsRef<Path>) -> Result<(), Error> {
        // SAFETY: self.x3f is valid for the lifetime of self.
        let de = unsafe { sys::x3f_get_thumb_jpeg(self.x3f.as_ptr()) };
        self.write_blob(out.as_ref(), de)
    }

    /// Dump the raw, undecoded image block (Huffman/TRUE-coded bytes — the
    /// caller is expected to have already loaded it via [`load_unconverted_raw`]
    /// so `data`/`data_size` are populated).
    ///
    /// [`load_unconverted_raw`]: Self::load_unconverted_raw
    pub fn dump_raw_block(&self, out: impl AsRef<Path>) -> Result<(), Error> {
        // SAFETY: self.x3f is valid.
        let de = unsafe { sys::x3f_get_raw(self.x3f.as_ptr()) };
        self.write_blob(out.as_ref(), de)
    }

    fn write_blob(&self, out: &Path, de: *mut sys::x3f_directory_entry_t) -> Result<(), Error> {
        if de.is_null() {
            return Err(Error::Library(LibraryError::Argument));
        }
        // SAFETY: de is non-null; image_data is the populated union arm for
        // both the embedded JPEG and the raw image directory entries (the
        // legacy x3f_dump.c assumed the same).
        let (data, size) = unsafe {
            let img = (*de).header.data_subsection.image_data;
            (img.data, (*de).input.size)
        };
        if data.is_null() {
            return Err(Error::Library(LibraryError::Internal));
        }
        // SAFETY: data points to size bytes inside an mmap or owned buffer
        // managed by the parent x3f_t and outlives this call.
        let bytes = unsafe { std::slice::from_raw_parts(data as *const u8, size as usize) };
        std::fs::write(out, bytes).map_err(|source| Error::Io {
            path: out.display().to_string(),
            source,
        })
    }

    /// Dump as DNG (LinearRaw with DNG-specific tags).
    pub fn dump_dng(&self, out: impl AsRef<Path>, opts: &ProcessOptions) -> Result<(), Error> {
        output::dng::write(self, out.as_ref(), opts)
    }

    /// Dump as TIFF (3×16-bit RGB).
    pub fn dump_tiff(&self, out: impl AsRef<Path>, opts: &ProcessOptions) -> Result<(), Error> {
        let out = out.as_ref();
        let image = self.get_image(opts)?;
        // Embed an ICC profile only in cineon mode — non-cineon output
        // already has the encoding's gamma curve baked in, and renderers
        // default to sRGB which matches the historical behaviour. The
        // cineon ICC declares the chosen primaries with a *linear* TRC
        // so colour-managed readers leave the log encoding intact (the
        // file is meant to look flat in viewers; the colourist supplies
        // their own log→linear input transform when grading).
        let icc = if opts.cineon {
            icc::cineon_log_profile(opts.color_encoding)
        } else {
            None
        };
        output::tiff::write(&image, out, opts.compress, icc.as_deref()).map_err(|source| {
            Error::Io {
                path: out.display().to_string(),
                source,
            }
        })
    }

    /// Dump as PPM. `binary` selects P6 (true) vs P3 (false).
    pub fn dump_ppm(
        &self,
        out: impl AsRef<Path>,
        opts: &ProcessOptions,
        binary: bool,
    ) -> Result<(), Error> {
        let out = out.as_ref();
        let image = self.get_image(opts)?;
        output::ppm::write(&image, out, binary).map_err(|source| Error::Io {
            path: out.display().to_string(),
            source,
        })
    }

    /// Dump per-channel histogram as CSV. `log_exposure` selects logarithmic
    /// binning (matches `-loghist`).
    pub fn dump_histogram(
        &self,
        out: impl AsRef<Path>,
        opts: &ProcessOptions,
        log_exposure: bool,
    ) -> Result<(), Error> {
        let cpath = path_to_cstring(out.as_ref())?;
        let cwb = wb_cstring(opts.wb.as_deref())?;
        let sgain = self.resolve_sgain(opts.apply_sgain);
        // SAFETY: see dump_dng.
        let r = unsafe {
            sys::x3f_dump_raw_data_as_histogram(
                self.x3f.as_ptr(),
                cpath.as_ptr() as *mut _,
                opts.color_encoding.to_raw(),
                opts.crop as i32,
                opts.fix_bad as i32,
                opts.denoise as i32,
                sgain,
                cwb_ptr(&cwb),
                log_exposure as i32,
            )
        };
        check(r)
    }

    /// Sigma's heuristic from the legacy CLI: pre-Quattro files default
    /// `apply_sgain=on`, Quattro defaults `off`. Caller can override.
    fn resolve_sgain(&self, override_: Option<bool>) -> i32 {
        if let Some(v) = override_ {
            return v as i32;
        }
        // X3F_VERSION_4_0 = 0x00040000 in the C header. Files older than 4.0
        // (i.e. pre-Quattro) get sgain on.
        const X3F_VERSION_4_0: u32 = 0x0004_0000;
        (self.header_version() < X3F_VERSION_4_0) as i32
    }
}

impl Drop for Reader {
    fn drop(&mut self) {
        // SAFETY: x3f and file are valid. x3f_delete frees the structure;
        // fclose closes the FILE* afterwards. Both are infallible from our
        // perspective (we ignore their return values).
        unsafe {
            sys::x3f_delete(self.x3f.as_ptr());
            libc::fclose(self.file.as_ptr());
        }
    }
}

pub(crate) fn wb_cstring(wb: Option<&str>) -> Result<Option<CString>, Error> {
    match wb {
        None => Ok(None),
        Some(s) => CString::new(s).map(Some).map_err(|_| Error::WbNul),
    }
}

pub(crate) fn cwb_ptr(cwb: &Option<CString>) -> *mut std::os::raw::c_char {
    match cwb {
        Some(s) => s.as_ptr() as *mut _,
        None => std::ptr::null_mut(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn library_error_ok_maps_to_ok() {
        assert!(LibraryError::from_raw(sys::x3f_return_e_X3F_OK).is_ok());
    }

    #[test]
    fn library_error_known_codes_map_to_named_variants() {
        let cases = [
            (sys::x3f_return_e_X3F_ARGUMENT_ERROR, LibraryError::Argument),
            (sys::x3f_return_e_X3F_INFILE_ERROR, LibraryError::Infile),
            (sys::x3f_return_e_X3F_OUTFILE_ERROR, LibraryError::Outfile),
            (sys::x3f_return_e_X3F_INTERNAL_ERROR, LibraryError::Internal),
        ];
        for (raw, expected) in cases {
            assert_eq!(LibraryError::from_raw(raw).unwrap_err(), expected);
        }
    }

    #[test]
    fn library_error_unknown_code_is_preserved() {
        let unknown: sys::x3f_return_t = 0x1234_5678;
        assert_eq!(
            LibraryError::from_raw(unknown).unwrap_err(),
            LibraryError::Unknown(unknown),
        );
    }

    #[test]
    fn color_encoding_round_trips_through_raw() {
        // The mapping is the only place a typo in either enum would land us
        // silently producing wrong output (e.g. Adobe RGB vs ProPhoto RGB).
        let pairs = [
            (ColorEncoding::None, sys::x3f_color_encoding_e_NONE),
            (ColorEncoding::Srgb, sys::x3f_color_encoding_e_SRGB),
            (ColorEncoding::AdobeRgb, sys::x3f_color_encoding_e_ARGB),
            (ColorEncoding::ProPhotoRgb, sys::x3f_color_encoding_e_PPRGB),
            (
                ColorEncoding::Unprocessed,
                sys::x3f_color_encoding_e_UNPROCESSED,
            ),
            (ColorEncoding::Qtop, sys::x3f_color_encoding_e_QTOP),
        ];
        for (enc, raw) in pairs {
            assert_eq!(enc.to_raw(), raw, "{enc:?}");
        }
    }

    #[test]
    fn wb_cstring_handles_none_and_string_inputs() {
        assert!(matches!(wb_cstring(None), Ok(None)));
        let cs = wb_cstring(Some("Daylight")).unwrap().unwrap();
        assert_eq!(cs.to_bytes(), b"Daylight");
    }

    #[test]
    fn wb_cstring_rejects_interior_nul() {
        assert!(matches!(wb_cstring(Some("Day\0light")), Err(Error::WbNul)));
    }
}
