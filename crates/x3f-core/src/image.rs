//! Processed-image extraction wrapper around the C `x3f_get_image`.
//!
//! The C API populates an `x3f_area16_t` whose `buf` is malloc'd. We copy
//! the pixels into a Rust-owned `Vec<u16>` and free the C buffer immediately,
//! so callers don't have to think about cross-boundary lifetimes.

use std::ptr;

use x3f_sys as sys;
// libc compat — see `x3f-sys/src/sysabi.rs`. Used here for `libc::free`
// on the C-allocated `x3f_area16_t.buf` / preview pixel buffer.
use x3f_sys::sysabi as libc;

use crate::{cwb_ptr, wb_cstring, Error, LibraryError, ProcessOptions, Reader};

/// Per-channel black point and saturation level returned alongside a
/// processed image. Matches the C `x3f_image_levels_t`. DNG output uses
/// these for the `BlackLevel` / `WhiteLevel` tags.
#[derive(Debug, Clone, Copy, Default)]
pub struct ImageLevels {
    /// Per-channel black point (DNG `BlackLevel`).
    pub black: [f64; 3],
    /// Per-channel saturation level (DNG `WhiteLevel`).
    pub white: [u32; 3],
}

/// A processed image: `channels` 16-bit samples per pixel, interleaved.
///
/// Pixel `(row, col)` channel `c` lives at
/// `data[row * row_stride + col * channels + c]`.
///
/// `row_stride` is in `u16` elements (not bytes) and may exceed
/// `columns * channels` when the C-side allocator pads rows.
pub struct Image {
    /// Interleaved 16-bit samples, `rows * row_stride` elements long.
    pub data: Vec<u16>,
    /// Image height in pixels.
    pub rows: u32,
    /// Image width in pixels.
    pub columns: u32,
    /// Samples per pixel (3 for RGB).
    pub channels: u32,
    /// Row pitch in `u16` elements (may exceed `columns * channels`).
    pub row_stride: u32,
    /// Per-channel black/white levels for this image.
    pub levels: ImageLevels,
    /// Snapshot of `x3f_get_dng_highlight_scale()` taken immediately
    /// after [`Reader::get_image`] returns, on the same thread that ran
    /// `apply_highlight_clip_dng`. The DNG writer adds `log2(scale)` to
    /// `BaselineExposure` so consumers restore brightness on import. We
    /// capture it here (instead of letting the writer re-read a global
    /// side-channel later) so nested-rayon work-stealing in batch mode
    /// can't have another file's `apply_highlight_clip_dng` clobber the
    /// thread-local cell between when this image was rendered and when
    /// the DNG writer needs the value.
    pub dng_highlight_scale: f64,
}

/// 8-bit preview image (3-channel interleaved). Produced by
/// [`Reader::get_preview`] for use as the IFD0 thumbnail in DNG output.
pub struct Preview {
    /// Interleaved 8-bit samples, `rows * row_stride` elements long.
    pub data: Vec<u8>,
    /// Preview height in pixels.
    pub rows: u32,
    /// Preview width in pixels.
    pub columns: u32,
    /// Samples per pixel (3 for RGB).
    pub channels: u32,
    /// Row pitch in bytes (may exceed `columns * channels`).
    pub row_stride: u32,
}

impl Image {
    /// Sample at `(row, col, channel)`. Bounds-checked via `Vec` indexing.
    #[inline]
    pub fn sample(&self, row: u32, col: u32, channel: u32) -> u16 {
        let idx = (row as usize) * (self.row_stride as usize)
            + (col as usize) * (self.channels as usize)
            + (channel as usize);
        self.data[idx]
    }

    /// Slice covering one full row, including any stride padding.
    #[inline]
    pub fn row(&self, row: u32) -> &[u16] {
        let start = (row as usize) * (self.row_stride as usize);
        let end = start + (self.row_stride as usize);
        &self.data[start..end]
    }
}

impl Reader {
    /// Run the full processing pipeline (white-balance, color-matrix, gamma,
    /// highlight-recovery, optional crop) and return the result as a Rust-owned
    /// 16-bit RGB image.
    pub fn get_image(&self, opts: &ProcessOptions) -> Result<Image, Error> {
        let cwb = wb_cstring(opts.wb.as_deref())?;
        let sgain = self.resolve_sgain(opts.apply_sgain);
        // Communicate the DNG highlight-recovery toggle to
        // `apply_highlight_clip_dng` via the thread-local FFI hook.
        // Set immediately before `x3f_get_image` so a stale value from
        // a prior conversion on this thread can't bleed into ours.
        // SAFETY: setter is a plain Cell write, no aliasing concerns.
        unsafe { sys::x3f_set_dng_highlight_recovery(opts.dng_highlight_recovery as libc::c_int) };
        // Same pattern for the Cineon-log TIFF mode toggle. Always
        // written (true *or* false) so a stale `true` from a previous
        // cineon conversion on this rayon worker can't leak into a
        // non-cineon call.
        unsafe { sys::x3f_set_cineon(opts.cineon as libc::c_int) };
        // SAFETY: zero-init is a valid x3f_area16_t (all-NULL pointers, all-0
        // dimensions). x3f_get_image overwrites every field on success.
        let mut area: sys::x3f_area16_t = unsafe { std::mem::zeroed() };
        let mut ilevels: sys::x3f_image_levels_t = unsafe { std::mem::zeroed() };

        // x3f_get_image early-returns for UNPROCESSED encoding with the
        // condition `return ilevels == NULL` — i.e. it requires a NULL
        // levels pointer in that mode and otherwise reports failure. We
        // mirror that contract here.
        let ilevels_ptr = if matches!(opts.color_encoding, crate::ColorEncoding::Unprocessed) {
            ptr::null_mut()
        } else {
            &mut ilevels
        };

        // SAFETY: x3f is valid; area is a stack sink x3f_get_image populates;
        // wb pointer (if any) outlives the call.
        let ok = unsafe {
            sys::x3f_get_image(
                self.x3f.as_ptr(),
                &mut area,
                ilevels_ptr,
                opts.color_encoding.to_raw(),
                opts.crop as i32,
                opts.fix_bad as i32,
                opts.denoise as i32,
                sgain,
                cwb_ptr(&cwb),
            )
        };
        if ok == 0 {
            return Err(Error::Library(LibraryError::Argument));
        }

        // Capture the DNG highlight scale RIGHT NOW, before any other
        // FFI call (notably `get_preview` or any nested-rayon work
        // inside the DNG writer's strip encoder) can let rayon
        // work-steal another file's `apply_highlight_clip_dng` onto
        // this thread and clobber the thread-local cell. The set+read
        // pair is safely co-located inside `x3f_get_image`'s body
        // here.
        let dng_highlight_scale = unsafe { sys::x3f_get_dng_highlight_scale() };

        let len = (area.rows as usize) * (area.row_stride as usize);
        // SAFETY: x3f_get_image returns a buffer the C accessor pattern reads
        // as data[0..rows*row_stride]; the same range is in-bounds for us.
        let data = unsafe { std::slice::from_raw_parts(area.data, len).to_vec() };

        // SAFETY: area.buf was malloc'd by the C library and is non-null on
        // a successful return; we are the unique owner now.
        unsafe { libc::free(area.buf) };

        Ok(Image {
            data,
            rows: area.rows,
            columns: area.columns,
            channels: area.channels,
            row_stride: area.row_stride,
            levels: ImageLevels {
                black: ilevels.black,
                white: ilevels.white,
            },
            dng_highlight_scale,
        })
    }

    /// Render an 8-bit downsampled preview of `image`. `max_width` caps the
    /// output dimensions (the legacy DNG writer uses 300). `image` must have
    /// been produced by [`Self::get_image`] so its `levels` are populated.
    pub fn get_preview(
        &self,
        image: &Image,
        opts: &ProcessOptions,
        max_width: u32,
    ) -> Result<Preview, Error> {
        let cwb = wb_cstring(opts.wb.as_deref())?;
        let sgain = self.resolve_sgain(opts.apply_sgain);

        // x3f_get_preview reads from `image` (data + dims) and writes to a
        // freshly-allocated preview struct. We synthesise a transient
        // x3f_area16_t whose `data` aliases our owned Vec — the C function
        // never frees `buf`, so leaving it null is safe.
        let mut area = sys::x3f_area16_t {
            data: image.data.as_ptr() as *mut u16,
            buf: ptr::null_mut(),
            rows: image.rows,
            columns: image.columns,
            channels: image.channels,
            row_stride: image.row_stride,
        };
        let mut ilevels = sys::x3f_image_levels_t {
            black: image.levels.black,
            white: image.levels.white,
        };
        let mut preview: sys::x3f_area8_t = unsafe { std::mem::zeroed() };

        // SAFETY: every pointer is non-null and outlives the call. The C
        // function reads from area+ilevels and populates preview.
        let ok = unsafe {
            sys::x3f_get_preview(
                self.x3f.as_ptr(),
                &mut area,
                &mut ilevels,
                sys::x3f_color_encoding_e_SRGB,
                sgain,
                cwb_ptr(&cwb),
                max_width,
                &mut preview,
            )
        };
        if ok == 0 {
            return Err(Error::Library(LibraryError::Argument));
        }

        let len = (preview.rows as usize) * (preview.row_stride as usize);
        // SAFETY: preview.data points to len bytes the C function just wrote.
        let data = unsafe { std::slice::from_raw_parts(preview.data, len).to_vec() };
        // SAFETY: preview.buf was malloc'd by the C library; we own it now.
        unsafe { libc::free(preview.buf as *mut _) };

        Ok(Preview {
            data,
            rows: preview.rows,
            columns: preview.columns,
            channels: preview.channels,
            row_stride: preview.row_stride,
        })
    }
}
