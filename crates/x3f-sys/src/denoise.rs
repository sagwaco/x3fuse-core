//! NLM and two-stage BM3D with portable Rust CPU implementations and an optional
//! native wgpu backend. Called directly by the pipeline (`run_denoising` in
//! `process.rs` and the two Quattro passes in `quattro.rs`).
//!
//! ## History
//!
//! The denoise pass originally linked against opencv-mobile's `cv::photo`
//! (`fastNlMeansDenoising`, `medianBlur`, `resize`), which shipped prebuilt
//! static archives for Apple / Linux / Windows / Android but not for `wasm32`
//! (and offline / docs.rs builds couldn't fetch it). This module was written as
//! a pure-Rust equivalent so denoise worked everywhere; it has since replaced
//! OpenCV entirely on every target. The C++ sources and the opencv-mobile fetch
//! in `build.rs` are gone.
//!
//! ## Fidelity
//!
//! The algorithm mirrors the original OpenCV pipeline closely — the same
//! fixed-point weight LUT, `BORDER_REFLECT_101` template/search windows,
//! per-channel `h`, L1 patch distance, V-channel 3×3 median, and the
//! low-frequency down/denoise/subtract/up/subtract refinement. It is **not**
//! byte-identical to the old OpenCV output (floating-point `exp`, INTER_AREA /
//! INTER_CUBIC fixed-point and rounding differ), and no parity baseline pins
//! it: every tier-2/tier-3 test runs with `-no-denoise`. The goal is a visually
//! equivalent denoise, not a bit-exact clone.
//!
//! GPU kernels retain the same distances and weights, using FP32 weighted
//! sums. Tests allow at most one u16 code of kernel error against the Rust
//! CPU reference. Automatic fallback and WASM use the CPU implementation.

use crate::Control;

use crate::quattro::Area16;

mod bm3d;
#[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
mod gpu;

/// Denoising algorithm, independent of execution backend.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(rename_all = "lowercase")
)]
pub enum DenoiseAlgorithm {
    #[default]
    Nlm,
    Bm3d,
}

impl std::str::FromStr for DenoiseAlgorithm {
    type Err = &'static str;
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "nlm" => Ok(Self::Nlm),
            "bm3d" => Ok(Self::Bm3d),
            _ => Err("denoiser must be nlm or bm3d"),
        }
    }
}

/// Full, odd NLM window sizes in pixels. These tune the main pass; the
/// legacy low-frequency chroma pass retains its 21-pixel search window.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(default, deny_unknown_fields, rename_all = "camelCase")
)]
pub struct NlmOptions {
    pub luma_patch: u8,
    pub chroma_patch: u8,
    pub luma_search: u8,
    pub chroma_search: u8,
}

impl NlmOptions {
    pub const DEFAULT: Self = Self {
        luma_patch: 3,
        chroma_patch: 3,
        luma_search: 11,
        chroma_search: 11,
    };
}
impl Default for NlmOptions {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// Sensor-space denoising. Defaults retain NLM's chroma-oriented Foveon
/// profile. `strength` multiplies the camera profile and intensity (0..10);
/// NLM windows do not affect BM3D's fixed 8x8 block profile.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(default, deny_unknown_fields, rename_all = "camelCase")
)]
pub struct DenoiseOptions {
    pub algorithm: DenoiseAlgorithm,
    pub luminance: bool,
    pub chroma: bool,
    pub strength: f32,
    pub nlm: NlmOptions,
}
impl DenoiseOptions {
    pub const DEFAULT: Self = Self {
        algorithm: DenoiseAlgorithm::Nlm,
        luminance: false,
        chroma: true,
        strength: 1.0,
        nlm: NlmOptions::DEFAULT,
    };

    pub fn validate(&self) -> Result<(), &'static str> {
        if !self.strength.is_finite() || !(0.1..=3.0).contains(&self.strength) {
            return Err("denoise strength must be finite and between 0.1 and 3.0");
        }
        for patch in [self.nlm.luma_patch, self.nlm.chroma_patch] {
            if !(1..=9).contains(&patch) || patch % 2 == 0 {
                return Err("NLM patch sizes must be odd, from 1 through 9");
            }
        }
        for search in [self.nlm.luma_search, self.nlm.chroma_search] {
            if !(1..=31).contains(&search) || search % 2 == 0 {
                return Err("NLM search sizes must be odd, from 1 through 31");
            }
        }
        Ok(())
    }
}
impl Default for DenoiseOptions {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// Execution preference for one conversion. `Auto` falls back to portable CPU
/// denoising; `Gpu` reports an error when acceleration cannot be used.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DenoiseBackend {
    /// Try the native GPU backend, retry on CPU if unavailable or unsuccessful.
    #[default]
    Auto,
    /// Always use the portable Rust implementation, without initializing a GPU.
    Cpu,
    /// Require GPU execution; return an error instead of falling back to CPU.
    Gpu,
}

impl std::str::FromStr for DenoiseBackend {
    type Err = &'static str;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "auto" => Ok(Self::Auto),
            "cpu" => Ok(Self::Cpu),
            "gpu" => Ok(Self::Gpu),
            _ => Err("denoise backend must be auto, cpu, or gpu"),
        }
    }
}

fn log(level: crate::x3f_verbosity_t, message: &str) {
    #[cfg(not(target_arch = "wasm32"))]
    if let Ok(message) = std::ffi::CString::new(message) {
        unsafe { crate::x3f_printf(level, c"%s\n".as_ptr(), message.as_ptr()) };
    }
    #[cfg(target_arch = "wasm32")]
    let _ = (level, message);
}

// ----------------------------------------------------------------------
// Denoise type table (mirror of `denoise_types[]` in x3f_denoise_utils.h)
// ----------------------------------------------------------------------

/// Sensor-type selector passed by callers (`denoise_area` / `denoise_active_area`),
/// mapped to a `DType` via `DType::from_u32`. Values match the legacy
/// `x3f_denoise_type_t` enum so existing call sites and behaviour are preserved.
pub(crate) const DENOISE_STD: u32 = 0;
pub(crate) const DENOISE_F20: u32 = 1;
pub(crate) const DENOISE_F23: u32 = 2;

/// The three BMT↔YUV transform families, selected by the sensor type.
/// Each carries the per-sensor base NLM sigma `h` from `denoise_types[]`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum DType {
    /// `X3F_DENOISE_STD` (Merrill): `h = 100`, luma = (B+M+T)/3.
    Std,
    /// `X3F_DENOISE_F20`: `h = 70`, luma = T.
    YisT,
    /// `X3F_DENOISE_F23` (Quattro): `h = 300`, luma = 4·T.
    Yis4T,
}

impl DType {
    fn from_u32(t: u32) -> Self {
        match t {
            1 => DType::YisT,
            2 => DType::Yis4T,
            // 0 (STD) and anything unexpected fall back to STD, matching the
            // C default where `denoise_types[0]` is the STD row.
            _ => DType::Std,
        }
    }

    /// Base sigma `h` from `denoise_types[]`.
    fn base_h(self) -> f64 {
        match self {
            DType::Std => 100.0,
            DType::YisT => 70.0,
            DType::Yis4T => 300.0,
        }
    }
}

/// Bias added to U/V so they stay non-negative in `u16` (matches the C
/// `O_UV` in `x3f_denoise_utils.cpp`).
const O_UV: i32 = 32768;

// ----------------------------------------------------------------------
// Tightly-packed 3-channel images (the working representation)
// ----------------------------------------------------------------------

/// Owned, tightly-packed (no row padding) interleaved 3-channel `u16` image.
/// `Area16` carries an arbitrary `row_stride`; we copy into this packed form
/// for the resize/median/NLM kernels and copy back.
#[derive(Clone)]
struct Img3 {
    rows: usize,
    cols: usize,
    /// `data[(r*cols + c)*3 + ch]`.
    data: Vec<u16>,
}

impl Img3 {
    fn new(rows: usize, cols: usize) -> Self {
        Img3 {
            rows,
            cols,
            data: vec![0u16; rows * cols * 3],
        }
    }

    fn area(&mut self) -> Area16 {
        Area16 {
            data: self.data.as_mut_ptr(),
            buf: std::ptr::null_mut(),
            rows: self.rows as u32,
            columns: self.cols as u32,
            channels: 3,
            row_stride: (self.cols * 3) as u32,
        }
    }
}

/// Signed counterpart of [`Img3`] for the `CV_16S` low-frequency residual.
struct Img3i16 {
    rows: usize,
    cols: usize,
    data: Vec<i16>,
}

/// Copy an `Area16` (possibly strided) into a packed [`Img3`].
///
/// # Safety
/// `area` must point to a valid 3-channel `x3f_area16_t` whose `data` spans
/// `rows * row_stride` `u16`s.
unsafe fn area_to_img3(area: *const Area16) -> Img3 {
    let a = unsafe { &*area };
    let rows = a.rows as usize;
    let cols = a.columns as usize;
    let stride = a.row_stride as usize;
    let mut img = Img3::new(rows, cols);
    let row_len = cols * 3;
    for r in 0..rows {
        let src = unsafe { a.data.add(r * stride) };
        let dst = &mut img.data[r * row_len..r * row_len + row_len];
        for (k, slot) in dst.iter_mut().enumerate() {
            *slot = unsafe { *src.add(k) };
        }
    }
    img
}

/// Copy a packed [`Img3`] back into a (possibly strided) `Area16` in place,
/// leaving any stride padding untouched.
///
/// # Safety
/// `area` must match `img`'s dimensions and point to a writable 3-channel
/// `x3f_area16_t`.
unsafe fn img3_to_area(img: &Img3, area: *mut Area16) {
    let a = unsafe { &*area };
    let rows = img.rows;
    let stride = a.row_stride as usize;
    let row_len = img.cols * 3;
    for r in 0..rows {
        let dst = unsafe { a.data.add(r * stride) };
        let src = &img.data[r * row_len..r * row_len + row_len];
        for (k, &v) in src.iter().enumerate() {
            unsafe { *dst.add(k) = v };
        }
    }
}

// ----------------------------------------------------------------------
// Small numeric helpers
// ----------------------------------------------------------------------

/// `BORDER_REFLECT_101`: `gfedcb|abcdefgh|gfedcba` — the edge sample is not
/// duplicated. Robust for offsets well beyond `[0, n)` (the search window can
/// reach `search_radius + template_radius` past an edge of a small image).
#[inline]
fn reflect_101(i: i32, n: i32) -> i32 {
    if n == 1 {
        return 0;
    }
    let period = 2 * (n - 1);
    let mut r = i.rem_euclid(period);
    if r >= n {
        r = period - r;
    }
    r
}

/// Smallest `p` such that `1 << p >= value` (OpenCV `getNearestPowerOf2`).
#[inline]
fn nearest_pow2(value: i32) -> u32 {
    let mut p = 0u32;
    while (1i64 << p) < value as i64 {
        p += 1;
    }
    p
}

/// `saturate_cast<uint16_t>` for an `i32` (clamp, no rounding).
#[inline]
fn sat_u16_i32(v: i32) -> u16 {
    v.clamp(0, 65535) as u16
}

/// `cvRound` (round half to even) of an `f64`, then clamp to `u16`.
#[inline]
fn round_sat_u16(v: f64) -> u16 {
    let r = v.round_ties_even();
    if r <= 0.0 {
        0
    } else if r >= 65535.0 {
        65535
    } else {
        r as u16
    }
}

/// Catmull-Rom cubic kernel with `a = -0.75` (OpenCV `INTER_CUBIC` default).
#[inline]
fn cubic_kernel(d: f64) -> f64 {
    const A: f64 = -0.75;
    let d = d.abs();
    if d <= 1.0 {
        ((A + 2.0) * d - (A + 3.0)) * d * d + 1.0
    } else if d < 2.0 {
        ((A * d - 5.0 * A) * d + 8.0 * A) * d - 4.0 * A
    } else {
        0.0
    }
}

/// Four cubic weights for taps at `floor(x)-1 .. floor(x)+2` given the
/// fractional offset `t = x - floor(x)`.
#[inline]
fn cubic_weights(t: f64) -> [f64; 4] {
    [
        cubic_kernel(1.0 + t),
        cubic_kernel(t),
        cubic_kernel(1.0 - t),
        cubic_kernel(2.0 - t),
    ]
}

// ----------------------------------------------------------------------
// Fast Non-Local Means (port of FastNlMeansDenoisingInvoker, CV_16UC3 +
// per-channel h + NORM_L1, the only instantiation x3f_denoise.cpp uses)
// ----------------------------------------------------------------------

/// Precomputed NLM state shared by all row bands.
struct NlmCtx<'a> {
    control: Control<'a>,
    /// `BORDER_REFLECT_101`-extended source, `ext_rows × ext_cols`, 3 channels.
    ext: &'a [u16],
    ext_cols: usize,
    rows: usize,
    cols: usize,
    border: i32,
    /// Template-window half size (radius).
    tr: i32,
    /// Search-window half size (radius).
    sr: i32,
    /// Template-window full size (`2*tr + 1`).
    tw: usize,
    /// Search-window full size (`2*sr + 1`).
    sw: usize,
    /// `>>` amount to turn a summed patch L1 distance into the LUT index.
    bin_shift: u32,
    /// `almost_dist2weight_`: per-channel fixed-point weight per quantized dist.
    lut: &'a [[i32; 3]],
}

impl NlmCtx<'_> {
    /// L1 distance (sum over 3 channels of `|a-b|`) between two pixels of the
    /// extended source, addressed by extended-image coordinates.
    #[inline]
    fn dist(&self, y1: i32, x1: i32, y2: i32, x2: i32) -> i32 {
        let i1 = (y1 as usize * self.ext_cols + x1 as usize) * 3;
        let i2 = (y2 as usize * self.ext_cols + x2 as usize) * 3;
        let e = self.ext;
        (e[i1] as i32 - e[i2] as i32).abs()
            + (e[i1 + 1] as i32 - e[i2 + 1] as i32).abs()
            + (e[i1 + 2] as i32 - e[i2 + 2] as i32).abs()
    }

    #[inline]
    fn ds(&self, y: i32, x: i32) -> usize {
        y as usize * self.sw + x as usize
    }

    #[inline]
    fn cds(&self, t: i32, y: i32, x: i32) -> usize {
        (t as usize * self.sw + y as usize) * self.sw + x as usize
    }

    #[inline]
    fn ucds(&self, j: i32, y: i32, x: i32) -> usize {
        (j as usize * self.sw + y as usize) * self.sw + x as usize
    }

    /// Process output rows `[row_from, row_to]` (inclusive), writing them into
    /// `dst` (packed, `(row - row_from)` indexing). Mirrors
    /// `FastNlMeansDenoisingInvoker::operator()` over one `cv::Range`.
    fn process_rows(&self, row_from: i32, row_to: i32, dst: &mut [u16]) {
        self.process_rows_until(row_from, row_to, dst, || self.control.check().is_err());
    }

    fn process_rows_until(
        &self,
        row_from: i32,
        row_to: i32,
        dst: &mut [u16],
        mut cancelled: impl FnMut() -> bool,
    ) {
        let sw = self.sw;
        let tw = self.tw as i32;
        let cols = self.cols as i32;
        let border = self.border;
        let tr = self.tr;
        let sr = self.sr;

        let mut dist_sums = vec![0i32; sw * sw];
        let mut col_dist_sums = vec![0i32; self.tw * sw * sw];
        let mut up_col_dist_sums = vec![0i32; self.cols * sw * sw];
        let mut first_col_num = -1i32;

        for i in row_from..=row_to {
            if cancelled() {
                return;
            }
            for j in 0..cols {
                if j % 64 == 0 && cancelled() {
                    return;
                }
                if j == 0 {
                    self.calc_first_in_row(
                        i,
                        &mut dist_sums,
                        &mut col_dist_sums,
                        &mut up_col_dist_sums,
                    );
                    first_col_num = 0;
                } else {
                    if i == row_from {
                        self.calc_first_in_first_row(
                            i,
                            j,
                            first_col_num,
                            &mut dist_sums,
                            &mut col_dist_sums,
                            &mut up_col_dist_sums,
                        );
                    } else {
                        // Incremental update: drop the column leaving the
                        // template window on the left, add the one entering on
                        // the right (reusing the previous row's column sums).
                        let ay = border + i;
                        let ax = border + j + tr;
                        let start_by = border + i - sr;
                        let start_bx = border + j - sr + tr;

                        for y in 0..sw as i32 {
                            for x in 0..sw as i32 {
                                let cidx = self.cds(first_col_num, y, x);
                                dist_sums[self.ds(y, x)] -= col_dist_sums[cidx];

                                let bx = start_bx + x;
                                let updown = self.dist(ay + tr, ax, start_by + tr + y, bx)
                                    - self.dist(ay - tr - 1, ax, start_by - tr - 1 + y, bx);
                                let new_col = up_col_dist_sums[self.ucds(j, y, x)] + updown;

                                col_dist_sums[cidx] = new_col;
                                dist_sums[self.ds(y, x)] += new_col;
                                up_col_dist_sums[self.ucds(j, y, x)] = new_col;
                            }
                        }
                    }
                    first_col_num = (first_col_num + 1) % tw;
                }

                // Weighted average of the search-window center pixels.
                let mut est = [0i64; 3];
                let mut wsum = [0i64; 3];
                let swy = i - sr;
                let swx = j - sr;
                for y in 0..sw as i32 {
                    for x in 0..sw as i32 {
                        let almost = (dist_sums[self.ds(y, x)] >> self.bin_shift) as usize;
                        let w = self.lut[almost.min(self.lut.len() - 1)];
                        let py = (border + swy + y) as usize;
                        let px = (border + swx + x) as usize;
                        let p = (py * self.ext_cols + px) * 3;
                        est[0] += w[0] as i64 * self.ext[p] as i64;
                        est[1] += w[1] as i64 * self.ext[p + 1] as i64;
                        est[2] += w[2] as i64 * self.ext[p + 2] as i64;
                        wsum[0] += w[0] as i64;
                        wsum[1] += w[1] as i64;
                        wsum[2] += w[2] as i64;
                    }
                }

                let o = ((i - row_from) as usize * self.cols + j as usize) * 3;
                for c in 0..3 {
                    // divByWeightsSum: rounded division, guarded against the
                    // (algorithmically impossible — the center matches itself)
                    // zero-weight case.
                    let v = if wsum[c] > 0 {
                        ((est[c] as u64 + wsum[c] as u64 / 2) / wsum[c] as u64) as i64
                    } else {
                        0
                    };
                    dst[o + c] = v.clamp(0, 65535) as u16;
                }
            }
        }
    }

    /// `calcDistSumsForFirstElementInRow` — full recompute at column 0.
    fn calc_first_in_row(
        &self,
        i: i32,
        dist_sums: &mut [i32],
        col_dist_sums: &mut [i32],
        up_col_dist_sums: &mut [i32],
    ) {
        let j = 0i32;
        let tr = self.tr;
        let sr = self.sr;
        let border = self.border;
        for y in 0..self.sw as i32 {
            for x in 0..self.sw as i32 {
                dist_sums[self.ds(y, x)] = 0;
                for t in 0..self.tw as i32 {
                    col_dist_sums[self.cds(t, y, x)] = 0;
                }
                let start_y = i + y - sr;
                let start_x = j + x - sr;
                for ty in -tr..=tr {
                    for tx in -tr..=tr {
                        let d = self.dist(
                            border + i + ty,
                            border + j + tx,
                            border + start_y + ty,
                            border + start_x + tx,
                        );
                        dist_sums[self.ds(y, x)] += d;
                        col_dist_sums[self.cds(tx + tr, y, x)] += d;
                    }
                }
                up_col_dist_sums[self.ucds(j, y, x)] =
                    col_dist_sums[self.cds(self.tw as i32 - 1, y, x)];
            }
        }
    }

    /// `calcDistSumsForElementInFirstRow` — slide one column at the top row of
    /// a band (no previous row to lean on, so recompute the entering column).
    fn calc_first_in_first_row(
        &self,
        i: i32,
        j: i32,
        first_col_num: i32,
        dist_sums: &mut [i32],
        col_dist_sums: &mut [i32],
        up_col_dist_sums: &mut [i32],
    ) {
        let tr = self.tr;
        let sr = self.sr;
        let border = self.border;
        let ay = border + i;
        let ax = border + j + tr;
        let start_by = border + i - sr;
        let start_bx = border + j - sr + tr;
        for y in 0..self.sw as i32 {
            for x in 0..self.sw as i32 {
                dist_sums[self.ds(y, x)] -= col_dist_sums[self.cds(first_col_num, y, x)];
                let by = start_by + y;
                let bx = start_bx + x;
                let mut s = 0i32;
                for ty in -tr..=tr {
                    s += self.dist(ay + ty, ax, by + ty, bx);
                }
                col_dist_sums[self.cds(first_col_num, y, x)] = s;
                dist_sums[self.ds(y, x)] += s;
                up_col_dist_sums[self.ucds(j, y, x)] = s;
            }
        }
    }
}

/// Port of `cv::fastNlMeansDenoising(src, dst, h, templateWindowSize,
/// searchWindowSize, NORM_L1)` for `CV_16UC3` with a per-channel `h`.
///
/// `template_window` / `search_window` are full (odd) window sizes, exactly
/// as passed to OpenCV (e.g. `3, 11`).
#[cfg(test)]
fn fast_nlm_denoise(src: &Img3, h: [f64; 3], template_window: i32, search_window: i32) -> Img3 {
    fast_nlm_cpu(src, h, template_window, search_window, Control::none()).unwrap()
}

fn fast_nlm_denoise_controlled(
    src: &Img3,
    h: [f64; 3],
    template_window: i32,
    search_window: i32,
    control: Control<'_>,
) -> crate::Result<Img3> {
    fast_nlm_with_gpu(src, h, template_window, search_window, control, || {
        #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
        return gpu::nlm(src, h, template_window, search_window, control);
        #[cfg(not(all(feature = "gpu", not(target_arch = "wasm32"))))]
        Err(crate::Error::Gpu("this build has no GPU denoiser".into()))
    })
}

fn fast_nlm_with_gpu(
    src: &Img3,
    h: [f64; 3],
    template_window: i32,
    search_window: i32,
    control: Control<'_>,
    accelerated: impl FnOnce() -> crate::Result<Img3>,
) -> crate::Result<Img3> {
    control.check()?;
    #[cfg(not(target_arch = "wasm32"))]
    let started = std::time::Instant::now();
    let mut backend = "cpu";
    let result = if control.denoise_backend == DenoiseBackend::Cpu {
        fast_nlm_cpu(src, h, template_window, search_window, control)
    } else {
        match accelerated() {
            Ok(image) => {
                backend = "gpu";
                Ok(image)
            }
            Err(crate::Error::Cancelled) => Err(crate::Error::Cancelled),
            Err(error) if control.denoise_backend == DenoiseBackend::Gpu => {
                log(crate::x3f_verbosity_t_ERR, &error.to_string());
                Err(error)
            }
            Err(error) => {
                log(
                    crate::x3f_verbosity_t_DEBUG,
                    &format!("NLM: CPU fallback ({error})"),
                );
                control.check()?;
                fast_nlm_cpu(src, h, template_window, search_window, control)
            }
        }
    };
    #[cfg(not(target_arch = "wasm32"))]
    log(
        crate::x3f_verbosity_t_DEBUG,
        &format!(
            "NLM: {backend}, {}x{}, search={search_window}, {:.3} ms",
            src.cols,
            src.rows,
            started.elapsed().as_secs_f64() * 1000.0,
        ),
    );
    #[cfg(target_arch = "wasm32")]
    let _ = backend;
    result
}

fn fast_nlm_cpu(
    src: &Img3,
    h: [f64; 3],
    template_window: i32,
    search_window: i32,
    control: Control<'_>,
) -> crate::Result<Img3> {
    control.check()?;
    let rows = src.rows;
    let cols = src.cols;
    if rows == 0 || cols == 0 {
        return Ok(Img3::new(rows, cols));
    }

    let tr = template_window / 2;
    let sr = search_window / 2;
    let tw = (tr * 2 + 1) as usize;
    let sw = (sr * 2 + 1) as usize;
    let border = sr + tr;

    // BORDER_REFLECT_101-extended source. Precompute the per-axis source-index
    // maps so the fill is plain lookups.
    let ext_rows = rows + 2 * border as usize;
    let ext_cols = cols + 2 * border as usize;
    let row_map: Vec<usize> = (0..ext_rows)
        .map(|ey| reflect_101(ey as i32 - border, rows as i32) as usize)
        .collect();
    let col_map: Vec<usize> = (0..ext_cols)
        .map(|ex| reflect_101(ex as i32 - border, cols as i32) as usize)
        .collect();
    let mut ext = vec![0u16; ext_rows * ext_cols * 3];
    for ey in 0..ext_rows {
        control.check()?;
        let sy = row_map[ey];
        for ex in 0..ext_cols {
            let sx = col_map[ex];
            let si = (sy * cols + sx) * 3;
            let di = (ey * ext_cols + ex) * 3;
            ext[di] = src.data[si];
            ext[di + 1] = src.data[si + 1];
            ext[di + 2] = src.data[si + 2];
        }
    }

    // Fixed-point scale (see invoker.hpp). For u16 + i32 weights this is
    // always INT_MAX for the window sizes we use, but compute it generically.
    let sample_max: i64 = 65535;
    let max_estimate_sum = (sw as i64) * (sw as i64) * sample_max;
    let fixed_point_mult = std::cmp::min(i64::MAX / max_estimate_sum, i32::MAX as i64);

    // Quantized-distance → weight LUT (`almost_dist2weight_`).
    let tw_sq = (tw * tw) as i32;
    let bin_shift = nearest_pow2(tw_sq);
    let mult = ((1i64 << bin_shift) as f64) / tw_sq as f64;
    let max_dist = 65535i32 * 3;
    let almost_max = ((max_dist as f64) / mult + 1.0) as usize;
    let threshold = 0.001 * fixed_point_mult as f64;
    let mut lut = vec![[0i32; 3]; almost_max.max(1)];
    for (ad, slot) in lut.iter_mut().enumerate() {
        if ad % 1024 == 0 {
            control.check()?;
        }
        let dist = ad as f64 * mult;
        for c in 0..3 {
            // DistAbs weight: exp(-dist^2 / (h^2 * channels)); h = 0 ⇒ the
            // division is 0/0 (dist 0) → NaN → 1.0, or -x/0 → -inf → 0.
            let hh = h[c] * h[c] * 3.0;
            let mut w = (-(dist * dist) / hh).exp();
            if w.is_nan() {
                w = 1.0;
            }
            let mut weight = (fixed_point_mult as f64 * w).round_ties_even() as i64;
            if (weight as f64) < threshold {
                weight = 0;
            }
            slot[c] = weight as i32;
        }
    }

    let ctx = NlmCtx {
        control,
        ext: &ext,
        ext_cols,
        rows,
        cols,
        border,
        tr,
        sr,
        tw,
        sw,
        bin_shift,
        lut: &lut,
    };

    let mut out = Img3::new(rows, cols);

    // OpenCV runs the invoker under `parallel_for_(Range(0, rows))`; mirror
    // that with rayon row bands. wasm32 has no thread pool, so run serially.
    #[cfg(not(target_arch = "wasm32"))]
    {
        use rayon::prelude::*;
        let threads = rayon::current_num_threads().max(1);
        let band = ((rows + 4 * threads - 1) / (4 * threads)).max(1);
        out.data
            .par_chunks_mut(band * cols * 3)
            .enumerate()
            .for_each(|(k, chunk)| {
                let row_from = (k * band) as i32;
                let nrows = chunk.len() / (cols * 3);
                ctx.process_rows(row_from, row_from + nrows as i32 - 1, chunk);
            });
    }
    #[cfg(target_arch = "wasm32")]
    {
        ctx.process_rows(0, rows as i32 - 1, &mut out.data);
    }

    control.check()?;
    Ok(out)
}

// ----------------------------------------------------------------------
// medianBlur (single channel, 3×3, BORDER_REPLICATE)
// ----------------------------------------------------------------------

/// Independent rows keep their existing arithmetic order. WASM remains serial.
fn for_rows_mut<T: Send>(data: &mut [T], stride: usize, f: impl Fn(usize, &mut [T]) + Sync + Send) {
    #[cfg(not(target_arch = "wasm32"))]
    {
        use rayon::prelude::*;
        data.par_chunks_mut(stride)
            .enumerate()
            .for_each(|(row, data)| f(row, data));
    }
    #[cfg(target_arch = "wasm32")]
    for (row, data) in data.chunks_mut(stride).enumerate() {
        f(row, data);
    }
}

/// In-place 3×3 median filter of one channel of `img` (OpenCV `medianBlur`
/// with `ksize = 3`, which uses `BORDER_REPLICATE`).
fn median_blur_channel_3x3(img: &mut Img3, ch: usize) {
    let rows = img.rows as i32;
    let cols = img.cols as i32;
    if rows == 0 || cols == 0 {
        return;
    }
    // Snapshot the channel plane so neighborhood reads aren't perturbed by
    // earlier writes.
    let mut plane = vec![0u16; img.rows * img.cols];
    for r in 0..img.rows {
        for c in 0..img.cols {
            plane[r * img.cols + c] = img.data[(r * img.cols + c) * 3 + ch];
        }
    }
    for_rows_mut(&mut img.data, cols as usize * 3, |r, row| {
        let r = r as i32;
        for c in 0..cols {
            let mut vals = [0u16; 9];
            let mut n = 0;
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let yy = (r + dy).clamp(0, rows - 1) as usize;
                    let xx = (c + dx).clamp(0, cols - 1) as usize;
                    vals[n] = plane[yy * cols as usize + xx];
                    n += 1;
                }
            }
            vals.sort_unstable();
            row[c as usize * 3 + ch] = vals[4];
        }
    });
}

// ----------------------------------------------------------------------
// resize (INTER_AREA downscale, INTER_CUBIC upscale)
// ----------------------------------------------------------------------

/// One axis of an area-resample: for each destination index, the source
/// indices it overlaps and their (normalized, summing to 1) weights.
fn build_area_tab(src_len: usize, dst_len: usize, scale: f64) -> Vec<Vec<(usize, f64)>> {
    let mut tab = Vec::with_capacity(dst_len);
    for d in 0..dst_len {
        let f1 = d as f64 * scale;
        let f2 = f1 + scale;
        let mut entries = Vec::new();
        let mut s = f1.floor() as i64;
        let s_end = f2.ceil() as i64;
        while s < s_end {
            let lo = (s as f64).max(f1);
            let hi = ((s + 1) as f64).min(f2);
            let overlap = hi - lo;
            if overlap > 0.0 {
                let si = s.clamp(0, src_len as i64 - 1) as usize;
                entries.push((si, overlap / scale));
            }
            s += 1;
        }
        tab.push(entries);
    }
    tab
}

/// `cv::resize(..., INTER_AREA)` for 3-channel `u16`, producing a
/// `dst_rows × dst_cols` image by area averaging. Separable (columns, then
/// rows) with an `f64` intermediate.
fn resize_area(src: &Img3, dst_rows: usize, dst_cols: usize) -> Img3 {
    let scale_x = src.cols as f64 / dst_cols as f64;
    let scale_y = src.rows as f64 / dst_rows as f64;
    let xtab = build_area_tab(src.cols, dst_cols, scale_x);
    let ytab = build_area_tab(src.rows, dst_rows, scale_y);

    // Horizontal pass: src.rows × dst_cols, kept in f64.
    let mut tmp = vec![0f64; src.rows * dst_cols * 3];
    for_rows_mut(&mut tmp, dst_cols * 3, |sy, row| {
        for (dx, taps) in xtab.iter().enumerate() {
            let mut acc = [0f64; 3];
            for &(sx, wx) in taps {
                let si = (sy * src.cols + sx) * 3;
                acc[0] += wx * src.data[si] as f64;
                acc[1] += wx * src.data[si + 1] as f64;
                acc[2] += wx * src.data[si + 2] as f64;
            }
            row[dx * 3..dx * 3 + 3].copy_from_slice(&acc);
        }
    });

    // Vertical pass: dst_rows × dst_cols → u16.
    let mut out = Img3::new(dst_rows, dst_cols);
    for_rows_mut(&mut out.data, dst_cols * 3, |dy, row| {
        let taps = &ytab[dy];
        for dx in 0..dst_cols {
            let mut acc = [0f64; 3];
            for &(sy, wy) in taps {
                let ti = (sy * dst_cols + dx) * 3;
                acc[0] += wy * tmp[ti];
                acc[1] += wy * tmp[ti + 1];
                acc[2] += wy * tmp[ti + 2];
            }
            row[dx * 3..dx * 3 + 3].copy_from_slice(&acc.map(round_sat_u16));
        }
    });
    out
}

/// One axis of a cubic resample: per destination index, the four (reflected)
/// source taps and their cubic weights.
fn build_cubic_tab(src_len: usize, dst_len: usize, scale: f64) -> Vec<([usize; 4], [f64; 4])> {
    let mut tab = Vec::with_capacity(dst_len);
    for d in 0..dst_len {
        let fx = (d as f64 + 0.5) * scale - 0.5;
        let isx = fx.floor() as i32;
        let t = fx - isx as f64;
        let w = cubic_weights(t);
        let taps = [
            reflect_101(isx - 1, src_len as i32) as usize,
            reflect_101(isx, src_len as i32) as usize,
            reflect_101(isx + 1, src_len as i32) as usize,
            reflect_101(isx + 2, src_len as i32) as usize,
        ];
        tab.push((taps, w));
    }
    tab
}

/// `cv::resize(..., INTER_CUBIC)` for a 3-channel `i16` image (the
/// `CV_16S` low-frequency residual), producing `dst_rows × dst_cols`.
fn resize_cubic_i16(src: &Img3i16, dst_rows: usize, dst_cols: usize) -> Img3i16 {
    let scale_x = src.cols as f64 / dst_cols as f64;
    let scale_y = src.rows as f64 / dst_rows as f64;
    let xtab = build_cubic_tab(src.cols, dst_cols, scale_x);
    let ytab = build_cubic_tab(src.rows, dst_rows, scale_y);

    // Horizontal pass: src.rows × dst_cols in f64.
    let mut tmp = vec![0f64; src.rows * dst_cols * 3];
    for_rows_mut(&mut tmp, dst_cols * 3, |sy, row| {
        for (dx, (taps, w)) in xtab.iter().enumerate() {
            let mut acc = [0f64; 3];
            for k in 0..4 {
                let si = (sy * src.cols + taps[k]) * 3;
                acc[0] += w[k] * src.data[si] as f64;
                acc[1] += w[k] * src.data[si + 1] as f64;
                acc[2] += w[k] * src.data[si + 2] as f64;
            }
            row[dx * 3..dx * 3 + 3].copy_from_slice(&acc);
        }
    });

    // Vertical pass → i16.
    let mut out = Img3i16 {
        rows: dst_rows,
        cols: dst_cols,
        data: vec![0i16; dst_rows * dst_cols * 3],
    };
    for_rows_mut(&mut out.data, dst_cols * 3, |dy, row| {
        let (taps, w) = &ytab[dy];
        for dx in 0..dst_cols {
            let mut acc = [0f64; 3];
            for k in 0..4 {
                let ti = (taps[k] * dst_cols + dx) * 3;
                acc[0] += w[k] * tmp[ti];
                acc[1] += w[k] * tmp[ti + 1];
                acc[2] += w[k] * tmp[ti + 2];
            }
            for c in 0..3 {
                let v = acc[c].round_ties_even();
                row[dx * 3 + c] = v.clamp(-32768.0, 32767.0) as i16;
            }
        }
    });
    out
}

// ----------------------------------------------------------------------
// denoise_nlm — orchestration mirroring x3f_denoise.cpp::denoise_nlm
// ----------------------------------------------------------------------

/// `subtract(sub, sub_dn, sub_res, CV_16S)`: signed per-element difference.
fn subtract_to_i16(sub: &Img3, sub_dn: &Img3) -> Img3i16 {
    let mut out = Img3i16 {
        rows: sub.rows,
        cols: sub.cols,
        data: vec![0i16; sub.data.len()],
    };
    for i in 0..sub.data.len() {
        let d = sub.data[i] as i32 - sub_dn.data[i] as i32;
        out.data[i] = d.clamp(-32768, 32767) as i16;
    }
    out
}

/// `subtract(out, res, out, CV_16U)`: `out -= res`, saturating to `u16`.
fn subtract_u16_inplace(out: &mut Img3, res: &Img3i16) {
    for i in 0..out.data.len() {
        let v = out.data[i] as i32 - res.data[i] as i32;
        out.data[i] = v.clamp(0, 65535) as u16;
    }
}

/// Port of `denoise_nlm(Mat& img, float h)`:
///   1. NLM on all channels with `h = {0, h, h}` (template 3, search 11);
///   2. 3×3 median on the V channel (channel 2);
///   3. low-frequency pass: ÷4 area-downscale → NLM `{0, h/8, h/4}`
///      (search 21) → signed residual → cubic upscale → subtract.
fn denoise_nlm(img: &mut Img3, h: f64, control: Control<'_>) -> crate::Result<()> {
    control.check()?;
    let options = control.denoise_options;
    // 1. main NLM.
    let mut out = configured_nlm(img, [h; 3], control)?;

    // 2. V-channel median.
    if options.chroma {
        median_blur_channel_3x3(&mut out, 2);
    }

    // 3. low-frequency denoising.
    let sub_rows = ((out.rows as f64) * 0.25).round_ties_even() as usize;
    let sub_cols = ((out.cols as f64) * 0.25).round_ties_even() as usize;
    if options.chroma && sub_rows >= 1 && sub_cols >= 1 {
        let sub = resize_area(&out, sub_rows, sub_cols);
        let sub_dn = fast_nlm_denoise_controlled(
            &sub,
            [0.0, h / 8.0, h / 4.0],
            options.nlm.chroma_patch as i32,
            21,
            control,
        )?;
        let sub_res = subtract_to_i16(&sub, &sub_dn);
        let res = resize_cubic_i16(&sub_res, out.rows, out.cols);
        subtract_u16_inplace(&mut out, &res);
    }

    preserve_disabled_channels(img, &mut out, options);
    control.check()?;
    *img = out;
    Ok(())
}

fn channel_strengths(mut h: [f64; 3], options: DenoiseOptions) -> [f64; 3] {
    if !options.luminance {
        h[0] = 0.0;
    }
    if !options.chroma {
        h[1] = 0.0;
        h[2] = 0.0;
    }
    h
}

fn denoise_bm3d(img: &Img3, scale: f32, control: Control<'_>) -> crate::Result<Img3> {
    let options = control.denoise_options;
    let noise = bm3d::estimate_noise(img, control)?;
    let sigma = channel_strengths(
        noise.map(|s| s * scale as f64 * options.strength as f64),
        options,
    );
    log(
        crate::x3f_verbosity_t_DEBUG,
        &format!("BM3D estimated sigma Y/U/V: {sigma:?}"),
    );
    bm3d::denoise(img, sigma, control)
}

fn preserve_disabled_channels(src: &Img3, out: &mut Img3, options: DenoiseOptions) {
    for (source, result) in src.data.chunks_exact(3).zip(out.data.chunks_exact_mut(3)) {
        if !options.luminance {
            result[0] = source[0];
        }
        if !options.chroma {
            result[1..].copy_from_slice(&source[1..]);
        }
    }
}

fn configured_nlm(img: &Img3, h: [f64; 3], control: Control<'_>) -> crate::Result<Img3> {
    let options = control.denoise_options;
    let h = channel_strengths(h, options);
    let nlm = options.nlm;
    let mut out = if options.chroma {
        let same = nlm.luma_patch == nlm.chroma_patch && nlm.luma_search == nlm.chroma_search;
        let strengths = if same { h } else { [0.0, h[1], h[2]] };
        let mut out = fast_nlm_denoise_controlled(
            img,
            strengths,
            nlm.chroma_patch as i32,
            nlm.chroma_search as i32,
            control,
        )?;
        if options.luminance && !same {
            let luma = fast_nlm_denoise_controlled(
                img,
                [h[0], 0.0, 0.0],
                nlm.luma_patch as i32,
                nlm.luma_search as i32,
                control,
            )?;
            for (result, source) in out.data.chunks_exact_mut(3).zip(luma.data.chunks_exact(3)) {
                result[0] = source[0];
            }
        }
        out
    } else if options.luminance {
        fast_nlm_denoise_controlled(
            img,
            h,
            nlm.luma_patch as i32,
            nlm.luma_search as i32,
            control,
        )?
    } else {
        img.clone()
    };
    preserve_disabled_channels(img, &mut out, options);
    Ok(out)
}

// ----------------------------------------------------------------------
// BMT ↔ YUV transforms (port of x3f_denoise_utils.cpp)
// ----------------------------------------------------------------------

/// `BMT_to_YUV_*` in place over an `Area16`. Writes `(Y, U+O_UV, V+O_UV)`.
///
/// # Safety
/// `area` must be a valid writable 3-channel `x3f_area16_t`.
unsafe fn bmt_to_yuv(area: *mut Area16, dt: DType) {
    let a = unsafe { &*area };
    let stride = a.row_stride as usize;
    for row in 0..a.rows as usize {
        let rp = unsafe { a.data.add(row * stride) };
        for col in 0..a.columns as usize {
            let p = unsafe { rp.add(col * 3) };
            let b = unsafe { *p } as i32;
            let m = unsafe { *p.add(1) } as i32;
            let t = unsafe { *p.add(2) } as i32;

            let y = match dt {
                DType::Std => (b + m + t) / 3,
                DType::YisT => t,
                DType::Yis4T => 4 * t,
            };
            let u = 2 * b - 2 * t;
            let v = b - 2 * m + t;

            unsafe { *p = sat_u16_i32(y) };
            unsafe { *p.add(1) = sat_u16_i32(u + O_UV) };
            unsafe { *p.add(2) = sat_u16_i32(v + O_UV) };
        }
    }
}

/// `YUV_to_BMT_*` in place over an `Area16` (the inverse of [`bmt_to_yuv`]).
///
/// # Safety
/// `area` must be a valid writable 3-channel `x3f_area16_t`.
unsafe fn yuv_to_bmt(area: *mut Area16, dt: DType) {
    let a = unsafe { &*area };
    let stride = a.row_stride as usize;
    for row in 0..a.rows as usize {
        let rp = unsafe { a.data.add(row * stride) };
        for col in 0..a.columns as usize {
            let p = unsafe { rp.add(col * 3) };
            let y = unsafe { *p } as i32;
            let u = unsafe { *p.add(1) } as i32 - O_UV;
            let v = unsafe { *p.add(2) } as i32 - O_UV;

            let (b, m, t) = match dt {
                DType::Std => (
                    (12 * y + 3 * u + 2 * v) / 12,
                    (3 * y - v) / 3,
                    (12 * y - 3 * u + 2 * v) / 12,
                ),
                DType::YisT => ((2 * y + u) / 2, (4 * y + u - 2 * v) / 4, y),
                DType::Yis4T => ((y + 2 * u + 2) / 4, (y + u - 2 * v + 2) / 4, (y + 2) / 4),
            };

            unsafe { *p = sat_u16_i32(b) };
            unsafe { *p.add(1) = sat_u16_i32(m) };
            unsafe { *p.add(2) = sat_u16_i32(t) };
        }
    }
}

// ----------------------------------------------------------------------
// Entry points — called directly by the pipeline (`run_denoising` in
// process.rs and the two Quattro passes in quattro.rs).
// ----------------------------------------------------------------------

/// Denoise B, M and T channels with the Quattro profile. Other cameras are unsupported.
/// Width and height use pixels. Row stride and slice length use u16 samples, not bytes.
/// The slice length must equal height times row stride. Width times channels must
/// not exceed row stride. Row padding is never changed.
/// Scale multiplies `DenoiseOptions::strength`. Noise levels use 16-bit camera codes.
/// Adjust scale for other code ranges. For X3I samples divided by S, divide scale by S.
/// The bridge compresses all channels by four before filtering to avoid YUV overflow.
/// It adds the filtered residual to the original codes in steps of four.
/// This can cause bands in deep shadows. At X3I S=11.4, one code is about
/// 0.00017 reference units and can be close to one noise standard deviation.
/// Zero scale or disabled luminance and chroma leaves the data unchanged.
/// Cancellation returns `Error::Cancelled` and leaves the input unchanged.
/// Zero dimensions, channels other than three, invalid stride or length, arithmetic
/// overflow, invalid denoise options, or a negative or nonfinite scale returns
/// `Error::InvalidData`. Backend errors are returned by the selected kernel.
pub fn denoise_u16(
    data: &mut [u16],
    width: u32,
    height: u32,
    channels: u32,
    row_stride: u32,
    scale: f32,
    control: Control<'_>,
) -> crate::Result<()> {
    control.check()?;
    let logical = width.checked_mul(channels);
    let length = (height as usize).checked_mul(row_stride as usize);
    if width == 0
        || height == 0
        || channels != 3
        || logical.is_none_or(|n| n > row_stride)
        || length != Some(data.len())
        || !scale.is_finite()
        || scale < 0.0
    {
        return Err(crate::Error::InvalidData("invalid u16 denoise raster"));
    }
    control
        .denoise_options
        .validate()
        .map_err(crate::Error::InvalidData)?;
    if scale == 0.0 || (!control.denoise_options.luminance && !control.denoise_options.chroma) {
        return Ok(());
    }
    let mut compressed: Vec<u16> = data.iter().map(|v| v / 4).collect();
    let mut area = Area16 {
        data: compressed.as_mut_ptr(),
        buf: std::ptr::null_mut(),
        rows: height,
        columns: width,
        channels,
        row_stride,
    };
    // SAFETY: the checks above cover each row. The kernel does not free this buffer.
    unsafe { denoise_area_controlled(&mut area, DENOISE_F23, scale / 4.0, control) }?;
    control.check()?;
    for (row, filtered) in data
        .chunks_exact_mut(row_stride as usize)
        .zip(compressed.chunks_exact(row_stride as usize))
    {
        for (original, filtered) in row[..logical.unwrap() as usize].iter_mut().zip(filtered) {
            let residual = (i32::from(*filtered) - i32::from(*original / 4)) * 4;
            *original = sat_u16_i32(i32::from(*original) + residual);
        }
    }
    Ok(())
}

/// BMT→YUV, full `denoise_nlm`, YUV→BMT (was the OpenCV `x3f_denoise`).
///
/// # Safety
/// `image` must be null or a valid 3-channel `x3f_area16_t`.
#[cfg(test)]
pub(crate) unsafe fn denoise_area(image: *mut Area16, dtype: u32, scale: f32) {
    let _ = unsafe { denoise_area_controlled(image, dtype, scale, Control::none()) };
}

pub(crate) unsafe fn denoise_area_controlled(
    image: *mut Area16,
    dtype: u32,
    scale: f32,
    control: Control<'_>,
) -> crate::Result<()> {
    control.check()?;
    if image.is_null()
        || scale <= 0.0
        || (!control.denoise_options.luminance && !control.denoise_options.chroma)
    {
        return Ok(());
    }
    debug_assert_eq!(unsafe { (*image).channels }, 3);
    #[cfg(not(target_arch = "wasm32"))]
    let started = std::time::Instant::now();
    let dt = DType::from_u32(dtype);
    let h = dt.base_h() * scale as f64 * control.denoise_options.strength as f64;

    let mut img = unsafe { area_to_img3(image) };
    // Work on owned storage: cancellation or a failed GPU attempt must never
    // leave the caller's sensor buffer in YUV or partially denoised.
    unsafe { bmt_to_yuv(&mut img.area(), dt) };
    if control.denoise_options.algorithm == DenoiseAlgorithm::Bm3d {
        img = denoise_bm3d(&img, scale, control)?;
    } else {
        denoise_nlm(&mut img, h, control)?;
    }
    unsafe { yuv_to_bmt(&mut img.area(), dt) };
    control.check()?;
    unsafe { img3_to_area(&img, image) };
    #[cfg(not(target_arch = "wasm32"))]
    log(
        crate::x3f_verbosity_t_DEBUG,
        &format!(
            "Denoise stage: sensor, {}x{}, {:.3} ms",
            img.cols,
            img.rows,
            started.elapsed().as_secs_f64() * 1000.0
        ),
    );
    Ok(())
}

/// Active-area Quattro pass: the area is already YUV (was the OpenCV
/// `x3f_denoise_active`).
/// `stage == 0` runs the full `denoise_nlm`; `stage != 0` runs only the
/// full-resolution NLM with per-channel `{0, h, 2h}` (template 3, search 11).
///
/// # Safety
/// `area` must be null or a valid 3-channel `x3f_area16_t`.
pub(crate) unsafe fn denoise_active_area(
    area: *mut Area16,
    dtype: u32,
    stage: i32,
    scale: f32,
    control: Control<'_>,
) -> crate::Result<()> {
    control.check()?;
    if area.is_null()
        || scale <= 0.0
        || (!control.denoise_options.luminance && !control.denoise_options.chroma)
    {
        return Ok(());
    }
    debug_assert_eq!(unsafe { (*area).channels }, 3);
    #[cfg(not(target_arch = "wasm32"))]
    let started = std::time::Instant::now();
    let dt = DType::from_u32(dtype);
    let sigma = dt.base_h() * scale as f64 * control.denoise_options.strength as f64;

    if control.denoise_options.algorithm == DenoiseAlgorithm::Bm3d {
        let img = unsafe { area_to_img3(area) };
        let out = denoise_bm3d(&img, scale, control)?;
        control.check()?;
        unsafe { img3_to_area(&out, area) };
    } else if stage == 0 {
        let mut img = unsafe { area_to_img3(area) };
        denoise_nlm(&mut img, sigma, control)?;
        control.check()?;
        unsafe { img3_to_area(&img, area) };
    } else {
        let img = unsafe { area_to_img3(area) };
        let out = configured_nlm(&img, [sigma, sigma, sigma * 2.0], control)?;
        control.check()?;
        unsafe { img3_to_area(&out, area) };
    }
    #[cfg(not(target_arch = "wasm32"))]
    log(
        crate::x3f_verbosity_t_DEBUG,
        &format!(
            "Denoise stage: quattro-{stage}, {}x{}, {:.3} ms",
            unsafe { (*area).columns },
            unsafe { (*area).rows },
            started.elapsed().as_secs_f64() * 1000.0
        ),
    );
    Ok(())
}

// ----------------------------------------------------------------------
// Tests (run on the host regardless of target gating)
// ----------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn u16_bridge_validates_and_preserves_cancelled_or_disabled_input() {
        use std::sync::atomic::AtomicBool;
        let source: Vec<u16> = (0..8 * 8 * 3)
            .map(|i| 3000 + (i * 71 % 200) as u16)
            .collect();
        let control = Control::none().with_denoise_backend(DenoiseBackend::Cpu);
        let mut data = source.clone();
        denoise_u16(&mut data, 8, 8, 3, 24, 0.0, control).unwrap();
        assert_eq!(data, source);
        for algorithm in [DenoiseAlgorithm::Nlm, DenoiseAlgorithm::Bm3d] {
            let options = DenoiseOptions {
                algorithm,
                ..Default::default()
            };
            let selected = control.with_denoise_options(options).unwrap();
            denoise_u16(&mut data, 8, 8, 3, 24, 0.3, selected).unwrap();
            assert!(data.iter().all(|&v| (1000..6000).contains(&v)));
        }
        let before = data.clone();
        let cancel = AtomicBool::new(true);
        assert!(matches!(
            denoise_u16(&mut data, 8, 8, 3, 24, 0.3, Control::new(&cancel)),
            Err(crate::Error::Cancelled)
        ));
        assert_eq!(data, before);
        for (width, height, channels, stride) in [
            (0, 8, 3, 24),
            (8, 0, 3, 24),
            (8, 8, 2, 24),
            (9, 8, 3, 24),
            (8, 9, 3, 24),
        ] {
            assert!(matches!(
                denoise_u16(&mut data, width, height, channels, stride, 0.3, control),
                Err(crate::Error::InvalidData(_))
            ));
        }
        assert!(denoise_u16(&mut data, 8, 8, 3, 24, f32::NAN, control).is_err());
        let mut padded = vec![60000; 8 * 30];
        for (row, input) in padded.chunks_exact_mut(30).zip(source.chunks_exact(24)) {
            row[..24].copy_from_slice(input);
        }
        denoise_u16(&mut padded, 8, 8, 3, 30, 0.3, control).unwrap();
        for row in padded.chunks_exact(30) {
            assert!(row[..24].iter().all(|&v| (1000..6000).contains(&v)));
            assert_eq!(&row[24..], &[60000; 6]);
        }
        for (length, stride) in [(8 * 23, 23), (8 * 24 + 1, 24)] {
            let mut invalid = vec![3000; length];
            assert!(matches!(
                denoise_u16(&mut invalid, 8, 8, 3, stride, 0.3, control),
                Err(crate::Error::InvalidData(_))
            ));
        }
    }

    #[test]
    fn u16_bridge_preserves_full_range_and_smooth_shadows() {
        let control = Control::none().with_denoise_backend(DenoiseBackend::Cpu);
        for algorithm in [DenoiseAlgorithm::Nlm, DenoiseAlgorithm::Bm3d] {
            let control = control
                .with_denoise_options(DenoiseOptions {
                    algorithm,
                    ..Default::default()
                })
                .unwrap();
            let mut bright = vec![60001; 8 * 8 * 3];
            denoise_u16(&mut bright, 8, 8, 3, 24, 0.3, control).unwrap();
            assert_eq!(bright, vec![60001; 8 * 8 * 3]);
            let mut dark: Vec<u16> = (0..8)
                .flat_map(|_| (0..32).flat_map(|x| [256 + x; 3]))
                .collect();
            let original = dark.clone();
            denoise_u16(&mut dark, 32, 8, 3, 96, 0.0, control).unwrap();
            assert_eq!(dark, original);
            denoise_u16(&mut dark, 32, 8, 3, 96, 0.3, control).unwrap();
            for row in dark.chunks_exact(96) {
                for pair in row.chunks_exact(3).collect::<Vec<_>>().windows(2) {
                    for c in 0..3 {
                        assert!(pair[0][c].abs_diff(pair[1][c]) <= 4);
                    }
                }
            }
        }
    }

    #[test]
    fn advanced_settings_validate_and_disabled_planes_are_exact() {
        for strength in [f32::NAN, f32::INFINITY, 0.0, 3.1] {
            assert!(DenoiseOptions {
                strength,
                ..Default::default()
            }
            .validate()
            .is_err());
        }
        for size in [0, 2, 10, 255] {
            let mut options = DenoiseOptions::default();
            options.nlm.luma_patch = size;
            assert!(options.validate().is_err());
            options.nlm = NlmOptions::default();
            options.nlm.chroma_search = size;
            assert!(options.validate().is_err());
        }
        let mut source = Img3::new(17, 19);
        for (i, pixel) in source.data.chunks_exact_mut(3).enumerate() {
            pixel.copy_from_slice(&[
                20000 + ((i * 71) % 200) as u16,
                32000 + ((i * 37) % 200) as u16,
                33000 + ((i * 23) % 200) as u16,
            ]);
        }
        for (luminance, chroma) in [(false, true), (true, false), (true, true), (false, false)] {
            let options = DenoiseOptions {
                luminance,
                chroma,
                nlm: NlmOptions {
                    luma_patch: 5,
                    luma_search: 7,
                    chroma_patch: 3,
                    chroma_search: 9,
                },
                ..Default::default()
            };
            let control = Control::none()
                .with_denoise_backend(DenoiseBackend::Cpu)
                .with_denoise_options(options)
                .unwrap();
            let mut out = source.clone();
            denoise_nlm(&mut out, 100.0, control).unwrap();
            for channel in 0..3 {
                let changed = source
                    .data
                    .chunks_exact(3)
                    .zip(out.data.chunks_exact(3))
                    .any(|(a, b)| a[channel] != b[channel]);
                assert_eq!(changed, if channel == 0 { luminance } else { chroma });
            }
        }
        let mut raw = source.clone();
        let control = Control::none()
            .with_denoise_options(DenoiseOptions {
                luminance: false,
                chroma: false,
                ..Default::default()
            })
            .unwrap();
        unsafe {
            denoise_area_controlled(&mut raw.area(), DENOISE_STD, 1.0, control).unwrap();
        }
        assert_eq!(raw.data, source.data);
    }

    #[test]
    fn cancellation_preserves_sensor_buffer_without_cancelling_another_call() {
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::time::{Duration, Instant};

        let cancel = AtomicBool::new(false);
        let mut data = vec![12000; 1024 * (1024 * 3 + 4)];
        let original = data.clone();
        let mut area = Area16 {
            data: data.as_mut_ptr(),
            buf: std::ptr::null_mut(),
            rows: 1024,
            columns: 1024,
            channels: 3,
            row_stride: 1024 * 3 + 4,
        };
        let started = Instant::now();
        std::thread::scope(|scope| {
            scope.spawn(|| {
                std::thread::sleep(Duration::from_millis(1));
                cancel.store(true, Ordering::Relaxed);
            });
            let result = unsafe {
                denoise_area_controlled(&mut area, DENOISE_STD, 1.0, Control::new(&cancel))
            };
            assert!(matches!(result, Err(crate::Error::Cancelled)));
        });
        assert!(started.elapsed() < Duration::from_secs(5));
        assert_eq!(
            data, original,
            "cancelled stage must preserve sensor data and padding"
        );
        let other = Img3::new(8, 8);
        let result =
            fast_nlm_denoise_controlled(&other, [0.0, 80.0, 80.0], 3, 11, Control::none()).unwrap();
        assert_eq!(result.data, other.data);
    }

    #[test]
    fn cancellation_interrupts_cpu_inside_a_row() {
        use std::sync::atomic::{AtomicBool, Ordering};
        let cancel = AtomicBool::new(false);
        let cols = 129;
        let rows = 4;
        let ext = vec![500; (rows + 12) * (cols + 12) * 3];
        let ctx = NlmCtx {
            control: Control::new(&cancel),
            ext: &ext,
            ext_cols: cols + 12,
            rows,
            cols,
            border: 6,
            tr: 1,
            sr: 5,
            tw: 3,
            sw: 11,
            bin_shift: 4,
            lut: &[[i32::MAX; 3]],
        };
        let mut out = vec![0; rows * cols * 3];
        let mut polls = 0;
        // The third poll is at column 64, after real NLM work. This does not
        // depend on thread scheduling or time spent copying the sensor data.
        ctx.process_rows_until(0, rows as i32 - 1, &mut out, || {
            polls += 1;
            if polls == 3 {
                cancel.store(true, Ordering::Relaxed);
            }
            ctx.control.check().is_err()
        });
        assert!(out[..64 * 3].iter().all(|&value| value == 500));
        assert!(out[64 * 3..].iter().all(|&value| value == 0));
        let other = NlmCtx {
            control: Control::none(),
            ..ctx
        };
        other.process_rows(0, rows as i32 - 1, &mut out);
        assert!(out.iter().all(|&value| value == 500));
    }

    #[test]
    fn failed_gpu_request_preserves_input_and_backend_selection() {
        let mut image = Img3::new(7, 13);
        for (i, value) in image.data.iter_mut().enumerate() {
            *value = 32700 + (i % 101) as u16;
        }
        let original = image.data.clone();
        let h = [0.0, 100.0, 100.0];
        let cpu = fast_nlm_cpu(&image, h, 3, 11, Control::none()).unwrap();
        for backend in [
            DenoiseBackend::Cpu,
            DenoiseBackend::Auto,
            DenoiseBackend::Gpu,
        ] {
            let result = fast_nlm_with_gpu(
                &image,
                h,
                3,
                11,
                Control::none().with_denoise_backend(backend),
                || {
                    assert_ne!(
                        backend,
                        DenoiseBackend::Cpu,
                        "CPU selection initialized a GPU"
                    );
                    Err(crate::Error::Gpu("injected device failure".into()))
                },
            );
            if backend == DenoiseBackend::Gpu {
                assert!(
                    matches!(result, Err(crate::Error::Gpu(message)) if message == "injected device failure")
                );
            } else {
                assert_eq!(result.unwrap().data, cpu.data);
            }
            assert_eq!(image.data, original);
        }
        let cancelled = fast_nlm_with_gpu(&image, h, 3, 11, Control::none(), || {
            Err(crate::Error::Cancelled)
        });
        assert!(matches!(cancelled, Err(crate::Error::Cancelled)));
    }

    /// Tiny deterministic LCG so tests don't need an RNG crate.
    struct Lcg(u64);
    impl Lcg {
        fn next_u16(&mut self) -> u16 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 33) as u16
        }
        /// Signed noise in `[-amp, amp]`.
        fn noise(&mut self, amp: i32) -> i32 {
            (self.next_u16() as i32 % (2 * amp + 1)) - amp
        }
    }

    #[test]
    fn reflect_101_matches_opencv() {
        // gfedcb|abcdefgh|gfedcba, n = 5: indices ... map symmetrically.
        assert_eq!(reflect_101(-1, 5), 1);
        assert_eq!(reflect_101(-2, 5), 2);
        assert_eq!(reflect_101(0, 5), 0);
        assert_eq!(reflect_101(4, 5), 4);
        assert_eq!(reflect_101(5, 5), 3);
        assert_eq!(reflect_101(6, 5), 2);
        // Far out of range (search radius can exceed a small image).
        assert!((0..5).contains(&reflect_101(-20, 5)));
        assert!((0..5).contains(&reflect_101(99, 5)));
        assert_eq!(reflect_101(7, 1), 0);
    }

    #[test]
    fn nearest_pow2_known() {
        assert_eq!(nearest_pow2(9), 4); // template_window_size^2 for tw=3
        assert_eq!(nearest_pow2(1), 0);
        assert_eq!(nearest_pow2(16), 4);
        assert_eq!(nearest_pow2(17), 5);
    }

    #[test]
    fn area_downscale_box_average() {
        // 4×4 constant block → 1×1 same value.
        let mut img = Img3::new(4, 4);
        for px in img.data.chunks_mut(3) {
            px[0] = 1000;
            px[1] = 2000;
            px[2] = 3000;
        }
        let out = resize_area(&img, 1, 1);
        assert_eq!(out.data, vec![1000, 2000, 3000]);

        // 2×2 with distinct values → average.
        let mut g = Img3::new(2, 2);
        let vals = [0u16, 100, 200, 300];
        for (i, &v) in vals.iter().enumerate() {
            g.data[i * 3] = v;
        }
        let avg = resize_area(&g, 1, 1);
        assert_eq!(avg.data[0], 150); // (0+100+200+300)/4
    }

    #[test]
    fn nlm_preserves_constant_image() {
        let mut img = Img3::new(20, 20);
        for px in img.data.chunks_mut(3) {
            px[0] = 12345;
            px[1] = 23456;
            px[2] = 34567;
        }
        let out = fast_nlm_denoise(&img, [0.0, 80.0, 80.0], 3, 11);
        // A flat field must come back unchanged (all weights average equal
        // values; channel 0 with h=0 keeps its self-match).
        assert_eq!(out.data, img.data);
    }

    #[test]
    fn nlm_reduces_chroma_noise() {
        // Flat chroma base + zero-mean noise on channels 1/2; channel 0 flat.
        // The noise amplitude must be modest relative to `h` (the NLM weight
        // decays as exp(-dist^2 / (h^2 * 3)), so patches far apart in value
        // get ~zero weight and nothing is averaged).
        let (rows, cols) = (40usize, 40usize);
        let mut noisy = Img3::new(rows, cols);
        let base = [20000i32, 30000, 40000];
        let mut rng = Lcg(0xC0FFEE);
        for px in noisy.data.chunks_mut(3) {
            px[0] = base[0] as u16;
            px[1] = (base[1] + rng.noise(150)).clamp(0, 65535) as u16;
            px[2] = (base[2] + rng.noise(150)).clamp(0, 65535) as u16;
        }

        let out = fast_nlm_denoise(&noisy, [0.0, 300.0, 300.0], 3, 11);

        // Variance of channel 1 must drop and the mean must be ~preserved.
        let var = |im: &Img3, ch: usize, mean: f64| {
            let n = (rows * cols) as f64;
            im.data
                .chunks(3)
                .map(|p| {
                    let d = p[ch] as f64 - mean;
                    d * d
                })
                .sum::<f64>()
                / n
        };
        let mean = |im: &Img3, ch: usize| {
            im.data.chunks(3).map(|p| p[ch] as f64).sum::<f64>() / (rows * cols) as f64
        };

        let m_in = mean(&noisy, 1);
        let m_out = mean(&out, 1);
        assert!((m_in - m_out).abs() < 200.0, "mean drift {m_in} -> {m_out}");
        let v_in = var(&noisy, 1, m_in);
        let v_out = var(&out, 1, m_out);
        assert!(
            v_out < v_in * 0.6,
            "variance not reduced: {v_in} -> {v_out}"
        );
    }

    #[test]
    fn median_removes_isolated_outlier() {
        let mut img = Img3::new(5, 5);
        for px in img.data.chunks_mut(3) {
            px[2] = 1000;
        }
        // Spike at the center of the V channel.
        img.data[(2 * 5 + 2) * 3 + 2] = 60000;
        median_blur_channel_3x3(&mut img, 2);
        assert_eq!(img.data[(2 * 5 + 2) * 3 + 2], 1000);
    }

    #[test]
    fn bmt_yuv_roundtrip_approx() {
        // YisT / Yis4T round-trip is exact for representable values; STD loses
        // a little to integer division. Values stay ≤16383 so the Yis4T
        // luma `Y = 4*T` doesn't saturate `u16` (which would be unrecoverable
        // — and is the documented limit of that transform).
        for dt in [DType::Std, DType::YisT, DType::Yis4T] {
            let mut data: Vec<u16> = vec![5000, 12000, 8000, 9000, 11000, 13000];
            let orig = data.clone();
            let mut area = Area16 {
                data: data.as_mut_ptr(),
                buf: std::ptr::null_mut(),
                rows: 1,
                columns: 2,
                channels: 3,
                row_stride: 6,
            };
            unsafe {
                bmt_to_yuv(&mut area, dt);
                yuv_to_bmt(&mut area, dt);
            }
            for (o, r) in orig.iter().zip(data.iter()) {
                let diff = (*o as i32 - *r as i32).abs();
                assert!(diff <= 4, "{dt:?} roundtrip diff {diff} ({o} -> {r})");
            }
        }
    }

    #[test]
    fn denoise_area_smoke() {
        // A near-flat patch with mild noise; denoise must run end-to-end and
        // leave values within range and close to the (flat) original.
        let (rows, cols) = (24usize, 24usize);
        let mut data = vec![0u16; rows * cols * 3];
        let mut rng = Lcg(0x1234_5678);
        for px in data.chunks_mut(3) {
            px[0] = (20000 + rng.noise(300)).clamp(0, 65535) as u16;
            px[1] = (20000 + rng.noise(300)).clamp(0, 65535) as u16;
            px[2] = (20000 + rng.noise(300)).clamp(0, 65535) as u16;
        }
        let mut area = Area16 {
            data: data.as_mut_ptr(),
            buf: std::ptr::null_mut(),
            rows: rows as u32,
            columns: cols as u32,
            channels: 3,
            row_stride: (cols * 3) as u32,
        };
        unsafe { denoise_area(&mut area, 0, 1.0) };

        // Mean should be roughly preserved (denoise of a flat field).
        let mean = data.iter().map(|&v| v as f64).sum::<f64>() / data.len() as f64;
        assert!((mean - 20000.0).abs() < 500.0, "mean drifted to {mean}");
    }

    #[test]
    fn cancelled_denoising_preserves_strided_sensor_buffer() {
        use std::sync::atomic::AtomicBool;
        let mut data = vec![12000; 7 * 10];
        let original = data.clone();
        let mut area = Area16 {
            data: data.as_mut_ptr(),
            buf: std::ptr::null_mut(),
            rows: 7,
            columns: 2,
            channels: 3,
            row_stride: 10,
        };
        let cancel = AtomicBool::new(true);
        let result =
            unsafe { denoise_area_controlled(&mut area, DENOISE_STD, 1.0, Control::new(&cancel)) };
        assert!(matches!(result, Err(crate::Error::Cancelled)));
        assert_eq!(data, original);
    }

    #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
    #[test]
    #[ignore = "requires a physical GPU; run explicitly on GPU test hosts"]
    fn gpu_matches_cpu_at_edges_and_tile_seams() {
        let control = Control::none().with_denoise_backend(DenoiseBackend::Gpu);
        for (rows, cols) in [(1, 1), (1, 29), (23, 1), (31, 43), (1037, 1049)] {
            for (search, h) in [(11, [0.0, 300.0, 600.0]), (21, [0.0, 37.5, 75.0])] {
                let mut rng = Lcg(12345);
                let mut img = Img3::new(rows, cols);
                // Coarse-pass noise must be low enough to admit neighbours,
                // otherwise every weight except the self-match would be zero.
                let divisor = if search == 21 { 8 } else { 1 };
                for (i, p) in img.data.chunks_exact_mut(3).enumerate() {
                    let edge = if i % cols < cols / 2 { 12000 } else { 58000 };
                    p[0] = (edge + rng.noise(100 / divisor)).clamp(0, 65535) as u16;
                    p[1] = (32768 + rng.noise(250 / divisor)) as u16;
                    p[2] = (32768 + rng.noise(200 / divisor)) as u16;
                }
                let expected = fast_nlm_cpu(&img, h, 3, search, Control::none()).unwrap();
                let actual = fast_nlm_denoise_controlled(&img, h, 3, search, control).unwrap();
                let max_error = expected
                    .data
                    .iter()
                    .zip(&actual.data)
                    .map(|(&a, &b)| a.abs_diff(b))
                    .max()
                    .unwrap();
                assert!(
                    max_error <= 1,
                    "{cols}x{rows}, search={search}: max error {max_error}"
                );
                let repeat = fast_nlm_denoise_controlled(&img, h, 3, search, control).unwrap();
                assert_eq!(actual.data, repeat.data);
            }
        }
    }

    #[test]
    fn unavailable_gpu_falls_back_only_in_auto_mode() {
        let image = Img3::new(7, 13);
        // Unsupported GPU window also exercises error fallback on GPU hosts.
        // CPU-only/WASM builds reject any forced GPU request before dispatch.
        let cpu = fast_nlm_cpu(&image, [0.0, 100.0, 100.0], 11, 7, Control::none()).unwrap();
        let auto = fast_nlm_denoise_controlled(&image, [0.0, 100.0, 100.0], 11, 7, Control::none())
            .unwrap();
        assert_eq!(cpu.data, auto.data);
        let forced = fast_nlm_denoise_controlled(
            &image,
            [0.0, 100.0, 100.0],
            11,
            7,
            Control::none().with_denoise_backend(DenoiseBackend::Gpu),
        );
        assert!(matches!(forced, Err(crate::Error::Gpu(_))));
    }
}
