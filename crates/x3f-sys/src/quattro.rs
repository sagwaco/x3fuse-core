//! Native Rust port of `x3f_expand_quattro`, replacing the M0 stub that
//! exited with code 2.
//!
//! Quattro sensors store the top (T) plane at full resolution and the middle
//! (M) and bottom (B) planes at half-resolution in each axis. This routine
//! reconstructs a full-resolution 3-channel BMT image by:
//!
//!  1. Converting the half-res BMT image to YUV (Yis4T variant: Y = 4*T).
//!  2. Bicubically upsampling 2x to the full-res grid.
//!  3. Replacing the upsampled Y plane with `qtop * 4` so the high-frequency
//!     T detail comes from the actual full-res sensor data, and only the
//!     chroma comes from the bilinear lift.
//!  4. Converting back to BMT.
//!
//! The original C++ used OpenCV's `INTER_CUBIC` resize. We implement the same
//! Catmull-Rom-with-a=-0.75 cubic kernel and BORDER_REFLECT_101 edge handling
//! that OpenCV uses; output is not bit-exact but visually indistinguishable.
//! No legacy Rust-build path exists to differ from (the M0 stub blocked
//! Quattro entirely).
//!
//! Symbol export: `x3f_expand_quattro` is `#[no_mangle] extern "C"`. The
//! legacy C call site in `src/x3f_process.c` resolves to this Rust function
//! at link time, after `denoise_stub.c` was edited to drop its stub.

use std::slice;

/// Mirror of `x3f_area16_t` from `x3f_io.h`. We accept raw pointers to the
/// C struct rather than reusing the bindgen-generated type because this
/// module is `pub mod quattro;` inside `x3f-sys` and including the bindgen
/// types here triggers a circular-include problem (the bindings live in
/// `OUT_DIR` and `include!` is at the crate root).
#[repr(C)]
pub(crate) struct Area16 {
    data: *mut u16,
    buf: *mut std::os::raw::c_void,
    rows: u32,
    columns: u32,
    channels: u32,
    row_stride: u32,
}

/// To match `denoise_utils.cpp`'s `O_UV` constant: a bias added to the U/V
/// planes so they fit in unsigned u16.
const O_UV: i32 = 32768;

/// Catmull-Rom cubic kernel with a=-0.75 (OpenCV's `INTER_CUBIC` default).
///
/// Returns four weights for taps at floor(x)-1, floor(x), floor(x)+1,
/// floor(x)+2 given the fractional offset `t = x - floor(x)` in [0, 1).
#[inline]
fn cubic_weights(t: f64) -> [f64; 4] {
    const A: f64 = -0.75;
    let t1 = t;
    let t2 = 1.0 - t;
    // Distances: |x_tap - x| for each of the four taps
    let d0 = 1.0 + t1; // tap at floor-1
    let d1 = t1; // tap at floor
    let d2 = t2; // tap at floor+1
    let d3 = 1.0 + t2; // tap at floor+2

    [
        cubic_kernel(d0, A),
        cubic_kernel(d1, A),
        cubic_kernel(d2, A),
        cubic_kernel(d3, A),
    ]
}

#[inline]
fn cubic_kernel(d: f64, a: f64) -> f64 {
    let d = d.abs();
    if d <= 1.0 {
        ((a + 2.0) * d - (a + 3.0)) * d * d + 1.0
    } else if d < 2.0 {
        ((a * d - 5.0 * a) * d + 8.0 * a) * d - 4.0 * a
    } else {
        0.0
    }
}

/// `BORDER_REFLECT_101`: `gfedcb|abcdefgh|gfedcba` — first/last not duplicated.
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

/// Saturating cast of an f64 to u16, matching `cv::saturate_cast<uint16_t>`.
#[inline]
fn sat_u16(v: f64) -> u16 {
    let r = v.round();
    if r <= 0.0 {
        0
    } else if r >= 65535.0 {
        65535
    } else {
        r as u16
    }
}

/// Bicubic 2x upscale of one channel from src (src_w x src_h) to dst
/// (2*src_w x 2*src_h). Both buffers are tightly packed (no per-row padding
/// beyond the channel data).
fn resize_bicubic_2x(src: &[u16], src_w: u32, src_h: u32, dst: &mut [u16]) {
    let sw = src_w as i32;
    let sh = src_h as i32;
    let dw = 2 * src_w as usize;
    let dh = 2 * src_h as usize;

    // Pre-compute the per-output-column 4-tap indices and weights. With a
    // fixed 2x scale, every other output column shares the same fractional
    // offset (t = 0.25 for odd dst_x, t = 0.75 for even dst_x), so we only
    // need to compute the kernel twice. Pre-compute taps to avoid repeating
    // the reflect + floor work for every row.
    let mut col_taps: Vec<[i32; 4]> = Vec::with_capacity(dw);
    let mut col_w: Vec<[f64; 4]> = Vec::with_capacity(dw);
    for dst_x in 0..dw {
        let sx = (dst_x as f64 + 0.5) * 0.5 - 0.5;
        let isx = sx.floor() as i32;
        let t = sx - isx as f64;
        let w = cubic_weights(t);
        let taps = [
            reflect_101(isx - 1, sw),
            reflect_101(isx, sw),
            reflect_101(isx + 1, sw),
            reflect_101(isx + 2, sw),
        ];
        col_taps.push(taps);
        col_w.push(w);
    }

    // Two-pass separable resize: first horizontal (src_h x dw), then
    // vertical (dh x dw). Intermediate buffer is f64 to retain precision.
    let mut tmp = vec![0.0f64; src_h as usize * dw];
    for y in 0..src_h as usize {
        let row_off = y * src_w as usize;
        let dst_row = y * dw;
        for dst_x in 0..dw {
            let taps = &col_taps[dst_x];
            let w = &col_w[dst_x];
            let v = (0..4)
                .map(|k| src[row_off + taps[k] as usize] as f64 * w[k])
                .sum::<f64>();
            tmp[dst_row + dst_x] = v;
        }
    }

    for dst_y in 0..dh {
        let sy = (dst_y as f64 + 0.5) * 0.5 - 0.5;
        let isy = sy.floor() as i32;
        let t = sy - isy as f64;
        let w = cubic_weights(t);
        let taps = [
            reflect_101(isy - 1, sh) as usize,
            reflect_101(isy, sh) as usize,
            reflect_101(isy + 1, sh) as usize,
            reflect_101(isy + 2, sh) as usize,
        ];
        let dst_row = dst_y * dw;
        for dst_x in 0..dw {
            let v = w[0] * tmp[taps[0] * dw + dst_x]
                + w[1] * tmp[taps[1] * dw + dst_x]
                + w[2] * tmp[taps[2] * dw + dst_x]
                + w[3] * tmp[taps[3] * dw + dst_x];
            dst[dst_row + dst_x] = sat_u16(v);
        }
    }
}

/// Walk an interleaved 3-channel `x3f_area16_t` and apply the BMT->YUV
/// transform used by the Quattro expansion (Yis4T variant from
/// `denoise_utils.cpp`):
///
/// ```text
/// Y =       +4*T
/// U = +2*B      -2*T
/// V =   +B -2*M   +T
/// ```
///
/// Writes (Y, U+O_UV, V+O_UV) back in place.
unsafe fn bmt_to_yuv_yis4t(area: *mut Area16) {
    let a = unsafe { &*area };
    debug_assert_eq!(a.channels, 3);
    let stride = a.row_stride as usize;
    for row in 0..a.rows as usize {
        let row_ptr = unsafe { a.data.add(row * stride) };
        for col in 0..a.columns as usize {
            let p = unsafe { row_ptr.add(col * 3) };
            let b = unsafe { *p } as i32;
            let m = unsafe { *p.add(1) } as i32;
            let t = unsafe { *p.add(2) } as i32;

            let y = 4 * t;
            let u = 2 * b - 2 * t;
            let v = b - 2 * m + t;

            unsafe { *p = sat_u16(y as f64) };
            unsafe { *p.add(1) = sat_u16((u + O_UV) as f64) };
            unsafe { *p.add(2) = sat_u16((v + O_UV) as f64) };
        }
    }
}

/// Inverse of [`bmt_to_yuv_yis4t`]:
///
/// ```text
/// B = ( +Y +2*U      + 2 ) / 4
/// M = ( +Y +U   -2*V + 2 ) / 4
/// T = ( +Y          + 2 ) / 4
/// ```
///
/// (The +2 then /4 matches OpenCV `cv::Mat` integer arithmetic with rounding
/// half-way values up; bit-equivalent to the C++ reference for u16 inputs.)
unsafe fn yuv_to_bmt_yis4t(area: *mut Area16) {
    let a = unsafe { &*area };
    debug_assert_eq!(a.channels, 3);
    let stride = a.row_stride as usize;
    for row in 0..a.rows as usize {
        let row_ptr = unsafe { a.data.add(row * stride) };
        for col in 0..a.columns as usize {
            let p = unsafe { row_ptr.add(col * 3) };
            let y = unsafe { *p } as i32;
            let u = unsafe { *p.add(1) } as i32 - O_UV;
            let v = unsafe { *p.add(2) } as i32 - O_UV;

            let b = (y + 2 * u + 2) / 4;
            let m = (y + u - 2 * v + 2) / 4;
            let t = (y + 2) / 4;

            unsafe { *p = sat_u16(b as f64) };
            unsafe { *p.add(1) = sat_u16(m as f64) };
            unsafe { *p.add(2) = sat_u16(t as f64) };
        }
    }
}

// `x3f_denoise_active` is implemented in C++ (src/x3f_denoise.cpp) when
// opencv-mobile is linked; on wasm32-unknown-unknown the matching no-op
// stub in csrc/denoise_stub.c resolves the symbol. Either way the call
// is safe to make unconditionally.
//
// The `denoise_type` arg is the C `x3f_denoise_type_t` enum from
// src/x3f_denoise.h; we hardcode 2 = X3F_DENOISE_F23 below since this
// is the only variant the Quattro path uses (the F23 row of
// `denoise_types[]` selects sigma h=300 and the Yis4T BMT/YUV transforms,
// which the Rust code already does in place above).
const X3F_DENOISE_F23: u32 = 2;
const STAGE_PRE_UPSAMPLE: i32 = 0;
const STAGE_POST_UPSAMPLE: i32 = 1;

extern "C" {
    fn x3f_denoise_active(area: *mut Area16, denoise_type: u32, stage: i32, scale: f32);
}

/// Native Rust replacement for the OpenCV-backed `x3f_expand_quattro`. The
/// resize + BMT/YUV transforms run here in Rust; the two NLM passes that
/// the legacy C++ embedded inside the upsampler are now thin FFI hops
/// into `x3f_denoise_active` so the entire algorithm orchestration stays
/// in Rust while the OpenCV NLM kernel stays in opencv-mobile.
///
/// # Safety
///
/// All non-null `*mut Area16` inputs must point to valid `x3f_area16_t`
/// structs whose `data` arrays span `rows * row_stride * sizeof(u16)` bytes.
/// `image` and `expanded` are mutated in place. `active` (when non-null)
/// must be a sub-area view into `image`'s data; `active_exp` must be a
/// sub-area view into `expanded`'s data — those invariants come from the
/// C call site in `x3f-sys::process::expand_quattro`. `scale` (0..1)
/// attenuates the NLM sigma of both Quattro denoise passes (1.0 = legacy
/// full strength); `expand_quattro` derives it from the 0..=10 intensity.
#[no_mangle]
pub(crate) unsafe extern "C" fn x3f_expand_quattro(
    image: *mut Area16,
    active: *mut Area16,
    qtop: *mut Area16,
    expanded: *mut Area16,
    active_exp: *mut Area16,
    scale: f32,
) {
    assert!(!image.is_null() && !qtop.is_null() && !expanded.is_null());

    let img = unsafe { &*image };
    let qt = unsafe { &*qtop };
    let exp = unsafe { &*expanded };

    assert_eq!(img.channels, 3, "image must be 3-channel");
    assert_eq!(qt.channels, 1, "qtop must be 1-channel");
    assert_eq!(exp.channels, 3, "expanded must be 3-channel");
    assert_eq!(qt.rows, exp.rows, "qtop and expanded must agree on rows");
    assert_eq!(
        qt.columns, exp.columns,
        "qtop and expanded must agree on columns"
    );

    // Step 1: BMT -> YUV in place on the (still half-res) image. After
    // this, `active` (a sub-area view into image->data) is also YUV.
    unsafe { bmt_to_yuv_yis4t(image) };

    // Step 1.5: pre-upsample NLM denoise on the half-res active region
    // (mirrors the legacy C++ `denoise_nlm(act, d->h)` call). No-op when
    // `active` is null (caller passed null because denoise was disabled)
    // or when the binary was built without opencv-mobile (WASM).
    if !active.is_null() {
        unsafe { x3f_denoise_active(active, X3F_DENOISE_F23, STAGE_PRE_UPSAMPLE, scale) };
    }

    // Step 2: bicubic 2x upsample, per channel, image -> expanded.
    // Source/destination strides are in units of u16 elements.
    let src_w = img.columns;
    let src_h = img.rows;
    let dst_w = exp.columns;
    let dst_h = exp.rows;
    assert_eq!(dst_w, 2 * src_w);
    assert_eq!(dst_h, 2 * src_h);

    // Deinterleave each channel into a tight buffer, resize, write back.
    let src_stride = img.row_stride as usize;
    let dst_stride = exp.row_stride as usize;
    let mut src_plane = vec![0u16; (src_w * src_h) as usize];
    let mut dst_plane = vec![0u16; (dst_w * dst_h) as usize];

    for c in 0..3usize {
        // Deinterleave channel c.
        for row in 0..src_h as usize {
            let row_ptr = unsafe { img.data.add(row * src_stride) };
            let out_row = row * src_w as usize;
            for col in 0..src_w as usize {
                src_plane[out_row + col] = unsafe { *row_ptr.add(col * 3 + c) };
            }
        }

        resize_bicubic_2x(&src_plane, src_w, src_h, &mut dst_plane);

        // Re-interleave into expanded's channel c.
        for row in 0..dst_h as usize {
            let row_ptr = unsafe { exp.data.add(row * dst_stride) };
            let in_row = row * dst_w as usize;
            for col in 0..dst_w as usize {
                unsafe { *row_ptr.add(col * 3 + c) = dst_plane[in_row + col] };
            }
        }
    }

    // Step 3: replace expanded's Y plane (channel 0) with qtop * 4 (saturating).
    // qtop is the high-resolution T plane that the Quattro sensor records
    // directly; multiplying by 4 puts it on the same scale as Y = 4*T.
    let qt_stride = qt.row_stride as usize;
    for row in 0..dst_h as usize {
        let qt_row = unsafe { slice::from_raw_parts(qt.data.add(row * qt_stride), dst_w as usize) };
        let exp_row = unsafe { exp.data.add(row * dst_stride) };
        for col in 0..dst_w as usize {
            let v = (qt_row[col] as u32) * 4;
            let clamped = if v > 65535 { 65535 } else { v as u16 };
            unsafe { *exp_row.add(col * 3) = clamped };
        }
    }

    // Step 3.5: post-upsample NLM denoise on the full-res active region
    // (mirrors the legacy C++ `fastNlMeansDenoising(act_exp, ...)` block).
    // Runs while `expanded` is still in YUV layout, before the final
    // YUV->BMT transform below.
    if !active_exp.is_null() {
        unsafe { x3f_denoise_active(active_exp, X3F_DENOISE_F23, STAGE_POST_UPSAMPLE, scale) };
    }

    // Step 4: YUV -> BMT in place on expanded.
    unsafe { yuv_to_bmt_yis4t(expanded) };
}

// Anchor `x3f_expand_quattro` so cross-crate dead-code elimination cannot
// strip it before the legacy C library's call site is linked. Taking the
// function's address in a `#[used]` static is the canonical way to keep a
// `#[no_mangle]` symbol alive in an rlib that no Rust code calls.
#[used]
static _ANCHOR_X3F_EXPAND_QUATTRO: unsafe extern "C" fn(
    *mut Area16,
    *mut Area16,
    *mut Area16,
    *mut Area16,
    *mut Area16,
    f32,
) = x3f_expand_quattro;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cubic_weights_sum_to_one() {
        for n in 0..=10 {
            let t = n as f64 / 10.0;
            let w = cubic_weights(t);
            let sum: f64 = w.iter().sum();
            assert!((sum - 1.0).abs() < 1e-9, "t={t}: weights sum to {sum}");
        }
    }

    #[test]
    fn cubic_weights_match_known_values() {
        // At t=0 the kernel reduces to a delta on the second tap.
        let w = cubic_weights(0.0);
        assert!((w[0]).abs() < 1e-12);
        assert!((w[1] - 1.0).abs() < 1e-12);
        assert!((w[2]).abs() < 1e-12);
        assert!((w[3]).abs() < 1e-12);
    }

    #[test]
    fn reflect_101_edges() {
        // n=8: gfedcb|abcdefgh|gfedcba
        // -1 reflects to 1, -2 to 2, etc.
        assert_eq!(reflect_101(-1, 8), 1);
        assert_eq!(reflect_101(-2, 8), 2);
        assert_eq!(reflect_101(0, 8), 0);
        assert_eq!(reflect_101(7, 8), 7);
        assert_eq!(reflect_101(8, 8), 6);
        assert_eq!(reflect_101(9, 8), 5);
    }

    #[test]
    fn reflect_101_single_pixel_image() {
        assert_eq!(reflect_101(0, 1), 0);
        assert_eq!(reflect_101(-1, 1), 0);
        assert_eq!(reflect_101(7, 1), 0);
    }

    #[test]
    fn sat_u16_clamps_to_range() {
        assert_eq!(sat_u16(-1.0), 0);
        assert_eq!(sat_u16(0.0), 0);
        assert_eq!(sat_u16(0.4), 0);
        assert_eq!(sat_u16(0.6), 1);
        assert_eq!(sat_u16(65535.0), 65535);
        assert_eq!(sat_u16(70000.0), 65535);
    }

    #[test]
    fn resize_constant_input_is_constant() {
        let src = vec![1234u16; 4 * 4];
        let mut dst = vec![0u16; 8 * 8];
        resize_bicubic_2x(&src, 4, 4, &mut dst);
        for &v in &dst {
            assert_eq!(v, 1234);
        }
    }

    #[test]
    fn yis4t_round_trip_preserves_bmt() {
        // Round-trip on a few representative pixels. Quantisation losses
        // exist (the C++ does the same /4 truncation), but for the values
        // OpenCV uses here the round-trip is exact within +/- 1.
        let mut data: Vec<u16> = vec![100, 200, 300, 1000, 2000, 3000, 5000, 10000, 12000];
        let area = Area16 {
            data: data.as_mut_ptr(),
            buf: std::ptr::null_mut(),
            rows: 1,
            columns: 3,
            channels: 3,
            row_stride: 9,
        };
        let area_ptr = &area as *const _ as *mut _;
        let original = data.clone();
        unsafe {
            bmt_to_yuv_yis4t(area_ptr);
            yuv_to_bmt_yis4t(area_ptr);
        }
        for (i, (got, want)) in data.iter().zip(original.iter()).enumerate() {
            let diff = (*got as i32 - *want as i32).abs();
            assert!(diff <= 1, "channel {i}: got {got} want {want}");
        }
    }
}
