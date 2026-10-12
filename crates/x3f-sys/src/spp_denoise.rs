//! SPP (Sigma Photo Pro) denoise kernels, ported from the native
//! `ReduceNoiseTiled.cpp` chain of `lib99_MerillDLLForMac.dylib`
//! (archived captures and field maps: x3f-docs
//! `followup-tiled-denoise/`). This is the explicit SPP denoise
//! option. The existing NLM/BM3D selection remains the default.
//!
//! Stage 7b ports the two preparation kernels, each verified
//! bit-for-bit against the native oracle battery
//! (`check_denoise_kernels.cpp`, `kernel-results.txt` in the
//! checkpoint archive):
//!
//! - `tanh_lut`: the table construction of the static
//!   `FPixelProc2_F20::GetTanhLUT(FColorFloat const&, FColorFloat
//!   const&)` (0x5d99c), without the native's process-wide LUT
//!   cache. The native builds three u16 channel tables of a
//!   power-of-two size, fills the first half with truncated
//!   cosine-windowed tanh values, mirrors the second half as the
//!   negated odd symmetry, and pins one boundary entry.
//! - `fast_bilateral_coeffs`: the histogram-bilateral bin
//!   construction of `FPixelProc2_F20::FastBilateral_F20::
//!   CalculateCoeff()` (0x30940): per 1024 bins a quadratic noise
//!   model gives a bin radius, and each bin's weight array holds a
//!   Gaussian `exp(-(4*delta)^2 / Q)` in float32.
//!
//! Transcendental parity: the native calls `exp`, `tanh` and `cos`
//! from the platform libm; Rust routes `f64::exp`, `f64::tanh` and
//! `f64::cos` through the same system libm, so the oracle battery
//! pins the values bit-for-bit on this platform. The hash pins
//! depend on that libm, so the transcendental tests are compiled
//! only for `aarch64` macOS, where the battery was recorded; other
//! platforms skip them instead of risking last-ulp differences.
//! `compute_resample_params` uses no transcendentals and its pins
//! are unconditional.
//!
//! Stage 7c adds `compute_resample_params`, the port of the free
//! function `ComputeResampleParams` (0x7ab70, the geometry builder
//! both `TiledDenoise` resample passes call). The native probes the
//! scale with a 64-entry position table, derives per-phase tent
//! kernels, detects the phase period by comparing kernel anchors,
//! and emits the kernel table, the phase count, the phase offsets
//! and the leading tap. The geometry is reproduced in the native's
//! exact operation order. The geometry and kernel table match all
//! 35 scales in the native test set, including fractional scales.
//!
//! `resample_image` applies these tables to RGB float tiles. It preserves
//! the native phase carry, row padding and four untouched destination rows.
//! Its 1,224 image cases match the native output bits.
//!
//! `compute_edge_map` starts stage 7d. Its 13,440 image cases match the
//! native tiled overload on arm64 macOS. Other targets compare NaN positions
//! and all non-NaN bits. The median helper matches 59,049 direct cases.

use crate::{Error, Result};

pub(crate) mod params;
pub(crate) mod pipeline;
pub(crate) mod pixmap;
pub(crate) mod residual;
pub(crate) mod tiling;

/// One `GetTanhLUT` result: the table size and the three channel
/// tables in the native object order `{A, C, B}` (the object stores
/// the pointers at +0/+8/+0x10; the channel assignment is pinned by
/// the `ApplyHistBilateralFilter` port).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TanhLut {
    pub size: u32,
    pub tables: [Vec<i16>; 3],
}

const PI: f64 = std::f64::consts::PI;

/// The tanh LUT tables for one noise-model pair. `cf1` is the
/// amplitude triple, `cf2` the rate triple (both are `FColorFloat`,
/// three doubles). The construction reproduces the native
/// arithmetic in exact operation order: the max ratio, the
/// `0.25/max` step (as `(0.5/max)*0.5`, two roundings), the
/// power-of-two size ladder, the cosine gate on `step*i < pi`,
/// `trunc` (never round) to the 16 low bits, the negated mirror
/// and the `table[half] = table[half + 1]` boundary fixup.
pub fn tanh_lut(cf1: [f64; 3], cf2: [f64; 3]) -> TanhLut {
    // max(cf1[c] / cf2[c]) with the native fcsel-gt semantics: the
    // first ratio is a plain division, the later ones select on
    // strict greater-than (unordered falls to the running max).
    let mut max_ratio = cf1[0] / cf2[0];
    let second = cf1[1] / cf2[1];
    if second > max_ratio {
        max_ratio = second;
    }
    let third = cf1[2] / cf2[2];
    if third > max_ratio {
        max_ratio = third;
    }
    // The native computes d0 = 0.5/max, then step = d0*0.5: two
    // roundings, not one 0.25/max division.
    let step = 0.5_f64 / max_ratio * 0.5;
    // Size: w8 = trunc(pi/step), quantized onto the power-of-two
    // ladder the native masks with (0x7f.. style indices).
    let raw = (PI / step) as i32;
    let size: u32 = if raw < 0x201 {
        0x200
    } else if raw < 0x401 {
        0x400
    } else if raw < 0x801 {
        0x800
    } else if raw as u32 <= 0x1000 {
        0x1000
    } else {
        0x2000
    };
    let half = size / 2;
    let mask = (size - 1) as usize;
    let mut tables: [Vec<i16>; 3] = [
        vec![0_i16; size as usize],
        vec![0_i16; size as usize],
        vec![0_i16; size as usize],
    ];
    let count = half.max(1);
    for i in 0..count {
        let index = i as f64;
        let phase = step * index;
        // b.pl skips the cosine when phase >= pi or unordered.
        let amplitude = if !(phase < PI) {
            0.0
        } else {
            (phase.cos() + 1.0) * 0.5
        };
        // Channel order {0, 1, 2} -> tables {A, C, B}.
        for (channel, table) in tables.iter_mut().enumerate() {
            let argument = (cf2[channel] * index) / cf1[channel];
            let value = (cf1[channel] * amplitude).mul_add(argument.tanh(), 0.5);
            // fcvtzs truncates; the strh keeps the low 16 bits.
            table[i as usize & mask] = value as i32 as i16;
        }
    }
    // Mirror: table[size - k] = -table[k] for k in 1..half.
    for table in tables.iter_mut() {
        for k in 1..half as usize {
            // The native `neg` wraps; i16::MIN mirrors itself.
            let value = table[k];
            table[mask + 1 - k] = value.wrapping_neg();
        }
        // Boundary fixup: table[half] = table[half + 1].
        let boundary = table[half as usize + 1];
        table[half as usize] = boundary;
    }
    TanhLut { size, tables }
}

/// One `CalculateCoeff` result: the 1024-entry bin bounds and the
/// per-bin Gaussian weight arrays.
#[derive(Clone, Debug, PartialEq)]
pub struct FastBilateralCoeffs {
    /// `low[i] = max(i - radius, 0)`: the first bin index.
    pub low: [i32; 1024],
    /// `high[i] = min(radius + i, 1023) + 1`: one past the last.
    pub high: [i32; 1024],
    /// `weights[i]` has `high[i] - low[i]` float32 entries.
    pub weights: Vec<Vec<f32>>,
}

/// One `ComputeResampleParams` result: the separable resampling
/// geometry for one scale.
#[derive(Clone, Debug, PartialEq)]
pub struct ResampleParams {
    /// The kernel table, `kernel_x * kernel_y` float32 entries,
    /// phase-major: phase `p` holds its taps at
    /// `table[p * kernel_x .. (p + 1) * kernel_x]`.
    pub table: Vec<f32>,
    /// The tap count per phase.
    pub kernel_x: u32,
    /// The distinct phase count.
    pub kernel_y: u32,
    /// `kernel_y` source-start deltas between consecutive phases;
    /// phase `p` starts at `tail + offsets[0..p]` summed.
    pub offsets: Vec<u32>,
    /// The first tap offset of phase 0 (`E[0]`, always negative or
    /// zero in the verified domain).
    pub tail: i32,
}

/// The `FastBilateral_F20::CalculateCoeff` histogram construction.
/// `scaling` is the object field +0x18 (the request noise scaling,
/// always nonzero when the native calls this), and `sigma_38`,
/// `sigma_40`, `sigma_48` are the object fields +0x38/+0x40/+0x48.
/// Per bin `i` the native computes, in this exact operation order:
/// `x = f64(4*i)`, `linear = sigma_40*x + sigma_38` (fused),
/// `quad = fma(sigma_48*x, x, linear)` (the `sigma_48*x` product
/// rounds first), `Q = quad / scaling`,
/// `radius = trunc(sqrt(Q) * 3.0) >> 2` (arithmetic), the bin
/// bounds, and `exp(-(4*delta)^2 / Q)` per float32 weight.
pub fn fast_bilateral_coeffs(
    scaling: f64,
    sigma_38: f64,
    sigma_40: f64,
    sigma_48: f64,
) -> FastBilateralCoeffs {
    let mut low = [0_i32; 1024];
    let mut high = [0_i32; 1024];
    let mut weights = Vec::with_capacity(1024);
    for i in 0..1024_i32 {
        let x = f64::from(i << 2);
        let linear = sigma_40.mul_add(x, sigma_38);
        let quad = (sigma_48 * x).mul_add(x, linear);
        let q = quad / scaling;
        // fcvtzs saturates (NaN -> 0); the shift is arithmetic.
        let radius = ((q.sqrt() * 3.0) as i32) >> 2;
        low[i as usize] = (i - radius).max(0);
        high[i as usize] = (radius + i).min(1023) + 1;
        let mut bin = Vec::with_capacity((high[i as usize] - low[i as usize]) as usize);
        for j in low[i as usize]..high[i as usize] {
            let delta = f64::from((i - j).wrapping_abs() << 2);
            // fnmul: the negation of the rounded product.
            let value = (-(delta * delta) / q).exp();
            bin.push(value as f32);
        }
        weights.push(bin);
    }
    FastBilateralCoeffs { low, high, weights }
}

/// The `ComputeResampleParams` port (the free function at 0x7ab70).
///
/// The native probes the scale over 64 output positions: position
/// `k` (1-based) maps to the source center
/// `f32(f64(f32(k / scale)) + 0.5 * (1 - 1/scale))` (the final add
/// is fused; the multiply is exact, so a plain add is identical).
/// Around each center it lays a tent kernel over
/// `count = ceil(window) + 2` taps starting at
/// `floor(center - window / 2)`, where `window = 2 / scale` when
/// scaling down and `2.0` when scaling up. Each downscaling tap
/// weight clamps `|center - tap| * scale` to [0, 1] in f32, then
/// computes `(1 - f64(clamped)) * f64(scale)` and narrows once.
/// Keep the scale factor before normalization to preserve float
/// rounding. The upscaling weight is the f32 tent
/// `1 - |center - tap|`, zero past 1. Every kernel is normalized
/// by its own tap sum and accumulated into one average buffer.
///
/// The outputs come from that average and the per-position kernels:
/// `kernel_x` is the nonzero span of the average, `kernel_y` the
/// smallest probe index (1..=64) whose kernel matches probe 0 at
/// the first nonzero average entry (the native compares
/// `|w - anchor| / max(|w|, |anchor|)` in f32 against 1e-3f, or
/// takes the first exact-zero match when the anchor is zero),
/// `tail` the first probe's floor start, `offsets` the consecutive
/// floor-start deltas, and `table` the normalized per-phase kernels
/// cropped to the nonzero span.
///
/// The geometry and table match all 35 native test scales. The
/// test set includes the request scales in eighths from 0.375 to
/// 1.0 and their reciprocals. `crp-complete-results.txt` in the
/// x3f-docs checkpoint records every scale and table as exact bits.
///
/// Domain contract. The average must contain a nonzero entry and
/// some probe in 1..=63 must match the anchor. The native scans
/// without a bound and, when no probe matches, reads one entry
/// past its probe table (garbage offsets; the recorded native
/// 0.374 case does this with `kernel_y` at 64); this port panics
/// with a scale-aware message instead. Zero, negative, NaN and
/// infinite scales are rejected up front where the native divides
/// without checking. The recorded battery and the request domain
/// (quantized eighths 0.375..=1.0 and their reciprocals) always
/// stay inside the contract.
pub fn compute_resample_params(scale: f32) -> ResampleParams {
    // The native divides by the scale without checking it; zero,
    // negative, NaN and infinites produce garbage geometry there.
    // The port refuses them instead (the request domain is finite
    // and positive).
    assert!(
        scale.is_finite() && scale > 0.0,
        "resample scale {scale} must be finite and positive"
    );
    let scale64 = f64::from(scale);
    let one_minus = 1.0 - 1.0 / scale64;
    // Probe centers for k = 1..=64: the f32 division by the scale,
    // widened to f64, fused with 0.5 * (1 - 1/scale), narrowed.
    let mut centers = [0_f32; 64];
    for (i, center) in centers.iter_mut().enumerate() {
        let quotient = (i as u32 + 1) as f32 / scale;
        *center = 0.5_f64.mul_add(one_minus, f64::from(quotient)) as f32;
    }
    let window = if scale < 1.0 { 2.0 / scale } else { 2.0 };
    let half_window = window * 0.5;
    let count = (window.ceil() + 2.0) as u32 as usize;
    // Per-position floor starts, tent kernels and the average.
    let mut starts = [0_i32; 64];
    let mut kernels = vec![vec![0_f32; count]; 64];
    let mut average = vec![0_f32; count];
    for i in 0..64 {
        // frintm plus fcvtzs on the f32 difference.
        starts[i] = (centers[i] - half_window).floor() as i32;
        let mut sum = 0_f32;
        for (j, weight) in kernels[i].iter_mut().enumerate() {
            let tap = starts[i].wrapping_add(j as i32);
            let distance = centers[i] - tap as f32;
            // The native magnitude is fneg plus fcsel (a branch
            // abs), not fabs; identical for the reachable values.
            let magnitude = if distance < 0.0 { -distance } else { distance };
            let value = if scale < 1.0 {
                // The saved d15 is f64(scale); apply it before the f32 rounding.
                let clamped = magnitude * scale;
                let clamped = if clamped < 0.0 { 0.0 } else { clamped };
                let clamped = if clamped > 1.0 { 1.0 } else { clamped };
                ((1.0 - f64::from(clamped)) * scale64) as f32
            } else if magnitude > 1.0 {
                // The up path compares first, then 1 - |d| in f32.
                0.0
            } else {
                1.0 - magnitude
            };
            *weight = value;
            sum += value;
        }
        for (j, weight) in kernels[i].iter_mut().enumerate() {
            *weight /= sum;
            average[j] += *weight;
        }
    }
    let first_nonzero = average
        .iter()
        .position(|&value| value != 0.0)
        .unwrap_or_else(|| panic!("resample scale {scale} has an all-zero average"));
    // The reverse scan always finds the leading entry itself, so
    // the fallback is unreachable; keep it defined anyway.
    let last_nonzero = (first_nonzero..count)
        .rev()
        .find(|&j| average[j] != 0.0)
        .unwrap_or(first_nonzero);
    let kernel_x = (last_nonzero - first_nonzero + 1) as u32;
    // Period detection against probe 0 at the anchor entry. The
    // native compares fabd(w, anchor) / max(|w|, |anchor|) in f32
    // against 1e-3f, or takes the first exact-zero match for a
    // zero anchor (0x7b8f0..0x7b91c).
    let anchor = kernels[0][first_nonzero];
    let mut kernel_y = 64_u32;
    for k in 1..64 {
        let weight = kernels[k][first_nonzero];
        let matched = if anchor != 0.0 {
            let reference = weight.abs().max(anchor.abs());
            (weight - anchor).abs() / reference < 1.0e-3_f32
        } else {
            weight == 0.0
        };
        if matched {
            kernel_y = k as u32;
            break;
        }
    }
    if kernel_y == 64 {
        // The native reads one entry past its 64-entry probe table
        // here and emits garbage offsets; refuse instead.
        panic!("resample scale {scale} has no phase period within the 64-probe table");
    }
    let offsets = (0..kernel_y as usize)
        .map(|i| starts[i + 1].wrapping_sub(starts[i]) as u32)
        .collect();
    let mut table = vec![0_f32; (kernel_x * kernel_y) as usize];
    for phase in 0..kernel_y as usize {
        let span = &kernels[phase][first_nonzero..=last_nonzero];
        table[phase * kernel_x as usize..(phase + 1) * kernel_x as usize].copy_from_slice(span);
    }
    ResampleParams {
        table,
        kernel_x,
        kernel_y,
        offsets,
        tail: starts[0],
    }
}

/// Resample an RGB float tile with the native `ResampleImage` operation order.
/// The destination size is `[columns, rows]`. Strides count float samples.
/// The last four destination rows and all destination padding stay unchanged.
/// Source reads can cross row padding. The source slice must contain every
/// sample the filter needs; the native function does not clamp its right edge.
/// A destination with exactly four rows needs no source reads.
///
/// Invalid layouts or parameters return an error before any destination write.
/// The native scratch arrays permit at most 24 taps. The parameter builder
/// supports at most 63 phases. This port owns its scratch storage.
pub fn resample_image(
    source: &[f32],
    source_stride: usize,
    destination: &mut [f32],
    destination_size: [usize; 2],
    destination_stride: usize,
    params: &ResampleParams,
) -> Result<()> {
    if source_stride < 3
        || source.len() < 3
        || destination_size[0] == 0
        || !destination_size[0]
            .checked_mul(3)
            .is_some_and(|width| width <= destination_stride)
        || !destination_size[1]
            .checked_mul(destination_stride)
            .is_some_and(|required| required <= destination.len())
        || destination_size[1] < 4
    {
        return Err(Error::InvalidData("invalid SPP resample image layout"));
    }
    let taps = params.kernel_x as usize;
    let phases = params.kernel_y as usize;
    if !(1..=24).contains(&taps)
        || !(1..=63).contains(&phases)
        || params.table.len() != taps * phases
        || params.offsets.len() != phases
        || params.offsets.iter().any(|&step| step > params.kernel_x)
    {
        return Err(Error::InvalidData("invalid SPP resample parameters"));
    }
    let [width, rows] = destination_size;
    let rows = rows - 4;
    if rows == 0 {
        return Ok(());
    }
    let half = taps / 2;
    let tail = i64::from(params.tail);
    if tail + (half as i64) < 0 {
        return Err(Error::InvalidData("SPP resample reads before source rows"));
    }
    let check_source_extent = |row: i64, last_column: i64| {
        let end = usize::try_from(last_column.max(0))
            .ok()
            .and_then(|column| column.checked_mul(3))
            .and_then(|column| column.checked_add(3))
            .and_then(|end| {
                usize::try_from(row)
                    .ok()?
                    .checked_mul(source_stride)?
                    .checked_add(end)
            });
        if end.is_some_and(|end| end <= source.len()) {
            Ok(())
        } else {
            Err(Error::InvalidData("SPP resample reads past source storage"))
        }
    };
    // The initial scratch rows carry the horizontal phase across row boundaries.
    let mut initial_phase = 0;
    let mut main_last_column = 0;
    for y in 0..half.max(1) {
        let mut column = tail;
        let mut phase = initial_phase;
        let mut last_column = 0;
        for _ in 0..width {
            last_column = column
                .checked_add(taps as i64 - 1)
                .ok_or(Error::InvalidData("SPP resample column overflow"))?;
            column = column
                .checked_add(i64::from(params.offsets[phase]))
                .ok_or(Error::InvalidData("SPP resample column overflow"))?;
            phase = (phase + 1) % phases;
        }
        if column < 0 {
            return Err(Error::InvalidData("SPP resample destination is too narrow"));
        }
        check_source_extent((tail + y as i64).max(0), last_column)?;
        if y == 0 {
            main_last_column = last_column;
        }
        initial_phase = phase;
    }
    let mut row = tail;
    for y in 0..rows {
        check_source_extent(row + taps as i64 - 1, main_last_column)?;
        row = row
            .checked_add(i64::from(params.offsets[y % phases]))
            .ok_or(Error::InvalidData("SPP resample row overflow"))?;
    }
    let row_samples = width * 3;
    let mut scratch = vec![vec![0.0_f32; row_samples]; taps];
    let horizontal = |source_row: usize, output: &mut [f32], phase: &mut usize| {
        let mut column = tail;
        for pixel in output.chunks_exact_mut(3) {
            let weights = &params.table[*phase * taps..(*phase + 1) * taps];
            pixel.fill(0.0);
            for (tap, &weight) in weights.iter().enumerate() {
                let x = (column + tap as i64).max(0) as usize;
                let input = &source[source_row * source_stride + x * 3..][..3];
                for channel in 0..3 {
                    // Separate multiplication and addition match the native rounding.
                    pixel[channel] += input[channel] * weight;
                }
            }
            column += i64::from(params.offsets[*phase]);
            *phase = (*phase + 1) % phases;
        }
    };
    let mut phase = 0;
    for (y, output) in scratch[..half].iter_mut().enumerate() {
        horizontal((tail + y as i64).max(0) as usize, output, &mut phase);
    }
    let mut pending = taps - half;
    let mut start = tail;
    for y in 0..rows {
        for (tap, output) in scratch.iter_mut().enumerate().skip(taps - pending) {
            horizontal((start + tap as i64) as usize, output, &mut 0);
        }
        let phase = y % phases;
        let weights = &params.table[phase * taps..(phase + 1) * taps];
        let output = &mut destination[y * destination_stride..][..row_samples];
        output.fill(0.0);
        for (input, &weight) in scratch.iter().zip(weights) {
            for (value, &sample) in output.iter_mut().zip(input) {
                *value += sample * weight;
            }
        }
        pending = params.offsets[phase] as usize;
        scratch.rotate_left(pending);
        start += pending as i64;
    }
    Ok(())
}

/// Parameters for the tiled native edge map.
#[derive(Clone, Copy, Debug)]
pub struct EdgeMapParams {
    /// Added to each source sample before the median filter.
    pub bias: f32,
    /// Multiplies source samples when `noise_model` is absent.
    pub scale: f32,
    /// Optional channel triples: constant terms, then linear terms.
    /// With a model, the source scale is 1 and each divisor is
    /// `constant + linear * median_sample`, evaluated in f64.
    pub noise_model: Option<[[f64; 3]; 2]>,
    /// Sum channel energies for each direction. Otherwise select the largest
    /// horizontal energy and the largest vertical energy separately.
    pub accumulate: bool,
    /// Apply the final five-sample cross average.
    pub smooth: bool,
    /// Exclude this many source rows and columns at each image edge.
    pub border: usize,
}

// The native comparison order also determines which signed zero is selected.
fn edge_median5([mut s0, mut s1, s2, mut s3, mut s4]: [f32; 5]) -> f32 {
    let mut s5 = if s0 > s1 { s1 } else { s0 };
    s0 = if s0 > s1 { s0 } else { s1 };
    let greater = s3 > s4;
    s1 = if greater { s4 } else { s3 };
    s3 = if greater { s3 } else { s4 };
    let greater = s5 > s1;
    s4 = if greater { s3 } else { s0 };
    s1 = if greater { s5 } else { s1 };
    s0 = if greater { s0 } else { s3 };
    s3 = if s4 < s1 { s4 } else { s1 };
    s5 = if s0 > s2 { s2 } else { s0 };
    s3 = if !(s1 < s2) { s3 } else { s5 };
    s0 = if s4 < s0 { s4 } else { s0 };
    s5 = if s1 > s2 { s2 } else { s1 };
    s0 = if !(s4 < s1) { s0 } else { s5 };
    if !(s4 < s2) {
        s3
    } else {
        s0
    }
}

/// Compute the native tiled edge map from RGB floats into a mono destination.
/// `size` is [columns, rows]. Both strides count f32 samples.
/// Only columns `[border + 1, width - border - 1)` and rows
/// `[border + 3, height - border - 3)` are written. Other samples stay intact.
/// The two scratch columns beside this region stay zero.
/// Both strides must be positive. Both slices must include the last row's padding.
/// Empty images or borders larger than half either dimension are rejected,
/// including some layouts where the native function returns without a write.
/// Invalid layouts return an error before any destination write.
pub fn compute_edge_map(
    source: &[f32],
    source_stride: usize,
    destination: &mut [f32],
    size: [usize; 2],
    destination_stride: usize,
    params: &EdgeMapParams,
) -> Result<()> {
    let [width, height] = size;
    let border = params.border;
    let row_samples = width.checked_mul(3);
    if width == 0
        || height == 0
        || !row_samples.is_some_and(|n| n <= source_stride)
        || destination_stride < width
        || !height
            .checked_mul(source_stride)
            .is_some_and(|n| n <= source.len())
        || !height
            .checked_mul(destination_stride)
            .is_some_and(|n| n <= destination.len())
        || border > width / 2
        || border > height / 2
    {
        return Err(Error::InvalidData("invalid SPP edge map layout"));
    }
    if width - 2 * border <= 2 || height - 2 * border <= 6 {
        return Ok(());
    }
    let mut rgb = vec![vec![0.0_f32; row_samples.unwrap()]; 5];
    let mut edges = vec![vec![0.0_f32; width]; 3];
    let scale = if params.noise_model.is_some() {
        1.0
    } else {
        params.scale
    };
    for y in border..height - border {
        for x in border + 1..width - border - 1 {
            let mut horizontal = [0.0_f32; 3];
            let mut vertical = [0.0_f32; 3];
            for c in 0..3 {
                let i = x * 3 + c;
                rgb[4][i] = source[y * source_stride + i] * scale + params.bias;
                rgb[2][i] = edge_median5([
                    rgb[3][i - 3],
                    rgb[3][i],
                    rgb[3][i + 3],
                    rgb[2][i],
                    rgb[4][i],
                ]);
                let divisor = if let Some(model) = params.noise_model {
                    (model[0][c] + model[1][c] * f64::from(rgb[1][i])) as f32
                } else {
                    rgb[1][i]
                };
                let inverse = 1.0_f32 / divisor;
                let dx = (rgb[1][i - 3] - rgb[1][i + 3]) * inverse;
                let dy = (rgb[0][i] - rgb[2][i]) * inverse;
                horizontal[c] = dx * dx;
                vertical[c] = dy * dy;
            }
            let energy = |v: [f32; 3]| {
                if params.accumulate {
                    (v[0] + v[1]) + v[2]
                } else {
                    let tail = if v[1] > v[2] { v[1] } else { v[2] };
                    if v[0] > tail {
                        v[0]
                    } else {
                        tail
                    }
                }
            };
            edges[2][x] = ((energy(horizontal) + energy(vertical)) * 0.5).sqrt();
            if y >= border + 6 {
                let value = if params.smooth {
                    let sum = edges[2][x]
                        + (edges[0][x] + (edges[1][x + 1] + (edges[1][x] + edges[1][x - 1])));
                    (f64::from(sum) * 0.2_f64) as f32
                } else {
                    edges[1][x]
                };
                destination[(y - 3) * destination_stride + x] = value;
            }
        }
        rgb.rotate_left(1);
        edges.rotate_left(1);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fnv1a_bytes(data: &[u8]) -> u64 {
        let mut hash = 0xcbf2_9ce4_8422_2325_u64;
        for &byte in data {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
        hash
    }

    fn fnv_i16(table: &[i16]) -> u64 {
        fnv1a_bytes(
            table
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>()
                .as_slice(),
        )
    }

    fn fnv_f32(values: &[f32]) -> u64 {
        fnv1a_bytes(
            values
                .iter()
                .flat_map(|v| v.to_bits().to_le_bytes())
                .collect::<Vec<_>>()
                .as_slice(),
        )
    }

    fn fnv_i32(values: &[i32]) -> u64 {
        fnv1a_bytes(
            values
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>()
                .as_slice(),
        )
    }

    #[test]
    fn edge_median_matches_native_special_values() {
        let values = [
            0xc080_0000,
            0xbf80_0000,
            0x8000_0000,
            0,
            0x3f80_0000,
            0x4080_0000,
            0x7fc1_2345,
            0x7f80_0000,
            0xff80_0000,
        ]
        .map(f32::from_bits);
        let mut output = Vec::with_capacity(59_049);
        for a in values {
            for b in values {
                for c in values {
                    for d in values {
                        for e in values {
                            output.push(edge_median5([a, b, c, d, e]));
                        }
                    }
                }
            }
        }
        let fixture = include_str!("../tests/fixtures/spp-edge-map-native.txt");
        let row = fixture
            .lines()
            .find(|line| line.starts_with("MEDIAN "))
            .unwrap();
        let fields: Vec<_> = row.split_whitespace().collect();
        assert_eq!(fields[..3], ["MEDIAN", "9", "59049"]);
        assert_eq!(
            fnv_f32(&output),
            u64::from_str_radix(fields[3], 16).unwrap()
        );
    }

    #[test]
    fn edge_map_matches_native_battery() {
        let fixture = include_str!("../tests/fixtures/spp-edge-map-native.txt");
        let mut cases = 0;
        for line in fixture.lines().filter(|line| line.starts_with("EDGE ")) {
            let fields: Vec<_> = line.split_whitespace().collect();
            let values: Vec<usize> = fields[1..7].iter().map(|v| v.parse().unwrap()).collect();
            let [width, height, border, padding, pattern, mode] = values[..] else {
                panic!()
            };
            let source_stride = width * 3 + padding;
            let destination_stride = width + padding;
            let mut source = vec![-123.25_f32; 8 + source_stride * height + 8];
            let mut destination = vec![-321.5_f32; 8 + destination_stride * height + 8];
            let mut state = 0x1234_5678_u32;
            for y in 0..height {
                for x in 0..width {
                    for c in 0..3 {
                        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                        let value = match pattern {
                            0 => ((state >> 16) as i32 - 32768) as f32 / 4096.0,
                            1 => (c + 1) as f32,
                            2 => {
                                if x == width / 2 && y == height / 2 {
                                    (16 * (c + 1)) as f32
                                } else {
                                    0.0
                                }
                            }
                            3 => (x + 2 * y + 3 * c) as f32 / 8.0,
                            4 => {
                                if (x + y + c) % 2 == 1 {
                                    -0.0
                                } else {
                                    0.0
                                }
                            }
                            5 => {
                                if (x + 3 * y + 5 * c) % 19 == 0 {
                                    f32::from_bits(0x7fc1_2345)
                                } else {
                                    ((state >> 16) as i32 - 32768) as f32 / 4096.0
                                }
                            }
                            _ => panic!("unknown pattern"),
                        };
                        source[8 + y * source_stride + x * 3 + c] = value;
                    }
                }
            }
            assert_eq!(
                fnv_f32(&source),
                u64::from_str_radix(fields[7], 16).unwrap(),
                "{line}"
            );
            let params = EdgeMapParams {
                bias: if mode & 8 != 0 { 16.7 } else { 8.0 },
                scale: if mode & 8 != 0 { -0.73 } else { 0.375 },
                noise_model: (mode & 2 != 0).then_some(if mode & 8 != 0 {
                    [[2.03, 3.17, 5.29], [0.12345, 0.23456, 0.45678]]
                } else {
                    [[2.0, 3.0, 5.0], [0.125, 0.25, 0.5]]
                }),
                accumulate: mode & 1 != 0,
                smooth: mode & 4 != 0,
                border,
            };
            compute_edge_map(
                &source[8..8 + height * source_stride],
                source_stride,
                &mut destination[8..8 + height * destination_stride],
                [width, height],
                destination_stride,
                &params,
            )
            .unwrap();
            #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
            assert_eq!(
                fnv_f32(&destination),
                u64::from_str_radix(fields[8], 16).unwrap(),
                "{line}"
            );
            // NaN payload and sign can differ between CPU architectures.
            for value in &mut destination {
                if value.is_nan() {
                    *value = f32::from_bits(0x7fc0_0000);
                }
            }
            assert_eq!(
                fnv_f32(&destination),
                u64::from_str_radix(fields[9], 16).unwrap(),
                "{line}"
            );
            cases += 1;
        }
        assert_eq!(cases, 13_440);
    }

    #[test]
    fn edge_map_rejects_invalid_layout_before_writes() {
        let source = [1.0; 3 * 7 * 13];
        let params = EdgeMapParams {
            bias: 8.0,
            scale: 0.375,
            noise_model: None,
            accumulate: true,
            smooth: true,
            border: 0,
        };
        for (source_len, source_stride, size, destination_stride, border) in [
            (source.len() - 1, 21, [7, 13], 7, 0),
            (source.len(), 20, [7, 13], 7, 0),
            (source.len(), 21, [7, 13], 6, 0),
            (source.len(), 21, [7, 14], 7, 0),
            (source.len(), 21, [0, 13], 7, 0),
            (source.len(), 21, [7, 0], 7, 0),
            (source.len(), 21, [7, 13], 7, 4),
            (source.len(), 21, [7, 13], 7, usize::MAX),
            (source.len(), 21, [usize::MAX, 13], 7, 0),
            (source.len(), usize::MAX, [7, 13], 7, 0),
            (source.len(), 21, [7, 13], usize::MAX, 0),
        ] {
            let mut destination = [-321.5; 7 * 13];
            let mut params = params;
            params.border = border;
            assert!(compute_edge_map(
                &source[..source_len],
                source_stride,
                &mut destination,
                size,
                destination_stride,
                &params
            )
            .is_err());
            assert_eq!(destination, [-321.5; 7 * 13]);
        }
        let mut destination = [-321.5; 7 * 13 - 1];
        assert!(compute_edge_map(&source, 21, &mut destination, [7, 13], 7, &params).is_err());
        assert_eq!(destination, [-321.5; 7 * 13 - 1]);
    }

    #[test]
    fn resample_image_matches_native_battery() {
        // The native oracle checks allocated and caller-owned scratch storage.
        let fixture = include_str!("../tests/fixtures/spp-resample-native.txt");
        let sentinel = f32::from_bits(0xc5a1_2345);
        let mut cases = 0;
        for line in fixture.lines().filter(|line| !line.starts_with('#')) {
            let fields: Vec<_> = line.split_whitespace().collect();
            let scale = f32::from_bits(u32::from_str_radix(fields[0], 16).unwrap());
            let width: usize = fields[1].parse().unwrap();
            let rows: usize = fields[2].parse().unwrap();
            let pattern: usize = fields[3].parse().unwrap();
            let padding: usize = fields[4].parse().unwrap();
            let expected = u64::from_str_radix(fields[5], 16).unwrap();
            let params = if scale == 0.0 {
                ResampleParams {
                    kernel_x: 17,
                    kernel_y: 3,
                    tail: -3,
                    offsets: vec![1, 2, 0],
                    table: (0..3)
                        .flat_map(|p| {
                            (0..17).map(move |t| (((p + 1) * (t + 1)) % 5 + 1) as f32 / 128.0)
                        })
                        .collect(),
                }
            } else {
                compute_resample_params(scale)
            };
            let source_columns = if pattern == 2 { 5 } else { 256 };
            let source_stride = source_columns * 3 + padding;
            let mut source = vec![sentinel; source_stride * 256];
            let mut random = 0x1234_5678_u32;
            for y in 0..256 {
                for x in 0..source_columns {
                    for channel in 0..3 {
                        random = random.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                        let value = if pattern == 3 {
                            -0.0
                        } else if pattern == 0 || pattern == 2 {
                            ((random >> 8) as i32 - 0x80_0000) as f32 / 256.0
                        } else if x == 0 || y == 0 || (x == 6 && y == 5) {
                            ((channel + 1) * 1024) as f32
                        } else {
                            -((channel + 1) as f32) / 8.0
                        };
                        source[y * source_stride + x * 3 + channel] = value;
                    }
                }
            }
            let stride = width * 3 + padding;
            let mut output = vec![sentinel; 16 + stride * rows + 16];
            resample_image(
                &source,
                source_stride,
                &mut output[16..16 + stride * rows],
                [width, rows],
                stride,
                &params,
            )
            .unwrap();
            assert_eq!(fnv_f32(&output), expected, "{line}");
            cases += 1;
        }
        assert_eq!(cases, 1224);
    }

    #[test]
    fn resample_image_rejects_invalid_input_before_writes() {
        let source = vec![1.0; 32 * 32 * 3];
        let params = compute_resample_params(0.5);
        let reject = |source_len,
                      source_stride,
                      destination_size,
                      destination_stride,
                      params: &ResampleParams| {
            let mut output = vec![-123.0; 8 * 12 * 3];
            assert!(resample_image(
                &source[..source_len],
                source_stride,
                &mut output,
                destination_size,
                destination_stride,
                params,
            )
            .is_err());
            assert!(output.iter().all(|&value| value == -123.0));
        };
        for (length, stride) in [(0, 96), (source.len(), 2), (source.len(), usize::MAX)] {
            reject(length, stride, [8, 12], 24, &params);
        }
        for (size, stride) in [
            ([0, 12], 24),
            ([8, 3], 24),
            ([8, 12], 23),
            ([8, 13], 24),
            ([usize::MAX, 5], usize::MAX),
        ] {
            reject(source.len(), 96, size, stride, &params);
        }
        // These bounds fail after earlier pixels or rows would have been valid.
        reject(15 * 3, 96, [8, 12], 24, &params);
        reject(15 * 96, 96, [8, 12], 24, &params);
        let mut bad = params.clone();
        bad.table.pop();
        reject(source.len(), 96, [8, 12], 24, &bad);
        for (taps, phases) in [(0, 1), (25, 1), (4, 0), (4, 64)] {
            let mut bad = params.clone();
            bad.kernel_x = taps;
            bad.kernel_y = phases;
            reject(source.len(), 96, [8, 12], 24, &bad);
        }
        let mut bad = params.clone();
        bad.offsets.clear();
        reject(source.len(), 96, [8, 12], 24, &bad);
        let mut bad = params.clone();
        bad.offsets[0] = 5;
        reject(source.len(), 96, [8, 12], 24, &bad);
        let mut bad = params.clone();
        bad.tail = -3;
        reject(source.len(), 96, [8, 12], 24, &bad);
        let upsample = compute_resample_params(8.0);
        reject(source.len(), 96, [1, 12], 3, &upsample);
    }

    #[test]
    fn resample_image_checks_the_carried_phase_source_extent() {
        let params = ResampleParams {
            kernel_x: 4,
            kernel_y: 3,
            tail: -1,
            offsets: vec![0, 0, 4],
            table: vec![0.25; 12],
        };
        let source = [1.0; 33];
        let mut output = [-123.0; 5 * 5 * 3];
        // Later fill rows need 27 samples; only the second initial row needs 33.
        assert!(resample_image(&source[..32], 3, &mut output, [5, 5], 15, &params).is_err());
        assert_eq!(output, [-123.0; 5 * 5 * 3]);
        resample_image(&source, 3, &mut output, [5, 5], 15, &params).unwrap();
        assert_eq!(output[..15], [1.0; 15]);
        assert_eq!(output[15..], [-123.0; 60]);
    }

    /// The native oracle battery: check_denoise_kernels.cpp against
    /// the shipped library, archived in followup-tiled-denoise/
    /// kernel-results.txt. Every hash below is a native run.
    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    #[test]
    fn tanh_lut_matches_native_battery() {
        let cases: &[(&str, [f64; 3], [f64; 3], u32, u64, u64, u64)] = &[
            (
                "uniform2/0.5",
                [2.0, 2.0, 2.0],
                [0.5, 0.5, 0.5],
                512,
                0x2ccfbc20bcc5a8e5,
                0x2ccfbc20bcc5a8e5,
                0x2ccfbc20bcc5a8e5,
            ),
            (
                "uniform1/0.25",
                [1.0, 1.0, 1.0],
                [0.25, 0.25, 0.25],
                512,
                0x2db71bcefe6cc8fe,
                0x2db71bcefe6cc8fe,
                0x2db71bcefe6cc8fe,
            ),
            (
                "ramp321",
                [3.0, 2.0, 1.0],
                [0.75, 0.5, 0.25],
                512,
                0xbaf660a9cb3c68ef,
                0x2ccfbc20bcc5a8e5,
                0x2db71bcefe6cc8fe,
            ),
            (
                "sd1-residual",
                [0.75, 0.6, 0.5],
                [0.7875, 0.575, 0.95],
                512,
                0x9edadcc940ed770a,
                0xa4d5797089736e79,
                0x51d88627df287325,
            ),
            (
                "dp2-residual",
                [0.75, 0.6, 0.5],
                [0.95, 0.6, 0.75],
                512,
                0x7e00703b224868cd,
                0xa4d5797089736e79,
                0x51d88627df287325,
            ),
            (
                "large",
                [5.0, 4.0, 3.0],
                [1.0, 1.0, 1.0],
                512,
                0xd15293038f10956a,
                0x97eb8d069d4f05e1,
                0x3ede7c76f5fdf09b,
            ),
            (
                "clamp512",
                [40.0, 40.0, 40.0],
                [8.0, 8.0, 8.0],
                512,
                0xe65f46917b5a9067,
                0xe65f46917b5a9067,
                0xe65f46917b5a9067,
            ),
            (
                "wide",
                [1.0, 0.5, 0.25],
                [0.0625, 0.03125, 0.015625],
                512,
                0x37bade2729069e3d,
                0x51d88627df287325,
                0x51d88627df287325,
            ),
        ];
        for (name, cf1, cf2, size, a, c, b) in cases {
            let lut = tanh_lut(*cf1, *cf2);
            assert_eq!(lut.size, *size, "{name} size");
            assert_eq!(lut.tables[0].len(), *size as usize);
            // The native object order {A, C, B} maps to tables
            // [0], [1], [2] per the construction at 0x5db78..0x5db9c.
            assert_eq!(fnv_i16(&lut.tables[0]), *a, "{name} table A");
            assert_eq!(fnv_i16(&lut.tables[1]), *c, "{name} table C");
            assert_eq!(fnv_i16(&lut.tables[2]), *b, "{name} table B");
        }
    }

    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    #[test]
    fn fast_bilateral_coeffs_matches_native_battery() {
        let cases: &[(f64, f64, f64, f64, u64, usize)] = &[
            (0.1, 0.0025, 0.12, 0.025, 0x975960cbfae1050b, 838861),
            (0.1, 0.0025, 0.12, 0.075, 0xdda29d5169d15b99, 902643),
            (0.1, 0.0025, 0.12, 1.0, 0x58da43ca30bb8ebe, 998131),
            (0.1, 0.0025, 0.25, 0.025, 0xc27b254fa0b4d115, 839268),
            (0.1, 0.0025, 0.25, 0.075, 0x140b16bb2f12154d, 902800),
            (0.1, 0.0025, 0.25, 1.0, 0x80d0068682816730, 998148),
            (0.1, 0.0025, 0.0, 0.025, 0x215248525f088e07, 838656),
            (0.1, 0.0025, 0.0, 0.075, 0x3dc3c41758289377, 902491),
            (0.1, 0.0025, 0.0, 1.0, 0x286a2ff64359ec67, 998115),
            (0.1, 0.0, 0.12, 0.025, 0xc8776020da8fab62, 838861),
            (0.1, 0.0, 0.12, 0.075, 0x7f1c82cc17f23ae0, 902643),
            (0.1, 0.0, 0.12, 1.0, 0xcde667f400a5e152, 998131),
            (0.1, 0.0, 0.25, 0.025, 0xb65cb6f2aae6f1f9, 839268),
            (0.1, 0.0, 0.25, 0.075, 0xc9249874b9816b45, 902800),
            (0.1, 0.0, 0.25, 1.0, 0x27958f5be88086e6, 998148),
            (0.1, 0.0, 0.0, 0.025, 0x168fbffc95558f9b, 838656),
            (0.1, 0.0, 0.0, 0.075, 0x810bff1387ffed3e, 902491),
            (0.1, 0.0, 0.0, 1.0, 0x0f0fdfd9942fea1a, 998115),
            (0.1, 0.025, 0.12, 0.025, 0x68abdec7635194ec, 838861),
            (0.1, 0.025, 0.12, 0.075, 0x04d54cc1965d7cfd, 902643),
            (0.1, 0.025, 0.12, 1.0, 0xf0ed3e93ccd4b6c8, 998131),
            (0.1, 0.025, 0.25, 0.025, 0x675edc59d434edd2, 839268),
            (0.1, 0.025, 0.25, 0.075, 0x21ccac3a374f661c, 902800),
            (0.1, 0.025, 0.25, 1.0, 0x262019edb38215a8, 998148),
            (0.1, 0.025, 0.0, 0.025, 0x73dee64a3df9a7df, 838656),
            (0.1, 0.025, 0.0, 0.075, 0x6ee938d49c4bbcd3, 902491),
            (0.1, 0.025, 0.0, 1.0, 0xd48c8fb8b3c9788e, 998115),
            (0.005, 0.0025, 0.12, 0.025, 0x9f019ed7c8d9f174, 980636),
            (0.005, 0.0025, 0.12, 0.075, 0x6772edda7c44a434, 1006743),
            (0.005, 0.0025, 0.12, 1.0, 0x8d8f4418a9e043b1, 1036014),
            (0.005, 0.0025, 0.25, 0.025, 0x3544bf5f7ea00250, 981197),
            (0.005, 0.0025, 0.25, 0.075, 0xbe85f437d57b220a, 1006942),
            (0.005, 0.0025, 0.25, 1.0, 0x6e88d1caabb660b1, 1036030),
            (0.005, 0.0025, 0.0, 0.025, 0xfcc7ac38f4bcb510, 980112),
            (0.005, 0.0025, 0.0, 0.075, 0x84583c351dc91565, 1006556),
            (0.005, 0.0025, 0.0, 1.0, 0x19db1841dd38a440, 1035997),
            (0.005, 0.0, 0.12, 0.025, 0x986ba282e19e90e8, 980636),
            (0.005, 0.0, 0.12, 0.075, 0x7d47d096fd52f617, 1006743),
            (0.005, 0.0, 0.12, 1.0, 0x1b4aee349d44c197, 1036014),
            (0.005, 0.0, 0.25, 0.025, 0x9e0871220f75ac43, 981197),
            (0.005, 0.0, 0.25, 0.075, 0x8c2488a2d5d376bf, 1006942),
            (0.005, 0.0, 0.25, 1.0, 0x3893dfc5a21bf613, 1036030),
            (0.005, 0.0, 0.0, 0.025, 0x6abfdb63e918e8d2, 980112),
            (0.005, 0.0, 0.0, 0.075, 0xdcc407d585fb875d, 1006556),
            (0.005, 0.0, 0.0, 1.0, 0x0d87146019d162d8, 1035997),
            (0.005, 0.025, 0.12, 0.025, 0xad8d6721088b10d1, 980643),
            (0.005, 0.025, 0.12, 0.075, 0x2b066b2ff112b6b3, 1006744),
            (0.005, 0.025, 0.12, 1.0, 0x1c8409fb8c7bf5a6, 1036015),
            (0.005, 0.025, 0.25, 0.025, 0xbb2dd4aaa1b1a25f, 981198),
            (0.005, 0.025, 0.25, 0.075, 0x4699fcf083d6227b, 1006944),
            (0.005, 0.025, 0.25, 1.0, 0xa19a8d8e045cfa5e, 1036031),
            (0.005, 0.025, 0.0, 0.025, 0x9289fd43f04c61ac, 980114),
            (0.005, 0.025, 0.0, 0.075, 0x59b277f83fc4359e, 1006558),
            (0.005, 0.025, 0.0, 1.0, 0xff0aff0efb3064c9, 1035998),
            (1.0, 0.0025, 0.12, 0.025, 0xc4df640a056534ea, 417453),
            (1.0, 0.0025, 0.12, 0.075, 0x16d468899dcb639e, 666822),
            (1.0, 0.0025, 0.12, 1.0, 0xdc2c9c58d6f91f43, 917248),
            (1.0, 0.0025, 0.25, 0.025, 0x7dad3772002fb9b1, 417973),
            (1.0, 0.0025, 0.25, 0.075, 0x03df28b571d0848e, 667105),
            (1.0, 0.0025, 0.25, 1.0, 0xf913f7abdfa53a36, 917248),
            (1.0, 0.0025, 0.0, 0.025, 0x5309a873675da174, 416963),
            (1.0, 0.0025, 0.0, 0.075, 0xe9e92b6a5df66c57, 666565),
            (1.0, 0.0025, 0.0, 1.0, 0x54aa0b3777e56a33, 917248),
            (1.0, 0.0, 0.12, 0.025, 0xe510381228125d1e, 417453),
            (1.0, 0.0, 0.12, 0.075, 0x5e7950b3c044453e, 666822),
            (1.0, 0.0, 0.12, 1.0, 0xc7e8eea5b7fabe2b, 917248),
            (1.0, 0.0, 0.25, 0.025, 0x1c4db02bafbf0f8d, 417973),
            (1.0, 0.0, 0.25, 0.075, 0x1f58b050324deae6, 667105),
            (1.0, 0.0, 0.25, 1.0, 0x2852fcf82c5346ba, 917248),
            (1.0, 0.0, 0.0, 0.025, 0x219d1b7ad173e133, 416963),
            (1.0, 0.0, 0.0, 0.075, 0xcfaf89c833e38172, 666565),
            (1.0, 0.0, 0.0, 1.0, 0xf741d491e7c9c0c7, 917248),
            (1.0, 0.025, 0.12, 0.025, 0xc2ddcf215ad8733e, 417453),
            (1.0, 0.025, 0.12, 0.075, 0xefd48ef102763990, 666822),
            (1.0, 0.025, 0.12, 1.0, 0x0d0e65c42102aabb, 917248),
            (1.0, 0.025, 0.25, 0.025, 0x5440a356d033f9b6, 417973),
            (1.0, 0.025, 0.25, 0.075, 0x6a77b1ef1dfb1fda, 667105),
            (1.0, 0.025, 0.25, 1.0, 0xa35d6a6466abb674, 917248),
            (1.0, 0.025, 0.0, 0.025, 0x4dc83800bc875a3f, 416963),
            (1.0, 0.025, 0.0, 0.075, 0x3d38cae350285931, 666565),
            (1.0, 0.025, 0.0, 1.0, 0x793602e258b9c4ea, 917248),
            (0.05, 0.0025, 0.12, 0.025, 0xe6ff1498f8b87c39, 880672),
            (0.05, 0.0025, 0.12, 0.075, 0xef2ee5cb54186e5c, 936168),
            (0.05, 0.0025, 0.12, 1.0, 0xd31ba22ab57e247f, 1011748),
            (0.05, 0.0025, 0.25, 0.025, 0x5c4a29b1f13cfaca, 881115),
            (0.05, 0.0025, 0.25, 0.075, 0x2fd6c08ec162053f, 936341),
            (0.05, 0.0025, 0.25, 1.0, 0x5dfbef0fd57c0197, 1011760),
            (0.05, 0.0025, 0.0, 0.025, 0x1598d56ee076aac6, 880259),
            (0.05, 0.0025, 0.0, 0.075, 0xf5d0596897ce0640, 936008),
            (0.05, 0.0025, 0.0, 1.0, 0xeb29e2c9c79c2ba2, 1011731),
            (0.05, 0.0, 0.12, 0.025, 0xed8be9eb58c695f9, 880672),
            (0.05, 0.0, 0.12, 0.075, 0x20476cd2f4ac2781, 936168),
            (0.05, 0.0, 0.12, 1.0, 0xfeda4b8b3fa89dd2, 1011748),
            (0.05, 0.0, 0.25, 0.025, 0xd188eb9128614407, 881115),
            (0.05, 0.0, 0.25, 0.075, 0x5a98c82807f977b2, 936341),
            (0.05, 0.0, 0.25, 1.0, 0x309296b6f942f41c, 1011760),
            (0.05, 0.0, 0.0, 0.025, 0xb8aaad1e22f25887, 880259),
            (0.05, 0.0, 0.0, 0.075, 0x7786598ec8d0c390, 936008),
            (0.05, 0.0, 0.0, 1.0, 0x08f4a967131c4ede, 1011731),
            (0.05, 0.025, 0.12, 0.025, 0x10c44103adae37b0, 880672),
            (0.05, 0.025, 0.12, 0.075, 0xc8ddd6a2ec229f44, 936168),
            (0.05, 0.025, 0.12, 1.0, 0xa74293f44fa7c55d, 1011748),
            (0.05, 0.025, 0.25, 0.025, 0x4dd127f6400f4535, 881119),
            (0.05, 0.025, 0.25, 0.075, 0x9e92441c9f628c74, 936342),
            (0.05, 0.025, 0.25, 1.0, 0x3dd5448cb48f951d, 1011761),
            (0.05, 0.025, 0.0, 0.025, 0xb6e274447f56dbab, 880259),
            (0.05, 0.025, 0.0, 0.075, 0xc7bc09d3c0a0f99a, 936008),
            (0.05, 0.025, 0.0, 1.0, 0x0a2b087e7603e896, 1011731),
        ];
        for (scaling, s38, s40, s48, expected, total) in cases {
            let coeffs = fast_bilateral_coeffs(*scaling, *s38, *s40, *s48);
            // The harness: h = fnv(A); h = h*prime ^ fnv(B); then per
            // bin h = h*prime ^ fnv(bin floats).
            let mut hash = fnv_i32(&coeffs.low);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3) ^ fnv_i32(&coeffs.high);
            let mut count = 0_usize;
            for (i, bin) in coeffs.weights.iter().enumerate() {
                assert_eq!(bin.len(), (coeffs.high[i] - coeffs.low[i]) as usize,);
                hash = hash.wrapping_mul(0x0000_0100_0000_01b3) ^ fnv_f32(bin);
                count += bin.len();
            }
            assert_eq!(hash, *expected, "scaling {scaling}");
            assert_eq!(count, *total, "scaling {scaling} total");
        }
    }

    /// The full entry-level dumps from the native battery pin the
    /// mirror and boundary-fixup semantics beyond the hashes.
    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    #[test]
    fn tanh_lut_matches_native_full_dump() {
        let lut = tanh_lut([3.0, 2.0, 1.0], [0.75, 0.5, 0.25]);
        assert_eq!(lut.size, 512);
        // kernel-results.txt FB FULL-style rows for the ramp321
        // pair, first eight entries of each table.
        let expected_a = [0, 1, 1, 2, 2, 2, 3, 3];
        let expected_c = [0, 0, 1, 1, 1, 2, 2, 2];
        let expected_b = [0, 0, 0, 1, 1, 1, 1, 1];
        for i in 0..8 {
            assert_eq!(lut.tables[0][i], expected_a[i], "A[{i}]");
            assert_eq!(lut.tables[1][i], expected_c[i], "C[{i}]");
            assert_eq!(lut.tables[2][i], expected_b[i], "B[{i}]");
        }
        // The mirror: table[size - k] == -table[k].
        for k in 1..256 {
            assert_eq!(lut.tables[0][512 - k], -lut.tables[0][k]);
        }
        // The boundary fixup: table[half] == table[half + 1].
        assert_eq!(lut.tables[0][256], lut.tables[0][257]);
    }

    /// All 35 scales from crp-complete-results.txt. Each table hash
    /// comes from the native library, including fractional scales.
    /// Scale 0.6 checks the zero-anchor period branch.
    #[test]
    fn compute_resample_params_matches_native_battery() {
        let cases: &[(f32, u32, u32, i32, &[u32], u64)] = &[
            (0.0625, 32, 1, -8, &[16], 0x7f9dd1e3e66f03c5),
            (0.05, 40, 1, -10, &[20], 0xf34027766c0f89d9),
            (0.125, 16, 1, -4, &[8], 0x3a0fe3d5da4854c5),
            (0.1875, 11, 3, -3, &[6, 5, 5], 0x54ba064f311ad2e5),
            (0.25, 8, 1, -2, &[4], 0xd076633462887ac5),
            (0.3, 7, 3, -2, &[4, 3, 3], 0xb594b657f46c02f4),
            (0.3125, 7, 5, -2, &[4, 3, 3, 3, 3], 0x4d3618e90de35ef4),
            (0.33333334, 5, 1, -1, &[3], 0x5e4dc14a2ff9cb0f),
            (0.36, 7, 1, -1, &[2], 0xab19293a7f9e4a07),
            (0.375, 6, 3, -1, &[2, 3, 3], 0x860e1fba1c86c10f),
            (
                0.376,
                6,
                47,
                -1,
                &[
                    2, 3, 3, 2, 3, 3, 2, 3, 3, 2, 3, 3, 2, 3, 3, 2, 3, 3, 2, 3, 3, 2, 3, 3, 2, 3,
                    2, 3, 3, 2, 3, 3, 2, 3, 3, 2, 3, 3, 2, 3, 3, 2, 3, 3, 2, 3, 3,
                ],
                0x3d7bef47aac1f945,
            ),
            (
                0.38,
                6,
                19,
                -1,
                &[2, 3, 3, 2, 3, 2, 3, 3, 2, 3, 3, 2, 3, 3, 2, 3, 2, 3, 3],
                0xc48a8ce554a80576,
            ),
            (0.4, 5, 2, -1, &[2, 3], 0x1bc4c38d0949e132),
            (0.4375, 5, 7, -1, &[2, 2, 3, 2, 2, 3, 2], 0x93fe54637fddbe0e),
            (
                0.45,
                5,
                9,
                -1,
                &[2, 2, 3, 2, 2, 2, 2, 3, 2],
                0x793c243d66b8a733,
            ),
            (0.5, 4, 1, -1, &[2], 0x9b2f1bb53e5bc065),
            (
                0.5625,
                4,
                9,
                -1,
                &[2, 2, 1, 2, 2, 2, 2, 1, 2],
                0xb86cc035a8515849,
            ),
            (0.6, 5, 1, -1, &[2], 0xacf6cd31327fd439),
            (0.625, 4, 5, -1, &[2, 1, 2, 2, 1], 0xfdc95829938b2fa2),
            (
                0.6875,
                3,
                11,
                -1,
                &[2, 1, 2, 1, 2, 1, 1, 2, 1, 2, 1],
                0xd8e5be1c00278267,
            ),
            (0.7, 3, 7, -1, &[2, 1, 2, 1, 1, 2, 1], 0xfdf17147c4a8f9b9),
            (0.75, 3, 3, -1, &[2, 1, 1], 0x4fa6e8c0799580e1),
            (
                0.8125,
                3,
                13,
                -1,
                &[2, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1],
                0xcfa3f4f65caa6a0c,
            ),
            (0.875, 3, 7, -1, &[2, 1, 1, 1, 1, 1, 1], 0xcedcbb349d494c96),
            (
                0.9375,
                3,
                15,
                -1,
                &[2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                0x6e8a6ae8cda2eaa0,
            ),
            (1.0, 1, 1, 0, &[1], 0x4b72477f9c5c2f98),
            (
                1.14285715,
                2,
                8,
                -1,
                &[1, 1, 1, 1, 1, 1, 1, 0],
                0xa325f76d9f09ba86,
            ),
            (1.3333334, 2, 4, -1, &[1, 1, 1, 0], 0x7833e11fbd04c4a5),
            (1.5, 2, 3, -1, &[1, 1, 0], 0x2903ab696398b73b),
            (1.6, 2, 8, -1, &[1, 1, 0, 1, 0, 1, 1, 0], 0xeb9cfa2244f62455),
            (2.0, 2, 2, -1, &[1, 0], 0x45d388a7237d30b5),
            (
                2.6666667,
                2,
                8,
                -1,
                &[1, 0, 0, 1, 0, 0, 1, 0],
                0x09e0cf5a9fc44455,
            ),
            (3.0, 2, 3, -1, &[1, 0, 0], 0x1b19e2987852d1af),
            (4.0, 2, 4, -1, &[0, 1, 0, 0], 0x701ec0880d000725),
            (8.0, 2, 8, -1, &[0, 0, 0, 1, 0, 0, 0, 0], 0x80760f1086d9ca35),
        ];
        for (scale, kernel_x, kernel_y, tail, offsets, table_hash) in cases {
            let params = compute_resample_params(*scale);
            assert_eq!(params.kernel_x, *kernel_x, "kernel_x at {scale}");
            assert_eq!(params.kernel_y, *kernel_y, "kernel_y at {scale}");
            assert_eq!(params.tail, *tail, "tail at {scale}");
            assert_eq!(params.offsets.as_slice(), *offsets, "offsets at {scale}");
            assert_eq!(
                params.table.len(),
                (*kernel_x * *kernel_y) as usize,
                "table length at {scale}"
            );
            assert_eq!(fnv_f32(&params.table), *table_hash, "table at {scale}");
        }
    }

    #[test]
    #[cfg(panic = "unwind")]
    fn compute_resample_params_rejects_invalid_scales() {
        for scale in [0.0, -1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(std::panic::catch_unwind(|| compute_resample_params(scale)).is_err());
        }
    }

    #[test]
    #[cfg(panic = "unwind")]
    #[should_panic(expected = "has no phase period within the 64-probe table")]
    fn compute_resample_params_rejects_native_period_overread() {
        compute_resample_params(0.374);
    }

    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    #[test]
    fn fast_bilateral_coeffs_matches_native_full_dump() {
        let coeffs = fast_bilateral_coeffs(0.1, 0.0025, 0.12, 0.025);
        let expected: &[(i32, i32, &[f32])] = &[
            (0, 1, &[1.0]),
            (0, 4, &[0.163158819, 1.0, 0.163158819, 0.000708667096]),
            (0, 6, &[0.0822854489, 0.535587907, 1.0, 0.535587907]),
            (0, 9, &[0.0575140305, 0.281053096, 0.728110135, 1.0]),
        ];
        for (bin, (low, high, weights)) in expected.iter().enumerate() {
            assert_eq!(coeffs.low[bin], *low, "bin {bin} low");
            assert_eq!(coeffs.high[bin], *high, "bin {bin} high");
            assert_eq!(
                &coeffs.weights[bin][..weights.len()],
                weights.as_ref(),
                "bin {bin}"
            );
        }
    }
}
