//! SPP (Sigma Photo Pro) denoise kernels, ported from the native
//! `ReduceNoiseTiled.cpp` chain of `lib99_MerillDLLForMac.dylib`
//! (archived captures and field maps: x3f-docs
//! `followup-tiled-denoise/`). This is the explicit SPP denoise
//! option under construction; the existing NLM/BM3D denoise stays
//! the default and untouched.
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
//! exact operation order and pinned on every scale of the native
//! battery; the kernel table is pinned bit-for-bit on the scales
//! listed in the function documentation (the fractional-downscale
//! table has an open one-ulp divergence, documented there).

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
/// computes `1 - clamped` in f64 and narrows once (the native
/// multiplies by a constant 1.0 gain, a bit-exact no-op, omitted
/// here); the upscaling weight is the f32 tent
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
/// Parity domain. The geometry (`kernel_x`, `kernel_y`, `tail`,
/// `offsets`, and the crop) matches the native bit-for-bit on
/// every scale of the battery below (35 rows, recorded across the
/// native sweep files of the stage 7c checkpoint). The kernel
/// table is bit-for-bit on every scale whose tent products are
/// exactly representable (0.0625, 0.125, 0.25, 0.5, 0.6), and on
/// every upsampling scale (1.0, 1.14285715, 1.3333334, 1.5, 1.6,
/// 2.0, 2.6666667, 3.0, 4.0, 8.0; the up path has no multiply by
/// the scale). On the remaining downscaling scales (0.05, 0.4,
/// 0.7, 0.375 and the other fractional eighths) the table differs
/// from the native by scattered one-ulp entries. The
/// investigation record (the stage 7c checkpoint in x3f-docs)
/// rules out a rounding-mode difference (the runtime FPCR is
/// round-to-nearest and forcing it reproduces the native table)
/// and an instruction misreading (both the arm64 and x86_64
/// slices decode to this exact pipeline, and in-process hardware
/// probes confirm the arithmetic). The native's tent weight for
/// the differing taps is unreachable by any single-rounding
/// variant of `1 - fl32(|d| * scale)`, so the native computes
/// those products by a subtly different formula that a later
/// checkpoint must recover (the anomalies correlate with
/// `|d| * scale` landing within one to two ulps of a dyadic
/// rational). The two retained SPP references only use verified
/// scales (identity and 0.5), so this open item does not affect
/// them.
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
                // Clamp in f32 (fmul, then fcsel against 0 and 1),
                // then 1 - clamped in f64, narrowed once by fcvt.
                let clamped = magnitude * scale;
                let clamped = if clamped < 0.0 { 0.0 } else { clamped };
                let clamped = if clamped > 1.0 { 1.0 } else { clamped };
                (1.0 - f64::from(clamped)) as f32
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

    /// The 16-scale ComputeResampleParams battery from the native
    /// oracle (kernel-results.txt plus the stage 7c sweeps, all
    /// deterministic across runs): scale, kernel_x, kernel_y, tail,
    /// the per-phase offsets, and the table FNV-1a over the f32
    /// bits for the bit-verified scales (None elsewhere; see the
    /// function documentation for the open divergence on scales
    /// with inexact tent products). Scale 0.6 exercises the
    /// zero-anchor period branch.
    #[test]
    fn compute_resample_params_matches_native_battery() {
        let cases: &[(f32, u32, u32, i32, &[u32], Option<u64>)] = &[
            (0.0625, 32, 1, -8, &[16], Some(0x7f9dd1e3e66f03c5)),
            (0.05, 40, 1, -10, &[20], None),
            (0.125, 16, 1, -4, &[8], Some(0x3a0fe3d5da4854c5)),
            (0.1875, 11, 3, -3, &[6, 5, 5], None),
            (0.25, 8, 1, -2, &[4], Some(0xd076633462887ac5)),
            (0.3, 7, 3, -2, &[4, 3, 3], None),
            (0.3125, 7, 5, -2, &[4, 3, 3, 3, 3], None),
            (0.33333334, 5, 1, -1, &[3], None),
            (0.36, 7, 1, -1, &[2], None),
            (0.375, 6, 3, -1, &[2, 3, 3], None),
            (
                0.376,
                6,
                47,
                -1,
                &[
                    2, 3, 3, 2, 3, 3, 2, 3, 3, 2, 3, 3, 2, 3, 3, 2, 3, 3, 2, 3, 3, 2, 3, 3, 2, 3,
                    2, 3, 3, 2, 3, 3, 2, 3, 3, 2, 3, 3, 2, 3, 3, 2, 3, 3, 2, 3, 3,
                ],
                None,
            ),
            (
                0.38,
                6,
                19,
                -1,
                &[2, 3, 3, 2, 3, 2, 3, 3, 2, 3, 3, 2, 3, 3, 2, 3, 2, 3, 3],
                None,
            ),
            (0.4, 5, 2, -1, &[2, 3], None),
            (0.4375, 5, 7, -1, &[2, 2, 3, 2, 2, 3, 2], None),
            (0.45, 5, 9, -1, &[2, 2, 3, 2, 2, 2, 2, 3, 2], None),
            (0.5, 4, 1, -1, &[2], Some(0x9b2f1bb53e5bc065)),
            (0.5625, 4, 9, -1, &[2, 2, 1, 2, 2, 2, 2, 1, 2], None),
            (0.6, 5, 1, -1, &[2], Some(0xacf6cd31327fd439)),
            (0.625, 4, 5, -1, &[2, 1, 2, 2, 1], None),
            (0.6875, 3, 11, -1, &[2, 1, 2, 1, 2, 1, 1, 2, 1, 2, 1], None),
            (0.7, 3, 7, -1, &[2, 1, 2, 1, 1, 2, 1], None),
            (0.75, 3, 3, -1, &[2, 1, 1], None),
            (
                0.8125,
                3,
                13,
                -1,
                &[2, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1],
                None,
            ),
            (0.875, 3, 7, -1, &[2, 1, 1, 1, 1, 1, 1], None),
            (
                0.9375,
                3,
                15,
                -1,
                &[2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                None,
            ),
            (1.0, 1, 1, 0, &[1], Some(0x4b72477f9c5c2f98)),
            (
                1.14285715,
                2,
                8,
                -1,
                &[1, 1, 1, 1, 1, 1, 1, 0],
                Some(0xa325f76d9f09ba86),
            ),
            (1.3333334, 2, 4, -1, &[1, 1, 1, 0], Some(0x7833e11fbd04c4a5)),
            (1.5, 2, 3, -1, &[1, 1, 0], Some(0x2903ab696398b73b)),
            (
                1.6,
                2,
                8,
                -1,
                &[1, 1, 0, 1, 0, 1, 1, 0],
                Some(0xeb9cfa2244f62455),
            ),
            (2.0, 2, 2, -1, &[1, 0], Some(0x45d388a7237d30b5)),
            (
                2.6666667,
                2,
                8,
                -1,
                &[1, 0, 0, 1, 0, 0, 1, 0],
                Some(0x09e0cf5a9fc44455),
            ),
            (3.0, 2, 3, -1, &[1, 0, 0], Some(0x1b19e2987852d1af)),
            (4.0, 2, 4, -1, &[0, 1, 0, 0], Some(0x701ec0880d000725)),
            (
                8.0,
                2,
                8,
                -1,
                &[0, 0, 0, 1, 0, 0, 0, 0],
                Some(0x80760f1086d9ca35),
            ),
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
            if let Some(expected) = table_hash {
                assert_eq!(fnv_f32(&params.table), *expected, "table at {scale}");
            }
        }
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
