//! Apply ColorDQ, Sigma's small per-pixel color correction, before the tone curve.
//!
//! ColorDQ uses the three channels of one pixel, not neighboring pixels. It works
//! after the camera's color matrix, on the tone calculation's 0..4095 brightness
//! scale. Its tables also accept negative and above-white values. Brightening
//! the input does not by itself increase the correction's strength.

use super::invalid;
use crate::Error;

// Sigma scales the correction strength by the capture-to-sensor ISO ratio,
// calculated with 32-bit floats and capped at four. The extra one-stop brightness
// gain in the camera profile must not also multiply this strength.
pub(super) const MAX_ISO_GAIN: f32 = 4.0;

// Reject correction strengths above 1024 brightness units before building the
// table. This bounds the metadata accepted here, not a user-adjustable setting.
const MAX_AMPLITUDE: f64 = 1024.0;

// tanh, the hyperbolic tangent, is a smooth S-shaped function that controls the
// correction near zero. A cosine envelope, a smooth taper down to zero, reduces
// the correction farther from zero. PHASE_SCALE controls that taper's width.
const TANH_SHAPE: f64 = 0.8;
const PHASE_SCALE: f64 = 0.25;

// Keep Sigma's exact stored constants for the taper cutoff and table span.
// Substituting exact pi or 2*pi changes rounding near the taper's edge.
const PHASE_CUTOFF: f64 = f64::from_bits(0x400921ff2e48e8a7);
const TABLE_SPAN: f64 = f64::from_bits(0x40192425aee631f9);
const TABLE_PERIODS: [usize; 5] = [512, 1024, 2048, 4096, 8192];
const MAX_TABLE_PERIOD: usize = 8192;

// Sigma rounds by adding a large floating-point constant and reading the low
// 32 bits as a signed integer. The small bias settles halfway cases. Keep this
// method rather than round(), which differs for some negative and halfway inputs
// and would therefore choose different correction-table entries.
const ROUNDING_BIAS: f64 = f64::from_bits(0x3e501b2b29a4692b);
const ROUNDING_MAGIC: f64 = f64::from_bits(0x4338000000000000);

pub(super) struct ColorDq {
    table: Vec<[i32; 3]>,
}

impl ColorDq {
    /// Build the correction table from the file's per-channel strengths after ISO
    /// scaling. Strengths and inputs use the same 0..4095 brightness scale.
    pub(super) fn new(amplitudes: [f64; 3]) -> Result<Self, Error> {
        if amplitudes
            .iter()
            .any(|v| !v.is_finite() || *v <= 0.0 || *v > MAX_AMPLITUDE)
        {
            return Err(invalid("invalid ColorDQ amplitude"));
        }
        let phase_scale = PHASE_SCALE
            / amplitudes
                .iter()
                .map(|v| v / TANH_SHAPE)
                .fold(0.0, f64::max);
        let needed = (TABLE_SPAN / phase_scale) as usize;
        let size = TABLE_PERIODS
            .into_iter()
            .find(|n| *n >= needed)
            .unwrap_or(MAX_TABLE_PERIOD);
        let table = (0..size)
            .map(|i| {
                let signed = if i < size / 2 {
                    i as i32
                } else {
                    i as i32 - size as i32
                };
                let magnitude = signed.unsigned_abs() as f64;
                let phase = phase_scale * magnitude;
                let envelope = if phase < PHASE_CUTOFF {
                    (1.0 + phase.cos()) * 0.5
                } else {
                    0.0
                };
                amplitudes.map(|a| {
                    signed.signum()
                        * (a * (TANH_SHAPE * magnitude / a).tanh() * envelope + 0.5).trunc() as i32
                })
            })
            .collect();
        Ok(Self { table })
    }

    /// Correct one pixel after the camera color matrix and before the tone curve.
    /// Supply finite channel values on the 0..4095 scale, not 16-bit display RGB.
    pub(super) fn apply(&self, values: [f64; 3]) -> [f64; 3] {
        let mask = self.table.len() - 1;
        let rounded =
            values.map(|v| ((v + ROUNDING_BIAS + ROUNDING_MAGIC).to_bits() as u32 as i32) as i64);
        let residual: [i64; 3] =
            std::array::from_fn(|c| rounded[c] - self.table[rounded[c] as usize & mask][c] as i64);
        // Estimate the shared brightness from the values left after the first
        // table lookup. Sigma gives the middle channel twice the weight of the
        // other channels. Shifting divides by four and rounds downward, including
        // for negative values, so ordinary integer division is not equivalent.
        let mean = (residual[0] + 2 * residual[1] + residual[2]) >> 2;
        std::array::from_fn(|c| {
            values[c] - self.table[(residual[c] - mean) as usize & mask][c] as f64
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn colordq_matches_native_tables_and_points() {
        // These FNV-1a checksums summarize every byte in Sigma's correction tables.
        // Encode each signed entry as a little-endian 32-bit integer so even a
        // one-entry rounding difference changes the checksum.
        for (amplitudes, expected) in [
            ([7.9921875; 3], 0xa12aaac649ea2818_u64),
            ([16.0; 3], 0x0a85ba2d5d7293e9_u64),
            ([4.0, 8.0, 16.0], 0x77bb8a9ee39543c8_u64),
        ] {
            let dq = ColorDq::new(amplitudes).unwrap();
            assert_eq!(dq.table.len(), 512);
            let mut hash = 0xcbf29ce484222325_u64;
            for value in dq.table.iter().flatten() {
                for byte in value.to_le_bytes() {
                    hash = (hash ^ byte as u64).wrapping_mul(0x100000001b3);
                }
            }
            assert_eq!(hash, expected);
        }
        let dq = ColorDq::new([16.0; 3]).unwrap();
        for (input, expected) in [
            ([-512.0, -512.0, -128.0], [-501.0, -501.0, -127.0]),
            ([-512.0, -512.0, 0.5], [-504.0, -504.0, 8.5]),
            ([-512.0, -0.5, 10.0], [-520.0, -8.5, 3.0]),
            ([-128.0, 40.0, -0.5], [-118.0, 26.0, -11.5]),
        ] {
            assert_eq!(dq.apply(input), expected);
        }
        for i in -4096..8192 {
            let gray = [i as f64 * 0.5; 3];
            assert_eq!(dq.apply(gray), gray);
        }
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(ColorDq::new([bad; 3]).is_err());
        }
    }
}
