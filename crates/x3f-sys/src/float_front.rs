//! Native float-image ApplyGains kernel for the ordered Merrill float
//! frontend. Verified bit-for-bit against the supplied Merrill library
//! (ARM64 0x39b38..0x3a42c) through the archived differential oracle.
//! Evidence: x3f-docs `spp-rgb-trace/followup-float-preprocess/`.
//!
//! The operation is
//! `out = f32((f64(sample) - black) * (gain * spatial_value))`
//! with no display clamp and no gamma. It works on sensor-count float
//! samples before any normalization. The ordered frontend follow-up
//! wires this kernel into the conversion path; production still uses
//! the normalized uint16 frontend.

use crate::control::{Error, Result};
use crate::Control;

/// One channel of the spatial gain table. `values` is row-major with
/// `rows` rows and `columns` knots per row. The CAMF builders in
/// `spatial_gain.rs` produce this layout.
#[derive(Clone, Debug)]
pub struct GainTable {
    pub values: Vec<f32>,
    pub columns: usize,
    pub rows: usize,
}

impl GainTable {
    pub fn new(values: Vec<f32>, columns: usize, rows: usize) -> Self {
        assert_eq!(values.len(), columns * rows, "gain table shape mismatch");
        assert!(
            columns >= 2 && rows >= 2,
            "gain table needs two knots per axis"
        );
        Self {
            values,
            columns,
            rows,
        }
    }
}

/// FNV-1a over the little-endian bytes of the f32 raster. Matches the
/// archived native-oracle hashes (`followup-float-preprocess`).
pub(crate) fn fnv1a_f32(values: &[f32]) -> u64 {
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(values.as_ptr() as *const u8, values.len() * 4) };
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

/// Native WB radial shading for the ordered Merrill float frontend,
/// stage 2. Reconstructed from the pinned preprocess instructions
/// (0xa64c8..0xa6520 geometry, 0xa8234..0xa83b4 arithmetic). The pass
/// runs over the gained float raster after ApplyGains, in
/// `activeMinusKeep` coordinates. C1 stays unchanged.
///
/// Geometry (0xa64c8..0xa6520):
/// - `activeMinusKeep = ActiveImageArea - KeepImageArea` component-wise
///   (top and bottom both subtract `keep.top`), a half-open rect.
/// - `margin = 32` when the resolution string contains "HI", else `16`
///   (native computes `scale * +-32` with scale 1.0 or 0.5, then
///   truncates toward zero).
/// - `valid.left = DarkShieldColRange[1] + 6`,
///   `valid.right = DarkShieldColRange[2] - 8` (matrix elements 1 and
///   2 of the 2x2 entry), `valid.top/bottom = activeMinusKeep.top/bottom
///   -/+ margin`.
/// - Half-extents truncate toward zero: `(right - left)` with a `+1`
///   correction when negative, then arithmetic shift by one.
/// - Center `= (valid.left + halfX, valid.top + halfY)`.
///
/// Arithmetic (0xa8234..0xa83b4), exact native order:
/// - Denominator `sqrt(contains_MED ? fma(2*halfY, 2*halfY, halfX*halfX)
///   : fma(halfY, halfY, halfX*halfX))` — the stretch literal is "MED"
///   (0xa6638), selected at 0xa82b8/0xa8364 with StrStr.
/// - Per pixel `dx = x - centerX`, `dy = y - centerY`,
///   `r2 = sqrt(stretch ? fma(2dy, 2dy, dx*dx) : fma(dy, dy, dx*dx)) /
///   denom`, `t = r2 * r2`, `t2 = t * t`.
/// - `factorC0 = f32(fma(a, t2, t*b) + 1.0)`, `factorC2 = f32(fma(c,
///   t2, t*d) + 1.0)`; each multiplies its f32 channel in place.
#[derive(Clone, Debug)]
pub struct RadialShading {
    a: f64,
    b: f64,
    c: f64,
    d: f64,
    center_x: f64,
    center_y: f64,
    denom: f64,
    stretch_y: bool,
}

impl RadialShading {
    /// Build the stage from CAMF rectangles and coefficients. `active`
    /// and `keep` are half-open `GetRectFromCamfile` rects; `col_range`
    /// holds the four elements of the DarkShieldColRange 2x2 matrix.
    pub fn new(
        active: [i32; 4],
        keep: [i32; 4],
        col_range: [i32; 4],
        coefficients: [f64; 4],
        resolution: &str,
    ) -> Result<Self> {
        // a5e50: 4-wide SIMD subtract, activeMinusKeep = active - keep.
        let minus_keep = [
            active[0] - keep[0],
            active[1] - keep[1],
            active[2] - keep[0],
            active[3] - keep[1],
        ];
        let hi = resolution.contains("HI");
        // 0xa64d4..0xa6508: scale 1.0 for HI else 0.5; valid.top/bottom =
        // fcvtzs(i32(activeMinusKeep.top/bottom) fmadd scale * -/+32).
        // The subtraction result is i32; scvtf converts it to f32, the
        // fmadd adds scale*(+-32) in one fused op, fcvtzs truncates.
        let scale = if hi { 1.0_f32 } else { 0.5_f32 };
        let valid_top = scale.mul_add(-32.0, minus_keep[1] as f32) as i32;
        let valid_bottom = scale.mul_add(32.0, minus_keep[3] as f32) as i32;
        let valid_left = col_range[1] + 6;
        let valid_right = col_range[2] - 8;
        // 0xa8234: (right - left) with +1 when negative (truncate toward
        // zero), then arithmetic shift by one.
        let truncate_half = |range: i32| {
            let delta = if range < 0 { range + 1 } else { range };
            delta >> 1
        };
        let half_x = truncate_half(valid_right - valid_left);
        let half_y = truncate_half(valid_bottom - valid_top);
        let [a, b, c, d] = coefficients;
        let stretch_y = resolution.contains("MED");
        let half_x_f = f64::from(half_x);
        let half_y_f = f64::from(half_y);
        let denom = if stretch_y {
            (2.0 * half_y_f)
                .mul_add(2.0 * half_y_f, half_x_f * half_x_f)
                .sqrt()
        } else {
            half_y_f.mul_add(half_y_f, half_x_f * half_x_f).sqrt()
        };
        if [a, b, c, d, denom].iter().any(|v| !v.is_finite()) || denom <= 0.0 {
            return Err(Error::InvalidData("invalid Merrill radial calibration"));
        }
        Ok(Self {
            a,
            b,
            c,
            d,
            center_x: f64::from(valid_left) + f64::from(half_x),
            center_y: f64::from(valid_top) + f64::from(half_y),
            denom,
            stretch_y,
        })
    }

    /// One pixel's factors, native order. Exposed for tests.
    fn factors(&self, x: i32, y: i32) -> (f32, f32) {
        let dx = f64::from(x) - self.center_x;
        let dy = f64::from(y) - self.center_y;
        let squared = dx * dx;
        let unstretched = dy.mul_add(dy, squared);
        let stretched = (dy + dy).mul_add(dy + dy, squared);
        let selected = if self.stretch_y {
            stretched
        } else {
            unstretched
        };
        let radius = selected.sqrt() / self.denom;
        let t = radius * radius;
        let t2 = t * t;
        let bottom = self.a.mul_add(t2, t * self.b) + 1.0;
        let top = self.c.mul_add(t2, t * self.d) + 1.0;
        (bottom as f32, top as f32)
    }

    /// Apply the pass in place over the gained float raster. `data`
    /// holds interleaved RGB f32 samples; `row_stride` may exceed
    /// `width * 3`.
    pub fn apply(
        &self,
        data: &mut [f32],
        width: usize,
        height: usize,
        row_stride: usize,
        control: Control<'_>,
    ) -> Result<()> {
        control.check()?;
        if width == 0 || height == 0 {
            return Err(Error::InvalidData("empty Merrill radial image"));
        }
        if row_stride < width * 3 || data.len() < row_stride * (height - 1) + width * 3 {
            return Err(Error::InvalidData("invalid Merrill radial image layout"));
        }
        for y in 0..height {
            let row = &mut data[y * row_stride..y * row_stride + width * 3];
            for x in 0..width {
                let (bottom, top) = self.factors(x as i32, y as i32);
                let pixel = &mut row[3 * x..3 * x + 3];
                pixel[0] *= bottom;
                pixel[2] *= top;
            }
        }
        Ok(())
    }
}

/// One camera-listed bad-pixel record. Layout matches the native
/// `FBadPixel_F20`: row, column, then a packed u16 flags word
/// (bits 0..7 neighbor mask, bits 8..11 badness, bits 12..14 channel
/// mask C0=4/C1=2/C2=1).
#[derive(Clone, Copy, Debug)]
pub struct BadPixelRecord {
    pub row: u16,
    pub column: u16,
    pub flags: u16,
}

/// Stage 3 of the ordered float frontend: listed bad-pixel repair,
/// the scalar reconstruction of the native `ReplaceBadPixelsF20`
/// (0x3b108). Verified bit-for-bit against the native export over all
/// 35,840 standalone cases (256 neighbor masks x 7 channel masks,
/// badness cutoffs, override bounds, zero/negative finite neighborhoods
/// and geometric skips) plus 8 sequential lists / 1,044 records:
/// x3f-docs `followup-float-front/check_listed_repair.cpp`.
#[derive(Clone, Debug)]
pub struct ListedRepair {
    minimum_badness: u32,
}

impl ListedRepair {
    /// `badness` is the selected badness label; `offset` is the
    /// metadata BPBadnessOffset. The native minimum is unsigned
    /// subtraction `badness == 0 ? 0 : badness - offset`, not
    /// saturating (tested offsets above badness wrap).
    pub fn new(badness: u32, offset: u32) -> Self {
        Self {
            minimum_badness: if badness == 0 {
                0
            } else {
                badness.wrapping_sub(offset)
            },
        }
    }

    pub fn minimum_badness(&self) -> u32 {
        self.minimum_badness
    }

    /// Repair one record in place. Mirrors the native skip conditions:
    /// out-of-halo coordinates (x or y below 2 or at the outermost
    /// border), badness below the request minimum, or an empty
    /// neighbor mask.
    #[allow(clippy::too_many_lines)]
    fn replace_one(
        &self,
        data: &mut [f32],
        width: usize,
        height: usize,
        row_stride: usize,
        record: BadPixelRecord,
        override_mask: Option<u32>,
    ) {
        let x = record.column as usize;
        let y = record.row as usize;
        let channel_mask = (record.flags >> 12) as u32;
        let mut neighbors = u32::from(record.flags & 0xff);
        // The per-record override applies only within the caller's
        // supplied count, when 1 <= override <= 254 and the override is
        // numerically below the original mask.
        if let Some(value) = override_mask {
            if (1..=254).contains(&value) && value < neighbors {
                neighbors = value;
            }
        }
        let badness = u32::from((record.flags >> 8) & 0xf);
        if x < 2
            || y < 2
            || x >= width.saturating_sub(1)
            || y >= height.saturating_sub(1)
            || badness < self.minimum_badness
            || neighbors == 0
        {
            return;
        }
        let at = |data: &[f32], dx: i32, dy: i32, c: usize| -> f32 {
            let index = (y as i32 + dy) as usize * row_stride + (x as i32 + dx) as usize * 3 + c;
            data[index]
        };
        let mut sums = [0.0_f32; 3];
        let mut lo = [0.0_f32; 3];
        let mut hi = [0.0_f32; 3];
        let mut original = [0.0_f32; 3];
        let mut first = true;
        let mut count = 0_u32;
        let mut bit = 0_u32;
        for dy in -1..=1 {
            for dx in -1..=1 {
                if dx == 0 && dy == 0 {
                    continue;
                }
                if neighbors & (1 << bit) == 0 {
                    bit += 1;
                    continue;
                }
                bit += 1;
                count += 1;
                // Native re-checks the all-zero RGB extrema sentinel
                // after every neighbor (0x3b284..0x3b300), so an
                // all-zero neighbor resets both extrema.
                let lo_zero = lo == [0.0, 0.0, 0.0];
                let hi_zero = hi == [0.0, 0.0, 0.0];
                for c in 0..3 {
                    let value = at(data, dx, dy, c);
                    sums[c] += value;
                    if first || lo_zero {
                        lo[c] = value;
                    } else {
                        lo[c] = lo[c].min(value);
                    }
                    if first || hi_zero {
                        hi[c] = value;
                    } else {
                        hi[c] = hi[c].max(value);
                    }
                }
                first = false;
            }
        }
        for c in 0..3 {
            if count > 2 {
                sums[c] -= lo[c] + hi[c];
            }
            original[c] = at(data, 0, 0, c);
        }
        let divisor = (if count > 2 { count - 2 } else { count }) as f32;
        // Native evaluates the channel-sum differences before dividing.
        let delta = |sums: &[f32; 3], a: usize, b: usize| -> f32 { (sums[a] - sums[b]) / divisor };
        let mut proposed = original;
        for c in 0..3 {
            if channel_mask & (4 >> c) == 0 {
                continue;
            }
            if channel_mask == 7 {
                // All three channels bad: trimmed mean per channel.
                proposed[c] = sums[c] / divisor;
            } else {
                // One or two bad channels: anchor on the good
                // channels plus the channel-sum differences.
                let mut total = 0.0_f32;
                let mut good = 0;
                for d in 0..3 {
                    if channel_mask & (4 >> d) == 0 {
                        let value = original[d] + delta(&sums, c, d);
                        if good == 0 {
                            total = value;
                        } else {
                            total += value;
                        }
                        good += 1;
                    }
                }
                proposed[c] = if good == 2 { total * 0.5 } else { total };
            }
        }
        if proposed.iter().all(|v| v.is_finite()) {
            let index = y * row_stride + x * 3;
            data[index..index + 3].copy_from_slice(&proposed);
        }
    }

    /// Apply a sequential list in place. The native changes the image
    /// between records, so later records observe earlier repairs.
    pub fn apply(
        &self,
        data: &mut [f32],
        width: usize,
        height: usize,
        row_stride: usize,
        records: &[BadPixelRecord],
        overrides: &[u32],
        control: Control<'_>,
    ) -> Result<()> {
        control.check()?;
        if width < 4 || height < 4 {
            // The halo requirement (>=2 and < dim-1) needs at least 4.
            return Err(Error::InvalidData("invalid Merrill repair image size"));
        }
        if row_stride < width * 3 || data.len() < row_stride * (height - 1) + width * 3 {
            return Err(Error::InvalidData("invalid Merrill repair image layout"));
        }
        for (index, record) in records.iter().enumerate() {
            control.check()?;
            let override_mask = overrides.get(index).copied();
            self.replace_one(data, width, height, row_stride, *record, override_mask);
        }
        Ok(())
    }
}

/// The ordered Merrill float frontend, stage 1: ApplyGains.
///
/// `black` is the per-channel DarkShieldTop trimmed mean, `gain` is the
/// GatherParams request gain (native order: `((baseGain * FNumberGainFact)
/// * TempGainFact) * SensorAdjustmentGainFact`, where `baseGain` is
/// `wb + wb` or `wb / 0.55` by process version), and `spatial` holds the
/// three native spatial gain tables or `None` to select the global gain
/// alone (the native all-three gate).
#[derive(Clone, Debug)]
pub struct FloatFront {
    black: [f64; 3],
    gain: [f64; 3],
    spatial: Option<[GainTable; 3]>,
}

impl FloatFront {
    pub fn new(black: [f64; 3], gain: [f64; 3], spatial: Option<[GainTable; 3]>) -> Result<Self> {
        let valid = |v: [f64; 3]| v.iter().all(|v| v.is_finite());
        if let Some(tables) = spatial.as_ref() {
            if tables.iter().any(|t| t.columns < 2 || t.rows < 2) {
                return Err(Error::InvalidData("invalid Merrill float spatial tables"));
            }
        }
        if !valid(black) || !valid(gain) {
            return Err(Error::InvalidData(
                "invalid Merrill float front calibration",
            ));
        }
        Ok(Self {
            black,
            gain,
            spatial,
        })
    }

    pub fn black(&self) -> [f64; 3] {
        self.black
    }

    pub fn gain(&self) -> [f64; 3] {
        self.gain
    }

    pub fn spatial(&self) -> Option<&[GainTable; 3]> {
        self.spatial.as_ref()
    }

    /// Apply stage 1 over decoded sensor counts. `decoded` holds
    /// interleaved RGB u16 samples with `row_stride` u16 elements per row;
    /// the u16 to f32 conversion is exact. The result is an owned f32
    /// raster with the same layout, gained but not normalized.
    ///
    /// Cancellation is checked once before the single native pass; the
    /// native ApplyGains has no intra-stage cancellation point.
    pub fn apply(
        &self,
        decoded: &[u16],
        width: usize,
        height: usize,
        row_stride: usize,
        control: Control<'_>,
    ) -> Result<Vec<f32>> {
        control.check()?;
        if height == 0 || width == 0 {
            return Err(Error::InvalidData("empty Merrill float image"));
        }
        if self.spatial.is_some() && height < 2 {
            return Err(Error::InvalidData("invalid Merrill float image size"));
        }
        if row_stride < width * 3 || decoded.len() < row_stride * (height - 1) + width * 3 {
            return Err(Error::InvalidData("invalid Merrill float image layout"));
        }
        let mut data = Vec::with_capacity(decoded.len());
        // u16 to f32 is exact: every u16 value is representable in f32.
        data.extend(decoded.iter().map(|&v| f32::from(v)));
        let refs: [Option<&GainTable>; 3] = match &self.spatial {
            None => [None, None, None],
            Some(t) => [Some(&t[0]), Some(&t[1]), Some(&t[2])],
        };
        apply_gains(
            &mut data, width, height, row_stride, self.black, self.gain, refs,
        );
        Ok(data)
    }
}

/// Apply black offsets, channel gains and spatial tables in one pass.
///
/// `data` holds interleaved RGB float samples with `row_stride` floats
/// per row. `row_stride` may exceed `width * 3`; the padding floats are
/// never read or written. When any entry of `tables` is `None`, the
/// native all-three gate applies and every channel uses `gain` alone.
///
/// The spatial path requires `height >= 2`. The native operation divides
/// by `height - 1` and produces NaN for a single row.
pub fn apply_gains(
    data: &mut [f32],
    width: usize,
    height: usize,
    row_stride: usize,
    black: [f64; 3],
    gain: [f64; 3],
    tables: [Option<&GainTable>; 3],
) {
    assert!(
        row_stride >= width * 3,
        "row stride smaller than the pixel row"
    );
    assert!(
        data.len() >= row_stride * (height - 1) + width * 3,
        "buffer smaller than the image"
    );
    let spatial = tables.iter().all(Option::is_some);
    for y in 0..height {
        // Per-channel interpolated gain row. The vertical coordinate is
        // float32, and each knot is blended with one float32 fma.
        let mut row = [Vec::new(), Vec::new(), Vec::new()];
        let mut edge = [0.0_f64; 3];
        let mut span = [0.0_f64; 3];
        let mut value = [0.0_f64; 3];
        let mut slope = [0.0_f64; 3];
        let mut knot = [0_usize; 3];
        if spatial {
            for c in 0..3 {
                let table = tables[c].expect("spatial path requires all tables");
                // Native 0x39f74: (f32(y) / f32(height-1)) * f32(rows-1).
                let position = (y as f32 / (height - 1) as f32) * (table.rows - 1) as f32;
                let lo = position.floor() as usize;
                let hi = position.ceil() as usize;
                let fraction = position - lo as f32;
                row[c].resize(table.columns, 0.0);
                for x in 0..table.columns {
                    let a = table.values[lo * table.columns + x];
                    let b = table.values[hi * table.columns + x];
                    row[c][x] = fraction.mul_add(b - a, a);
                }
                // Native horizontal span: width / (columns - 1), not
                // width - 1. The slope numerator is a rounded float32
                // difference before the double conversion.
                span[c] = width as f64 / (table.columns - 1) as f64;
                edge[c] = span[c];
                value[c] = row[c][0] as f64;
                slope[c] = (row[c][1] - row[c][0]) as f64 / span[c];
            }
        }
        let pixels = &mut data[y * row_stride..y * row_stride + width * 3];
        for x in 0..width {
            for c in 0..3 {
                if spatial && x as f64 >= edge[c] {
                    knot[c] += 1;
                    value[c] = row[c][knot[c]] as f64;
                    slope[c] = (row[c][knot[c] + 1] - row[c][knot[c]]) as f64 / span[c];
                    edge[c] += span[c];
                }
                let adjusted = if spatial { gain[c] * value[c] } else { gain[c] };
                let index = 3 * x + c;
                pixels[index] = ((pixels[index] as f64 - black[c]) * adjusted) as f32;
                if spatial {
                    value[c] += slope[c];
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Archived native results: (width, height, spatial, fnv1a hash).
    /// Source: followup-float-preprocess/apply-gains-results.txt, an
    /// independent repeat confirmed every hash.
    const NATIVE_CASES: &[(usize, usize, bool, u64)] = &[
        (1, 5, false, 0x138328c54eef5e5f),
        (2, 5, false, 0xe74038c3d75ccf3e),
        (3, 5, false, 0xcce4f8b3ddcaf15b),
        (4, 5, false, 0xaf74bebb544d557c),
        (5, 5, false, 0x5f4802a8b6ca169d),
        (9, 5, false, 0xe98c4c467bd4bc02),
        (9, 13, true, 0x975b9335a4245ddf),
        (37, 5, false, 0x3ea06b0be3260cbb),
        (37, 13, true, 0x7951b2f5927245f1),
        (97, 5, false, 0x595afa5a2e4e4c46),
        (97, 13, true, 0xb63b750844e59ce2),
    ];

    fn fnv1a(values: &[f32]) -> u64 {
        super::fnv1a_f32(values)
    }

    fn synthetic_tables() -> [GainTable; 3] {
        std::array::from_fn(|c| {
            let columns = 3 + 2 * c;
            let rows = 3 + c;
            let values = (0..columns * rows)
                .map(|i| 0.8_f32 + (((i * 7 + c * 3) % 17) as f32) * 0.071_f32)
                .collect();
            GainTable::new(values, columns, rows)
        })
    }

    /// Port of the archived differential oracle. The padding floats
    /// (-12345.25) must stay untouched, and the kernel must reproduce
    /// every archived native hash bit-for-bit.
    #[test]
    fn apply_gains_matches_native_oracle() {
        let black = [128.125_f64, 255.3, -64.75];
        let gain = [2.21234567_f64, 0.98765432, 4.4433322];
        let tables = synthetic_tables();
        let mut cases = 0;
        let mut samples = 0;
        for &(width, height, spatial, expected_hash) in NATIVE_CASES {
            let stride = width * 3 + 7;
            let mut data = vec![-12345.25_f32; stride * height];
            for y in 0..height {
                for x in 0..width {
                    for c in 0..3 {
                        let raw = ((x * 191 + y * 131 + c * 977) % 8192) as i32 - 128;
                        data[y * stride + 3 * x + c] = (raw as f32) * 0.75_f32;
                    }
                }
            }
            // The archived no-spatial cases pass table 0 and table 2
            // while table 1 is absent: the all-three gate must select
            // the global gain alone.
            let refs: [Option<&GainTable>; 3] = [
                Some(&tables[0]),
                if spatial { Some(&tables[1]) } else { None },
                Some(&tables[2]),
            ];
            apply_gains(&mut data, width, height, stride, black, gain, refs);
            for y in 0..height {
                for pad in width * 3..stride {
                    assert_eq!(data[y * stride + pad], -12345.25_f32, "padding changed");
                }
            }
            assert_eq!(
                fnv1a(&data),
                expected_hash,
                "case {width}x{height} spatial={spatial}"
            );
            cases += 1;
            samples += width * height * 3;
        }
        assert_eq!(cases, 11);
        assert_eq!(samples, 7947);
    }

    #[test]
    fn missing_any_table_selects_global_gain_only() {
        let tables = synthetic_tables();
        let mut with_all = vec![100.5_f32; 3 * 2 * 2];
        let mut without_middle = with_all.clone();
        let black = [1.0_f64; 3];
        let gain = [2.0_f64; 3];
        apply_gains(
            &mut with_all,
            2,
            2,
            6,
            black,
            gain,
            [Some(&tables[0]), Some(&tables[1]), Some(&tables[2])],
        );
        apply_gains(
            &mut without_middle,
            2,
            2,
            6,
            black,
            gain,
            [Some(&tables[0]), None, Some(&tables[2])],
        );
        // The all-three gate makes the missing-middle case equal to the
        // pure global gain, different from the spatial result.
        let pure = vec![((100.5_f64 - 1.0) * 2.0) as f32; 12];
        assert_eq!(without_middle, pure);
        assert_ne!(with_all, pure);
    }

    /// Port of the instruction-derived radial reference
    /// (x3f-docs followup-float-front/radial_reference.py). The geometry
    /// uses the documented DP2 example: native HI center (2416, 1604)
    /// and half-extents (2400, 1600) with the shown rect inputs.
    #[test]
    fn radial_factors_match_instruction_reference() {
        // The docs' DP2 HI example: native center (2416, 1604) and
        // half-extents (2400, 1600). valid.top = 4, valid.bottom = 3204
        // (activeMinusKeep.top 36 - 32 margin; 3236 + 32 - keep-relative
        // already applied), valid.left = 16, valid.right = 4816.
        // minus_keep[1] = active.top - keep.top = 36, minus_keep[3] =
        // active.bottom - keep.top = 3236.
        let active = [0, 68, 0, 3204];
        let keep = [0, 32, 0, 32];
        let col_range = [0, 10, 4824, 0];
        let coefficients = [1e-4, -2e-4, 2e-4, -1e-4];
        let radial = RadialShading::new(active, keep, col_range, coefficients, "HI").unwrap();
        assert_eq!(radial.center_x as i32, 2416);
        assert_eq!(radial.center_y as i32, 1604);

        // Instruction-order reference (radial_reference.py): dy squared
        // first through fma, radius sqrt/divide, square, fma factors.
        let reference = |x: i32, y: i32, stretch: f64| {
            let dx = x as f64 - 2416.0;
            let dy = y as f64 - 1604.0;
            let dy_s = stretch * dy;
            let radius = dy_s.mul_add(dy_s, dx * dx).sqrt()
                / (stretch * 1600.0)
                    .mul_add(stretch * 1600.0, 2400.0 * 2400.0)
                    .sqrt();
            let t = radius * radius;
            let t2 = t * t;
            let [a, b, c, d] = coefficients;
            let bottom = a.mul_add(t2, t * b) + 1.0;
            let top = c.mul_add(t2, t * d) + 1.0;
            (bottom as f32, top as f32)
        };
        for (x, y) in [(2416, 1604), (0, 0), (4815, 3203), (100, 200), (2416, 0)] {
            let (bottom, top) = radial.factors(x, y);
            let (r_bottom, r_top) = reference(x, y, 1.0);
            assert_eq!(bottom, r_bottom, "C0 factor at ({x},{y})");
            assert_eq!(top, r_top, "C2 factor at ({x},{y})");
        }
    }

    #[test]
    fn radial_med_stretches_the_vertical_axis() {
        let active = [0, 68, 0, 3204];
        let keep = [0, 32, 0, 32];
        let col_range = [0, 10, 4824, 0];
        let coefficients = [1e-4, -2e-4, 2e-4, -1e-4];
        let hi = RadialShading::new(active, keep, col_range, coefficients, "HI").unwrap();
        let med = RadialShading::new(active, keep, col_range, coefficients, "MED").unwrap();
        // A pixel directly above the center: dx = 0, so the radius is
        // pure dy. MED doubles dy in the numerator AND the denominator
        // half-extent, but dy (1604) exceeds halfY (1600), so the MED
        // radius is larger and its C0 factor differs from HI.
        let (hi_bottom, _) = hi.factors(2416, 0);
        let (med_bottom, _) = med.factors(2416, 0);
        assert_ne!(med_bottom, hi_bottom, "MED must change vertical geometry");
        // MED uses the 16 margin (scale 0.5): valid.top = 36-16 = 20,
        // valid.bottom = 3172+16 = 3188, halfY = (3188-20)/2 = 1584,
        // center_y = 20+1584 = 1604 (same symmetric center, different
        // half-extent: the denominator stretches by 2*1584).
        assert_eq!(med.center_y as i32, 1604);
    }

    #[test]
    fn radial_applies_factors_in_place_and_leaves_c1() {
        let active = [0, 0, 100, 100];
        let keep = [0, 0, 0, 0];
        let col_range = [0, 0, 92, 0];
        let radial =
            RadialShading::new(active, keep, col_range, [1e-3, -1e-3, 2e-3, -2e-3], "HI").unwrap();
        let stride = 10 * 3 + 3;
        let mut data = vec![10.0_f32; stride * 10];
        radial
            .apply(&mut data, 10, 10, stride, Control::none())
            .unwrap();
        for y in 0..10 {
            for x in 0..10 {
                let (bottom, top) = radial.factors(x as i32, y as i32);
                let pixel = &data[y * stride + 3 * x..y * stride + 3 * x + 3];
                assert_eq!(pixel[0], 10.0_f32 * bottom, "C0 at ({x},{y})");
                assert_eq!(pixel[1], 10.0_f32, "C1 unchanged at ({x},{y})");
                assert_eq!(pixel[2], 10.0_f32 * top, "C2 at ({x},{y})");
            }
            for pad in 10 * 3..stride {
                assert_eq!(data[y * stride + pad], 10.0_f32, "padding changed");
            }
        }
    }

    #[test]
    fn radial_rejects_degenerate_geometry() {
        // A collapsed valid rect (zero half-extents on both axes) gives
        // a zero denominator. valid.top == valid.bottom needs
        // active.bottom - active.top == -64 for the HI margin.
        let zero = RadialShading::new([0, 100, 0, 36], [0; 4], [0, 0, 14, 0], [1.0; 4], "HI");
        assert!(zero.is_err());
        // Non-finite coefficients reject.
        let bad = [f64::NAN, 0.0, 0.0, 0.0];
        let result = RadialShading::new([0, 0, 100, 100], [0; 4], [0, 0, 92, 0], bad, "HI");
        assert!(result.is_err());
    }

    /// The listed-repair kernel rules, checked one by one against the
    /// documented native behavior (the bit-for-bit native comparison is
    /// the archived C++ oracle: 35,840 standalone cases plus 8
    /// sequential lists / 1,044 records).
    #[test]
    fn listed_repair_matches_native_rules() {
        const W: usize = 11;
        const H: usize = 13;
        const STRIDE: usize = W * 3 + 7;
        let repair = ListedRepair::new(0, 1);
        let base = |x: usize, y: usize, c: usize| -> f32 {
            (((x * 31 + y * 17 + c * 7) * 131) % 100_000) as f32 * 0.015_625
        };
        // All-neighbor, all-channel interior record: the repaired value
        // must equal the trimmed mean of the eight neighbors.
        let mut data = vec![-31415.25_f32; STRIDE * H];
        for y in 0..H {
            for x in 0..W {
                for c in 0..3 {
                    data[y * STRIDE + 3 * x + c] = base(x, y, c);
                }
            }
        }
        let record = BadPixelRecord {
            row: 6,
            column: 5,
            flags: 0xff | (3 << 8) | (7 << 12),
        };
        let mut expected = [0.0_f32; 3];
        for c in 0..3 {
            let mut values: Vec<f32> = Vec::new();
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    values.push(base((5 + dx) as usize, (6 + dy) as usize, c));
                }
            }
            values.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let trimmed = &values[1..values.len() - 1];
            expected[c] = trimmed.iter().sum::<f32>() / trimmed.len() as f32;
        }
        repair
            .apply(&mut data, W, H, STRIDE, &[record], &[], Control::none())
            .unwrap();
        let pixel = &data[6 * STRIDE + 5 * 3..6 * STRIDE + 5 * 3 + 3];
        for c in 0..3 {
            assert_eq!(pixel[c], expected[c], "trimmed mean channel {c}");
        }

        // Halo skips: x or y below 2 or at the outermost border.
        for (row, column) in [(1_u16, 5_u16), (6, 1), (11, 5), (6, 10), (12, 5), (6, 11)] {
            let mut data = vec![1.0_f32; STRIDE * H];
            let record = BadPixelRecord {
                row,
                column,
                flags: 0xff | (0 << 8) | (7 << 12),
            };
            repair
                .apply(&mut data, W, H, STRIDE, &[record], &[], Control::none())
                .unwrap();
            let index = row as usize * STRIDE + column as usize * 3;
            assert_eq!(
                &data[index..index + 3],
                &[1.0; 3],
                "halo skip at ({row},{column})"
            );
        }

        // Badness below the request minimum skips.
        let mut data = vec![1.0_f32; STRIDE * H];
        let strict = ListedRepair::new(2, 1);
        let record = BadPixelRecord {
            row: 6,
            column: 5,
            flags: 0xff | (1 << 8) | (7 << 12),
        };
        strict
            .apply(&mut data, W, H, STRIDE, &[record], &[], Control::none())
            .unwrap();
        assert_eq!(
            &data[6 * STRIDE + 15..6 * STRIDE + 18],
            &[1.0; 3],
            "badness skip"
        );

        // Override rules: within the 1..=254 window and numerically
        // below the mask.
        let record = BadPixelRecord {
            row: 6,
            column: 5,
            flags: 0xff | (0 << 8) | (7 << 12),
        };
        let mut data = vec![2.0_f32; STRIDE * H];
        data[6 * STRIDE + 5 * 3..6 * STRIDE + 5 * 3 + 3].fill(9.0);
        // Override 255 is outside the 1..=254 window: the full mask
        // applies and the pixel changes.
        repair
            .apply(&mut data, W, H, STRIDE, &[record], &[255], Control::none())
            .unwrap();
        assert_eq!(
            &data[6 * STRIDE + 15..6 * STRIDE + 18],
            &[2.0; 3],
            "out-of-window override ignored, full mask repaired a uniform neighborhood to 2.0"
        );
        // Override above the original mask leaves the mask unchanged.
        let mut data = vec![2.0_f32; STRIDE * H];
        data[6 * STRIDE + 5 * 3..6 * STRIDE + 5 * 3 + 3].fill(9.0);
        repair
            .apply(&mut data, W, H, STRIDE, &[record], &[0xff], Control::none())
            .unwrap();
        assert_eq!(
            &data[6 * STRIDE + 15..6 * STRIDE + 18],
            &[2.0; 3],
            "higher override ignored"
        );
        // Override 0 does not meet the native >= 1 guard, so the
        // original full mask applies (the archived oracle passes 0 as
        // one of its override probes).
        let mut data = vec![2.0_f32; STRIDE * H];
        data[6 * STRIDE + 5 * 3..6 * STRIDE + 5 * 3 + 3].fill(9.0);
        repair
            .apply(&mut data, W, H, STRIDE, &[record], &[0], Control::none())
            .unwrap();
        assert_eq!(
            &data[6 * STRIDE + 15..6 * STRIDE + 18],
            &[2.0; 3],
            "zero override falls back to the original mask"
        );

        // All-zero RGB neighbors exercise the extrema sentinel.
        let mut data = vec![0.0_f32; STRIDE * H];
        data[6 * STRIDE + 5 * 3..6 * STRIDE + 5 * 3 + 3].fill(9.0);
        repair
            .apply(&mut data, W, H, STRIDE, &[record], &[], Control::none())
            .unwrap();
        assert_eq!(
            &data[6 * STRIDE + 15..6 * STRIDE + 18],
            &[0.0; 3],
            "zero sentinel trimmed mean"
        );

        // A non-finite proposal keeps the original pixel.
        let mut data = vec![1.0_f32; STRIDE * H];
        data[5 * STRIDE + 4 * 3] = f32::INFINITY;
        repair
            .apply(&mut data, W, H, STRIDE, &[record], &[], Control::none())
            .unwrap();
        assert_eq!(
            &data[6 * STRIDE + 15..6 * STRIDE + 18],
            &[1.0; 3],
            "non-finite keeps original"
        );
    }

    #[test]
    fn listed_repair_minimum_uses_unsigned_subtraction() {
        // Native wraps instead of saturating: badness 1 with offset 3
        // wraps to u32::MAX - 1 (0xFFFFFFFE).
        let repair = ListedRepair::new(1, 3);
        assert_eq!(repair.minimum_badness(), 0xFFFF_FFFE);
        // Badness 0 selects minimum 0 regardless of the offset.
        assert_eq!(ListedRepair::new(0, 3).minimum_badness(), 0);
        assert_eq!(ListedRepair::new(4, 1).minimum_badness(), 3);
    }

    #[test]
    fn listed_repair_sequential_lists_observe_earlier_repairs() {
        const W: usize = 11;
        const H: usize = 13;
        const STRIDE: usize = W * 3;
        let repair = ListedRepair::new(0, 1);
        let mut data = vec![0.0_f32; STRIDE * H];
        for y in 0..H {
            for x in 0..W {
                for c in 0..3 {
                    data[y * STRIDE + 3 * x + c] = ((x + y + c) % 7) as f32 + 1.0;
                }
            }
        }
        // Two adjacent records: the second must observe the first's
        // output.
        let first = BadPixelRecord {
            row: 6,
            column: 5,
            flags: 0xff | 0 << 8 | 7 << 12,
        };
        let second = BadPixelRecord {
            row: 6,
            column: 6,
            flags: 0xff | 0 << 8 | 7 << 12,
        };
        let mut sequential = data.clone();
        repair
            .apply(
                &mut sequential,
                W,
                H,
                STRIDE,
                &[first, second],
                &[],
                Control::none(),
            )
            .unwrap();
        let mut isolated = data.clone();
        repair
            .apply(&mut isolated, W, H, STRIDE, &[first], &[], Control::none())
            .unwrap();
        let seq_pixel = &sequential[6 * STRIDE + 6 * 3..6 * STRIDE + 6 * 3 + 3];
        let iso_pixel = &isolated[6 * STRIDE + 6 * 3..6 * STRIDE + 6 * 3 + 3];
        assert_ne!(
            seq_pixel, iso_pixel,
            "second record must see the first repair"
        );
    }

    #[test]
    fn listed_repair_rejects_invalid_layout() {
        let repair = ListedRepair::new(0, 1);
        let mut data = vec![0.0_f32; 3 * 3 * 3];
        // Width below 4 cannot host a halo record.
        assert!(repair
            .apply(
                &mut data,
                3,
                3,
                9,
                &[BadPixelRecord {
                    row: 1,
                    column: 1,
                    flags: 0xff
                }],
                &[],
                Control::none()
            )
            .is_err());
    }
}
