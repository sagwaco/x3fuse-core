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

pub mod repair;

/// One channel of the spatial gain table. `values` is row-major with
/// `rows` rows and `columns` knots per row. The CAMF builders in
/// `spatial_gain.rs` produce this layout.
///
/// A hand-built table with public fields bypasses the `new`
/// shape checks; `apply_gains` indexes `values` unchecked, and
/// panics on a shape mismatch.
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

/// FNV-1a over the little-endian f32 bits of the raster (the
/// recorded hashes use `to_le_bytes` order; this platform is
/// little-endian). Matches the archived native-oracle hashes
/// (`followup-float-preprocess`).
pub(crate) fn fnv1a_f32(values: &[f32]) -> u64 {
    let mut hash = 0xcbf29ce484222325_u64;
    for value in values {
        for byte in value.to_bits().to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x100000001b3);
        }
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
///   : fma(halfY, halfY, halfX*halfX))`: the stretch literal is "MED"
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
    /// The inclusive ValidPixRect (0xa64c8..0xa6520), shared by the
    /// radial and highlight stages: translated activeMinusKeep bounds,
    /// HI 32 / non-HI 16 margin, DarkShieldColRange[1]+6 and [2]-8
    /// horizontal bounds. The rect must carry a two-pixel halo.
    pub fn valid_pix_rect(
        active: [i32; 4],
        keep: [i32; 4],
        col_range: [i32; 4],
        resolution: &str,
    ) -> [i32; 4] {
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
        [valid_left, valid_top, valid_right, valid_bottom]
    }

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
        let [valid_left, valid_top, valid_right, valid_bottom] =
            Self::valid_pix_rect(active, keep, col_range, resolution);
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
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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

    /// Construct with the final badness minimum directly. Native
    /// `ReplaceBadPixelsF20` takes the caller-computed minimum as its
    /// fourth argument: the listed path derives it from the selected
    /// label and BPBadnessOffset, while the despeckle caller passes
    /// the constant 3.
    fn from_minimum(minimum: u32) -> Self {
        Self {
            minimum_badness: minimum,
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

/// Stage 4 of the ordered float frontend: native highlight restoration,
/// the scalar reconstruction of `F20RestoreHighlights` (0x43000) and its
/// `RepairPix` worker (0x42b6c). Verified bit-for-bit against the native
/// export at the full wrapper boundary: 40 cases / 174,231 channel
/// samples including saturation/dense maps, row padding and scratch
/// guards (x3f-docs
/// `followup-native-highlight/check_highlight.cpp`).
///
/// The saturation map is packed RGB bytes, `sat[3*(y*width+x)+channel]`,
/// decoded from the camera RLE or Full entries. `dense` is one byte per
/// pixel. `valid` is the inclusive ValidPixRect (the same rect the radial
/// stage uses) and must carry a two-pixel lower/right halo.
#[derive(Clone, Copy, Debug)]
pub struct HighlightRestoration {
    threshold: f64,
}

/// Statistics of one restoration pass, native counters.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HighlightStats {
    pub donors: u32,
    pub occupied: u32,
    pub removed: u32,
    pub dense_pixels: u32,
    pub have_lut: bool,
}

/// One RLE saturation-map entry: alternating gap and run lengths. The
/// trailing gap word is not consumed. `RLE2Map` at 0xaf4c0 takes a
/// second size argument that the native ignores; this port drops it.
pub fn decode_sat_map_rle(words: &[u32], sat: &mut [u8], channel: usize) {
    // Native adds u32 positions without bounds; the caller validates the
    // runs against the map size before this port decodes them.
    let mut position = 0_u64;
    let mut index = 0;
    while index + 1 < words.len() {
        let start = position + u64::from(words[index]);
        let end = start + u64::from(words[index + 1]);
        for pixel in start..end {
            let pixel = pixel as usize;
            if pixel < sat.len() / 3 {
                sat[3 * pixel + channel] = 1;
            }
        }
        position = end;
        index += 2;
    }
}

/// One Full saturation-map entry: 32 bits per word, most significant
/// first, writing both zeros and ones. `Full2Map` at 0xaf5a4. The final
/// word is ignored.
pub fn decode_sat_map_full(words: &[u32], sat: &mut [u8], channel: usize) {
    for (index, &word) in words.iter().enumerate().take(words.len().saturating_sub(1)) {
        for bit in 0..32 {
            let pixel = 32 * index + bit;
            if pixel < sat.len() / 3 {
                sat[3 * pixel + channel] = ((word >> (31 - bit)) & 1) as u8;
            }
        }
    }
}

impl HighlightRestoration {
    /// Native threshold at 0xa67f8..0xa68b0:
    /// `e = max(ExpEstimate + ExpNet - 1.5, 0)`,
    /// `mix = e > 1 ? 0 : 1 - e`, `threshold = fma(mix, 3000, 4000)`.
    pub fn new(exp_estimate: f32, exp_net: f32) -> Self {
        let e = (f64::from(exp_estimate) + f64::from(exp_net) - 1.5).max(0.0);
        let mix = if e > 1.0 { 0.0 } else { 1.0 - e };
        Self {
            threshold: mix.mul_add(3000.0, 4000.0),
        }
    }

    /// Direct threshold construction. Only the archived native-oracle
    /// mode matrix supplies a raw threshold; production code derives
    /// it through [`HighlightRestoration::new`].
    #[cfg(test)]
    fn from_threshold(threshold: f64) -> Self {
        Self { threshold }
    }

    pub fn threshold(&self) -> f64 {
        self.threshold
    }

    /// ARM64 FCVTZS followed by the native 0..255 clamp, including the
    /// ratio 0/0 case: NaN maps to zero, negatives to zero, saturation
    /// to 255.
    fn bin(value: f32) -> usize {
        if value.is_nan() || value < 0.0 {
            0
        } else if value >= 255.0 {
            255
        } else {
            value as usize
        }
    }

    /// Run the full stage in place. `sat` is the packed RGB saturation
    /// map (cleaned in place); `dense` receives the dilation map.
    #[allow(clippy::too_many_lines)]
    pub fn restore(
        &self,
        data: &mut [f32],
        width: usize,
        height: usize,
        row_stride: usize,
        sat: &mut [u8],
        dense: &mut [u8],
        valid: [i32; 4],
        control: Control<'_>,
    ) -> Result<HighlightStats> {
        control.check()?;
        if width < 4 || height < 4 {
            return Err(Error::InvalidData("invalid Merrill highlight image size"));
        }
        if sat.len() != width * height * 3 || dense.len() != width * height {
            return Err(Error::InvalidData("invalid Merrill highlight map size"));
        }
        if row_stride < width * 3 || data.len() < row_stride * (height - 1) + width * 3 {
            return Err(Error::InvalidData("invalid Merrill highlight image layout"));
        }
        let (left, top, right, bottom) = (valid[0], valid[1], valid[2], valid[3]);
        if left < 0 || top < 0 || right < left || bottom < top {
            return Err(Error::InvalidData("invalid Merrill highlight rectangle"));
        }
        // RepairPix reads a two-pixel lower/right halo in the dense map
        // with no upper/right clamp of its own. The halo must be
        // allocated: `right+2 <= width-1` and `bottom+2 <= height-1`.
        // The corner shows why the right bound is strict: with
        // `right+2 == width` the corner read is the flat index
        // `(bottom+2)*width + width`, one past the dense buffer when
        // `bottom+2 == height-1`.
        if (right + 2) as usize >= width || (bottom + 2) as usize >= height {
            return Err(Error::InvalidData("invalid Merrill highlight halo"));
        }
        let mut stats = HighlightStats::default();

        // 1. Saturation cleanup (0x43064..0x431f0): scan interior
        // rows/columns, keep a mark only if a same-channel eight-neighbor
        // has one; isolated marks are removed. Boundary flags stay.
        for y in 1..height - 1 {
            for x in 1..width - 1 {
                for c in 0..3 {
                    let index = 3 * (y * width + x) + c;
                    if sat[index] == 0 {
                        continue;
                    }
                    let mut any = false;
                    for dy in -1i32..=1 {
                        for dx in -1i32..=1 {
                            if dx == 0 && dy == 0 {
                                continue;
                            }
                            let neighbor = 3
                                * ((y as i32 + dy) as usize * width + (x as i32 + dx) as usize)
                                + c;
                            any |= sat[neighbor] != 0;
                        }
                    }
                    if !any {
                        stats.removed += 1;
                    }
                    sat[index] = u8::from(any);
                }
            }
        }

        // 2. Dense map (0x43250..0x433bc): zero, then dilate 5x5 around
        // interior centers with at least two marked channels.
        dense.fill(0);
        for y in 2..height - 2 {
            for x in 2..width - 2 {
                let base = 3 * (y * width + x);
                if u32::from(sat[base]) + u32::from(sat[base + 1]) + u32::from(sat[base + 2]) >= 2 {
                    for dy in -2i32..=2 {
                        for dx in -2i32..=2 {
                            let index =
                                ((y as i32 + dy) as usize) * width + (x as i32 + dx) as usize;
                            dense[index] = 1;
                        }
                    }
                }
            }
        }
        stats.dense_pixels = dense.iter().filter(|&&v| v == 1).count() as u32;

        // 3. Donor LUT (0x435e0..0x436b0) over the inclusive valid rect.
        // A donor is unsaturated in C2 with float(C0+C1) strictly above
        // 4000; accumulation is f32 in raster order.
        let mut lut = [0.0_f32; 256];
        let mut count = [0_u32; 256];
        for y in top as usize..=bottom as usize {
            let row = &data[y * row_stride..y * row_stride + width * 3];
            for x in left as usize..=right as usize {
                let pixel = &row[3 * x..3 * x + 3];
                if sat[3 * (y * width + x) + 2] == 0 && pixel[0] + pixel[1] > 4000.0 {
                    let a = pixel[0].max(1.0);
                    let b = pixel[1].max(1.0);
                    let c = pixel[2].max(1.0);
                    let index = Self::bin((a * 255.0) / (a + b));
                    lut[index] += c / b;
                    count[index] += 1;
                    stats.donors += 1;
                }
            }
        }
        let mut first = 256;
        for index in 0..256 {
            if lut[index] > 0.0 {
                lut[index] /= count[index] as f32;
                stats.occupied += 1;
                if first == 256 {
                    first = index;
                }
            }
        }
        stats.have_lut = first < 256;
        if stats.have_lut {
            // Forward fill: leading holes take the first occupied value,
            // later zero bins the previous value. No interpolation.
            let mut previous = lut[first];
            for value in &mut lut {
                if *value == 0.0 {
                    *value = previous;
                }
                previous = *value;
            }
        }

        // 4. RepairPix (0x42b6c) over the inclusive valid rect. The
        // native runs eight workers with disjoint rows; the scalar port
        // is row-serial, which changes no output (each pixel reads only
        // its own value and the completed maps).
        for y in top..=bottom {
            control.check()?;
            let y_index = y as usize;
            let row = &mut data[y_index * row_stride..y_index * row_stride + width * 3];
            for x in left..=right {
                let x_index = x as usize;
                let pixel = &mut row[3 * x_index..3 * x_index + 3];
                let (a, b, original_c) = (pixel[0], pixel[1], pixel[2]);
                let mut c = original_c;
                // Hole filling: marked C2 with an existing LUT. This uses
                // the original C0/C1 without the donor lower clamp; a
                // zero denominator maps NaN to bin 0 and saturates to 255.
                if stats.have_lut && sat[3 * (y_index * width + x_index) + 2] != 0 {
                    let index = Self::bin((a * 255.0) / (a + b));
                    c = (b * lut[index]).min(32767.0);
                }
                // Neutralization over the sorted current channels
                // (total_cmp: the pub function must not panic on a
                // non-finite sample).
                let mut sorted = [a, b, c];
                sorted.sort_by(|p, q| p.total_cmp(q).reverse());
                let hi = f64::from(sorted[0]);
                let mid = f64::from(sorted[1]);
                let lo = f64::from(sorted[2]);
                let mut t = hi.mul_add(2.0 / 3.0, mid * (1.0 / 3.0));
                t = lo.mul_add(0.0, t);
                t = ((t - self.threshold) / 1000.0).clamp(0.0, 1.0);
                let mut density = 0.0_f64;
                if x_index >= 2 && y_index >= 2 {
                    for dy in -2i32..=2 {
                        for dx in -2i32..=2 {
                            let index = ((y_index as i32 + dy) as usize) * width
                                + (x_index as i32 + dx) as usize;
                            density += f64::from(dense[index]);
                        }
                    }
                    density *= 0.04;
                }
                let weight = (t.mul_add(t, density * density) * 0.5).sqrt();
                let anchor = weight * f64::from(a);
                let keep = 1.0 - weight;
                pixel[0] = keep.mul_add(f64::from(a), anchor) as f32;
                pixel[1] = keep.mul_add(f64::from(b), anchor) as f32;
                pixel[2] = keep.mul_add(f64::from(c), anchor) as f32;
            }
        }
        Ok(stats)
    }
}

/// Stage 5 of the ordered float frontend: raster despeckle, the scalar
/// reconstruction of the native `DespeckleRas` wrapper (0x4a62c) with
/// the `DetectSpecklesRas` detector (0x49ac8), the `DoThreadedPix`
/// eight-way row split (0x62f48) and the mode-3 `ReplaceBadPixelsF20`
/// correction (native fourth argument 3 at 0x4aa30). Verified against
/// the archived native oracle (x3f-docs
/// `followup-despeckle/check_native.cpp`): 7 detector cases / 66,876
/// records exactly equal in count, order and flags, every LUT index
/// 0..4095 in all three channels straddled by nextafter probes on both
/// threshold boundaries, 2,597 mode-3 correction cases and 4 full
/// wrapper buffers byte-identical including row padding.
///
/// `gain` is the ApplyGains request gain triple (spatial factors
/// already changed the pixels), `adjust` is the typed CAMF `DespAdjust`
/// vector (default 15,15,5) scaled by the request strength, and `rect`
/// is ActiveImageArea translated by KeepImageArea's left/top, not the
/// ValidPixRect. The detector reads only the rect's left/top words.
#[derive(Clone, Debug)]
pub struct Despeckle {
    lut: [[f32; 4096]; 3],
    rect: [u32; 4],
}

impl Despeckle {
    /// Native LUT construction (0x49c6c): 4096 float thresholds per
    /// channel with explicit double fma and the exact native double
    /// constants (channel-B terms are exact doubles of rounded floats).
    /// The thresholds are float32 sample-count differences.
    fn make_lut(gain: [f64; 3], adjust: [f64; 3]) -> Result<[[f32; 4096]; 3]> {
        const K: [u64; 3] = [
            0x3d08_c511_7a7e_165e,
            0x3d00_e374_a4f8_e0b4,
            0x3cec_d2b2_97d8_89bc,
        ];
        const B: [u64; 3] = [
            0x3ea5_798e_e000_0000,
            0x3eab_2dd8_e000_0000,
            0x3e65_798e_e000_0000,
        ];
        const C1: u64 = 0x3f36_6666_6666_6666;
        const C2: u64 = 0x3c04_8ad3_caef_e280;
        const C3: u64 = 0x3c07_9ca1_0c92_4223;
        const C4: u64 = 0x3fe8_3886_594a_f4f1;
        let mut lut = [[0.0_f32; 4096]; 3];
        for channel in 0..3 {
            for index in 0..4096 {
                let a = f64::from((f64::from_bits(K[channel]) / gain[channel]) as f32);
                let mut t = ((f64::from(index as f32) * f64::from_bits(C1)) * a
                    / f64::from_bits(C2)) as f32;
                t = t.sqrt();
                t = ((f64::from(t) * f64::from_bits(C3)) / a) as f32;
                let inner = f64::from(t).mul_add(
                    f64::from(t) * f64::from_bits(C4),
                    f64::from_bits(B[channel]),
                );
                let mut u = inner.sqrt() as f32;
                u *= adjust[channel] as f32;
                u *= 4096.0_f32;
                lut[channel][index] = (f64::from(u) / 1.4) as f32;
            }
        }
        if lut
            .iter()
            .any(|channel| channel.iter().any(|v| !v.is_finite()))
        {
            return Err(Error::InvalidData("invalid Merrill despeckle thresholds"));
        }
        Ok(lut)
    }

    /// Build the stage from the request gains, the scaled adjust vector
    /// and the translated active rect `[left, top, right, bottom]`.
    pub fn new(gain: [f64; 3], adjust: [f64; 3], rect: [u32; 4]) -> Result<Self> {
        // The verified domain requires positive finite gains and
        // non-negative finite adjust values; other inputs reach native
        // non-finite thresholds.
        if gain.iter().any(|v| !v.is_finite() || *v <= 0.0)
            || adjust.iter().any(|v| !v.is_finite() || *v < 0.0)
        {
            return Err(Error::InvalidData("invalid Merrill despeckle calibration"));
        }
        // The column scan bounds need left >= 5 so the one-pixel
        // predicate halo stays inside the image; the row bounds depend
        // on the worker split and are validated per scan against the
        // effective range.
        if rect[0] < 5 {
            return Err(Error::InvalidData("invalid Merrill despeckle rectangle"));
        }
        let lut = Self::make_lut(gain, adjust)?;
        Ok(Self { lut, rect })
    }

    /// One LUT threshold, for tests and traces.
    pub fn lut(&self, channel: usize, index: usize) -> f32 {
        self.lut[channel][index]
    }

    /// The translated active rect `[left, top, right, bottom]`.
    pub fn rect(&self) -> [u32; 4] {
        self.rect
    }

    /// The per-pixel channel mask (C0 bit 4, C1 bit 2, C2 bit 1) of
    /// the native predicate: a strict horizontal maximum or minimum
    /// that clears the cross and diagonal neighbor thresholds. The
    /// maximum branch clamps the compared sample and the LUT index to
    /// 4095; the minimum branch compares the original center with an
    /// index truncated from the center clamped to 0..4095. Equality
    /// at a threshold fails the strict comparison.
    fn predicate(&self, data: &[f32], row_stride: usize, x: usize, y: usize) -> u32 {
        let at = |dx: i32, dy: i32, channel: usize| -> f32 {
            data[(y as i32 + dy) as usize * row_stride + (x as i32 + dx) as usize * 3 + channel]
        };
        let mut mask = 0_u32;
        for channel in 0..3 {
            let center = at(0, 0, channel);
            let left = at(-1, 0, channel);
            let right = at(1, 0, channel);
            if center > left && center > right {
                // Native clamps the compared sample as well as the LUT
                // index.
                let center = center.min(4095.0);
                let threshold = self.lut[channel][(center.max(0.0)) as usize];
                let cross = left
                    .max(right)
                    .max(at(0, -1, channel))
                    .max(at(0, 1, channel));
                let diagonal = at(-1, -1, channel)
                    .max(at(1, -1, channel))
                    .max(at(-1, 1, channel))
                    .max(at(1, 1, channel));
                if center > cross + threshold && center > diagonal + threshold {
                    mask |= 4 >> channel;
                }
            } else if center < left && center < right {
                let threshold = self.lut[channel][(center.clamp(0.0, 4095.0)) as usize];
                let cross = left
                    .min(right)
                    .min(at(0, -1, channel))
                    .min(at(0, 1, channel));
                let diagonal = at(-1, -1, channel)
                    .min(at(1, -1, channel))
                    .min(at(-1, 1, channel))
                    .min(at(1, 1, channel));
                if center < cross - threshold && center < diagonal - threshold {
                    mask |= 4 >> channel;
                }
            }
        }
        mask
    }

    /// One worker's detection scan over its inclusive row range. The
    /// native bounds (0x49ac8) read only the rect's first two words:
    /// rows intersect the worker range through
    /// `max(top - 4, start) ..= min(height - (top - 4), end)` and
    /// columns run `left - 4 ..= width - left + 2`; the detector
    /// stops at 624,999 marks (its 625,000 six-byte segment slots
    /// minus the four-byte count).
    fn detect_worker(
        &self,
        data: &[f32],
        width: usize,
        height: usize,
        row_stride: usize,
        start: usize,
        end: usize,
        out: &mut Vec<BadPixelRecord>,
        control: Control<'_>,
    ) -> Result<()> {
        control.check()?;
        let [left, top, _, _] = self.rect;
        let col_begin = i64::from(left) - 4;
        let col_end = width as i64 - i64::from(left) + 2;
        let row_begin = (i64::from(top) - 4).max(start as i64);
        let row_end = (height as i64 - (i64::from(top) - 4)).min(end as i64);
        // The column invariant (`left >= 5`) is owned by `new`. The
        // row range depends on the worker split, so it is validated
        // here: the predicate reads a one-pixel halo, and scans that
        // would leave the image are outside the verified contract (the
        // native reads without a clamp).
        if row_begin > row_end {
            // A worker outside the scan range detects nothing.
            return Ok(());
        }
        if row_begin < 1 || row_end > height as i64 - 2 {
            return Err(Error::InvalidData("invalid Merrill despeckle geometry"));
        }
        let mut marks = 0_usize;
        for y in row_begin..=row_end {
            control.check()?;
            let y = y as usize;
            for x in col_begin..=col_end {
                let x = x as usize;
                let mask = self.predicate(data, row_stride, x, y);
                if mask == 0 {
                    continue;
                }
                out.push(BadPixelRecord {
                    row: y as u16,
                    column: x as u16,
                    flags: (0x3ff | (mask << 12)) as u16,
                });
                marks += 1;
                if marks >= 624_999 {
                    return Ok(());
                }
            }
        }
        Ok(())
    }

    /// Detect over the native `DoThreadedPix` partition: eight
    /// nonoverlapping row ranges of `floor(height / 8)`, the remainder
    /// in the final worker, joined in worker order (raster order).
    pub fn detect(
        &self,
        data: &[f32],
        width: usize,
        height: usize,
        row_stride: usize,
        control: Control<'_>,
    ) -> Result<Vec<BadPixelRecord>> {
        control.check()?;
        if height < 8 {
            // floor(height / 8) cannot split fewer than eight rows.
            return Err(Error::InvalidData("invalid Merrill despeckle size"));
        }
        if width > u16::MAX as usize || height > u16::MAX as usize {
            return Err(Error::InvalidData("invalid Merrill despeckle size"));
        }
        if row_stride < width * 3 || data.len() < row_stride * (height - 1) + width * 3 {
            return Err(Error::InvalidData("invalid Merrill despeckle layout"));
        }
        let step = height / 8;
        let mut records = Vec::new();
        let mut start = 0;
        for worker in 0..8 {
            let end = if worker == 7 {
                height - 1
            } else {
                start + step - 1
            };
            self.detect_worker(
                data,
                width,
                height,
                row_stride,
                start,
                end,
                &mut records,
                control,
            )?;
            start += step;
        }
        Ok(records)
    }

    /// Run the full stage in place: detect all eight worker lists, join
    /// them in worker order, then correct sequentially with the mode-3
    /// `ReplaceBadPixelsF20` (minimum badness 3, all eight neighbors
    /// enabled by the 0x3ff record flags, no overrides). Detection
    /// finishes before correction starts; correction never re-detects.
    /// Returns the joined record count.
    pub fn apply(
        &self,
        data: &mut [f32],
        width: usize,
        height: usize,
        row_stride: usize,
        control: Control<'_>,
    ) -> Result<usize> {
        let records = self.detect(data, width, height, row_stride, control)?;
        let repair = ListedRepair::from_minimum(3);
        repair.apply(data, width, height, row_stride, &records, &[], control)?;
        Ok(records.len())
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
        let mut data = raw_float(decoded)?;
        self.apply_in_place(&mut data, width, height, row_stride, control)?;
        Ok(data)
    }

    /// Apply stage 1 to a raw float raster from `raw_float`. The detector
    /// stage reads this raster before the gains.
    pub fn apply_in_place(
        &self,
        data: &mut [f32],
        width: usize,
        height: usize,
        row_stride: usize,
        control: Control<'_>,
    ) -> Result<()> {
        control.check()?;
        if height == 0 || width == 0 {
            return Err(Error::InvalidData("empty Merrill float image"));
        }
        if self.spatial.is_some() && height < 2 {
            return Err(Error::InvalidData("invalid Merrill float image size"));
        }
        if row_stride < width * 3 || data.len() < row_stride * (height - 1) + width * 3 {
            return Err(Error::InvalidData("invalid Merrill float image layout"));
        }
        let refs: [Option<&GainTable>; 3] = match &self.spatial {
            None => [None, None, None],
            Some(t) => [Some(&t[0]), Some(&t[1]), Some(&t[2])],
        };
        apply_gains(data, width, height, row_stride, self.black, self.gain, refs);
        Ok(())
    }
}

/// Convert decoded sensor counts to the owned native float raster.
/// u16 to f32 is exact: every u16 value is representable in f32. Native
/// sign-extends the 16-bit sample before the float conversion; the result
/// is identical for valid 0..4095 sensor data (recorded in the stage 1 notes).
pub fn raw_float(decoded: &[u16]) -> Result<Vec<f32>> {
    let mut data = Vec::new();
    data.try_reserve_exact(decoded.len())
        .map_err(|_| Error::Allocation)?;
    data.extend(decoded.iter().map(|&v| f32::from(v)));
    Ok(data)
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
        // (activeMinusKeep.top 36 - 32 margin; 3172 + 32 - keep-relative
        // already applied), valid.left = 16, valid.right = 4816.
        // minus_keep[1] = active.top - keep.top = 36, minus_keep[3] =
        // active.bottom - keep.top = 3172.
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

    #[test]
    fn sat_map_rle_decodes_runs_and_ignores_the_trailing_gap() {
        let mut sat = vec![0_u8; 8 * 2 * 3];
        // Alternating gap/run words; the trailing gap word is not consumed.
        super::decode_sat_map_rle(&[2, 3, 1, 2, 5], &mut sat, 1);
        let mut marked = vec![false; 16];
        for pixel in 0..16 {
            marked[pixel] = sat[3 * pixel + 1] != 0;
        }
        for pixel in [0_usize, 1, 5] {
            assert!(!marked[pixel], "pixel {pixel} must stay clear");
        }
        for pixel in [2_usize, 3, 4, 6, 7] {
            assert!(marked[pixel], "pixel {pixel} must be marked");
        }
        // Only the selected channel is written.
        assert!(sat
            .iter()
            .enumerate()
            .all(|(i, &v)| { v == 0 || i % 3 == 1 }));
    }

    #[test]
    fn sat_map_full_decodes_words_most_significant_first() {
        let mut sat = vec![0_u8; 64 * 3];
        // 0xF0 repeats the pattern 11110000; the final word is ignored.
        super::decode_sat_map_full(&[0xf0f0_f0f0, u32::MAX], &mut sat, 0);
        for (pixel, expected) in [(0, 1), (4, 0), (7, 0), (8, 1), (31, 0), (32, 0), (63, 0)] {
            assert_eq!(u32::from(sat[3 * pixel]), expected, "pixel {pixel}");
        }
    }

    #[test]
    fn highlight_restore_is_identity_below_threshold_without_marks() {
        let restore = super::HighlightRestoration::new(0.0, 0.0);
        // Native threshold: e = max(0+0-1.5,0) = 0, mix = 1,
        // fma(1,3000,4000) = 7000.
        assert_eq!(restore.threshold(), 7000.0);
        let width = 8;
        let height = 8;
        let stride = width * 3;
        let mut data = vec![0.0_f32; stride * height];
        for row in 0..height {
            for column in 0..width {
                let pixel = &mut data[row * stride + 3 * column..row * stride + 3 * column + 3];
                pixel.copy_from_slice(&[100.0 + row as f32, 200.0, 300.0]);
            }
        }
        let original = data.clone();
        let mut sat = vec![0_u8; width * height * 3];
        let mut dense = vec![0_u8; width * height];
        let stats = restore
            .restore(
                &mut data,
                width,
                height,
                stride,
                &mut sat,
                &mut dense,
                [1, 1, 5, 5],
                Control::none(),
            )
            .unwrap();
        // No donors (a+b below 4000), no marks: the LUT stays empty and
        // every neutralization weight is zero, so the raster is the
        // identity.
        assert_eq!(stats.donors, 0);
        assert_eq!(stats.removed, 0);
        assert_eq!(stats.occupied, 0);
        assert_eq!(stats.dense_pixels, 0);
        assert!(!stats.have_lut);
        assert_eq!(data, original);
    }

    #[test]
    fn highlight_restore_fills_marked_c2_from_donors_and_neutralizes() {
        let restore = super::HighlightRestoration::new(0.0, 0.0);
        let width = 8;
        let height = 8;
        let stride = width * 3;
        let mut data = vec![0.0_f32; stride * height];
        // Donor: unsaturated with a+b = 8000 > 4000, so
        // bin = trunc(6000*255/8000) = 191 and lut[191] = c/b = 0.5.
        let donor = [6000.0_f32, 2000.0, 1000.0];
        // Two adjacent marked pixels survive the cleanup pass (each keeps
        // the other); a lone mark would be removed as isolated.
        let marked = [6000.0_f32, 2000.0, 5000.0];
        for (row, column, pixel) in [(1, 1, marked), (1, 2, marked), (2, 2, donor)] {
            let base = row * stride + 3 * column;
            data[base..base + 3].copy_from_slice(&pixel);
        }
        let mut sat = vec![0_u8; width * height * 3];
        for (row, column) in [(1, 1), (1, 2)] {
            sat[3 * (row * width + column) + 2] = 1;
        }
        let mut dense = vec![0_u8; width * height];
        let stats = restore
            .restore(
                &mut data,
                width,
                height,
                stride,
                &mut sat,
                &mut dense,
                [1, 1, 5, 5],
                Control::none(),
            )
            .unwrap();
        assert_eq!(stats.donors, 1);
        assert_eq!(stats.removed, 0);
        assert_eq!(stats.occupied, 1);
        // Only channel 2 is marked, so the 2-of-3 dilation stays empty.
        assert_eq!(stats.dense_pixels, 0);
        assert!(stats.have_lut);
        // Hole filling replaces C2 with min(b*lut[191], 32767) = 1000;
        // the sorted blend stays below the 7000 threshold, so the
        // neutralization weight is zero and the write is the identity.
        for (row, column) in [(1, 1), (1, 2)] {
            let base = row * stride + 3 * column;
            assert_eq!(
                data[base..base + 3],
                [6000.0, 2000.0, 1000.0],
                "marked pixel ({row},{column})"
            );
        }
        let donor_base = 2 * stride + 3 * 2;
        assert_eq!(data[donor_base..donor_base + 3], donor);
        // The cleanup pass kept both marks in place.
        assert_eq!(sat[3 * (1 * width + 1) + 2], 1);
        assert_eq!(sat[3 * (1 * width + 2) + 2], 1);
    }

    #[test]
    fn highlight_restore_rejects_invalid_layout() {
        let restore = super::HighlightRestoration::new(0.0, 0.0);
        let mut data = vec![0.0_f32; 8 * 8 * 3];
        // Map sizes must match the packed full image.
        let mut short_sat = vec![0_u8; 7 * 8 * 3];
        let mut dense = vec![0_u8; 8 * 8];
        assert!(restore
            .restore(
                &mut data,
                8,
                8,
                24,
                &mut short_sat,
                &mut dense,
                [1, 1, 5, 5],
                Control::none()
            )
            .is_err());
        // RepairPix reads a two-pixel lower/right halo in the dense map;
        // a rect touching the last row/column rejects.
        let mut sat = vec![0_u8; 8 * 8 * 3];
        assert!(restore
            .restore(
                &mut data,
                8,
                8,
                24,
                &mut sat,
                &mut dense,
                [1, 1, 6, 5],
                Control::none()
            )
            .is_err());
    }

    /// Minimal MT19937 (std::mt19937) so the port reproduces the
    /// archived native-oracle mode matrix inputs exactly.
    struct Mt19937 {
        state: [u32; 624],
        index: usize,
    }

    impl Mt19937 {
        fn new(seed: u32) -> Self {
            let mut state = [0_u32; 624];
            state[0] = seed;
            for i in 1..624 {
                let previous = state[i - 1];
                state[i] = 1812433253_u32
                    .wrapping_mul(previous ^ (previous >> 30))
                    .wrapping_add(i as u32);
            }
            Self { state, index: 624 }
        }

        fn next(&mut self) -> u32 {
            if self.index == 624 {
                for i in 0..624 {
                    let y =
                        (self.state[i] & 0x8000_0000) | (self.state[(i + 1) % 624] & 0x7fff_ffff);
                    self.state[i] = self.state[(i + 397) % 624]
                        ^ (y >> 1)
                        ^ if y & 1 != 0 { 0x9908_b0df } else { 0 };
                }
                self.index = 0;
            }
            let mut y = self.state[self.index];
            self.index += 1;
            y ^= y >> 11;
            y ^= (y << 7) & 0x9d2c_5680;
            y ^= (y << 15) & 0xefc6_0000;
            y ^ (y >> 18)
        }
    }

    /// The archived native-oracle mode matrix
    /// (followup-native-highlight/check_highlight.cpp and
    /// native-results.txt): 40 wrapper cases over eight flag/value
    /// regimes and a threshold sweep, seed 0x43000. The archived run
    /// compared every case bit-for-bit with the native
    /// F20RestoreHighlights wrapper; these are its per-mode stats, so
    /// this test pins the Rust kernel to that native-verified run
    /// without the library. Rows: (donors, occupied, removed, dense).
    #[test]
    fn highlight_restore_matches_archived_native_mode_matrix() {
        const EXPECTED: [(u32, u32, u32, u32); 40] = [
            (0, 0, 0, 0),
            (0, 0, 0, 0),
            (500, 1, 0, 0),
            (0, 0, 0, 0),
            (764, 227, 41, 1408),
            (1105, 245, 72, 0),
            (1043, 240, 0, 468),
            (485, 208, 4, 1456),
            (0, 0, 0, 0),
            (0, 0, 0, 0),
            (748, 1, 0, 0),
            (0, 0, 0, 0),
            (748, 235, 45, 1279),
            (1090, 242, 72, 0),
            (729, 224, 0, 368),
            (469, 203, 8, 1357),
            (0, 0, 0, 0),
            (0, 0, 0, 0),
            (736, 1, 0, 0),
            (0, 0, 0, 0),
            (898, 244, 62, 1677),
            (1305, 246, 75, 0),
            (712, 225, 0, 345),
            (381, 183, 5, 1256),
            (0, 0, 0, 0),
            (0, 0, 0, 0),
            (594, 1, 0, 0),
            (0, 0, 0, 0),
            (758, 236, 49, 1324),
            (1108, 244, 60, 0),
            (1017, 237, 0, 456),
            (562, 215, 8, 1643),
            (0, 0, 0, 0),
            (0, 0, 0, 0),
            (575, 1, 0, 0),
            (0, 0, 0, 0),
            (738, 224, 53, 1219),
            (1061, 243, 72, 0),
            (836, 242, 0, 408),
            (518, 209, 7, 1539),
        ];
        let mut random = Mt19937::new(0x43000);
        let (mut donors, mut removed) = (0_u32, 0_u32);
        let mut empty = 0_u32;
        let mut samples = 0_usize;
        for mode in 0..40_usize {
            let width = 43 + mode % 7;
            let height = 27 + mode % 11;
            let stride = width * 3 + 9;
            let mut data = vec![-12345.25_f32; stride * height];
            let mut sat = vec![0_u8; width * height * 3];
            let valid = [
                (mode % 3) as i32,
                (mode % 3) as i32,
                width as i32 - 3,
                height as i32 - 3,
            ];
            let threshold = 4000.0 + 750.0 * (mode % 5) as f64;
            for y in 0..height {
                for x in 0..width {
                    for c in 0..3 {
                        let mut v = ((random.next() % 100000) as i32 - 5000) as f32 * 0.125;
                        // No donors (a+b below 4000), neutralization runs.
                        if mode % 8 == 0 {
                            v = if c == 2 { 9000.0 } else { 100.0 };
                        }
                        // Exact 4000 stays excluded (strictly above).
                        if mode % 8 == 1 {
                            v = [3000.0, 1000.0, 2000.0][c];
                        }
                        // One-bin filling and the 32767 cap.
                        if mode % 8 == 2 {
                            v = [3000.0, 1001.0, 200000.0][c];
                        }
                        // Division 0/0 and infinities with marked C2.
                        if mode % 8 == 3 && x % 3 == 0 {
                            v = [100.0, -100.0, 1500.0][c];
                        }
                        // Ratio 0/0 (NaN bin) while a donor LUT exists.
                        if mode % 8 == 7 && x % 7 == 0 {
                            v = if c == 2 { 1500.0 } else { 0.0 };
                        }
                        data[y * stride + 3 * x + c] = v;
                        let flag = match mode % 8 {
                            0 | 1 => 0,
                            2 => u8::from(c == 2 && x >= width / 2),
                            3 => u8::from(c == 2),
                            4 => u8::from(random.next() % 3 == 0),
                            // Isolated flags must clear in the cleanup.
                            5 => u8::from(x % 8 == 3 && y % 8 == 3),
                            6 => u8::from(c != 1 && x > width / 2 && y > height / 2),
                            // Short-circuit: no extra draw when marked;
                            // the flag is the raw %2 value, not a
                            // comparison.
                            _ => {
                                if c == 2 && x % 7 == 0 {
                                    1
                                } else {
                                    (random.next() % 2) as u8
                                }
                            }
                        };
                        sat[3 * (y * width + x) + c] = flag;
                    }
                }
            }
            let restore = super::HighlightRestoration::from_threshold(threshold);
            let mut dense = vec![0_u8; width * height];
            let mut working = data.clone();
            let stats = restore
                .restore(
                    &mut working,
                    width,
                    height,
                    stride,
                    &mut sat,
                    &mut dense,
                    valid,
                    Control::none(),
                )
                .unwrap();
            assert_eq!(
                (
                    stats.donors,
                    stats.occupied,
                    stats.removed,
                    stats.dense_pixels
                ),
                EXPECTED[mode],
                "mode {mode} stats"
            );
            donors += stats.donors;
            removed += stats.removed;
            empty += u32::from(stats.donors == 0);
            samples += width * height * 3;
        }
        // Archived wrapper totals: 40 cases / 174,231 channel samples,
        // 15 empty-donor cases, 633 isolated marks removed, 19,480 donors.
        assert_eq!(samples, 174_231);
        assert_eq!(empty, 15);
        assert_eq!(removed, 633);
        assert_eq!(donors, 19_480);
    }

    /// The archived native despeckle oracle
    /// (followup-despeckle/check_native.cpp, seed 149902) in one
    /// continuous run: 7 detector modes (random regimes plus the LUT
    /// endpoint grid that straddles every index on both threshold
    /// boundaries), the mode-3 replacement draws that keep the stream
    /// aligned, 4 wrapper modes (independent channel gains, spatial
    /// variation, varying rects) and 8 controlled wheel probes. The
    /// archived run compared every case bit-for-bit with the native
    /// DetectSpecklesRas / DespeckleRas / mode-3 ReplaceBadPixelsF20;
    /// these are its per-mode record counts and LUT endpoints.
    #[test]
    fn despeckle_matches_archived_native_oracle() {
        const DETECTOR_RECORDS: [usize; 7] = [19_226, 19_299, 20_169, 4_086, 0, 0, 4_096];
        // C++ default ostream output rounds to 6 significant digits.
        let close6 = |actual: f32, printed: f64, what: &str| {
            let tolerance = printed.abs() * 5.1e-6 + 1.0e-9;
            assert!(
                (f64::from(actual) - printed).abs() <= tolerance,
                "{what}: {actual} vs {printed}"
            );
        };
        let mut random = Mt19937::new(149_902);
        // Detector modes 0..6: w=263, h=265, stride = w*3+7.
        let width = 263_usize;
        let height = 265_usize;
        let stride = width * 3 + 7;
        let rect = [5_u32, 5, 258, 260];
        for mode in 0..7_usize {
            let mut gain = [0.731_f64, 1.831, 3.173];
            let mut adjust = [10.0_f64, 10.0, 5.0];
            if mode == 1 {
                gain = [3.731, 0.331, 1.173];
            }
            if mode == 2 {
                adjust = [0.001, 1.331, 3.117];
            }
            if mode >= 3 {
                adjust = [1.0, 1.0, 1.0];
            }
            let despeckle = super::Despeckle::new(gain, adjust, rect).unwrap();
            let mut data = vec![123.0_f32; stride * height];
            if mode < 3 {
                for y in 0..height {
                    for x in 0..width {
                        for c in 0..3 {
                            let v = ((random.next() % 65_536) as i32 - 1024) as f32 * 0.125;
                            data[y * stride + 3 * x + c] = v;
                        }
                    }
                }
            } else {
                // Every LUT index straddled by nextafter probes on both
                // boundaries; mode&1 selects the direction, mode>=5 the
                // plus side.
                let next_down = |value: f32| -> f32 {
                    if value == 0.0 {
                        f32::from_bits(0x8000_0001)
                    } else if value < 0.0 {
                        f32::from_bits(value.to_bits() + 1)
                    } else {
                        f32::from_bits(value.to_bits() - 1)
                    }
                };
                let next_up = |value: f32| -> f32 { -next_down(-value) };
                for i in 0..4096_usize {
                    let x = 3 + (i % 64) * 4;
                    let y = 3 + (i / 64) * 4;
                    for c in 0..3 {
                        let p = i as f32;
                        let mut n = p - despeckle.lut(c, i);
                        n = if mode & 1 == 1 {
                            next_down(n)
                        } else {
                            next_up(n)
                        };
                        if mode >= 5 {
                            n = p + despeckle.lut(c, i);
                            n = if mode & 1 == 1 {
                                next_down(n)
                            } else {
                                next_up(n)
                            };
                        }
                        for dy in -1_i32..=1 {
                            for dx in -1_i32..=1 {
                                let value = if dx == 0 && dy == 0 { p } else { n };
                                data[(y as i32 + dy) as usize * stride
                                    + 3 * (x as i32 + dx) as usize
                                    + c] = value;
                            }
                        }
                    }
                }
            }
            // The harness ran one native worker over rows [2, h-3].
            let mut records = Vec::new();
            despeckle
                .detect_worker(
                    &data,
                    width,
                    height,
                    stride,
                    2,
                    height - 3,
                    &mut records,
                    Control::none(),
                )
                .unwrap();
            assert_eq!(
                records.len(),
                DETECTOR_RECORDS[mode],
                "mode {mode} detector records"
            );
            let lut0 = match mode {
                0 | 1 => 23.405_7_f64,
                2 => 0.002_340_57,
                _ => 2.340_57,
            };
            let lut4095 = match mode {
                0 => 107.845_f64,
                1 => 238.986,
                2 => 0.010_784_5,
                _ => 10.784_5,
            };
            close6(despeckle.lut(0, 0), lut0, "mode {mode} LUT0");
            close6(despeckle.lut(0, 4095), lut4095, "mode {mode} LUT4095");
            // The oracle's per-mode replacement battery: 7 masks x 53
            // seeds, 29 draws each. The mode-3 correction kernel is the
            // stage-3 port; here the draws keep the stream aligned and
            // the corrections exercise the 0x3ff flag path.
            let repair = super::ListedRepair::from_minimum(3);
            let mut scratch = vec![0.0_f32; stride * height];
            for mask in 1..=7_usize {
                for n in 0..53_usize {
                    let x = 4 + (random.next() % (width as u32 - 8)) as usize;
                    let y = 4 + (random.next() % (height as u32 - 8)) as usize;
                    let record = super::BadPixelRecord {
                        row: y as u16,
                        column: x as u16,
                        flags: (0x3ff | (mask << 12)) as u16,
                    };
                    scratch.copy_from_slice(&data);
                    for dy in -1_i32..=1 {
                        for dx in -1_i32..=1 {
                            for c in 0..3 {
                                let mut v = 1.0_f32 + (random.next() % 64_000) as f32 * 0.031_25;
                                if n % 3 == 0 && dy == -1 {
                                    v = 0.0;
                                }
                                if n % 3 == 1 {
                                    v = if dy < 0 || dx < 0 { 0.0 } else { -v };
                                }
                                let index = (y as i32 + dy) as usize * stride
                                    + 3 * (x as i32 + dx) as usize
                                    + c;
                                scratch[index] = v;
                            }
                        }
                    }
                    repair
                        .apply(
                            &mut scratch,
                            width,
                            height,
                            stride,
                            std::slice::from_ref(&record),
                            &[],
                            Control::none(),
                        )
                        .unwrap();
                }
            }
        }
        // Wrapper modes 0..3: w=93, h=101, stride = w*3+11, one draw
        // per sample, independent channel gains and spatial variation.
        let width = 93_usize;
        let height = 101_usize;
        let stride = width * 3 + 11;
        const WRAPPER_RECORDS: [usize; 4] = [2_682, 2_529, 2_448, 2_277];
        for mode in 0..4_usize {
            let gain = [0.91_f64, 2.735, 4.173];
            let adjust = [10.0_f64, 10.0, 5.0];
            let rect = [
                (8 + mode) as u32,
                (9 + mode) as u32,
                (width - 8 - mode) as u32,
                (height - 9 - mode) as u32,
            ];
            let despeckle = super::Despeckle::new(gain, adjust, rect).unwrap();
            let mut data = vec![-12_345.25_f32; stride * height];
            for y in 0..height {
                for x in 0..width {
                    for c in 0..3 {
                        let raw = ((random.next() % 3000) as i32 - 120) as f32;
                        let spatial = 0.81 + x as f64 * (c + 1) as f64 * 0.0011 + y as f64 * 0.0017;
                        data[y * stride + 3 * x + c] =
                            ((f64::from(raw) - 11.3) * gain[c] * spatial) as f32;
                    }
                }
            }
            let records = despeckle
                .detect(&data, width, height, stride, Control::none())
                .unwrap();
            assert_eq!(
                records.len(),
                WRAPPER_RECORDS[mode],
                "mode {mode} wrapper records"
            );
            // The eight-worker join equals the single full-range scan
            // (worker order is raster order).
            let mut single = Vec::new();
            despeckle
                .detect_worker(
                    &data,
                    width,
                    height,
                    stride,
                    0,
                    height - 1,
                    &mut single,
                    Control::none(),
                )
                .unwrap();
            assert_eq!(records, single, "mode {mode} partition join");
        }
        // Eight controlled wheel probes: the observed raw 3x3
        // neighborhood (deliberately not asserted to be native stage
        // input) marks C0 only; the correction returns 47/3 scaled by
        // the equal channel gain, the good channels unchanged, and is
        // independent of the adjust vector.
        const WHEEL: [[[f32; 3]; 3]; 3] = [
            [[16.0, 16.0, 21.0], [17.0, 17.0, 23.0], [17.0, 23.0, 22.0]],
            [[15.0, 18.0, 23.0], [203.0, 19.0, 20.0], [19.0, 22.0, 24.0]],
            [[16.0, 19.0, 20.0], [21.0, 18.0, 19.0], [16.0, 22.0, 23.0]],
        ];
        let mut previous_gain = None::<(f64, f32)>;
        for adjust_mode in 0..2_usize {
            for gain_value in [1.0_f64, 2.0, 4.0, 8.0] {
                let width = 64_usize;
                let height = 64_usize;
                let stride = width * 3;
                let gain = [gain_value; 3];
                let adjust = if adjust_mode == 0 {
                    [10.0_f64, 10.0, 5.0]
                } else {
                    [15.0, 15.0, 5.0]
                };
                let despeckle = super::Despeckle::new(gain, adjust, [8, 8, 55, 55]).unwrap();
                let mut pixels = vec![(20.0 * gain_value) as f32; stride * height];
                for y in 0..3_usize {
                    for x in 0..3_usize {
                        for c in 0..3_usize {
                            pixels[(y + 31) * stride + 3 * (x + 31) + c] =
                                WHEEL[y][x][c] * gain_value as f32;
                        }
                    }
                }
                let before = [
                    pixels[(32 * stride) + 3 * 32],
                    pixels[(32 * stride) + 3 * 32 + 1],
                    pixels[(32 * stride) + 3 * 32 + 2],
                ];
                let count = despeckle
                    .apply(&mut pixels, width, height, stride, Control::none())
                    .unwrap();
                assert_eq!(count, 1, "adjust {adjust_mode} gain {gain_value}");
                assert_eq!(
                    pixels[(32 * stride) + 3 * 32 + 1],
                    19.0 * gain_value as f32,
                    "C1 unchanged"
                );
                assert_eq!(
                    pixels[(32 * stride) + 3 * 32 + 2],
                    20.0 * gain_value as f32,
                    "C2 unchanged"
                );
                let after = pixels[(32 * stride) + 3 * 32];
                assert_ne!(after, before[0]);
                let expected = 47.0 / 3.0 * gain_value;
                assert!(
                    (f64::from(after) - expected).abs() <= expected * 1.0e-5,
                    "adjust {adjust_mode} gain {gain_value}: {after}"
                );
                if let Some((previous_gain_value, previous_after)) = previous_gain {
                    // Equal channel gains are powers of two, so the
                    // correction scales exactly.
                    assert_eq!(
                        gain_value / previous_gain_value,
                        f64::from(after) / f64::from(previous_after),
                        "exact gain scaling"
                    );
                }
                previous_gain = Some((gain_value, after));
            }
        }
    }
}
