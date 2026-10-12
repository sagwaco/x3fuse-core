//! Ordered Merrill float repair stages that run around listed repair:
//! `DetectBadReds` (0x4b6f4) on the raw float raster before ApplyGains, and
//! `RepairBadPixelClusters` (0x3d634) after the three listed repairs.
//! Evidence: x3f-docs `spp-rgb-trace/followup-repair/` and
//! `followup-tiled-denoise/production-repair/`.

use super::BadPixelRecord;
use crate::control::{Error, Result};
use crate::Control;

/// Native output allocation for detected records (6,000 bytes). The
/// detector keeps one slot free, so at most 999 records are produced.
pub const DETECTED_CAPACITY: usize = 1000;
/// Native primary override allocation (capacity * 1000 entries).
pub const OVERRIDE_CAPACITY: usize = DETECTED_CAPACITY * 1000;

const NEIGHBOR_BITS: [u32; 9] = [1, 2, 4, 8, 0, 16, 32, 64, 128];

fn check_layout(len: usize, width: usize, height: usize, row_stride: usize) -> Result<()> {
    let row = width.checked_mul(3);
    if width == 0
        || height == 0
        || !row.is_some_and(|row| row <= row_stride)
        || !(height - 1)
            .checked_mul(row_stride)
            .and_then(|n| n.checked_add(width * 3))
            .is_some_and(|n| n <= len)
    {
        return Err(Error::InvalidData("invalid Merrill repair image layout"));
    }
    Ok(())
}

// The native float multiplications round separately. Additions and the
// comparison are in f64. There is no FMA.
fn rejects(r: f32, b: f32, noise: &[f64; 9]) -> bool {
    let nr = (noise[0] + f64::from(r * noise[3] as f32)) + f64::from((r * r) * noise[6] as f32);
    let nb = (noise[2] + f64::from(b * noise[5] as f32)) + f64::from((b * b) * noise[8] as f32);
    f64::from(r.max(0.0)) - nr > nb + f64::from(b.max(0.0))
}

// Donor mask for a candidate at `position` in the listed 3x3 neighborhood.
// It excludes the listed center and the original disabled neighbors.
fn replacement_mask(position: usize, original: u32) -> u32 {
    let (px, py) = (position as i32 % 3 - 1, position as i32 / 3 - 1);
    let mut mask = 0;
    for j in (0..9).filter(|&j| j != 4) {
        let (x, y) = (px + j % 3 - 1, py + j / 3 - 1);
        if x == 0 && y == 0 {
            continue;
        }
        let inside = (-1..=1).contains(&x) && (-1..=1).contains(&y);
        if inside && original & NEIGHBOR_BITS[((y + 1) * 3 + x + 1) as usize] == 0 {
            continue;
        }
        mask |= NEIGHBOR_BITS[j as usize];
    }
    mask
}

/// Run the native `DetectBadReds` over one listed set and append to `output`.
///
/// `data` is the decoded raw float raster before black subtraction and gains.
/// Only enabled neighbors of in-bounds listed records are tested. Pixels do
/// not change. A rejected neighbor becomes a new record with badness 3,
/// channel C0 and the donor mask from its position. Appends stop at
/// `DETECTED_CAPACITY - 1` records in `output`.
///
/// `noise` is ColorNoiseModel: constant, linear and quadratic rows of
/// C0, C1, C2. C1 terms are unused. With `overrides`, entry `i` (below
/// `OVERRIDE_CAPACITY`) starts as the record mask and becomes the mask of
/// the neighbors that stayed, if any stayed.
#[allow(clippy::too_many_arguments)]
pub fn detect_bad_reds(
    data: &[f32],
    width: usize,
    height: usize,
    row_stride: usize,
    listed: &[BadPixelRecord],
    noise: &[f64; 9],
    output: &mut Vec<BadPixelRecord>,
    mut overrides: Option<&mut Vec<u32>>,
    control: Control<'_>,
) -> Result<()> {
    control.check()?;
    check_layout(data.len(), width, height, row_stride)?;
    if output.len() >= DETECTED_CAPACITY || !noise.iter().all(|v| v.is_finite()) {
        return Err(Error::InvalidData("invalid Merrill bad red request"));
    }
    if let Some(overrides) = overrides.as_mut() {
        overrides.clear();
        overrides.reserve(listed.len().min(OVERRIDE_CAPACITY));
    }
    for (i, record) in listed.iter().enumerate() {
        if i % 4096 == 0 {
            control.check()?;
        }
        let (x, y) = (usize::from(record.column), usize::from(record.row));
        let mask = u32::from(record.flags & 0xff);
        let mut remaining = mask;
        if let Some(overrides) = overrides.as_mut().filter(|_| i < OVERRIDE_CAPACITY) {
            overrides.push(mask);
        }
        if y < 2 || y + 1 >= height || x < 2 || x + 1 >= width || mask == 0 {
            continue;
        }
        for j in (0..9).filter(|&j| mask & NEIGHBOR_BITS[j] != 0) {
            let (xx, yy) = (x + j % 3 - 1, y + j / 3 - 1);
            let p = yy * row_stride + 3 * xx;
            if rejects(data[p], data[p + 2], noise) && output.len() < DETECTED_CAPACITY - 1 {
                output.push(BadPixelRecord {
                    row: yy as u16,
                    column: xx as u16,
                    flags: (0x4300 | replacement_mask(j, mask)) as u16,
                });
                remaining -= NEIGHBOR_BITS[j];
            }
        }
        if let Some(overrides) = overrides.as_mut().filter(|_| i < OVERRIDE_CAPACITY) {
            if remaining != 0 {
                overrides[i] = remaining;
            }
        }
    }
    Ok(())
}

/// Counts from one packed-cluster pass.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ClusterStats {
    pub records: usize,
    pub points: usize,
    pub zero_fallback: usize,
}

struct Cluster<'a> {
    x: usize,
    y: usize,
    rows: usize,
    columns: usize,
    left: usize,
    top: usize,
    right: usize,
    bottom: usize,
    payload: &'a [u16],
}

// Donor-to-target position maps for the six native patch orientations.
const ORIENTATIONS: [[usize; 9]; 6] = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8],
    [6, 3, 0, 7, 4, 1, 8, 5, 2],
    [2, 5, 8, 1, 4, 7, 0, 3, 6],
    [6, 7, 8, 3, 4, 5, 0, 1, 2],
    [2, 1, 0, 5, 4, 3, 8, 7, 6],
    [8, 7, 6, 5, 4, 3, 2, 1, 0],
];

// Select records and validate them before any write. The native returns
// from the whole call on the first selected record with invalid geometry.
fn plan_clusters<'a>(
    words: &'a [u16],
    width: usize,
    height: usize,
    badness: u32,
    request_margin: u32,
) -> Result<Vec<Cluster<'a>>> {
    let invalid = || Error::InvalidData("invalid Merrill packed cluster list");
    let mut clusters = Vec::new();
    let mut offset = 0;
    // The native counts a record at every start before the last word.
    while offset + 1 < words.len() {
        let header = words.get(offset..offset + 4).ok_or_else(invalid)?;
        let count = usize::from(header[3] & 0xff);
        let payload = words
            .get(offset + 4..offset + 4 + count)
            .ok_or_else(invalid)?;
        offset += 4 + count;
        if u32::from(header[0] >> 14) != badness {
            continue;
        }
        let (y, x) = (usize::from(header[1]), usize::from(header[2]));
        let (rows, columns) = (
            usize::from((header[0] >> 7) & 127),
            usize::from(header[0] & 127),
        );
        if y >= height || x >= width || rows > 25 || columns > 25 {
            break;
        }
        let margin = 5.max(u32::from(header[3] >> 8).min(request_margin)) as usize;
        let (left, top) = (x.saturating_sub(margin), y.saturating_sub(margin));
        let right = (width - 1).min(x + columns + margin);
        let bottom = (height - 1).min(y + rows + margin);
        let (w, h) = (right - left + 1, bottom - top + 1);
        if w * h > 10_000 {
            break;
        }
        // The scan rectangle and each 5x5 priority window must lie inside
        // the expanded bounds. Sites must be unique, in the scan rectangle,
        // and have a channel mask 1..7.
        if x - left < 1 || y - top < 1 || x - left + columns + 2 > w || y - top + rows + 2 > h {
            return Err(invalid());
        }
        let mut seen = std::collections::HashSet::new();
        for &word in payload {
            let (px, py) = (
                x - left + usize::from((word >> 4) & 63),
                y - top + usize::from(word >> 10),
            );
            let mask = word & 15;
            if !(1..=7).contains(&mask)
                || px > x - left + columns
                || py > y - top + rows
                || (px >= 2 && py >= 2 && (px + 2 >= w || py + 2 >= h))
                || !seen.insert((px, py))
            {
                return Err(invalid());
            }
        }
        clusters.push(Cluster {
            x,
            y,
            rows,
            columns,
            left,
            top,
            right,
            bottom,
            payload,
        });
    }
    Ok(clusters)
}

/// Run the native `RepairBadPixelClusters` over the packed `BadClustersF20`
/// words. Only records whose badness equals `badness` run.
///
/// Each record marks bad channels in a window around its rectangle. Marked
/// pixels are repaired in descending priority order. The priority is the
/// count of known channels in a 5x5 window. Each repair copies the full RGB
/// center of the clean 3x3 donor patch with the smallest masked difference
/// over six orientations. Without a donor below 1,000,000 the pixel becomes
/// zero. Later repairs see earlier results.
///
/// The native returns early on the first selected record with an out-of-range
/// base, an extent above 25 or a window above 10,000 pixels. Earlier records
/// stay repaired. Malformed lists return an error before any write.
#[allow(clippy::too_many_arguments)]
pub fn repair_bad_pixel_clusters(
    data: &mut [f32],
    width: usize,
    height: usize,
    row_stride: usize,
    words: &[u16],
    badness: u32,
    request_margin: u32,
    control: Control<'_>,
) -> Result<ClusterStats> {
    control.check()?;
    check_layout(data.len(), width, height, row_stride)?;
    let clusters = plan_clusters(words, width, height, badness, request_margin)?;
    let mut stats = ClusterStats::default();
    for cluster in &clusters {
        control.check()?;
        let (w, h) = (
            cluster.right - cluster.left + 1,
            cluster.bottom - cluster.top + 1,
        );
        let (cx0, cy0) = (cluster.x - cluster.left, cluster.y - cluster.top);
        let mut mask = vec![[1.0_f32; 3]; w * h];
        for &word in cluster.payload {
            let p = (cy0 + usize::from(word >> 10)) * w + cx0 + usize::from((word >> 4) & 63);
            for c in 0..3 {
                if word & (4 >> c) != 0 {
                    mask[p][c] = 0.0;
                }
            }
        }
        let clean: Vec<bool> = mask.iter().map(|m| m.iter().all(|&v| v != 0.0)).collect();
        let mut order = Vec::new();
        for py in cy0 - 1..cy0 + cluster.rows + 2 {
            for px in cx0 - 1..cx0 + cluster.columns + 2 {
                if clean[py * w + px] {
                    continue;
                }
                let mut sum = 0.0_f32;
                if py >= 2 && px >= 2 {
                    for yy in py - 2..=py + 2 {
                        for m in &mask[yy * w + px - 2..=yy * w + px + 2] {
                            sum += (m[0] + m[1]) + m[2];
                        }
                    }
                }
                order.push((sum as u32, py, px));
            }
        }
        // The plan admits only unique sites inside the scan rectangle.
        debug_assert_eq!(order.len(), cluster.payload.len());
        // Stable: equal priorities keep raster order.
        order.sort_by(|a, b| b.0.cmp(&a.0));
        // Donor eligibility uses the original marks for the whole record.
        let donors: Vec<(usize, usize)> = (cluster.top + 1..cluster.bottom - 1)
            .flat_map(|cy| (cluster.left + 1..cluster.right - 1).map(move |cx| (cy, cx)))
            .filter(|&(cy, cx)| {
                (0..3).all(|dy| {
                    (0..3).all(|dx| {
                        clean[(cy - cluster.top + dy - 1) * w + cx - cluster.left + dx - 1]
                    })
                })
            })
            .collect();
        let at = |data: &[f32], x: usize, y: usize| -> [f32; 3] {
            let i = y * row_stride + 3 * x;
            [data[i], data[i + 1], data[i + 2]]
        };
        for &(_, py, px) in &order {
            let (gx, gy) = (cluster.left + px, cluster.top + py);
            let mut weights = [[0.0_f32; 3]; 9];
            let mut negative = [[0.0_f32; 3]; 9];
            for t in 0..9 {
                weights[t] = mask[(py + t / 3 - 1) * w + px + t % 3 - 1];
            }
            // The native preload pairs the NE and E masks with the E and NE
            // samples. Keep this association.
            weights.swap(2, 5);
            for t in 0..9 {
                let target = at(data, gx + t % 3 - 1, gy + t / 3 - 1);
                for c in 0..3 {
                    negative[t][c] = -weights[t][c] * target[c];
                }
            }
            let (mut best, mut best_score) = ([0.0_f32; 3], 1_000_000.0_f32);
            for &(cy, cx) in &donors {
                let mut sums = [[0.0_f32; 9]; 9];
                for d in 0..9 {
                    let donor = at(data, cx + d % 3 - 1, cy + d / 3 - 1);
                    for t in 0..9 {
                        let q = |c: usize| donor[c].mul_add(weights[t][c], negative[t][c]).abs();
                        sums[d][t] = (q(0) + q(1)) + q(2);
                    }
                }
                for map in &ORIENTATIONS {
                    let score = (1..9).fold(sums[0][map[0]], |s, d| s + sums[d][map[d]]);
                    if score < best_score {
                        best_score = score;
                        best = at(data, cx, cy);
                    }
                }
            }
            if best_score == 1_000_000.0 {
                stats.zero_fallback += 1;
            }
            let i = gy * row_stride + 3 * gx;
            data[i..i + 3].copy_from_slice(&best);
            mask[py * w + px] = [1.0; 3];
            stats.points += 1;
        }
        stats.records += 1;
    }
    Ok(stats)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::float_front::fnv1a_f32;

    fn open_corpus(name: &str) -> Option<(*mut crate::x3f_t, *mut crate::sysabi::FILE)> {
        use crate::{x3f_get_camf, x3f_get_raw, x3f_load_data, x3f_new_from_file};
        let root = std::env::var_os("X3F_TEST_FILES")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| {
                std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../x3f_test_files")
            });
        let path = root.join(name);
        if !path.exists() {
            return None;
        }
        let path = std::ffi::CString::new(path.to_str().unwrap()).unwrap();
        unsafe {
            let file = crate::sysabi::fopen(path.as_ptr(), c"rb".as_ptr());
            assert!(!file.is_null());
            let x3f = x3f_new_from_file(file.cast());
            assert!(!x3f.is_null());
            assert_eq!(
                x3f_load_data(x3f, x3f_get_camf(x3f)),
                crate::x3f_return_e_X3F_OK
            );
            let prop = crate::x3f_get_prop(x3f);
            if !prop.is_null() {
                assert_eq!(x3f_load_data(x3f, prop), crate::x3f_return_e_X3F_OK);
            }
            assert_eq!(
                x3f_load_data(x3f, x3f_get_raw(x3f)),
                crate::x3f_return_e_X3F_OK
            );
            Some((x3f, file))
        }
    }

    fn write<T: Copy>(dir: &std::path::Path, name: &str, values: &[T]) {
        let bytes = unsafe {
            std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
        };
        std::fs::write(dir.join(name), bytes).unwrap();
    }

    fn record_words(records: &[BadPixelRecord]) -> Vec<u16> {
        records
            .iter()
            .flat_map(|r| [r.row, r.column, r.flags])
            .collect()
    }

    fn fnv_bytes<T: Copy>(values: &[T]) -> u64 {
        let bytes = unsafe {
            std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
        };
        let mut hash = 0xcbf2_9ce4_8422_2325_u64;
        for &byte in bytes {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
        hash
    }

    struct Rng(u32);
    impl Rng {
        fn next(&mut self) -> u32 {
            self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            self.0 >> 8
        }
    }

    /// The ordered native repair chain on synthetic padded rasters
    /// (x3f-docs `production-repair/native_synthetic.cpp`): DetectBadReds
    /// with overrides and the JPEG append, ApplyGains without tables, the
    /// three listed repairs and packed clusters.
    #[test]
    fn repair_chain_matches_native_battery() {
        let noise = [5.25, 1.0, 3.5, 0.0731, 0.5, 0.0457, 1.3e-5, 0.2, 2.9e-5];
        let fixture = include_str!("../../tests/fixtures/spp-repair-synthetic-native.txt");
        let mut cases = 0;
        for line in fixture.lines().filter(|l| l.starts_with("REPAIR ")) {
            let f: Vec<&str> = line.split_whitespace().collect();
            let v = |i: usize| f[i].parse::<u32>().unwrap();
            let x = |i: usize| u64::from_str_radix(f[i], 16).unwrap();
            let (w, h, pad, seed, label, margin) = (v(1), v(2), v(3), v(4), v(5), v(6));
            let (wu, hu) = (w as usize, h as usize);
            let stride = 3 * wu + pad as usize;
            let minimum = label.saturating_sub(1);
            let mut rng = Rng(0x5eed_0000 + 977 * seed + 31 * w + pad);
            let mut raw = vec![-77.5_f32; stride * hu];
            for y in 0..hu {
                for x in 0..wu {
                    for c in 0..3 {
                        raw[y * stride + 3 * x + c] = (rng.next() % 4096) as f32;
                    }
                    if rng.next() % 5 == 0 {
                        raw[y * stride + 3 * x] = (3000 + rng.next() % 1000) as f32;
                    }
                }
            }
            let mut make_list = |n: u32| -> Vec<BadPixelRecord> {
                (0..n)
                    .map(|_| {
                        let row = (rng.next() % h) as u16;
                        let column = (rng.next() % w) as u16;
                        let flags = (rng.next() & 0xff)
                            | ((rng.next() % 4) << 8)
                            | ((1 + rng.next() % 7) << 12);
                        BadPixelRecord {
                            row,
                            column,
                            flags: flags as u16,
                        }
                    })
                    .collect()
            };
            let primary = make_list(3 + w * h / 6);
            let jpeg = make_list(1 + w * h / 24);
            let mut words = Vec::new();
            if w > 12 && h > 12 {
                for _ in 0..3 {
                    let (rows, cols) = (rng.next() % 4, rng.next() % 4);
                    let y = 2 + rng.next() % (h - rows - 4);
                    let x = 2 + rng.next() % (w - cols - 4);
                    let (badness, stored) = (rng.next() % 4, rng.next() % 30);
                    let mut seen = std::collections::HashSet::new();
                    let mut payload = Vec::new();
                    for _ in 0..1 + rng.next() % ((rows + 1) * (cols + 1)) {
                        let (dy, dx) = (rng.next() % (rows + 1), rng.next() % (cols + 1));
                        if seen.insert((dy, dx)) {
                            payload.push(((dy << 10) | (dx << 4) | (1 + rng.next() % 7)) as u16);
                        }
                    }
                    words.extend([
                        ((badness << 14) | (rows << 7) | cols) as u16,
                        y as u16,
                        x as u16,
                        ((stored << 8) as usize | payload.len()) as u16,
                    ]);
                    words.extend(payload);
                }
            }
            assert_eq!(fnv_bytes(&raw), x(7), "{line}");
            let mut detected = Vec::new();
            let mut overrides = Vec::new();
            let none = Control::none();
            detect_bad_reds(
                &raw,
                wu,
                hu,
                stride,
                &primary,
                &noise,
                &mut detected,
                Some(&mut overrides),
                none,
            )
            .unwrap();
            detect_bad_reds(
                &raw,
                wu,
                hu,
                stride,
                &jpeg,
                &noise,
                &mut detected,
                None,
                none,
            )
            .unwrap();
            assert_eq!(detected.len(), v(8) as usize, "{line}");
            assert_eq!(fnv_bytes(&overrides), x(9), "{line}");
            let words16: Vec<u16> = detected
                .iter()
                .flat_map(|r| [r.row, r.column, r.flags])
                .collect();
            assert_eq!(fnv_bytes(&words16), x(10), "{line}");
            let front =
                super::super::FloatFront::new([64.5, 63.25, 65.125], [1.37, 1.0, 2.11], None)
                    .unwrap();
            let mut data = raw;
            front
                .apply_in_place(&mut data, wu, hu, stride, none)
                .unwrap();
            assert_eq!(fnv_bytes(&data), x(11), "{line}");
            let listed = super::super::ListedRepair::new(label, 1);
            listed
                .apply(&mut data, wu, hu, stride, &primary, &overrides, none)
                .unwrap();
            assert_eq!(fnv_bytes(&data), x(12), "{line}");
            listed
                .apply(&mut data, wu, hu, stride, &jpeg, &[], none)
                .unwrap();
            assert_eq!(fnv_bytes(&data), x(13), "{line}");
            listed
                .apply(&mut data, wu, hu, stride, &detected, &[], none)
                .unwrap();
            assert_eq!(fnv_bytes(&data), x(14), "{line}");
            repair_bad_pixel_clusters(&mut data, wu, hu, stride, &words, minimum, margin, none)
                .unwrap();
            assert_eq!(fnv_bytes(&data), x(15), "{line}");
            cases += 1;
        }
        assert_eq!(cases, 96);
    }

    /// Fixed-bit native witnesses (followup-repair/bad-reds/arithmetic-witnesses.json).
    /// Each rejects one wrong formula: reassociated f64 additions, r * (r * n6),
    /// or comparison in f32. Noise holds indices 0, 2, 3, 5, 6, 8.
    #[test]
    fn detector_predicate_matches_native_witnesses() {
        let witnesses: [(u32, u32, [u64; 6], bool); 6] = [
            (
                0x45707bec,
                0x44e7ee1e,
                [
                    0x4075de9dc12d8cd5,
                    0x4007fc7f6939956b,
                    0x3fac15bcca651b70,
                    0x3fc4e9da31cae970,
                    0x3f0d058306f37d48,
                    0x3f17462010908eca,
                ],
                false,
            ),
            (
                0x4522287a,
                0x440681a7,
                [
                    0x4099f147fbc49dcb,
                    0x3ff5f6f4ed88d2d6,
                    0x3fba29ff11a1647f,
                    0x3fbca85b7dc3141d,
                    0x3edad6708387f4ae,
                    0x3f17da6fff78fe29,
                ],
                false,
            ),
            (
                0x454b77dd,
                0x44a5b522,
                [
                    0x408642a69c000000,
                    0x0,
                    0x3fc55d53673fdf74,
                    0x3fbbc00bfb3d2a7f,
                    0x3f05d0518fc1ca3a,
                    0x3f0ab11c7a199d33,
                ],
                false,
            ),
            (
                0x45652718,
                0x44a2181a,
                [
                    0x4096ce6ccd800000,
                    0x0,
                    0x3fc1c3c5e2dac030,
                    0x3fbed084216cd9c2,
                    0x3ef26427d2515f85,
                    0x3ed75ad35dee9e8c,
                ],
                false,
            ),
            (
                0x4443b2f1,
                0x4300847a,
                [
                    0x408181ebfa3b3df5,
                    0x3fdd8e0ba61055ca,
                    0x3fa6d47dbbe4112a,
                    0x3fc54b0039f7168c,
                    0x3f0eef0265bb673a,
                    0x3f121190ab1d3a80,
                ],
                true,
            ),
            (
                0x45293839,
                0x43cdbbd8,
                [
                    0x40983d67b1ef7c00,
                    0x3fa3843841ffdfb8,
                    0x3f9c8058915bfea4,
                    0x3fc66e34ffbfe608,
                    0x3f151b21c5bdb8d9,
                    0x3f059f3af3c46777,
                ],
                true,
            ),
        ];
        for (r, b, n, native) in witnesses {
            let mut noise = [0.0; 9];
            for (slot, bits) in [0, 2, 3, 5, 6, 8].into_iter().zip(n) {
                noise[slot] = f64::from_bits(bits);
            }
            assert_eq!(
                rejects(f32::from_bits(r), f32::from_bits(b), &noise),
                native
            );
        }
    }

    #[test]
    fn invalid_repair_calls_leave_buffers_unchanged() {
        let none = Control::none();
        let noise = [0.0; 9];
        let data = vec![1.5_f32; 3 * 8 * 8];
        let record = BadPixelRecord {
            row: 3,
            column: 3,
            flags: 0x43ff,
        };
        for (len, width, height, stride) in [
            (191, 8, 8, 24),
            (192, 8, 8, 23),
            (192, 0, 8, 24),
            (192, 8, 0, 24),
        ] {
            let mut output = vec![record];
            let mut overrides = vec![7];
            assert!(detect_bad_reds(
                &data[..len],
                width,
                height,
                stride,
                &[record],
                &noise,
                &mut output,
                Some(&mut overrides),
                none
            )
            .is_err());
            assert_eq!((output.len(), overrides), (1, vec![7]));
        }
        let mut full = vec![record; DETECTED_CAPACITY];
        assert!(
            detect_bad_reds(&data, 8, 8, 24, &[record], &noise, &mut full, None, none).is_err()
        );
        assert!(detect_bad_reds(
            &data,
            8,
            8,
            24,
            &[record],
            &[f64::NAN; 9],
            &mut Vec::new(),
            None,
            none
        )
        .is_err());
        // Malformed packed lists: truncated header and payload, a zero
        // channel mask, a duplicate site, a site outside the extents, and a
        // window without the two-pixel halo.
        let site = |dy: u16, dx: u16, mask: u16| (dy << 10) | (dx << 4) | mask;
        for words in [
            vec![0, 3, 3],
            vec![0, 3, 3, 2, site(0, 0, 4)],
            vec![0, 3, 3, 1, site(0, 0, 0)],
            vec![1, 3, 3, 2, site(0, 0, 4), site(0, 0, 2)],
            vec![0, 3, 3, 1, site(0, 1, 4)],
            vec![0, 6, 3, 1, site(0, 0, 4)],
        ] {
            let mut image = data.clone();
            assert!(
                repair_bad_pixel_clusters(&mut image, 8, 8, 24, &words, 0, 25, none).is_err(),
                "{words:?}"
            );
            assert_eq!(image, data);
        }
        // A non-selected malformed record is skipped like the native.
        let mut image = data.clone();
        assert!(repair_bad_pixel_clusters(
            &mut image,
            8,
            8,
            24,
            &[0, 3, 3, 1, site(0, 0, 0)],
            1,
            25,
            none
        )
        .is_ok());
        assert_eq!(image, data);
    }

    /// The ordered Merrill float stages on the two retained reference files.
    /// The fixture pins each stage hash and detail. The native corpus harness
    /// (x3f-docs `followup-tiled-denoise/production-repair/native_corpus.cpp`)
    /// runs the exported native kernels from the dumped stage inputs and
    /// reproduces every pinned hash except radial and offset, which have no
    /// direct native export. The prepared row also pins the as-shot burst
    /// gate, raw saturation-map counts, and offset ISO. Set X3F_REPAIR_REGEN
    /// to print rows. Set X3F_REPAIR_DUMP to a directory to write the harness
    /// inputs.
    #[test]
    fn prepared_float_matches_corpus_pins() {
        let regen = std::env::var_os("X3F_REPAIR_REGEN").is_some();
        let dump = std::env::var_os("X3F_REPAIR_DUMP").map(std::path::PathBuf::from);
        let fixture = include_str!("../../tests/fixtures/spp-repair-corpus.tsv");
        let mut checked = 0;
        for name in ["sigma_sd1_merrill_15.x3f", "_P2M0358.X3F"] {
            let Some((x3f, file)) = open_corpus(name) else {
                continue;
            };
            let wb = unsafe { crate::x3f_get_wb(x3f) };
            let mut rows = Vec::new();
            let mut raw = Vec::new();
            let mut radial = Vec::new();
            let mut observe = |stage: &'static str, data: &[f32], detail: String| {
                if stage == "raw" {
                    raw = data.to_vec();
                }
                if stage == "radial" {
                    radial = data.to_vec();
                }
                rows.push(
                    format!("{name}\t{stage}\t{:016x}\t{detail}", fnv1a_f32(data))
                        .trim_end()
                        .to_owned(),
                );
            };
            let prepared = unsafe {
                crate::process::merrill_prepare_float(
                    x3f,
                    wb,
                    true,
                    true,
                    None,
                    Control::none(),
                    &mut observe,
                )
            }
            .unwrap();
            let (width, height, stride) = (prepared.width, prepared.height, prepared.row_stride);
            let burst = unsafe { crate::process::merrill_burst_exposure_adjust_included(x3f) };
            assert!(!burst, "{name} as-shot burst gate");
            let (active, keep, col_range) = unsafe { crate::process::merrill_rects(x3f) }.unwrap();
            let highlight = unsafe {
                crate::process::merrill_highlight_restoration(
                    x3f,
                    active,
                    keep,
                    col_range,
                    Control::none(),
                )
            }
            .unwrap()
            .expect("corpus saturation maps");
            let saturation =
                highlight
                    .sat
                    .chunks_exact(3)
                    .fold([0_usize; 3], |mut counts, channels| {
                        for channel in 0..3 {
                            counts[channel] += usize::from(channels[channel] != 0);
                        }
                        counts
                    });
            let offset = unsafe { crate::process::merrill_cr_offset(x3f, wb, Control::none()) }
                .unwrap()
                .expect("corpus offset calibration");
            rows.push(format!(
                "{name}\tprepared\t{:016x}\t{}x{} stride={} burst={burst} sat={saturation:?} offset_iso={:08x}",
                fnv1a_f32(&prepared.data),
                width,
                height,
                stride,
                offset.iso.to_bits()
            ));
            let primary = unsafe {
                crate::process::merrill_bad_pixel_list(x3f, c"BadPixelsF20", Control::none())
            }
            .unwrap()
            .unwrap();
            let jpeg = unsafe {
                crate::process::merrill_bad_pixel_list(x3f, c"Jpeg_BadClusters", Control::none())
            }
            .unwrap()
            .unwrap_or_default();
            let noise = unsafe { crate::process::merrill_color_noise_model(x3f) }.unwrap();
            let mut detected = Vec::new();
            let mut overrides = Vec::new();
            detect_bad_reds(
                &raw,
                width,
                height,
                stride,
                &primary,
                &noise,
                &mut detected,
                Some(&mut overrides),
                Control::none(),
            )
            .unwrap();
            detect_bad_reds(
                &raw,
                width,
                height,
                stride,
                &jpeg,
                &noise,
                &mut detected,
                None,
                Control::none(),
            )
            .unwrap();
            if name.starts_with("sigma_sd1") {
                // The archived native run on the pinned SD1 raw float copy
                // (followup-repair/bad-reds/FINDINGS.txt).
                let records: Vec<_> = detected
                    .iter()
                    .map(|r| (r.row, r.column, r.flags))
                    .collect();
                assert_eq!(
                    records,
                    [
                        (766, 4859, 17375),
                        (996, 4737, 17403),
                        (2004, 279, 17399),
                        (3041, 4832, 17403)
                    ]
                );
            }
            if let Some(dir) = &dump {
                let dir = dir.join(name);
                std::fs::create_dir_all(&dir).unwrap();
                write(&dir, "raw.f32", &raw);
                write(&dir, "radial.f32", &radial);
                write(&dir, "primary.u16", &record_words(&primary));
                write(&dir, "jpeg.u16", &record_words(&jpeg));
                write(&dir, "detected.u16", &record_words(&detected));
                write(&dir, "overrides.u32", &overrides);
                write(&dir, "noise.f64", &noise);
                let front = unsafe {
                    crate::process::merrill_float_gain_front(x3f, wb, true, Control::none())
                }
                .unwrap();
                let tables = front.spatial().unwrap();
                for (c, table) in tables.iter().enumerate() {
                    write(&dir, &format!("table{c}.f32"), &table.values);
                }
                let burst = unsafe { crate::process::merrill_burst_exposure_adjust_included(x3f) };
                let (label, offset) = unsafe {
                    crate::process::merrill_badness_selection(x3f, burst, Control::none())
                }
                .unwrap()
                .unwrap();
                let words = unsafe { crate::process::merrill_packed_cluster_words(x3f) }
                    .unwrap()
                    .unwrap_or_default();
                write(&dir, "clusters.u16", &words);
                let mut cluster_offset = 1_u32;
                let mut margin = 25_u32;
                let mut skip = 0_u32;
                unsafe {
                    crate::x3f_get_camf_unsigned(
                        x3f,
                        c"BCBadnessOffset".as_ptr().cast_mut(),
                        &mut cluster_offset,
                    );
                    crate::x3f_get_camf_unsigned(
                        x3f,
                        c"BadClusterMargin".as_ptr().cast_mut(),
                        &mut margin,
                    );
                    crate::x3f_get_camf_unsigned(
                        x3f,
                        c"SkipStepFlags".as_ptr().cast_mut(),
                        &mut skip,
                    );
                }
                let (active, keep, col_range) =
                    unsafe { crate::process::merrill_rects(x3f) }.unwrap();
                let highlight = unsafe {
                    crate::process::merrill_highlight_restoration(
                        x3f,
                        active,
                        keep,
                        col_range,
                        Control::none(),
                    )
                }
                .unwrap()
                .unwrap();
                write(&dir, "sat.u8", &highlight.sat);
                let (dgain, dadjust, drect) = unsafe {
                    crate::process::merrill_despeckle_inputs(x3f, wb, None, Control::none())
                }
                .unwrap()
                .unwrap();
                let bits = |v: &[f64]| {
                    v.iter()
                        .map(|v| format!("{:016x}", v.to_bits()))
                        .collect::<Vec<_>>()
                        .join(" ")
                };
                let meta = [
                    format!("size {width} {height} {stride}"),
                    format!("lists {} {} {}", primary.len(), jpeg.len(), detected.len()),
                    format!("badness {label} {offset} {cluster_offset} {margin} {skip}"),
                    format!("black {}", bits(&front.black())),
                    format!("gain {}", bits(&front.gain())),
                    format!(
                        "tables {}",
                        tables
                            .iter()
                            .map(|t| format!("{} {}", t.columns, t.rows))
                            .collect::<Vec<_>>()
                            .join(" ")
                    ),
                    format!(
                        "highlight {:016x} {:?}",
                        highlight.restore.threshold().to_bits(),
                        highlight.valid
                    ),
                    format!("despeckle {} {} {:?}", bits(&dgain), bits(&dadjust), drect),
                    format!("words {}", words.len()),
                ];
                std::fs::write(dir.join("meta.txt"), meta.join("\n") + "\n").unwrap();
                std::fs::write(dir.join("rust.tsv"), rows.join("\n") + "\n").unwrap();
            }
            unsafe {
                crate::x3f_delete(x3f);
                crate::sysabi::fclose(file);
            }
            if regen {
                println!("{}", rows.join("\n"));
            } else {
                let expected: Vec<_> = fixture
                    .lines()
                    .filter(|l| l.starts_with(&format!("{name}\t")))
                    .collect();
                assert_eq!(rows, expected, "{name}");
            }
            checked += 1;
        }
        eprintln!("Merrill prepared float: {checked}/2 corpus files present");
    }
}
