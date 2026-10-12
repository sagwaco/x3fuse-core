//! Native tile halos and reverse-order seam blending.

use crate::{Error, Result};

/// CopyImageToTile @0x7e18c. Extra destination columns and rows keep their values.
#[allow(clippy::too_many_arguments)]
pub(crate) fn copy_image_to_tile(
    source: &[f32],
    source_stride: usize,
    size: [usize; 2],
    grid: [usize; 2],
    halo: [usize; 2],
    index: usize,
    destination: &mut [f32],
    destination_stride: usize,
    tile: [usize; 2],
) -> Result<()> {
    let invalid = || Error::InvalidData("invalid SPP prefiltered tile layout");
    if grid.contains(&0)
        || size.contains(&0)
        || grid[0].checked_mul(grid[1]).is_none_or(|n| index >= n)
        || source_stride < size[0].saturating_mul(3)
        || source_stride
            .checked_mul(size[1])
            .is_none_or(|n| n > source.len())
        || destination_stride < tile[0].saturating_mul(3)
        || destination_stride
            .checked_mul(tile[1])
            .is_none_or(|n| n > destination.len())
    {
        return Err(invalid());
    }
    let core = [size[0] / grid[0], size[1] / grid[1]];
    if (0..2).any(|i| {
        halo[i] > core[i]
            || core[i] == 0
            || halo[i]
                .checked_mul(2)
                .and_then(|h| core[i].checked_add(h))
                .is_none_or(|n| tile[i] < n)
    }) {
        return Err(invalid());
    }
    let [tx, ty] = [index % grid[0], index / grid[0]];
    let first = [tx == 0, ty == 0];
    let last = [tx + 1 == grid[0], ty + 1 == grid[1]];
    let from = [
        (tx * core[0]).saturating_sub(halo[0]),
        (ty * core[1]).saturating_sub(halo[1]),
    ];
    let to = [
        if first[0] { halo[0] } else { 0 },
        if first[1] { halo[1] } else { 0 },
    ];
    let count: [usize; 2] = std::array::from_fn(|i| {
        core[i] + if first[i] { 0 } else { halo[i] } + if last[i] { 0 } else { halo[i] }
    });
    for y in 0..count[1] {
        let a = (from[1] + y) * source_stride + from[0] * 3;
        let b = (to[1] + y) * destination_stride + to[0] * 3;
        destination[b..b + count[0] * 3].copy_from_slice(&source[a..a + count[0] * 3]);
    }
    for y in 0..tile[1] {
        let from = if first[1] && y < halo[1] {
            Some(halo[1])
        } else if last[1] && y >= tile[1] - halo[1] {
            Some(tile[1] - halo[1] - 1)
        } else {
            None
        };
        if let Some(from) = from {
            destination.copy_within(
                from * destination_stride..from * destination_stride + tile[0] * 3,
                y * destination_stride,
            );
        }
    }
    for y in 0..tile[1] {
        for x in 0..tile[0] {
            let from = if first[0] && x < halo[0] {
                Some(halo[0])
            } else if last[0] && x >= tile[0] - halo[0] {
                Some(tile[0] - halo[0] - 1)
            } else {
                None
            };
            if let Some(from) = from {
                let from = y * destination_stride + from * 3;
                destination.copy_within(from..from + 3, y * destination_stride + x * 3);
            }
        }
    }
    Ok(())
}

/// Each dimension must divide by its grid count; native remainders stay unmerged.
pub(crate) struct TileLayout {
    pub size: [usize; 2],
    pub grid: [usize; 2],
    pub halo: [usize; 2],
    pub core: [usize; 2],
    pub tile: [usize; 2],
    pub samples: usize,
}

impl TileLayout {
    pub fn new(size: [usize; 2], grid: [usize; 2], halo: [usize; 2]) -> Result<Self> {
        let invalid = || Error::InvalidData("invalid SPP tile geometry");
        if size.contains(&0) || grid.contains(&0) || (0..2).any(|i| size[i] % grid[i] != 0) {
            return Err(invalid());
        }
        let core = [size[0] / grid[0], size[1] / grid[1]];
        if core[0] < 2
            || core[1] < 2
            || halo[1] < 2
            || (0..2).any(|i| halo[i] == 0 || halo[i] > core[i])
        {
            return Err(invalid());
        }
        let mut tile = [0; 2];
        for i in 0..2 {
            tile[i] = halo[i]
                .checked_mul(2)
                .and_then(|h| h.checked_add(core[i]))
                .ok_or_else(invalid)?;
        }
        let samples = tile[0]
            .checked_mul(tile[1])
            .and_then(|n| n.checked_mul(3))
            .ok_or_else(invalid)?;
        samples
            .checked_mul(grid[0])
            .and_then(|n| n.checked_mul(grid[1]))
            .ok_or_else(invalid)?;
        Ok(Self {
            size,
            grid,
            halo,
            core,
            tile,
            samples,
        })
    }

    pub fn sample_count(&self) -> usize {
        self.samples * self.grid[0] * self.grid[1]
    }

    pub fn initialize(&self, source: &[f32], stride: usize) -> Result<Vec<f32>> {
        if stride < self.size[0] * 3
            || stride
                .checked_mul(self.size[1])
                .is_none_or(|n| n > source.len())
        {
            return Err(Error::InvalidData("invalid SPP tile source"));
        }
        let mut result = Vec::new();
        result
            .try_reserve_exact(self.sample_count())
            .map_err(|_| Error::Allocation)?;
        // Native fresh caches start at zero; reused native caches retain old halo data.
        result.resize(self.sample_count(), 0.0);
        let [w, h] = self.core;
        let [hx, hy] = self.halo;
        let at = |tile, x, y| tile * self.samples + (y * self.tile[0] + x) * 3;
        for ty in 0..self.grid[1] {
            for tx in 0..self.grid[0] {
                for y in 0..h {
                    let from = (ty * h + y) * stride + tx * w * 3;
                    let to = at(ty * self.grid[0] + tx, hx, hy + y);
                    result[to..to + w * 3].copy_from_slice(&source[from..from + w * 3]);
                }
            }
        }
        for ty in 0..self.grid[1] {
            for tx in 0..self.grid[0] {
                let tile = ty * self.grid[0] + tx;
                let last_x = tx + 1 == self.grid[0];
                let last_y = ty + 1 == self.grid[1];
                for y in 0..hy {
                    let from = if ty == 0 {
                        at(tile, hx, hy + 1)
                    } else {
                        at(tile - self.grid[0], hx, h + y)
                    };
                    result.copy_within(from..from + w * 3, at(tile, hx, y));
                }
                for y in hy..hy + h {
                    if tx == 0 {
                        for x in 0..hx {
                            let from = at(tile, hx + 1, y);
                            result.copy_within(from..from + 3, at(tile, x, y));
                        }
                    } else {
                        let from = at(tile - 1, w, y);
                        result.copy_within(from..from + hx * 3, at(tile, 0, y));
                    }
                }
                for y in 0..hy {
                    let from = if last_y {
                        at(tile, hx, h + hy - 1)
                    } else {
                        at(tile + self.grid[0], hx, hy + y)
                    };
                    result.copy_within(from..from + w * 3, at(tile, hx, h + hy + y));
                }
                for y in hy..hy + h {
                    if last_x {
                        for x in w + hx..self.tile[0] {
                            let from = at(tile, w + hx - 1, y);
                            result.copy_within(from..from + 3, at(tile, x, y));
                        }
                    } else {
                        let from = at(tile + 1, hx, y);
                        result.copy_within(from..from + hx * 3, at(tile, w + hx, y));
                    }
                }
                for y in 0..hy {
                    let from = if tx > 0 && ty > 0 {
                        at(tile - self.grid[0] - 1, w, h + y)
                    } else {
                        at(tile, 0, hy + 1)
                    };
                    result.copy_within(from..from + hx * 3, at(tile, 0, y));
                }
                for y in 0..hy {
                    let from = if ty > 0 && !last_x {
                        at(tile - self.grid[0] + 1, hx, h + y)
                    } else {
                        at(tile, w + hx, hy + 1)
                    };
                    result.copy_within(from..from + hx * 3, at(tile, w + hx, y));
                }
                for y in 0..hy {
                    // The native boundary case reads the second bottom halo row.
                    let from = if !last_y && !last_x {
                        at(tile + self.grid[0] + 1, hx, hy + y)
                    } else {
                        at(tile, w + hx, h + hy + 1)
                    };
                    result.copy_within(from..from + hx * 3, at(tile, w + hx, h + hy + y));
                }
                for y in 0..hy {
                    let from = if tx > 0 && !last_y {
                        at(tile + self.grid[0] - 1, w, hy + y)
                    } else {
                        at(tile, 0, h + hy - 1)
                    };
                    result.copy_within(from..from + hx * 3, at(tile, 0, h + hy + y));
                }
            }
        }
        Ok(result)
    }

    /// Match the native blend order. Each f64 product narrows before addition.
    pub fn blend(
        &self,
        tiles: &[f32],
        destination: &mut [f32],
        stride: usize,
        border: usize,
    ) -> Result<()> {
        if tiles.len() < self.sample_count()
            || stride < self.size[0] * 3
            || stride
                .checked_mul(self.size[1])
                .is_none_or(|n| n > destination.len())
            || border >= self.halo[0]
            || border >= self.halo[1]
        {
            return Err(Error::InvalidData("invalid SPP tile blend layout"));
        }
        let [w, h] = self.core;
        let [hx, hy] = self.halo;
        let step = 1.0 / (2 * border) as f64;
        let mix = |a: f32, b: f32, weight: f64| {
            (a as f64 * weight) as f32 + (b as f64 * (1.0 - weight)) as f32
        };
        for index in (0..self.grid[0] * self.grid[1]).rev() {
            let tx = index % self.grid[0];
            let ty = index / self.grid[0];
            let last_col = tx + 1 == self.grid[0];
            let last_row = ty + 1 == self.grid[1];
            let left = if tx == 0 { 0 } else { border };
            let top = if ty == 0 { 0 } else { border };
            let sx0 = hx - left;
            let sx1 = w + hx - if last_col { 0 } else { border } - 1;
            let dx0 = tx * w - left;
            let at = |x, y| index * self.samples + (y * self.tile[0] + x) * 3;
            if !last_row && border != 0 && sx0 < sx1 {
                let mut weight = 0.0;
                for k in 0..=2 * border {
                    let sy = h + hy + border - 1 - k;
                    let dy = (ty + 1) * h + border - 1 - k;
                    for sx in sx0..sx1 {
                        let to = dy * stride + (dx0 + sx - sx0) * 3;
                        for c in 0..3 {
                            destination[to + c] =
                                mix(tiles[at(sx, sy) + c], destination[to + c], weight);
                        }
                    }
                    weight += step;
                }
            }
            let sy1 = h + hy - if last_row { 0 } else { border } - 1;
            let dy1 = (ty + 1) * h - if last_row { 0 } else { border } - 1;
            for sy in (hy - top..=sy1).rev() {
                let dy = dy1 - (sy1 - sy);
                let to = dy * stride + dx0 * 3;
                let from = at(sx0, sy);
                let count = (sx1 - sx0 + 1) * 3;
                destination[to..to + count].copy_from_slice(&tiles[from..from + count]);
                if !last_col && border != 0 {
                    let mut weight = 1.0;
                    for k in 0..=2 * border {
                        let sx = sx1 + 1 + k;
                        let dx = (tx + 1) * w - border + k;
                        for c in 0..3 {
                            let to = dy * stride + dx * 3 + c;
                            destination[to] = mix(tiles[at(sx, sy) + c], destination[to], weight);
                        }
                        weight -= step;
                    }
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn copied_tiles_match_native() {
        let hash = crate::float_front::fnv1a_f32;
        let mut key = [0; 8];
        let mut source = Vec::new();
        let mut source_hash = 0;
        let mut cases = 0;
        for line in include_str!("../../tests/fixtures/spp-copy-tile-native.txt").lines() {
            if line.starts_with('#') {
                continue;
            }
            let fields: Vec<_> = line.split_whitespace().collect();
            let p: Vec<usize> = fields[1..10].iter().map(|s| s.parse().unwrap()).collect();
            let [w, h, nx, ny, hx, hy, pad, extra, index] = p[..] else {
                unreachable!()
            };
            let stride = w * nx * 3 + pad;
            if key != p[..8] {
                key.copy_from_slice(&p[..8]);
                source = vec![-765.25; stride * h * ny + 11];
                let mut state = 0x9e3779b9_u32;
                for y in 0..h * ny {
                    for x in 0..w * nx * 3 {
                        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                        source[y * stride + x] = ((state >> 8) % 72001) as f32 / 13.0 - 171.0;
                    }
                }
                source_hash = hash(&source);
            }
            let tile = [w + hx * 2 + extra, h + hy * 2 + extra];
            let tile_stride = tile[0] * 3 + pad;
            let mut destination = vec![-987.125; tile_stride * tile[1] + 11];
            copy_image_to_tile(
                &source,
                stride,
                [w * nx, h * ny],
                [nx, ny],
                [hx, hy],
                index,
                &mut destination,
                tile_stride,
                tile,
            )
            .unwrap();
            assert_eq!(
                source_hash,
                u64::from_str_radix(fields[10], 16).unwrap(),
                "{line}"
            );
            assert_eq!(
                hash(&destination),
                u64::from_str_radix(fields[11], 16).unwrap(),
                "{line}"
            );
            cases += 1;
        }
        assert_eq!(cases, 1856);
    }

    #[test]
    fn invalid_copied_tile_does_not_write() {
        let source = [1.; 48];
        let mut out = [7.; 108];
        assert!(
            copy_image_to_tile(&source, 12, [4, 4], [0, 1], [1, 1], 0, &mut out, 18, [6, 6])
                .is_err()
        );
        assert!(
            copy_image_to_tile(&source, 12, [4, 4], [1, 1], [1, 1], 1, &mut out, 18, [6, 6])
                .is_err()
        );
        assert!(
            copy_image_to_tile(&source, 12, [4, 4], [1, 1], [1, 1], 0, &mut out, 18, [6, 7])
                .is_err()
        );
        assert!(out.iter().all(|&v| v == 7.));
    }

    #[test]
    fn tiles_match_native() {
        let hash = crate::float_front::fnv1a_f32;
        for row in include_str!("../../tests/fixtures/spp-tiling-native.txt").lines() {
            if row.starts_with('#') {
                continue;
            }
            let fields: Vec<_> = row.split_whitespace().collect();
            let p: Vec<usize> = fields[1..11].iter().map(|s| s.parse().unwrap()).collect();
            let [w, h, nx, ny, hx, hy, pad, border, modify, fill] = p[..] else {
                unreachable!()
            };
            let layout = TileLayout::new([w * nx, h * ny], [nx, ny], [hx, hy]).unwrap();
            let stride = w * nx * 3 + pad;
            let mut source = vec![-765.25; stride * h * ny];
            let mut state = 0x9e3779b9_u32;
            for y in 0..h * ny {
                for x in 0..w * nx * 3 {
                    state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                    source[y * stride + x] = ((state >> 8) % 72001) as f32 / 13.0 - 171.0;
                }
            }
            assert_eq!(
                hash(&source),
                u64::from_str_radix(fields[11], 16).unwrap(),
                "source: {row}"
            );
            let mut tiles = layout.initialize(&source, stride).unwrap();
            if fill != 0 {
                // Native keeps the prior cache values in these outer corners.
                for ty in 0..ny {
                    for tx in 0..nx {
                        if ty + 1 == ny || tx + 1 == nx {
                            for y in h + hy..layout.tile[1] {
                                for x in w + hx..layout.tile[0] {
                                    let i = (ty * nx + tx) * layout.samples
                                        + (y * layout.tile[0] + x) * 3;
                                    tiles[i..i + 3].fill(123.875);
                                }
                            }
                        }
                    }
                }
            }
            assert_eq!(
                hash(&tiles),
                u64::from_str_radix(fields[12], 16).unwrap(),
                "tiles: {row}"
            );
            if modify != 0 {
                for (i, v) in tiles.iter_mut().enumerate() {
                    let tile = i / layout.samples;
                    *v = *v * (0.73 + tile as f32 * 0.137) + (i % 11) as f32 * 0.17;
                }
            }
            let mut destination = vec![-987.125; source.len()];
            layout
                .blend(&tiles, &mut destination, stride, border)
                .unwrap();
            assert_eq!(
                hash(&destination),
                u64::from_str_radix(fields[13], 16).unwrap(),
                "blend: {row}"
            );
        }
    }

    #[test]
    fn invalid_tiles_do_not_change_output() {
        for (size, grid, halo) in [
            ([16, 16], [0, 1], [1, 1]),
            ([17, 16], [2, 2], [1, 1]),
            ([16, 16], [2, 2], [9, 1]),
            ([usize::MAX, 16], [1, 1], [1, 1]),
        ] {
            assert!(TileLayout::new(size, grid, halo).is_err());
        }
        let layout = TileLayout::new([16, 16], [2, 2], [2, 2]).unwrap();
        let source = vec![1.0; 16 * 16 * 3];
        let tiles = layout.initialize(&source, 48).unwrap();
        let mut destination = source.clone();
        assert!(layout.blend(&tiles, &mut destination, 48, 2).is_err());
        assert!(layout
            .blend(&tiles[..tiles.len() - 1], &mut destination, 48, 1)
            .is_err());
        assert!(layout.blend(&tiles, &mut destination, 47, 1).is_err());
        assert_eq!(source, destination);
    }
}
