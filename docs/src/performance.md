# Performance notes

The M0 baseline matched the legacy C binary line-for-line: scalar,
single-threaded, single-image-per-invocation. M7 walked through the
loop landscape and parallelised what could be parallelised.

End-to-end wall-time on Apple Silicon, single-image:

| Sensor / format            | M0 (s) |   M7 (s) |   Speedup |
| -------------------------- | -----: | -------: | --------: |
| SD1 Merrill, DNG           |   1.90 | **0.49** | **3.88×** |
| SD1 Merrill, TIFF          |   1.65 | **0.50** | **3.30×** |
| `_SDI8040`, DNG            |   1.67 | **0.53** | **3.15×** |
| `_SDI8040`, TIFF           |   1.56 | **0.51** | **3.06×** |
| `_SDI8284` (Quattro), DNG  |   1.30 | **0.85** | **1.53×** |
| `_SDI8284` (Quattro), TIFF |   1.38 | **0.93** | **1.48×** |

Quattro sees less benefit because the full-resolution top plane
carries most of the bitstream and the other two planes (binned
5424×3616) finish much sooner — single-thread bound.

15-file batch (DNG): ~19.5 s (sequential `for`-loop over the file
list) → **4.10 s** under `rayon::par_iter`, **4.4×** speedup with
~625 % CPU saturation.

## What was parallelised

In ROI order (each row is end-to-end Apple Silicon, not isolated
kernel time):

1. **M7a — `convert_data` + `apply_highlight_clip_dng` row-parallel.**
   Both hot loops are row-independent (verified by inspection of the
   per-pixel call graph: `x3f_calc_spatial_gain`,
   `chroma_lut_apply_pixel`, `reconstruct_highlights`,
   `repair_pix_apply_pixel`, `x3f_3x3_3x1_mul`, `x3f_LUT_lookup` —
   all read shared state and write only to the per-pixel slot).
   Per-row state is bundled into a `ConvCtx` / `DngCtx` with explicit
   `unsafe impl Send + Sync`; `img.data` is split with
   `par_chunks_mut(row_stride)`. The DNG path additionally
   parallelises the global-max reduction
   (`par_chunks().map().reduce(f64::max)`) and the divide-down second
   pass. SD1M DNG: 1.90 s → 1.38 s (1.38×).
2. **M7b — Top-level batch `par_iter`.** `args.files` becomes
   `args.files.par_iter().for_each(...)`, so every input file is
   processed end-to-end on its own rayon worker. The blocker that
   motivated M7d-before-M7b was `G_DNG_HIGHLIGHT_SCALE` — a
   process-wide `static mut` written by `apply_highlight_clip_dng`
   and read by the DNG writer. Two batched files would race; fixed
   by converting to `thread_local!` and snapshotting into the `Image`
   struct on the same worker thread (M5e).
3. **M7c — `preprocess_data` + `apply_wb_color_shading` row-parallel.**
   The four preprocess hot loops (per-pixel `out = scale*(raw -
black) + bias` clamp, Quattro top16 → image[2] downsample, full-res
   top preprocess, WB color-shading rewrite) are row-independent and
   each becomes `rayon::par_chunks_mut`. SD1M DNG: 1.38 s → 1.29 s.
4. \*\*M7d — Plane-parallel TRUE entropy decode + DNG strip parallelism
   - native inlinable mat3x1 / LUT helpers.\** The big win — entropy
     decode was ~92 % of single-image wall-time on Merrill TRUE inputs
     after M7c. The three planes have *independent\* Huffman bitstreams
     (`tru.plane_address[color]`) and write to disjoint output offsets
     (color 0/1/2 stagger by 1 u16 with stride `channels`=3, or color
     2 in Quattro layout writes to its own `q.top16.data`). Plus
     per-strip `par_iter` in `encode_strips` since each strip is an
     independent zlib payload. SD1M DNG: 1.29 s → 0.49 s (1.98× this
     step / 3.88× cumulative).

`interpolate_bad_pixels` stays serial — iterative passes with
neighbour reads are not naively row-parallel.

## What was _not_ SIMD'd

The M7 plan listed SIMD via the `wide` crate as the third lever after
row-parallelism and planar layout. With the `mat3x1` mul and gamma
LUT lookup inlined (M7d step 2), the auto-vectoriser's output is
already FMA-dense on aarch64, and explicit `f64x2` SIMD would force a
precision-preserving rewrite to keep MD5 parity (any FP reorder
breaks byte-identity). Projected gain is small relative to the
entropy-decode-bound regime we're now in. SIMD remains an option if
profiling on a future input shows the matrix kernel re-bottlenecking.

The planar-layout migration the M7 plan called for was also deferred.
The interleaved RGBRGB layout is genuinely worse cache-wise for the
channel-at-a-time highlight passes, but the cleanup-only nature
(no-op for tier-2 MD5; no-op for tier-3 ΔE) means it can land in any
post-port milestone without churning correctness gates.

## WASM perf

WASM is single-threaded by default in v1. `wasm-bindgen-rayon`
requires COOP/COEP HTTP headers downstream and is not worth forcing
on consumers. The full DNG pipeline runs on `wasm32-wasip1` under
wasmtime today; no benchmarks are committed yet.

## How to bench locally

```sh
# wall-time on a single file
time target/release/x3f_extract -dng input.X3F

# batch wall-time
time target/release/x3f_extract -dng *.X3F

# rayon thread count
RAYON_NUM_THREADS=1 time target/release/x3f_extract -dng input.X3F
RAYON_NUM_THREADS=8 time target/release/x3f_extract -dng input.X3F

# quick CPU saturation snapshot (macOS)
sudo dtruss -c target/release/x3f_extract -dng input.X3F 2>&1 | tail
```

Criterion benches and `cargo flamegraph` snapshots before/after each
M7 step are _not_ committed today — the wall-time deltas above came
from `time` runs against the
[`temp/highlight_recov_test/`](https://github.com/sagwaco/x3fuse-core/tree/master/temp)
corpus. Adding committed benches is open follow-up work.
