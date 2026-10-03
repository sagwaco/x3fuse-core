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
     independent payload. SD1M DNG: 1.29 s → 0.49 s (1.98× this
     step / 3.88× cumulative). (The strip parallelism now only helps
     the uncompressed path: `-compress` DNGs are a single full-height
     lossless-JPEG strip — the layout the dcraw-lineage decoders
     require — at roughly +0.3 s wall per file.)

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

## GPU denoising

The first acceleration update keeps the existing NLM algorithm and sensor
strengths. Native CLI builds enable the `gpu` feature by default. Library
consumers opt in with `x3f-core/gpu`; CPU-only CLI builds use
`cargo build --release -p x3f-cli --no-default-features`.

`-denoise-backend auto` selects a hardware compute adapter through wgpu 27
(Metal, Vulkan, or DX12). `cpu` avoids GPU initialization entirely. `gpu`
requires acceleration, so benchmarks cannot silently measure CPU fallback.
Forced-GPU failures reach Rust callers as `x3f_core::Error::Gpu`, including
conversion, scene-linear preparation, and histogram processing.
`-v` logs the adapter, each NLM call, and the full denoising stages. Settings
are local to each conversion, including the scene-linear and histogram APIs.

The implementation retains 16-bit samples, the fixed-point weight quantization,
both NLM windows, and the existing chroma/luma treatment. GPU weighted sums use
FP32. Subgroup shuffles share neighbouring patch distances when available; a
shared-memory shader handles other compute adapters and is retried if initial
pipeline creation fails. No FP16, floating-point
atomics, CUDA runtime, or downloaded model is required. Median and resize rows
also run in parallel on native CPUs without reordering each pixel's arithmetic.

GPU input/output/readback buffers are reused across 1024×1024 tiles, with
global-image `BORDER_REFLECT_101` halos. Fixed GPU buffers total about 26 MiB;
full-image CPU working buffers are additional. One shared device serializes GPU
calls to bound scratch memory in batch conversions. Cancellation is checked
between tiles and while waiting for readback or the device lock, including
waiting behind another caller's initialization. The initializing call itself
cannot interrupt a synchronous driver operation; it checks cancellation again
as soon as initialization returns. The original
stage input remains unchanged until success; automatic fallback retries that
input on CPU. A lost/failed device is disabled for the rest of the process.
The serialization bounds scratch memory; it is not a batch-throughput guarantee
on every GPU. `-denoise-backend cpu` remains available for such comparisons.

### Measured latency

Measured on 2026-10-03: Apple M1 Max, 64 GiB, macOS 27.0, Rust 1.98.1 release
build. Each row is 30 fresh CLI processes at strength 10, writing uncompressed
DNG to temporary storage through the universal binary embedded in X3Fuse.
Denoising includes all NLM passes, CPU median/resize/refinement, GPU device and
pipeline creation, upload, and readback. Quattro timings sum its pre-upsample
and post-upsample denoising stages. OS filesystem and driver shader caches
may be warm; this is not a claim about first launch after installation.

| Sample / raw raster | Backend | Median denoising | p95 denoising | Median full conversion | First full conversion |
| --- | --- | ---: | ---: | ---: | ---: |
| DP2 Merrill, 4704×3136 | CPU | 1689 ms | 1774 ms | 2237 ms | 2569 ms |
| DP2 Merrill, 4704×3136 | Metal GPU | 385 ms | 416 ms | 914 ms | 1081 ms |
| DP0 Quattro, 5424×3616 | CPU | 2200 ms | 2270 ms | 3020 ms | 3011 ms |
| DP0 Quattro, 5424×3616 | Metal GPU | 400 ms | 434 ms | 1206 ms | 1236 ms |

GPU denoising is 4.4× and 5.5× faster than the CPU backend in the same build;
complete conversion is about 2.4–2.5× faster. The provisional 500 ms p95
denoising target is met on these two images and this GPU. This does not
establish a bound across all images, resolutions, GPUs, or power modes.

A separate first conversion from the newly ad-hoc-signed app bundle took
595 ms of Quattro denoising. The table's p95 result should therefore be read
with its OS/driver cache conditions, not as a cold-start guarantee. The
universal Release app built successfully with local signing, and its bundled
converter completed a forced-Metal conversion.

### Correctness checks

Both WGSL kernels are validated without a GPU in ordinary CI. Physical-GPU
tests cover single rows/columns, odd sizes, image borders, both tile seams,
both search windows, repeatability, and cancellation followed by device reuse:

```sh
cargo test --release -p x3f-sys --features gpu denoise:: -- --include-ignored
cargo test -p x3f-sys --no-default-features denoise::
# CPU-only public API error propagation and histogram compatibility (real RAWs):
X3F_TEST_FILES=/path/to/corpus cargo test --release -p x3f-core --no-default-features --test conversion
```

The portable kernel crosses both horizontal and vertical tile seams. Tests
also cover CPU/GPU requests running concurrently, an uncaptured validation
error followed by automatic CPU fallback, failed/poisoned contexts, and
cancellation while waiting for initialization. CPU loop cancellation is
triggered after 64 output pixels rather than inferred from a timer. These do
not simulate every driver failure: actual device loss and GPU hangs still
require platform-specific validation. Native histograms retain their legacy
behavior and ignore DNG highlight/TIFF Cineon options; histogram dumping is
unsupported on WASM.

Kernel output must be within one 16-bit code value of the CPU reference.
Downstream matrix and tone processing can amplify that difference. On the
local `DP2M0981.X3F` and `DP0Q0010.X3F` samples, CPU output and denoising-disabled
output matched the pre-update binary (`00d326f`) byte for byte for DNG and TIFF.
GPU versus CPU raster differences at strength 10 were:

| Sample | Output | Maximum absolute difference (16-bit codes) | RMSE (16-bit codes) | PSNR against CPU |
| --- | --- | ---: | ---: | ---: |
| DP2 Merrill | DNG | 18 | 0.292 | 107.01 dB |
| DP0 Quattro | DNG | 5 | 0.201 | 110.27 dB |
| DP2 Merrill | sRGB TIFF | 575 | 8.165 | 78.09 dB |
| DP0 Quattro | sRGB TIFF | 438 | 10.084 | 76.26 dB |

These measure agreement with the CPU implementation, not denoising quality
against a clean reference. Metal was exercised on an Apple M1 Max; Vulkan,
DX12, Intel/AMD GPUs, and older Foveon sensors still need equivalent hardware
and image validation. The Intel binary slice was cross-compiled and checked;
Rosetta is unavailable on the test machine. The broader historical corpus is
not available locally.

### BM3D decision and follow-up plan

BM3D remains a quality candidate; it is not implemented or benchmarked in this
update. The speed work first accelerates the existing algorithm so its sensor
calibration and CPU fallback remain useful. The [C-BM3D paper](https://webpages.tuni.fi/foi/GCF-BM3D/CBM3D_ICIP_2007.pdf)
describes shared luminance-guided block matching for chroma and two transform
filtering stages. That suggests a possible fit for Foveon chroma noise, but
does not establish a speed advantage for this pipeline.

Before adopting BM3D:

1. Collect dark/flat fields and detail-rich scenes across Merrill, Quattro,
   older sensors, and ISO settings. Estimate noise in the actual sensor/YUV
   domain. Existing NLM `h` values are filter strengths, not calibrated BM3D
   noise standard deviations. Preserve the Quattro top-layer/luma behavior.
2. Compare a CPU reference on representative crops, retaining both
   hard-threshold and Wiener stages. The MIT-licensed
   [VapourSynth-BM3D implementation](https://github.com/HomeOfVapourSynthEvolution/VapourSynth-BM3D)
   is a reference candidate, not a new runtime dependency. Check residual
   chroma noise, fine texture, colored edges, clipping, and tile seams;
   synthetic noise scores alone do not validate real Foveon noise.
3. Only after a measured quality benefit, prototype tiled compute kernels
   using the current wgpu dependency. Measure block matching, both filtering
   stages, aggregation, transfers, initialization, and memory. The existing
   [CUDA gpu-bm3d project](https://github.com/JeffOwOSun/gpu-bm3d) can inform a
   comparison on NVIDIA hardware; it does not cover the macOS Metal target.
4. Add a user-selectable method only after CPU/GPU quality and latency gates
   pass. Keep NLM available and leave default selection unchanged until the
   new method is validated. A provisional target is p95 denoising at or below
   500 ms for a 15–25 MP image on the reference GPU; this is interactive still
   processing, not a 30/60 fps video guarantee.

## WASM perf

WASM is single-threaded by default in v1. `wasm-bindgen-rayon`
requires COOP/COEP HTTP headers downstream and is not worth forcing
on consumers. The full DNG pipeline runs on `wasm32-wasip1` under
wasmtime today; no benchmarks are committed yet.

GPU dependencies are excluded from all `wasm32-*` targets even with the `gpu`
feature enabled. `Auto` uses CPU; forced `Gpu` returns an error when a denoising
stage is requested. CI runs the portable denoiser under wasmtime and checks
that WASM artifacts have no unresolved `env` imports. Browser WebGPU would
require a separate asynchronous host integration and is deferred.

## How to bench locally

```sh
# wall-time on a single file
time target/release/x3f_extract -dng input.X3F

# batch wall-time
time target/release/x3f_extract -dng *.X3F

# Fresh-process CPU/GPU comparison; outputs are temporary, results are CSV.
# Includes GPU initialization/transfers and all denoising stages.
python3 scripts/bench_denoise.py --iterations 30 input.X3F

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
