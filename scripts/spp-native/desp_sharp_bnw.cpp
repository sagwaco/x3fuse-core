// Differential harness for the monochrome sharpening kernels of the Merrill
// engine:
//   - Despeckle3 (Flt32Mono <- Flt32RGB, row range) at 0x2ed50
//   - FPixelProc2_F20::SharpenBnW (16SMono <- Flt32Mono, row range) at 0x81768
//   - DoDespSharp (0x2fb7c), the MonochromeDenoise worker that runs both
//   - FPixelProc2_F20::ApplySharpLBnW (Flt32RGB += 16SMono, Stage2B) at 0x810b0
//
// Calls the native functions on deterministic synthetic images and prints one
// fixture line per case: every input as an exact parameter (floats and doubles
// as hex bits, images as seed + pattern of the generator below) and the FNV-1a
// hash of every output buffer, including the rows, columns and padding a call
// must leave untouched. The Rust port in
// crates/x3f-sys/src/spp_denoise/mono/sharpen_bnw.rs
// regenerates the same inputs and must match every hash bit for bit.
//
// Build and run (arm64 macOS):
//   clang++ -std=c++17 -arch arm64 -O1 -ffp-contract=off \
//     scripts/spp-native/desp_sharp_bnw.cpp -o /tmp/desp_sharp_bnw
//   /tmp/desp_sharp_bnw reference/merrill/lib99_MerillDLLForMac.arm64.dylib \
//     > crates/x3f-sys/tests/fixtures/spp-mono-desp-sharp-native.txt
//
// Nothing printed here comes from Sigma: inputs are synthetic, outputs are
// hashes.

#include "pixmap.h"

#include <cinttypes>
#include <cmath>
#include <limits>
#include <vector>

namespace {

using Despeckle3 = void (*)(merrill::Pixmap *, merrill::Pixmap *, unsigned, unsigned, float,
                            float);
using SharpenBnW = void (*)(merrill::Pixmap *, merrill::Pixmap *, unsigned, unsigned, float,
                            float);
using ApplySharpLBnW = void (*)(merrill::Pixmap *, merrill::Pixmap *, double);
using DoDespSharp = void (*)(void *);

const char *kDespeckle =
    "_Z10Despeckle3P19FPixmapTemplate_F20I24FPixelClassFlt32Mono_F20EPS_I23FPixelClassFlt32RGB_"
    "F20Ejjff";
const char *kSharpen =
    "_ZN15FPixelProc2_F2010SharpenBnWEP19FPixmapTemplate_F20I24FPixelClassFlt32Mono_F20EPS0_"
    "I22FPixelClass16SMono_F20Ejjff";
const char *kApply =
    "_ZN15FPixelProc2_F2014ApplySharpLBnWEP19FPixmapTemplate_F20I23FPixelClassFlt32RGB_F20EPS0_"
    "I22FPixelClass16SMono_F20Ed";
const char *kWorker = "_Z11DoDespSharpPv";

// Vtables (address points): Flt32Mono, Flt32RGB, 16SMono (_ZTV... + 0x10).
const uintptr_t kMonoVtable = 0xd0c20;
const uintptr_t kRgbVtable = 0xd0ca0;
const uintptr_t kS16Vtable = 0xd0d28;

const int16_t kS16Sentinel = 0x7e57;

uint32_t mix32(uint32_t x) {
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  x ^= x >> 16;
  return x;
}

float from_bits(uint32_t bits) {
  float f;
  memcpy(&f, &bits, sizeof f);
  return f;
}
uint32_t bits_of(float f) {
  uint32_t b;
  memcpy(&b, &f, sizeof b);
  return b;
}
uint64_t bits_of(double d) {
  uint64_t b;
  memcpy(&b, &d, sizeof b);
  return b;
}

// Float generator shared with the Rust test. Every value is exact (integer
// scaled by a power of two), so no rounding is involved.
//   0 wide:      0 .. 16384 in 1/256 steps
//   1 small:     -0.25 .. 1.75 in 2^-11 steps
//   2 plateau:   1000 / 1000.5 / 1001 with isolated bright and dark speckles
//   3 ceiling:   7000 .. 16500 plus 8191, 16383, 32767 and neighbors
//   4 gradient:  ramp along the buffer plus tiny noise
//   5 specials:  wide values, 1/16 negative, plus inf, NaN, -0, subnormal, max
//   6 small plateau: 0.5 with speckles near zero (despeckle threshold scale)
float pixel(uint32_t seed, uint32_t index, uint32_t pattern) {
  uint32_t h = mix32(seed * 0x9e3779b9u ^ index);
  uint32_t h2 = mix32(h ^ 0x5bd1e995u);
  uint32_t r = h2 >> 24;
  switch (pattern) {
    case 0:
      return static_cast<float>((h % 16384u) * 256u + (h2 & 255u)) * 0.00390625f;
    case 1:
      return static_cast<float>(static_cast<int32_t>(h % 4096u) - 512) * 0.00048828125f;
    case 2:
      if (r < 10) return static_cast<float>(1000u + h % 8192u);
      if (r < 20) return static_cast<float>(h % 1000u) * 0.5f;
      return 1000.0f + static_cast<float>(h2 % 3u) * 0.5f;
    case 3:
      if (h2 % 4u == 0) {
        static const float specials[] = {8191.0f,  8191.5f,  16383.0f, 16383.75f,
                                         32767.0f, 32766.5f, 0.0f,     8190.5f};
        return specials[(h >> 8) % 8u];
      }
      return static_cast<float>(7000u + h % 9500u);
    case 4:
      return static_cast<float>((index * 37u) % 16384u) +
             static_cast<float>(h % 16u) * 0.0625f;
    case 5: {
      if (r % 8u == 0) {
        static const uint32_t specials[] = {0x7f800000u, 0xff800000u, 0x7fc00000u, 0x80000000u,
                                            0x00000005u, 0x46fffe00u, 0x46fffe01u, 0x7f7fffffu};
        return from_bits(specials[h2 % 8u]);
      }
      float v = static_cast<float>((h % 16384u) * 256u + (h2 & 255u)) * 0.00390625f;
      return (h2 & 15u) == 0 ? -v : v;
    }
    default:
      if (r < 16) return 0.5f + static_cast<float>(h % 1024u) * 0.0009765625f;
      if (r < 32) return static_cast<float>(h % 256u) * 0.0009765625f;
      return 0.5f + static_cast<float>(h2 % 3u) * 0.000244140625f;
  }
}

// 16-bit sharpness generator: 0 full range, 1 realistic (+-6000), 2 small.
int16_t sharp(uint32_t seed, uint32_t index, uint32_t pattern) {
  uint32_t h = mix32(seed * 0x85ebca6bu ^ index ^ 0x27d4eb2fu);
  switch (pattern) {
    case 0:
      return static_cast<int16_t>(h & 0xffffu);
    case 1:
      return static_cast<int16_t>(static_cast<int32_t>(h % 12001u) - 6000);
    default:
      return static_cast<int16_t>(static_cast<int32_t>(h % 41u) - 20);
  }
}

// Initial mono contents for the despeckle output: 0 sentinel ramp (shows
// every write), 1 copy of RGB channel 2 as MonochromeDenoise prepares it.
std::vector<float> mono_init(const std::vector<float> &rgb, uint32_t w, uint32_t h,
                             uint32_t rgb_stride, uint32_t mono_stride, uint32_t init) {
  std::vector<float> mono(static_cast<size_t>(mono_stride) * h + 2);
  for (size_t i = 0; i < mono.size(); i++) mono[i] = -0.5f * static_cast<float>(1 + i % 97);
  if (init == 1) {
    for (uint32_t y = 0; y < h; y++)
      for (uint32_t x = 0; x < w; x++)
        mono[static_cast<size_t>(y) * mono_stride + x] =
            rgb[static_cast<size_t>(y) * rgb_stride + 3 * x + 2];
  }
  return mono;
}

std::vector<float> floats(uint32_t len, uint32_t seed, uint32_t pattern) {
  std::vector<float> v(len);
  for (uint32_t i = 0; i < len; i++) v[i] = pixel(seed, i, pattern);
  return v;
}

uint64_t hash(const std::vector<float> &v) {
  return merrill::fnv1a(v.data(), v.size() * sizeof(float));
}
uint64_t hash(const std::vector<int16_t> &v) {
  return merrill::fnv1a(v.data(), v.size() * sizeof(int16_t));
}

struct Native {
  merrill::Library lib;
  Despeckle3 despeckle = nullptr;
  SharpenBnW sharpen = nullptr;
  ApplySharpLBnW apply = nullptr;
  DoDespSharp worker = nullptr;

  merrill::Pixmap pixmap(uintptr_t vtable, void *data, uint32_t w, uint32_t h,
                         uint32_t row_bytes) {
    return merrill::Pixmap{lib.at(vtable), data, w, h, static_cast<int32_t>(row_bytes), 0};
  }
};

// DS name w h rgb_stride mono_stride first last seed pattern init floor level | hash
void despeckle_case(Native &n, const char *name, uint32_t w, uint32_t h, uint32_t rgb_pad,
                    uint32_t mono_pad, uint32_t first, uint32_t last, uint32_t seed,
                    uint32_t pattern, uint32_t init, float floor_, float level) {
  uint32_t rs = 3 * w + rgb_pad, ms = w + mono_pad;
  auto rgb = floats(rs * h + 3, seed, pattern);
  auto mono = mono_init(rgb, w, h, rs, ms, init);
  auto rgb_px = n.pixmap(kRgbVtable, rgb.data(), w, h, rs * 4);
  auto mono_px = n.pixmap(kMonoVtable, mono.data(), w, h, ms * 4);
  n.despeckle(&mono_px, &rgb_px, first, last, floor_, level);
  printf("DS %s %u %u %u %u %u %u %u %u %u %08x %08x | %016" PRIx64 "\n", name, w, h, rs, ms,
         first, last, seed, pattern, init, bits_of(floor_), bits_of(level), hash(mono));
}

// SB name w h mono_stride sl_stride first last seed pattern amount iso | hash
void sharpen_case(Native &n, const char *name, uint32_t w, uint32_t h, uint32_t mono_pad,
                  uint32_t sl_pad, uint32_t first, uint32_t last, uint32_t seed, uint32_t pattern,
                  float amount, float iso) {
  uint32_t ms = w + mono_pad, ss = w + sl_pad;
  auto mono = floats(ms * h + 2, seed, pattern);
  std::vector<int16_t> sl(static_cast<size_t>(ss) * h + 1, kS16Sentinel);
  auto mono_px = n.pixmap(kMonoVtable, mono.data(), w, h, ms * 4);
  auto sl_px = n.pixmap(kS16Vtable, sl.data(), w, h, ss * 2);
  n.sharpen(&mono_px, &sl_px, first, last, amount, iso);
  printf("SB %s %u %u %u %u %u %u %u %u %08x %08x | %016" PRIx64 "\n", name, w, h, ms, ss,
         first, last, seed, pattern, bits_of(amount), bits_of(iso), hash(sl));
}

// DD name w h rgb_stride mono_stride sl_stride first last seed pattern amount iso
//   | mono_hash sl_hash   (native DoDespSharp, mono initialized from RGB ch2)
void worker_case(Native &n, const char *name, uint32_t w, uint32_t h, uint32_t pad,
                 uint32_t first, uint32_t last, uint32_t seed, uint32_t pattern, float amount,
                 float iso) {
  uint32_t rs = 3 * w + pad, ms = w + pad, ss = w + pad;
  auto rgb = floats(rs * h + 3, seed, pattern);
  auto mono = mono_init(rgb, w, h, rs, ms, 1);
  std::vector<int16_t> sl(static_cast<size_t>(ss) * h + 1, kS16Sentinel);
  auto rgb_px = n.pixmap(kRgbVtable, rgb.data(), w, h, rs * 4);
  auto mono_px = n.pixmap(kMonoVtable, mono.data(), w, h, ms * 4);
  auto sl_px = n.pixmap(kS16Vtable, sl.data(), w, h, ss * 2);

  // NoiseParams: +0x04 level / ISO, +0x44 despeckle floor / sharpen amount.
  alignas(16) uint8_t noise[0x100] = {};
  memcpy(noise + 0x4, &iso, 4);
  memcpy(noise + 0x44, &amount, 4);
  // MonochromeDenoise context: +0x38 16SMono plane, +0x98 NoiseParams.
  alignas(16) uint8_t ctx[0x100] = {};
  void *sl_ptr = &sl_px, *noise_ptr = noise;
  memcpy(ctx + 0x38, &sl_ptr, 8);
  memcpy(ctx + 0x98, &noise_ptr, 8);
  // Worker argument: +0x08 mono, +0x10 rgb, +0x18 first, +0x1c last, +0x40 ctx.
  alignas(16) uint8_t arg[0x60] = {};
  void *mono_ptr = &mono_px, *rgb_ptr = &rgb_px, *ctx_ptr = ctx;
  memcpy(arg + 0x8, &mono_ptr, 8);
  memcpy(arg + 0x10, &rgb_ptr, 8);
  memcpy(arg + 0x18, &first, 4);
  memcpy(arg + 0x1c, &last, 4);
  memcpy(arg + 0x40, &ctx_ptr, 8);
  n.worker(arg);
  printf("DD %s %u %u %u %u %u %u %u %u %u %08x %08x | %016" PRIx64 " %016" PRIx64 "\n", name,
         w, h, rs, ms, ss, first, last, seed, pattern, bits_of(amount), bits_of(iso), hash(mono),
         hash(sl));
}

// AS name w h rgb_stride sl_stride seed pattern sl_pattern amount | hash
void apply_case(Native &n, const char *name, uint32_t w, uint32_t h, uint32_t rgb_pad,
                uint32_t sl_pad, uint32_t seed, uint32_t pattern, uint32_t sl_pattern,
                double amount) {
  uint32_t rs = 3 * w + rgb_pad, ss = w + sl_pad;
  auto rgb = floats(rs * h + 3, seed, pattern);
  std::vector<int16_t> sl(static_cast<size_t>(ss) * h + 1);
  for (size_t i = 0; i < sl.size(); i++) sl[i] = sharp(seed, static_cast<uint32_t>(i), sl_pattern);
  auto rgb_px = n.pixmap(kRgbVtable, rgb.data(), w, h, rs * 4);
  auto sl_px = n.pixmap(kS16Vtable, sl.data(), w, h, ss * 2);
  n.apply(&rgb_px, &sl_px, amount);
  printf("AS %s %u %u %u %u %u %u %u %016" PRIx64 " | %016" PRIx64 "\n", name, w, h, rs, ss,
         seed, pattern, sl_pattern, bits_of(amount), hash(rgb));
}

bool check(const merrill::Library &lib, void *fn, uintptr_t offset, const char *what) {
  if (!fn || reinterpret_cast<uintptr_t>(fn) - lib.base != offset) {
    fprintf(stderr, "%s not at 0x%lx\n", what, static_cast<unsigned long>(offset));
    return false;
  }
  return true;
}

}  // namespace

int main(int argc, char **argv) {
  const char *path = argc > 1 ? argv[1] : "reference/merrill/lib99_MerillDLLForMac.arm64.dylib";
  Native n;
  if (!n.lib.open(path)) return 1;
  n.despeckle = n.lib.sym<Despeckle3>(kDespeckle);
  n.sharpen = n.lib.sym<SharpenBnW>(kSharpen);
  n.apply = n.lib.sym<ApplySharpLBnW>(kApply);
  n.worker = n.lib.sym<DoDespSharp>(kWorker);
  if (!check(n.lib, reinterpret_cast<void *>(n.despeckle), 0x2ed50, "Despeckle3") ||
      !check(n.lib, reinterpret_cast<void *>(n.sharpen), 0x81768, "SharpenBnW") ||
      !check(n.lib, reinterpret_cast<void *>(n.apply), 0x810b0, "ApplySharpLBnW") ||
      !check(n.lib, reinterpret_cast<void *>(n.worker), 0x2fb7c, "DoDespSharp"))
    return 1;

  const float nanf = std::numeric_limits<float>::quiet_NaN();
  const float inff = std::numeric_limits<float>::infinity();

  printf("# Merrill monochrome despeckle + sharpen kernels, native outputs:\n");
  printf("#   DS Despeckle3 (Flt32Mono <- Flt32RGB ch2) 0x2ed50\n");
  printf("#   SB FPixelProc2_F20::SharpenBnW (16SMono <- Flt32Mono) 0x81768\n");
  printf("#   DD DoDespSharp worker 0x2fb7c (Despeckle3 then SharpenBnW)\n");
  printf("#   AS FPixelProc2_F20::ApplySharpLBnW (Flt32RGB += 16SMono) 0x810b0\n");
  printf("# Generated by scripts/spp-native/desp_sharp_bnw.cpp against the arm64 slice of\n");
  printf("# lib99_MerillDLLForMac.dylib; inputs are synthetic (seed/pattern generators in\n");
  printf("# that file and in spp_denoise/mono/sharpen_bnw.rs). Strides in elements, rows\n");
  printf("# as passed to the native (inclusive band). Floats/doubles as hex bits; hashes are\n");
  printf("# FNV-1a 64 over the whole output buffer.\n");
  printf("# DS name w h rgb_stride mono_stride first last seed pattern init floor level | hash\n");
  printf("# SB name w h mono_stride sl_stride first last seed pattern amount iso | hash\n");
  printf("# DD name w h rgb_stride mono_stride sl_stride first last seed pattern amount iso"
         " | mono_hash sl_hash\n");
  printf("# AS name w h rgb_stride sl_stride seed pattern sl_pattern amount | hash\n");

  uint32_t seed = 1;

  // --- Despeckle3 ---------------------------------------------------------
  const uint32_t dwidths[] = {5, 6, 7, 8, 9, 13, 64, 257};
  const uint32_t dheights[] = {4, 5, 6, 9};
  const uint32_t dpatterns[] = {0, 2, 6, 3};
  for (uint32_t w : dwidths)
    for (uint32_t h : dheights) {
      uint32_t p = dpatterns[seed % 4];
      despeckle_case(n, "full", w, h, 0, 0, 0, h - 1, seed++, p, 0, 0.0f, 100.0f);
      despeckle_case(n, "pad", w, h, 4, 3, 0, h - 1, seed++, 2, 1, 1.0f, 800.0f);
    }
  // Realistic ISO levels and floors, every pattern.
  const float levels[] = {50.0f, 100.0f, 200.0f, 400.0f, 800.0f, 1600.0f, 3200.0f, 6400.0f,
                          12800.0f};
  const float floors[] = {0.0f, 0.5f, 1.0f, 3.0f, 64.0f, 100.0f};
  for (float lv : levels)
    for (float fl : floors) {
      despeckle_case(n, "level", 23, 11, 1, 2, 0, 10, seed++, 6, 0, fl, lv);
      despeckle_case(n, "level_wide", 23, 11, 0, 0, 0, 10, seed++, 2, 1, fl, lv);
    }
  for (uint32_t p = 0; p < 7; p++) {
    despeckle_case(n, "pattern", 31, 17, 2, 1, 0, 16, seed++, p, 0, 0.5f, 400.0f);
    despeckle_case(n, "pattern_copy", 31, 17, 0, 0, 0, 16, seed++, p, 1, 2.0f, 1600.0f);
  }
  // Row bands as DoThreadedPix splits them (first, next_first - 1), plus edges.
  const uint32_t bands[][2] = {{0, 0},  {0, 3},  {0, 4},   {1, 5},  {3, 7},   {4, 9},
                               {5, 9},  {6, 6},  {10, 19}, {15, 19}, {19, 19}, {18, 40},
                               {9, 3},  {0, 100}, {25, 30}};
  for (auto &b : bands) {
    despeckle_case(n, "band", 19, 20, 1, 1, b[0], b[1], seed++, 2, 0, 0.5f, 200.0f);
  }
  despeckle_case(n, "band_hi", 19, 20, 0, 0, 0x7fffffffu, 0xffffffffu, seed++, 2, 0, 0.5f, 200.0f);
  despeckle_case(n, "band_neg", 19, 20, 0, 0, 0xfffffff0u, 5, seed++, 2, 0, 0.5f, 200.0f);
  despeckle_case(n, "band_wrap", 19, 20, 0, 0, 0, 0x7ffffffeu, seed++, 2, 0, 0.5f, 200.0f);
  // Odd parameters: negative / zero / huge / NaN thresholds.
  const float odd[][2] = {{-1.0f, -500.0f}, {0.0f, 0.0f},   {-64.0f, 100.0f}, {1e30f, 100.0f},
                          {0.0f, 1e30f},    {nanf, 100.0f}, {0.5f, nanf},     {inff, 100.0f},
                          {0.0f, -inff},    {6400.0f, 0.0f}};
  for (auto &o : odd) {
    despeckle_case(n, "odd", 17, 9, 0, 0, 0, 8, seed++, 2, 0, o[0], o[1]);
    despeckle_case(n, "odd_small", 17, 9, 0, 0, 0, 8, seed++, 6, 0, o[0], o[1]);
  }
  despeckle_case(n, "large", 301, 37, 5, 7, 0, 36, seed++, 0, 1, 1.0f, 400.0f);
  despeckle_case(n, "large_plateau", 301, 37, 0, 0, 3, 30, seed++, 2, 0, 0.0f, 3200.0f);

  // --- SharpenBnW ---------------------------------------------------------
  const uint32_t swidths[] = {2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 64, 257};
  const uint32_t sheights[] = {2, 3, 4, 7};
  for (uint32_t w : swidths)
    for (uint32_t h : sheights) {
      uint32_t p = seed % 5;
      sharpen_case(n, "full", w, h, 0, 0, 0, h - 1, seed++, p, 1.0f, 100.0f);
      sharpen_case(n, "pad", w, h, 3, 2, 0, h - 1, seed++, 0, 1.5f, 160.0f);
    }
  // Limit table: every interval, its edges and outside.
  const float isos[] = {0.0f,   50.0f,  99.9f,  100.0f, 100.5f, 110.0f, 124.9f,
                        125.0f, 140.0f, 159.9f, 160.0f, 175.0f, 199.99f, 200.0f,
                        400.0f, 6400.0f, -50.0f, inff,   -inff};
  const float amounts[] = {0.0f, 0.25f, 1.0f, 1.37f, 3.0f};
  for (float iso : isos)
    for (float a : amounts) {
      sharpen_case(n, "limit", 21, 6, 1, 3, 0, 5, seed++, 0, a, iso);
    }
  for (float iso : isos) sharpen_case(n, "limit_ramp", 21, 6, 0, 0, 0, 5, seed++, 4, 2.0f, iso);
  for (uint32_t p = 0; p < 7; p++) {
    sharpen_case(n, "pattern", 33, 9, 2, 1, 0, 8, seed++, p, 1.0f, 140.0f);
    sharpen_case(n, "pattern_neg", 33, 9, 0, 0, 0, 8, seed++, p, -2.5f, 100.0f);
    sharpen_case(n, "pattern_big", 33, 9, 0, 0, 0, 8, seed++, p, 64.0f, 200.0f);
  }
  const float samounts[] = {-0.0f, 1e-30f, 100.0f, -100.0f, 1e30f, nanf, inff, -inff};
  for (float a : samounts) sharpen_case(n, "amount", 19, 7, 0, 0, 0, 6, seed++, 3, a, 125.0f);
  for (auto &b : bands) {
    sharpen_case(n, "band", 19, 20, 1, 1, b[0], b[1], seed++, 0, 1.0f, 150.0f);
  }
  sharpen_case(n, "band_hi", 19, 20, 0, 0, 0x7fffffffu, 0xffffffffu, seed++, 0, 1.0f, 150.0f);
  sharpen_case(n, "band_neg", 19, 20, 0, 0, 0xfffffff0u, 5, seed++, 0, 1.0f, 150.0f);
  sharpen_case(n, "band_wrap", 19, 20, 0, 0, 0, 0x7ffffffeu, seed++, 0, 1.0f, 150.0f);
  sharpen_case(n, "large", 301, 37, 5, 7, 0, 36, seed++, 0, 1.2f, 180.0f);
  sharpen_case(n, "large_band", 301, 37, 0, 0, 12, 23, seed++, 4, 0.8f, 110.0f);

  // --- DoDespSharp (worker) -----------------------------------------------
  const float wamounts[] = {0.0f, 0.5f, 1.0f, 1.5f};
  const float wisos[] = {100.0f, 133.0f, 400.0f, 3200.0f};
  for (float a : wamounts)
    for (float iso : wisos) {
      uint32_t p = seed % 7;
      worker_case(n, "worker", 29, 13, 1, 0, 12, seed++, p, a, iso);
    }
  for (auto &b : bands) worker_case(n, "worker_band", 19, 20, 2, b[0], b[1], seed++, 2, 1.0f, 200.0f);
  worker_case(n, "worker_large", 257, 41, 3, 5, 30, seed++, 0, 1.0f, 800.0f);

  // --- ApplySharpLBnW -----------------------------------------------------
  const uint32_t awidths[] = {1, 2, 3, 7, 8, 9, 15, 16, 17, 33, 257};
  const uint32_t aheights[] = {1, 2, 5};
  for (uint32_t w : awidths)
    for (uint32_t h : aheights) {
      uint32_t p = seed % 7;
      apply_case(n, "full", w, h, 0, 0, seed++, p, 1, 1.0);
      apply_case(n, "pad", w, h, 5, 3, seed++, 3, 0, 0.75);
    }
  const double aamounts[] = {0.0,  -0.0, 0.25, 0.5,   1.0,  1.37, 2.0,  4.0,
                             -0.8, 1e-300, 1e300, -1e300};
  for (double a : aamounts)
    for (uint32_t sp = 0; sp < 3; sp++) apply_case(n, "amount", 19, 4, 1, 1, seed++, 3, sp, a);
  for (uint32_t p = 0; p < 7; p++) apply_case(n, "pattern", 23, 5, 0, 2, seed++, p, 0, 1.25);
  apply_case(n, "nan_amount", 9, 3, 0, 0, seed++, 0, 1, std::numeric_limits<double>::quiet_NaN());
  apply_case(n, "inf_amount", 9, 3, 0, 0, seed++, 0, 1, std::numeric_limits<double>::infinity());
  apply_case(n, "large", 301, 37, 5, 7, seed++, 0, 1, 1.1);
  return 0;
}
