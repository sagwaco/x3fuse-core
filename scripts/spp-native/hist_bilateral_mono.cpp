// Differential harness for FPixelProc2_F20::ApplyHistBilateralFilter, the
// Flt32Mono -> Flt32RGB overload at 0x3301c (the kernel of DoFastBilateralFilter
// inside MonochromeDenoise).
//
// Calls the native function on deterministic synthetic images and prints one
// fixture line per case: every input as an exact parameter (images as seed +
// pattern of the generator below, the noise scaling as f64 bits) and the FNV-1a
// hashes of the whole mono input and the whole RGB output buffer, including the
// rows, columns, channels and padding the call must leave untouched. The Rust
// port in crates/x3f-sys/src/spp_denoise/mono/hist_bilateral.rs
// regenerates the same inputs and must match every hash bit for bit.
//
// The FastBilateral_F20 argument is passed by value; AAPCS64 hands a 0xa0-byte
// aggregate over as a pointer to a caller copy in x4, after x0..x3 (the float
// goes to s0). Only its +0x50 (low), +0x58 (high) and +0x60 (weights) tables are
// read. Two kinds of tables are used:
//   mode 0: the native CalculateCoeff (0x30940, not exported) run on the same
//           object MonochromeDenoise builds: +0x18 the noise scaling as double,
//           +0x38/+0x40/+0x48 the noise model constants;
//   mode 1: synthetic tables (exact f32 arithmetic, no libm) so the Rust test
//           can also run off aarch64 macOS.
// Every case runs twice, with different values for the float and the two
// trailing unsigned arguments; the harness aborts if the outputs differ.
//
// Build and run (arm64 macOS):
//   clang++ -std=c++17 -arch arm64 -O1 -ffp-contract=off \
//     scripts/spp-native/hist_bilateral_mono.cpp -o /tmp/hist_bilateral_mono
//   /tmp/hist_bilateral_mono reference/merrill/lib99_MerillDLLForMac.arm64.dylib \
//     > crates/x3f-sys/tests/fixtures/spp-mono-hist-bilateral-native.txt
//
// Nothing printed here comes from Sigma: inputs are synthetic, outputs are hashes.

#include "pixmap.h"

#include <cinttypes>
#include <cmath>
#include <limits>
#include <vector>

namespace {

using HistBilateral = void (*)(merrill::Pixmap *, merrill::Pixmap *, unsigned, unsigned,
                               float, void *, unsigned, unsigned);
using CalculateCoeff = void (*)(void *);

const char *kSymbol =
    "_ZN15FPixelProc2_F2024ApplyHistBilateralFilterEP19FPixmapTemplate_F20I24FPixelClassFlt32Mono_"
    "F20EPS0_I23FPixelClassFlt32RGB_F20EjjfNS_17FastBilateral_F20Ejj";
const uintptr_t kCalculateCoeff = 0x30940;
const uintptr_t kMonoVtable = 0xd0c20;
const uintptr_t kRgbVtable = 0xd0ca0;

// Extra floats after the last mono row: the native reads one column past the
// image on the last column step (the result is discarded).
const size_t kTail = 64;

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

uint64_t f64_bits(double d) {
  uint64_t b;
  memcpy(&b, &d, sizeof b);
  return b;
}

uint32_t cell(uint32_t seed, uint32_t x, uint32_t y) {
  return mix32(seed ^ ((y * 65537u + x) * 0x9e3779b9u));
}

// One synthetic mono sample. Every expression rounds once per operation.
float sample(int pattern, uint32_t seed, uint32_t x, uint32_t y) {
  uint32_t u = cell(seed, x, y);
  switch (pattern) {
    case 0:  // the whole range up to 16383
      return static_cast<float>(u >> 8) * (16383.0f / 16777216.0f);
    case 1:  // near zero, negatives included
      return static_cast<float>(static_cast<int>(u % 81u) - 20) +
             static_cast<float>(u >> 24) * 0.00390625f;
    case 2: {  // smooth mid-range gradient with noise
      float v = 2000.0f + static_cast<float>(x * 3 + y * 2);
      return v + static_cast<float>(static_cast<int>(u >> 20) - 2048) * 0.1f;
    }
    case 3:  // just under the 8191 ceiling
      return 8191.0f - static_cast<float>(u >> 23) * 0.5f;
    case 4:  // across the 4095 bin clamp
      return 3900.0f + static_cast<float>(u >> 23) * 0.75f;
    case 5: {  // piecewise-constant blocks with noise
      float base = static_cast<float>(((x / 7 + y / 5) % 4) * 1100 + 300);
      return base + (static_cast<float>(u >> 24) * 0.5f - 64.0f);
    }
    case 6: {  // specials over the gradient
      switch (u % 37u) {
        case 0: return from_bits(0x7fc12345u);
        case 1: return std::numeric_limits<float>::infinity();
        case 2: return -std::numeric_limits<float>::infinity();
        case 3: return 1e30f;
        case 4: return -1e30f;
        case 5: return -0.0f;
        default: break;
      }
      float v = 2000.0f + static_cast<float>(x * 3 + y * 2);
      return v + static_cast<float>(static_cast<int>(u >> 20) - 2048) * 0.1f;
    }
    case 7:  // integers in the binned range
      return static_cast<float>(u % 4096u);
    default:  // just under 16383
      return 16383.0f - static_cast<float>(u >> 23);
  }
}

struct Tables {
  int32_t *low;
  int32_t *high;
  float **weights;
};

// The FastBilateral_F20 object, 0xa0 bytes (layout from MonochromeDenoise
// 0x2fd3c and the worker copy in DoFastBilateralFilter 0x2fbd0).
struct alignas(16) FastBilateral {
  uint8_t bytes[0xa0];
  void set_f64(size_t off, double v) { memcpy(bytes + off, &v, 8); }
  void set_ptr(size_t off, void *p) { memcpy(bytes + off, &p, 8); }
  void *ptr(size_t off) const {
    void *p;
    memcpy(&p, bytes + off, 8);
    return p;
  }
};
static_assert(sizeof(FastBilateral) == 0xa0, "FastBilateral_F20 is 0xa0 bytes");

void native_coeffs(const merrill::Library &lib, double scaling, FastBilateral *fb) {
  memset(fb->bytes, 0, sizeof fb->bytes);
  fb->set_f64(0x10, 0.1);
  fb->set_f64(0x18, scaling);
  fb->set_f64(0x20, 9.85389175535173);
  fb->set_f64(0x28, 0.0428362301201803);
  fb->set_f64(0x30, 1.73171583027778e-05);
  fb->set_f64(0x38, 9.85389175535173);
  fb->set_f64(0x40, 0.0428362301201803);
  fb->set_f64(0x48, 1.73171583027778e-05);
  reinterpret_cast<CalculateCoeff>(lib.at(kCalculateCoeff))(fb);
}

// Synthetic tables: radius ((7 i + 3 k) mod 13) + k, weight 1 / (1 + |d| / 4).
void synthetic_coeffs(int variant, FastBilateral *fb) {
  memset(fb->bytes, 0, sizeof fb->bytes);
  auto *low = static_cast<int32_t *>(malloc(1024 * sizeof(int32_t)));
  auto *high = static_cast<int32_t *>(malloc(1024 * sizeof(int32_t)));
  auto **weights = static_cast<float **>(malloc(1024 * sizeof(float *)));
  for (int i = 0; i < 1024; i++) {
    int r = (i * 7 + variant * 3) % 13 + variant;
    low[i] = i - r < 0 ? 0 : i - r;
    high[i] = (i + r > 1023 ? 1023 : i + r) + 1;
    weights[i] = static_cast<float *>(malloc((high[i] - low[i]) * sizeof(float)));
    for (int j = low[i]; j < high[i]; j++) {
      int d = i > j ? i - j : j - i;
      weights[i][j - low[i]] = 1.0f / (1.0f + static_cast<float>(d) * 0.25f);
    }
  }
  fb->set_ptr(0x50, low);
  fb->set_ptr(0x58, high);
  fb->set_ptr(0x60, weights);
}

uint64_t fnv_step(uint64_t h, const void *bytes, size_t n) {
  const uint8_t *b = static_cast<const uint8_t *>(bytes);
  for (size_t i = 0; i < n; i++) {
    h ^= b[i];
    h *= 0x100000001b3ull;
  }
  return h;
}

uint64_t coeff_hash(const FastBilateral &fb) {
  auto *low = static_cast<int32_t *>(fb.ptr(0x50));
  auto *high = static_cast<int32_t *>(fb.ptr(0x58));
  auto **weights = static_cast<float **>(fb.ptr(0x60));
  uint64_t h = 0xcbf29ce484222325ull;
  h = fnv_step(h, low, 1024 * sizeof(int32_t));
  h = fnv_step(h, high, 1024 * sizeof(int32_t));
  for (int i = 0; i < 1024; i++) h = fnv_step(h, weights[i], (high[i] - low[i]) * sizeof(float));
  return h;
}

struct Case {
  uint32_t w, h, mono_pad, rgb_pad, first, last;
  int pattern;
  uint32_t seed;
};

// first' = max(first, 9) and last' = min(last + 9, h - 1), both signed 32-bit;
// the native reads rows first' - 9 ..= first' + 9 and last' - 18 ..= last'.
bool in_domain(const Case &c) {
  int32_t first = static_cast<int32_t>(c.first) > 9 ? static_cast<int32_t>(c.first) : 9;
  int32_t last9 = static_cast<int32_t>(c.last + 9);
  int32_t last = static_cast<int32_t>(c.h) > last9 ? last9 : static_cast<int32_t>(c.h) - 1;
  return first + 9 <= static_cast<int32_t>(c.h) - 1 && last >= 18;
}

void run(const merrill::Library &lib, HistBilateral fn, const Case &c, const FastBilateral &fb,
         int mode, uint64_t param, float scaling) {
  const size_t ms = c.w + c.mono_pad, rs = 3 * c.w + c.rgb_pad;
  const size_t mono_len = ms * c.h, rgb_len = rs * c.h;
  std::vector<float> mono(mono_len + kTail, 100.0f);
  for (uint32_t y = 0; y < c.h; y++)
    for (uint32_t x = 0; x < ms; x++)
      mono[y * ms + x] = x < c.w ? sample(c.pattern, c.seed, x, y) : -777.25f;
  std::vector<float> rgb_init(rgb_len);
  for (size_t i = 0; i < rgb_len; i++) rgb_init[i] = -1000.0f - static_cast<float>(i % 97);
  const uint64_t mono_fnv = merrill::fnv1a(mono.data(), mono_len * sizeof(float));

  uint64_t out[2];
  for (int pass = 0; pass < 2; pass++) {
    std::vector<float> rgb = rgb_init;
    merrill::Pixmap pm{}, pr{};
    pm.vtable = lib.at(kMonoVtable);
    pm.data = mono.data();
    pm.width = c.w;
    pm.height = c.h;
    pm.row_bytes = static_cast<int32_t>(ms * sizeof(float));
    pr.vtable = lib.at(kRgbVtable);
    pr.data = rgb.data();
    pr.width = c.w;
    pr.height = c.h;
    pr.row_bytes = static_cast<int32_t>(rs * sizeof(float));
    FastBilateral copy = fb;  // the by-value argument
    if (pass == 0)
      fn(&pm, &pr, c.first, c.last, scaling, &copy, 0, 0);
    else
      fn(&pm, &pr, c.first, c.last, -1.5f, &copy, 0xdeadbeefu, 7);
    out[pass] = merrill::fnv1a(rgb.data(), rgb_len * sizeof(float));
  }
  if (out[0] != out[1]) {
    fprintf(stderr, "unused arguments changed the output\n");
    exit(1);
  }
  if (merrill::fnv1a(mono.data(), mono_len * sizeof(float)) != mono_fnv) {
    fprintf(stderr, "mono input changed\n");
    exit(1);
  }
  printf("HBF %u %u %u %u %u %u %d %08x %d %016" PRIx64 " %016" PRIx64 " %016" PRIx64 "\n", c.w,
         c.h, c.mono_pad, c.rgb_pad, c.first, c.last, c.pattern, c.seed, mode, param, mono_fnv,
         out[0]);
}

}  // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: %s lib99_MerillDLLForMac.arm64.dylib\n", argv[0]);
    return 2;
  }
  merrill::Library lib;
  if (!lib.open(argv[1])) return 1;
  auto fn = lib.sym<HistBilateral>(kSymbol);
  if (!fn) return 1;

  // Coefficient sets: native CalculateCoeff at several noise scalings, then
  // synthetic tables.
  const double scalings[] = {0.02, 0.1, 0.5, 1.0, 3.0, 20.0};
  const int variants[] = {0, 2, 9};
  struct Set {
    FastBilateral fb;
    int mode;
    uint64_t param;
    float scaling;
  };
  std::vector<Set> sets;
  for (double s : scalings) {
    Set set{};
    native_coeffs(lib, s, &set.fb);
    set.mode = 0;
    set.param = f64_bits(s);
    set.scaling = static_cast<float>(s);
    sets.push_back(set);
  }
  for (int v : variants) {
    Set set{};
    synthetic_coeffs(v, &set.fb);
    set.mode = 1;
    set.param = static_cast<uint64_t>(v);
    set.scaling = 1.0f;
    sets.push_back(set);
  }

  printf("# Native ApplyHistBilateralFilter (Flt32Mono -> Flt32RGB, 0x3301c) on synthetic\n");
  printf("# inputs, generated by scripts/spp-native/hist_bilateral_mono.cpp (see its header).\n");
  printf("# The float and the two trailing unsigned arguments were verified unused.\n");
  printf("# COEF mode param coeff_fnv (mode 0: CalculateCoeff, param = scaling f64 bits;\n");
  printf("#   mode 1: synthetic tables, param = variant)\n");
  printf("# HBF w h mono_pad rgb_pad first last pattern seed mode param mono_fnv rgb_out_fnv\n");
  for (const Set &set : sets)
    printf("COEF %d %016" PRIx64 " %016" PRIx64 "\n", set.mode, set.param, coeff_hash(set.fb));

  const uint32_t sizes[][2] = {{19, 19}, {20, 19}, {23, 24}, {31, 29},
                               {40, 33}, {57, 48}, {64, 64}};
  int index = 0, printed = 0;
  for (auto &s : sizes) {
    const uint32_t w = s[0], h = s[1];
    const uint32_t ranges[][2] = {
        {0, h},          {0, 0},          {9, h - 10},  {h / 3, 2 * h / 3}, {10, 11},
        {h - 10, h / 2}, {0, 1000},       {9, 9},       {h - 10, h - 10},   {0x80000000u, h},
    };
    for (auto &r : ranges) {
      for (int pattern = 0; pattern <= 8; pattern++) {
        Case c{w, h, static_cast<uint32_t>(index % 3 == 0 ? 0 : index % 3 + 2),
               static_cast<uint32_t>(index % 2 == 0 ? 0 : 5), r[0], r[1], pattern,
               0x5eed0000u + static_cast<uint32_t>(index)};
        // Rotate the coefficient set across ranges so each pattern meets several.
        const Set &set = sets[(index + index / 9) % sets.size()];
        index++;
        if (!in_domain(c)) continue;
        run(lib, fn, c, set.fb, set.mode, set.param, set.scaling);
        printed++;
      }
    }
  }
  // Every coefficient set on the larger image with every pattern.
  for (size_t k = 0; k < sets.size(); k++) {
    for (int pattern = 0; pattern <= 8; pattern++) {
      Case c{61, 52, 1, 2, 0, 52, pattern, 0xc0ffee00u + static_cast<uint32_t>(k * 16 + pattern)};
      run(lib, fn, c, sets[k].fb, sets[k].mode, sets[k].param, sets[k].scaling);
      printed++;
    }
  }
  printf("# %d cases\n", printed);
  return 0;
}
