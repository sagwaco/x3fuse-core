// Differential harness for FPixelProc2_F20::Sigma_HN, the Flt32Mono <- Flt32RGB
// overload at 0x42434 (the kernel of DoHNMixing inside MonochromeDenoise).
//
// Calls the native function on deterministic synthetic images and prints one
// fixture line per case: every input as an exact parameter (doubles and floats
// as hex bits, images as seed + pattern of the generator below) and the FNV-1a
// hash of the whole output buffer, including the rows and padding the call must
// leave untouched. The Rust port in
// crates/x3f-sys/src/spp_denoise/mono/hn.rs regenerates the
// same inputs and must match every hash bit for bit.
//
// Build and run (arm64 macOS):
//   clang++ -std=c++17 -arch arm64 -O1 -ffp-contract=off \
//     scripts/spp-native/sigma_hn.cpp -o /tmp/sigma_hn
//   /tmp/sigma_hn reference/merrill/lib99_MerillDLLForMac.arm64.dylib \
//     > crates/x3f-sys/tests/fixtures/spp-mono-hn-native.txt
//
// Nothing printed here comes from Sigma: inputs are synthetic, outputs are
// hashes (plus the first written value of each case).

#include "pixmap.h"

#include <cinttypes>
#include <cmath>
#include <limits>
#include <vector>

namespace {

struct FColorFloat32 {
  float r, g, b;
};

using SigmaHn = void (*)(merrill::Pixmap *, merrill::Pixmap *, unsigned, unsigned,
                         double *, double *, double, double, double, FColorFloat32);

const char *kSymbol =
    "_ZN15FPixelProc2_F208Sigma_HNEP19FPixmapTemplate_F20I24FPixelClassFlt32Mono_F20EPS0_"
    "I23FPixelClassFlt32RGB_F20EjjR11FColorFloatS8_ddd13FColorFloat32";

const float kSentinel = -12345.5f;

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

// Pixel generator shared with the Rust test (built from bits, no arithmetic).
//   pattern 0: wide, 2^-7 .. 2^17, 1/16 negative (exercises both clamps)
//   pattern 1: mid, 2^9 .. 2^13 (concentrated on the highlight ramp)
//   pattern 2: wide plus specials (inf, -inf, NaN, -0, subnormal, 32767, max)
float pixel(uint32_t seed, uint32_t index, uint32_t pattern) {
  uint32_t h = mix32(seed * 0x9e3779b9u ^ index);
  uint32_t h2 = mix32(h ^ 0x5bd1e995u);
  uint32_t mantissa = h & 0x7fffffu;
  if (pattern == 1) {
    uint32_t exp = 127 + 9 + h2 % 5;
    return from_bits(exp << 23 | mantissa);
  }
  if (pattern == 2 && (h2 >> 24) % 8 == 0) {
    static const uint32_t specials[] = {0x7f800000u, 0xff800000u, 0x7fc00000u, 0x80000000u,
                                        0x00000005u, 0x46fffe00u, 0x46fffe01u, 0x7f7fffffu};
    return from_bits(specials[h2 % 8]);
  }
  uint32_t exp = 120 + h2 % 25;
  uint32_t sign = (h2 >> 28) == 0 ? 0x80000000u : 0;
  return from_bits(sign | exp << 23 | mantissa);
}

struct Case {
  const char *name;
  uint32_t width, height;
  uint32_t in_stride, out_stride;  // floats per row
  uint32_t first, last;            // inclusive row range, as native
  uint32_t seed, pattern;
  double a[3], b[3];
  double start, range, mix;
  float w[3];
};

uint64_t hash_floats(const std::vector<float> &v) {
  return merrill::fnv1a(v.data(), v.size() * sizeof(float));
}

}  // namespace

int main(int argc, char **argv) {
  const char *path = argc > 1 ? argv[1] : "reference/merrill/lib99_MerillDLLForMac.arm64.dylib";
  merrill::Library lib;
  if (!lib.open(path)) return 1;
  auto fn = lib.sym<SigmaHn>(kSymbol);
  if (!fn || reinterpret_cast<uintptr_t>(fn) - lib.base != 0x42434) {
    fprintf(stderr, "Sigma_HN not at 0x42434\n");
    return 1;
  }

  const double inf = std::numeric_limits<double>::infinity();
  const double nan = std::numeric_limits<double>::quiet_NaN();
  // DoHNMixing's constants.
  const double A[3] = {0.03125, 0.03125, 0.9375};
  const double B[3] = {0.25, 0.5, 0.25};
  const float W[3] = {0.299f, 0.587f, 0.114f};
  const float W2[3] = {0.25f, 0.5f, 0.25f};
  const float W3[3] = {-0.75f, 1.5f, 0.3333333f};
  const float W4[3] = {1e30f, 1e30f, -1e30f};
  const float W0[3] = {0.0f, 0.0f, 0.0f};

  std::vector<Case> cases;
  auto add = [&](const char *name, uint32_t w, uint32_t h, uint32_t pad_in, uint32_t pad_out,
                 uint32_t first, uint32_t last, uint32_t seed, uint32_t pattern, const double *a,
                 const double *b, double start, double range, double mix, const float *wt) {
    Case c{name, w, h, w * 3 + pad_in, w + pad_out, first, last, seed, pattern,
           {a[0], a[1], a[2]}, {b[0], b[1], b[2]}, start, range, mix, {wt[0], wt[1], wt[2]}};
    cases.push_back(c);
  };

  // Production call (mix 0): plain clamp + weighted sum, every width tail.
  const uint32_t widths[] = {1, 2, 3, 4, 5, 7, 8, 13, 64, 257};
  uint32_t seed = 1;
  for (uint32_t w : widths) {
    add("prod", w, 5, 0, 0, 0, 4, seed++, 0, A, B, 2500, 950, 0, W);
    add("prod_pad", w, 6, 5, 3, 1, 4, seed++, 2, A, B, 2500, 950, 0, W2);
  }
  // Highlight blend path (mix > 0), every width tail.
  const double mixes[] = {1.0, 0.5, 0.25, 0.9999999, 2.0, 5e-324, 1e-300};
  for (uint32_t w : widths) {
    for (double m : mixes) {
      add("blend", w, 4, 0, 0, 0, 3, seed++, 1, A, B, 2500, 950, m, W);
    }
    add("blend_wide", w, 7, 2, 1, 2, 5, seed++, 0, A, B, 2500, 950, 0.5, W3);
    add("blend_special", w, 3, 1, 2, 0, 2, seed++, 2, A, B, 2500, 950, 0.75, W);
  }
  // Mix that is not > 0 takes the plain path.
  add("mix_neg", 9, 3, 0, 0, 0, 2, seed++, 1, A, B, 2500, 950, -0.5, W);
  add("mix_negzero", 9, 3, 0, 0, 0, 2, seed++, 1, A, B, 2500, 950, -0.0, W);
  add("mix_nan", 9, 3, 0, 0, 0, 2, seed++, 1, A, B, 2500, 950, nan, W);
  add("mix_inf", 9, 3, 0, 0, 0, 2, seed++, 1, A, B, 2500, 950, inf, W);
  // Row ranges: single row, last row, empty (first > last).
  add("row_single", 11, 6, 0, 0, 3, 3, seed++, 0, A, B, 2500, 950, 0.5, W);
  add("row_last", 11, 6, 0, 0, 5, 5, seed++, 0, A, B, 2500, 950, 0, W);
  add("row_empty", 11, 6, 0, 0, 4, 3, seed++, 0, A, B, 2500, 950, 0.5, W);
  add("row_empty0", 11, 6, 0, 0, 4, 3, seed++, 0, A, B, 2500, 950, 0, W);
  // Ramp parameters at the edges.
  const double A2[3] = {-0.5, 1.25, 0.125};
  const double B2[3] = {1.0 / 3, -2.0, 7.5};
  const double Z[3] = {0, 0, 0};
  const double H[3] = {1e300, 1e300, 1e300};
  add("range_zero", 17, 4, 0, 0, 0, 3, seed++, 1, A, B, 2500, 0, 0.5, W);
  add("range_zero_eq", 17, 4, 0, 0, 0, 3, seed++, 1, Z, B, 0, 0, 0.5, W);
  add("range_neg", 17, 4, 0, 0, 0, 3, seed++, 1, A, B, 2500, -950, 0.5, W);
  add("start_neg", 17, 4, 0, 0, 0, 3, seed++, 0, A, B, -1e6, 950, 0.5, W);
  add("start_huge", 17, 4, 0, 0, 0, 3, seed++, 0, A, B, 1e300, 1e-300, 0.5, W);
  add("odd_weights", 17, 4, 3, 1, 0, 3, seed++, 0, A2, B2, 1000, 3000, 0.6, W3);
  add("zero_weights", 17, 4, 0, 0, 0, 3, seed++, 0, Z, Z, 0, 1, 0.6, W0);
  add("huge_ab", 17, 4, 0, 0, 0, 3, seed++, 0, H, H, 2500, 950, 0.6, W);
  add("huge_out", 17, 4, 0, 0, 0, 3, seed++, 0, A, B, 2500, 950, 0.6, W4);
  add("huge_out0", 17, 4, 0, 0, 0, 3, seed++, 0, A, B, 2500, 950, 0, W4);
  add("special_plain", 33, 5, 0, 0, 0, 4, seed++, 2, A, B, 2500, 950, 0, W3);
  add("special_plain_narrow", 3, 9, 0, 0, 0, 8, seed++, 2, A, B, 2500, 950, 0, W);
  add("special_narrow", 1, 9, 0, 0, 0, 8, seed++, 2, A2, B2, 100, 50, 0.3, W);
  add("large", 301, 23, 4, 7, 2, 20, seed++, 0, A, B, 2500, 950, 0.5, W);
  add("large0", 301, 23, 4, 7, 2, 20, seed++, 0, A, B, 2500, 950, 0, W);

  printf("# FPixelProc2_F20::Sigma_HN (Flt32Mono <- Flt32RGB) at 0x42434, native outputs.\n");
  printf("# Generated by scripts/spp-native/sigma_hn.cpp against the arm64 slice of\n");
  printf("# lib99_MerillDLLForMac.dylib; inputs are synthetic (seed/pattern generator in\n");
  printf("# that file and in spp_denoise/mono/hn.rs). Strides in floats, rows inclusive.\n");
  printf("# HN name width height in_stride out_stride first last seed pattern a0 a1 a2 b0 b1 b2"
         " start range mix w0 w1 w2 | first_out_bits hash\n");

  for (const Case &c : cases) {
    std::vector<float> in(static_cast<size_t>(c.in_stride) * c.height + 3);
    for (size_t i = 0; i < in.size(); i++) in[i] = pixel(c.seed, static_cast<uint32_t>(i), c.pattern);
    std::vector<float> out(static_cast<size_t>(c.out_stride) * c.height, kSentinel);

    merrill::Pixmap rgb{lib.at(0xd0ca0), in.data(), c.width, c.height,
                        static_cast<int32_t>(c.in_stride * 4), 0};
    merrill::Pixmap mono{lib.at(0xd0c20), out.data(), c.width, c.height,
                         static_cast<int32_t>(c.out_stride * 4), 0};
    double a[3] = {c.a[0], c.a[1], c.a[2]};
    double b[3] = {c.b[0], c.b[1], c.b[2]};
    std::vector<float> in_copy = in;
    fn(&mono, &rgb, c.first, c.last, a, b, c.start, c.range, c.mix, {c.w[0], c.w[1], c.w[2]});

    if (memcmp(in.data(), in_copy.data(), in.size() * sizeof(float)) != 0 ||
        memcmp(a, c.a, sizeof a) != 0 || memcmp(b, c.b, sizeof b) != 0) {
      fprintf(stderr, "case %s: native modified an input\n", c.name);
      return 1;
    }
    uint32_t first_out = c.first <= c.last ? bits_of(out[c.first * c.out_stride]) : 0;
    printf("HN %s %u %u %u %u %u %u %u %u", c.name, c.width, c.height, c.in_stride, c.out_stride,
           c.first, c.last, c.seed, c.pattern);
    for (double v : c.a) printf(" %016" PRIx64, bits_of(v));
    for (double v : c.b) printf(" %016" PRIx64, bits_of(v));
    printf(" %016" PRIx64 " %016" PRIx64 " %016" PRIx64, bits_of(c.start), bits_of(c.range),
           bits_of(c.mix));
    for (float v : c.w) printf(" %08x", bits_of(v));
    printf(" | %08x %016" PRIx64 "\n", first_out, hash_floats(out));
  }
  return 0;
}
