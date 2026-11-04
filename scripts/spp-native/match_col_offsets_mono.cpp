// Differential harness for the monochrome overload of MatchColOffsets at
// 0x2f284 (the kernel of DoMonoBCS inside MonochromeDenoise):
//
//   MatchColOffsets(FPixmapTemplate_F20<Flt32Mono>*, FPixmapTemplate_F20<Flt32RGB>*,
//                   unsigned first, unsigned last, float, float iso,
//                   unsigned filter_size, bool, unsigned,
//                   FPixelProc2_F20::FastBilateral_F20, unsigned scale_index, bool)
//
// Calls the native function on deterministic synthetic images and prints one
// fixture line per case: every input as an exact parameter (floats as hex
// bits, images as seed + pattern of the generator below, weight tables as a
// kind + scale) and the FNV-1a hash of both whole buffers (mono and RGB,
// including the rows and padding the call must leave untouched). The Rust port
// in crates/x3f-sys/src/spp_denoise/mono/bcs.rs regenerates
// the same inputs from each line and must match every hash bit for bit.
//
// FastBilateral_F20 is 0xa0 bytes and travels by value: AAPCS64 makes a caller
// copy and passes its address in x7 (DoMonoBCS copies it from the worker
// context at +0xa0..+0x140). Only +0x00 (float *weights[0x1000]) and +0x08
// (int limits[0x1000]) are read below MatchColOffsets; the rest is filled like
// MonochromeDenoise does for realism.
//
// Build and run (arm64 macOS):
//   clang++ -std=c++17 -arch arm64 -O1 -ffp-contract=off \
//     scripts/spp-native/match_col_offsets_mono.cpp -o /tmp/match_col_offsets_mono
//   /tmp/match_col_offsets_mono reference/merrill/lib99_MerillDLLForMac.arm64.dylib \
//     > crates/x3f-sys/tests/fixtures/spp-mono-bcs-native.txt
// An optional second argument names a directory that receives the raw output
// planes (<name>.mono.f32, <name>.rgb.f32) for debugging a mismatch.
//
// Nothing printed here comes from Sigma: inputs are synthetic, outputs are
// hashes.

#include "pixmap.h"

#include <cinttypes>
#include <cmath>
#include <map>
#include <string>
#include <vector>

namespace {

struct FastBilateral {
  float **weights;        // +0x00: 0x1000 rows, row i has limits[i] + 1 entries
  int32_t *limits;        // +0x08
  double bcs_scale;       // +0x10: NoiseParams+0xac (0.1 when zero)
  double noise_scaling;   // +0x18: NoiseParams+0x28, CalculateCoeff input
  double coeff[6];        // +0x20: {c0, c1, c2} twice (second copy rotated)
  uint8_t rest[0x50];     // +0x50: CalculateCoeff outputs, WB gains, mixing
};
static_assert(sizeof(FastBilateral) == 0xa0, "FastBilateral_F20 is 0xa0 bytes");

using MatchColOffsetsMono = void (*)(merrill::Pixmap *, merrill::Pixmap *, unsigned, unsigned,
                                     float, float, unsigned, bool, unsigned, FastBilateral,
                                     unsigned, bool);

const char *kSymbol =
    "_Z15MatchColOffsetsP19FPixmapTemplate_F20I24FPixelClassFlt32Mono_F20EPS_I23FPixelClassFlt32RGB_"
    "F20EjjffjbjN15FPixelProc2_F2017FastBilateral_F20Ejb";

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
double from_bits64(uint64_t bits) {
  double d;
  memcpy(&d, &bits, sizeof d);
  return d;
}

// Pixel generator shared with the Rust test (integer hashing, exact f32 ops).
//   pattern 0: realistic, gradient 600..2400 + noise +-64 + column stripes
//              of +-22.5 (the offsets BCS removes)
//   pattern 1: wide, -500 .. 40000 in 1/1024 steps (both clamps, LUT clamp)
//   pattern 2: pattern 0 with 1/8 specials (inf, -inf, NaN, -0, huge, tiny)
float pixel(uint32_t seed, uint32_t x, uint32_t y, uint32_t pattern) {
  uint32_t h = mix32(seed * 0x9e3779b9u ^ (x * 0x10001u + y * 0x3c6ef372u));
  uint32_t h2 = mix32(h ^ 0x5bd1e995u);
  if (pattern == 1) {
    return static_cast<float>(static_cast<int32_t>(h % 40500u) - 500) +
           static_cast<float>(h2 % 1024u) * 0.0009765625f;
  }
  if (pattern == 2 && (h2 >> 24) % 8 == 0) {
    static const uint32_t kSpecials[10] = {0x7f800000u, 0xff800000u, 0x7fc00000u, 0x80000000u,
                                          0x7149f2cau, 0x00000001u, 0x46fffe00u, 0xf149f2cau,
                                          0x457ff800u, 0x45800000u};
    return from_bits(kSpecials[(h2 >> 8) % 10]);
  }
  int32_t level = 600 + static_cast<int32_t>((x * 37u + y * 11u) % 1800u);
  int32_t noise = static_cast<int32_t>(h % 129u) - 64;
  int32_t stripe = static_cast<int32_t>(mix32(seed * 0x85ebca6bu ^ x) % 61u) - 30;
  return static_cast<float>(level + noise) + static_cast<float>(stripe) * 0.75f;
}

// The weight tables, two kinds:
//   kind 0: built like MonochromeDenoise (0x30100..0x302a8) from the BCS noise
//           scale (NoiseParams+0xac); depends on the system libm exp.
//   kind 1: synthetic, with negative limits and all-zero rows, so the
//           0/0 fallback of DirectionalFastBF runs on finite inputs.
//   kind 2: null pointers (DirectionalFastBF returns early).
struct Tables {
  std::vector<std::vector<float>> rows;
  std::vector<float *> row_ptrs;
  std::vector<int32_t> limits;
};

const double kC0 = from_bits64(0x4023b5314cd71e93ull);  // 9.85389175535173
const double kC1 = from_bits64(0x3fa5eea15ee6741aull);  // 0.0428362301201803
const double kC2 = from_bits64(0x3ef2288a0e9f8189ull);  // 1.73171583027778e-05

Tables build_tables(uint32_t kind, float scale) {
  Tables t;
  t.rows.resize(0x1000);
  t.limits.resize(0x1000);
  for (int i = 0; i < 0x1000; i++) {
    if (kind == 0) {
      double d = static_cast<double>(i);
      double var = std::fma(kC2 * d, d, std::fma(kC1, d, kC0)) / static_cast<double>(scale);
      int32_t lim = static_cast<int32_t>(std::sqrt(var) * 3.0);
      t.limits[i] = lim;
      t.rows[i].resize(lim + 1);
      for (int32_t k = 0; k <= lim; k++) {
        t.rows[i][k] = static_cast<float>(std::exp(-static_cast<double>(k * k) / var));
      }
    } else {
      int32_t lim = i % 7 - 1;
      t.limits[i] = lim;
      t.rows[i].resize(lim + 1);
      for (int32_t k = 0; k <= lim; k++) {
        t.rows[i][k] =
            i % 3 == 0 ? 0.0f : 1.0f / static_cast<float>(k + 1 + i % 5);
      }
    }
  }
  t.row_ptrs.resize(0x1000);
  for (int i = 0; i < 0x1000; i++) t.row_ptrs[i] = t.rows[i].data();
  return t;
}

struct Case {
  const char *name;
  uint32_t width, height, mono_pad, rgb_pad, first, last, filter_size;
  float iso, unused_scale;
  bool unused_bool;
  uint32_t unused_uint, scale_index;
  bool unused_bool2;
  uint32_t table_kind;
  float table_scale;
  uint32_t seed, pattern;
};

}  // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: %s lib99_MerillDLLForMac.arm64.dylib\n", argv[0]);
    return 2;
  }
  merrill::Library lib;
  if (!lib.open(argv[1])) return 1;
  auto fn = lib.sym<MatchColOffsetsMono>(kSymbol);
  if (!fn || reinterpret_cast<uintptr_t>(fn) != lib.base + 0x2f284) {
    fprintf(stderr, "MatchColOffsets mono not at 0x2f284\n");
    return 1;
  }
  void *mono_vt = lib.at(0xd0c20);
  void *rgb_vt = lib.at(0xd0ca0);

  std::vector<Case> cases = {
      // Production shape: filter 27, scale index of NoiseParams+0xb4, one
      // DoThreadedPix slice of rows (last inclusive).
      {"prod", 96, 130, 0, 0, 0, 129, 27, 400.0f, 1.0f, false, 0, 2, false, 0, 1.0f, 1, 0},
      {"prod_slice", 96, 130, 3, 5, 41, 122, 27, 1600.0f, 1.0f, false, 7, 4, false, 0, 1.0f, 2, 0},
      {"prod_odd", 101, 97, 1, 2, 3, 95, 27, 800.0f, 1.0f, false, 0, 3, false, 0, 1.0f, 3, 0},
      {"prod_small", 33, 50, 2, 1, 0, 49, 27, 200.0f, 1.0f, false, 0, 4, false, 0, 1.0f, 4, 0},
      {"prod_narrow", 27, 45, 0, 0, 1, 44, 27, 3200.0f, 1.0f, false, 0, 4, false, 0, 1.0f, 5, 0},
      {"prod_tooarrow", 26, 45, 0, 0, 1, 44, 27, 3200.0f, 1.0f, false, 0, 4, false, 0, 1.0f, 6, 0},
      // Unused arguments: p5, p8, p9, p12 must not change anything.
      {"unused", 64, 60, 1, 1, 2, 57, 27, 1600.0f, 3.5f, true, 0xdeadbeef, 1, true, 0, 1.0f, 2, 0},
      // Row ranges around the 40-row block size.
      {"span39", 40, 60, 0, 0, 5, 44, 9, 800.0f, 1.0f, false, 0, 4, false, 0, 1.0f, 7, 0},
      {"span40", 40, 60, 0, 0, 5, 45, 9, 800.0f, 1.0f, false, 0, 4, false, 0, 1.0f, 8, 0},
      {"span79", 40, 90, 0, 0, 5, 84, 9, 800.0f, 1.0f, false, 0, 4, false, 0, 1.0f, 9, 0},
      {"span80", 40, 90, 0, 0, 5, 85, 9, 800.0f, 1.0f, false, 0, 4, false, 0, 1.0f, 10, 0},
      {"span123", 17, 140, 2, 3, 10, 133, 9, 800.0f, 1.0f, false, 0, 4, false, 0, 1.0f, 11, 0},
      {"one_row", 20, 5, 0, 0, 2, 2, 5, 800.0f, 1.0f, false, 0, 4, false, 0, 1.0f, 12, 0},
      // Every ISO interval of the threshold table, including the two upper
      // intervals that read past the 7 thresholds into the ISO points.
      {"iso50", 23, 45, 0, 0, 0, 44, 7, 50.0f, 1.0f, false, 0, 4, false, 0, 1.0f, 13, 0},
      {"iso100", 23, 45, 0, 0, 0, 44, 7, 100.0f, 1.0f, false, 0, 4, false, 0, 1.0f, 14, 0},
      {"iso101", 23, 45, 0, 0, 0, 44, 7, 100.5f, 1.0f, false, 0, 4, false, 0, 1.0f, 15, 0},
      {"iso150", 23, 45, 0, 0, 0, 44, 7, 150.0f, 1.0f, false, 0, 4, false, 0, 1.0f, 16, 0},
      {"iso250", 23, 45, 0, 0, 0, 44, 7, 250.0f, 1.0f, false, 0, 4, false, 0, 1.0f, 17, 0},
      {"iso640", 23, 45, 0, 0, 0, 44, 7, 640.0f, 1.0f, false, 0, 4, false, 0, 1.0f, 18, 0},
      {"iso1250", 23, 45, 0, 0, 0, 44, 7, 1250.0f, 1.0f, false, 0, 4, false, 0, 1.0f, 19, 0},
      {"iso2500", 23, 45, 0, 0, 0, 44, 7, 2500.0f, 1.0f, false, 0, 4, false, 0, 1.0f, 20, 0},
      {"iso5000", 23, 45, 0, 0, 0, 44, 7, 5000.0f, 1.0f, false, 0, 4, false, 0, 1.0f, 21, 0},
      {"iso10000", 23, 45, 0, 0, 0, 44, 7, 10000.0f, 1.0f, false, 0, 4, false, 0, 1.0f, 22, 0},
      {"iso12800", 23, 45, 0, 0, 0, 44, 7, 12800.0f, 1.0f, false, 0, 4, false, 0, 1.0f, 23, 0},
      {"iso25600", 23, 45, 0, 0, 0, 44, 7, 25600.0f, 1.0f, false, 0, 4, false, 0, 1.0f, 24, 0},
      // Scale index: 0..4 are 0.2..1.0, 5..8 read NaN (no correction).
      {"scale0", 31, 45, 0, 0, 0, 44, 11, 640.0f, 1.0f, false, 0, 0, false, 0, 1.0f, 25, 0},
      {"scale1", 31, 45, 0, 0, 0, 44, 11, 640.0f, 1.0f, false, 0, 1, false, 0, 1.0f, 26, 0},
      {"scale3", 31, 45, 0, 0, 0, 44, 11, 640.0f, 1.0f, false, 0, 3, false, 0, 1.0f, 27, 0},
      {"scale5", 31, 45, 0, 0, 0, 44, 11, 640.0f, 1.0f, false, 0, 5, false, 0, 1.0f, 28, 0},
      {"scale8", 31, 45, 0, 0, 0, 44, 11, 640.0f, 1.0f, false, 0, 8, false, 0, 1.0f, 29, 0},
      // Filter sizes: even sizes make the two radii differ (stale RGB
      // columns), 1 takes the strided copy path, 2 the 2-tap bilateral.
      {"fs26", 48, 50, 1, 2, 0, 49, 26, 800.0f, 1.0f, false, 0, 4, false, 0, 1.0f, 30, 0},
      {"fs4", 48, 50, 1, 2, 0, 49, 4, 800.0f, 1.0f, false, 0, 4, false, 0, 1.0f, 31, 0},
      {"fs3", 15, 50, 0, 0, 0, 49, 3, 800.0f, 1.0f, false, 0, 4, false, 0, 1.0f, 32, 0},
      {"fs2", 48, 50, 1, 2, 0, 49, 2, 800.0f, 1.0f, false, 0, 4, false, 0, 1.0f, 33, 0},
      {"fs1", 48, 50, 1, 2, 0, 48, 1, 800.0f, 1.0f, false, 0, 4, false, 0, 1.0f, 34, 0},
      {"fs1_pad", 9, 50, 9, 0, 3, 49, 1, 800.0f, 1.0f, false, 0, 4, false, 0, 1.0f, 35, 0},
      {"fs41", 90, 50, 0, 0, 0, 49, 41, 800.0f, 1.0f, false, 0, 4, false, 0, 1.0f, 36, 0},
      // BCS noise scale of the tables.
      {"tscale05", 64, 50, 0, 0, 0, 49, 27, 800.0f, 0.5f, false, 0, 4, false, 0, 0.5f, 37, 0},
      {"tscale2", 64, 50, 0, 0, 0, 49, 27, 800.0f, 2.0f, false, 0, 4, false, 0, 2.0f, 38, 0},
      {"tscale025", 64, 50, 0, 0, 0, 49, 27, 800.0f, 0.25f, false, 0, 4, false, 0, 0.25f, 39, 0},
      // Wide values: negative, above 32766, above the 4095 table index.
      {"wide", 57, 90, 2, 1, 0, 89, 27, 1600.0f, 1.0f, false, 0, 4, false, 0, 1.0f, 40, 1},
      {"wide_fs8", 57, 90, 2, 1, 0, 89, 8, 6400.0f, 1.0f, false, 0, 2, false, 0, 1.0f, 41, 1},
      // Specials (fallback to mono[r + 2i]); last row kept below the final
      // row so the fallback read stays inside the image.
      {"specials", 61, 92, 3, 2, 0, 90, 27, 800.0f, 1.0f, false, 0, 4, false, 0, 1.0f, 42, 2},
      {"specials_fs6", 61, 92, 3, 2, 1, 89, 6, 400.0f, 1.0f, false, 0, 4, false, 0, 1.0f, 43, 2},
      // Synthetic tables with empty and zero rows (0/0 fallback).
      {"sparse", 50, 60, 1, 1, 0, 58, 9, 800.0f, 1.0f, false, 0, 4, false, 1, 1.0f, 44, 1},
      {"sparse_fs27", 70, 60, 4, 0, 2, 57, 27, 1600.0f, 1.0f, false, 0, 4, false, 1, 1.0f, 45, 0},
      // Null tables: no filtering, the correction runs on the stale RGB.
      {"null", 40, 60, 0, 1, 0, 59, 27, 800.0f, 0.0f, false, 0, 4, false, 2, 0.0f, 46, 0},
      {"null_fs8", 40, 60, 0, 1, 0, 59, 8, 800.0f, 0.0f, false, 0, 4, false, 2, 0.0f, 47, 0},
  };

  printf("# lib99_MerillDLLForMac.dylib arm64 slice SHA-256 "
         "b728f4f1aa7d8c7598a04492b1cc5a0d29cf8a7fde7a34eb96b7cc7608ca496d\n");
  printf("# Regenerate: clang++ -std=c++17 -arch arm64 -O1 -ffp-contract=off "
         "scripts/spp-native/match_col_offsets_mono.cpp -o match_col_offsets_mono && "
         "./match_col_offsets_mono reference/merrill/lib99_MerillDLLForMac.arm64.dylib\n");
  printf("# MatchColOffsets (Flt32Mono, Flt32RGB) at 0x2f284, the DoMonoBCS kernel, native "
         "outputs.\n");
  printf("# Inputs are synthetic (generator in scripts/spp-native/match_col_offsets_mono.cpp and "
         "spp_denoise/mono/bcs.rs).\n");
  printf("# Strides in floats: mono width+mono_pad, rgb 3*width+rgb_pad. Rows first..=last.\n");
  printf("# Table kind 0 = MonochromeDenoise build from table_scale (libm exp), 1 = synthetic, "
         "2 = null.\n");
  printf("# TABLE scale | limits_hash weights_count weights_hash (kind 0, harness build)\n");
  printf("# BCS name width height mono_pad rgb_pad first last filter iso p5 p8 p9 scale_index "
         "p12 kind table_scale seed pattern | mono_count mono_hash rgb_count rgb_hash\n");

  std::map<uint32_t, Tables> tables;
  auto get_tables = [&](uint32_t kind, float scale) -> Tables & {
    uint32_t key = kind == 0 ? bits_of(scale) : 0xffffffffu - kind;
    auto it = tables.find(key);
    if (it == tables.end()) {
      it = tables.emplace(key, build_tables(kind, scale)).first;
      if (kind == 0) {
        std::vector<float> flat;
        for (auto &row : it->second.rows) flat.insert(flat.end(), row.begin(), row.end());
        printf("TABLE %08x | %016" PRIx64 " %zu %016" PRIx64 "\n", bits_of(scale),
               merrill::fnv1a(it->second.limits.data(), it->second.limits.size() * 4),
               flat.size(), merrill::fnv1a(flat.data(), flat.size() * 4));
      }
    }
    return it->second;
  };

  for (const Case &c : cases) {
    uint32_t ms = c.width + c.mono_pad;
    uint32_t rs = 3 * c.width + c.rgb_pad;
    size_t mono_len = static_cast<size_t>(ms) * c.height;
    size_t rgb_len = static_cast<size_t>(rs) * c.height;
    // Slack after both buffers so a native overread cannot fault; it is not
    // part of the hashes and the Rust slices end at the image.
    std::vector<float> mono(mono_len + 4 * ms + 64), rgb(rgb_len + 4 * rs + 64);
    for (size_t i = 0; i < mono.size(); i++)
      mono[i] = pixel(c.seed, i % ms, static_cast<uint32_t>(i / ms), c.pattern);
    for (size_t i = 0; i < rgb.size(); i++)
      rgb[i] = pixel(c.seed ^ 0xa5a5a5a5u, i % rs, static_cast<uint32_t>(i / rs), 0);

    FastBilateral fb{};
    if (c.table_kind != 2) {
      Tables &t = get_tables(c.table_kind, c.table_scale);
      fb.weights = t.row_ptrs.data();
      fb.limits = t.limits.data();
    }
    fb.bcs_scale = c.table_scale != 0.0f ? c.table_scale : 0.1;
    fb.noise_scaling = 0.0;
    fb.coeff[0] = kC0;
    fb.coeff[1] = kC1;
    fb.coeff[2] = kC2;
    fb.coeff[3] = kC0;
    fb.coeff[4] = kC1;
    fb.coeff[5] = kC2;

    merrill::Pixmap pm{mono_vt, mono.data(), c.width, c.height, static_cast<int32_t>(ms * 4), 0};
    merrill::Pixmap pr{rgb_vt, rgb.data(), c.width, c.height, static_cast<int32_t>(rs * 4), 0};
    fn(&pm, &pr, c.first, c.last, c.unused_scale, c.iso, c.filter_size, c.unused_bool,
       c.unused_uint, fb, c.scale_index, c.unused_bool2);

    printf("BCS %s %u %u %u %u %u %u %u %08x %08x %u %u %u %u %u %08x %u %u | %zu %016" PRIx64
           " %zu %016" PRIx64 "\n",
           c.name, c.width, c.height, c.mono_pad, c.rgb_pad, c.first, c.last, c.filter_size,
           bits_of(c.iso), bits_of(c.unused_scale), c.unused_bool ? 1u : 0u, c.unused_uint,
           c.scale_index, c.unused_bool2 ? 1u : 0u, c.table_kind, bits_of(c.table_scale), c.seed,
           c.pattern, mono_len, merrill::fnv1a(mono.data(), mono_len * 4), rgb_len,
           merrill::fnv1a(rgb.data(), rgb_len * 4));
    if (argc > 2) {
      std::string base = std::string(argv[2]) + "/" + c.name;
      if (FILE *f = fopen((base + ".mono.f32").c_str(), "wb")) {
        fwrite(mono.data(), 4, mono_len, f);
        fclose(f);
      }
      if (FILE *f = fopen((base + ".rgb.f32").c_str(), "wb")) {
        fwrite(rgb.data(), 4, rgb_len, f);
        fclose(f);
      }
    }
  }
  return 0;
}
