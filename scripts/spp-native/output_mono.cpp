// Differential harness for the monochrome OUTPUT stage of the Merrill engine:
//
//   * FPixelProc2_F20::ProcessToOutput @0x4f4e8 (the Stage4 overload with
//     FColorFloat toning gains and the grain/mono flags), which converts the
//     FColorMatrix to Q12 and runs ProcessToOutputRowTemplate<48RGB>::Do
//     @0x4e9d0 on every row (pixel type 0xf, the 16-bit TIFF output);
//   * FPixelProc2_F20::GenerateF20ToneCurve @0x57c64 with the mono flag off
//     and on (the "-ToneLUT-BnW-" curve);
//   * FPixelProc2_F20::CreateGammaLUT_F20 @0x561f4 (16-bit, as priMakeGammaLUT).
//
// It also prints the Stage4 toning conversion (FIPEngine2_F20::
// priProcessF20Stage4 @0xac7d4..0xac8b4, RGB gains -> Y/Cb/Cr in float32)
// evaluated with the same fused operations as the disassembly; that part is
// not a call into the library (it is inlined in a function we cannot drive),
// but its constants are read from the loaded library at 0xbf3b0.
//
// Every input is synthetic: doubles as hex bits, tables and images from the
// generator below (mirrored in crates/x3f-sys/src/
// merrill_mono.rs). Outputs are FNV-1a hashes of whole buffers plus a few
// sample values; nothing from Sigma is printed.
//
// Build and run (arm64 macOS):
//   clang++ -std=c++17 -arch arm64 -O1 -ffp-contract=off \
//     scripts/spp-native/output_mono.cpp -o /tmp/output_mono
//   /tmp/output_mono reference/merrill/lib99_MerillDLLForMac.arm64.dylib \
//     > crates/x3f-sys/tests/fixtures/merrill-mono-output-native.txt

#include "pixmap.h"

#include <cinttypes>
#include <cmath>
#include <vector>

namespace {

struct FColorFloat {
  double r, g, b;
};

struct Clut {  // FCLUTTemplate_F20<unsigned short>, 0x28 bytes
  uint16_t *t[3];
  uint32_t mask, size;
  uint64_t own;
};
static_assert(sizeof(Clut) == 0x28, "CLUT layout");

template <typename T> struct Lut {  // FLUTTemplate_F20<T>, 0x18 bytes
  T *t;
  uint32_t mask, size;
  uint64_t own;
};
static_assert(sizeof(Lut<uint16_t>) == 0x18, "LUT layout");

struct ToneParams {  // FPixelProc2_F20::ToneParams, 0x88 bytes
  double v[17];
};

using ProcessToOutput = void (*)(const merrill::Pixmap *, const merrill::Pixmap *, const Clut *,
                                 const Lut<int16_t> *, const Lut<uint16_t> *,
                                 const Lut<uint16_t> *, const double *, FColorFloat, double,
                                 double, merrill::Pixmap *, bool, bool, float);
using GenerateTone = void (*)(double, ToneParams *, double, double, double, double, FColorFloat,
                              unsigned, Clut *, bool);
using CreateGamma = void (*)(Lut<uint16_t> *, int, unsigned);

const char *kProcess =
    "_ZN15FPixelProc2_F2015ProcessToOutputEPK19FPixmapTemplate_F20I21FPixelClass48SRGB_F20EPKS0_"
    "I22FPixelClass16SMono_F20EPKNS_17FCLUTTemplate_F20ItEEPKNS_16FLUTTemplate_F20IsEEPKNSD_"
    "ItEESJ_RK12FColorMatrix11FColorFloatddP7FPixmapbbf";
const char *kTone =
    "_ZN15FPixelProc2_F2020GenerateF20ToneCurveEdRNS_10ToneParamsEdddd11FColorFloatjPNS_"
    "17FCLUTTemplate_F20ItEEb";
const char *kGamma =
    "_ZN15FPixelProc2_F2018CreateGammaLUT_F20EPNS_16FLUTTemplate_F20ItEE11FColorSpacej";

// The output pixmap only answers GetPixelType (vtable slot 2): 0xf = 48RGB.
int pixel_type_48rgb(void *) { return 0xf; }
void *fake_vtable[8] = {nullptr, nullptr, reinterpret_cast<void *>(&pixel_type_48rgb)};

uint64_t bits_of(double v) {
  uint64_t b;
  memcpy(&b, &v, 8);
  return b;
}
uint32_t bits_of(float v) {
  uint32_t b;
  memcpy(&b, &v, 4);
  return b;
}

// Generator shared with merrill_mono.rs: 32-bit LCG, high half.
struct Lcg {
  uint32_t s;
  uint16_t next() {
    s = s * 1664525u + 1013904223u;
    return static_cast<uint16_t>(s >> 16);
  }
};

// Table kinds: 0 = identity, 1 = LCG(seed), 2 = constant `seed`.
std::vector<uint16_t> table(uint32_t kind, uint32_t seed) {
  std::vector<uint16_t> t(65536);
  Lcg g{seed};
  for (uint32_t i = 0; i < 65536; i++)
    t[i] = kind == 0 ? static_cast<uint16_t>(i) : kind == 1 ? g.next() : static_cast<uint16_t>(seed);
  return t;
}

constexpr uint32_t kWidth = 65536;

// Pixel i of the single input row. R walks all 65536 patterns, G is a
// permutation of them; B (also the grain amplitude when grain is on) by mode.
void input_pixel(uint32_t i, uint32_t b_mode, int16_t *px) {
  px[0] = static_cast<int16_t>(static_cast<uint16_t>(i));
  px[1] = static_cast<int16_t>(static_cast<uint16_t>(i * 40503u + 12345u));
  int32_t b;
  switch (b_mode) {
    case 0: b = static_cast<int16_t>(static_cast<uint16_t>(i * 30011u + 777u)); break;
    case 1: b = static_cast<int32_t>((i * 37u) % 2001u) - 1000; break;
    default: b = 0; break;
  }
  px[2] = static_cast<int16_t>(b);
}
int16_t retinex_pixel(uint32_t i) { return static_cast<int16_t>(static_cast<uint16_t>(i * 2654435761u >> 7)); }

uint64_t fnv(const std::vector<uint16_t> &v) { return merrill::fnv1a(v.data(), v.size() * 2); }

struct RowCase {
  const char *name;
  bool mono, grain, retinex;
  double gains[3];
  double matrix[9];
  uint32_t tone_kind, tone_seed, gamma_kind, gamma_seed, retinex_kind, retinex_seed, b_mode;
};

// Stage4 @0xac848: float32 Y/Cb/Cr of the toning gains, stored as doubles.
void toning(const float *c, const double *g, double *out) {
  float r = static_cast<float>(g[0]), gg = static_cast<float>(g[1]), b = static_cast<float>(g[2]);
  // v3 = {0.587, -0.331} * G; fmla {0.299, -0.169} * R; fmla {0.114, 0.5} * B
  float y = c[0] * gg, cb = c[1] * gg;
  y = fmaf(c[2], r, y);
  cb = fmaf(c[3], r, cb);
  y = fmaf(c[4], b, y);
  cb = fmaf(c[5], b, cb);
  uint32_t k0 = 0xbed6872b, k1 = 0xbda5e354;
  float m0, m1;
  memcpy(&m0, &k0, 4);
  memcpy(&m1, &k1, 4);
  float cr = gg * m0;
  cr = fmaf(r, 0.5f, cr);
  cr = fmaf(b, m1, cr);
  out[0] = y;
  out[1] = cb;
  out[2] = cr;
}

}  // namespace

int main(int argc, char **argv) {
  merrill::Library lib;
  if (argc < 2 || !lib.open(argv[1])) return 1;
  auto process = lib.sym<ProcessToOutput>(kProcess);
  auto tone = lib.sym<GenerateTone>(kTone);
  auto gamma = lib.sym<CreateGamma>(kGamma);
  if (!process || !tone || !gamma) return 1;
  if (reinterpret_cast<uintptr_t>(process) != lib.base + 0x4f4e8 ||
      reinterpret_cast<uintptr_t>(tone) != lib.base + 0x57c64 ||
      reinterpret_cast<uintptr_t>(gamma) != lib.base + 0x561f4) {
    fprintf(stderr, "unexpected symbol addresses\n");
    return 1;
  }

  printf("# Merrill monochrome output stage, native outputs of the arm64 slice of\n");
  printf("# lib99_MerillDLLForMac.dylib. Generated by scripts/spp-native/output_mono.cpp;\n");
  printf("# inputs are synthetic (generator mirrored in src/merrill_mono.rs), doubles as\n");
  printf("# hex bits, outputs FNV-1a over little-endian u16 buffers. No Sigma data.\n");

  // --- Gamma LUT, CreateGammaLUT_F20(lut, colorspace, 16) ---------------------
  printf("# GAMMA colorspace hash\n");
  for (int cs : {1, 2, 3, 4, 8, 11, 12}) {
    std::vector<uint16_t> t(65536);
    Lut<uint16_t> l{t.data(), 0xffff, 0x10000, 0};
    gamma(&l, cs, 16);
    printf("GAMMA %d %016" PRIx64 "\n", cs, fnv(t));
  }

  // --- Tone curve, GenerateF20ToneCurve(psat, params, ex, sh, co, hi, cg, 16, lut, mono)
  printf("# TONE mono psat exposure shadows contrast highlights start end k1 k2 break scale"
         " | hash lut[1000] lut[4094] lut[65535]\n");
  struct ToneCase {
    bool mono;
    double psat, ex, sh, co, hi;
    double p[6];  // start end k1 k2 break scale
  };
  const double base[6] = {-1.17, 1.65, 3.0, 1.7, 0.1, 1.0};
  const double alt[6] = {-0.9, 1.4, 2.5, 2.1, 0.25, 0.8};
  const double nobreak[6] = {-1.2, 1.5, 3.1, 1.9, 1.0, 1.0};
  std::vector<ToneCase> tones;
  for (int mono = 0; mono < 2; mono++) {
    const double *ps[3] = {base, alt, nobreak};
    const double ctl[][5] = {
        {4095, 0, 0, 0, 0},       {4095, 0.5, 0, 0, 0},     {4095, 0, 1.0, 0, 0},
        {4095, 0, -1.0, 0, 0},    {4095, 0, 0, 0.7, 0},     {4095, 0, 0, -1.3, 0},
        {4095, 0, 0, 0, 1.0},     {4095, 0, 0, 0, -1.5},    {4095, -0.3, 2.5, 3.0, -2.5},
        {5000, 0.25, -0.4, 0.2, 0.6}, {4095, -1.0, 0.3, 0.3, 0.3},
    };
    for (int pi = 0; pi < 3; pi++)
      for (size_t ci = 0; ci < sizeof ctl / sizeof ctl[0]; ci++) {
        // every control on `base`; a few on the other parameter rows
        if (pi > 0 && ci != 0 && ci != 2 && ci != 7 && ci != 10) continue;
        const double *c = ctl[ci];
        ToneCase t{mono != 0, c[0], c[1], c[2], c[3], c[4], {}};
        memcpy(t.p, ps[pi], sizeof t.p);
        tones.push_back(t);
      }
  }
  for (const auto &t : tones) {
    std::vector<uint16_t> r(65536, 0xaaaa), g(65536, 0xbbbb), b(65536, 0xcccc);
    Clut clut{{r.data(), g.data(), b.data()}, 0xffff, 0x10000, 0};
    ToneParams tp{};
    tp.v[0] = t.p[0];
    tp.v[1] = t.p[1];
    tp.v[2] = t.p[2];
    tp.v[3] = t.p[3];
    tp.v[4] = t.p[4];
    tp.v[8] = t.p[5];
    tone(t.psat, &tp, t.ex, t.sh, t.co, t.hi, FColorFloat{0.9, 1.1, 0.7}, 16, &clut, t.mono);
    if (r != g || r != b) {
      fprintf(stderr, "tone channels differ\n");
      return 1;
    }
    printf("TONE %d", t.mono ? 1 : 0);
    for (double v : {t.psat, t.ex, t.sh, t.co, t.hi}) printf(" %016" PRIx64, bits_of(v));
    for (double v : t.p) printf(" %016" PRIx64, bits_of(v));
    printf(" | %016" PRIx64 " %u %u %u\n", fnv(r), r[1000], r[4094], r[65535]);
  }

  // --- Stage4 toning conversion (float32, same FMA chain) ----------------------
  float consts[6];
  memcpy(consts, lib.at(0xbf3b0), sizeof consts);
  printf("# TONING constants(0xbf3b0) %08x %08x %08x %08x %08x %08x\n", bits_of(consts[0]),
         bits_of(consts[1]), bits_of(consts[2]), bits_of(consts[3]), bits_of(consts[4]),
         bits_of(consts[5]));
  printf("# TONING r g b | y cb cr (not a library call: see header)\n");
  const double toning_in[][3] = {
      {1, 1, 1},           {1.0, 0.86, 0.66},   {1.05, 0.95, 0.75},  {0.8, 0.9, 1.2},
      {2, 0.5, 0.25},      {32768.0 / 32768, 30000.0 / 32768, 25000.0 / 32768},
      {65535.0 / 32768, 0, 1.0 / 32768},
  };
  for (const auto &g : toning_in) {
    double o[3];
    toning(consts, g, o);
    printf("TONING %016" PRIx64 " %016" PRIx64 " %016" PRIx64 " | %016" PRIx64 " %016" PRIx64
           " %016" PRIx64 "\n",
           bits_of(g[0]), bits_of(g[1]), bits_of(g[2]), bits_of(o[0]), bits_of(o[1]),
           bits_of(o[2]));
  }

  // --- ProcessToOutput rows ---------------------------------------------------
  const double ident[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  const double mixed[9] = {0.62, 0.31, 0.07, -0.21, 1.18, 0.03, 0.02, -0.4, 1.38};
  const double luma[9] = {0.299, 0.587, 0.114, 0.299, 0.587, 0.114, 0.299, 0.587, 0.114};
  const double wrap[9] = {30000.0, -30000.0, 0.5, -0.49999, 1.0 / 8192, 3.0, 0, 0, -1.0};
  double y1[3], sep[3], warm[3], cool[3];
  {
    const double a[3] = {1, 1, 1}, b[3] = {1.0, 0.86, 0.66}, c[3] = {1.05, 0.95, 0.75},
                 d[3] = {0.8, 0.9, 1.2};
    toning(consts, a, y1);
    toning(consts, b, sep);
    toning(consts, c, warm);
    toning(consts, d, cool);
  }
  std::vector<RowCase> rows = {
      // name mono grain retinex gains matrix tone(kind,seed) gamma(kind,seed) retinex(kind,seed) b
      {"mono_ident_id", true, false, false, {y1[0], y1[1], y1[2]}, {}, 0, 0, 0, 0, 2, 256, 0},
      {"mono_lcg", true, false, false, {y1[0], y1[1], y1[2]}, {}, 1, 11, 1, 21, 2, 256, 0},
      {"mono_lcg_luma", true, false, false, {y1[0], y1[1], y1[2]}, {}, 1, 12, 1, 22, 2, 256, 0},
      {"mono_sepia", true, false, false, {sep[0], sep[1], sep[2]}, {}, 1, 13, 1, 23, 2, 256, 0},
      {"mono_sepia_id", true, false, false, {sep[0], sep[1], sep[2]}, {}, 0, 0, 0, 0, 2, 256, 0},
      {"mono_warm", true, false, true, {warm[0], warm[1], warm[2]}, {}, 1, 14, 1, 24, 2, 256, 0},
      {"mono_cool_retinex", true, false, true, {cool[0], cool[1], cool[2]}, {}, 1, 15, 1, 25, 1, 31, 0},
      {"mono_raw_gains", true, false, false, {3.0, -0.4, 0.9}, {}, 1, 16, 1, 26, 2, 256, 0},
      {"mono_huge_chroma", true, false, false, {0.0, 1e6, -1e6}, {}, 1, 17, 1, 27, 2, 256, 0},
      {"mono_wrap", true, false, false, {y1[0], y1[1], y1[2]}, {}, 1, 18, 1, 28, 2, 256, 0},
      {"mono_grain", true, true, false, {y1[0], y1[1], y1[2]}, {}, 1, 19, 1, 29, 2, 256, 1},
      {"mono_grain_full", true, true, false, {sep[0], sep[1], sep[2]}, {}, 1, 20, 1, 30, 2, 256, 0},
      {"mono_grain_retinex", true, true, true, {warm[0], warm[1], warm[2]}, {}, 1, 21, 1, 31, 1, 32, 1},
      {"mono_grain_id", true, true, false, {y1[0], y1[1], y1[2]}, {}, 0, 0, 0, 0, 2, 256, 1},
      {"color_ident", false, false, false, {1, 1, 1}, {}, 0, 0, 0, 0, 2, 256, 0},
      {"color_lcg", false, false, false, {1, 1, 1}, {}, 1, 41, 1, 51, 2, 256, 0},
      {"color_gains", false, false, true, {1.7, 0.6, -2.2}, {}, 1, 42, 1, 52, 2, 256, 0},
      {"color_retinex", false, false, true, {1, 1, 1}, {}, 1, 43, 1, 53, 1, 33, 0},
      {"color_huge_gains", false, false, false, {1e12, -1e12, 70000.0}, {}, 1, 44, 1, 54, 2, 256, 0},
      {"color_wrap", false, false, false, {1, 1, 1}, {}, 1, 45, 1, 55, 2, 256, 2},
      {"color_grain", false, true, false, {1.2, 0.8, 1.0}, {}, 1, 46, 1, 56, 2, 256, 1},
  };
  for (auto &c : rows) {
    const double *m = mixed;
    if (strstr(c.name, "ident") || strstr(c.name, "_id")) m = ident;
    if (strstr(c.name, "luma") || strstr(c.name, "sepia") || strstr(c.name, "grain")) m = luma;
    if (strstr(c.name, "wrap")) m = wrap;
    memcpy(c.matrix, m, sizeof c.matrix);
  }

  printf("# ROW name mono grain retinex g0 g1 g2 m0..m8 tone_kind tone_seed gamma_kind gamma_seed"
         " retinex_kind retinex_seed b_mode | hash px0 px1 px32768 px65535\n");
  printf("# Tone channel c uses seed+c; retinex LUT kind 2 is the constant `seed` (256 = unit).\n");
  for (const auto &c : rows) {
    std::vector<int16_t> in(kWidth * 3), ret(kWidth);
    for (uint32_t i = 0; i < kWidth; i++) {
      input_pixel(i, c.b_mode, &in[i * 3]);
      ret[i] = retinex_pixel(i);
    }
    auto t0 = table(c.tone_kind, c.tone_seed), t1 = table(c.tone_kind, c.tone_seed + 1),
         t2 = table(c.tone_kind, c.tone_seed + 2), gm = table(c.gamma_kind, c.gamma_seed),
         rl = table(c.retinex_kind, c.retinex_seed), dummy = table(1, 999);
    Clut clut{{t0.data(), t1.data(), t2.data()}, 0xffff, 0x10000, 0};
    Lut<int16_t> rlut{reinterpret_cast<int16_t *>(rl.data()), 0xffff, 0x10000, 0};
    Lut<uint16_t> glut{gm.data(), 0xffff, 0x10000, 0};
    Lut<uint16_t> ulut{dummy.data(), 0xffff, 0x10000, 0};
    merrill::Pixmap pin{fake_vtable, in.data(), kWidth, 1, static_cast<int32_t>(kWidth * 6), 0};
    merrill::Pixmap pret{fake_vtable, ret.data(), kWidth, 1, static_cast<int32_t>(kWidth * 2), 0};
    std::vector<uint16_t> out(kWidth * 3 + 8, 0x5a5a);
    merrill::Pixmap pout{fake_vtable, out.data(), kWidth, 1, static_cast<int32_t>(kWidth * 6), 0};
    double matrix[16] = {};
    memcpy(matrix, c.matrix, sizeof c.matrix);
    // d3, d4 and s5 are not read by the 48RGB row; pass junk to show it.
    process(&pin, c.retinex ? &pret : nullptr, &clut, &rlut, &glut, &ulut, matrix,
            FColorFloat{c.gains[0], c.gains[1], c.gains[2]}, 123.0, -7.5, &pout, c.grain, c.mono,
            0.375f);
    for (size_t i = kWidth * 3; i < out.size(); i++)
      if (out[i] != 0x5a5a) {
        fprintf(stderr, "%s: wrote past the row\n", c.name);
        return 1;
      }
    out.resize(kWidth * 3);
    printf("ROW %s %d %d %d", c.name, c.mono, c.grain, c.retinex);
    for (double v : c.gains) printf(" %016" PRIx64, bits_of(v));
    for (double v : c.matrix) printf(" %016" PRIx64, bits_of(v));
    printf(" %u %u %u %u %u %u %u", c.tone_kind, c.tone_seed, c.gamma_kind, c.gamma_seed,
           c.retinex_kind, c.retinex_seed, c.b_mode);
    printf(" | %016" PRIx64, fnv(out));
    for (uint32_t p : {0u, 1u, 32768u, 65535u})
      printf(" %u,%u,%u", out[p * 3], out[p * 3 + 1], out[p * 3 + 2]);
    printf("\n");
  }
  return 0;
}
