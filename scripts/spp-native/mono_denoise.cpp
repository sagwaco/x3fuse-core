// End-to-end oracle for FPixelProc2_F20::MonochromeDenoise (0x2fd3c): runs the
// native function on a deterministic synthetic image and prints hashes of every
// output, optionally dumping the raw planes for a Rust comparison.
//
//   clang++ -std=c++17 -arch arm64 -O1 -ffp-contract=off scripts/spp-native/mono_denoise.cpp -o /tmp/mono_denoise
//   /tmp/mono_denoise <lib99 arm64 slice> [dump-dir]
//
// NoiseParams comes from the first IMAGE line of the upstream fixture
// crates/x3f-sys/tests/fixtures/spp-image-native.txt (native SetUpNoiseParamsF20
// output for a synthetic dp2 case), with the monochrome byte (+0x186) set.

#include "pixmap.h"

#include <cmath>
#include <string>
#include <vector>

using namespace merrill;

namespace {

// FRect is a non-trivial C++ class: passed by invisible reference (x5 holds a
// pointer to the 16 bytes), so the typedef takes a pointer.
struct FRect {
  int32_t left, top, right, bottom;
};
struct FColorFloat {
  double r, g, b;
};

typedef void (*MonochromeDenoise)(Pixmap *, Pixmap *, Pixmap *, void *, unsigned, const FRect *, uint8_t *,
                                  FColorFloat, bool, double, double, bool *, void *, bool);
typedef void (*ProgressRootCtor)(void *, double *);
typedef void (*MultiStepCtor)(void *, void *, unsigned, double *, bool);
typedef void (*SetRoot)(void *, void *);

std::vector<uint8_t> noise_params(const std::string &fixture) {
  // The full 0x1c0-byte blob is read from the fixture at run time.
  FILE *f = fopen(fixture.c_str(), "r");
  if (!f) return {};
  char line[8192];
  while (fgets(line, sizeof line, f)) {
    if (strncmp(line, "IMAGE dp2-s1 ", 13) != 0) continue;
    char *tok = strtok(line, " \n");
    for (int i = 0; tok && i < 14; i++) tok = strtok(nullptr, " \n");
    if (!tok || strlen(tok) != 0x1c0 * 2) break;
    std::vector<uint8_t> out(0x1c0);
    for (size_t i = 0; i < out.size(); i++) sscanf(tok + 2 * i, "%2hhx", &out[i]);
    fclose(f);
    return out;
  }
  fclose(f);
  return {};
}

float synthetic(uint32_t x, uint32_t y, uint32_t c, uint32_t seed) {
  // Smooth ramps, a few hard edges and hashed noise, in native count units.
  uint32_t h = (x * 73856093u) ^ (y * 19349663u) ^ (c * 83492791u) ^ (seed * 2654435761u);
  h ^= h >> 13;
  h *= 0x5bd1e995u;
  h ^= h >> 15;
  float noise = (float)(h % 2001) - 1000.f;
  float ramp = 300.f + 18.f * x + 7.f * y + 400.f * c;
  float edge = ((x / 23 + y / 17) & 1) ? 1800.f : 0.f;
  return std::fmax(0.f, ramp + edge + 0.35f * noise);
}

}  // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: %s <lib99 arm64 slice> [dump-dir]\n", argv[0]);
    return 2;
  }
  Library lib;
  if (!lib.open(argv[1])) return 1;
  std::string fixture =
      "crates/x3f-sys/tests/fixtures/spp-image-native.txt";
  std::vector<uint8_t> noise = noise_params(fixture);
  if (noise.empty()) {
    fprintf(stderr, "dp2 NoiseParams not found in %s (run from the repo root)\n", fixture.c_str());
    return 1;
  }
  noise[0x186] = 1;

  auto denoise = lib.sym<MonochromeDenoise>(
      "_ZN15FPixelProc2_F2017MonochromeDenoiseEP19FPixmapTemplate_F20I23FPixelClassFlt32RGB_F20EPS0_"
      "I22FPixelClass16SMono_F20ES3_Pvj5FRectPNS_11NoiseParamsE11FColorFloatbddPbP21MultiStepProgress_"
      "F20b");
  auto root_ctor = lib.sym<ProgressRootCtor>("_ZN16ProgressRoot_F20C1EPd");
  auto multi_ctor = lib.sym<MultiStepCtor>("_ZN21MultiStepProgress_F20C1EP16ProgressRoot_F20jPdb");
  auto set_root = lib.sym<SetRoot>("_ZN16ProgressRoot_F207SetRootEP12Progress_F20");
  if (!denoise || !root_ctor || !multi_ctor || !set_root) return 1;

  struct Case {
    uint32_t w, h;
    double wb[3], mix[3];
    bool adjust;
    double div[2];
  };
  // Production-like white balance (GatherParams normalizes by the minimum),
  // zero and non-zero mixer weights, the noise adjustment, and an image tall
  // enough for 8 DoThreadedPix bands.
  const Case cases[] = {
      {96, 64, {0, 0, 0}, {0, 0, 0}, false, {1, 1}},
      {131, 77, {0, 0, 0}, {0, 0, 0}, false, {1, 1}},
      {96, 64, {1.0, 1.21, 1.87}, {0, 0, 0}, false, {1, 1}},
      {131, 97, {1.0, 1.21, 1.87}, {0.3, 0.5, 0.2}, false, {1, 1}},
      {120, 700, {1.0, 1.35, 2.05}, {0, 0, 0}, false, {1, 1}},
      {77, 650, {1.13, 1.0, 1.64}, {0.2, 0.0, 0.9}, true, {0.75, 0.5}},
  };
  printf("# CASE width height wb[3] mix[3] adjust div[2] (f64 bits) | fnv(out) fnv(sharp)\n");
  for (const Case &k : cases) {
    const uint32_t w = k.w, h = k.h;
    Pixmap out = make(lib.at(0xd0ca0), w, h, 3);
    Pixmap src = make(lib.at(0xd0ca0), w, h, 3);
    // 16SMono: int16 samples, row bytes = 2 * width.
    Pixmap sharp{};
    sharp.vtable = lib.at(0xd0d28);  // FPixmapTemplate_F20<FPixelClass16SMono_F20>
    sharp.width = w;
    sharp.height = h;
    sharp.row_bytes = (int32_t)(w * 2);
    sharp.data = calloc((size_t)w * h, 2);
    float *in = (float *)src.data;
    float *o = (float *)out.data;
    for (uint32_t y = 0; y < h; y++)
      for (uint32_t x = 0; x < w; x++)
        for (uint32_t c = 0; c < 3; c++) {
          in[(y * w + x) * 3 + c] = synthetic(x, y, c, w);
          o[(y * w + x) * 3 + c] = in[(y * w + x) * 3 + c];
        }

    alignas(16) uint8_t root[256] = {};
    alignas(16) uint8_t multi[256] = {};
    double progress = 0, weights[4] = {1, 1, 1, 1};
    root_ctor(root, &progress);
    multi_ctor(multi, root, 4, weights, false);
    set_root(root, multi);

    std::vector<uint8_t> params = noise;
    memcpy(params.data() + 0x188, k.mix, 24);
    memcpy(params.data() + 0x1a8, k.wb, 24);
    bool cancel = false;
    FRect rect{0, 0, (int32_t)w, (int32_t)h};
    FColorFloat gains{1.711, 1.818, 2.410};
    denoise(&out, &sharp, &src, nullptr, 0, &rect, params.data(), gains, false, k.div[0], k.div[1],
            &cancel, multi, k.adjust);

    auto bits = [](double v) {
      uint64_t u;
      memcpy(&u, &v, 8);
      return (unsigned long long)u;
    };
    printf("CASE %u %u %016llx %016llx %016llx %016llx %016llx %016llx %d %016llx %016llx %016llx %016llx\n",
           w, h, bits(k.wb[0]), bits(k.wb[1]), bits(k.wb[2]), bits(k.mix[0]), bits(k.mix[1]),
           bits(k.mix[2]), k.adjust ? 1 : 0, bits(k.div[0]), bits(k.div[1]),
           (unsigned long long)fnv1a(out.data, (size_t)w * h * 12),
           (unsigned long long)fnv1a(sharp.data, (size_t)w * h * 2));
    if (!(o[0] == o[1] && o[1] == o[2])) fprintf(stderr, "not gray\n");
    if (argc > 2) {
      std::string base = std::string(argv[2]) + "/mono-" + std::to_string(w) + "x" + std::to_string(h);
      FILE *f = fopen((base + ".out.f32").c_str(), "wb");
      fwrite(out.data, 4, (size_t)w * h * 3, f);
      fclose(f);
      f = fopen((base + ".src.f32").c_str(), "wb");
      fwrite(src.data, 4, (size_t)w * h * 3, f);
      fclose(f);
    }
  }
  return 0;
}
