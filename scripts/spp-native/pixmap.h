// Minimal mirror of the Merrill engine's FPixmapTemplate_F20 objects, enough to
// hand synthetic images to exported functions of lib99_MerillDLLForMac.
//
// Layout from the decompiled engine (see piano-spp-monocromo-fpn.md):
//   +0x00 vtable, +0x08 data, +0x10 width, +0x14 height, +0x18 row bytes.
// The constructors are not exported, so the vtable pointer is taken from the
// loaded library: base + offset of the matching vtable in __DATA.
//
// Nothing here contains Sigma code or data; the library is loaded at run time
// from reference/ (gitignored).

#pragma once
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <mach-o/dyld.h>
#include <cstdio>

namespace merrill {

struct Pixmap {
  void *vtable;
  void *data;
  uint32_t width;
  uint32_t height;
  int32_t row_bytes;
  int32_t pad;
};
static_assert(sizeof(Pixmap) == 0x20, "native pixmaps are 0x20 bytes");

struct Library {
  void *handle = nullptr;
  uintptr_t base = 0;

  bool open(const char *path) {
    handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!handle) {
      fprintf(stderr, "dlopen: %s\n", dlerror());
      return false;
    }
    // GetTanhLUT is exported at 0x5d99c in the reference build.
    void *probe = dlsym(handle, "_ZN15FPixelProc2_F2010GetTanhLUTERK11FColorFloatS2_");
    if (!probe) {
      fprintf(stderr, "probe symbol missing: %s\n", dlerror());
      return false;
    }
    base = reinterpret_cast<uintptr_t>(probe) - 0x5d99c;
    return true;
  }

  void *at(uintptr_t offset) const { return reinterpret_cast<void *>(base + offset); }

  template <typename F> F sym(const char *mangled) const {
    void *p = dlsym(handle, mangled);
    if (!p) fprintf(stderr, "missing %s\n", mangled);
    return reinterpret_cast<F>(p);
  }
};

inline Pixmap make(void *vtable, uint32_t width, uint32_t height, uint32_t channels) {
  Pixmap p{};
  p.vtable = vtable;
  p.width = width;
  p.height = height;
  p.row_bytes = static_cast<int32_t>(width * channels * sizeof(float));
  p.data = calloc(static_cast<size_t>(width) * height * channels, sizeof(float));
  return p;
}

inline uint64_t fnv1a(const void *bytes, size_t n) {
  const uint8_t *b = static_cast<const uint8_t *>(bytes);
  uint64_t h = 0xcbf29ce484222325ull;
  for (size_t i = 0; i < n; i++) {
    h ^= b[i];
    h *= 0x100000001b3ull;
  }
  return h;
}

}  // namespace merrill
