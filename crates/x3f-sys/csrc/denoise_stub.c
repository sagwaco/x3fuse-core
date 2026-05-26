/* denoise_stub.c
 *
 * The Rust port drops the OpenCV-backed denoise path so the project compiles
 * without a 50+MB C++ image-processing dep. Legacy C call sites still
 * reference these symbols, so we provide stubs:
 *
 *   - x3f_denoise:         silent no-op. Image passes through unchanged.
 *                          Means `-no-denoise` and the default both produce
 *                          the un-denoised result. Documented in README.
 *                          A pure-Rust port returns in M9.
 *   - x3f_set_use_opencl:  no-op. The `-ocl` CLI flag becomes a silent
 *                          no-op until M9 brings GPU paths back.
 *
 * `x3f_expand_quattro` was previously a stub here too (exit(2)). In M5 it
 * was replaced with a native Rust implementation; see crates/x3f-sys/src/
 * quattro.rs. The symbol resolves at link time to the `#[no_mangle] extern
 * "C"` definition there.
 */

#include "x3f_denoise.h"

void x3f_denoise(x3f_area16_t *image, x3f_denoise_type_t type, float scale)
{
  (void)image;
  (void)type;
  (void)scale;
}

void x3f_denoise_active(x3f_area16_t *area, x3f_denoise_type_t type, int stage,
                        float scale)
{
  (void)area;
  (void)type;
  (void)stage;
  (void)scale;
}

void x3f_set_use_opencl(int flag)
{
  (void)flag;
}
