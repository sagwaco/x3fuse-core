/* X3F_DENOISE.H
 *
 * Library for denoising of X3F image data.
 *
 * Copyright 2015 - Roland and Erik Karlsson
 * BSD-style - see doc/copyright.txt
 *
 */

#ifndef X3F_DENOISE_H
#define X3F_DENOISE_H

#include "x3f_io.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  X3F_DENOISE_STD=0,
  X3F_DENOISE_F20=1,
  X3F_DENOISE_F23=2,
} x3f_denoise_type_t;

extern void x3f_denoise(x3f_area16_t *image, x3f_denoise_type_t type);
extern void x3f_expand_quattro(x3f_area16_t *image,
			       x3f_area16_t *active,
			       x3f_area16_t *qtop,
			       x3f_area16_t *expanded,
			       x3f_area16_t *active_exp);

/* x3f_denoise_active: NLM passes the Rust `x3f_expand_quattro` upsampler
 * calls into for Quattro files. The legacy C++ `x3f_expand_quattro` ran
 * two NLM passes inside its own body — once on the half-res active region
 * (YUV-encoded, before bicubic upsample) and once on the full-res active
 * region (YUV-encoded, after qtop merge). The Rust port (M5a) kept the
 * resize + BMT/YUV transforms but elided the NLM calls; this entry point
 * brings them back without dragging the upsampler back into C++.
 *
 *   stage=0: pre-upsample. Runs the full `denoise_nlm` pipeline (NLM +
 *            V-channel median + low-frequency subtraction) at sigma h.
 *   stage=1: post-upsample. Runs only `fastNlMeansDenoising` with
 *            per-channel weights {0, h, h*2} — matches the legacy
 *            "Quattro full-resolution denoising" inner block.
 *
 * `type` selects `denoise_types[type]` for sigma. For Quattro, callers
 * pass X3F_DENOISE_F23. */
extern void x3f_denoise_active(x3f_area16_t *area,
			       x3f_denoise_type_t type,
			       int stage);

extern void x3f_set_use_opencl(int flag);

#ifdef __cplusplus
}
#endif

#endif
