/* X3F_DENOISE.CPP
 *
 * Library for denoising of X3F image data.
 *
 * Copyright 2015 - Roland and Erik Karlsson
 * BSD-style - see doc/copyright.txt
 *
 */

#include <iostream>
#include <inttypes.h>

#include <opencv2/core.hpp>
#include <opencv2/photo.hpp>
#include <opencv2/imgproc.hpp>
// opencv-mobile strips the OpenCL module, so `<opencv2/core/ocl.hpp>` and
// the `cv::UMat` / `cv::ocl::*` API are not available. The Rust port falls
// back to the plain `cv::Mat` CPU path; the `-ocl` CLI flag becomes a
// silent no-op (see `x3f_set_use_opencl` below).

#include "x3f_denoise_utils.h"
#include "x3f_denoise.h"
#include "x3f_io.h"
#include "x3f_printf.h"

using namespace cv;

static void denoise_nlm(Mat& img, float h)
{
  Mat out, sub, sub_dn, sub_res, res;
  float h1[3] = {0.0, h, h}, h2[3] = {0.0, h/8, h/4};

  x3f_printf(DEBUG, "BEGIN denoising\n");
  fastNlMeansDenoising(img, out, std::vector<float>(h1, h1+3),
		       3, 11, NORM_L1);
  x3f_printf(DEBUG, "END denoising\n");

  x3f_printf(DEBUG, "BEGIN V median filtering\n");
  Mat V(out.size(), CV_16U);
  int get_V[2] = { 2,0 }, set_V[2] = { 0,2 };
  mixChannels(std::vector<Mat>(1, out), std::vector<Mat>(2, V), get_V, 1);
  medianBlur(V, V, 3);
  mixChannels(std::vector<Mat>(1, V), std::vector<Mat>(2, out), set_V, 1);
  x3f_printf(DEBUG, "END V median filtering\n");

  x3f_printf(DEBUG, "BEGIN low-frequency denoising\n");
  resize(out, sub, Size(), 1.0/4, 1.0/4, INTER_AREA);
  fastNlMeansDenoising(sub, sub_dn, std::vector<float>(h2, h2+3),
		       3, 21, NORM_L1);
  subtract(sub, sub_dn, sub_res, noArray(), CV_16S);
  resize(sub_res, res, out.size(), 0.0, 0.0, INTER_CUBIC);
  subtract(out, res, out, noArray(), CV_16U);
  x3f_printf(DEBUG, "END low-frequency denoising\n");

  out.copyTo(img);
}

void x3f_denoise_active(x3f_area16_t *area, x3f_denoise_type_t type, int stage,
                        float scale)
{
  // Entry point invoked by the Rust `x3f_expand_quattro` upsampler in
  // crates/x3f-sys/src/quattro.rs. Sees the area already in YUV layout —
  // we never touch BMT/YUV transforms here, the Rust caller owns those.
  assert(area->channels == 3);
  assert(type < sizeof(denoise_types)/sizeof(denoise_desc_t));
  const denoise_desc_t *d = &denoise_types[type];
  // `scale` (0..1) attenuates the per-sensor sigma; 1.0 == legacy strength.
  float sigma = d->h * scale;

  Mat act(area->rows, area->columns, CV_16UC3,
          area->data, sizeof(uint16_t)*area->row_stride);

  if (stage == 0) {
    // Half-res pre-upsample pass: full denoise_nlm pipeline.
    denoise_nlm(act, sigma);
  } else {
    // Full-res post-upsample pass: just fastNlMeansDenoising with the
    // legacy per-channel weights {0, h, h*2}.
    Mat out;
    float h[3] = {0.0f, sigma, sigma * 2.0f};
    x3f_printf(DEBUG, "BEGIN Quattro full-resolution denoising\n");
    fastNlMeansDenoising(act, out, std::vector<float>(h, h+3),
                         3, 11, NORM_L1);
    x3f_printf(DEBUG, "END Quattro full-resolution denoising\n");
    out.copyTo(act);
  }
}

void x3f_denoise(x3f_area16_t *image, x3f_denoise_type_t type, float scale)
{
  assert(image->channels == 3);
  assert(type < sizeof(denoise_types)/sizeof(denoise_desc_t));
  const denoise_desc_t *d = &denoise_types[type];

  d->BMT_to_YUV(image);

  Mat img(image->rows, image->columns, CV_16UC3,
	 image->data, sizeof(uint16_t)*image->row_stride);
  // `scale` (0..1) attenuates the per-sensor sigma; 1.0 == legacy strength.
  denoise_nlm(img, d->h * scale);

  d->YUV_to_BMT(image);
}

// x3f_expand_quattro is now implemented in Rust (src/quattro.rs) and
// exported as the canonical `#[no_mangle]` symbol; it orchestrates the
// resize + BMT/YUV transforms there and calls back into x3f_denoise_active
// above for the NLM passes. The legacy C++ body that used to live here was
// removed: keeping it produced a duplicate `x3f_expand_quattro` definition
// that GNU ld / lld reject at link time (only macOS's ld64 tolerated it).

void x3f_set_use_opencl(int flag)
{
  // opencv-mobile (https://github.com/nihui/opencv-mobile) strips the
  // OpenCL backend, so this is a no-op. The `-ocl` CLI flag is preserved
  // for command-line backwards compatibility but does nothing.
  if (flag) x3f_printf(WARN, "OpenCL is not available in this build\n");
  else x3f_printf(DEBUG, "OpenCL is disabled\n");
}
