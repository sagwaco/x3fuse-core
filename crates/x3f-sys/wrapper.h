// bindgen entrypoint. Pulls in every public header of the legacy C library
// that we need to expose to Rust callers. New headers should be added here.
//
// This file is consumed only by build.rs; it is not part of the C build.

#include "x3f_io.h"
#include "x3f_process.h"
#include "x3f_meta.h"
#include "x3f_print_meta.h"
#include "x3f_histogram.h"
#include "x3f_printf.h"
#include "x3f_image.h"
#include "x3f_matrix.h"
#include "x3f_spatial_gain.h"
