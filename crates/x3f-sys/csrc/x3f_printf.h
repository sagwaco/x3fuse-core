/* X3F_PRINTF.H
 *
 * Library for printout, depending on seriousity level.
 *
 * Copyright 2015 - Roland and Erik Karlsson
 * BSD-style - see doc/copyright.txt
 *
 */

#ifndef X3F_PRINTF_H
#define X3F_PRINTF_H

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {ERR=0, WARN=1, INFO=2, DEBUG=3} x3f_verbosity_t;

extern x3f_verbosity_t x3f_printf_level;

/* `__attribute__((format(printf, ...)))` is a GCC/Clang extension that lets
 * the compiler typecheck the variadic args against `fmt`. MSVC's cl.exe
 * doesn't understand it (it's a syntax error there), so expand it away on
 * non-GNU compilers — it's a diagnostic hint, not part of the ABI. */
#if defined(__GNUC__) || defined(__clang__)
#  define X3F_PRINTF_FORMAT(fmt_idx, args_idx) \
     __attribute__((format(printf, fmt_idx, args_idx)))
#else
#  define X3F_PRINTF_FORMAT(fmt_idx, args_idx)
#endif

extern void x3f_printf(x3f_verbosity_t level, const char *fmt, ...)
  X3F_PRINTF_FORMAT(2, 3);

/* Optional log routing hook. If non-NULL, x3f_printf formats the message
 * into a 2 KiB stack buffer and forwards it here instead of writing to
 * stdout/stderr. Used by Rust consumers (mobile/WASM/library embeds) to
 * capture diagnostic output. Default NULL = legacy stdout/stderr behaviour.
 *
 * The callback receives the verbosity level and a NUL-terminated message
 * string with no level prefix and no trailing newline normalisation. */
typedef void (*x3f_printf_callback_t)(x3f_verbosity_t level, const char *msg);
extern x3f_printf_callback_t x3f_printf_callback;

#ifdef __cplusplus
}
#endif

#endif
