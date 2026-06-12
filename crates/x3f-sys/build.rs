// build.rs — compile the tiny remaining C surface and emit Rust bindings.
//
// The X3F decode + processing pipeline (container parsing, CAMF, entropy
// decode, image processing, matrix math, histogram, metadata, denoise, and
// PPM/TIFF/DNG output) is all native Rust. The only C left lives in this
// crate's `csrc/` directory, kept self-contained so `cargo publish` packages
// it:
//
// What's compiled:
//  - csrc/x3f_printf.c, csrc/x3f_version.c — the log callback + version shim.
//    On wasm32 cc-rs is skipped entirely (Apple's bundled clang has no
//    wasm-libc sysroot); the variadic `x3f_printf` symbol referenced by the
//    bindgen output is satisfied by the Rust shim in `src/wasm_c_shims.rs`.
//  - bindgen against wrapper.h emits a single bindings.rs for the C
//    struct/enum layouts the Rust port mirrors via #[repr(C)].
//
// Denoise is pure Rust (`src/denoise.rs`), called directly by the pipeline on
// every target — no C/C++ and no FFI boundary.

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Translate a Rust target triple to the clang `-target` value bindgen
/// (and the iOS link line) expects. Rust uses `-sim` and `-macabi`
/// suffixes; clang uses `-simulator` and `-macabi`. Returns `None` for
/// host-style targets where no override is needed.
fn clang_target_for(rust_target: &str) -> Option<String> {
    if rust_target == "aarch64-apple-ios-sim" {
        Some("arm64-apple-ios-simulator".into())
    } else if rust_target == "x86_64-apple-ios-sim" || rust_target == "x86_64-apple-ios" {
        // Rust still uses `x86_64-apple-ios` for the simulator on Intel.
        Some("x86_64-apple-ios-simulator".into())
    } else {
        None
    }
}

/// wasm32 build path. The bulk of x3f-sys is pure Rust now (M4-M6); we
/// skip cc-rs entirely on wasm32 — Apple's bundled clang has no
/// wasi-libc / wasm-libc sysroot, and the workspace's only remaining
/// C files (`x3f_printf.c`, `x3f_version.c`) `#include <stdio.h>` /
/// `<inttypes.h>`. The variadic `x3f_printf` symbol is provided by a
/// `#[no_mangle]` Rust shim in `src/wasm_c_shims.rs`; denoise is pure Rust
/// in `src/denoise.rs`, called directly by the pipeline.
///
/// We still run bindgen so `x3f_t` / `x3f_directory_entry_t` /
/// `x3f_image_data_t` and friends keep their type definitions; the
/// underlying integer widths (`uint32_t`, `c_int`, etc.) are identical
/// on wasm32-unknown-unknown / wasip1 and the build host (both 32-bit
/// data model), so we run bindgen against the host clang headers and
/// the resulting bindings.rs is layout-compatible.
fn compile_wasm_only(manifest_dir: &Path, c_src: &Path, out_dir: &Path, host: &str) {
    println!("cargo:rerun-if-changed=wrapper.h");

    let bindings = run_bindgen(manifest_dir, c_src, /*clang_target=*/ Some(host));
    bindings
        .write_to_file(out_dir.join("bindings.rs"))
        .expect("write bindings.rs");
}

/// Build the full bindgen Builder, applied identically on every target.
/// `clang_target` is what we pass via `--target=…`; for wasm32 we override
/// to the host triple so `<inttypes.h>` resolves through the host's
/// libc/libclang headers rather than the missing wasm32 sysroot.
fn run_bindgen(manifest_dir: &Path, c_src: &Path, clang_target: Option<&str>) -> bindgen::Bindings {
    let mut b = bindgen::Builder::default()
        .header(manifest_dir.join("wrapper.h").to_string_lossy().to_string())
        .clang_arg(format!("-I{}", c_src.display()))
        .allowlist_file(format!("{}/x3f_.*\\.h", c_src.display()));
    if let Some(t) = clang_target {
        b = b.clang_arg(format!("--target={t}"));
    }
    apply_bindgen_blocklists(b)
        .derive_debug(true)
        .derive_default(true)
        .layout_tests(false)
        .generate_comments(false)
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("bindgen failed")
}

/// Resolve `xcrun --sdk <name> --show-sdk-path` to the absolute SDK
/// directory. Returns `None` if xcrun isn't on PATH or the SDK isn't
/// installed (the bindgen step then falls back to whatever `--target`
/// alone resolves, which generally still works for non-Apple builds).
fn apple_sdk_path(sdk: &str) -> Option<PathBuf> {
    let out = Command::new("xcrun")
        .args(["--sdk", sdk, "--show-sdk-path"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8(out.stdout).ok()?;
    let trimmed = s.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(PathBuf::from(trimmed))
    }
}

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    // All remaining C (the two log/version shims and the headers bindgen
    // consumes) lives inside the crate at `csrc/` so `cargo publish` packages
    // a self-contained crate.
    let c_src = manifest_dir.join("csrc");
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let target = env::var("TARGET").unwrap();
    let host = env::var("HOST").unwrap();
    let is_wasm = target.starts_with("wasm32");

    println!("cargo:rerun-if-changed=wrapper.h");
    println!("cargo:rerun-if-changed={}", c_src.display());

    let version = env::var("CARGO_PKG_VERSION").unwrap();

    // wasm32-unknown-unknown / wasm32-wasip* targets either lack a libc
    // sysroot (Apple's bundled clang doesn't ship wasi-libc / wasm-libc) or
    // lack libc altogether (the unknown-unknown ABI). Compile no C on wasm
    // targets; the variadic `x3f_printf` shim in `src/wasm_c_shims.rs`
    // satisfies the remaining symbol (denoise is pure Rust in `src/denoise.rs`).
    if is_wasm {
        compile_wasm_only(&manifest_dir, &c_src, &out_dir, &host);
        return;
    }

    let c_sources: &[&str] = &[
        // x3f_io.c — cleanup machinery (cleanup_*, x3f_delete) and the
        // directory-entry searchers (x3f_get_*) ported to Rust in M5e
        // (crates/x3f-sys/src/io.rs); the rest of the file was already
        // Rust through M4c/M4d. The .c file holds no function bodies
        // and is no longer compiled.
        // x3f_process.c — fully ported to Rust across M6a-M6e10; the
        // remaining typedefs in src/x3f_process.h are still consumed
        // by bindgen but the .c file holds no function bodies and is
        // no longer compiled.
        // x3f_meta.c — ported to Rust in M4a; see crates/x3f-sys/src/meta.rs.
        // x3f_image.c — ported to Rust in M6c; see crates/x3f-sys/src/image.rs.
        // x3f_spatial_gain.c — ported to Rust in M6d; see crates/x3f-sys/src/spatial_gain.rs.
        // x3f_histogram.c — ported to Rust in M6b; see crates/x3f-sys/src/histogram.rs.
        // x3f_print_meta.c — ported to Rust in M4b; see crates/x3f-sys/src/print_meta.rs.
        // x3f_matrix.c — ported to Rust in M6a; see crates/x3f-sys/src/matrix.rs.
        "x3f_printf.c",
        "x3f_version.c",
    ];

    let mut build = cc::Build::new();
    build
        .include(&c_src)
        .define("VERSION", format!("\"x3f-sys-{}\"", version).as_str())
        .flag_if_supported("-Wno-unused-parameter")
        .flag_if_supported("-Wno-unused-but-set-variable")
        .flag_if_supported("-Wno-unused-variable")
        .flag_if_supported("-Wno-unused-function")
        .flag_if_supported("-Wno-pointer-sign")
        .flag_if_supported("-Wno-deprecated-declarations")
        .flag_if_supported("-Wno-incompatible-pointer-types")
        .flag_if_supported("-Wno-format")
        .flag_if_supported("-Wno-format-security")
        .flag_if_supported("-Wno-unused-result")
        .flag_if_supported("-Wno-implicit-function-declaration")
        .flag_if_supported("-Wno-int-conversion");

    for src in c_sources {
        build.file(c_src.join(src));
    }

    build.compile("x3f");

    let mut bindgen_builder = bindgen::Builder::default()
        .header(manifest_dir.join("wrapper.h").to_string_lossy().to_string())
        .clang_arg(format!("-I{}", c_src.display()));

    // Cross-compilation: bindgen invokes clang on the build host but must
    // parse headers as the *target* would see them. For iOS we override
    // the clang triple (Rust's `-sim` suffix isn't recognised by clang)
    // and point at the matching SDK so `<inttypes.h>` etc. resolve.
    if let Some(clang_triple) = clang_target_for(&target) {
        bindgen_builder = bindgen_builder.clang_arg(format!("--target={}", clang_triple));
    }
    if target.contains("apple-ios-sim") {
        if let Some(sdk) = apple_sdk_path("iphonesimulator") {
            bindgen_builder = bindgen_builder.clang_arg(format!("-isysroot{}", sdk.display()));
        }
    } else if target.contains("apple-ios") {
        if let Some(sdk) = apple_sdk_path("iphoneos") {
            bindgen_builder = bindgen_builder.clang_arg(format!("-isysroot{}", sdk.display()));
        }
    }

    let bindings = apply_bindgen_blocklists(
        bindgen_builder.allowlist_file(format!("{}/x3f_.*\\.h", c_src.display())),
    )
    .derive_debug(true)
    .derive_default(true)
    .layout_tests(false)
    .generate_comments(false)
    .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
    .generate()
    .expect("bindgen failed");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap()).join("bindings.rs");
    bindings
        .write_to_file(&out_path)
        .expect("could not write bindings.rs");
}

/// Apply the (long, hand-curated) list of bindgen blocklists shared by
/// every target. Each blocklist entry tracks a port milestone where the
/// matching C symbol moved to Rust; we keep the C declaration in the
/// header (other Rust call sites pick up its forward decl from bindgen)
/// but block bindgen from emitting an FFI shim that would conflict with
/// the Rust `#[no_mangle]` definition.
fn apply_bindgen_blocklists(b: bindgen::Builder) -> bindgen::Builder {
    b
        // Restrict generated bindings to symbols defined in the x3f headers,
        // not transitively pulled-in stdlib types. The wasm32 path's
        // run_bindgen() applies the allowlist itself before calling here;
        // the host path applies it inline.
        .allowlist_recursively(true)
        // x3f_expand_quattro has a native Rust definition in src/quattro.rs
        // exported via #[no_mangle]; blocking the bindgen-generated FFI
        // binding avoids a duplicate-symbol conflict.
        .blocklist_function("x3f_expand_quattro")
        // M4a: x3f_meta.c was ported to Rust (src/meta.rs). Block the
        // generated FFI bindings for the same reason — the Rust impls
        // export these symbols, and the C source is no longer compiled.
        .blocklist_function("x3f_get_camf_text")
        .blocklist_function("x3f_get_camf_matrix_var")
        .blocklist_function("x3f_get_camf_matrix")
        .blocklist_function("x3f_get_camf_float")
        .blocklist_function("x3f_get_camf_float_vector")
        .blocklist_function("x3f_get_camf_unsigned")
        .blocklist_function("x3f_get_camf_signed")
        .blocklist_function("x3f_get_camf_signed_vector")
        .blocklist_function("x3f_get_camf_property_list")
        .blocklist_function("x3f_get_camf_property")
        .blocklist_function("x3f_get_prop_entry")
        .blocklist_function("x3f_get_wb")
        .blocklist_function("x3f_get_camf_matrix_for_wb")
        .blocklist_function("x3f_get_max_raw")
        // M4b: x3f_print_meta.c was ported to Rust (src/print_meta.rs).
        .blocklist_function("x3f_dump_meta_data")
        .blocklist_function("x3f_print_meta")
        .blocklist_var("max_printed_matrix_elements")
        // M4c: x3f_new_from_file was ported to Rust (src/io.rs).
        .blocklist_function("x3f_new_from_file")
        // M4d: x3f_load_data, x3f_load_image_block, and x3f_err were
        // ported to Rust (src/load.rs).
        .blocklist_function("x3f_load_data")
        .blocklist_function("x3f_load_image_block")
        .blocklist_function("x3f_err")
        // M5e: cleanup machinery + directory-entry searchers + the
        // legacy_offset / auto_legacy_offset globals all ported to
        // Rust (src/io.rs). x3f_io.c is now empty and dropped from
        // the cc-rs build above.
        .blocklist_function("x3f_delete")
        .blocklist_function("x3f_get_raw")
        .blocklist_function("x3f_get_thumb_plain")
        .blocklist_function("x3f_get_thumb_huffman")
        .blocklist_function("x3f_get_thumb_jpeg")
        .blocklist_function("x3f_get_camf")
        .blocklist_function("x3f_get_prop")
        .blocklist_var("legacy_offset")
        .blocklist_var("auto_legacy_offset")
        // M6b: x3f_histogram.c was ported to Rust (src/histogram.rs).
        .blocklist_function("x3f_dump_raw_data_as_histogram")
        // M6e1+ : process.c functions ported to Rust.
        .blocklist_function("x3f_get_gain")
        .blocklist_function("x3f_get_bmt_to_xyz")
        .blocklist_function("x3f_get_raw_to_xyz")
        // M6e9: x3f_get_dng_highlight_scale ported to Rust along with
        // the apply_highlight_clip_dng + g_dng_highlight_scale state.
        .blocklist_function("x3f_get_dng_highlight_scale")
        // M6e10: public entry points ported to Rust. With these gone
        // the entire x3f_process.c is empty and is dropped from cc-rs.
        .blocklist_function("x3f_get_image")
        .blocklist_function("x3f_get_preview")
        // M6d: x3f_spatial_gain.c was ported to Rust (src/spatial_gain.rs).
        .blocklist_function("x3f_get_merrill_type_spatial_gain")
        .blocklist_function("x3f_get_interp_merrill_type_spatial_gain")
        .blocklist_function("x3f_get_classic_spatial_gain")
        .blocklist_function("x3f_get_spatial_gain")
        .blocklist_function("x3f_cleanup_spatial_gain")
        .blocklist_function("x3f_calc_spatial_gain")
        // M6c: x3f_image.c was ported to Rust (src/image.rs).
        .blocklist_function("x3f_image_area")
        .blocklist_function("x3f_image_area_qtop")
        .blocklist_function("x3f_crop_area")
        .blocklist_function("x3f_crop_area8")
        .blocklist_function("x3f_get_camf_rect")
        .blocklist_function("x3f_crop_area_column")
        .blocklist_function("x3f_crop_area_camf")
        .blocklist_function("x3f_crop_area8_camf")
        // M6a: x3f_matrix.c was ported to Rust (src/matrix.rs).
        .blocklist_function("x3f_3x1_invert")
        .blocklist_function("x3f_3x1_comp_mul")
        .blocklist_function("x3f_scalar_3x1_mul")
        .blocklist_function("x3f_scalar_3x3_mul")
        .blocklist_function("x3f_3x3_3x1_mul")
        .blocklist_function("x3f_3x3_3x3_mul")
        .blocklist_function("x3f_3x3_inverse")
        .blocklist_function("x3f_3x3_identity")
        .blocklist_function("x3f_3x3_diag")
        .blocklist_function("x3f_3x3_ones")
        .blocklist_function("x3f_3x1_print")
        .blocklist_function("x3f_3x3_print")
        .blocklist_function("x3f_XYZ_to_ProPhotoRGB")
        .blocklist_function("x3f_ProPhotoRGB_to_XYZ")
        .blocklist_function("x3f_XYZ_to_AdobeRGB")
        .blocklist_function("x3f_AdobeRGB_to_XYZ")
        .blocklist_function("x3f_XYZ_to_sRGB")
        .blocklist_function("x3f_sRGB_to_XYZ")
        .blocklist_function("x3f_CIERGB_to_XYZ")
        .blocklist_function("x3f_Bradford_D50_to_D65")
        .blocklist_function("x3f_Bradford_D65_to_D50")
        .blocklist_function("x3f_sRGB_LUT")
        .blocklist_function("x3f_gamma_LUT")
        .blocklist_function("x3f_LUT_lookup")
}
