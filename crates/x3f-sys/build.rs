// build.rs — compile the small remaining C/C++ surface and emit Rust bindings.
//
// The X3F decode + processing pipeline (container parsing, CAMF, entropy
// decode, image processing, matrix math, histogram, metadata, and PPM/TIFF/
// DNG output) is all native Rust. The only C/C++ left lives in this crate's
// `csrc/` directory, kept self-contained so `cargo publish` packages it:
//
// What's compiled:
//  - csrc/x3f_printf.c, csrc/x3f_version.c — the log callback + version shim.
//  - The OpenCV-backed denoise .cpp files (x3f_denoise.cpp +
//    x3f_denoise_utils.cpp) on targets where we can fetch a prebuilt
//    opencv-mobile static framework (Apple, Linux, Windows, Android).
//    On wasm32 cc-rs is skipped entirely (Apple's bundled clang has no
//    wasm-libc sysroot); the four C symbols still referenced by the
//    bindgen output (x3f_printf, x3f_denoise, x3f_denoise_active,
//    x3f_set_use_opencl) are satisfied by Rust shims in
//    `src/wasm_c_shims.rs`. For non-wasm targets with no opencv-mobile
//    prebuilt (or offline / docs.rs builds), `csrc/denoise_stub.c`
//    provides the same no-op fallback.
//  - bindgen against wrapper.h emits a single bindings.rs for the C
//    struct/enum layouts the Rust port mirrors via #[repr(C)].
//
// opencv-mobile (https://github.com/nihui/opencv-mobile) ships static
// prebuilts for every target in PORT-PLAN goal #2 at ~5–20 MB per target
// (vs. 50+ MB upstream OpenCV). On Apple platforms it's an opencv2.framework
// containing a static `ar` archive; on Linux/Windows/Android it's a more
// conventional `lib/cmake/OpenCV/` layout. We pin to a release tag, fetch
// the matching asset into OUT_DIR on first build, and add the include
// path + link directives below.

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

const OPENCV_MOBILE_TAG: &str = "v35";
const OPENCV_MOBILE_VERSION: &str = "4.13.0";

/// Map a Rust target triple to an opencv-mobile release asset suffix.
/// Returns `None` for targets we don't have a prebuilt for, in which case
/// we fall back to the no-op denoise stub.
fn opencv_mobile_asset_for(target: &str) -> Option<&'static str> {
    if target.starts_with("wasm32") {
        // wasm32-unknown-unknown can't link C++ stdlib; the prebuilt WASM
        // bundle is Emscripten (wasm32-unknown-emscripten). When/if we
        // switch the WASM build target in M8 we can flip this to "webassembly".
        None
    } else if target.contains("apple-ios-sim") || target.ends_with("-ios-sim") {
        Some("ios-simulator")
    } else if target.contains("apple-ios-macabi") || target.contains("ios-macabi") {
        Some("mac-catalyst")
    } else if target.contains("apple-ios") {
        Some("ios")
    } else if target.contains("apple-darwin") {
        Some("macos")
    } else if target.contains("android") {
        Some("android")
    } else if target.contains("linux") {
        // The ubuntu-2404 prebuilt's static archives link against newer
        // glibc; for older distros the user can override the target.
        Some("ubuntu-2404")
    } else if target.contains("windows-msvc") {
        Some("windows-vs2022")
    } else {
        None
    }
}

/// Download (if absent) and extract the opencv-mobile prebuilt for the
/// given asset suffix into OUT_DIR. Returns the extracted directory root,
/// which on Apple platforms contains `opencv2.framework/` and on other
/// platforms an `include/` + `lib/` layout.
///
/// Returns `None` when the prebuilt can't be obtained (no network, `curl` /
/// `unzip` missing, or the build is running on docs.rs). The caller then
/// falls back to the no-op denoise stub, so an offline / sandboxed build
/// still succeeds — it just ships without the OpenCV NLM denoise.
fn fetch_opencv_mobile(out_dir: &Path, asset: &str) -> Option<PathBuf> {
    // docs.rs builds have no network access; go straight to the stub.
    if env::var_os("DOCS_RS").is_some() {
        println!("cargo:warning=DOCS_RS set; building without OpenCV denoise (no-op stub)");
        return None;
    }

    let zip_name = format!("opencv-mobile-{}-{}.zip", OPENCV_MOBILE_VERSION, asset);
    let extract_dir = out_dir.join(format!("opencv-mobile-{}-{}", OPENCV_MOBILE_VERSION, asset));
    let stamp = extract_dir.join(".extracted");
    if stamp.exists() {
        return Some(extract_dir);
    }

    let zip_path = out_dir.join(&zip_name);
    if !zip_path.exists() {
        let url = format!(
            "https://github.com/nihui/opencv-mobile/releases/download/{}/{}",
            OPENCV_MOBILE_TAG, zip_name
        );
        eprintln!("downloading {}", url);
        let status = Command::new("curl")
            .args(["-sLf", "-o"])
            .arg(&zip_path)
            .arg(&url)
            .status();
        if !matches!(status, Ok(ref s) if s.success()) {
            println!(
                "cargo:warning=could not fetch opencv-mobile ({url}): {status:?}; \
                 building without OpenCV denoise (no-op stub)"
            );
            return None;
        }
    }

    std::fs::create_dir_all(&extract_dir).ok()?;
    let status = Command::new("unzip")
        .args(["-q", "-o"])
        .arg(&zip_path)
        .arg("-d")
        .arg(&extract_dir)
        .status();
    if !matches!(status, Ok(ref s) if s.success()) {
        println!(
            "cargo:warning=could not extract {}: {status:?}; \
             building without OpenCV denoise (no-op stub)",
            zip_path.display()
        );
        return None;
    }
    std::fs::write(&stamp, b"ok").ok()?;
    Some(extract_dir)
}

/// Recursively search `base` (bounded by `max_depth`) for the directory that
/// directly contains `marker` — a relative path such as `opencv2/core.hpp`
/// (the include root we hand to `-I`) or a filename such as
/// `libopencv_core.a` (the link-search dir).
///
/// opencv-mobile's prebuilt zips differ across assets: the headers may sit at
/// `include/opencv2/` or, following OpenCV's CMake-install convention, at
/// `include/opencv4/opencv2/`; the static archives live under `lib/`; and the
/// whole tree may be wrapped in a top-level `opencv-mobile-<ver>-<asset>/`
/// folder (cross-compile assets add a further arch-triple level). Probing for
/// the real markers rather than assuming fixed subpaths keeps every layout
/// (including the Apple `opencv2.framework`) working.
fn find_under(base: &Path, marker: &Path, max_depth: usize) -> Option<PathBuf> {
    let mut stack = vec![(base.to_path_buf(), 0usize)];
    while let Some((dir, depth)) = stack.pop() {
        if dir.join(marker).exists() {
            return Some(dir);
        }
        if depth < max_depth {
            if let Ok(entries) = std::fs::read_dir(&dir) {
                for e in entries.flatten() {
                    let p = e.path();
                    if p.is_dir() {
                        stack.push((p, depth + 1));
                    }
                }
            }
        }
    }
    None
}

/// Configure cc::Build and emit cargo:rustc-link-* directives so the
/// denoise .cpp files can `#include <opencv2/...>` and the final
/// crate links against the prebuilt static archives.
fn configure_opencv(build: &mut cc::Build, target: &str, root: &Path) {
    if target.contains("apple") {
        // opencv-mobile ships `opencv2.framework` for Apple targets. With
        // `-F<dir>` clang resolves `<opencv2/photo.hpp>` through the
        // framework's `Headers/` directory. The framework's binary is a
        // static `ar` archive (universal arm64+x86_64 on macos), so the
        // linker pulls only used object files.
        let fw_dir = find_under(root, Path::new("opencv2.framework"), 4)
            .unwrap_or_else(|| root.to_path_buf());
        build.flag("-F").flag(fw_dir.to_str().unwrap());
        println!("cargo:rustc-link-search=framework={}", fw_dir.display());
        println!("cargo:rustc-link-lib=framework=opencv2");
        // OpenCV's image I/O uses Accelerate's vDSP/vImage on Apple.
        println!("cargo:rustc-link-lib=framework=Accelerate");
    } else {
        // Conventional layout, but the exact subpaths vary by asset (see
        // find_under). Locate the dir that actually holds <opencv2/core.hpp>
        // and the one holding the static archives instead of guessing.
        let include = find_under(root, &Path::new("opencv2").join("core.hpp"), 6)
            .unwrap_or_else(|| root.join("include"));
        build.include(&include);
        let lib_dir =
            find_under(root, Path::new("libopencv_core.a"), 6).unwrap_or_else(|| root.join("lib"));
        println!("cargo:rustc-link-search=native={}", lib_dir.display());
        // opencv-mobile's modules. Order matters for static linking on some
        // platforms (photo depends on imgproc which depends on core).
        for lib in ["opencv_photo", "opencv_imgproc", "opencv_core"] {
            println!("cargo:rustc-link-lib=static={}", lib);
        }
        // The Linux prebuilt parallelizes `cv::parallel_for_` with OpenMP, so
        // libopencv_core.a's parallel.cpp.o references GOMP_*/omp_* symbols.
        // Link the GNU OpenMP runtime (shipped with gcc) so they resolve;
        // emitted after the opencv archives so the static-archive references
        // are still open when the linker reaches it. (Android ships libomp
        // via the NDK and Windows MSVC pulls vcomp in via #pragma — neither
        // is built in CI, so scope this to glibc/musl Linux.)
        if target.contains("linux") {
            println!("cargo:rustc-link-lib=dylib=gomp");
        }
    }
}

/// The iOS deployment target opencv-mobile's iOS / iOS-simulator
/// prebuilts are compiled against. Linking with a lower deployment
/// target produces "object file built for newer iOS version than being
/// linked" warnings and resolves missing `___chkstk_darwin` symbols.
/// Apple's iOS 13.0 dropped 32-bit support, which matches opencv-mobile's
/// minimum.
const IPHONEOS_MIN_VERSION: &str = "13.0";

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
/// C/C++ files (`x3f_printf.c`, `x3f_denoise.cpp`, …) all `#include
/// <stdio.h>` / `<inttypes.h>`. The four C symbols still referenced
/// from the Rust port (`x3f_printf`, `x3f_denoise`, `x3f_denoise_active`,
/// `x3f_set_use_opencl`) are provided by `#[no_mangle]` Rust shims in
/// `src/wasm_c_shims.rs`.
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
    // All remaining C/C++ (the two log/version shims, the OpenCV denoise pair,
    // and the headers bindgen consumes) lives inside the crate at `csrc/` so
    // `cargo publish` packages a self-contained crate.
    let c_src = manifest_dir.join("csrc");
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let target = env::var("TARGET").unwrap();
    let host = env::var("HOST").unwrap();
    let is_wasm = target.starts_with("wasm32");

    // Pin IPHONEOS_DEPLOYMENT_TARGET for any iOS target so cc-rs and the
    // final linker pass agree with opencv-mobile's prebuilt floor. Lower
    // deployment targets produce missing `___chkstk_darwin` symbols at
    // link time. cc-rs reads this env var directly; for the final rustc
    // link pass (cdylib / bin) we additionally emit a `-platform_version`
    // link arg because rustc only forwards IPHONEOS_DEPLOYMENT_TARGET
    // when the deployment target is left unset on the linker command line
    // — and bypassing it ensures the prebuilt object files from
    // opencv-mobile are linked against a matching platform-version load
    // command.
    if target.contains("apple-ios") {
        if env::var_os("IPHONEOS_DEPLOYMENT_TARGET").is_none() {
            env::set_var("IPHONEOS_DEPLOYMENT_TARGET", IPHONEOS_MIN_VERSION);
            println!(
                "cargo:rustc-env=IPHONEOS_DEPLOYMENT_TARGET={}",
                IPHONEOS_MIN_VERSION
            );
        }
        // For cdylib / bin link passes only — iOS staticlib production
        // doesn't invoke the linker and doesn't need this. We emit the
        // platform_version ld arg so opencv-mobile's iOS-13+ object files
        // resolve `___chkstk_darwin` against the matching libSystem.
        // Platform name strings ("ios" / "ios-simulator" / "mac-catalyst")
        // are accepted by both Apple ld64 and lld.
        let platform = if target.contains("apple-ios-sim") {
            "ios-simulator"
        } else if target.contains("apple-ios-macabi") {
            "mac-catalyst"
        } else {
            "ios"
        };
        let link_arg = format!(
            "-Wl,-platform_version,{platform},{IPHONEOS_MIN_VERSION},{IPHONEOS_MIN_VERSION}"
        );
        println!("cargo:rustc-cdylib-link-arg={link_arg}");
    }

    println!("cargo:rerun-if-changed=wrapper.h");
    println!("cargo:rerun-if-changed=csrc/denoise_stub.c");
    println!("cargo:rerun-if-changed={}", c_src.display());

    let version = env::var("CARGO_PKG_VERSION").unwrap();

    // wasm32-unknown-unknown / wasm32-wasip* targets either lack a libc
    // sysroot (Apple's bundled clang doesn't ship wasi-libc / wasm-libc) or
    // lack libc altogether (the unknown-unknown ABI). Compile no C/C++ on
    // wasm targets and let the matching Rust shims in `src/wasm_shim.rs`
    // satisfy x3f_printf / x3f_version / x3f_denoise / x3f_set_use_opencl.
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

    // Denoise: fetch opencv-mobile and compile the denoise .cpp files for
    // supported targets; otherwise fall back to the no-op stub so the
    // x3f_denoise / x3f_set_use_opencl symbols still resolve.
    let denoise_cpp_sources = ["x3f_denoise.cpp", "x3f_denoise_utils.cpp"];
    let opencv_root =
        opencv_mobile_asset_for(&target).and_then(|asset| fetch_opencv_mobile(&out_dir, asset));
    if let Some(opencv_root) = opencv_root {
        // Compile the denoise files with cc-rs as a separate cpp build
        // object so we can flip cpp(true) and add the OpenCV include
        // path without polluting the C build above.
        let mut cpp_build = cc::Build::new();
        cpp_build
            .cpp(true)
            .std("c++14")
            .include(&c_src)
            .flag_if_supported("-Wno-unused-parameter")
            .flag_if_supported("-Wno-unused-variable")
            .flag_if_supported("-Wno-deprecated-declarations");
        configure_opencv(&mut cpp_build, &target, &opencv_root);
        for src in &denoise_cpp_sources {
            cpp_build.file(c_src.join(src));
        }
        cpp_build.compile("x3f_denoise");
    } else {
        // No prebuilt available (unsupported target, offline, or docs.rs):
        // link the no-op denoise stub so x3f_denoise / x3f_set_use_opencl
        // still resolve.
        build.file(manifest_dir.join("csrc/denoise_stub.c"));
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
