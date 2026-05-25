//! Smoke tests that the cbindgen-emitted `x3f.h` is C-compilable and
//! references the same symbols the Rust source declares. We invoke the
//! host C compiler (`cc`) on a tiny fixture; if `cc` isn't available the
//! tests skip with a one-line note rather than failing.
//!
//! Linking against the actual staticlib is intentionally out of scope
//! here: that requires libz / libtiff / libjpeg / libc++ / Accelerate
//! to be wired into the link line, plus a real corpus file to feed
//! `x3f_reader_open`. That end-to-end smoke test belongs with the M8b
//! (iOS) / M8c (Android) per-platform CI jobs once those are stood up.

use std::path::PathBuf;
use std::process::Command;

/// Locate `target/<profile>/include/x3f.h`. The test runs with
/// `CARGO_MANIFEST_DIR` set to the crate root; we walk up to the
/// workspace root and probe both common profiles. We don't rely on
/// `CARGO_TARGET_TMPDIR` because cargo only sets it for integration
/// tests that use the `tempdir` API, not all integration tests.
fn header_path() -> Option<PathBuf> {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    // crates/x3f-ffi-c → workspace root is two parents up.
    let workspace = manifest.parent()?.parent()?;
    // Honour CARGO_TARGET_DIR if set; otherwise <workspace>/target.
    let target = std::env::var_os("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| workspace.join("target"));
    for profile in ["release", "debug"] {
        let header = target.join(profile).join("include").join("x3f.h");
        if header.is_file() {
            return Some(header);
        }
    }
    None
}

/// Returns the cc binary if usable, else `None`. Lets us skip cleanly on
/// CI runners or developer machines without a working host C toolchain.
fn host_cc() -> Option<&'static str> {
    let out = Command::new("cc").arg("--version").output().ok()?;
    out.status.success().then_some("cc")
}

#[test]
fn header_is_c99_clean() {
    let Some(header) = header_path() else {
        eprintln!("(skip: x3f.h not built — run `cargo build -p x3f-ffi-c` first)");
        return;
    };
    let Some(cc) = host_cc() else {
        eprintln!("(skip: host `cc` not available)");
        return;
    };

    let tmp = std::env::temp_dir().join(format!("x3f-ffi-c-smoke-{}", std::process::id()));
    std::fs::create_dir_all(&tmp).expect("create tmp");
    let src = tmp.join("smoke.c");

    // Touches every public symbol so any rename / signature drift between
    // Rust source and cbindgen header surfaces here as a -Werror=implicit-
    // function-declaration / type-mismatch failure.
    std::fs::write(
        &src,
        r#"
#include "x3f.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

int main(void) {
    (void)x3f_last_error();
    X3FReader *r = x3f_reader_open(NULL);
    (void)r;
    static const uint8_t empty[1] = { 0 };
    X3FReader *rb = x3f_reader_open_from_bytes(empty, 0);
    (void)rb;
    x3f_reader_close(NULL);
    (void)x3f_reader_header_version(NULL);
    (void)x3f_reader_dump_meta(NULL, NULL);
    (void)x3f_reader_dump_jpeg_thumbnail(NULL, NULL);
    return 0;
}
"#,
    )
    .expect("write smoke.c");

    let include_dir = header.parent().expect("header parent");
    let status = Command::new(cc)
        .args(["-std=c99", "-Wall", "-Werror", "-c"])
        .arg("-I")
        .arg(include_dir)
        .arg("-o")
        .arg(tmp.join("smoke.o"))
        .arg(&src)
        .status()
        .expect("invoke cc");
    assert!(
        status.success(),
        "x3f.h failed to compile cleanly under -std=c99 -Wall -Werror"
    );
}

#[test]
fn header_is_cpp_clean() {
    // Same as above but compiled as C++. Catches issues with `extern "C"`
    // guards, name mangling, and type punning that would bite anyone
    // including x3f.h from an Objective-C++ or C++-only consumer (e.g.
    // an iOS app's Swift-bridge header).
    let Some(header) = header_path() else {
        eprintln!("(skip: x3f.h not built)");
        return;
    };

    let cxx = match Command::new("c++").arg("--version").output() {
        Ok(o) if o.status.success() => "c++",
        _ => {
            eprintln!("(skip: host `c++` not available)");
            return;
        }
    };

    let tmp = std::env::temp_dir().join(format!("x3f-ffi-c-smoke-cpp-{}", std::process::id()));
    std::fs::create_dir_all(&tmp).expect("create tmp");
    let src = tmp.join("smoke.cpp");
    std::fs::write(
        &src,
        r#"
#include "x3f.h"
int main() {
    (void)x3f_last_error();
    return 0;
}
"#,
    )
    .expect("write smoke.cpp");

    let include_dir = header.parent().expect("header parent");
    let status = Command::new(cxx)
        .args(["-std=c++14", "-Wall", "-Werror", "-c"])
        .arg("-I")
        .arg(include_dir)
        .arg("-o")
        .arg(tmp.join("smoke.o"))
        .arg(&src)
        .status()
        .expect("invoke c++");
    assert!(status.success(), "x3f.h failed to compile cleanly as C++");
}
