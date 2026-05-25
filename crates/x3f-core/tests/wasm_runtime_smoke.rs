//! End-to-end wasm32 runtime smoke. Cross-compiles the
//! `wasi_from_bytes` example to `wasm32-wasip1`, runs it via
//! `wasmtime`, and asserts the parsed X3F header version matches what
//! `Reader::open` produces on the host.
//!
//! Skips with a one-line notice when prerequisites are missing
//! (`wasmtime` or `cargo` not on PATH; rust toolchain doesn't have
//! `wasm32-wasip1` installed; corpus directory missing). Same skip
//! pattern as the tier-2/tier-3 corpus tests.

use std::path::PathBuf;
use std::process::Command;

fn corpus_dir() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("X3F_TEST_FILES") {
        let p = PathBuf::from(p);
        return p.is_dir().then_some(p);
    }
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest.parent()?.parent()?.to_path_buf();
    let candidate = workspace_root.join("x3f_test_files");
    candidate.is_dir().then_some(candidate)
}

fn pick_corpus_file() -> Option<PathBuf> {
    let dir = corpus_dir()?;
    let mut entries: Vec<PathBuf> = std::fs::read_dir(&dir)
        .ok()?
        .filter_map(|r| r.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension()
                .and_then(|e| e.to_str())
                .is_some_and(|e| e.eq_ignore_ascii_case("x3f"))
        })
        .collect();
    entries.sort();
    entries.into_iter().next()
}

fn tool_available(name: &str, args: &[&str]) -> bool {
    Command::new(name)
        .args(args)
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

#[test]
fn wasm32_wasip1_reader_from_bytes_runs_under_wasmtime() {
    let Some(input) = pick_corpus_file() else {
        eprintln!("  (skip: no .X3F corpus file found)");
        return;
    };
    if !tool_available("wasmtime", &["--version"]) {
        eprintln!("  (skip: wasmtime not on PATH; install via `brew install wasmtime`)");
        return;
    }
    if !tool_available("rustup", &["target", "list", "--installed"]) {
        eprintln!("  (skip: rustup not on PATH)");
        return;
    }
    let installed = Command::new("rustup")
        .args(["target", "list", "--installed"])
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).into_owned())
        .unwrap_or_default();
    if !installed.contains("wasm32-wasip1") {
        eprintln!(
            "  (skip: wasm32-wasip1 rustup target not installed; run `rustup target add wasm32-wasip1`)"
        );
        return;
    }

    // Compile the example for wasm32-wasip1. Use `cargo build` from the
    // env CARGO so the test honours custom toolchains.
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_string());
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest.parent().unwrap().parent().unwrap().to_path_buf();
    let status = Command::new(&cargo)
        .current_dir(&workspace_root)
        .args([
            "build",
            "--release",
            "-p",
            "x3f-core",
            "--target",
            "wasm32-wasip1",
            "--example",
            "wasi_from_bytes",
        ])
        .status()
        .expect("spawn cargo");
    assert!(status.success(), "cargo build failed");

    let wasm = workspace_root.join("target/wasm32-wasip1/release/examples/wasi_from_bytes.wasm");
    assert!(wasm.is_file(), "expected {} to be built", wasm.display());

    // Reference: parse the header version on host.
    let host_v = x3f_core::Reader::open(&input)
        .expect("Reader::open")
        .header_version();
    let expected = format!("OK {host_v:#010x}");

    let dir_arg = format!("--dir={}", input.parent().unwrap().display());
    let out = Command::new("wasmtime")
        .args(["run", &dir_arg])
        .arg(&wasm)
        .arg(&input)
        .output()
        .expect("spawn wasmtime");
    assert!(
        out.status.success(),
        "wasmtime failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
    let stdout = String::from_utf8_lossy(&out.stdout).trim().to_string();
    assert_eq!(
        stdout, expected,
        "wasm output `{stdout}` doesn't match host header version `{expected}`"
    );
}
