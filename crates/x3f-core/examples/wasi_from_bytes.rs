//! Tiny wasi-targetable smoke binary that exercises
//! `Reader::from_bytes`. Reads an X3F file from `argv[1]` (preopened
//! via `wasmtime --dir`), parses it through the buffer-based API,
//! and prints `OK <header_version>` on success.
//!
//! Used by the `wasm32-wasip1` runtime smoke test in CI:
//! ```sh
//! cargo build --release --example wasi_from_bytes \
//!     -p x3f-core --target wasm32-wasip1
//! wasmtime run --dir=$X3F_TEST_FILES \
//!     target/wasm32-wasip1/release/examples/wasi_from_bytes.wasm \
//!     -- $X3F_TEST_FILES/some.X3F
//! ```
//!
//! The source code path here is identical on host and on wasm32-wasip1:
//! `Reader::from_bytes` dispatches through libc's `fmemopen` on host
//! and through the M8d-α-2 `MemFile` shim on wasm32-unknown-unknown.
//! On wasm32-wasip1 it goes through wasi-libc's `fmemopen` (yes, that
//! exists in wasi-sdk's libc port). All three paths resolve to the
//! same Rust call sequence, so this smoke test exercising the wasi
//! path validates the wasm32 cdylib's behaviour by proxy.

fn main() -> std::process::ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let Some(path) = args.get(1) else {
        eprintln!("usage: wasi_from_bytes <path-to-x3f>");
        return std::process::ExitCode::from(64);
    };

    let bytes = match std::fs::read(path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("read `{path}`: {e}");
            return std::process::ExitCode::from(74);
        }
    };

    match x3f_core::Reader::from_bytes(&bytes) {
        Ok(r) => {
            println!("OK {:#010x}", r.header_version());
            std::process::ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("from_bytes failed: {e}");
            std::process::ExitCode::from(70)
        }
    }
}
