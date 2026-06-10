# FFI surface

The [`x3f-ffi-c`](../../crates/x3f-ffi-c/) crate is the entry point
for non-Rust consumers — iOS / macOS / Mac Catalyst, Android, browser
WASM, command-line C / C++ programs, and any other host that can dlopen
a shared library. It wraps `x3f-core` in a minimal, opaque-handle C
ABI.

The crate produces a `staticlib` *and* a `cdylib` *and* an `rlib`
([`Cargo.toml`](../../crates/x3f-ffi-c/Cargo.toml)), so you can pick
the linkage that fits the target:

- **iOS / macOS / Catalyst** — staticlib, bundled into an
  `.xcframework`.
- **Android** — cdylib (`.so`) under `jniLibs/<abi>/`.
- **Browser WASM** — cdylib, instantiated as a `.wasm` module.
- **Server WASM (wasmtime / wasmer)** — cdylib targeting
  `wasm32-wasip1`.
- **Linux / Windows / macOS host C programs** — staticlib or cdylib.

A [`cbindgen.toml`](../../crates/x3f-ffi-c/cbindgen.toml) +
[`build.rs`](../../crates/x3f-ffi-c/build.rs) generate `x3f.h` into
`OUT_DIR` and `target/<profile>/include/x3f.h` on every build.

## C ABI surface

The current surface is intentionally small — enough to satisfy the
PORT-PLAN's per-platform smoke test (load file → metadata dump →
thumbnail extract) without committing to a heavier API ahead of seeing
what's idiomatic on each platform.

```c
typedef struct X3FReader X3FReader;

// Open / close (host filesystem path)
X3FReader* x3f_reader_open(const char *path);
void       x3f_reader_close(X3FReader *handle);

// Open from in-memory buffer (Unix-likes + WASM only;
// returns NULL with an error on Windows — use _open there).
X3FReader* x3f_reader_open_from_bytes(const uint8_t *data, size_t len);

// Header inspection
uint32_t   x3f_reader_header_version(const X3FReader *handle);

// Conversion entry points (return 0 on success, non-zero on error)
int        x3f_reader_dump_meta(X3FReader *handle, const char *out_path);
int        x3f_reader_dump_jpeg_thumbnail(X3FReader *handle, const char *out_path);

// Thread-local last error (NULL when no error since last successful call)
const char* x3f_last_error(void);
```

Conventions:

- **Handles** are opaque pointers owned by the library. Allocate with
  `x3f_reader_open` / `_open_from_bytes`; free with
  `x3f_reader_close`. Mixing the library's allocator with the
  caller's malloc/free across handles is undefined.
- **Strings** are UTF-8 NUL-terminated. Output strings owned by the
  library (`x3f_last_error`) live until the next call on the same
  thread.
- **Threading**: each `X3FReader` is single-threaded; callers must not
  share a handle across threads without external synchronisation.
  Distinct handles are independent.
- **Errors**: 0 = success, non-zero = error. On error, callers may pull
  a human-readable message from `x3f_last_error()` (thread-local;
  cleared on the next successful call).

Full doc-comments live on each `#[no_mangle] extern "C"` function in
[`crates/x3f-ffi-c/src/lib.rs`](../../crates/x3f-ffi-c/src/lib.rs);
cbindgen renders them into the generated `x3f.h`.

The heavier conversion entry points (TIFF / DNG / PPM output) are
deferred until the per-platform build wiring stabilises — there's no
point committing to the surface before we know what's idiomatic on
each host.

## iOS — XCFramework

The build script
[`crates/x3f-ffi-c/scripts/build-xcframework.sh`](../../crates/x3f-ffi-c/scripts/build-xcframework.sh)
cross-compiles the staticlib for `aarch64-apple-ios` (device) and
`aarch64-apple-ios-sim` (Apple-silicon simulator), stages the cbindgen
header, and bundles the slices into `target/X3F.xcframework`:

```sh
./crates/x3f-ffi-c/scripts/build-xcframework.sh
ls target/X3F.xcframework/
# Info.plist
# ios-arm64/{libx3f.a, Headers/x3f.h}
# ios-arm64-simulator/{libx3f.a, Headers/x3f.h}
```

Drop the `.xcframework` into Xcode (Add Files → Embed & Sign for
binary frameworks) and Swift / Obj-C can call the C ABI directly:

```swift
import X3F  // (or import a manual bridging header)

guard let h = x3f_reader_open(path) else {
    let msg = x3f_last_error().map { String(cString: $0) } ?? "unknown"
    throw X3FError(msg)
}
defer { x3f_reader_close(h) }
print(String(format: "0x%08x", x3f_reader_header_version(h)))
```

A build-system tweak lives in
[`crates/x3f-sys/build.rs`](../../crates/x3f-sys/build.rs) to make
this work: bindgen invokes clang with the correct `--target` +
`-isysroot` for the simulator's `arm64-apple-ios-simulator` triple.
(Earlier revisions also pinned `IPHONEOS_DEPLOYMENT_TARGET` to 13.0 for
opencv-mobile's iOS prebuilt; with OpenCV removed the crate is pure Rust
plus two tiny C shims, so the rustc default deployment target is fine.)

Mac Catalyst (`aarch64-apple-ios-macabi`) is wired into the `build.rs`
but not yet added to the script; adding it is a ~3-line change once
we have a use case.

## Android — `jniLibs/`

`M8c` is currently *blocked* on having an Android NDK installed in the
build environment (~3–5 GB). Once the NDK is on disk, an analogous
shell script under
[`crates/x3f-ffi-c/scripts/build-jnilibs.sh`](../../crates/x3f-ffi-c/scripts/build-jnilibs.sh)
will cross-compile the cdylib to:

```
target/jniLibs/
├── arm64-v8a/libx3f.so
├── armeabi-v7a/libx3f.so
└── x86_64/libx3f.so
```

drop-in for `<app>/src/main/jniLibs/`. The crate is pure Rust plus two
tiny C shims (no external native dependency), so the only missing piece
is the NDK + cargo-ndk wrapper.

## WASM

Two targets are supported; both produce a cdylib `.wasm` plus the
matching `libx3f.a` and `x3f.h`.

### `wasm32-wasip1` (server-side WASM)

Use this in wasmtime / wasmer / any wasi runtime. It links against
`wasi-libc` so the regular libc surface (`fopen`, `fread`, …) is
available, and the
[`Reader::open(path)`](../../crates/x3f-core/src/lib.rs) constructor
works against the WASI-mapped filesystem.

```sh
cargo build -p x3f-ffi-c --target wasm32-wasip1 --release
ls target/wasm32-wasip1/release/
# libx3f.a x3f.wasm include/x3f.h
```

### `wasm32-unknown-unknown` (browser WASM)

This target has no host filesystem, no libc, no syscalls — the wasm
module can only manipulate bytes the embedder hands to it. The crate
ships a pure-Rust `sysabi` shim
([`crates/x3f-sys/src/sysabi.rs`](../../crates/x3f-sys/src/sysabi.rs))
that replaces `libc` on this target: the allocator routes through
`std::alloc`, `memcpy` / `memset` lower to wasm `memory.copy` /
`memory.fill`, file-I/O is satisfied by an internal `MemFile` cursor
backing `fmemopen`. The variadic `x3f_printf` is no-op'd by a Rust shim in
[`crates/x3f-sys/src/wasm_c_shims.rs`](../../crates/x3f-sys/src/wasm_c_shims.rs);
denoise is the pure-Rust Non-Local Means in
[`crates/x3f-sys/src/denoise.rs`](../../crates/x3f-sys/src/denoise.rs), called
directly by the pipeline — so it runs in the browser just like on any other
target. Together this leaves the wasm cdylib with **zero**
unresolved `(import "env" ...)` entries — fully self-contained, ready for
`WebAssembly.instantiateStreaming` with no host-function shims.

```sh
cargo build -p x3f-ffi-c --target wasm32-unknown-unknown --release
wasm-tools print target/wasm32-unknown-unknown/release/x3f.wasm \
  | grep '(import "env"'   # should be empty — invariant gated in CI
```

The browser-runtime entry point is `x3f_reader_open_from_bytes`:

```js
const wasm = await WebAssembly.instantiateStreaming(fetch('x3f.wasm'));
const { x3f_reader_open_from_bytes, x3f_reader_close,
        x3f_reader_header_version, malloc } = wasm.instance.exports;

const buf = await (await fetch('photo.X3F')).arrayBuffer();
const len = buf.byteLength;
const ptr = malloc(len);
new Uint8Array(wasm.instance.exports.memory.buffer, ptr, len)
  .set(new Uint8Array(buf));

const handle = x3f_reader_open_from_bytes(ptr, len);
console.log(x3f_reader_header_version(handle).toString(16));
x3f_reader_close(handle);
```

End-to-end runtime smoke test:
[`crates/x3f-core/tests/wasm_runtime_smoke.rs`](../../crates/x3f-core/tests/wasm_runtime_smoke.rs)
cross-compiles
[`crates/x3f-core/examples/wasi_from_bytes.rs`](../../crates/x3f-core/examples/wasi_from_bytes.rs)
to `wasm32-wasip1`, runs it under wasmtime, and asserts the parsed
header version matches what `Reader::open` produces on the host.

## Logging callback

C-side log output (the `x3f_printf` family) routes through a Rust
function pointer set by
`x3f_core::globals::set_log_callback`. Mobile / WASM consumers should
plug in a sink appropriate for their platform:

- **iOS** — forward to `os_log` / `NSLog`.
- **Android** — forward to `__android_log_print`.
- **Browser WASM** — forward to `console.log` via a JS-side shim.

Without a callback set, log output is dropped on wasm32 (since the
sysabi stubs out `printf`); on host targets it falls through to
stdout.
