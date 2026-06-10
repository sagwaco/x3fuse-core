#!/usr/bin/env bash
#
# Build the x3f xcframework bundling the iOS device + simulator
# staticlibs produced by `cargo build -p x3f-ffi-c --target …`.
#
# Output: target/X3F.xcframework
#
# Requires:
#   - macOS host (xcodebuild must be on PATH)
#   - rustup targets `aarch64-apple-ios` and `aarch64-apple-ios-sim`
#   - Xcode with iOS SDK
#
# Usage:
#   scripts/build-xcframework.sh [--release|--debug]   # default: --release
#
# Skips device-only builds when CI sets `X3F_XCFRAMEWORK_SIM_ONLY=1`,
# which is useful on PR builds where signing identity isn't available
# (device builds only need the rustc toolchain — no signing — but the
# guard is here for parity with downstream workflows).

set -euo pipefail

PROFILE="${1:---release}"
case "$PROFILE" in
    --release) PROFILE_DIR="release"; CARGO_FLAGS=(--release) ;;
    --debug)   PROFILE_DIR="debug";   CARGO_FLAGS=() ;;
    *)
        echo "usage: $0 [--release|--debug]" >&2
        exit 64
        ;;
esac

# Resolve workspace root: this script lives at
# crates/x3f-ffi-c/scripts/build-xcframework.sh.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
TARGET_DIR="${CARGO_TARGET_DIR:-$WORKSPACE_ROOT/target}"

# Build each target. The crate is pure Rust plus two tiny C shims, so the
# rustc default iOS deployment target is fine; nothing needs pinning here.
TARGETS=(aarch64-apple-ios aarch64-apple-ios-sim)
if [[ "${X3F_XCFRAMEWORK_SIM_ONLY:-}" == "1" ]]; then
    TARGETS=(aarch64-apple-ios-sim)
fi

for TRIPLE in "${TARGETS[@]}"; do
    echo "==> building $TRIPLE staticlib"
    cargo build "${CARGO_FLAGS[@]}" -p x3f-ffi-c --target "$TRIPLE"
done

# The cbindgen header is the same across targets — it lands in
# <profile>/include/x3f.h on whichever target ran most recently, but
# we always rebuild the host crate too via the `cargo build` calls
# (each target's build.rs writes the header into its own profile dir).
# We just need any one of them.
HEADER=""
for TRIPLE in "${TARGETS[@]}"; do
    candidate="$TARGET_DIR/$TRIPLE/$PROFILE_DIR/include/x3f.h"
    if [[ -f "$candidate" ]]; then
        HEADER="$candidate"
        break
    fi
done
if [[ -z "$HEADER" ]]; then
    # Fall back to the host build's header.
    HEADER="$TARGET_DIR/$PROFILE_DIR/include/x3f.h"
fi
if [[ ! -f "$HEADER" ]]; then
    echo "error: no x3f.h found under $TARGET_DIR/*/(\$PROFILE)/include/" >&2
    exit 1
fi

# xcframework expects a directory of headers, not a single file. We stage
# one Headers/ dir per slice.
STAGE="$TARGET_DIR/xcframework-staging"
rm -rf "$STAGE"
mkdir -p "$STAGE/headers"
cp "$HEADER" "$STAGE/headers/x3f.h"

XCFW="$TARGET_DIR/X3F.xcframework"
rm -rf "$XCFW"

ARGS=(-create-xcframework)
for TRIPLE in "${TARGETS[@]}"; do
    LIB="$TARGET_DIR/$TRIPLE/$PROFILE_DIR/libx3f.a"
    if [[ ! -f "$LIB" ]]; then
        echo "error: $LIB not found — did the build succeed?" >&2
        exit 1
    fi
    ARGS+=(-library "$LIB" -headers "$STAGE/headers")
done
ARGS+=(-output "$XCFW")

echo "==> running xcodebuild ${ARGS[*]}"
xcodebuild "${ARGS[@]}"

echo
echo "✔ wrote $XCFW"
ls -la "$XCFW"
