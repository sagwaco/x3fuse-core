# Architecture

This file used to hold the long-form architecture overview. That
content has been promoted into a proper [mdbook](https://rust-lang.github.io/mdBook/)
under [docs/](docs/) so it can grow without drowning the repo root.

To browse the book locally:

```sh
cargo install mdbook       # one-time, if not already installed
mdbook serve docs --open
```

Or read the source directly under [docs/src/](docs/src/). The
chapters that replaced this file:

- [Conversion pipeline](docs/src/pipeline.md) — how an X3F file
  becomes a DNG/TIFF/PPM, stage by stage.
- [Workspace layout](docs/src/workspace.md) — the four-crate split
  and which C source files survive.
- [X3F format reference](docs/src/format.md) — header, directory,
  sections, CAMF, entropy.
- [FFI surface](docs/src/ffi.md) — C ABI for iOS / Android / WASM
  consumers.
- [Performance notes](docs/src/performance.md) — what was
  parallelised and why.
- [Contributor guide](docs/src/contributing.md) — port conventions,
  parity gates, test corpus.
- [Port plan](docs/PORT-PLAN.md) — milestone breakdown.
