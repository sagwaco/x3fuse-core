# X3F format reference

This chapter is a quick byte-level orientation. The authoritative
implementation is the parser in
[`crates/x3f-sys/src/io.rs`](../../crates/x3f-sys/src/io.rs) and the
section loaders in
[`crates/x3f-sys/src/load.rs`](../../crates/x3f-sys/src/load.rs); this
text exists so a new contributor can look at one before diving into
the other.

> The format is little-endian throughout. Multi-byte fields are stored
> in native (LE) order; multi-byte streams (e.g. UTF-16 strings, raw
> Huffman bit-streams) are byte-LE but element-native. All offsets are
> from the start of the file.

## High-level layout

An X3F file is a section-table container. The table sits at the *end*
of the file, similar to a ZIP central directory. To parse a file:

1. Read the **header** at offset 0.
2. Read the **footer** (last 4 bytes) — it's the offset to the
   directory section.
3. At that offset, read the **directory header** + N **directory
   entries** (one per section).
4. Each directory entry is a `(fourCC, offset, length)` triple — seek
   to it on demand.

```
+------------------+   offset 0
|  header          |   "FOVb" magic + version + camera id + WB string +
|  (varies; ~232)  |   color mode + extended-data table (v2.1+)
+------------------+
|  payload data    |   image bytes, CAMF blob, property table —
|  (most of file)  |   referenced by the directory below
|                  |
+------------------+
|  directory       |   "SECd" + count + N × (fourCC, offset, length)
+------------------+
|  footer (u32)    |   absolute offset of the directory section
+------------------+   end of file
```

## Header

The header magic is `FOVb` = `0x6256_4f46` (little-endian uint32).
Following fields, in order:

| Field | Size | Notes |
|-------|------|-------|
| `magic` | u32 | `0x6256_4f46` (`"FOVb"`) |
| `version` | u32 | `(major << 16) \| minor`, e.g. `0x0002_0001` for 2.1 |
| `unique_identifier` | 16 bytes | per-camera serial-ish blob |
| `mark_bits` | u32 | rotation flags |
| `columns`, `rows` | u32 × 2 | sensor pixel dimensions |
| `rotation` | u32 | 0/90/180/270 |
| `white_balance` | 32 bytes | NUL-padded ASCII (v2.1+ only) |
| `color_mode` | 32 bytes | NUL-padded ASCII (v2.3+ only) |
| `extended_data` | 32 × (u8 + f32) | per-slot type + value (v2.1+; expands to 64 slots in v3.0+) |

Versions seen in the wild and what they unlock:

| Major.minor | Hex | First seen on |
|-------------|-----|---------------|
| 2.0 | `0x0002_0000` | SD9 / SD10 |
| 2.1 | `0x0002_0001` | adds `white_balance` + 32-slot extended data |
| 2.2 | `0x0002_0002` | minor metadata addition |
| 2.3 | `0x0002_0003` | adds `color_mode` |
| 3.0 | `0x0003_0000` | extended-data table grows to 64 slots |
| 4.0 | `0x0004_0000` | Quattro layout (no header `extended_data` after this point) |
| 4.1 | `0x0004_0001` | sd Quattro / sd Quattro H |

The `x3f_extended_types_t` enum (slot kinds) lists exposure /
contrast / shadow / highlight / saturation / sharpness / red / green
/ blue / fill-light adjust in slots 1–10; see
[`src/x3f_io.h`](../../src/x3f_io.h) for the exact mapping.

## Directory

The directory section starts with the 4-byte fourCC `SECd`
(`0x6443_4553`), a u32 version, and a u32 entry count. Each entry is:

```
struct x3f_directory_entry {
    uint32_t offset;   // absolute file offset of the section payload
    uint32_t length;   // length in bytes of the section payload
    uint32_t type;     // section fourCC: PROP / IMAG / IMA2 / CAMF / SPPA
};
```

The footer is a single u32 at the very end of the file, holding the
absolute offset of the `SECd` directory section.

## Section types

Each directory entry's payload begins with its own fourCC header
identifying the encoding:

### Property list

Section type `PROP` (`0x504f_5250`); payload starts with `SECp`
(`0x7043_4553`). Encodes the camera's name → value property table:

- `num_properties` (u32) — count
- `character_format` (u32) — `0` for UTF-16LE
- 24-byte header
- `num_properties` × `(name_offset, value_offset)` u32 pairs
- a UTF-16LE string blob the offsets index into

Both names and values are UTF-16LE, NUL-terminated. The Rust loader
in [`crates/x3f-sys/src/load.rs`](../../crates/x3f-sys/src/load.rs)
converts to UTF-8 via `std::char::decode_utf16`, replacing the
iconv path the legacy C parser used.

### Image / RAW

Section type `IMAG` / `IMA2` (`0x4641_4d49` / `0x3241_4d49`);
payload starts with `SECi` (`0x6943_4553`). The 28-byte image
header carries:

| Field | Notes |
|-------|-------|
| `type` | top u16: kind (RAW=3, THUMB=2); bottom u16: encoding |
| `format` | 0 for ABGR, 1 for grey, etc. |
| `columns`, `rows`, `row_stride` | u32 × 3 |

The full 32-bit `type` field selects the decoder. The encodings in
the corpus today:

| Hex | Meaning | Decoder |
|-----|---------|---------|
| `0x0001_001e` | RAW Merrill | TRUE entropy decoder |
| `0x0001_0023` | RAW Quattro | TRUE + 2×2 plane expansion |
| `0x0001_0025` | RAW sd Quattro (SDQ) | TRUE + Quattro |
| `0x0001_0027` | RAW sd Quattro H (SDQH) | TRUE + Quattro |
| `0x0002_0003` | THUMB plain | uncompressed RGB triples |
| `0x0002_000b` | THUMB Huffman | Huffman, with per-row table |
| `0x0002_0012` | THUMB JPEG | embedded JFIF (byte-blob copy) |
| `0x0002_0019` | THUMB SDQ | (TODO — under-investigated) |
| `0x0003_0005` | RAW Huffman X530 | Huffman, classic SD9/SD10 path |
| `0x0003_0006` | RAW Huffman 10BIT | Huffman, 10-bit |
| `0x0003_001e` | RAW TRUE | early TRUE |

The TRUE decoder (Sigma's predictor + Huffman) lives in
[`crates/x3f-sys/src/entropy.rs`](../../crates/x3f-sys/src/entropy.rs);
the three planes have independent bitstreams (`tru.plane_address[c]`)
and write to disjoint output offsets, which M7d exploits for
plane-parallel decode. The Quattro 2×2 expansion that lifts the
half-resolution M and B planes up to T's resolution lives in
[`crates/x3f-sys/src/quattro.rs`](../../crates/x3f-sys/src/quattro.rs)
— a Catmull-Rom bicubic upsampler (`a = -0.75`) mirroring OpenCV's
`INTER_CUBIC`.

### CAMF — camera metadata

Section type `CAMF` (`0x464d_4143`); payload starts with `SECc`
(`0x6343_4553`). The 28-byte CAMF header carries an `info_type`
(2 / 4 / 5) and a `data_size`. Three encodings are in the wild:

- **Type 2** — stream cipher with a key derived from a 32-bit value
  in the header. Used on early bodies.
- **Type 4** — zigzag predictor + Huffman. Used on Merrill and later.
- **Type 5** — 1-D predictor + Huffman. Used on Quattro.

After the encoded blob is decoded into a flat byte buffer, it
contains a stream of CAMF entries each tagged with a per-entry
fourCC:

| Hex | fourCC | Meaning |
|-----|--------|---------|
| `0x4346_624d` | `CMb` | generic blob |
| `0x4346_624d` + `P` | `CMbP` | named property entry |
| `0x4346_624d` + `T` | `CMbT` | named text entry |
| `0x4346_624d` + `M` | `CMbM` | named matrix entry |

Named-entry lookup is done by
[`crates/x3f-sys/src/meta.rs`](../../crates/x3f-sys/src/meta.rs)
(`x3f_get_camf_text`, `x3f_get_camf_property`, `x3f_get_camf_matrix`,
`x3f_get_camf_float_array`, …). The CAMF blob is where most of the
camera's calibration data lives — black-shield rectangles, white
balance gains, color correction matrices, spatial gain LUTs, bad-pixel
maps, sat-map thresholds. See the project memory entries for an
inventory of unread CAMF entries the Sigma firmware writes that we
don't currently consume.

### SPPA

Section type `SPPA` (`0x4150_5053`); payload starts with `SECs`
(`0x7343_4553`). Reserved for sd Quattro variants; not currently
consumed.

## Camera IDs

The header carries no camera ID directly — it's recovered from CAMF
property `CAMERA_ID` (or sometimes `CameraID`). The IDs we know:

| ID | Body |
|----|------|
| 40 | sd Quattro (SDQ) |
| 41 | sd Quattro H (SDQH) |
| 77 | DP1 Merrill |
| 78 | DP2 Merrill / DP3 Merrill (shared) |
| 80 | DP1 Quattro |
| 81 | DP2 Quattro |
| 82 | DP3 Quattro |
| 83 | DP0 Quattro |

The classic SD9 / SD10 / SD14 / SD15 / DP1 / DP2 / DP1x / DP2x bodies
predate the CAMF ID convention; their image type fourCC distinguishes
them.

## Where to dig deeper

- [`crates/x3f-sys/src/io.rs`](../../crates/x3f-sys/src/io.rs) —
  header + directory + footer parser.
- [`crates/x3f-sys/src/load.rs`](../../crates/x3f-sys/src/load.rs) —
  per-section dispatch, CAMF type-2/4/5 decoders, property-list
  UTF-16 conversion.
- [`crates/x3f-sys/src/entropy.rs`](../../crates/x3f-sys/src/entropy.rs)
  — Huffman / TRUE / simple_decode.
- [`crates/x3f-sys/src/meta.rs`](../../crates/x3f-sys/src/meta.rs) —
  named-entry CAMF accessors.
- [`src/x3f_io.h`](../../src/x3f_io.h) — the original C-side struct
  definitions, kept around because they're still consumed by bindgen.
