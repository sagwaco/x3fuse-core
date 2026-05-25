# DNG flat-fielding opcodes

Pre-rendered binary `OpcodeList3` blobs for Sigma's per-camera /
per-aperture flat-field corrections. When passed to `x3f_extract` via
`-opcodes-dir <path>`, the matching blob is embedded verbatim into the
DNG raw IFD's `OpcodeList3` tag, enabling automatic flat-field
correction in any DNG-aware processor (Lightroom, Capture One, RawTherapee, etc.).

```
x3f_extract -opcodes-dir opcodes -dng input.X3F
```

## Layout

```
<MODEL>[_<LENSID>]_FF_DNG_Opcodelist3_<APERTURE>
```

- `MODEL`: one of `DP1M`, `DP2M`, `DP3M`, `SD1M`.
- `LENSID`: present only for `SD1M` (interchangeable lens). The bundled
  set covers the SD1 Merrill 30mm prime as `Unknown_(32776)_30mm`.
- `APERTURE`: f-number with one decimal place, matching standard
  third-stop steps (`2.8`, `3.2`, `3.5`, `4.0`, …, `14.0`).

| Body  | Apertures                                                          |
|-------|--------------------------------------------------------------------|
| DP1M  | f/4.0 – f/16.0 (16 stops)                                          |
| DP2M  | f/2.8 – f/16.0 (16 stops)                                          |
| DP3M  | f/2.8 – f/14.0 (15 stops)                                          |
| SD1M  | f/1.4 – f/14.0 (22 stops, 30mm prime only)                         |

Quattro and pre-Merrill bodies (SD9/SD10/SD14) are not covered.

## Format

Each file is a self-contained DNG `OpcodeList3` byte stream — a 4-byte
big-endian opcode count followed by one DNG `GainMap` (opcode ID 9)
record. The writer doesn't parse or modify them; the bytes are written
into the raw IFD as-is.

## Provenance and licensing

These calibration blobs originated as freely-distributed flat-field
files from the Sigma photographer community and were collected and
shipped (with no upstream license terms attached) by
[x3fuse](https://github.com/sagwaco/x3fuse). They are treated as
public-domain calibration data; redistribution is allowed.

If you have a higher-quality calibration for any of these bodies (or
calibrations for additional models / lenses), drop them into this
directory using the same filename convention and they'll be picked up
automatically.
