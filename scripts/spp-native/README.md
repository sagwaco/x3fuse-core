# Native oracles for the Merrill monochrome branch

These programs load the arm64 slice of SIGMA Photo Pro 6.9.0.0's
`lib99_MerillDLLForMac.dylib` with `dlopen`. Each runs one native function on
deterministic synthetic inputs and prints the fixture lines in
`crates/x3f-sys/tests/fixtures/`. They contain no Sigma code. The library is
not part of this repository and is never loaded by x3fuse at run time.

Extract the slice from an installed SIGMA Photo Pro 6.9 and check it:

```sh
lipo -thin arm64 \
  "/Applications/SIGMA_PhotoPro6.app/Contents/MonoBundle/Sensor/lib99_MerillDLLForMac.dylib" \
  -output reference/merrill/lib99_MerillDLLForMac.arm64.dylib
shasum -a 256 reference/merrill/lib99_MerillDLLForMac.arm64.dylib
# b728f4f1aa7d8c7598a04492b1cc5a0d29cf8a7fde7a34eb96b7cc7608ca496d
```

The universal library's SHA-256 is
`037419dc04bb1afe86dddad7a14888a1b36d6c00a0231ec50a6974a9963ce5e6`.

Build every oracle with `-ffp-contract=off`. Otherwise clang can fuse the
input generators' multiply-adds and change the synthetic inputs by one ulp.

```sh
clang++ -std=c++17 -arch arm64 -O1 -ffp-contract=off scripts/spp-native/sigma_hn.cpp -o /tmp/sigma_hn
/tmp/sigma_hn reference/merrill/lib99_MerillDLLForMac.arm64.dylib
```

| Oracle | Native function | Fixture |
| --- | --- | --- |
| `sigma_hn.cpp` | `Sigma_HN` 0x42434 | `spp-mono-hn-native.txt` |
| `match_col_offsets_mono.cpp` | mono `MatchColOffsets` 0x2f284 | `spp-mono-bcs-native.txt` |
| `hist_bilateral_mono.cpp` | `ApplyHistBilateralFilter` 0x3301c | `spp-mono-hist-bilateral-native.txt` |
| `desp_sharp_bnw.cpp` | `Despeckle3` 0x2ed50, `SharpenBnW` 0x81768, `ApplySharpLBnW` 0x810b0 | `spp-mono-desp-sharp-native.txt` |
| `mono_denoise.cpp` | `MonochromeDenoise` 0x2fd3c | `spp-mono-denoise-native.txt` |
| `output_mono.cpp` | `ProcessToOutput` 0x4f4e8 (row 0x4e9d0), `GenerateF20ToneCurve` 0x57c64, `CreateGammaLUT_F20` 0x561f4 | `merrill-mono-output-native.txt` |

Addresses are offsets from the slice's image base, computed at run time from the
exported `GetTanhLUT` (0x5d99c). `mono_denoise.cpp` runs from the repository
root because it reads a NoiseParams blob from `spp-image-native.txt`.
