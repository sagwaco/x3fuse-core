# Embedding a DNG look

Use `-dng-look FILE` to add a DCP look table and tone curve to a DNG:

```sh
x3f_extract -dng -compress -dng-look look.dcp input.X3F
```

In Rust, set `ProcessOptions::dng_look` and use `Reader::dump_dng` or
`convert_file` with `OutputFormat::Dng`. The CLI and `convert_file` reject the
option for other formats. The lower-level TIFF and PPM writers do not apply it.

The option is off by default and does not enable highlight recovery. Without a
look, output is unchanged.

## What changes

The look belongs to the default embedded profile. Its tone curve replaces the
existing curve. Import copies only these fields:

- `ProfileLookTableDims`
- `ProfileLookTableData`
- `ProfileLookTableEncoding`
- `ProfileToneCurve`

Camera calibration, white balance, exposure metadata, raw samples, and the other
embedded profiles stay unchanged. The DNG backward version becomes 1.4 to support
the look encoding tag. This is look import, not replacement of the camera profile.

The selected look applies to every DNG conversion, regardless of camera model.
The DCP's `UniqueCameraModel` is neither required nor used to restrict import.
A look's appearance on other cameras still needs visual review.

## Accepted profiles

The reader accepts little-endian and big-endian extended-profile DCP files up to
16 MiB. A file must have a profile name, a complete look table, and a tone curve.
The curve must increase strictly from
`(0,0)` to `(1,1)`. Table values must be finite, scales must be nonnegative, and
neutral value scales must equal one. Supported encodings are linear and sRGB.
An omitted encoding means linear.

Malformed files, unsupported look values, and profiles that forbid embedding
produce an error. The converter does not substitute another look. Rust callers
can inspect a file with `dcp::DcpLook::open` or parse bytes with `from_bytes`.

## Previews

The option changes metadata, not the small embedded thumbnail. A viewer must
render the DNG using its embedded profile to show the look. A separate preview
renderer can use the completed DNG without applying the DCP again.

## Camera color modes

Use `-dng-color-mode <MODE>` to generate and embed a DNG camera profile from the color-mode parameters stored in a DP1, DP2 or DP3 Merrill X3F's camera metadata (CAMF). Select `Standard`, `Neutral`, `Vivid`, `Portrait`, `Landscape` or `FCBlue` (Foveon Classic Blue). The Rust API uses `ProcessOptions::dng_color_mode: Option<ColorMode>`. The converter requires no external DCP file or Sigma Photo Pro (SPP) runtime. The option is disabled by default. The recipe uses a ×2 setup gain, the selected mode's matrix and tone contrast, and metadata-derived ColorDQ. The file's `COLORSPACE` property selects native sRGB or Adobe RGB internal routing, independently of the renderer's output space. It does not reproduce the complete SPP pipeline.

```sh
x3f_extract -dng -wb Auto -denoise 10 -dng-highlight-recovery -dng-highlight-mapping linear -dng-color-mode Standard input.X3F
```

The writer embeds a coarse `[72,33,65]` look table and a separate 513-point tone curve. Baking uses the DNG's published calibration and rounded exposure. The tone curve carries neutral brightness, including compensation for Adobe's white-preserving negative-exposure adjustment. Its highlight tail can be flat. The look table carries the color residual around Adobe's hue-preserving tone operator. The selected mode's matrix right-multiplies the white-balance camera matrix before ColorDQ. Its contrast compensation adjusts the tone slopes. ColorDQ follows the combined matrix and setup gain, before tone. The setup gain does not multiply ColorDQ's correction amplitude. No `BaselineExposureOffset` is written.

Camera samples, calibration, white balance, `BaselineExposure`, headroom storage and generated previews remain identical to conversion with the same processing options and this flag disabled. Highlight recovery, metadata-based bad-pixel repair and the existing denoiser remain separate upstream operations. This option adds no sharpening, Fill Light, lens correction or native SPP detail reconstruction. The small embedded preview does not show the selected color-mode profile. Render the DNG with its default embedded profile to review it.

The option requires DNG output and linear highlight mapping on a DP1, DP2 or DP3 Merrill. Any white-balance preset with valid calibration in the file is supported, including its stored Custom setting. Use the preset names from the file's CAMF lists. The existing Daylight alias resolves to Sunlight when needed. The rendering recipe is shared across these cameras. Each file supplies the calibration for its camera and selected white balance. Use `-wb Auto` to select the file's Auto matrix, gains and color-shading parameters together, or `-wb Daylight` for daylight calibration. Auto selects stored camera calibration rather than running a new WB estimator. Do not combine the recipe with `-dng-look`. Missing source color space, missing mode data, unsupported calibration or unsupported tone parameters return an error. Non-Standard modes require shared CAMF tone-shape rows. The converter rejects differing rows rather than guessing their mapping. Multi-axis mode tables are not supported.

Standard review covers low-ISO examples from all three cameras under different lighting, with Daylight or Auto white balance and sRGB or Adobe RGB source metadata. Other color modes have a smaller DP2 Auto-WB review set. Their overall appearance follows the metadata-derived recipe, but color residuals remain. High-ISO color differences remain unresolved. The mode renders after the converter's existing denoising. Native SPP interleaves its mode matrix with denoising and reconstruction. Native denoising, banding suppression and sharpening are not reproduced by this camera profile. Adobe clips some values before profile finishing, and the coarse table interpolates the native correction. Exact native equivalence is not guaranteed. Other white-balance edits and readers need separate review.
