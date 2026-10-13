# Recovery mask

A recovery mask marks pixels whose source layers approach or reach clipping. Use
it to exclude those pixels from color-profile training.

## Export a mask

```sh
x3f_extract -dng -compress -dng-highlight-recovery -dng-recovery-mask \
  -o output input.X3F
```

Create the `output` directory before running the command. The converter writes:

- `output/input.X3F.dng`
- `output/input.X3F.mask.pgm`

Without `-o`, both files go beside the input. Batch conversion uses a separate
name for each input and rejects duplicate destinations.

Mask export is off by default. It requires DNG output with highlight recovery
already enabled. The mask option does not enable recovery or select its mapping.
The mask is a separate file, not an embedded DNG image.

## Read the mask

The file uses binary Portable Graymap (PGM) format, with a `P5` header and a maximum
value of 255. It covers the native active image area before orientation:

| Value | Meaning |
| --- | --- |
| `0` | All three source layers are fully reliable. |
| `255` | At least one layer has lower reliability. |

Reliability describes source measurements, not output appearance. A marked pixel
may have partially clipped layers or no usable recovery information. The mask is
not a before-and-after difference image or an estimate of recoverable detail.
An unmarked pixel is not a guarantee of accurate color.

Camera clipping maps contribute only when the `X3F_CAMERA_CLIP_MAPS` environment
variable is set to `1` and the camera provides supported maps. Mask export does not
enable these maps.

Mask export reads the same layer-reliability data as recovery without changing the
DNG image or its metadata. The exporter does not rotate, resize, or expand the
marked regions. Apply the same orientation and alignment to the mask as to the
training image.
Choose an exclusion margin large enough to cover any resampling or neighborhood
operations in the training process.

## Rust API

Set a file path in the `ProcessOptions::dng_recovery_mask` field:

```rust,no_run
use x3f_core::{ProcessOptions, Reader};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut reader = Reader::open("input.X3F")?;
    let options = ProcessOptions {
        dng_highlight_recovery: true,
        dng_recovery_mask: Some("input.X3F.mask.pgm".into()),
        ..ProcessOptions::default()
    };
    reader.dump_dng("input.X3F.dng", &options)?;
    Ok(())
}
```

The path names one file, not a directory or a template. It must differ from the
DNG destination. Concurrent conversions must use distinct output paths.

The DNG writer selects sensor-space processing. Direct calls to the
`Reader::get_image` method require `ColorEncoding::None` and `cineon: false`.
The request and export result belong to that conversion, including work performed
on parallel threads.

## Failures and partial files

The exporter needs native layer-reliability data. Quattro's expanded layers do not
provide that data, so a mask request fails on that path. The exporter checks the
available data rather than assuming that a camera name is sufficient.

An existing mask, including a symlink, is never overwritten. Invalid options,
missing layer data, and file creation or write failures return conversion errors.

The converter writes the mask before completing the DNG. The command-line tool
writes its DNG to a temporary file and then renames it. The two output files do not
succeed or fail as a pair: a partial mask, a complete mask without a DNG, or a
temporary DNG can remain after failure. The converter does not delete the mask
when a later step fails.

Check the conversion result, not just whether the mask exists. Inspect failed
outputs before retrying with fresh destinations. Before training, verify the mask's
size and alignment and keep a record that ties it to the source image and DNG.
