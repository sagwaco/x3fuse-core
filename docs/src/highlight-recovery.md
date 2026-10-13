# DNG highlight recovery

Highlight recovery estimates brightness and color where sensor layers approach
clipping.

Recovery is off by default. Enable it when writing a Digital Negative (DNG) file:

```sh
x3f_extract -dng -compress -dng-highlight-recovery input.X3F
```

This writes `input.X3F.dng`. Use `-o output` to choose an existing output directory.
The white-balance, denoise, and lens-shading options still apply. Recovery does not
require a particular white-balance preset or a separate calibration file.

## Sensor support

The camera model selects the reconstruction method:

| Camera family | Method |
| --- | --- |
| DP1 Merrill, DP2 Merrill, DP3 Merrill, SD1 Merrill | Estimate brightness, reconstruct its spatial detail, then add color from nearby reliable pixels. |
| Older Foveon cameras | Estimate clipped layers from local layer ratios and the shared color lookup tables, with color safeguards. |
| Quattro | Apply lookup-table and fallback reconstruction after expanding the lower-resolution layers. |

Quattro's expanded layers are not independent full-resolution measurements, so
Quattro does not use the Merrill gradient method. The same output mapping options
are available for all three paths.

## Storing recovered highlights

Recovery can estimate values above the sensor’s recorded clipping limit. For example, a layer records 1.0 because it clipped. Recovery might estimate that its value would have been 1.4 without clipping.  

The mapping controls how these estimates fit into a 16-bit DNG.

### Linear mapping

Linear mapping is the default:

```sh
x3f_extract -dng -compress -dng-highlight-recovery \
  -dng-highlight-mapping linear input.X3F
```

The converter measures the largest recovered value in the active image area. It
uses that value as a shared scale, limited to the range one to 16, and divides the
image by that scale before encoding. Values within this range keep their relative
brightness differences. Values beyond the supported range clip during encoding.

The DNG writer adds the matching exposure adjustment to the `BaselineExposure` tag.
For a scale of two, the stored samples are half as large and the adjustment is one
stop. A reader that ignores this tag displays a darker image. Apply the corresponding
positive exposure adjustment in that reader.

The tags are `BlackLevel = 0`, `WhiteLevel = 65535`, and `LinearResponseLimit = 1`.
Even where recovery leaves a pixel alone, this shared scaling and 16-bit rounding
can make its stored samples differ from a recovery-off DNG. If an uncropped output
includes sensor borders outside the active area, recovery encodes those borders
as black.

### Shoulder mapping

Shoulder mapping compresses the brightest values instead of using an image-wide
exposure scale:

```sh
x3f_extract -dng -compress -dng-highlight-recovery \
  -dng-highlight-mapping shoulder input.X3F
```

The curve starts at `0.85` by default. The `X3F_DNG_SHOULDER_KNEE` environment
variable sets that starting point. Each pixel uses one scale for all three layers,
which preserves their ratios while compressing brightness.

The writer records the knee in `LinearResponseLimit` when compression is needed.
It does not add the linear mapping's headroom adjustment to `BaselineExposure`.
The compression is part of the stored image and cannot be undone by changing a tag.

Both mappings are available, but the Merrill image checks cover linear mapping.
Choosing a mapping with recovery off does not change the recovery-off result.

## Merrill reconstruction

Foveon records light in three overlapping sensor layers. These measurements are
not independent red, green, and blue image channels. Recovery uses a reliability
value for each layer: 255 means fully reliable, zero means clipped, and values in
between describe the approach to clipping.

The three stages form one Merrill recovery pipeline and run together. First, the
converter estimates brightness for each affected pixel from its surviving sensor
measurements. It refines brightness using spatial changes in reliable neighboring
measurements. Finally, a color field combines surviving same-pixel layer ratios
with color from reliable pixels around the affected region.

Brightness and color are estimated in separate stages. The color field uses
reconstructed brightness and surviving layer ratios to determine how color
evidence connects across the image. Its estimated layer ratios are then applied
at the reconstructed brightness for each pixel.

### 1. Estimate brightness (tone anchor)

The tone "anchor" gives each affected pixel a starting brightness. Where enough
signal remains, it uses brightness calculated from the camera calibration. When
at least two layers have low reliability, it shifts toward an estimate from the
remaining measurements.

Camera luminance can approach zero through cancellation between positive and
negative calibration coefficients. The matrix anchor loses influence smoothly
near that boundary. Severe recovery blends from the complete previous estimate,
including its strength, so a one-step reliability change cannot switch an
unusable matrix anchor directly to a full-strength dark result.

Where all layers are clipped, the method can estimate a neutral brightness from
the recorded levels, but those levels provide no measured texture. The estimate
is an assumption about brightness, not a physical measurement of the missing light.

### 2. Reconstruct brightness detail

The tone anchor estimates brightness separately for each affected pixel. This stage
uses spatial changes in the surviving layer measurements to reconstruct brightness
detail that the individual estimates may miss. A *gradient* describes the local
change between neighboring pixels. Here, the change is relative rather than absolute.

For example, two adjacent pixels may have the same recorded value in a clipped
layer, while a reliable layer has a 20% stronger signal at the second pixel. That
ratio supports an estimate that the second pixel is 20% brighter, rather than
equally bright. Other layers and neighbors may support different ratios. A layer
change can also reflect a color difference, so the ratio is evidence for a brightness
change, not proof.

The stage combines this evidence from each pixel's four immediate neighbors at
native resolution. A layer contributes only when both measurements are finite and
positive. The lower of the two reliability values sets the pair's influence.
Zero reliability gives the pair no influence.

The *solver* is the numerical routine that refines brightness estimates across the
affected region together. The anchors provide an overall brightness reference,
while reliable neighboring measurements can move the estimates away from their
starting values. A pixel without usable neighbor evidence keeps its anchor as-is.
Other recovery stages calculate the reliability values and reconstruct highlight
color.

Fully reliable pixels and pixels outside the reconstruction remain fixed during
this calculation. Negative rendered RGB alone does not trigger Merrill recovery.
The solver stops conversion if it cannot produce a valid result. It does not
silently switch to another reconstruction method.

### 3. Add highlight color

The color stage reconstructs a *color field*, a grid of logarithmic layer ratios
with 8-pixel spacing. Intact pixels supply color ratios at the boundaries of
affected regions. These pixels are the color *donors*. Reliable pairs of layers
at the same pixel also constrain the field. One surviving layer cannot determine
hue. A balanced pair does not establish white, because the missing third layer
could distinguish a colored surface from a neutral one.

The field connects affected regions according to agreement in reconstructed
brightness and surviving layer ratios. The donor search has no fixed radius.
The stage also checks boundary influence against the brightest cell in each
connected region, using both brightness and surviving ratios. A gradual path to
a much darker surface is not sufficient evidence for borrowing its color.
Unaffected low-signal cells cannot connect unrelated donors. These checks limit
color transfer, but they do not identify surfaces or materials.

Before averaging cells, the stage builds a noise-aware boundary guide from
native log-layer measurements and their pairwise differences. A normalized
Gaussian with sigma 0.85 pixels smooths that guide using reliability weights.
Partly clipped and dark layers have reduced influence; repaired samples supply
none. The guide and color equations share fourth-power reliability weighting and
a cubic dark-signal taper from 0.002 to 0.02. The source pixels and the brightness reconstruction are not blurred.

Strong boundaries exceed three estimated noise scales, with minimum log steps
of 0.035 for layers and 0.025 for ratios. Weak boundaries at 60% of that threshold
survive only when connected to a strong boundary by four-neighbor hysteresis.
The noise estimate is the median absolute smoothing residual on bright,
reliable samples, scaled by 0.35/0.6745. This scale is an empirical guide from
the exp-10 trials; scene texture can raise it, and it is not sensor calibration.

A native step above 20% in any fully reliable layer above 0.02 signal supplies an
additional strong boundary. This preserves fine stripes that smoothing erases
and that the residual-based noise estimate mistakes for noise. Where two such
native transitions lie inside the seven-pixel kernel footprint along an axis,
their smoothed guides can overlap. In that narrow region, a smoothed cut also
needs a native feature step exceeding the strong threshold. Native strong cuts
always remain. This localizes the barrier without cutting a diagonal stripe's
interior off from its donors. Boundaries elsewhere retain the smoothed guide.

Cells touching the wider guide footprint do not donate their mixed color or
connect the coarse field. Affected edge cells use the existing color equations
at native resolution, with no neighbor contribution across a localized boundary.
Other cells retain the coarse calculation. The share of native work depends on
scene texture and clipping; it can dominate on detailed scenes. Keeping the
wider refinement footprint avoids reintroducing mixed coarse averages while
localizing a fine boundary.

Removing mixed cells can leave a smooth coarse component without donors even
when a path through native pixels exists. Such a component uses its surviving
ratios and the neutral prior. Extending native refinement to those components
was tested on the three review images: it recovered color in a controlled
synthetic case but increased visible stripe spill and processing time. It is
not enabled. More donor connectivity is not, by itself, evidence of better
recovery.

These boundaries do not identify materials. Lighting changes can also stop color
borrowing; weak or fully clipped boundaries can remain undetected. A stripe still
needs surviving color evidence because a boundary alone cannot determine its
missing hue. This variant does not add the experimental confidence-based color
fade or directional/segmented donor selection.

The edge builder processes one feature at a time, stores threshold membership in
one byte per pixel, and uses scanline buffers for its separable Gaussian passes.
These storage choices do not change the convolution. The native-resolution
refinement can still require substantial memory for large clipped regions.

A *neutral prior* adds a soft constraint toward the neutral layer ratios from the
camera calibration. The constraint is strongest where pair evidence is absent,
but it does not classify a balanced pair as white. Regions without reachable
donors still use their surviving ratios and the neutral prior.

The prior uses a *color metric*, a measure of color difference derived from the
camera matrix. Spatial differences use the same metric, including connections to
intact boundaries. Equal distances in raw log-layer coordinates do not represent
equal color differences. Treating those coordinates as equivalent can rotate hue
when blending a partly clipped colored region toward a neutral neighbor. The
shared metric avoids that conflict without a hue-specific penalty. It does not
resolve every missing-layer ambiguity or correct optical fringing. Green colors
and defocus fringes are not inherently recovery errors.

When bad-pixel correction is enabled, the color stage excludes camera-marked
repair sites from its donors. Interpolated replacement values are not independent
color measurements. The pipeline tracks these repair sites separately from layer
clipping reliability, without changing the repair operation, tone reconstruction,
or exported source-reliability mask. This exclusion applies only to Merrill color
recovery, not to processing with recovery disabled or to other sensor families.

Interpolation brings the field back to native resolution, with weights based on
brightness and surviving layer ratios. The color stage checks each proposal
against trustworthy ratios at the target pixel. Disagreement shifts the proposal
toward calibrated neutral. The stage fits both the adjusted and unadjusted
proposals to the same-pixel measurements, then limits the adjustment: each pair's
reliability-weighted ratio error can increase by at most 0.035 log units relative
to the unadjusted fit. This bound uses the pair-agreement scale to limit the loss
of measured evidence. It is a heuristic, not a calibrated noise interval or a test
for white surfaces.

The color stage scales the fitted direction to the reconstructed brightness,
using luminance from the camera calibration. A direction whose nearly cancelled
luminance would require a layer above the encoder's 16-times headroom is rejected;
the stage tries the field direction, then calibrated neutral, retaining the tone
estimate if neither fits. Layer ratios are soft constraints,
not promises to retain absolute surviving-layer values alongside an independently
reconstructed brightness. Recovery strength increases smoothly as layer reliability
falls. The stage applies this blend once, not once per reconstruction step. Fully
reliable pixels retain their normalized sensor samples before shared output
scaling and 16-bit encoding.

`X3F_NO_CHROMA_LUT=1` disables the Merrill color field as well as the legacy
lookup-table color reconstruction. The Merrill brightness stages still run when
recovery is enabled. The override does not enable recovery on its own.

Headroom measurement and encoding read the same immutable color field. The solver
checks cancellation and recomputes the equation error before accepting its result.
If roundoff makes the accumulated color-solver residual optimistic, it restarts
from the recomputed residual within the original iteration budget and tolerance.
Invalid calibration or failure to converge stops conversion rather than selecting
another donor algorithm.

## Rust API

Use the `ProcessOptions` struct with the `Reader::dump_dng` method:

```rust,no_run
use x3f_core::{DngHighlightMapping, ProcessOptions, Reader};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut reader = Reader::open("input.X3F")?;
    let options = ProcessOptions {
        dng_highlight_recovery: true,
        dng_highlight_mapping: DngHighlightMapping::Linear,
        compress: true,
        ..ProcessOptions::default()
    };
    reader.dump_dng("input.X3F.dng", &options)?;
    Ok(())
}
```

The writer selects sensor-space processing. Direct `Reader::get_image` calls need
`ColorEncoding::None` and `cineon: false` to use DNG recovery. TIFF and PPM color
processing do not use the Merrill DNG method. The public C wrapper exposes metadata
and thumbnail access, not DNG conversion.

The library calculates Merrill brightness coefficients from the chosen white
balance and camera profile. It uses the same rounding as the DNG calibration tags.

## Implementation

The following files are under `crates/x3f-sys/src/`:

| File | Responsibility |
| --- | --- |
| `process.rs` | Select the processing path, build shared data, measure headroom, and encode pixels. |
| `highlight_recovery.rs` | Store layer reliability and donor data, and reconstruct older native-sensor highlights. |
| `tone_anchor.rs` | Estimate each affected pixel's starting brightness. |
| `gradient_tone.rs` | Solve for brightness using neighboring layer differences. |
| `color_field.rs` | Reconstruct highlight color from intact boundaries, surviving ratios, and a calibrated-neutral prior. |
| `color_edges.rs` | Build noise-aware boundaries used by coarse-cell selection and the native color solver. |
| `highlight_color.rs` | Apply those ratios at the chosen brightness. |
| `recovery_mask.rs` | Write the optional source-reliability mask. |

The pipeline builds the gradient and color fields before measuring headroom or
encoding pixels. Both passes read the same immutable fields and source reliability.
Encoding writes pixels in place without reading neighbors that it might have
overwritten.

Calibration helpers live under `crates/x3f-core/src/output/dng/`. The
`Reader::get_image` method passes their result into processing for each Merrill
conversion. Processing takes that result before starting parallel pixel work.

### Solver iterations

The solver starts with the tone anchors as its brightness estimates, then refines
the affected region:

1. **Compare the estimates with the evidence.** The solver checks how the estimated
   brightness differences agree with the surviving layer measurements, while
   accounting for departures from the anchors.
2. **Calculate a coordinated adjustment.** The solver finds changes to the affected
   pixels that reduce the combined disagreement. Reliable measurements have more
   influence than uncertain ones.
3. **Update and repeat.** The solver applies the adjustment, checks the revised
   estimates, and calculates another adjustment if needed.
4. **Validate the result.** The solver accepts the estimates when they satisfy the
   numerical checks. If the solver cannot reach an acceptable result within its
   processing limit, conversion returns an error.

The measurements, reliability values, and anchors remain unchanged throughout these
iterations. Fully reliable pixels and pixels outside the reconstruction remain
fixed. Only the brightness estimates under reconstruction change.

Neighboring measurements can conflict, so the solver balances their contributions
rather than reproducing every measured ratio exactly. It does not repeatedly
average or blur neighboring pixels. Passing the numerical checks confirms that the
solver completed its calculation, not that the estimates match the original scene.

## Masks, errors, and limits

Use the [recovery mask](./recovery-mask.md) to exclude pixels with stressed source
layers from color-profile training. The mask does not say which output pixels
changed or how much detail is recoverable.

Invalid calibration, missing required layer data, and numerical solver failures
return conversion errors. Recovery errors use the library's verbosity and log
callback settings. A failed conversion must not be treated as a usable output.

Recovery estimates brightness and color separately, and either estimate can be
wrong. Differences between neighboring layer measurements may come from a change
in color rather than brightness. Nearby pixels used to estimate highlight color
may belong to a different material.

These errors can make recovered areas too dark or give them the wrong color.
Review the recovered highlights and their transitions into surrounding areas in
the intended DNG reader. A file that opens successfully does not necessarily
contain convincing recovery.
