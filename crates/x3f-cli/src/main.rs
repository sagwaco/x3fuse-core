//! `x3f_extract` — command-line frontend over `x3f-core`.
//!
//! M0 deliberately preserves the legacy single-dash flag syntax
//! (`-tiff`, `-no-denoise`, `-color sRGB`, …) so existing test corpora and
//! shell scripts continue to work. A modern `--long-flag` interface and
//! subcommand layout will be introduced in a later milestone after the test
//! harness stabilises in M2.

mod x3i_report;

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use x3f_core::{
    frame_ev, is_x3i, set_max_printed_matrix_elements, set_offset_legacy, set_verbosity,
    X3iContainer,
};
use x3f_core::{ColorEncoding, ColorMode, DngHighlightMapping, ProcessOptions, Reader, Verbosity};

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum FileType {
    Meta,
    Jpeg,
    Raw,
    Tiff,
    Dng,
    PpmP3,
    PpmP6,
    Histogram,
}

impl FileType {
    /// Extension suffix appended to the input filename (with leading dot,
    /// matching the legacy C `extension[]` table).
    fn extension(self) -> &'static str {
        match self {
            FileType::Meta => ".meta",
            FileType::Jpeg => ".jpg",
            FileType::Raw => ".raw",
            FileType::Tiff => ".tif",
            FileType::Dng => ".dng",
            FileType::PpmP3 | FileType::PpmP6 => ".ppm",
            FileType::Histogram => ".csv",
        }
    }
}

#[derive(Debug)]
struct Args {
    x3i_frame: Option<usize>,
    x3i_extract: bool,
    x3i_report: bool,
    x3i_ghost_map: bool,
    output_explicit: bool,
    extract_jpg: bool,
    extract_raw: bool,
    extract_unconverted_raw: bool,
    crop: bool,
    fix_bad: bool,
    /// Denoise strength, 0..=10. 0 disables denoise (legacy `-no-denoise`),
    /// 10 is full strength (legacy default).
    denoise_intensity: u8,
    denoise_backend: x3f_core::DenoiseBackend,
    spp_denoise: bool,
    rgb_chroma_lut: Option<bool>,
    denoise_options: x3f_core::DenoiseOptions,
    apply_sgain: Option<bool>,
    file_type: FileType,
    color_encoding: ColorEncoding,
    color_mode: Option<ColorMode>,
    log_hist: bool,
    wb: Option<String>,
    compress: bool,
    use_opencl: bool,
    outdir: Option<PathBuf>,
    opcodes_dir: Option<PathBuf>,
    files: Vec<PathBuf>,
    verbosity: Option<Verbosity>,
    legacy_offset: Option<i32>,
    matrix_max: Option<u32>,
    dng_highlight_recovery: bool,
    dng_highlight_mapping: DngHighlightMapping,
    dng_highlight_mapping_explicit: bool,
    cineon: bool,
    /// Set when the user explicitly passed `-color <space>`. Used to
    /// resolve `-cineon` alone → ProPhotoRGB without overriding an
    /// explicit user choice.
    color_explicit: bool,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            x3i_frame: None,
            x3i_extract: false,
            x3i_report: false,
            x3i_ghost_map: false,
            output_explicit: false,
            extract_jpg: false,
            extract_raw: true,
            extract_unconverted_raw: false,
            crop: true,
            fix_bad: true,
            denoise_intensity: 10,
            denoise_backend: x3f_core::DenoiseBackend::Auto,
            spp_denoise: false,
            rgb_chroma_lut: None,
            denoise_options: x3f_core::DenoiseOptions::default(),
            apply_sgain: None,
            file_type: FileType::Dng,
            color_encoding: ColorEncoding::Srgb,
            color_mode: None,
            log_hist: false,
            wb: None,
            compress: false,
            use_opencl: false,
            outdir: None,
            opcodes_dir: None,
            files: Vec::new(),
            verbosity: None,
            legacy_offset: None,
            matrix_max: None,
            dng_highlight_recovery: false,
            dng_highlight_mapping: DngHighlightMapping::Linear,
            dng_highlight_mapping_explicit: false,
            cineon: false,
            color_explicit: false,
        }
    }
}

fn usage(progname: &str) -> ! {
    eprintln!(
        "usage: {progname} <SWITCHES> <file1> ...\n\
         \x20  -o <DIR>        Use <DIR> as output directory\n\
         \x20  -v              Verbose output for debugging\n\
         \x20  -q              Suppress all messages except errors\n\
         \x20  -x3i-ghost-map  Write research ghost weights as per-frame PGM images\n\
         \x20  -x3i-report     Write X3I radiometry and peak RSS beside the DNG\n\
         \x20  -x3i-frame <n>  Convert X3I frame n (zero based)\n\
         \x20  -x3i-extract    Copy all X3I frames to new X3F files\n\
         ONE OFF THE FORMAT SWITCHWES\n\
         \x20  -meta           Dump metadata\n\
         \x20  -jpg            Dump embedded JPEG\n\
         \x20  -raw            Dump RAW area undecoded\n\
         \x20  -tiff           Dump RAW/color as 3x16 bit TIFF\n\
         \x20  -dng            Dump RAW as DNG LinearRaw (default)\n\
         \x20  -ppm-ascii      Dump RAW/color as 3x16 bit PPM/P3 (ascii)\n\
         \x20                  NOTE: 16 bit PPM/P3 is not generally supported\n\
         \x20  -ppm            Dump RAW/color as 3x16 bit PPM/P6 (binary)\n\
         \x20  -histogram      Dump histogram as csv file\n\
         \x20  -loghist        Dump histogram as csv file, with log exposure\n\
         APPROPRIATE COMBINATIONS OF MODIFIER SWITCHES\n\
         \x20  -color <COLOR>  Convert to RGB color space\n\
         \x20                  (none, sRGB, AdobeRGB, ProPhotoRGB)\n\
         \x20                  'none' means neither scaling, applying gamma\n\
         \x20                  nor converting color space.\n\
         \x20                  This switch does not affect DNG output\n\
         \x20  -color-mode <MODE>  Select a camera color mode at normal strength\n\
         \x20                  as-shot, Standard, Vivid, Neutral, Portrait,\n\
         \x20                  Landscape, FOV Classic Blue; dp Quattro also supports\n\
         \x20                  Sunset Red, Forest Green, FOV Classic Yellow, Cinema,\n\
         \x20                  Teal and Orange. Quote names that contain spaces.\n\
         \x20                  Short names such as FCBlue remain accepted.\n\
         \x20                  Color TIFF applies the look to pixels. DNG embeds\n\
         \x20                  selectable profiles with this requested default.\n\
         \x20                  DNG looks are approximations at export white balance.\n\
         \x20                  Omitting the option preserves existing output.\n\
         \x20  -unprocessed    Dump RAW without any preprocessing\n\
         \x20  -qtop           Dump Quattro top layer without preprocessing\n\
         \x20  -no-crop        Do not crop to active area\n\
         \x20  -spp-pipeline   Use Merrill SPP recovery and denoise on CPU\n\
         \x20                  Neutral noise sliders; TIFF, PPM or histogram\n\
         \x20                  Requires default denoise options and -denoise 10\n\
         \x20  -no-denoise     Disable denoising (same as -denoise 0)\n\
         \x20  -denoiser <nlm|bm3d>  Select algorithm (default: nlm)\n\
         \x20  -denoise-luma / -no-denoise-luma  Luminance (default: off)\n\
         \x20  -denoise-chroma / -no-denoise-chroma  Chroma (default: on)\n\
         \x20  -denoise-strength <0.1-3>  Sensor strength multiplier (default: 1)\n\
         \x20  -nlm-luma-patch / -nlm-chroma-patch <1-9>  Odd patch size\n\
         \x20  -nlm-luma-search / -nlm-chroma-search <1-31>  Odd search size\n\
         \x20  -denoise-backend <auto|cpu|gpu>  Select execution backend\n\
         \x20  -denoise <0-10> Denoise intensity. 0 = off,\n\
         \x20                  10 = full strength (default). Intermediate\n\
         \x20                  values scale the selected filter strength.\n\
         \x20  -no-sgain       Do not apply spatial gain (color compensation)\n\
         \x20  -no-fix-bad     Do not fix bad pixels\n\
         \x20  -chroma-lut     Enable scene-derived RGB highlight recovery\n\
         \x20  -no-chroma-lut  Disable that LUT; retain prior recovery and gate\n\
         \x20                  Both override X3F_NO_CHROMA_LUT for rendered\n\
         \x20                  sRGB/AdobeRGB/ProPhotoRGB output.\n\
         \x20                  Default: LUT on unless the environment disables it.\n\
         \x20                  -chroma-lut cannot combine with -spp-pipeline.\n\
         \x20                  DNG, -color none, raw and Cineon ignore these flags.\n\
         \x20                  The environment still controls the separate DNG\n\
         \x20                  recovery LUT, including -color none output.\n\
         \x20  -sgain          Apply spatial gain (default except for Quattro)\n\
         \x20  -wb <WB>        Select white balance preset\n\
         \x20  -compress       Enable lossless compression (DNG: lossless\n\
         \x20                  JPEG, TIFF: ZIP/Deflate)\n\
         \x20  -opcodes-dir <DIR>  Directory of pre-rendered DNG OpcodeList3\n\
         \x20                  flat-fielding blobs. When set, the matching\n\
         \x20                  per-(model, aperture[, lens]) blob is embedded\n\
         \x20                  into the DNG raw IFD's OpcodeList3 tag. Files\n\
         \x20                  follow the x3fuse layout: <MODEL>[_<LENS>]_FF_DNG_Opcodelist3_<APERTURE>.\n\
         \x20  -dng-highlight-recovery\n\
         \x20                  Recover clipped Foveon highlights when writing\n\
         \x20                  DNG, preserving linear highlight headroom.\n\
         \x20                  BaselineExposure restores default brightness;\n\
         \x20                  readers that ignore it show a darker image.\n\
         \x20                  Default: off (matches the pre-Rust C writer).\n\
         \x20  -dng-highlight-mapping <linear|shoulder>\n\
         \x20                  Choose recovered highlight mapping (default:\n\
         \x20                  linear). Requires -dng-highlight-recovery to\n\
         \x20                  have an effect. shoulder bakes the legacy soft\n\
         \x20                  shoulder into the raster; its knee is controlled\n\
         \x20                  by X3F_DNG_SHOULDER_KNEE (default 0.85).\n\
         \x20  -cineon         Write a 16-bit TIFF with a Cineon-style log tone\n\
         \x20                  curve (lifted shadows, pulled highlights, flat\n\
         \x20                  midtones) baked into the pixels and the Foveon\n\
         \x20                  highlight-recovery passes (chroma LUT, RepairPix)\n\
         \x20                  bypassed. The file looks flat in any viewer;\n\
         \x20                  apply your own log→linear input transform in the\n\
         \x20                  grading app. Combine with -color to pick the\n\
         \x20                  primaries; defaults to ProPhotoRGB. An ICC\n\
         \x20                  profile declaring those primaries with a *linear*\n\
         \x20                  TRC is embedded so colour-managed readers leave\n\
         \x20                  the log encoding intact. Override the curve scale\n\
         \x20                  with X3F_CINEON_SCALE (default 100; 12 ≈ Cineon\n\
         \x20                  film-print, 50 ≈ moderate log; larger is flatter\n\
         \x20                  still at the cost of shadow noise). Requires -tiff.\n\
         \x20  -ocl            Ignored (no OpenCL backend)\n\
         \n\
         STRANGE STUFF\n\
         \x20  -offset <OFF>   Offset for SD14 and older\n\
         \x20                  NOTE: If not given, then offset is automatic\n\
         \x20  -matrixmax <M>  Max num matrix elements in metadata (def=100)\n",
    );
    std::process::exit(1);
}

/// Parse command-line arguments, mirroring the legacy x3f_extract.c semantics.
///
/// Notable quirks preserved from the original:
///  - Format flags (`-tiff`, `-dng`, …) are mutually exclusive; only the last
///    wins. They also reset the `extract_*` booleans so the previous format
///    does not bleed through.
///  - `-color <COLOR>` and `-unprocessed`/`-qtop` all write to the same field;
///    last one wins.
///  - Anything starting with `-` that isn't recognised triggers usage; the
///    first non-flag argument starts the file list.
fn parse_args(argv: &[String]) -> Args {
    let progname = argv.first().map(String::as_str).unwrap_or("x3f_extract");
    let mut args = Args::default();
    let argv = &argv[1..];
    let mut i = 0;
    while i < argv.len() {
        let a = argv[i].as_str();
        // Reset closure for the format flags' Z-macro behaviour.
        let reset = |args: &mut Args| {
            args.output_explicit = true;
            args.extract_jpg = false;
            args.extract_raw = false;
            args.extract_unconverted_raw = false;
        };
        match a {
            "-x3i-frame" => {
                i += 1;
                args.x3i_frame = Some(
                    argv.get(i)
                        .and_then(|n| n.parse().ok())
                        .unwrap_or_else(|| usage(progname)),
                );
            }
            "-x3i-extract" => args.x3i_extract = true,
            "-x3i-report" => args.x3i_report = true,
            "-x3i-ghost-map" => args.x3i_ghost_map = true,
            "-jpg" => {
                reset(&mut args);
                args.extract_jpg = true;
                args.file_type = FileType::Jpeg;
            }
            "-meta" => {
                reset(&mut args);
                args.file_type = FileType::Meta;
            }
            "-raw" => {
                reset(&mut args);
                args.extract_unconverted_raw = true;
                args.file_type = FileType::Raw;
            }
            "-tiff" => {
                reset(&mut args);
                args.extract_raw = true;
                args.file_type = FileType::Tiff;
            }
            "-dng" => {
                reset(&mut args);
                args.extract_raw = true;
                args.file_type = FileType::Dng;
            }
            "-ppm-ascii" => {
                reset(&mut args);
                args.extract_raw = true;
                args.file_type = FileType::PpmP3;
            }
            "-ppm" => {
                reset(&mut args);
                args.extract_raw = true;
                args.file_type = FileType::PpmP6;
            }
            "-histogram" => {
                reset(&mut args);
                args.extract_raw = true;
                args.file_type = FileType::Histogram;
                args.log_hist = false;
            }
            "-loghist" => {
                reset(&mut args);
                args.extract_raw = true;
                args.file_type = FileType::Histogram;
                args.log_hist = true;
            }
            "-color" => {
                i += 1;
                let v = argv
                    .get(i)
                    .map(String::as_str)
                    .unwrap_or_else(|| usage(progname));
                args.color_encoding = match v {
                    "none" => ColorEncoding::None,
                    "sRGB" => ColorEncoding::Srgb,
                    "AdobeRGB" => ColorEncoding::AdobeRgb,
                    "ProPhotoRGB" => ColorEncoding::ProPhotoRgb,
                    _ => {
                        eprintln!("Unknown color encoding: {v}");
                        usage(progname);
                    }
                };
                args.color_explicit = true;
            }
            "-color-mode" => {
                i += 1;
                let value = argv.get(i).unwrap_or_else(|| usage(progname));
                args.color_mode = Some(value.parse().unwrap_or_else(|_| {
                    eprintln!("Unknown camera color mode: {value}");
                    usage(progname)
                }));
            }
            "-cineon" => args.cineon = true,
            "-o" => {
                i += 1;
                let v = argv.get(i).unwrap_or_else(|| usage(progname));
                args.outdir = Some(PathBuf::from(v));
            }
            "-v" => args.verbosity = Some(Verbosity::Debug),
            "-q" => args.verbosity = Some(Verbosity::Error),
            "-unprocessed" => {
                args.color_encoding = ColorEncoding::Unprocessed;
                args.color_explicit = true;
            }
            "-qtop" => {
                args.color_encoding = ColorEncoding::Qtop;
                args.color_explicit = true;
            }
            "-no-crop" => args.crop = false,
            "-no-fix-bad" => args.fix_bad = false,
            "-chroma-lut" => args.rgb_chroma_lut = Some(true),
            "-no-chroma-lut" => args.rgb_chroma_lut = Some(false),
            "-no-denoise" => args.denoise_intensity = 0,
            "-spp-pipeline" => args.spp_denoise = true,
            "-denoiser" => {
                i += 1;
                args.denoise_options.algorithm = argv
                    .get(i)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or_else(|| usage(progname));
            }
            "-denoise-luma" => args.denoise_options.luminance = true,
            "-no-denoise-luma" => args.denoise_options.luminance = false,
            "-denoise-chroma" => args.denoise_options.chroma = true,
            "-no-denoise-chroma" => args.denoise_options.chroma = false,
            "-denoise-strength" => {
                i += 1;
                args.denoise_options.strength = argv
                    .get(i)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or_else(|| usage(progname));
                args.denoise_options
                    .validate()
                    .unwrap_or_else(|_| usage(progname));
            }
            "-nlm-luma-patch" | "-nlm-chroma-patch" | "-nlm-luma-search" | "-nlm-chroma-search" => {
                let flag = argv[i].as_str();
                i += 1;
                let value = argv
                    .get(i)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or_else(|| usage(progname));
                match flag {
                    "-nlm-luma-patch" => args.denoise_options.nlm.luma_patch = value,
                    "-nlm-chroma-patch" => args.denoise_options.nlm.chroma_patch = value,
                    "-nlm-luma-search" => args.denoise_options.nlm.luma_search = value,
                    _ => args.denoise_options.nlm.chroma_search = value,
                }
                args.denoise_options
                    .validate()
                    .unwrap_or_else(|_| usage(progname));
            }
            "-denoise-backend" => {
                i += 1;
                let v = argv.get(i).unwrap_or_else(|| usage(progname));
                args.denoise_backend = v.parse().unwrap_or_else(|_| usage(progname));
            }
            "-denoise" => {
                i += 1;
                let v = argv.get(i).unwrap_or_else(|| usage(progname));
                match v.parse::<u8>() {
                    Ok(n) if n <= 10 => args.denoise_intensity = n,
                    _ => usage(progname),
                }
            }
            "-no-sgain" => args.apply_sgain = Some(false),
            "-sgain" => args.apply_sgain = Some(true),
            "-wb" => {
                i += 1;
                let v = argv.get(i).unwrap_or_else(|| usage(progname));
                args.wb = Some(v.clone());
            }
            "-compress" => args.compress = true,
            "-opcodes-dir" => {
                i += 1;
                let v = argv.get(i).unwrap_or_else(|| usage(progname));
                args.opcodes_dir = Some(PathBuf::from(v));
            }
            "-dng-highlight-recovery" => args.dng_highlight_recovery = true,
            "-dng-highlight-mapping" => {
                args.dng_highlight_mapping_explicit = true;
                i += 1;
                let v = argv
                    .get(i)
                    .map(String::as_str)
                    .unwrap_or_else(|| usage(progname));
                args.dng_highlight_mapping = match v {
                    "linear" => DngHighlightMapping::Linear,
                    "shoulder" => DngHighlightMapping::Shoulder,
                    _ => {
                        eprintln!("Unknown DNG highlight mapping: {v}");
                        usage(progname);
                    }
                };
            }
            "-ocl" => args.use_opencl = true,
            "-offset" => {
                i += 1;
                let v = argv.get(i).unwrap_or_else(|| usage(progname));
                args.legacy_offset = Some(v.parse().unwrap_or_else(|_| usage(progname)));
            }
            "-matrixmax" => {
                i += 1;
                let v = argv.get(i).unwrap_or_else(|| usage(progname));
                args.matrix_max = Some(v.parse().unwrap_or_else(|_| usage(progname)));
            }
            other if other.starts_with('-') => usage(progname),
            _ => {
                // Start of file list; consume everything that remains.
                args.files = argv[i..].iter().map(PathBuf::from).collect();
                normalize(&mut args);
                return args;
            }
        }
        i += 1;
    }
    normalize(&mut args);
    args
}

/// Post-parse defaults that depend on combinations of flags rather than a
/// single switch. Currently: `-cineon` alone (no explicit `-color`) defaults
/// to ProPhotoRGB primaries — the widest standard gamut among the matrices
/// already in the pipeline, which is what color-grading workflows expect.
fn normalize(args: &mut Args) {
    if args.cineon && !args.color_explicit {
        args.color_encoding = ColorEncoding::ProPhotoRgb;
    }
}

/// Reject unsupported flag combinations. Check X3I inputs by their magic.
fn validate_args(args: &Args) -> Result<(), String> {
    if args.x3i_ghost_map
        && (!matches!(
            args.file_type,
            FileType::Dng | FileType::Tiff | FileType::PpmP3 | FileType::PpmP6
        ) || args.x3i_frame.is_some()
            || args.x3i_extract)
    {
        return Err("-x3i-ghost-map requires an X3I merge with -dng, -tiff or -ppm".into());
    }
    if args.x3i_report
        && (args.file_type != FileType::Dng || args.x3i_frame.is_some() || args.x3i_extract)
    {
        return Err("-x3i-report requires an X3I merge with -dng".into());
    }
    if args.x3i_extract && (args.output_explicit || args.x3i_frame.is_some()) {
        return Err("-x3i-extract cannot combine with output flags or -x3i-frame".into());
    }
    if args.x3i_frame.is_some()
        || args.x3i_extract
        || args.x3i_report
        || args.x3i_ghost_map
        || args.spp_denoise
    {
        for path in &args.files {
            match is_x3i(path) {
                Ok(x3i) => validate_x3i_input(args, x3i)?,
                Err(error)
                    if args.x3i_frame.is_some()
                        || args.x3i_extract
                        || args.x3i_report
                        || args.x3i_ghost_map =>
                {
                    return Err(format!("{}: {error}", path.display()));
                }
                // The conversion reports ordinary input errors as before.
                Err(_) => {}
            }
        }
    }
    if args.color_mode.is_some() {
        if !matches!(args.file_type, FileType::Tiff | FileType::Dng) {
            return Err("-color-mode requires color TIFF or DNG output".into());
        }
        if args.cineon
            || (args.file_type == FileType::Tiff
                && !matches!(
                    args.color_encoding,
                    ColorEncoding::Srgb | ColorEncoding::AdobeRgb | ColorEncoding::ProPhotoRgb
                ))
        {
            return Err("-color-mode requires color TIFF without Cineon, or DNG output".into());
        }
    }
    if args.spp_denoise {
        if args.rgb_chroma_lut == Some(true) {
            return Err(
                "-spp-pipeline cannot combine with -chroma-lut; select one pipeline".into(),
            );
        }
        if !matches!(
            args.file_type,
            FileType::Tiff | FileType::PpmP3 | FileType::PpmP6 | FileType::Histogram
        ) || args.cineon
            || !matches!(
                args.color_encoding,
                ColorEncoding::Srgb | ColorEncoding::AdobeRgb | ColorEncoding::ProPhotoRgb
            )
        {
            return Err("-spp-pipeline requires color TIFF, PPM or histogram output".into());
        }
        if args.denoise_intensity != 10 {
            return Err(
                "-spp-pipeline uses neutral SPP noise sliders and requires -denoise 10".into(),
            );
        }
        if args.denoise_backend == x3f_core::DenoiseBackend::Gpu {
            return Err("-spp-pipeline supports -denoise-backend auto or cpu".into());
        }
        if args.denoise_options != x3f_core::DenoiseOptions::default() {
            return Err("-spp-pipeline requires default denoise options".into());
        }
    }
    if args.cineon && args.file_type != FileType::Tiff {
        return Err(
            "-cineon is only meaningful with -tiff (other formats apply their own gamma/profile)"
                .to_string(),
        );
    }
    if args.cineon
        && (args.color_encoding == ColorEncoding::Unprocessed
            || args.color_encoding == ColorEncoding::Qtop)
    {
        return Err(
            "-cineon cannot combine with -unprocessed or -qtop (they bypass the matrix pipeline)"
                .to_string(),
        );
    }
    Ok(())
}

fn validate_x3i_input(args: &Args, x3i: bool) -> Result<(), String> {
    if !x3i
        && (args.x3i_frame.is_some() || args.x3i_extract || args.x3i_report || args.x3i_ghost_map)
    {
        return Err(
            "-x3i-frame, -x3i-extract, -x3i-report and -x3i-ghost-map require an X3I input".into(),
        );
    }
    if x3i && args.spp_denoise {
        return Err("-spp-pipeline does not support X3I input".into());
    }
    Ok(())
}

/// Produce `(tmpfile, outfile)` paths from an input filename, replicating
/// the legacy `make_paths()` exactly.
///
/// The legacy semantics are:
/// - If `outdir` is given: outfile = `<outdir>/<basename(infile)><ext>`
/// - Otherwise:            outfile = `<infile><ext>`
/// - tmpfile = `<outfile>.tmp`
///
/// Note that `ext` already includes the leading dot (e.g. `.meta`) and is
/// *appended* to the full filename — `input.X3F` becomes `input.X3F.meta`,
/// not `input.meta`. Tests rely on this exact spelling.
fn make_paths(infile: &Path, outdir: Option<&Path>, ext: &str) -> (PathBuf, PathBuf) {
    use std::ffi::OsString;
    let mut outfile = OsString::new();
    if let Some(dir) = outdir {
        outfile.push(dir);
        // `as_encoded_bytes()` (stable since 1.74) is byte-identical to the
        // Unix-only `OsStrExt::as_bytes()` on Unix, but also compiles on
        // Windows — keeping the legacy `/`-separator spelling cross-platform.
        // ASCII bytes like `/` survive the encoding intact on every host.
        let dir_bytes = dir.as_os_str().as_encoded_bytes();
        if !dir_bytes.ends_with(b"/") {
            outfile.push("/");
        }
        outfile.push(infile.file_name().unwrap_or_default());
    } else {
        outfile.push(infile.as_os_str());
    }
    outfile.push(ext);
    let mut tmpfile = outfile.clone();
    tmpfile.push(".tmp");
    (PathBuf::from(tmpfile), PathBuf::from(outfile))
}

fn convert_one(infile: &Path, args: &Args) -> Result<(), String> {
    let opts = ProcessOptions {
        color_encoding: args.color_encoding,
        color_mode: args.color_mode,
        crop: args.crop,
        fix_bad: args.fix_bad,
        denoise_intensity: args.denoise_intensity,
        denoise_backend: args.denoise_backend,
        spp_denoise: args.spp_denoise,
        rgb_chroma_lut: args.rgb_chroma_lut,
        denoise_options: args.denoise_options,
        apply_sgain: args.apply_sgain,
        wb: args.wb.clone(),
        compress: args.compress,
        opcodes_dir: args.opcodes_dir.clone(),
        dng_highlight_recovery: args.dng_highlight_recovery,
        dng_highlight_mapping: args.dng_highlight_mapping,
        cineon: args.cineon,
        x3i_ghost_maps: args.x3i_ghost_map,
    };

    let container =
        if is_x3i(infile).map_err(|e| format!("could not open `{}`: {e}", infile.display()))? {
            Some(X3iContainer::open(infile).map_err(|e| e.to_string())?)
        } else {
            None
        };
    if args.x3i_extract {
        let container = container
            .as_ref()
            .ok_or("-x3i-extract requires an X3I input")?;
        let directory = args
            .outdir
            .as_deref()
            .unwrap_or_else(|| infile.parent().unwrap_or(Path::new(".")));
        for index in 0..container.frames().len() {
            let mut name = infile.file_stem().unwrap_or_default().to_os_string();
            name.push(format!("_{index}.X3F"));
            container
                .extract_frame(index, &directory.join(name))
                .map_err(|e| e.to_string())?;
        }
        return Ok(());
    }
    if let (Some(container), None) = (&container, args.x3i_frame) {
        if matches!(
            args.file_type,
            FileType::Dng | FileType::Tiff | FileType::PpmP3 | FileType::PpmP6
        ) {
            let format = match args.file_type {
                FileType::Dng => x3f_core::OutputFormat::Dng,
                FileType::Tiff => x3f_core::OutputFormat::Tiff,
                FileType::PpmP3 => x3f_core::OutputFormat::PpmAscii,
                FileType::PpmP6 => x3f_core::OutputFormat::Ppm,
                _ => unreachable!(),
            };
            let (tmpfile, outfile) =
                make_paths(infile, args.outdir.as_deref(), args.file_type.extension());
            let _ = std::fs::remove_file(&tmpfile);
            let mut report = x3f_core::convert_file(
                infile,
                &tmpfile,
                format,
                &opts,
                &std::sync::atomic::AtomicBool::new(false),
                |_| {},
            )
            .map_err(|e| e.to_string())?;
            add_x3i_warnings(args, &mut report.warnings);
            for warning in &report.warnings {
                if show_x3i_warning(args, warning) {
                    eprintln!("{warning}");
                }
            }
            std::fs::rename(&tmpfile, &outfile).map_err(|e| e.to_string())?;
            for map in &report.ghost_maps {
                let (map_tmp, map_out) = make_paths(
                    infile,
                    args.outdir.as_deref(),
                    &format!(".ghost{}.pgm", map.frame_index),
                );
                let _ = std::fs::remove_file(&map_tmp);
                let result = (|| -> std::io::Result<()> {
                    use std::io::Write;
                    let mut file = std::fs::File::create(&map_tmp)?;
                    write!(file, "P5\n{} {}\n255\n", map.width, map.height)?;
                    file.write_all(&map.data)?;
                    drop(file);
                    std::fs::rename(&map_tmp, &map_out)
                })();
                if let Err(error) = result {
                    let _ = std::fs::remove_file(&map_tmp);
                    return Err(format!(
                        "X3I ghost map failed: {error}; output saved to `{}`",
                        outfile.display()
                    ));
                }
            }
            if args.x3i_report {
                let (report_tmp, report_out) =
                    make_paths(infile, args.outdir.as_deref(), ".x3i-report");
                let _ = std::fs::remove_file(&report_tmp);
                let result = (|| {
                    let mut text = x3i_report::radiometry(container, &opts)?;
                    for line in &report.warnings {
                        text.push_str(line);
                        text.push('\n');
                    }
                    text.push_str(&x3i_report::peak_rss()?);
                    std::fs::write(&report_tmp, text).map_err(|e| e.to_string())?;
                    std::fs::rename(&report_tmp, &report_out).map_err(|e| e.to_string())
                })();
                if let Err(error) = result {
                    let _ = std::fs::remove_file(&report_tmp);
                    return Err(format!(
                        "X3I report failed: {error}; DNG saved to `{}`",
                        outfile.display()
                    ));
                }
            }
            return Ok(());
        }
        if args.file_type != FileType::Meta {
            return Err(
                "X3I input: use -dng, -tiff, -ppm, -x3i-frame <n>, -x3i-extract or -meta".into(),
            );
        }
    }
    let mut table = String::new();
    let mut reader = match (&container, args.x3i_frame, args.file_type) {
        (Some(container), Some(index), _) => container.open_frame(index),
        (Some(container), None, FileType::Meta) => {
            table.push_str("X3I frames\nindex offset length exposure_us EV\n");
            let reference = container.frame_exposure_us(0).ok();
            for (index, entry) in container.frames().iter().enumerate() {
                let us = if index == 0 {
                    reference
                } else {
                    container.frame_exposure_us(index).ok()
                };
                let ev = reference
                    .zip(us)
                    .map(|(reference, us)| format!("{:+.6}", frame_ev(reference, us)))
                    .unwrap_or_else(|| "-".into());
                let exposure = us.map(|us| us.to_string()).unwrap_or_else(|| "-".into());
                table.push_str(&format!(
                    "{index} {} {} {exposure} {ev}\n",
                    entry.offset, entry.length
                ));
            }
            table.push('\n');
            container.open_frame(0)
        }
        (Some(_), None, _) => unreachable!("X3I merge handled above"),
        _ => Reader::open(infile),
    }
    .map_err(|e| format!("could not open `{}`: {e}", infile.display()))?;

    if args.extract_jpg {
        reader.load_thumbnail_jpeg().map_err(|e| {
            format!(
                "could not load JPEG thumbnail from `{}`: {e}",
                infile.display()
            )
        })?;
    }

    let extract_meta = args.file_type == FileType::Meta
        || args.file_type == FileType::Dng
        || (args.extract_raw
            && (args.crop
                || (args.color_encoding != ColorEncoding::Unprocessed
                    && args.color_encoding != ColorEncoding::Qtop)));

    if extract_meta {
        reader
            .load_camf()
            .map_err(|e| format!("could not load CAMF from `{}`: {e}", infile.display()))?;
        reader
            .load_property_list()
            .map_err(|e| format!("could not load PROP from `{}`: {e}", infile.display()))?;
    }

    if args.extract_raw {
        reader
            .load_raw()
            .map_err(|e| format!("could not load RAW from `{}`: {e}", infile.display()))?;
    }

    if args.extract_unconverted_raw {
        reader.load_unconverted_raw().map_err(|e| {
            format!(
                "could not load unconverted RAW from `{}`: {e}",
                infile.display()
            )
        })?;
    }

    let extension = match args.x3i_frame {
        Some(index) => format!("_{index}{}", args.file_type.extension()),
        None => args.file_type.extension().to_string(),
    };
    let (tmpfile, outfile) = make_paths(infile, args.outdir.as_deref(), &extension);

    // Match legacy behavior: unlink the temp first so we never write through
    // a stale file from a previous failed run.
    let _ = std::fs::remove_file(&tmpfile);

    let result = match args.file_type {
        FileType::Meta => reader.dump_meta(&tmpfile),
        FileType::Jpeg => reader.dump_jpeg(&tmpfile),
        FileType::Raw => reader.dump_raw_block(&tmpfile),
        FileType::Tiff => reader.dump_tiff(&tmpfile, &opts),
        FileType::Dng => reader.dump_dng(&tmpfile, &opts),
        FileType::PpmP3 => reader.dump_ppm(&tmpfile, &opts, false),
        FileType::PpmP6 => reader.dump_ppm(&tmpfile, &opts, true),
        FileType::Histogram => reader.dump_histogram(&tmpfile, &opts, args.log_hist),
    };

    result.map_err(|e| format!("could not dump to `{}`: {e}", tmpfile.display()))?;
    if !table.is_empty() {
        let metadata = std::fs::read(&tmpfile).map_err(|e| e.to_string())?;
        let mut text = table.into_bytes();
        text.extend(metadata);
        std::fs::write(&tmpfile, text).map_err(|e| e.to_string())?;
    }
    std::fs::rename(&tmpfile, &outfile).map_err(|e| {
        format!(
            "could not rename `{}` to `{}`: {e}",
            tmpfile.display(),
            outfile.display()
        )
    })?;
    Ok(())
}

fn add_x3i_warnings(args: &Args, warnings: &mut Vec<String>) {
    if args.dng_highlight_recovery || args.dng_highlight_mapping_explicit {
        warnings.push(
            "-dng-highlight-recovery and -dng-highlight-mapping have no effect on X3I merges"
                .into(),
        );
    }
}

fn show_x3i_warning(args: &Args, warning: &str) -> bool {
    !warning.contains("-bin ")
        && (args.verbosity != Some(Verbosity::Error) || !warning.starts_with("X3I frame "))
}

fn for_each_input(files: &[PathBuf], convert: impl Fn(&Path) + Sync) {
    use rayon::prelude::*;
    let (serial, parallel): (Vec<_>, Vec<_>) =
        files.iter().partition(|path| is_x3i(path).unwrap_or(false));
    parallel.par_iter().for_each(|path| convert(path));
    // X3I merges must not overlap with another conversion's image buffers.
    for path in serial {
        convert(path);
    }
}

fn main() -> ExitCode {
    let argv: Vec<String> = std::env::args().collect();
    let args = parse_args(&argv);
    if let Err(msg) = validate_args(&args) {
        eprintln!("{msg}");
        usage(argv.first().map(String::as_str).unwrap_or("x3f_extract"));
    }

    if let Some(v) = args.verbosity {
        set_verbosity(v);
    }
    set_offset_legacy(args.legacy_offset);
    if let Some(n) = args.matrix_max {
        set_max_printed_matrix_elements(n);
    }
    // -ocl is a no-op, preserved only so legacy scripts don't error. There is
    // no OpenCL backend (native acceleration uses wgpu in x3f-sys).
    let _ = args.use_opencl;

    if let Some(dir) = args.outdir.as_deref() {
        if !dir.is_dir() {
            eprintln!("Could not find outdir {}", dir.display());
            usage(argv.first().map(String::as_str).unwrap_or("x3f_extract"));
        }
    }

    if args.files.is_empty() {
        usage(argv.first().map(String::as_str).unwrap_or("x3f_extract"));
    }

    eprintln!(
        "X3F TOOLS VERSION = x3f-rust-{}\n",
        env!("CARGO_PKG_VERSION")
    );

    // X3F files share the Rayon pool. X3I files run serially after that group.
    // Each conversion stays on one calling thread for its thread-local state.
    use std::sync::atomic::{AtomicUsize, Ordering};
    let errors = AtomicUsize::new(0);
    for_each_input(&args.files, |f| {
        if let Err(e) = convert_one(f, &args) {
            eprintln!("{e}");
            errors.fetch_add(1, Ordering::Relaxed);
        }
    });
    let errors = errors.load(Ordering::Relaxed);

    if errors > 0 {
        eprintln!("{errors}/{} files had errors", args.files.len());
        ExitCode::from(1)
    } else {
        ExitCode::SUCCESS
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(args: &[&str]) -> Args {
        let mut argv = vec!["x3f_extract".to_string()];
        argv.extend(args.iter().map(|s| s.to_string()));
        parse_args(&argv)
    }

    #[test]
    fn x3i_flags_validate_input_magic_and_output_combinations() {
        let path = std::env::temp_dir().join(format!("x3i-cli-flags-{}.bin", std::process::id()));
        std::fs::write(&path, b"FOVi").unwrap();
        let name = path.to_str().unwrap();
        for format in [
            "-meta",
            "-jpg",
            "-raw",
            "-tiff",
            "-dng",
            "-ppm-ascii",
            "-ppm",
            "-histogram",
            "-loghist",
        ] {
            let args = parse(&[format, "-x3i-frame", "2", name]);
            assert_eq!(args.x3i_frame, Some(2));
            assert!(validate_args(&args).is_ok(), "{format}");
            for flags in [
                vec![format, "-x3i-extract", name],
                vec!["-x3i-extract", format, name],
            ] {
                assert!(validate_args(&parse(&flags)).is_err(), "{flags:?}");
            }
        }
        for flags in [
            vec!["-x3i-extract", name],
            vec!["-x3i-frame", "0", name],
            vec!["-meta", name],
            vec!["-tiff", name],
            vec!["-ppm", name],
            vec!["-ppm-ascii", name],
        ] {
            assert!(validate_args(&parse(&flags)).is_ok());
        }
        assert!(validate_args(&parse(&["-x3i-frame", "0", "-x3i-extract", name])).is_err());
        assert!(validate_args(&parse(&["-tiff", "-spp-pipeline", name]))
            .unwrap_err()
            .contains("X3I"));
        assert!(
            validate_args(&parse(&["-tiff", "-spp-pipeline", "-x3i-frame", "0", name])).is_err()
        );
        std::fs::write(&path, b"FOVb").unwrap();
        for flags in [vec!["-x3i-frame", "0", name], vec!["-x3i-extract", name]] {
            assert!(validate_args(&parse(&flags)).unwrap_err().contains("X3I"));
        }
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn x3i_ghost_map_requires_whole_container_merge_output() {
        let path = std::env::temp_dir().join(format!("x3i-ghost-flags-{}.bin", std::process::id()));
        std::fs::write(&path, b"FOVi").unwrap();
        let name = path.to_str().unwrap();
        for format in ["-dng", "-tiff", "-ppm", "-ppm-ascii"] {
            let args = parse(&[format, "-x3i-ghost-map", name]);
            assert!(args.x3i_ghost_map);
            assert!(validate_args(&args).is_ok());
        }
        assert!(validate_args(&parse(&["-x3i-ghost-map", name])).is_ok());
        assert!(validate_args(&parse(&["-dng", "-x3i-report", "-x3i-ghost-map", name])).is_ok());
        for format in ["-meta", "-jpg", "-raw", "-histogram"] {
            assert!(
                validate_args(&parse(&[format, "-x3i-ghost-map", name])).is_err(),
                "{format}"
            );
        }
        for flags in [
            vec!["-x3i-frame", "0", "-x3i-ghost-map", name],
            vec!["-x3i-extract", "-x3i-ghost-map", name],
        ] {
            assert!(validate_args(&parse(&flags)).is_err());
        }
        std::fs::write(&path, b"FOVb").unwrap();
        assert!(validate_args(&parse(&["-dng", "-x3i-ghost-map", name])).is_err());
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn x3i_report_requires_a_whole_container_dng() {
        let path =
            std::env::temp_dir().join(format!("x3i-report-flags-{}.bin", std::process::id()));
        std::fs::write(&path, b"FOVi").unwrap();
        let name = path.to_str().unwrap();
        for flags in [vec!["-x3i-report", name], vec!["-dng", "-x3i-report", name]] {
            let args = parse(&flags);
            assert!(args.x3i_report);
            assert!(validate_args(&args).is_ok());
        }
        for flag in [
            "-meta",
            "-jpg",
            "-raw",
            "-tiff",
            "-ppm",
            "-ppm-ascii",
            "-histogram",
            "-x3i-extract",
        ] {
            assert!(
                validate_args(&parse(&[flag, "-x3i-report", name])).is_err(),
                "{flag}"
            );
        }
        assert!(validate_args(&parse(&["-x3i-frame", "0", "-x3i-report", name])).is_err());
        std::fs::write(&path, b"FOVb").unwrap();
        assert!(validate_args(&parse(&["-x3i-report", name]))
            .unwrap_err()
            .contains("X3I"));
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn x3i_highlight_flags_warn_without_rejecting_conversion() {
        let path =
            std::env::temp_dir().join(format!("x3i-highlight-flags-{}.bin", std::process::id()));
        std::fs::write(&path, b"FOVi").unwrap();
        let name = path.to_str().unwrap();
        for flags in [
            vec!["-dng", "-dng-highlight-recovery", name],
            vec!["-dng", "-dng-highlight-mapping", "linear", name],
            vec!["-dng", "-dng-highlight-mapping", "shoulder", name],
            vec![
                "-dng",
                "-dng-highlight-recovery",
                "-dng-highlight-mapping",
                "shoulder",
                name,
            ],
        ] {
            let args = parse(&flags);
            assert!(validate_args(&args).is_ok(), "{flags:?}");
            let mut warnings = Vec::new();
            add_x3i_warnings(&args, &mut warnings);
            assert_eq!(warnings.len(), 1);
            assert!(warnings[0].contains("have no effect on X3I merges"));
        }
        let args = parse(&["-dng", name]);
        assert!(!args.dng_highlight_mapping_explicit);
        let mut warnings = Vec::new();
        add_x3i_warnings(&args, &mut warnings);
        assert!(warnings.is_empty());
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn quiet_hides_only_x3i_frame_rows() {
        for (flags, show_frames) in [
            (vec!["in.X3I"], true),
            (vec!["-v", "in.X3I"], true),
            (vec!["-q", "in.X3I"], false),
            (vec!["-q", "-v", "in.X3I"], true),
            (vec!["-v", "-q", "in.X3I"], false),
        ] {
            let args = parse(&flags);
            assert_eq!(
                show_x3i_warning(&args, "X3I frame 0: shutter_us=7813 scale=1"),
                show_frames
            );
            assert!(!show_x3i_warning(
                &args,
                "X3I frame 1: aligned-ratio-bin lower=0.05"
            ));
            assert!(!show_x3i_warning(
                &args,
                "X3I frame 1: ghost-bin lower=0.05"
            ));
            for line in [
                "X3I merge: frames=7 s=1",
                "X3I fallback samples: shortest exposure 1",
                "X3I saturation warning: frame=6 plane=B falls back to 1.0",
                "X3I denoise is off; set X3F_X3I_DENOISE_SCALE=0.3 to enable the experimental post-merge filter",
                "-dng-highlight-mapping has no effect on X3I merges",
            ] {
                assert!(show_x3i_warning(&args, line));
            }
        }
    }

    #[test]
    fn batch_runs_x3i_inputs_serially_after_x3f_inputs() {
        use std::sync::{
            atomic::{AtomicUsize, Ordering},
            Mutex,
        };
        let dir = std::env::temp_dir().join(format!("x3i-batch-{}", std::process::id()));
        std::fs::create_dir(&dir).unwrap();
        let files: Vec<_> = ["first.X3F", "first.X3I", "second.bin", "second.raw"]
            .iter()
            .map(|name| dir.join(name))
            .collect();
        for (i, path) in files.iter().enumerate() {
            std::fs::write(path, if i % 2 == 0 { b"FOVi" } else { b"FOVb" }).unwrap();
        }
        let active = AtomicUsize::new(0);
        let visited = Mutex::new(Vec::new());
        rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap()
            .install(|| {
                for_each_input(&files, |path| {
                    let x3i = is_x3i(path).unwrap();
                    if x3i {
                        assert_eq!(active.fetch_add(1, Ordering::SeqCst), 0);
                    } else {
                        assert_eq!(active.load(Ordering::SeqCst), 0);
                    }
                    visited.lock().unwrap().push((x3i, path.to_path_buf()));
                    std::thread::sleep(std::time::Duration::from_millis(10));
                    if x3i {
                        active.fetch_sub(1, Ordering::SeqCst);
                    }
                });
            });
        let visited = visited.into_inner().unwrap();
        assert_eq!(visited.len(), 4);
        assert!(visited[..2].iter().all(|(x3i, _)| !x3i));
        assert_eq!(
            visited[2..],
            [(true, files[0].clone()), (true, files[2].clone())]
        );
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn defaults_match_legacy() {
        let a = parse(&["in.X3F"]);
        assert_eq!(a.file_type, FileType::Dng);
        assert_eq!(a.color_encoding, ColorEncoding::Srgb);
        assert!(a.crop && a.fix_bad);
        assert_eq!(a.denoise_intensity, 10);
        assert!(!a.compress);
        assert!(!a.dng_highlight_recovery);
        assert_eq!(a.dng_highlight_mapping, DngHighlightMapping::Linear);
        assert_eq!(a.files, vec![PathBuf::from("in.X3F")]);
    }

    #[test]
    fn rgb_chroma_lut_override_is_explicit_and_last_wins() {
        assert_eq!(parse(&["in.X3F"]).rgb_chroma_lut, None);
        for (flags, expected) in [
            (vec!["-chroma-lut", "in.X3F"], true),
            (vec!["-no-chroma-lut", "in.X3F"], false),
            (vec!["-chroma-lut", "-no-chroma-lut", "in.X3F"], false),
            (vec!["-no-chroma-lut", "-chroma-lut", "in.X3F"], true),
        ] {
            assert_eq!(parse(&flags).rgb_chroma_lut, Some(expected));
        }
    }

    #[test]
    fn dng_recovery_defaults_to_linear_headroom() {
        let args = parse(&["-dng-highlight-recovery", "in.X3F"]);
        assert!(args.dng_highlight_recovery);
        assert_eq!(args.dng_highlight_mapping, DngHighlightMapping::Linear);
    }

    #[test]
    fn dng_shoulder_mapping_is_independent_of_recovery_flag_order() {
        for flags in [
            vec![
                "-dng-highlight-recovery",
                "-dng-highlight-mapping",
                "shoulder",
                "in.X3F",
            ],
            vec![
                "-dng-highlight-mapping",
                "shoulder",
                "-dng-highlight-recovery",
                "in.X3F",
            ],
        ] {
            let args = parse(&flags);
            assert!(args.dng_highlight_recovery);
            assert_eq!(args.dng_highlight_mapping, DngHighlightMapping::Shoulder);
        }
    }

    #[test]
    fn dng_mapping_alone_does_not_enable_recovery() {
        let args = parse(&["-dng-highlight-mapping", "shoulder", "in.X3F"]);
        assert!(!args.dng_highlight_recovery);
        assert_eq!(args.dng_highlight_mapping, DngHighlightMapping::Shoulder);
    }

    #[test]
    fn dng_mapping_last_selection_wins() {
        let args = parse(&[
            "-dng-highlight-mapping",
            "shoulder",
            "-dng-highlight-mapping",
            "linear",
            "in.X3F",
        ]);
        assert_eq!(args.dng_highlight_mapping, DngHighlightMapping::Linear);
    }

    #[test]
    fn format_flags_are_mutually_exclusive_last_wins() {
        // The legacy Z-macro behaviour: each format flag resets all the
        // extract_* booleans, so flag ordering doesn't leak through.
        let a = parse(&["-tiff", "-jpg", "-meta", "in.X3F"]);
        assert_eq!(a.file_type, FileType::Meta);
        assert!(!a.extract_jpg);
        assert!(!a.extract_raw);
        assert!(!a.extract_unconverted_raw);
    }

    #[test]
    fn color_keyword_table() {
        let cases = [
            ("none", ColorEncoding::None),
            ("sRGB", ColorEncoding::Srgb),
            ("AdobeRGB", ColorEncoding::AdobeRgb),
            ("ProPhotoRGB", ColorEncoding::ProPhotoRgb),
        ];
        for (kw, expected) in cases {
            let a = parse(&["-color", kw, "in.X3F"]);
            assert_eq!(a.color_encoding, expected, "color keyword {kw}");
        }
    }

    #[test]
    fn unprocessed_and_qtop_overwrite_color_encoding() {
        let a = parse(&["-color", "sRGB", "-unprocessed", "in.X3F"]);
        assert_eq!(a.color_encoding, ColorEncoding::Unprocessed);
        let a = parse(&["-color", "sRGB", "-qtop", "in.X3F"]);
        assert_eq!(a.color_encoding, ColorEncoding::Qtop);
    }

    #[test]
    fn color_mode_is_separate_from_output_encoding_and_opt_in() {
        assert_eq!(parse(&["in.X3F"]).color_mode, None);
        let args = parse(&[
            "-tiff",
            "-color",
            "AdobeRGB",
            "-color-mode",
            "Vivid",
            "in.X3F",
        ]);
        assert_eq!(args.color_mode, Some(ColorMode::Vivid));
        assert_eq!(args.color_encoding, ColorEncoding::AdobeRgb);
        assert!(validate_args(&args).is_ok());
        let args = parse(&["-dng", "-color", "none", "-color-mode", "as-shot", "in.X3F"]);
        assert_eq!(args.color_mode, Some(ColorMode::AsShot));
        assert!(validate_args(&args).is_ok());
        for name in ["FOV Classic Blue", "FCBlue"] {
            let args = parse(&["-dng", "-color-mode", name, "in.X3F"]);
            assert_eq!(args.color_mode, Some(ColorMode::FCBlue));
            assert!(validate_args(&args).is_ok());
        }
    }

    #[test]
    fn color_mode_rejects_outputs_outside_tiff_and_dng() {
        for flags in [
            vec!["-jpg"],
            vec!["-raw"],
            vec!["-meta"],
            vec!["-ppm"],
            vec!["-histogram"],
            vec!["-tiff", "-cineon"],
            vec!["-tiff", "-unprocessed"],
            vec!["-tiff", "-qtop"],
            vec!["-tiff", "-color", "none"],
        ] {
            let mut command = flags;
            command.extend(["-color-mode", "Standard", "in.X3F"]);
            assert!(validate_args(&parse(&command)).is_err(), "{command:?}");
        }
    }

    #[test]
    fn cineon_alone_defaults_to_prophotorgb() {
        let a = parse(&["-tiff", "-cineon", "in.X3F"]);
        assert!(a.cineon);
        assert_eq!(a.color_encoding, ColorEncoding::ProPhotoRgb);
        assert!(validate_args(&a).is_ok());
    }

    #[test]
    fn cineon_respects_explicit_color() {
        let a = parse(&["-tiff", "-color", "AdobeRGB", "-cineon", "in.X3F"]);
        assert!(a.cineon);
        assert_eq!(a.color_encoding, ColorEncoding::AdobeRgb);
        let a = parse(&["-tiff", "-cineon", "-color", "sRGB", "in.X3F"]);
        assert!(a.cineon);
        assert_eq!(a.color_encoding, ColorEncoding::Srgb);
    }

    #[test]
    fn cineon_requires_tiff() {
        let a = parse(&["-dng", "-cineon", "in.X3F"]);
        assert!(a.cineon);
        let err = validate_args(&a).unwrap_err();
        assert!(err.contains("-tiff"), "got: {err}");
    }

    #[test]
    fn cineon_rejects_unprocessed_and_qtop() {
        let a = parse(&["-tiff", "-cineon", "-unprocessed", "in.X3F"]);
        assert!(validate_args(&a).is_err());
        let a = parse(&["-tiff", "-cineon", "-qtop", "in.X3F"]);
        assert!(validate_args(&a).is_err());
    }

    #[test]
    fn cineon_with_color_none_is_allowed() {
        // -cineon -color none means "no matrix, log curve baked into raw
        // BMT samples, no ICC" — a valid camera-native log dump.
        let a = parse(&["-tiff", "-cineon", "-color", "none", "in.X3F"]);
        assert_eq!(a.color_encoding, ColorEncoding::None);
        assert!(validate_args(&a).is_ok());
    }

    #[test]
    fn modifier_flags_capture_typed_values() {
        let a = parse(&[
            "-no-crop",
            "-no-fix-bad",
            "-no-denoise",
            "-no-sgain",
            "-compress",
            "-wb",
            "Daylight",
            "-offset",
            "1024",
            "-matrixmax",
            "200",
            "-v",
            "in.X3F",
        ]);
        assert!(!a.crop && !a.fix_bad);
        assert_eq!(a.denoise_intensity, 0);
        assert_eq!(a.apply_sgain, Some(false));
        assert!(a.compress);
        assert_eq!(a.wb.as_deref(), Some("Daylight"));
        assert_eq!(a.legacy_offset, Some(1024));
        assert_eq!(a.matrix_max, Some(200));
        assert_eq!(a.verbosity, Some(Verbosity::Debug));
    }

    #[test]
    fn denoise_intensity_parses_explicit_value() {
        assert_eq!(parse(&["-denoise", "5", "in.X3F"]).denoise_intensity, 5);
        assert_eq!(parse(&["-denoise", "0", "in.X3F"]).denoise_intensity, 0);
        assert_eq!(parse(&["-denoise", "10", "in.X3F"]).denoise_intensity, 10);
        // `-no-denoise` is the alias for the off end of the scale.
        assert_eq!(parse(&["-no-denoise", "in.X3F"]).denoise_intensity, 0);
    }

    #[test]
    fn denoise_backend_is_per_conversion() {
        use x3f_core::DenoiseBackend;
        assert_eq!(parse(&["in.X3F"]).denoise_backend, DenoiseBackend::Auto);
        assert_eq!(
            parse(&["-denoise-backend", "cpu", "in.X3F"]).denoise_backend,
            DenoiseBackend::Cpu
        );
        assert_eq!(
            parse(&["-denoise-backend", "gpu", "in.X3F"]).denoise_backend,
            DenoiseBackend::Gpu
        );
        assert!("opencl".parse::<DenoiseBackend>().is_err());
    }

    #[test]
    fn spp_pipeline_is_explicit_and_rejects_unsupported_modes() {
        assert!(!parse(&["in.X3F"]).spp_denoise);
        for format in ["-tiff", "-ppm-ascii", "-ppm", "-histogram"] {
            let args = parse(&[format, "-color", "sRGB", "-spp-pipeline", "in.X3F"]);
            assert!(args.spp_denoise);
            assert!(validate_args(&args).is_ok(), "{format}");
        }
        for flags in [
            vec!["-dng"],
            vec!["-jpg"],
            vec!["-meta"],
            vec!["-tiff", "-cineon"],
            vec!["-tiff", "-color", "none"],
            vec!["-tiff", "-no-denoise"],
            vec!["-tiff", "-denoise", "5"],
            vec!["-tiff", "-denoise-backend", "gpu"],
        ] {
            let mut command = flags;
            command.extend(["-spp-pipeline", "in.X3F"]);
            assert!(validate_args(&parse(&command)).is_err(), "{command:?}");
        }
        assert!(validate_args(&parse(&[
            "-tiff",
            "-spp-pipeline",
            "-denoise-backend",
            "cpu",
            "in.X3F"
        ]))
        .is_ok());
    }

    #[test]
    fn spp_pipeline_requires_default_denoise_options() {
        for flags in [
            vec!["-denoiser", "bm3d"],
            vec!["-denoise-luma"],
            vec!["-no-denoise-chroma"],
            vec!["-denoise-strength", "1.5"],
            vec!["-nlm-luma-patch", "7"],
            vec!["-nlm-chroma-patch", "7"],
            vec!["-nlm-luma-search", "21"],
            vec!["-nlm-chroma-search", "21"],
        ] {
            for spp_first in [false, true] {
                let mut command = vec!["-tiff"];
                if spp_first {
                    command.push("-spp-pipeline");
                }
                command.extend(&flags);
                if !spp_first {
                    command.push("-spp-pipeline");
                }
                command.push("in.X3F");
                let error = validate_args(&parse(&command)).unwrap_err();
                assert!(
                    error.contains("default denoise options"),
                    "{command:?}: {error}"
                );
            }
        }
        let args = parse(&[
            "-tiff",
            "-spp-pipeline",
            "-denoiser",
            "nlm",
            "-no-denoise-luma",
            "-denoise-chroma",
            "-denoise-strength",
            "1",
            "-nlm-luma-patch",
            "3",
            "-nlm-chroma-patch",
            "3",
            "-nlm-luma-search",
            "11",
            "-nlm-chroma-search",
            "11",
            "in.X3F",
        ]);
        assert!(validate_args(&args).is_ok());
    }

    #[test]
    fn recovery_pipelines_are_separate() {
        for flags in [
            vec!["-chroma-lut", "-spp-pipeline"],
            vec!["-spp-pipeline", "-chroma-lut"],
        ] {
            let mut command = vec!["-tiff"];
            command.extend(flags);
            command.push("in.X3F");
            let error = validate_args(&parse(&command)).unwrap_err();
            assert!(error.contains("select one pipeline"), "{error}");
        }
        for flags in [
            vec!["-chroma-lut", "-no-denoise"],
            vec!["-chroma-lut", "-denoise", "5"],
            vec!["-chroma-lut", "-denoise-backend", "gpu"],
            vec!["-spp-pipeline", "-no-chroma-lut"],
            vec!["-spp-pipeline", "-chroma-lut", "-no-chroma-lut"],
        ] {
            let mut command = vec!["-tiff"];
            command.extend(flags);
            command.push("in.X3F");
            assert!(validate_args(&parse(&command)).is_ok(), "{command:?}");
        }
    }

    #[test]
    fn algorithm_and_advanced_denoise_flags_are_independent() {
        let options = parse(&[
            "-denoiser",
            "bm3d",
            "-denoise-luma",
            "-no-denoise-chroma",
            "-denoise-strength",
            "1.5",
            "-nlm-luma-patch",
            "7",
            "-nlm-chroma-search",
            "21",
            "in.X3F",
        ])
        .denoise_options;
        assert_eq!(options.algorithm, x3f_core::DenoiseAlgorithm::Bm3d);
        assert!(options.luminance);
        assert!(!options.chroma);
        assert_eq!(options.strength, 1.5);
        assert_eq!(options.nlm.luma_patch, 7);
        assert_eq!(options.nlm.chroma_search, 21);
        assert_eq!(
            parse(&["in.X3F"]).denoise_options,
            x3f_core::DenoiseOptions::default()
        );
    }

    #[test]
    fn first_non_flag_starts_file_list() {
        // Critical: anything after the first positional argument is treated
        // as a filename, even if it looks like a flag. Mirrors getopt-less C.
        let a = parse(&["-tiff", "in.X3F", "-jpg", "another.X3F"]);
        assert_eq!(
            a.files,
            vec![
                PathBuf::from("in.X3F"),
                PathBuf::from("-jpg"),
                PathBuf::from("another.X3F")
            ],
        );
        assert_eq!(a.file_type, FileType::Tiff);
    }

    #[test]
    fn make_paths_no_outdir() {
        let (tmp, out) = make_paths(Path::new("dir/in.X3F"), None, ".dng");
        assert_eq!(out, PathBuf::from("dir/in.X3F.dng"));
        assert_eq!(tmp, PathBuf::from("dir/in.X3F.dng.tmp"));
    }

    #[test]
    fn make_paths_with_outdir_strips_input_dirname() {
        let (tmp, out) = make_paths(Path::new("dir/in.X3F"), Some(Path::new("/out")), ".tif");
        assert_eq!(out, PathBuf::from("/out/in.X3F.tif"));
        assert_eq!(tmp, PathBuf::from("/out/in.X3F.tif.tmp"));
    }

    #[test]
    fn make_paths_with_outdir_trailing_slash_does_not_double() {
        let (_, out) = make_paths(Path::new("in.X3F"), Some(Path::new("/out/")), ".meta");
        assert_eq!(out, PathBuf::from("/out/in.X3F.meta"));
    }

    #[test]
    fn file_type_extensions_match_legacy() {
        // Extensions are appended to the *full* input filename; `input.X3F`
        // becomes `input.X3F.meta`. Existing test corpora rely on this.
        assert_eq!(FileType::Meta.extension(), ".meta");
        assert_eq!(FileType::Jpeg.extension(), ".jpg");
        assert_eq!(FileType::Raw.extension(), ".raw");
        assert_eq!(FileType::Tiff.extension(), ".tif");
        assert_eq!(FileType::Dng.extension(), ".dng");
        assert_eq!(FileType::PpmP3.extension(), ".ppm");
        assert_eq!(FileType::PpmP6.extension(), ".ppm");
        assert_eq!(FileType::Histogram.extension(), ".csv");
    }
}
