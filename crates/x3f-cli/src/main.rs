//! `x3f_extract` — command-line frontend over `x3f-core`.
//!
//! M0 deliberately preserves the legacy single-dash flag syntax
//! (`-tiff`, `-no-denoise`, `-color sRGB`, …) so existing test corpora and
//! shell scripts continue to work. A modern `--long-flag` interface and
//! subcommand layout will be introduced in a later milestone after the test
//! harness stabilises in M2.

use std::os::unix::ffi::OsStrExt;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use x3f_core::{set_max_printed_matrix_elements, set_offset_legacy, set_verbosity};
use x3f_core::{ColorEncoding, ProcessOptions, Reader, Verbosity};

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
    extract_jpg: bool,
    extract_raw: bool,
    extract_unconverted_raw: bool,
    crop: bool,
    fix_bad: bool,
    /// Denoise strength, 0..=10. 0 disables denoise (legacy `-no-denoise`),
    /// 10 is full strength (legacy default).
    denoise_intensity: u8,
    apply_sgain: Option<bool>,
    file_type: FileType,
    color_encoding: ColorEncoding,
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
    cineon: bool,
    /// Set when the user explicitly passed `-color <space>`. Used to
    /// resolve `-cineon` alone → ProPhotoRGB without overriding an
    /// explicit user choice.
    color_explicit: bool,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            extract_jpg: false,
            extract_raw: true,
            extract_unconverted_raw: false,
            crop: true,
            fix_bad: true,
            denoise_intensity: 10,
            apply_sgain: None,
            file_type: FileType::Dng,
            color_encoding: ColorEncoding::Srgb,
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
         \x20  -unprocessed    Dump RAW without any preprocessing\n\
         \x20  -qtop           Dump Quattro top layer without preprocessing\n\
         \x20  -no-crop        Do not crop to active area\n\
         \x20  -no-denoise     Disable NLM denoise (same as -denoise 0)\n\
         \x20  -denoise <0-10> NLM denoise intensity. 0 = off,\n\
         \x20                  10 = full strength (default). Intermediate\n\
         \x20                  values linearly scale the NLM sigma.\n\
         \x20  -no-sgain       Do not apply spatial gain (color compensation)\n\
         \x20  -no-fix-bad     Do not fix bad pixels\n\
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
         \x20                  Apply Foveon highlight recovery (per-channel\n\
         \x20                  chroma-LUT reconstruction, L*p fallback,\n\
         \x20                  matrix-pathology gate) when writing DNG.\n\
         \x20                  Recovered overshoot is folded back under\n\
         \x20                  WhiteLevel with a soft highlight shoulder baked\n\
         \x20                  into the raster (knee tunable via\n\
         \x20                  X3F_DNG_SHOULDER_KNEE, default 0.85), so the\n\
         \x20                  result renders identically in every DNG reader\n\
         \x20                  (Adobe Camera Raw, Lightroom, LibRaw/RawTherapee,\n\
         \x20                  Capture One, Apple RAW Engine).\n\
         \x20                  Default: off (matches the pre-Rust C writer).\n\
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
            args.extract_jpg = false;
            args.extract_raw = false;
            args.extract_unconverted_raw = false;
        };
        match a {
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
            "-no-denoise" => args.denoise_intensity = 0,
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

/// Reject argument combinations that the pipeline cannot honour. Returns
/// `Err` with a human-readable message; the caller turns that into a
/// usage()-style exit. Kept as a pure function so the unit tests can
/// exercise it without `process::exit`.
fn validate_args(args: &Args) -> Result<(), String> {
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
        let dir_bytes = dir.as_os_str().as_bytes();
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
    let mut reader =
        Reader::open(infile).map_err(|e| format!("could not open `{}`: {e}", infile.display()))?;

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

    let (tmpfile, outfile) = make_paths(infile, args.outdir.as_deref(), args.file_type.extension());

    // Match legacy behavior: unlink the temp first so we never write through
    // a stale file from a previous failed run.
    let _ = std::fs::remove_file(&tmpfile);

    let opts = ProcessOptions {
        color_encoding: args.color_encoding,
        crop: args.crop,
        fix_bad: args.fix_bad,
        denoise_intensity: args.denoise_intensity,
        apply_sgain: args.apply_sgain,
        wb: args.wb.clone(),
        compress: args.compress,
        opcodes_dir: args.opcodes_dir.clone(),
        dng_highlight_recovery: args.dng_highlight_recovery,
        cineon: args.cineon,
    };

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
    std::fs::rename(&tmpfile, &outfile).map_err(|e| {
        format!(
            "could not rename `{}` to `{}`: {e}",
            tmpfile.display(),
            outfile.display()
        )
    })?;
    Ok(())
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
    // no OpenCL backend (the denoise pass is the pure-Rust NLM in x3f-sys).
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

    // Batch parallelism (M7b). Each file is processed end-to-end on a
    // single rayon worker, so the per-image thread-local
    // DNG_HIGHLIGHT_SCALE in x3f-sys::process stays consistent. The
    // inner per-image rayon (M7a) recursively shares the global pool
    // and rejoins before any thread-local write.
    use rayon::prelude::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    let errors = AtomicUsize::new(0);
    args.files.par_iter().for_each(|f| {
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
    fn defaults_match_legacy() {
        let a = parse(&["in.X3F"]);
        assert_eq!(a.file_type, FileType::Dng);
        assert_eq!(a.color_encoding, ColorEncoding::Srgb);
        assert!(a.crop && a.fix_bad);
        assert_eq!(a.denoise_intensity, 10);
        assert!(!a.compress);
        assert_eq!(a.files, vec![PathBuf::from("in.X3F")]);
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
