//! `OpcodeList3` (DNG flat-fielding) blob lookup.
//!
//! Sigma's lens flat-field corrections are pre-rendered as binary DNG
//! `OpcodeList3` byte streams and shipped alongside x3fuse. We don't
//! bundle the blobs into this crate (license question hasn't been
//! resolved); instead the user passes a directory via
//! `ProcessOptions::opcodes_dir` and we look up the matching file by
//! `(model, aperture, lens_id)` and embed its bytes verbatim into the
//! DNG raw IFD's `OpcodeList3` tag.
//!
//! Filename convention (matches `_local/x3fuse/X3Fuse/opcodes/`):
//!
//! ```text
//! <MODEL>[_<LENSID>]_FF_DNG_Opcodelist3_<APERTURE>
//! ```
//!
//! - `MODEL` ∈ `{DP1M, DP2M, DP3M, SD1M}` (no Quattro opcodes shipped).
//! - `LENSID` is omitted for fixed-lens DP cameras; for SD1M it's the
//!   `Unknown_(<numeric>)_30mm` form Sigma uses internally.
//! - `APERTURE` is the f-stop with one decimal (`"2.8"`, `"5.6"`,
//!   `"14.0"`).
//!
//! On a miss we log a warning and emit no `OpcodeList3` tag — the DNG
//! still validates, just without the flat-field correction. Failing the
//! conversion would be too aggressive (the user might have legitimate
//! `(model, aperture)` combinations the directory doesn't cover).

use std::path::Path;

use super::exif::CaptureMetadata;

/// Look up and read the `OpcodeList3` bytes for `meta`'s capture
/// settings. Returns `None` when no directory was supplied, when we
/// can't derive a model/aperture, or when the matching file isn't on
/// disk. Emits a warning on the latter case so the user knows their
/// directory was queried but nothing matched.
pub(crate) fn load_for(meta: &CaptureMetadata, dir: &Path) -> Option<Vec<u8>> {
    let model_id = derive_model_id(meta)?;
    let aperture = format_aperture(meta)?;
    let lens_id = if model_id == "SD1M" {
        Some(format_sd1m_lens_id(meta))
    } else {
        None
    };

    let filename = match lens_id {
        Some(lid) => format!("{model_id}_{lid}_FF_DNG_Opcodelist3_{aperture}"),
        None => format!("{model_id}_FF_DNG_Opcodelist3_{aperture}"),
    };
    let path = dir.join(&filename);
    match std::fs::read(&path) {
        Ok(bytes) => Some(bytes),
        Err(_) => {
            eprintln!(
                "x3f: DNG opcode file not found in `{}` (looked for `{}`); flat-field correction not embedded",
                dir.display(),
                filename
            );
            None
        }
    }
}

/// Map a `Make`/`Model` pair to the short identifier the opcode files
/// use (`"DP1M"` / `"DP2M"` / `"DP3M"` / `"SD1M"`). Returns `None` for
/// camera bodies the x3fuse opcode set doesn't cover (Quattro, SD9/SD10/
/// SD14, etc.).
fn derive_model_id(meta: &CaptureMetadata) -> Option<&'static str> {
    let model = meta.model.as_deref()?;
    let lc = model.to_ascii_lowercase();
    if lc.contains("dp1") && lc.contains("merrill") {
        Some("DP1M")
    } else if lc.contains("dp2") && lc.contains("merrill") {
        Some("DP2M")
    } else if lc.contains("dp3") && lc.contains("merrill") {
        Some("DP3M")
    } else if lc.contains("sd1") && lc.contains("merrill") {
        Some("SD1M")
    } else {
        None
    }
}

/// Format the f-number as `"X.Y"` (one decimal). Returns `None` if we
/// don't have an FNumber rational to draw from.
fn format_aperture(meta: &CaptureMetadata) -> Option<String> {
    let (n, d) = meta.f_number?;
    if d == 0 {
        return None;
    }
    let f = n as f64 / d as f64;
    if !f.is_finite() || f <= 0.0 {
        return None;
    }
    Some(format!("{:.1}", f))
}

/// SD1M opcodes use Sigma's internal lens-id format
/// `"Unknown_(<numeric>)_30mm"`. The numeric is what the camera writes
/// into PROP[LENSMODEL] (e.g. `"32776"` for the 30mm prime). The
/// trailing `30mm` is fixed in the x3fuse opcode set — Sigma only
/// published SD1 opcodes for the 30mm prime.
///
/// `meta.lens_model` carries the raw PROP[LENSMODEL] string when
/// available; we fall back to `Unknown_(32776)_30mm` when it isn't,
/// which is what x3fuse does too. A wrong-lens lookup just misses,
/// logs a warning, and emits the DNG without OpcodeList3.
fn format_sd1m_lens_id(meta: &CaptureMetadata) -> String {
    let code = meta
        .lens_model
        .as_deref()
        .and_then(|s| s.trim().parse::<u32>().ok())
        .unwrap_or(32776);
    format!("Unknown_({})_30mm", code)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta_with(model: Option<&str>, fnum: Option<(u32, u32)>) -> CaptureMetadata {
        CaptureMetadata {
            model: model.map(str::to_string),
            f_number: fnum,
            ..Default::default()
        }
    }

    #[test]
    fn model_id_recognises_merrill_bodies() {
        assert_eq!(
            derive_model_id(&meta_with(Some("SIGMA DP1 Merrill"), None)),
            Some("DP1M")
        );
        assert_eq!(
            derive_model_id(&meta_with(Some("SIGMA DP2 Merrill"), None)),
            Some("DP2M")
        );
        assert_eq!(
            derive_model_id(&meta_with(Some("SIGMA DP3 Merrill"), None)),
            Some("DP3M")
        );
        assert_eq!(
            derive_model_id(&meta_with(Some("SIGMA SD1 Merrill"), None)),
            Some("SD1M")
        );
    }

    #[test]
    fn model_id_rejects_quattro_and_unknown() {
        assert_eq!(
            derive_model_id(&meta_with(Some("SIGMA dp0 Quattro"), None)),
            None
        );
        assert_eq!(derive_model_id(&meta_with(Some("SIGMA SD9"), None)), None);
        assert_eq!(derive_model_id(&meta_with(None, None)), None);
    }

    #[test]
    fn aperture_format_matches_x3fuse_filenames() {
        let m = meta_with(None, Some((28, 10)));
        assert_eq!(format_aperture(&m).as_deref(), Some("2.8"));
        let m = meta_with(None, Some((4, 1)));
        assert_eq!(format_aperture(&m).as_deref(), Some("4.0"));
        let m = meta_with(None, Some((140, 10)));
        assert_eq!(format_aperture(&m).as_deref(), Some("14.0"));
    }

    #[test]
    fn sd1m_lens_id_reads_lens_model_or_falls_back() {
        let mut m = meta_with(Some("SIGMA SD1 Merrill"), Some((28, 10)));
        m.lens_model = Some("32776".into());
        assert_eq!(format_sd1m_lens_id(&m), "Unknown_(32776)_30mm");
        m.lens_model = None;
        assert_eq!(format_sd1m_lens_id(&m), "Unknown_(32776)_30mm");
    }
}
