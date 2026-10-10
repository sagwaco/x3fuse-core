//! M6d — native Rust port of `src/x3f_spatial_gain.c`.
//!
//! Per-pixel "spatial gain" (lens shading) correction tables. The Sigma
//! cameras encode a 4D family of correction tables in CAMF: per-aperture,
//! per-focus-distance entries. We pick the four nearest neighbours in
//! (1/aperture, lens-position) space, build bilinear interpolation
//! weights, and either stash the raw tables for `x3f_calc_spatial_gain`
//! to combine on-the-fly, or pre-interpolate them into one dense table
//! per channel.
//!
//! Quattro HP: six channels (R, G, B0..B3) where B0..B3 are 2×2
//! sub-sampled with row/col offsets.
//!
//! Memory ownership: when `corr->malloc != 0`, `corr->gain` is owned by
//! `libc::malloc` (paired with `x3f_cleanup_spatial_gain` →
//! `libc::free`). When `malloc == 0`, `corr->gain` aliases the CAMF
//! payload and the cleanup is a no-op.
//!
//! The `alloca`-backed linked list of candidate gains in the C source is
//! replaced by a `Vec<MerrillSpatialGain>`; iteration order is preserved
//! by walking the Vec in reverse so any `<`-tiebreaks resolve identically
//! to the C version.
#![allow(clippy::missing_safety_doc)]

use std::ffi::CStr;
use std::mem;
use std::ptr;

use crate::*;
// libc compat — see `sysabi.rs`.
use crate::sysabi as libc;

/// Merrill's float32 spatial tables, before per-image interpolation.
/// Native ConvertGains / InterpFStop / GenerateSpatialGainTables: 0xaf750..0xb1ce4.
#[derive(Clone, Debug)]
pub(crate) struct MerrillSpatialTable {
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    pub(crate) values: Vec<f32>,
}

fn invalid_merrill_spatial() -> Error {
    Error::InvalidData("unsupported Merrill spatial gain calibration")
}

impl MerrillSpatialTable {
    fn validate(&self) -> Result<()> {
        if self.cols < 2
            || self.rows == 0
            || self.rows.checked_mul(self.cols) != Some(self.values.len())
            || self.values.iter().any(|v| !v.is_finite())
        {
            return Err(invalid_merrill_spatial());
        }
        Ok(())
    }

    fn blend(mut self, other: Self, a: f64, b: f64, fused: bool) -> Result<Self> {
        if self.rows != other.rows || self.cols != other.cols || !a.is_finite() || !b.is_finite() {
            return Err(invalid_merrill_spatial());
        }
        for (x, y) in self.values.iter_mut().zip(other.values) {
            let low = (f64::from(*x) * a) as f32;
            *x = if fused {
                (b as f32).mul_add(y, low)
            } else {
                low + (f64::from(y) * b) as f32
            };
        }
        self.validate()?;
        Ok(self)
    }
}

/// The image overload of ApplyGains interpolates vertically in f32, then
/// advances each horizontal knot in f64 using width/(table_columns-1).
pub(crate) struct MerrillSpatialRow {
    rows: [Vec<f32>; 3],
    span: [f64; 3],
    edge: [f64; 3],
    value: [f64; 3],
    slope: [f64; 3],
    knot: [usize; 3],
    column: usize,
    columns: usize,
}

impl MerrillSpatialRow {
    pub(crate) fn new(
        tables: &[MerrillSpatialTable; 3],
        row: usize,
        rows: usize,
        cols: usize,
    ) -> Result<Self> {
        if rows < 2 || row >= rows || cols == 0 {
            return Err(invalid_merrill_spatial());
        }
        let mut result = Self {
            rows: std::array::from_fn(|_| Vec::new()),
            span: [0.0; 3],
            edge: [0.0; 3],
            value: [0.0; 3],
            slope: [0.0; 3],
            knot: [0; 3],
            column: 0,
            columns: cols,
        };
        for (c, table) in tables.iter().enumerate() {
            // Tables are validated once when built. Check dimensions here so
            // synthetic/test callers cannot turn row interpolation into OOB reads.
            if table.cols < 2
                || table.rows == 0
                || table.rows.checked_mul(table.cols) != Some(table.values.len())
            {
                return Err(invalid_merrill_spatial());
            }
            let position = (row as f32 / (rows - 1) as f32) * (table.rows - 1) as f32;
            let low = position.floor() as usize;
            let high = position.ceil() as usize;
            if high >= table.rows {
                return Err(invalid_merrill_spatial());
            }
            let fraction = position - low as f32;
            result.rows[c]
                .try_reserve_exact(table.cols)
                .map_err(|_| Error::Allocation)?;
            for x in 0..table.cols {
                let a = table.values[low * table.cols + x];
                let b = table.values[high * table.cols + x];
                result.rows[c].push(fraction.mul_add(b - a, a));
            }
            result.span[c] = cols as f64 / (table.cols - 1) as f64;
            result.edge[c] = result.span[c];
            result.value[c] = f64::from(result.rows[c][0]);
            result.slope[c] = f64::from(result.rows[c][1] - result.rows[c][0]) / result.span[c];
        }
        Ok(result)
    }
}

impl Iterator for MerrillSpatialRow {
    type Item = [f64; 3];

    fn next(&mut self) -> Option<Self::Item> {
        if self.column == self.columns {
            return None;
        }
        for c in 0..3 {
            // Native increments once, including when the table is wider than
            // the raster; replacing this with direct bilinear lookup changes it.
            if self.column as f64 >= self.edge[c] {
                self.knot[c] += 1;
                let k = self.knot[c];
                self.value[c] = f64::from(self.rows[c][k]);
                self.slope[c] = f64::from(self.rows[c][k + 1] - self.rows[c][k]) / self.span[c];
                self.edge[c] += self.span[c];
            }
        }
        let result = self.value;
        for c in 0..3 {
            self.value[c] += self.slope[c];
        }
        self.column += 1;
        Some(result)
    }
}

unsafe fn merrill_camf_entry<'a>(x3f: *mut x3f_t, name: &CStr) -> Option<&'a camf_entry_t> {
    let de = unsafe { x3f_get_camf(x3f) };
    if de.is_null() {
        return None;
    }
    let table = unsafe { (*de).header.data_subsection.camf.entry_table };
    if table.element.is_null() || table.size == 0 {
        return None;
    }
    unsafe { std::slice::from_raw_parts(table.element, table.size as usize) }
        .iter()
        .find(|e| !e.name_address.is_null() && unsafe { CStr::from_ptr(e.name_address) } == name)
}

unsafe fn merrill_gain_table(
    x3f: *mut x3f_t,
    name: &CStr,
    channel: &CStr,
) -> Result<MerrillSpatialTable> {
    let mut key = [0; 32];
    let len = format_into(&mut key, "GainsTable", channel.to_bytes());
    let key = CStr::from_bytes_with_nul(&key[..=len]).unwrap();
    let mut target = ptr::null_mut();
    if unsafe {
        x3f_get_camf_property(
            x3f,
            name.as_ptr().cast_mut(),
            key.as_ptr().cast_mut(),
            &mut target,
        )
    } == 0
        || target.is_null()
    {
        return Err(invalid_merrill_spatial());
    }
    let entry = unsafe { merrill_camf_entry(x3f, CStr::from_ptr(target)) }
        .ok_or_else(invalid_merrill_spatial)?;
    if entry.id != 0x4d62_4d43 || entry.matrix_type != 5 || entry.matrix_dim != 2 {
        return Err(invalid_merrill_spatial());
    }
    let (mut data, mut rows, mut cols, mut minimum, mut delta) =
        (ptr::null_mut(), -1, -1, 0.0, 0.0);
    if unsafe {
        get_merrill_type_gains_table(
            x3f,
            name.as_ptr().cast_mut(),
            channel,
            &mut data,
            &mut rows,
            &mut cols,
            &mut minimum,
            &mut delta,
        )
    } == 0
        || rows < 1
        || cols < 2
        || !minimum.is_finite()
        || !delta.is_finite()
        || minimum == 0.0
        || delta == 0.0
    {
        return Err(invalid_merrill_spatial());
    }
    let len = (rows as usize)
        .checked_mul(cols as usize)
        .ok_or(Error::Allocation)?;
    if data.is_null() || len > isize::MAX as usize / mem::size_of::<u32>() {
        return Err(invalid_merrill_spatial());
    }
    let bytes = unsafe { std::slice::from_raw_parts(data, len) };
    let mut values = Vec::new();
    values
        .try_reserve_exact(len)
        .map_err(|_| Error::Allocation)?;
    for &byte in bytes {
        if byte > 255 {
            return Err(invalid_merrill_spatial());
        }
        // ConvertGains @0xaf9c4 rounds the product before adding the f32 minimum.
        values.push(minimum as f32 + (delta * f64::from(byte)) as f32);
    }
    let result = MerrillSpatialTable {
        rows: rows as usize,
        cols: cols as usize,
        values,
    };
    result.validate()?;
    Ok(result)
}

fn merrill_aperture_table(
    aperture: f32,
    stops: &[f64],
    squared: bool,
    mut get: impl FnMut(usize) -> Result<MerrillSpatialTable>,
) -> Result<MerrillSpatialTable> {
    if !aperture.is_finite()
        || aperture <= 0.0
        || stops.len() < 2
        || stops.iter().any(|x| !x.is_finite() || *x <= 0.0)
        || stops.windows(2).any(|p| p[0] >= p[1])
    {
        return Err(invalid_merrill_spatial());
    }
    let f = f64::from(aperture);
    if f <= stops[0] {
        return get(0);
    }
    if squared && f >= stops[stops.len() - 1] {
        return get(stops.len() - 1);
    }
    let high = stops
        .iter()
        .position(|x| *x >= f)
        .unwrap_or(stops.len() - 1);
    let a = if squared {
        let upper = stops[high] * stops[high];
        (upper - f64::from(aperture * aperture)) / (upper - stops[high - 1] * stops[high - 1])
    } else {
        (1.0 / f - 1.0 / stops[high]) / (1.0 / stops[high - 1] - 1.0 / stops[high])
    };
    get(high - 1)?.blend(get(high)?, a, f64::from((1.0 - a) as f32), true)
}

fn merrill_cra_weights(aperture: f32, cra: f32) -> (&'static str, f64, f64) {
    let q = 0.5 / f64::from(aperture);
    let phi = |a: f64, b: f64, low: f32| {
        ((q + a).min(b) + f64::from(((a - q) as f32).max(low)))
            .mul_add(0.5, 0.0)
            .atan()
    };
    let p50 = phi(
        f64::from_bits(0x3fc3_c50a_f08f_bd56),
        f64::from_bits(0x3fdd_7343_4719_bb36),
        f32::from_bits(0xbd28_cf0c),
    );
    let t = (std::f64::consts::PI / p50) * f64::from(cra / 180.0);
    if aperture >= 2.0 && t >= 1.0 {
        let r = phi(
            f64::from_bits(0x3fcf_a730_ea29_9de5),
            f64::from_bits(0x3fdd_23b6_93fe_1ec4),
            f32::from_bits(0xbd34_fbe9),
        ) / p50;
        ("85", (r - t) / (r - 1.0), (t - 1.0) / (r - 1.0))
    } else {
        let r = phi(
            f64::from_bits(0x3f8a_cf4e_4e15_9df0),
            f64::from_bits(0x3fd4_17db_677f_f305),
            f32::from_bits(0xbccc_8007),
        ) / p50;
        ("30", (t - r) / (1.0 - r), (1.0 - t) / (1.0 - r))
    }
}

/// Exact table construction for the supported Merrill CAMF layouts. `aperture`
/// is the as-shot F-number rounded to f32 by native preprocess @0xa6618.
/// None means the native table-family selector found no applicable calibration.
/// SkipStepFlags bit4 and the caller's spatial-gain option are handled by caller.
pub(crate) unsafe fn merrill_spatial_gain(
    x3f: *mut x3f_t,
    aperture: f32,
    control: Control<'_>,
) -> Result<Option<[MerrillSpatialTable; 3]>> {
    control.check()?;
    if !aperture.is_finite() || aperture <= 0.0 {
        return Err(invalid_merrill_spatial());
    }
    let mut cra = 0.0;
    unsafe {
        x3f_get_camf_float(x3f, c"ChiefRayAngle".as_ptr().cast_mut(), &mut cra);
    }
    if !cra.is_finite() {
        return Err(invalid_merrill_spatial());
    }
    let cra = cra as f32;
    let mut stops = Vec::new();
    let (family, a, b) = if cra > 0.0 {
        let (mut names, mut values, mut count) = (ptr::null_mut(), ptr::null_mut(), 0);
        if unsafe {
            x3f_get_camf_property_list(
                x3f,
                c"SpatialGainsProps_1.4_30".as_ptr().cast_mut(),
                &mut names,
                &mut values,
                &mut count,
            )
        } == 0
            || count == 0
        {
            return Ok(None);
        }
        stops.extend([1.4, 2.0, 2.8, 8.0]);
        merrill_cra_weights(aperture, cra)
    } else {
        let (mut data, mut count) = (ptr::null_mut(), 0);
        if unsafe {
            x3f_get_camf_matrix_var(
                x3f,
                c"SpatialGain_Fstop".as_ptr().cast_mut(),
                &mut count,
                ptr::null_mut(),
                ptr::null_mut(),
                matrix_type_t_M_FLOAT,
                &mut data,
            )
        } == 0
        {
            if unsafe { merrill_camf_entry(x3f, c"SpatialGain_Fstop") }.is_some() {
                return Err(invalid_merrill_spatial());
            }
            return Ok(None);
        }
        if !(2..=10).contains(&count) {
            return Err(invalid_merrill_spatial());
        }
        stops.extend_from_slice(unsafe {
            std::slice::from_raw_parts(data.cast::<f64>(), count as usize)
        });
        let mut focus = 0;
        unsafe {
            x3f_get_camf_unsigned(x3f, c"ScaleENC".as_ptr().cast_mut(), &mut focus);
        }
        let a = (31.0 - f64::from(focus as i32)) / 31.0;
        ("MOD", a, 1.0 - a)
    };
    let mut tables = Vec::new();
    for channel in [c"R", c"G", c"B"] {
        control.check()?;
        let get = |focus: &str| {
            merrill_aperture_table(aperture, &stops, cra > 0.0, |i| {
                let label = if cra > 0.0 {
                    ["1.4", "2.0", "2.8", "8.0"][i].to_owned()
                } else {
                    i.to_string()
                };
                let name =
                    std::ffi::CString::new(format!("SpatialGainsProps_{label}_{focus}")).unwrap();
                unsafe { merrill_gain_table(x3f, &name, channel) }
            })
        };
        tables.push(get(if cra > 0.0 { "50" } else { "INF" })?.blend(get(family)?, a, b, false)?);
    }
    control.check()?;
    Ok(Some(tables.try_into().unwrap()))
}

#[derive(Clone)]
struct MerrillSpatialGain {
    name: *mut libc::c_char,
    x: f64,
    y: f64,
}

#[inline]
fn lens_position(focal_length: f64, object_distance: f64) -> f64 {
    1.0 / (1.0 / focal_length - 1.0 / object_distance)
}

unsafe fn get_focal_length(x3f: *mut x3f_t) -> f64 {
    let mut flength: *mut libc::c_char = ptr::null_mut();
    if unsafe { x3f_get_prop_entry(x3f, c"FLENGTH".as_ptr() as *mut _, &mut flength) } != 0 {
        unsafe { libc::atof(flength) }
    } else {
        let focal_length = 30.0_f64;
        unsafe {
            x3f_printf(
                x3f_verbosity_t_WARN,
                c"Could not get focal length, assuming %g mm\n".as_ptr(),
                focal_length,
            );
        }
        focal_length
    }
}

unsafe fn get_object_distance(x3f: *mut x3f_t) -> f64 {
    let mut object_distance: f64 = 0.0;
    if unsafe {
        x3f_get_camf_float(
            x3f,
            c"ObjectDistance".as_ptr() as *mut _,
            &mut object_distance,
        )
    } != 0
    {
        // Convert cm to mm.
        object_distance * 10.0
    } else {
        let object_distance = f64::INFINITY;
        unsafe {
            x3f_printf(
                x3f_verbosity_t_WARN,
                c"Could not get object distance, assuming %g mm\n".as_ptr(),
                object_distance,
            );
        }
        object_distance
    }
}

unsafe fn get_mod(x3f: *mut x3f_t) -> f64 {
    let mut lens_information: i32 = 0;
    if unsafe {
        x3f_get_camf_signed(
            x3f,
            c"LensInformation".as_ptr() as *mut _,
            &mut lens_information,
        )
    } == 0
    {
        lens_information = -1;
    }
    match lens_information {
        1003 => 200.0, // DP1 Merrill
        1004 => 280.0, // DP2 Merrill
        1005 => 226.0, // DP3 Merrill
        _ => {
            let mod_ = 280.0_f64;
            unsafe {
                x3f_printf(
                    x3f_verbosity_t_WARN,
                    c"Could not get MOD, assuming %g mm\n".as_ptr(),
                    mod_,
                );
            }
            mod_
        }
    }
}

/// Read `GainsTable<chan>` / `MinGains<chan>` / `Delta<chan>` for a
/// named CAMF block. On success, populates `*mgain`, asserts row/col
/// dimensions match across calls (or fills them on first call), and
/// fills `mingain` / `delta`. Returns 1 on success, 0 on failure.
unsafe fn get_merrill_type_gains_table(
    x3f: *mut x3f_t,
    name: *mut libc::c_char,
    chan: &CStr,
    mgain: *mut *mut u32,
    rows: *mut libc::c_int,
    cols: *mut libc::c_int,
    mingain: *mut f64,
    delta: *mut f64,
) -> libc::c_int {
    let mut buf = [0u8; 32];
    let mut val: *mut libc::c_char = ptr::null_mut();

    // GainsTable<chan>
    let n = format_into(&mut buf, "GainsTable", chan.to_bytes());
    let key = unsafe { CStr::from_bytes_with_nul_unchecked(&buf[..n + 1]) };
    let mut rows_tmp: libc::c_int = 0;
    let mut cols_tmp: libc::c_int = 0;
    let ok = unsafe {
        x3f_get_camf_property(x3f, name, key.as_ptr() as *mut _, &mut val) != 0
            && x3f_get_camf_matrix_var(
                x3f,
                val,
                &mut rows_tmp,
                &mut cols_tmp,
                ptr::null_mut(),
                matrix_type_t_M_UINT,
                mgain as *mut *mut libc::c_void,
            ) != 0
    };
    if !ok {
        return 0;
    }
    unsafe {
        if (*rows != -1 && *rows != rows_tmp) || (*cols != -1 && *cols != cols_tmp) {
            return 0;
        }
        *rows = rows_tmp;
        *cols = cols_tmp;
    }

    // MinGains<chan>
    let n = format_into(&mut buf, "MinGains", chan.to_bytes());
    let key = unsafe { CStr::from_bytes_with_nul_unchecked(&buf[..n + 1]) };
    if unsafe { x3f_get_camf_property(x3f, name, key.as_ptr() as *mut _, &mut val) } == 0 {
        return 0;
    }
    unsafe {
        *mingain = libc::atof(val);
    }

    // Delta<chan>
    let n = format_into(&mut buf, "Delta", chan.to_bytes());
    let key = unsafe { CStr::from_bytes_with_nul_unchecked(&buf[..n + 1]) };
    if unsafe { x3f_get_camf_property(x3f, name, key.as_ptr() as *mut _, &mut val) } == 0 {
        return 0;
    }
    unsafe {
        *delta = libc::atof(val);
    }

    1
}

/// Concatenate `prefix` + `suffix` into `buf` and NUL-terminate.
/// Panics if the result would overflow `buf`.
fn format_into(buf: &mut [u8; 32], prefix: &str, suffix: &[u8]) -> usize {
    let pb = prefix.as_bytes();
    let n = pb.len() + suffix.len();
    assert!(n + 1 <= buf.len());
    buf[..pb.len()].copy_from_slice(pb);
    buf[pb.len()..n].copy_from_slice(suffix);
    buf[n] = 0;
    n
}

/// Parse `"SpatialGainHPProps_<int><EOF>"`. Returns `Some(int)` if the
/// pattern matches with no trailing characters; `None` otherwise.
fn parse_hp_props_index(s: &CStr) -> Option<u32> {
    let bytes = s.to_bytes();
    let prefix = b"SpatialGainHPProps_";
    let rest = bytes.strip_prefix(prefix.as_slice())?;
    if rest.is_empty() {
        return None;
    }
    let s = std::str::from_utf8(rest).ok()?;
    s.parse::<u32>().ok()
}

/// Parse `"SpatialGainsProps_<int>_<3char><EOF>"`. The C version uses
/// `sscanf("SpatialGainsProps_%d_%3s%c", &idx, focus, &dummy) == 2` —
/// 2 means: int + 3-char string matched, dummy did NOT match (i.e. no
/// trailing chars). Returns `Some((idx, [u8; 3]))` on match.
fn parse_focus_props(s: &CStr) -> Option<(u32, [u8; 3])> {
    let bytes = s.to_bytes();
    let prefix = b"SpatialGainsProps_";
    let rest = bytes.strip_prefix(prefix.as_slice())?;
    let underscore = rest.iter().position(|&b| b == b'_')?;
    let int_part = std::str::from_utf8(&rest[..underscore]).ok()?;
    let idx: u32 = int_part.parse().ok()?;
    let after = &rest[underscore + 1..];
    if after.len() != 3 {
        return None;
    }
    let mut focus = [0u8; 3];
    focus.copy_from_slice(after);
    Some((idx, focus))
}

/// Parse `"SpatialGainsProps_<dbl>_<dbl><EOF>"`. The C version uses
/// `sscanf("SpatialGainsProps_%lf_%lf%c", &a, &b, &dummy) == 2`.
/// Returns `Some((aperture, lenspos))`.
fn parse_aperture_lenspos_props(s: &CStr) -> Option<(f64, f64)> {
    let bytes = s.to_bytes();
    let prefix = b"SpatialGainsProps_";
    let rest = bytes.strip_prefix(prefix.as_slice())?;
    // sscanf "%lf" in C will greedily consume digits, optional sign,
    // decimal point, and exponent. We delegate to f64::from_str on the
    // "longest match" boundary: locate the underscore between the two
    // numbers.
    let underscore = rest.iter().position(|&b| b == b'_')?;
    let a_str = std::str::from_utf8(&rest[..underscore]).ok()?;
    let b_str = std::str::from_utf8(&rest[underscore + 1..]).ok()?;
    let a: f64 = a_str.parse().ok()?;
    let b: f64 = b_str.parse().ok()?;
    Some((a, b))
}

#[no_mangle]
pub unsafe extern "C" fn x3f_get_merrill_type_spatial_gain(
    x3f: *mut x3f_t,
    hp_flag: libc::c_int,
    corr: *mut x3f_spatial_gain_corr_t,
) -> libc::c_int {
    let mut capture_aperture: f64 = 0.0;
    if unsafe {
        x3f_get_camf_float(
            x3f,
            c"CaptureAperture".as_ptr() as *mut _,
            &mut capture_aperture,
        )
    } == 0
    {
        return 0;
    }

    let mut include_blocks: *mut *mut libc::c_char = ptr::null_mut();
    let mut include_blocks_val: *mut *mut libc::c_char = ptr::null_mut();
    let mut include_blocks_num: u32 = 0;
    if unsafe {
        x3f_get_camf_property_list(
            x3f,
            c"IncludeBlocks".as_ptr() as *mut _,
            &mut include_blocks,
            &mut include_blocks_val,
            &mut include_blocks_num,
        )
    } == 0
    {
        return 0;
    }

    let mut spatial_gain_fstop: *mut f64 = ptr::null_mut();
    let mut num_fstop: libc::c_int = 0;
    let mut corr_num: libc::c_int = 3;
    let mut gains: Vec<MerrillSpatialGain> = Vec::new();
    let x;
    let y;

    if hp_flag != 0 {
        // Quattro HP — 6 channels (R, G, B0, B1, B2, B3).
        let ok = unsafe {
            x3f_get_camf_matrix_var(
                x3f,
                c"SpatialGainHP_Fstop".as_ptr() as *mut _,
                &mut num_fstop,
                ptr::null_mut(),
                ptr::null_mut(),
                matrix_type_t_M_FLOAT,
                &mut spatial_gain_fstop as *mut *mut f64 as *mut *mut libc::c_void,
            )
        };
        if ok == 0 {
            return 0;
        }
        corr_num = 6;

        for i in 0..include_blocks_num {
            let block = unsafe { *include_blocks.offset(i as isize) };
            let block_cstr = unsafe { CStr::from_ptr(block) };
            let Some(aperture_index) = parse_hp_props_index(block_cstr) else {
                continue;
            };
            let mut names: *mut *mut libc::c_char = ptr::null_mut();
            let mut values: *mut *mut libc::c_char = ptr::null_mut();
            let mut num: u32 = 0;
            let pl_ok = unsafe {
                x3f_get_camf_property_list(x3f, block, &mut names, &mut values, &mut num)
            };
            if pl_ok != 0 && (aperture_index as libc::c_int) < num_fstop {
                let fstop_value = unsafe { *spatial_gain_fstop.offset(aperture_index as isize) };
                gains.push(MerrillSpatialGain {
                    name: block,
                    x: 1.0 / fstop_value,
                    y: 0.0,
                });
            }
        }

        x = 1.0 / capture_aperture;
        y = 0.0;
    } else {
        // Merrill or non-HP Quattro.
        let ok = unsafe {
            x3f_get_camf_matrix_var(
                x3f,
                c"SpatialGain_Fstop".as_ptr() as *mut _,
                &mut num_fstop,
                ptr::null_mut(),
                ptr::null_mut(),
                matrix_type_t_M_FLOAT,
                &mut spatial_gain_fstop as *mut *mut f64 as *mut *mut libc::c_void,
            )
        };
        if ok != 0 {
            for i in 0..include_blocks_num {
                let block = unsafe { *include_blocks.offset(i as isize) };
                let block_cstr = unsafe { CStr::from_ptr(block) };
                let Some((aperture_index, focus)) = parse_focus_props(block_cstr) else {
                    continue;
                };
                let mut names: *mut *mut libc::c_char = ptr::null_mut();
                let mut values: *mut *mut libc::c_char = ptr::null_mut();
                let mut num: u32 = 0;
                let pl_ok = unsafe {
                    x3f_get_camf_property_list(x3f, block, &mut names, &mut values, &mut num)
                };
                if pl_ok == 0 || (aperture_index as libc::c_int) >= num_fstop {
                    continue;
                }
                let lenspos = if &focus == b"INF" {
                    lens_position(unsafe { get_focal_length(x3f) }, f64::INFINITY)
                } else if &focus == b"MOD" {
                    lens_position(unsafe { get_focal_length(x3f) }, unsafe { get_mod(x3f) })
                } else {
                    continue;
                };
                let fstop_value = unsafe { *spatial_gain_fstop.offset(aperture_index as isize) };
                gains.push(MerrillSpatialGain {
                    name: block,
                    x: 1.0 / fstop_value,
                    y: lenspos,
                });
            }
        } else {
            for i in 0..include_blocks_num {
                let block = unsafe { *include_blocks.offset(i as isize) };
                let block_cstr = unsafe { CStr::from_ptr(block) };
                let Some((aperture, lenspos)) = parse_aperture_lenspos_props(block_cstr) else {
                    continue;
                };
                let mut names: *mut *mut libc::c_char = ptr::null_mut();
                let mut values: *mut *mut libc::c_char = ptr::null_mut();
                let mut num: u32 = 0;
                let pl_ok = unsafe {
                    x3f_get_camf_property_list(x3f, block, &mut names, &mut values, &mut num)
                };
                if pl_ok == 0 {
                    continue;
                }
                gains.push(MerrillSpatialGain {
                    name: block,
                    x: 1.0 / aperture,
                    y: lenspos,
                });
            }
        }

        x = 1.0 / capture_aperture;
        y = lens_position(unsafe { get_focal_length(x3f) }, unsafe {
            get_object_distance(x3f)
        });
    }

    // Find the closest neighbour in each of the 4 quadrants. C
    // traverses a linked list built by prepending, so iteration is
    // last-pushed-first. Mirror that with `.iter().rev()`.
    let mut q_closest: [Option<&MerrillSpatialGain>; 4] = [None; 4];
    let mut q_closest_dx = [
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
        f64::INFINITY,
    ];
    let mut q_closest_dy = [
        f64::INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
    ];
    let mut q_closest_d2 = [f64::INFINITY; 4];

    for g in gains.iter().rev() {
        let dx = g.x - x;
        let dy = g.y - y;
        let d2 = dx * dx + dy * dy;
        let q = if dx > 0.0 && dy > 0.0 {
            0
        } else if dx > 0.0 {
            3
        } else if dy > 0.0 {
            1
        } else {
            2
        };
        if d2 < q_closest_d2[q] {
            q_closest[q] = Some(g);
            q_closest_dx[q] = dx;
            q_closest_dy[q] = dy;
            q_closest_d2[q] = d2;
        }
    }

    let mut q_weight_x = [0.0_f64; 4];
    let mut q_weight_y = [0.0_f64; 4];
    let mut q_weight = [0.0_f64; 4];

    q_weight_x[0] = q_closest_dx[1] / (q_closest_dx[1] - q_closest_dx[0]);
    q_weight_x[1] = q_closest_dx[0] / (q_closest_dx[0] - q_closest_dx[1]);
    q_weight_x[2] = q_closest_dx[3] / (q_closest_dx[3] - q_closest_dx[2]);
    q_weight_x[3] = q_closest_dx[2] / (q_closest_dx[2] - q_closest_dx[3]);

    q_weight_y[0] = q_closest_dy[3] / (q_closest_dy[3] - q_closest_dy[0]);
    q_weight_y[1] = q_closest_dy[2] / (q_closest_dy[2] - q_closest_dy[1]);
    q_weight_y[2] = q_closest_dy[1] / (q_closest_dy[1] - q_closest_dy[2]);
    q_weight_y[3] = q_closest_dy[0] / (q_closest_dy[0] - q_closest_dy[3]);

    unsafe {
        x3f_printf(x3f_verbosity_t_DEBUG, c"x = %f y = %f\n".as_ptr(), x, y);
        for i in 0..4 {
            if let Some(g) = q_closest[i] {
                x3f_printf(
                    x3f_verbosity_t_DEBUG,
                    c"q = %d name = %s x = %f y = %f\n".as_ptr(),
                    i as libc::c_int,
                    g.name,
                    g.x,
                    g.y,
                );
            } else {
                x3f_printf(
                    x3f_verbosity_t_DEBUG,
                    c"q = %d name = NULL\n".as_ptr(),
                    i as libc::c_int,
                );
            }
            x3f_printf(
                x3f_verbosity_t_DEBUG,
                c"q = %d dx = %f dy = %f d2 = %f wx = %f wy = %f\n".as_ptr(),
                i as libc::c_int,
                q_closest_dx[i],
                q_closest_dy[i],
                q_closest_d2[i],
                q_weight_x[i],
                q_weight_y[i],
            );
        }
    }

    for i in 0..4 {
        if q_weight_x[i].is_nan() {
            q_weight_x[i] = 1.0;
        }
        if q_weight_y[i].is_nan() {
            q_weight_y[i] = 1.0;
        }
        q_weight[i] = q_weight_x[i] * q_weight_y[i];
    }

    unsafe {
        for i in 0..4 {
            let name = match q_closest[i] {
                Some(g) => g.name,
                None => c"NULL".as_ptr() as *mut _,
            };
            x3f_printf(
                x3f_verbosity_t_DEBUG,
                c"q = %d name = %s w = %f\n".as_ptr(),
                i as libc::c_int,
                name,
                q_weight[i],
            );
        }
    }

    unsafe {
        for i in 0..corr_num as usize {
            let c = corr.add(i);
            (*c).gain = ptr::null_mut();
            (*c).malloc = 0;
            (*c).rows = -1;
            (*c).cols = -1;
            (*c).rowoff = 0;
            (*c).coloff = 0;
            (*c).rowpitch = 1;
            (*c).colpitch = 1;
            (*c).chan = i as libc::c_int;
            (*c).channels = 1;
            (*c).mgain_num = 0;
        }
    }

    if hp_flag != 0 {
        unsafe {
            (*corr.add(2)).rowoff = 0;
            (*corr.add(2)).coloff = 0;
            (*corr.add(2)).rowpitch = 2;
            (*corr.add(2)).colpitch = 2;
            (*corr.add(2)).chan = 2;

            (*corr.add(3)).rowoff = 0;
            (*corr.add(3)).coloff = 1;
            (*corr.add(3)).rowpitch = 2;
            (*corr.add(3)).colpitch = 2;
            (*corr.add(3)).chan = 2;

            (*corr.add(4)).rowoff = 1;
            (*corr.add(4)).coloff = 0;
            (*corr.add(4)).rowpitch = 2;
            (*corr.add(4)).colpitch = 2;
            (*corr.add(4)).chan = 2;

            (*corr.add(5)).rowoff = 1;
            (*corr.add(5)).coloff = 1;
            (*corr.add(5)).rowpitch = 2;
            (*corr.add(5)).colpitch = 2;
            (*corr.add(5)).chan = 2;
        }
    }

    let channels_normal: [&CStr; 6] = [c"R", c"G", c"B", c"", c"", c""];
    let channels_hp: [&CStr; 6] = [c"R", c"G", c"B0", c"B1", c"B2", c"B3"];
    let channels: &[&CStr; 6] = if hp_flag != 0 {
        &channels_hp
    } else {
        &channels_normal
    };

    for i in 0..4 {
        let Some(g) = q_closest[i] else {
            continue;
        };
        for j in 0..corr_num as usize {
            unsafe {
                let c = corr.add(j);
                let m_idx = (*c).mgain_num as usize;
                (*c).mgain_num += 1;
                let m = (*c).mgain.as_mut_ptr().add(m_idx);
                (*m).weight = q_weight[i];
                let ok = get_merrill_type_gains_table(
                    x3f,
                    g.name,
                    channels[j],
                    &mut (*m).gain,
                    &mut (*c).rows,
                    &mut (*c).cols,
                    &mut (*m).mingain,
                    &mut (*m).delta,
                );
                if ok == 0 {
                    return 0;
                }
            }
        }
    }

    unsafe {
        for i in 0..corr_num as usize {
            if (*corr.add(i)).mgain_num == 0 {
                return 0;
            }
        }
    }

    corr_num
}

#[no_mangle]
pub unsafe extern "C" fn x3f_get_interp_merrill_type_spatial_gain(
    x3f: *mut x3f_t,
    hp_flag: libc::c_int,
    corr: *mut x3f_spatial_gain_corr_t,
) -> libc::c_int {
    let corr_num = unsafe { x3f_get_merrill_type_spatial_gain(x3f, hp_flag, corr) };

    for i in 0..corr_num as usize {
        unsafe {
            let c = corr.add(i);
            let num = ((*c).rows * (*c).cols * (*c).channels) as usize;
            (*c).gain = libc::malloc(num * mem::size_of::<f64>()) as *mut f64;
            (*c).malloc = 1;

            for j in 0..num {
                let mut sum = 0.0_f64;
                for g in 0..(*c).mgain_num as usize {
                    let m = (*c).mgain.as_ptr().add(g);
                    sum += (*m).weight * ((*m).mingain + (*m).delta * (*(*m).gain.add(j)) as f64);
                }
                *(*c).gain.add(j) = sum;
            }
        }
    }

    corr_num
}

#[no_mangle]
pub unsafe extern "C" fn x3f_get_classic_spatial_gain(
    x3f: *mut x3f_t,
    wb: *mut libc::c_char,
    corr: *mut x3f_spatial_gain_corr_t,
) -> libc::c_int {
    let mut gain_name: *mut libc::c_char = ptr::null_mut();
    let prop_ok = unsafe {
        x3f_get_camf_property(
            x3f,
            c"SpatialGainTables".as_ptr() as *mut _,
            wb,
            &mut gain_name,
        ) != 0
            && x3f_get_camf_matrix_var(
                x3f,
                gain_name,
                &mut (*corr).rows,
                &mut (*corr).cols,
                &mut (*corr).channels,
                matrix_type_t_M_FLOAT,
                &mut (*corr).gain as *mut *mut f64 as *mut *mut libc::c_void,
            ) != 0
    };

    let fallback_ok = if prop_ok {
        true
    } else {
        unsafe {
            x3f_get_camf_matrix_var(
                x3f,
                c"SpatialGain".as_ptr() as *mut _,
                &mut (*corr).rows,
                &mut (*corr).cols,
                &mut (*corr).channels,
                matrix_type_t_M_FLOAT,
                &mut (*corr).gain as *mut *mut f64 as *mut *mut libc::c_void,
            ) != 0
        }
    };

    if !prop_ok && !fallback_ok {
        return 0;
    }

    unsafe {
        (*corr).malloc = 0;
        (*corr).rowoff = 0;
        (*corr).coloff = 0;
        (*corr).rowpitch = 1;
        (*corr).colpitch = 1;
        (*corr).chan = 0;
        (*corr).mgain_num = 0;
    }
    1
}

#[no_mangle]
pub unsafe extern "C" fn x3f_get_spatial_gain(
    x3f: *mut x3f_t,
    wb: *mut libc::c_char,
    corr: *mut x3f_spatial_gain_corr_t,
) -> libc::c_int {
    let mut corr_num = 0;
    corr_num +=
        unsafe { x3f_get_interp_merrill_type_spatial_gain(x3f, 0, corr.add(corr_num as usize)) };
    if corr_num == 0 {
        corr_num += unsafe { x3f_get_classic_spatial_gain(x3f, wb, corr.add(corr_num as usize)) };
    }
    corr_num
}

#[no_mangle]
pub unsafe extern "C" fn x3f_cleanup_spatial_gain(
    corr: *mut x3f_spatial_gain_corr_t,
    corr_num: libc::c_int,
) {
    for i in 0..corr_num as usize {
        unsafe {
            let c = corr.add(i);
            if (*c).malloc != 0 {
                libc::free((*c).gain as *mut libc::c_void);
            }
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn x3f_calc_spatial_gain(
    corr: *mut x3f_spatial_gain_corr_t,
    corr_num: libc::c_int,
    row: libc::c_int,
    col: libc::c_int,
    chan: libc::c_int,
    rows: libc::c_int,
    cols: libc::c_int,
) -> f64 {
    let mut gain = 1.0_f64;
    let rrel = row as f64 / rows as f64;
    let crel = col as f64 / cols as f64;

    for i in 0..corr_num as usize {
        let c = unsafe { &mut *corr.add(i) };
        let ch = chan - c.chan;
        if ch < 0 || ch >= c.channels {
            continue;
        }
        if row % c.rowpitch != c.rowoff {
            continue;
        }
        if col % c.colpitch != c.coloff {
            continue;
        }

        let rc = rrel * (c.rows - 1) as f64;
        let ri = rc.floor() as libc::c_int;
        let rf = rc - ri as f64;

        let cc = crel * (c.cols - 1) as f64;
        let ci = cc.floor() as libc::c_int;
        let cf = cc - ci as f64;

        let r1: *const f64;
        let r2: *const f64;
        unsafe {
            if ri < 0 {
                r1 = c.gain;
                r2 = c.gain;
            } else if ri >= c.rows - 1 {
                let off = ((c.rows - 1) * c.cols * c.channels) as isize;
                r1 = c.gain.offset(off);
                r2 = c.gain.offset(off);
            } else {
                r1 = c.gain.offset((ri * c.cols * c.channels) as isize);
                r2 = c.gain.offset(((ri + 1) * c.cols * c.channels) as isize);
            }
        }

        // NB: the C source has a known bug — the first branch of the
        // ci<0 check uses an `if` (not `else if`) for `ci>=c->cols-1`,
        // so when ci<0 we fall through and overwrite co1/co2 from the
        // (c->cols-1) branch. We preserve the bug verbatim for byte-
        // identity with the legacy converter. See src/x3f_spatial_gain.c
        // line 457-460 in the original.
        let co1: libc::c_int;
        let co2: libc::c_int;
        if ci < 0 {
            // C wrote `co1 = co2 = ch;` here, but then the next `if`
            // (not `else if`) immediately reassigns. Mimic that.
            let _co1_intermediate = ch;
            let _co2_intermediate = ch;
            if ci >= c.cols - 1 {
                // Unreachable when ci<0 and c.cols>=1, but emit the
                // assignment for fidelity.
                co1 = (c.cols - 1) * c.channels + ch;
                co2 = (c.cols - 1) * c.channels + ch;
            } else {
                co1 = ci * c.channels + ch;
                co2 = (ci + 1) * c.channels + ch;
            }
        } else if ci >= c.cols - 1 {
            co1 = (c.cols - 1) * c.channels + ch;
            co2 = (c.cols - 1) * c.channels + ch;
        } else {
            co1 = ci * c.channels + ch;
            co2 = (ci + 1) * c.channels + ch;
        }

        let r1c1 = unsafe { *r1.offset(co1 as isize) };
        let r1c2 = unsafe { *r1.offset(co2 as isize) };
        let r2c1 = unsafe { *r2.offset(co1 as isize) };
        let r2c2 = unsafe { *r2.offset(co2 as isize) };

        let gr1 = r1c1 + cf * (r1c2 - r1c1);
        let gr2 = r2c1 + cf * (r2c2 - r2c1);

        gain *= gr1 + rf * (gr2 - gr1);
    }
    gain
}

// Symbol anchors so cross-crate dead-code elimination can't strip the
// Rust definitions before the still-C call sites in x3f_process.c link.
#[used]
static _A_GET_MERRILL: unsafe extern "C" fn(
    *mut x3f_t,
    libc::c_int,
    *mut x3f_spatial_gain_corr_t,
) -> libc::c_int = x3f_get_merrill_type_spatial_gain;
#[used]
static _A_GET_INTERP_MERRILL: unsafe extern "C" fn(
    *mut x3f_t,
    libc::c_int,
    *mut x3f_spatial_gain_corr_t,
) -> libc::c_int = x3f_get_interp_merrill_type_spatial_gain;
#[used]
static _A_GET_CLASSIC: unsafe extern "C" fn(
    *mut x3f_t,
    *mut libc::c_char,
    *mut x3f_spatial_gain_corr_t,
) -> libc::c_int = x3f_get_classic_spatial_gain;
#[used]
static _A_GET_SGAIN: unsafe extern "C" fn(
    *mut x3f_t,
    *mut libc::c_char,
    *mut x3f_spatial_gain_corr_t,
) -> libc::c_int = x3f_get_spatial_gain;
#[used]
static _A_CLEANUP_SGAIN: unsafe extern "C" fn(*mut x3f_spatial_gain_corr_t, libc::c_int) =
    x3f_cleanup_spatial_gain;
#[used]
static _A_CALC_SGAIN: unsafe extern "C" fn(
    *mut x3f_spatial_gain_corr_t,
    libc::c_int,
    libc::c_int,
    libc::c_int,
    libc::c_int,
    libc::c_int,
    libc::c_int,
) -> f64 = x3f_calc_spatial_gain;

#[cfg(test)]
mod tests {
    use super::*;

    fn constant_table(value: f32) -> MerrillSpatialTable {
        MerrillSpatialTable {
            rows: 1,
            cols: 2,
            values: vec![value; 2],
        }
    }

    fn float_hash(values: impl IntoIterator<Item = f32>) -> u64 {
        values
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .fold(0xcbf29ce484222325, |h, v| {
                (h ^ u64::from(v)).wrapping_mul(0x100000001b3)
            })
    }

    #[test]
    fn merrill_spatial_aperture_endpoints_and_extrapolation() {
        let stops = [2.0, 4.0, 8.0];
        let get = |i| Ok(constant_table([1.0, 2.0, 4.0][i]));
        assert_eq!(
            merrill_aperture_table(1.0, &stops, false, get)
                .unwrap()
                .values,
            [1.0; 2]
        );
        assert_eq!(
            merrill_aperture_table(4.0, &stops, false, get)
                .unwrap()
                .values,
            [2.0; 2]
        );
        // DP extrapolates the final reciprocal-aperture interval; CRA clamps.
        assert_eq!(
            merrill_aperture_table(16.0, &stops, false, get)
                .unwrap()
                .values,
            [5.0; 2]
        );
        assert_eq!(
            merrill_aperture_table(16.0, &stops, true, get)
                .unwrap()
                .values,
            [4.0; 2]
        );
        for stops in [
            &[2.0][..],
            &[2.0, 2.0],
            &[4.0, 2.0],
            &[0.0, 4.0],
            &[2.0, f64::NAN],
        ] {
            assert!(merrill_aperture_table(2.8, stops, false, get).is_err());
        }
        for aperture in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            assert!(merrill_aperture_table(aperture, &stops, false, get).is_err());
        }
        assert!(constant_table(1.0)
            .blend(
                MerrillSpatialTable {
                    rows: 2,
                    cols: 2,
                    values: vec![1.0; 4]
                },
                0.5,
                0.5,
                true
            )
            .is_err());
    }

    #[test]
    fn merrill_spatial_missing_and_malformed_camf_are_distinct() {
        unsafe {
            let mut values = [2.8_f64, 4.0, 8.0];
            let mut dim: camf_dim_entry_t = std::mem::zeroed();
            dim.size = 3;
            let mut entry: camf_entry_t = std::mem::zeroed();
            entry.id = u32::from_le_bytes(*b"CMbM");
            entry.name_address = c"Other".as_ptr().cast_mut();
            entry.matrix_type = 3;
            entry.matrix_dim = 1;
            entry.matrix_dim_entry = &mut dim;
            entry.matrix_elements = 3;
            entry.matrix_decoded_type = matrix_type_t_M_FLOAT;
            entry.matrix_decoded = values.as_mut_ptr().cast();
            let mut section: x3f_directory_entry_t = std::mem::zeroed();
            section.header.identifier = u32::from_le_bytes(*b"SECc");
            section.header.data_subsection.camf.entry_table = camf_entry_table_t {
                size: 1,
                element: &mut entry,
            };
            let mut x3f: x3f_t = std::mem::zeroed();
            x3f.directory_section.num_directory_entries = 1;
            x3f.directory_section.directory_entry = &mut section;
            assert!(merrill_spatial_gain(&mut x3f, 4.0, Control::none())
                .unwrap()
                .is_none());
            let entry_ptr = section.header.data_subsection.camf.entry_table.element;
            (*entry_ptr).name_address = c"SpatialGain_Fstop".as_ptr().cast_mut();
            (*entry_ptr).matrix_elements = 2;
            assert!(matches!(
                merrill_spatial_gain(&mut x3f, 4.0, Control::none()),
                Err(Error::InvalidData(_))
            ));
            (*entry_ptr).matrix_elements = 3;
            (*entry_ptr).matrix_decoded_type = matrix_type_t_M_UINT;
            assert!(matches!(
                merrill_spatial_gain(&mut x3f, 4.0, Control::none()),
                Err(Error::InvalidData(_))
            ));
            (*entry_ptr).matrix_decoded_type = matrix_type_t_M_FLOAT;
            // A valid selector with missing channel tables is malformed rather
            // than a partially applied correction.
            assert!(matches!(
                merrill_spatial_gain(&mut x3f, 4.0, Control::none()),
                Err(Error::InvalidData(_))
            ));
            let cancelled = std::sync::atomic::AtomicBool::new(true);
            assert!(matches!(
                merrill_spatial_gain(&mut x3f, 4.0, Control::new(&cancelled)),
                Err(Error::Cancelled)
            ));
        }
    }

    #[test]
    fn merrill_spatial_row_matches_native_apply_gains() {
        // Original exported image ApplyGains, SHA256 037419dc...6ce5e6.
        // Retained oracle: spp-rgb-trace/float-preprocess/check_apply_gains.cpp.
        let tables = std::array::from_fn(|c| {
            let (cols, rows) = (3 + 2 * c, 3 + c);
            MerrillSpatialTable {
                rows,
                cols,
                values: (0..cols * rows)
                    .map(|i| 0.8_f32 + ((i * 7 + c * 3) % 17) as f32 * 0.071_f32)
                    .collect(),
            }
        });
        for (width, expected) in [
            (9, 0x975b9335a4245ddf),
            (37, 0x7951b2f5927245f1),
            (97, 0xb63b750844e59ce2),
        ] {
            let height = 13;
            let stride = width * 3 + 7;
            let mut pixels = vec![-12345.25_f32; stride * height];
            for y in 0..height {
                assert_eq!(
                    MerrillSpatialRow::new(&tables, y, height, width)
                        .unwrap()
                        .count(),
                    width
                );
                for (x, spatial) in MerrillSpatialRow::new(&tables, y, height, width)
                    .unwrap()
                    .enumerate()
                {
                    for c in 0..3 {
                        let raw =
                            (((x * 191 + y * 131 + c * 977) % 8192) as i32 - 128) as f32 * 0.75;
                        let black = [128.125, 255.3, -64.75][c];
                        let gain = [2.21234567, 0.98765432, 4.4433322][c];
                        pixels[y * stride + 3 * x + c] =
                            ((f64::from(raw) - black) * (gain * spatial[c])) as f32;
                    }
                }
            }
            assert_eq!(float_hash(pixels), expected, "width {width}");
        }
        assert!(MerrillSpatialRow::new(&tables, 0, 1, 9).is_err());
        assert!(MerrillSpatialRow::new(&tables, 13, 13, 9).is_err());
        assert!(MerrillSpatialRow::new(&tables, 0, 13, 0).is_err());
        let broken = std::array::from_fn(|_| MerrillSpatialTable {
            rows: 2,
            cols: 3,
            values: vec![1.0; 5],
        });
        assert!(MerrillSpatialRow::new(&broken, 0, 13, 9).is_err());
    }

    #[test]
    fn merrill_spatial_real_camf_matches_native_tables() {
        // The supplied aperture here is explicitly CAMF CaptureAperture. This
        // checks table math, independently of managed SPP's aperture selection.
        let root = std::env::var_os("X3F_TEST_FILES")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| {
                std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../x3f_test_files")
            });
        let mut checked = 0;
        for line in include_str!("../tests/fixtures/merrill-spatial-native.tsv")
            .lines()
            .filter(|line| !line.starts_with('#'))
        {
            let fields: Vec<_> = line.split('\t').collect();
            assert_eq!(fields.len(), 4);
            let name = fields[0];
            let expected: [u64; 3] =
                std::array::from_fn(|c| u64::from_str_radix(fields[c + 1], 16).unwrap());
            let path = root.join(name);
            if !path.exists() {
                continue;
            }
            let path = std::ffi::CString::new(path.to_str().unwrap()).unwrap();
            unsafe {
                let file = libc::fopen(path.as_ptr(), c"rb".as_ptr());
                assert!(!file.is_null());
                let x3f = x3f_new_from_file(file.cast());
                assert!(!x3f.is_null());
                assert_eq!(x3f_load_data(x3f, x3f_get_camf(x3f)), x3f_return_e_X3F_OK);
                let mut aperture = 0.0;
                assert_ne!(
                    x3f_get_camf_float(x3f, c"CaptureAperture".as_ptr().cast_mut(), &mut aperture),
                    0
                );
                let tables = merrill_spatial_gain(x3f, aperture as f32, Control::none())
                    .unwrap()
                    .unwrap();
                assert_eq!(
                    tables
                        .each_ref()
                        .map(|t| float_hash(t.values.iter().copied())),
                    expected,
                    "{name}"
                );
                assert_eq!(
                    tables.each_ref().map(|t| [t.cols, t.rows]),
                    [[241, 161], [78, 52], [78, 52]]
                );
                x3f_delete(x3f);
                libc::fclose(file);
            }
            checked += 1;
        }
        eprintln!(
            "Merrill native table oracle: {checked}/53 corpus fixtures present at {}",
            root.display()
        );
    }

    #[test]
    fn lens_position_at_infinity_returns_focal_length() {
        // 1/(1/30 - 1/inf) = 30
        assert!((lens_position(30.0, f64::INFINITY) - 30.0).abs() < 1e-9);
    }

    #[test]
    fn parse_hp_props_index_works() {
        assert_eq!(parse_hp_props_index(c"SpatialGainHPProps_3"), Some(3));
        assert_eq!(parse_hp_props_index(c"SpatialGainHPProps_12"), Some(12));
        // Trailing garbage rejected by parse::<u32>.
        assert_eq!(parse_hp_props_index(c"SpatialGainHPProps_3x"), None);
        assert_eq!(parse_hp_props_index(c"SomethingElse_3"), None);
    }

    #[test]
    fn parse_focus_props_works() {
        assert_eq!(
            parse_focus_props(c"SpatialGainsProps_4_INF"),
            Some((4, *b"INF"))
        );
        assert_eq!(
            parse_focus_props(c"SpatialGainsProps_2_MOD"),
            Some((2, *b"MOD"))
        );
        // Trailing chars after 3-char focus → reject.
        assert_eq!(parse_focus_props(c"SpatialGainsProps_4_INFX"), None);
        // Wrong shape.
        assert_eq!(parse_focus_props(c"SpatialGainsProps_INF"), None);
    }

    #[test]
    fn parse_aperture_lenspos_props_works() {
        assert_eq!(
            parse_aperture_lenspos_props(c"SpatialGainsProps_4.0_30.5"),
            Some((4.0, 30.5))
        );
        // C99 sscanf %lf delegates to strtod, which parses "INF" as
        // f64::INFINITY. f64::from_str matches that behaviour, so
        // this case does match — even though in practice the older-
        // CAMF code path that consumes lenspos values from the block
        // name never sees "INF" in the second slot (the INF/MOD
        // tokens only appear in the newer SpatialGain_Fstop path,
        // which uses parse_focus_props instead).
        let parsed = parse_aperture_lenspos_props(c"SpatialGainsProps_2.8_INF").unwrap();
        assert!((parsed.0 - 2.8).abs() < 1e-12);
        assert!(parsed.1.is_infinite() && parsed.1 > 0.0);
        // Garbled input rejected.
        assert!(parse_aperture_lenspos_props(c"SpatialGainsProps_2.8x_INF").is_none());
    }
}
