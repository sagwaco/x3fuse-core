use std::collections::BTreeMap;
use std::ffi::CString;
use std::io::{Cursor, Seek, SeekFrom};
use std::path::PathBuf;

#[path = "color_metrics.rs"]
mod color_metrics;

use super::*;
use crate::output::dng::tiff_writer::TiffWriter;

#[derive(Clone, Copy)]
pub(super) enum Order {
    Little,
    Big,
}

impl Order {
    fn u16(self, data: &[u8]) -> u16 {
        match self {
            Self::Little => u16::from_le_bytes(data.try_into().unwrap()),
            Self::Big => u16::from_be_bytes(data.try_into().unwrap()),
        }
    }

    fn u32(self, data: &[u8]) -> u32 {
        match self {
            Self::Little => u32::from_le_bytes(data.try_into().unwrap()),
            Self::Big => u32::from_be_bytes(data.try_into().unwrap()),
        }
    }
}

pub(super) struct Entry<'a> {
    kind: u16,
    data: &'a [u8],
    order: Order,
}

impl Entry<'_> {
    fn shorts(&self) -> Vec<u16> {
        assert_eq!(self.kind, 3);
        self.data
            .as_chunks::<2>()
            .0
            .iter()
            .map(|v| self.order.u16(v))
            .collect()
    }
    fn text(&self) -> &str {
        assert_eq!(self.kind, 2);
        assert_eq!(self.data.last(), Some(&0));
        std::str::from_utf8(&self.data[..self.data.len() - 1]).unwrap()
    }

    pub(super) fn longs(&self) -> Vec<u32> {
        assert_eq!(self.kind, 4);
        self.data
            .as_chunks::<4>()
            .0
            .iter()
            .map(|v| self.order.u32(v))
            .collect()
    }

    fn floats(&self) -> Vec<f32> {
        assert_eq!(self.kind, 11);
        self.data
            .as_chunks::<4>()
            .0
            .iter()
            .map(|v| f32::from_bits(self.order.u32(v)))
            .collect()
    }

    fn matrix(&self) -> [f32; 9] {
        assert_eq!(self.kind, 10);
        self.data
            .as_chunks::<8>()
            .0
            .iter()
            .map(|v| {
                let n = self.order.u32(&v[..4]) as i32;
                let d = self.order.u32(&v[4..]) as i32;
                assert_ne!(d, 0);
                n as f32 / d as f32
            })
            .collect::<Vec<_>>()
            .try_into()
            .unwrap()
    }

    fn rationals(&self) -> Vec<f64> {
        assert!(self.kind == 5 || self.kind == 10);
        self.data
            .as_chunks::<8>()
            .0
            .iter()
            .map(|v| {
                let (n, d) = (self.order.u32(&v[..4]), self.order.u32(&v[4..]));
                assert_ne!(d, 0);
                if self.kind == 10 {
                    (n as i32) as f64 / (d as i32) as f64
                } else {
                    n as f64 / d as f64
                }
            })
            .collect()
    }

    fn value(&self) -> Value {
        let short = || {
            self.data
                .as_chunks::<2>()
                .0
                .iter()
                .map(|v| self.order.u16(v))
                .collect()
        };
        let long = || {
            self.data
                .as_chunks::<4>()
                .0
                .iter()
                .map(|v| self.order.u32(v))
                .collect()
        };
        let rational = || {
            self.data
                .as_chunks::<8>()
                .0
                .iter()
                .map(|v| (self.order.u32(&v[..4]), self.order.u32(&v[4..])))
                .collect::<Vec<_>>()
        };
        match self.kind {
            1 => Value::Byte(self.data.to_vec()),
            2 => Value::Ascii(CString::new(self.text()).unwrap()),
            3 => Value::Short(short()),
            4 => Value::Long(long()),
            5 => Value::Rational(rational()),
            7 => Value::Undefined(self.data.to_vec()),
            10 => Value::SRational(
                rational()
                    .into_iter()
                    .map(|(n, d)| (n as i32, d as i32))
                    .collect(),
            ),
            11 => Value::Float(self.floats()),
            _ => panic!("unsupported fixture field type {}", self.kind),
        }
    }
}

pub(super) fn directory(
    data: &[u8],
    base: usize,
    relative: usize,
    order: Order,
) -> BTreeMap<u16, Entry<'_>> {
    let start = base.checked_add(relative).unwrap();
    let count = order.u16(&data[start..start + 2]) as usize;
    let end = start.checked_add(2 + count * 12 + 4).unwrap();
    assert!(end <= data.len(), "IFD exceeds file");
    let mut result = BTreeMap::new();
    let mut previous = 0;
    for i in 0..count {
        let entry = &data[start + 2 + i * 12..start + 2 + (i + 1) * 12];
        let tag = order.u16(&entry[..2]);
        assert!(tag > previous, "duplicate or unsorted tag {tag}");
        previous = tag;
        let kind = order.u16(&entry[2..4]);
        let elements = order.u32(&entry[4..8]) as usize;
        let size = match kind {
            1 | 2 | 6 | 7 => 1,
            3 | 8 => 2,
            4 | 9 | 11 => 4,
            5 | 10 | 12 => 8,
            _ => panic!("invalid TIFF field type {kind}"),
        };
        let bytes = elements.checked_mul(size).unwrap();
        let value = if bytes <= 4 {
            &entry[8..8 + bytes]
        } else {
            let offset = order.u32(&entry[8..]) as usize;
            assert!(offset.is_multiple_of(2), "unaligned tag {tag}");
            let offset = base.checked_add(offset).unwrap();
            let value_end = offset.checked_add(bytes).unwrap();
            assert!(
                value_end <= data.len(),
                "payload for tag {tag} exceeds file"
            );
            &data[offset..value_end]
        };
        result.insert(
            tag,
            Entry {
                kind,
                data: value,
                order,
            },
        );
    }
    result
}

pub(super) fn root(data: &[u8]) -> BTreeMap<u16, Entry<'_>> {
    assert_eq!(&data[..4], b"II*\0");
    directory(
        data,
        0,
        Order::Little.u32(&data[4..8]) as usize,
        Order::Little,
    )
}

fn extra(data: &[u8], base: usize) -> BTreeMap<u16, Entry<'_>> {
    let order = match &data[base..base + 2] {
        b"MM" => Order::Big,
        b"II" => Order::Little,
        _ => panic!("invalid camera-profile header"),
    };
    assert_eq!(order.u16(&data[base + 2..base + 4]), 0x4352);
    directory(
        data,
        base,
        order.u32(&data[base + 4..base + 8]) as usize,
        order,
    )
}

fn assert_curve(curve: &[f32]) {
    assert!(curve.len() >= 4 && curve.len().is_multiple_of(2));
    assert_eq!(&curve[..2], &[0.0, 0.0]);
    assert_eq!(&curve[curve.len() - 2..], &[1.0, 1.0]);
    assert!(curve
        .iter()
        .all(|v| v.is_finite() && (0.0..=1.0).contains(v)));
    assert!(curve
        .as_chunks::<2>()
        .0
        .iter()
        .collect::<Vec<_>>()
        .windows(2)
        .all(|p| p[0][0] < p[1][0] && p[0][1] <= p[1][1]));
}

fn assert_raw_storage_unchanged(source: &[u8], output: &[u8]) {
    let primary = root(output);
    let raw_offset = primary[&t::SUB_IFDS].longs()[0] as usize;
    assert_eq!(root(source)[&t::SUB_IFDS].longs(), [raw_offset as u32]);
    let raw = directory(output, 0, raw_offset, Order::Little);
    let old_raw = directory(source, 0, raw_offset, Order::Little);
    assert_eq!(
        raw.keys().collect::<Vec<_>>(),
        old_raw.keys().collect::<Vec<_>>()
    );
    for (tag, entry) in &raw {
        assert_eq!(entry.data, old_raw[tag].data);
    }
    for (offset, count) in raw[&t::STRIP_OFFSETS]
        .longs()
        .into_iter()
        .zip(raw[&t::STRIP_BYTE_COUNTS].longs())
    {
        let range = offset as usize..(offset + count) as usize;
        assert_eq!(&source[range.clone()], &output[range]);
    }
}

fn little_classic_profile(
    name: &str,
    color: [f32; 9],
    forward: [f32; 9],
    curve: &[f32],
) -> Vec<u8> {
    let source = build_mmcr_profile(name, color, forward, Some(curve), None);
    let entries = extra(&source, 0);
    let mut output = source.clone();
    output[..4].copy_from_slice(b"IIRC");
    output[4..8].copy_from_slice(&8u32.to_le_bytes());
    output[8..10].copy_from_slice(&(entries.len() as u16).to_le_bytes());
    for (index, (&tag, entry)) in entries.iter().enumerate() {
        let pos = 10 + index * 12;
        let count = Order::Big.u32(&source[pos + 4..pos + 8]);
        output[pos..pos + 2].copy_from_slice(&tag.to_le_bytes());
        output[pos + 2..pos + 4].copy_from_slice(&entry.kind.to_le_bytes());
        output[pos + 4..pos + 8].copy_from_slice(&count.to_le_bytes());
        let start = if entry.data.len() <= 4 {
            pos + 8
        } else {
            let offset = Order::Big.u32(&source[pos + 8..pos + 12]);
            output[pos + 8..pos + 12].copy_from_slice(&offset.to_le_bytes());
            offset as usize
        };
        let size = match entry.kind {
            1 | 2 | 7 => 1,
            3 => 2,
            4 | 5 | 10 | 11 => 4,
            _ => panic!("unsupported diagnostic profile type {}", entry.kind),
        };
        for (index, chunk) in entry.data.chunks_exact(size).enumerate() {
            let target = &mut output[start + index * size..start + (index + 1) * size];
            match size {
                1 => target.copy_from_slice(chunk),
                2 => target.copy_from_slice(&Order::Big.u16(chunk).to_le_bytes()),
                4 => target.copy_from_slice(&Order::Big.u32(chunk).to_le_bytes()),
                _ => unreachable!(),
            }
        }
    }
    let parsed = extra(&output, 0);
    assert_eq!(parsed[&t::PROFILE_NAME].text(), name);
    assert_eq!(parsed[&t::COLOR_MATRIX1].matrix(), color);
    assert_eq!(parsed[&t::FORWARD_MATRIX1].matrix(), forward);
    assert_eq!(parsed[&t::PROFILE_TONE_CURVE].floats(), curve);
    output
}

#[test]
#[ignore = "writes a temporary classic-matrix AsShotWhiteXY neutral-cast diagnostic"]
fn write_lightroom_white_xy_cast_diagnostic() {
    let source = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../_local/spp-alignment/chroma-lut-option/pins/sigma_sd1_merrill_15.x3f.dng");
    let source = std::fs::read(source).unwrap();
    let old = root(&source);
    let color = old[&t::COLOR_MATRIX1].matrix();
    let forward = old[&t::FORWARD_MATRIX1].matrix();
    let neutral: [f64; 3] = old[&t::AS_SHOT_NEUTRAL].rationals().try_into().unwrap();
    let white = super::super::rgb_table::multiply(&mat3_inverse(&color.map(f64::from)), neutral);
    assert!(white.iter().all(|v| v.is_finite() && *v > 0.0));
    let sum: f64 = white.iter().sum();
    let xy = [white[0] / sum, white[1] / sum];
    assert!(xy[0] > 0.0 && xy[1] > 0.0 && xy[0] + xy[1] < 1.0);
    let d: [f64; 3] = [1.0 / 1.4, 1.0 / 0.7, 1.0 / 1.4];
    assert!(d.iter().all(|v| v.is_finite() && *v > 0.0));
    let cast = std::array::from_fn(|i| (f64::from(color[i]) * d[i / 3]) as f32);
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis()
        .to_string();
    let name = format!("x3fuse C3 Base {nonce}");
    let controls = [
        (
            format!("x3fuse C3 Identity II {nonce}"),
            color,
            vec![0.0, 0.0, 1.0, 1.0],
            "identity",
        ),
        (
            format!("x3fuse C3 Tone Bright II {nonce}"),
            color,
            vec![0.0, 0.0, 0.05, 0.8, 1.0, 1.0],
            "tone-bright",
        ),
        (
            format!("x3fuse C3 Cast Magenta II {nonce}"),
            cast,
            vec![0.0, 0.0, 1.0, 1.0],
            "magenta-cast",
        ),
    ];
    let destination = PathBuf::from("/private/tmp/x3f-dng-lightroom-diagnostic");
    std::fs::create_dir_all(&destination).unwrap();
    let path = destination.join(format!("C3-white-xy-cast-{nonce}.dng"));
    let mut output = Cursor::new(source.clone());
    let mut writer = TiffWriter::new(&mut output).unwrap();
    writer.writer_mut().seek(SeekFrom::End(0)).unwrap();
    let mut offsets = Vec::new();
    let mut manifest = format!(
        "file\tprofile\tplacement\tcontrol\n{}\t{name}\tprimary\tidentity\n",
        path.display()
    );
    for (name, matrix, curve, control) in &controls {
        let matrix = matrix.map(|v| (f64::from(v) * 10_000.0).round() as f32 / 10_000.0);
        let blob = little_classic_profile(name, matrix, forward, curve);
        offsets.push(writer.write_data(&blob).unwrap());
        manifest.push_str(&format!("{}\t{name}\textra\t{control}\n", path.display()));
    }
    let mut primary = DirectoryWriter::new();
    for (&tag, entry) in &old {
        if !matches!(
            tag,
            t::AS_SHOT_NEUTRAL
                | t::AS_SHOT_WHITE_XY
                | t::EXTRA_CAMERA_PROFILES
                | t::AS_SHOT_PROFILE_NAME
                | t::PROFILE_NAME
                | t::PROFILE_TONE_CURVE
                | t::PROFILE_HUE_SAT_MAP_DIMS
                | t::PROFILE_HUE_SAT_MAP_DATA1
                | t::PROFILE_HUE_SAT_MAP_ENCODING
                | t::RGB_TABLES
                | t::BASELINE_EXPOSURE_OFFSET
        ) {
            primary.add(tag, entry.value());
        }
    }
    add_profile_tags(&mut primary, &name, color, forward);
    primary.add(t::CALIBRATION_ILLUMINANT1, Value::Short(vec![21]));
    primary.add(
        t::PROFILE_TONE_CURVE,
        Value::Float(vec![0.0, 0.0, 1.0, 1.0]),
    );
    primary.add(
        t::AS_SHOT_PROFILE_NAME,
        Value::Ascii(CString::new(name).unwrap()),
    );
    primary.add(
        t::AS_SHOT_WHITE_XY,
        Value::Rational(
            xy.map(|v| ((v * 1_000_000_000.0).round() as u32, 1_000_000_000))
                .to_vec(),
        ),
    );
    primary.add(
        t::BASELINE_EXPOSURE_OFFSET,
        srational_from_floats(&[0.0], 10_000),
    );
    primary.add(t::EXTRA_CAMERA_PROFILES, Value::Long(offsets));
    for tag in [t::DNG_VERSION, t::DNG_BACKWARD_VERSION] {
        primary.add(tag, Value::Byte(t::DNG_VERSION_1_4_0_0.to_vec()));
    }
    let offset = primary.build(&mut writer).unwrap();
    writer.finalize(offset).unwrap();
    let data = output.into_inner();
    assert_raw_storage_unchanged(&source, &data);
    let primary = root(&data);
    assert!(!primary.contains_key(&t::AS_SHOT_NEUTRAL));
    assert_eq!(primary[&t::AS_SHOT_WHITE_XY].rationals().len(), 2);
    for tag in [t::COLOR_MATRIX1, t::FORWARD_MATRIX1, t::BASELINE_EXPOSURE] {
        assert_eq!(primary[&tag].data, old[&tag].data);
    }
    for (offset, (name, matrix, curve, _)) in primary[&t::EXTRA_CAMERA_PROFILES]
        .longs()
        .into_iter()
        .zip(&controls)
    {
        assert_eq!(&data[offset as usize..offset as usize + 4], b"IIRC");
        let profile = extra(&data, offset as usize);
        assert_eq!(profile[&t::PROFILE_NAME].text(), name);
        assert!(!profile.contains_key(&t::RGB_TABLES));
        assert_eq!(profile[&t::FORWARD_MATRIX1].matrix(), forward);
        assert_eq!(profile[&t::PROFILE_TONE_CURVE].floats(), *curve);
        assert_curve(curve);
        for (actual, expected) in profile[&t::COLOR_MATRIX1].matrix().into_iter().zip(matrix) {
            assert!((actual - expected).abs() <= 0.000051);
        }
    }
    std::fs::write(&path, data).unwrap();
    std::fs::write(destination.join("profiles-C3.tsv"), manifest).unwrap();
    eprintln!(
        "{}: xy={xy:?}, positive d={d:?}; raw-strip, classic IIRC and matrix checks passed",
        path.display()
    );
}

fn assert_mode_profiles(data: &[u8], selected: &str, modes: &[x3f_sys::ColorMode]) {
    let primary = root(data);
    assert_eq!(primary[&t::PROFILE_NAME].text(), selected);
    assert_eq!(
        primary[&t::AS_SHOT_PROFILE_NAME].text(),
        primary[&t::PROFILE_NAME].text()
    );
    assert_eq!(primary[&t::DNG_VERSION].data, [1, 4, 0, 0]);
    assert_eq!(primary[&t::DNG_BACKWARD_VERSION].data, [1, 4, 0, 0]);
    assert!(!primary.contains_key(&t::AS_SHOT_NEUTRAL));
    assert_eq!(primary[&t::AS_SHOT_WHITE_XY].rationals().len(), 2);
    let mut profiles = vec![primary];
    let offsets = profiles[0][&t::EXTRA_CAMERA_PROFILES].longs();
    assert!(offsets.windows(2).all(|w| w[0] < w[1]));
    profiles.extend(offsets.into_iter().map(|o| extra(data, o as usize)));
    let mut names = Vec::new();
    for profile in &profiles {
        let name = profile[&t::PROFILE_NAME].text().to_owned();
        assert!(!profile.contains_key(&t::RGB_TABLES));
        assert!(!names.contains(&name), "duplicate profile {name}");
        names.push(name.clone());
        if modes.iter().any(|mode| mode.name() == name) {
            assert_eq!(profile[&t::CALIBRATION_ILLUMINANT1].shorts(), [21]);
            assert_eq!(profile[&t::PROFILE_LOOK_TABLE_DIMS].longs(), [72, 17, 65]);
            assert_eq!(profile[&t::PROFILE_LOOK_TABLE_ENCODING].longs(), [1]);
            let map = profile[&t::PROFILE_LOOK_TABLE_DATA].floats();
            assert_eq!(map.len(), 72 * 17 * 65 * 3);
            for (i, p) in map.as_chunks::<3>().0.iter().enumerate() {
                assert!(p.iter().all(|v| v.is_finite()));
                assert!((-180.0..=180.0).contains(&p[0]));
                assert!((0.0..=8.0).contains(&p[1]));
                assert!(p[2] >= 0.0);
                if i % 17 == 0 {
                    assert_eq!(p[2], 1.0);
                }
            }
            let curve = profile[&t::PROFILE_TONE_CURVE].floats();
            assert_eq!(curve.len(), 8194);
            assert_eq!(&curve[..2], &[0.0, 0.0]);
            assert_eq!(&curve[curve.len() - 2..], &[1.0, 1.0]);
            assert!(curve
                .iter()
                .all(|v| v.is_finite() && (0.0..=1.0).contains(v)));
            assert!(curve
                .as_chunks::<2>()
                .0
                .windows(2)
                .all(|p| p[1][0] > p[0][0] && p[1][1] >= p[0][1]));
            let total = profiles[0][&t::BASELINE_EXPOSURE].rationals()[0]
                + profile[&t::BASELINE_EXPOSURE_OFFSET].rationals()[0];
            assert_eq!(total, 0.0, "serialized exposure must match the fit");
            let (model, _, _) = classic_model(profile, &profiles[0]);
            assert!(model.white.into_iter().all(|v| (0.001..=1.0).contains(&v)));
        } else {
            assert!(!profile.contains_key(&t::PROFILE_LOOK_TABLE_DATA));
        }
    }
    let mut expected: Vec<_> = modes
        .iter()
        .map(|mode| mode.name())
        .chain(PROFILES[1..].iter().map(|profile| profile.name))
        .collect();
    names.sort();
    expected.sort();
    assert_eq!(names, expected);
}

fn classic_model(
    profile: &BTreeMap<u16, Entry<'_>>,
    primary: &BTreeMap<u16, Entry<'_>>,
) -> (
    super::super::classic_profile::Matrices,
    super::super::classic_profile::ClassicFit,
    f64,
) {
    use super::super::classic_profile::{
        sdk_color, sdk_forward, xy_xyz, ClassicFit, Matrices, ToneCurve,
    };
    let xy = primary[&t::AS_SHOT_WHITE_XY].rationals();
    let xy = [xy[0], xy[1]].map(|v| (v * 1_000_000.0).round() / 1_000_000.0);
    let color = sdk_color(profile[&t::COLOR_MATRIX1].rationals().try_into().unwrap());
    let forward =
        sdk_forward(profile[&t::FORWARD_MATRIX1].rationals().try_into().unwrap()).unwrap();
    let white = multiply(&color, xy_xyz(xy));
    let maximum = white.into_iter().fold(0.0, f64::max);
    let white = white.map(|v| (v / maximum).clamp(0.001, 1.0));
    let camera_to_prophoto = mat3_mul(
        &xyz_d50_to_prophoto(),
        &mat3_mul(&forward, &mat3_diag(&white.map(|v| 1.0 / v))),
    );
    let curve = ToneCurve::from_knots(profile[&t::PROFILE_TONE_CURVE].floats()).unwrap();
    let gray_endpoint = f64::from(*curve.knots.last().unwrap());
    let map = profile[&t::PROFILE_LOOK_TABLE_DATA].floats();
    let fit = ClassicFit {
        curve,
        map,
        capped: 0,
        gray_endpoint,
    };
    let exposure = primary
        .get(&t::BASELINE_EXPOSURE)
        .map_or(0.0, |e| e.rationals()[0])
        + profile[&t::BASELINE_EXPOSURE_OFFSET].rationals()[0];
    assert!(exposure >= 0.0);
    (
        Matrices {
            color,
            forward,
            white,
            camera_to_prophoto,
            diagonal: [1.0; 3],
        },
        fit,
        exposure,
    )
}

#[test]
fn iirc_classic_look_table_has_independent_payloads_and_offsets() {
    let matrix = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let map: Vec<_> = (0..65)
        .flat_map(|v| {
            (0..72).flat_map(move |h| {
                (0..17).flat_map(move |s| [h as f32, s as f32 / 16.0, 1.0 + v as f32 / 64.0])
            })
        })
        .collect();
    let profile = build_profile_blob(
        "Classic payload control",
        matrix,
        matrix,
        Some(&[0.0, 0.0, 1.0, 1.0]),
        None,
        Some(ProfileLook::Classic(&map)),
        -3.0,
    );
    let parsed = extra(&profile, 0);
    assert_eq!(&profile[..2], b"II");
    assert_eq!(Order::Little.u16(&profile[2..4]), 0x4352);
    assert_eq!(parsed[&t::PROFILE_LOOK_TABLE_DIMS].longs(), [72, 17, 65]);
    assert_eq!(parsed[&t::PROFILE_LOOK_TABLE_ENCODING].longs(), [1]);
    assert_eq!(parsed[&t::PROFILE_LOOK_TABLE_DATA].floats(), map);
    assert_eq!(
        parsed[&t::PROFILE_TONE_CURVE].floats(),
        [0.0, 0.0, 1.0, 1.0]
    );
    assert_eq!(parsed[&t::BASELINE_EXPOSURE_OFFSET].rationals(), [-3.0]);
    assert_eq!(parsed[&t::CALIBRATION_ILLUMINANT1].shorts(), [21]);
    assert!(!parsed.contains_key(&t::RGB_TABLES));
}

fn decoded_raw(data: &[u8]) -> Vec<u16> {
    let primary = root(data);
    let raw = directory(
        data,
        0,
        primary[&t::SUB_IFDS].longs()[0] as usize,
        Order::Little,
    );
    let compression = raw[&t::COMPRESSION].shorts()[0];
    let mut decoded = Vec::new();
    for (offset, count) in raw[&t::STRIP_OFFSETS]
        .longs()
        .into_iter()
        .zip(raw[&t::STRIP_BYTE_COUNTS].longs())
    {
        let strip = &data[offset as usize..(offset + count) as usize];
        match compression {
            1 => decoded.extend(
                strip
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|v| u16::from_le_bytes(*v)),
            ),
            7 => decoded.extend(super::super::ljpeg::tests::decode(strip).0),
            _ => panic!("unexpected raw compression {compression}"),
        }
    }
    assert_eq!(
        decoded.len(),
        raw[&t::IMAGE_WIDTH].longs()[0] as usize * raw[&t::IMAGE_LENGTH].longs()[0] as usize * 3
    );
    decoded
}

fn assert_raw_equal(left: &[u8], right: &[u8]) {
    assert_eq!(
        decoded_raw(left),
        decoded_raw(right),
        "decoded camera samples changed with selected mode"
    );
    let a = root(left);
    let b = root(right);
    let mut tags = vec![t::BASELINE_EXPOSURE, t::LINEAR_RESPONSE_LIMIT];
    for tag in [t::AS_SHOT_NEUTRAL, t::AS_SHOT_WHITE_XY] {
        if a.contains_key(&tag) && b.contains_key(&tag) {
            tags.push(tag);
        }
    }
    for tag in tags {
        assert_eq!(
            a.get(&tag).map(|e| e.data),
            b.get(&tag).map(|e| e.data),
            "calibration or exposure tag {tag} changed"
        );
    }
    let a = directory(left, 0, a[&t::SUB_IFDS].longs()[0] as usize, Order::Little);
    let b = directory(right, 0, b[&t::SUB_IFDS].longs()[0] as usize, Order::Little);
    for tag in [
        t::BLACK_LEVEL,
        t::WHITE_LEVEL,
        t::ACTIVE_AREA,
        t::DEFAULT_CROP_ORIGIN,
        t::DEFAULT_CROP_SIZE,
        t::DEFAULT_USER_CROP,
    ] {
        assert_eq!(
            a.get(&tag).map(|e| e.data),
            b.get(&tag).map(|e| e.data),
            "raw level or crop tag {tag} changed"
        );
    }
}

fn loaded_reader(path: &std::path::Path) -> Reader {
    let mut reader = Reader::open(path).unwrap();
    reader.load_property_list().unwrap();
    reader.load_camf().unwrap();
    reader.load_raw().unwrap();
    reader
}

/// SDQH5085's Auto white is outside the classic profile model, so the
/// audits use its Sunlight preset. Other files use the as-shot WB.
fn audit_wb(file: &str, reader: &Reader) -> String {
    if file == "SDQH5085.X3F" {
        "Sunlight".into()
    } else {
        reader.dng_default_wb()
    }
}

/// Load a supplemental corpus file, or report a skip when it is absent.
fn optional_reader(corpus: &std::path::Path, name: &str) -> Option<Reader> {
    let path = corpus.join(name);
    if !path.is_file() {
        eprintln!("skip: missing {}", path.display());
        return None;
    }
    Some(loaded_reader(&path))
}

#[test]
#[ignore = "checks legacy profile camera whites after the WhiteXY row correction"]
fn legacy_profiles_keep_the_as_shot_camera_white() {
    use super::super::classic_profile::{sdk_color, serialized, xy_xyz};
    let corpus = PathBuf::from(std::env::var_os("X3F_TEST_FILES").expect("X3F_TEST_FILES"));
    for file in [
        "sigma_sd1_merrill_15.x3f",
        "_SDI8040.X3F",
        "_SDI8284.X3F",
        "DP0Q0010.X3F",
        "SDIM0002.X3F",
        "SDQH5085.X3F",
        "_DQH1429.X3F",
    ] {
        let Some(reader) = optional_reader(&corpus, file) else {
            continue;
        };
        let wb = audit_wb(file, &reader);
        let calibration = ColorCalibration::new(&reader, &wb).unwrap();
        let color0 = sdk_color(serialized(
            calibration
                .fold_color_matrix(&mat3_inverse(&reader.dng_bmt_to_xyz(Some(&wb)).unwrap())),
        ));
        let profiles = PreparedColorModes::resolve(
            &reader,
            &wb,
            x3f_sys::ColorMode::Standard,
            &mut Vec::new(),
            x3f_sys::Control::none(),
        )
        .unwrap()
        .finish(
            &reader,
            &wb,
            &calibration,
            1.0,
            &mut Vec::new(),
            x3f_sys::Control::none(),
        )
        .unwrap_or_else(|e| panic!("{file}: {e}"));
        let data = serialized_profile_metadata(&reader, &calibration, &profiles, 1.0);
        let primary = root(&data);
        let xy = profiles
            .white_xy
            .map(|v| (v * 1_000_000.0).round() / 1_000_000.0);
        let wanted = multiply(&color0, xy_xyz(xy));
        let max = wanted.into_iter().fold(0.0, f64::max);
        let wanted = wanted.map(|v| v / max);
        // Each stored ColorMatrix entry is rounded to 1/10000, then the SDK
        // read rounds it again. Two half steps per entry move a component
        // by at most 1e-4 * sum|XYZ|; max normalization can double that.
        let budget = 2.0e-4 * xy_xyz(xy).iter().map(|v| v.abs()).sum::<f64>() / max;
        // Keep the measured limit for the original fixtures.
        let limit = if ["SDIM0002.X3F", "SDQH5085.X3F", "_DQH1429.X3F"].contains(&file) {
            budget
        } else {
            0.0001
        };
        let mut count = 0;
        let mut error = 0.0_f64;
        for off in primary[&t::EXTRA_CAMERA_PROFILES].longs() {
            let profile = extra(&data, off as usize);
            if profile.contains_key(&t::PROFILE_LOOK_TABLE_DATA) {
                assert_eq!(&data[off as usize..off as usize + 2], b"II");
                continue;
            }
            assert_eq!(&data[off as usize..off as usize + 2], b"MM");
            let color = sdk_color(profile[&t::COLOR_MATRIX1].rationals().try_into().unwrap());
            let actual = multiply(&color, xy_xyz(xy));
            let max = actual.into_iter().fold(0.0, f64::max);
            let actual = actual.map(|v| v / max);
            error = error.max(
                actual
                    .into_iter()
                    .zip(wanted)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0, f64::max),
            );
            count += 1;
        }
        assert_eq!(count, 6);
        assert!(
            error <= limit,
            "{file} legacy white error={error} limit={limit}"
        );
        eprintln!("{file}: all 6 legacy camera whites match physical as-shot white within {error:.8} (rounding budget {budget:.8})");
    }
}

#[test]
#[ignore = "audits classic native matrix coordinates on the four camera fixtures"]
fn classic_color_mode_matrix_audit() {
    use super::super::classic_profile::{sdk_color, sdk_forward, serialized, white_xy, Matrices};
    use super::super::rgb_table::srgb_to_xyz_d50;
    let corpus = PathBuf::from(std::env::var_os("X3F_TEST_FILES").expect("X3F_TEST_FILES"));
    let destination = PathBuf::from("/private/tmp/x3f-dng-classic");
    std::fs::create_dir_all(&destination).unwrap();
    let mut report = String::from("file\tmode\tbasis\td\tcamera_white\tpin_distance\tquantized_relative_error\tforward_difference\n");
    for file in [
        "sigma_sd1_merrill_15.x3f",
        "_SDI8040.X3F",
        "_SDI8284.X3F",
        "DP0Q0010.X3F",
        "SDIM0002.X3F",
        "SDQH5085.X3F",
        "_DQH1429.X3F",
    ] {
        let Some(reader) = optional_reader(&corpus, file) else {
            continue;
        };
        let wb = audit_wb(file, &reader);
        let calibration = ColorCalibration::new(&reader, &wb).unwrap();
        let prepared = PreparedColorModes::resolve(
            &reader,
            &wb,
            x3f_sys::ColorMode::Standard,
            &mut Vec::new(),
            x3f_sys::Control::none(),
        )
        .unwrap();
        let bmt = reader.dng_bmt_to_xyz(Some(&wb)).unwrap();
        let color0 = sdk_color(serialized(
            calibration.fold_color_matrix(&mat3_inverse(&bmt)),
        ));
        let forward0 = sdk_forward(serialized(
            sdk_forward(mat3_mul(&bradford_d65_to_d50(), &bmt)).unwrap(),
        ))
        .unwrap();
        let neutral = calibration
            .neutral
            .map(|v| (v * 10_000.0).round().max(1.0) / 10_000.0);
        let maximum = neutral.into_iter().fold(0.0, f64::max);
        let neutral = neutral.map(|v| (v / maximum).clamp(0.001, 1.0));
        let xy = white_xy(color0, neutral)
            .unwrap_or_else(|e| panic!("{file}: {e}; CM0={color0:?} neutral={neutral:?}"));
        let basis = mat3_mul(
            &srgb_to_xyz_d50(),
            &super::super::rgb_table::quantized_output(prepared.native.output_to_srgb),
        );
        let standard = prepared
            .modes
            .iter()
            .find(|p| p.mode == x3f_sys::ColorMode::Standard)
            .unwrap();
        let (counts0, _) = prepared.native.render_matrix(standard);
        let count_scale = multiply(&mat3_mul(&basis, &counts0), neutral)[1];
        let native_standard =
            NativeRendering::new(&prepared.native, standard, x3f_sys::Control::none()).unwrap();
        let mut tint = Vec::new();
        for i in 1..=1024 {
            let rgb = multiply(
                &native_standard.prophoto_to_srgb(),
                native_standard.sensor_to_prophoto_exact(neutral.map(|v| v * i as f64 / 1024.0)),
            );
            let lab = color_metrics::lab(rgb);
            tint.push(color_metrics::delta_e(lab, [lab[0], 0.0, 0.0]));
        }
        eprintln!("{file}: Standard native physical-neutral tint dE2000 p50/p95/p99/max {:?}; count_scale={count_scale:.8}; O={:?}", color_metrics::percentiles(tint), prepared.native.output_to_srgb);
        for params in &prepared.modes {
            let cc = prepared.native.sensor_to_srgb;
            let creative = params
                .creative_matrix
                .unwrap_or([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
            let relative = mat3_mul(&cc, &mat3_mul(&creative, &mat3_inverse(&cc)));
            let relative = mat3_mul(&basis, &mat3_mul(&relative, &mat3_inverse(&basis)));
            let relative = mat3_mul(&relative, &forward0);
            let absolute =
                Matrices::new(&prepared.native, params, color0, neutral, xy, count_scale)
                    .unwrap_or_else(|e| {
                        let (counts, offset) = prepared.native.render_matrix(params);
                        panic!(
                            "{file} {}: {e}; counts={counts:?} offset={offset:?}",
                            params.mode
                        )
                    });
            let relative = Matrices::from_absolute(relative, color0, xy).unwrap();
            let (counts, _) = prepared.native.render_matrix(params);
            let wanted = mat3_mul(&xyz_d50_to_prophoto(), &mat3_mul(&basis, &counts))
                .map(|v| v / count_scale);
            for (label, matrices) in [("absolute", &absolute), ("relative", &relative)] {
                let predicted = matrices.camera_to_prophoto;
                let maximum = matrices
                    .diagonal
                    .into_iter()
                    .zip(neutral)
                    .map(|(d, n)| d * n)
                    .fold(0.0, f64::max);
                let error = predicted
                    .iter()
                    .zip(wanted)
                    .map(|(a, b)| (a / maximum - b).abs())
                    .fold(0.0, f64::max)
                    / wanted.iter().map(|v| v.abs()).fold(0.0, f64::max);
                let forward_difference = matrices
                    .forward
                    .iter()
                    .zip(forward0)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0, f64::max);
                let pin_distance = matrices.white.into_iter().fold(f64::INFINITY, f64::min) - 0.001;
                report.push_str(&format!("{file}\t{}\t{label}\t{:?}\t{:?}\t{pin_distance:.8}\t{error:.8}\t{forward_difference:.8}\n", params.mode.name(), matrices.diagonal, matrices.white));
                assert!(pin_distance > 0.0);
                assert!(matrices.color.iter().all(|v| v.is_finite()));
                if label == "absolute" {
                    assert!(
                        error < 0.001,
                        "{file} {} serialization error {error}",
                        params.mode
                    );
                }
            }
        }
    }
    std::fs::write(destination.join("matrix-audit.tsv"), report).unwrap();
}

#[test]
#[ignore = "measures classic curve and LookTable residuals against the native renderer"]
fn classic_color_mode_residual_audit() {
    use super::super::classic_profile::{sdk_color, serialized, white_xy, ClassicFit, Matrices};
    use super::super::rgb_table::{quantized_output, srgb_to_xyz_d50};
    let corpus = PathBuf::from(std::env::var_os("X3F_TEST_FILES").expect("X3F_TEST_FILES"));
    let destination = PathBuf::from("/private/tmp/x3f-dng-classic");
    std::fs::create_dir_all(&destination).unwrap();
    let mut report =
        String::from("file\tmode\tclass\tsamples\tp50\tp95\tp99\tmax\tcap_nodes\tgray_endpoint\n");
    let mut wb_report = String::from("file\tmode\tpreset\tclass\tsamples\tp50\tp95\tp99\tmax\n");
    for file in [
        "sigma_sd1_merrill_15.x3f",
        "_SDI8284.X3F",
        "_SDI8040.X3F",
        "DP0Q0010.X3F",
        "SDIM0002.X3F",
        "SDQH5085.X3F",
        "_DQH1429.X3F",
    ] {
        let Some(reader) = optional_reader(&corpus, file) else {
            continue;
        };
        let wb = audit_wb(file, &reader);
        let calibration = ColorCalibration::new(&reader, &wb).unwrap();
        let prepared = PreparedColorModes::resolve(
            &reader,
            &wb,
            x3f_sys::ColorMode::Standard,
            &mut Vec::new(),
            x3f_sys::Control::none(),
        )
        .unwrap();
        let bmt = reader.dng_bmt_to_xyz(Some(&wb)).unwrap();
        let color0 = sdk_color(serialized(
            calibration.fold_color_matrix(&mat3_inverse(&bmt)),
        ));
        let neutral = calibration
            .neutral
            .map(|v| (v * 10_000.0).round().max(1.0) / 10_000.0);
        let maximum = neutral.into_iter().fold(0.0, f64::max);
        let neutral = neutral.map(|v| (v / maximum).clamp(0.001, 1.0));
        let xy = white_xy(color0, neutral)
            .unwrap_or_else(|e| panic!("{file}: {e}; CM0={color0:?} neutral={neutral:?}"));
        let standard = prepared
            .modes
            .iter()
            .find(|p| p.mode == x3f_sys::ColorMode::Standard)
            .unwrap();
        let (counts0, _) = prepared.native.render_matrix(standard);
        let basis = mat3_mul(
            &srgb_to_xyz_d50(),
            &quantized_output(prepared.native.output_to_srgb),
        );
        let count_scale = multiply(&mat3_mul(&basis, &counts0), neutral)[1];
        let mut image = reader
            .get_image(&crate::ProcessOptions {
                color_encoding: crate::ColorEncoding::None,
                denoise_intensity: 0,
                crop: false,
                wb: Some(wb),
                ..Default::default()
            })
            .unwrap();
        super::super::equalize_levels_controlled(&mut image, x3f_sys::Control::none()).unwrap();
        let (top, left, rows, columns) = super::super::active_area_crop(&reader, &image)
            .unwrap_or((0, 0, image.rows, image.columns));
        let samples: Vec<[f64; 3]> = (0..8192)
            .map(|i| {
                let pixel = i as u64 * u64::from(rows) * u64::from(columns) / 8192;
                let (row, col) = (
                    top + (pixel / u64::from(columns)) as u32,
                    left + (pixel % u64::from(columns)) as u32,
                );
                std::array::from_fn(|c| f64::from(image.sample(row, col, c as u32)) / 65535.0)
            })
            .collect();
        let probe = loaded_reader(&corpus.join(file));
        let presets: Vec<_> = ["Sunlight", "Overcast", "Incandescent"]
            .into_iter()
            .map(|preset| {
                let physical = ColorCalibration::new(&probe, preset).unwrap();
                let neutral = physical
                    .neutral
                    .map(|v| (v * 10_000.0).round().max(1.0) / 10_000.0);
                let xy = white_xy(color0, neutral)
                    .unwrap()
                    .map(|v| (v * 1_000_000.0).round() / 1_000_000.0);
                let name = CString::new(preset).unwrap();
                let calibration = unsafe {
                    x3f_sys::color_mode_calibration(
                        probe.x3f.as_ptr(),
                        name.as_ptr().cast_mut(),
                        x3f_sys::Control::none(),
                    )
                }
                .unwrap();
                (preset, xy, calibration)
            })
            .collect();
        for params in &prepared.modes {
            let matrices =
                Matrices::new(&prepared.native, params, color0, neutral, xy, count_scale)
                    .unwrap_or_else(|e| {
                        let (counts, offset) = prepared.native.render_matrix(params);
                        panic!(
                            "{file} {}: {e}; counts={counts:?} offset={offset:?}",
                            params.mode
                        )
                    });
            let native =
                NativeRendering::new(&prepared.native, params, x3f_sys::Control::none()).unwrap();
            let fit = ClassicFit::new(
                &matrices,
                &native,
                image.dng_highlight_scale,
                0.0,
                x3f_sys::Control::none(),
            )
            .unwrap();
            assert_eq!(fit.curve.knots.len(), 8194);
            let mut classes: BTreeMap<&str, Vec<f64>> = BTreeMap::new();
            for i in 0..8192 {
                let pixel = i as u64 * u64::from(rows) * u64::from(columns) / 8192;
                let (row, col) = (
                    top + (pixel / u64::from(columns)) as u32,
                    left + (pixel % u64::from(columns)) as u32,
                );
                let sensor =
                    std::array::from_fn(|c| f64::from(image.sample(row, col, c as u32)) / 65535.0);
                let unclipped = multiply(&matrices.camera_to_prophoto, sensor);
                let camera_clipped = sensor.iter().zip(matrices.white).any(|(s, w)| *s > w);
                let input = matrices.input(sensor, 0.0);
                let expected =
                    native.sensor_to_prophoto_exact(sensor.map(|v| v * image.dng_highlight_scale));
                let output = fit.apply(input);
                let expected_lab =
                    color_metrics::lab(multiply(&native.prophoto_to_srgb(), expected));
                let output_lab = color_metrics::lab(multiply(&native.prophoto_to_srgb(), output));
                let de = color_metrics::delta_e(expected_lab, output_lab);
                if let Some(hue) = color_metrics::hue_error(expected_lab, output_lab) {
                    classes.entry("hue-degrees").or_default().push(hue);
                }
                classes.entry("all").or_default().push(de);
                classes
                    .entry(if camera_clipped {
                        "camera-white-clipped"
                    } else if unclipped.iter().any(|v| *v < 0.0 || *v > 1.0) {
                        "prophoto-clipped"
                    } else {
                        "in-gamut"
                    })
                    .or_default()
                    .push(de);
                if expected_lab[0] < 10.0 {
                    classes.entry("shadow-L<10").or_default().push(de);
                }
                if expected_lab[1].hypot(expected_lab[2]) < 3.0 {
                    classes.entry("near-neutral-C<3").or_default().push(de);
                }
                if expected_lab[1].hypot(expected_lab[2]) > 30.0 {
                    classes.entry("saturated-C>30").or_default().push(de);
                }
            }
            let mut neutral_errors = Vec::new();
            let mut worst_neutral = (0.0, 0, [0.0; 3], [0.0; 3], false);
            for i in 1..=1024 {
                let sensor = neutral.map(|v| v * i as f64 / 1024.0 / image.dng_highlight_scale);
                let input = matrices.input(sensor, 0.0);
                let expected =
                    native.sensor_to_prophoto_exact(sensor.map(|v| v * image.dng_highlight_scale));
                let output = fit.apply(input);
                let de = color_metrics::delta_e(
                    color_metrics::lab(multiply(&native.prophoto_to_srgb(), expected)),
                    color_metrics::lab(multiply(&native.prophoto_to_srgb(), output)),
                );
                neutral_errors.push(de);
                if de > worst_neutral.0 {
                    worst_neutral = (
                        de,
                        i,
                        input,
                        expected,
                        sensor.iter().zip(matrices.white).any(|(s, w)| *s > w),
                    );
                }
            }
            classes.insert("scene-neutral-ramp", neutral_errors);
            eprintln!("{file} {}: native gray endpoint={:.8}, cap nodes={}, all dE {:?}, in-gamut dE {:?}, neutral {:?}; worst neutral {:?}",params.mode,fit.gray_endpoint,fit.capped,color_metrics::percentiles(classes["all"].clone()),color_metrics::percentiles(classes.get("in-gamut").cloned().unwrap_or_default()),color_metrics::percentiles(classes["scene-neutral-ramp"].clone()),worst_neutral);
            for (class, values) in classes {
                let count = values.len();
                let [p50, p95, p99, max] = color_metrics::percentiles(values);
                report.push_str(&format!("{file}\t{}\t{class}\t{count}\t{p50:.8}\t{p95:.8}\t{p99:.8}\t{max:.8}\t{}\t{:.8}\n",params.mode,fit.capped,fit.gray_endpoint));
            }
            for (preset, xy, calibration) in &presets {
                let target =
                    NativeRendering::new(calibration, params, x3f_sys::Control::none()).unwrap();
                let white = multiply(
                    &sdk_color(matrices.color),
                    super::super::classic_profile::xy_xyz(*xy),
                );
                let maximum = white.into_iter().fold(0.0, f64::max);
                let white = white.map(|v| (v / maximum).clamp(0.001, 1.0));
                let forward = super::super::classic_profile::sdk_forward(matrices.forward).unwrap();
                let edited = Matrices {
                    color: matrices.color,
                    forward: matrices.forward,
                    diagonal: matrices.diagonal,
                    white,
                    camera_to_prophoto: mat3_mul(
                        &xyz_d50_to_prophoto(),
                        &mat3_mul(&forward, &mat3_diag(&white.map(|v| 1.0 / v))),
                    ),
                };
                let mut values: BTreeMap<&str, Vec<f64>> = BTreeMap::new();
                for sensor in samples.iter().step_by(4) {
                    let wanted = target
                        .sensor_to_prophoto_exact(sensor.map(|v| v * image.dng_highlight_scale));
                    let actual = fit.apply(edited.input(*sensor, 0.0));
                    let wanted_lab =
                        color_metrics::lab(multiply(&target.prophoto_to_srgb(), wanted));
                    let actual_lab =
                        color_metrics::lab(multiply(&target.prophoto_to_srgb(), actual));
                    let de = color_metrics::delta_e(wanted_lab, actual_lab);
                    let input = multiply(&edited.camera_to_prophoto, *sensor);
                    let clipped = sensor.iter().zip(white).any(|(s, w)| *s > w)
                        || input.iter().any(|v| *v < 0.0 || *v > 1.0);
                    values
                        .entry(if clipped { "clipped" } else { "in-gamut" })
                        .or_default()
                        .push(de);
                    if wanted_lab[1].hypot(wanted_lab[2]) > 30.0 {
                        values.entry("saturated-C>30").or_default().push(de);
                    }
                    if let Some(hue) = color_metrics::hue_error(wanted_lab, actual_lab) {
                        values.entry("hue-degrees").or_default().push(hue);
                    }
                }
                for i in 1..=1024 {
                    let x = ColorCalibration::new(&probe, preset).unwrap().neutral;
                    let maximum = x.into_iter().fold(0.0, f64::max);
                    let sensor =
                        x.map(|v| v / maximum * i as f64 / 1024.0 / image.dng_highlight_scale);
                    let wanted = target
                        .sensor_to_prophoto_exact(sensor.map(|v| v * image.dng_highlight_scale));
                    let actual = fit.apply(edited.input(sensor, 0.0));
                    values
                        .entry("scene-neutral-ramp")
                        .or_default()
                        .push(color_metrics::delta_e(
                            color_metrics::lab(multiply(&target.prophoto_to_srgb(), wanted)),
                            color_metrics::lab(multiply(&target.prophoto_to_srgb(), actual)),
                        ));
                }
                for (class, values) in values {
                    let count = values.len();
                    let [p50, p95, p99, max] = color_metrics::percentiles(values);
                    wb_report.push_str(&format!("{file}\t{}\t{preset}\t{class}\t{count}\t{p50:.8}\t{p95:.8}\t{p99:.8}\t{max:.8}\n",params.mode));
                }
            }
        }
    }
    std::fs::write(destination.join("residual-audit.tsv"), report).unwrap();
    std::fs::write(destination.join("wb-audit.tsv"), wb_report).unwrap();
}

fn serialized_profile_metadata(
    reader: &Reader,
    calibration: &ColorCalibration,
    profiles: &ColorModeProfiles,
    scale: f64,
) -> Vec<u8> {
    let mut ifd = DirectoryWriter::new();
    profiles.write_default(&mut ifd);
    super::super::add_dng_top_level_tags(
        reader,
        &super::super::exif::CaptureMetadata::from_reader(reader),
        calibration,
        scale,
        &mut ifd,
        Some(profiles.white_xy),
    );
    let mut output = Cursor::new(Vec::new());
    let mut writer = TiffWriter::new(&mut output).unwrap();
    let base = writer.write_data(&profiles.extra_blob).unwrap();
    ifd.add(
        t::EXTRA_CAMERA_PROFILES,
        Value::Long(profiles.extra_offsets.iter().map(|v| base + v).collect()),
    );
    let offset = ifd.build(&mut writer).unwrap();
    writer.finalize(offset).unwrap();
    output.into_inner()
}

#[test]
#[ignore = "checks recovery headroom on the same scene values in the camera corpus"]
fn color_mode_recovery_scale_is_restored_once() {
    let Some(corpus) = std::env::var_os("X3F_TEST_FILES").map(PathBuf::from) else {
        return;
    };
    for source in ["sigma_sd1_merrill_15.x3f", "_SDI8284.X3F"] {
        let reader = loaded_reader(&corpus.join(source));
        let wb = reader.dng_default_wb();
        let calibration = ColorCalibration::new(&reader, &wb).unwrap();
        let prepared = PreparedColorModes::resolve(
            &reader,
            &wb,
            x3f_sys::ColorMode::Standard,
            &mut Vec::new(),
            x3f_sys::Control::none(),
        )
        .unwrap();
        for scale in [1.0, 2.0, 4.0] {
            let copy = PreparedColorModes {
                modes: prepared.modes.clone(),
                selected: prepared.selected,
                monochrome: prepared.monochrome,
                native: prepared.native.clone(),
            };
            let profiles = copy
                .finish(
                    &reader,
                    &wb,
                    &calibration,
                    scale,
                    &mut Vec::new(),
                    x3f_sys::Control::none(),
                )
                .unwrap();
            let bytes = serialized_profile_metadata(&reader, &calibration, &profiles, scale);
            let primary = root(&bytes);
            let offsets = primary[&t::EXTRA_CAMERA_PROFILES].longs();
            let mut entries = vec![primary];
            entries.extend(offsets.into_iter().map(|o| extra(&bytes, o as usize)));
            for profile in &entries {
                if !profile.contains_key(&t::PROFILE_LOOK_TABLE_DATA) {
                    continue;
                }
                let (matrix, fit, exposure) = classic_model(profile, &entries[0]);
                let mode_name = profile[&t::PROFILE_NAME].text();
                let params = prepared
                    .modes
                    .iter()
                    .find(|p| p.mode.name() == mode_name)
                    .unwrap();
                let native =
                    NativeRendering::new(&prepared.native, params, x3f_sys::Control::none())
                        .unwrap();
                let inverse = mat3_inverse(&matrix.camera_to_prophoto);
                for scene in [[0.01, 0.02, 0.03], [0.02, 0.01, 0.04], [0.03, 0.05, 0.01]] {
                    let input = multiply(&matrix.camera_to_prophoto, scene.map(|v| v / scale))
                        .map(|v| v * exposure.exp2());
                    let restored = multiply(&inverse, input).map(|v| v * scale / exposure.exp2());
                    assert!(restored
                        .into_iter()
                        .zip(scene)
                        .all(|(a, b)| (a - b).abs() < 1e-12));
                    assert_eq!(
                        native.sensor_to_srgb16(restored),
                        native.sensor_to_srgb16(scene)
                    );
                }
                for index in [16, 128, 512, 2048, 3072] {
                    let x = f64::from(fit.curve.knots[index * 2]);
                    let sensor = multiply(&inverse, [x; 3]).map(|v| v * scale / exposure.exp2());
                    let target = native.sensor_to_prophoto(sensor);
                    let y = 0.2880 * target[0] + 0.7119 * target[1] + 0.0001 * target[2];
                    assert!(
                        (f64::from(fit.curve.knots[index * 2 + 1]) - y).abs() < 1e-6,
                        "{source}, {mode_name}, h{scale}, knot{index}"
                    );
                }
            }
        }
    }
}

#[test]
#[ignore = "perturbs DigitalISOGain on the sd Quattro H corpus file in memory"]
fn sd_quattro_digital_iso_gain_enters_mode_profiles_once() {
    let Some(corpus) = std::env::var_os("X3F_TEST_FILES").map(PathBuf::from) else {
        eprintln!("skip: set X3F_TEST_FILES to run the sd Quattro DigitalISOGain check");
        return;
    };
    let measure = |reader: &Reader| {
        let wb = reader.dng_default_wb();
        let calibration = ColorCalibration::new(reader, &wb).unwrap();
        let prepared = PreparedColorModes::resolve(
            reader,
            &wb,
            x3f_sys::ColorMode::Standard,
            &mut Vec::new(),
            x3f_sys::Control::none(),
        )
        .unwrap();
        let native = prepared.native.clone();
        let profiles = prepared
            .finish(
                reader,
                &wb,
                &calibration,
                1.0,
                &mut Vec::new(),
                x3f_sys::Control::none(),
            )
            .unwrap();
        let data = serialized_profile_metadata(reader, &calibration, &profiles, 1.0);
        let primary = root(&data);
        let baseline = primary[&t::BASELINE_EXPOSURE].rationals()[0];
        let mut entries = vec![primary];
        entries.extend(
            entries[0][&t::EXTRA_CAMERA_PROFILES]
                .longs()
                .into_iter()
                .map(|o| extra(&data, o as usize)),
        );
        for profile in entries
            .iter()
            .filter(|p| p.contains_key(&t::PROFILE_LOOK_TABLE_DATA))
        {
            assert_eq!(
                baseline + profile[&t::BASELINE_EXPOSURE_OFFSET].rationals()[0],
                0.0,
                "{}",
                profile[&t::PROFILE_NAME].text()
            );
        }
        (baseline, calibration.neutral, native)
    };
    let name = "_DQH1429.X3F";
    let Some(reader) = optional_reader(&corpus, name) else {
        return;
    };
    let gain = [4.0, 4.0, 2.0];
    assert_eq!(reader.dng_digital_iso_gain(), gain);
    let (baseline, neutral, native) = measure(&reader);
    // Same file and calibration, with only DigitalISOGain replaced by unity.
    // SAFETY: the reader owns the decoded CAMF matrix and nothing borrows it.
    unsafe {
        let (mut size, mut data) = (0, std::ptr::null_mut());
        assert_ne!(
            x3f_sys::x3f_get_camf_matrix_var(
                reader.x3f.as_ptr(),
                c"DigitalISOGain".as_ptr().cast_mut(),
                &mut size,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                x3f_sys::matrix_type_t_M_FLOAT,
                &mut data,
            ),
            0
        );
        assert_eq!(size, 3);
        std::slice::from_raw_parts_mut(data.cast::<f64>(), 3).fill(1.0);
    }
    assert_eq!(reader.dng_digital_iso_gain(), [1.0; 3]);
    let (unity_baseline, unity_neutral, unity_native) = measure(&reader);
    // The raw is not scaled by the gain. BaselineExposure carries its scalar
    // part once, and the neutral carries its relative part once.
    let mean = (gain[0] * gain[1] * gain[2]).cbrt();
    assert!((baseline - unity_baseline - mean.log2()).abs() <= 0.0001);
    for (c, g) in gain.iter().enumerate() {
        assert!((neutral[c] * g / mean / unity_neutral[c] - 1.0).abs() < 1e-12);
    }
    // Native Tiger preprocessing applies the gain to raw counts before WB,
    // CC, ISO and tone: counts = DIG * 100/SensorISO * (raw - RawOffset).
    for (c, g) in gain.iter().enumerate() {
        let ratio = native.normalized_to_counts[c] / unity_native.normalized_to_counts[c];
        assert!((ratio / g - 1.0).abs() < 1e-12, "count span {c}: {ratio}");
        if unity_native.count_offset[c] != 0.0 {
            let ratio = native.count_offset[c] / unity_native.count_offset[c];
            assert!((ratio / g - 1.0).abs() < 1e-12, "count offset {c}: {ratio}");
        }
    }
    assert_eq!(native.wb_gain, unity_native.wb_gain);
    assert_eq!(native.iso_gain, unity_native.iso_gain);
    assert_eq!(native.sensor_to_srgb, unity_native.sensor_to_srgb);
    eprintln!("{name}: BaselineExposure {baseline:.4} vs unity {unity_baseline:.4}; native spans and offsets scale by {gain:?}; WB, ISO and CC unchanged");
}

#[test]
#[ignore = "decodes Merrill and Quattro corpus exports across compression and recovery settings"]
fn color_mode_corpus_preserves_decoded_raw_and_calibration() {
    let corpus = std::env::var_os("X3F_TEST_FILES")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../x3f_test_files"));
    if !corpus.is_dir() {
        eprintln!("skip: set X3F_TEST_FILES to run DNG Color Mode raw invariance");
        return;
    }
    let destination =
        std::env::temp_dir().join(format!("x3f-dng-mode-invariance-{}", std::process::id()));
    std::fs::create_dir_all(&destination).unwrap();
    for name in [
        "sigma_sd1_merrill_15.x3f",
        "_SDI8284.X3F",
        "SDIM0002.X3F",
        "_DQH1429.X3F",
        "SDQH5085.X3F",
    ] {
        let input = corpus.join(name);
        if !input.is_file() {
            eprintln!("skip: missing {}", input.display());
            continue;
        }
        let wb = (name == "SDQH5085.X3F").then(|| "Sunlight".to_owned());
        let modes = loaded_reader(&input).available_color_modes().unwrap();
        assert!(
            modes.contains(&x3f_sys::ColorMode::Standard)
                && modes.contains(&x3f_sys::ColorMode::Vivid)
        );
        for compress in [false, true] {
            for recovery in [false, true] {
                let mut reference: Option<Vec<u8>> = None;
                let selections = if compress || recovery {
                    vec![x3f_sys::ColorMode::Standard, x3f_sys::ColorMode::Vivid]
                } else {
                    modes.clone()
                };
                for mode in selections {
                    let reader = loaded_reader(&input);
                    let camera_mode = reader.dng_color_mode();
                    let output = destination
                        .join(format!("{name}-{compress}-{recovery}-{}.dng", mode.name()));
                    let opts = crate::ProcessOptions {
                        color_mode: Some(mode),
                        compress,
                        dng_highlight_recovery: recovery,
                        denoise_intensity: 0,
                        wb: wb.clone(),
                        ..crate::ProcessOptions::default()
                    };
                    reader.dump_dng(&output, &opts).unwrap();
                    assert_eq!(
                        reader.dng_color_mode(),
                        camera_mode,
                        "selection mutated X3F header"
                    );
                    let data = std::fs::read(&output).unwrap();
                    assert_mode_profiles(&data, mode.name(), &modes);
                    if let Some(first) = &reference {
                        assert_raw_equal(first, &data);
                        if mode == x3f_sys::ColorMode::Vivid {
                            let a = root(first);
                            let b = root(&data);
                            let preview = |root: &BTreeMap<u16, Entry<'_>>, bytes: &[u8]| {
                                let start = root[&t::STRIP_OFFSETS].longs()[0] as usize;
                                let count = root[&t::STRIP_BYTE_COUNTS].longs()[0] as usize;
                                bytes[start..start + count].to_vec()
                            };
                            assert_ne!(
                                preview(&a, first),
                                preview(&b, &data),
                                "selected Color Mode did not reach preview"
                            );
                        }
                    } else {
                        reference = Some(data);
                    }
                    std::fs::remove_file(output).unwrap();
                }
                if !compress && !recovery {
                    let output = destination.join(format!("{name}-legacy.dng"));
                    loaded_reader(&input)
                        .dump_dng(
                            &output,
                            &crate::ProcessOptions {
                                denoise_intensity: 0,
                                wb: wb.clone(),
                                ..crate::ProcessOptions::default()
                            },
                        )
                        .unwrap();
                    let legacy = std::fs::read(&output).unwrap();
                    assert_raw_equal(reference.as_ref().unwrap(), &legacy);
                    let pin = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                        .join("../../_local/spp-alignment/chroma-lut-option/pins")
                        .join(format!("{name}.dng"));
                    if pin.is_file() {
                        assert!(
                            legacy == std::fs::read(&pin).unwrap(),
                            "no-option DNG differs from {}",
                            pin.display()
                        );
                    }
                    std::fs::remove_file(output).unwrap();
                }
                eprintln!("{name}: decoded raw, levels, WB, exposure and profile payloads passed; compress={compress}, recovery={recovery}");
            }
        }
    }
    std::fs::remove_dir(destination).unwrap();
}

#[test]
#[ignore = "checks the optional dp0 Quattro sample with a native Cinema tone reversal"]
fn quattro_native_tone_reversal_keeps_all_profiles() {
    let corpus = std::env::var_os("X3F_TEST_FILES")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../x3f_test_files"));
    let input = corpus.join("DP0Q0130.X3F");
    if !input.is_file() {
        eprintln!("skip: missing {}", input.display());
        return;
    }
    let modes = loaded_reader(&input).available_color_modes().unwrap();
    assert_eq!(modes.len(), 11);
    let destination =
        std::env::temp_dir().join(format!("x3f-dng-tone-reversal-{}", std::process::id()));
    std::fs::create_dir_all(&destination).unwrap();
    let mut reference: Option<Vec<u8>> = None;
    for (selection, name) in [
        (Some(x3f_sys::ColorMode::AsShot), "Sunset Red"),
        (Some(x3f_sys::ColorMode::Standard), "Standard"),
        (None, "legacy"),
    ] {
        let path = destination.join(format!("{name}.dng"));
        let report = crate::convert_file(
            &input,
            &path,
            crate::OutputFormat::Dng,
            &crate::ProcessOptions {
                color_mode: selection,
                denoise_intensity: 0,
                ..Default::default()
            },
            &std::sync::atomic::AtomicBool::new(false),
            |_| {},
        )
        .unwrap();
        assert!(report.warnings.is_empty(), "{:?}", report.warnings);
        let data = std::fs::read(&path).unwrap();
        if selection.is_some() {
            assert_mode_profiles(&data, name, &modes);
        }
        if let Some(first) = &reference {
            assert_raw_equal(first, &data);
        } else {
            reference = Some(data);
        }
        std::fs::remove_file(path).unwrap();
    }
    std::fs::remove_dir(destination).unwrap();
}

#[test]
#[cfg(any(unix, target_arch = "wasm32"))]
fn monochrome_as_shot_metadata_uses_header_or_picture_mode() {
    for (version, header, picture, expected) in [
        (0x0003_0000_u32, "Monochrome", "Standard", true),
        (0x0003_0000, "Standard", "Monochrome", false),
        (0x0003_0000, "Sepia", "Monochrome", false),
        (0x0003_0000, "", "Monochrome", false),
        (0x0004_0001, "Standard", "Monochrome", true),
        (0x0004_0001, "Monochrome", "Standard", false),
        (0x0004_0001, "Monochrome", "Sepia", false),
        (0x0004_0001, "Monochrome", "", false),
    ] {
        // IFD0 -> EXIF -> Sigma MakerNote with a 16-byte PictureMode value.
        let mut tiff = b"II*\0\x08\0\0\0".to_vec();
        for (tag, kind, count, offset) in [(0x8769_u16, 4_u16, 1_u32, 26_u32), (0x927c, 7, 44, 44)]
        {
            tiff.extend(1_u16.to_le_bytes());
            tiff.extend(tag.to_le_bytes());
            tiff.extend(kind.to_le_bytes());
            tiff.extend(count.to_le_bytes());
            tiff.extend(offset.to_le_bytes());
            tiff.extend(0_u32.to_le_bytes());
        }
        tiff.extend(b"SIGMA\0\0\0\0\x03");
        tiff.extend(1_u16.to_le_bytes());
        tiff.extend(0x003d_u16.to_le_bytes());
        tiff.extend(2_u16.to_le_bytes());
        tiff.extend(16_u32.to_le_bytes());
        tiff.extend(72_u32.to_le_bytes());
        tiff.extend(0_u32.to_le_bytes());
        tiff.extend(picture.bytes());
        tiff.resize(88, 0);
        let mut jpeg = b"\xff\xd8\xff\xe1".to_vec();
        jpeg.extend(((tiff.len() + 8) as u16).to_be_bytes());
        jpeg.extend(b"Exif\0\0");
        jpeg.extend(tiff);

        let mut bytes = vec![0; if version < 0x0004_0000 { 424 } else { 24 }];
        bytes[..4].copy_from_slice(b"FOVb");
        bytes[4..8].copy_from_slice(&version.to_le_bytes());
        let image = bytes.len() as u32;
        bytes.extend(b"SECi");
        for value in [0x0002_0000_u32, 2, 18, 1, 1, 0] {
            bytes.extend(value.to_le_bytes());
        }
        bytes.extend(&jpeg);
        let directory = bytes.len() as u32;
        bytes.extend(b"SECd");
        for value in [0x0002_0000, 1, image, jpeg.len() as u32 + 28] {
            bytes.extend(value.to_le_bytes());
        }
        bytes.extend(b"IMA2");
        bytes.extend(directory.to_le_bytes());
        let reader = Reader::from_bytes(&bytes).unwrap();
        // SAFETY: the reader owns this header and no other references exist.
        unsafe {
            let field = &mut (*reader.x3f.as_ptr()).header.color_mode;
            for (out, byte) in field.iter_mut().zip(header.bytes()) {
                *out = byte as _;
            }
        }
        assert_eq!(reader.dng_thumb_jpeg_bytes().unwrap(), jpeg);
        assert_eq!(
            as_shot_monochrome(&reader),
            expected,
            "{version:x}/{header}/{picture}"
        );
    }
}

#[test]
#[ignore = "writes and checks Monochrome DNGs from the optional Merrill and dp2 Quattro corpus"]
fn write_monochrome_as_shot_corpus_and_check_profiles() {
    fn reader_with_mode(path: &std::path::Path, mode: &str) -> Reader {
        let mut reader = loaded_reader(path);
        if reader.header_version() >= 0x0004_0000 {
            reader.load_thumbnail_jpeg().unwrap();
            // SAFETY: the loaded JPEG is owned only by this reader. The fixed-size
            // edit changes the fixture's PictureMode without changing its offsets.
            let jpeg = unsafe {
                let de = x3f_sys::x3f_get_thumb_jpeg(reader.x3f.as_ptr());
                let image = &mut (*de).header.data_subsection.image_data;
                std::slice::from_raw_parts_mut(image.data.cast::<u8>(), image.data_size as usize)
            };
            fn field(tiff: &[u8], ifd: usize, tag: u16, order: Order) -> (usize, usize) {
                let count = order.u16(&tiff[ifd..ifd + 2]) as usize;
                let entry = tiff[ifd + 2..]
                    .as_chunks::<12>()
                    .0
                    .iter()
                    .take(count)
                    .find(|entry| order.u16(&entry[..2]) == tag)
                    .unwrap();
                (
                    order.u32(&entry[4..8]) as usize,
                    order.u32(&entry[8..]) as usize,
                )
            }
            let base = jpeg.windows(6).position(|v| v == b"Exif\0\0").unwrap() + 6;
            let tiff = &jpeg[base..];
            let order = match &tiff[..2] {
                b"II" => Order::Little,
                b"MM" => Order::Big,
                _ => panic!("invalid fixture TIFF header"),
            };
            let ifd = order.u32(&tiff[4..8]) as usize;
            let (_, exif) = field(tiff, ifd, 0x8769, order);
            let (_, maker) = field(tiff, exif, 0x927c, order);
            assert_eq!(&tiff[maker..maker + 8], b"SIGMA\0\0\0");
            let (count, offset) = field(tiff, maker + 10, 0x003d, order);
            assert!(count > mode.len());
            let slot = &mut jpeg[base + offset..base + offset + count];
            slot.fill(0);
            slot[..mode.len()].copy_from_slice(mode.as_bytes());
            assert_eq!(
                x3f_sys::jpeg_exif::parse_jpeg_exif(jpeg)
                    .unwrap()
                    .picture_mode
                    .as_deref(),
                Some(mode)
            );
        } else {
            assert_eq!(reader.dng_color_mode(), "Monochrome");
            // SAFETY: the test owns the reader and preserves the source file.
            unsafe {
                let field = &mut (*reader.x3f.as_ptr()).header.color_mode;
                field.fill(0);
                for (out, byte) in field.iter_mut().zip(mode.bytes()) {
                    *out = byte as _;
                }
            }
        }
        reader
    }

    let corpus = std::env::var_os("X3F_TEST_FILES")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../x3f_test_files"));
    let destination =
        std::env::temp_dir().join(format!("x3f-dng-monochrome-{}", std::process::id()));
    std::fs::create_dir_all(&destination).unwrap();
    for (name, compress, count) in [
        ("DP1M0815.X3F", false, 6),
        ("DP2M0981.X3F", true, 6),
        ("_SDI8284.X3F", false, 11),
    ] {
        let source = corpus.join(name);
        if !source.is_file() {
            eprintln!("skip: missing {}", source.display());
            continue;
        }
        let mut data = Vec::new();
        for selection in [x3f_sys::ColorMode::Standard, x3f_sys::ColorMode::AsShot] {
            let mut reader = reader_with_mode(&source, "Monochrome");
            assert!(as_shot_monochrome(&reader));
            let header = reader.dng_color_mode();
            let jpeg = reader.dng_thumb_jpeg_bytes();
            let modes = reader.available_color_modes().unwrap();
            assert_eq!(modes.len(), count);
            let path = destination.join(format!("{name}-{}.dng", selection.name()));
            let mut warnings = Vec::new();
            super::super::write_controlled(
                &reader,
                &path,
                &crate::ProcessOptions {
                    color_mode: Some(selection),
                    compress,
                    denoise_intensity: 0,
                    ..Default::default()
                },
                x3f_sys::Control::none(),
                &mut || {},
                &mut warnings,
            )
            .unwrap();
            assert!(warnings.is_empty(), "{warnings:?}");
            assert_eq!(reader.dng_color_mode(), header);
            assert_eq!(reader.dng_thumb_jpeg_bytes(), jpeg);
            let bytes = std::fs::read(&path).unwrap();
            assert_mode_profiles(
                &bytes,
                if selection == x3f_sys::ColorMode::AsShot {
                    "Monochrome"
                } else {
                    "Standard"
                },
                &modes,
            );
            data.push(bytes);
        }
        assert_raw_equal(&data[0], &data[1]);
        let directories: Vec<_> = data
            .iter()
            .map(|bytes| {
                let primary = root(bytes);
                let offsets = primary[&t::EXTRA_CAMERA_PROFILES].longs();
                let start = primary[&t::STRIP_OFFSETS].longs()[0] as usize;
                let size = primary[&t::STRIP_BYTE_COUNTS].longs()[0] as usize;
                let preview = &bytes[start..start + size];
                let mut profiles = vec![primary];
                profiles.extend(offsets.into_iter().map(|off| extra(bytes, off as usize)));
                (preview, profiles)
            })
            .collect();
        assert_eq!(
            directories[0].0, directories[1].0,
            "Standard preview changed"
        );
        for original in &directories[0].1 {
            let name = original[&t::PROFILE_NAME].text();
            let promoted = directories[1]
                .1
                .iter()
                .find(|profile| profile[&t::PROFILE_NAME].text() == name)
                .unwrap();
            for tag in [
                t::COLOR_MATRIX1,
                t::FORWARD_MATRIX1,
                t::BASELINE_EXPOSURE_OFFSET,
            ] {
                assert_eq!(
                    original.get(&tag).map(Entry::rationals),
                    promoted.get(&tag).map(Entry::rationals),
                    "{name}/{tag}"
                );
            }
            for tag in [t::PROFILE_TONE_CURVE, t::PROFILE_LOOK_TABLE_DATA] {
                assert_eq!(
                    original.get(&tag).map(Entry::floats),
                    promoted.get(&tag).map(Entry::floats),
                    "{name}/{tag}"
                );
            }
            for tag in [
                t::DEFAULT_BLACK_RENDER,
                t::PROFILE_LOOK_TABLE_DIMS,
                t::PROFILE_LOOK_TABLE_ENCODING,
            ] {
                assert_eq!(
                    original.get(&tag).map(Entry::longs),
                    promoted.get(&tag).map(Entry::longs),
                    "{name}/{tag}"
                );
            }
        }
        let options = crate::ProcessOptions {
            color_mode: Some(x3f_sys::ColorMode::AsShot),
            denoise_intensity: 0,
            ..Default::default()
        };
        let tiff = destination.join(format!("{name}-unsupported.tif"));
        let error = reader_with_mode(&source, "Monochrome")
            .dump_tiff(&tiff, &options)
            .unwrap_err();
        assert!(
            error.to_string().contains("as-shot monochrome and sepia"),
            "{error}"
        );
        assert!(!tiff.exists());
        let sepia = destination.join(format!("{name}-sepia.dng"));
        let error = reader_with_mode(&source, "Sepia")
            .dump_dng(&sepia, &options)
            .unwrap_err();
        assert!(
            error.to_string().contains("as-shot monochrome and sepia"),
            "{error}"
        );
        assert!(!sepia.exists());
        eprintln!("{name}: Monochrome default, all {count} native modes, unchanged profiles/raw/preview; TIFF and Sepia still reject");
    }
    eprintln!("Monochrome DNG validation files: {}", destination.display());
}

#[test]
#[ignore = "checks the sd Quattro H Auto white that the classic profile model cannot represent"]
fn sd_quattro_h_unrepresentable_as_shot_white_is_rejected() {
    let Some(corpus) = std::env::var_os("X3F_TEST_FILES").map(PathBuf::from) else {
        eprintln!("skip: set X3F_TEST_FILES to run the SDQH5085 white check");
        return;
    };
    let name = "SDQH5085.X3F";
    let Some(reader) = optional_reader(&corpus, name) else {
        return;
    };
    // Auto gains [0.839935, 1.04999, 1.54372] give negative Z through CM0.
    assert_eq!(reader.dng_default_wb(), "Auto");
    let destination = std::env::temp_dir().join(format!("x3f-dng-white-{}", std::process::id()));
    std::fs::create_dir_all(&destination).unwrap();
    let path = destination.join("standard.dng");
    let error = reader
        .dump_dng(
            &path,
            &crate::ProcessOptions {
                color_mode: Some(x3f_sys::ColorMode::Standard),
                denoise_intensity: 0,
                ..Default::default()
            },
        )
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("invalid classic profile as-shot white"),
        "{error}"
    );
    assert!(!path.exists());
    std::fs::remove_dir(destination).unwrap();
}

#[test]
#[ignore = "checks real Quattro PictureMode values against explicit named DNG selection"]
fn quattro_as_shot_matches_recorded_profile_and_raw() {
    let Some(corpus) = std::env::var_os("X3F_TEST_FILES").map(PathBuf::from) else {
        eprintln!("skip: set X3F_TEST_FILES to run Quattro AsShot equivalence");
        return;
    };
    for (name, recorded, named) in [
        ("_SDI8284.X3F", "Standard", x3f_sys::ColorMode::Standard),
        ("SDIM0002.X3F", "Standard", x3f_sys::ColorMode::Standard),
        ("_DQH1429.X3F", "SunsetRed", x3f_sys::ColorMode::SunsetRed),
    ] {
        let input = corpus.join(name);
        let Some(mut reader) = optional_reader(&corpus, name) else {
            continue;
        };
        reader.load_thumbnail_jpeg().unwrap();
        let jpeg = reader.dng_thumb_jpeg_bytes().unwrap();
        let exif = x3f_sys::jpeg_exif::parse_jpeg_exif(&jpeg).unwrap();
        assert_eq!(exif.picture_mode.as_deref(), Some(recorded), "{name}");
        let resolved =
            unsafe { x3f_sys::resolve_color_mode(reader.x3f.as_ptr(), x3f_sys::ColorMode::AsShot) }
                .unwrap();
        assert_eq!(resolved.mode, named, "{name}");
        assert!(!resolved.as_shot_fallback, "{name}");
        let modes = reader.available_color_modes().unwrap();
        assert!(modes.contains(&named), "{name}: {modes:?}");
        drop(reader);

        let destination =
            std::env::temp_dir().join(format!("x3f-dng-as-shot-{}", std::process::id()));
        std::fs::create_dir_all(&destination).unwrap();
        let mut outputs = Vec::new();
        for selection in [x3f_sys::ColorMode::AsShot, named] {
            let reader = loaded_reader(&input);
            let header_mode = reader.dng_color_mode();
            let path = destination.join(format!("{name}-{}.dng", selection.name()));
            reader
                .dump_dng(
                    &path,
                    &crate::ProcessOptions {
                        color_mode: Some(selection),
                        compress: true,
                        denoise_intensity: 0,
                        ..crate::ProcessOptions::default()
                    },
                )
                .unwrap();
            assert_eq!(reader.dng_color_mode(), header_mode);
            let data = std::fs::read(&path).unwrap();
            assert_mode_profiles(&data, named.name(), &modes);
            outputs.push(data);
            std::fs::remove_file(path).unwrap();
        }
        assert_raw_equal(&outputs[0], &outputs[1]);
        assert!(
            outputs[0] == outputs[1],
            "{name}: AsShot and {} DNGs differ in raw, preview, profiles or default",
            named.name()
        );
        std::fs::remove_dir(destination).unwrap();
        eprintln!("{name}: recorded PictureMode {recorded}, no fallback; AsShot and named DNGs are byte-identical with independently verified raw/profiles/default");
    }
}

#[test]
fn profile_exposure_uses_the_serialized_rational_sum() {
    let identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let table = vec![0.0f32; 72 * 17 * 65 * 3];
    for baseline in [3.000123, -3.000123, 4.00005, -0.00005] {
        for target in [0.0, 0.307085, 3.188639] {
            let (offset, total) = profile_exposure(baseline, target).unwrap();
            let blob = build_profile_blob(
                "x3fuse Exposure",
                identity,
                identity,
                Some(&[0.0, 0.0, 1.0, 1.0]),
                None,
                Some(ProfileLook::Classic(&table)),
                offset,
            );
            let parsed = extra(&blob, 0);
            let (n, d) = srational_pair(f64::from(baseline), 10000);
            let sdk_total =
                f64::from(n) / f64::from(d) + parsed[&t::BASELINE_EXPOSURE_OFFSET].rationals()[0];
            assert_eq!(total, sdk_total);
            assert!(total >= 0.0 && (total - target).abs() <= 0.000051);
        }
    }
    for invalid in [f64::NAN, f64::INFINITY, -1.0, 1e12] {
        assert!(profile_exposure(3.0, invalid).is_err());
    }
    assert!(profile_exposure(f32::NAN, 0.0).is_err());
}

#[test]
#[ignore = "writes full Color Mode profiles and SDK pixel probes from the camera corpus"]
fn write_color_mode_camera_fixtures() {
    let Some(corpus) = std::env::var_os("X3F_TEST_FILES").map(PathBuf::from) else {
        return;
    };
    let destination = PathBuf::from("/private/tmp/x3f-dng-classic");
    std::fs::create_dir_all(&destination).unwrap();
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis();
    let mut names = String::from("file\tprofile\tplacement\n");
    let mut probes =
        String::from("file\tprofile\twidth\theight\trow\tcol\tlinear_r\tlinear_g\tlinear_b\n");
    for (source, name) in [
        ("sigma_sd1_merrill_15.x3f", "sd1-merrill"),
        ("_SDI8284.X3F", "dp2-quattro"),
        ("_SDI8040.X3F", "dp2-merrill-adobe"),
        ("DP0Q0010.X3F", "dp0-quattro"),
    ] {
        let mut reader = loaded_reader(&corpus.join(source));
        let modes = reader.available_color_modes().unwrap();
        let name = format!("{name}-classic-color-modes-{nonce}.dng");
        let output = destination.join(&name);
        reader
            .dump_dng(
                &output,
                &crate::ProcessOptions {
                    color_mode: Some(x3f_sys::ColorMode::Standard),
                    compress: true,
                    denoise_intensity: 0,
                    ..crate::ProcessOptions::default()
                },
            )
            .unwrap();
        let data = std::fs::read(&output).unwrap();
        assert_mode_profiles(&data, x3f_sys::ColorMode::Standard.name(), &modes);
        let primary = root(&data);
        let raw = directory(
            &data,
            0,
            primary[&t::SUB_IFDS].longs()[0] as usize,
            Order::Little,
        );
        let samples = decoded_raw(&data);
        let (width, height) = (
            raw[&t::IMAGE_WIDTH].longs()[0],
            raw[&t::IMAGE_LENGTH].longs()[0],
        );
        let mut entries = vec![primary];
        entries.extend(
            entries[0][&t::EXTRA_CAMERA_PROFILES]
                .longs()
                .into_iter()
                .map(|o| extra(&data, o as usize)),
        );
        for (index, profile) in entries.iter().enumerate() {
            let profile_name = profile[&t::PROFILE_NAME].text();
            names.push_str(&format!(
                "{name}\t{profile_name}\t{}\n",
                if index == 0 { "primary" } else { "extra" }
            ));
            if !profile.contains_key(&t::PROFILE_LOOK_TABLE_DATA) {
                continue;
            }
            let (matrix, fit, exposure) = classic_model(profile, &entries[0]);
            for pixel in (0..width as usize * height as usize)
                .step_by((width as usize * height as usize / 1024).max(1))
            {
                let sensor = std::array::from_fn(|c| f64::from(samples[pixel * 3 + c]) / 65535.0);
                let expected = fit.apply(matrix.input(sensor, exposure));
                probes.push_str(&format!(
                    "{name}\t{}\t{width}\t{height}\t{}\t{}\t{:.9}\t{:.9}\t{:.9}\n",
                    profile[&t::PROFILE_NAME].text(),
                    pixel / width as usize,
                    pixel % width as usize,
                    expected[0],
                    expected[1],
                    expected[2]
                ));
            }
        }
        eprintln!(
            "{}: primary/as-shot Standard; all supported profiles parsed; lossless raw decoded",
            output.display()
        );
    }
    std::fs::write(destination.join("sdk-pixel-probes.tsv"), probes).unwrap();
    std::fs::write(destination.join("profiles.tsv"), names).unwrap();
}

#[test]
fn prebuilt_x3i_scale_preserves_raw_and_disables_preview_gain() {
    let corpus = std::env::var_os("X3F_TEST_FILES")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../x3f_test_files"));
    let input = corpus.join("_SDI8284.X3F");
    if !input.is_file() {
        eprintln!("skip: missing _SDI8284.X3F for the prebuilt DNG check");
        return;
    }
    let reader = loaded_reader(&input);
    let control = x3f_sys::Control::none();
    let source_opts = crate::ProcessOptions {
        denoise_intensity: 0,
        apply_sgain: Some(false),
        ..Default::default()
    };
    let source_inputs =
        super::super::WriterInputs::resolve(&reader, &source_opts, control, &mut Vec::new())
            .unwrap();
    let source = reader
        .get_image_with_control(&source_inputs.opts, control)
        .unwrap();
    drop(reader);
    let destination = std::env::temp_dir().join(format!("x3f-prebuilt-{}", std::process::id()));
    std::fs::create_dir_all(&destination).unwrap();
    let mut outputs = Vec::new();
    let mut baselines = Vec::new();
    for (i, scale, gain) in [(0, 1.0, false), (1, 4.0, false), (2, 4.0, true)] {
        let opts = crate::ProcessOptions {
            denoise_intensity: 0,
            apply_sgain: Some(gain),
            ..Default::default()
        };
        let reader = loaded_reader(&input);
        let inputs =
            super::super::WriterInputs::resolve(&reader, &opts, control, &mut Vec::new()).unwrap();
        // Keep the same raster in all runs. Only writer settings and S differ.
        let image = crate::Image {
            data: source.data.clone(),
            dng_highlight_scale: scale,
            ..source
        };
        let path = destination.join(format!("{i}.dng"));
        super::super::write_prepared(
            &reader,
            inputs,
            image,
            &path,
            control,
            &mut Vec::new(),
            true,
        )
        .unwrap();
        let bytes = std::fs::read(&path).unwrap();
        let root_offset = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
        let root = directory(&bytes, 0, root_offset, Order::Little);
        baselines.push(root[&t::BASELINE_EXPOSURE].rationals()[0]);
        outputs.push(bytes);
        std::fs::remove_file(path).unwrap();
    }
    fn raw_strips(bytes: &[u8]) -> Vec<&[u8]> {
        let primary = root(bytes);
        let raw = directory(
            bytes,
            0,
            primary[&t::SUB_IFDS].longs()[0] as usize,
            Order::Little,
        );
        raw[&t::STRIP_OFFSETS]
            .longs()
            .into_iter()
            .zip(raw[&t::STRIP_BYTE_COUNTS].longs())
            .map(|(offset, count)| &bytes[offset as usize..(offset + count) as usize])
            .collect()
    }
    assert_eq!(raw_strips(&outputs[0]), raw_strips(&outputs[1]));
    assert_eq!(baselines[1] - baselines[0], 4.0f64.log2());
    assert_eq!(
        outputs[1], outputs[2],
        "prebuilt preview applied spatial gain twice"
    );
    std::fs::remove_dir(destination).unwrap();
}

#[test]
fn prebuilt_color_mode_uses_extended_scale_for_profile_and_preview() {
    let corpus = std::env::var_os("X3F_TEST_FILES")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../x3f_test_files"));
    let input = corpus.join("_SDI8284.X3F");
    if !input.is_file() {
        eprintln!("skip: missing _SDI8284.X3F for the prebuilt Color Mode check");
        return;
    }
    let destination =
        std::env::temp_dir().join(format!("x3f-prebuilt-mode-{}", std::process::id()));
    std::fs::create_dir_all(&destination).unwrap();
    let mut outputs = Vec::new();
    for (i, scale) in [1.0, 4.0].into_iter().enumerate() {
        let reader = loaded_reader(&input);
        let opts = crate::ProcessOptions {
            color_mode: Some(x3f_sys::ColorMode::Standard),
            denoise_intensity: 0,
            apply_sgain: Some(true),
            ..Default::default()
        };
        let control = x3f_sys::Control::none();
        let inputs =
            super::super::WriterInputs::resolve(&reader, &opts, control, &mut Vec::new()).unwrap();
        let image = crate::Image {
            data: vec![512; 16 * 16 * 3],
            rows: 16,
            columns: 16,
            channels: 3,
            row_stride: 48,
            levels: crate::ImageLevels {
                black: [0.0; 3],
                white: [65535; 3],
            },
            dng_highlight_scale: scale,
            dng_shoulder_ceiling: 1.0,
        };
        let path = destination.join(format!("{i}.dng"));
        super::super::write_prepared(
            &reader,
            inputs,
            image,
            &path,
            control,
            &mut Vec::new(),
            true,
        )
        .unwrap();
        outputs.push(std::fs::read(&path).unwrap());
        std::fs::remove_file(path).unwrap();
    }
    assert_eq!(decoded_raw(&outputs[0]), decoded_raw(&outputs[1]));
    let first = root(&outputs[0]);
    let extended = root(&outputs[1]);
    assert_eq!(
        extended[&t::BASELINE_EXPOSURE].rationals()[0]
            - first[&t::BASELINE_EXPOSURE].rationals()[0],
        2.0
    );
    let preview = |bytes: &[u8], root: &BTreeMap<u16, Entry<'_>>| {
        let offset = root[&t::STRIP_OFFSETS].longs()[0] as usize;
        let count = root[&t::STRIP_BYTE_COUNTS].longs()[0] as usize;
        bytes[offset..offset + count].to_vec()
    };
    assert_ne!(
        preview(&outputs[0], &first),
        preview(&outputs[1], &extended)
    );
    std::fs::remove_dir(destination).unwrap();
}

#[test]
fn x3i_corpus_dispatch_writes_scaled_linear_dng_and_preserves_contracts() {
    use crate::{convert_file, ConversionStage, Error, OutputFormat, ProcessOptions, X3iContainer};
    use std::sync::atomic::{AtomicBool, Ordering};

    let corpus = std::env::var_os("X3F_TEST_FILES")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../x3f_test_files"));
    let destination = std::env::temp_dir().join(format!("x3i-wiring-{}", std::process::id()));
    std::fs::create_dir_all(&destination).unwrap();
    let cancel = AtomicBool::new(false);
    for name in ["DP0Q0002.X3I", "_DQH0978.X3I"] {
        let input = corpus.join(name);
        if !input.is_file() {
            eprintln!("skip: missing {name} for X3I DNG integration");
            continue;
        }
        let container = X3iContainer::open(&input).unwrap();
        assert_eq!(container.frames().len(), 7);
        let output = destination.join(format!("{name}.dng"));
        let options = ProcessOptions::default();
        let mut processed = 0;
        let report = convert_file(
            &input,
            &output,
            OutputFormat::Dng,
            &options,
            &cancel,
            |stage| {
                processed += usize::from(stage == ConversionStage::Process);
            },
        )
        .unwrap();
        assert_eq!(processed, 7);
        assert!(report
            .warnings
            .iter()
            .any(|s| s.starts_with("X3I denoise is off; set X3F_X3I_DENOISE_SCALE=")));
        for i in 0..7 {
            let exposure = container.frame_exposure_us(i).unwrap();
            assert!(report
                .warnings
                .iter()
                .any(|s| s.starts_with(&format!("X3I frame {i}: shutter_us={exposure} "))));
        }
        let summary = report
            .warnings
            .iter()
            .find(|s| s.starts_with("X3I merge:"))
            .unwrap();
        let scale: f64 = summary
            .split_whitespace()
            .find_map(|s| s.strip_prefix("s="))
            .unwrap()
            .parse()
            .unwrap();
        assert!(scale >= 1.0);
        let bytes = std::fs::read(&output).unwrap();
        let primary = root(&bytes);
        let raw = directory(
            &bytes,
            0,
            primary[&t::SUB_IFDS].longs()[0] as usize,
            Order::Little,
        );
        let merged_baseline = primary[&t::BASELINE_EXPOSURE].rationals()[0];
        assert_eq!(raw[&t::BITS_PER_SAMPLE].shorts(), [16, 16, 16]);
        assert_eq!(raw[&t::SAMPLES_PER_PIXEL].shorts(), [3]);
        assert_eq!(raw[&t::COMPRESSION].shorts(), [1]);
        assert_eq!(primary[&t::LINEAR_RESPONSE_LIMIT].rationals(), [1.0]);
        assert!(raw[&t::WHITE_LEVEL].longs().iter().all(|&v| v == 65535));
        assert!(raw[&t::BLACK_LEVEL].rationals().iter().all(|&v| v == 0.0));
        let mut samples = 0;
        let mut maximum = 0;
        for (offset, count) in raw[&t::STRIP_OFFSETS]
            .longs()
            .into_iter()
            .zip(raw[&t::STRIP_BYTE_COUNTS].longs())
        {
            let strip = &bytes[offset as usize..(offset + count) as usize];
            assert_eq!(strip.len() % 2, 0);
            for chunk in strip.as_chunks::<2>().0 {
                samples += 1;
                maximum = maximum.max(u16::from_le_bytes(*chunk));
            }
        }
        let rows = raw[&t::IMAGE_LENGTH].longs()[0];
        let cols = raw[&t::IMAGE_WIDTH].longs()[0];
        assert_eq!(samples, rows as usize * cols as usize * 3);
        let raster_maximum: f64 = summary
            .split_whitespace()
            .find_map(|s| s.strip_prefix("maximum="))
            .unwrap()
            .parse()
            .unwrap();
        assert_eq!(
            maximum,
            (raster_maximum / scale * 65535.0).round() as u16,
            "{name}"
        );
        if scale > 1.0 {
            assert_eq!(maximum, 65535, "{name}");
        }
        let merged_raw = (name == "_DQH0978.X3I").then(|| decoded_raw(&bytes));
        drop(bytes);
        // A second call must fail before it reads or modifies the destination.
        assert!(convert_file(
            &input,
            &output,
            OutputFormat::Dng,
            &options,
            &cancel,
            |_| panic!("existing output")
        )
        .is_err());
        std::fs::remove_file(&output).unwrap();

        let mut frame0 = container.open_frame(0).unwrap();
        frame0.load_property_list().unwrap();
        frame0.load_camf().unwrap();
        frame0.load_raw().unwrap();
        let no_denoise = ProcessOptions {
            denoise_intensity: 0,
            ..Default::default()
        };
        frame0.dump_dng(&output, &no_denoise).unwrap();
        let reference = std::fs::read(&output).unwrap();
        let reference_baseline = root(&reference)[&t::BASELINE_EXPOSURE].rationals()[0];
        assert!((merged_baseline - reference_baseline - scale.log2()).abs() < 1e-6);
        if let Some(merged_raw) = merged_raw {
            assert!(scale > 1.0, "sd Quattro H must retain extended highlights");
            let reference_raw = decoded_raw(&reference);
            assert_eq!(merged_raw.len(), reference_raw.len());
            let mut histogram = vec![0usize; 3 * usize::from(u16::MAX) + 1];
            for pixel in reference_raw.as_chunks::<3>().0 {
                histogram[pixel.iter().map(|&v| usize::from(v)).sum::<usize>()] += 1;
            }
            let target = (reference_raw.len() / 3).div_ceil(100);
            let mut selected = 0;
            let threshold = (0..histogram.len())
                .rev()
                .find(|&sum| {
                    selected += histogram[sum];
                    selected >= target
                })
                .unwrap();
            let mut reference_planes: [Vec<u16>; 3] = std::array::from_fn(|_| Vec::new());
            let mut merged_planes: [Vec<u16>; 3] = std::array::from_fn(|_| Vec::new());
            for (reference, merged) in reference_raw
                .as_chunks::<3>()
                .0
                .iter()
                .zip(merged_raw.as_chunks::<3>().0)
            {
                if reference.iter().map(|&v| usize::from(v)).sum::<usize>() < threshold {
                    continue;
                }
                for c in 0..3 {
                    reference_planes[c].push(reference[c]);
                    merged_planes[c].push(merged[c]);
                }
            }
            let median = |values: &mut [u16]| {
                values.sort_unstable();
                let middle = values.len() / 2;
                if values.len().is_multiple_of(2) {
                    (f64::from(values[middle - 1]) + f64::from(values[middle])) / 2.0
                } else {
                    f64::from(values[middle])
                }
            };
            for c in 0..3 {
                let reference = median(&mut reference_planes[c]);
                let merged = median(&mut merged_planes[c]) * scale;
                eprintln!(
                    "{name} top 1% channel {c}: pixels={selected} threshold={threshold} frame0={reference} merged={merged:.6} ratio={:.9}",
                    merged / reference
                );
                assert!((merged / reference - 1.0).abs() <= 0.05, "channel {c}");
            }
        }
        drop(reference);
        std::fs::remove_file(&output).unwrap();

        let mode_options = ProcessOptions {
            color_mode: Some(crate::ColorMode::Standard),
            ..no_denoise.clone()
        };
        let result = convert_file(
            &input,
            &output,
            OutputFormat::Dng,
            &mode_options,
            &cancel,
            |_| {},
        );
        result.unwrap();
        let modes = {
            let mut frame = container.open_frame(0).unwrap();
            frame.load_property_list().unwrap();
            frame.load_camf().unwrap();
            frame.available_color_modes().unwrap()
        };
        // The merged BaselineExposure uses 1e-6 steps and each profile
        // offset uses 1e-4 steps, so their sum is zero within 5.1e-5 EV.
        let bytes = std::fs::read(&output).unwrap();
        let primary = root(&bytes);
        let baseline = primary[&t::BASELINE_EXPOSURE].rationals()[0];
        assert_eq!(primary[&t::PROFILE_NAME].text(), "Standard");
        assert_eq!(primary[&t::AS_SHOT_PROFILE_NAME].text(), "Standard");
        let mut names: Vec<String> = primary[&t::EXTRA_CAMERA_PROFILES]
            .longs()
            .into_iter()
            .map(|o| extra(&bytes, o as usize))
            .chain([primary])
            .filter(|p| p.contains_key(&t::PROFILE_LOOK_TABLE_DATA))
            .map(|p| {
                let total = baseline + p[&t::BASELINE_EXPOSURE_OFFSET].rationals()[0];
                assert!(total.abs() <= 5.1e-5, "{name}: exposure sum {total}");
                p[&t::PROFILE_NAME].text().to_owned()
            })
            .collect();
        let mut expected: Vec<_> = modes.iter().map(|m| m.name().to_owned()).collect();
        names.sort();
        expected.sort();
        assert_eq!(names, expected, "{name}");
        std::fs::remove_file(&output).unwrap();
        // Cancel during preparation of the first frame and a later frame.
        for cancel_frame in [1, 2] {
            let (sender, receiver) = std::sync::mpsc::channel();
            let mut processes = 0;
            std::thread::scope(|scope| {
                let cancel_ref = &cancel;
                let worker = scope.spawn(move || {
                    receiver
                        .recv_timeout(std::time::Duration::from_secs(180))
                        .unwrap();
                    std::thread::sleep(std::time::Duration::from_millis(10));
                    cancel_ref.store(true, Ordering::Relaxed);
                });
                let result = convert_file(
                    &input,
                    &output,
                    OutputFormat::Dng,
                    &no_denoise,
                    &cancel,
                    |stage| {
                        if stage == ConversionStage::Process {
                            processes += 1;
                            if processes == cancel_frame {
                                sender.send(()).unwrap();
                            }
                        }
                    },
                );
                worker.join().unwrap();
                assert!(matches!(result, Err(Error::Cancelled)), "{result:?}");
            });
            assert!(!output.exists());
            cancel.store(false, Ordering::Relaxed);
        }
    }
    std::fs::remove_dir(destination).unwrap();
}
