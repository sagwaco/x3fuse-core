//! Native checks of the whole-image prefilter. The oracle is x3f-docs
//! `spp-rgb-trace/followup-tiled-denoise/prefilter/native.cpp`. It composes the
//! exported native functions in the TiledDenoise order: ResampleImage,
//! DoThreadedPix(XDoBilateralFilter), ComputeEdgeMap, ApplyEdgeLUT with border 0,
//! and LocalMean with border 4. It fills the LUT globals with the native formulas.

use super::*;
use std::sync::atomic::AtomicBool;

fn fnv(values: &[f32]) -> u64 {
    crate::float_front::fnv1a_f32(values)
}

fn next(state: &mut u32) -> u32 {
    *state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *state
}

// The three oracle parameter sets: filter size, subsample, noise scaling,
// locality, edge strength, decay, edge bias, accumulate, smooth, model and
// sensor scale.
fn params(set: usize, scale: f32) -> (NoiseParams, f32) {
    let mut p = NoiseParams {
        scale,
        ..NoiseParams::default()
    };
    let sensor = match set {
        0 => 1.0,
        1 => {
            p.filter_size = 9;
            p.subsample = false;
            p.noise_scaling = 0.73;
            p.locality = 0.837;
            p.edge_strength = 0.413;
            p.decay = 0.913;
            p.edge_bias = 3.7;
            p.accumulate = true;
            p.smooth_edge = false;
            p.color_model = [
                [2.03, 3.17, 5.29],
                [0.12345, 0.23456, 0.45678],
                [1e-4, 2e-5, 3e-6],
            ];
            1.37
        }
        _ => {
            p.filter_size = 5;
            p.noise_scaling = 1.0;
            p.locality = 1.9;
            p.edge_strength = 0.05;
            p.decay = 0.71;
            p.edge_bias = 0.0;
            p.color_model = [
                [0.8, 1.1, 0.9],
                [0.013, 0.021, 0.017],
                [1.3e-5, 2.1e-5, 0.7e-5],
            ];
            0.83
        }
    };
    (p, sensor)
}

fn pipeline(params: &NoiseParams, sensor: f32) -> Pipeline {
    let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    Pipeline::new(params.clone(), identity, [4.0; 3], sensor, [1.0; 3]).unwrap()
}

struct Case {
    buffer: Vec<f32>,
    origin: usize,
    stride: usize,
    reduced: [usize; 2],
    params: NoiseParams,
    pipeline: Pipeline,
}

// Rebuild one PREFILTER fixture input and check its source hash.
fn case(line: &str) -> (Case, Vec<u64>) {
    let f: Vec<&str> = line.split_whitespace().collect();
    let u = |i: usize| f[i].parse::<usize>().unwrap();
    let hashes: Vec<u64> = f[11..15]
        .iter()
        .map(|v| u64::from_str_radix(v, 16).unwrap())
        .collect();
    let (fw, fh) = (u(1), u(2));
    let scale = f32::from_bits(u32::from_str_radix(f[3], 16).unwrap());
    let (ox, oy, pad, set, pattern) = (u(4), u(5), u(6), u(7), u(8));
    let layout = usize::from(pad != 0);
    let stride = 3 * (fw + ox + 2 * layout) + pad;
    let rows = fh + oy + 3 * layout;
    let mut buffer = vec![-17.25_f32; stride * rows];
    let mut state = (0x2545_f491_u32)
        .wrapping_add(97 * fw as u32)
        .wrapping_add(13 * set as u32)
        .wrapping_add(pattern as u32);
    for y in 0..fh {
        for x in 0..fw {
            for c in 0..3 {
                let s = next(&mut state);
                buffer[stride * (oy + y) + 3 * (ox + x) + c] = if pattern == 0 {
                    150.0 + (s >> 16) as f32 * 0.0473
                } else {
                    400.0
                        + x as f32 * 3.7
                        + y as f32 * 2.3
                        + c as f32 * 41.9
                        + ((s >> 16) & 31) as f32 * 0.61
                };
            }
        }
    }
    assert_eq!(fnv(&buffer), hashes[0], "{line}");
    let (params, sensor) = params(set, scale);
    let reduced = sub_size([fw, fh], scale);
    assert_eq!(reduced, [u(9), u(10)], "{line}");
    let pipeline = pipeline(&params, sensor);
    (
        Case {
            buffer,
            origin: stride * oy + 3 * ox,
            stride,
            reduced,
            params,
            pipeline,
        },
        hashes,
    )
}

// The exp and edge LUTs call the platform expf and tanhf, like the native.
// The pins were recorded with the arm64 macOS libm.
#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
#[test]
fn prefilter_matches_native_composition() {
    let fixture = include_str!("../../../tests/fixtures/spp-prefilter-native.txt");
    let mut cases = 0;
    for line in fixture.lines().filter(|l| l.starts_with("PREFILTER ")) {
        let (case, hashes) = case(line);
        assert_eq!(fnv(case.pipeline.exp_lut()), hashes[1], "{line}");
        let output = prefilter(
            &case.buffer[case.origin..],
            case.stride,
            case.reduced,
            &case.params,
            &case.pipeline,
            Control::none(),
        )
        .unwrap();
        assert_eq!(output.len(), 3 * case.reduced[0] * case.reduced[1]);
        assert_eq!(fnv(&output), hashes[3], "{line}");
        cases += 1;
    }
    assert_eq!(cases, 72);
}

#[test]
fn prefilter_cancellation_and_bounds() {
    let fixture = include_str!("../../../tests/fixtures/spp-prefilter-native.txt");
    let line = fixture
        .lines()
        .find(|l| l.starts_with("PREFILTER 130 90 3f000000 3 2 5 2 0 "))
        .unwrap();
    let (case, _) = case(line);
    let source = &case.buffer[case.origin..];
    let cancel = AtomicBool::new(true);
    assert!(matches!(
        prefilter(
            source,
            case.stride,
            case.reduced,
            &case.params,
            &case.pipeline,
            Control::new(&cancel)
        ),
        Err(crate::Error::Cancelled)
    ));
    // The native blend at 0x7fe3c runs only for alpha != 1. The port rejects it.
    let blended = NoiseParams {
        alpha: 0.75,
        ..case.params.clone()
    };
    assert!(matches!(
        prefilter(
            source,
            case.stride,
            case.reduced,
            &blended,
            &case.pipeline,
            Control::none()
        ),
        Err(crate::Error::InvalidData(
            "SPP prefilter requires neutral alpha"
        ))
    ));
    // LocalMean border 4 needs 10 rows and columns, and the bilateral filter
    // needs 5 rows. Empty, oversized and short-source requests fail.
    for size in [[9, 30], [30, 9], [0, 0], [usize::MAX, 2]] {
        assert!(
            prefilter(
                source,
                case.stride,
                size,
                &case.params,
                &case.pipeline,
                Control::none()
            )
            .is_err(),
            "{size:?}"
        );
    }
    assert!(prefilter(
        &source[..case.stride * 10],
        case.stride,
        case.reduced,
        &case.params,
        &case.pipeline,
        Control::none()
    )
    .is_err());
}

/// The actual SD1 export input: the prepared float raster and the native
/// noise setup, reduced at scale 0.5 to 2464x1632. Release builds on arm64 macOS.
/// Set X3F_PREFILTER_DUMP to a directory to write the native oracle input.
#[cfg(all(not(debug_assertions), target_arch = "aarch64", target_os = "macos"))]
#[test]
fn prefilter_matches_native_sd1() {
    let root = std::env::var_os("X3F_TEST_FILES")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../x3f_test_files")
        });
    let path = root.join("sigma_sd1_merrill_15.x3f");
    if !path.exists() {
        eprintln!("SPP prefilter SD1: skipped, reference corpus is absent");
        return;
    }
    let fixture = include_str!("../../../tests/fixtures/spp-prefilter-native.txt");
    let expected: Vec<&str> = fixture
        .lines()
        .find(|l| l.starts_with("CORPUS "))
        .unwrap()
        .split_whitespace()
        .collect();
    let path = std::ffi::CString::new(path.to_str().unwrap()).unwrap();
    unsafe {
        let file = libc::fopen(path.as_ptr(), c"rb".as_ptr());
        assert!(!file.is_null());
        let x3f = crate::x3f_new_from_file(file.cast());
        assert!(!x3f.is_null());
        for section in [
            crate::x3f_get_camf(x3f),
            crate::x3f_get_prop(x3f),
            crate::x3f_get_raw(x3f),
        ] {
            assert!(!section.is_null());
            assert_eq!(
                crate::x3f_load_data(x3f, section),
                crate::x3f_return_e_X3F_OK
            );
        }
        let wb = x3f_get_wb(x3f);
        let control = Control::none();
        let gain = merrill_request_gain(x3f, wb, control).unwrap();
        let exif = merrill_exif(x3f, control).unwrap();
        let noise = noise_params(x3f, gain, &exif).unwrap();
        assert_eq!(noise.scale, 0.5);
        let matrix = color_matrix(x3f, wb, noise.iso_gain).unwrap();
        let sensor = typed_scalar(x3f, c"SensorISO", 3, 0.) as f32;
        let sensor_scale = if sensor == 0. { 1. } else { sensor / 100. };
        let mut strengths = [0.; 3];
        assert_ne!(
            x3f_get_camf_float_vector(
                x3f,
                c"ColorDQCamRGB".as_ptr().cast_mut(),
                strengths.as_mut_ptr()
            ),
            0
        );
        let dq_scale = f64::from(noise.color_dq_scale.min(noise.iso_gain));
        let pipeline = Pipeline::new(
            noise.clone(),
            matrix,
            strengths.map(|v| v * dq_scale),
            sensor_scale,
            gain.map(|v| v as f32),
        )
        .unwrap();
        let prepared = merrill_prepare_float(
            x3f,
            wb,
            true,
            true,
            Some(noise.despeckle),
            control,
            &mut |_, _, _| {},
        )
        .unwrap();
        crate::x3f_delete(x3f);
        libc::fclose(file);
        let full = [prepared.width, prepared.height];
        let reduced = sub_size(full, noise.scale);
        assert_eq!(reduced, [2464, 1632]);
        if let Some(dir) = std::env::var_os("X3F_PREFILTER_DUMP") {
            let dir = std::path::PathBuf::from(dir);
            std::fs::create_dir_all(&dir).unwrap();
            let bytes: Vec<u8> = prepared.data.iter().flat_map(|v| v.to_le_bytes()).collect();
            std::fs::write(dir.join("prepared.f32"), bytes).unwrap();
            let hex = |v: f32| format!("{:08x}", v.to_bits());
            let model: Vec<String> = noise
                .color_model
                .iter()
                .flatten()
                .map(|v| format!("{:016x}", v.to_bits()))
                .collect();
            let meta = format!(
                "{} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}\n",
                full[0],
                full[1],
                prepared.row_stride,
                hex(noise.scale),
                noise.filter_size,
                u8::from(noise.subsample),
                hex(noise.noise_scaling),
                hex(noise.locality),
                hex(noise.edge_strength),
                hex(noise.decay),
                hex(noise.edge_bias),
                hex(noise.alpha),
                u8::from(noise.accumulate),
                u8::from(noise.smooth_edge),
                hex(sensor_scale),
                model.join(" ")
            );
            std::fs::write(dir.join("meta.txt"), meta).unwrap();
        }
        assert_eq!(noise.alpha, 1.0);
        let output = prefilter(
            &prepared.data,
            prepared.row_stride,
            reduced,
            &noise,
            &pipeline,
            control,
        )
        .unwrap();
        let hash = |i: usize| u64::from_str_radix(expected[i], 16).unwrap();
        assert_eq!(fnv(&prepared.data), hash(6));
        assert_eq!(fnv(pipeline.exp_lut()), hash(7));
        assert_eq!(fnv(&output), hash(9));
    }
}
