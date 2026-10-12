//! Whole-orchestration checks of `denoise_image` against the direct native
//! TiledDenoise (0x7ee90) and Stage2B Sharpen (0x80b34). The fixture comes from
//! x3f-docs `spp-rgb-trace/followup-tiled-denoise/image/gen.py`, which runs the
//! reviewer wrapper `tiled.cpp` in file mode. It stores the native NoiseParams
//! bytes and native output hashes only.

use super::*;

fn fnv(values: &[f32]) -> u64 {
    crate::float_front::fnv1a_f32(values)
}

fn hex_u64(s: &str) -> u64 {
    u64::from_str_radix(s, 16).unwrap()
}

fn hex_f32(s: &str) -> f32 {
    f32::from_bits(u32::from_str_radix(s, 16).unwrap())
}

/// Decode the native 0x1c0-byte NoiseParams with the offsets that the native
/// SetUpNoiseParamsF20 oracle pins (followup-tiled-denoise/params/native.cpp).
/// `iso_gain` is +0x44, the ComputeEdgeMap scale argument.
fn decode(bytes: &[u8]) -> NoiseParams {
    let u = |o: usize| u32::from_le_bytes(bytes[o..o + 4].try_into().unwrap());
    let f = |o: usize| f32::from_bits(u(o));
    let d = |o: usize| f64::from_le_bytes(bytes[o..o + 8].try_into().unwrap());
    NoiseParams {
        iso: f(0x04),
        iso_gain: f(0x44),
        despeckle: f(0x14),
        scale: f(0x18),
        filter_size: u(0x20),
        subsample: bytes[0x24] != 0,
        noise_scaling: f(0x28),
        alpha: f(0x30),
        locality: f(0x34),
        edge_bias: f(0x38),
        edge_strength: f(0x3c),
        smooth_edge: bytes[0x4a] != 0,
        accumulate: bytes[0x4b] != 0,
        decay: f(0x4c),
        local_mean_iters: u(0x50),
        aniso_iters: u(0x60),
        aniso_k: u(0x68),
        lambda: f(0x6c),
        aniso_noise_scale: f(0x70),
        color_dq_scale: f(0x74),
        epsilon: f(0x80),
        mixing: f(0x84),
        color_mixing: f(0x88),
        residual_strength: f(0x8c),
        column_noise_scale: f(0xac),
        rcps_threshold: f(0xb8),
        rcps_map_threshold: f(0xbc),
        rcps_blending: f(0xc0),
        perm_half: u(0xc4),
        sharpen_amount: f(0xcc),
        sharpen_limit: f(0xd0),
        color_model: std::array::from_fn(|r| std::array::from_fn(|c| d(0xe0 + 8 * (3 * r + c)))),
        intensity_model: [f(0x170), f(0x174), f(0x178)],
        fpn_suppression: u(0x180),
    }
}

// Native fields this port does not read. The fixture blobs must keep the
// values that the port assumes: median and directional flags, dark
// neutralization and its thresholds, NM diffusion iterations, residual
// smoothing columns and limits, the BCS filter size, and the monochrome flag.
fn check_fixed_fields(bytes: &[u8], line: &str) {
    let u = |o: usize| u32::from_le_bytes(bytes[o..o + 4].try_into().unwrap());
    assert_eq!([bytes[0x48], bytes[0x49]], [1, 0], "{line}");
    assert_eq!([u(0x54), u(0x58), u(0x5c)], [0, 0, 0], "{line}");
    assert_eq!(
        [u(0x64), u(0x90), u(0xb0), u(0xb4)],
        [1, 3, 27, 2],
        "{line}"
    );
    assert_eq!(
        [u(0x94), u(0x98)],
        [0.12_f32.to_bits(), 0.0025_f32.to_bits()],
        "{line}"
    );
    assert_eq!(bytes[0x186], 0, "{line}");
}

// The generator's integer-valued input: exact in f32 on every target.
fn pixels(width: usize, height: usize) -> Vec<f32> {
    let mut state = 0x1357_9bdf_u32;
    let mut out = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            for c in 0..3 {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let value = 300
                    + (x * 13 + y * 7 + c * 101) % 1900
                    + ((x / 16 + y / 16) % 2) * 450
                    + (state >> 9) as usize % 61;
                out.push(value as f32);
            }
        }
    }
    out
}

struct Constants {
    matrix: [[f64; 3]; 3],
    dq: [f64; 3],
    sensor: f32,
}

fn constants(fixture: &str) -> Constants {
    let line = |key: &str| -> Vec<&str> {
        fixture
            .lines()
            .find(|l| l.starts_with(key))
            .unwrap()
            .split_whitespace()
            .skip(1)
            .collect()
    };
    let m: Vec<f64> = line("MATRIX ")
        .iter()
        .map(|s| f64::from_bits(hex_u64(s)))
        .collect();
    let dq: Vec<f64> = line("DQ ")
        .iter()
        .map(|s| f64::from_bits(hex_u64(s)))
        .collect();
    Constants {
        matrix: [[m[0], m[1], m[2]], [m[3], m[4], m[5]], [m[6], m[7], m[8]]],
        dq: [dq[0], dq[1], dq[2]],
        sensor: hex_f32(line("SENSOR ")[0]),
    }
}

/// Run one fixture case. Returns the output hash and the native sharpened hash.
fn run_case(line: &str, constants: &Constants, control: Control<'_>) -> crate::Result<(u64, u64)> {
    let f: Vec<&str> = line.split_whitespace().collect();
    let u = |i: usize| f[i].parse::<usize>().unwrap();
    let (width, height) = (u(3), u(4));
    let active = [u(5), u(6), u(7), u(8)];
    let scale = hex_f32(f[9]);
    let gain = [hex_f32(f[10]), hex_f32(f[11]), hex_f32(f[12])];
    let bytes: Vec<u8> = (0..f[14].len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&f[14][i..i + 2], 16).unwrap())
        .collect();
    assert_eq!(bytes.len(), 0x1c0, "{line}");
    check_fixed_fields(&bytes, line);
    let noise = decode(&bytes);
    assert_eq!(noise.scale, scale, "{line}");
    let mut data = pixels(width, height);
    assert_eq!(fnv(&data), hex_u64(f[13]), "{line}");
    let pipeline = Pipeline::new(
        noise.clone(),
        constants.matrix,
        constants.dq,
        constants.sensor,
        gain,
    )?;
    denoise_image(
        &mut data,
        width * 3,
        [width, height],
        active,
        &noise,
        &pipeline,
        control,
    )?;
    Ok((fnv(&data), hex_u64(f[16])))
}

// Pipeline::new builds the exp and edge LUTs with the platform expf and
// tanhf, like the native. The pins were recorded with the arm64 macOS libm.
#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
#[test]
fn denoise_image_matches_native_tiled_denoise() {
    let fixture = include_str!("../../../tests/fixtures/spp-image-native.txt");
    let constants = constants(fixture);
    let mut cases = 0;
    for line in fixture.lines().filter(|l| l.starts_with("IMAGE ")) {
        let (actual, native) = run_case(line, &constants, Control::none()).unwrap();
        assert_eq!(actual, native, "{line}");
        cases += 1;
    }
    assert_eq!(cases, 8);
}

/// Native TiledDenoise accepts tile cores smaller than the 40-pixel halo
/// (320x240 and 256x192 with an 8x8 grid). `TileLayout::new` rejects them.
/// If the port lifts this restriction, move these records to IMAGE.
#[test]
fn denoise_image_rejects_native_small_core_cases() {
    let fixture = include_str!("../../../tests/fixtures/spp-image-native.txt");
    let constants = constants(fixture);
    let mut cases = 0;
    for line in fixture.lines().filter(|l| l.starts_with("NATIVE_ONLY ")) {
        assert!(
            matches!(
                run_case(line, &constants, Control::none()),
                Err(crate::Error::InvalidData("invalid SPP tile geometry"))
            ),
            "{line}"
        );
        cases += 1;
    }
    assert_eq!(cases, 2);
}

#[test]
fn denoise_image_cancellation_and_layout_errors() {
    let fixture = include_str!("../../../tests/fixtures/spp-image-native.txt");
    let constants = constants(fixture);
    let line = fixture
        .lines()
        .find(|l| l.starts_with("IMAGE sd1-s0.5-edges "))
        .unwrap();
    let cancel = std::sync::atomic::AtomicBool::new(true);
    assert!(matches!(
        run_case(line, &constants, Control::new(&cancel)),
        Err(crate::Error::Cancelled)
    ));
    // A 320x320 image whose active bottom edge leaves more rows than one tile.
    let f: Vec<&str> = line.split_whitespace().collect();
    let bytes: Vec<u8> = (0..f[14].len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&f[14][i..i + 2], 16).unwrap())
        .collect();
    let noise = decode(&bytes);
    let pipeline = Pipeline::new(
        noise.clone(),
        constants.matrix,
        constants.dq,
        constants.sensor,
        [1.0; 3],
    )
    .unwrap();
    let mut data = pixels(320, 320);
    let original = data.clone();
    assert!(denoise_image(
        &mut data,
        320 * 3,
        [320, 320],
        [0, 0, 320, 100],
        &noise,
        &pipeline,
        Control::none()
    )
    .is_err());
    // Sizes the 8x8 grid cannot split, and a short image.
    for (size, len) in [([319, 320], 319 * 320 * 3), ([320, 320], 320 * 320 * 3 - 1)] {
        let mut data = pixels(320, 320);
        data.truncate(len);
        assert!(denoise_image(
            &mut data,
            size[0] * 3,
            size,
            [0, 0, size[0], size[1]],
            &noise,
            &pipeline,
            Control::none()
        )
        .is_err());
    }
    // Validation runs before the image changes.
    assert_eq!(fnv(&data), fnv(&original));
}

// Write the mapped fields of `noise` into a native NoiseParams blob. The
// blob keeps every field the port does not map (checked by
// `check_fixed_fields`).
fn encode(noise: &NoiseParams, bytes: &mut [u8]) {
    let mut put = |o: usize, v: &[u8]| bytes[o..o + v.len()].copy_from_slice(v);
    for (o, v) in [
        (0x04, noise.iso),
        (0x44, noise.iso_gain),
        (0x14, noise.despeckle),
        (0x18, noise.scale),
        (0x28, noise.noise_scaling),
        (0x30, noise.alpha),
        (0x34, noise.locality),
        (0x38, noise.edge_bias),
        (0x3c, noise.edge_strength),
        (0x4c, noise.decay),
        (0x6c, noise.lambda),
        (0x70, noise.aniso_noise_scale),
        (0x74, noise.color_dq_scale),
        (0x80, noise.epsilon),
        (0x84, noise.mixing),
        (0x88, noise.color_mixing),
        (0x8c, noise.residual_strength),
        (0xac, noise.column_noise_scale),
        (0xb8, noise.rcps_threshold),
        (0xbc, noise.rcps_map_threshold),
        (0xc0, noise.rcps_blending),
        (0xcc, noise.sharpen_amount),
        (0xd0, noise.sharpen_limit),
        (0x170, noise.intensity_model[0]),
        (0x174, noise.intensity_model[1]),
        (0x178, noise.intensity_model[2]),
    ] {
        put(o, &v.to_le_bytes());
    }
    for (o, v) in [
        (0x20, noise.filter_size),
        (0x50, noise.local_mean_iters),
        (0x60, noise.aniso_iters),
        (0x68, noise.aniso_k),
        (0xc4, noise.perm_half),
        (0x180, noise.fpn_suppression),
    ] {
        put(o, &v.to_le_bytes());
    }
    put(0x24, &[u8::from(noise.subsample)]);
    put(0x4a, &[u8::from(noise.smooth_edge)]);
    put(0x4b, &[u8::from(noise.accumulate)]);
    for (i, v) in noise.color_model.iter().flatten().enumerate() {
        put(0xe0 + 8 * i, &v.to_le_bytes());
    }
}

/// The two reference files through the production setup and `denoise_image`,
/// compared with the direct native wrapper on the same Keep input. Release
/// only and ignored for cost. With X3F_IMAGE_DUMP, it writes the wrapper
/// file-mode input (spec.txt, input-rgb.f32, noiseparams.bin) per file. The
/// base blob is X3F_IMAGE_BASE_<SD1|DP2>, a native SetUpNoiseParamsF20 +
/// GatherParams result; mapped fields come from the Rust setup and any
/// difference from the native blob is printed. Native hashes are REAL records.
#[cfg(all(not(debug_assertions), target_arch = "aarch64", target_os = "macos"))]
#[test]
#[ignore]
fn denoise_image_matches_native_reference_files() {
    let root = std::env::var_os("X3F_TEST_FILES")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../x3f_test_files")
        });
    let fixture = include_str!("../../../tests/fixtures/spp-image-native.txt");
    let dump = std::env::var_os("X3F_IMAGE_DUMP").map(std::path::PathBuf::from);
    let mut checked = 0;
    for (name, key) in [("sigma_sd1_merrill_15.x3f", "SD1"), ("_P2M0358.X3F", "DP2")] {
        let path = root.join(name);
        if !path.exists() {
            continue;
        }
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
            let dq = strengths.map(|v| v * f64::from(noise.color_dq_scale.min(noise.iso_gain)));
            let adjusted = gain.map(|v| v as f32);
            let pipeline =
                Pipeline::new(noise.clone(), matrix, dq, sensor_scale, adjusted).unwrap();
            let mut prepared = merrill_prepare_float(
                x3f,
                wb,
                true,
                true,
                Some(noise.despeckle),
                control,
                &mut |_, _, _| {},
            )
            .unwrap();
            let size = [prepared.width, prepared.height];
            assert_eq!(prepared.row_stride, 3 * size[0]);
            let (active, keep, _) = merrill_rects(x3f).unwrap();
            let active = active_in_keep(active, keep, size).unwrap();
            crate::x3f_delete(x3f);
            libc::fclose(file);
            let input_hash = fnv(&prepared.data);
            if let Some(dir) = &dump {
                let dir = dir.join(key);
                std::fs::create_dir_all(&dir).unwrap();
                let base =
                    std::fs::read(std::env::var_os(format!("X3F_IMAGE_BASE_{key}")).unwrap())
                        .unwrap();
                let mut blob = base[..0x1c0].to_vec();
                check_fixed_fields(&blob, name);
                let native = decode(&blob);
                for (field, rust, native) in [
                    ("iso", noise.iso, native.iso),
                    ("scale", noise.scale, native.scale),
                    ("despeckle", noise.despeckle, native.despeckle),
                    ("noise_scaling", noise.noise_scaling, native.noise_scaling),
                    ("locality", noise.locality, native.locality),
                    ("edge_strength", noise.edge_strength, native.edge_strength),
                    ("decay", noise.decay, native.decay),
                    (
                        "sharpen_amount",
                        noise.sharpen_amount,
                        native.sharpen_amount,
                    ),
                ] {
                    if rust.to_bits() != native.to_bits() {
                        eprintln!("{key} {field}: rust {rust} native blob {native}");
                    }
                }
                if noise.color_model != native.color_model {
                    eprintln!(
                        "{key} color_model: rust {:?} native blob {:?}",
                        noise.color_model, native.color_model
                    );
                }
                encode(&noise, &mut blob);
                std::fs::write(dir.join("noiseparams.bin"), &blob).unwrap();
                let bytes: Vec<u8> = prepared.data.iter().flat_map(|v| v.to_le_bytes()).collect();
                std::fs::write(dir.join("input-rgb.f32"), bytes).unwrap();
                let hex = |v: f64| format!("{v:e}");
                let spec = format!(
                    "{} {} 8 8 40 40\n{}\n{}\n{}\n{}\n{} {} {} {}\n1 1 1\n{}\n",
                    size[0],
                    size[1],
                    matrix
                        .iter()
                        .flatten()
                        .map(|v| hex(*v))
                        .collect::<Vec<_>>()
                        .join(" "),
                    dq.map(hex).join(" "),
                    hex(f64::from(sensor_scale)),
                    adjusted.map(|v| hex(f64::from(v))).join(" "),
                    active[0],
                    active[1],
                    active[2],
                    active[3],
                    hex(f64::from(noise.sharpen_amount)),
                );
                std::fs::write(dir.join("spec.txt"), spec).unwrap();
                eprintln!(
                    "{key} input_fnv={input_hash:016x} dumped to {}",
                    dir.display()
                );
            }
            denoise_image(
                &mut prepared.data,
                3 * size[0],
                size,
                active,
                &noise,
                &pipeline,
                control,
            )
            .unwrap();
            let output = fnv(&prepared.data);
            eprintln!("{key} rust_sharpened_fnv={output:016x}");
            if let Some(line) = fixture
                .lines()
                .find(|l| l.starts_with(&format!("REAL {key} ")))
            {
                let f: Vec<&str> = line.split_whitespace().collect();
                assert_eq!(input_hash, hex_u64(f[2]), "{line}");
                assert_eq!(output, hex_u64(f[3]), "{line}");
                checked += 1;
            }
        }
    }
    eprintln!("SPP whole image: {checked} reference files compared with native");
}
