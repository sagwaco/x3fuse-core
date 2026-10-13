//! Bounded, headless GPU NLM. All GPU resources are shared by a process;
//! settings, cancellation, and the uncommitted output belong to each call.
use std::sync::{mpsc, Arc, Mutex, TryLockError};
use std::time::{Duration, Instant};

use super::{reflect_101, Img3};
use crate::{Control, Error, Result};
#[path = "bm3d_gpu.rs"]
mod bm3d_gpu;

const TILE: usize = 1024;
const INPUT_BYTES: u64 = ((TILE + 38) * (TILE + 38) * 8) as u64;
const OUTPUT_BYTES: u64 = (TILE * TILE * 8) as u64;
const LUT_BYTES: u64 = 196_609 * 16;

type SharedGpu = Mutex<Option<std::result::Result<Gpu, String>>>;
static GPU: SharedGpu = Mutex::new(None);

struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    general_pipeline: wgpu::ComputePipeline,
    bindings: wgpu::BindGroup,
    source: wgpu::Buffer,
    destination: wgpu::Buffer,
    readback: wgpu::Buffer,
    weights: wgpu::Buffer,
    params: wgpu::Buffer,
    failure: Arc<Mutex<Option<String>>>,
    workgroup: [u32; 2],
    bm3d: std::sync::OnceLock<std::result::Result<bm3d_gpu::Resources, String>>,
}

fn gpu_error(error: impl std::fmt::Display) -> Error {
    Error::Gpu(error.to_string())
}

impl Gpu {
    fn new(allow_subgroups: bool) -> std::result::Result<Self, String> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::METAL | wgpu::Backends::VULKAN | wgpu::Backends::DX12,
            ..Default::default()
        });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        }))
        .map_err(|e| e.to_string())?;
        let info = adapter.get_info();
        if info.device_type == wgpu::DeviceType::Cpu {
            return Err("adapter is a software renderer".into());
        }
        let subgroups = allow_subgroups && adapter.features().contains(wgpu::Features::SUBGROUP);
        match Self::from_adapter(&adapter, subgroups) {
            Err(first) if subgroups => {
                // Reuse the adapter, and retry only if subgroups were requested.
                // Both diagnostics matter if the portable pipeline also fails.
                Self::from_adapter(&adapter, false).map_err(|second| {
                    format!("subgroup initialization: {first}; portable initialization: {second}")
                })
            }
            result => result,
        }
    }

    fn from_adapter(adapter: &wgpu::Adapter, subgroups: bool) -> std::result::Result<Self, String> {
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("x3f denoising"),
            required_features: if subgroups {
                wgpu::Features::SUBGROUP
            } else {
                wgpu::Features::empty()
            },
            ..Default::default()
        }))
        .map_err(|e| e.to_string())?;
        let failure = Arc::new(Mutex::new(None));
        let errors = failure.clone();
        device.on_uncaptured_error(Arc::new(move |error: wgpu::Error| {
            errors
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .get_or_insert_with(|| error.to_string());
        }));
        let lost = failure.clone();
        device.set_device_lost_callback(move |reason, message| {
            *lost.lock().unwrap_or_else(|e| e.into_inner()) =
                Some(format!("device lost ({reason:?}): {message}"));
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("x3f NLM"),
            source: wgpu::ShaderSource::Wgsl(
                if subgroups {
                    include_str!("nlm_subgroup.wgsl")
                } else {
                    include_str!("nlm.wgsl")
                }
                .into(),
            ),
        });
        if let Some(error) = failure.lock().unwrap_or_else(|e| e.into_inner()).as_ref() {
            return Err(error.clone());
        }
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("x3f NLM"),
            layout: None,
            module: &shader,
            entry_point: Some("nlm"),
            compilation_options: Default::default(),
            cache: None,
        });
        if let Some(error) = failure.lock().unwrap_or_else(|e| e.into_inner()).as_ref() {
            return Err(error.clone());
        }
        let buffer = |label, size, usage| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage,
                mapped_at_creation: false,
            })
        };
        let general_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("x3f configurable NLM"),
            source: wgpu::ShaderSource::Wgsl(include_str!("nlm_general.wgsl").into()),
        });
        let general_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("NLM shared layout"),
            bind_group_layouts: &[&pipeline.get_bind_group_layout(0)],
            push_constant_ranges: &[],
        });
        let general_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("x3f configurable NLM"),
            layout: Some(&general_layout),
            module: &general_shader,
            entry_point: Some("nlm"),
            compilation_options: Default::default(),
            cache: None,
        });
        let source = buffer(
            "NLM input",
            INPUT_BYTES,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        let destination = buffer(
            "NLM output",
            OUTPUT_BYTES,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let readback = buffer(
            "NLM readback",
            OUTPUT_BYTES,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );
        let weights = buffer(
            "NLM weights",
            LUT_BYTES,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        let params = buffer(
            "NLM dimensions",
            32,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("NLM buffers"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: source.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: destination.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: weights.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params.as_entire_binding(),
                },
            ],
        });
        let gpu = Self {
            device,
            queue,
            pipeline,
            general_pipeline,
            bindings,
            source,
            destination,
            readback,
            weights,
            params,
            failure,
            workgroup: if subgroups { [128, 1] } else { [16, 16] },
            bm3d: std::sync::OnceLock::new(),
        };
        gpu.check().map_err(|e| e.to_string())?;
        let info = adapter.get_info();
        super::log(
            crate::x3f_verbosity_t_DEBUG,
            &format!("NLM GPU: {} ({:?})", info.name, info.backend),
        );
        Ok(gpu)
    }

    fn check(&self) -> Result<()> {
        match self
            .failure
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
        {
            Some(error) => Err(gpu_error(error)),
            None => Ok(()),
        }
    }

    fn tile(
        &self,
        pixels: &[[u32; 2]],
        dimensions: [u32; 8],
        control: Control<'_>,
        output: &mut Img3,
        x: usize,
        y: usize,
    ) -> Result<()> {
        self.check()?;
        control.check()?;
        let width = dimensions[2] as usize;
        let height = dimensions[3] as usize;
        let bytes = (width * height * 8) as u64;
        self.queue
            .write_buffer(&self.source, 0, bytemuck::cast_slice(pixels));
        self.queue
            .write_buffer(&self.params, 0, bytemuck::cast_slice(&dimensions));
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            // Zero in slot 6 is the original 3x3 dispatch used by kernel tests.
            let general = dimensions[6] != 0 || dimensions[4] > 10;
            let workgroup = if general { [8, 8] } else { self.workgroup };
            pass.set_pipeline(if general {
                &self.general_pipeline
            } else {
                &self.pipeline
            });
            pass.set_bind_group(0, &self.bindings, &[]);
            pass.dispatch_workgroups(
                (width as u32).div_ceil(workgroup[0]),
                (height as u32).div_ceil(workgroup[1]),
                1,
            );
        }
        encoder.copy_buffer_to_buffer(&self.destination, 0, &self.readback, 0, bytes);
        self.queue.submit([encoder.finish()]);
        let slice = self.readback.slice(..bytes);
        let (send, recv) = mpsc::sync_channel(1);
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = send.send(result);
        });
        let started = Instant::now();
        let result = (|| {
            loop {
                control.check()?;
                self.check()?;
                match self.device.poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: Some(Duration::from_millis(2)),
                }) {
                    Ok(_) | Err(wgpu::PollError::Timeout) => {}
                    Err(error) => return Err(gpu_error(error)),
                }
                match recv.try_recv() {
                    Ok(result) => {
                        result.map_err(gpu_error)?;
                        break;
                    }
                    Err(mpsc::TryRecvError::Disconnected) => {
                        return Err(gpu_error("readback disconnected"))
                    }
                    Err(mpsc::TryRecvError::Empty) => {}
                }
                if started.elapsed() > Duration::from_secs(30) {
                    *self.failure.lock().unwrap_or_else(|e| e.into_inner()) =
                        Some("GPU readback timed out".into());
                    return Err(gpu_error("GPU readback timed out"));
                }
            }
            self.check()?;
            let mapped = slice.get_mapped_range();
            let pixels: &[[u32; 2]] = bytemuck::cast_slice(&mapped);
            for row in 0..height {
                let dest = &mut output.data[((y + row) * output.cols + x) * 3..][..width * 3];
                for (out, p) in dest
                    .chunks_exact_mut(3)
                    .zip(&pixels[row * width..][..width])
                {
                    out.copy_from_slice(&[p[0] as u16, (p[0] >> 16) as u16, p[1] as u16]);
                }
            }
            Ok(())
        })();
        // Also cancels a pending mapping when conversion cancellation wins.
        self.readback.unmap();
        result
    }
}

pub(super) fn nlm(
    src: &Img3,
    h: [f64; 3],
    template: i32,
    search: i32,
    control: Control<'_>,
) -> Result<Img3> {
    nlm_on(&GPU, src, h, template, search, control)
}

fn nlm_on(
    shared: &SharedGpu,
    src: &Img3,
    h: [f64; 3],
    template: i32,
    search: i32,
    control: Control<'_>,
) -> Result<Img3> {
    control.check()?;
    if !(1..=9).contains(&template)
        || template % 2 == 0
        || !(1..=31).contains(&search)
        || search % 2 == 0
    {
        return Err(gpu_error("unsupported NLM window"));
    }
    if src.rows == 0 || src.cols == 0 {
        return Ok(Img3::new(src.rows, src.cols));
    }
    with_gpu(shared, control, |gpu| {
        run_nlm_windows(gpu, src, h, template, search, control)
    })
}

pub(super) fn bm3d(src: &Img3, sigma: [f64; 3], control: Control<'_>) -> Result<Img3> {
    with_gpu(&GPU, control, |gpu| {
        bm3d_gpu::denoise(gpu, src, sigma, control)
    })
}

fn with_gpu<T>(
    shared: &SharedGpu,
    control: Control<'_>,
    run: impl FnOnce(&Gpu) -> Result<T>,
) -> Result<T> {
    // ponytail: one GPU call at a time bounds scratch memory; add a resource
    // pool only if measured batch throughput warrants concurrent dispatch.
    let mut guard = loop {
        control.check()?;
        match shared.try_lock() {
            Ok(guard) => break guard,
            Err(TryLockError::Poisoned(_)) => return Err(gpu_error("GPU context poisoned")),
            Err(TryLockError::WouldBlock) => std::thread::sleep(Duration::from_millis(1)),
        }
    };
    // Initialize under the same cancellable lock as dispatch. Other callers
    // can cancel while a driver is creating the first device/pipeline.
    let gpu = guard
        .get_or_insert_with(|| Gpu::new(true))
        .as_ref()
        .map_err(gpu_error)?;
    control.check()?;
    gpu.check()?;
    run(gpu)
}

fn run_nlm(gpu: &Gpu, src: &Img3, h: [f64; 3], search: i32, control: Control<'_>) -> Result<Img3> {
    run_nlm_windows(gpu, src, h, 3, search, control)
}

fn run_nlm_windows(
    gpu: &Gpu,
    src: &Img3,
    h: [f64; 3],
    template: i32,
    search: i32,
    control: Control<'_>,
) -> Result<Img3> {
    gpu.check()?;
    // Above this distance every channel's fixed-point weight is zero. Keep a
    // zero sentinel so clamping larger indices exactly matches the CPU LUT.
    let shift = super::nearest_pow2(template * template);
    let step = (1 << shift) as f64 / (template * template) as f64;
    let cutoff = h.into_iter().fold(0.0_f64, f64::max) * (3.0 * 1000.0_f64.ln()).sqrt();
    let len = ((cutoff / step).ceil() as usize + 2).min((LUT_BYTES / 16) as usize);
    let mut weights = vec![[0_f32; 4]; len];
    for (i, weight) in weights.iter_mut().enumerate() {
        let distance = i as f64 * step;
        for c in 0..3 {
            let value = if h[c] == 0.0 {
                f64::from(i == 0)
            } else {
                (-(distance * distance) / (h[c] * h[c] * 3.0)).exp()
            };
            let fixed = (value * i32::MAX as f64).round_ties_even();
            if fixed >= 0.001 * i32::MAX as f64 {
                weight[c] = (fixed / i32::MAX as f64) as f32;
            }
        }
    }
    gpu.queue
        .write_buffer(&gpu.weights, 0, bytemuck::cast_slice(&weights));
    let mut out = Img3::new(src.rows, src.cols);
    let halo = (search / 2 + template / 2) as usize;
    let mut pixels = Vec::with_capacity((TILE + 2 * halo).pow(2));
    for y in (0..src.rows).step_by(TILE) {
        let height = TILE.min(src.rows - y);
        let rows: Vec<_> = (0..height + 2 * halo)
            .map(|r| reflect_101((y + r) as i32 - halo as i32, src.rows as i32) as usize)
            .collect();
        for x in (0..src.cols).step_by(TILE) {
            control.check()?;
            let width = TILE.min(src.cols - x);
            let cols: Vec<_> = (0..width + 2 * halo)
                .map(|c| reflect_101((x + c) as i32 - halo as i32, src.cols as i32) as usize)
                .collect();
            pixels.clear();
            for &row in &rows {
                for &col in &cols {
                    let p = &src.data[(row * src.cols + col) * 3..];
                    pixels.push([p[0] as u32 | (p[1] as u32) << 16, p[2] as u32]);
                }
            }
            gpu.tile(
                &pixels,
                [
                    (width + 2 * halo) as u32,
                    (height + 2 * halo) as u32,
                    width as u32,
                    height as u32,
                    search as u32 / 2,
                    len as u32,
                    if template == 3 && search <= 21 {
                        0
                    } else {
                        template as u32 / 2 + 1
                    },
                    shift,
                ],
                control,
                &mut out,
                x,
                y,
            )?;
        }
    }
    control.check()?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires a physical GPU"]
    fn configurable_windows_and_bm3d_match_cpu() {
        let gpu = Gpu::new(true).expect("physical GPU");
        let mut seed = 42_u32;
        let mut source = Img3::new(37, 39);
        for pixel in source.data.chunks_exact_mut(3) {
            for value in pixel {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                *value = 32000 + ((seed >> 16) % 800) as u16;
            }
        }
        for (patch, search) in [(1, 1), (1, 7), (5, 7), (7, 15), (9, 31), (3, 31)] {
            let cpu =
                crate::denoise::fast_nlm_cpu(&source, [100.; 3], patch, search, Control::none())
                    .unwrap();
            let actual =
                run_nlm_windows(&gpu, &source, [100.; 3], patch, search, Control::none()).unwrap();
            let error = cpu
                .data
                .iter()
                .zip(actual.data)
                .map(|(a, b)| a.abs_diff(b))
                .max()
                .unwrap();
            assert!(error <= 1, "NLM patch={patch} search={search}: {error}");
        }
        for (rows, cols) in [(1, 1), (3, 5), (37, 39), (259, 263)] {
            let mut source = Img3::new(rows, cols);
            for (i, pixel) in source.data.chunks_exact_mut(3).enumerate() {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                pixel.copy_from_slice(&[i as u16, 32000 + ((seed >> 16) % 800) as u16, 32768]);
            }
            let cpu = crate::denoise::bm3d::denoise_cpu(&source, [0., 230., 0.], Control::none())
                .unwrap();
            let actual = bm3d_gpu::denoise(&gpu, &source, [0., 230., 0.], Control::none()).unwrap();
            let repeated =
                bm3d_gpu::denoise(&gpu, &source, [0., 230., 0.], Control::none()).unwrap();
            assert!(
                actual.data == repeated.data,
                "BM3D must reproduce the same pixels on repeated execution"
            );
            let error = cpu
                .data
                .iter()
                .zip(&actual.data)
                .map(|(a, b)| a.abs_diff(*b))
                .max()
                .unwrap();
            // FP32 transforms and quantized weights can perturb match ordering. Bound
            // both the worst pixel and RMS, rather than requiring bit identity.
            let rms = (cpu
                .data
                .iter()
                .zip(&actual.data)
                .map(|(a, b)| (*a as f64 - *b as f64).powi(2))
                .sum::<f64>()
                / cpu.data.len() as f64)
                .sqrt();
            assert!(
                error <= 8 && rms < 0.6,
                "BM3D {cols}x{rows}: max={error}, RMS={rms}"
            );
            assert!(source
                .data
                .chunks_exact(3)
                .zip(actual.data.chunks_exact(3))
                .all(|(a, b)| a[0] == b[0] && a[2] == b[2]));
        }
    }

    #[test]
    fn failed_or_poisoned_context_returns_an_error_without_a_gpu() {
        let src = Img3::new(7, 13);
        let failed = Mutex::new(Some(Err("injected initialization failure".to_string())));
        let result = nlm_on(&failed, &src, [0.0, 100.0, 100.0], 3, 11, Control::none());
        assert!(
            matches!(result, Err(Error::Gpu(message)) if message == "injected initialization failure")
        );

        let poisoned = Mutex::new(None);
        let _ = std::panic::catch_unwind(|| {
            let _guard = poisoned.lock().unwrap();
            panic!("injected context panic");
        });
        let result = nlm_on(&poisoned, &src, [0.0, 100.0, 100.0], 3, 11, Control::none());
        assert!(matches!(result, Err(Error::Gpu(message)) if message.contains("poisoned")));
    }

    #[test]
    fn cancellation_interrupts_waiting_for_initialization() {
        use std::sync::atomic::{AtomicBool, Ordering};
        let shared = Mutex::new(None);
        let initializing = shared.lock().unwrap();
        let cancel = AtomicBool::new(false);
        let src = Img3::new(7, 13);
        let started = Instant::now();
        std::thread::scope(|scope| {
            scope.spawn(|| {
                std::thread::sleep(Duration::from_millis(2));
                cancel.store(true, Ordering::Relaxed);
            });
            let result = nlm_on(
                &shared,
                &src,
                [0.0, 100.0, 100.0],
                3,
                11,
                Control::new(&cancel),
            );
            assert!(matches!(result, Err(Error::Cancelled)));
        });
        assert!(started.elapsed() < Duration::from_secs(5));
        assert!(initializing.is_none());
    }

    #[test]
    fn shaders_validate_without_a_gpu() {
        use wgpu::naga::{
            front::wgsl,
            valid::{Capabilities, ValidationFlags, Validator},
        };
        for (source, caps) in [
            (include_str!("nlm.wgsl"), Capabilities::empty()),
            (include_str!("nlm_general.wgsl"), Capabilities::empty()),
            (include_str!("bm3d.wgsl"), Capabilities::empty()),
            (include_str!("nlm_subgroup.wgsl"), Capabilities::SUBGROUP),
        ] {
            let module = wgsl::parse_str(source).unwrap();
            Validator::new(ValidationFlags::all(), caps)
                .validate(&module)
                .unwrap();
        }
    }

    #[test]
    #[ignore = "requires a physical GPU; run explicitly on GPU test hosts"]
    fn gpu_portable_shader_matches_cpu() {
        let gpu = Gpu::new(false).unwrap();
        let mut source = Img3::new(1031, 1037);
        for (i, pixel) in source.data.chunks_exact_mut(3).enumerate() {
            pixel.copy_from_slice(&[
                if i % source.cols < source.cols / 2 {
                    500
                } else {
                    65535
                },
                32700 + ((i * 37) % 400) as u16,
                32600 + ((i * 101) % 600) as u16,
            ]);
        }
        for (search, h) in [(11, [0.0, 300.0, 600.0]), (21, [0.0, 37.5, 75.0])] {
            let expected =
                super::super::fast_nlm_cpu(&source, h, 3, search, Control::none()).unwrap();
            let actual = run_nlm(&gpu, &source, h, search, Control::none()).unwrap();
            assert!(expected
                .data
                .iter()
                .zip(&actual.data)
                .all(|(&a, &b)| a.abs_diff(b) <= 1));
        }
    }

    #[test]
    #[ignore = "requires a physical GPU; run explicitly on GPU test hosts"]
    fn gpu_uncaptured_error_falls_back_and_preserves_input() {
        let gpu = Gpu::new(false).unwrap();
        // Invalid MAP_READ usage produces a validation error without allocating
        // a large buffer or relying on a particular driver failure.
        let _ = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("injected invalid usage"),
            size: 16,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        assert!(matches!(gpu.check(), Err(Error::Gpu(_))));
        let mut src = Img3::new(7, 13);
        for (i, value) in src.data.iter_mut().enumerate() {
            *value = 32700 + (i % 101) as u16;
        }
        let original = src.data.clone();
        let h = [0.0, 100.0, 100.0];
        let expected = super::super::fast_nlm_cpu(&src, h, 3, 11, Control::none()).unwrap();
        let actual = super::super::fast_nlm_with_gpu(&src, h, 3, 11, Control::none(), || {
            run_nlm(&gpu, &src, h, 11, Control::none())
        })
        .unwrap();
        assert_eq!(actual.data, expected.data);
        assert_eq!(src.data, original);
        assert!(run_nlm(&gpu, &src, h, 11, Control::none()).is_err());
    }

    #[test]
    #[ignore = "requires a physical GPU; run explicitly on GPU test hosts"]
    fn concurrent_cpu_and_gpu_requests_keep_their_backends() {
        let mut src = Img3::new(31, 43);
        for (i, value) in src.data.iter_mut().enumerate() {
            *value = 32700 + (i % 101) as u16;
        }
        let h = [0.0, 100.0, 100.0];
        std::thread::scope(|scope| {
            let cpu = scope.spawn(|| {
                super::super::fast_nlm_with_gpu(
                    &src,
                    h,
                    3,
                    11,
                    Control::none().with_denoise_backend(crate::DenoiseBackend::Cpu),
                    || panic!("CPU request touched the GPU"),
                )
                .unwrap()
            });
            let gpu = scope.spawn(|| {
                super::super::fast_nlm_denoise_controlled(
                    &src,
                    h,
                    3,
                    11,
                    Control::none().with_denoise_backend(crate::DenoiseBackend::Gpu),
                )
                .unwrap()
            });
            let cpu = cpu.join().unwrap();
            let gpu = gpu.join().unwrap();
            assert!(cpu
                .data
                .iter()
                .zip(gpu.data)
                .all(|(&a, b)| a.abs_diff(b) <= 1));
        });
    }

    #[test]
    #[ignore = "requires a physical GPU; run explicitly on GPU test hosts"]
    fn gpu_readback_can_be_cancelled_and_reused() {
        use std::sync::atomic::{AtomicBool, Ordering};
        let gpu = Gpu::new(false).unwrap();
        let cancel = AtomicBool::new(false);
        let pixels = vec![[12000 | (32768 << 16), 32768]; (TILE + 22).pow(2)];
        gpu.queue
            .write_buffer(&gpu.weights, 0, bytemuck::cast_slice(&[[1_f32; 4]]));
        let mut output = Img3::new(TILE, TILE);
        std::thread::scope(|scope| {
            scope.spawn(|| {
                std::thread::sleep(Duration::from_millis(2));
                cancel.store(true, Ordering::Relaxed);
            });
            let result = gpu.tile(
                &pixels,
                [
                    (TILE + 22) as u32,
                    (TILE + 22) as u32,
                    TILE as u32,
                    TILE as u32,
                    10,
                    1,
                    0,
                    0,
                ],
                Control::new(&cancel),
                &mut output,
                0,
                0,
            );
            assert!(matches!(result, Err(Error::Cancelled)), "{result:?}");
        });
        assert!(output.data.iter().all(|&value| value == 0));
        let mut source = Img3::new(1, 1);
        source.data.copy_from_slice(&[12000, 32768, 32768]);
        let resumed = run_nlm(&gpu, &source, [0.0, 100.0, 100.0], 11, Control::none()).unwrap();
        assert_eq!(resumed.data, source.data);
    }
}
