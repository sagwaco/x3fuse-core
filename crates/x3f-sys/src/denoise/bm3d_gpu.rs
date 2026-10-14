//! GPU BM3D uses the CPU profile and a global reference grid. Tiles include
//! every reference that can contribute, so overlap aggregation has no seams.
use super::{gpu_error, Gpu};
use crate::denoise::{bm3d, reflect_101, round_sat_u16, Img3};
use crate::{Control, Result};
use std::sync::mpsc;
use std::time::{Duration, Instant};

const TILE: usize = 256;
const HALO: usize = 32;
const INPUT: u64 = ((TILE + 2 * HALO) * (TILE + 2 * HALO) * 4) as u64;
const OUTPUT: u64 = (TILE * TILE * 4) as u64;

pub(super) struct Resources {
    filter: wgpu::ComputePipeline,
    finalize: wgpu::ComputePipeline,
    bindings: wgpu::BindGroup,
    noisy: wgpu::Buffer,
    guide: wgpu::Buffer,
    sums: wgpu::Buffer,
    params: wgpu::Buffer,
    output: wgpu::Buffer,
    readback: wgpu::Buffer,
}

impl Resources {
    pub(super) fn new(gpu: &Gpu) -> std::result::Result<Self, String> {
        let device = &gpu.device;
        device.push_error_scope(wgpu::ErrorFilter::Validation);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("x3f BM3D"),
            source: wgpu::ShaderSource::Wgsl(include_str!("bm3d.wgsl").into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("BM3D buffers"),
            entries: &(0..6)
                .map(|binding| wgpu::BindGroupLayoutEntry {
                    binding,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    count: None,
                    ty: wgpu::BindingType::Buffer {
                        ty: if binding == 4 {
                            wgpu::BufferBindingType::Uniform
                        } else {
                            wgpu::BufferBindingType::Storage {
                                read_only: matches!(binding, 0 | 1 | 3),
                            }
                        },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                })
                .collect::<Vec<_>>(),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("BM3D"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let pipeline = |name| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(name),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(name),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let filter = pipeline("filter_groups");
        let finalize = pipeline("finalize");
        let buffer = |label, size, usage| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage,
                mapped_at_creation: false,
            })
        };
        let storage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let noisy = buffer("BM3D noisy", INPUT, storage);
        let guide = buffer("BM3D guide", INPUT, storage);
        let sums = buffer("BM3D accumulation", OUTPUT * 3, storage);
        let tables = buffer("BM3D transforms", 128 * 4, storage);
        let params = buffer(
            "BM3D params",
            64,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let output = buffer(
            "BM3D output",
            OUTPUT,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let readback = buffer(
            "BM3D readback",
            OUTPUT,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );
        let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("BM3D"),
            layout: &layout,
            entries: &[&noisy, &guide, &sums, &tables, &params, &output]
                .iter()
                .enumerate()
                .map(|(i, buffer)| wgpu::BindGroupEntry {
                    binding: i as u32,
                    resource: buffer.as_entire_binding(),
                })
                .collect::<Vec<_>>(),
        });
        if let Some(error) = pollster::block_on(device.pop_error_scope()) {
            return Err(error.to_string());
        }
        let coefficients: Vec<f32> = bm3d::dct_basis()
            .iter()
            .flatten()
            .chain(bm3d::window())
            .copied()
            .collect();
        gpu.queue
            .write_buffer(&tables, 0, bytemuck::cast_slice(&coefficients));
        gpu.check().map_err(|e| e.to_string())?;
        Ok(Self {
            filter,
            finalize,
            bindings,
            noisy,
            guide,
            sums,
            params,
            output,
            readback,
        })
    }

    fn read(&self, gpu: &Gpu, bytes: u64, control: Control<'_>) -> Result<Vec<f32>> {
        let slice = self.readback.slice(..bytes);
        let (send, recv) = mpsc::sync_channel(1);
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = send.send(result);
        });
        let started = Instant::now();
        let result = (|| {
            loop {
                control.check()?;
                gpu.check()?;
                match gpu.device.poll(wgpu::PollType::Wait {
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
                        return Err(gpu_error("BM3D readback disconnected"))
                    }
                    Err(mpsc::TryRecvError::Empty) => {}
                }
                if started.elapsed() > Duration::from_secs(30) {
                    *gpu.failure.lock().unwrap_or_else(|e| e.into_inner()) =
                        Some("BM3D GPU readback timed out".into());
                    return Err(gpu_error("BM3D GPU readback timed out"));
                }
            }
            gpu.check()?;
            Ok(bytemuck::cast_slice::<u8, f32>(&slice.get_mapped_range()).to_vec())
        })();
        self.readback.unmap();
        result
    }

    fn stage(
        &self,
        gpu: &Gpu,
        noisy: &[f32],
        guide: &[f32],
        width: usize,
        height: usize,
        sigma: f32,
        wiener: bool,
        control: Control<'_>,
    ) -> Result<Vec<f32>> {
        let mut output = vec![0.0; width * height];
        let mut tile_noisy = Vec::with_capacity(INPUT as usize / 4);
        let mut tile_guide = Vec::with_capacity(INPUT as usize / 4);
        for y in (0..height).step_by(TILE) {
            let h = (height - y).min(TILE);
            for x in (0..width).step_by(TILE) {
                control.check()?;
                gpu.check()?;
                let w = (width - x).min(TILE);
                tile_noisy.clear();
                tile_guide.clear();
                let cols: Vec<usize> = (0..w + 2 * HALO)
                    .map(|i| reflect_101((x + i) as i32 - HALO as i32, width as i32) as usize)
                    .collect();
                for row in 0..h + 2 * HALO {
                    let sy = reflect_101((y + row) as i32 - HALO as i32, height as i32) as usize;
                    for &sx in &cols {
                        tile_noisy.push(noisy[sy * width + sx]);
                        tile_guide.push(guide[sy * width + sx]);
                    }
                }
                let rx = x.saturating_sub(15).div_ceil(4) * 4;
                let ry = y.saturating_sub(15).div_ceil(4) * 4;
                let nx = ((x + w + 8).min(width) - rx).div_ceil(4);
                let ny = ((y + h + 8).min(height) - ry).div_ceil(4);
                let params = [
                    (w + 2 * HALO) as u32,
                    width as u32,
                    height as u32,
                    w as u32,
                    h as u32,
                    x as u32,
                    y as u32,
                    rx as u32,
                    ry as u32,
                    nx as u32,
                    ny as u32,
                    u32::from(wiener),
                    sigma.to_bits(),
                    0,
                    0,
                    0,
                ];
                gpu.queue
                    .write_buffer(&self.noisy, 0, bytemuck::cast_slice(&tile_noisy));
                gpu.queue
                    .write_buffer(&self.guide, 0, bytemuck::cast_slice(&tile_guide));
                gpu.queue
                    .write_buffer(&self.params, 0, bytemuck::cast_slice(&params));
                let mut encoder = gpu
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
                encoder.clear_buffer(&self.sums, 0, None);
                {
                    let mut pass =
                        encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                    pass.set_pipeline(&self.filter);
                    pass.set_bind_group(0, &self.bindings, &[]);
                    pass.dispatch_workgroups((nx * ny) as u32, 1, 1);
                }
                {
                    let mut pass =
                        encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                    pass.set_pipeline(&self.finalize);
                    pass.set_bind_group(0, &self.bindings, &[]);
                    pass.dispatch_workgroups(((w * h) as u32).div_ceil(256), 1, 1);
                }
                let bytes = (w * h * 4) as u64;
                encoder.copy_buffer_to_buffer(&self.output, 0, &self.readback, 0, bytes);
                gpu.queue.submit([encoder.finish()]);
                let values = self.read(gpu, bytes, control)?;
                for row in 0..h {
                    output[(y + row) * width + x..][..w].copy_from_slice(&values[row * w..][..w]);
                }
            }
        }
        control.check()?;
        Ok(output)
    }
}

pub(super) fn denoise(
    gpu: &Gpu,
    src: &Img3,
    sigma: [f64; 3],
    control: Control<'_>,
) -> Result<Img3> {
    let resources = gpu
        .bm3d
        .get_or_init(|| Resources::new(gpu))
        .as_ref()
        .map_err(gpu_error)?;
    let mut out = src.clone();
    for (channel, sigma) in sigma.into_iter().enumerate() {
        control.check()?;
        if sigma == 0.0 {
            continue;
        }
        let noisy: Vec<f32> = src
            .data
            .chunks_exact(3)
            .map(|p| p[channel] as f32)
            .collect();
        let basic = resources.stage(
            gpu,
            &noisy,
            &noisy,
            src.cols,
            src.rows,
            sigma as f32,
            false,
            control,
        )?;
        let final_image = resources.stage(
            gpu,
            &noisy,
            &basic,
            src.cols,
            src.rows,
            sigma as f32,
            true,
            control,
        )?;
        for (pixel, value) in out.data.chunks_exact_mut(3).zip(final_image) {
            pixel[channel] = round_sat_u16(value as f64);
        }
    }
    control.check()?;
    Ok(out)
}
