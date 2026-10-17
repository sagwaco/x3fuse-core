//! Per-frame merge export for X3I conversion.

use crate::{sys, Error, ProcessOptions, Reader};

impl Reader {
    /// Prepare one Quattro exposure. Call this once on a fresh reader.
    /// This consumes the decoded RAW samples. The export uses the recorded WB.
    /// This ignores `wb`, `crop`, `color_mode`, and `dng_highlight_recovery`.
    /// Normalized camera samples do not depend on the WB override. The writer applies WB.
    pub fn merge_frame(
        &self,
        options: &ProcessOptions,
        control: sys::Control<'_>,
    ) -> Result<sys::MergeFrame, Error> {
        let control = self.with_control(control);
        control.check()?;
        if options.spp_denoise || options.denoise_intensity > 0 {
            return Err(Error::InvalidData(
                "merge export requires denoise off".into(),
            ));
        }
        // SAFETY: this reader owns the sections throughout preparation.
        unsafe {
            let x3f = self.x3f.as_ptr();
            let prop = sys::x3f_get_prop(x3f);
            if !prop.is_null() {
                sys::load_data(x3f, prop, control)?;
            }
            sys::load_data(x3f, sys::x3f_get_camf(x3f), control)?;
            sys::load_data(x3f, sys::x3f_get_raw(x3f), control)?;
            sys::get_merge_frame_controlled(
                x3f,
                options.fix_bad,
                self.resolve_sgain(options.apply_sgain) != 0,
                control,
            )
            .map_err(Into::into)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{path::PathBuf, sync::atomic::AtomicBool};

    fn corpus_file(name: &str) -> Option<PathBuf> {
        let root = std::env::var_os("X3F_TEST_FILES")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../x3f_test_files")
            });
        let path = root.join(name);
        if path.is_file() {
            Some(path)
        } else {
            eprintln!("skip: {name} is required for merge export coverage");
            None
        }
    }

    #[test]
    fn quattro_merge_contract_and_public_linear_active_region() {
        let Some(path) = corpus_file("_SDI8284.X3F") else {
            return;
        };
        let options = ProcessOptions {
            denoise_intensity: 0,
            apply_sgain: Some(false),
            wb: Some("ignored merge WB".into()),
            color_mode: Some(crate::ColorMode::Vivid),
            dng_highlight_recovery: true,
            ..ProcessOptions::default()
        };
        let mut reader = Reader::open(&path).unwrap();
        reader.load_property_list().unwrap();
        reader.load_camf().unwrap();
        reader.load_raw().unwrap();
        // Force known CAMF-listed defects high. Repair must remove their clipping evidence.
        unsafe {
            let mut top = std::mem::zeroed();
            let mut half = std::mem::zeroed();
            assert_ne!(sys::x3f_image_area_qtop(reader.x3f.as_ptr(), &mut top), 0);
            assert_ne!(sys::x3f_image_area(reader.x3f.as_ptr(), &mut half), 0);
            assert!(top.rows > 29 && top.columns > 913);
            assert!(half.rows > 14 && half.columns > 311 && half.channels >= 2);
            *top.data
                .add(29 * top.row_stride as usize + 913 * top.channels as usize) = 16383;
            for c in 0..2 {
                *half
                    .data
                    .add(14 * half.row_stride as usize + 311 * half.channels as usize + c) = 16383;
            }
        }
        let frame = reader.merge_frame(&options, sys::Control::none()).unwrap();
        assert_eq!(frame.frame_index, 0);
        assert_eq!(frame.width, 2 * frame.raw_half_width);
        assert_eq!(frame.height, 2 * frame.raw_half_height);
        let pixels = frame.width as usize * frame.height as usize;
        assert_eq!(frame.bmt.len(), pixels * 3);
        assert_eq!(frame.raw_top.len(), pixels);
        assert_eq!(frame.raw_half.len(), pixels / 2);
        assert!(frame.gain.is_none());
        assert!((frame.raw_top[29 * frame.width as usize + 913] as f64) < frame.raw_white[2]);
        for c in 0..2 {
            let count = frame.raw_half[(14 * frame.raw_half_width as usize + 311) * 2 + c];
            assert!((count as f64) < frame.raw_white[c]);
        }
        assert!(frame.bmt.iter().all(|v| v.is_finite()));
        assert!(frame.sigma.iter().all(|v| v.is_finite() && *v >= 0.0));
        for c in 0..3 {
            assert!(frame.raw_black[c] < frame.raw_white[c]);
        }
        assert!(frame.raw_black.iter().all(|v| v.is_finite()));
        assert!(frame.raw_white.iter().all(|v| v.is_finite()));
        assert_eq!(frame.shutter_us, 17539);
        assert_eq!(reader.dng_camf_float("CaptureShutter"), Some(17539.0));
        assert_eq!(
            reader.dng_camf_float("CaptureExpTime").unwrap().round(),
            258938.0
        );
        let meta_path =
            std::env::temp_dir().join(format!("x3f-merge-meta-{}.txt", std::process::id()));
        reader.dump_meta(&meta_path).unwrap();
        let meta = std::fs::read_to_string(&meta_path).unwrap();
        std::fs::remove_file(&meta_path).unwrap();
        let shutter_block = meta
            .split("BEGIN: CAMF matrix meta data (CaptureShutter)")
            .nth(1)
            .unwrap()
            .split("END:")
            .next()
            .unwrap();
        assert_eq!(
            shutter_block
                .split_whitespace()
                .last()
                .unwrap()
                .parse::<u32>()
                .unwrap(),
            frame.shutter_us
        );

        let mut area = sys::x3f_area16_t {
            rows: frame.height,
            columns: frame.width,
            ..unsafe { std::mem::zeroed() }
        };
        let mut rect = [0; 4];
        assert_ne!(
            unsafe {
                sys::x3f_get_camf_rect(
                    reader.x3f.as_ptr(),
                    c"ActiveImageArea".as_ptr() as *mut _,
                    &mut area,
                    0,
                    rect.as_mut_ptr(),
                )
            },
            0
        );
        let mut source = Reader::open(&path).unwrap();
        source.load_property_list().unwrap();
        source.load_camf().unwrap();
        source.load_raw().unwrap();
        let linear = unsafe {
            sys::get_scene_linear_controlled(
                source.x3f.as_ptr(),
                options.fix_bad,
                0,
                false,
                false,
                sys::Control::none(),
            )
        }
        .unwrap();
        assert_eq!(
            (linear.width, linear.height),
            (rect[2] - rect[0] + 1, rect[3] - rect[1] + 1)
        );
        for row in 0..linear.height as usize {
            for col in 0..linear.width as usize {
                let off =
                    ((row + rect[1] as usize) * frame.width as usize + col + rect[0] as usize) * 3;
                let source = &frame.bmt[off..off + 3];
                let expected = &linear.data[(row * linear.width as usize + col) * 3..][..3];
                for (c, expected) in expected.iter().enumerate() {
                    let terms = std::array::from_fn::<_, 3, _>(|k| {
                        frame.camera_to_rgb[3 * c + k] * source[k] as f64
                    });
                    let rgb: f64 = terms.iter().sum();
                    let scale = terms.iter().map(|v| v.abs()).sum::<f64>().max(1e-6);
                    assert!(
                        (rgb - *expected as f64).abs()
                            <= 1e-4 * (*expected as f64).abs() + 4.0 * f32::EPSILON as f64 * scale
                    );
                }
            }
        }
    }

    #[test]
    fn every_x3i_frame_applies_its_own_window() {
        for name in ["DP0Q0002.X3I", "_DQH0978.X3I"] {
            let Some(path) = corpus_file(name) else {
                continue;
            };
            let container = crate::X3iContainer::open(path).unwrap();
            let options = ProcessOptions {
                denoise_intensity: 0,
                ..Default::default()
            };
            for i in 0..container.frames().len() {
                let frame = container
                    .open_frame(i)
                    .unwrap()
                    .merge_frame(&options, sys::Control::none())
                    .unwrap();
                assert_eq!(
                    frame.shutter_us,
                    container.frame_exposure_us(i).unwrap(),
                    "{name} frame {i}"
                );
                assert_eq!(frame.raw_half_width * 2, frame.width);
                if i == 0 {
                    let mut reader = container.open_frame(i).unwrap();
                    reader.load_camf().unwrap();
                    let shutter = reader.dng_camf_float("CaptureShutter").unwrap();
                    let exp_time = reader.dng_camf_float("CaptureExpTime").unwrap();
                    eprintln!("{name}: CaptureShutter={shutter} CaptureExpTime={exp_time}");
                    assert_eq!(shutter.round() as u32, frame.shutter_us);
                    assert_ne!(shutter, exp_time);
                }
            }
        }
    }

    #[test]
    fn merge_rejects_merrill_denoise_and_cancelled_work() {
        let Some(path) = corpus_file("_SDI8040.X3F") else {
            return;
        };
        let reader = Reader::open(path).unwrap();
        let mut options = ProcessOptions::default();
        assert!(matches!(
            reader.merge_frame(&options, sys::Control::none()),
            Err(Error::InvalidData(_))
        ));
        options.denoise_intensity = 0;
        options.spp_denoise = true;
        assert!(matches!(
            reader.merge_frame(&options, sys::Control::none()),
            Err(Error::InvalidData(_))
        ));
        options.spp_denoise = false;
        let cancel = AtomicBool::new(true);
        assert!(matches!(
            reader.merge_frame(&options, sys::Control::new(&cancel)),
            Err(Error::Cancelled)
        ));
        assert!(matches!(
            reader.merge_frame(&options, sys::Control::none()),
            Err(Error::InvalidData(_))
        ));
    }
}
