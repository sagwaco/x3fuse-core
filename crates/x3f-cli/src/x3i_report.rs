//! Raw-plane diagnostics for the phase 1 X3I merge.

use x3f_core::{MergeFrame, ProcessOptions, X3iContainer};

fn sample(frame: &MergeFrame, plane: usize, index: usize) -> f64 {
    let value = if plane == 2 {
        frame.raw_top[index]
    } else {
        frame.raw_half[index * 2 + plane]
    };
    (f64::from(value) - frame.raw_black[plane]) / (frame.raw_white[plane] - frame.raw_black[plane])
}

fn plane_stats(
    reference: &MergeFrame,
    frame: &MergeFrame,
    plane: usize,
    margin: f64,
) -> (f64, Option<f64>) {
    let count = if plane == 2 {
        frame.raw_top.len()
    } else {
        frame.raw_half.len() / 2
    };
    let mut clipped = 0usize;
    let mut ratios = Vec::new();
    for i in 0..count {
        let value = sample(frame, plane, i);
        clipped += usize::from(value >= 1.0 - margin);
        let ref_value = sample(reference, plane, i);
        if (0.05..=0.6).contains(&ref_value) && value > 0.0 && value < 1.0 - margin {
            ratios.push(ref_value / value);
        }
    }
    let median = if ratios.is_empty() {
        None
    } else {
        let even = ratios.len() % 2 == 0;
        let mid = ratios.len() / 2;
        let (lower, centre, _) = ratios.select_nth_unstable_by(mid, f64::total_cmp);
        Some(if even {
            (*centre + lower.iter().copied().max_by(f64::total_cmp).unwrap()) / 2.0
        } else {
            *centre
        })
    };
    (clipped as f64 / count as f64, median)
}

pub(super) fn radiometry(
    container: &X3iContainer,
    options: &ProcessOptions,
) -> Result<String, String> {
    let mut options = options.clone();
    options.denoise_intensity = 0;
    let margin = x3f_core::x3i_merge::MergeOptions::from_env()
        .map_err(|e| e.to_string())?
        .validity_margin as f64;
    let load = |i| -> Result<MergeFrame, String> {
        let mut frame = container
            .open_frame(i)
            .map_err(|e| e.to_string())?
            .merge_frame(&options, Default::default())
            .map_err(|e| e.to_string())?;
        frame.bmt = Vec::new();
        frame.gain = None;
        Ok(frame)
    };
    let reference = load(0)?;
    let mut text = format!("X3I raw-plane report\nplanes: B M T; clipping threshold={:.9}\nratio fit: median x_ref/x_i; reference in [0.05,0.60], both valid; no refinement or alignment\nindex shutter_us scale sigma_B sigma_M sigma_T clip_B clip_M clip_T ratio_B ratio_M ratio_T\n", 1.0 - margin);
    for i in 0..container.frames().len() {
        let owned;
        let frame = if i == 0 {
            &reference
        } else {
            owned = load(i)?;
            &owned
        };
        if (frame.width, frame.height) != (reference.width, reference.height) {
            return Err("X3I report frame dimensions differ".into());
        }
        let stats: [_; 3] =
            std::array::from_fn(|plane| plane_stats(&reference, frame, plane, margin));
        text.push_str(&format!(
            "{i} {} {:.12}",
            frame.shutter_us,
            f64::from(reference.shutter_us) / f64::from(frame.shutter_us)
        ));
        for sigma in frame.sigma {
            text.push_str(&format!(" {sigma:.12}"));
        }
        for (clipped, _) in stats {
            text.push_str(&format!(" {clipped:.12}"));
        }
        for (_, ratio) in stats {
            text.push_str(
                &ratio
                    .map(|v| format!(" {v:.12}"))
                    .unwrap_or_else(|| " -".into()),
            );
        }
        text.push('\n');
    }
    Ok(text)
}

pub(super) fn peak_rss() -> Result<String, String> {
    #[cfg(unix)]
    {
        // SAFETY: getrusage initializes this structure on success.
        let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
        if unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) } != 0 {
            return Err(std::io::Error::last_os_error().to_string());
        }
        let bytes = usage.ru_maxrss as u64;
        #[cfg(not(target_os = "macos"))]
        let bytes = bytes * 1024;
        Ok(format!(
            "peak_rss_bytes={bytes} (process high-water mark)\n"
        ))
    }
    #[cfg(not(unix))]
    {
        Ok("peak_rss_bytes=unavailable\n".into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raw_ratios_use_black_subtraction_validity_and_exact_median() {
        let reference = MergeFrame {
            raw_black: [10.0; 3],
            raw_white: [1010.0; 3],
            raw_top: vec![110, 310, 1010, 10],
            raw_half: vec![110, 110, 310, 310, 1010, 1010, 10, 10],
            ..Default::default()
        };
        let mut frame = reference.clone();
        frame.raw_top = vec![110, 110, 1010, 10];
        frame.raw_half = vec![110, 110, 110, 110, 1010, 1010, 10, 10];
        for plane in 0..3 {
            let (clipped, ratio) = plane_stats(&reference, &frame, plane, 0.02);
            assert_eq!(clipped, 0.25);
            assert!((ratio.unwrap() - 2.0).abs() < 1e-12);
        }
        frame.raw_top = vec![60, 110, 160, 10];
        let (clipped, ratio) = plane_stats(&reference, &frame, 2, 0.02);
        assert_eq!(clipped, 0.0);
        assert!((ratio.unwrap() - 2.5).abs() < 1e-12);
        frame.raw_top.fill(10);
        assert_eq!(plane_stats(&reference, &frame, 2, 0.02), (0.0, None));
        assert!(peak_rss().unwrap().starts_with("peak_rss_bytes="));
    }
}
