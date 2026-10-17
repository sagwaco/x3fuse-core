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

#[derive(Clone, Copy)]
struct PlaneStats {
    clipped: f64,
    ratio: Option<f64>,
    bins: [Option<f64>; 8],
}

fn median(values: &mut [f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let even = values.len().is_multiple_of(2);
    let mid = values.len() / 2;
    let (lower, centre, _) = values.select_nth_unstable_by(mid, f64::total_cmp);
    Some(if even {
        (*centre + lower.iter().copied().max_by(f64::total_cmp).unwrap()) / 2.0
    } else {
        *centre
    })
}

fn plane_stats(
    reference: &MergeFrame,
    frame: &MergeFrame,
    plane: usize,
    margin: f64,
    ceiling: f64,
) -> PlaneStats {
    let count = if plane == 2 {
        frame.raw_top.len()
    } else {
        frame.raw_half.len() / 2
    };
    let threshold = (1.0 - margin) * ceiling;
    let bin_for = |ref_value: f64, value: f64| {
        let reference_level = ref_value / ceiling;
        if (0.05..=0.6).contains(&reference_level) && value > 0.0 && value < threshold {
            Some((((reference_level - 0.05) / (0.6 - 0.05) * 8.0) as usize).min(7))
        } else {
            None
        }
    };
    let mut clipped = 0usize;
    let mut counts = [0usize; 8];
    for i in 0..count {
        let value = sample(frame, plane, i);
        clipped += usize::from(value >= threshold);
        if let Some(bin) = bin_for(sample(reference, plane, i), value) {
            counts[bin] += 1;
        }
    }
    let mut offsets = [0usize; 9];
    for bin in 0..8 {
        offsets[bin + 1] = offsets[bin] + counts[bin];
    }
    // Group ratios by bin in one buffer. Reuse it for the global median.
    let mut ratios = vec![0.0; offsets[8]];
    let mut next = offsets;
    for i in 0..count {
        let value = sample(frame, plane, i);
        let ref_value = sample(reference, plane, i);
        if let Some(bin) = bin_for(ref_value, value) {
            ratios[next[bin]] = ref_value / value;
            next[bin] += 1;
        }
    }
    let bins = std::array::from_fn(|bin| median(&mut ratios[offsets[bin]..offsets[bin + 1]]));
    PlaneStats {
        clipped: clipped as f64 / count as f64,
        ratio: median(&mut ratios),
        bins,
    }
}

pub(super) fn radiometry(
    container: &X3iContainer,
    options: &ProcessOptions,
) -> Result<String, String> {
    let mut options = options.clone();
    options.denoise_intensity = 0;
    let mut merge_options =
        x3f_core::x3i_merge::MergeOptions::from_env().map_err(|e| e.to_string())?;
    merge_options.process = options.clone();
    let saturation = merge_options
        .saturation_for_container(container)
        .map_err(|e| e.to_string())?;
    let margin = merge_options.validity_margin as f64;
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
    let mut text = format!("X3I raw-plane report\nplanes: B M T; clipping threshold relative to ceiling={:.9}\nsaturation: B={:.9} M={:.9} T={:.9}\nratio fit: median x_ref/x_i; reference r/ceiling in [0.05,0.60], both valid; no refinement or alignment\nratio bins: 8 equal reference r/ceiling bins; final bin includes 0.60\nindex shutter_us scale sigma_B sigma_M sigma_T clip_B clip_M clip_T ratio_B ratio_M ratio_T\n", 1.0 - margin, saturation[0], saturation[1], saturation[2]);
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
        let stats: [_; 3] = std::array::from_fn(|plane| {
            plane_stats(
                &reference,
                frame,
                plane,
                margin,
                f64::from(saturation[plane]),
            )
        });
        text.push_str(&format!(
            "{i} {} {:.12}",
            frame.shutter_us,
            f64::from(reference.shutter_us) / f64::from(frame.shutter_us)
        ));
        for sigma in frame.sigma {
            text.push_str(&format!(" {sigma:.12}"));
        }
        for stat in stats {
            text.push_str(&format!(" {:.12}", stat.clipped));
        }
        for stat in stats {
            let ratio = stat.ratio;
            text.push_str(
                &ratio
                    .map(|v| format!(" {v:.12}"))
                    .unwrap_or_else(|| " -".into()),
            );
        }
        text.push('\n');
        for bin in 0..8 {
            let lower = 0.05 + (0.6 - 0.05) * bin as f64 / 8.0;
            let upper = 0.05 + (0.6 - 0.05) * (bin + 1) as f64 / 8.0;
            text.push_str(&format!(
                "ratio-bin frame={i} lower={lower:.5} upper={upper:.5}"
            ));
            for stat in stats {
                text.push_str(
                    &stat.bins[bin]
                        .map(|v| format!(" {v:.12}"))
                        .unwrap_or_else(|| " -".into()),
                );
            }
            text.push('\n');
        }
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
            let stats = plane_stats(&reference, &frame, plane, 0.02, 1.0);
            assert_eq!(stats.clipped, 0.25);
            assert!((stats.ratio.unwrap() - 2.0).abs() < 1e-12);
            assert_eq!(stats.bins[0], Some(1.0));
            assert!((stats.bins[3].unwrap() - 3.0).abs() < 1e-12);
        }
        frame.raw_top = vec![60, 110, 160, 10];
        let stats = plane_stats(&reference, &frame, 2, 0.02, 1.0);
        assert_eq!(stats.clipped, 0.0);
        assert!((stats.ratio.unwrap() - 2.5).abs() < 1e-12);
        frame.raw_top = vec![110, 510, 1000, 10];
        let stats = plane_stats(&reference, &frame, 2, 0.02, 0.5);
        assert_eq!(stats.clipped, 0.5);
        assert_eq!(stats.ratio, Some(1.0));
        assert_eq!(stats.bins[2], Some(1.0));
        assert_eq!(stats.bins[7], None);
        frame.raw_top.fill(10);
        let stats = plane_stats(&reference, &frame, 2, 0.02, 1.0);
        assert_eq!(stats.clipped, 0.0);
        assert_eq!(stats.ratio, None);
        assert_eq!(stats.bins, [None; 8]);
        assert!(peak_rss().unwrap().starts_with("peak_rss_bytes="));
    }
}
