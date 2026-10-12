//! Scene chroma estimates for the native Merrill highlight stage.

use super::{chroma_lut_defaults, chroma_lut_proposal, chroma_lut_t, env_atoi, env_present};
use crate::{Control, Error, Result};

struct Proposal {
    offset: usize,
    strength: f64,
    value: f64,
}

pub(crate) struct Refinement {
    proposals: Vec<Proposal>,
    next: usize,
}

impl Refinement {
    /// Learn from samples before native highlight restoration changes them.
    /// `white` has the same black, gain, spatial and radial transforms as `data`.
    /// `sat` is the original camera mask; `prior` uses gained sensor units.
    pub(crate) fn new(
        data: &[f32],
        white: &[f32],
        size: [usize; 2],
        stride: usize,
        sat: &[u8],
        valid: [i32; 4],
        prior: [f64; 3],
        control: Control<'_>,
    ) -> Result<Self> {
        control.check()?;
        let [width, height] = size;
        let samples = width
            .checked_mul(height)
            .and_then(|v| v.checked_mul(3))
            .ok_or(Error::InvalidData("invalid SPP LUT image size"))?;
        let length = height.checked_mul(stride).ok_or(Error::Allocation)?;
        let [left, top, right, bottom] = valid;
        if width < 4
            || height < 4
            || stride < width * 3
            || data.len() < length
            || white.len() < length
            || sat.len() != samples
            || left < 0
            || top < 0
            || right < left
            || bottom < top
            || right as usize >= width - 2
            || bottom as usize >= height - 2
            || !prior.iter().all(|v| v.is_finite() && *v > 0.)
        {
            return Err(Error::InvalidData("invalid SPP LUT input"));
        }
        let mut lut: chroma_lut_t = unsafe { std::mem::zeroed() };
        chroma_lut_defaults(&mut lut, true);
        if ![
            lut.sat_threshold,
            lut.recovery_cap,
            lut.asymmetric_max,
            lut.soft_window,
            lut.blend_divisor,
        ]
        .iter()
        .all(|v| v.is_finite() && *v > 0.)
            || !lut.donor_min_brightness.is_finite()
            || !lut.blend_threshold.is_finite()
        {
            return Err(Error::InvalidData("invalid SPP LUT settings"));
        }
        let normalized = |offset: usize| -> Result<[f64; 3]> {
            let mut out = [0.; 3];
            for c in 0..3 {
                let scale = f64::from(white[offset + c]);
                if !scale.is_finite() || scale <= 0. {
                    return Err(Error::InvalidData("invalid SPP LUT white level"));
                }
                out[c] = f64::from(data[offset + c]) / scale;
            }
            Ok(out)
        };
        let mut sums = [0.; 256];
        let mut counts = [0_usize; 256];
        for y in top as usize..=bottom as usize {
            control.check()?;
            for x in left as usize..=right as usize {
                let s = normalized(y * stride + x * 3)?;
                let flags = &sat[3 * (y * width + x)..][..3];
                if flags.iter().all(|&v| v == 0)
                    && s.iter()
                        .all(|v| v.is_finite() && *v > 0. && *v < lut.sat_threshold)
                    && s[0] + s[1] > lut.donor_min_brightness
                {
                    let bin = bin(s);
                    sums[bin] += s[2] / s[1];
                    counts[bin] += 1;
                }
            }
        }
        let radius = env_atoi("X3F_CHROMA_LUT_FILL_DIST")
            .unwrap_or(16)
            .clamp(0, 255);
        let mut supported = [false; 256];
        for bin in 0..256 {
            let donor = (0..=radius).find_map(|distance| {
                [bin as i32 - distance, bin as i32 + distance]
                    .into_iter()
                    .find(|&i| (0..256).contains(&i) && counts[i as usize] != 0)
            });
            if let Some(donor) = donor {
                lut.lut[bin] = (sums[donor as usize] / counts[donor as usize] as f64) as f32;
                supported[bin] = true;
            }
        }
        lut.valid = 1;
        let mut proposals = Vec::new();
        for y in top as usize..=bottom as usize {
            control.check()?;
            for x in left as usize..=right as usize {
                let offset = y * stride + x * 3;
                let s = normalized(offset)?;
                let flags = &sat[3 * (y * width + x)..][..3];
                if flags[0] != 0
                    || flags[1] != 0
                    || flags[2] == 0
                    || !s.iter().all(|v| v.is_finite() && *v > 0.)
                    || !supported[bin(s)]
                {
                    continue;
                }
                // The neutral direction must use the same local units as the samples.
                lut.neutral_tm = (prior[2] / f64::from(white[offset + 2]))
                    / (prior[1] / f64::from(white[offset + 1]));
                if !lut.neutral_tm.is_finite() || lut.neutral_tm <= 0. {
                    return Err(Error::InvalidData("invalid SPP LUT neutral direction"));
                }
                if let Some((strength, recovered)) =
                    unsafe { chroma_lut_proposal(s, &lut, std::ptr::null_mut()) }
                {
                    let value = (recovered * f64::from(white[offset + 2])).min(32767.);
                    if !strength.is_finite() || !value.is_finite() {
                        return Err(Error::InvalidData("invalid SPP LUT estimate"));
                    }
                    proposals.try_reserve(1).map_err(|_| Error::Allocation)?;
                    proposals.push(Proposal {
                        offset: offset + 2,
                        strength,
                        value,
                    });
                }
            }
        }
        if env_present("X3F_CHROMA_LUT_TRACE") {
            eprintln!(
                "SPP chroma LUT: donors={} bins={} proposals={}",
                counts.iter().sum::<usize>(),
                counts.iter().filter(|&&v| v != 0).count(),
                proposals.len()
            );
        }
        Ok(Self { proposals, next: 0 })
    }

    /// Native restoration visits these offsets in raster order.
    pub(crate) fn apply(&mut self, offset: usize, baseline: f32) -> f32 {
        // Saturation cleanup can remove an earlier proposal from the native pass.
        while self
            .proposals
            .get(self.next)
            .is_some_and(|p| p.offset < offset)
        {
            self.next += 1;
        }
        let Some(proposal) = self.proposals.get(self.next).filter(|p| p.offset == offset) else {
            return baseline;
        };
        self.next += 1;
        ((1. - proposal.strength) * f64::from(baseline) + proposal.strength * proposal.value) as f32
    }
}

fn bin(s: [f64; 3]) -> usize {
    ((s[0] * 255. / (s[0] + s[1])) as usize).min(255)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SIZE: [usize; 2] = [8, 8];
    const STRIDE: usize = 27;
    const VALID: [i32; 4] = [2, 2, 5, 5];
    const TARGET: usize = 3 * STRIDE + 3 * 3;

    fn fixture(local_white: bool) -> (Vec<f32>, Vec<f32>, Vec<u8>) {
        let mut data = vec![-12345.; STRIDE * SIZE[1]];
        let mut white = data.clone();
        let mut sat = vec![0; SIZE[0] * SIZE[1] * 3];
        for y in 0..SIZE[1] {
            for x in 0..SIZE[0] {
                let offset = y * STRIDE + x * 3;
                for c in 0..3 {
                    white[offset + c] = if local_white {
                        [4096., 2048., 8192.][c] * (1. + (x + y) as f32 / 16.)
                    } else {
                        4000.
                    };
                    data[offset + c] = [0.25, 0.25, 0.75][c] * white[offset + c];
                }
            }
        }
        for x in [3, 4] {
            let offset = 3 * STRIDE + x * 3;
            for c in 0..3 {
                data[offset + c] = [0.5, 0.5, 1.][c] * white[offset + c];
            }
            sat[3 * (3 * SIZE[0] + x) + 2] = 1;
        }
        (data, white, sat)
    }

    fn prepare(data: &[f32], white: &[f32], sat: &[u8]) -> Refinement {
        Refinement::new(
            data,
            white,
            SIZE,
            STRIDE,
            sat,
            VALID,
            [1.; 3],
            Control::none(),
        )
        .unwrap()
    }

    #[test]
    fn estimate_uses_local_white_and_does_not_read_padding() {
        for local_white in [false, true] {
            let (data, white, sat) = fixture(local_white);
            let mut refinement = prepare(&data, &white, &sat);
            assert_eq!(refinement.proposals.len(), 2);
            assert_eq!(refinement.apply(TARGET + 2, 4000.), white[TARGET + 2] * 1.5);
        }
    }

    #[test]
    fn partial_strength_blends_once_from_the_spp_estimate() {
        let (mut data, white, sat) = fixture(false);
        data[TARGET + 2] = 3400.;
        let mut refinement = prepare(&data, &white, &sat);
        assert_eq!(refinement.apply(TARGET + 2, 5000.), 5300.);
    }

    #[test]
    fn saturation_cleanup_does_not_block_later_proposals() {
        let (mut data, white, mut sat) = fixture(false);
        let isolated = STRIDE + 3;
        for c in 0..3 {
            data[isolated + c] = [0.5, 0.5, 1.][c] * white[isolated + c];
        }
        sat[3 * (SIZE[0] + 1) + 2] = 1;
        let mut refinement = Refinement::new(
            &data,
            &white,
            SIZE,
            STRIDE,
            &sat,
            [0, 0, 5, 5],
            [1.; 3],
            Control::none(),
        )
        .unwrap();
        assert_eq!(refinement.proposals.len(), 3);
        let mut dense = vec![0; 64];
        let stats = crate::float_front::HighlightRestoration::new(0., 0.)
            .restore_with_refinement(
                &mut data,
                8,
                8,
                STRIDE,
                &mut sat,
                &mut dense,
                [0, 0, 5, 5],
                Control::none(),
                |offset, value| refinement.apply(offset, value),
            )
            .unwrap();
        assert_eq!(stats.removed, 1);
        assert_eq!(data[isolated + 2], 4000.);
        assert_eq!(data[TARGET + 2], 6000.);
        assert_eq!(data[TARGET + 5], 6000.);
    }

    #[test]
    fn partial_refinement_receives_the_native_fill() {
        let (mut data, mut white, mut sat) = fixture(false);
        for y in 0..8 {
            for x in 0..8 {
                let offset = y * STRIDE + x * 3;
                white[offset..offset + 3].fill(10000.);
                data[offset..offset + 3].copy_from_slice(&[3000., 3000., 9000.]);
            }
        }
        for x in [3, 4] {
            let offset = 3 * STRIDE + 3 * x;
            data[offset..offset + 3].copy_from_slice(&[5000., 5000., 8500.]);
        }
        for x in [2, 3] {
            data[2 * STRIDE + 3 * x + 2] = 3000.;
            sat[3 * (2 * SIZE[0] + x)] = 1;
        }
        let mut refinement = prepare(&data, &white, &sat);
        let proposal = &refinement.proposals[0];
        assert_eq!(proposal.offset, TARGET + 2);
        assert!(proposal.strength > 0. && proposal.strength < 1.);
        let (strength, learned) = (proposal.strength, proposal.value);
        let mut observed = None;
        let stats = crate::float_front::HighlightRestoration::new(0., 0.)
            .restore_with_refinement(
                &mut data,
                8,
                8,
                STRIDE,
                &mut sat,
                &mut [0; 64],
                VALID,
                Control::none(),
                |offset, baseline| {
                    let value = refinement.apply(offset, baseline);
                    if offset == TARGET + 2 {
                        observed = Some((baseline, value));
                    }
                    value
                },
            )
            .unwrap();
        assert!(stats.have_lut);
        let (native, refined) = observed.unwrap();
        assert_ne!(native, 8500.);
        assert_ne!(f64::from(native), learned);
        assert_eq!(
            refined,
            ((1. - strength) * f64::from(native) + strength * learned) as f32
        );
    }

    #[test]
    fn no_evidence_or_reliable_pair_leaves_the_baseline() {
        let (data, white, sat) = fixture(false);
        for mode in 0..4 {
            let mut data = data.clone();
            let mut flags = sat.clone();
            match mode {
                0 => flags.fill(1),
                1 => flags.fill(0),
                2 => {
                    for y in 0..8 {
                        for x in 0..8 {
                            data[y * STRIDE + x * 3 + 2] = data[y * STRIDE + x * 3 + 1];
                        }
                    }
                    data[TARGET + 2] = 4000.;
                }
                _ => {
                    data[TARGET] = 500.;
                    data[TARGET + 1] = 3000.;
                }
            }
            let mut refinement = prepare(&data, &white, &flags);
            assert_eq!(
                refinement.apply(TARGET + 2, 4321.25),
                4321.25,
                "mode {mode}"
            );
        }
    }

    #[test]
    fn invalid_layout_white_and_cancellation_return_errors() {
        let (data, mut white, sat) = fixture(false);
        assert!(Refinement::new(
            &data,
            &white,
            SIZE,
            23,
            &sat,
            VALID,
            [1.; 3],
            Control::none()
        )
        .is_err());
        white[2 * STRIDE + 6] = 0.;
        assert!(Refinement::new(
            &data,
            &white,
            SIZE,
            STRIDE,
            &sat,
            VALID,
            [1.; 3],
            Control::none()
        )
        .is_err());
        let cancel = std::sync::atomic::AtomicBool::new(true);
        assert!(matches!(
            Refinement::new(
                &data,
                &white,
                SIZE,
                STRIDE,
                &sat,
                VALID,
                [1.; 3],
                Control::new(&cancel)
            ),
            Err(Error::Cancelled)
        ));
    }

    #[test]
    fn refinement_keeps_native_neutralization_and_empty_refinement_is_exact() {
        let (data, white, mut sat) = fixture(false);
        sat[3 * (4 * SIZE[0] + 3)] = 1;
        sat[3 * (4 * SIZE[0] + 3) + 1] = 1;
        sat[3 * (4 * SIZE[0] + 4)] = 1;
        sat[3 * (4 * SIZE[0] + 4) + 1] = 1;
        let run = |mut image: Vec<f32>, mut refinement: Option<Refinement>| {
            let mut flags = sat.clone();
            let mut dense = vec![0; 64];
            crate::float_front::HighlightRestoration::new(0., 0.)
                .restore_with_refinement(
                    &mut image,
                    8,
                    8,
                    STRIDE,
                    &mut flags,
                    &mut dense,
                    VALID,
                    Control::none(),
                    |offset, value| {
                        refinement
                            .as_mut()
                            .map_or(value, |r| r.apply(offset, value))
                    },
                )
                .unwrap();
            image
        };
        let refinement = prepare(&data, &white, &sat);
        let mut expected = data.clone();
        for p in &refinement.proposals {
            expected[p.offset] = p.value as f32;
        }
        let actual = run(data.clone(), Some(refinement));
        assert_eq!(actual, run(expected, None));
        assert!(
            actual[TARGET + 2] < 6000.,
            "native dense-region protection was bypassed"
        );
        assert_ne!(actual, run(data.clone(), None));
        let empty = Refinement {
            proposals: Vec::new(),
            next: 0,
        };
        assert_eq!(run(data.clone(), Some(empty)), run(data, None));
    }
}
