//! `ProfileToneCurve` synthesis from Sigma's `CMCC_<mode>` parameter.
//!
//! Each in-camera color mode has a corresponding `CMCC_<mode>` CAMF entry
//! that's a single float in `[-0.3, +0.3]` describing how strongly the
//! contrast should bend off identity. The reference Sigma-aware DNG build
//! ("CPP") turns this float into a 256-point `ProfileToneCurve` —
//! anchored at (0, 0) and (1, 1), bending the midtones up for positive
//! values (Vivid / Landscape / FCBlue) and down for negative values
//! (Neutral / Portrait).
//!
//! Empirical inspection of the CPP tone curves (extracted from
//! `3_2_DP0Q0006-CPP.X3F.dng`) shows three load-bearing properties:
//!
//! - The shape function is **asymmetric** between the positive and
//!   negative sides — the lift curve is a different family from the
//!   compression curve, not just a sign flip.
//! - Within each side, all curves are **linear scalings** of a single
//!   reference. Landscape (CMCC=0.25) = (0.25/0.3) × Vivid (CMCC=0.3)
//!   to within 0.1% across the full range; same for Portrait /
//!   Neutral.
//! - At CMCC = 0 the curve is identity, and the CPP build *omits*
//!   `ProfileToneCurve` entirely rather than writing an explicit
//!   identity curve.
//!
//! So we ship two 256-point reference *delta* arrays (`VIVID_DELTAS` for
//! +0.3, `NEUTRAL_DELTAS` for −0.3) and synthesise any curve in
//! `[-0.3, +0.3]` by scaling. Outside that range we clamp the scale
//! factor so we don't extrapolate past Sigma's tested envelope.

/// 256 input samples uniformly spanning [0, 1]. Each `(input, output)`
/// pair in `ProfileToneCurve` is `(SAMPLES[i], SAMPLES[i] +
/// scaled_delta[i])`. The first sample is exactly 0 and the last
/// exactly 1 so the curve anchors at (0, 0) and (1, 1).
const N_POINTS: usize = 256;
fn sample(i: usize) -> f32 {
    i as f32 / (N_POINTS - 1) as f32
}

/// Reference contrast parameter the bundled curves were measured at.
const REFERENCE_CMCC: f64 = 0.3;

/// 256-point delta curve for `CMCC = +0.3` (lift midtones — used by
/// Vivid / Landscape / FCBlue / and any future positive-CMCC mode).
/// Extracted from `3_2_DP0Q0006-CPP.X3F.dng`.
const VIVID_DELTAS: [f32; 256] = [
    0.000000e+00,
    4.159574e-03,
    6.926687e-03,
    9.255_84e-3,
    1.131831e-02,
    1.319127e-02,
    1.491852e-02,
    1.652821e-02,
    1.803989e-02,
    1.946785e-02,
    2.082302e-02,
    2.211398e-02,
    2.334761e-02,
    2.452958e-02,
    2.566463e-02,
    2.675677e-02,
    2.780939e-02,
    2.882548e-02,
    2.980_76e-2,
    3.075802e-02,
    3.167875e-02,
    3.257156e-02,
    3.343805e-02,
    3.427964e-02,
    3.509764e-02,
    3.589321e-02,
    3.666741e-02,
    3.742122e-02,
    3.815552e-02,
    3.887112e-02,
    3.956877e-02,
    4.024916e-02,
    4.091293e-02,
    4.156066e-02,
    4.219289e-02,
    4.281014e-02,
    4.341288e-02,
    4.400152e-02,
    4.457_65e-2,
    4.513817e-02,
    4.568_69e-2,
    4.622301e-02,
    4.674682e-02,
    4.725862e-02,
    4.775867e-02,
    4.824722e-02,
    4.872453e-02,
    4.919_08e-2,
    4.964624e-02,
    5.009107e-02,
    5.052546e-02,
    5.094959e-02,
    5.136357e-02,
    5.176763e-02,
    5.216189e-02,
    5.254649e-02,
    5.292152e-02,
    5.328713e-02,
    5.364342e-02,
    5.399053e-02,
    5.432852e-02,
    5.465747e-02,
    5.497752e-02,
    5.528875e-02,
    5.559117e-02,
    5.588493e-02,
    5.617005e-02,
    5.644664e-02,
    5.671471e-02,
    5.697438e-02,
    5.722564e-02,
    5.746859e-02,
    5.770326e-02,
    5.792966e-02,
    5.814791e-02,
    5.835798e-02,
    5.855995e-02,
    5.875385e-02,
    5.893967e-02,
    5.911_75e-2,
    5.928737e-02,
    5.944926e-02,
    5.960321e-02,
    5.974931e-02,
    5.988_75e-2,
    6.001785e-02,
    6.014037e-02,
    6.025508e-02,
    6.036201e-02,
    6.046116e-02,
    6.055_26e-2,
    6.063628e-02,
    6.071228e-02,
    6.078058e-02,
    6.084_12e-2,
    6.089416e-02,
    6.093949e-02,
    6.097719e-02,
    6.100729e-02,
    6.102982e-02,
    6.104475e-02,
    6.105214e-02,
    6.105199e-02,
    6.104434e-02,
    6.102914e-02,
    6.100649e-02,
    6.097636e-02,
    6.093878e-02,
    6.089377e-02,
    6.084135e-02,
    6.078154e-02,
    6.071433e-02,
    6.063_98e-2,
    6.055793e-02,
    6.046876e-02,
    6.037226e-02,
    6.026855e-02,
    6.015757e-02,
    6.003937e-02,
    5.991396e-02,
    5.978146e-02,
    5.964175e-02,
    5.949494e-02,
    5.934104e-02,
    5.918011e-02,
    5.901_22e-2,
    5.883721e-02,
    5.865535e-02,
    5.846643e-02,
    5.827075e-02,
    5.806816e-02,
    5.785871e-02,
    5.764252e-02,
    5.741_96e-2,
    5.718994e-02,
    5.695361e-02,
    5.671066e-02,
    5.646116e-02,
    5.620503e-02,
    5.594248e-02,
    5.567342e-02,
    5.539799e-02,
    5.511618e-02,
    5.482805e-02,
    5.453366e-02,
    5.423307e-02,
    5.392635e-02,
    5.361342e-02,
    5.329454e-02,
    5.296957e-02,
    5.263871e-02,
    5.230194e-02,
    5.195934e-02,
    5.161095e-02,
    5.125_69e-2,
    5.089712e-02,
    5.053_18e-2,
    5.016094e-02,
    4.978_46e-2,
    4.940289e-02,
    4.901582e-02,
    4.862_35e-2,
    4.822_6e-2,
    4.782331e-02,
    4.741561e-02,
    4.700291e-02,
    4.658526e-02,
    4.616278e-02,
    4.573554e-02,
    4.530364e-02,
    4.486704e-02,
    4.442596e-02,
    4.398036e-02,
    4.353_04e-2,
    4.307_61e-2,
    4.261756e-02,
    4.215491e-02,
    4.168814e-02,
    4.121745e-02,
    4.074275e-02,
    4.026431e-02,
    3.978205e-02,
    3.929621e-02,
    3.880674e-02,
    3.831381e-02,
    3.781748e-02,
    3.731781e-02,
    3.681493e-02,
    3.630888e-02,
    3.579_98e-2,
    3.528_78e-2,
    3.477287e-02,
    3.425515e-02,
    3.373474e-02,
    3.321177e-02,
    3.268623e-02,
    3.215832e-02,
    3.162807e-02,
    3.109556e-02,
    3.056091e-02,
    3.002423e-02,
    2.948558e-02,
    2.894503e-02,
    2.840275e-02,
    2.785873e-02,
    2.731317e-02,
    2.676612e-02,
    2.621764e-02,
    2.566785e-02,
    2.511686e-02,
    2.456468e-02,
    2.401155e-02,
    2.345747e-02,
    2.290249e-02,
    2.234679e-02,
    2.179044e-02,
    2.123_35e-2,
    2.067614e-02,
    2.011_83e-2,
    1.956022e-02,
    1.900196e-02,
    1.844352e-02,
    1.788515e-02,
    1.732677e-02,
    1.676857e-02,
    1.621068e-02,
    1.565301e-02,
    1.509583e-02,
    1.453918e-02,
    1.398307e-02,
    1.342767e-02,
    1.287305e-02,
    1.231927e-02,
    1.176643e-02,
    1.121461e-02,
    1.066393e-02,
    1.011437e-02,
    9.566128e-03,
    9.019196e-03,
    8.473694e-03,
    7.929683e-03,
    7.387_28e-3,
    6.846488e-03,
    6.307483e-03,
    5.770266e-03,
    5.234897e-03,
    4.701495e-03,
    4.170_12e-3,
    3.640831e-03,
    3.113747e-03,
    2.588809e-03,
    2.066195e-03,
    1.545966e-03,
    1.028121e-03,
    5.127788e-04,
    0.000000e+00,
];

/// 256-point delta curve for `CMCC = -0.3` (compress midtones — used
/// by Neutral / Portrait / and any future negative-CMCC mode).
/// Extracted from `3_2_DP0Q0006-CPP.X3F.dng`.
const NEUTRAL_DELTAS: [f32; 256] = [
    0.000000e+00,
    -1.542121e-03,
    -2.781929e-03,
    -3.896647e-03,
    -4.928137e-03,
    -5.896975e-03,
    -6.815562e-03,
    -7.692227e-03,
    -8.532969e-03,
    -9.342315e-03,
    -1.012_38e-2,
    -1.088028e-02,
    -1.161408e-02,
    -1.232716e-02,
    -1.302116e-02,
    -1.369_75e-2,
    -1.435_74e-2,
    -1.500192e-02,
    -1.563201e-02,
    -1.624_85e-2,
    -1.685212e-02,
    -1.744354e-02,
    -1.802334e-02,
    -1.859207e-02,
    -1.915_02e-2,
    -1.969817e-02,
    -2.023638e-02,
    -2.076519e-02,
    -2.128494e-02,
    -2.179592e-02,
    -2.229841e-02,
    -2.279267e-02,
    -2.327894e-02,
    -2.375741e-02,
    -2.422828e-02,
    -2.469175e-02,
    -2.514798e-02,
    -2.559711e-02,
    -2.603929e-02,
    -2.647465e-02,
    -2.690_33e-2,
    -2.732536e-02,
    -2.774094e-02,
    -2.815_01e-2,
    -2.855293e-02,
    -2.894954e-02,
    -2.933997e-02,
    -2.972429e-02,
    -3.010255e-02,
    -3.047483e-02,
    -3.084114e-02,
    -3.120154e-02,
    -3.155608e-02,
    -3.190477e-02,
    -3.224766e-02,
    -3.258477e-02,
    -3.291611e-02,
    -3.324173e-02,
    -3.356162e-02,
    -3.387579e-02,
    -3.418428e-02,
    -3.448707e-02,
    -3.478418e-02,
    -3.507562e-02,
    -3.536139e-02,
    -3.564148e-02,
    -3.591588e-02,
    -3.618462e-02,
    -3.644767e-02,
    -3.670503e-02,
    -3.695671e-02,
    -3.720269e-02,
    -3.744295e-02,
    -3.767_75e-2,
    -3.790632e-02,
    -3.812939e-02,
    -3.834671e-02,
    -3.855827e-02,
    -3.876406e-02,
    -3.896406e-02,
    -3.915823e-02,
    -3.934661e-02,
    -3.952914e-02,
    -3.970584e-02,
    -3.987_67e-2,
    -4.004166e-02,
    -4.020074e-02,
    -4.035389e-02,
    -4.050115e-02,
    -4.064247e-02,
    -4.077786e-02,
    -4.090729e-02,
    -4.103073e-02,
    -4.114822e-02,
    -4.125968e-02,
    -4.136515e-02,
    -4.146_46e-2,
    -4.155803e-02,
    -4.164541e-02,
    -4.172674e-02,
    -4.180202e-02,
    -4.187125e-02,
    -4.193437e-02,
    -4.199144e-02,
    -4.204_24e-2,
    -4.208729e-02,
    -4.212609e-02,
    -4.215878e-02,
    -4.218537e-02,
    -4.220584e-02,
    -4.222023e-02,
    -4.222852e-02,
    -4.223_07e-2,
    -4.222676e-02,
    -4.221675e-02,
    -4.220062e-02,
    -4.217845e-02,
    -4.215017e-02,
    -4.211581e-02,
    -4.207_54e-2,
    -4.202_89e-2,
    -4.197639e-02,
    -4.191783e-02,
    -4.185328e-02,
    -4.178271e-02,
    -4.170614e-02,
    -4.162359e-02,
    -4.153511e-02,
    -4.144_07e-2,
    -4.134035e-02,
    -4.123411e-02,
    -4.112199e-02,
    -4.100403e-02,
    -4.088026e-02,
    -4.075068e-02,
    -4.061532e-02,
    -4.047424e-02,
    -4.032743e-02,
    -4.017496e-02,
    -4.001683e-02,
    -3.985_31e-2,
    -3.968382e-02,
    -3.950894e-02,
    -3.932858e-02,
    -3.914279e-02,
    -3.895152e-02,
    -3.875488e-02,
    -3.855294e-02,
    -3.834569e-02,
    -3.813314e-02,
    -3.791541e-02,
    -3.769255e-02,
    -3.746456e-02,
    -3.723_15e-2,
    -3.699_35e-2,
    -3.675_05e-2,
    -3.650254e-02,
    -3.624982e-02,
    -3.599226e-02,
    -3.572994e-02,
    -3.546304e-02,
    -3.519142e-02,
    -3.491527e-02,
    -3.463465e-02,
    -3.434956e-02,
    -3.406012e-02,
    -3.376639e-02,
    -3.346843e-02,
    -3.316623e-02,
    -3.285998e-02,
    -3.254962e-02,
    -3.223538e-02,
    -3.191715e-02,
    -3.159517e-02,
    -3.126937e-02,
    -3.093994e-02,
    -3.060687e-02,
    -3.027028e-02,
    -2.993023e-02,
    -2.958679e-02,
    -2.924001e-02,
    -2.889007e-02,
    -2.853692e-02,
    -2.818072e-02,
    -2.782154e-02,
    -2.745944e-02,
    -2.709454e-02,
    -2.672684e-02,
    -2.635652e-02,
    -2.598357e-02,
    -2.560818e-02,
    -2.523035e-02,
    -2.485019e-02,
    -2.446783e-02,
    -2.408326e-02,
    -2.369_66e-2,
    -2.330804e-02,
    -2.291751e-02,
    -2.252519e-02,
    -2.213115e-02,
    -2.173549e-02,
    -2.133822e-02,
    -2.093953e-02,
    -2.053946e-02,
    -2.013814e-02,
    -1.973563e-02,
    -1.933199e-02,
    -1.892734e-02,
    -1.852173e-02,
    -1.811534e-02,
    -1.770818e-02,
    -1.730031e-02,
    -1.689196e-02,
    -1.648307e-02,
    -1.607376e-02,
    -1.566422e-02,
    -1.525444e-02,
    -1.484454e-02,
    -1.443458e-02,
    -1.402467e-02,
    -1.361489e-02,
    -1.320535e-02,
    -1.279_61e-2,
    -1.238728e-02,
    -1.197892e-02,
    -1.157111e-02,
    -1.116401e-02,
    -1.075763e-02,
    -1.035202e-02,
    -9.947_36e-3,
    -9.543657e-03,
    -9.141088e-03,
    -8.739_65e-3,
    -8.339405e-03,
    -7.940531e-03,
    -7.542968e-03,
    -7.146955e-03,
    -6.752491e-03,
    -6.359577e-03,
    -5.968451e-03,
    -5.579054e-03,
    -5.191565e-03,
    -4.805923e-03,
    -4.422367e-03,
    -4.040837e-03,
    -3.661454e-03,
    -3.284276e-03,
    -2.909362e-03,
    -2.536774e-03,
    -2.166629e-03,
    -1.798987e-03,
    -1.433849e-03,
    -1.071334e-03,
    -7.115006e-04,
    -3.543496e-04,
    0.000000e+00,
];

/// Build a 256-point `ProfileToneCurve` payload (interleaved
/// (input, output) f32 pairs, total 512 floats) for the given CMCC
/// value. Returns `None` for `cmcc.abs() < 1e-3`, since the curve
/// would be effectively identity and the CPP reference build omits
/// `ProfileToneCurve` entirely in that case.
pub(crate) fn build_curve(cmcc: f64) -> Option<Vec<f32>> {
    if cmcc.abs() < 1e-3 {
        return None;
    }
    // Clamp the scale factor to [0, 1] so a future CMCC outside ±0.3
    // doesn't extrapolate the bundled reference into territory it was
    // never sampled in. Negative CMCCs use the Neutral curve, positive
    // use Vivid; the reference is itself signed so we pick by the
    // sign of CMCC and use |scale| against the matching reference.
    let scale = (cmcc.abs() / REFERENCE_CMCC).clamp(0.0, 1.0) as f32;
    let reference: &[f32; N_POINTS] = if cmcc > 0.0 {
        &VIVID_DELTAS
    } else {
        &NEUTRAL_DELTAS
    };
    let mut out: Vec<f32> = Vec::with_capacity(N_POINTS * 2);
    for (i, &delta) in reference.iter().enumerate() {
        let x = sample(i);
        let y = (x + delta * scale).clamp(0.0, 1.0);
        out.push(x);
        out.push(y);
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cmcc_zero_returns_none() {
        assert!(build_curve(0.0).is_none());
        assert!(build_curve(0.0001).is_none());
    }

    #[test]
    fn endpoints_anchor_at_0_and_1() {
        for &cmcc in &[-0.3, -0.25, 0.25, 0.3] {
            let c = build_curve(cmcc).unwrap();
            // First pair = (0, 0), last pair = (1, 1).
            assert_eq!(c[0], 0.0);
            assert_eq!(c[1], 0.0);
            assert_eq!(c[c.len() - 2], 1.0);
            assert_eq!(c[c.len() - 1], 1.0);
        }
    }

    #[test]
    fn vivid_curve_lifts_midtones() {
        // CMCC = +0.3 lifts the midtones above the diagonal.
        let c = build_curve(0.3).unwrap();
        let mid = c.len() / 2; // index 256 = pair (input[128], output[128])
        let input = c[mid];
        let output = c[mid + 1];
        assert!(
            output > input,
            "Vivid midtone should lift, got input={input}, output={output}",
        );
    }

    #[test]
    fn neutral_curve_compresses_midtones() {
        let c = build_curve(-0.3).unwrap();
        let mid = c.len() / 2;
        let input = c[mid];
        let output = c[mid + 1];
        assert!(
            output < input,
            "Neutral midtone should compress, got input={input}, output={output}",
        );
    }

    #[test]
    fn landscape_is_linear_scale_of_vivid() {
        // CMCC=0.25 should be (0.25/0.3) of the Vivid lift at every point.
        let v = build_curve(0.3).unwrap();
        let l = build_curve(0.25).unwrap();
        for i in 0..N_POINTS {
            let v_delta = v[i * 2 + 1] - v[i * 2];
            let l_delta = l[i * 2 + 1] - l[i * 2];
            let expected = v_delta * (0.25_f32 / 0.3_f32);
            // Allow a couple of f32 ULP for the rounding; outside the
            // ±0.05 band of large deltas this is ≪ 1e-6.
            let abs_diff = (l_delta - expected).abs();
            assert!(
                abs_diff < 1e-5,
                "i={i}: vivid_delta={v_delta}, landscape_delta={l_delta}, expected={expected}",
            );
        }
    }

    #[test]
    fn input_axis_is_uniform_0_to_1() {
        let c = build_curve(0.3).unwrap();
        for i in 0..N_POINTS {
            let expected = i as f32 / (N_POINTS - 1) as f32;
            assert!(
                (c[i * 2] - expected).abs() < 1e-6,
                "input[{i}] = {} != {}",
                c[i * 2],
                expected,
            );
        }
    }
}
