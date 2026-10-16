//! Test metrics for clipped linear sRGB output under D65.
//! CIEDE2000: https://hajim.rochester.edu/ece/sites/gsharma/ciede2000/

pub(super) fn lab(rgb: [f64; 3]) -> [f64; 3] {
    let rgb = rgb.map(|v| v.clamp(0.0, 1.0));
    let xyz = super::super::super::rgb_table::multiply(
        &[
            0.4124564, 0.3575761, 0.1804375, 0.2126729, 0.7151522, 0.0721750, 0.0193339, 0.1191920,
            0.9503041,
        ],
        rgb,
    );
    let f: [f64; 3] = std::array::from_fn(|i| {
        let t = xyz[i] / [0.95047, 1.0, 1.08883][i];
        if t > 216.0 / 24389.0 {
            t.cbrt()
        } else {
            (24389.0 / 27.0 * t + 16.0) / 116.0
        }
    });
    [
        116.0 * f[1] - 16.0,
        500.0 * (f[0] - f[1]),
        200.0 * (f[1] - f[2]),
    ]
}

pub(super) fn delta_e(a: [f64; 3], b: [f64; 3]) -> f64 {
    let chroma = (a[1].hypot(a[2]) + b[1].hypot(b[2])) * 0.5;
    let g = 0.5 * (1.0 - (chroma.powi(7) / (chroma.powi(7) + 25.0_f64.powi(7))).sqrt());
    let (a1, a2) = ((1.0 + g) * a[1], (1.0 + g) * b[1]);
    let (c1, c2) = (a1.hypot(a[2]), a2.hypot(b[2]));
    let hue = |x: f64, y: f64| {
        if x == 0.0 && y == 0.0 {
            0.0
        } else {
            y.atan2(x).to_degrees().rem_euclid(360.0)
        }
    };
    let (h1, h2) = (hue(a1, a[2]), hue(a2, b[2]));
    let dh = if c1 * c2 == 0.0 {
        0.0
    } else if (h2 - h1).abs() <= 180.0 {
        h2 - h1
    } else if h2 > h1 {
        h2 - h1 - 360.0
    } else {
        h2 - h1 + 360.0
    };
    let mean_hue = if c1 * c2 == 0.0 {
        h1 + h2
    } else if (h1 - h2).abs() <= 180.0 {
        (h1 + h2) * 0.5
    } else if h1 + h2 < 360.0 {
        (h1 + h2 + 360.0) * 0.5
    } else {
        (h1 + h2 - 360.0) * 0.5
    };
    let cosine = |v: f64| v.to_radians().cos();
    let t = 1.0 - 0.17 * cosine(mean_hue - 30.0)
        + 0.24 * cosine(2.0 * mean_hue)
        + 0.32 * cosine(3.0 * mean_hue + 6.0)
        - 0.20 * cosine(4.0 * mean_hue - 63.0);
    let mean_l = (a[0] + b[0]) * 0.5;
    let mean_c = (c1 + c2) * 0.5;
    let dl = (b[0] - a[0])
        / (1.0 + 0.015 * (mean_l - 50.0).powi(2) / (20.0 + (mean_l - 50.0).powi(2)).sqrt());
    let dc = (c2 - c1) / (1.0 + 0.045 * mean_c);
    let dh = 2.0 * (c1 * c2).sqrt() * (dh * 0.5).to_radians().sin() / (1.0 + 0.015 * mean_c * t);
    let rotation = -2.0
        * (mean_c.powi(7) / (mean_c.powi(7) + 25.0_f64.powi(7))).sqrt()
        * (60.0 * (-((mean_hue - 275.0) / 25.0).powi(2)).exp())
            .to_radians()
            .sin();
    (dl * dl + dc * dc + dh * dh + rotation * dc * dh)
        .max(0.0)
        .sqrt()
}

pub(super) fn hue_error(a: [f64; 3], b: [f64; 3]) -> Option<f64> {
    if a[1].hypot(a[2]) < 1.0 || b[1].hypot(b[2]) < 1.0 {
        return None;
    }
    let angle = (a[2].atan2(a[1]) - b[2].atan2(b[1]))
        .to_degrees()
        .abs()
        .rem_euclid(360.0);
    Some(angle.min(360.0 - angle))
}

pub(super) fn percentiles(mut values: Vec<f64>) -> [f64; 4] {
    if values.is_empty() {
        return [0.0; 4];
    }
    values.sort_by(f64::total_cmp);
    [0.5, 0.95, 0.99, 1.0].map(|p| values[((values.len() - 1) as f64 * p).round() as usize])
}

#[test]
fn cie2000_matches_published_sharma_pairs() {
    for (a, b, expected) in [
        ([50.0, 2.6772, -79.7751], [50.0, 0.0, -82.7485], 2.0425),
        ([50.0, 3.1571, -77.2803], [50.0, 0.0, -82.7485], 2.8615),
        ([50.0, 0.0, 0.0], [50.0, -1.0, 2.0], 2.3669),
        ([50.0, 2.49, -0.001], [50.0, -2.49, 0.0009], 7.1792),
        ([50.0, 2.49, -0.001], [50.0, -2.49, 0.0011], 7.2195),
        ([50.0, -0.001, 2.49], [50.0, 0.0011, -2.49], 4.7461),
    ] {
        assert!((delta_e(a, b) - expected).abs() < 0.0001);
        assert!((delta_e(b, a) - expected).abs() < 0.0001);
        assert_eq!(delta_e(a, a), 0.0);
    }
    assert!(lab([0.0; 3]).iter().all(|v| v.abs() < 1e-12));
    assert!((lab([1.0; 3])[0] - 100.0).abs() < 1e-5);
}
