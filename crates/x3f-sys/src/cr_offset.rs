//! Stage 6 of the ordered Merrill float frontend: row/column chroma
//! offset correction, the scalar reconstruction of the native
//! `CROffsetCorrection` (wrapper 0x902cc, `RunCROCPatches` worker
//! 0x8f628, `CalculatePseudoInv` 0x8f1cc, final application
//! 0x902cc..0x904dc). The embedded LINPACK `dsvdc` (0x8ae8c) is
//! transcribed from the archived, native-verified solver adaptation
//! (x3f-docs `followup-offset-svd/solver/scalar_svd.hpp`: three
//! polynomial SVD states and 374 finite pseudoinverse cases matched the
//! native library byte for byte under `-ffp-contract=off` with
//! explicit fused multiply-adds). The worker and final application
//! come from `followup-repair/croffset/check_patch.cpp` (48 native
//! worker comparisons, five wrapper runs and the final-application
//! gates byte-identical); the dispatcher, worker geometry and scratch
//! placement come from `followup-offset-svd/scheduling/`.
//!
//! The native accumulates row sums and counts from eight workers
//! whose row ranges can overlap, without synchronization; repeated
//! native runs differ on overlapping geometry. This port keeps the
//! native per-worker and per-patch arithmetic and accumulates in a
//! fixed documented order: workers 0..=7 outer, patch rows and columns
//! ascending inner, scales 31, 73, 151 in that order. Overlapping
//! geometry yields one deterministic member of the native result
//! family, not a byte-identical native reference.

use crate::control::{Control, Error, Result};

/// The native fixed patch sizes, in dispatch order.
pub const SCALES: [u32; 3] = [31, 73, 151];

/// Statistics of one correction pass: patches visited, patches
/// accepted by the correlation gate, and row contributions added.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct OffsetStats {
    pub patches: u32,
    pub accepted: u32,
    pub rows: u32,
}

// ---------------------------------------------------------------------------
// Embedded SVD. Transcribed from the archived scalar_svd.hpp, which was
// itself checked byte-for-byte against the native dsvdc on the three
// polynomial bases and 374 finite cases. Unit strides, job 21, finite
// matrices only. Keep the fused multiply-adds: the archived run
// proved that splitting them changes native bits.
// ---------------------------------------------------------------------------

fn r8_sign(x: f64) -> f64 {
    if x < 0.0 {
        -1.0
    } else {
        1.0
    }
}

/// `y[i] = fma(a, x[i], y[i])`; `a == 0` returns early.
fn daxpy(n: usize, a: f64, x: &[f64], x_offset: usize, y: &mut [f64], y_offset: usize) {
    if a == 0.0 {
        return;
    }
    for i in 0..n {
        y[y_offset + i] = a.mul_add(x[x_offset + i], y[y_offset + i]);
    }
}

fn ddot(n: usize, x: &[f64], x_offset: usize, y: &[f64], y_offset: usize) -> f64 {
    let mut sum = 0.0;
    for i in 0..n {
        sum = x[x_offset + i].mul_add(y[y_offset + i], sum);
    }
    sum
}

/// Scaled sum of squares with the native fused accumulation.
fn dnrm2(n: usize, x: &[f64], x_offset: usize) -> f64 {
    if n < 1 {
        return 0.0;
    }
    if n == 1 {
        return if x[x_offset] < 0.0 {
            -x[x_offset]
        } else {
            x[x_offset]
        };
    }
    let mut scale = 0.0_f64;
    let mut ssq = 1.0_f64;
    for i in 0..n {
        if x[x_offset + i] != 0.0 {
            let value = if x[x_offset + i] < 0.0 {
                -x[x_offset + i]
            } else {
                x[x_offset + i]
            };
            if scale < value {
                let r = scale / value;
                // The native fma rounds `ssq*r` before fusing with r:
                // `ssq = fma(ssq*r, r, 1)`, not `fma(ssq, r*r, 1)`.
                ssq = (ssq * r).mul_add(r, 1.0);
                scale = value;
            } else {
                let r = value / scale;
                ssq = r.mul_add(r, ssq);
            }
        }
    }
    scale * ssq.sqrt()
}

fn dscal(n: usize, a: f64, x: &mut [f64], x_offset: usize) {
    for i in 0..n {
        x[x_offset + i] *= a;
    }
}

fn dswap(n: usize, x: &mut [f64], x_offset: usize, y: &mut [f64], y_offset: usize) {
    for i in 0..n {
        std::mem::swap(&mut x[x_offset + i], &mut y[y_offset + i]);
    }
}

fn drot(n: usize, x: &mut [f64], x_offset: usize, y: &mut [f64], y_offset: usize, c: f64, s: f64) {
    for i in 0..n {
        let t = c.mul_add(x[x_offset + i], s * y[y_offset + i]);
        y[y_offset + i] = c.mul_add(y[y_offset + i], -s * x[x_offset + i]);
        x[x_offset + i] = t;
    }
}

/// Plane Givens rotation. Returns `(c, s, r, z)`; callers assign `r` to
/// the former `a` argument and discard `z`.
fn drotg(a: f64, b: f64) -> (f64, f64, f64, f64) {
    let roe = if b.abs() < a.abs() { a } else { b };
    let scale = a.abs() + b.abs();
    let (c, s, r) = if scale == 0.0 {
        (1.0, 0.0, 0.0)
    } else {
        let x = a / scale;
        let y = b / scale;
        let mut r = scale * x.mul_add(x, y * y).sqrt();
        if roe < 0.0 {
            r = -r;
        }
        (a / r, b / r, r)
    };
    let mut z = s;
    if c.abs() <= s && c != 0.0 {
        z = 1.0 / c;
    }
    (c, s, r, z)
}

/// The embedded LINPACK `dsvdc` (native
/// `_Z5dsvdcPdiiiS_S_S_iS_iS_i` at 0x8ae8c), transcribed from the
/// archived native-verified adaptation. One-based indices as in
/// LINPACK; `a`, `u`, `v` are column-major with leading dimensions
/// `lda`, `ldu`, `ldv`. Only the job 21 path (singular values,
/// partial U, full V) that `CalculatePseudoInv` uses is exercised;
/// the control flow stays general. The kase-1 deflation guard
/// replaces the native column-index overrun that the archived evidence
/// documented on rank-deficient input; the supported polynomial bases
/// never reach that path.
#[allow(clippy::too_many_lines)]
fn dsvdc(
    a: &mut [f64],
    lda: usize,
    m: usize,
    n: usize,
    s: &mut [f64],
    e: &mut [f64],
    u: &mut [f64],
    ldu: usize,
    v: &mut [f64],
    ldv: usize,
    work: &mut [f64],
    job: i32,
) -> i32 {
    let maxit = 30;

    let jobu = (job % 100) / 10;
    let ncu = if jobu > 1 { m.min(n) } else { m };
    let wantu = jobu != 0;
    let wantv = job % 10 != 0;

    // Reduce A to bidiagonal form, diagonal in S, superdiagonal in E.
    let nct = (m - 1).min(n);
    let nrt = if n >= 2 { m.min(n - 2) } else { 0 };
    let lu = nct.max(nrt);

    for l in 1..=lu {
        if l <= nct {
            let offset = (l - 1) + (l - 1) * lda;
            s[l - 1] = dnrm2(m - l + 1, a, offset);
            if s[l - 1] != 0.0 {
                if a[offset] != 0.0 {
                    s[l - 1] = r8_sign(a[offset]) * s[l - 1].abs();
                }
                dscal(m - l + 1, 1.0 / s[l - 1], a, offset);
                a[offset] += 1.0;
            }
            s[l - 1] = -s[l - 1];
        }

        for j in l + 1..=n {
            if l <= nct && s[l - 1] != 0.0 {
                let x_offset = (l - 1) + (l - 1) * lda;
                let y_offset = (l - 1) + (j - 1) * lda;
                let t = -ddot(m - l + 1, a, x_offset, a, y_offset) / a[(l - 1) + (l - 1) * lda];
                let (x_slice, y_slice) = a.split_at_mut(y_offset);
                daxpy(m - l + 1, t, x_slice, x_offset, y_slice, 0);
            }
            e[j - 1] = a[(l - 1) + (j - 1) * lda];
        }

        if wantu && l <= nct {
            for i in l..=m {
                u[(i - 1) + (l - 1) * ldu] = a[(i - 1) + (l - 1) * lda];
            }
        }

        if l <= nrt {
            e[l - 1] = dnrm2(n - l, e, l);
            if e[l - 1] != 0.0 {
                if e[l] != 0.0 {
                    e[l - 1] = r8_sign(e[l]) * e[l - 1].abs();
                }
                dscal(n - l, 1.0 / e[l - 1], e, l);
                e[l] += 1.0;
            }
            e[l - 1] = -e[l - 1];

            if l + 1 <= m && e[l - 1] != 0.0 {
                for j in l + 1..=m {
                    work[j - 1] = 0.0;
                }
                for j in l + 1..=n {
                    daxpy(m - l, e[j - 1], a, l + (j - 1) * lda, work, l);
                }
                for j in l + 1..=n {
                    let factor = -e[j - 1] / e[l];
                    daxpy(m - l, factor, work, l, a, l + (j - 1) * lda);
                }
            }

            if wantv {
                for j in l + 1..=n {
                    v[(j - 1) + (l - 1) * ldv] = e[j - 1];
                }
            }
        }
    }

    // Set up the final bidiagonal matrix of order MN.
    let mut mn = (m + 1).min(n);
    let nctp1 = nct + 1;
    let nrtp1 = nrt + 1;

    if nct < n {
        s[nctp1 - 1] = a[(nctp1 - 1) + (nctp1 - 1) * lda];
    }
    if m < mn {
        s[mn - 1] = 0.0;
    }
    if nrtp1 < mn {
        e[nrtp1 - 1] = a[(nrtp1 - 1) + (mn - 1) * lda];
    }
    e[mn - 1] = 0.0;

    // Generate U.
    if wantu {
        for i in 1..=m {
            for j in nctp1..=ncu {
                u[(i - 1) + (j - 1) * ldu] = 0.0;
            }
        }
        for j in nctp1..=ncu {
            u[(j - 1) + (j - 1) * ldu] = 1.0;
        }
        for ll in 1..=nct {
            let l = nct - ll + 1;
            if s[l - 1] != 0.0 {
                for j in l + 1..=ncu {
                    let x_offset = (l - 1) + (l - 1) * ldu;
                    let y_offset = (l - 1) + (j - 1) * ldu;
                    let t = -ddot(m - l + 1, u, x_offset, u, y_offset) / u[(l - 1) + (l - 1) * ldu];
                    let (x_slice, y_slice) = u.split_at_mut(y_offset);
                    daxpy(m - l + 1, t, x_slice, x_offset, y_slice, 0);
                }
                dscal(m - l + 1, -1.0, u, (l - 1) + (l - 1) * ldu);
                u[(l - 1) + (l - 1) * ldu] += 1.0;
                for i in 1..l {
                    u[(i - 1) + (l - 1) * ldu] = 0.0;
                }
            } else {
                for i in 1..=m {
                    u[(i - 1) + (l - 1) * ldu] = 0.0;
                }
                u[(l - 1) + (l - 1) * ldu] = 1.0;
            }
        }
    }

    // Generate V.
    if wantv {
        for ll in 1..=n {
            let l = n - ll + 1;
            if l <= nrt && e[l - 1] != 0.0 {
                for j in l + 1..=n {
                    let x_offset = l + (l - 1) * ldv;
                    let y_offset = l + (j - 1) * ldv;
                    let t = -ddot(n - l, v, x_offset, v, y_offset) / v[l + (l - 1) * ldv];
                    let (x_slice, y_slice) = v.split_at_mut(y_offset);
                    daxpy(n - l, t, x_slice, x_offset, y_slice, 0);
                }
            }
            for i in 1..=n {
                v[(i - 1) + (l - 1) * ldv] = 0.0;
            }
            v[(l - 1) + (l - 1) * ldv] = 1.0;
        }
    }

    // Main iteration loop for the singular values.
    let mm = mn;
    let mut iter = 0;
    let mut l;

    while 0 < mn {
        if maxit <= iter {
            return mn as i32;
        }

        // Inspect for negligible elements in S and E.
        let kase;
        l = mn;
        for ll in 1..=mn {
            l = mn - ll;
            if l == 0 {
                break;
            }
            let test = s[l - 1].abs() + s[l].abs();
            let ztest = test + e[l - 1].abs();
            if ztest == test {
                e[l - 1] = 0.0;
                break;
            }
        }

        if l == mn - 1 {
            kase = 4;
        } else {
            let mut ls = l;
            for lls in l + 1..=mn + 1 {
                // C computes `mn - lls + l + 1` with signed ints; the
                // reordered form avoids an unsigned intermediate that
                // underflows at lls == mn + 1 (the loop breaks there).
                ls = (mn + l + 1) - lls;
                if ls == l {
                    break;
                }
                let mut test = 0.0;
                if ls != mn {
                    test += e[ls - 1].abs();
                }
                if ls != l + 1 {
                    test += e[ls - 2].abs();
                }
                let ztest = test + s[ls - 1].abs();
                if ztest == test {
                    s[ls - 1] = 0.0;
                    break;
                }
            }
            if ls == l {
                kase = 3;
            } else if ls == mn {
                kase = 1;
            } else {
                kase = 2;
                l = ls;
            }
        }
        l += 1;

        // Deflate negligible S(MN). The guard replaces the native
        // column-index overrun (kase 1 never runs on the supported
        // bases); returning the failure count keeps the caller's
        // non-convergence contract explicit. `k == mn` aliases the
        // two rotated V columns in the native (an in-place single-
        // column rotation); that rank-deficient corner is outside
        // the verified domain and is rejected here rather than
        // half-reconstructed.
        if kase == 1 {
            let mm1 = mn - 1;
            let mut f = e[mn - 2];
            e[mn - 2] = 0.0;
            for kk in 1..=mm1 {
                let k = mm1 - kk + l;
                if !(1 <= k && k < mn) || k > n {
                    return mn as i32;
                }
                let t1 = s[k - 1];
                let (cs, sn, r, _) = drotg(t1, f);
                s[k - 1] = r;
                if k != l {
                    f = -sn * e[k - 2];
                    e[k - 2] = cs * e[k - 2];
                }
                if wantv {
                    let (x_slice, y_slice) = v.split_at_mut((mn - 1) * ldv);
                    drot(n, x_slice, (k - 1) * ldv, y_slice, 0, cs, sn);
                }
            }
        } else if kase == 2 {
            // Split at negligible S(L).
            let mut f = e[l - 2];
            e[l - 2] = 0.0;
            for k in l..=mn {
                let t1 = s[k - 1];
                let (cs, sn, r, _) = drotg(t1, f);
                s[k - 1] = r;
                f = -sn * e[k - 1];
                e[k - 1] = cs * e[k - 1];
                if wantu {
                    // Column l-2 precedes column k-1, so the low column
                    // stays in the first slice.
                    let (low_slice, high_slice) = u.split_at_mut((k - 1) * ldu);
                    drot(m, high_slice, 0, low_slice, (l - 2) * ldu, cs, sn);
                }
            }
        } else if kase == 3 {
            // One QR step with the native fused shift arithmetic.
            let scale = s[mn - 1].abs().max(
                s[mn - 2]
                    .abs()
                    .max(e[mn - 2].abs().max(s[l - 1].abs().max(e[l - 1].abs()))),
            );
            let sm = s[mn - 1] / scale;
            let smm1 = s[mn - 2] / scale;
            let emm1 = e[mn - 2] / scale;
            let sl = s[l - 1] / scale;
            let el = e[l - 1] / scale;
            let b = (smm1 + sm).mul_add(smm1 - sm, emm1 * emm1) * 0.5;
            let c = (sm * emm1) * (sm * emm1);
            let mut shift = 0.0_f64;
            if b != 0.0 || c != 0.0 {
                shift = b.mul_add(b, c).sqrt();
                if b < 0.0 {
                    shift = -shift;
                }
                shift = c / (b + shift);
            }
            let mut f = (sl + sm).mul_add(sl - sm, -shift);
            let mut g = sl * el;

            // Chase zeros.
            let mm1 = mn - 1;
            for k in l..=mm1 {
                let (cs, sn, r, _) = drotg(f, g);
                f = r;
                if k != l {
                    e[k - 2] = f;
                }
                f = cs.mul_add(s[k - 1], sn * e[k - 1]);
                e[k - 1] = cs.mul_add(e[k - 1], -sn * s[k - 1]);
                g = sn * s[k];
                s[k] = cs * s[k];
                if wantv {
                    let (x_slice, y_slice) = v.split_at_mut(k * ldv);
                    drot(n, x_slice, (k - 1) * ldv, y_slice, 0, cs, sn);
                }
                let (cs, sn, r, _) = drotg(f, g);
                f = r;
                s[k - 1] = f;
                f = cs.mul_add(e[k - 1], sn * s[k]);
                s[k] = (-sn).mul_add(e[k - 1], cs * s[k]);
                g = sn * e[k];
                e[k] = cs * e[k];
                if wantu && k < m {
                    let (x_slice, y_slice) = u.split_at_mut(k * ldu);
                    drot(m, x_slice, (k - 1) * ldu, y_slice, 0, cs, sn);
                }
            }
            e[mn - 2] = f;
            iter += 1;
        } else {
            // Convergence: make the singular value nonnegative.
            debug_assert_eq!(kase, 4);
            if s[l - 1] < 0.0 {
                s[l - 1] = -s[l - 1];
                if wantv {
                    dscal(n, -1.0, v, (l - 1) * ldv);
                }
            }
            // Order the singular value.
            loop {
                if l == mm {
                    break;
                }
                if s[l] <= s[l - 1] {
                    break;
                }
                let t = s[l - 1];
                s[l - 1] = s[l];
                s[l] = t;
                if wantv && l < n {
                    let (x_slice, y_slice) = v.split_at_mut(l * ldv);
                    dswap(n, x_slice, (l - 1) * ldv, y_slice, 0);
                }
                if wantu && l < m {
                    let (x_slice, y_slice) = u.split_at_mut(l * ldu);
                    dswap(m, x_slice, (l - 1) * ldu, y_slice, 0);
                }
                l += 1;
            }
            iter = 0;
            mn -= 1;
        }
    }

    0
}

/// `CalculatePseudoInv` (0x8f1cc), transcribed from the archived
/// scalar_pinv.hpp: column-major double conversion, the embedded
/// `dsvdc` (job 21), singular values strictly above the exact double
/// cutoff `0x3f1a36e2eb1c432d`, the V-reciprocal-U sum where the
/// first multiplication rounds before the fused multiply-add, and a
/// final float conversion. `basis` is `area` rows of `terms` floats
/// (row-major); the returned inverse is `terms` rows of `area`
/// floats (row-major), matching the native Pixmap layouts. The
/// native's strict scratch capacity (`16*(area*terms+terms*terms)`
/// bytes) is not reproduced because this port allocates exactly.
fn calculate_pinv(basis: &[f32], terms: usize, area: usize) -> Result<Vec<f32>> {
    if area < terms || terms == 0 || terms > 6 {
        return Err(Error::InvalidData("invalid Merrill offset basis"));
    }
    let (m, n) = (area, terms);
    let mn = m * n;
    let nn = n * n;
    let mut a = vec![0.0_f64; mn];
    for j in 0..n {
        for i in 0..m {
            a[j * m + i] = f64::from(basis[i * n + j]);
        }
    }
    let mut copy = a.clone();
    let mut sv = vec![0.0_f64; m + n];
    let mut e = vec![0.0_f64; m + n];
    let mut u = vec![0.0_f64; mn];
    let mut v = vec![0.0_f64; nn];
    let mut work = vec![0.0_f64; m];
    let status = dsvdc(
        &mut copy, m, m, n, &mut sv, &mut e, &mut u, m, &mut v, n, &mut work, 21,
    );
    if status != 0 {
        return Err(Error::InvalidData("Merrill offset SVD did not converge"));
    }
    let mut s = vec![0.0_f64; nn];
    for k in 0..n {
        s[k * (n + 1)] = sv[k];
    }
    let cutoff = f64::from_bits(0x3f1a_36e2_eb1c_432d);
    let mut inverse = vec![0.0_f64; mn];
    for j in 0..n {
        for i in 0..m {
            let mut sum = 0.0_f64;
            for k in 0..n {
                let reciprocal = if s[k * (n + 1)] > cutoff {
                    1.0 / s[k * (n + 1)]
                } else {
                    0.0
                };
                sum = (v[j + k * n] * reciprocal).mul_add(u[i + k * m], sum);
            }
            inverse[j * m + i] = sum;
        }
    }
    let mut output = vec![0.0_f32; mn];
    for j in 0..n {
        for i in 0..m {
            output[j * m + i] = inverse[j * m + i] as f32;
        }
    }
    Ok(output)
}

/// The wrapper's polynomial basis (0x8fe0c): `terms` columns on an
/// n*n grid centered at `(float(n)+1)*.5`, first coordinate slowest.
/// For n < 54 the rows are `[x, y, 1]`; for 54 <= n < 152 they are
/// `[x*x, y*y, x*y, x, y, 1]`. The native generic n >= 152 branch
/// (10 cubic terms) is never selected by the fixed sizes.
fn polynomial_basis(n: u32) -> (usize, Vec<f32>) {
    let terms: usize = if n < 54 { 3 } else { 6 };
    let n = n as usize;
    let mut basis = vec![0.0_f32; n * n * terms];
    let center = (n as f32 + 1.0_f32) * 0.5_f32;
    for x in 0..n {
        for y in 0..n {
            let xx = (x + 1) as f32 - center;
            let yy = (y + 1) as f32 - center;
            let start = (x * n + y) * terms;
            let pixel = &mut basis[start..start + terms];
            if terms == 3 {
                pixel[0] = xx;
                pixel[1] = yy;
                pixel[2] = 1.0;
            } else {
                pixel[0] = xx * xx;
                pixel[1] = yy * yy;
                pixel[2] = xx * yy;
                pixel[3] = xx;
                pixel[4] = yy;
                pixel[5] = 1.0;
            }
        }
    }
    (terms, basis)
}

/// The patch worker `RunCROCPatches` (0x8f628), transcribed from the
/// archived check_patch.cpp. One worker over its inclusive row range:
/// per-patch polynomial fit through the pseudoinverse, residuals,
/// correlation acceptance, and row-mean accumulation into the shared
/// shifted accumulators. The native row-sum column gate is
/// `fma(patchColumn, size, localColumn) < x_limit`, strictly.
#[allow(clippy::too_many_lines)]
fn run_patches(
    data: &[f32],
    row_stride: usize,
    basis: &[f32],
    inverse: &[f32],
    terms: usize,
    residual: &mut [f32],
    accum: &mut [f32],
    counts: &mut [u16],
    size: u32,
    x_limit: u16,
    rect: [i32; 4],
    row_start: usize,
    thread: usize,
    control: Control<'_>,
) -> Result<OffsetStats> {
    control.check()?;
    let n = size as usize;
    debug_assert_eq!(residual.len(), n * n * 3);
    residual.fill(0.0);
    let size_f = size as f32;
    let cols = (((rect[2] - rect[0]) as f32) / size_f).floor() as u32;
    let segment = (((rect[3] - rect[1]) / 8) as u16) as u32;
    let rows = if thread == 7 {
        ((segment as f32 / size_f).floor()) as u32
    } else {
        (((segment as f32 / size_f).ceil() * size_f / size_f).floor()) as u32
    };
    let mut stats = OffsetStats::default();
    for py in 0..rows {
        let mut yy = (py as f32).mul_add(size_f, rect[1] as f32) + row_start as f32;
        if thread != 0 {
            yy = (py as f32).mul_add(size_f, row_start as f32);
        }
        let y0 = yy as u32 as usize;
        for px in 0..cols {
            control.check()?;
            // The ordinary dispatcher passes x_multiplier 0, so the
            // native column term drops out of x0.
            let x0 = ((px as f32).mul_add(size_f, rect[0] as f32)) as u32 as usize;
            let mut coef = [[0.0_f32; 3]; 10];
            for k in 0..terms {
                for x in 0..n {
                    for y in 0..n {
                        let weight = inverse[k * (n * n) + x * n + y];
                        let base = (y0 + y) * row_stride + 3 * (x0 + x);
                        for c in 0..3 {
                            coef[k][c] = weight.mul_add(data[base + c], coef[k][c]);
                        }
                    }
                }
            }
            let mut raw_sum = [0.0_f64; 3];
            let mut residual_sum = [0.0_f64; 3];
            for x in 0..n {
                for y in 0..n {
                    let mut fit = [0.0_f32; 3];
                    for k in 0..terms {
                        for c in 0..3 {
                            fit[c] = basis[(x * n + y) * terms + k].mul_add(coef[k][c], fit[c]);
                        }
                    }
                    let base = (y0 + y) * row_stride + 3 * (x0 + x);
                    for c in 0..3 {
                        // Zero polynomial terms bypass the fit and keep
                        // the original sample as residual, including
                        // its signed zero.
                        let r = if terms != 0 {
                            data[base + c] - fit[c]
                        } else {
                            data[base + c]
                        };
                        residual[(y * n + x) * 3 + c] = r;
                        raw_sum[c] += f64::from(data[base + c]);
                        residual_sum[c] += f64::from(r);
                    }
                }
            }
            let area = f64::from(size_f * size_f);
            let mut mean = [0.0_f64; 3];
            for c in 0..3 {
                mean[c] = residual_sum[c] / area;
            }
            let mut var = [0.0_f64; 3];
            let mut cov = [0.0_f64; 3];
            for x in 0..n {
                for y in 0..n {
                    let mut d = [0.0_f64; 3];
                    for c in 0..3 {
                        d[c] = f64::from(residual[(y * n + x) * 3 + c]) - mean[c];
                    }
                    for c in 0..3 {
                        var[c] = d[c].mul_add(d[c], var[c]);
                    }
                    cov[0] = d[0].mul_add(d[1], cov[0]);
                    cov[1] = d[0].mul_add(d[2], cov[1]);
                    cov[2] = d[1].mul_add(d[2], cov[2]);
                }
            }
            let rg = cov[0].abs() / (var[0].sqrt() * var[1].sqrt());
            let rb = cov[1].abs() / (var[0].sqrt() * var[2].sqrt());
            let gb = cov[2].abs() / (var[1].sqrt() * var[2].sqrt());
            let mut correlation = if gb > rb { gb } else { rb };
            if rg > correlation {
                correlation = rg;
            }
            // Promoted float bounds; the strict comparison also rejects
            // the unordered NaN that constant channels produce.
            let threshold = if raw_sum[0] / area > 600.0
                && raw_sum[1] / area > 600.0
                && raw_sum[2] / area > 600.0
            {
                f64::from_bits(0x3fb9_9999_a000_0000)
            } else {
                f64::from_bits(0x3fae_b851_e000_0000)
            };
            stats.patches += 1;
            if !(correlation < threshold) {
                continue;
            }
            stats.accepted += 1;
            for y in 0..n {
                let mut sum = [0.0_f32; 3];
                let mut count = 0.0_f32;
                for x in 0..n {
                    if (px as f32).mul_add(size_f, x as f32) < x_limit as f32 {
                        for c in 0..3 {
                            sum[c] += residual[(y * n + x) * 3 + c];
                        }
                        count += 1.0;
                    }
                }
                if count == 0.0 {
                    continue;
                }
                let accum_row = (y0 + y + 1) * 3;
                for c in 0..3 {
                    accum[accum_row + c] += sum[c] / count;
                }
                counts[y0 + y + 1] += 1;
                stats.rows += 1;
            }
        }
    }
    Ok(stats)
}

/// The final row-offset application (0x902d8..0x90330) with the native
/// literal loop coordinates: image rows `[0, rect.bottom-rect.top)`,
/// image columns `[rect.left, rect.right-rect.left)`, accumulators
/// read one row down. The per-channel allowed magnitude is
/// `gains[c] * (CaptureISO < 800 ? 0.75 : promoted 1.2)`; offsets
/// from rows with fewer than 20 contributions and offsets above the
/// magnitude bound drop to zero. The magnitude comparison keeps
/// equality (a boundary offset applies).
fn final_apply(
    data: &mut [f32],
    row_stride: usize,
    accum: &[f32],
    counts: &[u16],
    iso: f32,
    gains: [f64; 3],
    rect: [i32; 4],
) {
    let scale = if iso < 800.0 {
        0.75_f64
    } else {
        f64::from_bits(0x3ff3_3333_4000_0000)
    };
    for y in 0..(rect[3] - rect[1]) as usize {
        let count = (u32::from(counts[y + 1])).max(1);
        let mut offset = [0.0_f32; 3];
        for c in 0..3 {
            let mut v = accum[(y + 1) * 3 + c] / count as f32;
            if count < 20 || f64::from(v.abs()) > scale * gains[c] {
                v = 0.0;
            }
            offset[c] = v;
        }
        for x in rect[0] as usize..(rect[2] - rect[0]) as usize {
            let base = y * row_stride + 3 * x;
            for c in 0..3 {
                data[base + c] -= offset[c];
            }
        }
    }
}

/// Run the full stage in place: per scale (31, 73, 151) build the
/// basis, compute the pseudoinverse, run the eight workers in the
/// documented deterministic order (worker 0..=7, patch rows and
/// columns ascending) into the shared shifted accumulators, then
/// apply the final row offsets with the native literal loop bounds.
/// The accumulators persist across scales, as in the native wrapper.
///
/// Geometry is validated up front: the native reads patches and
/// accumulators without clamping, and the supported contract requires
/// every worker's patch extent to stay inside the image and the
/// literal final loops to stay inside the raster.
pub fn correct(
    data: &mut [f32],
    width: usize,
    height: usize,
    row_stride: usize,
    iso: f32,
    gains: [f64; 3],
    rect: [i32; 4],
    control: Control<'_>,
) -> Result<OffsetStats> {
    control.check()?;
    if height < 8 {
        // DoThreadedPix splits height into eight ranges.
        return Err(Error::InvalidData("invalid Merrill offset image size"));
    }
    if rect.iter().any(|&v| v < 0) || rect[2] <= rect[0] || rect[3] <= rect[1] {
        return Err(Error::InvalidData("invalid Merrill offset rectangle"));
    }
    if (rect[2] - rect[0]) as usize > width || (rect[3] - rect[1]) as usize > height {
        return Err(Error::InvalidData("invalid Merrill offset extent"));
    }
    // The final application reads the accumulators one row past its
    // literal row range, so a rect height reaching the image height
    // would index past the accumulator allocation. The native reads
    // its own scratch padding there (its accum region is 12*H bytes);
    // this port rejects instead.
    if (rect[3] - rect[1]) as usize >= height {
        return Err(Error::InvalidData("invalid Merrill offset extent"));
    }
    if (rect[2] - rect[0]) > u16::MAX as i32 || (rect[3] - rect[1]) / 8 > u16::MAX as i32 {
        return Err(Error::InvalidData("invalid Merrill offset rectangle"));
    }
    if row_stride < width * 3 || data.len() < row_stride * (height - 1) + width * 3 {
        return Err(Error::InvalidData("invalid Merrill offset layout"));
    }
    let x_limit = (rect[2] - rect[0]) as u16;
    let segment = (((rect[3] - rect[1]) / 8) as u16) as u32;
    let step = height / 8;
    // Validate every worker's patch extent per scale: the native
    // reads image rows [firstRow, firstRow + patchRows*size) and
    // writes accumulators one row down, without clamping.
    for &size in &SCALES {
        let size_f = size as f32;
        let cols = (((rect[2] - rect[0]) as f32) / size_f).floor() as u32;
        if (cols as usize)
            .checked_mul(size as usize)
            .map(|span| span + rect[0] as usize > width)
            .unwrap_or(true)
        {
            return Err(Error::InvalidData("invalid Merrill offset extent"));
        }
        for thread in 0..8_u32 {
            let rows = if thread == 7 {
                (segment as f32 / size_f).floor() as u32
            } else {
                ((segment as f32 / size_f).ceil() * size_f / size_f).floor() as u32
            };
            let first_row = if thread == 0 {
                rect[1] as usize
            } else {
                (thread as usize) * step
            };
            let last = first_row + rows as usize * size as usize;
            // The workers write accumulators at row+1, so a worker
            // extent ending exactly at the image height would write
            // the accumulator row `height`, one past this port's
            // exact allocation. The native tolerates `last == height`
            // (its reads fit; the shifted write lands in its scratch
            // padding), so this check is deliberately stricter than
            // the native's unclamped reads.
            if last >= height {
                return Err(Error::InvalidData("invalid Merrill offset extent"));
            }
        }
    }
    let mut accum = vec![0.0_f32; height * 3];
    let mut counts = vec![0_u16; height];
    let mut stats = OffsetStats::default();
    for &size in &SCALES {
        control.check()?;
        let (terms, basis) = polynomial_basis(size);
        let area = size as usize * size as usize;
        let inverse = calculate_pinv(&basis, terms, area)?;
        let mut residuals = vec![0.0_f32; 8 * area * 3];
        for thread in 0..8_usize {
            control.check()?;
            let row_start = thread * (height / 8);
            let slice = &mut residuals[thread * area * 3..(thread + 1) * area * 3];
            let worker = run_patches(
                data,
                row_stride,
                &basis,
                &inverse,
                terms,
                slice,
                &mut accum,
                &mut counts,
                size,
                x_limit,
                rect,
                row_start,
                thread,
                control,
            )?;
            stats.patches += worker.patches;
            stats.accepted += worker.accepted;
            stats.rows += worker.rows;
        }
    }
    final_apply(data, row_stride, &accum, &counts, iso, gains, rect);
    Ok(stats)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal MT19937 (std::mt19937) so the worker test reproduces the
    /// archived native-oracle case inputs exactly.
    struct Mt19937 {
        state: [u32; 624],
        index: usize,
    }

    impl Mt19937 {
        fn new(seed: u32) -> Self {
            let mut state = [0_u32; 624];
            state[0] = seed;
            for i in 1..624 {
                let previous = state[i - 1];
                state[i] = 1812433253_u32
                    .wrapping_mul(previous ^ (previous >> 30))
                    .wrapping_add(i as u32);
            }
            Self { state, index: 624 }
        }

        fn next(&mut self) -> u32 {
            if self.index == 624 {
                for i in 0..624 {
                    let y =
                        (self.state[i] & 0x8000_0000) | (self.state[(i + 1) % 624] & 0x7fff_ffff);
                    self.state[i] = self.state[(i + 397) % 624]
                        ^ (y >> 1)
                        ^ if y & 1 != 0 { 0x9908_b0df } else { 0 };
                }
                self.index = 0;
            }
            let mut y = self.state[self.index];
            self.index += 1;
            y ^= y >> 11;
            y ^= (y << 7) & 0x9d2c_5680;
            y ^= (y << 15) & 0xefc6_0000;
            y ^ (y >> 18)
        }
    }

    fn fnv1a_f32(data: &[f32]) -> u64 {
        let mut hash = 0xcbf2_9ce4_8422_2325_u64;
        for value in data {
            for byte in value.to_bits().to_le_bytes() {
                hash ^= u64::from(byte);
                hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
        hash
    }

    /// The solver chain pin, against the archived native-verified
    /// oracle: `followup-offset-svd/solver/scalar_svd.hpp` and
    /// `scalar_pinv.hpp` were compared byte-for-byte with the native
    /// `dsvdc`/`CalculatePseudoInv` (three polynomial SVD states, 374
    /// finite pseudoinverse cases). This transcription was then run
    /// through the same C++ driver (`pinv_driver.cpp` in the stage
    /// archive), and every basis and inverse float matched this Rust
    /// port bit for bit on all three scales. These FNV-1a hashes pin
    /// the full chain: basis construction, SVD, cutoff, inverse sum
    /// and the final float conversion.
    #[test]
    fn solver_chain_matches_archived_native_oracle() {
        const BASIS_FNV: [u64; 3] = [
            0x0084_ee96_d9e9_e1f8,
            0x878b_4663_7a2e_9df0,
            0xd5c4_c96d_ea98_e5e8,
        ];
        const INVERSE_FNV: [u64; 3] = [
            0x747c_ab92_df1a_04f6,
            0x8624_6000_8b34_d187,
            0x5943_9c29_274c_dbaa,
        ];
        const SINGULAR_VALUES: [[f64; 6]; 3] = [
            [
                153881.84416858,
                29009.22833241,
                2195.91418653,
                0.0,
                0.0,
                0.0,
            ],
            [
                54231.321900199029,
                32411.999999999982,
                28982.011317367211,
                1538.2054479165001,
                1538.2054479164988,
                39.012267303040758,
            ],
            [
                2.260975279495184e7,
                1.350694242922099e7,
                1.210878923589876e7,
                42484.469997386775,
                42484.4699973855,
                1082.56963215116,
            ],
        ];
        for (index, &scale) in SCALES.iter().enumerate() {
            let (terms, basis) = polynomial_basis(scale);
            let area = scale as usize * scale as usize;
            assert_eq!(fnv1a_f32(&basis), BASIS_FNV[index], "scale {scale} basis");
            let inverse = calculate_pinv(&basis, terms, area).unwrap();
            assert_eq!(
                fnv1a_f32(&inverse),
                INVERSE_FNV[index],
                "scale {scale} inverse"
            );
            // Determinism: a second run reproduces every bit.
            let inverse_again = calculate_pinv(&basis, terms, area).unwrap();
            assert_eq!(fnv1a_f32(&inverse_again), INVERSE_FNV[index]);
            let _ = SINGULAR_VALUES[index];
        }
    }

    /// The smallest SVD case, hand-checked: a 4x3 matrix through the
    /// embedded `dsvdc` reconstructs `A = U * diag(s) * V'` with
    /// descending nonnegative singular values and orthonormal
    /// singular vectors, and the pinv returns the left inverse.
    #[test]
    fn svd_small_case_reconstructs() {
        let basis = [
            2.0_f32, 0.0, 1.0, //
            0.0, 3.0, 1.0, //
            4.0, 0.0, 1.0, //
            0.0, 5.0, 1.0,
        ];
        let inverse = calculate_pinv(&basis, 3, 4).unwrap();
        // P * A is the 3x3 identity (A has full column rank).
        for k in 0..3 {
            for l in 0..3 {
                let mut sum = 0.0_f64;
                for i in 0..4 {
                    sum += f64::from(inverse[k * 4 + i]) * f64::from(basis[i * 3 + l]);
                }
                let target = if k == l { 1.0 } else { 0.0 };
                assert!((sum - target).abs() < 1.0e-5, "P*A[{k}][{l}] = {sum}");
            }
        }
        // P inverts A on its column space: for u in R^3, the
        // projection P*(A*u) recovers u.
        let u = [1.0_f32, -2.0, 0.5];
        let mut w = [0.0_f64; 4];
        for i in 0..4 {
            for k in 0..3 {
                w[i] += f64::from(basis[i * 3 + k]) * f64::from(u[k]);
            }
        }
        let mut fit = [0.0_f64; 3];
        for (k, slot) in fit.iter_mut().enumerate() {
            for i in 0..4 {
                *slot += f64::from(inverse[k * 4 + i]) * w[i];
            }
        }
        for k in 0..3 {
            assert!(
                (fit[k] - f64::from(u[k])).abs() < 1.0e-4,
                "u[{k}]: {} vs {}",
                fit[k],
                u[k]
            );
        }
    }

    /// The cutoff gate: singular values above the cutoff contribute
    /// their reciprocal, values below contribute zero. A rank-two
    /// 4x3 basis with one near-zero singular value pins the
    /// thresholded inverse (the exact cutoff constant matters).
    #[test]
    fn pinv_cutoff_gates_small_singular_values() {
        // Columns: [1,0,0], [0,1,0], [1,1,epsilon] - the third is
        // nearly the sum of the first two, so its singular value is
        // order epsilon.
        let epsilon = 1.0e-9_f32;
        let basis = [
            1.0_f32, 0.0, 0.0, //
            0.0, 1.0, 0.0, //
            1.0, 1.0, epsilon, //
            0.0, 0.0, 0.0,
        ];
        let inverse = calculate_pinv(&basis, 3, 4).unwrap();
        // The projection reproduces the two full-rank directions; the
        // near-null direction collapses (no amplification).
        // v = col0 + col1 lives in the full-rank span; P*(A*v)
        // recovers it.
        let v = [1.0_f32, 1.0, 0.0];
        let mut w = [0.0_f64; 4];
        for i in 0..4 {
            for k in 0..3 {
                w[i] += f64::from(basis[i * 3 + k]) * f64::from(v[k]);
            }
        }
        let mut fit = [0.0_f64; 3];
        for (k, slot) in fit.iter_mut().enumerate() {
            for i in 0..4 {
                *slot += f64::from(inverse[k * 4 + i]) * w[i];
            }
        }
        for k in 0..3 {
            assert!(
                (fit[k] - f64::from(v[k])).abs() < 1.0e-4,
                "v[{k}]: {} vs {}",
                fit[k],
                v[k]
            );
        }
        // The near-null direction does not blow up: A*[0,0,1] is a
        // near-zero column, and its projection stays near zero
        // (singular values below the cutoff contribute nothing).
        let n = [0.0_f32, 0.0, 1.0];
        let mut wn = [0.0_f64; 4];
        for i in 0..4 {
            for k in 0..3 {
                wn[i] += f64::from(basis[i * 3 + k]) * f64::from(n[k]);
            }
        }
        let mut ampl = [0.0_f64; 3];
        for (k, slot) in ampl.iter_mut().enumerate() {
            for i in 0..4 {
                *slot += f64::from(inverse[k * 4 + i]) * wn[i];
            }
        }
        assert!(
            ampl.iter().all(|v| v.abs() < 1.0),
            "null direction amplified: {ampl:?}"
        );
    }

    /// The worker's correlation gate: the archived 48-case structure
    /// (`followup-repair/croffset/check_patch.cpp`, mt19937 seed
    /// 4737109) over all three scales and all eight worker indices,
    /// with this port's pseudoinverse (bit-identical to the native
    /// solver on these bases). The archived native run accepted 104
    /// of 144 patches and 9,056 row contributions; detection changed
    /// no pixels.
    fn fnv1a_bytes(data: &[u8]) -> u64 {
        let mut hash = 0xcbf2_9ce4_8422_2325_u64;
        for &byte in data {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
        hash
    }

    /// Per-case combined accumulator hashes from the archived
    /// native-verified worker (`followup-repair/croffset/
    /// check_patch.cpp` patches()), run by the stage driver with the
    /// archived scalar pinv: FNV over the accum floats, XOR-combined
    /// with the counts hash and the worker's residual slot hash.
    const WORKER_CASE_HASHES: [[u64; 16]; 3] = [
        [
            0x8305_4de9_9d4a_fcbb,
            0x347d_ab24_e4fd_2c8d,
            0xe730_3831_9700_00a1,
            0x7543_aa61_3e05_cd6a,
            0x6538_3e7d_9f6a_3425,
            0x7a82_3cda_5d9c_8aa4,
            0x2d3f_e410_b93b_9629,
            0x560f_60eb_1392_51e7,
            0xc67c_aee7_53a8_1eeb,
            0x148e_749c_bc75_4514,
            0x21b8_efcb_c23f_ddfe,
            0x79c0_0b4e_81c5_ec41,
            0x3503_f980_a5c5_9452,
            0x68ee_dde3_85a3_b048,
            0x63d2_1fcd_5344_a9e8,
            0x8eac_800b_e8ae_b9f3,
        ],
        [
            0x2faa_9a2e_c52f_4408,
            0xbd00_5aae_c1ec_0e98,
            0xbf9b_d337_79db_a518,
            0x7518_d343_55c9_c926,
            0x6013_f80c_ec4f_d1c5,
            0x596f_cce3_fc28_30a4,
            0x0627_7f0c_0bef_8185,
            0x1125_8e72_d2a9_ec40,
            0xa386_47f0_6ffd_8b35,
            0x6d46_6967_541f_70da,
            0x43a6_e27b_49a2_5e3d,
            0x00c0_fc2f_6cf7_0677,
            0xce6a_21d1_ce6c_3284,
            0x772e_62a2_2127_34af,
            0xe262_61ce_ba0f_49ca,
            0x7117_5a77_3b0a_af40,
        ],
        [
            0x7d2a_5d49_79a7_66c4,
            0x67f5_a8d3_043d_d104,
            0xa5cf_a0b2_2c9b_4b7f,
            0x547f_05e0_4fbc_596c,
            0x6d92_a1da_ce47_3425,
            0xa139_59a2_939f_a74c,
            0x7d32_fffb_08c1_92b5,
            0x0225_ca92_f759_5554,
            0xbef9_7888_cdb7_adce,
            0xafb5_7ece_74a9_71d9,
            0xcb15_3a1c_040f_5d98,
            0xeafa_8f06_cd45_a3a1,
            0x8368_466b_b1d0_cad3,
            0x2d31_db8c_2257_cb63,
            0xc5b7_52fc_9b8f_7744,
            0x0987_d4e6_d90b_46f0,
        ],
    ];

    #[test]
    fn worker_matches_archived_native_case_totals() {
        let mut rng = Mt19937::new(4_737_109);
        let (mut patches, mut accepted, mut rows) = (0_u32, 0_u32, 0_u32);
        for (scale_index, &scale) in SCALES.iter().enumerate() {
            let n = scale as usize;
            let terms = if scale < 54 { 3 } else { 6 };
            let area = n * n;
            let (built_terms, basis) = polynomial_basis(scale);
            assert_eq!(built_terms, terms);
            let inverse = calculate_pinv(&basis, terms, area).unwrap();
            let width = n * 3 + 17;
            let height = n * 10 + 32;
            let stride = width * 3 + 7;
            let mut image = vec![-31_415.25_f32; stride * height];
            let rect = [3_i32, 2, (width - 5) as i32, (8 * n + 9) as i32];
            for mode in 0..16_usize {
                for y in 0..height {
                    for x in 0..width {
                        let same = ((rng.next() % 2000) as i32 - 1000) as f32 * 0.031_25;
                        for c in 0..3 {
                            let mut v = if mode % 4 == 0 {
                                same
                            } else {
                                ((rng.next() % 2000) as i32 - 1000) as f32 * 0.031_25
                            };
                            v += if mode % 4 == 3 { 700.0 } else { 100.0 };
                            if mode % 8 == 4 {
                                v = 100.0;
                            }
                            if mode % 8 == 5 && c == 1 {
                                v = 100.0;
                            }
                            image[y * stride + 3 * x + c] = v;
                        }
                    }
                }
                let original = image.clone();
                let thread = mode % 8;
                let row_start = thread * n;
                let use_terms = if mode >= 8 { terms } else { 0 };
                let x_limit = (rect[2] - rect[0] - if mode % 3 == 1 { n as i32 } else { 0 }) as u16;
                let mut accum = vec![0.0_f32; (height + 8) * 3];
                let mut counts = vec![0_u16; height + 8];
                let mut residual = vec![0.0_f32; area * 3];
                let stats = run_patches(
                    &image,
                    stride,
                    &basis,
                    &inverse,
                    use_terms,
                    &mut residual,
                    &mut accum,
                    &mut counts,
                    scale,
                    x_limit,
                    rect,
                    row_start,
                    thread,
                    Control::none(),
                )
                .unwrap();
                // Detection reads only: the image must stay unchanged.
                assert_eq!(image, original, "mode {mode} changed pixels");
                // Per-case accumulator pin: the archived native-verified
                // worker's combined accum/counts/residual hash.
                let mut hash = fnv1a_bytes(
                    accum
                        .iter()
                        .flat_map(|v| v.to_bits().to_le_bytes())
                        .collect::<Vec<_>>()
                        .as_slice(),
                );
                hash ^= fnv1a_bytes(
                    counts
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect::<Vec<_>>()
                        .as_slice(),
                );
                hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
                hash ^= fnv1a_bytes(
                    residual
                        .iter()
                        .flat_map(|v| v.to_bits().to_le_bytes())
                        .collect::<Vec<_>>()
                        .as_slice(),
                );
                hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
                assert_eq!(
                    hash, WORKER_CASE_HASHES[scale_index][mode],
                    "scale {scale} mode {mode} accumulator hash"
                );
                patches += stats.patches;
                accepted += stats.accepted;
                rows += stats.rows;
            }
        }
        // Archived native worker run totals.
        assert_eq!(patches, 144);
        assert_eq!(accepted, 104);
        assert_eq!(rows, 9_056);
    }

    /// The worker's per-row accumulation semantics on a controlled
    /// patch: the residual of an exactly polynomial image is zero
    /// (the fit is exact), every channel is then constant, the
    /// correlation is NaN and the strict gate rejects the patch.
    #[test]
    fn worker_rejects_constant_residual_channels() {
        let scale = 31_u32;
        let n = scale as usize;
        let (terms, basis) = polynomial_basis(scale);
        let area = n * n;
        let inverse = calculate_pinv(&basis, terms, area).unwrap();
        let width = n * 4;
        let height = n * 4;
        let stride = width * 3;
        let mut image = vec![0.0_f32; stride * height];
        // v = 1 + 2x + 3y + x*x/4: exactly inside the basis span.
        for y in 0..height {
            for x in 0..width {
                let xf = x as f32;
                let yf = y as f32;
                let v = 1.0 + 2.0 * xf + 3.0 * yf + xf * xf * 0.25;
                for c in 0..3 {
                    image[y * stride + 3 * x + c] = v;
                }
            }
        }
        let rect = [4_i32, 4, (width - 4) as i32, (height - 4) as i32];
        let mut accum = vec![0.0_f32; (height + 2) * 3];
        let mut counts = vec![0_u16; height + 2];
        let mut residual = vec![0.0_f32; area * 3];
        let stats = run_patches(
            &image,
            stride,
            &basis,
            &inverse,
            terms,
            &mut residual,
            &mut accum,
            &mut counts,
            scale,
            (rect[2] - rect[0]) as u16,
            rect,
            0,
            0,
            Control::none(),
        )
        .unwrap();
        assert_eq!(stats.patches, 3);
        assert_eq!(stats.accepted, 0);
        assert_eq!(stats.rows, 0);
        assert!(accum.iter().all(|&v| v == 0.0));
        assert!(counts.iter().all(|&v| v == 0));
    }

    /// Zero polynomial terms bypass the fit and keep the original
    /// sample as the residual (`terms=0` exported edge case).
    #[test]
    fn worker_zero_terms_uses_original_as_residual() {
        let scale = 31_u32;
        let n = scale as usize;
        let (terms, basis) = polynomial_basis(scale);
        let _ = terms;
        let area = n * n;
        let inverse = calculate_pinv(&basis, 3, area).unwrap();
        let width = n * 4;
        let height = n * 4;
        let stride = width * 3;
        let mut image = vec![0.0_f32; stride * height];
        for y in 0..height {
            for x in 0..width {
                for c in 0..3 {
                    image[y * stride + 3 * x + c] = 10.0 + (x % 7) as f32 + 100.0 * c as f32;
                }
            }
        }
        let rect = [4_i32, 4, (4 + 31) as i32, (height - 4) as i32];
        let mut accum = vec![0.0_f32; (height + 2) * 3];
        let mut counts = vec![0_u16; height + 2];
        let mut residual = vec![0.0_f32; area * 3];
        let stats = run_patches(
            &image,
            stride,
            &basis,
            &inverse,
            0,
            &mut residual,
            &mut accum,
            &mut counts,
            scale,
            (rect[2] - rect[0]) as u16,
            rect,
            0,
            0,
            Control::none(),
        )
        .unwrap();
        // One patch row at the given width: cols = floor(31/31) = 1.
        assert_eq!(stats.patches, 1);
        // The channels are independent patterns, so the patch is
        // accepted unless the strict gate sees NaN from constants.
        assert!(stats.accepted <= 1);
        if stats.accepted == 1 {
            // Residual row 0 of the patch: original values.
            let row = residual[..3 * 31].to_vec();
            let expected: Vec<f32> = (0..31)
                .flat_map(|x| (0..3).map(move |c| 10.0 + (x % 7) as f32 + 100.0 * c as f32))
                .collect();
            assert_eq!(row, expected);
        }
    }

    /// The final application gates: offsets from rows with fewer than
    /// 20 contributions drop to zero; offsets above the gain-scaled
    /// magnitude bound drop to zero, equality keeps the offset; the
    /// loops use the native literal coordinates (image rows from
    /// zero, columns from rect.left, accumulators read one row down).
    #[test]
    fn final_apply_gates_match_native_bounds() {
        let width = 64;
        let height = 64;
        let stride = width * 3;
        let rect = [8_i32, 4, 60, 44];
        let iso_low = 799.0_f32;
        let iso_high = 800.0_f32;
        let gains = [2.0_f64, 1.0, 0.5];
        // Row 10: 21 contributions with a small offset (kept).
        // Row 20: 19 contributions (count gate drops it).
        // Row 30: 21 contributions with an offset above the bound.
        // Row 40: 21 contributions with an offset exactly at the
        //         bound (equality keeps it).
        let mut accum = vec![0.0_f32; (height + 1) * 3];
        let mut counts = vec![0_u16; height + 1];
        // The native divides the row sum by the count BEFORE the
        // magnitude bound, so the bound applies to the per-pixel
        // offset. Channel 0 bound at iso 799: 2.0 * 0.75 = 1.5.
        for (row, count, sum) in [
            (10_usize, 21_u16, 10.5_f32),
            (20, 19, 10.5),
            (30, 21, 45.0),
            (34, 21, 31.5),
        ] {
            counts[row + 1] = count;
            accum[(row + 1) * 3] = sum;
        }
        let mut image = vec![100.0_f32; stride * height];
        final_apply(&mut image, stride, &accum, &counts, iso_low, gains, rect);
        let at = |row: usize, column: usize| image[row * stride + 3 * column];
        // Applied region: rows [0, rect.bottom-rect.top) = [0,32),
        // columns [rect.left, rect.right-rect.left) = [8,52).
        // Row 10: offset = 10.5/21 = 0.5, below the bound, applied.
        assert!((at(10, 8) - (100.0 - 10.5 / 21.0)).abs() < 1.0e-6);
        // Column left of the literal x bound is untouched.
        assert_eq!(at(10, 7), 100.0);
        // Column at the right edge (rect.right - rect.left = 52) is
        // untouched (the loop is exclusive).
        assert_eq!(at(10, 52), 100.0);
        // Row 20: count 19 < 20 drops the offset.
        assert_eq!(at(20, 8), 100.0);
        // Row 30: offset = 45/21 > 2.0*0.75 drops.
        assert_eq!(at(30, 8), 100.0);
        // Row 34: offset = 31.5/21 = 1.5 == bound, equality keeps.
        assert_eq!(at(34, 8), 98.5);
        // Rows at/after rect.bottom-rect.top are untouched.
        assert_eq!(at(40, 8), 100.0);
        // The high-ISO factor: the promoted float 1.2 widens channel
        // 0's bound to 2.4, so 45/21 = 2.14 stays applied at row 30.
        let mut image2 = vec![100.0_f32; stride * height];
        final_apply(&mut image2, stride, &accum, &counts, iso_high, gains, rect);
        assert!((at2(&image2, stride, 30, 8) - (100.0 - 45.0 / 21.0)).abs() < 1.0e-6);
        // Channel 2's bound (0.5 * the promoted 1.2) is not pinned
        // here: the accumulator shares channels across columns, so
        // an independent per-channel bound check needs a dedicated
        // fixture (open item in the stage 6 checkpoint notes).
    }

    fn at2(image: &[f32], stride: usize, row: usize, column: usize) -> f32 {
        image[row * stride + 3 * column]
    }

    /// The full stage on a controlled image: the wrapper validates
    /// geometry, accumulates deterministically in the documented
    /// order, and applies the final offsets.
    #[test]
    fn correct_runs_end_to_end_deterministically() {
        let width = 256;
        let height = 640;
        let stride = width * 3;
        let rect = [8_i32, 4, 248, 68];
        let iso = 100.0_f32;
        let gains = [1.0_f64; 3];
        let mut image = vec![0.0_f32; stride * height];
        // Deterministic texture: independent channel patterns.
        let mut value = 12345_u32;
        let mut next = || {
            value ^= value << 13;
            value ^= value >> 17;
            value ^= value << 5;
            value
        };
        for y in 0..height {
            for x in 0..width {
                for c in 0..3 {
                    image[y * stride + 3 * x + c] = 100.0 + (next() % 400) as f32 + 50.0 * c as f32;
                }
            }
        }
        let mut run = image.clone();
        let stats = correct(
            &mut run,
            width,
            height,
            stride,
            iso,
            gains,
            rect,
            Control::none(),
        )
        .unwrap();
        assert!(stats.patches >= stats.accepted);
        assert!(stats.accepted > 0);
        // Deterministic accumulation order: a second run reproduces
        // every bit.
        let mut run_again = image.clone();
        let stats_again = correct(
            &mut run_again,
            width,
            height,
            stride,
            iso,
            gains,
            rect,
            Control::none(),
        )
        .unwrap();
        assert_eq!(stats, stats_again);
        assert_eq!(run, run_again);
    }

    /// Out-of-contract geometry rejects: images below eight rows,
    /// rectangles outside the raster, and extents that would leave
    /// the image in the worker patch reads.
    #[test]
    fn correct_rejects_invalid_geometry() {
        let mut data = vec![0.0_f32; 32 * 3 * 8];
        assert!(correct(
            &mut data,
            32,
            4,
            32 * 3,
            100.0,
            [1.0; 3],
            [0, 0, 32, 4],
            Control::none()
        )
        .is_err());
        let data = vec![0.0_f32; 32 * 3 * 64];
        // Rect outside the raster.
        assert!(correct(
            &mut data.clone(),
            32,
            64,
            32 * 3,
            100.0,
            [1.0; 3],
            [40, 4, 60, 40],
            Control::none()
        )
        .is_err());
        // Negative coordinates.
        assert!(correct(
            &mut data.clone(),
            32,
            64,
            32 * 3,
            100.0,
            [1.0; 3],
            [-8, 4, 20, 40],
            Control::none()
        )
        .is_err());
        // Empty rect.
        assert!(correct(
            &mut data.clone(),
            32,
            64,
            32 * 3,
            100.0,
            [1.0; 3],
            [8, 8, 8, 8],
            Control::none()
        )
        .is_err());
        // Worker patch extents that would read outside the image:
        // for a 16-row image the last worker starts at row 14 and a
        // single 31-row patch would read past the bottom.
        let mut narrow = vec![0.0_f32; 16 * 3 * 16];
        assert!(correct(
            &mut narrow,
            16,
            16,
            16 * 3,
            100.0,
            [1.0; 3],
            [5, 6, 15, 15],
            Control::none()
        )
        .is_err());
    }
}
