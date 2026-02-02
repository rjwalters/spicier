//! GMRES iterative solver for linear systems.
//!
//! Provides both complex and real-valued GMRES solvers. The complex solver
//! uses SIMD-accelerated conjugate dot products for Gram-Schmidt orthogonalization.
//!
//! # Usage
//!
//! ```ignore
//! use spicier_solver::{solve_gmres, solve_gmres_real, GmresConfig};
//!
//! // Complex system
//! let result = solve_gmres(&complex_operator, &complex_rhs, &GmresConfig::default());
//!
//! // Real system
//! let result = solve_gmres_real(&real_operator, &real_rhs, &GmresConfig::default());
//! ```

use crate::operator::{ComplexOperator, RealOperator};
use crate::preconditioner::{ComplexPreconditioner, RealPreconditioner};
use num_complex::Complex64 as C64;
use spicier_simd::{complex_conjugate_dot_product, real_dot_product, SimdCapability};

/// GMRES solver configuration.
#[derive(Debug, Clone)]
pub struct GmresConfig {
    /// Maximum number of iterations.
    pub max_iter: usize,
    /// Convergence tolerance (relative residual).
    pub tol: f64,
    /// Restart parameter (Krylov subspace dimension before restart).
    pub restart: usize,
}

impl Default for GmresConfig {
    fn default() -> Self {
        Self {
            max_iter: 500,
            tol: 1e-8,
            restart: 30,
        }
    }
}

/// Result of a complex GMRES solve.
#[derive(Debug, Clone)]
pub struct GmresResult {
    /// Solution vector.
    pub x: Vec<C64>,
    /// Number of iterations performed.
    pub iterations: usize,
    /// Final relative residual.
    pub residual: f64,
    /// Whether the solver converged.
    pub converged: bool,
}

/// Result of a real-valued GMRES solve.
#[derive(Debug, Clone)]
pub struct RealGmresResult {
    /// Solution vector.
    pub x: Vec<f64>,
    /// Number of iterations performed.
    pub iterations: usize,
    /// Final relative residual.
    pub residual: f64,
    /// Whether the solver converged.
    pub converged: bool,
}

/// Solve A*x = b using restarted GMRES.
///
/// Uses SIMD-accelerated conjugate dot products for Gram-Schmidt
/// orthogonalization when available (AVX-512, AVX2 on x86/x86_64).
pub fn solve_gmres(op: &dyn ComplexOperator, b: &[C64], config: &GmresConfig) -> GmresResult {
    let simd_cap = SimdCapability::detect();

    let n = op.dim();
    assert_eq!(b.len(), n, "RHS dimension mismatch");

    let b_norm = vec_norm(b, simd_cap);
    if b_norm < 1e-30 {
        return GmresResult {
            x: vec![C64::new(0.0, 0.0); n],
            iterations: 0,
            residual: 0.0,
            converged: true,
        };
    }

    let mut x = vec![C64::new(0.0, 0.0); n];
    let mut total_iter = 0;

    for _restart_cycle in 0..config.max_iter {
        // Compute residual r = b - A*x
        let mut ax = vec![C64::new(0.0, 0.0); n];
        op.apply(&x, &mut ax);
        let mut r: Vec<C64> = b
            .iter()
            .zip(ax.iter())
            .map(|(&bi, &axi)| bi - axi)
            .collect();
        let r_norm = vec_norm(&r, simd_cap);

        if r_norm / b_norm < config.tol {
            return GmresResult {
                x,
                iterations: total_iter,
                residual: r_norm / b_norm,
                converged: true,
            };
        }

        // Arnoldi process with modified Gram-Schmidt
        let m = config.restart.min(n);
        let mut v: Vec<Vec<C64>> = Vec::with_capacity(m + 1);
        let mut h = vec![vec![C64::new(0.0, 0.0); m + 1]; m];

        // v[0] = r / ||r||
        let inv_r_norm = 1.0 / r_norm;
        for ri in r.iter_mut() {
            *ri *= inv_r_norm;
        }
        v.push(r);

        // g = ||r|| * e_1
        let mut g = vec![C64::new(0.0, 0.0); m + 1];
        g[0] = C64::new(r_norm, 0.0);

        // Givens rotation storage
        let mut cs = vec![C64::new(0.0, 0.0); m];
        let mut sn = vec![C64::new(0.0, 0.0); m];

        let mut k = 0;
        while k < m {
            total_iter += 1;
            if total_iter > config.max_iter {
                break;
            }

            // w = A * v[k]
            let mut w = vec![C64::new(0.0, 0.0); n];
            op.apply(&v[k], &mut w);

            // Modified Gram-Schmidt
            for j in 0..=k {
                let hij = complex_conjugate_dot_product(&v[j], &w, simd_cap);
                h[k][j] = hij;
                for i in 0..n {
                    w[i] -= hij * v[j][i];
                }
            }

            let w_norm = vec_norm(&w, simd_cap);
            h[k][k + 1] = C64::new(w_norm, 0.0);

            if w_norm < 1e-30 {
                // Lucky breakdown
                k += 1;
                break;
            }

            let inv_w = 1.0 / w_norm;
            let vk1: Vec<C64> = w.iter().map(|&wi| wi * inv_w).collect();
            v.push(vk1);

            // Apply previous Givens rotations to h[k]
            for j in 0..k {
                let temp = cs[j].conj() * h[k][j] + sn[j].conj() * h[k][j + 1];
                h[k][j + 1] = -sn[j] * h[k][j] + cs[j] * h[k][j + 1];
                h[k][j] = temp;
            }

            // Compute new Givens rotation
            let (c, s) = givens_rotation(h[k][k], h[k][k + 1]);
            cs[k] = c;
            sn[k] = s;

            let temp = c.conj() * h[k][k] + s.conj() * h[k][k + 1];
            h[k][k + 1] = C64::new(0.0, 0.0);
            h[k][k] = temp;

            let temp_g = c.conj() * g[k] + s.conj() * g[k + 1];
            g[k + 1] = -s * g[k] + c * g[k + 1];
            g[k] = temp_g;

            let rel_res = g[k + 1].norm() / b_norm;
            if rel_res < config.tol {
                k += 1;
                break;
            }

            k += 1;
        }

        // Back-substitution to find y from H*y = g
        let mut y = vec![C64::new(0.0, 0.0); k];
        for i in (0..k).rev() {
            let mut sum = g[i];
            for j in (i + 1)..k {
                sum -= h[j][i] * y[j];
            }
            if h[i][i].norm() > 1e-30 {
                y[i] = sum / h[i][i];
            }
        }

        // Update x = x + V * y
        for i in 0..k {
            for j in 0..n {
                x[j] += v[i][j] * y[i];
            }
        }

        // Check convergence
        let mut ax_final = vec![C64::new(0.0, 0.0); n];
        op.apply(&x, &mut ax_final);
        let final_res: f64 = b
            .iter()
            .zip(ax_final.iter())
            .map(|(&bi, &axi)| (bi - axi).norm_sqr())
            .sum::<f64>()
            .sqrt();

        if final_res / b_norm < config.tol {
            return GmresResult {
                x,
                iterations: total_iter,
                residual: final_res / b_norm,
                converged: true,
            };
        }

        if total_iter >= config.max_iter {
            return GmresResult {
                x,
                iterations: total_iter,
                residual: final_res / b_norm,
                converged: false,
            };
        }
    }

    // Should not reach here
    GmresResult {
        x,
        iterations: total_iter,
        residual: f64::NAN,
        converged: false,
    }
}

// ============================================================================
// Real-valued GMRES
// ============================================================================

/// Solve A*x = b using restarted GMRES for real-valued systems.
///
/// Uses SIMD-accelerated dot products for Gram-Schmidt orthogonalization
/// when available (AVX-512, AVX2 on x86/x86_64).
///
/// This is more efficient than using the complex GMRES for real systems
/// since it avoids complex arithmetic overhead.
pub fn solve_gmres_real(
    op: &dyn RealOperator,
    b: &[f64],
    config: &GmresConfig,
) -> RealGmresResult {
    let simd_cap = SimdCapability::detect();

    let n = op.dim();
    assert_eq!(b.len(), n, "RHS dimension mismatch");

    let b_norm = real_vec_norm(b, simd_cap);
    if b_norm < 1e-30 {
        return RealGmresResult {
            x: vec![0.0; n],
            iterations: 0,
            residual: 0.0,
            converged: true,
        };
    }

    let mut x = vec![0.0; n];
    let mut total_iter = 0;

    for _restart_cycle in 0..config.max_iter {
        // Compute residual r = b - A*x
        let mut ax = vec![0.0; n];
        op.apply(&x, &mut ax);
        let mut r: Vec<f64> = b.iter().zip(ax.iter()).map(|(&bi, &axi)| bi - axi).collect();
        let r_norm = real_vec_norm(&r, simd_cap);

        if r_norm / b_norm < config.tol {
            return RealGmresResult {
                x,
                iterations: total_iter,
                residual: r_norm / b_norm,
                converged: true,
            };
        }

        // Arnoldi process with modified Gram-Schmidt
        let m = config.restart.min(n);
        let mut v: Vec<Vec<f64>> = Vec::with_capacity(m + 1);
        let mut h = vec![vec![0.0; m + 1]; m];

        // v[0] = r / ||r||
        let inv_r_norm = 1.0 / r_norm;
        for ri in r.iter_mut() {
            *ri *= inv_r_norm;
        }
        v.push(r);

        // g = ||r|| * e_1
        let mut g = vec![0.0; m + 1];
        g[0] = r_norm;

        // Givens rotation storage
        let mut cs = vec![0.0; m];
        let mut sn = vec![0.0; m];

        let mut k = 0;
        while k < m {
            total_iter += 1;
            if total_iter > config.max_iter {
                break;
            }

            // w = A * v[k]
            let mut w = vec![0.0; n];
            op.apply(&v[k], &mut w);

            // Modified Gram-Schmidt
            for j in 0..=k {
                let hij = real_dot_product(&v[j], &w, simd_cap);
                h[k][j] = hij;
                for i in 0..n {
                    w[i] -= hij * v[j][i];
                }
            }

            let w_norm = real_vec_norm(&w, simd_cap);
            h[k][k + 1] = w_norm;

            if w_norm < 1e-30 {
                // Lucky breakdown
                k += 1;
                break;
            }

            let inv_w = 1.0 / w_norm;
            let vk1: Vec<f64> = w.iter().map(|&wi| wi * inv_w).collect();
            v.push(vk1);

            // Apply previous Givens rotations to h[k]
            for j in 0..k {
                let temp = cs[j] * h[k][j] + sn[j] * h[k][j + 1];
                h[k][j + 1] = -sn[j] * h[k][j] + cs[j] * h[k][j + 1];
                h[k][j] = temp;
            }

            // Compute new Givens rotation
            let (c, s) = real_givens_rotation(h[k][k], h[k][k + 1]);
            cs[k] = c;
            sn[k] = s;

            let temp = c * h[k][k] + s * h[k][k + 1];
            h[k][k + 1] = 0.0;
            h[k][k] = temp;

            let temp_g = c * g[k] + s * g[k + 1];
            g[k + 1] = -s * g[k] + c * g[k + 1];
            g[k] = temp_g;

            let rel_res = g[k + 1].abs() / b_norm;
            if rel_res < config.tol {
                k += 1;
                break;
            }

            k += 1;
        }

        // Back-substitution to find y from H*y = g
        let mut y = vec![0.0; k];
        for i in (0..k).rev() {
            let mut sum = g[i];
            for j in (i + 1)..k {
                sum -= h[j][i] * y[j];
            }
            if h[i][i].abs() > 1e-30 {
                y[i] = sum / h[i][i];
            }
        }

        // Update x = x + V * y
        for i in 0..k {
            for j in 0..n {
                x[j] += v[i][j] * y[i];
            }
        }

        // Check convergence
        let mut ax_final = vec![0.0; n];
        op.apply(&x, &mut ax_final);
        let final_res: f64 = b
            .iter()
            .zip(ax_final.iter())
            .map(|(&bi, &axi)| (bi - axi).powi(2))
            .sum::<f64>()
            .sqrt();

        if final_res / b_norm < config.tol {
            return RealGmresResult {
                x,
                iterations: total_iter,
                residual: final_res / b_norm,
                converged: true,
            };
        }

        if total_iter >= config.max_iter {
            return RealGmresResult {
                x,
                iterations: total_iter,
                residual: final_res / b_norm,
                converged: false,
            };
        }
    }

    // Should not reach here
    RealGmresResult {
        x,
        iterations: total_iter,
        residual: f64::NAN,
        converged: false,
    }
}

// ============================================================================
// Preconditioned GMRES
// ============================================================================

/// Solve A*x = b using right-preconditioned GMRES for real systems.
///
/// Solves the system A*M^(-1)*y = b, then x = M^(-1)*y.
/// Right preconditioning preserves the residual norm ||b - Ax||.
///
/// # Arguments
/// * `op` - The matrix operator A
/// * `precond` - The preconditioner M (approximates A)
/// * `b` - Right-hand side vector
/// * `config` - GMRES configuration
pub fn solve_gmres_real_preconditioned(
    op: &dyn RealOperator,
    precond: &dyn RealPreconditioner,
    b: &[f64],
    config: &GmresConfig,
) -> RealGmresResult {
    let simd_cap = SimdCapability::detect();

    let n = op.dim();
    assert_eq!(b.len(), n, "RHS dimension mismatch");
    assert_eq!(precond.dim(), n, "Preconditioner dimension mismatch");

    let b_norm = real_vec_norm(b, simd_cap);
    if b_norm < 1e-30 {
        return RealGmresResult {
            x: vec![0.0; n],
            iterations: 0,
            residual: 0.0,
            converged: true,
        };
    }

    let mut x = vec![0.0; n];
    let mut total_iter = 0;

    // Workspace for preconditioner application
    let mut precond_work = vec![0.0; n];

    for _restart_cycle in 0..config.max_iter {
        // Compute residual r = b - A*x
        let mut ax = vec![0.0; n];
        op.apply(&x, &mut ax);
        let mut r: Vec<f64> = b.iter().zip(ax.iter()).map(|(&bi, &axi)| bi - axi).collect();
        let r_norm = real_vec_norm(&r, simd_cap);

        if r_norm / b_norm < config.tol {
            return RealGmresResult {
                x,
                iterations: total_iter,
                residual: r_norm / b_norm,
                converged: true,
            };
        }

        // Arnoldi process with modified Gram-Schmidt
        let m = config.restart.min(n);
        let mut v: Vec<Vec<f64>> = Vec::with_capacity(m + 1);
        let mut z: Vec<Vec<f64>> = Vec::with_capacity(m); // z[k] = M^(-1) * v[k]
        let mut h = vec![vec![0.0; m + 1]; m];

        // v[0] = r / ||r||
        let inv_r_norm = 1.0 / r_norm;
        for ri in r.iter_mut() {
            *ri *= inv_r_norm;
        }
        v.push(r);

        // g = ||r|| * e_1
        let mut g = vec![0.0; m + 1];
        g[0] = r_norm;

        // Givens rotation storage
        let mut cs = vec![0.0; m];
        let mut sn = vec![0.0; m];

        let mut k = 0;
        while k < m {
            total_iter += 1;
            if total_iter > config.max_iter {
                break;
            }

            // z[k] = M^(-1) * v[k]
            precond.apply(&v[k], &mut precond_work);
            z.push(precond_work.clone());

            // w = A * z[k] = A * M^(-1) * v[k]
            let mut w = vec![0.0; n];
            op.apply(&z[k], &mut w);

            // Modified Gram-Schmidt
            for j in 0..=k {
                let hij = real_dot_product(&v[j], &w, simd_cap);
                h[k][j] = hij;
                for i in 0..n {
                    w[i] -= hij * v[j][i];
                }
            }

            let w_norm = real_vec_norm(&w, simd_cap);
            h[k][k + 1] = w_norm;

            if w_norm < 1e-30 {
                k += 1;
                break;
            }

            let inv_w = 1.0 / w_norm;
            let vk1: Vec<f64> = w.iter().map(|&wi| wi * inv_w).collect();
            v.push(vk1);

            // Apply previous Givens rotations to h[k]
            for j in 0..k {
                let temp = cs[j] * h[k][j] + sn[j] * h[k][j + 1];
                h[k][j + 1] = -sn[j] * h[k][j] + cs[j] * h[k][j + 1];
                h[k][j] = temp;
            }

            // Compute new Givens rotation
            let (c, s) = real_givens_rotation(h[k][k], h[k][k + 1]);
            cs[k] = c;
            sn[k] = s;

            let temp = c * h[k][k] + s * h[k][k + 1];
            h[k][k + 1] = 0.0;
            h[k][k] = temp;

            let temp_g = c * g[k] + s * g[k + 1];
            g[k + 1] = -s * g[k] + c * g[k + 1];
            g[k] = temp_g;

            let rel_res = g[k + 1].abs() / b_norm;
            if rel_res < config.tol {
                k += 1;
                break;
            }

            k += 1;
        }

        // Back-substitution to find y from H*y = g
        let mut y = vec![0.0; k];
        for i in (0..k).rev() {
            let mut sum = g[i];
            for j in (i + 1)..k {
                sum -= h[j][i] * y[j];
            }
            if h[i][i].abs() > 1e-30 {
                y[i] = sum / h[i][i];
            }
        }

        // Update x = x + Z * y where Z = [z[0], z[1], ..., z[k-1]]
        for i in 0..k {
            for j in 0..n {
                x[j] += z[i][j] * y[i];
            }
        }

        // Check convergence
        let mut ax_final = vec![0.0; n];
        op.apply(&x, &mut ax_final);
        let final_res: f64 = b
            .iter()
            .zip(ax_final.iter())
            .map(|(&bi, &axi)| (bi - axi).powi(2))
            .sum::<f64>()
            .sqrt();

        if final_res / b_norm < config.tol {
            return RealGmresResult {
                x,
                iterations: total_iter,
                residual: final_res / b_norm,
                converged: true,
            };
        }

        if total_iter >= config.max_iter {
            return RealGmresResult {
                x,
                iterations: total_iter,
                residual: final_res / b_norm,
                converged: false,
            };
        }
    }

    RealGmresResult {
        x,
        iterations: total_iter,
        residual: f64::NAN,
        converged: false,
    }
}

/// Solve A*x = b using right-preconditioned GMRES for complex systems.
///
/// Solves the system A*M^(-1)*y = b, then x = M^(-1)*y.
pub fn solve_gmres_preconditioned(
    op: &dyn ComplexOperator,
    precond: &dyn ComplexPreconditioner,
    b: &[C64],
    config: &GmresConfig,
) -> GmresResult {
    let simd_cap = SimdCapability::detect();

    let n = op.dim();
    assert_eq!(b.len(), n, "RHS dimension mismatch");
    assert_eq!(precond.dim(), n, "Preconditioner dimension mismatch");

    let b_norm = vec_norm(b, simd_cap);
    if b_norm < 1e-30 {
        return GmresResult {
            x: vec![C64::new(0.0, 0.0); n],
            iterations: 0,
            residual: 0.0,
            converged: true,
        };
    }

    let mut x = vec![C64::new(0.0, 0.0); n];
    let mut total_iter = 0;
    let mut precond_work = vec![C64::new(0.0, 0.0); n];

    for _restart_cycle in 0..config.max_iter {
        let mut ax = vec![C64::new(0.0, 0.0); n];
        op.apply(&x, &mut ax);
        let mut r: Vec<C64> = b
            .iter()
            .zip(ax.iter())
            .map(|(&bi, &axi)| bi - axi)
            .collect();
        let r_norm = vec_norm(&r, simd_cap);

        if r_norm / b_norm < config.tol {
            return GmresResult {
                x,
                iterations: total_iter,
                residual: r_norm / b_norm,
                converged: true,
            };
        }

        let m = config.restart.min(n);
        let mut v: Vec<Vec<C64>> = Vec::with_capacity(m + 1);
        let mut z: Vec<Vec<C64>> = Vec::with_capacity(m);
        let mut h = vec![vec![C64::new(0.0, 0.0); m + 1]; m];

        let inv_r_norm = 1.0 / r_norm;
        for ri in r.iter_mut() {
            *ri *= inv_r_norm;
        }
        v.push(r);

        let mut g = vec![C64::new(0.0, 0.0); m + 1];
        g[0] = C64::new(r_norm, 0.0);

        let mut cs = vec![C64::new(0.0, 0.0); m];
        let mut sn = vec![C64::new(0.0, 0.0); m];

        let mut k = 0;
        while k < m {
            total_iter += 1;
            if total_iter > config.max_iter {
                break;
            }

            precond.apply(&v[k], &mut precond_work);
            z.push(precond_work.clone());

            let mut w = vec![C64::new(0.0, 0.0); n];
            op.apply(&z[k], &mut w);

            for j in 0..=k {
                let hij = complex_conjugate_dot_product(&v[j], &w, simd_cap);
                h[k][j] = hij;
                for i in 0..n {
                    w[i] -= hij * v[j][i];
                }
            }

            let w_norm = vec_norm(&w, simd_cap);
            h[k][k + 1] = C64::new(w_norm, 0.0);

            if w_norm < 1e-30 {
                k += 1;
                break;
            }

            let inv_w = 1.0 / w_norm;
            let vk1: Vec<C64> = w.iter().map(|&wi| wi * inv_w).collect();
            v.push(vk1);

            for j in 0..k {
                let temp = cs[j].conj() * h[k][j] + sn[j].conj() * h[k][j + 1];
                h[k][j + 1] = -sn[j] * h[k][j] + cs[j] * h[k][j + 1];
                h[k][j] = temp;
            }

            let (c, s) = givens_rotation(h[k][k], h[k][k + 1]);
            cs[k] = c;
            sn[k] = s;

            let temp = c.conj() * h[k][k] + s.conj() * h[k][k + 1];
            h[k][k + 1] = C64::new(0.0, 0.0);
            h[k][k] = temp;

            let temp_g = c.conj() * g[k] + s.conj() * g[k + 1];
            g[k + 1] = -s * g[k] + c * g[k + 1];
            g[k] = temp_g;

            let rel_res = g[k + 1].norm() / b_norm;
            if rel_res < config.tol {
                k += 1;
                break;
            }

            k += 1;
        }

        let mut y = vec![C64::new(0.0, 0.0); k];
        for i in (0..k).rev() {
            let mut sum = g[i];
            for j in (i + 1)..k {
                sum -= h[j][i] * y[j];
            }
            if h[i][i].norm() > 1e-30 {
                y[i] = sum / h[i][i];
            }
        }

        for i in 0..k {
            for j in 0..n {
                x[j] += z[i][j] * y[i];
            }
        }

        let mut ax_final = vec![C64::new(0.0, 0.0); n];
        op.apply(&x, &mut ax_final);
        let final_res: f64 = b
            .iter()
            .zip(ax_final.iter())
            .map(|(&bi, &axi)| (bi - axi).norm_sqr())
            .sum::<f64>()
            .sqrt();

        if final_res / b_norm < config.tol {
            return GmresResult {
                x,
                iterations: total_iter,
                residual: final_res / b_norm,
                converged: true,
            };
        }

        if total_iter >= config.max_iter {
            return GmresResult {
                x,
                iterations: total_iter,
                residual: final_res / b_norm,
                converged: false,
            };
        }
    }

    GmresResult {
        x,
        iterations: total_iter,
        residual: f64::NAN,
        converged: false,
    }
}

// ============================================================================
// Helper functions
// ============================================================================

/// Compute the 2-norm of a complex vector using SIMD-accelerated dot product.
fn vec_norm(v: &[C64], cap: SimdCapability) -> f64 {
    complex_conjugate_dot_product(v, v, cap).re.sqrt()
}

/// Compute the 2-norm of a real vector using SIMD-accelerated dot product.
fn real_vec_norm(v: &[f64], cap: SimdCapability) -> f64 {
    real_dot_product(v, v, cap).sqrt()
}

/// Compute Givens rotation coefficients for complex values.
fn givens_rotation(a: C64, b: C64) -> (C64, C64) {
    if b.norm() < 1e-30 {
        return (C64::new(1.0, 0.0), C64::new(0.0, 0.0));
    }
    let r = (a.norm_sqr() + b.norm_sqr()).sqrt();
    let c = a / r;
    let s = b / r;
    (c, s)
}

/// Compute Givens rotation coefficients for real values.
fn real_givens_rotation(a: f64, b: f64) -> (f64, f64) {
    if b.abs() < 1e-30 {
        return (1.0, 0.0);
    }
    let r = (a * a + b * b).sqrt();
    let c = a / r;
    let s = b / r;
    (c, s)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Simple diagonal operator for testing.
    struct DiagOp {
        diag: Vec<C64>,
    }

    impl ComplexOperator for DiagOp {
        fn dim(&self) -> usize {
            self.diag.len()
        }

        fn apply(&self, x: &[C64], y: &mut [C64]) {
            for i in 0..self.diag.len() {
                y[i] = self.diag[i] * x[i];
            }
        }
    }

    /// Dense matrix operator for testing.
    struct DenseOp {
        matrix: Vec<Vec<C64>>,
        n: usize,
    }

    impl DenseOp {
        fn new(matrix: Vec<Vec<C64>>) -> Self {
            let n = matrix.len();
            Self { matrix, n }
        }
    }

    impl ComplexOperator for DenseOp {
        fn dim(&self) -> usize {
            self.n
        }

        #[allow(clippy::needless_range_loop)]
        fn apply(&self, x: &[C64], y: &mut [C64]) {
            for i in 0..self.n {
                y[i] = C64::new(0.0, 0.0);
                for j in 0..self.n {
                    y[i] += self.matrix[i][j] * x[j];
                }
            }
        }
    }

    #[test]
    fn gmres_diagonal_system() {
        let n = 10;
        let diag: Vec<C64> = (1..=n)
            .map(|i| C64::new(i as f64, 0.5 * i as f64))
            .collect();
        let op = DiagOp { diag: diag.clone() };

        let b: Vec<C64> = diag.iter().map(|d| d * C64::new(1.0, 1.0)).collect();

        let config = GmresConfig::default();
        let result = solve_gmres(&op, &b, &config);

        assert!(result.converged, "GMRES did not converge");
        assert!(result.residual < 1e-6);

        for xi in &result.x {
            assert!((xi - C64::new(1.0, 1.0)).norm() < 1e-6);
        }
    }

    #[test]
    fn gmres_zero_rhs() {
        let n = 5;
        let diag: Vec<C64> = (1..=n).map(|i| C64::new(i as f64, 0.0)).collect();
        let op = DiagOp { diag };

        let b = vec![C64::new(0.0, 0.0); n];
        let config = GmresConfig::default();
        let result = solve_gmres(&op, &b, &config);

        assert!(result.converged);
        assert_eq!(result.iterations, 0);
        for xi in &result.x {
            assert!(xi.norm() < 1e-15);
        }
    }

    #[test]
    fn gmres_identity_operator() {
        let n = 5;
        let diag = vec![C64::new(1.0, 0.0); n];
        let op = DiagOp { diag };

        let b: Vec<C64> = (1..=n)
            .map(|i| C64::new(i as f64, -0.5 * i as f64))
            .collect();
        let config = GmresConfig::default();
        let result = solve_gmres(&op, &b, &config);

        assert!(result.converged);
        for (xi, bi) in result.x.iter().zip(b.iter()) {
            assert!((xi - bi).norm() < 1e-10);
        }
    }

    #[test]
    fn gmres_real_symmetric_positive_definite() {
        let matrix = vec![
            vec![C64::new(4.0, 0.0), C64::new(1.0, 0.0)],
            vec![C64::new(1.0, 0.0), C64::new(3.0, 0.0)],
        ];
        let op = DenseOp::new(matrix);

        let b = vec![C64::new(5.0, 0.0), C64::new(4.0, 0.0)];
        let config = GmresConfig::default();
        let result = solve_gmres(&op, &b, &config);

        assert!(result.converged);
        assert!((result.x[0] - C64::new(1.0, 0.0)).norm() < 1e-8);
        assert!((result.x[1] - C64::new(1.0, 0.0)).norm() < 1e-8);
    }

    #[test]
    fn gmres_complex_hermitian() {
        let matrix = vec![
            vec![C64::new(2.0, 0.0), C64::new(1.0, -1.0)],
            vec![C64::new(1.0, 1.0), C64::new(3.0, 0.0)],
        ];
        let op = DenseOp::new(matrix);

        let b = vec![
            C64::new(2.0, 0.0) + C64::new(1.0, -1.0),
            C64::new(1.0, 1.0) + C64::new(3.0, 0.0),
        ];
        let config = GmresConfig::default();
        let result = solve_gmres(&op, &b, &config);

        assert!(result.converged);
        assert!((result.x[0] - C64::new(1.0, 0.0)).norm() < 1e-8);
        assert!((result.x[1] - C64::new(1.0, 0.0)).norm() < 1e-8);
    }

    #[test]
    fn gmres_restart_behavior() {
        let n = 50;
        let diag: Vec<C64> = (1..=n).map(|i| C64::new(i as f64, 0.5)).collect();
        let op = DiagOp { diag: diag.clone() };

        let b: Vec<C64> = diag.iter().map(|d| d * C64::new(1.0, 1.0)).collect();

        let config = GmresConfig {
            max_iter: 200,
            tol: 1e-8,
            restart: 5,
        };
        let result = solve_gmres(&op, &b, &config);

        assert!(result.converged);
        assert!(result.residual < 1e-6);
    }

    #[test]
    fn gmres_config_default() {
        let config = GmresConfig::default();
        assert_eq!(config.max_iter, 500);
        assert!((config.tol - 1e-8).abs() < 1e-15);
        assert_eq!(config.restart, 30);
    }

    #[test]
    fn gmres_single_element() {
        let a = C64::new(3.0, 4.0);
        let b_val = C64::new(6.0, 8.0);
        let expected_x = b_val / a;

        let op = DiagOp { diag: vec![a] };
        let b = vec![b_val];
        let config = GmresConfig::default();
        let result = solve_gmres(&op, &b, &config);

        assert!(result.converged);
        assert!((result.x[0] - expected_x).norm() < 1e-10);
    }

    #[test]
    fn vec_norm_test() {
        let cap = SimdCapability::detect();
        let v = vec![C64::new(3.0, 4.0)];
        assert!((vec_norm(&v, cap) - 5.0).abs() < 1e-15);
    }

    #[test]
    fn givens_rotation_test() {
        let a = C64::new(3.0, 0.0);
        let b = C64::new(4.0, 0.0);
        let (c, s) = givens_rotation(a, b);

        let new_b = -s * a + c * b;
        assert!(new_b.norm() < 1e-10);
    }

    // ========================================================================
    // Real GMRES tests
    // ========================================================================

    /// Simple diagonal real operator for testing.
    struct RealDiagOp {
        diag: Vec<f64>,
    }

    impl RealOperator for RealDiagOp {
        fn dim(&self) -> usize {
            self.diag.len()
        }

        fn apply(&self, x: &[f64], y: &mut [f64]) {
            for i in 0..self.diag.len() {
                y[i] = self.diag[i] * x[i];
            }
        }
    }

    /// Dense real matrix operator for testing.
    struct RealDenseOp {
        matrix: Vec<Vec<f64>>,
        n: usize,
    }

    impl RealDenseOp {
        fn new(matrix: Vec<Vec<f64>>) -> Self {
            let n = matrix.len();
            Self { matrix, n }
        }
    }

    impl RealOperator for RealDenseOp {
        fn dim(&self) -> usize {
            self.n
        }

        #[allow(clippy::needless_range_loop)]
        fn apply(&self, x: &[f64], y: &mut [f64]) {
            for i in 0..self.n {
                y[i] = 0.0;
                for j in 0..self.n {
                    y[i] += self.matrix[i][j] * x[j];
                }
            }
        }
    }

    #[test]
    fn gmres_real_diagonal_system() {
        let n = 10;
        let diag: Vec<f64> = (1..=n).map(|i| i as f64).collect();
        let op = RealDiagOp { diag: diag.clone() };

        // b = A * [1, 1, ..., 1]
        let b: Vec<f64> = diag.iter().map(|&d| d * 1.0).collect();

        let config = GmresConfig::default();
        let result = solve_gmres_real(&op, &b, &config);

        assert!(result.converged, "Real GMRES did not converge");
        assert!(result.residual < 1e-6);

        for xi in &result.x {
            assert!((xi - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn gmres_real_zero_rhs() {
        let n = 5;
        let diag: Vec<f64> = (1..=n).map(|i| i as f64).collect();
        let op = RealDiagOp { diag };

        let b = vec![0.0; n];
        let config = GmresConfig::default();
        let result = solve_gmres_real(&op, &b, &config);

        assert!(result.converged);
        assert_eq!(result.iterations, 0);
        for xi in &result.x {
            assert!(xi.abs() < 1e-15);
        }
    }

    #[test]
    fn gmres_real_identity_operator() {
        let n = 5;
        let diag = vec![1.0; n];
        let op = RealDiagOp { diag };

        let b: Vec<f64> = (1..=n).map(|i| i as f64).collect();
        let config = GmresConfig::default();
        let result = solve_gmres_real(&op, &b, &config);

        assert!(result.converged);
        for (xi, &bi) in result.x.iter().zip(b.iter()) {
            assert!((xi - bi).abs() < 1e-10);
        }
    }

    #[test]
    fn gmres_real_spd_system() {
        // Symmetric positive definite matrix:
        // [4 1]
        // [1 3]
        let matrix = vec![vec![4.0, 1.0], vec![1.0, 3.0]];
        let op = RealDenseOp::new(matrix);

        // b = A * [1, 1] = [5, 4]
        let b = vec![5.0, 4.0];
        let config = GmresConfig::default();
        let result = solve_gmres_real(&op, &b, &config);

        assert!(result.converged);
        assert!((result.x[0] - 1.0).abs() < 1e-8);
        assert!((result.x[1] - 1.0).abs() < 1e-8);
    }

    #[test]
    fn gmres_real_tridiagonal() {
        // Tridiagonal matrix:
        // [2 -1  0]
        // [-1 2 -1]
        // [0 -1  2]
        let matrix = vec![
            vec![2.0, -1.0, 0.0],
            vec![-1.0, 2.0, -1.0],
            vec![0.0, -1.0, 2.0],
        ];
        let op = RealDenseOp::new(matrix);

        // b = A * [1, 2, 3] = [0, 0, 4]
        let b = vec![0.0, 0.0, 4.0];
        let config = GmresConfig::default();
        let result = solve_gmres_real(&op, &b, &config);

        assert!(result.converged);
        assert!((result.x[0] - 1.0).abs() < 1e-8);
        assert!((result.x[1] - 2.0).abs() < 1e-8);
        assert!((result.x[2] - 3.0).abs() < 1e-8);
    }

    #[test]
    fn gmres_real_restart_behavior() {
        let n = 50;
        let diag: Vec<f64> = (1..=n).map(|i| i as f64 + 0.5).collect();
        let op = RealDiagOp { diag: diag.clone() };

        let b: Vec<f64> = diag.iter().map(|&d| d * 1.0).collect();

        let config = GmresConfig {
            max_iter: 200,
            tol: 1e-8,
            restart: 5,
        };
        let result = solve_gmres_real(&op, &b, &config);

        assert!(result.converged);
        assert!(result.residual < 1e-6);
    }

    #[test]
    fn real_vec_norm_test() {
        let cap = SimdCapability::detect();
        let v = vec![3.0, 4.0];
        assert!((real_vec_norm(&v, cap) - 5.0).abs() < 1e-15);
    }

    #[test]
    fn real_givens_rotation_test() {
        let (c, s) = real_givens_rotation(3.0, 4.0);

        // After rotation, b component should be zero
        let new_b = -s * 3.0 + c * 4.0;
        assert!(new_b.abs() < 1e-10);

        // Check normalization: c^2 + s^2 = 1
        assert!((c * c + s * s - 1.0).abs() < 1e-15);
    }

    // ========================================================================
    // Preconditioned GMRES tests
    // ========================================================================

    use crate::preconditioner::{IdentityPreconditioner, JacobiPreconditioner};

    #[test]
    fn preconditioned_gmres_real_with_identity() {
        // Identity preconditioner should give same result as unpreconditioned
        let n = 10;
        let diag: Vec<f64> = (1..=n).map(|i| i as f64).collect();
        let op = RealDiagOp { diag: diag.clone() };
        let precond = IdentityPreconditioner::new(n);

        let b: Vec<f64> = diag.iter().map(|&d| d * 1.0).collect();
        let config = GmresConfig::default();

        let result = solve_gmres_real_preconditioned(&op, &precond, &b, &config);

        assert!(result.converged);
        for xi in &result.x {
            assert!((xi - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn preconditioned_gmres_real_with_jacobi() {
        // Jacobi preconditioner on a diagonal system should converge in 1 iteration
        let n = 10;
        let diag: Vec<f64> = (1..=n).map(|i| i as f64).collect();
        let op = RealDiagOp { diag: diag.clone() };
        let precond = JacobiPreconditioner::from_diagonal(&diag);

        let b: Vec<f64> = diag.iter().map(|&d| d * 1.0).collect();
        let config = GmresConfig::default();

        let result = solve_gmres_real_preconditioned(&op, &precond, &b, &config);

        assert!(result.converged);
        // With perfect Jacobi preconditioner on diagonal matrix, should converge fast
        assert!(result.iterations <= 2);
        for xi in &result.x {
            assert!((xi - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn preconditioned_gmres_real_spd_system() {
        // SPD system with Jacobi preconditioning
        let matrix = vec![vec![4.0, 1.0], vec![1.0, 3.0]];
        let op = RealDenseOp::new(matrix);

        // Jacobi uses diagonal: [4, 3]
        let precond = JacobiPreconditioner::from_diagonal(&[4.0, 3.0]);

        let b = vec![5.0, 4.0]; // Solution is [1, 1]
        let config = GmresConfig::default();

        let result = solve_gmres_real_preconditioned(&op, &precond, &b, &config);

        assert!(result.converged);
        assert!((result.x[0] - 1.0).abs() < 1e-6);
        assert!((result.x[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn preconditioned_gmres_complex_with_identity() {
        let n = 5;
        let diag: Vec<C64> = (1..=n)
            .map(|i| C64::new(i as f64, 0.5 * i as f64))
            .collect();
        let op = DiagOp { diag: diag.clone() };
        let precond = IdentityPreconditioner::new(n);

        let b: Vec<C64> = diag.iter().map(|d| d * C64::new(1.0, 1.0)).collect();
        let config = GmresConfig::default();

        let result = solve_gmres_preconditioned(&op, &precond, &b, &config);

        assert!(result.converged);
        for xi in &result.x {
            assert!((xi - C64::new(1.0, 1.0)).norm() < 1e-6);
        }
    }

    #[test]
    fn preconditioned_gmres_zero_rhs() {
        let n = 5;
        let diag: Vec<f64> = (1..=n).map(|i| i as f64).collect();
        let op = RealDiagOp { diag: diag.clone() };
        let precond = JacobiPreconditioner::from_diagonal(&diag);

        let b = vec![0.0; n];
        let config = GmresConfig::default();

        let result = solve_gmres_real_preconditioned(&op, &precond, &b, &config);

        assert!(result.converged);
        assert_eq!(result.iterations, 0);
        for xi in &result.x {
            assert!(xi.abs() < 1e-15);
        }
    }

    #[test]
    fn preconditioned_gmres_from_triplets() {
        // Test Jacobi preconditioner built from triplets
        let triplets = vec![
            (0, 0, 4.0),
            (0, 1, 1.0),
            (1, 0, 1.0),
            (1, 1, 3.0),
        ];
        let matrix = vec![vec![4.0, 1.0], vec![1.0, 3.0]];
        let op = RealDenseOp::new(matrix);
        let precond = JacobiPreconditioner::from_triplets(2, &triplets);

        let b = vec![5.0, 4.0];
        let config = GmresConfig::default();

        let result = solve_gmres_real_preconditioned(&op, &precond, &b, &config);

        assert!(result.converged);
        assert!((result.x[0] - 1.0).abs() < 1e-6);
        assert!((result.x[1] - 1.0).abs() < 1e-6);
    }
}
