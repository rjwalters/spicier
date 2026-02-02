//! Real-valued GMRES solver.

use crate::operator::RealOperator;
use crate::preconditioner::RealPreconditioner;
use spicier_simd::{SimdCapability, real_dot_product};

use super::GmresConfig;
use super::helpers::{real_givens_rotation, real_vec_norm};

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

/// Solve A*x = b using restarted GMRES for real-valued systems.
///
/// Uses SIMD-accelerated dot products for Gram-Schmidt orthogonalization
/// when available (AVX-512, AVX2 on x86/x86_64).
///
/// This is more efficient than using the complex GMRES for real systems
/// since it avoids complex arithmetic overhead.
pub fn solve_gmres_real(op: &dyn RealOperator, b: &[f64], config: &GmresConfig) -> RealGmresResult {
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
        let mut r: Vec<f64> = b
            .iter()
            .zip(ax.iter())
            .map(|(&bi, &axi)| bi - axi)
            .collect();
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
        let mut r: Vec<f64> = b
            .iter()
            .zip(ax.iter())
            .map(|(&bi, &axi)| bi - axi)
            .collect();
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::preconditioner::{IdentityPreconditioner, JacobiPreconditioner};

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
        let matrix = vec![vec![4.0, 1.0], vec![1.0, 3.0]];
        let op = RealDenseOp::new(matrix);

        let b = vec![5.0, 4.0];
        let config = GmresConfig::default();
        let result = solve_gmres_real(&op, &b, &config);

        assert!(result.converged);
        assert!((result.x[0] - 1.0).abs() < 1e-8);
        assert!((result.x[1] - 1.0).abs() < 1e-8);
    }

    #[test]
    fn gmres_real_tridiagonal() {
        let matrix = vec![
            vec![2.0, -1.0, 0.0],
            vec![-1.0, 2.0, -1.0],
            vec![0.0, -1.0, 2.0],
        ];
        let op = RealDenseOp::new(matrix);

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
    fn preconditioned_gmres_real_with_identity() {
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
        let n = 10;
        let diag: Vec<f64> = (1..=n).map(|i| i as f64).collect();
        let op = RealDiagOp { diag: diag.clone() };
        let precond = JacobiPreconditioner::from_diagonal(&diag);

        let b: Vec<f64> = diag.iter().map(|&d| d * 1.0).collect();
        let config = GmresConfig::default();

        let result = solve_gmres_real_preconditioned(&op, &precond, &b, &config);

        assert!(result.converged);
        assert!(result.iterations <= 2);
        for xi in &result.x {
            assert!((xi - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn preconditioned_gmres_real_spd_system() {
        let matrix = vec![vec![4.0, 1.0], vec![1.0, 3.0]];
        let op = RealDenseOp::new(matrix);
        let precond = JacobiPreconditioner::from_diagonal(&[4.0, 3.0]);

        let b = vec![5.0, 4.0];
        let config = GmresConfig::default();

        let result = solve_gmres_real_preconditioned(&op, &precond, &b, &config);

        assert!(result.converged);
        assert!((result.x[0] - 1.0).abs() < 1e-6);
        assert!((result.x[1] - 1.0).abs() < 1e-6);
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
        let triplets = vec![(0, 0, 4.0), (0, 1, 1.0), (1, 0, 1.0), (1, 1, 3.0)];
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
