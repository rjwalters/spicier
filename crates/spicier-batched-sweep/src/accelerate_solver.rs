//! Apple Accelerate framework-based batched LU solver.
//!
//! This module provides CPU-based batched LU solving using Apple's Accelerate
//! framework, which includes highly optimized LAPACK routines tuned for Apple
//! Silicon and Intel Macs.
//!
//! Accelerate is only available on macOS/iOS. This module uses direct FFI
//! bindings to avoid dependency conflicts with lapack-src.

use crate::error::{BatchedSweepError, Result};
use crate::solver::{BackendType, BatchedLuSolver, BatchedSolveResult, GpuBatchConfig};

// Direct FFI bindings to Accelerate's LAPACK routines.
// These avoid the lapack-src version conflicts mentioned in workspace Cargo.toml.
#[link(name = "Accelerate", kind = "framework")]
unsafe extern "C" {
    /// LAPACK dgesv: Solve a general system of linear equations A*X = B.
    ///
    /// Arguments:
    /// - n: Number of linear equations (matrix dimension)
    /// - nrhs: Number of right-hand sides (columns of B)
    /// - a: Matrix A (overwritten with LU factors), column-major
    /// - lda: Leading dimension of A
    /// - ipiv: Pivot indices (output)
    /// - b: Right-hand side matrix B (overwritten with solution), column-major
    /// - ldb: Leading dimension of B
    /// - info: 0 = success, < 0 = invalid arg, > 0 = singular matrix
    fn dgesv_(
        n: *const i32,
        nrhs: *const i32,
        a: *mut f64,
        lda: *const i32,
        ipiv: *mut i32,
        b: *mut f64,
        ldb: *const i32,
        info: *mut i32,
    );
}

/// Accelerate-backed batched LU solver.
///
/// Uses Apple's Accelerate framework for high-performance LU decomposition
/// with optimizations for Apple Silicon (M1/M2/M3) and Intel Macs.
pub struct AccelerateBatchedSolver {
    config: GpuBatchConfig,
}

impl AccelerateBatchedSolver {
    /// Create a new Accelerate batched solver.
    pub fn new(config: GpuBatchConfig) -> Self {
        Self { config }
    }
}

impl BatchedLuSolver for AccelerateBatchedSolver {
    fn solve_batch(
        &self,
        matrices: &[f64],
        rhs: &[f64],
        n: usize,
        batch_size: usize,
    ) -> Result<BatchedSolveResult> {
        let expected_matrix_len = batch_size * n * n;
        let expected_rhs_len = batch_size * n;

        if matrices.len() != expected_matrix_len {
            return Err(BatchedSweepError::InvalidDimension(format!(
                "Expected {} matrix elements, got {}",
                expected_matrix_len,
                matrices.len()
            )));
        }

        if rhs.len() != expected_rhs_len {
            return Err(BatchedSweepError::InvalidDimension(format!(
                "Expected {} RHS elements, got {}",
                expected_rhs_len,
                rhs.len()
            )));
        }

        let mut solutions = Vec::with_capacity(expected_rhs_len);
        let mut singular_indices = Vec::new();

        // LAPACK parameters
        let n_i32 = n as i32;
        let nrhs: i32 = 1; // One right-hand side per system
        let lda = n_i32;
        let ldb = n_i32;

        // Pivot array (reused for each system)
        let mut ipiv = vec![0i32; n];

        for i in 0..batch_size {
            // Copy matrix data (dgesv overwrites it with LU factors)
            let mat_start = i * n * n;
            let mut a_copy: Vec<f64> = matrices[mat_start..mat_start + n * n].to_vec();

            // Copy RHS data (dgesv overwrites it with solution)
            let rhs_start = i * n;
            let mut b_copy: Vec<f64> = rhs[rhs_start..rhs_start + n].to_vec();

            let mut info: i32 = 0;

            // Call Accelerate's dgesv
            // Safety: All pointers are valid, dimensions match, and memory is properly sized
            unsafe {
                dgesv_(
                    &n_i32,
                    &nrhs,
                    a_copy.as_mut_ptr(),
                    &lda,
                    ipiv.as_mut_ptr(),
                    b_copy.as_mut_ptr(),
                    &ldb,
                    &mut info,
                );
            }

            if info == 0 {
                // Success: b_copy now contains the solution
                solutions.extend(b_copy);
            } else if info > 0 {
                // Singular matrix (U(info,info) is exactly zero)
                solutions.extend(std::iter::repeat(0.0).take(n));
                singular_indices.push(i);
            } else {
                // Negative info indicates invalid argument - this shouldn't happen
                // with our validated inputs, but handle it gracefully
                return Err(BatchedSweepError::Backend(format!(
                    "dgesv returned error code {} for system {}",
                    info, i
                )));
            }
        }

        Ok(BatchedSolveResult {
            solutions,
            singular_indices,
            n,
            batch_size,
        })
    }

    fn should_use_gpu(&self, _matrix_size: usize, _batch_size: usize) -> bool {
        false // Accelerate is CPU-based (though it may use Apple's Neural Engine)
    }

    fn backend_type(&self) -> BackendType {
        BackendType::Accelerate
    }

    fn config(&self) -> &GpuBatchConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_accelerate_solver_identity() {
        let solver = AccelerateBatchedSolver::new(GpuBatchConfig::default());

        let n = 2;
        let batch_size = 2;

        // Two 2x2 identity matrices in column-major order
        let matrices = vec![
            1.0, 0.0, 0.0, 1.0, // Identity 0
            1.0, 0.0, 0.0, 1.0, // Identity 1
        ];

        let rhs = vec![
            1.0, 2.0, // b0 = [1, 2]
            3.0, 4.0, // b1 = [3, 4]
        ];

        let result = solver.solve_batch(&matrices, &rhs, n, batch_size).unwrap();

        assert_eq!(result.batch_size, 2);
        assert!(result.singular_indices.is_empty());

        let sol0 = result.solution(0).unwrap();
        assert!((sol0[0] - 1.0).abs() < 1e-10);
        assert!((sol0[1] - 2.0).abs() < 1e-10);

        let sol1 = result.solution(1).unwrap();
        assert!((sol1[0] - 3.0).abs() < 1e-10);
        assert!((sol1[1] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_accelerate_solver_simple_system() {
        let solver = AccelerateBatchedSolver::new(GpuBatchConfig::default());

        let n = 2;
        let batch_size = 1;

        // Matrix [[2, 1], [1, 3]] in column-major: [2, 1, 1, 3]
        let matrices = vec![2.0, 1.0, 1.0, 3.0];
        let rhs = vec![5.0, 10.0]; // b = [5, 10]

        // Solution: x = [1, 3]
        // Check: 2*1 + 1*3 = 5, 1*1 + 3*3 = 10

        let result = solver.solve_batch(&matrices, &rhs, n, batch_size).unwrap();

        assert!(result.singular_indices.is_empty());
        let sol = result.solution(0).unwrap();
        assert!((sol[0] - 1.0).abs() < 1e-10);
        assert!((sol[1] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_accelerate_solver_singular() {
        let solver = AccelerateBatchedSolver::new(GpuBatchConfig::default());

        let n = 2;
        let batch_size = 2;

        // Matrix 0: identity (non-singular)
        // Matrix 1: [[1, 2], [1, 2]] (singular) - column-major: [1, 1, 2, 2]
        let matrices = vec![
            1.0, 0.0, 0.0, 1.0, // Identity
            1.0, 1.0, 2.0, 2.0, // Singular
        ];
        let rhs = vec![1.0, 2.0, 1.0, 2.0];

        let result = solver.solve_batch(&matrices, &rhs, n, batch_size).unwrap();

        assert!(result.is_singular(1));
        assert!(!result.is_singular(0));
    }

    #[test]
    fn test_accelerate_solver_larger_system() {
        let solver = AccelerateBatchedSolver::new(GpuBatchConfig::default());

        let n = 3;
        let batch_size = 1;

        // 3x3 matrix in column-major:
        // [[1, 2, 3],
        //  [0, 1, 4],
        //  [5, 6, 0]]
        // Column-major: [1, 0, 5, 2, 1, 6, 3, 4, 0]
        let matrices = vec![1.0, 0.0, 5.0, 2.0, 1.0, 6.0, 3.0, 4.0, 0.0];
        let rhs = vec![1.0, 2.0, 3.0];

        let result = solver.solve_batch(&matrices, &rhs, n, batch_size).unwrap();

        assert!(result.singular_indices.is_empty());

        // Verify solution satisfies Ax = b
        let sol = result.solution(0).unwrap();
        let ax0 = 1.0 * sol[0] + 2.0 * sol[1] + 3.0 * sol[2];
        let ax1 = 0.0 * sol[0] + 1.0 * sol[1] + 4.0 * sol[2];
        let ax2 = 5.0 * sol[0] + 6.0 * sol[1] + 0.0 * sol[2];

        assert!((ax0 - 1.0).abs() < 1e-10);
        assert!((ax1 - 2.0).abs() < 1e-10);
        assert!((ax2 - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_accelerate_backend_type() {
        let solver = AccelerateBatchedSolver::new(GpuBatchConfig::default());
        assert_eq!(solver.backend_type(), BackendType::Accelerate);
    }
}
