//! Faer-based high-performance batched LU solver.
//!
//! This module provides CPU-based batched LU solving using the faer crate,
//! which is a modern, high-performance linear algebra library for Rust.
//! Faer uses SIMD optimizations for efficient computation on modern CPUs.

use crate::error::{BatchedSweepError, Result};
use crate::solver::{BackendType, BatchedLuSolver, BatchedSolveResult, GpuBatchConfig};
use faer::prelude::*;

/// Faer-backed batched LU solver.
///
/// Uses the faer crate for high-performance LU decomposition with SIMD
/// optimizations. This provides excellent CPU performance without requiring
/// external BLAS/LAPACK dependencies.
pub struct FaerBatchedSolver {
    config: GpuBatchConfig,
}

impl FaerBatchedSolver {
    /// Create a new Faer batched solver.
    pub fn new(config: GpuBatchConfig) -> Self {
        Self { config }
    }
}

impl BatchedLuSolver for FaerBatchedSolver {
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

        for i in 0..batch_size {
            // Extract matrix (column-major order) - faer uses column-major
            let mat_start = i * n * n;
            let mat_data = &matrices[mat_start..mat_start + n * n];

            // Create faer matrix from column-major data
            let matrix = Mat::<f64>::from_fn(n, n, |row, col| mat_data[col * n + row]);

            // Extract RHS
            let rhs_start = i * n;
            let rhs_data = &rhs[rhs_start..rhs_start + n];
            let b = Col::<f64>::from_fn(n, |row| rhs_data[row]);

            // Compute LU factorization and solve
            let plu = matrix.partial_piv_lu();
            let x = plu.solve(&b);

            // Check for singularity by looking for NaN/Inf in solution
            let mut is_singular = false;
            for j in 0..n {
                if !x[j].is_finite() {
                    is_singular = true;
                    break;
                }
            }

            if is_singular {
                solutions.extend(std::iter::repeat(0.0).take(n));
                singular_indices.push(i);
            } else {
                // Extract solution
                for j in 0..n {
                    solutions.push(x[j]);
                }
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
        false // Faer is CPU-based, never uses GPU
    }

    fn backend_type(&self) -> BackendType {
        BackendType::Faer
    }

    fn config(&self) -> &GpuBatchConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_faer_solver_identity() {
        let solver = FaerBatchedSolver::new(GpuBatchConfig::default());

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
    fn test_faer_solver_simple_system() {
        let solver = FaerBatchedSolver::new(GpuBatchConfig::default());

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
    fn test_faer_solver_singular() {
        let solver = FaerBatchedSolver::new(GpuBatchConfig::default());

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
    fn test_faer_backend_type() {
        let solver = FaerBatchedSolver::new(GpuBatchConfig::default());
        assert_eq!(solver.backend_type(), BackendType::Faer);
    }
}
