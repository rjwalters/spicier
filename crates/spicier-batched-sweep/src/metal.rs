//! Metal backend implementation for batched LU solving.
//!
//! This module provides GPU-accelerated batched LU solving on Apple Silicon
//! using Metal via WebGPU (wgpu).
//!
//! Note: Full Metal batched LU support is not yet implemented. This module
//! currently provides a CPU fallback but correctly detects Metal availability.

use crate::error::{BatchedSweepError, Result};
use crate::solver::{BackendType, BatchedLuSolver, BatchedSolveResult, GpuBatchConfig};

/// Metal-accelerated batched LU solver.
///
/// Uses WebGPU/Metal for efficient parallel solving on Apple GPUs (M1, M2, M3, etc.).
///
/// Note: Full batched LU using Metal compute shaders is not yet implemented.
/// This solver currently falls back to CPU but provides the infrastructure
/// for future Metal acceleration.
pub struct MetalBatchedSolver {
    config: GpuBatchConfig,
    #[allow(dead_code)]
    available: bool,
}

impl MetalBatchedSolver {
    /// Create a new Metal batched solver.
    ///
    /// # Errors
    /// Returns an error if Metal/WebGPU initialization fails.
    pub fn new(config: GpuBatchConfig) -> Result<Self> {
        // Check if Metal/WebGPU is available
        let available = spicier_backend_metal::context::WgpuContext::is_available();

        if !available {
            return Err(BatchedSweepError::BackendInit(
                "Metal/WebGPU not available on this system".to_string(),
            ));
        }

        Ok(Self { config, available })
    }

    /// Try to create a Metal solver, returning None if unavailable.
    pub fn try_new(config: GpuBatchConfig) -> Option<Self> {
        Self::new(config).ok()
    }
}

impl BatchedLuSolver for MetalBatchedSolver {
    fn solve_batch(
        &self,
        matrices: &[f64],
        rhs: &[f64],
        n: usize,
        batch_size: usize,
    ) -> Result<BatchedSolveResult> {
        // TODO: Implement Metal batched LU using MPS
        // For now, fall back to CPU implementation
        log::warn!("Metal batched LU not yet implemented, falling back to CPU");

        use nalgebra::{DMatrix, DVector};

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
            let mat_start = i * n * n;
            let mat_data = &matrices[mat_start..mat_start + n * n];
            let matrix = DMatrix::from_column_slice(n, n, mat_data);

            let rhs_start = i * n;
            let rhs_data = &rhs[rhs_start..rhs_start + n];
            let b = DVector::from_column_slice(rhs_data);

            let lu = nalgebra::linalg::LU::new(matrix);
            match lu.solve(&b) {
                Some(solution) => {
                    solutions.extend(solution.iter());
                }
                None => {
                    solutions.extend(std::iter::repeat(0.0).take(n));
                    singular_indices.push(i);
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

    fn should_use_gpu(&self, matrix_size: usize, batch_size: usize) -> bool {
        // TODO: Enable when Metal batched LU is implemented
        // For now, return false to trigger CPU fallback
        let _ = (matrix_size, batch_size);
        false
    }

    fn backend_type(&self) -> BackendType {
        BackendType::Metal
    }

    fn config(&self) -> &GpuBatchConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn try_create_solver() -> Option<MetalBatchedSolver> {
        MetalBatchedSolver::try_new(GpuBatchConfig::default())
    }

    #[test]
    fn test_metal_solver_identity() {
        let solver = match try_create_solver() {
            Some(s) => s,
            None => {
                eprintln!("Skipping test: Metal not available");
                return;
            }
        };

        let n = 2;
        let batch_size = 2;

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
    }

    #[test]
    fn test_metal_backend_type() {
        let solver = match try_create_solver() {
            Some(s) => s,
            None => {
                eprintln!("Skipping test: Metal not available");
                return;
            }
        };

        assert_eq!(solver.backend_type(), BackendType::Metal);
    }
}
