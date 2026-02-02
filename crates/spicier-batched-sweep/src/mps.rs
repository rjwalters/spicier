//! MPS backend implementation for batched LU solving.
//!
//! This module provides GPU-accelerated batched LU solving on Apple Silicon
//! using Metal Performance Shaders (MPS) for optimized linear algebra operations.

use crate::error::{BatchedSweepError, Result};
use crate::solver::{BackendType, BatchedLuSolver, BatchedSolveResult, GpuBatchConfig};
use spicier_backend_mps::{MpsBatchedLuSolver as MpsSolver, MpsContext};
use std::sync::Arc;

/// MPS-accelerated batched LU solver.
///
/// Uses Apple's Metal Performance Shaders for highly optimized parallel solving
/// on Apple GPUs (M1, M2, M3, etc.). MPS typically provides better performance
/// than custom compute shaders due to Apple's internal optimizations.
pub struct MpsBatchedSolver {
    solver: MpsSolver,
    config: GpuBatchConfig,
}

impl MpsBatchedSolver {
    /// Create a new MPS batched solver.
    ///
    /// # Errors
    /// Returns an error if MPS initialization fails (e.g., on non-macOS platforms).
    pub fn new(config: GpuBatchConfig) -> Result<Self> {
        let ctx = MpsContext::new().map_err(|e| {
            BatchedSweepError::BackendInit(format!("MPS context creation failed: {}", e))
        })?;

        let mps_config = spicier_backend_mps::MpsBatchConfig {
            min_batch_size: config.min_batch_size,
            min_matrix_size: config.min_matrix_size,
        };

        let solver = MpsSolver::with_config(Arc::new(ctx), mps_config).map_err(|e| {
            BatchedSweepError::BackendInit(format!("MPS solver creation failed: {}", e))
        })?;

        Ok(Self { solver, config })
    }

    /// Try to create an MPS solver, returning None if unavailable.
    pub fn try_new(config: GpuBatchConfig) -> Option<Self> {
        Self::new(config).ok()
    }
}

impl BatchedLuSolver for MpsBatchedSolver {
    fn solve_batch(
        &self,
        matrices: &[f64],
        rhs: &[f64],
        n: usize,
        batch_size: usize,
    ) -> Result<BatchedSolveResult> {
        let mps_result = self
            .solver
            .solve_batch(matrices, rhs, n, batch_size)
            .map_err(|e| BatchedSweepError::Backend(format!("MPS solve failed: {}", e)))?;

        Ok(BatchedSolveResult {
            solutions: mps_result.solutions,
            singular_indices: mps_result.singular_indices,
            n: mps_result.n,
            batch_size: mps_result.batch_size,
        })
    }

    fn should_use_gpu(&self, matrix_size: usize, batch_size: usize) -> bool {
        self.solver.should_use_gpu(matrix_size, batch_size)
    }

    fn backend_type(&self) -> BackendType {
        BackendType::Mps
    }

    fn config(&self) -> &GpuBatchConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn try_create_solver() -> Option<MpsBatchedSolver> {
        MpsBatchedSolver::try_new(GpuBatchConfig::default())
    }

    #[test]
    fn test_mps_solver_identity() {
        let solver = match try_create_solver() {
            Some(s) => s,
            None => {
                eprintln!("Skipping test: MPS not available");
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
        assert!((sol0[0] - 1.0).abs() < 1e-4);
        assert!((sol0[1] - 2.0).abs() < 1e-4);
    }

    #[test]
    fn test_mps_backend_type() {
        let solver = match try_create_solver() {
            Some(s) => s,
            None => {
                eprintln!("Skipping test: MPS not available");
                return;
            }
        };

        assert_eq!(solver.backend_type(), BackendType::Mps);
    }

    #[test]
    fn test_mps_should_use_gpu() {
        let solver = match try_create_solver() {
            Some(s) => s,
            None => {
                eprintln!("Skipping test: MPS not available");
                return;
            }
        };

        // Uses GpuBatchConfig defaults from batched-sweep: min_batch=16, min_matrix=32
        assert!(!solver.should_use_gpu(16, 100)); // Matrix too small (16 < 32)
        assert!(!solver.should_use_gpu(64, 8)); // Batch too small (8 < 16)
        assert!(solver.should_use_gpu(64, 32)); // Both OK (64 >= 32, 32 >= 16)
    }
}
