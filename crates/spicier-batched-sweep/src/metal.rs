//! Metal backend implementation for batched LU solving.
//!
//! This module provides GPU-accelerated batched LU solving on Apple Silicon
//! using Metal via WebGPU (wgpu) compute shaders.

use crate::error::{BatchedSweepError, Result};
use crate::solver::{BackendType, BatchedLuSolver, BatchedSolveResult, GpuBatchConfig};
use spicier_backend_metal::{MetalBatchedLuSolver as MetalSolver, WgpuContext};
use std::sync::Arc;

/// Metal-accelerated batched LU solver.
///
/// Uses wgpu/Metal compute shaders for efficient parallel solving
/// on Apple GPUs (M1, M2, M3, etc.).
pub struct MetalBatchedSolver {
    solver: MetalSolver,
    config: GpuBatchConfig,
}

impl MetalBatchedSolver {
    /// Create a new Metal batched solver.
    ///
    /// # Errors
    /// Returns an error if Metal/WebGPU initialization fails.
    pub fn new(config: GpuBatchConfig) -> Result<Self> {
        let ctx = WgpuContext::new().map_err(|e| {
            BatchedSweepError::BackendInit(format!("Metal/WebGPU context creation failed: {}", e))
        })?;

        let metal_config = spicier_backend_metal::GpuBatchConfig {
            min_batch_size: config.min_batch_size,
            min_matrix_size: config.min_matrix_size,
            max_matrix_size: spicier_backend_metal::MAX_MATRIX_SIZE,
        };

        let solver = MetalSolver::with_config(Arc::new(ctx), metal_config).map_err(|e| {
            BatchedSweepError::BackendInit(format!("Metal solver creation failed: {}", e))
        })?;

        Ok(Self { solver, config })
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
        let metal_result = self.solver.solve_batch(matrices, rhs, n, batch_size).map_err(|e| {
            BatchedSweepError::Backend(format!("Metal solve failed: {}", e))
        })?;

        Ok(BatchedSolveResult {
            solutions: metal_result.solutions,
            singular_indices: metal_result.singular_indices,
            n: metal_result.n,
            batch_size: metal_result.batch_size,
        })
    }

    fn should_use_gpu(&self, matrix_size: usize, batch_size: usize) -> bool {
        self.solver.should_use_gpu(matrix_size, batch_size)
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
        assert!((sol0[0] - 1.0).abs() < 1e-4);
        assert!((sol0[1] - 2.0).abs() < 1e-4);
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

    #[test]
    fn test_metal_should_use_gpu() {
        let solver = match try_create_solver() {
            Some(s) => s,
            None => {
                eprintln!("Skipping test: Metal not available");
                return;
            }
        };

        // Uses GpuBatchConfig defaults from batched-sweep: min_batch=16, min_matrix=32
        assert!(!solver.should_use_gpu(16, 100));  // Matrix too small (16 < 32)
        assert!(!solver.should_use_gpu(64, 8));    // Batch too small (8 < 16)
        assert!(solver.should_use_gpu(64, 32));    // Both OK (64 >= 32, 32 >= 16)
    }
}
