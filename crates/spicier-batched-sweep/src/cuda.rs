//! CUDA backend implementation for batched LU solving.

use crate::error::{BatchedSweepError, Result};
use crate::solver::{BackendType, BatchedLuSolver, BatchedSolveResult, GpuBatchConfig};
use spicier_backend_cuda::{CudaBatchedLuSolver as CudaSolver, CudaContext};
use std::sync::Arc;

/// CUDA-accelerated batched LU solver.
///
/// Uses cuBLAS batched LU operations for efficient parallel solving
/// on NVIDIA GPUs.
pub struct CudaBatchedSolver {
    solver: CudaSolver,
    config: GpuBatchConfig,
}

impl CudaBatchedSolver {
    /// Create a new CUDA batched solver.
    ///
    /// # Errors
    /// Returns an error if CUDA initialization fails (no CUDA device available).
    pub fn new(config: GpuBatchConfig) -> Result<Self> {
        // cudarc panics if CUDA is not installed, so we need to catch that
        let ctx_result = std::panic::catch_unwind(CudaContext::new);

        let ctx = match ctx_result {
            Ok(Ok(ctx)) => ctx,
            Ok(Err(e)) => {
                return Err(BatchedSweepError::BackendInit(format!(
                    "CUDA context creation failed: {}",
                    e
                )));
            }
            Err(_) => {
                return Err(BatchedSweepError::BackendInit(
                    "CUDA not available (library not found)".to_string(),
                ));
            }
        };

        let cuda_config = spicier_backend_cuda::GpuBatchedSweepConfig {
            min_batch_size: config.min_batch_size,
            min_matrix_size: config.min_matrix_size,
            max_batch_per_launch: config.max_batch_per_launch,
        };

        let solver = CudaSolver::with_config(Arc::new(ctx), cuda_config);

        Ok(Self { solver, config })
    }

    /// Try to create a CUDA solver, returning None if unavailable.
    pub fn try_new(config: GpuBatchConfig) -> Option<Self> {
        Self::new(config).ok()
    }
}

impl BatchedLuSolver for CudaBatchedSolver {
    fn solve_batch(
        &self,
        matrices: &[f64],
        rhs: &[f64],
        n: usize,
        batch_size: usize,
    ) -> Result<BatchedSolveResult> {
        let cuda_result = self.solver.solve_batch(matrices, rhs, n, batch_size)?;

        Ok(BatchedSolveResult {
            solutions: cuda_result.solutions,
            singular_indices: cuda_result.singular_indices,
            n: cuda_result.n,
            batch_size: cuda_result.batch_size,
        })
    }

    fn should_use_gpu(&self, matrix_size: usize, batch_size: usize) -> bool {
        self.solver.should_use_gpu(matrix_size, batch_size)
    }

    fn backend_type(&self) -> BackendType {
        BackendType::Cuda
    }

    fn config(&self) -> &GpuBatchConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn try_create_solver() -> Option<CudaBatchedSolver> {
        CudaBatchedSolver::try_new(GpuBatchConfig::default())
    }

    #[test]
    fn test_cuda_solver_identity() {
        let solver = match try_create_solver() {
            Some(s) => s,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        let n = 2;
        let batch_size = 2;

        // Two 2×2 identity matrices in column-major order
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
    fn test_cuda_backend_type() {
        let solver = match try_create_solver() {
            Some(s) => s,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        assert_eq!(solver.backend_type(), BackendType::Cuda);
    }
}
