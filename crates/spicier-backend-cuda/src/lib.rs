//! CUDA backend for Spicier GPU-accelerated operators.

pub mod batched_lu;
pub mod batched_sweep;
pub mod context;
pub mod dense_operator;
pub mod error;

pub use batched_lu::{
    BatchedMatrices, BatchedPivots, BatchedSolveResult, BatchedVectors, CudaBatchedLuSolver,
    GpuBatchedSweepConfig, MAX_BATCH_SIZE, MIN_BATCH_SIZE, MIN_MATRIX_SIZE,
};
pub use batched_sweep::{solve_batched_sweep_gpu, GpuBatchedSweepResult};
pub use context::CudaContext;
pub use error::{CudaError, Result};
