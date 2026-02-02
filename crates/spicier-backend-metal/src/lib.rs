//! Metal/WebGPU backend for Spicier GPU-accelerated operators.

mod batch_layout;
pub mod batched_lu;
pub mod context;
pub mod dense_operator;
pub mod error;

pub use batched_lu::{
    BatchedSolveResult, GpuBatchConfig, MAX_MATRIX_SIZE, MIN_BATCH_SIZE, MIN_MATRIX_SIZE,
    MetalBatchedLuSolver,
};
pub use context::WgpuContext;
pub use error::{Result, WgpuError};
