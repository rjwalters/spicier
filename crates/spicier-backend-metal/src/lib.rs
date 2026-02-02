//! Metal/WebGPU backend for Spicier GPU-accelerated operators.
//!
//! This crate provides GPU acceleration for circuit simulation:
//! - Batched LU solving for parameter sweeps
//! - Device evaluation kernels (MOSFET, diode, BJT) for massively parallel sweeps
//! - Matrix assembly kernels for parallel stamping across sweep points
//! - Batched sparse matrix-vector multiplication for iterative solvers

mod batch_layout;
pub mod batched_lu;
pub mod batched_spmv;
pub mod context;
pub mod dense_operator;
pub mod device_eval;
pub mod error;
pub mod matrix_assembly;

pub use batched_lu::{
    BatchedSolveResult, GpuBatchConfig, MAX_MATRIX_SIZE, MIN_BATCH_SIZE, MIN_MATRIX_SIZE,
    MetalBatchedLuSolver,
};
pub use context::WgpuContext;
pub use device_eval::{
    BjtEvalResult, DiodeEvalResult, GpuBjtEvaluator, GpuBjtParams, GpuDiodeEvaluator,
    GpuDiodeParams, GpuMosfetEvaluator, GpuMosfetParams, MosfetEvalResult,
};
pub use error::{Result, WgpuError};
pub use batched_spmv::{BatchedCsrMatrix, GpuBatchedSpmv};
pub use matrix_assembly::{ConductanceStamp, CurrentStamp, GpuMatrixAssembler, GpuRhsAssembler};
