//! Error types for batched sweep operations.

use thiserror::Error;

/// Errors that can occur during batched sweep operations.
#[derive(Debug, Error)]
pub enum BatchedSweepError {
    /// No GPU backend is available.
    #[error("No GPU backend available")]
    NoBackendAvailable,

    /// Backend initialization failed.
    #[error("Backend initialization failed: {0}")]
    BackendInit(String),

    /// Memory allocation failed.
    #[error("Memory allocation failed: {0}")]
    MemoryAlloc(String),

    /// Data transfer failed.
    #[error("Data transfer failed: {0}")]
    Transfer(String),

    /// Invalid dimensions provided.
    #[error("Invalid dimensions: {0}")]
    InvalidDimension(String),

    /// Batch size exceeds backend limit.
    #[error("Batch size {size} exceeds maximum {max}")]
    BatchTooLarge { size: usize, max: usize },

    /// Some matrices in the batch were singular.
    #[error("{} of {} matrices were singular", indices.len(), total)]
    SingularBatch { indices: Vec<usize>, total: usize },

    /// Backend-specific error.
    #[error("Backend error: {0}")]
    Backend(String),

    /// Solver error from spicier-solver.
    #[error("Solver error: {0}")]
    Solver(#[from] spicier_solver::Error),
}

/// Result type for batched sweep operations.
pub type Result<T> = std::result::Result<T, BatchedSweepError>;

#[cfg(feature = "cuda")]
impl From<spicier_backend_cuda::CudaError> for BatchedSweepError {
    fn from(e: spicier_backend_cuda::CudaError) -> Self {
        match e {
            spicier_backend_cuda::CudaError::BatchTooLarge { size, max } => {
                BatchedSweepError::BatchTooLarge { size, max }
            }
            spicier_backend_cuda::CudaError::SingularBatch { indices } => {
                BatchedSweepError::SingularBatch {
                    indices: indices.clone(),
                    total: 0, // Will be set by caller
                }
            }
            spicier_backend_cuda::CudaError::MemoryAlloc(msg) => {
                BatchedSweepError::MemoryAlloc(msg)
            }
            spicier_backend_cuda::CudaError::Transfer(msg) => BatchedSweepError::Transfer(msg),
            spicier_backend_cuda::CudaError::InvalidDimension(msg) => {
                BatchedSweepError::InvalidDimension(msg)
            }
            other => BatchedSweepError::Backend(other.to_string()),
        }
    }
}
