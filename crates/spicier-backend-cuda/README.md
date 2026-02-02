# spicier-backend-cuda

CUDA backend for Spicier GPU-accelerated operators.

## Features

- **cuBLAS integration** - GPU-accelerated BLAS operations via cudarc
- **Dense operators** - Real and complex matrix-vector multiplication
- **Dynamic loading** - No compile-time CUDA dependency
- **Automatic fallback** - Falls back to CPU for small matrices

## Requirements

- NVIDIA GPU with CUDA support
- CUDA toolkit installed (runtime only)

## Usage

```rust
use spicier_backend_cuda::{CudaContext, CudaRealDenseOperator};
use spicier_solver::RealOperator;

// Check CUDA availability
if CudaContext::is_available() {
    let ctx = CudaContext::new()?;

    // Create GPU operator
    let matrix = vec![1.0, 2.0, 3.0, 4.0];
    let op = CudaRealDenseOperator::new(ctx, matrix, 2)?;

    // Apply on GPU
    let x = vec![1.0, 1.0];
    let mut y = vec![0.0, 0.0];
    op.apply(&x, &mut y);
}
```

## License

Licensed under either of Apache License, Version 2.0 or MIT license at your option.
