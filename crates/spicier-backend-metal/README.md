# spicier-backend-metal

Metal/WebGPU backend for Spicier GPU-accelerated operators.

## Features

- **WebGPU/wgpu** - Cross-platform GPU compute via wgpu
- **Metal support** - Native acceleration on macOS
- **WGSL shaders** - Custom compute shaders for matvec
- **Automatic fallback** - Falls back to CPU for small matrices or missing GPU

## Requirements

- macOS with Metal support, or
- Windows/Linux with Vulkan support

## Usage

```rust
use spicier_backend_metal::{WgpuContext, WgpuRealDenseOperator};
use spicier_solver::RealOperator;

// Check GPU availability
if WgpuContext::is_available() {
    let ctx = WgpuContext::new()?;

    // Create GPU operator
    let matrix = vec![1.0, 2.0, 3.0, 4.0];
    let op = WgpuRealDenseOperator::from_matrix(ctx, matrix, 2)?;

    // Apply on GPU
    let x = vec![1.0, 1.0];
    let mut y = vec![0.0, 0.0];
    op.apply(&x, &mut y);
}
```

## License

Licensed under either of Apache License, Version 2.0 or MIT license at your option.
