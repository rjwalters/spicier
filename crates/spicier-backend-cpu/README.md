# spicier-backend-cpu

CPU backend for Spicier with SIMD-accelerated dense operators.

## Features

- **Real dense operator** - SIMD-accelerated f64 matrix-vector multiplication
- **Complex dense operator** - SIMD-accelerated complex64 matvec
- **Implements operator traits** - Works with GMRES iterative solver

## Usage

```rust
use spicier_backend_cpu::{CpuRealDenseOperator, CpuComplexDenseOperator};
use spicier_solver::RealOperator;

// Create operator from dense matrix
let matrix = vec![1.0, 2.0, 3.0, 4.0]; // 2x2 row-major
let op = CpuRealDenseOperator::new(matrix, 2);

// Apply operator: y = A * x
let x = vec![1.0, 1.0];
let mut y = vec![0.0, 0.0];
op.apply(&x, &mut y);
```

## License

Licensed under either of Apache License, Version 2.0 or MIT license at your option.
