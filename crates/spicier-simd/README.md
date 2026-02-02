# spicier-simd

SIMD-accelerated numerical kernels for the Spicier circuit simulator.

## Features

- **Runtime detection** - Automatic AVX-512, AVX2, or scalar selection
- **Real dot product** - SIMD-accelerated f64 dot product
- **Complex dot product** - SIMD-accelerated complex64 operations
- **Matrix-vector** - Dense matvec with SIMD acceleration

## Usage

```rust
use spicier_simd::{SimdCapability, real_dot_product, complex_dot_product};

// Detect CPU capabilities
let capability = SimdCapability::detect();
println!("SIMD: {:?}", capability);

// Compute dot product
let a = vec![1.0, 2.0, 3.0, 4.0];
let b = vec![5.0, 6.0, 7.0, 8.0];
let result = real_dot_product(&a, &b);
```

## License

Licensed under either of Apache License, Version 2.0 or MIT license at your option.
