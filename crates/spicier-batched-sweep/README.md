# spicier-batched-sweep

Unified GPU-accelerated batched sweep solving for SPICE circuit simulation.

This crate provides a common API for GPU-accelerated batched LU solving across different backends:
- **CUDA** - NVIDIA GPUs via cuBLAS batched LU operations
- **Metal** - Apple Silicon GPUs (M1/M2/M3) via wgpu compute shaders
- **CPU** - Fallback using nalgebra's LU decomposition

## Features

- Unified `BatchedLuSolver` trait for backend abstraction
- Automatic backend detection and selection
- Graceful fallback when preferred GPU unavailable
- Efficient parallel solving for Monte Carlo, corner analysis, and parameter sweeps

## Usage

```rust
use spicier_batched_sweep::{solve_batched_sweep_gpu, BackendSelector};
use spicier_solver::{
    ConvergenceCriteria, DispatchConfig, MonteCarloGenerator, ParameterVariation,
};

// Automatic backend selection (CUDA → Metal → CPU)
let backend = BackendSelector::auto();

// Or prefer a specific backend
// let backend = BackendSelector::prefer_cuda();
// let backend = BackendSelector::prefer_metal();
// let backend = BackendSelector::cpu_only();

let result = solve_batched_sweep_gpu(
    &backend,
    &factory,
    &generator,
    &variations,
    &ConvergenceCriteria::default(),
    &DispatchConfig::default(),
)?;

println!("Backend used: {}", result.backend_used);
println!("Converged: {}/{}", result.converged_count, result.total_count);
```

## Benchmark Results

Benchmarked on Mac Studio M3 Ultra (96GB RAM):

| Batch × Matrix | CPU Time | Metal GPU Time | Speedup |
|----------------|----------|----------------|---------|
| 100 × 10       | 68 µs    | 8.5 ms         | 0.008x (CPU faster) |
| 100 × 50       | 1.2 ms   | 15.6 ms        | 0.08x (CPU faster) |
| 100 × 100      | 7.2 ms   | 37.6 ms        | 0.19x (CPU faster) |
| 500 × 10       | 340 µs   | 9.3 ms         | 0.04x (CPU faster) |
| 500 × 50       | 6.4 ms   | 19.4 ms        | 0.33x (CPU faster) |
| 500 × 100      | 41 ms    | 54.6 ms        | 0.75x (CPU faster) |
| 1000 × 10      | 680 µs   | 10.1 ms        | 0.07x (CPU faster) |
| 1000 × 50      | 12.5 ms  | 23.1 ms        | 0.54x (CPU faster) |
| 1000 × 100     | 73.8 ms  | 74.9 ms        | **0.99x** (nearly equal) |

### Analysis

The current Metal implementation uses a simple single-threaded-per-matrix approach where each workgroup processes one matrix sequentially. This results in:

1. **High overhead**: GPU kernel launch, memory transfer (~8-10ms baseline)
2. **No intra-matrix parallelism**: LU factorization is done sequentially within each matrix
3. **Global memory latency**: Every operation accesses global GPU memory

The GPU only approaches CPU performance for large problems (1000 matrices × 100×100).

### CUDA Performance (on NVIDIA GPUs)

The CUDA backend uses cuBLAS batched LU operations which are highly optimized:
- `cublasDgetrfBatched` - Batched LU factorization (parallelized internally)
- `cublasDgetrsBatched` - Batched triangular solve

Expected speedups of 10-100x on NVIDIA GPUs for large batches with medium-sized matrices.

### Future Optimizations

To make Metal competitive with CPU, the shader would need:
1. **Workgroup parallelism** - Multiple threads cooperating on each matrix
2. **Shared memory tiling** - Cache matrix blocks in fast workgroup memory
3. **Blocked algorithms** - Better memory access patterns

## Feature Flags

| Feature | Description |
|---------|-------------|
| `cuda`  | Enable NVIDIA CUDA backend |
| `metal` | Enable Apple Metal backend |

## Current Thresholds

The Metal backend will only use GPU when:
- Batch size ≥ 2000
- Matrix size ≥ 100×100

This ensures GPU is only used when it's likely to be beneficial.

## License

MIT OR Apache-2.0
