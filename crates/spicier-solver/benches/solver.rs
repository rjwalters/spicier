//! Benchmarks for linear solvers.

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use nalgebra::{DMatrix, DVector};
use spicier_solver::linear::{solve_dense, solve_sparse};

fn bench_solve_dense(c: &mut Criterion) {
    let mut group = c.benchmark_group("solve_dense");

    for size in [10, 50, 100, 500] {
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &size,
            |bencher, &size| {
                // Create a diagonally dominant matrix (guaranteed non-singular)
                let a = DMatrix::from_fn(size, size, |i, j| {
                    if i == j {
                        (size as f64) + 1.0
                    } else {
                        1.0 / ((i as f64 - j as f64).abs() + 1.0)
                    }
                });
                let rhs = DVector::from_fn(size, |i, _| (i + 1) as f64);

                bencher.iter(|| solve_dense(black_box(&a), black_box(&rhs)).unwrap());
            },
        );
    }

    group.finish();
}

fn bench_solve_sparse(c: &mut Criterion) {
    let mut group = c.benchmark_group("solve_sparse");

    for size in [10, 50, 100, 500] {
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &size,
            |bencher, &size| {
                // Build sparse MNA-like matrix: each node connects to ~4 neighbors
                let mut triplets = Vec::new();
                for i in 0..size {
                    // Diagonal: large value for diagonal dominance
                    triplets.push((i, i, (size as f64) + 1.0));
                    // Off-diagonal: sparse connections (band structure)
                    for &offset in &[1_usize, 2] {
                        if i + offset < size {
                            let v = 1.0 / (offset as f64 + 1.0);
                            triplets.push((i, i + offset, v));
                            triplets.push((i + offset, i, v));
                        }
                    }
                }
                let rhs = DVector::from_fn(size, |i, _| (i + 1) as f64);

                bencher.iter(|| {
                    solve_sparse(black_box(size), black_box(&triplets), black_box(&rhs)).unwrap()
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_solve_dense, bench_solve_sparse);
criterion_main!(benches);
