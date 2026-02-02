//! Faer-based sparse batched LU solver with cached symbolic factorization.
//!
//! This module provides a CPU-based sparse batched solver that caches the
//! symbolic factorization (elimination tree, fill-in pattern) and only
//! recomputes the numeric factorization for each sweep point.
//!
//! Since all sweep points share the same circuit topology (sparsity pattern),
//! the symbolic factorization can be computed once and reused, providing
//! significant speedup for large sweeps.

use crate::error::{BatchedSweepError, Result};
use crate::solver::{BackendType, BatchedLuSolver, BatchedSolveResult, GpuBatchConfig};
use faer::prelude::*;
use faer::sparse::linalg::solvers::{Lu, SymbolicLu};
use faer::sparse::{SparseColMat, Triplet};
use std::sync::{Arc, RwLock};

/// Faer-backed sparse batched LU solver with symbolic caching.
///
/// This solver is optimized for parameter sweeps where the circuit topology
/// (sparsity pattern) is fixed and only component values change.
///
/// The workflow is:
/// 1. First sweep point: compute sparsity pattern + symbolic factorization
/// 2. Subsequent points: reuse symbolic factorization, only do numeric factorization
///
/// Uses interior mutability (`RwLock`) to cache the symbolic factorization
/// while implementing the `BatchedLuSolver` trait (which uses `&self`).
pub struct FaerSparseCachedBatchedSolver {
    config: GpuBatchConfig,
    /// Cached symbolic factorization (computed on first batch)
    cached: RwLock<CachedSymbolic>,
}

/// Internal cache for symbolic factorization.
struct CachedSymbolic {
    symbolic: Option<Arc<SymbolicLu<usize>>>,
    expected_size: Option<usize>,
}

impl FaerSparseCachedBatchedSolver {
    /// Create a new sparse cached solver.
    pub fn new(config: GpuBatchConfig) -> Self {
        Self {
            config,
            cached: RwLock::new(CachedSymbolic {
                symbolic: None,
                expected_size: None,
            }),
        }
    }

    /// Reset the cached symbolic factorization.
    ///
    /// Call this when the circuit topology changes.
    pub fn reset_cache(&self) {
        let mut cache = self.cached.write().unwrap();
        cache.symbolic = None;
        cache.expected_size = None;
    }

    /// Check if symbolic factorization is cached.
    pub fn has_cached_symbolic(&self) -> bool {
        let cache = self.cached.read().unwrap();
        cache.symbolic.is_some()
    }
}

impl BatchedLuSolver for FaerSparseCachedBatchedSolver {
    fn solve_batch(
        &self,
        matrices: &[f64],
        rhs: &[f64],
        n: usize,
        batch_size: usize,
    ) -> Result<BatchedSolveResult> {
        let expected_matrix_len = batch_size * n * n;
        let expected_rhs_len = batch_size * n;

        if matrices.len() != expected_matrix_len {
            return Err(BatchedSweepError::InvalidDimension(format!(
                "Expected {} matrix elements, got {}",
                expected_matrix_len,
                matrices.len()
            )));
        }

        if rhs.len() != expected_rhs_len {
            return Err(BatchedSweepError::InvalidDimension(format!(
                "Expected {} RHS elements, got {}",
                expected_rhs_len,
                rhs.len()
            )));
        }

        let mut solutions = Vec::with_capacity(expected_rhs_len);
        let mut singular_indices = Vec::new();

        // Try to get cached symbolic factorization, or build one
        let symbolic = {
            // First, try to read from cache
            let cache_read = self.cached.read().unwrap();
            if let Some(ref sym) = cache_read.symbolic {
                if cache_read.expected_size != Some(n) {
                    return Err(BatchedSweepError::InvalidDimension(format!(
                        "Matrix size {} doesn't match cached size {:?}",
                        n, cache_read.expected_size
                    )));
                }
                sym.clone()
            } else {
                // Need to build symbolic - drop read lock and acquire write lock
                drop(cache_read);

                // Build symbolic from first matrix
                let mat_data = &matrices[0..n * n];
                let triplets = dense_to_sparse_triplets(mat_data, n);
                let sparse_mat = SparseColMat::<usize, f64>::try_new_from_triplets(n, n, &triplets)
                    .map_err(|e| {
                        BatchedSweepError::Backend(format!(
                            "Failed to build sparse matrix: {:?}",
                            e
                        ))
                    })?;

                let new_symbolic = SymbolicLu::try_new(sparse_mat.symbolic()).map_err(|e| {
                    BatchedSweepError::Backend(format!("Symbolic factorization failed: {:?}", e))
                })?;

                let symbolic = Arc::new(new_symbolic);

                // Cache the symbolic factorization for future batches
                let mut cache_write = self.cached.write().unwrap();
                cache_write.symbolic = Some(symbolic.clone());
                cache_write.expected_size = Some(n);

                symbolic
            }
        };

        // Solve each system using the cached symbolic factorization
        for i in 0..batch_size {
            let mat_start = i * n * n;
            let mat_data = &matrices[mat_start..mat_start + n * n];

            let rhs_start = i * n;
            let rhs_data = &rhs[rhs_start..rhs_start + n];

            match solve_with_symbolic(&symbolic, mat_data, rhs_data, n) {
                Ok(sol) => {
                    solutions.extend(sol);
                }
                Err(_) => {
                    solutions.extend(std::iter::repeat(0.0).take(n));
                    singular_indices.push(i);
                }
            }
        }

        Ok(BatchedSolveResult {
            solutions,
            singular_indices,
            n,
            batch_size,
        })
    }

    fn should_use_gpu(&self, _matrix_size: usize, _batch_size: usize) -> bool {
        false // This is a CPU-based solver
    }

    fn backend_type(&self) -> BackendType {
        BackendType::Faer
    }

    fn config(&self) -> &GpuBatchConfig {
        &self.config
    }
}

/// Convert dense matrix (column-major) to sparse triplets.
fn dense_to_sparse_triplets(data: &[f64], n: usize) -> Vec<Triplet<usize, usize, f64>> {
    let mut triplets = Vec::new();
    for col in 0..n {
        for row in 0..n {
            let val = data[col * n + row];
            if val.abs() > 1e-15 {
                triplets.push(Triplet::new(row, col, val));
            }
        }
    }
    triplets
}

/// Solve a single system using cached symbolic factorization.
fn solve_with_symbolic(
    symbolic: &SymbolicLu<usize>,
    mat_data: &[f64],
    rhs_data: &[f64],
    n: usize,
) -> std::result::Result<Vec<f64>, ()> {
    // Convert dense to sparse
    let triplets = dense_to_sparse_triplets(mat_data, n);
    let sparse_mat =
        SparseColMat::<usize, f64>::try_new_from_triplets(n, n, &triplets).map_err(|_| ())?;

    // Numeric factorization using cached symbolic
    let lu = Lu::try_new_with_symbolic(symbolic.clone(), sparse_mat.as_ref()).map_err(|_| ())?;

    // Solve
    let b = Col::<f64>::from_fn(n, |i| rhs_data[i]);
    let x = lu.solve(&b);

    // Check for NaN/Inf
    let mut result = Vec::with_capacity(n);
    for i in 0..n {
        let val = x[i];
        if !val.is_finite() {
            return Err(());
        }
        result.push(val);
    }

    Ok(result)
}

/// Batched solver that accepts triplets directly (avoiding dense→sparse conversion).
///
/// This is more efficient when the stamper can generate triplets directly.
pub struct FaerTripletBatchedSolver {
    config: GpuBatchConfig,
}

impl FaerTripletBatchedSolver {
    /// Create a new triplet-based batched solver.
    pub fn new(config: GpuBatchConfig) -> Self {
        Self { config }
    }

    /// Solve a batch of systems from triplet format with cached symbolic factorization.
    ///
    /// All systems must have the same sparsity pattern (same (row, col) pairs).
    ///
    /// # Arguments
    /// * `triplets_per_system` - Triplets for each system: Vec<Vec<(row, col, value)>>
    /// * `rhs_per_system` - RHS vectors for each system
    /// * `n` - System size
    pub fn solve_batch_triplets(
        &self,
        triplets_per_system: &[Vec<(usize, usize, f64)>],
        rhs_per_system: &[Vec<f64>],
        n: usize,
    ) -> Result<BatchedSolveResult> {
        let batch_size = triplets_per_system.len();

        if rhs_per_system.len() != batch_size {
            return Err(BatchedSweepError::InvalidDimension(format!(
                "Triplets batch size {} doesn't match RHS batch size {}",
                batch_size,
                rhs_per_system.len()
            )));
        }

        if batch_size == 0 {
            return Ok(BatchedSolveResult {
                solutions: vec![],
                singular_indices: vec![],
                n,
                batch_size: 0,
            });
        }

        // Build symbolic factorization from first system
        let first_triplets: Vec<Triplet<usize, usize, f64>> = triplets_per_system[0]
            .iter()
            .map(|&(r, c, v)| Triplet::new(r, c, v))
            .collect();

        let first_sparse = SparseColMat::<usize, f64>::try_new_from_triplets(n, n, &first_triplets)
            .map_err(|e| {
                BatchedSweepError::Backend(format!("Failed to build sparse matrix: {:?}", e))
            })?;

        let symbolic = SymbolicLu::try_new(first_sparse.symbolic()).map_err(|e| {
            BatchedSweepError::Backend(format!("Symbolic factorization failed: {:?}", e))
        })?;

        let mut solutions = Vec::with_capacity(batch_size * n);
        let mut singular_indices = Vec::new();

        for (i, (triplets, rhs)) in triplets_per_system
            .iter()
            .zip(rhs_per_system.iter())
            .enumerate()
        {
            if rhs.len() != n {
                return Err(BatchedSweepError::InvalidDimension(format!(
                    "RHS {} has size {}, expected {}",
                    i,
                    rhs.len(),
                    n
                )));
            }

            let faer_triplets: Vec<Triplet<usize, usize, f64>> = triplets
                .iter()
                .map(|&(r, c, v)| Triplet::new(r, c, v))
                .collect();

            let sparse_mat =
                match SparseColMat::<usize, f64>::try_new_from_triplets(n, n, &faer_triplets) {
                    Ok(m) => m,
                    Err(_) => {
                        solutions.extend(std::iter::repeat(0.0).take(n));
                        singular_indices.push(i);
                        continue;
                    }
                };

            let lu = match Lu::try_new_with_symbolic(symbolic.clone(), sparse_mat.as_ref()) {
                Ok(lu) => lu,
                Err(_) => {
                    solutions.extend(std::iter::repeat(0.0).take(n));
                    singular_indices.push(i);
                    continue;
                }
            };

            let b = Col::<f64>::from_fn(n, |j| rhs[j]);
            let x = lu.solve(&b);

            let mut is_singular = false;
            for j in 0..n {
                if !x[j].is_finite() {
                    is_singular = true;
                    break;
                }
            }

            if is_singular {
                solutions.extend(std::iter::repeat(0.0).take(n));
                singular_indices.push(i);
            } else {
                for j in 0..n {
                    solutions.push(x[j]);
                }
            }
        }

        Ok(BatchedSolveResult {
            solutions,
            singular_indices,
            n,
            batch_size,
        })
    }

    /// Get the configuration.
    pub fn config(&self) -> &GpuBatchConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparse_cached_solver_identity() {
        let solver = FaerSparseCachedBatchedSolver::new(GpuBatchConfig::default());

        let n = 2;
        let batch_size = 2;

        // Two 2x2 identity matrices in column-major order
        let matrices = vec![
            1.0, 0.0, 0.0, 1.0, // Identity 0
            1.0, 0.0, 0.0, 1.0, // Identity 1
        ];

        let rhs = vec![
            1.0, 2.0, // b0 = [1, 2]
            3.0, 4.0, // b1 = [3, 4]
        ];

        let result = solver.solve_batch(&matrices, &rhs, n, batch_size).unwrap();

        assert_eq!(result.batch_size, 2);
        assert!(result.singular_indices.is_empty());

        let sol0 = result.solution(0).unwrap();
        assert!((sol0[0] - 1.0).abs() < 1e-10);
        assert!((sol0[1] - 2.0).abs() < 1e-10);

        let sol1 = result.solution(1).unwrap();
        assert!((sol1[0] - 3.0).abs() < 1e-10);
        assert!((sol1[1] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_sparse_cached_solver_simple() {
        let solver = FaerSparseCachedBatchedSolver::new(GpuBatchConfig::default());

        let n = 2;
        let batch_size = 1;

        // Matrix [[2, 1], [1, 3]] in column-major: [2, 1, 1, 3]
        let matrices = vec![2.0, 1.0, 1.0, 3.0];
        let rhs = vec![5.0, 10.0]; // b = [5, 10]

        // Solution: x = [1, 3]
        let result = solver.solve_batch(&matrices, &rhs, n, batch_size).unwrap();

        assert!(result.singular_indices.is_empty());
        let sol = result.solution(0).unwrap();
        assert!((sol[0] - 1.0).abs() < 1e-10);
        assert!((sol[1] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_triplet_solver_batch() {
        let solver = FaerTripletBatchedSolver::new(GpuBatchConfig::default());

        let n = 2;

        // Two systems with same sparsity but different values
        // System 0: [[2, 1], [1, 3]] * x = [5, 10] -> x = [1, 3]
        // System 1: [[4, 2], [2, 6]] * x = [10, 20] -> x = [1, 3] (scaled)
        let triplets_per_system = vec![
            vec![(0, 0, 2.0), (0, 1, 1.0), (1, 0, 1.0), (1, 1, 3.0)],
            vec![(0, 0, 4.0), (0, 1, 2.0), (1, 0, 2.0), (1, 1, 6.0)],
        ];

        let rhs_per_system = vec![vec![5.0, 10.0], vec![10.0, 20.0]];

        let result = solver
            .solve_batch_triplets(&triplets_per_system, &rhs_per_system, n)
            .unwrap();

        assert_eq!(result.batch_size, 2);
        assert!(result.singular_indices.is_empty());

        let sol0 = result.solution(0).unwrap();
        assert!((sol0[0] - 1.0).abs() < 1e-10);
        assert!((sol0[1] - 3.0).abs() < 1e-10);

        let sol1 = result.solution(1).unwrap();
        assert!((sol1[0] - 1.0).abs() < 1e-10);
        assert!((sol1[1] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_sparse_vs_dense_consistency() {
        use crate::faer_solver::FaerBatchedSolver;

        let sparse_solver = FaerSparseCachedBatchedSolver::new(GpuBatchConfig::default());
        let dense_solver = FaerBatchedSolver::new(GpuBatchConfig::default());

        let n = 5;
        let batch_size = 10;

        // Random-ish diagonally dominant matrices
        let mut matrices = Vec::with_capacity(batch_size * n * n);
        let mut rhs = Vec::with_capacity(batch_size * n);

        for batch_idx in 0..batch_size {
            for col in 0..n {
                for row in 0..n {
                    let val = if row == col {
                        10.0 + (batch_idx as f64) * 0.1
                    } else {
                        1.0 / ((row as f64 - col as f64).abs() + 1.0)
                    };
                    matrices.push(val);
                }
            }
            for i in 0..n {
                rhs.push((i + 1) as f64 + (batch_idx as f64) * 0.5);
            }
        }

        let sparse_result = sparse_solver
            .solve_batch(&matrices, &rhs, n, batch_size)
            .unwrap();
        let dense_result = dense_solver
            .solve_batch(&matrices, &rhs, n, batch_size)
            .unwrap();

        assert_eq!(
            sparse_result.singular_indices,
            dense_result.singular_indices
        );

        for i in 0..batch_size {
            let sparse_sol = sparse_result.solution(i).unwrap();
            let dense_sol = dense_result.solution(i).unwrap();
            for j in 0..n {
                assert!(
                    (sparse_sol[j] - dense_sol[j]).abs() < 1e-10,
                    "Batch {}, element {}: sparse={}, dense={}",
                    i,
                    j,
                    sparse_sol[j],
                    dense_sol[j]
                );
            }
        }
    }

    #[test]
    fn test_symbolic_caching_across_batches() {
        // Test that symbolic factorization is cached and reused across multiple solve_batch calls
        let solver = FaerSparseCachedBatchedSolver::new(GpuBatchConfig::default());

        assert!(!solver.has_cached_symbolic());

        let n = 3;

        // First batch: 3x3 diagonally dominant matrix
        let matrices1 = vec![
            5.0, 1.0, 0.0, // col 0
            1.0, 5.0, 1.0, // col 1
            0.0, 1.0, 5.0, // col 2
        ];
        let rhs1 = vec![6.0, 7.0, 6.0];

        let result1 = solver.solve_batch(&matrices1, &rhs1, n, 1).unwrap();
        assert!(result1.singular_indices.is_empty());

        // After first batch, symbolic should be cached
        assert!(solver.has_cached_symbolic());

        // Second batch: same sparsity pattern, different values
        let matrices2 = vec![
            10.0, 2.0, 0.0, // col 0
            2.0, 10.0, 2.0, // col 1
            0.0, 2.0, 10.0, // col 2
        ];
        let rhs2 = vec![12.0, 14.0, 12.0];

        let result2 = solver.solve_batch(&matrices2, &rhs2, n, 1).unwrap();
        assert!(result2.singular_indices.is_empty());

        // Symbolic should still be cached
        assert!(solver.has_cached_symbolic());

        // Verify solutions are correct (x = [1, 1, 1] for both scaled systems)
        let sol1 = result1.solution(0).unwrap();
        let sol2 = result2.solution(0).unwrap();
        for i in 0..n {
            assert!((sol1[i] - 1.0).abs() < 1e-10, "sol1[{}] = {}", i, sol1[i]);
            assert!((sol2[i] - 1.0).abs() < 1e-10, "sol2[{}] = {}", i, sol2[i]);
        }

        // Reset cache and verify
        solver.reset_cache();
        assert!(!solver.has_cached_symbolic());
    }
}
