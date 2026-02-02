//! Unified GPU-accelerated batched sweep solving.

use crate::error::Result;
use crate::solver::{BackendSelector, BackendType, BatchedLuSolver};
use nalgebra::DVector;
#[cfg(test)]
use spicier_solver::SweepStamper;
use spicier_solver::{
    ConvergenceCriteria, DispatchConfig, ParameterVariation, SweepPoint, SweepPointGenerator,
    SweepStamperFactory, SweepStatistics,
};
#[cfg(test)]
use std::sync::Arc;

/// Result of a GPU-accelerated batched sweep analysis.
#[derive(Debug, Clone)]
pub struct GpuBatchedSweepResult {
    /// Solutions for each sweep point.
    pub solutions: Vec<DVector<f64>>,
    /// Parameter values for each point.
    pub points: Vec<SweepPoint>,
    /// Number of converged points.
    pub converged_count: usize,
    /// Total number of points.
    pub total_count: usize,
    /// Indices of singular systems.
    pub singular_indices: Vec<usize>,
    /// Whether GPU was actually used (vs CPU fallback).
    pub used_gpu: bool,
    /// Which backend was used.
    pub backend_used: BackendType,
}

impl GpuBatchedSweepResult {
    /// Get the solution at a specific point.
    pub fn solution(&self, index: usize) -> Option<&DVector<f64>> {
        self.solutions.get(index)
    }

    /// Get all node voltages at a specific node across all points.
    pub fn node_voltages(&self, node_index: usize) -> Vec<f64> {
        self.solutions.iter().map(|s| s[node_index]).collect()
    }

    /// Calculate statistics for a node voltage across all points.
    pub fn statistics(&self, node_index: usize) -> SweepStatistics {
        let voltages = self.node_voltages(node_index);
        SweepStatistics::from_samples(&voltages)
    }

    /// Check if a specific system was singular.
    pub fn is_singular(&self, index: usize) -> bool {
        self.singular_indices.contains(&index)
    }
}

/// Execute a GPU-accelerated batched sweep analysis.
///
/// Uses the appropriate GPU backend (CUDA or Metal) for efficient parallel solving
/// of Monte Carlo, corner analysis, and parameter sweeps.
///
/// Falls back to CPU solving if:
/// - No GPU backend is available
/// - Matrix size is below threshold (< 32)
/// - Batch size is below threshold (< 16)
///
/// # Arguments
/// * `backend` - Backend selector for choosing CUDA, Metal, or CPU
/// * `factory` - Factory for creating stampers with varied parameters
/// * `generator` - Generator for sweep points
/// * `variations` - Parameter variations to sweep
/// * `criteria` - Convergence criteria (for nonlinear circuits)
/// * `config` - Dispatch configuration with GPU thresholds
///
/// # Example
///
/// ```ignore
/// use spicier_batched_sweep::{solve_batched_sweep_gpu, BackendSelector};
/// use spicier_solver::{
///     DispatchConfig, ConvergenceCriteria, MonteCarloGenerator,
///     ParameterVariation,
/// };
///
/// let backend = BackendSelector::auto();
/// let config = DispatchConfig::default();
/// let generator = MonteCarloGenerator::new(100);
/// let variations = vec![ParameterVariation::new("R1", 1000.0)];
///
/// let result = solve_batched_sweep_gpu(
///     &backend,
///     &factory,
///     &generator,
///     &variations,
///     &ConvergenceCriteria::default(),
///     &config,
/// )?;
///
/// println!("Used GPU: {}", result.used_gpu);
/// println!("Backend: {}", result.backend_used);
/// ```
pub fn solve_batched_sweep_gpu(
    backend: &BackendSelector,
    factory: &dyn SweepStamperFactory,
    generator: &dyn SweepPointGenerator,
    variations: &[ParameterVariation],
    _criteria: &ConvergenceCriteria,
    config: &DispatchConfig,
) -> Result<GpuBatchedSweepResult> {
    let points = generator.generate(variations);
    let total_count = points.len();

    if total_count == 0 {
        return Ok(GpuBatchedSweepResult {
            solutions: vec![],
            points: vec![],
            converged_count: 0,
            total_count: 0,
            singular_indices: vec![],
            used_gpu: false,
            backend_used: BackendType::Cpu,
        });
    }

    // Get system size from first point
    let first_stamper = factory.create_stamper(
        points
            .first()
            .map(|p| p.parameters.as_slice())
            .unwrap_or(&[]),
    );
    let system_size = first_stamper.num_nodes() + first_stamper.num_vsources();

    // Check if GPU should be used
    let use_gpu = config.use_gpu_batch(system_size, total_count);

    // Create the solver
    let solver = backend.create_solver()?;

    if !use_gpu || solver.backend_type() == BackendType::Cpu {
        log::debug!(
            "Using CPU for batched sweep (size={}, batch={})",
            system_size,
            total_count
        );
        return solve_with_solver(&*solver, factory, &points, system_size);
    }

    log::info!(
        "Using {} for batched sweep (size={}, batch={})",
        solver.backend_type(),
        system_size,
        total_count
    );

    solve_with_solver(&*solver, factory, &points, system_size)
}

/// Solve sweep using a specific solver backend.
fn solve_with_solver(
    solver: &dyn BatchedLuSolver,
    factory: &dyn SweepStamperFactory,
    points: &[SweepPoint],
    system_size: usize,
) -> Result<GpuBatchedSweepResult> {
    let total_count = points.len();

    // Build all matrices and RHS vectors into contiguous buffers
    let mut matrices = Vec::with_capacity(total_count * system_size * system_size);
    let mut rhs_vectors = Vec::with_capacity(total_count * system_size);

    for point in points {
        let stamper = factory.create_stamper(&point.parameters);

        let mut matrix = nalgebra::DMatrix::zeros(system_size, system_size);
        let mut rhs = DVector::zeros(system_size);

        stamper.stamp_linear(&mut matrix, &mut rhs);

        // Convert to column-major (nalgebra stores column-major by default)
        for col in 0..system_size {
            for row in 0..system_size {
                matrices.push(matrix[(row, col)]);
            }
        }

        // Copy RHS vector
        for i in 0..system_size {
            rhs_vectors.push(rhs[i]);
        }
    }

    // Solve batch
    let batch_result = solver.solve_batch(&matrices, &rhs_vectors, system_size, total_count)?;

    // Convert solutions back to DVector format
    let mut solutions = Vec::with_capacity(total_count);
    for i in 0..total_count {
        if let Some(sol_slice) = batch_result.solution(i) {
            solutions.push(DVector::from_vec(sol_slice.to_vec()));
        } else {
            solutions.push(DVector::zeros(system_size));
        }
    }

    let converged_count = total_count - batch_result.singular_indices.len();
    let used_gpu = solver.backend_type() != BackendType::Cpu;

    Ok(GpuBatchedSweepResult {
        solutions,
        points: points.to_vec(),
        converged_count,
        total_count,
        singular_indices: batch_result.singular_indices,
        used_gpu,
        backend_used: solver.backend_type(),
    })
}

/// Convenience function to solve with automatic backend selection.
pub fn solve_batched_sweep_auto(
    factory: &dyn SweepStamperFactory,
    generator: &dyn SweepPointGenerator,
    variations: &[ParameterVariation],
    criteria: &ConvergenceCriteria,
    config: &DispatchConfig,
) -> Result<GpuBatchedSweepResult> {
    let backend = BackendSelector::auto();
    solve_batched_sweep_gpu(&backend, factory, generator, variations, criteria, config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use spicier_solver::{CornerGenerator, LinearSweepGenerator, MonteCarloGenerator};

    struct SimpleDividerFactory {
        r2_nominal: f64,
    }

    impl SweepStamperFactory for SimpleDividerFactory {
        fn create_stamper(&self, parameters: &[f64]) -> Arc<dyn SweepStamper> {
            let r1 = parameters.first().copied().unwrap_or(1000.0);
            Arc::new(SimpleDividerStamper {
                r1,
                r2: self.r2_nominal,
                v_source: 10.0,
            })
        }
    }

    struct SimpleDividerStamper {
        r1: f64,
        r2: f64,
        v_source: f64,
    }

    impl SweepStamper for SimpleDividerStamper {
        fn stamp_linear(&self, matrix: &mut nalgebra::DMatrix<f64>, rhs: &mut DVector<f64>) {
            let g1 = 1.0 / self.r1;
            let g2 = 1.0 / self.r2;

            // Stamp R1 between node 0 and 1
            matrix[(0, 0)] += g1;
            matrix[(1, 1)] += g1;
            matrix[(0, 1)] -= g1;
            matrix[(1, 0)] -= g1;

            // Stamp R2 between node 1 and ground
            matrix[(1, 1)] += g2;

            // Stamp voltage source at node 0 (branch current at index 2)
            matrix[(0, 2)] += 1.0;
            matrix[(2, 0)] += 1.0;
            rhs[2] = self.v_source;
        }

        fn num_nodes(&self) -> usize {
            2
        }

        fn num_vsources(&self) -> usize {
            1
        }
    }

    #[test]
    fn test_sweep_cpu_fallback() {
        let backend = BackendSelector::cpu_only();
        let config = DispatchConfig::default();
        let factory = SimpleDividerFactory { r2_nominal: 1000.0 };
        let generator = LinearSweepGenerator::new(5);
        let variations = vec![ParameterVariation::new("R1", 1000.0).with_bounds(500.0, 1500.0)];

        let result = solve_batched_sweep_gpu(
            &backend,
            &factory,
            &generator,
            &variations,
            &ConvergenceCriteria::default(),
            &config,
        );

        assert!(result.is_ok());
        let result = result.unwrap();

        assert!(!result.used_gpu);
        assert_eq!(result.backend_used, BackendType::Cpu);
        assert_eq!(result.total_count, 5);
        assert_eq!(result.converged_count, 5);
    }

    #[test]
    fn test_sweep_auto_backend() {
        let config = DispatchConfig::default();
        let factory = SimpleDividerFactory { r2_nominal: 1000.0 };
        let generator = LinearSweepGenerator::new(5);
        let variations = vec![ParameterVariation::new("R1", 1000.0).with_bounds(500.0, 1500.0)];

        let result = solve_batched_sweep_auto(
            &factory,
            &generator,
            &variations,
            &ConvergenceCriteria::default(),
            &config,
        );

        assert!(result.is_ok());
        let result = result.unwrap();

        // Should fallback to CPU for small batch
        assert_eq!(result.total_count, 5);
        assert_eq!(result.converged_count, 5);
    }

    #[test]
    fn test_sweep_monte_carlo() {
        let backend = BackendSelector::auto();
        let config = DispatchConfig::default();
        let factory = SimpleDividerFactory { r2_nominal: 1000.0 };
        let generator = MonteCarloGenerator::new(100).with_seed(42);
        let variations = vec![
            ParameterVariation::new("R1", 1000.0)
                .with_bounds(500.0, 1500.0)
                .with_sigma(0.1),
        ];

        let result = solve_batched_sweep_gpu(
            &backend,
            &factory,
            &generator,
            &variations,
            &ConvergenceCriteria::default(),
            &config,
        );

        assert!(result.is_ok());
        let result = result.unwrap();

        assert_eq!(result.total_count, 100);
        assert!(result.converged_count >= 95);

        // Check statistics
        let stats = result.statistics(1);
        // V(1) should be around 5V for voltage divider with R1 ≈ R2
        assert!(
            stats.mean > 3.0 && stats.mean < 7.0,
            "Mean was {}",
            stats.mean
        );
    }

    #[test]
    fn test_sweep_corners() {
        let backend = BackendSelector::auto();
        let config = DispatchConfig::default();
        let factory = SimpleDividerFactory { r2_nominal: 1000.0 };
        let generator = CornerGenerator;
        let variations = vec![ParameterVariation::new("R1", 1000.0).with_bounds(500.0, 1500.0)];

        let result = solve_batched_sweep_gpu(
            &backend,
            &factory,
            &generator,
            &variations,
            &ConvergenceCriteria::default(),
            &config,
        );

        assert!(result.is_ok());
        let result = result.unwrap();

        assert_eq!(result.total_count, 2); // 2^1 corners
        assert_eq!(result.converged_count, 2);
    }
}
