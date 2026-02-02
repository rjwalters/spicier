//! Solver dispatch configuration and backend integration.
//!
//! Provides a unified configuration for selecting:
//! - Compute backend (CPU, CUDA, Metal)
//! - Solver strategy (Direct LU, Iterative GMRES, Auto)
//! - Size thresholds for dispatch decisions

use crate::backend::ComputeBackend;
use crate::gmres::GmresConfig;

/// Solver dispatch configuration.
///
/// Controls how the solver selects between different backends and algorithms
/// based on system size and available hardware.
#[derive(Debug, Clone)]
pub struct DispatchConfig {
    /// Compute backend to use.
    pub backend: ComputeBackend,
    /// Solver strategy selection.
    pub strategy: SolverDispatchStrategy,
    /// Size threshold below which CPU is always used (even if GPU available).
    pub cpu_threshold: usize,
    /// Size threshold above which GMRES is preferred over direct LU.
    pub gmres_threshold: usize,
    /// GMRES configuration for iterative solving.
    pub gmres_config: GmresConfig,
}

impl Default for DispatchConfig {
    fn default() -> Self {
        Self {
            backend: ComputeBackend::Cpu,
            strategy: SolverDispatchStrategy::Auto,
            cpu_threshold: 1000,      // < 1k nodes: always CPU
            gmres_threshold: 10_000,  // >= 10k nodes: prefer GMRES
            gmres_config: GmresConfig::default(),
        }
    }
}

impl DispatchConfig {
    /// Create a CPU-only configuration.
    pub fn cpu() -> Self {
        Self {
            backend: ComputeBackend::Cpu,
            ..Default::default()
        }
    }

    /// Create a configuration with CUDA backend.
    pub fn cuda(device_id: usize) -> Self {
        Self {
            backend: ComputeBackend::Cuda { device_id },
            ..Default::default()
        }
    }

    /// Create a configuration with Metal backend.
    pub fn metal(adapter_name: impl Into<String>) -> Self {
        Self {
            backend: ComputeBackend::Metal {
                adapter_name: adapter_name.into(),
            },
            ..Default::default()
        }
    }

    /// Set the solver strategy.
    pub fn with_strategy(mut self, strategy: SolverDispatchStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Set the CPU threshold.
    pub fn with_cpu_threshold(mut self, threshold: usize) -> Self {
        self.cpu_threshold = threshold;
        self
    }

    /// Set the GMRES threshold.
    pub fn with_gmres_threshold(mut self, threshold: usize) -> Self {
        self.gmres_threshold = threshold;
        self
    }

    /// Set the GMRES configuration.
    pub fn with_gmres_config(mut self, config: GmresConfig) -> Self {
        self.gmres_config = config;
        self
    }

    /// Decide whether to use GPU for a given system size.
    pub fn use_gpu(&self, size: usize) -> bool {
        if size < self.cpu_threshold {
            return false;
        }
        !matches!(self.backend, ComputeBackend::Cpu)
    }

    /// Decide whether to use GMRES for a given system size.
    pub fn use_gmres(&self, size: usize) -> bool {
        match self.strategy {
            SolverDispatchStrategy::DirectLU => false,
            SolverDispatchStrategy::IterativeGmres => true,
            SolverDispatchStrategy::Auto => size >= self.gmres_threshold,
        }
    }

    /// Get a human-readable description of the dispatch decision for a size.
    pub fn describe(&self, size: usize) -> String {
        let backend = if self.use_gpu(size) {
            self.backend.name()
        } else {
            "CPU"
        };
        let solver = if self.use_gmres(size) {
            "GMRES"
        } else {
            "Direct LU"
        };
        format!("{} with {}", solver, backend)
    }
}

/// Solver strategy for dispatch decisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SolverDispatchStrategy {
    /// Automatically select based on system size.
    #[default]
    Auto,
    /// Always use direct LU factorization.
    DirectLU,
    /// Always use iterative GMRES.
    IterativeGmres,
}

impl SolverDispatchStrategy {
    /// Parse from a string.
    pub fn from_name(name: &str) -> Option<Self> {
        match name.to_lowercase().as_str() {
            "auto" => Some(Self::Auto),
            "lu" | "direct" | "directlu" => Some(Self::DirectLU),
            "gmres" | "iterative" => Some(Self::IterativeGmres),
            _ => None,
        }
    }
}

/// Result of a dispatched solve operation.
#[derive(Debug, Clone)]
pub struct DispatchedSolveInfo {
    /// Backend that was actually used.
    pub backend_used: String,
    /// Solver that was actually used.
    pub solver_used: String,
    /// Number of iterations (for iterative solvers).
    pub iterations: Option<usize>,
    /// Final residual (for iterative solvers).
    pub residual: Option<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config() {
        let config = DispatchConfig::default();
        assert_eq!(config.backend, ComputeBackend::Cpu);
        assert_eq!(config.cpu_threshold, 1000);
        assert_eq!(config.gmres_threshold, 10_000);
    }

    #[test]
    fn use_gpu_decision() {
        let cpu_config = DispatchConfig::cpu();
        assert!(!cpu_config.use_gpu(500));
        assert!(!cpu_config.use_gpu(5000));

        let cuda_config = DispatchConfig::cuda(0).with_cpu_threshold(1000);
        assert!(!cuda_config.use_gpu(500));  // Below threshold
        assert!(cuda_config.use_gpu(1500));  // Above threshold
    }

    #[test]
    fn use_gmres_decision() {
        let config = DispatchConfig::default();

        // Auto: uses threshold
        assert!(!config.use_gmres(5000));
        assert!(config.use_gmres(15000));

        // Force direct LU
        let lu_config = config.clone().with_strategy(SolverDispatchStrategy::DirectLU);
        assert!(!lu_config.use_gmres(15000));

        // Force GMRES
        let gmres_config = config.clone().with_strategy(SolverDispatchStrategy::IterativeGmres);
        assert!(gmres_config.use_gmres(500));
    }

    #[test]
    fn describe_output() {
        let config = DispatchConfig::cuda(0)
            .with_cpu_threshold(1000)
            .with_gmres_threshold(5000);

        // Small: CPU + LU
        assert_eq!(config.describe(500), "Direct LU with CPU");

        // Medium: GPU + LU
        assert_eq!(config.describe(2000), "Direct LU with CUDA");

        // Large: GPU + GMRES
        assert_eq!(config.describe(10000), "GMRES with CUDA");
    }

    #[test]
    fn strategy_from_name() {
        assert_eq!(
            SolverDispatchStrategy::from_name("auto"),
            Some(SolverDispatchStrategy::Auto)
        );
        assert_eq!(
            SolverDispatchStrategy::from_name("LU"),
            Some(SolverDispatchStrategy::DirectLU)
        );
        assert_eq!(
            SolverDispatchStrategy::from_name("gmres"),
            Some(SolverDispatchStrategy::IterativeGmres)
        );
        assert_eq!(SolverDispatchStrategy::from_name("invalid"), None);
    }
}
