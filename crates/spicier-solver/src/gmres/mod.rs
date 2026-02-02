//! GMRES iterative solver for linear systems.
//!
//! Provides both complex and real-valued GMRES solvers. The complex solver
//! uses SIMD-accelerated conjugate dot products for Gram-Schmidt orthogonalization.
//!
//! # Usage
//!
//! ```ignore
//! use spicier_solver::{solve_gmres, solve_gmres_real, GmresConfig};
//!
//! // Complex system
//! let result = solve_gmres(&complex_operator, &complex_rhs, &GmresConfig::default());
//!
//! // Real system
//! let result = solve_gmres_real(&real_operator, &real_rhs, &GmresConfig::default());
//! ```
//!
//! # Module Structure
//!
//! - [`complex`] - Complex-valued GMRES solvers
//! - [`real`] - Real-valued GMRES solvers
//! - [`helpers`] - Givens rotation and vector norm utilities

pub mod complex;
pub mod helpers;
pub mod real;

// Re-export main types and functions
pub use complex::{GmresResult, solve_gmres, solve_gmres_preconditioned};
pub use real::{RealGmresResult, solve_gmres_real, solve_gmres_real_preconditioned};

/// GMRES solver configuration.
#[derive(Debug, Clone)]
pub struct GmresConfig {
    /// Maximum number of iterations.
    pub max_iter: usize,
    /// Convergence tolerance (relative residual).
    pub tol: f64,
    /// Restart parameter (Krylov subspace dimension before restart).
    pub restart: usize,
}

impl Default for GmresConfig {
    fn default() -> Self {
        Self {
            max_iter: 500,
            tol: 1e-8,
            restart: 30,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gmres_config_default() {
        let config = GmresConfig::default();
        assert_eq!(config.max_iter, 500);
        assert!((config.tol - 1e-8).abs() < 1e-15);
        assert_eq!(config.restart, 30);
    }
}
