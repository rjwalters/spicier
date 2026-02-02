//! Preconditioners for iterative solvers.
//!
//! Preconditioners improve the convergence rate of iterative solvers like GMRES
//! by transforming the linear system into one with better spectral properties.

use num_complex::Complex64 as C64;

/// A preconditioner for real-valued linear systems.
///
/// Given a linear system Ax = b, a preconditioner M approximates A^(-1).
/// The preconditioned system M^(-1)Ax = M^(-1)b (left preconditioning) has
/// better convergence properties.
pub trait RealPreconditioner: Send + Sync {
    /// Apply the preconditioner: y = M^(-1) * x.
    fn apply(&self, x: &[f64], y: &mut [f64]);

    /// Dimension of the preconditioner.
    fn dim(&self) -> usize;
}

/// A preconditioner for complex-valued linear systems.
pub trait ComplexPreconditioner: Send + Sync {
    /// Apply the preconditioner: y = M^(-1) * x.
    fn apply(&self, x: &[C64], y: &mut [C64]);

    /// Dimension of the preconditioner.
    fn dim(&self) -> usize;
}

// ============================================================================
// Jacobi (Diagonal) Preconditioner
// ============================================================================

/// Jacobi (diagonal) preconditioner for real systems.
///
/// Uses M = diag(A), so M^(-1) * x = x / diag(A).
/// Simple but effective for diagonally dominant systems like MNA matrices.
///
/// # Construction
///
/// From triplets, extracts the diagonal elements. Zero or near-zero diagonal
/// entries are replaced with 1.0 to avoid division issues.
pub struct JacobiPreconditioner {
    /// Inverse of diagonal elements.
    inv_diag: Vec<f64>,
}

impl JacobiPreconditioner {
    /// Create from matrix triplets.
    ///
    /// Extracts diagonal elements and computes their inverses.
    /// Near-zero diagonal entries (< 1e-30) are treated as 1.0.
    pub fn from_triplets(size: usize, triplets: &[(usize, usize, f64)]) -> Self {
        let mut diag = vec![0.0; size];

        // Sum diagonal entries (triplets may have duplicates)
        for &(row, col, value) in triplets {
            if row == col && row < size {
                diag[row] += value;
            }
        }

        // Compute inverses, handling near-zero entries
        let inv_diag: Vec<f64> = diag
            .iter()
            .map(|&d| {
                if d.abs() < 1e-30 {
                    1.0 // Don't scale entries with zero diagonal
                } else {
                    1.0 / d
                }
            })
            .collect();

        Self { inv_diag }
    }

    /// Create from a diagonal vector.
    pub fn from_diagonal(diag: &[f64]) -> Self {
        let inv_diag: Vec<f64> = diag
            .iter()
            .map(|&d| if d.abs() < 1e-30 { 1.0 } else { 1.0 / d })
            .collect();

        Self { inv_diag }
    }
}

impl RealPreconditioner for JacobiPreconditioner {
    fn apply(&self, x: &[f64], y: &mut [f64]) {
        assert_eq!(x.len(), self.inv_diag.len());
        assert_eq!(y.len(), self.inv_diag.len());

        for (i, (&xi, &inv_di)) in x.iter().zip(self.inv_diag.iter()).enumerate() {
            y[i] = xi * inv_di;
        }
    }

    fn dim(&self) -> usize {
        self.inv_diag.len()
    }
}

/// Jacobi (diagonal) preconditioner for complex systems.
pub struct ComplexJacobiPreconditioner {
    /// Inverse of diagonal elements.
    inv_diag: Vec<C64>,
}

impl ComplexJacobiPreconditioner {
    /// Create from matrix triplets.
    pub fn from_triplets(size: usize, triplets: &[(usize, usize, C64)]) -> Self {
        let mut diag = vec![C64::new(0.0, 0.0); size];

        for &(row, col, value) in triplets {
            if row == col && row < size {
                diag[row] += value;
            }
        }

        let inv_diag: Vec<C64> = diag
            .iter()
            .map(|&d| {
                if d.norm() < 1e-30 {
                    C64::new(1.0, 0.0)
                } else {
                    C64::new(1.0, 0.0) / d
                }
            })
            .collect();

        Self { inv_diag }
    }

    /// Create from a diagonal vector.
    pub fn from_diagonal(diag: &[C64]) -> Self {
        let inv_diag: Vec<C64> = diag
            .iter()
            .map(|&d| {
                if d.norm() < 1e-30 {
                    C64::new(1.0, 0.0)
                } else {
                    C64::new(1.0, 0.0) / d
                }
            })
            .collect();

        Self { inv_diag }
    }
}

impl ComplexPreconditioner for ComplexJacobiPreconditioner {
    fn apply(&self, x: &[C64], y: &mut [C64]) {
        assert_eq!(x.len(), self.inv_diag.len());
        assert_eq!(y.len(), self.inv_diag.len());

        for (i, (&xi, &inv_di)) in x.iter().zip(self.inv_diag.iter()).enumerate() {
            y[i] = xi * inv_di;
        }
    }

    fn dim(&self) -> usize {
        self.inv_diag.len()
    }
}

// ============================================================================
// Identity Preconditioner (no preconditioning)
// ============================================================================

/// Identity preconditioner (no-op).
///
/// Useful as a baseline or when no preconditioning is desired.
pub struct IdentityPreconditioner {
    size: usize,
}

impl IdentityPreconditioner {
    /// Create an identity preconditioner of the given size.
    pub fn new(size: usize) -> Self {
        Self { size }
    }
}

impl RealPreconditioner for IdentityPreconditioner {
    fn apply(&self, x: &[f64], y: &mut [f64]) {
        y.copy_from_slice(x);
    }

    fn dim(&self) -> usize {
        self.size
    }
}

impl ComplexPreconditioner for IdentityPreconditioner {
    fn apply(&self, x: &[C64], y: &mut [C64]) {
        y.copy_from_slice(x);
    }

    fn dim(&self) -> usize {
        self.size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jacobi_from_triplets() {
        // 3x3 diagonal matrix with diag = [2, 4, 5]
        let triplets = vec![(0, 0, 2.0), (1, 1, 4.0), (2, 2, 5.0)];
        let precond = JacobiPreconditioner::from_triplets(3, &triplets);

        let x = vec![2.0, 8.0, 10.0];
        let mut y = vec![0.0; 3];
        precond.apply(&x, &mut y);

        // y = x / diag = [1, 2, 2]
        assert!((y[0] - 1.0).abs() < 1e-15);
        assert!((y[1] - 2.0).abs() < 1e-15);
        assert!((y[2] - 2.0).abs() < 1e-15);
    }

    #[test]
    fn jacobi_handles_duplicate_triplets() {
        // Diagonal element 2.0 + 3.0 = 5.0
        let triplets = vec![(0, 0, 2.0), (0, 0, 3.0)];
        let precond = JacobiPreconditioner::from_triplets(1, &triplets);

        let x = vec![10.0];
        let mut y = vec![0.0];
        precond.apply(&x, &mut y);

        assert!((y[0] - 2.0).abs() < 1e-15); // 10 / 5 = 2
    }

    #[test]
    fn jacobi_handles_zero_diagonal() {
        let triplets = vec![(0, 0, 0.0), (1, 1, 2.0)];
        let precond = JacobiPreconditioner::from_triplets(2, &triplets);

        let x = vec![5.0, 4.0];
        let mut y = vec![0.0; 2];
        precond.apply(&x, &mut y);

        // Zero diagonal treated as 1.0
        assert!((y[0] - 5.0).abs() < 1e-15);
        assert!((y[1] - 2.0).abs() < 1e-15);
    }

    #[test]
    fn jacobi_with_off_diagonal() {
        // Matrix with off-diagonal entries (ignored by Jacobi)
        let triplets = vec![
            (0, 0, 4.0),
            (0, 1, 1.0), // off-diagonal, ignored
            (1, 0, 1.0), // off-diagonal, ignored
            (1, 1, 2.0),
        ];
        let precond = JacobiPreconditioner::from_triplets(2, &triplets);

        let x = vec![8.0, 6.0];
        let mut y = vec![0.0; 2];
        precond.apply(&x, &mut y);

        assert!((y[0] - 2.0).abs() < 1e-15); // 8 / 4
        assert!((y[1] - 3.0).abs() < 1e-15); // 6 / 2
    }

    #[test]
    fn complex_jacobi_basic() {
        let triplets = vec![
            (0, 0, C64::new(2.0, 0.0)),
            (1, 1, C64::new(0.0, 4.0)), // pure imaginary
        ];
        let precond = ComplexJacobiPreconditioner::from_triplets(2, &triplets);

        let x = vec![C64::new(4.0, 0.0), C64::new(0.0, 8.0)];
        let mut y = vec![C64::new(0.0, 0.0); 2];
        precond.apply(&x, &mut y);

        // y[0] = 4 / 2 = 2
        assert!((y[0] - C64::new(2.0, 0.0)).norm() < 1e-15);
        // y[1] = 8i / 4i = 2
        assert!((y[1] - C64::new(2.0, 0.0)).norm() < 1e-15);
    }

    #[test]
    fn identity_preconditioner_real() {
        let precond = IdentityPreconditioner::new(3);
        let x = vec![1.0, 2.0, 3.0];
        let mut y = vec![0.0; 3];

        RealPreconditioner::apply(&precond, &x, &mut y);

        assert_eq!(y, x);
    }

    #[test]
    fn identity_preconditioner_complex() {
        let precond = IdentityPreconditioner::new(2);
        let x = vec![C64::new(1.0, 2.0), C64::new(3.0, 4.0)];
        let mut y = vec![C64::new(0.0, 0.0); 2];

        ComplexPreconditioner::apply(&precond, &x, &mut y);

        assert_eq!(y, x);
    }

    #[test]
    fn preconditioner_dim() {
        let jacobi = JacobiPreconditioner::from_diagonal(&[1.0, 2.0, 3.0]);
        assert_eq!(jacobi.dim(), 3);

        let identity = IdentityPreconditioner::new(5);
        assert_eq!(RealPreconditioner::dim(&identity), 5);
    }
}
