//! Sparse matrix operator wrappers for iterative solvers.
//!
//! These wrappers implement [`RealOperator`] and [`ComplexOperator`] traits for
//! faer's sparse column matrices, enabling their use with GMRES and other iterative solvers.

use crate::operator::{ComplexOperator, RealOperator};
use faer::prelude::*;
use faer::sparse::{SparseColMat, Triplet};
use num_complex::Complex64 as C64;

/// Sparse real-valued operator for iterative solvers.
///
/// Wraps a faer `SparseColMat<usize, f64>` and implements `RealOperator`,
/// enabling its use with real-valued iterative solvers.
pub struct SparseRealOperator {
    matrix: SparseColMat<usize, f64>,
}

impl SparseRealOperator {
    /// Create from an existing sparse matrix.
    pub fn from_matrix(matrix: SparseColMat<usize, f64>) -> Self {
        Self { matrix }
    }

    /// Create from triplets (row, col, value).
    ///
    /// Duplicate entries at the same position are summed.
    pub fn from_triplets(size: usize, triplets: &[(usize, usize, f64)]) -> Option<Self> {
        let faer_triplets: Vec<_> = triplets
            .iter()
            .map(|&(r, c, v)| Triplet::new(r, c, v))
            .collect();

        SparseColMat::<usize, f64>::try_new_from_triplets(size, size, &faer_triplets)
            .ok()
            .map(|matrix| Self { matrix })
    }

    /// Get a reference to the underlying matrix.
    pub fn matrix(&self) -> &SparseColMat<usize, f64> {
        &self.matrix
    }
}

impl RealOperator for SparseRealOperator {
    fn dim(&self) -> usize {
        self.matrix.nrows()
    }

    fn apply(&self, x: &[f64], y: &mut [f64]) {
        let n = self.matrix.nrows();
        assert_eq!(x.len(), n);
        assert_eq!(y.len(), n);

        // Zero output
        y.iter_mut().for_each(|yi| *yi = 0.0);

        // CSC matrix-vector multiplication: y = A * x
        // For each column j, add A[:, j] * x[j] to y
        let mat_ref = self.matrix.as_ref();
        let col_ptrs = mat_ref.col_ptr();
        let row_indices = mat_ref.row_idx();
        let values = mat_ref.val();

        for j in 0..n {
            let col_start = col_ptrs[j];
            let col_end = col_ptrs[j + 1];
            let xj = x[j];

            for idx in col_start..col_end {
                let i = row_indices[idx];
                let aij = values[idx];
                y[i] += aij * xj;
            }
        }
    }
}

/// Sparse complex-valued operator for iterative solvers.
///
/// Wraps a faer `SparseColMat<usize, c64>` and implements `ComplexOperator`,
/// enabling its use with complex-valued iterative solvers like GMRES.
pub struct SparseComplexOperator {
    matrix: SparseColMat<usize, c64>,
}

impl SparseComplexOperator {
    /// Create from an existing sparse matrix.
    pub fn from_matrix(matrix: SparseColMat<usize, c64>) -> Self {
        Self { matrix }
    }

    /// Create from triplets (row, col, value).
    ///
    /// Duplicate entries at the same position are summed.
    pub fn from_triplets(size: usize, triplets: &[(usize, usize, C64)]) -> Option<Self> {
        let faer_triplets: Vec<_> = triplets
            .iter()
            .map(|&(r, c, v)| Triplet::new(r, c, c64::new(v.re, v.im)))
            .collect();

        SparseColMat::<usize, c64>::try_new_from_triplets(size, size, &faer_triplets)
            .ok()
            .map(|matrix| Self { matrix })
    }

    /// Get a reference to the underlying matrix.
    pub fn matrix(&self) -> &SparseColMat<usize, c64> {
        &self.matrix
    }
}

impl ComplexOperator for SparseComplexOperator {
    fn dim(&self) -> usize {
        self.matrix.nrows()
    }

    fn apply(&self, x: &[C64], y: &mut [C64]) {
        let n = self.matrix.nrows();
        assert_eq!(x.len(), n);
        assert_eq!(y.len(), n);

        // Zero output
        y.iter_mut().for_each(|yi| *yi = C64::new(0.0, 0.0));

        // CSC matrix-vector multiplication: y = A * x
        let mat_ref = self.matrix.as_ref();
        let col_ptrs = mat_ref.col_ptr();
        let row_indices = mat_ref.row_idx();
        let values = mat_ref.val();

        for j in 0..n {
            let col_start = col_ptrs[j];
            let col_end = col_ptrs[j + 1];
            let xj = x[j];

            for idx in col_start..col_end {
                let i = row_indices[idx];
                let aij = values[idx];
                y[i] += C64::new(aij.re, aij.im) * xj;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sparse_real_identity() {
        // 3x3 identity matrix
        let triplets = vec![(0, 0, 1.0), (1, 1, 1.0), (2, 2, 1.0)];
        let op = SparseRealOperator::from_triplets(3, &triplets).unwrap();

        assert_eq!(op.dim(), 3);

        let x = vec![1.0, 2.0, 3.0];
        let mut y = vec![0.0; 3];
        op.apply(&x, &mut y);

        assert!((y[0] - 1.0).abs() < 1e-15);
        assert!((y[1] - 2.0).abs() < 1e-15);
        assert!((y[2] - 3.0).abs() < 1e-15);
    }

    #[test]
    fn sparse_real_tridiagonal() {
        // Tridiagonal matrix:
        // [ 2 -1  0]
        // [-1  2 -1]
        // [ 0 -1  2]
        let triplets = vec![
            (0, 0, 2.0),
            (0, 1, -1.0),
            (1, 0, -1.0),
            (1, 1, 2.0),
            (1, 2, -1.0),
            (2, 1, -1.0),
            (2, 2, 2.0),
        ];
        let op = SparseRealOperator::from_triplets(3, &triplets).unwrap();

        let x = vec![1.0, 2.0, 3.0];
        let mut y = vec![0.0; 3];
        op.apply(&x, &mut y);

        // y[0] = 2*1 - 1*2 = 0
        // y[1] = -1*1 + 2*2 - 1*3 = 0
        // y[2] = -1*2 + 2*3 = 4
        assert!((y[0] - 0.0).abs() < 1e-15);
        assert!((y[1] - 0.0).abs() < 1e-15);
        assert!((y[2] - 4.0).abs() < 1e-15);
    }

    #[test]
    fn sparse_complex_identity() {
        let triplets = vec![(0, 0, C64::new(1.0, 0.0)), (1, 1, C64::new(1.0, 0.0))];
        let op = SparseComplexOperator::from_triplets(2, &triplets).unwrap();

        assert_eq!(op.dim(), 2);

        let x = vec![C64::new(1.0, 2.0), C64::new(3.0, 4.0)];
        let mut y = vec![C64::new(0.0, 0.0); 2];
        op.apply(&x, &mut y);

        assert!((y[0] - x[0]).norm() < 1e-15);
        assert!((y[1] - x[1]).norm() < 1e-15);
    }

    #[test]
    fn sparse_complex_with_imaginary() {
        // Matrix: [[1+i, 0], [0, 2-i]]
        let triplets = vec![(0, 0, C64::new(1.0, 1.0)), (1, 1, C64::new(2.0, -1.0))];
        let op = SparseComplexOperator::from_triplets(2, &triplets).unwrap();

        let x = vec![C64::new(1.0, 0.0), C64::new(0.0, 1.0)];
        let mut y = vec![C64::new(0.0, 0.0); 2];
        op.apply(&x, &mut y);

        // y[0] = (1+i) * 1 = 1+i
        // y[1] = (2-i) * i = 2i - i^2 = 1 + 2i
        assert!((y[0] - C64::new(1.0, 1.0)).norm() < 1e-15);
        assert!((y[1] - C64::new(1.0, 2.0)).norm() < 1e-15);
    }

    #[test]
    fn sparse_real_as_trait_object() {
        let triplets = vec![(0, 0, 2.0), (1, 1, 3.0)];
        let op = SparseRealOperator::from_triplets(2, &triplets).unwrap();
        let op_ref: &dyn RealOperator = &op;

        let x = vec![5.0, 7.0];
        let mut y = vec![0.0; 2];
        op_ref.apply(&x, &mut y);

        assert!((y[0] - 10.0).abs() < 1e-15);
        assert!((y[1] - 21.0).abs() < 1e-15);
    }

    #[test]
    fn sparse_complex_as_trait_object() {
        let triplets = vec![(0, 0, C64::new(2.0, 0.0)), (1, 1, C64::new(3.0, 0.0))];
        let op = SparseComplexOperator::from_triplets(2, &triplets).unwrap();
        let op_ref: &dyn ComplexOperator = &op;

        let x = vec![C64::new(5.0, 0.0), C64::new(7.0, 0.0)];
        let mut y = vec![C64::new(0.0, 0.0); 2];
        op_ref.apply(&x, &mut y);

        assert!((y[0] - C64::new(10.0, 0.0)).norm() < 1e-15);
        assert!((y[1] - C64::new(21.0, 0.0)).norm() < 1e-15);
    }
}
