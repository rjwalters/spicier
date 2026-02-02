//! Helper functions for GMRES solver.

use num_complex::Complex64 as C64;
use spicier_simd::{SimdCapability, complex_conjugate_dot_product, real_dot_product};

/// Compute the 2-norm of a complex vector using SIMD-accelerated dot product.
pub fn complex_vec_norm(v: &[C64], cap: SimdCapability) -> f64 {
    complex_conjugate_dot_product(v, v, cap).re.sqrt()
}

/// Compute the 2-norm of a real vector using SIMD-accelerated dot product.
pub fn real_vec_norm(v: &[f64], cap: SimdCapability) -> f64 {
    real_dot_product(v, v, cap).sqrt()
}

/// Compute Givens rotation coefficients for complex values.
///
/// Returns (c, s) such that:
/// ```text
/// [ c* s* ] [ a ]   [ r ]
/// [-s  c  ] [ b ] = [ 0 ]
/// ```
pub fn complex_givens_rotation(a: C64, b: C64) -> (C64, C64) {
    if b.norm() < 1e-30 {
        return (C64::new(1.0, 0.0), C64::new(0.0, 0.0));
    }
    let r = (a.norm_sqr() + b.norm_sqr()).sqrt();
    let c = a / r;
    let s = b / r;
    (c, s)
}

/// Compute Givens rotation coefficients for real values.
///
/// Returns (c, s) such that:
/// ```text
/// [ c  s ] [ a ]   [ r ]
/// [-s  c ] [ b ] = [ 0 ]
/// ```
pub fn real_givens_rotation(a: f64, b: f64) -> (f64, f64) {
    if b.abs() < 1e-30 {
        return (1.0, 0.0);
    }
    let r = (a * a + b * b).sqrt();
    let c = a / r;
    let s = b / r;
    (c, s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_complex_vec_norm() {
        let cap = SimdCapability::detect();
        let v = vec![C64::new(3.0, 4.0)];
        assert!((complex_vec_norm(&v, cap) - 5.0).abs() < 1e-15);
    }

    #[test]
    fn test_real_vec_norm() {
        let cap = SimdCapability::detect();
        let v = vec![3.0, 4.0];
        assert!((real_vec_norm(&v, cap) - 5.0).abs() < 1e-15);
    }

    #[test]
    fn test_complex_givens_rotation() {
        let a = C64::new(3.0, 0.0);
        let b = C64::new(4.0, 0.0);
        let (c, s) = complex_givens_rotation(a, b);

        let new_b = -s * a + c * b;
        assert!(new_b.norm() < 1e-10);
    }

    #[test]
    fn test_real_givens_rotation() {
        let (c, s) = real_givens_rotation(3.0, 4.0);

        // After rotation, b component should be zero
        let new_b = -s * 3.0 + c * 4.0;
        assert!(new_b.abs() < 1e-10);

        // Check normalization: c^2 + s^2 = 1
        assert!((c * c + s * s - 1.0).abs() < 1e-15);
    }
}
