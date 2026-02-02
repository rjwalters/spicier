//! Window functions for spectral analysis.
//!
//! Window functions reduce spectral leakage when analyzing finite-length signals.
//! Different windows trade off between main lobe width and side lobe level.

use std::f64::consts::PI;

/// Window function types for FFT analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WindowFunction {
    /// Rectangular (no windowing). Best frequency resolution, worst leakage.
    Rectangular,
    /// Hanning window. Good general-purpose choice.
    #[default]
    Hanning,
    /// Hamming window. Similar to Hanning with slightly different coefficients.
    Hamming,
    /// Blackman window. Best side lobe suppression, widest main lobe.
    Blackman,
}

impl WindowFunction {
    /// Apply the window function to a signal.
    ///
    /// Returns a new vector with the windowed signal.
    pub fn apply(&self, signal: &[f64]) -> Vec<f64> {
        let n = signal.len();
        if n == 0 {
            return Vec::new();
        }

        match self {
            WindowFunction::Rectangular => signal.to_vec(),
            WindowFunction::Hanning => signal
                .iter()
                .enumerate()
                .map(|(i, &x)| {
                    let w = 0.5 * (1.0 - (2.0 * PI * i as f64 / (n - 1) as f64).cos());
                    x * w
                })
                .collect(),
            WindowFunction::Hamming => signal
                .iter()
                .enumerate()
                .map(|(i, &x)| {
                    let w = 0.54 - 0.46 * (2.0 * PI * i as f64 / (n - 1) as f64).cos();
                    x * w
                })
                .collect(),
            WindowFunction::Blackman => signal
                .iter()
                .enumerate()
                .map(|(i, &x)| {
                    let a0 = 0.42;
                    let a1 = 0.5;
                    let a2 = 0.08;
                    let phase = 2.0 * PI * i as f64 / (n - 1) as f64;
                    let w = a0 - a1 * phase.cos() + a2 * (2.0 * phase).cos();
                    x * w
                })
                .collect(),
        }
    }

    /// Compute the coherent gain of the window function.
    ///
    /// This is used to normalize the FFT magnitude. It's the sum of the window
    /// coefficients divided by N.
    pub fn coherent_gain(&self, n: usize) -> f64 {
        if n == 0 {
            return 0.0;
        }

        match self {
            WindowFunction::Rectangular => 1.0,
            WindowFunction::Hanning => {
                // Sum of Hanning window / N ≈ 0.5
                let sum: f64 = (0..n)
                    .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f64 / (n - 1) as f64).cos()))
                    .sum();
                sum / n as f64
            }
            WindowFunction::Hamming => {
                // Sum of Hamming window / N ≈ 0.54
                let sum: f64 = (0..n)
                    .map(|i| 0.54 - 0.46 * (2.0 * PI * i as f64 / (n - 1) as f64).cos())
                    .sum();
                sum / n as f64
            }
            WindowFunction::Blackman => {
                let a0 = 0.42;
                let a1 = 0.5;
                let a2 = 0.08;
                let sum: f64 = (0..n)
                    .map(|i| {
                        let phase = 2.0 * PI * i as f64 / (n - 1) as f64;
                        a0 - a1 * phase.cos() + a2 * (2.0 * phase).cos()
                    })
                    .sum();
                sum / n as f64
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rectangular_window() {
        let signal = vec![1.0, 2.0, 3.0, 4.0];
        let windowed = WindowFunction::Rectangular.apply(&signal);
        assert_eq!(windowed, signal);
    }

    #[test]
    fn test_hanning_window_endpoints() {
        // Hanning window should be 0 at endpoints
        let signal = vec![1.0; 100];
        let windowed = WindowFunction::Hanning.apply(&signal);
        assert!(windowed[0].abs() < 1e-10);
        assert!(windowed[99].abs() < 1e-10);
        // Middle should be approximately 1.0
        assert!((windowed[50] - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_hamming_window_endpoints() {
        // Hamming window should be ~0.08 at endpoints (not zero)
        let signal = vec![1.0; 100];
        let windowed = WindowFunction::Hamming.apply(&signal);
        assert!((windowed[0] - 0.08).abs() < 0.01);
        assert!((windowed[99] - 0.08).abs() < 0.01);
    }

    #[test]
    fn test_blackman_window_endpoints() {
        // Blackman window should be very small at endpoints
        let signal = vec![1.0; 100];
        let windowed = WindowFunction::Blackman.apply(&signal);
        assert!(windowed[0].abs() < 0.001);
        assert!(windowed[99].abs() < 0.001);
    }

    #[test]
    fn test_coherent_gain_rectangular() {
        let gain = WindowFunction::Rectangular.coherent_gain(1024);
        assert!((gain - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_coherent_gain_hanning() {
        let gain = WindowFunction::Hanning.coherent_gain(1024);
        assert!((gain - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_empty_signal() {
        let signal: Vec<f64> = Vec::new();
        let windowed = WindowFunction::Hanning.apply(&signal);
        assert!(windowed.is_empty());
    }
}
