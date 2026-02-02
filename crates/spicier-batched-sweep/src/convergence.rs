//! Convergence tracking for batched Newton-Raphson sweeps.
//!
//! This module provides infrastructure for tracking per-point convergence status
//! in batched nonlinear circuit simulation. When running Monte Carlo or parameter
//! sweeps on nonlinear circuits, different parameter combinations may converge at
//! different rates. This module enables early termination for converged points,
//! avoiding wasted computation.
//!
//! # Design
//!
//! Two strategies are supported:
//!
//! 1. **Masking**: Keep all points in the batch, but skip computation for converged ones.
//!    Simpler implementation, some wasted memory bandwidth.
//!
//! 2. **Compaction**: Remove converged points from the active set, keeping only
//!    unconverged points. More complex, but more efficient for large batches.
//!
//! # Example
//!
//! ```
//! use spicier_batched_sweep::convergence::{ConvergenceTracker, ConvergenceStatus};
//!
//! let mut tracker = ConvergenceTracker::new(1000); // 1000 sweep points
//!
//! // After NR iteration, mark converged points
//! tracker.mark_converged(42);
//! tracker.mark_converged(100);
//!
//! // Check status
//! assert_eq!(tracker.status(42), ConvergenceStatus::Converged);
//! assert_eq!(tracker.status(0), ConvergenceStatus::Active);
//!
//! // Get active indices for next iteration
//! let active = tracker.active_indices();
//! assert_eq!(active.len(), 998);
//! ```

/// Convergence status for a single sweep point.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConvergenceStatus {
    /// Point is still iterating (not yet converged).
    Active,
    /// Point has converged successfully.
    Converged,
    /// Point failed to converge within iteration limit.
    Failed,
    /// Point was singular (matrix not invertible).
    Singular,
}

impl ConvergenceStatus {
    /// Returns true if this point needs more iterations.
    #[inline]
    pub fn is_active(&self) -> bool {
        matches!(self, ConvergenceStatus::Active)
    }

    /// Returns true if this point has finished (converged, failed, or singular).
    #[inline]
    pub fn is_finished(&self) -> bool {
        !self.is_active()
    }

    /// Returns true if this point converged successfully.
    #[inline]
    pub fn is_converged(&self) -> bool {
        matches!(self, ConvergenceStatus::Converged)
    }
}

/// Tracks convergence status for a batch of sweep points.
///
/// This is the main type for managing early termination in batched NR solves.
#[derive(Debug, Clone)]
pub struct ConvergenceTracker {
    /// Status of each point.
    status: Vec<ConvergenceStatus>,
    /// Iteration count for each point.
    iterations: Vec<u32>,
    /// Maximum allowed iterations.
    max_iterations: u32,
    /// Number of currently active (unconverged) points.
    active_count: usize,
}

impl ConvergenceTracker {
    /// Create a new convergence tracker for the given batch size.
    ///
    /// All points start as `Active` with zero iterations.
    pub fn new(batch_size: usize) -> Self {
        Self::with_max_iterations(batch_size, 50)
    }

    /// Create a new convergence tracker with custom iteration limit.
    pub fn with_max_iterations(batch_size: usize, max_iterations: u32) -> Self {
        Self {
            status: vec![ConvergenceStatus::Active; batch_size],
            iterations: vec![0; batch_size],
            max_iterations,
            active_count: batch_size,
        }
    }

    /// Total number of points in the batch.
    #[inline]
    pub fn batch_size(&self) -> usize {
        self.status.len()
    }

    /// Number of currently active (unconverged) points.
    #[inline]
    pub fn active_count(&self) -> usize {
        self.active_count
    }

    /// Number of converged points.
    pub fn converged_count(&self) -> usize {
        self.status
            .iter()
            .filter(|s| s.is_converged())
            .count()
    }

    /// Number of failed points.
    pub fn failed_count(&self) -> usize {
        self.status
            .iter()
            .filter(|s| matches!(s, ConvergenceStatus::Failed | ConvergenceStatus::Singular))
            .count()
    }

    /// Check if all points have finished (none active).
    #[inline]
    pub fn all_finished(&self) -> bool {
        self.active_count == 0
    }

    /// Get the status of a specific point.
    #[inline]
    pub fn status(&self, index: usize) -> ConvergenceStatus {
        self.status[index]
    }

    /// Get the iteration count for a specific point.
    #[inline]
    pub fn iterations(&self, index: usize) -> u32 {
        self.iterations[index]
    }

    /// Mark a point as converged.
    ///
    /// Returns true if the point was previously active.
    pub fn mark_converged(&mut self, index: usize) -> bool {
        if self.status[index].is_active() {
            self.status[index] = ConvergenceStatus::Converged;
            self.active_count -= 1;
            true
        } else {
            false
        }
    }

    /// Mark a point as failed (exceeded iteration limit).
    ///
    /// Returns true if the point was previously active.
    pub fn mark_failed(&mut self, index: usize) -> bool {
        if self.status[index].is_active() {
            self.status[index] = ConvergenceStatus::Failed;
            self.active_count -= 1;
            true
        } else {
            false
        }
    }

    /// Mark a point as singular (matrix not invertible).
    ///
    /// Returns true if the point was previously active.
    pub fn mark_singular(&mut self, index: usize) -> bool {
        if self.status[index].is_active() {
            self.status[index] = ConvergenceStatus::Singular;
            self.active_count -= 1;
            true
        } else {
            false
        }
    }

    /// Increment iteration count for a point.
    ///
    /// If the point exceeds max_iterations, it's marked as failed.
    /// Returns the new iteration count.
    pub fn increment_iteration(&mut self, index: usize) -> u32 {
        self.iterations[index] += 1;
        if self.iterations[index] >= self.max_iterations && self.status[index].is_active() {
            self.mark_failed(index);
        }
        self.iterations[index]
    }

    /// Increment iteration count for all active points.
    ///
    /// Points that exceed max_iterations are marked as failed.
    pub fn increment_all_active(&mut self) {
        for i in 0..self.status.len() {
            if self.status[i].is_active() {
                self.increment_iteration(i);
            }
        }
    }

    /// Get indices of all active (unconverged) points.
    ///
    /// This is useful for the masking strategy where you process only
    /// active points but keep the full batch structure.
    pub fn active_indices(&self) -> Vec<usize> {
        self.status
            .iter()
            .enumerate()
            .filter(|(_, s)| s.is_active())
            .map(|(i, _)| i)
            .collect()
    }

    /// Get indices of all converged points.
    pub fn converged_indices(&self) -> Vec<usize> {
        self.status
            .iter()
            .enumerate()
            .filter(|(_, s)| s.is_converged())
            .map(|(i, _)| i)
            .collect()
    }

    /// Get indices of all failed points.
    pub fn failed_indices(&self) -> Vec<usize> {
        self.status
            .iter()
            .enumerate()
            .filter(|(_, s)| matches!(s, ConvergenceStatus::Failed | ConvergenceStatus::Singular))
            .map(|(i, _)| i)
            .collect()
    }

    /// Create a boolean mask where true = active.
    ///
    /// Useful for GPU masking operations.
    pub fn active_mask(&self) -> Vec<bool> {
        self.status.iter().map(|s| s.is_active()).collect()
    }

    /// Create a u32 mask where 1 = active, 0 = inactive.
    ///
    /// Useful for GPU shader operations.
    pub fn active_mask_u32(&self) -> Vec<u32> {
        self.status
            .iter()
            .map(|s| if s.is_active() { 1 } else { 0 })
            .collect()
    }

    /// Check convergence for multiple points based on solution change.
    ///
    /// Marks points as converged if their maximum absolute solution change
    /// is below the tolerance.
    ///
    /// # Arguments
    /// * `solution_changes` - Flattened array of solution changes (batch_size * n)
    /// * `n` - System size (number of variables per point)
    /// * `abstol` - Absolute tolerance for convergence
    /// * `reltol` - Relative tolerance for convergence
    /// * `solutions` - Current solutions for relative tolerance check
    ///
    /// # Returns
    /// Number of newly converged points.
    pub fn check_convergence(
        &mut self,
        solution_changes: &[f64],
        n: usize,
        abstol: f64,
        reltol: f64,
        solutions: &[f64],
    ) -> usize {
        let batch_size = self.batch_size();
        assert_eq!(solution_changes.len(), batch_size * n);
        assert_eq!(solutions.len(), batch_size * n);

        let mut newly_converged = 0;

        for i in 0..batch_size {
            if !self.status[i].is_active() {
                continue;
            }

            let offset = i * n;
            let mut converged = true;

            for j in 0..n {
                let delta = solution_changes[offset + j].abs();
                let value = solutions[offset + j].abs();
                let tol = abstol + reltol * value;

                if delta > tol {
                    converged = false;
                    break;
                }
            }

            if converged {
                self.mark_converged(i);
                newly_converged += 1;
            }
        }

        newly_converged
    }

    /// Check convergence for multiple points based on residual norm.
    ///
    /// Marks points as converged if their residual norm is below tolerance.
    ///
    /// # Arguments
    /// * `residuals` - Flattened array of residuals (batch_size * n)
    /// * `n` - System size (number of variables per point)
    /// * `tolerance` - Residual tolerance for convergence
    ///
    /// # Returns
    /// Number of newly converged points.
    pub fn check_residual_convergence(
        &mut self,
        residuals: &[f64],
        n: usize,
        tolerance: f64,
    ) -> usize {
        let batch_size = self.batch_size();
        assert_eq!(residuals.len(), batch_size * n);

        let mut newly_converged = 0;

        for i in 0..batch_size {
            if !self.status[i].is_active() {
                continue;
            }

            let offset = i * n;
            let mut norm_sq = 0.0;

            for j in 0..n {
                norm_sq += residuals[offset + j].powi(2);
            }

            if norm_sq.sqrt() < tolerance {
                self.mark_converged(i);
                newly_converged += 1;
            }
        }

        newly_converged
    }

    /// Get summary statistics about convergence.
    pub fn summary(&self) -> ConvergenceSummary {
        let mut total_iterations = 0u64;
        let mut max_iter = 0u32;

        for (i, &status) in self.status.iter().enumerate() {
            if status.is_converged() {
                total_iterations += self.iterations[i] as u64;
                max_iter = max_iter.max(self.iterations[i]);
            }
        }

        let converged = self.converged_count();
        let avg_iterations = if converged > 0 {
            total_iterations as f64 / converged as f64
        } else {
            0.0
        };

        ConvergenceSummary {
            total_points: self.batch_size(),
            converged_count: converged,
            failed_count: self.failed_count(),
            active_count: self.active_count,
            average_iterations: avg_iterations,
            max_iterations: max_iter,
        }
    }
}

/// Summary statistics about convergence.
#[derive(Debug, Clone)]
pub struct ConvergenceSummary {
    /// Total number of points in the batch.
    pub total_points: usize,
    /// Number of points that converged successfully.
    pub converged_count: usize,
    /// Number of points that failed to converge.
    pub failed_count: usize,
    /// Number of points still active.
    pub active_count: usize,
    /// Average iterations for converged points.
    pub average_iterations: f64,
    /// Maximum iterations among converged points.
    pub max_iterations: u32,
}

impl std::fmt::Display for ConvergenceSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Converged: {}/{} ({:.1}%), Failed: {}, Avg iterations: {:.1}",
            self.converged_count,
            self.total_points,
            100.0 * self.converged_count as f64 / self.total_points as f64,
            self.failed_count,
            self.average_iterations
        )
    }
}

/// Compact active points into a contiguous buffer.
///
/// This is useful for the compaction strategy where only active points
/// are processed in subsequent iterations.
///
/// # Arguments
/// * `data` - Flattened data array (batch_size * item_size)
/// * `item_size` - Size of each item (e.g., system_size for solutions)
/// * `active_indices` - Indices of active points
///
/// # Returns
/// Compacted data containing only active points.
pub fn compact_active<T: Copy>(data: &[T], item_size: usize, active_indices: &[usize]) -> Vec<T> {
    let mut result = Vec::with_capacity(active_indices.len() * item_size);

    for &idx in active_indices {
        let offset = idx * item_size;
        result.extend_from_slice(&data[offset..offset + item_size]);
    }

    result
}

/// Expand compacted results back to full batch size.
///
/// # Arguments
/// * `compacted` - Compacted data array (active_count * item_size)
/// * `item_size` - Size of each item
/// * `active_indices` - Indices where active data should be placed
/// * `batch_size` - Total batch size
/// * `default` - Default value for inactive positions
///
/// # Returns
/// Full-size array with active results placed at their original indices.
pub fn expand_active<T: Copy>(
    compacted: &[T],
    item_size: usize,
    active_indices: &[usize],
    batch_size: usize,
    default: T,
) -> Vec<T> {
    let mut result = vec![default; batch_size * item_size];

    for (compact_idx, &original_idx) in active_indices.iter().enumerate() {
        let src_offset = compact_idx * item_size;
        let dst_offset = original_idx * item_size;
        result[dst_offset..dst_offset + item_size]
            .copy_from_slice(&compacted[src_offset..src_offset + item_size]);
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tracker_basic() {
        let tracker = ConvergenceTracker::new(100);

        assert_eq!(tracker.batch_size(), 100);
        assert_eq!(tracker.active_count(), 100);
        assert_eq!(tracker.converged_count(), 0);
        assert!(!tracker.all_finished());
    }

    #[test]
    fn test_mark_converged() {
        let mut tracker = ConvergenceTracker::new(10);

        assert!(tracker.mark_converged(0));
        assert!(tracker.mark_converged(5));
        assert!(tracker.mark_converged(9));

        assert_eq!(tracker.active_count(), 7);
        assert_eq!(tracker.converged_count(), 3);
        assert_eq!(tracker.status(0), ConvergenceStatus::Converged);
        assert_eq!(tracker.status(1), ConvergenceStatus::Active);

        // Can't mark already converged point
        assert!(!tracker.mark_converged(0));
    }

    #[test]
    fn test_mark_failed() {
        let mut tracker = ConvergenceTracker::new(10);

        tracker.mark_failed(3);
        tracker.mark_singular(7);

        assert_eq!(tracker.active_count(), 8);
        assert_eq!(tracker.failed_count(), 2);
        assert_eq!(tracker.status(3), ConvergenceStatus::Failed);
        assert_eq!(tracker.status(7), ConvergenceStatus::Singular);
    }

    #[test]
    fn test_iteration_limit() {
        let mut tracker = ConvergenceTracker::with_max_iterations(5, 3);

        for _ in 0..3 {
            tracker.increment_all_active();
        }

        // All points should have exceeded limit and be marked failed
        assert!(tracker.all_finished());
        assert_eq!(tracker.failed_count(), 5);
    }

    #[test]
    fn test_active_indices() {
        let mut tracker = ConvergenceTracker::new(5);

        tracker.mark_converged(1);
        tracker.mark_converged(3);

        let active = tracker.active_indices();
        assert_eq!(active, vec![0, 2, 4]);
    }

    #[test]
    fn test_active_mask() {
        let mut tracker = ConvergenceTracker::new(5);

        tracker.mark_converged(1);
        tracker.mark_failed(3);

        let mask = tracker.active_mask();
        assert_eq!(mask, vec![true, false, true, false, true]);

        let mask_u32 = tracker.active_mask_u32();
        assert_eq!(mask_u32, vec![1, 0, 1, 0, 1]);
    }

    #[test]
    fn test_check_convergence() {
        let mut tracker = ConvergenceTracker::new(3);
        let n = 2;

        // Three 2-element solutions
        let solutions = vec![1.0, 2.0, 10.0, 20.0, 100.0, 200.0];

        // Small changes for points 0 and 2, large change for point 1
        let changes = vec![
            1e-8, 1e-8, // Point 0: small change
            1.0, 1.0, // Point 1: large change
            1e-9, 1e-9, // Point 2: tiny change
        ];

        let newly_converged = tracker.check_convergence(&changes, n, 1e-6, 1e-6, &solutions);

        assert_eq!(newly_converged, 2);
        assert_eq!(tracker.status(0), ConvergenceStatus::Converged);
        assert_eq!(tracker.status(1), ConvergenceStatus::Active);
        assert_eq!(tracker.status(2), ConvergenceStatus::Converged);
    }

    #[test]
    fn test_check_residual_convergence() {
        let mut tracker = ConvergenceTracker::new(3);
        let n = 2;

        // Residuals: point 0 small, point 1 large, point 2 small
        let residuals = vec![
            1e-8, 1e-8, // Point 0: ||r|| ≈ 1.4e-8
            1.0, 1.0, // Point 1: ||r|| ≈ 1.4
            1e-9, 1e-9, // Point 2: ||r|| ≈ 1.4e-9
        ];

        let newly_converged = tracker.check_residual_convergence(&residuals, n, 1e-6);

        assert_eq!(newly_converged, 2);
        assert!(tracker.status(0).is_converged());
        assert!(tracker.status(1).is_active());
        assert!(tracker.status(2).is_converged());
    }

    #[test]
    fn test_compact_active() {
        let data = vec![
            1.0, 2.0, // Point 0
            3.0, 4.0, // Point 1
            5.0, 6.0, // Point 2
            7.0, 8.0, // Point 3
        ];

        let active_indices = vec![0, 2]; // Points 0 and 2 are active
        let compacted = compact_active(&data, 2, &active_indices);

        assert_eq!(compacted, vec![1.0, 2.0, 5.0, 6.0]);
    }

    #[test]
    fn test_expand_active() {
        let compacted = vec![1.0, 2.0, 5.0, 6.0]; // Two active points
        let active_indices = vec![0, 2];
        let batch_size = 4;

        let expanded = expand_active(&compacted, 2, &active_indices, batch_size, 0.0);

        assert_eq!(
            expanded,
            vec![
                1.0, 2.0, // Point 0 (active)
                0.0, 0.0, // Point 1 (inactive, default)
                5.0, 6.0, // Point 2 (active)
                0.0, 0.0, // Point 3 (inactive, default)
            ]
        );
    }

    #[test]
    fn test_summary() {
        let mut tracker = ConvergenceTracker::new(100);

        // Simulate some iterations
        for i in 0..50 {
            for _ in 0..3 {
                tracker.increment_iteration(i);
            }
            tracker.mark_converged(i);
        }

        for i in 50..60 {
            for _ in 0..10 {
                tracker.increment_iteration(i);
            }
            tracker.mark_converged(i);
        }

        for i in 60..65 {
            tracker.mark_failed(i);
        }

        let summary = tracker.summary();

        assert_eq!(summary.total_points, 100);
        assert_eq!(summary.converged_count, 60);
        assert_eq!(summary.failed_count, 5);
        assert_eq!(summary.active_count, 35);
        assert!(summary.average_iterations > 0.0);
        assert_eq!(summary.max_iterations, 10);

        // Test Display
        let s = format!("{}", summary);
        assert!(s.contains("60/100"));
    }
}
