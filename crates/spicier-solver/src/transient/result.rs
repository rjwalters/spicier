//! Result types for transient analysis.

use nalgebra::DVector;

/// A single timepoint in a transient simulation result.
#[derive(Debug, Clone)]
pub struct TimePoint {
    /// Time value (s).
    pub time: f64,
    /// Solution vector at this time.
    pub solution: DVector<f64>,
}

/// Result of a transient simulation.
#[derive(Debug, Clone)]
pub struct TransientResult {
    /// All computed timepoints.
    pub points: Vec<TimePoint>,
    /// Number of nodes (excluding ground).
    pub num_nodes: usize,
}

impl TransientResult {
    /// Get the voltage at a node across all timepoints.
    pub fn voltage_waveform(&self, node_idx: usize) -> Vec<(f64, f64)> {
        self.points
            .iter()
            .map(|tp| (tp.time, tp.solution[node_idx]))
            .collect()
    }

    /// Get all time values.
    pub fn times(&self) -> Vec<f64> {
        self.points.iter().map(|tp| tp.time).collect()
    }

    /// Interpolate the solution at a specific time.
    ///
    /// Uses linear interpolation between the two nearest timepoints.
    /// Returns None if time is outside the simulation range.
    pub fn interpolate_at(&self, time: f64) -> Option<DVector<f64>> {
        if self.points.is_empty() {
            return None;
        }

        // Handle boundary cases
        if time <= self.points[0].time {
            return Some(self.points[0].solution.clone());
        }
        if time >= self.points.last()?.time {
            return Some(self.points.last()?.solution.clone());
        }

        // Find the interval containing time
        for i in 0..self.points.len() - 1 {
            let t0 = self.points[i].time;
            let t1 = self.points[i + 1].time;

            if time >= t0 && time <= t1 {
                // Linear interpolation
                let alpha = (time - t0) / (t1 - t0);
                let v0 = &self.points[i].solution;
                let v1 = &self.points[i + 1].solution;
                return Some(v0 * (1.0 - alpha) + v1 * alpha);
            }
        }

        None
    }

    /// Sample the waveform at evenly-spaced times.
    ///
    /// Returns a new TransientResult with timepoints at regular intervals.
    ///
    /// # Arguments
    /// * `tstep` - Time step between samples
    /// * `tstart` - Start time (default 0.0)
    /// * `tstop` - Stop time (uses simulation end time if None)
    pub fn sample_at_times(
        &self,
        tstep: f64,
        tstart: Option<f64>,
        tstop: Option<f64>,
    ) -> TransientResult {
        let tstart = tstart.unwrap_or(0.0);
        let tstop = tstop.unwrap_or_else(|| self.points.last().map(|p| p.time).unwrap_or(0.0));

        let mut sampled_points = Vec::new();
        let mut t = tstart;

        while t <= tstop + tstep * 0.001 {
            if let Some(solution) = self.interpolate_at(t) {
                sampled_points.push(TimePoint { time: t, solution });
            }
            t += tstep;
        }

        TransientResult {
            points: sampled_points,
            num_nodes: self.num_nodes,
        }
    }

    /// Get the voltage at a node at a specific time (interpolated).
    pub fn voltage_at(&self, node_idx: usize, time: f64) -> Option<f64> {
        self.interpolate_at(time).map(|sol| sol[node_idx])
    }
}

/// Result of adaptive transient simulation with statistics.
#[derive(Debug, Clone)]
pub struct AdaptiveTransientResult {
    /// All computed timepoints.
    pub points: Vec<TimePoint>,
    /// Number of nodes (excluding ground).
    pub num_nodes: usize,
    /// Total number of timesteps taken.
    pub total_steps: usize,
    /// Number of rejected timesteps.
    pub rejected_steps: usize,
    /// Minimum timestep used.
    pub min_step_used: f64,
    /// Maximum timestep used.
    pub max_step_used: f64,
}

impl AdaptiveTransientResult {
    /// Get the voltage at a node across all timepoints.
    pub fn voltage_waveform(&self, node_idx: usize) -> Vec<(f64, f64)> {
        self.points
            .iter()
            .map(|tp| (tp.time, tp.solution[node_idx]))
            .collect()
    }

    /// Get all time values.
    pub fn times(&self) -> Vec<f64> {
        self.points.iter().map(|tp| tp.time).collect()
    }

    /// Interpolate the solution at a specific time.
    ///
    /// Uses linear interpolation between the two nearest timepoints.
    /// Returns None if time is outside the simulation range.
    pub fn interpolate_at(&self, time: f64) -> Option<DVector<f64>> {
        if self.points.is_empty() {
            return None;
        }

        // Handle boundary cases
        if time <= self.points[0].time {
            return Some(self.points[0].solution.clone());
        }
        if time >= self.points.last()?.time {
            return Some(self.points.last()?.solution.clone());
        }

        // Find the interval containing time
        for i in 0..self.points.len() - 1 {
            let t0 = self.points[i].time;
            let t1 = self.points[i + 1].time;

            if time >= t0 && time <= t1 {
                // Linear interpolation
                let alpha = (time - t0) / (t1 - t0);
                let v0 = &self.points[i].solution;
                let v1 = &self.points[i + 1].solution;
                return Some(v0 * (1.0 - alpha) + v1 * alpha);
            }
        }

        None
    }

    /// Sample the waveform at evenly-spaced times.
    ///
    /// Returns a new TransientResult with timepoints at regular intervals.
    /// This is useful for producing output at uniform time steps from an
    /// adaptive simulation that used variable step sizes.
    ///
    /// # Arguments
    /// * `tstep` - Time step between samples
    /// * `tstart` - Start time (default 0.0)
    /// * `tstop` - Stop time (uses simulation end time if None)
    pub fn sample_at_times(
        &self,
        tstep: f64,
        tstart: Option<f64>,
        tstop: Option<f64>,
    ) -> TransientResult {
        let tstart = tstart.unwrap_or(0.0);
        let tstop = tstop.unwrap_or_else(|| self.points.last().map(|p| p.time).unwrap_or(0.0));

        let mut sampled_points = Vec::new();
        let mut t = tstart;

        while t <= tstop + tstep * 0.001 {
            if let Some(solution) = self.interpolate_at(t) {
                sampled_points.push(TimePoint { time: t, solution });
            }
            t += tstep;
        }

        TransientResult {
            points: sampled_points,
            num_nodes: self.num_nodes,
        }
    }

    /// Get the voltage at a node at a specific time (interpolated).
    pub fn voltage_at(&self, node_idx: usize, time: f64) -> Option<f64> {
        self.interpolate_at(time).map(|sol| sol[node_idx])
    }
}
