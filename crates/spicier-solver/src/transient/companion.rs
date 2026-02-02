//! Companion models for reactive elements in transient analysis.

use nalgebra::DVector;
use spicier_core::mna::MnaSystem;

use super::types::{IntegrationMethod, TRBDF2_GAMMA};

/// State of a capacitor for companion model.
#[derive(Debug, Clone)]
pub struct CapacitorState {
    /// Capacitance (F).
    pub capacitance: f64,
    /// Voltage at previous timestep.
    pub v_prev: f64,
    /// Current at previous timestep (for trapezoidal).
    pub i_prev: f64,
    /// Voltage at two timesteps ago (for TR-BDF2).
    pub v_prev_prev: f64,
    /// Positive node MNA index (None for ground).
    pub node_pos: Option<usize>,
    /// Negative node MNA index (None for ground).
    pub node_neg: Option<usize>,
}

impl CapacitorState {
    /// Create a new capacitor state.
    pub fn new(capacitance: f64, node_pos: Option<usize>, node_neg: Option<usize>) -> Self {
        Self {
            capacitance,
            v_prev: 0.0,
            i_prev: 0.0,
            v_prev_prev: 0.0,
            node_pos,
            node_neg,
        }
    }

    /// Stamp the companion model for Backward Euler.
    ///
    /// C is replaced by: G_eq = C/h in parallel with I_eq = C/h * V_prev
    pub fn stamp_be(&self, mna: &mut MnaSystem, h: f64) {
        let geq = self.capacitance / h;
        let ieq = geq * self.v_prev;

        mna.stamp_conductance(self.node_pos, self.node_neg, geq);
        // Current source: ieq flows from neg to pos (charging)
        mna.stamp_current_source(self.node_neg, self.node_pos, ieq);
    }

    /// Stamp the companion model for Trapezoidal rule.
    ///
    /// C is replaced by: G_eq = 2C/h in parallel with I_eq = 2C/h * V_prev + I_prev
    pub fn stamp_trap(&self, mna: &mut MnaSystem, h: f64) {
        let geq = 2.0 * self.capacitance / h;
        let ieq = geq * self.v_prev + self.i_prev;

        mna.stamp_conductance(self.node_pos, self.node_neg, geq);
        mna.stamp_current_source(self.node_neg, self.node_pos, ieq);
    }

    /// Update state after solving a timestep.
    pub fn update(&mut self, v_new: f64, h: f64, method: IntegrationMethod) {
        match method {
            IntegrationMethod::BackwardEuler => {
                self.i_prev = self.capacitance / h * (v_new - self.v_prev);
            }
            IntegrationMethod::Trapezoidal => {
                self.i_prev = 2.0 * self.capacitance / h * (v_new - self.v_prev) - self.i_prev;
            }
            IntegrationMethod::TrBdf2 => {
                // TR-BDF2 update after full step completion
                // Current is computed from the BDF2 formula
                let gamma = TRBDF2_GAMMA;
                let alpha = (1.0 - gamma) / (gamma * (2.0 - gamma));
                self.i_prev = self.capacitance / h
                    * ((1.0 + alpha) * v_new - (1.0 + 2.0 * alpha) * self.v_prev
                        + alpha * self.v_prev_prev);
            }
        }
        self.v_prev_prev = self.v_prev;
        self.v_prev = v_new;
    }

    /// Update state after TR-BDF2 intermediate (Trapezoidal) step.
    ///
    /// Called after the first stage of TR-BDF2 with the intermediate voltage.
    pub fn update_trbdf2_intermediate(&mut self, v_gamma: f64, h: f64) {
        let gamma = TRBDF2_GAMMA;
        let h_gamma = gamma * h;
        // Store current v_prev as v_prev_prev for BDF2 stage
        self.v_prev_prev = self.v_prev;
        // Update i_prev using trapezoidal current
        self.i_prev = 2.0 * self.capacitance / h_gamma * (v_gamma - self.v_prev_prev) - self.i_prev;
        // Update v_prev to intermediate value
        self.v_prev = v_gamma;
    }

    /// Stamp companion model for TR-BDF2 BDF2 stage.
    ///
    /// Uses v_prev (at γ*h) and v_prev_prev (at 0) for BDF2 formula.
    pub fn stamp_trbdf2_bdf2(&self, mna: &mut MnaSystem, h: f64) {
        let gamma = TRBDF2_GAMMA;
        // BDF2 coefficients for non-uniform step: h1 = γ*h, h2 = (1-γ)*h
        // The step we're taking is h2 = (1-γ)*h
        let h2 = (1.0 - gamma) * h;
        let h1 = gamma * h;
        let rho = h2 / h1; // ratio of step sizes

        // BDF2 for non-uniform steps:
        // y_{n+1} = a1 * y_n + a2 * y_{n-1} + b0 * h2 * y'_{n+1}
        // where:
        //   a1 = (1+ρ)² / (1+2ρ)
        //   a2 = -ρ² / (1+2ρ)
        //   b0 = (1+ρ) / (1+2ρ)
        let denom = 1.0 + 2.0 * rho;
        let a1 = (1.0 + rho).powi(2) / denom;
        let a2 = -rho * rho / denom;
        let b0 = (1.0 + rho) / denom;

        // For capacitor: i = C * dv/dt
        // Geq = C / (b0 * h2)
        let geq = self.capacitance / (b0 * h2);
        // Ieq represents the history terms: current = Geq * (a1*v_n + a2*v_{n-1})
        let ieq = geq * (a1 * self.v_prev + a2 * self.v_prev_prev);

        mna.stamp_conductance(self.node_pos, self.node_neg, geq);
        mna.stamp_current_source(self.node_neg, self.node_pos, ieq);
    }

    /// Estimate Local Truncation Error for the capacitor voltage.
    ///
    /// Uses the difference between Trapezoidal and Backward Euler predictions.
    /// For a capacitor: LTE ≈ h²/12 * C * d²v/dt²
    ///
    /// This method computes the LTE estimate using the "Milne device":
    /// LTE ≈ |v_trap - v_be| / 3
    pub fn estimate_lte(&self, v_new: f64, h: f64) -> f64 {
        // Current computed by Trapezoidal
        let i_trap = 2.0 * self.capacitance / h * (v_new - self.v_prev) - self.i_prev;

        // Current computed by Backward Euler
        let i_be = self.capacitance / h * (v_new - self.v_prev);

        // The difference gives an error estimate
        // For Trapezoidal with Milne device: LTE ≈ |i_trap - i_be| / 3
        // This estimates the error in the capacitor current
        (i_trap - i_be).abs() / 3.0
    }

    /// Get voltage across capacitor from solution vector.
    pub fn voltage_from_solution(&self, solution: &DVector<f64>) -> f64 {
        let vp = self.node_pos.map(|i| solution[i]).unwrap_or(0.0);
        let vn = self.node_neg.map(|i| solution[i]).unwrap_or(0.0);
        vp - vn
    }
}

/// State of an inductor for companion model.
#[derive(Debug, Clone)]
pub struct InductorState {
    /// Inductance (H).
    pub inductance: f64,
    /// Current at previous timestep.
    pub i_prev: f64,
    /// Voltage at previous timestep (for trapezoidal).
    pub v_prev: f64,
    /// Current at two timesteps ago (for TR-BDF2).
    pub i_prev_prev: f64,
    /// Positive node MNA index (None for ground).
    pub node_pos: Option<usize>,
    /// Negative node MNA index (None for ground).
    pub node_neg: Option<usize>,
}

impl InductorState {
    /// Create a new inductor state.
    pub fn new(inductance: f64, node_pos: Option<usize>, node_neg: Option<usize>) -> Self {
        Self {
            inductance,
            i_prev: 0.0,
            v_prev: 0.0,
            i_prev_prev: 0.0,
            node_pos,
            node_neg,
        }
    }

    /// Stamp the companion model for Backward Euler.
    ///
    /// L is replaced by: G_eq = h/L in parallel with I_eq = I_prev
    /// The inductor current flows from node_pos to node_neg.
    pub fn stamp_be(&self, mna: &mut MnaSystem, h: f64) {
        let geq = h / self.inductance;
        let ieq = self.i_prev;

        mna.stamp_conductance(self.node_pos, self.node_neg, geq);
        // Current source ieq flows from node_pos to node_neg (same direction as i_prev)
        mna.stamp_current_source(self.node_pos, self.node_neg, ieq);
    }

    /// Stamp the companion model for Trapezoidal rule.
    ///
    /// L is replaced by: G_eq = h/(2L) in parallel with I_eq = I_prev + h/(2L) * V_prev
    /// The inductor current flows from node_pos to node_neg.
    pub fn stamp_trap(&self, mna: &mut MnaSystem, h: f64) {
        let geq = h / (2.0 * self.inductance);
        let ieq = self.i_prev + geq * self.v_prev;

        mna.stamp_conductance(self.node_pos, self.node_neg, geq);
        // Current source ieq flows from node_pos to node_neg (same direction as i_prev)
        mna.stamp_current_source(self.node_pos, self.node_neg, ieq);
    }

    /// Update state after solving a timestep.
    pub fn update(&mut self, v_new: f64, h: f64, method: IntegrationMethod) {
        match method {
            IntegrationMethod::BackwardEuler => {
                self.i_prev += h / self.inductance * v_new;
            }
            IntegrationMethod::Trapezoidal => {
                self.i_prev += h / (2.0 * self.inductance) * (v_new + self.v_prev);
            }
            IntegrationMethod::TrBdf2 => {
                // TR-BDF2 update after full step completion
                let gamma = TRBDF2_GAMMA;
                let alpha = (1.0 - gamma) / (gamma * (2.0 - gamma));
                let di = h / self.inductance
                    * ((1.0 + alpha) * v_new - (1.0 + 2.0 * alpha) * self.v_prev
                        + alpha * self.v_prev);
                self.i_prev += di;
            }
        }
        self.i_prev_prev = self.i_prev;
        self.v_prev = v_new;
    }

    /// Update state after TR-BDF2 intermediate (Trapezoidal) step.
    pub fn update_trbdf2_intermediate(&mut self, v_gamma: f64, h: f64) {
        let gamma = TRBDF2_GAMMA;
        let h_gamma = gamma * h;
        // Save current i_prev for BDF2 stage
        self.i_prev_prev = self.i_prev;
        // Trapezoidal update for intermediate step
        self.i_prev += h_gamma / (2.0 * self.inductance) * (v_gamma + self.v_prev);
        // v_prev updated to intermediate value (don't update yet, done in main loop)
    }

    /// Stamp companion model for TR-BDF2 BDF2 stage.
    pub fn stamp_trbdf2_bdf2(&self, mna: &mut MnaSystem, h: f64) {
        let gamma = TRBDF2_GAMMA;
        let h2 = (1.0 - gamma) * h;
        let h1 = gamma * h;
        let rho = h2 / h1;

        // BDF2 coefficients for non-uniform steps
        // i_{n+1} = a1 * i_n + a2 * i_{n-1} + b0 * h2 / L * v_{n+1}
        let denom = 1.0 + 2.0 * rho;
        let a1 = (1.0 + rho).powi(2) / denom;
        let a2 = -rho * rho / denom;
        let b0 = (1.0 + rho) / denom;

        // For inductor: L * di/dt = v
        // Geq = b0 * h2 / L (conductance seen by the circuit)
        let geq = b0 * h2 / self.inductance;
        // Ieq represents the history terms
        let ieq = a1 * self.i_prev + a2 * self.i_prev_prev;

        mna.stamp_conductance(self.node_pos, self.node_neg, geq);
        // Current source ieq flows from node_pos to node_neg (same direction as i_prev)
        mna.stamp_current_source(self.node_pos, self.node_neg, ieq);
    }

    /// Estimate Local Truncation Error for the inductor current.
    ///
    /// Uses the difference between Trapezoidal and Backward Euler predictions.
    /// For an inductor: LTE ≈ h²/12 * L * d²i/dt²
    pub fn estimate_lte(&self, v_new: f64, h: f64) -> f64 {
        // Current increment by Trapezoidal
        let di_trap = h / (2.0 * self.inductance) * (v_new + self.v_prev);

        // Current increment by Backward Euler
        let di_be = h / self.inductance * v_new;

        // The difference gives an error estimate
        (di_trap - di_be).abs() / 3.0
    }

    /// Get voltage across inductor from solution vector.
    pub fn voltage_from_solution(&self, solution: &DVector<f64>) -> f64 {
        let vp = self.node_pos.map(|i| solution[i]).unwrap_or(0.0);
        let vn = self.node_neg.map(|i| solution[i]).unwrap_or(0.0);
        vp - vn
    }
}
