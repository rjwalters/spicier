//! Type definitions for transient analysis.

use std::collections::HashMap;

use nalgebra::DVector;

/// TR-BDF2 gamma parameter: γ = 2 - √2 ≈ 0.5858 for the fraction of step using Trapezoidal.
///
/// The method takes a Trapezoidal step of size γ*h, then a BDF2 step of size (1-γ)*h.
/// This value maximizes order of accuracy while maintaining L-stability.
pub const TRBDF2_GAMMA: f64 = 2.0 - std::f64::consts::SQRT_2;

/// Initial conditions for transient analysis.
///
/// Stores node name -> voltage mappings from .IC commands.
#[derive(Debug, Clone, Default)]
pub struct InitialConditions {
    /// Node voltages keyed by node name.
    pub voltages: HashMap<String, f64>,
}

impl InitialConditions {
    /// Create an empty initial conditions set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a node voltage initial condition.
    pub fn set_voltage(&mut self, node: &str, voltage: f64) {
        self.voltages.insert(node.to_string(), voltage);
    }

    /// Apply initial conditions to a solution vector.
    ///
    /// The `node_map` maps node names to their MNA matrix indices.
    pub fn apply(&self, solution: &mut DVector<f64>, node_map: &HashMap<String, usize>) {
        for (node, &voltage) in &self.voltages {
            if let Some(&idx) = node_map.get(node) {
                solution[idx] = voltage;
            }
        }
    }

    /// Check if any initial conditions are set.
    pub fn is_empty(&self) -> bool {
        self.voltages.is_empty()
    }
}

/// Integration method for transient analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntegrationMethod {
    /// Backward Euler (first order, A-stable).
    BackwardEuler,
    /// Trapezoidal (second order, A-stable).
    Trapezoidal,
    /// TR-BDF2 (second order, L-stable, good for stiff circuits).
    ///
    /// A composite method that uses Trapezoidal for γ*h (γ ≈ 0.2929),
    /// then BDF2 for the remaining (1-γ)*h. Provides L-stability
    /// without the numerical ringing issues of pure Trapezoidal.
    TrBdf2,
}

/// Transient analysis parameters.
#[derive(Debug, Clone)]
pub struct TransientParams {
    /// Stop time (s).
    pub tstop: f64,
    /// Maximum timestep (s).
    pub tstep: f64,
    /// Integration method.
    pub method: IntegrationMethod,
}

/// Parameters for adaptive timestep control.
#[derive(Debug, Clone)]
pub struct AdaptiveTransientParams {
    /// Stop time (s).
    pub tstop: f64,
    /// Initial timestep (s).
    pub h_init: f64,
    /// Minimum timestep (s).
    pub h_min: f64,
    /// Maximum timestep (s).
    pub h_max: f64,
    /// Relative tolerance for LTE.
    pub reltol: f64,
    /// Absolute tolerance for LTE.
    pub abstol: f64,
    /// Integration method.
    pub method: IntegrationMethod,
}

impl Default for AdaptiveTransientParams {
    fn default() -> Self {
        Self {
            tstop: 1e-3,
            h_init: 1e-9,
            h_min: 1e-15,
            h_max: 1e-6,
            reltol: 1e-3,
            abstol: 1e-6,
            method: IntegrationMethod::Trapezoidal,
        }
    }
}

impl AdaptiveTransientParams {
    /// Create parameters for a specific stop time with defaults.
    pub fn for_tstop(tstop: f64) -> Self {
        Self {
            tstop,
            h_max: tstop / 100.0, // At least 100 points
            ..Default::default()
        }
    }
}
