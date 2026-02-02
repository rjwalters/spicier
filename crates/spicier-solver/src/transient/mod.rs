//! Transient analysis engine.
//!
//! This module provides time-domain simulation for circuits with reactive elements
//! (capacitors and inductors). It supports multiple integration methods:
//!
//! - **Backward Euler**: First-order, A-stable, most robust
//! - **Trapezoidal**: Second-order, A-stable, good for oscillators
//! - **TR-BDF2**: Second-order, L-stable, good for stiff circuits
//!
//! # Module Structure
//!
//! - [`types`] - Configuration types and parameters
//! - [`companion`] - Companion models for capacitors and inductors
//! - [`result`] - Result types with interpolation support
//! - [`solver`] - Main solver functions

pub mod companion;
pub mod result;
pub mod solver;
pub mod types;

// Re-export main types and functions
pub use companion::{CapacitorState, InductorState};
pub use result::{AdaptiveTransientResult, TimePoint, TransientResult};
pub use solver::{
    TransientStamper, solve_transient, solve_transient_adaptive, solve_transient_dispatched,
};
pub use types::{
    AdaptiveTransientParams, InitialConditions, IntegrationMethod, TRBDF2_GAMMA, TransientParams,
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dispatch::DispatchConfig;
    use nalgebra::DVector;
    use spicier_core::mna::MnaSystem;

    #[test]
    fn test_transient_dispatched() {
        // Simple RC circuit with dispatched solver
        struct SimpleRcStamper;
        impl TransientStamper for SimpleRcStamper {
            fn stamp_at_time(&self, mna: &mut MnaSystem, _time: f64) {
                mna.stamp_voltage_source(Some(0), None, 0, 5.0);
                mna.stamp_conductance(Some(0), Some(1), 1.0 / 1000.0);
            }
            fn num_nodes(&self) -> usize {
                2
            }
            fn num_vsources(&self) -> usize {
                1
            }
        }

        let mut caps = vec![CapacitorState::new(1e-6, Some(1), None)];
        let params = TransientParams {
            tstop: 1e-3,
            tstep: 100e-6,
            method: IntegrationMethod::BackwardEuler,
        };
        let dc = DVector::from_vec(vec![5.0, 0.0, -0.005]);
        let config = DispatchConfig::default();

        let result =
            solve_transient_dispatched(&SimpleRcStamper, &mut caps, &mut [], &params, &dc, &config)
                .unwrap();

        // Should have 11 points (0 to 1ms in 100us steps)
        assert_eq!(result.points.len(), 11);
        // Capacitor should be charging
        assert!(result.points.last().unwrap().solution[1] > 0.0);
    }

    /// Simple RC circuit stamper: V1 -- R -- node0 -- C -- GND
    struct RcCircuitStamper {
        voltage: f64,
        resistance: f64,
    }

    impl TransientStamper for RcCircuitStamper {
        fn stamp_at_time(&self, mna: &mut MnaSystem, _time: f64) {
            // Voltage source at node 0, current var index 0
            mna.stamp_voltage_source(Some(0), None, 0, self.voltage);
            // Resistor from node 0 to node 1
            let g = 1.0 / self.resistance;
            mna.stamp_conductance(Some(0), Some(1), g);
        }

        fn num_nodes(&self) -> usize {
            2
        }

        fn num_vsources(&self) -> usize {
            1
        }
    }

    #[test]
    fn test_rc_charging_be() {
        // RC circuit: V1=5V, R=1k, C=1uF
        // Time constant: tau = RC = 1k * 1uF = 1ms
        let stamper = RcCircuitStamper {
            voltage: 5.0,
            resistance: 1000.0,
        };

        let capacitance = 1e-6;
        let mut caps = vec![CapacitorState::new(capacitance, Some(1), None)];

        let params = TransientParams {
            tstop: 5e-3,  // 5 time constants
            tstep: 10e-6, // 10us steps
            method: IntegrationMethod::BackwardEuler,
        };

        let dc = DVector::from_vec(vec![5.0, 0.0, -0.005]); // V(0)=5, V(1)=0, I(V1)=-5mA

        let result = solve_transient(&stamper, &mut caps, &mut [], &params, &dc).unwrap();

        // After 5 tau, capacitor should be nearly charged to 5V
        let final_voltage = result.points.last().unwrap().solution[1];
        assert!(
            (final_voltage - 5.0).abs() < 0.05,
            "Final V(cap) = {} (expected ≈ 5.0)",
            final_voltage
        );

        // At t = tau (1ms), voltage should be ~3.16V (= 5 * (1 - e^-1))
        let tau_step = (1e-3 / params.tstep).round() as usize;
        let v_at_tau = result.points[tau_step].solution[1];
        let expected_v_tau = 5.0 * (1.0 - (-1.0_f64).exp());
        assert!(
            (v_at_tau - expected_v_tau).abs() < 0.2,
            "V(cap) at tau = {} (expected ≈ {})",
            v_at_tau,
            expected_v_tau
        );
    }

    #[test]
    fn test_rc_charging_trapezoidal() {
        let stamper = RcCircuitStamper {
            voltage: 5.0,
            resistance: 1000.0,
        };

        let mut caps = vec![CapacitorState::new(1e-6, Some(1), None)];

        let params = TransientParams {
            tstop: 5e-3,
            tstep: 10e-6,
            method: IntegrationMethod::Trapezoidal,
        };

        let dc = DVector::from_vec(vec![5.0, 0.0, -0.005]);

        let result = solve_transient(&stamper, &mut caps, &mut [], &params, &dc).unwrap();

        let final_voltage = result.points.last().unwrap().solution[1];
        assert!(
            (final_voltage - 5.0).abs() < 0.05,
            "Final V(cap) = {} (expected ≈ 5.0)",
            final_voltage
        );

        // Trapezoidal should be more accurate at tau
        let tau_step = (1e-3 / params.tstep).round() as usize;
        let v_at_tau = result.points[tau_step].solution[1];
        let expected_v_tau = 5.0 * (1.0 - (-1.0_f64).exp());
        assert!(
            (v_at_tau - expected_v_tau).abs() < 0.1,
            "V(cap) at tau = {} (expected ≈ {}) [trapezoidal]",
            v_at_tau,
            expected_v_tau
        );
    }

    #[test]
    fn test_capacitor_companion_be() {
        let cap = CapacitorState {
            capacitance: 1e-6,
            v_prev: 2.5,
            i_prev: 0.0,
            v_prev_prev: 0.0,
            node_pos: Some(0),
            node_neg: None,
        };

        let mut mna = MnaSystem::new(1, 0);
        let h = 1e-6;
        cap.stamp_be(&mut mna, h);
        let matrix = mna.to_dense_matrix();

        // Geq = C/h = 1e-6/1e-6 = 1.0
        assert!(
            (matrix[(0, 0)] - 1.0).abs() < 1e-10,
            "Geq = {} (expected 1.0)",
            matrix[(0, 0)]
        );

        // Ieq = Geq * V_prev = 1.0 * 2.5 = 2.5
        assert!(
            (mna.rhs()[0] - 2.5).abs() < 1e-10,
            "Ieq = {} (expected 2.5)",
            mna.rhs()[0]
        );
    }

    #[test]
    fn test_adaptive_rc_charging() {
        // RC circuit: V1=5V, R=1k, C=1uF, tau=1ms
        let stamper = RcCircuitStamper {
            voltage: 5.0,
            resistance: 1000.0,
        };

        let capacitance = 1e-6;
        let mut caps = vec![CapacitorState::new(capacitance, Some(1), None)];

        let params = AdaptiveTransientParams {
            tstop: 5e-3, // 5 time constants
            h_init: 1e-7,
            h_min: 1e-9,
            h_max: 1e-4,
            reltol: 1e-3,
            abstol: 1e-6,
            method: IntegrationMethod::Trapezoidal,
        };

        let dc = DVector::from_vec(vec![5.0, 0.0, -0.005]);

        let result = solve_transient_adaptive(&stamper, &mut caps, &mut [], &params, &dc).unwrap();

        // After 5 tau, capacitor should be nearly charged to 5V
        let final_voltage = result.points.last().unwrap().solution[1];
        assert!(
            (final_voltage - 5.0).abs() < 0.05,
            "Final V(cap) = {} (expected ≈ 5.0)",
            final_voltage
        );

        // Adaptive should use fewer steps than fixed timestep
        // With fixed 10us steps, we'd need 500 steps
        // Adaptive should use fewer
        assert!(
            result.total_steps < 200,
            "Adaptive used {} steps (expected < 200 for efficiency)",
            result.total_steps
        );

        // Timestep should increase as capacitor approaches steady state
        assert!(
            result.max_step_used > params.h_init * 10.0,
            "Max step {} should grow from initial {}",
            result.max_step_used,
            params.h_init
        );

        println!(
            "Adaptive transient: {} total steps, {} rejected, h: [{:.2e}, {:.2e}]",
            result.total_steps, result.rejected_steps, result.min_step_used, result.max_step_used
        );
    }

    #[test]
    fn test_lte_estimation() {
        // Test that LTE estimate is reasonable for a smooth (constant rate) change.
        // For a constant dV/dt, the capacitor current is constant: i = C * dV/dt.
        // Trapezoidal and Backward Euler should agree, giving near-zero LTE.
        let h = 1e-6;
        let dv_dt = 1e5; // 0.1V per microsecond
        let v_prev = 0.0;
        let v_new = v_prev + dv_dt * h; // = 0.1V

        // Current is constant at C * dV/dt for linear voltage ramp
        let capacitance = 1e-6;
        let i_const = capacitance * dv_dt; // = 0.1 A

        let cap = CapacitorState {
            capacitance,
            v_prev,
            i_prev: i_const, // Current at previous step (same as current step for linear ramp)
            v_prev_prev: 0.0,
            node_pos: Some(0),
            node_neg: None,
        };

        let lte = cap.estimate_lte(v_new, h);

        // LTE should be non-negative
        assert!(lte >= 0.0, "LTE should be non-negative: {}", lte);

        // For a perfectly linear ramp with consistent i_prev, LTE should be very small
        assert!(
            lte < 1e-6,
            "LTE {} seems too large for constant-rate change",
            lte
        );
    }

    #[test]
    fn test_interpolate_at() {
        // Create a simple result with known values
        let points = vec![
            TimePoint {
                time: 0.0,
                solution: DVector::from_vec(vec![0.0, 0.0]),
            },
            TimePoint {
                time: 1.0,
                solution: DVector::from_vec(vec![1.0, 2.0]),
            },
            TimePoint {
                time: 2.0,
                solution: DVector::from_vec(vec![2.0, 4.0]),
            },
        ];

        let result = TransientResult {
            points,
            num_nodes: 2,
        };

        // Test interpolation at midpoint
        let interp = result.interpolate_at(0.5).unwrap();
        assert!((interp[0] - 0.5).abs() < 1e-10);
        assert!((interp[1] - 1.0).abs() < 1e-10);

        // Test interpolation at 1.5
        let interp = result.interpolate_at(1.5).unwrap();
        assert!((interp[0] - 1.5).abs() < 1e-10);
        assert!((interp[1] - 3.0).abs() < 1e-10);

        // Test at exact points
        let interp = result.interpolate_at(1.0).unwrap();
        assert!((interp[0] - 1.0).abs() < 1e-10);
        assert!((interp[1] - 2.0).abs() < 1e-10);

        // Test voltage_at helper
        assert!((result.voltage_at(0, 0.5).unwrap() - 0.5).abs() < 1e-10);
        assert!((result.voltage_at(1, 1.5).unwrap() - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_sample_at_times() {
        // Create a result with 3 points at t=0, 0.3, 1.0
        let points = vec![
            TimePoint {
                time: 0.0,
                solution: DVector::from_vec(vec![0.0]),
            },
            TimePoint {
                time: 0.3,
                solution: DVector::from_vec(vec![0.3]),
            },
            TimePoint {
                time: 1.0,
                solution: DVector::from_vec(vec![1.0]),
            },
        ];

        let result = TransientResult {
            points,
            num_nodes: 1,
        };

        // Sample at tstep=0.25
        let sampled = result.sample_at_times(0.25, None, None);

        // Should have 5 points: 0.0, 0.25, 0.5, 0.75, 1.0
        assert_eq!(sampled.points.len(), 5);

        // Check times
        assert!((sampled.points[0].time - 0.0).abs() < 1e-10);
        assert!((sampled.points[1].time - 0.25).abs() < 1e-10);
        assert!((sampled.points[2].time - 0.5).abs() < 1e-10);
        assert!((sampled.points[3].time - 0.75).abs() < 1e-10);
        assert!((sampled.points[4].time - 1.0).abs() < 1e-10);

        // Check interpolated values (linear from 0 to 1)
        assert!((sampled.points[0].solution[0] - 0.0).abs() < 1e-10);
        assert!((sampled.points[4].solution[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_rc_charging_trbdf2() {
        // Same RC circuit as other tests but using TR-BDF2
        let stamper = RcCircuitStamper {
            voltage: 5.0,
            resistance: 1000.0,
        };

        let mut caps = vec![CapacitorState::new(1e-6, Some(1), None)];

        let params = TransientParams {
            tstop: 5e-3,
            tstep: 10e-6,
            method: IntegrationMethod::TrBdf2,
        };

        let dc = DVector::from_vec(vec![5.0, 0.0, -0.005]);

        let result = solve_transient(&stamper, &mut caps, &mut [], &params, &dc).unwrap();

        // TR-BDF2 should produce reasonable results (similar to Trapezoidal)
        // At 5τ, voltage should be ~99.3% of final (very close to 5.0)
        let final_voltage = result.points.last().unwrap().solution[1];
        assert!(
            (final_voltage - 5.0).abs() < 0.15,
            "Final V(cap) = {} (expected ≈ 5.0)",
            final_voltage
        );

        // Check voltage at tau (time constant = RC = 1ms)
        // TR-BDF2 with 10µs steps should be within 20% at tau
        let tau_step = (1e-3 / params.tstep).round() as usize;
        let v_at_tau = result.points[tau_step].solution[1];
        let expected_v_tau = 5.0 * (1.0 - (-1.0_f64).exp()); // ~3.16V
        assert!(
            (v_at_tau - expected_v_tau).abs() < 0.6,
            "V(cap) at tau = {} (expected ≈ {}) [TR-BDF2]",
            v_at_tau,
            expected_v_tau
        );
    }

    /// Simple LC circuit stamper for oscillation test.
    /// Circuit: Initial voltage on capacitor, connected to inductor.
    /// Node 0: capacitor top / inductor top
    /// Ground: capacitor bottom / inductor bottom
    /// The inductor uses companion model (conductance + current source), not branch current.
    struct LcOscillatorStamper;

    impl TransientStamper for LcOscillatorStamper {
        fn stamp_at_time(&self, _mna: &mut MnaSystem, _time: f64) {
            // No static elements - capacitor and inductor are handled by companion models
            // The LC circuit has only reactive elements
        }

        fn num_nodes(&self) -> usize {
            1 // Just node 0 (top of L and C)
        }

        fn num_vsources(&self) -> usize {
            0 // No voltage sources - inductor uses companion model
        }
    }

    #[test]
    fn test_lc_oscillation() {
        // LC circuit: L = 1mH, C = 1µF
        // Resonant frequency: f = 1/(2π√(LC)) = 1/(2π√(1e-3 * 1e-6)) = 5033 Hz
        // Period: T = 1/f ≈ 0.199 ms ≈ 200 µs
        let inductance = 1e-3; // 1 mH
        let capacitance = 1e-6; // 1 µF

        let lc_product: f64 = inductance * capacitance;
        let expected_freq = 1.0 / (2.0 * std::f64::consts::PI * lc_product.sqrt());
        let expected_period: f64 = 1.0 / expected_freq;

        // Initial conditions: capacitor charged to 5V, zero inductor current
        // dc_solution: [V(0)] - just node voltage, no branch currents
        let dc = DVector::from_vec(vec![5.0]);

        // Create state for reactive elements
        let mut caps = vec![CapacitorState::new(capacitance, Some(0), None)];
        let mut inds = vec![InductorState::new(inductance, Some(0), None)];

        // Simulate for 5 periods using Trapezoidal (good for oscillators)
        let params = TransientParams {
            tstop: 5.0 * expected_period,
            tstep: expected_period / 50.0, // 50 points per period
            method: IntegrationMethod::Trapezoidal,
        };

        let result =
            solve_transient(&LcOscillatorStamper, &mut caps, &mut inds, &params, &dc).unwrap();

        // Find zero crossings to measure the period
        let voltages: Vec<f64> = result.points.iter().map(|p| p.solution[0]).collect();
        let times: Vec<f64> = result.points.iter().map(|p| p.time).collect();

        // Find first zero crossing from positive to negative (after initial positive)
        let mut zero_crossings = Vec::new();
        for i in 1..voltages.len() {
            if voltages[i - 1] > 0.0 && voltages[i] <= 0.0 {
                // Linear interpolation for more accurate crossing time
                let t_cross = times[i - 1]
                    + (0.0 - voltages[i - 1]) * (times[i] - times[i - 1])
                        / (voltages[i] - voltages[i - 1]);
                zero_crossings.push(t_cross);
            }
        }

        // Need at least 2 zero crossings to measure a full period
        assert!(
            zero_crossings.len() >= 2,
            "Not enough zero crossings found: {}",
            zero_crossings.len()
        );

        // Measure period from consecutive positive-to-negative zero crossings
        // These are separated by exactly one full period
        let measured_period = zero_crossings[1] - zero_crossings[0];
        let measured_freq = 1.0 / measured_period;

        // Check frequency is within 5% of expected
        let freq_error = (measured_freq - expected_freq).abs() / expected_freq;
        assert!(
            freq_error < 0.05,
            "LC oscillation frequency {} Hz differs from expected {} Hz by {:.1}%",
            measured_freq,
            expected_freq,
            freq_error * 100.0
        );

        // Check that amplitude is preserved (energy conservation)
        // For ideal LC, max voltage should stay close to initial voltage
        let max_voltage = voltages.iter().cloned().fold(0.0_f64, f64::max);
        let min_voltage = voltages.iter().cloned().fold(0.0_f64, f64::min);
        let amplitude = (max_voltage - min_voltage) / 2.0;

        assert!(
            (amplitude - 5.0).abs() < 0.5,
            "LC amplitude {} differs from expected 5.0 (amplitude decay)",
            amplitude
        );

        println!(
            "LC oscillator: expected f={:.1} Hz, measured f={:.1} Hz, error={:.2}%",
            expected_freq,
            measured_freq,
            freq_error * 100.0
        );
        println!(
            "LC oscillator: expected T={:.2} µs, measured T={:.2} µs",
            expected_period * 1e6,
            measured_period * 1e6
        );
    }
}
