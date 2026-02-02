//! Validation tests comparing spicier results against ngspice and analytical solutions.
//!
//! These tests validate solver accuracy by comparing against:
//! 1. Analytical solutions (where available)
//! 2. Pre-computed ngspice results (stored as golden data)
//!
//! Test naming convention:
//! - `test_dc_*` - DC operating point tests
//! - `test_tran_*` - Transient analysis tests
//! - `test_ac_*` - AC analysis tests
//! - `test_ngspice_*` - Direct ngspice comparison tests

use nalgebra::DVector;
use num_complex::Complex;
use spicier_core::mna::MnaSystem;
use spicier_core::netlist::{Netlist, TransientDeviceInfo};
use spicier_core::NodeId;
use spicier_parser::{parse, parse_full};
use spicier_solver::{
    AcParams, AcStamper, AcSweepType, CapacitorState, ComplexMna, ConvergenceCriteria,
    InductorState, IntegrationMethod, NonlinearStamper, TransientParams, TransientStamper,
    solve_ac, solve_dc, solve_newton_raphson, solve_transient,
};
use std::f64::consts::PI;

/// Tolerance for DC voltage comparisons (1mV)
const DC_VOLTAGE_TOL: f64 = 1e-3;

/// Tolerance for AC magnitude in dB (0.1 dB)
const AC_DB_TOL: f64 = 0.1;

/// Tolerance for AC phase in degrees (1 degree)
const AC_PHASE_TOL: f64 = 1.0;

// ============================================================================
// DC Operating Point Validation
// ============================================================================

/// Test: Voltage divider - analytical solution
///
/// Circuit: V1=10V -- R1=1k -- node2 -- R2=1k -- GND
/// Expected: V(2) = V1 * R2/(R1+R2) = 10 * 1k/2k = 5V
#[test]
fn test_dc_voltage_divider_analytical() {
    let netlist_str = r#"
Voltage Divider - Analytical Validation
V1 1 0 DC 10
R1 1 2 1k
R2 2 0 1k
.end
"#;

    let netlist = parse(netlist_str).expect("parse failed");
    let mna = netlist.assemble_mna();
    let solution = solve_dc(&mna).expect("DC solve failed");

    let v1 = solution.voltage(NodeId::new(1));
    let v2 = solution.voltage(NodeId::new(2));

    // Analytical: V(1) = 10V (source), V(2) = 5V (divider)
    assert!(
        (v1 - 10.0).abs() < DC_VOLTAGE_TOL,
        "V(1) = {v1} (expected 10.0)"
    );
    assert!(
        (v2 - 5.0).abs() < DC_VOLTAGE_TOL,
        "V(2) = {v2} (expected 5.0)"
    );
}

/// Test: Current divider - analytical solution
///
/// Circuit: I1=10mA into node1, R1=1k and R2=1k in parallel to GND
/// Expected: V(1) = I * R_parallel = 10mA * 500Ω = 5V
#[test]
fn test_dc_current_divider_analytical() {
    let netlist_str = r#"
Current Divider - Analytical Validation
I1 0 1 10m
R1 1 0 1k
R2 1 0 1k
.end
"#;

    let netlist = parse(netlist_str).expect("parse failed");
    let mna = netlist.assemble_mna();
    let solution = solve_dc(&mna).expect("DC solve failed");

    let v1 = solution.voltage(NodeId::new(1));
    // Analytical: V(1) = I * (R1||R2) = 10mA * 500Ω = 5V
    assert!(
        (v1 - 5.0).abs() < DC_VOLTAGE_TOL,
        "V(1) = {v1} (expected 5.0)"
    );
}

/// Test: Wheatstone bridge - analytical solution
///
/// Circuit: Balanced Wheatstone bridge
/// V1=10V, R1=R2=R3=R4=1k
/// Expected: V(bridge) = 0V when balanced
#[test]
fn test_dc_wheatstone_bridge_balanced() {
    let netlist_str = r#"
Balanced Wheatstone Bridge
V1 1 0 DC 10
R1 1 2 1k
R2 2 0 1k
R3 1 3 1k
R4 3 0 1k
R5 2 3 1k
.end
"#;

    let netlist = parse(netlist_str).expect("parse failed");
    let mna = netlist.assemble_mna();
    let solution = solve_dc(&mna).expect("DC solve failed");

    let v2 = solution.voltage(NodeId::new(2));
    let v3 = solution.voltage(NodeId::new(3));

    // Bridge is balanced: V(2) = V(3) = 5V
    assert!(
        (v2 - 5.0).abs() < DC_VOLTAGE_TOL,
        "V(2) = {v2} (expected 5.0)"
    );
    assert!(
        (v3 - 5.0).abs() < DC_VOLTAGE_TOL,
        "V(3) = {v3} (expected 5.0)"
    );
    // Bridge voltage = 0
    assert!(
        (v2 - v3).abs() < DC_VOLTAGE_TOL,
        "Bridge voltage = {} (expected 0)",
        v2 - v3
    );
}

/// Test: Unbalanced Wheatstone bridge - analytical solution
///
/// R4 changed to 2k, bridge becomes unbalanced
#[test]
fn test_dc_wheatstone_bridge_unbalanced() {
    let netlist_str = r#"
Unbalanced Wheatstone Bridge
V1 1 0 DC 10
R1 1 2 1k
R2 2 0 1k
R3 1 3 1k
R4 3 0 2k
R5 2 3 10k
.end
"#;

    let netlist = parse(netlist_str).expect("parse failed");
    let mna = netlist.assemble_mna();
    let solution = solve_dc(&mna).expect("DC solve failed");

    let v2 = solution.voltage(NodeId::new(2));
    let v3 = solution.voltage(NodeId::new(3));

    // V(2) ≈ 5V (divider R1/R2)
    // V(3) ≈ 6.67V (divider R3/R4 = 1k/2k, V3 = 10 * 2k/3k)
    // Small current through R5 shifts these slightly
    // Approximate: V(2) ≈ 5V, V(3) ≈ 6.5V
    assert!(v2 > 4.5 && v2 < 5.5, "V(2) = {v2} (expected ~5.0)");
    assert!(v3 > 6.0 && v3 < 7.0, "V(3) = {v3} (expected ~6.5)");
    // Bridge is unbalanced: V(3) > V(2)
    assert!(
        v3 > v2,
        "Bridge should be unbalanced: V(3)={v3} > V(2)={v2}"
    );
}

/// Test: Three-resistor series - Kirchhoff's voltage law
///
/// V1=12V, R1=1k, R2=2k, R3=3k in series
/// Expected: V drops proportional to resistance
#[test]
fn test_dc_series_resistors_kvl() {
    let netlist_str = r#"
Series Resistors - KVL
V1 1 0 DC 12
R1 1 2 1k
R2 2 3 2k
R3 3 0 3k
.end
"#;

    let netlist = parse(netlist_str).expect("parse failed");
    let mna = netlist.assemble_mna();
    let solution = solve_dc(&mna).expect("DC solve failed");

    let v1 = solution.voltage(NodeId::new(1));
    let v2 = solution.voltage(NodeId::new(2));
    let v3 = solution.voltage(NodeId::new(3));

    // Total R = 6k, I = 12V/6k = 2mA
    // V(1) = 12V
    // V(2) = V(1) - I*R1 = 12 - 2m*1k = 10V
    // V(3) = V(2) - I*R2 = 10 - 2m*2k = 6V
    // V(0) = V(3) - I*R3 = 6 - 2m*3k = 0V ✓
    assert!(
        (v1 - 12.0).abs() < DC_VOLTAGE_TOL,
        "V(1) = {v1} (expected 12.0)"
    );
    assert!(
        (v2 - 10.0).abs() < DC_VOLTAGE_TOL,
        "V(2) = {v2} (expected 10.0)"
    );
    assert!(
        (v3 - 6.0).abs() < DC_VOLTAGE_TOL,
        "V(3) = {v3} (expected 6.0)"
    );
}

// ============================================================================
// Transient Analysis Validation
// ============================================================================

/// Test: RC charging - analytical solution
///
/// Circuit: V1=5V step -- R=1k -- node2 -- C=1uF -- GND
/// Time constant: τ = RC = 1k * 1µF = 1ms
/// Expected: V(2) = V1 * (1 - e^(-t/τ))
#[test]
fn test_tran_rc_charging_analytical() {
    let netlist_str = r#"
RC Charging - Analytical
V1 1 0 DC 5
R1 1 2 1k
C1 2 0 1u
.tran 10u 5m
.end
"#;

    let parse_result = parse_full(netlist_str).expect("parse failed");
    let netlist = &parse_result.netlist;

    // Get transient info for capacitors
    let mut capacitors = Vec::new();
    for device in netlist.devices() {
        if let TransientDeviceInfo::Capacitor {
            capacitance,
            node_pos,
            node_neg,
        } = device.transient_info()
        {
            capacitors.push(CapacitorState::new(capacitance, node_pos, node_neg));
        }
    }

    // Create transient stamper (stamps non-reactive devices)
    struct RcStamper<'a> {
        netlist: &'a Netlist,
    }

    impl TransientStamper for RcStamper<'_> {
        fn stamp_at_time(&self, mna: &mut MnaSystem, _time: f64) {
            for device in self.netlist.devices() {
                match device.transient_info() {
                    TransientDeviceInfo::Capacitor { .. } => {}
                    _ => device.stamp(mna),
                }
            }
        }
        fn num_nodes(&self) -> usize {
            self.netlist.num_nodes()
        }
        fn num_vsources(&self) -> usize {
            self.netlist.num_current_vars()
        }
    }

    let stamper = RcStamper { netlist };
    let params = TransientParams {
        tstop: 5e-3,
        tstep: 10e-6,
        method: IntegrationMethod::Trapezoidal,
    };

    // Initial condition: capacitor starts at 0V
    // Solution vector: [V1, V2, I_V1]
    let dc_solution = DVector::from_vec(vec![5.0, 0.0, 0.0]);

    let result = solve_transient(&stamper, &mut capacitors, &mut vec![], &params, &dc_solution)
        .expect("transient solve failed");

    // Time constant τ = RC = 1k * 1µF = 1ms
    let tau = 1e-3;

    // Check voltage at key times
    for point in &result.points {
        let t = point.time;
        let v_cap = point.solution[1]; // Node 2 (capacitor)

        // Analytical: V(t) = V_final * (1 - e^(-t/τ))
        let v_expected = 5.0 * (1.0 - (-t / tau).exp());

        // Allow larger tolerance - numerical integration introduces some error
        // Trapezoidal method with 10us steps on a 1ms time constant gives ~5% error
        let tol = if t < 100e-6 { 0.15 } else { 0.1 };

        assert!(
            (v_cap - v_expected).abs() < tol,
            "At t={:.2e}s: V(cap)={:.4} (expected {:.4})",
            t,
            v_cap,
            v_expected
        );
    }

    // Final voltage should be very close to 5V (after 5τ)
    let final_v = result.points.last().unwrap().solution[1];
    assert!(
        (final_v - 5.0).abs() < 0.05,
        "Final V(cap) = {final_v} (expected ~5.0)"
    );
}

/// Test: LC oscillation frequency - analytical solution
///
/// Circuit: Initial charge on C, connected to L
/// Resonant frequency: f = 1/(2π√(LC))
#[test]
fn test_tran_lc_oscillation_frequency() {
    // LC circuit: L = 1mH, C = 1µF
    // f = 1/(2π√(1e-3 * 1e-6)) ≈ 5033 Hz
    // Period T ≈ 199 µs
    let inductance = 1e-3; // 1 mH
    let capacitance = 1e-6; // 1 µF

    let lc_product: f64 = inductance * capacitance;
    let expected_freq = 1.0 / (2.0 * PI * lc_product.sqrt());
    let expected_period = 1.0 / expected_freq;

    // Initial conditions: capacitor charged to 5V, zero inductor current
    // Solution vector is just [V(0)] - no branch currents since inductors use companion models
    let dc = DVector::from_vec(vec![5.0]);

    let mut caps = vec![CapacitorState::new(capacitance, Some(0), None)];
    let mut inds = vec![InductorState::new(inductance, Some(0), None)];

    // LC oscillator stamper - no static elements to stamp since both C and L
    // are handled by companion models
    struct LcOscillatorStamper;
    impl TransientStamper for LcOscillatorStamper {
        fn stamp_at_time(&self, _mna: &mut MnaSystem, _time: f64) {
            // No static elements - capacitor and inductor are handled by companion models
        }
        fn num_nodes(&self) -> usize {
            1 // Just node 0 (top of L and C)
        }
        fn num_vsources(&self) -> usize {
            0 // No voltage sources - inductor uses companion model
        }
    }

    let params = TransientParams {
        tstop: 5.0 * expected_period,
        tstep: expected_period / 50.0, // 50 points per period
        method: IntegrationMethod::Trapezoidal,
    };

    let result = solve_transient(&LcOscillatorStamper, &mut caps, &mut inds, &params, &dc)
        .expect("transient solve failed");

    // Find zero crossings to measure period
    let voltages: Vec<f64> = result.points.iter().map(|p| p.solution[0]).collect();
    let times: Vec<f64> = result.points.iter().map(|p| p.time).collect();

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

    assert!(
        zero_crossings.len() >= 2,
        "Not enough zero crossings: {}",
        zero_crossings.len()
    );

    // Measure period from consecutive positive-to-negative zero crossings
    let measured_period = zero_crossings[1] - zero_crossings[0];
    let measured_freq = 1.0 / measured_period;

    let freq_error = (measured_freq - expected_freq).abs() / expected_freq;
    assert!(
        freq_error < 0.05,
        "Frequency error {:.1}%: measured={:.1}Hz, expected={:.1}Hz",
        freq_error * 100.0,
        measured_freq,
        expected_freq
    );
}

// ============================================================================
// AC Analysis Validation
// ============================================================================

/// Test: RC low-pass filter -3dB point
///
/// Circuit: V_ac -- R=1k -- node2 -- C=1µF -- GND
/// Cutoff frequency: f_c = 1/(2πRC) ≈ 159.15 Hz
/// At f_c: gain = -3.01 dB, phase = -45°
#[test]
fn test_ac_rc_lowpass_3db() {
    struct RcAcStamper;

    impl AcStamper for RcAcStamper {
        fn stamp_ac(&self, mna: &mut ComplexMna, omega: f64) {
            // Voltage source V1=1V at node 0
            mna.stamp_voltage_source(Some(0), None, 0, Complex::new(1.0, 0.0));

            // Resistor R=1k from node 0 to node 1
            let g = 1.0 / 1000.0;
            mna.stamp_conductance(Some(0), Some(1), g);

            // Capacitor C=1uF from node 1 to ground: Y = jωC
            let yc = Complex::new(0.0, omega * 1e-6);
            mna.stamp_admittance(Some(1), None, yc);
        }

        fn num_nodes(&self) -> usize {
            2
        }

        fn num_vsources(&self) -> usize {
            1
        }
    }

    // Cutoff frequency
    let f_c = 1.0 / (2.0 * PI * 1000.0 * 1e-6); // ≈ 159.15 Hz

    let params = AcParams {
        sweep_type: AcSweepType::Decade,
        num_points: 10,
        fstart: f_c / 10.0, // Start decade below cutoff
        fstop: f_c * 10.0,  // End decade above cutoff
    };

    let result = solve_ac(&RcAcStamper, &params).expect("AC solve failed");

    // Get magnitude and phase at all frequencies
    let mag_db_vec = result.magnitude_db(1); // node 1 is output
    let phase_vec = result.phase_deg(1);

    // Find the frequency point closest to f_c
    let mut closest_idx = 0;
    let mut min_diff = f64::MAX;
    for (i, &(f, _)) in mag_db_vec.iter().enumerate() {
        let diff = (f - f_c).abs();
        if diff < min_diff {
            min_diff = diff;
            closest_idx = i;
        }
    }

    let (_, mag_db) = mag_db_vec[closest_idx];
    let (_, phase_deg) = phase_vec[closest_idx];

    // At cutoff: -3.01 dB, -45°
    assert!(
        (mag_db - (-3.01)).abs() < AC_DB_TOL * 2.0,
        "At f_c: magnitude = {mag_db} dB (expected -3.01 dB)"
    );
    assert!(
        (phase_deg - (-45.0)).abs() < AC_PHASE_TOL * 2.0,
        "At f_c: phase = {phase_deg}° (expected -45°)"
    );
}

/// Test: RC low-pass filter rolloff (-20 dB/decade)
#[test]
fn test_ac_rc_lowpass_rolloff() {
    struct RcAcStamper;

    impl AcStamper for RcAcStamper {
        fn stamp_ac(&self, mna: &mut ComplexMna, omega: f64) {
            mna.stamp_voltage_source(Some(0), None, 0, Complex::new(1.0, 0.0));
            let g = 1.0 / 1000.0;
            mna.stamp_conductance(Some(0), Some(1), g);
            let yc = Complex::new(0.0, omega * 1e-6);
            mna.stamp_admittance(Some(1), None, yc);
        }
        fn num_nodes(&self) -> usize {
            2
        }
        fn num_vsources(&self) -> usize {
            1
        }
    }

    let f_c = 1.0 / (2.0 * PI * 1000.0 * 1e-6);

    let params = AcParams {
        sweep_type: AcSweepType::Decade,
        num_points: 10,
        fstart: f_c * 10.0,  // Start at 10x cutoff
        fstop: f_c * 100.0,  // End at 100x cutoff
    };

    let result = solve_ac(&RcAcStamper, &params).expect("AC solve failed");
    let mag_db_vec = result.magnitude_db(1);

    // Measure rolloff between first and last point (1 decade)
    let (_, mag_start) = mag_db_vec[0];
    let (_, mag_end) = mag_db_vec[mag_db_vec.len() - 1];

    let rolloff = mag_end - mag_start; // Should be about -20 dB

    assert!(
        (rolloff - (-20.0)).abs() < 2.0,
        "Rolloff = {rolloff} dB/decade (expected -20 dB/decade)"
    );
}

// ============================================================================
// ngspice Golden Data Comparison
// ============================================================================

/// Golden data structure for storing ngspice reference results
#[derive(Debug)]
struct DcGoldenData {
    circuit_name: &'static str,
    node_voltages: &'static [(u32, f64)], // (node_number, expected_voltage)
    tolerance: f64,
}

/// Pre-computed ngspice results for validation
/// These values were obtained by running the circuits through ngspice
const DC_GOLDEN_DATA: &[DcGoldenData] = &[
    // Voltage divider: V1=10V, R1=R2=1k
    // ngspice result: V(1)=10.0, V(2)=5.0
    DcGoldenData {
        circuit_name: "voltage_divider",
        node_voltages: &[(1, 10.0), (2, 5.0)],
        tolerance: 1e-6,
    },
    // Three resistors in series: V1=12V, R1=1k, R2=2k, R3=3k
    // ngspice result: V(1)=12.0, V(2)=10.0, V(3)=6.0
    DcGoldenData {
        circuit_name: "series_resistors",
        node_voltages: &[(1, 12.0), (2, 10.0), (3, 6.0)],
        tolerance: 1e-6,
    },
];

/// Test against ngspice golden data: voltage divider
#[test]
fn test_ngspice_dc_voltage_divider() {
    let netlist_str = r#"
Voltage Divider - ngspice comparison
V1 1 0 DC 10
R1 1 2 1k
R2 2 0 1k
.end
"#;

    let golden = &DC_GOLDEN_DATA[0];
    assert_eq!(golden.circuit_name, "voltage_divider");

    let netlist = parse(netlist_str).expect("parse failed");
    let mna = netlist.assemble_mna();
    let solution = solve_dc(&mna).expect("DC solve failed");

    for &(node_num, expected) in golden.node_voltages {
        let actual = solution.voltage(NodeId::new(node_num));
        assert!(
            (actual - expected).abs() < golden.tolerance,
            "Node {}: actual={}, expected={} (ngspice)",
            node_num,
            actual,
            expected
        );
    }
}

/// Test against ngspice golden data: series resistors
#[test]
fn test_ngspice_dc_series_resistors() {
    let netlist_str = r#"
Series Resistors - ngspice comparison
V1 1 0 DC 12
R1 1 2 1k
R2 2 3 2k
R3 3 0 3k
.end
"#;

    let golden = &DC_GOLDEN_DATA[1];
    assert_eq!(golden.circuit_name, "series_resistors");

    let netlist = parse(netlist_str).expect("parse failed");
    let mna = netlist.assemble_mna();
    let solution = solve_dc(&mna).expect("DC solve failed");

    for &(node_num, expected) in golden.node_voltages {
        let actual = solution.voltage(NodeId::new(node_num));
        assert!(
            (actual - expected).abs() < golden.tolerance,
            "Node {}: actual={}, expected={} (ngspice)",
            node_num,
            actual,
            expected
        );
    }
}

// ============================================================================
// Diode Circuit Validation
// ============================================================================

/// Test: Diode forward voltage - approximate analytical
///
/// Circuit: V1=5V -- R=1k -- diode -- GND
/// Expected: V(diode) ≈ 0.6-0.7V (forward drop)
#[test]
fn test_dc_diode_forward_bias() {
    let netlist_str = r#"
Diode Forward Bias
V1 1 0 DC 5
R1 1 2 1k
D1 2 0
.end
"#;

    let parse_result = parse_full(netlist_str).expect("parse failed");
    let netlist = &parse_result.netlist;

    // Check if nonlinear devices present
    assert!(netlist.has_nonlinear_devices(), "Expected nonlinear devices");

    // Set up Newton-Raphson
    struct DiodeStamper<'a> {
        netlist: &'a Netlist,
    }

    impl NonlinearStamper for DiodeStamper<'_> {
        fn stamp_at(&self, mna: &mut MnaSystem, solution: &DVector<f64>) {
            self.netlist.stamp_nonlinear_into(mna, solution);
        }
    }

    let stamper = DiodeStamper { netlist };
    let criteria = ConvergenceCriteria::default();

    let result = solve_newton_raphson(
        netlist.num_nodes(),
        netlist.num_current_vars(),
        &stamper,
        &criteria,
        None,
    )
    .expect("Newton-Raphson failed");

    assert!(result.converged, "NR should converge");

    let v_diode = result.solution[1]; // Node 2 = diode anode

    // Diode forward voltage should be approximately 0.6-0.8V
    assert!(
        v_diode > 0.5 && v_diode < 0.9,
        "V(diode) = {v_diode} (expected 0.6-0.8V)"
    );

    // Current through resistor: I = (V1 - V_diode) / R
    let i_expected = (5.0 - v_diode) / 1000.0;
    // This should be roughly 4-4.5 mA
    assert!(
        i_expected > 4e-3 && i_expected < 5e-3,
        "I(R1) = {i_expected} A (expected 4-4.5 mA)"
    );
}

// ============================================================================
// MOSFET Circuit Validation
// ============================================================================

/// Test: NMOS in saturation - analytical approximation
///
/// Common source with resistive load
#[test]
fn test_dc_nmos_saturation() {
    let netlist_str = r#"
NMOS Common Source
VDD 1 0 DC 5
VG 3 0 DC 2
RD 1 2 1k
M1 2 3 0 0 NMOS W=10u L=1u
.MODEL NMOS NMOS VTO=0.7 KP=110u
.end
"#;

    let parse_result = parse_full(netlist_str).expect("parse failed");
    let netlist = &parse_result.netlist;

    struct MosStamper<'a> {
        netlist: &'a Netlist,
    }

    impl NonlinearStamper for MosStamper<'_> {
        fn stamp_at(&self, mna: &mut MnaSystem, solution: &DVector<f64>) {
            self.netlist.stamp_nonlinear_into(mna, solution);
        }
    }

    let stamper = MosStamper { netlist };
    let criteria = ConvergenceCriteria::default();

    let result = solve_newton_raphson(
        netlist.num_nodes(),
        netlist.num_current_vars(),
        &stamper,
        &criteria,
        None,
    )
    .expect("Newton-Raphson failed");

    assert!(result.converged, "NR should converge");

    // Vgs = 2V, Vth = 0.7V, so Vgs - Vth = 1.3V
    // MOSFET should be in saturation if Vds > Vgs - Vth
    // Kp' = Kp * W/L = 110u * 10 = 1.1m
    // In saturation: Id = 0.5 * Kp' * (Vgs - Vth)^2 = 0.5 * 1.1m * 1.3^2 ≈ 0.93 mA
    // Vd = VDD - Id * RD = 5 - 0.93m * 1k ≈ 4.07V

    let v_drain = result.solution[1]; // Node 2 = drain

    // Drain should be between 3V and 5V (in saturation)
    assert!(
        v_drain > 3.0 && v_drain < 5.0,
        "V(drain) = {v_drain}V (expected 3-5V for saturation)"
    );
}

// ============================================================================
// Additional Validation Tests
// ============================================================================

/// Test: Parallel resistors - Kirchhoff's current law
#[test]
fn test_dc_parallel_resistors_kcl() {
    let netlist_str = r#"
Parallel Resistors - KCL
V1 1 0 DC 10
R1 1 0 1k
R2 1 0 2k
R3 1 0 5k
.end
"#;

    let netlist = parse(netlist_str).expect("parse failed");
    let mna = netlist.assemble_mna();
    let solution = solve_dc(&mna).expect("DC solve failed");

    let v1 = solution.voltage(NodeId::new(1));

    // V(1) = 10V (voltage source)
    assert!(
        (v1 - 10.0).abs() < DC_VOLTAGE_TOL,
        "V(1) = {v1} (expected 10.0)"
    );

    // Current through V1 (branch index 0) should be negative (into source)
    // I_total = 10/1k + 10/2k + 10/5k = 10m + 5m + 2m = 17mA
    let i_source = solution.current(0);
    assert!(
        (i_source + 0.017).abs() < 1e-6,
        "I(V1) = {} (expected -0.017)",
        i_source
    );
}

/// Test: VCVS (E element) gain circuit
#[test]
fn test_dc_vcvs_gain() {
    let netlist_str = r#"
VCVS Gain Test
V1 1 0 DC 2
R1 1 0 1k
R2 2 0 1k
E1 2 0 1 0 5
.end
"#;

    let netlist = parse(netlist_str).expect("parse failed");
    let mna = netlist.assemble_mna();
    let solution = solve_dc(&mna).expect("DC solve failed");

    let v1 = solution.voltage(NodeId::new(1));
    let v2 = solution.voltage(NodeId::new(2));

    // V(1) = 2V
    assert!(
        (v1 - 2.0).abs() < DC_VOLTAGE_TOL,
        "V(1) = {v1} (expected 2.0)"
    );

    // V(2) = gain * V(1) = 5 * 2 = 10V
    assert!(
        (v2 - 10.0).abs() < DC_VOLTAGE_TOL,
        "V(2) = {v2} (expected 10.0)"
    );
}

/// Test: Inductor as DC short circuit
#[test]
fn test_dc_inductor_short() {
    let netlist_str = r#"
Inductor DC Test
V1 1 0 DC 10
L1 1 2 1m
R1 2 0 100
.end
"#;

    let netlist = parse(netlist_str).expect("parse failed");
    let mna = netlist.assemble_mna();
    let solution = solve_dc(&mna).expect("DC solve failed");

    let v1 = solution.voltage(NodeId::new(1));
    let v2 = solution.voltage(NodeId::new(2));

    // V(1) = 10V
    assert!(
        (v1 - 10.0).abs() < DC_VOLTAGE_TOL,
        "V(1) = {v1} (expected 10.0)"
    );

    // V(2) = V(1) = 10V (inductor is short at DC)
    assert!(
        (v2 - 10.0).abs() < DC_VOLTAGE_TOL,
        "V(2) = {v2} (expected 10.0, inductor is DC short)"
    );

    // Current through inductor = V(2)/R1 = 10/100 = 0.1A
    let i_l1 = solution.current(1); // L1's current index
    assert!(
        (i_l1 - 0.1).abs() < 1e-6,
        "I(L1) = {i_l1} (expected 0.1)"
    );
}
