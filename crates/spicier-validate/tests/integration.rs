//! Integration tests for spicier-validate.
//!
//! These tests require ngspice to be installed.

use spicier_validate::{
    ComparisonConfig, NgspiceConfig, is_ngspice_available, compare_simulators,
};

fn ngspice_available() -> bool {
    is_ngspice_available(&NgspiceConfig::default())
}

#[test]
#[ignore = "requires ngspice"]
fn test_voltage_divider_dc() {
    if !ngspice_available() {
        eprintln!("ngspice not available, skipping test");
        return;
    }

    let netlist = "Voltage Divider\nV1 1 0 DC 10\nR1 1 2 1k\nR2 2 0 1k\n.op\n.end\n";

    let config = ComparisonConfig::default();
    let report = compare_simulators(netlist, &config).unwrap();

    println!("Report:\n{}", report.to_text());
    assert!(report.passed, "DC voltage divider should match");

    // Check specific values
    for comp in &report.comparisons {
        if comp.name.contains("V(2)") {
            // Should be 5V (half of 10V)
            assert!(comp.passed, "V(2) should match expected 5V");
        }
    }
}

#[test]
#[ignore = "requires ngspice"]
fn test_series_resistors() {
    if !ngspice_available() {
        return;
    }

    let netlist = "Series Resistors\nI1 0 1 DC 1m\nR1 1 2 1k\nR2 2 3 2k\nR3 3 0 3k\n.op\n.end\n";

    let config = ComparisonConfig::default();
    let report = compare_simulators(netlist, &config).unwrap();

    println!("Report:\n{}", report.to_text());
    assert!(report.passed, "Series resistors should match");
}

#[test]
#[ignore = "requires ngspice"]
fn test_parallel_resistors() {
    if !ngspice_available() {
        return;
    }

    let netlist = "Parallel Resistors\nI1 0 1 DC 10m\nR1 1 0 1k\nR2 1 0 1k\n.op\n.end\n";

    let config = ComparisonConfig::default();
    let report = compare_simulators(netlist, &config).unwrap();

    println!("Report:\n{}", report.to_text());
    assert!(report.passed, "Parallel resistors should match");
}

#[test]
fn test_spicier_only_dc() {
    // Test that spicier can solve a simple circuit without ngspice
    let netlist = "Voltage Divider\nV1 1 0 DC 10\nR1 1 2 1k\nR2 2 0 1k\n.op\n.end\n";

    let result = spicier_validate::run_spicier(netlist).unwrap();

    match result {
        spicier_validate::SpicierResult::DcOp(dc) => {
            let v2 = dc.voltage("V(2)").expect("V(2) should exist");
            assert!((v2 - 5.0).abs() < 1e-6, "V(2) should be 5.0, got {}", v2);

            let v1 = dc.voltage("V(1)").expect("V(1) should exist");
            assert!((v1 - 10.0).abs() < 1e-6, "V(1) should be 10.0, got {}", v1);
        }
        _ => panic!("Expected DC operating point result"),
    }
}

#[test]
fn test_values_match_function() {
    use spicier_validate::values_match;

    // Exact match
    assert!(values_match(1.0, 1.0, 1e-9, 1e-9));

    // Within absolute tolerance
    assert!(values_match(0.0, 1e-10, 1e-9, 1e-9));

    // Within relative tolerance
    assert!(values_match(100.0, 100.001, 1e-9, 1e-3));

    // Outside both tolerances
    assert!(!values_match(100.0, 101.0, 1e-9, 1e-4));
}

#[test]
fn test_config_builder() {
    use spicier_validate::{AcTolerances, ComparisonConfig, DcTolerances};

    let config = ComparisonConfig::default()
        .with_dc_tolerances(DcTolerances {
            voltage_abs: 1e-3,
            voltage_rel: 1e-2,
            current_abs: 1e-6,
            current_rel: 1e-2,
        })
        .with_ac_tolerances(AcTolerances {
            magnitude_db: 0.5,
            phase_deg: 5.0,
        })
        .with_variables(vec!["V(1)".to_string(), "V(2)".to_string()]);

    assert!((config.dc.voltage_abs - 1e-3).abs() < 1e-12);
    assert!((config.ac.magnitude_db - 0.5).abs() < 1e-12);
    assert_eq!(config.variables.unwrap().len(), 2);
}
