//! Transient time-domain analysis.

use anyhow::Result;
use nalgebra::DVector;
use spicier_core::NodeId;
use spicier_parser::{InitialCondition, OutputVariable};
use spicier_solver::{
    ConvergenceCriteria, InitialConditions, IntegrationMethod, TransientParams, TransientStamper,
    solve_dc, solve_newton_raphson, solve_transient,
};
use std::collections::HashMap;

use crate::output::get_dc_print_nodes;
use crate::stampers::{build_transient_state, NetlistNonlinearStamper, NetlistTransientStamper};

/// Run transient time-domain analysis.
#[allow(clippy::too_many_arguments)]
pub fn run_transient(
    netlist: &spicier_core::Netlist,
    tstep: f64,
    tstop: f64,
    tstart: f64,
    uic: bool,
    initial_conditions: &[InitialCondition],
    node_map: &HashMap<String, NodeId>,
    print_vars: &[&OutputVariable],
) -> Result<()> {
    println!(
        "Transient Analysis (.TRAN {} {} {}{})",
        tstep,
        tstop,
        tstart,
        if uic { " UIC" } else { "" }
    );
    println!("==========================================");
    println!();

    // 1. Get initial conditions - either from DC operating point or from .IC values (if UIC)
    let mut dc_solution = if uic {
        // UIC: Skip DC operating point, start from zero and apply .IC values
        println!("UIC: Skipping DC operating point calculation.");
        DVector::zeros(netlist.num_nodes() + netlist.num_current_vars())
    } else if netlist.has_nonlinear_devices() {
        let stamper = NetlistNonlinearStamper { netlist };
        let criteria = ConvergenceCriteria::default();
        let nr_result = solve_newton_raphson(
            netlist.num_nodes(),
            netlist.num_current_vars(),
            &stamper,
            &criteria,
            None,
        )
        .map_err(|e| anyhow::anyhow!("DC operating point error: {}", e))?;
        nr_result.solution
    } else {
        let mna = netlist.assemble_mna();
        let dc = solve_dc(&mna).map_err(|e| anyhow::anyhow!("DC operating point error: {}", e))?;
        // Reconstruct full solution vector
        let mut full = DVector::zeros(netlist.num_nodes() + netlist.num_current_vars());
        for i in 0..dc.num_nodes {
            full[i] = dc.node_voltages[i];
        }
        for i in 0..dc.branch_currents.len() {
            full[dc.num_nodes + i] = dc.branch_currents[i];
        }
        full
    };

    // 1b. Apply .IC initial conditions (override DC solution)
    if !initial_conditions.is_empty() {
        // Convert parser's InitialCondition to solver's InitialConditions
        let mut ic = InitialConditions::new();
        for parsed_ic in initial_conditions {
            ic.set_voltage(&parsed_ic.node, parsed_ic.voltage);
        }
        // Build MNA index map from node_map
        // NodeId.as_u32() is the node number (1-based), MNA index is (node_number - 1)
        let mna_index_map: HashMap<String, usize> = node_map
            .iter()
            .filter_map(|(name, node_id)| {
                if node_id.is_ground() {
                    None
                } else {
                    Some((name.clone(), node_id.as_u32() as usize - 1))
                }
            })
            .collect();
        ic.apply(&mut dc_solution, &mna_index_map);

        println!("Applied initial conditions:");
        for parsed_ic in initial_conditions {
            println!("  V({}) = {} V", parsed_ic.node, parsed_ic.voltage);
        }
        println!();
    }

    // 2. Build reactive element state vectors
    let (mut caps, mut inds) = build_transient_state(netlist);

    // 3. Build transient stamper (stamps non-reactive devices)
    let stamper = NetlistTransientStamper { netlist };

    // 4. Run transient simulation with Trapezoidal method
    let params = TransientParams {
        tstop,
        tstep,
        method: IntegrationMethod::Trapezoidal,
    };

    // Adjust DC solution size if inductor companion models change MNA dimensions
    let tran_size = stamper.num_nodes() + stamper.num_vsources();
    let dc_for_tran = if dc_solution.len() != tran_size {
        let mut adjusted = DVector::zeros(tran_size);
        for i in 0..tran_size.min(dc_solution.len()) {
            adjusted[i] = dc_solution[i];
        }
        adjusted
    } else {
        dc_solution
    };

    let result = solve_transient(&stamper, &mut caps, &mut inds, &params, &dc_for_tran)
        .map_err(|e| anyhow::anyhow!("Transient error: {}", e))?;

    // 5. Print tabular output
    let nodes_to_print = get_dc_print_nodes(print_vars, node_map, netlist.num_nodes());

    // Header
    print!("{:>14}", "Time");
    for (name, _) in &nodes_to_print {
        print!("{:>14}", format!("V({})", name));
    }
    println!();

    let width = 14 * (1 + nodes_to_print.len());
    println!("{}", "-".repeat(width));

    // Data (skip points before tstart)
    for point in &result.points {
        if point.time < tstart - tstep * 0.5 {
            continue;
        }
        print!("{:>14.6e}", point.time);
        for (_, node_id) in &nodes_to_print {
            let idx = (node_id.as_u32() - 1) as usize;
            let v = if idx < point.solution.len() {
                point.solution[idx]
            } else {
                0.0
            };
            print!("{:>14.6}", v);
        }
        println!();
    }

    println!();
    println!(
        "Transient analysis complete ({} points).",
        result.points.len()
    );
    println!();
    Ok(())
}
