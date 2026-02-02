//! Output formatting and print variable handling.

use spicier_core::NodeId;
use spicier_parser::OutputVariable;
use spicier_solver::DcSolution;
use std::collections::{HashMap, HashSet};

/// Get list of (name, NodeId) pairs to print based on .PRINT variables.
/// If print_vars is empty, prints all nodes.
pub fn get_dc_print_nodes(
    print_vars: &[&OutputVariable],
    node_map: &HashMap<String, NodeId>,
    num_nodes: usize,
) -> Vec<(String, NodeId)> {
    if print_vars.is_empty() {
        // Print all nodes
        (1..=num_nodes)
            .map(|i| (i.to_string(), NodeId::new(i as u32)))
            .collect()
    } else {
        // Print only specified nodes
        print_vars
            .iter()
            .filter_map(|v| {
                if let OutputVariable::Voltage { node, node2: None } = v {
                    // Try to find node in node_map, or parse as number
                    if let Some(node_id) = node_map.get(node) {
                        if !node_id.is_ground() {
                            return Some((node.clone(), *node_id));
                        }
                    } else if let Ok(n) = node.parse::<u32>() {
                        if n > 0 && n <= num_nodes as u32 {
                            return Some((node.clone(), NodeId::new(n)));
                        }
                    }
                }
                None
            })
            .collect()
    }
}

/// Get list of (name, NodeId) pairs to print based on .PRINT AC variables.
/// Handles V(), VM(), VP(), VDB(), VR(), VI() output types.
/// If print_vars is empty, prints all nodes.
pub fn get_ac_print_nodes(
    print_vars: &[&OutputVariable],
    node_map: &HashMap<String, NodeId>,
    num_nodes: usize,
) -> Vec<(String, NodeId)> {
    if print_vars.is_empty() {
        // Print all nodes
        (1..=num_nodes)
            .map(|i| (i.to_string(), NodeId::new(i as u32)))
            .collect()
    } else {
        // Extract unique nodes from all AC output variable types
        let mut seen = HashSet::new();
        let mut result = Vec::new();

        for v in print_vars {
            let node = match v {
                OutputVariable::Voltage { node, node2: None } => Some(node),
                OutputVariable::VoltageMag { node } => Some(node),
                OutputVariable::VoltagePhase { node } => Some(node),
                OutputVariable::VoltageDb { node } => Some(node),
                OutputVariable::VoltageReal { node } => Some(node),
                OutputVariable::VoltageImag { node } => Some(node),
                _ => None,
            };

            if let Some(node) = node {
                if seen.contains(node) {
                    continue; // Skip duplicates
                }

                // Try to find node in node_map, or parse as number
                if let Some(node_id) = node_map.get(node) {
                    if !node_id.is_ground() {
                        seen.insert(node.clone());
                        result.push((node.clone(), *node_id));
                    }
                } else if let Ok(n) = node.parse::<u32>() {
                    if n > 0 && n <= num_nodes as u32 {
                        seen.insert(node.clone());
                        result.push((node.clone(), NodeId::new(n)));
                    }
                }
            }
        }
        result
    }
}

/// Print DC solution in tabular format.
pub fn print_dc_solution(
    netlist: &spicier_core::Netlist,
    solution: &DcSolution,
    print_vars: &[&OutputVariable],
    node_map: &HashMap<String, NodeId>,
) {
    let nodes_to_print = get_dc_print_nodes(print_vars, node_map, netlist.num_nodes());

    println!("Node Voltages:");
    for (name, node_id) in &nodes_to_print {
        let voltage = solution.voltage(*node_id);
        println!("  V({}) = {:.6} V", name, voltage);
    }

    if netlist.num_current_vars() > 0 && print_vars.is_empty() {
        // Only print currents if no specific print vars (or if I() was specified)
        println!();
        println!("Branch Currents:");
        for i in 0..netlist.num_current_vars() {
            let current = solution.current(i);
            println!("  I(branch{}) = {:.6} A", i, current);
        }
    }
    println!();
}
