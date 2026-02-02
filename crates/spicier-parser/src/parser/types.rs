//! Public types for the SPICE parser.

use std::collections::HashMap;

use spicier_core::{Netlist, NodeId};

/// AC sweep type parsed from netlist.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum AcSweepType {
    /// Linear frequency spacing.
    Lin,
    /// Logarithmic spacing per decade.
    Dec,
    /// Logarithmic spacing per octave.
    Oct,
}

/// A single DC sweep specification.
#[derive(Debug, Clone)]
pub struct DcSweepSpec {
    /// Name of the source to sweep.
    pub source_name: String,
    /// Start value.
    pub start: f64,
    /// Stop value.
    pub stop: f64,
    /// Step size.
    pub step: f64,
}

/// An analysis command parsed from the netlist.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum AnalysisCommand {
    /// DC operating point (.OP).
    Op,
    /// DC sweep (.DC source start stop step [source2 start2 stop2 step2]).
    ///
    /// Supports nested sweeps: the first sweep is the outer (slow) sweep,
    /// the second (if present) is the inner (fast) sweep.
    Dc {
        /// One or two sweep specifications.
        sweeps: Vec<DcSweepSpec>,
    },
    /// AC sweep (.AC type npoints fstart fstop).
    Ac {
        sweep_type: AcSweepType,
        num_points: usize,
        fstart: f64,
        fstop: f64,
    },
    /// Transient analysis (.TRAN tstep tstop \[tstart\] \[tmax\] \[UIC\]).
    Tran {
        tstep: f64,
        tstop: f64,
        tstart: f64,
        /// Use Initial Conditions - skip DC operating point, use .IC values directly.
        uic: bool,
    },
}

/// Initial condition for a node voltage.
#[derive(Debug, Clone)]
pub struct InitialCondition {
    /// Node name (e.g., "1", "out").
    pub node: String,
    /// Initial voltage value.
    pub voltage: f64,
}

/// Type of analysis for .PRINT command.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum PrintAnalysisType {
    /// DC operating point or DC sweep.
    Dc,
    /// AC analysis.
    Ac,
    /// Transient analysis.
    Tran,
}

/// An output variable specification from .PRINT command.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum OutputVariable {
    /// Node voltage: V(node) or V(node1, node2) for differential.
    Voltage { node: String, node2: Option<String> },
    /// Device current: I(device).
    Current { device: String },
    /// Real part of voltage (AC): VR(node).
    VoltageReal { node: String },
    /// Imaginary part of voltage (AC): VI(node).
    VoltageImag { node: String },
    /// Magnitude of voltage (AC): VM(node).
    VoltageMag { node: String },
    /// Phase of voltage (AC): VP(node).
    VoltagePhase { node: String },
    /// Magnitude in dB (AC): VDB(node).
    VoltageDb { node: String },
}

/// A .PRINT command specifying output variables for an analysis type.
#[derive(Debug, Clone)]
pub struct PrintCommand {
    /// Type of analysis this print applies to.
    pub analysis_type: PrintAnalysisType,
    /// Variables to output.
    pub variables: Vec<OutputVariable>,
}

/// A raw element line stored in a subcircuit definition.
///
/// We store element lines as strings to be re-parsed during expansion,
/// allowing proper node name substitution.
#[derive(Debug, Clone)]
pub struct RawElementLine {
    /// The full element line (e.g., "R1 1 2 1k").
    pub line: String,
}

/// A subcircuit definition from .SUBCKT/.ENDS block.
#[derive(Debug, Clone)]
pub struct SubcircuitDefinition {
    /// Subcircuit name (e.g., "NAND", "OPAMP").
    pub name: String,
    /// Port node names in order (external interface).
    pub ports: Vec<String>,
    /// Element lines inside the subcircuit (stored as raw text).
    pub elements: Vec<RawElementLine>,
    /// Nested subcircuit instantiations (X lines).
    pub instances: Vec<RawElementLine>,
}

impl SubcircuitDefinition {
    pub(super) fn new(name: String, ports: Vec<String>) -> Self {
        Self {
            name,
            ports,
            elements: Vec::new(),
            instances: Vec::new(),
        }
    }
}

/// Result of parsing a SPICE netlist.
///
/// Contains both the circuit (Netlist) and analysis commands.
#[derive(Debug)]
pub struct ParseResult {
    /// The circuit netlist.
    pub netlist: Netlist,
    /// Analysis commands found in the netlist.
    pub analyses: Vec<AnalysisCommand>,
    /// Initial conditions from .IC commands.
    pub initial_conditions: Vec<InitialCondition>,
    /// Node name to NodeId mapping.
    pub node_map: HashMap<String, NodeId>,
    /// Print commands specifying output variables.
    pub print_commands: Vec<PrintCommand>,
    /// Subcircuit definitions from .SUBCKT blocks.
    pub subcircuits: HashMap<String, SubcircuitDefinition>,
    /// Parameters from .PARAM commands (name -> value).
    pub parameters: HashMap<String, f64>,
}
