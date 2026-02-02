//! Subcircuit parsing and expansion (.SUBCKT/.ENDS, X instances).

use std::collections::HashMap;

use spicier_core::units::parse_value;
use spicier_devices::diode::{Diode, DiodeParams};
use spicier_devices::mosfet::{Mosfet, MosfetParams, MosfetType};
use spicier_devices::passive::{Capacitor, Inductor, Resistor};
use spicier_devices::sources::{CurrentSource, VoltageSource};

use crate::error::{Error, Result};
use crate::lexer::{Lexer, SpannedToken, Token};

use super::types::RawElementLine;
use super::{ModelDefinition, Parser};

impl<'a> Parser<'a> {
    /// Parse Xname node1 node2 ... subckt_name
    ///
    /// If we're inside a subcircuit definition, store as raw line.
    /// Otherwise, expand the subcircuit inline.
    pub(super) fn parse_subcircuit_instance(&mut self, name: &str, line: usize) -> Result<()> {
        self.advance(); // consume instance name

        // Collect all node names until we hit EOL
        // The last name is the subcircuit name, others are connections
        let mut tokens: Vec<String> = Vec::new();
        loop {
            match self.peek() {
                Token::Eol | Token::Eof => break,
                Token::Name(n) | Token::Value(n) => {
                    tokens.push(n.clone());
                    self.advance();
                }
                _ => {
                    self.advance();
                }
            }
        }

        if tokens.is_empty() {
            return Err(Error::ParseError {
                line,
                message: format!("Subcircuit instance {} requires nodes and subcircuit name", name),
            });
        }

        // Last token is the subcircuit name
        let subckt_name = tokens.pop().unwrap().to_uppercase();
        let connection_nodes = tokens;

        // Build the raw line for storage
        let raw_line = format!("{} {} {}", name, connection_nodes.join(" "), subckt_name);

        if self.current_subckt.is_some() {
            // Inside a subcircuit definition - store raw line
            let subckt = self.current_subckt.as_mut().unwrap();
            subckt.instances.push(RawElementLine { line: raw_line });
        } else {
            // At top level - first register connection nodes to ensure they get proper IDs
            // before any internal subcircuit nodes are created
            for node_name in &connection_nodes {
                self.get_or_create_node(node_name);
            }
            // Then expand the subcircuit
            self.expand_subcircuit(name, &connection_nodes, &subckt_name, line)?;
        }

        self.skip_to_eol();
        Ok(())
    }

    /// Expand a subcircuit instance into the netlist.
    fn expand_subcircuit(
        &mut self,
        instance_name: &str,
        connections: &[String],
        subckt_name: &str,
        line: usize,
    ) -> Result<()> {
        // Look up subcircuit definition
        let subckt = match self.subcircuits.get(subckt_name) {
            Some(s) => s.clone(),
            None => {
                return Err(Error::ParseError {
                    line,
                    message: format!("Unknown subcircuit: {}", subckt_name),
                });
            }
        };

        // Verify port count matches
        if connections.len() != subckt.ports.len() {
            return Err(Error::ParseError {
                line,
                message: format!(
                    "Subcircuit {} expects {} ports but {} provided",
                    subckt_name,
                    subckt.ports.len(),
                    connections.len()
                ),
            });
        }

        // Build node mapping: port_name (uppercase) -> connection_node
        let mut node_map: HashMap<String, String> = HashMap::new();
        for (port, conn) in subckt.ports.iter().zip(connections.iter()) {
            // Store with uppercase key for case-insensitive lookup
            node_map.insert(port.to_uppercase(), conn.clone());
        }

        // Expand element lines with node substitution
        for elem in &subckt.elements {
            let expanded = self.expand_element_line(instance_name, &elem.line, &node_map);
            self.parse_expanded_element(&expanded, line)?;
        }

        // Expand nested subcircuit instances
        for inst in &subckt.instances {
            let expanded = self.expand_element_line(instance_name, &inst.line, &node_map);
            self.parse_expanded_element(&expanded, line)?;
        }

        Ok(())
    }

    /// Expand a single element line with node substitution.
    fn expand_element_line(
        &self,
        instance_prefix: &str,
        line: &str,
        node_map: &HashMap<String, String>,
    ) -> String {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.is_empty() {
            return line.to_string();
        }

        let mut expanded = Vec::new();

        // First part is element name - preserve element type prefix, add instance hierarchy
        // e.g., R1 in instance X1 becomes R_X1_1 (preserving 'R' as first char)
        let elem_name = parts[0];
        let first_char = elem_name.chars().next().unwrap_or('R');
        let rest = if elem_name.len() > 1 { &elem_name[1..] } else { "" };
        expanded.push(format!("{}{}_{}", first_char, instance_prefix, rest));

        // Remaining parts: substitute nodes if in port map, otherwise prefix internal nodes
        for part in &parts[1..] {
            if let Some(mapped) = node_map.get(&part.to_uppercase()) {
                // Port node - use the external connection
                expanded.push(mapped.clone());
            } else if part.parse::<f64>().is_ok() || part.contains('=') || parse_value(part).is_some() {
                // Value (including SPICE suffixes like 1k, 1u) or parameter - keep as-is
                expanded.push(part.to_string());
            } else if part.to_uppercase() == "0" || part.to_uppercase() == "GND" {
                // Ground - keep as-is
                expanded.push(part.to_string());
            } else if part.chars().next().map(|c| c.is_alphabetic()).unwrap_or(false) {
                // Possibly a model name or internal node
                // Check if it looks like a model reference (for D, M elements)
                let upper = part.to_uppercase();
                if self.models.contains_key(&upper) || self.subcircuits.contains_key(&upper) {
                    // Model or subcircuit reference - keep as-is
                    expanded.push(part.to_string());
                } else {
                    // Internal node - prefix with instance name (use _ as separator)
                    expanded.push(format!("{}_{}", instance_prefix, part));
                }
            } else {
                // Internal node with numeric start - prefix (use _ as separator)
                expanded.push(format!("{}_{}", instance_prefix, part));
            }
        }

        expanded.join(" ")
    }

    /// Parse an expanded element line.
    ///
    /// This creates a mini-parser to handle the expanded line and adds the
    /// resulting device to this parser's netlist.
    fn parse_expanded_element(&mut self, line: &str, source_line: usize) -> Result<()> {
        // Tokenize the expanded line
        let lexer = Lexer::new(line);
        let tokens = lexer.tokenize()?;

        if tokens.is_empty() {
            return Ok(());
        }

        // Get the element name from first token
        let name = match &tokens[0].token {
            Token::Name(n) => n.clone(),
            _ => return Ok(()),
        };

        let first_char = name.chars().next().unwrap_or(' ').to_ascii_uppercase();

        // Mini-parser: manually extract nodes and values from tokens
        // This is a simplified approach that handles common element types
        match first_char {
            'R' => {
                // Resistor: R name node1 node2 value
                if tokens.len() >= 4 {
                    let node_pos = self.get_or_create_node(&Self::token_to_string(&tokens[1]));
                    let node_neg = self.get_or_create_node(&Self::token_to_string(&tokens[2]));
                    if let Some(value) = parse_value(&Self::token_to_string(&tokens[3])) {
                        let r = Resistor::new(&name, node_pos, node_neg, value);
                        self.netlist.register_node(node_pos);
                        self.netlist.register_node(node_neg);
                        self.netlist.add_device(r);
                    }
                }
            }
            'C' => {
                // Capacitor: C name node1 node2 value
                if tokens.len() >= 4 {
                    let node_pos = self.get_or_create_node(&Self::token_to_string(&tokens[1]));
                    let node_neg = self.get_or_create_node(&Self::token_to_string(&tokens[2]));
                    if let Some(value) = parse_value(&Self::token_to_string(&tokens[3])) {
                        let c = Capacitor::new(&name, node_pos, node_neg, value);
                        self.netlist.register_node(node_pos);
                        self.netlist.register_node(node_neg);
                        self.netlist.add_device(c);
                    }
                }
            }
            'L' => {
                // Inductor: L name node1 node2 value
                if tokens.len() >= 4 {
                    let node_pos = self.get_or_create_node(&Self::token_to_string(&tokens[1]));
                    let node_neg = self.get_or_create_node(&Self::token_to_string(&tokens[2]));
                    if let Some(value) = parse_value(&Self::token_to_string(&tokens[3])) {
                        let idx = self.next_current_index;
                        self.next_current_index += 1;
                        let l = Inductor::new(&name, node_pos, node_neg, value, idx);
                        self.netlist.register_node(node_pos);
                        self.netlist.register_node(node_neg);
                        self.netlist.add_device(l);
                    }
                }
            }
            'V' => {
                // Voltage source: V name node+ node- value
                if tokens.len() >= 4 {
                    let node_pos = self.get_or_create_node(&Self::token_to_string(&tokens[1]));
                    let node_neg = self.get_or_create_node(&Self::token_to_string(&tokens[2]));
                    if let Some(value) = parse_value(&Self::token_to_string(&tokens[3])) {
                        let idx = self.next_current_index;
                        self.next_current_index += 1;
                        let v = VoltageSource::new(&name, node_pos, node_neg, value, idx);
                        self.netlist.register_node(node_pos);
                        self.netlist.register_node(node_neg);
                        self.netlist.add_device(v);
                    }
                }
            }
            'I' => {
                // Current source: I name node+ node- value
                if tokens.len() >= 4 {
                    let node_pos = self.get_or_create_node(&Self::token_to_string(&tokens[1]));
                    let node_neg = self.get_or_create_node(&Self::token_to_string(&tokens[2]));
                    if let Some(value) = parse_value(&Self::token_to_string(&tokens[3])) {
                        let i = CurrentSource::new(&name, node_pos, node_neg, value);
                        self.netlist.register_node(node_pos);
                        self.netlist.register_node(node_neg);
                        self.netlist.add_device(i);
                    }
                }
            }
            'D' => {
                // Diode: D name anode cathode [model]
                if tokens.len() >= 3 {
                    let anode = self.get_or_create_node(&Self::token_to_string(&tokens[1]));
                    let cathode = self.get_or_create_node(&Self::token_to_string(&tokens[2]));

                    let params = if tokens.len() >= 4 {
                        let model_name = Self::token_to_string(&tokens[3]).to_uppercase();
                        if let Some(ModelDefinition::Diode(p)) = self.models.get(&model_name) {
                            p.clone()
                        } else {
                            DiodeParams::default()
                        }
                    } else {
                        DiodeParams::default()
                    };

                    let d = Diode::with_params(&name, anode, cathode, params);
                    self.netlist.register_node(anode);
                    self.netlist.register_node(cathode);
                    self.netlist.add_device(d);
                }
            }
            'M' => {
                // MOSFET: M name drain gate source bulk [model] [W=val L=val]
                if tokens.len() >= 5 {
                    let drain = self.get_or_create_node(&Self::token_to_string(&tokens[1]));
                    let gate = self.get_or_create_node(&Self::token_to_string(&tokens[2]));
                    let source = self.get_or_create_node(&Self::token_to_string(&tokens[3]));
                    let _bulk = self.get_or_create_node(&Self::token_to_string(&tokens[4]));

                    // Parse model and W/L
                    let mut params = MosfetParams::nmos_default();
                    let mut mos_type = MosfetType::Nmos;

                    for i in 5..tokens.len() {
                        let s = Self::token_to_string(&tokens[i]);
                        let upper = s.to_uppercase();
                        if let Some(ModelDefinition::Nmos(p)) = self.models.get(&upper) {
                            params = p.clone();
                            mos_type = MosfetType::Nmos;
                        } else if let Some(ModelDefinition::Pmos(p)) = self.models.get(&upper) {
                            params = p.clone();
                            mos_type = MosfetType::Pmos;
                        } else if upper.starts_with("W=") {
                            if let Some(v) = parse_value(&s[2..]) {
                                params.w = v;
                            }
                        } else if upper.starts_with("L=") {
                            if let Some(v) = parse_value(&s[2..]) {
                                params.l = v;
                            }
                        }
                    }

                    let m = Mosfet::with_params(&name, drain, gate, source, mos_type, params);
                    self.netlist.register_node(drain);
                    self.netlist.register_node(gate);
                    self.netlist.register_node(source);
                    self.netlist.add_device(m);
                }
            }
            'X' => {
                // Nested subcircuit instance - need to recursively expand
                // Collect only Name and Value tokens (skip Eol, Eof, etc.)
                let mut node_names: Vec<String> = Vec::new();
                for i in 1..tokens.len() {
                    match &tokens[i].token {
                        Token::Name(s) | Token::Value(s) => {
                            node_names.push(s.clone());
                        }
                        Token::Eol | Token::Eof => break,
                        _ => {}
                    }
                }
                if !node_names.is_empty() {
                    let subckt_name = node_names.pop().unwrap().to_uppercase();
                    self.expand_subcircuit(&name, &node_names, &subckt_name, source_line)?;
                }
            }
            _ => {
                // Unknown element type in subcircuit - skip
            }
        }

        Ok(())
    }

    /// Helper to convert a SpannedToken to its string value.
    fn token_to_string(token: &SpannedToken) -> String {
        match &token.token {
            Token::Name(s) | Token::Value(s) => s.clone(),
            _ => String::new(),
        }
    }
}
