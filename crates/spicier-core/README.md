# spicier-core

Core circuit representation and MNA (Modified Nodal Analysis) matrix structures for the Spicier circuit simulator.

## Features

- **Circuit graph** - Node and branch representation for circuit topology
- **MNA system** - Sparse matrix assembly with stamping methods
- **Netlist** - High-level circuit assembly with automatic node management
- **Units** - SI unit parsing (k, M, u, n, p, etc.)

## Usage

```rust
use spicier_core::{Circuit, MnaSystem, NodeId};

// Create a circuit and add nodes
let mut circuit = Circuit::new();
let n1 = circuit.add_node(Some("vdd".to_string()));
let n2 = circuit.add_node(Some("out".to_string()));

// Build MNA system
let mut mna = MnaSystem::new(2, 1); // 2 nodes, 1 voltage source
mna.stamp_conductance(Some(0), Some(1), 1e-3); // 1k resistor
mna.stamp_voltage_source(Some(0), None, 0, 5.0); // 5V source
```

## License

Licensed under either of Apache License, Version 2.0 or MIT license at your option.
