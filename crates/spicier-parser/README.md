# spicier-parser

SPICE netlist parser for the Spicier circuit simulator.

## Supported Elements

| Element | Syntax | Description |
|---------|--------|-------------|
| R | `R1 n1 n2 value` | Resistor |
| C | `C1 n1 n2 value` | Capacitor |
| L | `L1 n1 n2 value` | Inductor |
| V | `V1 n+ n- [DC val] [AC mag]` | Voltage source |
| I | `I1 n+ n- [DC val]` | Current source |
| D | `D1 anode cathode [model]` | Diode |
| M | `M1 d g s b model [W=w L=l]` | MOSFET |
| E | `E1 n+ n- nc+ nc- gain` | VCVS |
| G | `G1 n+ n- nc+ nc- gm` | VCCS |
| F | `F1 n+ n- Vsource gain` | CCCS |
| H | `H1 n+ n- Vsource rm` | CCVS |
| B | `B1 n+ n- V=expr` or `I=expr` | Behavioral |
| X | `X1 n1 n2 ... subckt` | Subcircuit instance |

## Commands

- `.OP` - DC operating point
- `.DC source start stop step` - DC sweep
- `.AC type npts fstart fstop` - AC analysis
- `.TRAN tstep tstop [tstart] [UIC]` - Transient
- `.PRINT type var1 var2 ...` - Output selection
- `.IC V(node)=value` - Initial conditions
- `.MODEL name type (params)` - Device models
- `.SUBCKT` / `.ENDS` - Subcircuit definition

## Usage

```rust
use spicier_parser::parse_full;

let netlist = r#"
Voltage Divider
V1 1 0 DC 10
R1 1 2 1k
R2 2 0 1k
.OP
.END
"#;

let result = parse_full(netlist)?;
println!("Nodes: {}", result.netlist.num_nodes());
```

## License

Licensed under either of Apache License, Version 2.0 or MIT license at your option.
