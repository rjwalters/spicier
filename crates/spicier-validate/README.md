# spicier-validate

Cross-simulator validation tool comparing Spicier with ngspice.

## Features

- **Cross-simulator comparison** - Run netlists through both simulators
- **Tolerance checking** - Configurable absolute/relative tolerances
- **Golden data validation** - JSON-based reference data
- **Multiple analysis types** - DC, AC, and transient comparison

## CLI Usage

```bash
# Check ngspice availability
spicier-validate check

# Compare a netlist
spicier-validate compare circuit.sp

# Run validation suite
spicier-validate suite --golden-dir tests/golden_data

# Generate golden data from ngspice
spicier-validate generate circuit.sp -o golden.json
```

## Library Usage

```rust
use spicier_validate::{compare_simulators, ComparisonConfig};

let netlist = "V1 1 0 DC 10\nR1 1 2 1k\nR2 2 0 1k\n.op\n.end\n";
let config = ComparisonConfig::default();

let report = compare_simulators(netlist, &config)?;
if report.passed {
    println!("Results match within tolerance!");
} else {
    println!("{}", report.to_text());
}
```

## Requirements

- ngspice installed and in PATH (for cross-simulator comparison)

## License

Licensed under either of Apache License, Version 2.0 or MIT license at your option.
